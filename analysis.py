"""Analysis — the 'predict' phase: per-week load, overload / idle, deadline risk.

Per-person and per-project metrics are independent, so they are computed in
parallel with a thread pool — the direct analog of Hermes's batch_runner
(`_process_batch_worker`) and of delegate_task spawning isolated subagents.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import timedelta

import math

from engine import week_of, _unavailable_weeks
from models import Person, Project, Plan


@dataclass
class PersonStats:
    person: Person
    booked_weeks: int = 0          # person-weeks of load (sum over weeks of concurrent projects)
    available_weeks: int = 0
    peak_load: int = 0             # max concurrent projects in any week
    util: float = 0.0              # booked / (capacity * available) — can exceed 100%
    overloaded: bool = False       # some week has load > capacity
    idle: bool = False             # utilization < 30%


@dataclass
class ProjectRisk:
    project: Project
    risk: str          # "ok" | "unstaffable" | "late" | "tight"
    start_week: int | None = None
    done_week: int | None = None
    done_date: str | None = None
    days_left: int | None = None   # days from expected completion to deadline


def _person_stats(args: tuple[Person, Plan]) -> PersonStats:
    person, plan = args
    weeks = plan.committed.get(person.id, {})
    booked = sum(len(s) for s in weeks.values())          # person-weeks (counts overlap)
    peak = max((len(s) for s in weeks.values()), default=0)
    off = len(_unavailable_weeks(person, plan.start_date) & set(range(plan.horizon_weeks)))
    available = plan.horizon_weeks - off
    denom = person.capacity * available
    util = booked / denom if denom else 0.0
    return PersonStats(
        person=person,
        booked_weeks=booked,
        available_weeks=available,
        peak_load=peak,
        util=util,
        overloaded=peak > person.capacity,
        idle=util < 0.3,
    )


def analyze(plan: Plan, people: list[Person], projects: list[Project]) -> tuple[list[PersonStats], list[ProjectRisk]]:
    with ThreadPoolExecutor() as pool:
        person_stats = list(pool.map(_person_stats, [(p, plan) for p in people]))
    risks = _project_risks(plan, projects)
    return person_stats, risks


def _project_risks(plan: Plan, projects: list[Project]) -> list[ProjectRisk]:
    by_id = {a.project_id: a for a in plan.assignments}
    unstaffable_ids = {pid for pid, _ in plan.unstaffable}
    out: list[ProjectRisk] = []
    for proj in projects:
        if proj.id in unstaffable_ids:
            out.append(ProjectRisk(proj, "unstaffable"))
            continue
        a = by_id.get(proj.id)
        if a is None:
            out.append(ProjectRisk(proj, "unstaffable"))
            continue
        done_week = a.start_week + proj.duration_weeks
        done_date = plan.start_date + timedelta(weeks=done_week)
        days_left = (proj.deadline - done_date).days
        if days_left < 0:
            risk = "late"
        elif days_left < 7:
            risk = "tight"
        else:
            risk = "ok"
        out.append(ProjectRisk(proj, risk, a.start_week, done_week,
                               done_date.isoformat(), days_left))
    return out


def _remaining_weeks(person: Person, plan: Plan) -> int:
    """Free person-weeks (capacity - load) over the horizon, excluding leave."""
    cap = max(1, math.ceil(person.capacity))
    off = _unavailable_weeks(person, plan.start_date)
    free = 0
    for w in range(plan.horizon_weeks):
        if w in off:
            continue
        load = len(plan.committed.get(person.id, {}).get(w, set()))
        free += max(0, cap - load)
    return free


def find_bottlenecks(plan: Plan, people: list[Person], projects: list[Project],
                     risks: list[ProjectRisk], pstats: list[PersonStats] | None = None
                     ) -> list[dict]:
    """Scarce (role, level) that actually cause lateness or dropped projects.

    A role is a bottleneck only when a late/unstaffable project needs it AND
    every qualified person is packed (or there is a single busy specialist).
    Idle qualified people (e.g. Ben the PM) mean that role is not the constraint.
    """
    util = {s.person.id: s.util for s in (pstats or [])}
    troubled = {r.project.id for r in risks if r.risk in ("late", "unstaffable")}
    troubled.update(pid for pid, _ in plan.unstaffable)

    demand: dict[tuple[str, int], list[str]] = {}
    for proj in projects:
        if proj.id not in troubled:
            continue
        for role in proj.roles():
            key = (role, proj.level_for(role))
            demand.setdefault(key, []).append(proj.name)

    out: list[dict] = []
    seen: set[tuple[str, int]] = set()
    for (role, lvl), names in sorted(demand.items()):
        qualified = [p for p in people if p.level(role) >= lvl]
        if (role, lvl) in seen:
            continue
        seen.add((role, lvl))
        qnames = ", ".join(p.name for p in qualified) if qualified else ""
        if not qualified:
            out.append({
                "role": role, "level": lvl,
                "detail": f"không ai đủ {role}≥{lvl}; ảnh hưởng: {', '.join(names)}",
            })
            continue
        slack = [p for p in qualified if _remaining_weeks(p, plan) >= 3]
        remaining = _remaining_weeks(qualified[0], plan) if qualified else 0
        busy_unique = (
            len(qualified) == 1
            and util.get(qualified[0].id, 0) >= 0.6
            and remaining <= 3
        )
        all_packed = not slack
        if busy_unique or all_packed:
            if len(qualified) == 1:
                who = qualified[0].name
                pct = round(util.get(qualified[0].id, 0) * 100)
                detail = (f"chỉ {who} đủ {role}≥{lvl}"
                          + (f" (tải {pct}%)" if pct else "")
                          + f"; {len(names)} dự án trễ/không xếp được cần vai trò này")
            else:
                detail = (f"{qnames} đủ {role}≥{lvl} đều kín lịch; "
                          f"ảnh hưởng: {', '.join(names)}")
            out.append({"role": role, "level": lvl, "detail": detail})
    return out
