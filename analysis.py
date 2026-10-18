"""Analysis — the 'predict' phase: per-week load, overload / idle, deadline risk.

Per-person and per-project metrics are independent, so they are computed in
parallel with a thread pool — the direct analog of Hermes's batch_runner
(`_process_batch_worker`) and of delegate_task spawning isolated subagents.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import timedelta

import methodology
from engine import week_of, week_capacity, WEEK, DAY
from models import Person, Project, Plan


@dataclass
class PersonStats:
    person: Person
    booked_weeks: float = 0        # full-time person-weeks of work booked in the active window
    available_weeks: float = 0     # working weeks in the window (leave / not-started days excluded)
    capacity_weeks: float = 0      # available_weeks x capacity: what could be booked
    peak_load: float = 0           # busiest week, in full-time roles (1.0 = one full-time role)
    util: float = 0.0              # booked / capacity_weeks over the active window
    overloaded: bool = False       # some week is booked past capacity + allowed cover
    idle: bool = False             # utilization < 30%
    cover_days: int = 0            # person-days someone else must cover (short leave)


@dataclass
class ProjectRisk:
    project: Project
    risk: str          # "ok" | "unstaffable" | "late" | "tight"
    start_week: int | None = None
    done_week: int | None = None
    done_date: str | None = None
    days_left: int | None = None   # days from expected completion to deadline
    milestones: list | None = None  # [{stage, due, planned, days_left}] for waterfall stage due dates

    @property
    def late_days(self) -> int:
        """Days late by the worst of the deadline and the stage due dates (0 if not late)."""
        if self.risk != "late":
            return 0
        return max([0, -(self.days_left or 0)] + [-m["days_left"] for m in self.milestones or []])


def active_weeks(plan: Plan, projects: list[Project]) -> int:
    """Weeks that actually carry work: up to the last finish or last deadline.

    The solver horizon pads ~6+ weeks of slack past the last deadline; dividing
    load by it made a half-busy engineer look 20% utilized.
    """
    dur = {p.id: p.duration_weeks for p in projects}
    last_finish = max((a.start_week + dur.get(a.project_id, 0) for a in plan.assignments), default=0)
    last_deadline = max((week_of(p.deadline, plan.start_date) + 1 for p in projects), default=0)
    return max(1, min(plan.horizon_weeks, max(last_finish, last_deadline)))


def _person_stats(args: tuple[Person, Plan, int]) -> PersonStats:
    person, plan, window = args
    load = plan.load.get(person.id, {})
    booked = sum(u for w, u in load.items() if w < window)
    caps = [week_capacity(person, plan.start_date, w) for w in range(window)]
    avail = sum(a for a, _ in caps)
    cover = sum(max(0, load.get(w, 0) - a) for w, (a, _) in enumerate(caps))
    over = any(load.get(w, 0) > a + c for w, (a, c) in enumerate(caps))
    util = booked / avail if avail else 0.0
    return PersonStats(
        person=person,
        booked_weeks=round(booked / WEEK, 1),
        available_weeks=round(avail / (WEEK * person.capacity), 1) if person.capacity else 0,
        capacity_weeks=round(avail / WEEK, 1),
        peak_load=round(max(load.values(), default=0) / WEEK, 2),
        util=util,
        overloaded=over,
        idle=util < 0.3,
        cover_days=cover // DAY,
    )


def analyze(plan: Plan, people: list[Person], projects: list[Project]) -> tuple[list[PersonStats], list[ProjectRisk]]:
    with ThreadPoolExecutor() as pool:
        window = active_weeks(plan, projects)
        person_stats = list(pool.map(_person_stats, [(p, plan, window) for p in people]))
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
        last_week = a.start_week + max(proj.duration_weeks, 1) - 1
        # Weeks start Monday; finish = Friday of the last occupied week.
        done_week = last_week
        done_date = plan.start_date + timedelta(weeks=last_week, days=4)
        days_left = (proj.deadline - done_date).days
        if days_left < 0:
            risk = "late"
        elif days_left < 7:
            risk = "tight"
        else:
            risk = "ok"
        ms = []
        for stage, due in proj.milestones.items():
            end = methodology.stage_end(proj.stages, stage)
            if end is None:
                continue
            planned = plan.start_date + timedelta(weeks=a.start_week + end, days=4)
            ms.append({"stage": stage, "due": due.isoformat(), "planned": planned.isoformat(),
                       "days_left": (due - planned).days})
        if risk != "late" and any(m["days_left"] < 0 for m in ms):
            risk = "late"   # a missed contractual stage date is a late project
        out.append(ProjectRisk(proj, risk, a.start_week, done_week,
                               done_date.isoformat(), days_left, ms))
    return out


def _remaining_in_window(person: Person, plan: Plan, last_week: int) -> float:
    """Free full-time person-weeks inside [0, last_week), leave excluded."""
    free = 0
    end = min(max(last_week, 0), plan.horizon_weeks)
    for w in range(end):
        avail, _ = week_capacity(person, plan.start_date, w)
        free += max(0, avail - plan.load.get(person.id, {}).get(w, 0))
    return free / WEEK


def find_bottlenecks(plan: Plan, people: list[Person], projects: list[Project],
                     risks: list[ProjectRisk], pstats: list[PersonStats] | None = None
                     ) -> list[dict]:
    """Scarce (role, level) that actually cause lateness or dropped projects.

    Unique specialists needed by late work are bottlenecks even if they have
    slack later in a long horizon. Idle qualified people (e.g. Ben the PM)
    still mean that role is not the constraint.
    """
    util = {s.person.id: s.util for s in (pstats or [])}
    troubled = {r.project.id for r in risks if r.risk in ("late", "unstaffable")}
    troubled.update(pid for pid, _ in plan.unstaffable)

    demand: dict[tuple[str, int], list[str]] = {}
    crit = 0
    pmeta = {p.id: p for p in projects}
    for proj in projects:
        if proj.id not in troubled:
            continue
        crit = max(crit, week_of(proj.deadline, plan.start_date) + 2)
        for role in proj.roles():
            key = (role, proj.level_for(role))
            demand.setdefault(key, []).append(proj.name)
    for a in plan.assignments:
        if a.project_id in troubled and a.project_id in pmeta:
            j = pmeta[a.project_id]
            crit = max(crit, a.start_week + j.duration_weeks)
    crit = min(max(crit, 8), plan.horizon_weeks)

    out: list[dict] = []
    seen: set[tuple[str, int]] = set()
    seen_groups: set[tuple[str, frozenset]] = set()
    ranked = sorted(demand.items(), key=lambda kv: (-len(kv[1]), kv[0][0], -kv[0][1]))
    for (role, lvl), names in ranked:
        if (role, lvl) in seen:
            continue
        seen.add((role, lvl))
        qualified = [p for p in people if p.level(role) >= lvl]
        # analyst≥4 and analyst≥3 held by the same people are one bottleneck
        group = (role, frozenset(p.id for p in qualified))
        if qualified and group in seen_groups:
            continue
        seen_groups.add(group)
        qnames = ", ".join(p.name for p in qualified) if qualified else ""
        if not qualified:
            out.append({
                "role": role, "level": lvl,
                "detail": f"nobody has {role}≥{lvl}; affects: {', '.join(names)}",
            })
            continue
        slack = [p for p in qualified if _remaining_in_window(p, plan, crit) >= 3]
        unique = len(qualified) == 1
        packed = not slack
        # Unique person for a role late work needs = bottleneck (Ivy / QA≥3).
        # Several people with slack in the critical window (Ben the PM) are not.
        if unique or packed:
            if unique:
                who = qualified[0].name
                pct = round(util.get(qualified[0].id, 0) * 100)
                detail = (f"only {who} has {role}≥{lvl}"
                          + (f" (load {pct}%)" if pct else "")
                          + f"; {len(names)} late/unstaffed project(s) need this role")
            else:
                detail = (f"{qnames} have {role}≥{lvl} but are fully booked; "
                          f"affects: {', '.join(names)}")
            out.append({"role": role, "level": lvl, "detail": detail})
    return out
