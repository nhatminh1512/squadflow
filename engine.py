"""Assignment engine — the 'plan/optimize' phase of the pipeline.

Greedy placement, priority-then-urgency ordered, with two hard constraints
per person (skill + availability) and one SOFT constraint (capacity):

  1. skill: the person holds the role at >= the required proficiency
     (matching is by skill, not primary role — a person may hold several)
  2. available: not on leave any week the project runs
  3. capacity (soft): prefer people with spare capacity; if none exists for a
     role, the least-loaded qualified person is still assigned and flagged as
     overloaded, rather than the project being silently dropped. Overload then
     becomes visible in analysis (>100% in some week) instead of hidden.
"""
from __future__ import annotations

from collections import defaultdict
from datetime import date

from models import Person, Project, Assignment, Plan


def week_of(d: date, start: date) -> int:
    return (d - start).days // 7


def _unavailable_weeks(person: Person, start: date) -> set[int]:
    out: set[int] = set()
    for a, b in person.leave:
        wa, wb = week_of(a, start), week_of(b, start)
        out.update(range(wa, wb + 1))
    return out


def _horizon_weeks(projects: list[Project], start: date) -> int:
    return max((week_of(p.deadline, start) for p in projects), default=0) + 1


def _load(committed: dict[str, dict[int, set[str]]], pid: str, week: int) -> int:
    return len(committed.get(pid, {}).get(week, set()))


def _candidate(
    people: list[Person],
    role: str,
    min_level: int,
    start_week: int,
    duration: int,
    committed: dict[str, dict[int, set[str]]],
    unavailable: dict[str, set[int]],
    exclude: set[str] | None = None,
) -> tuple[Person | None, dict | None]:
    """Pick the best person for (role, min_level) over [start, start+duration).

    Skill + availability are hard; capacity is soft. Preference:
    least-qualified-acceptable (reserve seniors), then least-loaded.
    Returns (person, decision_meta) where meta carries the grounding facts
    for a later explanation (so an LLM can phrase, never invent).
    """
    weeks = range(start_week, start_week + duration)
    exclude = exclude or set()
    eligible = [
        p for p in people
        if p.id not in exclude
        and p.level(role) >= min_level
        and all(w >= 0 and w not in unavailable.get(p.id, set()) for w in weeks)
    ]
    if not eligible:
        return None, None

    feasible = [
        p for p in eligible
        if all(_load(committed, p.id, w) < p.capacity for w in weeks)
    ]
    overloaded = not feasible
    pool = feasible if feasible else eligible
    chosen = min(pool, key=lambda p: (
        p.level(role),
        sum(_load(committed, p.id, w) for w in weeks),
    ))
    meta = {
        "role": role,
        "min_level": min_level,
        "chosen_level": chosen.level(role),
        "n_eligible": len(eligible),
        "n_feasible": len(feasible),
        "overloaded": overloaded,
    }
    return chosen, meta


def assign(people: list[Person], projects: list[Project], start_date: date) -> Plan:
    horizon = _horizon_weeks(projects, start_date)
    unavailable = {p.id: _unavailable_weeks(p, start_date) for p in people}
    committed: dict[str, dict[int, set[str]]] = defaultdict(lambda: defaultdict(set))
    assignments: list[Assignment] = []
    unstaffable: list[tuple[str, str]] = []
    decisions: list[dict] = []
    finish: dict[str, int] = {}
    pname = {p.id: p.name for p in projects}

    # priority desc, then urgency (latest feasible start — tighter first)
    ordered = sorted(
        projects,
        key=lambda p: (-p.priority, week_of(p.deadline, start_date) - p.duration_weeks),
    )

    for proj in ordered:
        duration = proj.duration_weeks
        latest_start = week_of(proj.deadline, start_date) - duration

        dep_start = 0
        blocked = False
        for dep in proj.depends_on:
            if dep not in finish:
                unstaffable.append(
                    (proj.id, f"đang chờ dự án “{pname.get(dep, dep)}” chưa xếp được")
                )
                blocked = True
                break
            dep_start = max(dep_start, finish[dep] + 1)
        if blocked:
            continue

        if latest_start < dep_start:
            unstaffable.append((proj.id, "hết thời gian: phụ thuộc xong quá trễ so với deadline"))
            continue

        # prefer a feasible (no-overload) start week; only overload when the
        # deadline leaves no feasible start at all. This levels load instead of
        # dumping every project into week 0.
        placed_feasible: tuple | None = None
        placed_overload: tuple | None = None
        missing: tuple[str, int] | None = None
        for start_week in range(dep_start, latest_start + 1):
            squad: dict[str, list[str]] = {}
            metas: dict[str, list[dict]] = {}
            overloaded = False
            incomplete = False
            for role in proj.roles():
                min_level = proj.level_for(role)
                count = proj.count_for(role)
                squad[role] = []
                metas[role] = []
                excluded: set[str] = set()
                for _ in range(count):
                    person, meta = _candidate(people, role, min_level, start_week, duration,
                                              committed, unavailable, exclude=excluded)
                    if person is None:
                        incomplete = True
                        missing = (role, min_level)
                        break
                    squad[role].append(person.id)
                    metas[role].append(meta)
                    excluded.add(person.id)
                    overloaded = overloaded or meta["overloaded"]
                if incomplete:
                    break
            if incomplete:
                continue
            if not overloaded:
                placed_feasible = (start_week, squad, metas)
                break
            if placed_overload is None:
                placed_overload = (start_week, squad, metas)

        placed = placed_feasible or placed_overload

        if placed is None:
            if missing:
                role, lvl = missing
                unstaffable.append((proj.id, f"thiếu người {role} trình độ ≥ {lvl} trước deadline"))
            else:
                unstaffable.append((proj.id, "không xếp được trước deadline"))
            continue

        start_week, squad, metas = placed
        for role, pids in squad.items():
            for pid, meta in zip(pids, metas[role]):
                meta.update({"project_id": proj.id, "person_id": pid, "start_week": start_week})
                decisions.append(meta)
                for w in range(start_week, start_week + duration):
                    committed[pid][w].add(proj.id)
        assignments.append(Assignment(proj.id, start_week, squad))
        finish[proj.id] = start_week + duration - 1

    return Plan(start_date, horizon, assignments, unstaffable, dict(committed), decisions)
