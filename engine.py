"""Assignment engine — the 'plan/optimize' phase of the pipeline.

Greedy placement, priority-then-urgency ordered, with three HARD constraints
per person and a SOFT deadline:

  1. skill: the person holds the role at >= the required proficiency
     (matching is by skill, not primary role — a person may hold several)
  2. capacity (hard), counted in person-days per week: each role books
     effort% x 5 days in the weeks of its phase; a person offers
     capacity x 5 days minus leave days. A short leave may leave up to
     MAX_COVER_DAYS per week uncovered (someone covers; penalized), a long
     one blocks the week.
  3. deadline (soft): a project may start after its latest on-time week and
     is flagged late; it is only dropped when skill/availability/dependency
     make it impossible, never because it would miss the deadline.
"""
from __future__ import annotations

from collections import defaultdict
from datetime import date, timedelta

from models import Person, Project, Assignment, Plan

# Extra weeks past the latest deadline so a late project still has a slot.
SLACK_WEEKS = 6

DAY = 100            # load unit: one person-day of full-time work
WEEK = 5 * DAY       # a full-time role for one week
MAX_COVER_DAYS = 2   # a short leave may leave this many days per week to cover


def week_of(d: date, start: date) -> int:
    return (d - start).days // 7


def role_load(proj: Project, role: str, s: int) -> dict[int, int]:
    """week -> units one person in `role` books, for proj starting at week s."""
    return {w: e * WEEK // 100 for w, e in proj.role_effort(role, s).items()}


def _workdays(start: date, w: int) -> list[date]:
    mon = start + timedelta(weeks=w)
    return [d for d in (mon + timedelta(days=i) for i in range(7)) if d.weekday() < 5]


def week_capacity(person: Person, start: date, w: int) -> tuple[int, int]:
    """(available units, coverable shortfall) for week w.

    Days before a new hire's first day are simply absent (nothing to cover);
    leave days reduce availability and allow up to MAX_COVER_DAYS of cover."""
    present = leave_days = 0
    for d in _workdays(start, w):
        if person.available_from and d < person.available_from:
            continue
        if any(a <= d <= b for a, b in person.leave):
            leave_days += 1
        else:
            present += 1
    return round(person.capacity * DAY * present), min(MAX_COVER_DAYS, leave_days) * DAY


def leave_days_in_week(person: Person, start: date, w: int) -> int:
    return sum(1 for d in _workdays(start, w) if any(a <= d <= b for a, b in person.leave))


def _unavailable_weeks(person: Person, start: date) -> set[int]:
    """Weeks with no working day at all (whole-week leave, or not started yet)."""
    last = max([week_of(b, start) for _, b in person.leave]
               + ([week_of(person.available_from, start)] if person.available_from else [])
               + [-1])
    return {w for w in range(0, last + 1) if week_capacity(person, start, w)[0] == 0}


def book(load: dict, committed: dict, proj: Project, role: str, pid: str, s: int) -> None:
    """Record pid working `role` on proj (project starting week s)."""
    for w, need in role_load(proj, role, s).items():
        load[pid][w] += need
        committed[pid][w].add(proj.id)


def role_fits(person: Person, proj: Project, role: str, s: int, start: date,
              load: dict | None = None) -> bool:
    """Can person take `role` on proj starting week s, on top of `load`?"""
    for w, need in role_load(proj, role, s).items():
        if w < 0:
            return False
        avail, cover = week_capacity(person, start, w)
        if ((load or {}).get(person.id, {}).get(w, 0)) + need > avail + cover:
            return False
    return True


def shortfall(person: Person, proj: Project, role: str, s: int, start: date,
              load: dict | None = None) -> int:
    """Units of cover needed (over availability, within the allowed cover)."""
    out = 0
    for w, need in role_load(proj, role, s).items():
        avail, _ = week_capacity(person, start, w)
        out += max(0, (load or {}).get(person.id, {}).get(w, 0) + need - avail)
    return out


def _horizon_weeks(projects: list[Project], start: date) -> int:
    """Long enough to place every project sequentially, even a few weeks late."""
    max_deadline = max((week_of(p.deadline, start) for p in projects), default=0)
    max_dur = max((p.duration_weeks for p in projects), default=1)
    total_dur = sum(p.duration_weeks for p in projects)
    return max(max_deadline + SLACK_WEEKS, max_dur, total_dur) + 1


def _candidate(
    people: list[Person],
    proj: Project,
    role: str,
    start_week: int,
    start: date,
    load: dict[str, dict[int, int]],
    exclude: set[str] | None = None,
) -> tuple[Person | None, dict | None]:
    """Pick the best person for `role` of proj starting at start_week.

    Skill, availability and capacity are all hard. Preference: least cover
    needed, then least-qualified-acceptable (reserve seniors), then least-loaded.
    Returns (person, decision_meta) where meta carries the grounding facts
    for a later explanation (so an LLM can phrase, never invent).
    """
    min_level = proj.level_for(role)
    weeks = proj.role_weeks(role, start_week)
    exclude = exclude or set()
    eligible = [
        p for p in people
        if p.id not in exclude
        and p.level(role) >= min_level
        and role_fits(p, proj, role, start_week, start)
    ]
    if not eligible:
        return None, None

    feasible = [p for p in eligible if role_fits(p, proj, role, start_week, start, load)]
    if not feasible:
        return None, None
    chosen = min(feasible, key=lambda p: (
        shortfall(p, proj, role, start_week, start, load),
        p.level(role),
        sum(load[p.id][w] for w in weeks),
    ))
    meta = {
        "role": role,
        "min_level": min_level,
        "chosen_level": chosen.level(role),
        "n_eligible": len(eligible),
        "n_feasible": len(feasible),
        "overloaded": False,
    }
    return chosen, meta


def assign(people: list[Person], projects: list[Project], start_date: date,
           freeze: list[Assignment] | None = None,
           pins: list | None = None) -> Plan:
    jmeta = {j.id: j for j in projects}
    horizon = _horizon_weeks(projects, start_date)
    freeze_map = {a.project_id: a for a in (freeze or []) if a.project_id in jmeta}
    for a in freeze_map.values():
        j = jmeta[a.project_id]
        horizon = max(horizon, int(a.start_week) + j.duration_weeks + SLACK_WEEKS)
    committed: dict[str, dict[int, set[str]]] = defaultdict(lambda: defaultdict(set))
    load: dict[str, dict[int, int]] = defaultdict(lambda: defaultdict(int))
    assignments: list[Assignment] = []
    unstaffable: list[tuple[str, str]] = []
    decisions: list[dict] = []
    finish: dict[str, int] = {}
    pname = {p.id: p.name for p in projects}

    for jid, a in freeze_map.items():
        j = jmeta[jid]
        s = int(a.start_week)
        squad = {
            r: list(pids if isinstance(pids, list) else [pids])
            for r, pids in a.squad.items()
        }
        assignments.append(Assignment(jid, s, squad))
        finish[jid] = s + j.duration_weeks - 1
        for role, pids in squad.items():
            for pid in pids:
                decisions.append({
                    "project_id": jid, "role": role, "person_id": pid,
                    "start_week": s, "solver": "freeze", "overloaded": False,
                    "min_level": j.level_for(role),
                    "chosen_level": next((p.level(role) for p in people if p.id == pid), 0),
                    "n_eligible": 1, "n_feasible": 1,
                })
                book(load, committed, j, role, pid, s)

    # priority desc, then urgency (latest feasible start — tighter first)
    ordered = sorted(
        [p for p in projects if p.id not in freeze_map],
        key=lambda p: (-p.priority, week_of(p.deadline, start_date) - p.duration_weeks),
    )

    for proj in ordered:
        duration = proj.duration_weeks
        max_start = max(0, horizon - duration)

        dep_start = 0
        blocked = False
        for dep in proj.depends_on:
            if dep not in finish:
                unstaffable.append(
                    (proj.id, f"waiting on “{pname.get(dep, dep)}”, which could not be staffed")
                )
                blocked = True
                break
            dep_start = max(dep_start, finish[dep] + 1)
        if blocked:
            continue

        # Deadline is SOFT: search past the on-time window, never drop for lateness.
        placed: tuple | None = None
        missing: tuple[str, int] | None = None
        for start_week in range(dep_start, max_start + 1):
            squad: dict[str, list[str]] = {}
            metas: dict[str, list[dict]] = {}
            incomplete = False
            # tentative bookings so a 2nd person in the same squad sees the 1st
            trial: dict[str, dict[int, int]] = defaultdict(lambda: defaultdict(int))
            for pid, weeks in load.items():
                trial[pid].update(weeks)
            excluded: set[str] = set()
            for role in proj.roles():
                min_level = proj.level_for(role)
                count = proj.count_for(role)
                squad[role] = []
                metas[role] = []
                for _ in range(count):
                    person, meta = _candidate(people, proj, role, start_week, start_date,
                                              trial, exclude=excluded)
                    if person is None:
                        incomplete = True
                        missing = (role, min_level)
                        break
                    for w, need in role_load(proj, role, start_week).items():
                        trial[person.id][w] += need
                    squad[role].append(person.id)
                    metas[role].append(meta)
                    excluded.add(person.id)
                if incomplete:
                    break
            if incomplete:
                continue
            placed = (start_week, squad, metas)
            break

        if placed is None:
            if missing:
                role, lvl = missing
                qualified = [p for p in people if p.level(role) >= lvl]
                if not qualified:
                    unstaffable.append(
                        (proj.id, f"no {role} with level ≥ {lvl} (nobody qualifies)")
                    )
                else:
                    names = ", ".join(p.name for p in qualified)
                    unstaffable.append(
                        (proj.id,
                         f"not enough {role} with level ≥ {lvl} ({names}) — "
                         f"all busy or on leave, no free slot")
                    )
            else:
                unstaffable.append((proj.id, "could not be staffed (no free slot in the schedule)"))
            continue

        start_week, squad, metas = placed
        for role, pids in squad.items():
            for pid, meta in zip(pids, metas[role]):
                meta.update({"project_id": proj.id, "person_id": pid, "start_week": start_week})
                decisions.append(meta)
                book(load, committed, proj, role, pid, start_week)
        assignments.append(Assignment(proj.id, start_week, squad))
        finish[proj.id] = start_week + duration - 1

    return Plan(start_date, horizon, assignments, unstaffable, dict(committed), decisions,
                {pid: {w: u for w, u in weeks.items() if u} for pid, weeks in load.items()})


def compatible_freeze(prev, body: dict, projects=None,
                      today: date = date(2026, 10, 19)) -> list:
    """Selective freeze for what-if / replan:
    - Keep on-time projects frozen unless directly touched by leave/remove/slip/pin.
    - Unlock LATE projects when a new person is hired (or skills/capacity updated)
      so the new hire can actually absorb work and reduce lateness.
    - Unlock projects whose squad member goes on leave or is removed, plus
      their transitive dependents.
    """
    from datetime import timedelta
    if not isinstance(body, dict) or body.get("reoptimize") or body.get("freeze_existing") is False:
        return []

    jmeta = {p.id: p for p in (projects or []) if getattr(p, "id", None)}
    late_ids: set[str] = set()
    for a in prev or []:
        proj = jmeta.get(a.project_id)
        if not proj:
            continue
        last_week = int(a.start_week) + max(proj.duration_weeks, 1) - 1
        done_date = today + timedelta(weeks=last_week, days=4)
        if (proj.deadline - done_date).days < 0:
            late_ids.add(a.project_id)

    leave_ids = {x.get("person_id") for x in (body.get("leave") or []) if isinstance(x, dict)}
    slip_ids = {x.get("project_id") for x in (body.get("slip") or []) if isinstance(x, dict)}
    pin_list = [x for x in (body.get("pin") or []) if isinstance(x, dict)]
    pin_ids = {x.get("project_id") for x in pin_list}
    pin_people = {x.get("person_id") for x in pin_list}
    unpin_ids = {x.get("project_id") for x in (body.get("unpin") or []) if isinstance(x, dict)}
    removed_people = set(body.get("remove_person") or [])
    removed_projects = set(body.get("remove_project") or [])
    up = body.get("update_person") if isinstance(body.get("update_person"), dict) else None
    updated_pid = up.get("id") if up else None
    ap = body.get("add_person") if isinstance(body.get("add_person"), dict) else None

    touched_people = leave_ids | removed_people | ({updated_pid} if updated_pid else set())
    affected: set = set(slip_ids) | set(pin_ids) | set(unpin_ids) | set(removed_projects)

    # A hire / skill change only unlocks LATE projects that need a role the
    # person can fill; on-time projects and unrelated roles stay frozen.
    for person in (ap, up):
        skills = (person or {}).get("skills")
        if not isinstance(skills, dict):
            continue
        for jid in late_ids:
            proj = jmeta[jid]
            if any(int(skills.get(r) or 0) >= proj.level_for(r) for r in proj.roles()):
                affected.add(jid)

    for a in prev or []:
        used: list[str] = []
        for pids in a.squad.values():
            used.extend(pids if isinstance(pids, list) else [pids])
        if touched_people & set(used):
            affected.add(a.project_id)
        if (pin_people & set(used)) and a.project_id in late_ids:
            affected.add(a.project_id)

    graph = {p.id: list(p.depends_on or []) for p in (projects or [])}
    changed = True
    while changed:
        changed = False
        for pid, deps in graph.items():
            if pid in affected:
                continue
            if any(d in affected for d in deps):
                affected.add(pid)
                changed = True

    out = []
    for a in prev or []:
        if a.project_id in affected or a.project_id in removed_projects:
            continue
        out.append(a)
    return out


def sanitize_freeze(prev, people: list[Person], projects: list[Project],
                    pins=None, today: date = date(2026, 10, 19)) -> list[Assignment] | None:
    """Return valid frozen assignments from session state so /api/assign
    preserves approved assignments and only schedules unstaffed projects."""
    if not prev:
        return None
    jmeta = {p.id: p for p in projects}
    pidx = {p.id: p for p in people}
    pin_map: dict[tuple[str, str], set[str]] = defaultdict(set)
    for item in pins or []:
        if isinstance(item, (list, tuple)) and len(item) == 3:
            pin_map[(item[0], item[1])].add(item[2])
        elif isinstance(item, dict):
            pin_map[(item.get("project_id"), item.get("role"))].add(item.get("person_id"))

    load: dict[str, dict[int, int]] = defaultdict(lambda: defaultdict(int))
    candidates: dict[str, Assignment] = {}
    for a in prev:
        j = jmeta.get(a.project_id)
        if not j:
            continue
        s = int(a.start_week)
        if s < 0:
            continue
        squad = {
            r: list(pids if isinstance(pids, list) else [pids])
            for r, pids in (a.squad or {}).items()
        }
        valid = True
        for role in j.roles():
            pids = squad.get(role) or []
            if len(pids) != j.count_for(role):
                valid = False
                break
            req_pins = pin_map.get((j.id, role), set())
            if not req_pins.issubset(set(pids)):
                valid = False
                break
            for pid in pids:
                person = pidx.get(pid)
                if (not person or person.level(role) < j.level_for(role)
                        or not role_fits(person, j, role, s, today)):
                    valid = False
                    break
            if not valid:
                break
        if valid:
            candidates[j.id] = Assignment(j.id, s, squad)

    # Check dependency order and capacity
    changed = True
    while changed:
        changed = False
        for jid in list(candidates.keys()):
            j = jmeta[jid]
            a = candidates[jid]
            for dep in j.depends_on or []:
                if dep not in jmeta:
                    continue
                if dep not in candidates or a.start_week < candidates[dep].start_week + jmeta[dep].duration_weeks:
                    del candidates[jid]
                    changed = True
                    break

    kept: list[Assignment] = []
    for a in prev:
        if a.project_id not in candidates:
            continue
        cand = candidates[a.project_id]
        j = jmeta[cand.project_id]
        fits = all(role_fits(pidx[pid], j, role, cand.start_week, today, load)
                   for role, pids in cand.squad.items() for pid in pids)
        if fits:
            committed: dict = defaultdict(lambda: defaultdict(set))
            for role, pids in cand.squad.items():
                for pid in pids:
                    book(load, committed, j, role, pid, cand.start_week)
            kept.append(cand)

    return kept if kept else None

