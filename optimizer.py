"""CP-SAT optimizer — global optimum instead of greedy (OR-Tools).

Model: assign each project a start week and each required role a qualified,
available person, so that no one exceeds capacity and dependencies hold.
Capacity is counted in person-days per week (engine.DAY units): a role books
effort% of a full-time week during its own phase only; a short leave may
leave up to engine.MAX_COVER_DAYS per week to cover, at OVER_W per day.
Deadlines are SOFT: a project may run past its deadline, but lateness is
penalized (LATE_W * priority * weeks-late), so the solver prefers on-time
placement yet still schedules (and flags) a late project instead of silently
dropping it. A project is only dropped when it is skill- or
dependency-infeasible. Returns the same Plan shape as engine.assign.

ortools is imported lazily — the rest of squadflow runs without it.
"""
from __future__ import annotations

import threading
from collections import defaultdict
from datetime import date

from engine import week_of, SLACK_WEEKS, DAY, WEEK, role_load, week_capacity, book, role_fits
from models import Person, Project, Assignment, Plan

# OR-Tools is not safe to run two CP-SAT models on one process at once
# (two judge sessions hitting assign/suggest concurrently).
_SOLVE_LOCK = threading.Lock()

BIG = 10_000        # dropping a project always beats scheduling it
LATE_W = 100        # one week late == this many weeks of start-delay (soft deadline)
BALANCE_W = 2       # per percentage-point of peak util within a role (<< LATE_W)
SKILL_W = 2         # prefer the least-qualified acceptable person
STICKY_W = 120      # cost of swapping a person off a previously-staffed role
TIE_W = 3           # FIFO: earlier projects in the list win exact ties
OVER_W = 8          # per person-day someone must cover for a short leave (<< a week late)


def _nonzero(load: dict) -> dict[str, dict[int, int]]:
    return {pid: {w: u for w, u in weeks.items() if u} for pid, weeks in load.items()}


def has_cp_sat() -> bool:
    try:
        import ortools.sat.python.cp_model  # noqa: F401
        return True
    except ImportError:
        return False


def _why_dropped(j: Project, people: list[Person], projects: list[Project] | None = None,
                 scheduled_val: dict | None = None, start_val: dict | None = None,
                 pins: list[tuple[str, str, str]] | None = None) -> str:
    """Name the real blocker: dependency, infeasible pin, then scarcest role."""
    projects = projects or []
    scheduled_val = scheduled_val or {}
    start_val = start_val or {}
    jmeta = {p.id: p for p in projects}
    pidx = {p.id: p for p in people}

    for dep in j.depends_on:
        depj = jmeta.get(dep)
        name = depj.name if depj else dep
        if scheduled_val.get(dep, 1) == 0:
            return f"waiting on “{name}”, which could not be staffed"

    for jid, role, pid in pins or []:
        if jid != j.id:
            continue
        person = pidx.get(pid)
        who = person.name if person else pid
        if role not in j.roles():
            return f"{who} is pinned to {role}, but “{j.name}” does not need that role"
        if person is not None and person.level(role) < j.level_for(role):
            return (f"{who} ({role} {person.level(role)}) is pinned to “{j.name}” "
                    f"but it needs {role}≥{j.level_for(role)}")
        return (f"{who} is pinned to {role} on “{j.name}” but has no free slot "
                f"(on leave, not started yet, or already booked)")

    ranked: list[tuple] = []
    for role in j.roles():
        lvl = j.level_for(role)
        count = j.count_for(role)
        qualified = [p for p in people if p.level(role) >= lvl]
        names = ", ".join(p.name for p in qualified) if qualified else ""
        if not qualified:
            ranked.append((0, 0, role, lvl, count, names, "none"))
        elif len(qualified) < count:
            ranked.append((1, len(qualified), role, lvl, count, names, "short"))
        else:
            ranked.append((2, len(qualified), role, lvl, count, names, "busy"))
    if not ranked:
        return "not enough people"
    ranked.sort(key=lambda t: (t[0], t[1]))
    _, n, role, lvl, count, names, kind = ranked[0]
    if kind == "none":
        return f"no {role} with level ≥ {lvl} (nobody qualifies)"
    if kind == "short":
        return (f"not enough {role} with level ≥ {lvl}: only {n} "
                f"({names}) but {count} needed")
    return (f"not enough {role} with level ≥ {lvl} ({names}) — "
            f"all busy or on leave, no free slot")


def cp_sat_assign(people: list[Person], projects: list[Project],
                  start_date: date, time_limit_s: float = 15.0,
                  prev_assignments: list[Assignment] | None = None,
                  pins: list[tuple[str, str, str]] | None = None,
                  freeze: list[Assignment] | None = None,
                  keep_squads: bool = False) -> Plan:
    with _SOLVE_LOCK:
        return _cp_sat_assign(people, projects, start_date, time_limit_s,
                              prev_assignments, pins, freeze, keep_squads)


def _cp_sat_assign(people: list[Person], projects: list[Project],
                   start_date: date, time_limit_s: float = 15.0,
                   prev_assignments: list[Assignment] | None = None,
                   pins: list[tuple[str, str, str]] | None = None,
                   freeze: list[Assignment] | None = None,
                   keep_squads: bool = False) -> Plan:
    """keep_squads: previously staffed roles must keep the same people
    (only dates may move) — the 'keep squads' what-if option."""
    from ortools.sat.python import cp_model

    pidx = {p.id: p for p in people}
    jmeta = {j.id: j for j in projects}
    deadline_week = {j.id: week_of(j.deadline, start_date) for j in projects}

    # Freeze is a HARD copy of old assignments: those projects are not variables.
    # Remaining capacity is what is left after the frozen load is subtracted.
    freeze_map: dict[str, Assignment] = {}
    for a in freeze or []:
        if a.project_id in jmeta:
            freeze_map[a.project_id] = a
    frozen_load: dict[str, dict[int, int]] = defaultdict(lambda: defaultdict(int))  # units
    frozen_assignments: list[Assignment] = []
    frozen_committed: dict[str, dict[int, set[str]]] = defaultdict(lambda: defaultdict(set))
    frozen_decisions: list[dict] = []
    freeze_finish: dict[str, int] = {}
    for jid, a in freeze_map.items():
        j = jmeta[jid]
        s = int(a.start_week)
        squad = {
            r: list(pids if isinstance(pids, list) else [pids])
            for r, pids in a.squad.items()
        }
        frozen_assignments.append(Assignment(jid, s, squad))
        freeze_finish[jid] = s + j.duration_weeks
        for role, pids in squad.items():
            lvl = j.level_for(role)
            elig = [p for p in people if p.level(role) >= lvl]
            n_elig = len(elig)
            min_elig = min((p.level(role) for p in elig), default=lvl)
            for pid in pids:
                person = pidx.get(pid)
                chosen_lvl = person.level(role) if person else lvl
                frozen_decisions.append({
                    "project_id": jid, "role": role, "min_level": lvl,
                    "person_id": pid, "chosen_level": chosen_lvl,
                    "start_week": s, "overloaded": False,
                    "n_eligible": n_elig, "n_feasible": n_elig, "solver": "cp_sat",
                    "least_qualified": chosen_lvl == min_elig,
                    "frozen": True,
                })
                book(frozen_load, frozen_committed, j, role, pid, s)

    active = [j for j in projects if j.id not in freeze_map]

    # horizon: room to schedule every project sequentially, even a few weeks late
    max_deadline = max((week_of(j.deadline, start_date) for j in projects), default=0)
    max_dur = max((j.duration_weeks for j in projects), default=1)
    total_dur = sum(j.duration_weeks for j in projects)
    horizon = max(max_deadline + SLACK_WEEKS, max_dur, total_dur) + 1
    for a in freeze_map.values():
        j = jmeta[a.project_id]
        horizon = max(horizon, int(a.start_week) + j.duration_weeks + SLACK_WEEKS)

    if not active:
        return Plan(start_date, horizon, frozen_assignments, [], dict(frozen_committed),
                    frozen_decisions, _nonzero(frozen_load))

    m = cp_model.CpModel()
    wcap = {p.id: [week_capacity(p, start_date, w) for w in range(horizon)] for p in people}

    # ---- variables (unfrozen projects only) ----
    # y[(jid, role, pid, s)] : person pid does role of project jid starting week s
    y: dict = {}
    max_start = {}
    for j in active:
        dur = j.duration_weeks
        max_start[j.id] = horizon - dur
        for r in j.roles():
            lvl = j.level_for(r)
            for p in people:
                if p.level(r) < lvl:
                    continue
                for s in range(0, max_start[j.id] + 1):
                    # static check: leave / not-started weeks alone already rule it out
                    if not role_fits(p, j, r, s, start_date):
                        continue
                    y[(j.id, r, p.id, s)] = m.NewBoolVar(f"y_{j.id}_{r}_{p.id}_{s}")

    scheduled = {j.id: m.NewBoolVar(f"sched_{j.id}") for j in active}
    start_week = {j.id: m.NewIntVar(0, max(max_start[j.id], 0), f"start_{j.id}")
                  for j in active}
    late = {j.id: m.NewIntVar(0, horizon, f"late_{j.id}") for j in active}
    start_at = {}
    for j in active:
        for s in range(0, max_start[j.id] + 1):
            start_at[(j.id, s)] = m.NewBoolVar(f"sa_{j.id}_{s}")

    # ---- project start consistency + lateness (soft deadline) ----
    for j in active:
        for s in range(0, max_start[j.id] + 1):
            m.Add(start_at[(j.id, s)] <= scheduled[j.id])
        m.Add(sum(start_at[(j.id, s)] for s in range(0, max_start[j.id] + 1))
              == scheduled[j.id])
        m.Add(start_week[j.id] == sum(
            s * start_at[(j.id, s)] for s in range(0, max_start[j.id] + 1)))
        # late >= start + duration - deadline_week (0 when on time)
        # last occupied week is start + duration - 1; Friday of that week is still that week
        m.Add(late[j.id] >= start_week[j.id] + j.duration_weeks - 1 - deadline_week[j.id])

    # ---- stage milestones (waterfall): soft, weighed like the deadline ----
    import methodology
    milestone_late = []
    for j in active:
        for stage, due in j.milestones.items():
            end = methodology.stage_end(j.stages, stage)
            if end is None:
                continue
            ml = m.NewIntVar(0, horizon, f"ml_{j.id}_{stage}")
            m.Add(ml >= start_week[j.id] + end - week_of(due, start_date))
            milestone_late.append(j.priority * ml)

    # ---- role filled exactly `count` times per scheduled project ----
    for j in active:
        for r in j.roles():
            count = j.count_for(r)
            terms = [y[k] for k in y if k[0] == j.id and k[1] == r]
            if not terms:
                # no eligible person at all -> project cannot be scheduled
                m.Add(scheduled[j.id] == 0)
                continue
            m.Add(sum(terms) == count * scheduled[j.id])

    # ---- a person does at most one role per project, and only at project's start ----
    for j in active:
        for p in people:
            terms = [y[k] for k in y if k[0] == j.id and k[2] == p.id]
            if terms:
                m.Add(sum(terms) <= 1)
        for s in range(0, max_start[j.id] + 1):
            for r in j.roles():
                for p in people:
                    if (j.id, r, p.id, s) in y:
                        m.Add(y[(j.id, r, p.id, s)] <= start_at[(j.id, s)])

    # ---- capacity HARD (person-days per week) + soft cover for short leave ----
    by_person: dict[str, list[tuple]] = defaultdict(list)
    for (jid, r, pid, s), var in y.items():
        j = jmeta[jid]
        by_person[pid].append((var, role_load(j, r, s)))
    over_days = []
    for p in people:
        for w in range(horizon):
            terms = [wl[w] * var for var, wl in by_person[p.id] if w in wl]
            if not terms:
                continue
            avail, cover = wcap[p.id][w]
            booked = sum(terms) + frozen_load.get(p.id, {}).get(w, 0)
            m.Add(booked <= avail + cover)
            if cover:
                od = m.NewIntVar(0, cover // DAY, f"over_{p.id}_{w}")
                m.Add(DAY * od >= booked - avail)
                over_days.append(od)

    # ---- dependencies (frozen deps pin the earliest start of active projects) ----
    for j in active:
        for dep in j.depends_on:
            if dep in freeze_map:
                m.Add(start_week[j.id] >= freeze_finish[dep]
                      ).OnlyEnforceIf(scheduled[j.id])
            elif dep in scheduled:
                m.Add(scheduled[j.id] <= scheduled[dep])
                m.Add(start_week[j.id] >= start_week[dep] + jmeta[dep].duration_weeks
                      ).OnlyEnforceIf(scheduled[j.id])

    # ---- pins: hard-lock a person onto a project's role ----
    for jid, role, pid in pins or []:
        terms = [var for (jj, r, pp, s), var in y.items()
                 if jj == jid and r == role and pp == pid]
        if jid not in scheduled:
            continue
        if not terms:
            m.Add(scheduled[jid] == 0)
        else:
            m.Add(sum(terms) == scheduled[jid])

    # ---- load (person-weeks, in % of a full-time week) for balance ----
    load = {}
    for p in people:
        terms = [(sum(wl.values()) * 100 // WEEK, var) for var, wl in by_person[p.id]]
        cap_total = sum(a + c for a, c in wcap[p.id]) * 100 // WEEK
        load[p.id] = m.NewIntVar(0, max(cap_total, 1), f"load_{p.id}")
        if terms:
            m.Add(load[p.id] == sum(c * t for c, t in terms))
        else:
            m.Add(load[p.id] == 0)

    # peak utilization (%) per primary role — spreads Eve/Frank/Grace instead of
    # dumping everything on the highest-capacity person.
    peak_util_terms = []
    by_role: dict[str, list[Person]] = defaultdict(list)
    for p in people:
        by_role[p.role].append(p)
    for role, members in by_role.items():
        if len(members) < 2:
            continue
        utils = []
        for p in members:
            # available person-weeks (in %) over the horizon; leave and pre-start excluded
            denom = max(1, sum(a for a, _ in wcap[p.id]) * 100 // WEEK)
            u = m.NewIntVar(0, 400, f"util_{p.id}")
            m.AddDivisionEquality(u, load[p.id] * 100, denom)
            utils.append(u)
        peak = m.NewIntVar(0, 200, f"peak_{role}")
        m.AddMaxEquality(peak, utils)
        peak_util_terms.append(peak)

    # prefer least-qualified-acceptable (small weight, never beats lateness)
    skill_cost = sum(
        pidx[pid].level(r) * var
        for (jid, r, pid, s), var in y.items()
    )

    # sticky: keep previous squad members on the same project/role
    sticky_missing = []
    prev_squad: dict[tuple[str, str], list[str]] = defaultdict(list)
    for a in prev_assignments or []:
        for role, pids in a.squad.items():
            prev_squad[(a.project_id, role)].extend(pids)
    for (jid, role), pids in prev_squad.items():
        if jid not in scheduled:
            continue
        kept_terms = [var for (jj, r, pid, s), var in y.items()
                      if jj == jid and r == role and pid in pids]
        kept = m.NewIntVar(0, max(len(pids), 1), f"kept_{jid}_{role}")
        if kept_terms:
            m.Add(kept == sum(kept_terms))
        else:
            m.Add(kept == 0)
        missing = m.NewIntVar(0, max(len(pids), 1), f"miss_{jid}_{role}")
        m.Add(missing == len(pids) * scheduled[jid] - kept)
        sticky_missing.append(missing)
        if keep_squads:
            m.Add(missing == 0)

    # ---- objective: schedule (esp. high priority), then on-time, then early,
    #      then keep old squads, then spread load, then prefer juniors.
    #      FIFO tie-break: earlier items in the projects list win exact ties. ----
    nproj = max(len(active), 1)
    m.Minimize(
        sum(
            p.priority * (BIG * (1 - scheduled[p.id]) + start_week[p.id] + LATE_W * late[p.id])
            + TIE_W * (nproj - i) * (1 - scheduled[p.id])
            for i, p in enumerate(active)
        )
        + STICKY_W * sum(sticky_missing)
        + LATE_W * sum(milestone_late)
        + OVER_W * sum(over_days)
        + BALANCE_W * sum(peak_util_terms)
        + SKILL_W * skill_cost
    )

    # ---- solve ----
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = time_limit_s
    solver.parameters.num_search_workers = 1  # deterministic
    solver.parameters.random_seed = 0
    status = solver.Solve(m)

    if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        # Keep the frozen plan intact. Drop only the unfrozen projects rather
        # than silently unfreezing everything and reshuffling old squads.
        if freeze_map:
            unstaffable = [
                (j.id, _why_dropped(j, people, projects, {}, {}, pins))
                for j in active
            ]
            return Plan(
                start_date, horizon, frozen_assignments, unstaffable,
                dict(frozen_committed), frozen_decisions,
            )
        if keep_squads:
            # every old squad cannot be kept (e.g. someone is gone for weeks):
            # report that rather than silently swapping people
            return Plan(start_date, horizon, frozen_assignments,
                        [(j.id, "cannot keep the same squad (someone is unavailable)")
                         for j in active],
                        dict(frozen_committed), frozen_decisions, _nonzero(frozen_load))
        from engine import assign
        return assign(people, projects, start_date, freeze=freeze, pins=pins)

    # ---- extract ----
    assignments: list[Assignment] = list(frozen_assignments)
    unstaffable: list[tuple[str, str]] = []
    decisions: list[dict] = list(frozen_decisions)
    committed: dict[str, dict[int, set[str]]] = defaultdict(lambda: defaultdict(set))
    for pid, weeks in frozen_committed.items():
        for w, jobs in weeks.items():
            committed[pid][w].update(jobs)
    booked: dict[str, dict[int, int]] = defaultdict(lambda: defaultdict(int))
    for pid, weeks in frozen_load.items():
        booked[pid].update(weeks)

    scheduled_val = {j.id: solver.Value(scheduled[j.id]) for j in active}
    start_val = {j.id: solver.Value(start_week[j.id]) for j in active}

    for j in active:
        if scheduled_val[j.id] == 0:
            unstaffable.append((j.id, _why_dropped(
                j, people, projects, scheduled_val, start_val, pins,
            )))
            continue
        s = solver.Value(start_week[j.id])
        squad: dict[str, list[str]] = {}
        for r in j.roles():
            lvl = j.level_for(r)
            pids = [p.id for p in people
                    if (j.id, r, p.id, s) in y and solver.Value(y[(j.id, r, p.id, s)]) == 1]
            squad[r] = pids
            elig_levels = [p.level(r) for p in people if p.level(r) >= lvl]
            n_elig = len(elig_levels)
            min_elig = min(elig_levels) if elig_levels else lvl
            for pid in pids:
                chosen_lvl = pidx[pid].level(r)
                decisions.append({
                    "project_id": j.id, "role": r, "min_level": lvl,
                    "person_id": pid, "chosen_level": chosen_lvl,
                    "start_week": s, "overloaded": False,
                    "n_eligible": n_elig, "n_feasible": n_elig, "solver": "cp_sat",
                    "least_qualified": chosen_lvl == min_elig,
                })
                book(booked, committed, j, r, pid, s)
        assignments.append(Assignment(j.id, s, squad))

    return Plan(start_date, horizon, assignments, unstaffable, dict(committed), decisions,
                _nonzero(booked))
