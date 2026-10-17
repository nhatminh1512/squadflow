"""CP-SAT optimizer — global optimum instead of greedy (OR-Tools).

Model: assign each project a start week and each required role a qualified,
available person, so that no one exceeds capacity and dependencies hold.
Deadlines are SOFT: a project may run past its deadline, but lateness is
penalized (LATE_W * priority * weeks-late), so the solver prefers on-time
placement yet still schedules (and flags) a late project instead of silently
dropping it. A project is only dropped when it is skill- or
dependency-infeasible. Returns the same Plan shape as engine.assign.

ortools is imported lazily — the rest of squadflow runs without it.
"""
from __future__ import annotations

import math
import threading
from collections import defaultdict
from datetime import date

from engine import week_of, _unavailable_weeks
from models import Person, Project, Assignment, Plan

# OR-Tools is not safe to run two CP-SAT models on one process at once
# (two judge sessions hitting assign/suggest concurrently).
_SOLVE_LOCK = threading.Lock()

BIG = 10_000        # dropping a project always beats scheduling it
LATE_W = 250        # one week late == this many weeks of start-delay (soft deadline)
BALANCE_W = 2       # per percentage-point of peak util within a role (<< LATE_W)
SKILL_W = 2         # prefer the least-qualified acceptable person
STICKY_W = 120      # cost of swapping a person off a previously-staffed role
TIE_W = 3           # FIFO: earlier projects in the list win exact ties
SLACK_WEEKS = 4     # extra horizon weeks past the latest deadline


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
            return f"đang chờ dự án “{name}” chưa xếp được"

    for jid, role, pid in pins or []:
        if jid != j.id:
            continue
        person = pidx.get(pid)
        who = person.name if person else pid
        if role not in j.roles():
            return f"ghim {who} vào {role} nhưng “{j.name}” không cần vai trò đó"
        if person is not None and person.level(role) < j.level_for(role):
            return (f"ghim {who} ({role} {person.level(role)}) vào “{j.name}” "
                    f"nhưng cần {role}≥{j.level_for(role)}")
        return (f"ghim {who} vào {role} của “{j.name}” nhưng không xếp được lịch "
                f"(nghỉ phép hoặc trùng việc)")

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
        return "thiếu nhân sự"
    ranked.sort(key=lambda t: (t[0], t[1]))
    _, n, role, lvl, count, names, kind = ranked[0]
    if kind == "none":
        return f"thiếu người {role} trình độ ≥ {lvl} (không ai đủ chuẩn)"
    if kind == "short":
        return (f"thiếu người {role} trình độ ≥ {lvl}: chỉ có {n} người "
                f"({names}) trong khi cần {count}")
    return (f"thiếu người {role} trình độ ≥ {lvl} ({names}) — "
            f"đều bận hoặc nghỉ, không đủ slot trống")


def cp_sat_assign(people: list[Person], projects: list[Project],
                  start_date: date, time_limit_s: float = 15.0,
                  prev_assignments: list[Assignment] | None = None,
                  pins: list[tuple[str, str, str]] | None = None,
                  freeze: list[Assignment] | None = None) -> Plan:
    with _SOLVE_LOCK:
        return _cp_sat_assign(people, projects, start_date, time_limit_s,
                              prev_assignments, pins, freeze)


def _cp_sat_assign(people: list[Person], projects: list[Project],
                   start_date: date, time_limit_s: float = 15.0,
                   prev_assignments: list[Assignment] | None = None,
                   pins: list[tuple[str, str, str]] | None = None,
                   freeze: list[Assignment] | None = None) -> Plan:
    from ortools.sat.python import cp_model

    unavailable = {p.id: _unavailable_weeks(p, start_date) for p in people}
    pidx = {p.id: p for p in people}
    jmeta = {j.id: j for j in projects}
    deadline_week = {j.id: week_of(j.deadline, start_date) for j in projects}

    # horizon: room to schedule every project, even a few weeks late
    max_deadline = max((week_of(j.deadline, start_date) for j in projects), default=0)
    max_dur = max((j.duration_weeks for j in projects), default=1)
    horizon = max(max_deadline + SLACK_WEEKS, max_dur) + 1

    m = cp_model.CpModel()

    # ---- variables ----
    # y[(jid, role, pid, s)] : person pid does role of project jid starting week s
    y: dict = {}
    max_start = {}
    for j in projects:
        dur = j.duration_weeks
        max_start[j.id] = horizon - dur
        for r in j.roles():
            lvl = j.level_for(r)
            for p in people:
                if p.level(r) < lvl:
                    continue
                for s in range(0, max_start[j.id] + 1):
                    weeks = range(s, s + dur)
                    if any(w < 0 or w in unavailable[p.id] for w in weeks):
                        continue
                    y[(j.id, r, p.id, s)] = m.NewBoolVar(f"y_{j.id}_{r}_{p.id}_{s}")

    scheduled = {j.id: m.NewBoolVar(f"sched_{j.id}") for j in projects}
    start_week = {j.id: m.NewIntVar(0, max(max_start[j.id], 0), f"start_{j.id}")
                  for j in projects}
    late = {j.id: m.NewIntVar(0, horizon, f"late_{j.id}") for j in projects}
    start_at = {}
    for j in projects:
        for s in range(0, max_start[j.id] + 1):
            start_at[(j.id, s)] = m.NewBoolVar(f"sa_{j.id}_{s}")

    # ---- project start consistency + lateness (soft deadline) ----
    for j in projects:
        for s in range(0, max_start[j.id] + 1):
            m.Add(start_at[(j.id, s)] <= scheduled[j.id])
        m.Add(sum(start_at[(j.id, s)] for s in range(0, max_start[j.id] + 1))
              == scheduled[j.id])
        m.Add(start_week[j.id] == sum(
            s * start_at[(j.id, s)] for s in range(0, max_start[j.id] + 1)))
        # late >= start + duration - deadline_week (0 when on time)
        m.Add(late[j.id] >= start_week[j.id] + j.duration_weeks - deadline_week[j.id])

    # ---- role filled exactly `count` times per scheduled project ----
    for j in projects:
        for r in j.roles():
            count = j.count_for(r)
            terms = [y[k] for k in y if k[0] == j.id and k[1] == r]
            if not terms:
                # no eligible person at all -> project cannot be scheduled
                m.Add(scheduled[j.id] == 0)
                continue
            m.Add(sum(terms) == count * scheduled[j.id])

    # ---- a person does at most one role per project, and only at project's start ----
    for j in projects:
        for p in people:
            terms = [y[k] for k in y if k[0] == j.id and k[2] == p.id]
            if terms:
                m.Add(sum(terms) <= 1)
        for s in range(0, max_start[j.id] + 1):
            for r in j.roles():
                for p in people:
                    if (j.id, r, p.id, s) in y:
                        m.Add(y[(j.id, r, p.id, s)] <= start_at[(j.id, s)])

    # ---- capacity: concurrent load <= capacity per person per week ----
    cap = {p.id: max(1, math.ceil(p.capacity)) for p in people}
    for p in people:
        for w in range(horizon):
            terms = []
            for (jid, r, pid, s), var in y.items():
                if pid == p.id and s <= w < s + jmeta[jid].duration_weeks:
                    terms.append(var)
            if terms:
                m.Add(sum(terms) <= cap[p.id])

    # ---- dependencies ----
    for j in projects:
        for dep in j.depends_on:
            if dep in scheduled:
                m.Add(scheduled[j.id] <= scheduled[dep])
                # only enforce the start-ordering when j is actually scheduled
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

    # ---- freeze: keep old projects at the exact start week and squad ----
    for a in freeze or []:
        jid = a.project_id
        if jid not in scheduled:
            continue
        s = int(a.start_week)
        if (jid, s) not in start_at:
            continue
        m.Add(scheduled[jid] == 1)
        m.Add(start_at[(jid, s)] == 1)
        for role, pids in a.squad.items():
            arr = pids if isinstance(pids, list) else [pids]
            for pid in arr:
                key = (jid, role, pid, s)
                if key in y:
                    m.Add(y[key] == 1)

    # ---- load (person-weeks) for balance + skill preference ----
    load = {}
    for p in people:
        terms = []
        coeffs = []
        for (jid, r, pid, s), var in y.items():
            if pid == p.id:
                terms.append(var)
                coeffs.append(jmeta[jid].duration_weeks)
        load[p.id] = m.NewIntVar(0, horizon * cap[p.id], f"load_{p.id}")
        if terms:
            m.Add(load[p.id] == sum(c * t for c, t in zip(coeffs, terms)))
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
            off = len(unavailable[p.id] & set(range(horizon)))
            avail = max(1, horizon - off)
            denom = max(1, cap[p.id] * avail)
            u = m.NewIntVar(0, 200, f"util_{p.id}")
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

    # ---- objective: schedule (esp. high priority), then on-time, then early,
    #      then keep old squads, then spread load, then prefer juniors.
    #      FIFO tie-break: earlier items in the projects list win exact ties. ----
    nproj = max(len(projects), 1)
    m.Minimize(
        sum(
            p.priority * (BIG * (1 - scheduled[p.id]) + start_week[p.id] + LATE_W * late[p.id])
            + TIE_W * (nproj - i) * (1 - scheduled[p.id])
            for i, p in enumerate(projects)
        )
        + STICKY_W * sum(sticky_missing)
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
        # fall back to greedy on any failure
        from engine import assign
        return assign(people, projects, start_date)

    # ---- extract ----
    assignments: list[Assignment] = []
    unstaffable: list[tuple[str, str]] = []
    decisions: list[dict] = []
    committed: dict[str, dict[int, set[str]]] = defaultdict(lambda: defaultdict(set))

    scheduled_val = {j.id: solver.Value(scheduled[j.id]) for j in projects}
    start_val = {j.id: solver.Value(start_week[j.id]) for j in projects}

    for j in projects:
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
                for w in range(s, s + j.duration_weeks):
                    committed[pid][w].add(j.id)
        assignments.append(Assignment(j.id, s, squad))

    return Plan(start_date, horizon, assignments, unstaffable, dict(committed), decisions)
