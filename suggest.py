"""Auto-try 2–3 remedies for the current bottleneck and rank them.

Each candidate is a replan payload run on a trial copy. Ranking is
total late-days, then number of unstaffable projects.
"""
from __future__ import annotations

import copy
import time
from datetime import date, timedelta

import pipeline
from analysis import active_weeks
from models import HIRE_LEAD_WEEKS, Person, Project, ROLES


def _role_label(role: str) -> str:
    return role.upper() if role in ("pm", "qa") else role


def _late_days(risks) -> int:
    return sum(r.late_days for r in risks)


def _apply(payload: dict, people: list[Person], projects: list[Project], pins: list) -> None:
    if payload.get("add_person"):
        people.append(Person.from_dict(payload["add_person"]))
    for sl in payload.get("slip") or []:
        proj = next(p for p in projects if p.id == sl["project_id"])
        proj.deadline = date.fromisoformat(sl["deadline"])
    for pin in payload.get("pin") or []:
        t = (pin["project_id"], pin["role"], pin["person_id"])
        if t not in pins:
            pins.append(t)
    up = payload.get("update_person")
    if up:
        person = next(p for p in people if p.id == up["id"])
        person.skills = dict(up["skills"])


def _infer_bottlenecks(people: list[Person], risks) -> list[dict]:
    """If analysis returned no bottlenecks, pick the scarcest role on late work."""
    late = [r for r in (risks or []) if getattr(r, "risk", None) in ("late", "unstaffable")]
    demand: dict[tuple[str, int], int] = {}
    for r in late:
        proj = r.project
        for role in proj.roles():
            key = (role, proj.level_for(role))
            demand[key] = demand.get(key, 0) + 1
    ranked = []
    for (role, lvl), n in demand.items():
        if role not in ROLES:
            continue
        n_qual = sum(1 for p in people if p.level(role) >= lvl)
        ranked.append((n_qual, -n, -lvl, {"role": role, "level": lvl}))
    ranked.sort()
    return [row[3] for row in ranked]


def suggest_fixes(people: list[Person], projects: list[Project], today,
                  prev_assignments=None, pins=None, bottlenecks=None,
                  risks=None) -> list[dict]:
    if not bottlenecks:
        bottlenecks = _infer_bottlenecks(people, risks)
    if not bottlenecks:
        return []
    b = bottlenecks[0]
    role, lvl = b["role"], int(b["level"])
    if role not in ROLES:
        return []
    pins = list(pins or [])
    late_risks = [r for r in (risks or []) if r.risk == "late"]
    needing_late = [r for r in late_risks if role in r.project.roles()]
    specialist = next((p for p in people if p.level(role) >= lvl), None)

    candidates: list[tuple[str, dict]] = []
    hire_id = f"hire_{role}_{int(time.time()) % 100000}"
    hire_start = today + timedelta(weeks=HIRE_LEAD_WEEKS)
    candidates.append((
        f"Hire a {_role_label(role)} at level {lvl} (starts {hire_start:%b} {hire_start.day})",
        {"add_person": {
            "id": hire_id,
            "name": f"New {role.upper()}",
            "role": role,
            "skills": {role: lvl},
            "capacity": 1.0,
            "available_from": hire_start.isoformat(),
        }},
    ))

    # Cheaper than hiring: train the strongest person just below the bar.
    near = [p for p in people if 0 < p.level(role) < lvl]
    if near:
        cand = max(near, key=lambda p: (p.level(role), p.id))
        candidates.append((
            f"Upskill {cand.name} from {_role_label(role)} {cand.level(role)} to {lvl} (training / mentoring)",
            {"update_person": {"id": cand.id, "skills": {**cand.skills, role: lvl}}},
        ))

    if needing_late:
        worst = max(needing_late, key=lambda r: r.late_days)
        new_dl = worst.project.deadline + timedelta(days=14)
        candidates.append((
            f"Move the “{worst.project.name}” deadline by 14 days (→ {new_dl.isoformat()})",
            {"slip": [{"project_id": worst.project.id, "deadline": new_dl.isoformat()}]},
        ))
        top = max(needing_late, key=lambda r: (r.project.priority, r.late_days))
        need_lvl = top.project.level_for(role)
        if (specialist and specialist.level(role) >= need_lvl):
            candidates.append((
                f"Pin {specialist.name} to “{top.project.name}” ({_role_label(role)})",
                {"pin": [{
                    "project_id": top.project.id,
                    "role": role,
                    "person_id": specialist.id,
                }]},
            ))

    ranked: list[dict] = []
    from engine import compatible_freeze
    for title, payload in candidates:
        ppl = copy.deepcopy(people)
        prj = copy.deepcopy(projects)
        pn = list(pins)
        freeze = (
            compatible_freeze(prev_assignments, payload, projects, today=today)
            if prev_assignments else None
        )
        try:
            _apply(payload, ppl, prj, pn)
        except Exception:
            continue
        plan, _, risks2 = pipeline.run(
            ppl, prj, today,
            prev_assignments=prev_assignments, pins=pn,
            freeze=freeze,
            time_limit_s=5.0,
        )
        late = _late_days(risks2)
        n_un = len(plan.unstaffable)
        note = ""
        if payload.get("add_person"):
            # how much of the new hire is actually used — full-time vs short contract
            # only weeks after the start date count: before it the hire cannot work
            first = max(0, (hire_start - today).days // 7)
            window = max(1, active_weeks(plan, prj) - first)
            busy = sum(1 for w in plan.committed.get(hire_id, {}) if first <= w < first + window)
            pct = round(100 * busy / window)
            note = (f"{HIRE_LEAD_WEEKS} weeks to recruit and onboard; then has work "
                    f"{busy}/{window} weeks ({pct}%)")
            if pct < 70:
                note += " — consider a short-term contract / freelancer instead of a full-time hire"
        elif payload.get("update_person"):
            note = "no new headcount, only training time"
        ranked.append({
            "title": title,
            "payload": payload,
            "total_late_days": late,
            "unstaffable": n_un,
            "late_projects": [r.project.name for r in risks2 if r.risk == "late"],
            "note": note,
            "score": late + 1000 * n_un,
        })
    ranked.sort(key=lambda x: (x["score"], x["title"]))
    for i, row in enumerate(ranked, 1):
        row["rank"] = i
    return ranked

