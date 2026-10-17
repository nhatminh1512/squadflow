"""Auto-try 2–3 remedies for the current bottleneck and rank them.

Each candidate is a replan payload run on a trial copy. Ranking is
total late-days, then number of unstaffable projects.
"""
from __future__ import annotations

import copy
import time
from datetime import date, timedelta

import pipeline
from models import Person, Project, ROLES


def _late_days(risks) -> int:
    return sum(max(0, -(r.days_left or 0)) for r in risks)


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


def suggest_fixes(people: list[Person], projects: list[Project], today,
                  prev_assignments=None, pins=None, bottlenecks=None,
                  risks=None) -> list[dict]:
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
    candidates.append((
        f"Thuê thêm {role} trình độ {lvl}",
        {"add_person": {
            "id": hire_id,
            "name": f"Người {role.upper()} mới",
            "role": role,
            "skills": {role: lvl},
            "capacity": 1.0,
        }},
    ))

    if needing_late:
        worst = min(needing_late, key=lambda r: r.days_left if r.days_left is not None else 0)
        new_dl = worst.project.deadline + timedelta(days=14)
        candidates.append((
            f"Dời hạn “{worst.project.name}” thêm 14 ngày (→ {new_dl.isoformat()})",
            {"slip": [{"project_id": worst.project.id, "deadline": new_dl.isoformat()}]},
        ))
        top = max(needing_late, key=lambda r: (r.project.priority, -(r.days_left or 0)))
        need_lvl = top.project.level_for(role)
        if (specialist and specialist.level(role) >= need_lvl):
            candidates.append((
                f"Ghim {specialist.name} vào “{top.project.name}” ({role})",
                {"pin": [{
                    "project_id": top.project.id,
                    "role": role,
                    "person_id": specialist.id,
                }]},
            ))

    ranked: list[dict] = []
    for title, payload in candidates[:3]:
        ppl = copy.deepcopy(people)
        prj = copy.deepcopy(projects)
        pn = list(pins)
        try:
            _apply(payload, ppl, prj, pn)
        except Exception:
            continue
        plan, _, risks2 = pipeline.run(
            ppl, prj, today,
            prev_assignments=prev_assignments, pins=pn,
            time_limit_s=5.0,
        )
        late = _late_days(risks2)
        n_un = len(plan.unstaffable)
        ranked.append({
            "title": title,
            "payload": payload,
            "total_late_days": late,
            "unstaffable": n_un,
            "late_projects": [r.project.name for r in risks2 if r.risk == "late"],
            "score": late + 1000 * n_un,
        })
    ranked.sort(key=lambda x: (x["score"], x["title"]))
    for i, row in enumerate(ranked, 1):
        row["rank"] = i
    return ranked
