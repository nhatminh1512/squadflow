"""Explanation — render grounded, human-readable reasons from the solver log.

The FACTS come from engine.decisions (recorded at solve time), so any LLM
wrapping this can phrase them freely but cannot invent the underlying reason:
who was chosen, their level, how many qualified, and whether it was overload.
"""
from __future__ import annotations

from analysis import ProjectRisk
from models import Person, Plan, Project


def explain_decision(plan: Plan, people: list[Person], meta: dict) -> str:
    name = {p.id: p.name for p in people}
    role = meta["role"]
    min_level = meta["min_level"]
    person_name = name.get(meta["person_id"], meta["person_id"])
    level = meta["chosen_level"]
    n_eligible = meta["n_eligible"]
    if meta.get("overloaded"):
        return (f"{person_name} ({role} {level}) — OVERLOADED: nobody else with "
                f"{role}≥{min_level} is free")
    if n_eligible == 1:
        return f"{person_name} ({role} {level}) — the only person with {role}≥{min_level}"
    if meta.get("least_qualified") or (
        meta.get("solver") != "cp_sat" and not meta.get("overloaded")
    ):
        return (f"{person_name} ({role} {level}) — qualified; least senior of "
                f"{n_eligible} candidates (seniors kept for harder work)")
    return (f"{person_name} ({role} {level}) — has {role}≥{min_level}; "
            f"picked by the global optimizer among {n_eligible} candidates "
            f"(load balance / keep previous squad / deadlines)")


EMPTY_PLAN_EXPLANATION = (
    "There are no projects in the plan to assign yet. "
    "Add a project, or press Reset to restore the sample data."
)


def explain_plan(plan: Plan, people: list[Person], projects: list[Project],
                 risks: list[ProjectRisk] | None = None) -> str:
    if not projects or (not plan.assignments and not plan.unstaffable):
        return EMPTY_PLAN_EXPLANATION

    pname = {p.id: p.name for p in projects}
    lines: list[str] = []

    by_project: dict[str, list[dict]] = {}
    for d in plan.decisions:
        by_project.setdefault(d["project_id"], []).append(d)

    late = [r for r in (risks or []) if r.risk == "late"]
    if late:
        lines.append("LATE PROJECTS:")
        for r in late:
            deadline = r.project.deadline.isoformat()
            if r.days_left is not None and r.days_left < 0:
                lines.append(f"  • {r.project.name} — LATE by {r.late_days} days "
                             f"(done {r.done_date}, deadline {deadline})")
            else:
                lines.append(f"  • {r.project.name} — LATE by {r.late_days} days "
                             f"(deadline {deadline} met, but a stage date is missed)")
            for m in r.milestones or []:
                if (m.get("days_left") or 0) < 0:
                    lines.append(f"    stage “{m['stage']}” due {m['due']} "
                                 f"(planned {m['planned']}) — late by {-m['days_left']} days")
        lines.append("")

    for a in sorted(plan.assignments, key=lambda x: x.start_week):
        lines.append(f"**{pname.get(a.project_id, a.project_id)}** (week {a.start_week}):")
        for d in by_project.get(a.project_id, []):
            lines.append(f"  • {explain_decision(plan, people, d)}")

    for pid, reason in plan.unstaffable:
        lines.append(f"**{pname.get(pid, pid)}** — UNSTAFFABLE: {reason}")

    return "\n".join(lines) if lines else EMPTY_PLAN_EXPLANATION


def explain_plan_llm(plan: Plan, people: list[Person], projects: list[Project],
                     risks: list[ProjectRisk] | None = None) -> str:
    """LLM rephrases the grounded solver log into natural language.

    The facts come from explain_plan (the solver log); the LLM only phrases,
    so it cannot invent a reason that isn't there.
    """
    if not projects or (not plan.assignments and not plan.unstaffable):
        return EMPTY_PLAN_EXPLANATION
    import llm
    facts = explain_plan(plan, people, projects, risks)
    if not facts.strip() or facts == EMPTY_PLAN_EXPLANATION:
        return EMPTY_PLAN_EXPLANATION
    user = (
        "Below is the scheduler's decision log (real facts). "
        "Rewrite it as a short, natural explanation in English. "
        "Use only the facts in the log; do not invent anything. "
        "If the log has a LATE PROJECTS section, name those projects and how many days late. "
        "If a late stage date is listed under a project, name that stage and its due date. "
        "Never say 'nobody is overloaded' or 'everything is on time' if the log lists a LATE project. "
        "Never say someone was picked as 'least senior' unless the log says exactly that.\n\n"
        + facts
    )
    try:
        text = llm.complete([
            {"role": "system", "content": "You explain staffing plans. Be brief and precise, and never make things up."},
            {"role": "user", "content": user},
        ], timeout=20, fast=True, max_tokens=800)
        if text and str(text).strip():
            return text
    except Exception:
        pass
    return facts

