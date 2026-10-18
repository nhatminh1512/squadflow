"""Report — the 'final response' of the pipeline. Renders assignments,
utilization, crunch/idle flags, and deadline risk as text.
"""
from __future__ import annotations

from datetime import timedelta

from analysis import PersonStats, ProjectRisk, find_bottlenecks
from models import Person, Plan, Project


def _d(plan: Plan, week: int) -> str:
    return (plan.start_date + timedelta(weeks=week)).isoformat()


def render(plan: Plan, people: list[Person], person_stats: list[PersonStats],
           risks: list[ProjectRisk], projects: list[Project] | None = None) -> str:
    name = {p.id: p.name for p in people}
    pname = {p.id: p.name for p in (projects or [])}
    def jn(pid: str) -> str:
        return pname.get(pid, pid)
    lines: list[str] = []
    lines.append(f"Horizon: {plan.start_date} .. {plan.end_date()}  ({plan.horizon_weeks} weeks)")
    lines.append("")

    # ---- assignments ----
    lines.append("ASSIGNMENTS")
    lines.append("-" * 78)
    for a in sorted(plan.assignments, key=lambda x: x.start_week):
        squad = ", ".join(
            f"{role}:{', '.join(name.get(pid, pid) for pid in pids)}"
            for role, pids in a.squad.items()
        )
        lines.append(f"{jn(a.project_id):<28} wk{a.start_week:>2} [{_d(plan, a.start_week)}]  {squad}")
    if not plan.assignments:
        lines.append("(no projects staffed)")

    # ---- unstaffable / at-risk ----
    if plan.unstaffable:
        lines.append("")
        lines.append("UNSTAFFABLE")
        lines.append("-" * 78)
        for pid, reason in plan.unstaffable:
            lines.append(f"  {jn(pid):<28} {reason}")

    # ---- utilization ----
    lines.append("")
    lines.append("UTILIZATION")
    lines.append("-" * 78)
    lines.append(f"{'person':<18}{'role':<10}{'booked':>7}{'avail':>7}{'util':>7}  flag")
    for s in sorted(person_stats, key=lambda x: (-x.util, -x.booked_weeks, x.person.id)):
        flags = []
        if s.overloaded:
            flags.append("OVERLOADED")
        if s.idle:
            flags.append("IDLE")
        lines.append(
            f"{s.person.name:<18}{s.person.role:<10}"
            f"{s.booked_weeks:>6}w{s.available_weeks:>6}w{s.util:>6.0%}  {', '.join(flags) or '-'}"
        )

    # ---- bottleneck roles (scarce skills causing late / dropped work) ----
    if projects:
        bottlenecks = find_bottlenecks(plan, people, projects, risks, person_stats)
        if bottlenecks:
            lines.append("")
            lines.append("BOTTLENECKS (roles with unmet demand)")
            lines.append("-" * 78)
            for b in bottlenecks:
                lines.append(f"  {b['role']} >= {b['level']}: {b.get('detail') or 'fully booked / short of people'}")

    # ---- deadline risk ----
    lines.append("")
    lines.append("DEADLINE RISK")
    lines.append("-" * 78)
    for r in sorted(risks, key=lambda x: x.project.deadline):
        if r.risk == "ok":
            continue
        note = r.risk
        if r.days_left is not None:
            note += f" ({r.days_left} days left)"
        lines.append(f"  {r.project.name:<28} due {r.project.deadline}  -> {note}")

    lines.append("")
    return "\n".join(lines)
