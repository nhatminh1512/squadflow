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
        return (f"{person_name} ({role} {level}) — QUÁ TẢI: không còn ai đủ "
                f"{role}≥{min_level} trống lịch")
    if n_eligible == 1:
        return f"{person_name} ({role} {level}) — người duy nhất đủ {role}≥{min_level}"
    if meta.get("least_qualified") or (
        meta.get("solver") != "cp_sat" and not meta.get("overloaded")
    ):
        return (f"{person_name} ({role} {level}) — đủ chuẩn, chọn người ít kinh nghiệm "
                f"nhất trong {n_eligible} người (dành senior cho việc khó)")
    return (f"{person_name} ({role} {level}) — đủ {role}≥{min_level}, "
            f"chọn bởi tối ưu toàn cục trong {n_eligible} người "
            f"(cân tải / giữ phân công cũ / hạn chót)")


def explain_plan(plan: Plan, people: list[Person], projects: list[Project],
                 risks: list[ProjectRisk] | None = None) -> str:
    pname = {p.id: p.name for p in projects}
    lines: list[str] = []

    by_project: dict[str, list[dict]] = {}
    for d in plan.decisions:
        by_project.setdefault(d["project_id"], []).append(d)

    late = [r for r in (risks or []) if r.risk == "late"]
    if late:
        lines.append("DỰ ÁN TRỄ HẠN:")
        for r in late:
            days = r.days_left if r.days_left is not None else 0
            lines.append(f"  • {r.project.name} — TRỄ {abs(days)} ngày "
                         f"(xong {r.done_date}, hạn {r.project.deadline.isoformat()})")
        lines.append("")

    for a in sorted(plan.assignments, key=lambda x: x.start_week):
        lines.append(f"**{pname.get(a.project_id, a.project_id)}** (tuần {a.start_week}):")
        for d in by_project.get(a.project_id, []):
            lines.append(f"  • {explain_decision(plan, people, d)}")

    for pid, reason in plan.unstaffable:
        lines.append(f"**{pname.get(pid, pid)}** — KHÔNG XẾP ĐƯỢC: {reason}")

    return "\n".join(lines)


def explain_plan_llm(plan: Plan, people: list[Person], projects: list[Project],
                     risks: list[ProjectRisk] | None = None) -> str:
    """LLM rephrases the grounded solver log into natural language.

    The facts come from explain_plan (the solver log); the LLM only phrases,
    so it cannot invent a reason that isn't there.
    """
    import llm
    facts = explain_plan(plan, people, projects, risks)
    user = (
        "Dưới đây là log quyết định của bộ xếp lịch (dữ kiện có thật). "
        "Viết lại thành một đoạn giải thích tự nhiên, ngắn gọn, bằng tiếng Việt. "
        "Chỉ dùng đúng dữ kiện trong log, không bịa thêm. "
        "Nếu log có mục DỰ ÁN TRỄ HẠN thì phải nêu tên các dự án đó và số ngày trễ. "
        "Không được nói 'đều không quá tải' hay 'mọi dự án đúng hạn' nếu log ghi có dự án TRỄ. "
        "Không được nói ai được chọn vì 'ít kinh nghiệm nhất' trừ khi log ghi đúng cụm đó.\n\n"
        + facts
    )
    return llm.complete([
        {"role": "system", "content": "Bạn là trợ lý giải thích kế hoạch phân công nhân sự, nói ngắn gọn, chính xác, không bịa."},
        {"role": "user", "content": user},
    ], timeout=40)
