"""Web UI — a thin HTTP layer over the same engine the CLI uses.

Stdlib-only (http.server), so it runs anywhere with `python3 web.py` and
serves the single-page UI plus a JSON API. The pipeline underneath is
unchanged: this is the "web dashboard" surface in front of the same core.
"""
from __future__ import annotations

import copy
import json
import math
import mimetypes
import re
import sys
import threading
import time
import traceback
import uuid
from datetime import date
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

import analysis
import forecast
import llm
import pipeline
import suggest
from explain import explain_plan_llm
from models import Person, Project, ROLES

ROOT = Path(__file__).resolve().parent
TODAY: date = date(2026, 10, 19)

# Per-browser sessions so two judges don't overwrite each other.
SESSIONS: dict[str, dict] = {}
SESS_LOCK = threading.Lock()
SESSION_TTL_S = 6 * 3600


def _fresh_state() -> dict:
    return {
        "people": pipeline.load_people(ROOT / "data" / "people.json"),
        "projects": pipeline.load_projects(ROOT / "data" / "projects.json"),
        "pins": [],
        "prev_assignments": [],
        "touched": time.time(),
        "lock": threading.RLock(),
    }


def _get_session(sid: str | None) -> tuple[str, dict, bool]:
    now = time.time()
    with SESS_LOCK:
        dead = [k for k, v in SESSIONS.items() if now - v.get("touched", 0) > SESSION_TTL_S]
        for k in dead:
            SESSIONS.pop(k, None)
        if sid and sid in SESSIONS:
            SESSIONS[sid]["touched"] = now
            return sid, SESSIONS[sid], False
        sid = uuid.uuid4().hex
        SESSIONS[sid] = _fresh_state()
        return sid, SESSIONS[sid], True


# Rate limiting — intake and explain are separate buckets so one doesn't starve the other.
_llm_lock = threading.Lock()
_llm_times: dict[str, list[float]] = {"intake": [], "explain": []}
_LLM_WINDOW = {"intake": 20.0, "explain": 15.0}
_LLM_MAX = {"intake": 6, "explain": 4}


def _llm_allowed(kind: str = "intake") -> bool:
    now = time.time()
    window = _LLM_WINDOW.get(kind, 12.0)
    cap = _LLM_MAX.get(kind, 4)
    with _llm_lock:
        bucket = _llm_times.setdefault(kind, [])
        bucket[:] = [t for t in bucket if now - t < window]
        if len(bucket) >= cap:
            return False
        bucket.append(now)
        return True


def _iso(d: date) -> str:
    return d.isoformat()


def _parse_int(v, field: str, lo: int | None = None, hi: int | None = None) -> int:
    """Strict integer parse — rejects 3.5, bools, and junk without leaking Python errors."""
    if isinstance(v, bool) or v is None:
        raise ValueError(f"{field} phải là số nguyên")
    if isinstance(v, float):
        if not v.is_integer():
            raise ValueError(f"{field} phải là số nguyên, không nhận {v}")
        v = int(v)
    elif isinstance(v, str):
        s = v.strip()
        if not re.fullmatch(r"-?\d+", s):
            raise ValueError(f"{field} phải là số nguyên")
        v = int(s)
    elif not isinstance(v, int):
        raise ValueError(f"{field} phải là số nguyên")
    if lo is not None and v < lo:
        raise ValueError(f"{field} phải ≥ {lo}")
    if hi is not None and v > hi:
        raise ValueError(f"{field} phải ≤ {hi}")
    return v


def _parse_date(v, field: str) -> date:
    try:
        return date.fromisoformat(str(v))
    except (TypeError, ValueError):
        raise ValueError(f"{field} sai định dạng (cần YYYY-MM-DD)")


def build_result(people: list[Person], projects: list[Project],
                 prev_assignments=None, pins=None, freeze=None) -> dict:
    pins = list(pins or [])
    plan, pstats, risks = pipeline.run(
        people, projects, TODAY,
        prev_assignments=None if freeze else (prev_assignments or None),
        pins=pins,
        freeze=freeze,
    )
    name = {p.id: p.name for p in people}
    pmeta = {p.id: p for p in projects}

    from datetime import timedelta
    week_dates = [plan.start_date + timedelta(weeks=w) for w in range(plan.horizon_weeks)]

    bottlenecks = analysis.find_bottlenecks(plan, people, projects, risks, pstats)
    pin_set = {(a, b, c) for a, b, c in pins}

    return {
        "today": _iso(TODAY),
        "horizon_weeks": plan.horizon_weeks,
        "week_dates": [_iso(d) for d in week_dates],
        "people": [
            {"id": p.id, "name": p.name, "role": p.role,
             "skills": p.skills, "capacity": p.capacity,
             "leave": [[_iso(a), _iso(b)] for a, b in p.leave]}
            for p in people
        ],
        "projects": [
            {"id": p.id, "name": p.name, "required": p.required,
             "duration_weeks": p.duration_weeks, "deadline": _iso(p.deadline),
             "priority": p.priority, "depends_on": p.depends_on}
            for p in projects
        ],
        "assignments": [
            {"project_id": a.project_id, "start_week": a.start_week,
             "end_week": a.start_week + pmeta[a.project_id].duration_weeks,
             "name": pmeta[a.project_id].name, "squad": a.squad}
            for a in plan.assignments
        ],
        "unstaffable": [
            {"project_id": pid, "name": pmeta[pid].name if pid in pmeta else pid, "reason": reason}
            for pid, reason in plan.unstaffable
        ],
        "utilization": [
            {"id": s.person.id, "name": s.person.name, "role": s.person.role,
             "booked": s.booked_weeks, "available": s.available_weeks,
             "slots": max(1, math.ceil(s.person.capacity)) * s.available_weeks,
             "peak": s.peak_load, "util": round(s.util, 3),
             "idle": s.idle, "overloaded": s.overloaded}
            for s in pstats
        ],
        "bottlenecks": bottlenecks,
        "risks": [
            {"project_id": r.project.id, "name": r.project.name, "risk": r.risk,
             "start_week": r.start_week, "done_date": r.done_date,
             "days_left": r.days_left, "deadline": _iso(r.project.deadline)}
            for r in risks
        ],
        "committed": {
            pid: {str(w): sorted(s) for w, s in weeks.items()}
            for pid, weeks in plan.committed.items()
        },
        "decisions": [
            {**d, "person_name": name.get(d["person_id"], d["person_id"])}
            for d in plan.decisions
        ],
        "miss_prob": forecast.monte_carlo(plan, projects),
        "pins": [
            {"project_id": jid, "role": role, "person_id": pid,
             "person_name": name.get(pid, pid),
             "project_name": pmeta[jid].name if jid in pmeta else jid}
            for jid, role, pid in sorted(pin_set)
        ],
        "_plan": plan,  # stripped before sending JSON
    }


def _public(result: dict) -> dict:
    out = dict(result)
    out.pop("_plan", None)
    return out


def _apply_changes_to(body: dict, people: list[Person], projects: list[Project],
                      pins: list) -> None:
    for lv in body.get("leave", []) or []:
        person = next(p for p in people if p.id == lv["person_id"])
        person.leave.append((_parse_date(lv["from"], "ngày bắt đầu"),
                             _parse_date(lv["to"], "ngày kết thúc")))

    for sl in body.get("slip", []) or []:
        proj = next(p for p in projects if p.id == sl["project_id"])
        proj.deadline = _parse_date(sl["deadline"], "deadline")

    if body.get("add_person"):
        people.append(Person.from_dict(body["add_person"]))

    if body.get("add_project"):
        projects.append(Project.from_dict(body["add_project"]))
    for aj in body.get("add_projects") or []:
        projects.append(Project.from_dict(aj))

    for pid in body.get("remove_project", []) or []:
        projects[:] = [p for p in projects if p.id != pid]

    for pid in body.get("remove_person", []) or []:
        people[:] = [p for p in people if p.id != pid]
        pins[:] = [t for t in pins if t[2] != pid]

    up = body.get("update_person")
    if up:
        person = next(p for p in people if p.id == up["id"])
        if "name" in up and up["name"]:
            person.name = str(up["name"]).strip()
        if "role" in up and up["role"] in ROLES:
            person.role = up["role"]
        if "skills" in up and isinstance(up["skills"], dict):
            person.skills = {k: _parse_int(v, f"trình độ {k}", 1, 5) for k, v in up["skills"].items()}
        if "capacity" in up:
            cap = float(up["capacity"])
            _check_capacity(cap)
            person.capacity = cap

    for pin in body.get("pin", []) or []:
        t = (pin["project_id"], pin["role"], pin["person_id"])
        if t not in pins:
            pins.append(t)

    for pin in body.get("unpin", []) or []:
        t = (pin["project_id"], pin["role"], pin["person_id"])
        pins[:] = [x for x in pins if x != t]


def _check_person(d: dict) -> None:
    if not (d.get("id") and d.get("name")):
        raise ValueError("thiếu id hoặc tên")
    if d.get("role") not in ROLES:
        raise ValueError(f"vai trò không hợp lệ: {d.get('role')}")
    skills = d.get("skills") or {}
    if not isinstance(skills, dict):
        raise ValueError("kỹ năng phải là đối tượng role→trình độ")
    for k, v in skills.items():
        if k not in ROLES:
            raise ValueError(f"kỹ năng không hợp lệ: {k}")
        _parse_int(v, f"trình độ '{k}'", 1, 5)
    try:
        cap = float(d.get("capacity", 1.0))
    except (TypeError, ValueError):
        raise ValueError("sức chứa phải là số")
    _check_capacity(cap)
    for pair in d.get("leave", []) or []:
        if not (isinstance(pair, (list, tuple)) and len(pair) == 2):
            raise ValueError("lịch nghỉ phải là [từ, đến]")
        a = _parse_date(pair[0], "ngày bắt đầu nghỉ")
        b = _parse_date(pair[1], "ngày kết thúc nghỉ")
        if a > b:
            raise ValueError("ngày bắt đầu nghỉ phải trước hoặc bằng ngày kết thúc")
        if a < TODAY:
            raise ValueError("ngày bắt đầu nghỉ không được trước hôm nay")


def _check_capacity(cap: float) -> None:
    if cap > 10:
        raise ValueError(
            "sức chứa là số dự án làm song song (1 hoặc 2), không phải phần trăm — thử 1"
        )
    if not (0.5 <= cap <= 4):
        raise ValueError("sức chứa phải trong 0.5..4 (số dự án làm cùng lúc)")


def _check_project(d: dict) -> None:
    if not (d.get("id") and d.get("name")):
        raise ValueError("thiếu id hoặc tên")
    required = d.get("required") or {}
    if not required:
        raise ValueError("thiếu yêu cầu nhân sự (role)")
    if not isinstance(required, dict):
        raise ValueError("required phải là đối tượng role→trình độ")
    for role, spec in required.items():
        if role not in ROLES:
            raise ValueError(f"vai trò không hợp lệ: {role}")
        level = spec.get("level") if isinstance(spec, dict) else spec
        _parse_int(level, f"trình độ '{role}'", 1, 5)
        if isinstance(spec, dict):
            _parse_int(spec.get("count", 1), f"số lượng '{role}'", 1, 10)
    _parse_int(d.get("duration_weeks", 0), "thời lượng", 1, 52)
    deadline = _parse_date(d.get("deadline"), "deadline")
    if deadline < TODAY:
        raise ValueError("deadline đã qua (phải từ hôm nay trở đi)")
    _parse_int(d.get("priority", 3), "priority", 1, 5)


def _validate_changes(body: dict, people: list[Person], projects: list[Project]) -> list[str]:
    errors: list[str] = []
    if not isinstance(body, dict):
        return ["body phải là JSON object (không phải danh sách/chuỗi)"]
    pids = {p.id for p in people}
    jids = {p.id for p in projects}

    for key in ("leave", "slip", "remove_project", "remove_person", "pin", "unpin"):
        v = body.get(key, [])
        if v is None:
            continue
        if not isinstance(v, list):
            errors.append(f"{key} phải là một danh sách")

    for lv in body.get("leave", []) if isinstance(body.get("leave", []), list) else []:
        if not isinstance(lv, dict):
            errors.append("nghỉ phép: mỗi mục phải là đối tượng")
            continue
        if lv.get("person_id") not in pids:
            errors.append("nghỉ phép: không tìm thấy người đó")
        else:
            try:
                a = _parse_date(lv.get("from"), "ngày bắt đầu nghỉ")
                b = _parse_date(lv.get("to"), "ngày kết thúc nghỉ")
                if a > b:
                    errors.append("nghỉ phép: ngày bắt đầu phải trước hoặc bằng ngày kết thúc")
                elif a < TODAY:
                    errors.append("nghỉ phép: ngày bắt đầu không được trước hôm nay")
            except ValueError as e:
                errors.append(f"nghỉ phép: {e}")

    for sl in body.get("slip", []) if isinstance(body.get("slip", []), list) else []:
        if not isinstance(sl, dict):
            errors.append("dời hạn: mỗi mục phải là đối tượng")
            continue
        if sl.get("project_id") not in jids:
            errors.append("dời hạn: không tìm thấy dự án đó")
        else:
            try:
                dl = _parse_date(sl.get("deadline"), "deadline")
                if dl < TODAY:
                    errors.append("dời hạn: deadline đã qua (phải từ hôm nay trở đi)")
            except ValueError as e:
                errors.append(f"dời hạn: {e}")

    ap = body.get("add_person")
    if ap:
        if not isinstance(ap, dict):
            errors.append("thêm người: dữ liệu không hợp lệ")
        elif ap.get("id") in pids:
            errors.append("thêm người: id đã tồn tại")
        else:
            try:
                _check_person(ap)
            except ValueError as e:
                errors.append(f"thêm người: {e}")

    incoming = []
    if body.get("add_project"):
        incoming.append(body["add_project"])
    if isinstance(body.get("add_projects"), list):
        incoming.extend(body["add_projects"])
    seen_new = set()
    for aj in incoming:
        if not isinstance(aj, dict):
            errors.append("thêm dự án: dữ liệu không hợp lệ")
            continue
        pid = aj.get("id")
        if not pid:
            errors.append("thêm dự án: thiếu id")
            continue
        if pid in jids or pid in seen_new:
            errors.append(f"thêm dự án: id đã tồn tại ({pid})")
            continue
        seen_new.add(pid)
        try:
            _check_project(aj)
        except ValueError as e:
            errors.append(f"thêm dự án “{aj.get('name') or pid}”: {e}")

    up = body.get("update_person")
    if up:
        if not isinstance(up, dict) or up.get("id") not in pids:
            errors.append("sửa người: không tìm thấy người đó")
        else:
            try:
                if "role" in up and up["role"] not in ROLES:
                    raise ValueError(f"vai trò không hợp lệ: {up['role']}")
                if "skills" in up:
                    if not isinstance(up["skills"], dict):
                        raise ValueError("kỹ năng phải là đối tượng")
                    for k, v in up["skills"].items():
                        if k not in ROLES:
                            raise ValueError(f"kỹ năng không hợp lệ: {k}")
                        _parse_int(v, f"trình độ '{k}'", 1, 5)
                if "capacity" in up:
                    cap = float(up["capacity"])
                    _check_capacity(cap)
            except (TypeError, ValueError) as e:
                msg = str(e)
                if "invalid literal" in msg or "could not convert" in msg:
                    msg = "giá trị không hợp lệ (cần số nguyên / số dương)"
                errors.append(f"sửa người: {msg}")

    for pid in body.get("remove_person", []) if isinstance(body.get("remove_person", []), list) else []:
        if pid not in pids:
            errors.append("xóa người: không tìm thấy người đó")

    for key in ("pin", "unpin"):
        for pin in body.get(key, []) if isinstance(body.get(key, []), list) else []:
            if not isinstance(pin, dict):
                errors.append("ghim: dữ liệu không hợp lệ")
                continue
            if pin.get("person_id") not in pids:
                errors.append("ghim: không tìm thấy người đó")
            if pin.get("project_id") not in jids:
                errors.append("ghim: không tìm thấy dự án đó")
            if pin.get("role") not in ROLES:
                errors.append("ghim: vai trò không hợp lệ")
                continue
            if key == "pin":
                person = next((p for p in people if p.id == pin.get("person_id")), None)
                proj = next((p for p in projects if p.id == pin.get("project_id")), None)
                role = pin.get("role")
                if person and proj and role in ROLES:
                    if role not in proj.roles():
                        errors.append(
                            f"ghim: “{proj.name}” không cần vai trò {role}"
                        )
                    elif person.level(role) < proj.level_for(role):
                        have = person.level(role)
                        if have == 0:
                            errors.append(
                                f"ghim: {person.name} không có kỹ năng {role} "
                                f"(cần ≥{proj.level_for(role)} cho “{proj.name}”)"
                            )
                        else:
                            errors.append(
                                f"ghim: {person.name} ({role} {have}) "
                                f"không đủ {role}≥{proj.level_for(role)} cho “{proj.name}”"
                            )

    return errors


def _diff(before: dict, after: dict) -> dict:
    """Compact before/after for the what-if table."""
    b_asg = {a["project_id"]: a for a in before["assignments"]}
    a_asg = {a["project_id"]: a for a in after["assignments"]}
    b_risk = {r["project_id"]: r for r in before["risks"]}
    a_risk = {r["project_id"]: r for r in after["risks"]}
    b_un = {u["project_id"]: u for u in before["unstaffable"]}
    a_un = {u["project_id"]: u for u in after["unstaffable"]}
    names = {p["id"]: p["name"] for p in after["projects"]}
    names.update({p["id"]: p["name"] for p in before["projects"]})
    people_b = {p["id"]: p["name"] for p in before["people"]}
    people_a = {p["id"]: p["name"] for p in after["people"]}
    b_dead = {p["id"]: p.get("deadline") for p in before["projects"]}
    a_dead = {p["id"]: p.get("deadline") for p in after["projects"]}

    def squad_txt(block, people_map):
        if not block:
            return "—"
        bits = []
        for role, pids in (block.get("squad") or {}).items():
            arr = pids if isinstance(pids, list) else [pids]
            bits.append(role + ": " + ", ".join(people_map.get(x, x) for x in arr))
        return " · ".join(bits)

    def status(pid, asg, risk, un):
        if pid in un:
            return "không xếp được"
        r = risk.get(pid) or {}
        if r.get("risk") == "late":
            days = r.get("days_left")
            return f"TRỄ {abs(days)} ngày" if days is not None else "TRỄ"
        if r.get("risk") == "tight":
            return "GẤP"
        if pid in asg:
            return "đúng hạn"
        return "—"

    rows = []
    for pid in sorted(set(b_asg) | set(a_asg) | set(b_un) | set(a_un) | set(names),
                      key=lambda x: names.get(x, x)):
        b_dl = b_dead.get(pid)
        a_dl = a_dead.get(pid)
        changed = (
            (pid in b_asg) != (pid in a_asg)
            or (b_asg.get(pid) or {}).get("squad") != (a_asg.get(pid) or {}).get("squad")
            or (b_asg.get(pid) or {}).get("start_week") != (a_asg.get(pid) or {}).get("start_week")
            or status(pid, b_asg, b_risk, b_un) != status(pid, a_asg, a_risk, a_un)
            or b_dl != a_dl
        )
        rows.append({
            "id": pid,
            "name": names.get(pid, pid),
            "before_squad": squad_txt(b_asg.get(pid), people_b),
            "after_squad": squad_txt(a_asg.get(pid), people_a),
            "before_start": (b_asg[pid]["start_week"] if pid in b_asg else None),
            "after_start": (a_asg[pid]["start_week"] if pid in a_asg else None),
            "before_status": status(pid, b_asg, b_risk, b_un),
            "after_status": status(pid, a_asg, a_risk, a_un),
            "before_deadline": b_dl,
            "after_deadline": a_dl,
            "changed": changed,
        })

    util = []
    b_u = {u["id"]: u for u in before["utilization"]}
    a_u = {u["id"]: u for u in after["utilization"]}
    for uid in sorted(set(b_u) | set(a_u), key=lambda i: (a_u.get(i) or b_u.get(i) or {}).get("name", i)):
        bu, au = b_u.get(uid), a_u.get(uid)
        util.append({
            "id": uid,
            "name": (au or bu)["name"],
            "before": None if not bu else round(bu["util"] * 100),
            "after": None if not au else round(au["util"] * 100),
        })

    n_changed = sum(1 for r in rows if r["changed"])
    causes = _delta_causes(before, after)
    bits = []
    if n_changed:
        bits.append(f"{n_changed} dự án thay đổi so với kế hoạch hiện tại")
    n_dl = sum(1 for r in rows if r.get("before_deadline") != r.get("after_deadline"))
    if n_dl:
        bits.append(f"{n_dl} hạn chót được dời")
    if not bits:
        bits.append("không đổi phân công hay hạn chót")
    return {
        "projects": rows,
        "utilization": util,
        "summary": " · ".join(bits),
        "causes": causes,
    }


def _delta_causes(before: dict, after: dict) -> list[str]:
    """Human reasons for extra lateness (deps + people newly on leave)."""
    b_risk = {r["project_id"]: r for r in before.get("risks") or []}
    a_risk = {r["project_id"]: r for r in after.get("risks") or []}
    a_proj = {p["id"]: p for p in after.get("projects") or []}
    b_proj = {p["id"]: p for p in before.get("projects") or []}
    b_people = {p["id"]: p for p in before.get("people") or []}
    a_people = {p["id"]: p for p in after.get("people") or []}
    a_un = {u["project_id"]: u for u in after.get("unstaffable") or []}

    on_leave = []
    for pid, p in a_people.items():
        old = b_people.get(pid)
        if old is not None and len(p.get("leave") or []) > len(old.get("leave") or []):
            on_leave.append(p["name"])

    lines: list[str] = []
    for pid, ar in a_risk.items():
        br = b_risk.get(pid) or {}
        a_late = max(0, -(ar.get("days_left") or 0)) if ar.get("risk") == "late" else 0
        b_late = max(0, -(br.get("days_left") or 0)) if br.get("risk") == "late" else 0
        extra = a_late - b_late
        proj = a_proj.get(pid) or b_proj.get(pid) or {}
        name = proj.get("name") or ar.get("name") or pid
        deps = proj.get("depends_on") or []
        dep_bits = []
        for dep in deps:
            depn = (a_proj.get(dep) or b_proj.get(dep) or {}).get("name", dep)
            dr = a_risk.get(dep) or {}
            dep_late = dr.get("risk") == "late"
            dep_un = dep in a_un
            if dep_late or dep_un or extra > 0:
                piece = f"phụ thuộc “{depn}”"
                if dep_late and dr.get("days_left") is not None:
                    piece += f" (trễ {abs(dr['days_left'])} ngày)"
                elif dep_un:
                    piece += " chưa xếp được"
                if on_leave:
                    piece += f", mà “{depn}” chờ {', '.join(on_leave)}"
                dep_bits.append(piece)
        if extra > 0:
            why = " vì " + "; ".join(dep_bits) if dep_bits else (
                f" vì chờ {', '.join(on_leave)}" if on_leave else ""
            )
            lines.append(f"“{name}” trễ thêm {extra} ngày{why}")
        elif pid in a_un:
            why = dep_bits[0] if dep_bits else (a_un[pid].get("reason") or "")
            lines.append(f"“{name}” không xếp được" + (f" — {why}" if why else ""))
    return lines


def _capacity_notes(before: dict, after: dict, new_ids: set) -> list[str]:
    notes: list[str] = []
    b_u = {u["id"]: u for u in before.get("utilization") or []}
    for u in after.get("utilization") or []:
        bu = b_u.get(u["id"])
        if not bu:
            notes.append(f"{u['name']} (mới) — tải {round(u['util'] * 100)}%")
            continue
        if u["overloaded"]:
            notes.append(f"{u['name']} vượt sức chứa ({round(u['util'] * 100)}%)")
        elif u["util"] - bu["util"] >= 0.12:
            notes.append(
                f"{u['name']}: tải {round(bu['util'] * 100)}% → {round(u['util'] * 100)}%"
            )
    for u in after.get("unstaffable") or []:
        if u["project_id"] in new_ids:
            notes.append(f"“{u['name']}” không xếp được: {u['reason']}")
    late = [r for r in after.get("risks") or []
            if r.get("project_id") in new_ids and r.get("risk") == "late"]
    for r in late:
        days = r.get("days_left")
        extra = f" (trễ {abs(days)} ngày)" if days is not None else ""
        notes.append(f"“{r['name']}” xếp được nhưng trễ hạn{extra}")
    return notes


def _why_package_late(asg: dict, after: dict, prepared: list) -> str:
    """Short cause for a newly added package that is late."""
    pid = asg.get("project_id")
    proj = next((p for p in (prepared or after.get("projects") or []) if p.get("id") == pid), None)
    names = {p["id"]: p["name"] for p in after.get("projects") or []}
    bits = []
    for dep in (proj or {}).get("depends_on") or []:
        depn = names.get(dep, dep)
        dr = next((r for r in after.get("risks") or [] if r["project_id"] == dep), None)
        if dr and dr.get("risk") == "late":
            bits.append(f"phụ thuộc “{depn}” (trễ {abs(dr.get('days_left') or 0)} ngày)")
        elif any(u["project_id"] == dep for u in after.get("unstaffable") or []):
            bits.append(f"phụ thuộc “{depn}” chưa xếp được")
        else:
            bits.append(f"phụ thuộc “{depn}”")
    people = {p["id"]: p["name"] for p in after.get("people") or []}
    util = {u["id"]: u for u in after.get("utilization") or []}
    week_dates = after.get("week_dates") or []
    start = asg.get("start_week") or 0
    for role_block in asg.get("squad") or []:
        for pname in role_block.get("people") or []:
            pid_p = next((i for i, n in people.items() if n == pname), None)
            if not pid_p:
                continue
            committed = after.get("committed", {}).get(pid_p) or {}
            busy_before = [int(w) for w in committed if int(w) < start and committed[w]]
            if busy_before:
                last = max(busy_before)
                when = week_dates[last][5:] if last < len(week_dates) else f"tuần {last}"
                bits.append(f"chờ {pname} tới {when}")
            u = util.get(pid_p)
            if u and u.get("overloaded"):
                bits.append(f"{pname} vượt sức chứa")
    # unique
    out = []
    for b in bits:
        if b not in out:
            out.append(b)
    return "; ".join(out[:3])


class Handler(BaseHTTPRequestHandler):
    def _bind_session(self) -> dict:
        raw = self.headers.get("Cookie", "")
        sid = None
        for part in raw.split(";"):
            k, _, v = part.strip().partition("=")
            if k == "sf":
                sid = v
                break
        sid, sess, is_new = _get_session(sid)
        self._sid = sid
        self._set_cookie = (
            f"sf={sid}; Path=/; SameSite=Lax; Max-Age={SESSION_TTL_S}" if is_new else None
        )
        return sess

    def _send(self, code: int, body: bytes, ctype: str) -> None:
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        cookie = getattr(self, "_set_cookie", None)
        if cookie:
            self.send_header("Set-Cookie", cookie)
        self.end_headers()
        self.wfile.write(body)

    def _json(self, obj: dict) -> None:
        self._send(200, json.dumps(obj, ensure_ascii=False).encode(), "application/json")

    def _json_status(self, code: int, obj: dict) -> None:
        self._send(code, json.dumps(obj, ensure_ascii=False).encode(), "application/json")

    def do_GET(self) -> None:
        self._bind_session()
        path = urlparse(self.path).path
        if path in ("/", "/index.html"):
            html = (ROOT / "static" / "index.html").read_text()
            self._send(200, html.encode(), "text/html; charset=utf-8")
            return
        if path.startswith("/static/"):
            fp = (ROOT / path.lstrip("/")).resolve()
            static_root = (ROOT / "static").resolve()
            if static_root not in fp.parents and fp != static_root:
                self._send(403, b"forbidden", "text/plain")
                return
            if not fp.is_file():
                self._send(404, b"not found", "text/plain")
                return
            ctype = mimetypes.guess_type(str(fp))[0] or "application/octet-stream"
            if fp.suffix == ".js":
                ctype = "application/javascript; charset=utf-8"
            self._send(200, fp.read_bytes(), ctype)
            return
        # /api/state used to dump the whole roster on GET — removed on purpose.
        self._send(404, b"not found", "text/plain")

    def do_POST(self) -> None:
        try:
            self._do_POST()
        except Exception:
            traceback.print_exc()
            try:
                self._json_status(500, {"error": ["lỗi máy chủ khi xếp lịch"]})
            except Exception:
                pass

    def _do_POST(self) -> None:
        sess = self._bind_session()
        path = urlparse(self.path).path
        if path == "/api/assign":
            with sess["lock"]:
                result = build_result(
                    sess["people"], sess["projects"],
                    prev_assignments=sess.get("prev_assignments") or None,
                    pins=sess.get("pins"),
                )
                sess["prev_assignments"] = list(result["_plan"].assignments)
            self._json(_public(result))
            return
        if path == "/api/replan":
            length = int(self.headers.get("Content-Length", 0))
            try:
                body = json.loads(self.rfile.read(length) or b"{}")
            except json.JSONDecodeError:
                self._json_status(400, {"error": ["JSON không hợp lệ"]})
                return
            if not isinstance(body, dict):
                self._json_status(400, {"error": ["body phải là JSON object"]})
                return
            preview = bool(body.get("preview"))
            with sess["lock"]:
                errors = _validate_changes(body, sess["people"], sess["projects"])
                if errors:
                    self._json_status(400, {"error": errors})
                    return
                if preview:
                    people = copy.deepcopy(sess["people"])
                    projects = copy.deepcopy(sess["projects"])
                    pins = list(sess["pins"])
                    try:
                        _apply_changes_to(body, people, projects, pins)
                    except ValueError as e:
                        self._json_status(400, {"error": [str(e)]})
                        return
                    before = _public(build_result(
                        sess["people"], sess["projects"],
                        prev_assignments=sess.get("prev_assignments") or None,
                        pins=sess.get("pins"),
                    ))
                    after_full = build_result(
                        people, projects,
                        prev_assignments=sess.get("prev_assignments") or None,
                        pins=pins,
                    )
                    after = _public(after_full)
                    self._json({
                        "preview": True,
                        "before": before,
                        "after": after,
                        "delta": _diff(before, after),
                    })
                    return
                freeze = list(sess.get("prev_assignments") or []) if body.get("freeze_existing") else None
                try:
                    _apply_changes_to(body, sess["people"], sess["projects"], sess["pins"])
                except ValueError as e:
                    self._json_status(400, {"error": [str(e)]})
                    return
                result = build_result(
                    sess["people"], sess["projects"],
                    prev_assignments=sess.get("prev_assignments") or None,
                    pins=sess.get("pins"),
                    freeze=freeze,
                )
                sess["prev_assignments"] = list(result["_plan"].assignments)
            self._json(_public(result))
            return
        if path == "/api/reset":
            with sess["lock"]:
                fresh = _fresh_state()
                sess["people"] = fresh["people"]
                sess["projects"] = fresh["projects"]
                sess["pins"] = []
                sess["prev_assignments"] = []
                result = build_result(sess["people"], sess["projects"])
                sess["prev_assignments"] = list(result["_plan"].assignments)
            self._json(_public(result))
            return
        if path == "/api/intake":
            length = int(self.headers.get("Content-Length", 0))
            try:
                body = json.loads(self.rfile.read(length) or b"{}")
            except json.JSONDecodeError:
                self._json_status(400, {"error": "JSON không hợp lệ"})
                return
            brief = (body.get("brief") or "") if isinstance(body, dict) else ""
            needs_llm = not llm._looks_like_whatif(brief)
            if needs_llm and not _llm_allowed("intake"):
                self._json_status(429, {"error": "quá nhiều yêu cầu AI — thử lại sau ít giây"})
                return
            try:
                out = llm.parse_input(brief, today=TODAY)
            except Exception:
                out = {"intent": "invalid", "reason": "không xử lý được đầu vào"}
            if out.get("intent") == "what_if":
                with sess["lock"]:
                    resolved = llm.resolve_whatif(
                        out.get("change") or {},
                        sess["people"], sess["projects"], TODAY,
                        original_text=brief,
                    )
                out.update(resolved)
            self._json(out)
            return
        if path == "/api/decompose":
            length = int(self.headers.get("Content-Length", 0))
            try:
                body = json.loads(self.rfile.read(length) or b"{}")
            except json.JSONDecodeError:
                self._json_status(400, {"error": "JSON không hợp lệ"})
                return
            doc = (body.get("document") or body.get("brief") or "") if isinstance(body, dict) else ""
            if not str(doc).strip():
                self._json_status(400, {"error": "tài liệu trống"})
                return
            if not _llm_allowed("intake"):
                self._json_status(429, {"error": "quá nhiều yêu cầu AI — thử lại sau ít giây"})
                return
            out = llm.decompose_document(str(doc), today=TODAY)
            if out.get("error") and not out.get("tasks"):
                self._json_status(400, {"error": out["error"]})
                return
            self._json(out)
            return
        if path == "/api/propose-staff":
            length = int(self.headers.get("Content-Length", 0))
            try:
                body = json.loads(self.rfile.read(length) or b"{}")
            except json.JSONDecodeError:
                self._json_status(400, {"error": ["JSON không hợp lệ"]})
                return
            tasks = (body.get("projects") or body.get("tasks") or []) if isinstance(body, dict) else []
            if not isinstance(tasks, list) or not tasks:
                self._json_status(400, {"error": ["chưa có gói việc để xếp người"]})
                return
            prepared = []
            for i, t in enumerate(tasks):
                if not isinstance(t, dict):
                    self._json_status(400, {"error": ["gói việc không hợp lệ"]})
                    return
                item = dict(t)
                item.setdefault("id", f"jdoc_{int(time.time())}_{i}")
                if not item.get("deadline"):
                    from datetime import timedelta as _td
                    item["deadline"] = (TODAY + _td(days=21)).isoformat()
                prepared.append(item)
            with sess["lock"]:
                errors = _validate_changes({"add_projects": prepared}, sess["people"], sess["projects"])
                if errors:
                    self._json_status(400, {"error": errors})
                    return
                people = copy.deepcopy(sess["people"])
                projects = copy.deepcopy(sess["projects"])
                pins = list(sess["pins"])
                freeze = list(sess.get("prev_assignments") or [])
                try:
                    _apply_changes_to({"add_projects": prepared}, people, projects, pins)
                except ValueError as e:
                    self._json_status(400, {"error": [str(e)]})
                    return
                before = _public(build_result(
                    sess["people"], sess["projects"],
                    prev_assignments=sess.get("prev_assignments") or None,
                    pins=sess.get("pins"),
                ))
                after_full = build_result(
                    people, projects,
                    prev_assignments=None,
                    pins=pins,
                    freeze=freeze,
                )
                after = _public(after_full)
            new_ids = {p["id"] for p in prepared}
            names = {p["id"]: p["name"] for p in after["people"]}
            new_asg = []
            for a in after["assignments"]:
                if a["project_id"] not in new_ids:
                    continue
                risk = next((r for r in after["risks"] if r["project_id"] == a["project_id"]), None)
                squad = []
                for role, pids in (a.get("squad") or {}).items():
                    arr = pids if isinstance(pids, list) else [pids]
                    squad.append({"role": role, "people": [names.get(x, x) for x in arr]})
                new_asg.append({
                    "project_id": a["project_id"], "name": a["name"],
                    "start_week": a["start_week"],
                    "squad": squad,
                    "risk": (risk or {}).get("risk"),
                    "days_left": (risk or {}).get("days_left"),
                    "done_date": (risk or {}).get("done_date"),
                })
            unstaff = [u for u in after["unstaffable"] if u["project_id"] in new_ids]
            notes = _capacity_notes(before, after, new_ids)
            for a in new_asg:
                if a.get("risk") == "late":
                    a["why"] = _why_package_late(a, after, prepared)
            self._json({
                "projects": prepared,
                "assignments": new_asg,
                "unstaffable": unstaff,
                "notes": notes,
                "delta": _diff(before, after),
            })
            return
        if path == "/api/explain":
            if not _llm_allowed("explain"):
                self._json_status(429, {"error": "quá nhiều yêu cầu AI — thử lại sau ít giây"})
                return
            with sess["lock"]:
                people = list(sess["people"])
                projects = list(sess["projects"])
                prev = sess.get("prev_assignments") or None
                pins = list(sess.get("pins") or [])
            try:
                plan, _, risks = pipeline.run(
                    people, projects, TODAY, prev_assignments=prev, pins=pins,
                )
                text = explain_plan_llm(plan, people, projects, risks)
                self._json({"explanation": text})
            except Exception:
                self._json({"error": "giải thích thất bại"})
            return
        if path == "/api/suggest":
            with sess["lock"]:
                people = copy.deepcopy(sess["people"])
                projects = copy.deepcopy(sess["projects"])
                pins = list(sess.get("pins") or [])
                prev = sess.get("prev_assignments") or None
                base = build_result(people, projects, prev_assignments=prev, pins=pins)
                sess["prev_assignments"] = list(base["_plan"].assignments)
                _, risk_objs = analysis.analyze(base["_plan"], people, projects)
                ranked = suggest.suggest_fixes(
                    people, projects, TODAY,
                    prev_assignments=sess["prev_assignments"],
                    pins=pins,
                    bottlenecks=base.get("bottlenecks"),
                    risks=risk_objs,
                )
            baseline_late = sum(
                max(0, -(r.get("days_left") or 0))
                for r in base.get("risks") or [] if r.get("risk") == "late"
            )
            self._json({
                "baseline_late_days": baseline_late,
                "suggestions": ranked,
            })
            return
        self._send(404, b"not found", "text/plain")

    def log_message(self, fmt, *args) -> None:
        pass  # quiet access log

    def log_error(self, fmt, *args) -> None:
        sys.stderr.write("HTTP error: " + (fmt % args) + "\n")


def main() -> None:
    port = 8091
    # warm up CP-SAT on startup so the first request is not a cold start
    try:
        import optimizer
        if optimizer.has_cp_sat():
            warm = _fresh_state()
            optimizer.cp_sat_assign(warm["people"], warm["projects"], TODAY)
    except Exception:
        pass
    print(f"squadflow UI → http://localhost:{port}")
    ThreadingHTTPServer(("0.0.0.0", port), Handler).serve_forever()


if __name__ == "__main__":
    main()
