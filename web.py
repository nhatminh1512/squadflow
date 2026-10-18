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
from datetime import date, timedelta
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

import analysis
import forecast
import llm
import pipeline
import suggest
from engine import WEEK, role_load, week_capacity
from explain import EMPTY_PLAN_EXPLANATION, explain_plan_llm
from models import HIRE_LEAD_WEEKS, Person, Project, ROLES

# Client gone mid-response — never treat this as an application failure.
_DISCONNECT = (BrokenPipeError, ConnectionResetError, ConnectionAbortedError)

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
        "result": None,
        "gen": 0,                 # bumped by /api/reset; stale changes are refused
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


# Rate limiting — socket IP only (X-Forwarded-For is spoofable).
# Global + per-kind buckets so 40 bursts in 2 minutes are blocked.
_llm_lock = threading.Lock()
_llm_times: dict[str, list[float]] = {}
_LLM_WINDOW_S = 60.0
_LLM_MAX_GLOBAL = 8


def _client_ip(handler) -> str:
    try:
        return handler.client_address[0] or "unknown"
    except Exception:
        return "unknown"


def _llm_allowed(kind: str = "intake", ip: str = "") -> bool:
    now = time.time()
    ip = ip or "unknown"
    with _llm_lock:
        bucket = _llm_times.setdefault(ip, [])
        bucket[:] = [t for t in bucket if now - t < _LLM_WINDOW_S]
        if len(bucket) >= _LLM_MAX_GLOBAL:
            return False
        bucket.append(now)
        return True


def _llm_retry_after(ip: str = "") -> int:
    now = time.time()
    ip = ip or "unknown"
    with _llm_lock:
        bucket = [t for t in _llm_times.get(ip, []) if now - t < _LLM_WINDOW_S]
        if not bucket:
            return int(_LLM_WINDOW_S)
        wait = math.ceil(_LLM_WINDOW_S - (now - min(bucket)))
        return max(1, min(int(_LLM_WINDOW_S), wait))


def _iso(d: date) -> str:
    return d.isoformat()


def _parse_int(v, field: str, lo: int | None = None, hi: int | None = None) -> int:
    """Strict integer parse — rejects 3.5, bools, and junk without leaking Python errors."""
    if isinstance(v, bool) or v is None:
        raise ValueError(f"{field} must be a whole number")
    if isinstance(v, float):
        if not v.is_integer():
            raise ValueError(f"{field} must be a whole number, got {v}")
        v = int(v)
    elif isinstance(v, str):
        s = v.strip()
        if not re.fullmatch(r"-?\d+", s):
            raise ValueError(f"{field} must be a whole number")
        v = int(s)
    elif not isinstance(v, int):
        raise ValueError(f"{field} must be a whole number")
    if lo is not None and v < lo:
        raise ValueError(f"{field} must be ≥ {lo}")
    if hi is not None and v > hi:
        raise ValueError(f"{field} must be ≤ {hi}")
    return v


def _parse_date(v, field: str) -> date:
    try:
        return date.fromisoformat(str(v))
    except (TypeError, ValueError):
        raise ValueError(f"{field} has the wrong format (expected YYYY-MM-DD)")


def build_result(people: list[Person], projects: list[Project],
                 prev_assignments=None, pins=None, freeze=None,
                 keep_squads: bool = False) -> dict:
    pins = list(pins or [])
    plan, pstats, risks = pipeline.run(
        people, projects, TODAY,
        prev_assignments=prev_assignments or None,
        pins=pins,
        freeze=freeze,
        keep_squads=keep_squads,
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
             "leave": [[_iso(a), _iso(b)] for a, b in p.leave],
             "available_from": _iso(p.available_from) if p.available_from else None}
            for p in people
        ],
        "projects": [
            {"id": p.id, "name": p.name, "required": p.required,
             "duration_weeks": p.duration_weeks, "deadline": _iso(p.deadline),
             "priority": p.priority, "depends_on": p.depends_on,
             "method": p.method, "stages": p.stages,
             "milestones": {k: _iso(v) for k, v in p.milestones.items()}}
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
             "slots": s.capacity_weeks,
             "peak": s.peak_load, "util": round(s.util, 3),
             "idle": s.idle, "overloaded": s.overloaded, "cover_days": s.cover_days}
            for s in pstats
        ],
        "bottlenecks": bottlenecks,
        "risks": [
            {"project_id": r.project.id, "name": r.project.name, "risk": r.risk,
             "start_week": r.start_week, "done_date": r.done_date,
             "days_left": r.days_left, "deadline": _iso(r.project.deadline),
             "late_days": r.late_days,
             "milestones": r.milestones or []}
            for r in risks
        ],
        "committed": {
            pid: {str(w): sorted(s) for w, s in weeks.items()}
            for pid, weeks in plan.committed.items()
        },
        # % of a full-time week booked per person per week (100 = one full-time role)
        "load": {
            pid: {str(w): u * 100 // WEEK for w, u in weeks.items() if u}
            for pid, weeks in plan.load.items()
        },
        "decisions": [
            {**d, "person_name": name.get(d["person_id"], d["person_id"])}
            for d in plan.decisions
        ],
        "miss_prob": forecast.monte_carlo(plan, projects),
        "p80_finish": forecast.p80_finish(plan, projects),
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
        person.leave.append((_parse_date(lv["from"], "start date"),
                             _parse_date(lv["to"], "end date")))

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
            person.skills = {k: _parse_int(v, f"level {k}", 1, 5) for k, v in up["skills"].items()}
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
        raise ValueError("missing id or name")
    if d.get("role") not in ROLES:
        raise ValueError(f"invalid role: {d.get('role')}")
    skills = d.get("skills") or {}
    if not isinstance(skills, dict):
        raise ValueError("skills must be an object of role→level")
    for k, v in skills.items():
        if k not in ROLES:
            raise ValueError(f"invalid skill: {k}")
        _parse_int(v, f"level '{k}'", 1, 5)
    try:
        cap = float(d.get("capacity", 1.0))
    except (TypeError, ValueError):
        raise ValueError("capacity must be a number")
    _check_capacity(cap)
    for pair in d.get("leave", []) or []:
        if not (isinstance(pair, (list, tuple)) and len(pair) == 2):
            raise ValueError("leave must be [from, to]")
        a = _parse_date(pair[0], "leave start date")
        b = _parse_date(pair[1], "leave end date")
        if a > b:
            raise ValueError("leave start must be on or before leave end")
        if a < TODAY:
            raise ValueError("leave cannot start before today")
    if d.get("available_from"):
        _parse_date(d["available_from"], "start date")


def _check_capacity(cap: float) -> None:
    if cap > 10:
        raise ValueError(
            "capacity is in full-time roles (1 = one full-time role, 2 = two), not a percentage — try 1"
        )
    if not (0.5 <= cap <= 4):
        raise ValueError("capacity must be within 0.5..4 (full-time roles at the same time)")


def _cycle_errors(projects) -> list[str]:
    """Self-deps and cycles (DFS)."""
    graph = {p.id: [d for d in (p.depends_on or [])] for p in projects}
    names = {p.id: getattr(p, "name", p.id) for p in projects}
    errors: list[str] = []
    for pid, deps in graph.items():
        if pid in deps:
            errors.append(f"“{names.get(pid, pid)}” cannot depend on itself")
    WHITE, GRAY, BLACK = 0, 1, 2
    color = {i: WHITE for i in graph}
    stack: list[str] = []

    def dfs(u: str) -> None:
        color[u] = GRAY
        stack.append(u)
        for v in graph.get(u, []):
            if v not in graph:
                continue
            if color[v] == GRAY:
                cyc = stack[stack.index(v):] + [v]
                label = " → ".join(names.get(x, x) for x in cyc)
                errors.append(f"dependency cycle: {label}")
            elif color[v] == WHITE:
                dfs(v)
        stack.pop()
        color[u] = BLACK

    for u in graph:
        if color[u] == WHITE:
            dfs(u)
    # unique
    out = []
    for e in errors:
        if e not in out:
            out.append(e)
    return out


def _commit(sess: dict, result: dict) -> dict:
    """Store an approved plan. /api/assign, /api/suggest, /api/explain and every
    'before' column read this saved result back instead of re-solving."""
    sess["result"] = result
    sess["prev_assignments"] = list(result["_plan"].assignments)
    return result


def _saved(sess: dict) -> dict:
    """This session's saved plan; solved once if missing (staffed projects stay
    hard-frozen, only projects without a squad are scheduled)."""
    if sess.get("result") is None:
        from engine import sanitize_freeze
        prev = sess.get("prev_assignments") or None
        freeze = sanitize_freeze(prev, sess["people"], sess["projects"],
                                 sess.get("pins"), today=TODAY)
        _commit(sess, build_result(
            sess["people"], sess["projects"],
            prev_assignments=prev, pins=sess.get("pins"), freeze=freeze,
        ))
    return sess["result"]


def _ensure_prev(sess: dict) -> list:
    """Baseline assignments to freeze: the saved plan."""
    return list(_saved(sess)["_plan"].assignments)


def _norm_name(s: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^\w]+", " ", (s or "").lower()).replace("_", " ")).strip()


_BILINGUAL_PHRASES: list[tuple[str, str]] = [
    ("cổng thanh toán", "payment gateway"),
    ("nền tảng thanh toán", "payment gateway"),
    ("hệ thống thanh toán", "payment gateway"),
    ("thanh toán", "payment"),
    ("ứng dụng di động", "mobile app"),
    ("app di động", "mobile app"),
    ("di động", "mobile"),
    ("kho dữ liệu", "data warehouse"),
    ("dữ liệu", "data"),
    ("kiểm toán bảo mật", "security audit"),
    ("đánh giá bảo mật", "security audit"),
    ("rà soát bảo mật", "security audit"),
    ("bảo mật", "security"),
    ("an ninh", "security"),
    ("kiểm toán", "audit"),
    ("tích hợp crm", "crm integration"),
    ("quản lý khách hàng", "crm"),
    ("chăm sóc khách hàng", "crm"),
    ("bảng điều khiển nội bộ", "internal dashboard"),
    ("dashboard nội bộ", "internal dashboard"),
    ("bảng điều khiển", "dashboard"),
    ("nội bộ", "internal"),
    ("cổng api", "api gateway"),
    ("xác thực", "auth"),
    ("đăng nhập", "auth"),
    ("phân quyền", "auth"),
    ("thông báo", "notification"),
    ("hóa đơn", "billing"),
    ("thu phí", "billing"),
    ("tìm kiếm", "search"),
    ("báo cáo", "report"),
    ("phân tích", "analytics"),
    ("di chuyển", "migration"),
    ("chuyển đổi", "migration"),
    ("nâng cấp", "migration"),
    ("thiết kế lại", "redesign"),
    ("làm lại", "redesign"),
    ("tích hợp", "integration"),
]

_GENERIC_MODIFIERS = {
    "migration", "redesign", "integration", "upgrade", "update",
    "project", "package", "module", "system", "platform", "service",
    "dự", "án", "gói", "hệ", "thống", "nền", "tảng", "dịch", "vụ",
    "mới", "new",
}


def _canonical_tokens(s: str) -> tuple[set[str], set[str]]:
    norm = _norm_name(s)
    if not norm:
        return set(), set()
    padded = f" {norm} "
    for vn, en in _BILINGUAL_PHRASES:
        padded = padded.replace(f" {vn} ", f" {en} ")
    tokens = set(padded.split())
    core = {t for t in tokens if t not in _GENERIC_MODIFIERS}
    return tokens, (core or tokens)


def _names_close(a: str, b: str) -> bool:
    na, nb = _norm_name(a), _norm_name(b)
    if not na or not nb:
        return False
    if na == nb or na in nb or nb in na:
        return True
    ta0, tb0 = set(na.split()), set(nb.split())
    if ta0 and tb0 and len(ta0 & tb0) / len(ta0 | tb0) >= 0.5:
        return True
    ta, ca = _canonical_tokens(a)
    tb, cb = _canonical_tokens(b)
    if ca and cb:
        if ca == cb:
            return True
        if len(ca) >= 2 and (ca <= cb or cb <= ca):
            return True
        if len(ca & cb) / len(ca | cb) >= 0.5:
            return True
    if ta and tb and len(ta & tb) / len(ta | tb) >= 0.5:
        return True
    return False


def _roles_key(required: dict) -> tuple:
    out = []
    for role, spec in (required or {}).items():
        if isinstance(spec, dict):
            out.append((role, int(spec.get("level") or 0), int(spec.get("count") or 1)))
        else:
            out.append((role, int(spec or 0), 1))
    return tuple(sorted(out))


def _llm_duplicate_matches(unmatched_names: list[str], existing_names: list[str]) -> dict[str, str]:
    """Use LLM (cached) to catch cross-language duplicates not in the bilingual dictionary."""
    if not unmatched_names or not existing_names:
        return {}
    key_txt = json.dumps({"new": unmatched_names, "old": existing_names}, ensure_ascii=False, sort_keys=True)
    cached = llm.cache_get("dup_names", key_txt, TODAY)
    if cached is not None:
        return cached.get("matches") or {}
    prompt = (
        "Compare the new work packages with the existing projects (names may be in a different "
        "language, Vietnamese or English — e.g. 'Cổng thanh toán' duplicates 'Payment gateway migration').\n"
        f"New packages: {json.dumps(unmatched_names, ensure_ascii=False)}\n"
        f"Existing projects: {json.dumps(existing_names, ensure_ascii=False)}\n"
        "Return plain JSON: {\"matches\": {\"<new package name>\": \"<existing project with the same meaning>\"}}\n"
        "Only list pairs that are truly the same piece of work. If none, return {\"matches\": {}}."
    )
    try:
        data = llm.complete_json(
            [{"role": "system", "content": "You detect duplicate project names across Vietnamese and English. Reply with plain JSON only."},
             {"role": "user", "content": prompt}],
            temperature=0.0, max_tokens=300, attempts=1, timeout=8,
        )
        matches = data.get("matches") if isinstance(data, dict) else {}
        if not isinstance(matches, dict):
            matches = {}
        valid_old = set(existing_names)
        clean = {str(k): str(v) for k, v in matches.items() if str(v) in valid_old}
        llm.cache_put("dup_names", key_txt, TODAY, {"matches": clean})
        return clean
    except Exception:
        return {}


def _overlap_warnings(incoming: list, existing: list, use_llm: bool = True) -> list[str]:
    """Warn when a new package duplicates (or nearly duplicates) a live project."""
    warnings: list[str] = []
    unmatched: list[str] = []
    existing_names: list[str] = [
        getattr(old, "name", None) or (old.get("name") if isinstance(old, dict) else "")
        for old in existing
    ]
    existing_names = [n for n in existing_names if n]

    for aj in incoming:
        if not isinstance(aj, dict):
            continue
        name = aj.get("name") or ""
        req = aj.get("required") or {}
        dur = aj.get("duration_weeks")
        dl = str(aj.get("deadline") or "")
        matched = False
        for old in existing:
            old_name = getattr(old, "name", None) or (old.get("name") if isinstance(old, dict) else "")
            old_req = getattr(old, "required", None) or (old.get("required") if isinstance(old, dict) else {})
            old_dur = getattr(old, "duration_weeks", None)
            if isinstance(old, dict):
                old_dl = str(old.get("deadline") or "")
            else:
                old_dl = old.deadline.isoformat() if getattr(old, "deadline", None) else ""
            close_name = _names_close(name, old_name)
            same_shape = (
                _roles_key(req) == _roles_key(old_req)
                and dur is not None and old_dur is not None
                and abs(int(dur) - int(old_dur)) <= 1
            )
            if close_name:
                warnings.append(
                    f"package “{name}” duplicates or nearly duplicates existing project “{old_name}”"
                )
                matched = True
                break
            gap = None
            if same_shape and dl and old_dl:
                try:
                    gap = abs((date.fromisoformat(str(dl)[:10]) - date.fromisoformat(str(old_dl)[:10])).days)
                except ValueError:
                    gap = None
            if gap is not None and gap <= 21:
                warnings.append(
                    f"package “{name}” looks like “{old_name}” (same roles, ~{old_dur} weeks)"
                )
                matched = True
                break
        if not matched and name:
            unmatched.append(name)

    if use_llm and unmatched and existing_names:
        llm_matches = _llm_duplicate_matches(unmatched, existing_names)
        for name in unmatched:
            old_name = llm_matches.get(name)
            if old_name:
                warnings.append(
                    f"package “{name}” duplicates or nearly duplicates existing project “{old_name}”"
                )
    return warnings


def _added_ids(body: dict) -> set:
    out = {(body.get("add_project") or {}).get("id")}
    out |= {p.get("id") for p in body.get("add_projects") or [] if isinstance(p, dict)}
    return {x for x in out if x}


def _compatible_freeze(prev, body: dict, projects=None) -> list:
    from engine import compatible_freeze
    return compatible_freeze(prev, body, projects, today=TODAY)



def _check_project(d: dict) -> None:
    if not (d.get("id") and d.get("name")):
        raise ValueError("missing id or name")
    required = d.get("required") or {}
    if not required:
        raise ValueError("missing staffing requirements (roles)")
    if not isinstance(required, dict):
        raise ValueError("required must be an object of role→level")
    for role, spec in required.items():
        if role not in ROLES:
            raise ValueError(f"invalid role: {role}")
        level = spec.get("level") if isinstance(spec, dict) else spec
        _parse_int(level, f"level '{role}'", 1, 5)
    dur = _parse_int(d.get("duration_weeks", 0), "duration", 1, 52)
    for role, spec in required.items():
        if not isinstance(spec, dict):
            continue
        _parse_int(spec.get("count", 1), f"count '{role}'", 1, 10)
        _parse_int(spec.get("effort", 100), f"effort '{role}' (%)", 5, 100)
        start = _parse_int(spec.get("start", 0) or 0, f"start week '{role}'", 0, dur - 1)
        if spec.get("weeks") not in (None, ""):
            _parse_int(spec["weeks"], f"weeks '{role}'", 1, dur - start)
        phases = spec.get("phases")
        if phases is not None:
            if not isinstance(phases, list):
                raise ValueError(f"phases '{role}' must be a list")
            for ph in phases:
                if not isinstance(ph, dict):
                    raise ValueError(f"phases '{role}': each phase must be an object")
                ps = _parse_int(ph.get("start", 0) or 0, f"phase start '{role}'", 0, dur - 1)
                _parse_int(ph.get("weeks", 1), f"phase weeks '{role}'", 1, dur - ps)
                _parse_int(ph.get("effort", 100), f"phase effort '{role}' (%)", 5, 100)
    method = d.get("method") or "simple"
    if method not in ("simple", "waterfall"):
        raise ValueError("method must be 'simple' or 'waterfall'")
    milestones = d.get("milestones") or {}
    if not isinstance(milestones, dict):
        raise ValueError("milestones must be an object of stage → date")
    if milestones:
        import methodology
        known = set(methodology.MILESTONE_STAGES)
        if method != "waterfall" or not methodology.stage_plan(dur):
            raise ValueError("stage milestones need a waterfall project of at least "
                             f"{methodology.MIN_WEEKS} weeks")
        for stage, due in milestones.items():
            if stage not in known:
                raise ValueError(f"unknown stage '{stage}' (this project has: {', '.join(sorted(known))})")
            _parse_date(due, f"{stage} milestone")
    deadline = _parse_date(d.get("deadline"), "deadline")
    if deadline < TODAY:
        raise ValueError("deadline is in the past (must be today or later)")
    _parse_int(d.get("priority", 3), "priority", 1, 5)


def _validate_changes(body: dict, people: list[Person], projects: list[Project]) -> list[str]:
    errors: list[str] = []
    if not isinstance(body, dict):
        return ["body must be a JSON object (not a list/string)"]
    pids = {p.id for p in people}
    jids = {p.id for p in projects}

    for key in ("leave", "slip", "remove_project", "remove_person", "pin", "unpin"):
        v = body.get(key, [])
        if v is None:
            continue
        if not isinstance(v, list):
            errors.append(f"{key} must be a list")

    for lv in body.get("leave", []) if isinstance(body.get("leave", []), list) else []:
        if not isinstance(lv, dict):
            errors.append("leave: each item must be an object")
            continue
        if lv.get("person_id") not in pids:
            errors.append("leave: person not found")
        else:
            try:
                a = _parse_date(lv.get("from"), "leave start date")
                b = _parse_date(lv.get("to"), "leave end date")
                if a > b:
                    errors.append("leave: start must be on or before end")
                elif a < TODAY:
                    errors.append("leave: cannot start before today")
            except ValueError as e:
                errors.append(f"leave: {e}")

    for sl in body.get("slip", []) if isinstance(body.get("slip", []), list) else []:
        if not isinstance(sl, dict):
            errors.append("deadline change: each item must be an object")
            continue
        if sl.get("project_id") not in jids:
            errors.append("deadline change: project not found")
        else:
            try:
                dl = _parse_date(sl.get("deadline"), "deadline")
                if dl < TODAY:
                    errors.append("deadline change: deadline is in the past (must be today or later)")
            except ValueError as e:
                errors.append(f"deadline change: {e}")

    ap = body.get("add_person")
    if ap:
        if not isinstance(ap, dict):
            errors.append("add person: invalid data")
        elif ap.get("id") in pids:
            errors.append("add person: id already exists")
        else:
            try:
                _check_person(ap)
            except ValueError as e:
                errors.append(f"add person: {e}")

    incoming = []
    if body.get("add_project"):
        incoming.append(body["add_project"])
    if isinstance(body.get("add_projects"), list):
        incoming.extend(body["add_projects"])
    seen_new = set()
    for aj in incoming:
        if not isinstance(aj, dict):
            errors.append("add project: invalid data")
            continue
        pid = aj.get("id")
        if not pid:
            errors.append("add project: missing id")
            continue
        if pid in jids or pid in seen_new:
            errors.append(f"add project: id already exists ({pid})")
            continue
        seen_new.add(pid)
        try:
            _check_project(aj)
        except ValueError as e:
            errors.append(f"add project “{aj.get('name') or pid}”: {e}")

    up = body.get("update_person")
    if up:
        if not isinstance(up, dict) or up.get("id") not in pids:
            errors.append("edit person: person not found")
        else:
            try:
                if "role" in up and up["role"] not in ROLES:
                    raise ValueError(f"invalid role: {up['role']}")
                if "skills" in up:
                    if not isinstance(up["skills"], dict):
                        raise ValueError("skills must be an object")
                    for k, v in up["skills"].items():
                        if k not in ROLES:
                            raise ValueError(f"invalid skill: {k}")
                        _parse_int(v, f"level '{k}'", 1, 5)
                if "capacity" in up:
                    cap = float(up["capacity"])
                    _check_capacity(cap)
            except (TypeError, ValueError) as e:
                msg = str(e)
                if "invalid literal" in msg or "could not convert" in msg:
                    msg = "invalid value (expected a whole / positive number)"
                errors.append(f"edit person: {msg}")

    for pid in body.get("remove_person", []) if isinstance(body.get("remove_person", []), list) else []:
        if pid not in pids:
            errors.append("remove person: person not found")

    for key in ("pin", "unpin"):
        for pin in body.get(key, []) if isinstance(body.get(key, []), list) else []:
            if not isinstance(pin, dict):
                errors.append("pin: invalid data")
                continue
            if pin.get("person_id") not in pids:
                errors.append("pin: person not found")
            if pin.get("project_id") not in jids:
                errors.append("pin: project not found")
            if pin.get("role") not in ROLES:
                errors.append("pin: invalid role")
                continue
            if key == "pin":
                person = next((p for p in people if p.id == pin.get("person_id")), None)
                proj = next((p for p in projects if p.id == pin.get("project_id")), None)
                role = pin.get("role")
                if person and proj and role in ROLES:
                    if role not in proj.roles():
                        errors.append(
                            f"pin: “{proj.name}” does not need the {role} role"
                        )
                    elif person.level(role) < proj.level_for(role):
                        have = person.level(role)
                        if have == 0:
                            errors.append(
                                f"pin: {person.name} has no {role} skill "
                                f"(“{proj.name}” needs ≥{proj.level_for(role)})"
                            )
                        else:
                            errors.append(
                                f"pin: {person.name} ({role} {have}) "
                                f"does not meet {role}≥{proj.level_for(role)} for “{proj.name}”"
                            )

    class _Node:
        def __init__(self, pid, name, deps):
            self.id = pid
            self.name = name
            self.depends_on = list(deps or [])

    merged = [_Node(p.id, p.name, p.depends_on) for p in projects]
    for aj in incoming:
        if isinstance(aj, dict) and aj.get("id"):
            merged.append(_Node(aj["id"], aj.get("name") or aj["id"], aj.get("depends_on") or []))
    errors.extend(_cycle_errors(merged))

    known_ids = set(jids) | seen_new
    for aj in incoming:
        if not isinstance(aj, dict):
            continue
        deps = aj.get("depends_on") or []
        if isinstance(deps, str):
            deps = [deps]
        if deps and not isinstance(deps, list):
            errors.append(
                f"add project “{aj.get('name') or aj.get('id')}”: depends_on must be a list of ids"
            )
            continue
        for dep in deps:
            if dep not in known_ids:
                label = aj.get("name") or aj.get("id") or "new"
                errors.append(
                    f"add project “{label}”: depends on unknown id ({dep})"
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
    b_weeks = before.get("week_dates") or []
    a_weeks = after.get("week_dates") or []

    def week_iso(weeks, sw):
        if sw is None:
            return None
        try:
            sw = int(sw)
        except (TypeError, ValueError):
            return None
        if 0 <= sw < len(weeks):
            return weeks[sw]
        return None

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
            return "unstaffable"
        r = risk.get(pid) or {}
        if r.get("risk") == "late":
            return f"LATE {_late_days(r)} days"
        if r.get("risk") == "tight":
            return "TIGHT"
        if pid in asg:
            return "on time"
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
        b_sw = (b_asg[pid]["start_week"] if pid in b_asg else None)
        a_sw = (a_asg[pid]["start_week"] if pid in a_asg else None)
        rows.append({
            "id": pid,
            "name": names.get(pid, pid),
            "before_squad": squad_txt(b_asg.get(pid), people_b),
            "after_squad": squad_txt(a_asg.get(pid), people_a),
            "before_start": b_sw,
            "after_start": a_sw,
            "before_start_date": week_iso(b_weeks, b_sw),
            "after_start_date": week_iso(a_weeks, a_sw),
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

    old_ids = {p["id"] for p in before.get("projects") or []}
    reasons = _change_reasons(before, after)
    for r in rows:
        r["is_new"] = r["id"] not in old_ids
        r["reason"] = reasons.get(r["id"], "") if r["changed"] and not r["is_new"] else ""
    n_changed_old = sum(1 for r in rows if r["changed"] and not r["is_new"])
    n_new = sum(1 for r in rows if r["is_new"])
    causes = _delta_causes(before, after, reasons,
                           {r["id"] for r in rows if r["changed"] and not r["is_new"]})
    bits = []
    if n_changed_old == 0:
        bits.append("existing plan unchanged")
    else:
        bits.append(f"{n_changed_old} existing project(s) changed")
    if n_new:
        bits.append(f"{n_new} new package(s)")
    n_dl = sum(
        1 for r in rows
        if not r["is_new"] and r.get("before_deadline") and r.get("after_deadline")
        and r["before_deadline"] != r["after_deadline"]
    )
    if n_dl:
        bits.append(f"{n_dl} existing deadline(s) moved")
    return {
        "projects": rows,
        "utilization": util,
        "summary": " · ".join(bits),
        "causes": causes,
    }


def _person_leave_weeks(person_dict: dict) -> set[int]:
    out: set[int] = set()
    for pair in person_dict.get("leave") or []:
        if not (isinstance(pair, (list, tuple)) and len(pair) == 2):
            continue
        try:
            wa = (date.fromisoformat(str(pair[0])) - TODAY).days // 7
            wb = (date.fromisoformat(str(pair[1])) - TODAY).days // 7
            out.update(range(wa, wb + 1))
        except ValueError:
            continue
    return out


def _md(d: date | str) -> str:
    """'Oct 19' — same format as the UI."""
    d = date.fromisoformat(d) if isinstance(d, str) else d
    return f"{d:%b} {d.day}"


def _wk(w: int) -> str:
    """Monday of week w."""
    return _md(TODAY + timedelta(weeks=w))


def _wk_fri(w: int) -> str:
    """Friday of week w."""
    return _md(TODAY + timedelta(weeks=w, days=4))


def _late_days(risk: dict | None) -> int:
    if not risk or risk.get("risk") != "late":
        return 0
    if risk.get("late_days") is not None:
        return risk["late_days"]
    return max(0, -(risk.get("days_left") or 0))


def _change_reasons(before: dict, after: dict) -> dict[str, str]:
    """project id -> why it changed, built only from before/after facts
    (deadline, leave, busy squad, dependency, new hire). Dates come from the
    data, never from an LLM. Every changed old project gets an entry."""
    b_asg = {a["project_id"]: a for a in before.get("assignments") or []}
    a_asg = {a["project_id"]: a for a in after.get("assignments") or []}
    b_proj = {p["id"]: p for p in before.get("projects") or []}
    a_proj = {p["id"]: p for p in after.get("projects") or []}
    a_un = {u["project_id"]: u for u in after.get("unstaffable") or []}
    b_un = {u["project_id"]: u for u in before.get("unstaffable") or []}
    a_people = {p["id"]: p for p in after.get("people") or []}
    b_people = {p["id"]: p for p in before.get("people") or []}
    b_comm = before.get("committed") or {}
    a_comm = after.get("committed") or {}
    people_obj = {pid: Person.from_dict(p) for pid, p in a_people.items()}
    projs = {jid: Project.from_dict(p) for jid, p in a_proj.items()}
    # load is sent as % of a full-time week; back to engine units (exact)
    a_load = {pid: {int(w): pct * WEEK // 100 for w, pct in weeks.items()}
              for pid, weeks in (after.get("load") or {}).items()}

    def pname(pid):
        return (a_people.get(pid) or b_people.get(pid) or {}).get("name", pid)

    def jname(jid):
        return (a_proj.get(jid) or b_proj.get(jid) or {}).get("name", jid)

    def end_week(asg, proj):
        return int(asg["start_week"]) + int(proj.get("duration_weeks") or 0)

    def members(asg):
        out: dict[str, list[str]] = {}
        for role, pids in ((asg or {}).get("squad") or {}).items():
            out[role] = list(pids if isinstance(pids, list) else [pids])
        return out

    def others(comm, pid, w, self_id):
        return [j for j in (comm.get(pid) or {}).get(str(w)) or [] if j != self_id]

    def own_units(pid, w, jid):
        """Units project jid itself books for pid in week w (after plan)."""
        aa, j = a_asg.get(jid), projs.get(jid)
        if not aa or not j:
            return 0
        return sum(role_load(j, r, int(aa["start_week"])).get(w, 0)
                   for r, ps in members(aa).items() if pid in ps)

    def blocked_weeks(jid, pid, role, s0, s1):
        """Weeks in which pid could not give `role` of jid, for starts in [s0, s1)."""
        person, j = people_obj.get(pid), projs.get(jid)
        if not person or not j or s1 <= s0:
            return []
        out: set[int] = set()
        for s in range(s0, s1):
            for w, need in role_load(j, role, s).items():
                if w < 0 or w in out:
                    continue
                avail, cover = week_capacity(person, TODAY, w)
                if a_load.get(pid, {}).get(w, 0) - own_units(pid, w, jid) + need > avail + cover:
                    out.add(w)
        return sorted(out)

    def level_for(proj, role):
        v = (proj.get("required") or {}).get(role)
        return int((v.get("level") if isinstance(v, dict) else v) or 0)

    def no_stand_in(jid, proj, role, squad, w0, as_):
        """Why nobody else with the skill could take `role` for any start in [w0, as_)."""
        lvl = level_for(proj, role)
        facts = []
        for cid, c in a_people.items():
            if cid in squad or int((c.get("skills") or {}).get(role) or 0) < lvl:
                continue
            free_at = next((s for s in range(w0, as_)
                            if not blocked_weeks(jid, cid, role, s, s + 1)), None)
            if free_at is not None:
                facts.append(f"{c['name']} was free from {_wk(free_at)}")
            else:
                facts.extend(blockers(jid, [(cid, role)], w0, as_))
        if not facts:
            return f"nobody else has {role} ≥ {lvl}"
        return f"other {role} ≥ {lvl}: " + " · ".join(facts[:4])

    def blockers(jid, pairs, s0, s1):
        """Leave / not-started / busy facts behind the blocked weeks of each
        (person, role) for project starts in [s0, s1). A short leave that only
        needs cover is not a blocker and is not listed."""
        leave_bits: list[str] = []
        busy: dict[str, list[str]] = {}
        for pid, role in pairs:
            weeks = blocked_weeks(jid, pid, role, s0, s1)
            person = people_obj.get(pid)
            if not weeks or not person:
                continue
            for a, b in person.leave:
                if any(a <= TODAY + timedelta(weeks=w, days=4) and b >= TODAY + timedelta(weeks=w)
                       for w in weeks):
                    back = b + timedelta(days=1)
                    while back.weekday() >= 5:
                        back += timedelta(days=1)
                    bit = f"{person.name} on leave {_md(a)} → {_md(b)} (back {_md(back)})"
                    if bit not in leave_bits:
                        leave_bits.append(bit)
            af = person.available_from
            if af and any(TODAY + timedelta(weeks=w) < af for w in weeks):
                bit = f"{person.name} starts {_md(af)} (new hire)"
                if bit not in leave_bits:
                    leave_bits.append(bit)
            for w in weeks:
                for o in others(a_comm, pid, w, jid):
                    busy.setdefault(o, [])
                    if pid not in busy[o]:
                        busy[o].append(pid)
        busy_bits = []
        for o, who in sorted(busy.items(),
                             key=lambda kv: end_week(a_asg[kv[0]], a_proj[kv[0]])
                             if kv[0] in a_asg and kv[0] in a_proj else 0):
            till = (f" until {_wk_fri(end_week(a_asg[o], a_proj[o]) - 1)}"
                    if o in a_asg and o in a_proj else "")
            busy_bits.append(f"{', '.join(pname(p) for p in who)} busy on “{jname(o)}”{till}")
        return leave_bits + busy_bits

    out: dict[str, str] = {}
    for jid, proj in a_proj.items():
        if jid not in b_proj:
            continue  # new package — explained in its own table
        bits: list[str] = []
        b_dl, a_dl = b_proj[jid].get("deadline"), proj.get("deadline")
        if b_dl and a_dl and b_dl != a_dl:
            gap = (date.fromisoformat(a_dl) - date.fromisoformat(b_dl)).days
            if gap < 0:
                bits.append(f"deadline pulled in by {-gap} days ({_md(b_dl)} → {_md(a_dl)})")
            else:
                bits.append(f"deadline extended by {gap} days ({_md(b_dl)} → {_md(a_dl)})")

        ba, aa = b_asg.get(jid), a_asg.get(jid)
        if jid in a_un:
            if jid not in b_un:
                bits.append("unstaffable: " + (a_un[jid].get("reason") or ""))
        elif aa and not ba:
            bits.append("now staffed (was unstaffable before)")
        elif aa and ba:
            bs, as_ = int(ba["start_week"]), int(aa["start_week"])
            dur = int(proj.get("duration_weeks") or 0)
            b_m, a_m = members(ba), members(aa)
            a_pids = [p for ps in a_m.values() for p in ps]
            if as_ > bs:
                w0 = bs
                for dep in proj.get("depends_on") or []:
                    da, dp = a_asg.get(dep), a_proj.get(dep)
                    if not da or not dp:
                        bits.append(f"waiting on “{jname(dep)}” (unstaffed)")
                        continue
                    d_end = end_week(da, dp)
                    db = b_asg.get(dep)
                    b_end = end_week(db, b_proj.get(dep) or dp) if db else 0
                    if d_end > bs and d_end > b_end:
                        bits.append(f"waiting for “{jname(dep)}” to finish ({_wk_fri(d_end - 1)})")
                        w0 = max(w0, d_end)
                # People kept on the squad are what pinned the start; newcomers
                # (e.g. a swapped-in engineer) did not cause the delay.
                kept = {r: [p for p in a_m.get(r, []) if p in b_m.get(r, [])] for r in a_m}
                kept = {r: ps for r, ps in kept.items() if ps}
                suspects = ([(p, r) for r, ps in kept.items() for p in ps]
                            or [(p, r) for r, ps in a_m.items() for p in ps])
                found = blockers(jid, suspects, w0, as_)
                blocked = [(r, p) for p, r in suspects if kept and blocked_weeks(jid, p, r, w0, as_)]
                head = f"start pushed {_wk(bs)} → {_wk(as_)}"
                if blocked and dur > 1:
                    who = " and ".join(f"{pname(p)} ({r})" for r, p in blocked)
                    head += f": needs {who} free {dur} weeks in a row"
                bits.append(head)
                bits.extend(found)
                squad_ids = set(a_pids) | {p for ps in b_m.values() for p in ps}
                for r in dict.fromkeys(r for r, _ in blocked):
                    bits.append(no_stand_in(jid, proj, r, squad_ids, w0, as_))
                if not found and w0 == bs:
                    bits.append("solver pushed it back to cut total late days across the plan")
            elif as_ < bs:
                bits.append(f"starts earlier {_wk(bs)} → {_wk(as_)}")
                # who was tied up in the new window before, and by what
                freed: dict[str, list[str]] = {}
                for pid in a_pids:
                    for w in range(as_, as_ + dur):
                        for o in others(b_comm, pid, w, jid):
                            if o not in others(a_comm, pid, w, jid):
                                freed.setdefault(o, [])
                                if pid not in freed[o]:
                                    freed[o].append(pid)
                for o, who in freed.items():
                    bits.append(f"{', '.join(pname(p) for p in who)} freed up because “{jname(o)}” was rescheduled")
            for role in sorted(set(b_m) | set(a_m)):
                old = [p for p in b_m.get(role, []) if p not in a_m.get(role, [])]
                new = [p for p in a_m.get(role, []) if p not in b_m.get(role, [])]
                for i in range(max(len(old), len(new))):
                    o = old[i] if i < len(old) else None
                    n = new[i] if i < len(new) else None
                    why = []
                    if o and o not in a_people:
                        why.append(f"{pname(o)} was removed")
                    elif o:
                        why.extend(blockers(jid, [(o, role)], as_, as_ + 1))
                    if n and n not in b_people:
                        why.append(f"{pname(n)} is a new hire")
                    swap = f"{role}: {pname(o) if o else '—'} → {pname(n) if n else '—'}"
                    bits.append(swap + (f" ({'; '.join(why)})" if why else ""))
        if bits:
            out[jid] = "; ".join(bits)
    return out


def _delta_causes(before: dict, after: dict, reasons: dict[str, str] | None = None,
                  changed: set[str] | None = None) -> list[str]:
    """One line per changed old project: what moved and why."""
    reasons = _change_reasons(before, after) if reasons is None else reasons
    b_risk = {r["project_id"]: r for r in before.get("risks") or []}
    a_risk = {r["project_id"]: r for r in after.get("risks") or []}
    a_un = {u["project_id"] for u in after.get("unstaffable") or []}
    names = {p["id"]: p["name"] for p in before.get("projects") or []}
    lines = []
    for jid in sorted(changed if changed is not None else reasons, key=lambda j: names.get(j, j)):
        if jid not in names:
            continue
        b_late, a_late = _late_days(b_risk.get(jid)), _late_days(a_risk.get(jid))
        if jid in a_un:
            head = "unstaffable"
        elif a_late > b_late:
            head = f"{a_late - b_late} more days late"
        elif a_late < b_late:
            head = "no longer late" if a_late == 0 else f"{b_late - a_late} fewer days late"
        else:
            head = "changed"
        why = reasons.get(jid)
        lines.append(f"“{names[jid]}” {head}" + (f" — {why}" if why else ""))
    return lines


def _capacity_notes(before: dict, after: dict, new_ids: set) -> list[str]:
    notes: list[str] = []
    b_u = {u["id"]: u for u in before.get("utilization") or []}
    for u in after.get("utilization") or []:
        bu = b_u.get(u["id"])
        if not bu:
            notes.append(f"{u['name']} (new) — load {round(u['util'] * 100)}%")
            continue
        if u["overloaded"]:
            notes.append(f"{u['name']} over capacity ({round(u['util'] * 100)}%)")
        elif u["util"] - bu["util"] >= 0.12:
            notes.append(
                f"{u['name']}: load {round(bu['util'] * 100)}% → {round(u['util'] * 100)}%"
            )
    for u in after.get("unstaffable") or []:
        if u["project_id"] in new_ids:
            notes.append(f"“{u['name']}” unstaffable: {u['reason']}")
    return notes


def _why_package_late(asg: dict, after: dict, prepared: list) -> str:
    """Blame the scarce role or a late dependency — not a free PM on the squad."""
    pid = asg.get("project_id")
    proj = next((p for p in (prepared or after.get("projects") or []) if p.get("id") == pid), None)
    names = {p["id"]: p["name"] for p in after.get("projects") or []}
    a_asg = {a["project_id"]: a for a in after.get("assignments") or []}
    a_proj = {p["id"]: p for p in after.get("projects") or []}
    start = int(asg.get("start_week") or 0)

    dep_bits = []
    dep_finish_w = 0
    for dep in (proj or {}).get("depends_on") or []:
        depn = names.get(dep, dep)
        dep_a = a_asg.get(dep)
        dep_dur = int((a_proj.get(dep) or {}).get("duration_weeks") or 0)
        if dep_a is not None and dep_dur:
            dep_finish_w = max(dep_finish_w, int(dep_a["start_week"]) + dep_dur)
        dr = next((r for r in after.get("risks") or [] if r["project_id"] == dep), None)
        if dr and dr.get("risk") == "late":
            dep_bits.append(f"depends on “{depn}” (late {abs(dr.get('days_left') or 0)} days)")
        elif any(u["project_id"] == dep for u in after.get("unstaffable") or []):
            dep_bits.append(f"depends on “{depn}”, which is unstaffed")
    if dep_bits and start <= dep_finish_w:
        return "; ".join(dep_bits)

    people_by_id = {p["id"]: p for p in after.get("people") or []}
    util = {u["id"]: u for u in after.get("utilization") or []}
    week_dates = after.get("week_dates") or []
    required = (proj or {}).get("required") or {}
    scored: list[tuple] = []
    for role_block in asg.get("squad") or []:
        role = role_block.get("role")
        spec = required.get(role) if isinstance(required.get(role), dict) else {}
        lvl = int(spec.get("level") or (required.get(role) if isinstance(required.get(role), int) else 1))
        qualified = [
            p for p in people_by_id.values()
            if int((p.get("skills") or {}).get(role) or 0) >= lvl
        ]
        unique = len(qualified) == 1
        for pname in role_block.get("people") or []:
            person = next((p for p in people_by_id.values() if p.get("name") == pname), None)
            if not person:
                continue
            u = util.get(person["id"]) or {}
            committed = after.get("committed", {}).get(person["id"]) or {}
            lv_weeks = _person_leave_weeks(person)
            busy_before = [
                int(w) for w in committed
                if dep_finish_w <= int(w) < start and committed[w]
            ] + [w for w in lv_weeks if dep_finish_w <= w < start]
            first_free = (max(busy_before) + 1) if busy_before else 0
            # Available strictly before start — not the reason the package waited until start.
            if first_free < start:
                continue
            slack = u.get("util", 0) < 0.45
            if slack and not unique:
                continue
            when = ""
            if first_free < len(week_dates):
                when = _md(week_dates[first_free])
            score = (10 if unique else 0) + int((u.get("util") or 0) * 10)
            text = f"waiting for {pname} until {when}" if when else f"waiting for {pname}"
            if unique:
                text += f" (only {pname} has {role}≥{lvl})"
            scored.append((score, text))
    scored.sort(reverse=True)
    if scored:
        if dep_bits:
            return "; ".join(dep_bits) + "; " + scored[0][1]
        return scored[0][1]
    if dep_bits:
        return "; ".join(dep_bits)
    dur = (proj or {}).get("duration_weeks")
    if start == 0 and dur:
        return f"needs {dur} weeks of work"
    return "no free slot before the deadline"



def _plan_cost(result: dict) -> int:
    """Total late days, an unstaffed project counting as 1000."""
    return (sum(_late_days(r) for r in result.get("risks") or [])
            + 1000 * len(result.get("unstaffable") or []))


def _build_with_additions(people, projects, pins, sticky, freeze, new_ids: set,
                          keep_squads: bool = False) -> tuple[dict, list[str]]:
    """Solve with the approved plan frozen; if a NEW project then ends up late
    or unstaffed, unlock the frozen projects that have not started yet and hold
    someone qualified for a role it needs, and keep that plan only if total
    lateness drops. Returns (result, notes naming the moved projects)."""
    first = build_result(people, projects, prev_assignments=sticky, pins=pins,
                         freeze=freeze, keep_squads=keep_squads)
    if not freeze or not new_ids:
        return first, []
    stuck = [r for r in first["risks"] if r["project_id"] in new_ids and r["risk"] in ("late", "unstaffable")]
    if not stuck:
        return first, []
    pmeta = {p.id: p for p in projects}
    need = {(role, pmeta[r["project_id"]].level_for(role))
            for r in stuck for role in pmeta[r["project_id"]].roles()}
    qualified = {p.id for p in people for role, lvl in need if p.level(role) >= lvl}
    # running projects (start this week) never move
    keep = [a for a in freeze
            if int(a.start_week) == 0
            or not qualified & {pid for pids in a.squad.values() for pid in (pids if isinstance(pids, list) else [pids])}]
    if len(keep) == len(freeze):
        return first, []
    second = build_result(people, projects, prev_assignments=sticky, pins=pins,
                          freeze=keep, keep_squads=keep_squads)
    if _plan_cost(second) >= _plan_cost(first):
        return first, []
    b = {a.project_id: (a.start_week, a.squad) for a in freeze}
    moved = [pmeta[a["project_id"]].name for a in second["assignments"]
             if a["project_id"] in b and (a["start_week"], a["squad"]) != b[a["project_id"]]]
    note = ("To fit the new work, existing projects that have not started yet were rescheduled: "
            + ", ".join(f"“{n}”" for n in moved) + f" (total late days {_plan_cost(first)} → {_plan_cost(second)})")
    return second, [note] if moved else []


def _scenario_options(before: dict, people, projects, pins, prev, selective) -> list[dict]:
    """Solve one scenario three ways so the manager picks the trade-off.

    Each option carries the replan flags that reproduce it on Apply, its
    before/after delta and a few comparable numbers. Options that land on
    the same plan as an earlier one are dropped."""
    variants = [
        ("least", "Least disruption",
         "Only projects the change touches may move; everything else stays as approved.",
         {"freeze_existing": True}, dict(freeze=selective)),
        ("best", "Best outcome",
         "Every project may move or change people if that cuts lateness.",
         {"freeze_existing": False}, dict(freeze=None)),
        ("keep", "Keep every squad",
         "Same people on every project; only start dates may move.",
         {"freeze_existing": True, "keep_squads": True}, dict(freeze=selective, keep_squads=True)),
    ]
    b_asg = {a["project_id"]: a for a in before.get("assignments") or []}
    out, seen = [], []
    for key, label, blurb, flags, kw in variants:
        after = _public(build_result(copy.deepcopy(people), copy.deepcopy(projects),
                                     prev_assignments=prev or None, pins=pins, **kw))
        sig = sorted((a["project_id"], a["start_week"], json.dumps(a["squad"], sort_keys=True))
                     for a in after["assignments"])
        if sig in seen:
            continue
        seen.append(sig)
        swaps = 0
        for a in after["assignments"]:
            old = b_asg.get(a["project_id"])
            if not old:
                continue
            for role, pids in a["squad"].items():
                swaps += len(set(pids) - set(old["squad"].get(role) or []))
        delta = _diff(before, after)
        out.append({
            "key": key, "label": label, "blurb": blurb, "apply": flags,
            "late_days": sum(_late_days(r) for r in after.get("risks") or []),
            "unstaffable": len(after.get("unstaffable") or []),
            "changed": sum(1 for p in delta["projects"] if p["changed"] and not p["is_new"]),
            "swaps": swaps,
            "cover_days": sum(u.get("cover_days") or 0 for u in after.get("utilization") or []),
            "delta": delta,
        })
    return out


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

    def _send(self, code: int, body: bytes, ctype: str,
              extra_headers: dict[str, str] | None = None) -> None:
        if getattr(self, "_sent", False):
            return
        try:
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            for hk, hv in (extra_headers or {}).items():
                self.send_header(hk, str(hv))
            cookie = getattr(self, "_set_cookie", None)
            if cookie:
                self.send_header("Set-Cookie", cookie)
            self.end_headers()
            self.wfile.write(body)
            self._sent = True
        except _DISCONNECT:
            self.close_connection = True
            self._sent = True

    def _json(self, obj: dict) -> None:
        self._send(200, json.dumps(obj, ensure_ascii=False).encode(), "application/json")

    def _json_status(self, code: int, obj: dict,
                     extra_headers: dict[str, str] | None = None) -> None:
        self._send(code, json.dumps(obj, ensure_ascii=False).encode(), "application/json",
                   extra_headers=extra_headers)

    def _rate_limited(self, ip: str) -> None:
        wait = _llm_retry_after(ip)
        self._json_status(
            429,
            {"error": f"too many AI requests — please retry in {wait} seconds (60-second window)",
             "retry_after": wait},
            extra_headers={"Retry-After": str(wait)},
        )

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
        except _DISCONNECT:
            self.close_connection = True
        except Exception:
            traceback.print_exc()
            try:
                self._json_status(500, {"error": ["server error while scheduling"]})
            except Exception:
                pass

    def _do_POST(self) -> None:
        sess = self._bind_session()
        path = urlparse(self.path).path
        if path == "/api/assign":
            # Read back the approved plan; never re-solve what was approved.
            with sess["lock"]:
                result = _saved(sess)
            self._json({**_public(result), "gen": sess["gen"]})
            return
        if path == "/api/replan":
            length = int(self.headers.get("Content-Length", 0))
            try:
                body = json.loads(self.rfile.read(length) or b"{}")
            except json.JSONDecodeError:
                self._json_status(400, {"error": ["invalid JSON"]})
                return
            if not isinstance(body, dict):
                self._json_status(400, {"error": ["body must be a JSON object"]})
                return
            preview = bool(body.get("preview"))
            with sess["lock"]:
                # a change made on a plan that has since been reset must not land on the new one
                if not preview and body.get("base_gen") not in (None, sess["gen"]):
                    self._json_status(409, {"error": ["the plan was reset — this change was not applied"]})
                    return
                errors = _validate_changes(body, sess["people"], sess["projects"])
                if errors:
                    self._json_status(400, {"error": errors})
                    return
                freeze_now = _ensure_prev(sess)
                reopt = bool(body.get("reoptimize"))
                want_freeze = False if reopt else bool(body.get("freeze_existing", True))
                freeze = _compatible_freeze(freeze_now, body, sess["projects"]) if want_freeze else None
                sticky = None if reopt else (freeze_now or None)
                keep = bool(body.get("keep_squads")) and not reopt
                if preview and body.get("options") and not reopt:
                    people = copy.deepcopy(sess["people"])
                    projects = copy.deepcopy(sess["projects"])
                    pins = list(sess["pins"])
                    try:
                        _apply_changes_to(body, people, projects, pins)
                    except ValueError as e:
                        self._json_status(400, {"error": [str(e)]})
                        return
                    before = _public(_saved(sess))
                    sel = _compatible_freeze(freeze_now, body, sess["projects"])
                    self._json({"preview": True, "reoptimize": False, "before": before,
                                "options": _scenario_options(before, people, projects, pins,
                                                             freeze_now, sel)})
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
                    before = _public(_saved(sess))
                    after_full, notes = _build_with_additions(
                        people, projects, pins, sticky, freeze, _added_ids(body), keep)
                    after = _public(after_full)
                    self._json({
                        "preview": True,
                        "reoptimize": reopt,
                        "before": before,
                        "after": after,
                        "notes": notes,
                        "delta": _diff(before, after),
                    })
                    return
                try:
                    _apply_changes_to(body, sess["people"], sess["projects"], sess["pins"])
                except ValueError as e:
                    self._json_status(400, {"error": [str(e)]})
                    return
                # same rule as the preview, so Apply saves what was reviewed
                built, notes = _build_with_additions(
                    sess["people"], sess["projects"], sess.get("pins"), sticky, freeze,
                    _added_ids(body), keep)
                result = _commit(sess, built)
            # same bytes as /api/assign unless something needs saying
            self._json({**_public(result), **({"notes": notes} if notes else {}), "gen": sess["gen"]})
            return
        if path == "/api/reset":
            with sess["lock"]:
                fresh = _fresh_state()
                sess["people"] = fresh["people"]
                sess["projects"] = fresh["projects"]
                sess["pins"] = []
                sess["prev_assignments"] = []
                sess["result"] = None
                sess["gen"] += 1
                result = _saved(sess)
            self._json({**_public(result), "gen": sess["gen"]})
            return
        if path == "/api/intake":
            length = int(self.headers.get("Content-Length", 0))
            try:
                body = json.loads(self.rfile.read(length) or b"{}")
            except json.JSONDecodeError:
                self._json_status(400, {"error": "invalid JSON"})
                return
            brief = (body.get("brief") or "") if isinstance(body, dict) else ""
            ip = _client_ip(self)
            if not _llm_allowed("intake", ip):
                self._rate_limited(ip)
                return
            try:
                out = copy.deepcopy(llm.parse_input(brief, today=TODAY))
            except Exception:
                out = {"intent": "invalid", "reason": "could not process the input"}
            if out.get("intent") == "what_if":
                with sess["lock"]:
                    resolved = llm.resolve_whatif(
                        out.get("change") or {},
                        sess["people"], sess["projects"], TODAY,
                        original_text=brief,
                    )
                out.update(resolved)
            elif out.get("intent") == "add_project" and isinstance(out.get("project"), dict):
                with sess["lock"]:
                    existing_projs = list(sess["projects"])
                dup_warns = _overlap_warnings([out["project"]], existing_projs)
                if dup_warns:
                    wlist = list(out.get("warnings") or [])
                    for w in dup_warns:
                        if w not in wlist:
                            wlist.append(w)
                    out["warnings"] = wlist
            self._json(out)
            return
        if path == "/api/decompose":
            length = int(self.headers.get("Content-Length", 0))
            try:
                body = json.loads(self.rfile.read(length) or b"{}")
            except json.JSONDecodeError:
                self._json_status(400, {"error": "invalid JSON"})
                return
            doc = (body.get("document") or body.get("brief") or "") if isinstance(body, dict) else ""
            if not str(doc).strip():
                self._json({"error": "document is empty", "tasks": [], "manual": True})
                return
            ip = _client_ip(self)
            if not _llm_allowed("intake", ip):
                self._rate_limited(ip)
                return
            out = copy.deepcopy(llm.decompose_document(str(doc), today=TODAY))
            if out.get("error") and not out.get("tasks"):
                out.setdefault("manual", True)
            if out.get("tasks"):
                with sess["lock"]:
                    existing_projs = list(sess["projects"])
                dup_warns = _overlap_warnings(out["tasks"], existing_projs)
                if dup_warns:
                    wlist = list(out.get("warnings") or [])
                    for w in dup_warns:
                        if w not in wlist:
                            wlist.append(w)
                    out["warnings"] = wlist
                for t in out["tasks"]:
                    tw = _overlap_warnings([t], existing_projs, use_llm=False)
                    if tw:
                        t_warns = list(t.get("warnings") or [])
                        for w in tw:
                            if w not in t_warns:
                                t_warns.append(w)
                        t["warnings"] = t_warns
            self._json(out)
            return
        if path == "/api/propose-staff":
            length = int(self.headers.get("Content-Length", 0))
            try:
                body = json.loads(self.rfile.read(length) or b"{}")
            except json.JSONDecodeError:
                self._json_status(400, {"error": ["invalid JSON"]})
                return
            tasks = (body.get("projects") or body.get("tasks") or []) if isinstance(body, dict) else []
            if not isinstance(tasks, list) or not tasks:
                self._json_status(400, {"error": ["no work packages to staff yet"]})
                return
            prepared = []
            for i, t in enumerate(tasks):
                if not isinstance(t, dict):
                    self._json_status(400, {"error": ["invalid work package"]})
                    return
                item = dict(t)
                item.setdefault("id", f"jdoc_{int(time.time())}_{i}")
                prepared.append(item)
            with sess["lock"]:
                errors = _validate_changes({"add_projects": prepared}, sess["people"], sess["projects"])
                if errors:
                    self._json_status(400, {"error": errors})
                    return
                people = copy.deepcopy(sess["people"])
                projects = copy.deepcopy(sess["projects"])
                pins = list(sess["pins"])
                freeze = _ensure_prev(sess)
                try:
                    _apply_changes_to({"add_projects": prepared}, people, projects, pins)
                except ValueError as e:
                    self._json_status(400, {"error": [str(e)]})
                    return
                before = _public(_saved(sess))
                after_full, relax_notes = _build_with_additions(
                    people, projects, pins, freeze, freeze, {p["id"] for p in prepared})
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
                    "late_days": _late_days(risk),
                    "done_date": (risk or {}).get("done_date"),
                })
            unstaff = [u for u in after["unstaffable"] if u["project_id"] in new_ids]
            notes = relax_notes + _capacity_notes(before, after, new_ids)
            for a in new_asg:
                if a.get("risk") == "late":
                    a["why"] = _why_package_late(a, after, prepared)
            warnings = _overlap_warnings(prepared, sess["projects"])
            notes = notes + [w for w in warnings if w not in notes]
            self._json({
                "projects": prepared,
                "assignments": new_asg,
                "unstaffable": unstaff,
                "notes": notes,
                "warnings": warnings,
                "delta": _diff(before, after),
            })
            return
        if path == "/api/explain":
            with sess["lock"]:
                people = list(sess["people"])
                projects = list(sess["projects"])
                plan = _saved(sess)["_plan"]
            # Empty plan: fixed answer, never reaches the LLM (or the rate limit).
            if not projects or (not plan.assignments and not plan.unstaffable):
                self._json({"explanation": EMPTY_PLAN_EXPLANATION})
                return
            ip = _client_ip(self)
            if not _llm_allowed("explain", ip):
                self._rate_limited(ip)
                return
            try:
                _, risks = analysis.analyze(plan, people, projects)
                text = explain_plan_llm(plan, people, projects, risks)
                self._json({"explanation": text or ""})
            except _DISCONNECT:
                self.close_connection = True
            except Exception:
                self._json({"error": "explanation failed"})
            return
        if path == "/api/suggest":
            with sess["lock"]:
                people = copy.deepcopy(sess["people"])
                projects = copy.deepcopy(sess["projects"])
                pins = list(sess.get("pins") or [])
                base = _saved(sess)
                _, risk_objs = analysis.analyze(base["_plan"], people, projects)
                ranked = suggest.suggest_fixes(
                    people, projects, TODAY,
                    prev_assignments=sess["prev_assignments"],
                    pins=pins,
                    bottlenecks=base.get("bottlenecks"),
                    risks=risk_objs,
                )
            baseline_late = sum(_late_days(r) for r in base.get("risks") or [])
            self._json({
                "baseline_late_days": baseline_late,
                "suggestions": ranked,
            })
            return
        self._send(404, b"not found", "text/plain")


    def log_message(self, fmt, *args) -> None:
        pass  # quiet access log

    def handle(self) -> None:
        try:
            super().handle()
        except _DISCONNECT:
            self.close_connection = True

    def finish(self) -> None:
        try:
            super().finish()
        except _DISCONNECT:
            pass

    def log_error(self, fmt, *args) -> None:
        sys.stderr.write("HTTP error: " + (fmt % args) + "\n")


def main() -> None:
    port = 8091
    # warm up CP-SAT on startup so the first request is not a cold start
    import optimizer
    if not optimizer.has_cp_sat():
        sys.stderr.write(
            "WARNING: OR-Tools is missing — falling back to the greedy scheduler; results DIFFER from CP-SAT.\n"
            "  Run with: .venv/bin/python web.py\n"
        )
    try:
        if optimizer.has_cp_sat():
            warm = _fresh_state()
            optimizer.cp_sat_assign(warm["people"], warm["projects"], TODAY)
    except Exception:
        pass
    print(f"squadflow UI → http://localhost:{port}")
    ThreadingHTTPServer(("0.0.0.0", port), Handler).serve_forever()


if __name__ == "__main__":
    main()
