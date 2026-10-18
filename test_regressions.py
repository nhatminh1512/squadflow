"""Regression tests for freeze, suggestions, and rate limit."""
from __future__ import annotations

from datetime import date

import analysis
import pipeline
import suggest
import web
from models import Person, Project


TODAY = date(2026, 10, 19)


def _squad_key(a):
    squad = {
        r: list(pids if isinstance(pids, list) else [pids])
        for r, pids in a.squad.items()
    }
    return (a.project_id, a.start_week, tuple(sorted((r, tuple(v)) for r, v in squad.items())))


def test_freeze_keeps_old_plan_identical():
    people = pipeline.load_people("data/people.json")
    projects = pipeline.load_projects("data/projects.json")
    before, _, _ = pipeline.run(people, projects, TODAY)
    old = {_squad_key(a) for a in before.assignments}

    extra = [
        Project.from_dict({
            "id": "jdoc_pay",
            "name": "Cổng thanh toán",
            "required": {"pm": {"level": 4, "count": 1},
                         "engineer": {"level": 4, "count": 1},
                         "qa": {"level": 3, "count": 1}},
            "duration_weeks": 4,
            "deadline": "2026-11-27",
            "priority": 5,
        }),
        Project.from_dict({
            "id": "jdoc_api",
            "name": "API khách",
            "required": {"engineer": {"level": 4, "count": 1},
                         "qa": {"level": 3, "count": 1}},
            "duration_weeks": 6,
            "deadline": "2026-12-11",
            "priority": 4,
            "depends_on": ["jdoc_pay"],
        }),
        Project.from_dict({
            "id": "jdoc_dash",
            "name": "Dashboard ACME",
            "required": {"engineer": {"level": 2, "count": 1},
                         "analyst": {"level": 2, "count": 1}},
            "duration_weeks": 3,
            "deadline": "2026-12-25",
            "priority": 2,
        }),
    ]
    after, stats, _ = pipeline.run(
        people, list(projects) + extra, TODAY,
        freeze=list(before.assignments),
    )
    new_old = {_squad_key(a) for a in after.assignments if a.project_id in {p.id for p in projects}}
    assert new_old == old, f"old plan changed: {old ^ new_old}"
    assert not any(s.overloaded for s in stats)


def test_suggest_when_late_even_without_packed_people():
    people = pipeline.load_people("data/people.json")
    projects = pipeline.load_projects("data/projects.json")
    plan, pstats, risks = pipeline.run(people, projects, TODAY)
    late_days = sum(max(0, -(r.days_left or 0)) for r in risks)
    assert late_days >= 70
    bottlenecks = analysis.find_bottlenecks(plan, people, projects, risks, pstats)
    assert bottlenecks, "expected a QA/specialist bottleneck on the late plan"
    ranked = suggest.suggest_fixes(
        people, projects, TODAY,
        prev_assignments=plan.assignments,
        pins=[],
        bottlenecks=[],  # UI used to skip when this was empty
        risks=risks,
    )
    assert ranked, "suggestions must not be empty when the plan is late"
    hire = next((r for r in ranked if "Hire a" in r["title"]), None)
    assert hire is not None
    # a real hire needs HIRE_LEAD_WEEKS before day one, so it may not help this plan
    assert "weeks to recruit" in hire["note"] and "(starts " in hire["title"], hire
    assert hire["total_late_days"] <= late_days
    # ...while the same person available today does cut lateness (the solver uses them)
    now = list(people) + [Person("qa9", "Ready QA", "qa", {"qa": 3})]
    _, _, risks_now = pipeline.run(now, projects, TODAY)
    assert sum(max(0, -(r.days_left or 0)) for r in risks_now) < late_days


def test_rate_limit_blocks_burst():
    web._llm_times.clear()
    ip = "203.0.113.9"
    allowed = sum(1 for _ in range(40) if web._llm_allowed("intake", ip))
    assert allowed == web._LLM_MAX_GLOBAL
    assert allowed < 40
    assert web._llm_allowed("explain", ip) is False
    # re-reading failed document parts keeps its own, larger budget
    assert web._llm_allowed("retry", ip) is True
    web._llm_times.clear()
    # A different IP still has budget.
    assert web._llm_allowed("explain", "198.51.100.2") is True


def test_overlap_warns_on_acme_payment_gateway():
    existing = pipeline.load_projects("data/projects.json")
    incoming = [{
        "id": "jdoc_pay",
        "name": "Cổng thanh toán",
        "required": {"pm": {"level": 4, "count": 1},
                     "engineer": {"level": 4, "count": 1},
                     "qa": {"level": 3, "count": 1}},
        "duration_weeks": 4,
        "deadline": "2026-11-27",
    }]
    warns = web._overlap_warnings(incoming, existing)
    assert warns, "should warn that Cổng thanh toán duplicates Payment gateway"
    assert any("Payment gateway" in w or "thanh toán" in w for w in warns)


def test_hire_qa_whatif_does_not_reshuffle_unrelated():
    people = pipeline.load_people("data/people.json")
    projects = pipeline.load_projects("data/projects.json")
    before, _, _ = pipeline.run(people, projects, TODAY)
    crm = next(a for a in before.assignments if a.project_id == "p5")
    freeze = web._compatible_freeze(
        list(before.assignments),
        {"add_person": {"id": "hire_qa", "name": "New QA", "role": "qa",
                        "skills": {"qa": 3}, "capacity": 1.0}},
        projects,
    )
    people2 = list(people) + [Person.from_dict({
        "id": "hire_qa", "name": "New QA", "role": "qa",
        "skills": {"qa": 3}, "capacity": 1.0,
    })]
    after, stats2, risks2 = pipeline.run(
        people2, projects, TODAY,
        prev_assignments=list(before.assignments),
        freeze=freeze,
    )
    crm2 = next(a for a in after.assignments if a.project_id == "p5")
    assert crm.squad == crm2.squad, f"CRM squad changed: {crm.squad} -> {crm2.squad}"
    assert crm.start_week == crm2.start_week
    late_after = sum(max(0, -(r.days_left or 0)) for r in risks2)
    assert late_after == 35, f"expected 35 total late days after hiring QA, got {late_after}"
    qa_stat = next(s for s in stats2 if s.person.id == "hire_qa")
    assert qa_stat.util > 0, "newly hired QA must be utilized in selective freeze"


def test_persisted_plan_survives_assign_and_suggest():
    """Approving new decomposed packages must persist the plan so /api/assign and /api/suggest do not reshuffle."""
    import engine
    sess = web._fresh_state()
    # Initial /api/assign
    f0 = engine.sanitize_freeze(sess["prev_assignments"], sess["people"], sess["projects"], sess["pins"], today=TODAY)
    d0 = web.build_result(sess["people"], sess["projects"], prev_assignments=sess["prev_assignments"], pins=sess["pins"], freeze=f0)
    sess["prev_assignments"] = list(d0["_plan"].assignments)
    p1_risk_before = next(r for r in d0["risks"] if r["project_id"] == "p1")
    assert p1_risk_before["days_left"] >= 0, "Payment gateway should start on time"

    # Approve 3 decomposed packages with freeze_existing=True
    extra_raw = [
        {
            "id": "jdoc_pay",
            "name": "Cổng thanh toán",
            "required": {"pm": {"level": 4, "count": 1},
                         "engineer": {"level": 4, "count": 1},
                         "qa": {"level": 3, "count": 1}},
            "duration_weeks": 4,
            "deadline": "2026-11-27",
            "priority": 5,
        },
        {
            "id": "jdoc_api",
            "name": "API khách",
            "required": {"engineer": {"level": 4, "count": 1},
                         "qa": {"level": 3, "count": 1}},
            "duration_weeks": 6,
            "deadline": "2026-12-11",
            "priority": 4,
            "depends_on": ["jdoc_pay"],
        },
        {
            "id": "jdoc_dash",
            "name": "Dashboard ACME",
            "required": {"engineer": {"level": 2, "count": 1},
                         "analyst": {"level": 2, "count": 1}},
            "duration_weeks": 3,
            "deadline": "2026-12-25",
            "priority": 2,
        },
    ]
    for p in extra_raw:
        sess["projects"].append(Project.from_dict(p))
    frz = web._compatible_freeze(sess["prev_assignments"], {"add_projects": extra_raw}, sess["projects"])
    d_approved = web.build_result(sess["people"], sess["projects"], prev_assignments=sess["prev_assignments"], pins=sess["pins"], freeze=frz)
    sess["prev_assignments"] = list(d_approved["_plan"].assignments)
    late_approved = sum(max(0, -(r["days_left"] or 0)) for r in d_approved["risks"])
    old_map_approved = {
        a["project_id"]: (a["start_week"], a["squad"])
        for a in d_approved["assignments"]
        if not a["project_id"].startswith("jdoc_")
    }

    # Simulate background /api/suggest call followed by page reload /api/assign
    f_sug = engine.sanitize_freeze(sess["prev_assignments"], sess["people"], sess["projects"], sess["pins"], today=TODAY)
    d_sug = web.build_result(sess["people"], sess["projects"], prev_assignments=sess["prev_assignments"], pins=sess["pins"], freeze=f_sug)
    sess["prev_assignments"] = list(d_sug["_plan"].assignments)

    f_asg = engine.sanitize_freeze(sess["prev_assignments"], sess["people"], sess["projects"], sess["pins"], today=TODAY)
    d_reload = web.build_result(sess["people"], sess["projects"], prev_assignments=sess["prev_assignments"], pins=sess["pins"], freeze=f_asg)
    sess["prev_assignments"] = list(d_reload["_plan"].assignments)

    late_reload = sum(max(0, -(r["days_left"] or 0)) for r in d_reload["risks"])
    old_map_reload = {
        a["project_id"]: (a["start_week"], a["squad"])
        for a in d_reload["assignments"]
        if not a["project_id"].startswith("jdoc_")
    }
    assert old_map_reload == old_map_approved, "reload/assign reshuffled approved projects!"
    assert late_reload == late_approved, f"total late days jumped from {late_approved} to {late_reload}"


def test_bilingual_overlap_without_heuristic_match():
    existing = pipeline.load_projects("data/projects.json")
    # Different duration (6 vs 4) and different roles (only engineer) so duration/role heuristic won't trigger
    incoming = [{
        "id": "new_pg",
        "name": "Cổng thanh toán",
        "required": {"engineer": {"level": 3, "count": 1}},
        "duration_weeks": 6,
        "deadline": "2026-12-15",
    }]
    warns = web._overlap_warnings(incoming, existing, use_llm=False)
    assert warns and any("Payment gateway migration" in w for w in warns), f"expected bilingual warning, got {warns}"


def test_delta_causes_pulled_deadline_and_busy_squad():
    b_people = pipeline.load_people("data/people.json")
    b_projects = pipeline.load_projects("data/projects.json")
    before = web.build_result(b_people, b_projects)
    b_asg = list(before["_plan"].assignments)

    # Case 1: Pull ETL (p3, Data warehouse ETL) deadline earlier from 2026-11-20 to 2026-11-06
    a_projects = pipeline.load_projects("data/projects.json")
    for p in a_projects:
        if p.id == "p3":
            p.deadline = date.fromisoformat("2026-11-06")
    frz = web._compatible_freeze(b_asg, {"slip": [{"project_id": "p3", "deadline": "2026-11-06"}]}, a_projects)
    after = web.build_result(b_people, a_projects, prev_assignments=b_asg, freeze=frz)
    delta = web._diff(before, after)
    etl_causes = [c for c in delta["causes"] if "Data warehouse ETL" in c]
    assert etl_causes, f"expected cause for ETL, got {delta['causes']}"
    assert any("deadline pulled in by" in c for c in etl_causes), f"expected pulled deadline cause, got {etl_causes}"
    assert not any("“Payment gateway" in c for c in etl_causes), f"should not blame Payment gateway when deadline pulled: {etl_causes}"

    # Case 2: When ETL is delayed by 56 days (to week 12) while Carol & Grace are busy on Mobile app redesign (weeks 7..12)
    from models import Assignment
    after_asg = []
    for a in b_asg:
        if a.project_id == "p2":
            after_asg.append(Assignment("p2", 7, {"pm": ["pm1"], "engineer": ["eng3"], "analyst": ["an1"], "qa": ["qa2"]}))
        elif a.project_id == "p3":
            after_asg.append(Assignment("p3", 12, {"analyst": ["an1"], "engineer": ["eng3"]}))
        else:
            after_asg.append(Assignment(a.project_id, a.start_week, dict(a.squad)))
    after_busy = web.build_result(b_people, b_projects, freeze=after_asg)
    delta_busy = web._diff(before, after_busy)
    etl_busy_causes = [c for c in delta_busy["causes"] if "Data warehouse ETL" in c]
    assert etl_busy_causes, f"expected busy squad cause for ETL, got {delta_busy['causes']}"
    assert any("Mobile app redesign" in c and "Carol" in c and "Grace" in c for c in etl_busy_causes), (
        f"expected Carol & Grace busy on Mobile app redesign, got: {etl_busy_causes}"
    )
    assert not any("Payment gateway" in c for c in etl_busy_causes), (
        f"should not blame Payment gateway when Mobile app redesign is the blocker: {etl_busy_causes}"
    )


def test_empty_plan_explanation_and_429_retry_after():
    import explain
    from models import Plan
    empty_plan = Plan(start_date=TODAY, horizon_weeks=12, assignments=[], unstaffable=[])
    text = explain.explain_plan_llm(empty_plan, [], [], [])
    assert text == explain.EMPTY_PLAN_EXPLANATION
    assert "log" not in text.lower()

    web._llm_times.clear()
    ip = "192.0.2.55"
    for _ in range(web._LLM_MAX_GLOBAL):
        assert web._llm_allowed("explain", ip) is True
    assert web._llm_allowed("explain", ip) is False
    retry_after = web._llm_retry_after(ip)
    assert 1 <= retry_after <= int(web._LLM_WINDOW_S)


class _Client:
    """In-process HTTP server + one cookie, so tests hit the real handlers."""

    def __init__(self):
        import threading
        from http.server import ThreadingHTTPServer
        self.srv = ThreadingHTTPServer(("127.0.0.1", 0), web.Handler)
        threading.Thread(target=self.srv.serve_forever, daemon=True).start()
        self.cookie = None

    def post(self, path, body=None):
        import http.client
        import json
        c = http.client.HTTPConnection("127.0.0.1", self.srv.server_address[1], timeout=120)
        h = {"Content-Type": "application/json"}
        if self.cookie:
            h["Cookie"] = self.cookie
        c.request("POST", path, json.dumps(body or {}), h)
        r = c.getresponse()
        raw = r.read()
        if r.getheader("Set-Cookie"):
            self.cookie = r.getheader("Set-Cookie").split(";")[0]
        return raw, json.loads(raw)

    def close(self):
        self.srv.shutdown()


ACME = [
    {"id": "jdoc_pay", "name": "Cổng thanh toán",
     "required": {"pm": {"level": 4, "count": 1}, "engineer": {"level": 4, "count": 1},
                  "qa": {"level": 3, "count": 1}},
     "duration_weeks": 4, "deadline": "2026-11-27", "priority": 5},
    {"id": "jdoc_api", "name": "API khách",
     "required": {"engineer": {"level": 4, "count": 1}, "qa": {"level": 3, "count": 1}},
     "duration_weeks": 6, "deadline": "2026-12-11", "priority": 4, "depends_on": ["jdoc_pay"]},
    {"id": "jdoc_dash", "name": "Dashboard ACME",
     "required": {"engineer": {"level": 2, "count": 1}, "analyst": {"level": 2, "count": 1}},
     "duration_weeks": 3, "deadline": "2026-12-25", "priority": 2},
]


def test_approve_then_assign_is_byte_identical():
    """Approved plan is saved: assign / suggest / reload return it byte-for-byte,
    and the 'Before' column of a full re-optimize is that saved plan."""
    cl = _Client()
    try:
        cl.post("/api/assign")
        _, prop = cl.post("/api/propose-staff", {"projects": ACME})
        approved_raw, approved = cl.post("/api/replan",
                                         {"add_projects": prop["projects"], "freeze_existing": True})
        assert "assignments" in approved, approved
        cl.post("/api/suggest")
        again_raw, again = cl.post("/api/assign")
        # the approve response may add one-off "notes"/"moved" (projects moved
        # to fit); the saved plan itself is identical
        assert again == {k: v for k, v in approved.items() if k not in ("notes", "moved")}, \
            "assign after approve must return the saved plan unchanged"
        again2_raw, _ = cl.post("/api/assign")
        assert again2_raw == again_raw
        _, reopt = cl.post("/api/replan", {"reoptimize": True, "freeze_existing": False, "preview": True})
        assert reopt["before"]["assignments"] == approved["assignments"]
        assert reopt["before"]["risks"] == approved["risks"]
    finally:
        cl.close()


def test_hire_qa4_whatif_moves_only_late_projects():
    cl = _Client()
    try:
        _, base = cl.post("/api/assign")
        _, r = cl.post("/api/replan", {
            "add_person": {"id": "newqa", "name": "New QA", "role": "qa",
                           "skills": {"qa": 4}, "capacity": 1},
            "freeze_existing": True, "preview": True,
        })
        late_before = sum(max(0, -(x["days_left"] or 0)) for x in base["risks"] if x["risk"] == "late")
        late_after = sum(max(0, -(x["days_left"] or 0)) for x in r["after"]["risks"] if x["risk"] == "late")
        assert late_after < late_before, (late_before, late_after)
        util = {u["id"]: u["util"] for u in r["after"]["utilization"]}
        assert util["newqa"] > 0, "the new QA must take work"
        on_time = {x["project_id"] for x in base["risks"] if x["risk"] != "late"}
        b_asg = {a["project_id"]: a for a in base["assignments"]}
        for a in r["after"]["assignments"]:
            if a["project_id"] in on_time:
                assert a == b_asg[a["project_id"]], f"on-time project reshuffled: {a['name']}"
    finally:
        cl.close()


def test_every_changed_project_has_a_factual_reason():
    """Ivy on leave: each changed project gets one reason, with real dates."""
    cl = _Client()
    try:
        cl.post("/api/assign")
        _, r = cl.post("/api/replan", {
            "leave": [{"person_id": "qa1", "from": "2026-10-26", "to": "2026-11-08"}],
            "freeze_existing": True, "preview": True,
        })
        changed = [p for p in r["delta"]["projects"] if p["changed"] and not p["is_new"]]
        assert changed
        assert all(p["reason"] for p in changed), [p["name"] for p in changed if not p["reason"]]
        assert len(r["delta"]["causes"]) == len(changed)
        pay = next(p for p in changed if p["id"] == "p1")
        assert "Ivy on leave Oct 26 → Nov 8" in pay["reason"], pay["reason"]
        assert "Oct 19" not in pay["reason"].split("start pushed")[-1].split(";", 1)[-1]
    finally:
        cl.close()


def test_pushed_start_names_the_binding_people_not_newcomers():
    """Ben off a whole week + Jack's leave push CRM 5 weeks: the reason must
    name Ben and Jack as the constraint and not list the swapped-in
    engineer's other work as if it were a cause."""
    from models import Assignment
    people = pipeline.load_people("data/people.json")
    people.append(Person(id="qa3", name="New QA", role="qa", skills={"qa": 3}))
    projects = pipeline.load_projects("data/projects.json")
    base = {
        "p1": Assignment("p1", 0, {"pm": ["pm1"], "engineer": ["eng1"], "qa": ["qa1"]}),
        "p4": Assignment("p4", 0, {"pm": ["pm1"], "engineer": ["eng1"], "qa": ["qa3"]}),
        "p6": Assignment("p6", 0, {"engineer": ["eng4"], "analyst": ["an2"]}),
        "p7": Assignment("p7", 2, {"engineer": ["eng1"], "qa": ["qa3"]}),
        "p3": Assignment("p3", 4, {"analyst": ["an1"], "engineer": ["eng3"]}),
        "p2": Assignment("p2", 7, {"pm": ["pm1"], "engineer": ["eng2"], "analyst": ["an1"], "qa": ["qa2"]}),
    }
    before = web.build_result(people, projects, freeze=list(base.values()) + [
        Assignment("p5", 0, {"pm": ["pm2"], "engineer": ["eng3"], "qa": ["qa2"]})])
    ben = next(p for p in people if p.id == "pm2")
    ben.leave = [(date(2026, 10, 19), date(2026, 10, 23))]
    after = web.build_result(people, projects, freeze=list(base.values()) + [
        Assignment("p5", 5, {"pm": ["pm2"], "engineer": ["eng1"], "qa": ["qa2"]})])
    crm = next(p for p in web._diff(before, after)["projects"] if p["id"] == "p5")
    why = crm["reason"]
    assert "start pushed Oct 19 → Nov 23: needs Ben (pm) and Jack (qa) free 3 weeks in a row" in why, why
    assert "Ben on leave Oct 19 → Oct 23 (back Oct 26)" in why, why
    assert "Jack on leave Nov 9 → Nov 20" in why, why
    assert "Eve busy" not in why, why
    assert "engineer: Grace → Eve (Grace busy on “Data warehouse ETL”" in why, why
    assert "other pm ≥ 3: Alice" in why and "other qa ≥ 2:" in why, why


def test_short_leave_is_covered_not_a_whole_week_lost():
    """Ben off 2 days: CRM keeps its start (someone covers 2 days);
    a whole week off still blocks the week."""
    people = pipeline.load_people("data/people.json")
    projects = pipeline.load_projects("data/projects.json")
    plan, _, _ = pipeline.run(people, projects, TODAY)
    crm = next(a for a in plan.assignments if a.project_id == "p5")
    assert crm.start_week == 0 and crm.squad["pm"] == ["pm2"]
    ben = next(p for p in people if p.id == "pm2")
    ben.leave = [(date(2026, 10, 19), date(2026, 10, 20))]
    frz = web._compatible_freeze(plan.assignments, {"leave": [{"person_id": "pm2"}]}, projects)
    plan2, stats, _ = pipeline.run(people, projects, TODAY, prev_assignments=plan.assignments, freeze=frz)
    crm2 = next(a for a in plan2.assignments if a.project_id == "p5")
    assert crm2.start_week == 0, crm2
    if crm2.squad["pm"] == ["pm2"]:
        assert next(s for s in stats if s.person.id == "pm2").cover_days == 2

    from engine import week_capacity
    assert week_capacity(ben, TODAY, 0) == (300, 200)      # 3 days present, 2 coverable
    ben.leave = [(date(2026, 10, 19), date(2026, 10, 23))]
    assert week_capacity(ben, TODAY, 0) == (0, 200)        # a full week off: 500 needed > 200


def test_role_phase_and_effort_only_book_their_weeks():
    from models import Project
    j = Project.from_dict({"id": "x", "name": "X", "duration_weeks": 5, "deadline": "2026-12-31",
                           "required": {"pm": {"level": 3, "effort": 20},
                                        "qa": {"level": 2, "start": 3, "weeks": 2}}})
    assert j.phase_for("pm") == (0, 5) and j.effort_for("pm") == 20
    assert list(j.role_weeks("qa", 2)) == [5, 6]
    people = [Person("pm", "P", "pm", {"pm": 3}), Person("q", "Q", "qa", {"qa": 2})]
    plan, _, _ = pipeline.run(people, [j], TODAY)
    assert sorted(plan.committed["q"]) == [3, 4]                   # QA only in its phase
    assert plan.load["pm"] == {w: 100 for w in range(5)}           # 20% = 1 day a week
    # five 20% PM roles fit one full-time PM at the same time
    many = [Project.from_dict({"id": f"m{i}", "name": f"M{i}", "duration_weeks": 2,
                               "deadline": "2026-11-06", "required": {"pm": {"level": 3, "effort": 20}}})
            for i in range(5)]
    plan, _, risks = pipeline.run(people[:1], many, TODAY)
    assert all(a.start_week == 0 for a in plan.assignments) and len(plan.assignments) == 5


def test_hire_cannot_work_before_start_date():
    people = pipeline.load_people("data/people.json")
    people.append(Person("qa3", "New QA", "qa", {"qa": 3}, 1.0, [], date(2026, 11, 30)))
    plan, _, _ = pipeline.run(people, pipeline.load_projects("data/projects.json"), TODAY)
    assert all(w >= 6 for w in plan.committed.get("qa3", {})), plan.committed.get("qa3")


def test_long_document_is_read_in_parts_and_merged():
    """A 30k-char spec: each part goes to the model with the outline, the same
    deliverable named in two parts becomes one package, dependencies resolve
    across parts, and stated stage dates become milestones (never guessed)."""
    import llm
    doc = ("REQUIREMENTS DOCUMENT\n\n1. Background\n" + "Context sentence. " * 600
           + "\n\n2. Payment gateway\nPM level 4, 2 senior engineers, QA 3, 12 weeks. "
           "UAT complete by 15/12. Go-live 20/01/2027.\n" + "Gateway detail. " * 600
           + "\n\n3. Mobile app\nEngineer 4 and QA 3 for 10 weeks, after the payment gateway.\n"
           + "Screen detail. " * 600 + "\n\n4. Payment gateway hardening\n" + "More gateway. " * 300)
    seen = []

    def fake(messages, **kw):
        part = messages[-1]["content"]
        seen.append(part)
        tasks = []
        if "UAT complete" in part:
            tasks.append({"name": "Payment gateway", "duration_weeks": 12, "deadline": "2027-01-20",
                          "required": {"pm": {"level": 4}, "engineer": {"level": 4, "count": 2}, "qa": {"level": 3}}})
        if "Screen detail" in part and "Mobile app" in part:
            tasks.append({"name": "Mobile app", "duration_weeks": 10, "deadline": "2027-03-01",
                          "required": {"engineer": {"level": 4}, "qa": {"level": 3}},
                          "depends_on": ["Payment gateway"]})
        if "More gateway" in part:
            tasks.append({"name": "payment gateway", "duration_weeks": 14, "deadline": "2027-02-01",
                          "required": {"engineer": {"level": 5}}})
        return {"summary": "part", "item_count": len(tasks), "tasks": tasks}

    orig, llm.complete_json = llm.complete_json, fake
    llm._CACHE.clear()
    try:
        out = llm.decompose_document(doc, today=TODAY)
    finally:
        llm.complete_json = orig
    assert out["parts"] >= 3 and len(seen) == out["parts"], out["parts"]
    assert all("Outline of the whole document" in p for p in seen)
    names = [t["name"] for t in out["tasks"]]
    assert names == ["Payment gateway", "Mobile app"], names          # the two gateway mentions merged
    gw, app = out["tasks"]
    assert gw["required"]["engineer"] == {"level": 4, "count": 2}     # first/own reading wins, never maxed
    assert gw["required"]["pm"] == {"level": 4, "count": 1}
    assert gw["duration_weeks"] == 12 and gw["deadline"] == "2027-01-20"
    assert app["depends_on"] == [gw["id"]]
    assert gw["milestones"] == {"uat": "2026-12-15", "deploy": "2027-01-20"}, gw.get("milestones")
    assert "2. Payment gateway" in gw["source"] and "Long document" in out["warnings"][0]


def test_stage_milestones_are_read_not_guessed():
    import llm
    cases = {
        "Hoàn thành UAT trước 18/12/2026, go-live 15/01/2027.": {"uat": "2026-12-18", "deploy": "2027-01-15"},
        "Go-live 20/01/2027; UAT by 15/12": {"deploy": "2027-01-20", "uat": "2026-12-15"},
        "SIT 1/12 và UAT 10/12": {"sit": "2026-12-01", "uat": "2026-12-10"},
        "Ký duyệt yêu cầu (BRD) ngày 30/10": {"requirements": "2026-10-30"},
        "UAT will be scheduled later": {},
        "Deploy scripts must be idempotent": {},
    }
    for text, want in cases.items():
        assert llm._stage_milestones(text, TODAY) == want, (text, llm._stage_milestones(text, TODAY))


def test_waterfall_stages_book_roles_only_when_needed():
    from models import Project
    import methodology
    wf = Project.from_dict({"id": "w", "name": "W", "duration_weeks": 16, "deadline": "2027-03-01",
                            "method": "waterfall", "milestones": {"uat": "2027-01-29"},
                            "required": {"pm": {"level": 3, "effort": 20}, "analyst": {"level": 3},
                                         "engineer": {"level": 3}, "qa": {"level": 2, "start": 13, "weeks": 2}}})
    names = [st["stage"] for st in wf.stages]
    assert names == ["requirements", "gate", "design", "build", "sit", "uat", "gate", "deploy"], names
    assert sum(st["weeks"] for st in wf.stages) == 16
    build = next(st for st in wf.stages if st["stage"] == "build")
    an = wf.role_effort("analyst", 0)
    assert all(w not in an for w in range(build["start"], build["start"] + build["weeks"]))  # BA free in build
    assert wf.role_effort("pm", 0)[0] == 20                         # stated effort replaces the template
    assert sorted(wf.role_effort("qa", 0)) == [13, 14]              # stated weeks win over the template
    people = [Person("p", "P", "pm", {"pm": 3}), Person("a", "A", "analyst", {"analyst": 3}),
              Person("e", "E", "engineer", {"engineer": 3}), Person("q", "Q", "qa", {"qa": 2})]
    side = Project.from_dict({"id": "s", "name": "Side", "duration_weeks": 3, "deadline": "2026-12-31",
                              "required": {"analyst": {"level": 3}}})
    plan, _, risks = pipeline.run(people, [wf, side], TODAY)
    s_start = next(a.start_week for a in plan.assignments if a.project_id == "s")
    assert s_start < 10, s_start                                      # analyst reused while W builds
    r = next(x for x in risks if x.project.id == "w")
    assert r.milestones and r.milestones[0]["stage"] == "uat"
    late_ms = Project.from_dict({**{"id": "w2", "name": "W2", "duration_weeks": 16, "deadline": "2027-03-01",
                                   "method": "waterfall", "milestones": {"uat": "2026-11-06"},
                                   "required": {"engineer": {"level": 3}}}})
    _, _, risks = pipeline.run(people, [late_ms], TODAY)
    assert risks[0].risk == "late" and risks[0].milestones[0]["days_left"] < 0
    # late by the milestone while the deadline still has room: late_days is positive, never "-24d"
    assert risks[0].days_left > 0 and risks[0].late_days == -risks[0].milestones[0]["days_left"] > 0
    assert web._late_days({"risk": "late", "days_left": 24, "late_days": risks[0].late_days}) == risks[0].late_days
    for bad in ({"method": "agile"}, {"milestones": {"uat": "2026-12-01"}},
                {"method": "waterfall", "milestones": {"qa": "2026-12-01"}}):
        try:
            web._check_project({"id": "x", "name": "x", "duration_weeks": 8, "deadline": "2026-12-31",
                                "required": {"pm": 3}, **bad})
            raise AssertionError(f"accepted {bad}")
        except ValueError:
            pass


def _fake_decompose(doc, packages, cap=None):
    """Run decompose_document with a scripted model: `packages` is a list of
    task dicts; each part gets the ones whose evidence is in it."""
    import llm
    def fake(messages, **kw):
        part = messages[-1]["content"]
        return {"summary": "s", "tasks": [t for t in packages if t["evidence"][:30] in part]}
    orig, llm.complete_json = llm.complete_json, fake
    old_cap = llm.MAX_PACKAGES_SHORT
    if cap:
        llm.MAX_PACKAGES_SHORT = cap
    llm._CACHE.clear()
    try:
        return llm.decompose_document(doc, today=TODAY)
    finally:
        llm.complete_json, llm.MAX_PACKAGES_SHORT = orig, old_cap


PRD = """# Payments Platform PRD

## 1. Overview
The bank needs a modern payments platform for retail customers.

## 2. Scope
### 2.1 Payment gateway
PM level 4, 2 senior engineers, QA 3, 8 weeks. Connects Napas and card schemes.
### 2.2 Partner API
Engineer 4 and QA 3 for 6 weeks. The partner API is built on the payment gateway.
### 2.3 Security audit
One engineer level 5 for 3 weeks. The security audit must be finished before the partner API go-live.
### 2.4 Audit logging
Engineer 3 for 2 weeks: write an audit log entry for every admin action.

## 3. Release timeline
UAT complete by 15/12. Go-live 20/01/2027.

## Appendix A. Technical appendix
Audit logging format, glossary of terms and the technical appendix tables.
"""

PRD_TASKS = [
    {"name": "Payment gateway", "duration_weeks": 8, "deadline": "2027-01-20", "priority": 5,
     "required": {"pm": {"level": 4}, "engineer": {"level": 4, "count": 2}, "qa": {"level": 3}},
     "evidence": "PM level 4, 2 senior engineers, QA 3, 8 weeks."},
    {"name": "Partner API", "duration_weeks": 6, "deadline": "2027-01-20",
     "required": {"engineer": {"level": 4}, "qa": {"level": 3}}, "depends_on": ["Payment gateway"],
     "evidence": "Engineer 4 and QA 3 for 6 weeks."},
    {"name": "Security audit", "duration_weeks": 3, "deadline": "2027-01-20", "priority": 2,
     "required": {"engineer": {"level": 5}}, "depends_on": ["Partner API"],          # reversed by the model
     "evidence": "One engineer level 5 for 3 weeks."},
    {"name": "Audit logging", "duration_weeks": 2, "deadline": "2027-01-20",
     "required": {"engineer": {"level": 3}},
     "evidence": "Engineer 3 for 2 weeks: write an audit log entry for every admin action."},
    {"name": "Release timeline", "duration_weeks": 1, "deadline": "2027-01-20",
     "required": {"pm": {"level": 3}}, "evidence": "UAT complete by 15/12. Go-live 20/01/2027."},
    {"name": "Go-live of the whole platform", "duration_weeks": 2, "deadline": "2027-01-20",
     "required": {"engineer": {"level": 3}}, "evidence": "Go-live 20/01/2027."},
    {"name": "Technical appendix tables", "duration_weeks": 2, "deadline": "2027-01-20",
     "required": {"engineer": {"level": 3}}, "evidence": "Audit logging format, glossary of terms and the technical appendix tables."},
]


def test_prd_with_several_packages_gets_dates_sources_and_dependency_order():
    out = _fake_decompose(PRD, PRD_TASKS)
    by = {t["name"]: t for t in out["tasks"]}
    assert list(by) == ["Payment gateway", "Partner API", "Security audit", "Audit logging"], list(by)
    # 1: project-wide stage dates reach packages whose own text states none;
    # a shared go-live that contradicts a document dependency is not assigned
    for name in ("Partner API", "Audit logging"):
        assert by[name]["milestones"] == {"uat": "2026-12-15", "deploy": "2027-01-20"}, by[name]
    for name in ("Payment gateway", "Security audit"):
        assert "milestones" not in by[name], (name, by[name])
        assert any("after “Partner API” starts" in w for w in by[name]["warnings"]), by[name]["warnings"]
    assert "Release timeline" in out["warnings"][-1] and "appendix" in out["warnings"][-1]
    # 3: source = the package's own heading, without markdown marks
    assert by["Payment gateway"]["source"] == "2.1 Payment gateway"
    assert by["Security audit"]["source"] == "2.3 Security audit"
    # 4: "audit must be finished before the partner API go-live" beats the model's reversed link
    gw, api, audit = by["Payment gateway"]["id"], by["Partner API"]["id"], by["Security audit"]["id"]
    assert by["Security audit"]["depends_on"] == [], by["Security audit"]
    assert set(by["Partner API"]["depends_on"]) == {gw, audit}
    assert any("opposite order" in w for w in by["Security audit"]["warnings"])
    # 2: "Security audit" and "Audit logging" are different deliverables
    assert by["Security audit"]["duration_weeks"] == 3 and by["Security audit"]["required"] == {"engineer": {"level": 5, "count": 1}}


def test_roles_the_section_does_not_staff_are_removed():
    tasks = [dict(PRD_TASKS[2], required={"pm": {"level": 3}, "analyst": {"level": 3}, "engineer": {"level": 5}},
                  depends_on=[])]
    out = _fake_decompose(PRD, tasks)
    t = out["tasks"][0]
    assert t["required"] == {"engineer": {"level": 5, "count": 1}}, t["required"]
    assert any("pm removed" in w for w in t["warnings"]) and any("analyst removed" in w for w in t["warnings"])


def test_over_the_cap_keeps_the_best_supported_packages():
    out = _fake_decompose(PRD, PRD_TASKS, cap=3)
    names = [t["name"] for t in out["tasks"]]
    # all four have their own section; the audit logging package has a staffing
    # phrase ("Engineer 3") while the security audit has only stage wording, so
    # it is the one cut — and the reason carries its score
    assert names == ["Payment gateway", "Partner API", "Audit logging"], names
    assert "Security audit (over the 3-package limit, score 6" in out["warnings"][-1], out["warnings"][-1]


def test_overview_fragments_do_not_take_the_package_cap():
    """25 packages, 10 of them read under “1. Tổng quan / 1.1 / 1.2”: the four
    main 2.x packages stay in the 20 kept, and the drop reason carries a score."""
    import llm
    overview = ["Customer onboarding", "Merchant directory", "Fee calculation",
                "Fraud screening", "Settlement reports", "Refund handling",
                "Dispute tracking", "Currency conversion", "Statement export",
                "Notification centre"]
    main = ["Payment gateway", "Partner API", "Security audit", "Audit logging"]
    other = ["Ledger service", "Card vault", "Tokenisation", "Terminal fleet",
             "Batch uploader", "Reconciliation", "Chargeback API", "Merchant portal",
             "Sandbox tools", "Webhook relay", "Data warehouse"]
    lines = ["# Big PRD\n", "## 1. Tổng quan\nIntro of the whole programme.\n",
             "### 1.1 Bối cảnh\n"]
    packages = []
    for name in overview[:4]:
        ev = f"{name}. 1 engineer for 2 weeks. UAT by 15/12."
        lines.append(f"- {ev}\n")
        packages.append({"name": name, "duration_weeks": 2, "deadline": "2027-01-10",
                         "required": {"engineer": {"level": 3}}, "evidence": ev})
    lines.append("### 1.2 Mục tiêu\n")
    for name in overview[4:]:
        ev = f"{name}. 1 engineer for 2 weeks. UAT by 15/12."
        lines.append(f"- {ev}\n")
        packages.append({"name": name, "duration_weeks": 2, "deadline": "2027-01-10",
                         "required": {"engineer": {"level": 3}}, "evidence": ev})
    lines.append("## 2. Phạm vi\n")
    for i, name in enumerate(main, 1):
        ev = f"{name}. PM level 4, 2 senior engineers, QA 3, 8 weeks."
        lines.append(f"### 2.{i} {name}\n{ev}\n")
        packages.append({"name": name, "duration_weeks": 8, "deadline": "2027-01-10",
                         "priority": 4, "required": {"pm": {"level": 4},
                                                     "engineer": {"level": 4, "count": 2},
                                                     "qa": {"level": 3}}, "evidence": ev})
    lines.append("## 3. Delivery\n")
    for i, name in enumerate(other, 1):
        ev = f"{name}. 1 QA for 3 weeks."
        lines.append(f"### 3.{i} {name}\n{ev}\n")
        packages.append({"name": name, "duration_weeks": 3, "deadline": "2027-01-10",
                         "required": {"qa": {"level": 3}}, "evidence": ev})
    lines.append("## 4. Notes\n" + ("Additional background detail. " * 800) + "\n")
    doc = "".join(lines)
    assert len(doc) > llm.SINGLE_PASS_CHARS and len(packages) == 25
    out = _fake_decompose(doc, packages)
    names = [t["name"] for t in out["tasks"]]
    assert len(names) == 20, (len(names), names)
    assert all(n in names for n in main), names
    omitted = " ".join(out["omitted"])
    assert "overview section" in omitted and "score" in omitted, omitted


def test_overview_duplicate_merges_into_the_section_package():
    """The overview mentions the same deliverable; the content section keeps the
    values (2 engineers, 8 weeks), the overview reading is only a duplicate."""
    doc = ("# PRD\n\n## 1. Tổng quan\nWe will build the Payment gateway first.\n\n"
           "## 2. Scope\n### 2.1 Payment gateway\n2 engineer level 4, 8 weeks. UAT 10/12.\n")
    packages = [
        {"name": "Payment gateway", "duration_weeks": 4, "deadline": "2027-01-01",
         "priority": 5, "required": {"engineer": {"level": 3, "count": 1},
                                      "qa": {"level": 3, "count": 3}},
         "evidence": "We will build the Payment gateway first."},
        {"name": "Payment gateway", "duration_weeks": 8, "deadline": "2027-01-10",
         "required": {"engineer": {"level": 4, "count": 2}},
         "evidence": "2 engineer level 4, 8 weeks."},
    ]
    out = _fake_decompose(doc, packages)
    assert len(out["tasks"]) == 1, out["tasks"]
    gw = out["tasks"][0]
    assert gw["duration_weeks"] == 8 and gw["required"] == {"engineer": {"level": 4, "count": 2}}
    assert gw["source"] == "2.1 Payment gateway" and gw["milestones"] == {"uat": "2026-12-10"}


def test_parent_intro_does_not_overwrite_child_staffing():
    """The package is named in the parent intro (“Đội dự án gồm 1 engineer…”)
    but has a 2.1 section: the child section supplies the staffing (2 engineers),
    not the intro; roles the child does not staff are removed."""
    doc = ("# PRD\n\n## 1. Tổng quan\nIntro.\n\n## 2. Phạm vi\n"
           "Đội dự án gồm 1 engineer senior, 2 QA và 1 PM.\n\n"
           "### 2.1 Cổng thanh toán\n2 engineer level 4, 8 tuần. UAT 10/12.\n")
    packages = [{"name": "Cổng thanh toán", "duration_weeks": 8, "deadline": "2027-01-10",
                 "required": {"engineer": {"level": 4, "count": 1},
                              "qa": {"level": 3, "count": 2},
                              "pm": {"level": 3, "count": 1}},
                 "evidence": "Đội dự án gồm 1 engineer senior, 2 QA và 1 PM."}]
    out = _fake_decompose(doc, packages)
    t = out["tasks"][0]
    assert t["required"] == {"engineer": {"level": 4, "count": 2}}, t["required"]
    assert t["source"] == "2.1 Cổng thanh toán"
    assert t["milestones"] == {"uat": "2026-12-10"}, t["milestones"]
    assert not any("section intro" in w for w in t["warnings"]), t["warnings"]


def test_ai_dependencies_are_suggestions_until_the_document_confirms_them():
    """depends_on_names without an ordering sentence: the solver sees no edge;
    the package carries it as a suggested dependency instead."""
    doc = "# P\n\n## 1. Alpha\nAlpha does X.\n\n## 2. Beta\nBeta does Y.\n"
    packages = [
        {"name": "Alpha", "duration_weeks": 3, "deadline": "2027-01-10",
         "required": {"engineer": {"level": 3}}, "depends_on": ["Beta"],
         "evidence": "Alpha does X."},
        {"name": "Beta", "duration_weeks": 3, "deadline": "2027-01-10",
         "required": {"engineer": {"level": 3}}, "evidence": "Beta does Y."},
    ]
    out = _fake_decompose(doc, packages)
    by = {t["name"]: t for t in out["tasks"]}
    assert by["Alpha"]["depends_on"] == []
    assert [s["id"] for s in by["Alpha"]["suggested_deps"]] == [by["Beta"]["id"]]
    assert by["Alpha"]["suggested_deps"][0]["why"] == "AI guess"
    assert by["Beta"]["depends_on"] == [] and by["Beta"]["suggested_deps"] == []


def test_shared_milestone_contradicting_a_document_dependency_is_dropped():
    """Alpha must finish before Beta; the project-wide go-live (15/03) cannot
    be Alpha's own date when Beta's UAT is 01/02. Alpha gets no dates and a
    warning; the unrelated Gamma still takes the shared go-live."""
    doc = ("# P\n\n## 1. Timeline\nGo-live 15/03/2027.\n\n"
           "## 2. Alpha\nAlpha does X. Alpha must finish before Beta starts.\n\n"
           "## 3. Beta\nBeta does Y. UAT 01/02/2027.\n\n"
           "## 4. Gamma\nGamma does G.\n")
    packages = [
        {"name": "Alpha", "duration_weeks": 3, "deadline": "2027-05-01",
         "required": {"engineer": {"level": 3}}, "evidence": "Alpha does X."},
        {"name": "Beta", "duration_weeks": 3, "deadline": "2027-05-01",
         "required": {"engineer": {"level": 3}}, "evidence": "Beta does Y."},
        {"name": "Gamma", "duration_weeks": 3, "deadline": "2027-05-01",
         "required": {"engineer": {"level": 3}}, "evidence": "Gamma does G."},
    ]
    out = _fake_decompose(doc, packages)
    by = {t["name"]: t for t in out["tasks"]}
    assert by["Beta"]["depends_on"] == [by["Alpha"]["id"]], by["Beta"]
    assert "milestones" not in by["Alpha"], by["Alpha"]
    assert any("after “Beta” starts (2027-02-01)" in w for w in by["Alpha"]["warnings"]), by["Alpha"]["warnings"]
    assert by["Gamma"]["milestones"] == {"deploy": "2027-03-15"}, by["Gamma"]


def test_package_without_any_stage_dates_is_flagged():
    doc = "# P\n\n## 1. Delta\nDelta does D.\n"
    out = _fake_decompose(doc, [{"name": "Delta", "duration_weeks": 2,
                                 "deadline": "2027-01-10", "evidence": "Delta does D.",
                                 "required": {"engineer": {"level": 3}}}])
    t = out["tasks"][0]
    assert any("no stage dates in the document" in w for w in t["warnings"]), t["warnings"]


def test_ids_follow_the_package_name_and_cut_at_word_boundaries():
    import llm
    assert llm._slug("Payment gateway integration and certification", max_len=24) == "payment-gateway"
    assert llm._slug("A very long package name that would be cut in the middle of a word",
                     max_len=30) == "a-very-long-package-name-that"
    doc = "# P\n\n## 2.1 Deliverables\nPartner API. 1 engineer for 2 weeks. UAT 10/12.\n"
    out = _fake_decompose(doc, [{"name": "Partner API", "duration_weeks": 2,
                                 "deadline": "2027-01-10",
                                 "evidence": "Partner API. 1 engineer for 2 weeks. UAT 10/12.",
                                 "required": {"engineer": {"level": 3}}}])
    assert out["tasks"][0]["id"] == "doc-2-1-partner-api", out["tasks"][0]["id"]


def test_complete_retries_429_with_retry_after():
    """A 429 with Retry-After: 1 is waited out and retried, not converted into
    an immediate plain-JSON call."""
    import io
    import os
    import urllib.error
    import llm
    calls, sleeps = [], []

    class Resp:
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def read(self):
            return b'{"choices":[{"message":{"content":"ok"}}]}'

    def fake_urlopen(req, timeout=0):
        calls.append(timeout)
        if len(calls) == 1:
            raise urllib.error.HTTPError(req.full_url, 429, "rate",
                                        {"Retry-After": "1"}, io.BytesIO(b""))
        return Resp()

    old_key = os.environ.get("DEEPSEEK_API_KEY")
    os.environ["DEEPSEEK_API_KEY"] = "test"
    orig_open, orig_sleep = llm.urllib.request.urlopen, llm.time.sleep
    llm.urllib.request.urlopen = fake_urlopen
    llm.time.sleep = lambda s: sleeps.append(s)
    try:
        assert llm.complete([{"role": "user", "content": "hi"}], json_mode=True) == "ok"
    finally:
        llm.urllib.request.urlopen = orig_open
        llm.time.sleep = orig_sleep
        if old_key is None:
            os.environ.pop("DEEPSEEK_API_KEY", None)
        else:
            os.environ["DEEPSEEK_API_KEY"] = old_key
    assert len(calls) == 2, calls
    assert sleeps and abs(sleeps[0] - 1.0) < 0.01, sleeps


def test_complete_gives_up_after_three_429_retries():
    """One initial call + three retries, then the HTTP error is raised so the
    part is reported in failed_parts."""
    import io
    import os
    import urllib.error
    import llm
    calls = []

    def fake_urlopen(req, timeout=0):
        calls.append(1)
        raise urllib.error.HTTPError(req.full_url, 429, "rate",
                                    {"Retry-After": "0"}, io.BytesIO(b""))

    old_key = os.environ.get("DEEPSEEK_API_KEY")
    os.environ["DEEPSEEK_API_KEY"] = "test"
    orig_open, orig_sleep = llm.urllib.request.urlopen, llm.time.sleep
    llm.urllib.request.urlopen = fake_urlopen
    llm.time.sleep = lambda s: None
    try:
        try:
            llm.complete([{"role": "user", "content": "hi"}], json_mode=True)
            raise AssertionError("complete should raise after the retries")
        except urllib.error.HTTPError as e:
            assert e.code == 429
    finally:
        llm.urllib.request.urlopen = orig_open
        llm.time.sleep = orig_sleep
        if old_key is None:
            os.environ.pop("DEEPSEEK_API_KEY", None)
        else:
            os.environ["DEEPSEEK_API_KEY"] = old_key
    assert len(calls) == 4, calls


def test_failed_part_is_reported_and_retried_from_cache():
    """One part fails: its title + a real reason are returned in failed_parts,
    the good parts are cached, and a retry asks the AI only for the failed part."""
    import urllib.error
    import llm
    doc = "".join(
        f"## {i}. Section {i}\nPackage {i}. 1 engineer for 2 weeks. UAT by 15/12.\n\n"
        + ("Filler sentence. " * 300) + "\n\n"
        for i in (1, 2, 3)
    )
    assert len(doc) > llm.SINGLE_PASS_CHARS
    calls = []

    def fake_part(title):
        return {"summary": "s", "tasks": [{
            "name": title.split(". ", 1)[-1], "duration_weeks": 2, "deadline": "2027-01-10",
            "required": {"engineer": {"level": 3}}, "evidence": "UAT by 15/12."}]}

    def fake_fail(skill, today, part, idx, n, outline, total, deadline=None):
        calls.append(part[0])
        if part[0].startswith("2."):
            raise urllib.error.HTTPError("u", 429, "rate", {}, None)
        return fake_part(part[0])

    def fake_ok(skill, today, part, idx, n, outline, total, deadline=None):
        calls.append(part[0])
        return fake_part(part[0])

    orig_call, orig_sleep = llm._call_part, llm.time.sleep
    llm.time.sleep = lambda s: None
    llm._CACHE.clear()
    try:
        llm._call_part = fake_fail
        first = llm.decompose_document(doc, today=TODAY)
        assert {t["name"] for t in first["tasks"]} == {"Section 1", "Section 3"}
        assert len(first["failed_parts"]) == 1, first["failed_parts"]
        assert first["failed_parts"][0]["title"] == "2. Section 2"
        assert "HTTP 429" in first["failed_parts"][0]["error"]
        calls.clear()
        llm._call_part = fake_ok
        second = llm.decompose_document(doc, today=TODAY)
    finally:
        llm._call_part, llm.time.sleep = orig_call, orig_sleep
    assert {t["name"] for t in second["tasks"]} == {"Section 1", "Section 2", "Section 3"}
    assert second["failed_parts"] == []
    assert calls == ["2. Section 2"], calls


def test_new_urgent_work_can_move_a_project_that_has_not_started():
    from models import Assignment, Project
    q = Person("q", "Q", "qa", {"qa": 3})
    old = Project.from_dict({"id": "old", "name": "Old", "duration_weeks": 6, "deadline": "2027-06-01",
                             "priority": 1, "required": {"qa": {"level": 3}}})
    new = Project.from_dict({"id": "new", "name": "New", "duration_weeks": 4, "deadline": "2026-11-13",
                             "priority": 5, "required": {"qa": {"level": 3}}})
    frozen = [Assignment("old", 1, {"qa": ["q"]})]
    result, notes, moved = web._build_with_additions([q], [old, new], [], frozen, frozen, {"new"})
    start = {a["project_id"]: a["start_week"] for a in result["assignments"]}
    assert start["new"] == 0 and start["old"] >= 4, start
    assert notes and "“Old”" in notes[0]
    assert moved == ["Old"], moved          # the approve API reports what it moved
    # a project already running (starts this week) is never moved
    running = [Assignment("old", 0, {"qa": ["q"]})]
    result, notes, moved = web._build_with_additions([q], [old, new], [], running, running, {"new"})
    assert {a["project_id"]: a["start_week"] for a in result["assignments"]}["old"] == 0 and not notes
    assert moved == []


def test_change_made_before_a_reset_does_not_land_after_it():
    cl = _Client()
    try:
        _, d = cl.post("/api/assign")
        gen = d["gen"]
        add = {"add_project": {"id": "late1", "name": "Late add", "duration_weeks": 2, "deadline": "2026-12-31",
                               "required": {"engineer": {"level": 2}}}, "freeze_existing": True}
        _, r = cl.post("/api/replan", {**add, "base_gen": gen})
        assert len(r["projects"]) == 8
        _, r = cl.post("/api/reset")
        assert r["gen"] == gen + 1 and len(r["projects"]) == 7
        # the same page sends another change it started before the reset
        _, r = cl.post("/api/replan", {**add, "base_gen": gen})
        assert "reset" in r["error"][0]
        _, d = cl.post("/api/assign")
        assert len(d["projects"]) == 7
    finally:
        cl.close()


def test_leave_dates_follow_the_sentence():
    """'2 weeks from 2/11' is Nov 2–15, never one day, never from today."""
    import llm
    people = pipeline.load_people("data/people.json")
    projects = pipeline.load_projects("data/projects.json")
    cases = {"Ivy nghỉ 2 tuần từ 2/11": ("2026-11-02", "2026-11-15"),
             "Nếu Ivy nghỉ 1 tuần kể từ 2/11 thì sao?": ("2026-11-02", "2026-11-08"),
             "Ivy nghỉ 10 ngày từ ngày 2 tháng 11": ("2026-11-02", "2026-11-11"),
             "Ivy nghỉ 10 ngày từ 02/11/2026": ("2026-11-02", "2026-11-11"),
             "Ivy nghỉ phép từ 2-11 đến 6-11": ("2026-11-02", "2026-11-06"),
             "Ivy nghỉ 3 ngày": ("2026-10-19", "2026-10-21")}
    for q, want in cases.items():
        change = llm._extract_whatif_local(q, TODAY)
        lv = llm.resolve_whatif(change, people, projects, TODAY, original_text=q)["payload"]["leave"][0]
        assert (lv["from"], lv["to"]) == want, (q, lv)
    assert llm._extract_dates("starting 2 November, back Nov 16", TODAY) == ("2026-11-02", "2026-11-16")


def test_empty_plan_explain_skips_llm():
    cl = _Client()
    try:
        cl.post("/api/assign")
        ids = ["p1", "p2", "p3", "p4", "p5", "p6", "p7"]
        cl.post("/api/replan", {"remove_project": ids, "freeze_existing": True})
        for _ in range(2):
            _, r = cl.post("/api/explain")
            assert r["explanation"] == web.EMPTY_PLAN_EXPLANATION
    finally:
        cl.close()


def test_utilization_uses_active_window_and_upskill_is_offered():
    people = pipeline.load_people("data/people.json")
    projects = pipeline.load_projects("data/projects.json")
    plan, pstats, risks = pipeline.run(people, projects, TODAY)
    util = {s.person.name: s.util for s in pstats}
    assert util["Ivy"] >= 0.99, "the only QA≥3 is fully booked during the active window"
    assert util["Eve"] >= 0.45, f"engineer load must not be diluted by solver slack: {util['Eve']}"
    bottlenecks = analysis.find_bottlenecks(plan, people, projects, risks, pstats)
    ranked = suggest.suggest_fixes(people, projects, TODAY, prev_assignments=plan.assignments,
                                   bottlenecks=bottlenecks, risks=risks)
    hire = next(r for r in ranked if "add_person" in r["payload"])
    assert "weeks" in hire["note"], hire["note"]
    upskill = next(r for r in ranked if "update_person" in r["payload"])
    assert upskill["payload"]["update_person"]["id"] == "qa2"
    assert upskill["total_late_days"] < 77


def test_merging_one_package_never_inflates_headcount():
    """A second reading from another part may not max level/count/duration:
    the quoted reading in the home span wins, and the home text's own
    '1 engineer and 1 QA' pulls the headcount back to 1 + 1."""
    import llm
    doc = ("# Programme\n\n## 1. Partner API\n"
           "Partner API: 1 engineer and 1 QA for 6 weeks. This is the main deliverable.\n\n"
           + "Scope detail. " * 3000
           + "\n\nAdditional notes. The team will refine the scope.\n")
    calls = []

    def fake(messages, **kw):
        part = messages[-1]["content"]
        calls.append(part)
        tasks = []
        if "1 engineer and 1 QA" in part:
            tasks.append({"name": "Partner API", "duration_weeks": 6, "deadline": "2027-01-15",
                          "required": {"engineer": {"level": 3, "count": 1},
                                       "qa": {"level": 3, "count": 1}},
                          "evidence": "Partner API: 1 engineer and 1 QA for 6 weeks."})
        if "Additional notes" in part:
            tasks.append({"name": "Partner API", "duration_weeks": 9, "deadline": "2027-03-01",
                          "required": {"engineer": {"level": 5, "count": 2},
                                       "qa": {"level": 4, "count": 3}},
                          "evidence": "Additional notes. The team will refine the scope."})
        return {"summary": "part", "tasks": tasks}

    orig, llm.complete_json = llm.complete_json, fake
    llm._CACHE.clear()
    try:
        out = llm.decompose_document(doc, today=TODAY)
    finally:
        llm.complete_json = orig
    assert len(calls) >= 2 and out["parts"] >= 2
    assert len(out["tasks"]) == 1, out["tasks"]
    t = out["tasks"][0]
    assert t["required"]["engineer"] == {"level": 3, "count": 1}, t["required"]
    assert t["required"]["qa"] == {"level": 3, "count": 1}, t["required"]
    assert t["duration_weeks"] == 6 and t["deadline"] == "2027-01-15"


def test_documented_order_keeps_ai_edges_out_of_the_graph():
    """Alpha→Beta→Gamma guessed by the model: those stay suggestions. The
    document's own sentence (Alpha before Gamma) is the only edge the solver
    gets, so the guessed cycle cannot reach it."""
    import llm
    doc = ("1. Alpha\nAlpha does X. Alpha must finish before Gamma starts.\n\n"
           "2. Beta\nBeta does Y.\n\n"
           "3. Gamma\nGamma does Z.\n")
    packages = [
        {"name": "Alpha", "duration_weeks": 3, "deadline": "2027-01-10",
         "required": {"engineer": {"level": 3}}, "depends_on": ["Beta"],
         "evidence": "Alpha does X. Alpha must finish before Gamma starts."},
        {"name": "Beta", "duration_weeks": 3, "deadline": "2027-01-10",
         "required": {"engineer": {"level": 3}}, "depends_on": ["Gamma"],
         "evidence": "Beta does Y."},
        {"name": "Gamma", "duration_weeks": 3, "deadline": "2027-01-10",
         "required": {"engineer": {"level": 3}}, "evidence": "Gamma does Z."},
    ]

    def fake(messages, **kw):
        part = messages[-1]["content"]
        return {"summary": "s", "tasks": [t for t in packages if t["evidence"] in part]}

    orig, llm.complete_json = llm.complete_json, fake
    llm._CACHE.clear()
    try:
        out = llm.decompose_document(doc, today=TODAY)
    finally:
        llm.complete_json = orig
    by = {t["name"]: t for t in out["tasks"]}
    assert by["Alpha"]["depends_on"] == [], by["Alpha"]
    assert by["Beta"]["depends_on"] == [], by["Beta"]
    assert by["Gamma"]["depends_on"] == [by["Alpha"]["id"]], by["Gamma"]
    assert [s["id"] for s in by["Alpha"]["suggested_deps"]] == [by["Beta"]["id"]]
    assert [s["id"] for s in by["Beta"]["suggested_deps"]] == [by["Gamma"]["id"]]
    assert by["Gamma"]["suggested_deps"] == []
    warnings = [w for t in out["tasks"] for w in t["warnings"]]
    assert any("from the document" in w for w in warnings), warnings
    assert not any("check by hand" in w for w in warnings), warnings


def test_adding_a_sentence_keeps_other_ids_and_part_cache():
    """Part boundaries follow headings: a new sentence in Alpha re-reads only
    Alpha; Beta/Gamma keep their cache entries and their package ids."""
    import llm
    base = ("# Programme\n\n## 1. Alpha\nAlpha deliverable. 1 engineer for 4 weeks. UAT by 15/12.\n\n"
            + "Alpha detail. " * 900
            + "\n\n## 2. Beta\nBeta deliverable. 1 QA for 3 weeks. UAT by 20/12.\n\n"
            + "Beta detail. " * 900
            + "\n\n## 3. Gamma\nGamma deliverable. 1 engineer for 2 weeks. UAT by 25/12.\n\n"
            + "Gamma detail. " * 900)
    changed = base.replace("## 1. Alpha\nAlpha deliverable.",
                           "## 1. Alpha\nAlpha deliverable, now clearer.")
    assert changed != base and len(base) > llm.SINGLE_PASS_CHARS
    calls = []

    def fake(messages, **kw):
        part = messages[-1]["content"]
        calls.append(part)
        for name, heading, line, req in [
            ("Alpha", "## 1. Alpha", "1 engineer for 4 weeks.", {"engineer": {"level": 3}}),
            ("Beta", "## 2. Beta", "1 QA for 3 weeks.", {"qa": {"level": 3}}),
            ("Gamma", "## 3. Gamma", "1 engineer for 2 weeks.", {"engineer": {"level": 3}}),
        ]:
            if heading in part:
                return {"summary": "s", "tasks": [{"name": name, "duration_weeks": 4,
                                                   "deadline": "2027-01-10", "required": req,
                                                   "evidence": line}]}
        return {"summary": "s", "tasks": []}

    orig, llm.complete_json = llm.complete_json, fake
    llm._CACHE.clear()
    try:
        first = llm.decompose_document(base, today=TODAY)
        n_first = len(calls)
        ids_first = [t["id"] for t in first["tasks"]]
        second = llm.decompose_document(changed, today=TODAY)
        rerun = calls[n_first:]
    finally:
        llm.complete_json = orig
    assert [t["id"] for t in second["tasks"]] == ids_first
    assert [t["name"] for t in second["tasks"]] == ["Alpha", "Beta", "Gamma"]
    assert len(rerun) == 1, len(rerun)
    assert "Part — 1. Alpha:" in rerun[0]
    assert "Part — 2. Beta:" not in rerun[0] and "Part — 3. Gamma:" not in rerun[0]


def test_bullets_under_one_heading_get_their_own_milestones_and_source():
    """Three packages in one section: each reads the stage dates from its own
    bullet, and the source is that bullet - never the shared section title."""
    import llm
    doc = ("# PayHub programme\n\n"
           "- Payments core. 1 engineer. UAT by 15/12.\n"
           "- Partner API. 1 engineer and 1 QA. UAT by 20/12; go-live 10/01/2027.\n"
           "- Audit module. 1 engineer. Go-live 25/01/2027.\n")
    packages = [
        {"name": "Payments core", "duration_weeks": 4, "deadline": "2027-01-10",
         "required": {"engineer": {"level": 3}},
         "evidence": "Payments core. 1 engineer. UAT by 15/12."},
        {"name": "Partner API", "duration_weeks": 6, "deadline": "2027-01-10",
         "required": {"engineer": {"level": 3}, "qa": {"level": 3}},
         "evidence": "Partner API. 1 engineer and 1 QA. UAT by 20/12; go-live 10/01/2027."},
        {"name": "Audit module", "duration_weeks": 3, "deadline": "2027-01-25",
         "required": {"engineer": {"level": 3}},
         "evidence": "Audit module. 1 engineer. Go-live 25/01/2027."},
        # no quote: it must not inherit the first bullet's stage dates
        {"name": "Mystery module", "duration_weeks": 2, "deadline": "2027-02-01",
         "required": {"engineer": {"level": 3}},
         "evidence": "This sentence is not in the document."},
    ]

    def fake(messages, **kw):
        part = messages[-1]["content"]
        tasks = [t for t in packages if t["evidence"] in part]
        tasks += [t for t in packages if t["name"] == "Mystery module"]
        return {"summary": "s", "tasks": tasks}

    orig, llm.complete_json = llm.complete_json, fake
    llm._CACHE.clear()
    try:
        out = llm.decompose_document(doc, today=TODAY)
    finally:
        llm.complete_json = orig
    by = {t["name"]: t for t in out["tasks"]}
    assert by["Payments core"]["milestones"] == {"uat": "2026-12-15"}
    assert by["Partner API"]["milestones"] == {"uat": "2026-12-20", "deploy": "2027-01-10"}
    assert by["Audit module"]["milestones"] == {"deploy": "2027-01-25"}
    for name, t in by.items():
        if name == "Mystery module":
            continue
        assert t["source"] != "PayHub programme", (name, t["source"])
        assert t["source"].startswith(name), (name, t["source"])
    mystery = by["Mystery module"]
    assert "milestones" not in mystery, mystery.get("milestones")
    assert mystery["source"] == "", mystery["source"]
    assert any("no quote found" in w for w in mystery["warnings"]), mystery["warnings"]



def test_a_bare_analyst_mention_does_not_keep_the_role():
    """'phân tích' in prose is not staffing: only a number + role keeps it."""
    doc = ("# PRD\n\n## 1. Integration\n"
           "1 engineer and 1 QA. The analyst will review the data.\n")
    tasks = [{"name": "Integration", "duration_weeks": 3, "deadline": "2027-01-10",
              "required": {"analyst": {"level": 3}, "engineer": {"level": 3},
                           "qa": {"level": 3}},
              "evidence": "1 engineer and 1 QA."}]
    out = _fake_decompose(doc, tasks)
    assert out["tasks"][0]["required"] == {
        "engineer": {"level": 3, "count": 1},
        "qa": {"level": 3, "count": 1},
    }, out["tasks"][0]["required"]
    assert any("analyst removed" in w for w in out["tasks"][0]["warnings"])


def test_risk_sections_and_stage_only_names_are_not_packages():
    """Risks/assumptions sections are context; a name that is only a delivery
    stage (Vietnamese or English) is dropped instead of becoming a package."""
    import llm
    doc = ("# PRD\n\n"
           "## Risks and assumptions\nProject risk list. Assumption A.\n\n"
           "## UAT and go-live of the whole system\nOne engineer. UAT 15/12.\n\n"
           "## 1. Payments core\n1 engineer and 1 QA. UAT 20/12.\n")
    tasks = [
        {"name": "Risks and assumptions", "duration_weeks": 1, "deadline": "2027-01-10",
         "required": {"engineer": {"level": 3}},
         "evidence": "Project risk list. Assumption A."},
        {"name": "Giai đoạn UAT và go-live toàn hệ thống", "duration_weeks": 1,
         "deadline": "2027-01-10", "required": {"engineer": {"level": 3}},
         "evidence": "One engineer. UAT 15/12."},
        {"name": "Payments core", "duration_weeks": 4, "deadline": "2027-01-10",
         "required": {"engineer": {"level": 3}, "qa": {"level": 3}},
         "evidence": "1 engineer and 1 QA. UAT 20/12."},
    ]

    def fake(messages, **kw):
        part = messages[-1]["content"]
        return {"summary": "s", "tasks": [t for t in tasks if t["evidence"] in part]}

    orig, llm.complete_json = llm.complete_json, fake
    llm._CACHE.clear()
    try:
        out = llm.decompose_document(doc, today=TODAY)
    finally:
        llm.complete_json = orig
    assert [t["name"] for t in out["tasks"]] == ["Payments core"], out["tasks"]
    joined = " ".join(out["warnings"])
    assert "Risks and assumptions (appendix / cross-cutting section" in joined, joined
    assert "a delivery stage" in joined, joined
    assert out["tasks"][0]["milestones"] == {"uat": "2026-12-20"}


def test_explain_late_days_uses_the_missed_stage_date():
    """A project late only by a stage date is late by that gap, not by the
    (positive) distance to the deadline; the stage and its due date are named."""
    import explain
    from analysis import ProjectRisk
    from models import Plan, Project
    w = Project.from_dict({"id": "w", "name": "W", "duration_weeks": 2,
                           "deadline": "2026-12-31", "required": {"engineer": {"level": 3}}})
    plan = Plan(start_date=TODAY, horizon_weeks=10, assignments=[],
                unstaffable=[("w", "no slot")])
    risk = ProjectRisk(w, "late", start_week=0, done_week=1, done_date="2026-11-06",
                       days_left=55,
                       milestones=[{"stage": "uat", "due": "2026-12-01",
                                    "planned": "2026-12-10", "days_left": -9}])
    text = explain.explain_plan(plan, [], [w], [risk])
    assert "LATE by 9 days" in text, text
    assert "LATE by 55 days" not in text, text
    assert "stage “uat” due 2026-12-01" in text, text
    assert "late by 9 days" in text, text
    # the web-side fallback (no precomputed late_days) uses the stage too
    assert web._late_days({"risk": "late", "days_left": 55,
                           "milestones": [{"days_left": -9}]}) == 9


if __name__ == "__main__":
    test_freeze_keeps_old_plan_identical()
    print("ok freeze")
    test_suggest_when_late_even_without_packed_people()
    print("ok suggest")
    test_rate_limit_blocks_burst()
    print("ok rate limit")
    test_overlap_warns_on_acme_payment_gateway()
    print("ok overlap")
    test_hire_qa_whatif_does_not_reshuffle_unrelated()
    print("ok hire-qa freeze")
    test_persisted_plan_survives_assign_and_suggest()
    print("ok persisted plan")
    test_bilingual_overlap_without_heuristic_match()
    print("ok bilingual overlap")
    test_delta_causes_pulled_deadline_and_busy_squad()
    print("ok delta causes")
    test_empty_plan_explanation_and_429_retry_after()
    print("ok empty explain & 429 retry-after")
    test_approve_then_assign_is_byte_identical()
    print("ok approve -> assign byte-identical")
    test_hire_qa4_whatif_moves_only_late_projects()
    print("ok hire QA4 what-if")
    test_every_changed_project_has_a_factual_reason()
    print("ok change reasons")
    test_pushed_start_names_the_binding_people_not_newcomers()
    print("ok pushed-start reason")
    test_short_leave_is_covered_not_a_whole_week_lost()
    print("ok short leave covered")
    test_role_phase_and_effort_only_book_their_weeks()
    print("ok role phase & effort")
    test_hire_cannot_work_before_start_date()
    print("ok hire start date")
    test_long_document_is_read_in_parts_and_merged()
    print("ok long document")
    test_stage_milestones_are_read_not_guessed()
    print("ok stage milestones")
    test_waterfall_stages_book_roles_only_when_needed()
    print("ok waterfall")
    test_prd_with_several_packages_gets_dates_sources_and_dependency_order()
    print("ok PRD dates/sources/deps")
    test_over_the_cap_keeps_the_best_supported_packages()
    print("ok cap keeps go-live work")
    test_overview_fragments_do_not_take_the_package_cap()
    print("ok overview cap")
    test_overview_duplicate_merges_into_the_section_package()
    print("ok overview duplicate merge")
    test_parent_intro_does_not_overwrite_child_staffing()
    print("ok parent intro vs child staffing")
    test_ai_dependencies_are_suggestions_until_the_document_confirms_them()
    print("ok AI deps are suggestions")
    test_shared_milestone_contradicting_a_document_dependency_is_dropped()
    print("ok shared milestone conflict")
    test_package_without_any_stage_dates_is_flagged()
    print("ok no stage dates warning")
    test_ids_follow_the_package_name_and_cut_at_word_boundaries()
    print("ok ids from package name")
    test_complete_retries_429_with_retry_after()
    print("ok 429 retry-after")
    test_complete_gives_up_after_three_429_retries()
    print("ok 429 gives up after retries")
    test_failed_part_is_reported_and_retried_from_cache()
    print("ok failed part + retry from cache")
    test_roles_the_section_does_not_staff_are_removed()
    print("ok unstated roles removed")
    test_new_urgent_work_can_move_a_project_that_has_not_started()
    print("ok selective unlock")
    test_change_made_before_a_reset_does_not_land_after_it()
    print("ok reset refuses stale changes")
    test_leave_dates_follow_the_sentence()
    print("ok leave dates")
    test_empty_plan_explain_skips_llm()
    print("ok empty explain via API")
    test_utilization_uses_active_window_and_upskill_is_offered()
    print("ok active-window util & upskill")
    test_merging_one_package_never_inflates_headcount()
    print("ok merge keeps headcount")
    test_documented_order_keeps_ai_edges_out_of_the_graph()
    print("ok doc edge breaks AI cycle")
    test_adding_a_sentence_keeps_other_ids_and_part_cache()
    print("ok stable ids & part cache")
    test_bullets_under_one_heading_get_their_own_milestones_and_source()
    print("ok per-bullet milestones & source")
    test_risk_sections_and_stage_only_names_are_not_packages()
    print("ok risks/stage names dropped")
    test_a_bare_analyst_mention_does_not_keep_the_role()
    print("ok bare analyst mention")
    test_explain_late_days_uses_the_missed_stage_date()
    print("ok late days by stage")
    print("ALL OK")
