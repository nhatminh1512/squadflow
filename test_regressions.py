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
        # the approve response may add one-off "notes" (projects moved to fit); the plan itself is identical
        assert again == {k: v for k, v in approved.items() if k != "notes"}, \
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
    assert gw["required"]["engineer"] == {"level": 5, "count": 2}     # max of both parts
    assert gw["duration_weeks"] == 14 and gw["deadline"] == "2027-01-20"
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
    # 1: project-wide stage dates reach every package; schedules / appendix are not packages
    for t in by.values():
        assert t["milestones"] == {"uat": "2026-12-15", "deploy": "2027-01-20"}, t
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


def test_over_the_cap_keeps_go_live_work_in_document_order():
    out = _fake_decompose(PRD, PRD_TASKS, cap=3)
    names = [t["name"] for t in out["tasks"]]
    # Audit logging (no go-live / must wording) is the one cut, not the security audit
    assert names == ["Payment gateway", "Partner API", "Security audit"], names
    assert "Audit logging (over the 3-package limit)" in out["warnings"][-1]


def test_new_urgent_work_can_move_a_project_that_has_not_started():
    from models import Assignment, Project
    q = Person("q", "Q", "qa", {"qa": 3})
    old = Project.from_dict({"id": "old", "name": "Old", "duration_weeks": 6, "deadline": "2027-06-01",
                             "priority": 1, "required": {"qa": {"level": 3}}})
    new = Project.from_dict({"id": "new", "name": "New", "duration_weeks": 4, "deadline": "2026-11-13",
                             "priority": 5, "required": {"qa": {"level": 3}}})
    frozen = [Assignment("old", 1, {"qa": ["q"]})]
    result, notes = web._build_with_additions([q], [old, new], [], frozen, frozen, {"new"})
    start = {a["project_id"]: a["start_week"] for a in result["assignments"]}
    assert start["new"] == 0 and start["old"] >= 4, start
    assert notes and "“Old”" in notes[0]
    # a project already running (starts this week) is never moved
    running = [Assignment("old", 0, {"qa": ["q"]})]
    result, notes = web._build_with_additions([q], [old, new], [], running, running, {"new"})
    assert {a["project_id"]: a["start_week"] for a in result["assignments"]}["old"] == 0 and not notes


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
    test_over_the_cap_keeps_go_live_work_in_document_order()
    print("ok cap keeps go-live work")
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
    print("ALL OK")

