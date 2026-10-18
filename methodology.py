"""Waterfall delivery template.

Turns one project (total weeks + roles) into sequential stages with client
sign-off gates, and books each role only in the stages it really works, at a
stage-specific effort. The AI only extracts what a document states; this
module applies the method, so every plan is consistent and testable.

All numbers below are defaults a delivery manager can tune in one place.
"""
from __future__ import annotations

# Stage order and share of the working weeks (sign-off gates excluded).
STAGES = [("requirements", 15), ("design", 15), ("build", 40),
          ("sit", 15), ("uat", 10), ("deploy", 5)]
# Short projects: analysis+design, build, test (UAT/deploy folded into test).
COMPACT_STAGES = [("requirements", 25), ("build", 50), ("sit", 25)]
# Client sign-off wait (weeks) after a stage. Only for projects long enough
# to have a formal gate; shorter ones sign off inside the stage.
GATES = {"requirements": 1, "uat": 1}
GATE_MIN_WEEKS = 10
MIN_WEEKS = 4                     # shorter than this: plan as one block

# Effort % of one person per stage; missing = not needed in that stage.
# "gate" = waiting for sign-off (only the PM chases it).
ROLE_STAGE_EFFORT: dict[str, dict[str, int]] = {
    "pm":       {"requirements": 50, "design": 50, "build": 50, "sit": 50,
                 "uat": 50, "deploy": 50, "gate": 20},
    "analyst":  {"requirements": 100, "design": 50, "uat": 50},
    "engineer": {"design": 50, "build": 100, "sit": 50, "deploy": 100},
    "qa":       {"design": 20, "build": 20, "sit": 100, "uat": 50},
}

# How much longer than planned a stage may run (Monte Carlo upper bound).
# Testing stages overrun most: defects found late cause rework.
STAGE_OVERRUN = {"requirements": 0.2, "design": 0.2, "build": 0.3, "sit": 0.5,
                 "uat": 0.5, "deploy": 0.2, "gate": 1.0}

STAGE_LABELS = {"requirements": "Requirements", "design": "Design", "build": "Build",
                "sit": "SIT", "uat": "UAT", "deploy": "Deploy", "gate": "Sign-off"}


def _split(work: int, stages: list[tuple[str, int]]) -> dict[str, int]:
    """Largest-remainder split of `work` weeks by percentage, every stage ≥ 1."""
    raw = {s: pct * work / 100 for s, pct in stages}
    out = {s: int(x) for s, x in raw.items()}
    left = work - sum(out.values())
    for s in sorted(raw, key=lambda s: -(raw[s] - int(raw[s])))[:left]:
        out[s] += 1
    for s in out:                       # no empty stage: borrow from the biggest
        if out[s] == 0:
            big = max(out, key=out.get)
            out[big] -= 1
            out[s] = 1
    return out


def stage_plan(duration: int) -> list[dict]:
    """Sequential stages covering weeks [0, duration): [{stage, start, weeks}].

    Gates appear as {"stage": "gate", "after": <stage>}. Empty when the
    project is too short for staged delivery."""
    if duration < MIN_WEEKS:
        return []
    gates = GATES if duration >= GATE_MIN_WEEKS else {}
    work = duration - sum(gates.values())
    stages = STAGES if work >= len(STAGES) else COMPACT_STAGES
    weeks = _split(work, stages)
    plan, w = [], 0
    for s, _ in stages:
        plan.append({"stage": s, "start": w, "weeks": weeks[s]})
        w += weeks[s]
        if s in gates:
            plan.append({"stage": "gate", "after": s, "start": w, "weeks": gates[s]})
            w += gates[s]
    return plan


def role_phases(role: str, plan: list[dict], duration: int,
                effort: int | None = None) -> list[dict]:
    """[{start, weeks, effort}] for a role across the stage plan.

    `effort` (stated in the document) replaces the template effort in every
    working stage the role is part of; gates keep their own effort."""
    table = ROLE_STAGE_EFFORT.get(role, {})
    phases: list[dict] = []
    for st in plan:
        e = table.get(st["stage"], 0)
        if e and effort and st["stage"] != "gate":
            e = effort
        if not e:
            continue
        last = phases[-1] if phases else None
        if last and last["effort"] == e and last["start"] + last["weeks"] == st["start"]:
            last["weeks"] += st["weeks"]
        else:
            phases.append({"start": st["start"], "weeks": st["weeks"], "effort": e})
    if not phases:  # role the template does not know: whole project
        phases = [{"start": 0, "weeks": duration, "effort": effort or 100}]
    return phases


# A compact plan folds stages together; a milestone on a folded stage is
# due when the stage it was folded into ends.
FOLDED_INTO = {"design": "requirements", "uat": "sit", "deploy": "sit"}
MILESTONE_STAGES = [s for s, _ in STAGES]


def stage_end(plan: list[dict], stage: str) -> int | None:
    """Last week offset of `stage` (None if the plan has no stages)."""
    ends = [s["start"] + s["weeks"] - 1 for s in plan if s["stage"] == stage]
    if not ends and stage in FOLDED_INTO:
        return stage_end(plan, FOLDED_INTO[stage])
    return max(ends) if ends else None


if __name__ == "__main__":
    for d in (3, 4, 5, 8, 10, 12, 16, 26):
        p = stage_plan(d)
        assert not p or (p[0]["start"] == 0 and p[-1]["start"] + p[-1]["weeks"] == d), (d, p)
        assert all(s["weeks"] >= 1 for s in p), (d, p)
    p = stage_plan(12)
    qa = role_phases("qa", p, 12)
    assert sum(ph["weeks"] for ph in qa) < 12 and max(ph["effort"] for ph in qa) == 100
    assert role_phases("pm", p, 12, effort=20)[0]["effort"] == 20
    assert stage_end(stage_plan(5), "uat") == stage_end(stage_plan(5), "sit") == 4
    print("methodology ok:", [(s["stage"], s["weeks"]) for s in p])
