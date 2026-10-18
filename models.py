"""Domain model — the "memory" layer of squadflow.

Persistent facts about the team and incoming projects, kept as plain
dataclasses. This mirrors Hermes's memory/state: declarative facts, no logic.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, timedelta
from typing import Any

# Roles a project can require. Everyone has exactly one primary role.
ROLES = ("pm", "analyst", "engineer", "qa")

# Default weeks from "hire" to first working day (recruiting + onboarding).
HIRE_LEAD_WEEKS = 6


def _as_id_list(value) -> list[str]:
    """depends_on is a list of ids; a lone string must not be split into characters."""
    if not value:
        return []
    if isinstance(value, str):
        return [value]
    return [str(x) for x in value]


@dataclass
class Person:
    id: str
    name: str
    role: str                       # one of ROLES
    skills: dict[str, int]          # role -> proficiency 1..5 (a person may hold several)
    capacity: float = 1.0           # load units: 1.0 = one full-time role, 2.0 = two
    leave: list[tuple[date, date]] = field(default_factory=list)  # date ranges off work
    available_from: date | None = None  # new hire: first working day (None = already on the team)

    def level(self, role: str) -> int:
        return self.skills.get(role, 0)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Person":
        leave = [
            (date.fromisoformat(a), date.fromisoformat(b))
            for a, b in d.get("leave", [])
        ]
        start = d.get("available_from")
        return cls(
            d["id"],
            d["name"],
            d["role"],
            {k: int(v) for k, v in d.get("skills", {}).items()},
            float(d.get("capacity", 1.0)),
            leave,
            date.fromisoformat(start) if start else None,
        )


@dataclass
class Project:
    id: str
    name: str
    # role -> {"level", "count", "effort" (% of one full-time person, default 100),
    #          "start" (week offset inside the project, default 0),
    #          "weeks" (how long the role is needed, default: to the end),
    #          "phases" ([{start, weeks, effort}] — several windows, e.g. from
    #                   the waterfall template; overrides start/weeks/effort)}
    required: dict[str, dict]
    duration_weeks: int
    deadline: date
    priority: int = 3               # 1..5, higher = staff first
    depends_on: list[str] = field(default_factory=list)  # project ids that must finish first
    method: str = "simple"          # "simple" | "waterfall" (methodology.py stages + gates)
    stages: list[dict] = field(default_factory=list)       # [{stage, start, weeks}] when waterfall
    milestones: dict[str, date] = field(default_factory=dict)  # stage -> due date (e.g. uat)

    def level_for(self, role: str) -> int:
        spec = self.required.get(role)
        if isinstance(spec, dict):
            return int(spec.get("level", 0))
        return int(spec or 0)

    def count_for(self, role: str) -> int:
        spec = self.required.get(role)
        if isinstance(spec, dict):
            return max(1, int(spec.get("count", 1)))
        return 1 if spec else 0

    def effort_for(self, role: str) -> int:
        """Percent of one full-time person this role needs each week it runs."""
        spec = self.required.get(role)
        return int(spec.get("effort", 100)) if isinstance(spec, dict) else 100

    def phase_for(self, role: str) -> tuple[int, int]:
        """(first week offset, number of weeks) the role is needed, clamped to the project."""
        spec = self.required.get(role)
        spec = spec if isinstance(spec, dict) else {}
        dur = max(1, self.duration_weeks)
        start = min(max(0, int(spec.get("start") or 0)), dur - 1)
        weeks = spec.get("weeks")
        weeks = dur - start if weeks in (None, "") else min(max(1, int(weeks)), dur - start)
        return start, weeks

    def role_phases(self, role: str) -> list[tuple[int, int, int]]:
        """[(week offset, weeks, effort %)] windows the role works, clamped to the project."""
        spec = self.required.get(role)
        spec = spec if isinstance(spec, dict) else {}
        dur = max(1, self.duration_weeks)
        if spec.get("phases"):
            out = []
            for ph in spec["phases"]:
                s = min(max(0, int(ph.get("start") or 0)), dur - 1)
                n = min(max(1, int(ph.get("weeks") or 1)), dur - s)
                out.append((s, n, int(ph.get("effort") or 100)))
            return out
        s, n = self.phase_for(role)
        return [(s, n, self.effort_for(role))]

    def role_effort(self, role: str, project_start: int) -> dict[int, int]:
        """week -> effort % the role needs, for a project starting at project_start."""
        out: dict[int, int] = {}
        for s, n, e in self.role_phases(role):
            for w in range(project_start + s, project_start + s + n):
                out[w] = max(out.get(w, 0), e)
        return out

    def role_weeks(self, role: str, project_start: int) -> list[int]:
        return sorted(self.role_effort(role, project_start))

    def roles(self) -> list[str]:
        return list(self.required.keys())

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Project":
        required: dict[str, dict] = {}
        for role, spec in d.get("required", {}).items():
            if isinstance(spec, int):
                required[role] = {"level": spec, "count": 1}
            else:
                required[role] = {
                    "level": int(spec.get("level", 1)),
                    "count": max(1, int(spec.get("count", 1))),
                }
                # only store non-defaults so old plans/JSON stay byte-identical
                if int(spec.get("effort") or 100) != 100:
                    required[role]["effort"] = int(spec["effort"])
                if int(spec.get("start") or 0):
                    required[role]["start"] = int(spec["start"])
                if spec.get("weeks") not in (None, ""):
                    required[role]["weeks"] = int(spec["weeks"])
                if spec.get("phases"):
                    required[role]["phases"] = [
                        {"start": int(ph.get("start") or 0), "weeks": int(ph.get("weeks") or 1),
                         "effort": int(ph.get("effort") or 100)} for ph in spec["phases"]]
        duration = int(d["duration_weeks"])
        method = d.get("method") or "simple"
        stages: list[dict] = []
        if method == "waterfall":
            import methodology
            stages = methodology.stage_plan(duration)
            for role, spec in required.items():
                # a role whose weeks the document states keeps them; the rest follow the template
                if stages and not ({"phases", "start", "weeks"} & set(spec)):
                    spec["phases"] = methodology.role_phases(role, stages, duration, spec.get("effort"))
        return cls(
            d["id"],
            d["name"],
            required,
            duration,
            date.fromisoformat(d["deadline"]),
            int(d.get("priority", 3)),
            _as_id_list(d.get("depends_on")),
            method,
            stages,
            {k: date.fromisoformat(v) for k, v in (d.get("milestones") or {}).items()},
        )


@dataclass
class Assignment:
    project_id: str
    start_week: int                # week offset from the planning horizon start
    squad: dict[str, list[str]]    # role -> list of person ids (count people per role)


@dataclass
class Plan:
    """Result of one planning run (the 'final response' of the pipeline)."""
    start_date: date
    horizon_weeks: int
    assignments: list[Assignment] = field(default_factory=list)
    unstaffable: list[tuple[str, str]] = field(default_factory=list)  # (project_id, reason)
    committed: dict[str, dict[int, set[str]]] = field(default_factory=dict)  # person -> week -> project ids
    decisions: list[dict] = field(default_factory=list)  # solver log: why each role went to each person
    load: dict[str, dict[int, int]] = field(default_factory=dict)  # person -> week -> booked units (engine.DAY per person-day)

    def end_date(self) -> date:
        return self.start_date + timedelta(weeks=self.horizon_weeks)
