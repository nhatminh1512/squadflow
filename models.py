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


@dataclass
class Person:
    id: str
    name: str
    role: str                       # one of ROLES
    skills: dict[str, int]          # role -> proficiency 1..5 (a person may hold several)
    capacity: float = 1.0           # fraction of full time: 1.0 = one project at a time
    leave: list[tuple[date, date]] = field(default_factory=list)  # date ranges off work

    def level(self, role: str) -> int:
        return self.skills.get(role, 0)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Person":
        leave = [
            (date.fromisoformat(a), date.fromisoformat(b))
            for a, b in d.get("leave", [])
        ]
        return cls(
            d["id"],
            d["name"],
            d["role"],
            {k: int(v) for k, v in d.get("skills", {}).items()},
            float(d.get("capacity", 1.0)),
            leave,
        )


@dataclass
class Project:
    id: str
    name: str
    required: dict[str, dict]        # role -> {"level": int, "count": int}
    duration_weeks: int
    deadline: date
    priority: int = 3               # 1..5, higher = staff first
    depends_on: list[str] = field(default_factory=list)  # project ids that must finish first

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
        return cls(
            d["id"],
            d["name"],
            required,
            int(d["duration_weeks"]),
            date.fromisoformat(d["deadline"]),
            int(d.get("priority", 3)),
            list(d.get("depends_on", [])),
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

    def end_date(self) -> date:
        return self.start_date + timedelta(weeks=self.horizon_weeks)
