"""Pipeline — the turn loop of squadflow.

One planning pass decomposes into the same phases Hermes decomposes a turn:
  ingest  -> load people/projects JSON
  model   -> parse into domain objects
  plan    -> engine.assign (deterministic placement)
  predict -> analysis.analyze (crunch / idle / deadline risk)
  report  -> render the plan (report.py)

Each phase is a separate module (facade + siblings), so a phase can be
swapped or re-run independently — e.g. replanning re-runs only plan+predict.
"""
from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import optimizer
from analysis import analyze
from engine import assign
from models import Person, Project, Plan


def load_people(path: str | Path) -> list[Person]:
    raw = json.loads(Path(path).read_text())
    return [Person.from_dict(d) for d in raw]


def load_projects(path: str | Path) -> list[Project]:
    raw = json.loads(Path(path).read_text())
    return [Project.from_dict(d) for d in raw]


def run(
    people: list[Person],
    projects: list[Project],
    start_date: date,
    use_cp_sat: bool | None = None,
    prev_assignments=None,
    pins: list | None = None,
    freeze=None,
    time_limit_s: float = 15.0,
    keep_squads: bool = False,
):
    """Run the pipeline. use_cp_sat defaults to auto: CP-SAT if ortools is
    installed, greedy otherwise. keep_squads (CP-SAT only): previously staffed
    roles keep their people, only dates move."""
    if use_cp_sat is None:
        use_cp_sat = optimizer.has_cp_sat()
    if use_cp_sat:
        plan = optimizer.cp_sat_assign(
            people, projects, start_date,
            time_limit_s=time_limit_s,
            prev_assignments=prev_assignments,
            pins=pins, freeze=freeze, keep_squads=keep_squads,
        )
    else:
        plan = assign(people, projects, start_date, freeze=freeze, pins=pins)
    person_stats, risks = analyze(plan, people, projects)
    return plan, person_stats, risks

