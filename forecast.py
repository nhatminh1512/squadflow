"""Forecast — Monte Carlo deadline risk + per-week crunch forecast.

Two outputs, both pure data (no LLM):

  1. P(miss deadline) per project — simulate each project's actual duration
     under a lateness factor (e.g. projects historically run 0-30% long),
     count how often it finishes after its deadline.

  2. crunch forecast — per-week load per person (from the plan), so a week
     where someone is at/over capacity is visible BEFORE it happens.
"""
from __future__ import annotations

import random
from datetime import timedelta

from engine import week_of
from models import Person, Plan, Project


def monte_carlo(
    plan: Plan,
    projects: list[Project],
    n_sims: int = 500,
    lateness_lo: float = 0.0,
    lateness_hi: float = 0.3,
    seed: int = 0,
) -> dict[str, float]:
    """Per-project probability of missing its deadline.

    lateness_lo/hi bound the historical 'runs long' factor: each project's
    duration is multiplied by 1+U(lo,hi). Learn these from history in
    production; here they are parameters.
    """
    rng = random.Random(seed)
    pmeta = {p.id: p for p in projects}
    assigned = {a.project_id: a for a in plan.assignments}
    misses = {p.id: 0 for p in projects}

    for _ in range(n_sims):
        for pid, a in assigned.items():
            proj = pmeta[pid]
            f = rng.uniform(lateness_lo, lateness_hi)
            dur = max(1, round(proj.duration_weeks * (1 + f)))
            done_date = plan.start_date + timedelta(weeks=a.start_week + dur - 1)
            if done_date > proj.deadline:
                misses[pid] += 1

    return {pid: misses[pid] / n_sims for pid in misses}


def crunch_forecast(plan: Plan, people: list[Person]) -> dict[str, dict[int, int]]:
    """Per-person per-week concurrent load (>= capacity means a crunch week)."""
    return {
        pid: {w: len(s) for w, s in weeks.items()}
        for pid, weeks in plan.committed.items()
    }
