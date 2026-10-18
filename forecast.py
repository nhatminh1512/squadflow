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


def _sample_weeks(proj: Project, rng: random.Random, lo: float, hi: float) -> int:
    """One simulated duration. Waterfall projects overrun per stage (testing
    stages most, methodology.STAGE_OVERRUN); others by one factor in [lo, hi]."""
    if proj.stages:
        import methodology
        return max(1, round(sum(st["weeks"] * (1 + rng.uniform(0, methodology.STAGE_OVERRUN.get(st["stage"], hi)))
                                for st in proj.stages)))
    return max(1, round(proj.duration_weeks * (1 + rng.uniform(lo, hi))))


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
            dur = _sample_weeks(proj, rng, lateness_lo, lateness_hi)
            done_date = plan.start_date + timedelta(weeks=a.start_week + dur - 1, days=4)
            if done_date > proj.deadline:
                misses[pid] += 1

    return {pid: misses[pid] / n_sims for pid in misses}


def p80_finish(plan: Plan, projects: list[Project], n_sims: int = 500,
               lateness_lo: float = 0.0, lateness_hi: float = 0.3,
               seed: int = 0) -> dict[str, str]:
    """Date each staffed project finishes by in 80% of the same simulated runs."""
    rng = random.Random(seed)
    pmeta = {p.id: p for p in projects}
    runs: dict[str, list[int]] = {a.project_id: [] for a in plan.assignments}
    for _ in range(n_sims):
        for a in plan.assignments:
            proj = pmeta[a.project_id]
            runs[a.project_id].append(a.start_week + _sample_weeks(proj, rng, lateness_lo, lateness_hi) - 1)
    out = {}
    for pid, weeks in runs.items():
        weeks.sort()
        w = weeks[min(len(weeks) - 1, int(0.8 * len(weeks)))]
        out[pid] = (plan.start_date + timedelta(weeks=w, days=4)).isoformat()
    return out


def crunch_forecast(plan: Plan, people: list[Person]) -> dict[str, dict[int, float]]:
    """Per-person per-week load in full-time roles (> capacity means a crunch week)."""
    from engine import WEEK
    return {
        pid: {w: u / WEEK for w, u in weeks.items()}
        for pid, weeks in plan.load.items()
    }
