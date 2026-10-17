# squadflow

A project-to-squad assignment engine for an inter-disciplinary team with
shifting availability and varied expertise. It answers, mechanically, the
question the team currently answers by hand: *who takes what, and what breaks
if someone leaves or a deadline slips?*

## The problem it solves

Incoming customer projects (each with size, deadline, required expertise)
must each be staffed by a squad drawn from PMs, analysts, engineers, and QA.
Requirements shift: people go on leave, deadlines slip, and the crunch is
only ever discovered after it has already happened. squadflow assigns the
projects, flags who is drowning vs. idle, and re-plans automatically when the
picture changes.

## Architecture — Hermes Agent's task-splitting, applied

squadflow is deliberately shaped like Hermes Agent's core. One planning pass
decomposes into the same kind of phases Hermes decomposes a turn:

| Hermes Agent | squadflow | where |
|---|---|---|
| turn loop (`agent/turn_*.py`) — a turn is a pipeline of discrete phases | one pass = `ingest → model → plan → predict → report` | `pipeline.py` |
| facade + siblings — god files split by topic | one module per phase / concern | `models.py` `engine.py` `analysis.py` `report.py` `cli.py` |
| `batch_runner._process_batch_worker` / `delegate_task` — independent subtasks in parallel | per-person & per-project metrics computed in a `ThreadPoolExecutor` | `analysis.py` |
| memory / state — declarative facts | domain objects are plain dataclasses, no logic | `models.py` |
| skills — procedural knowledge loaded on demand | what-if extractor; document → work packages (manager) | `skills/whatif.md` `skills/decompose.md` |
| cron / delivery queue — scheduled jobs, incidents | re-plan on change + unstaffable "incidents" surface as a report | `cli.py replan`, `analysis.py` |
| tools registry — small discoverable subcommands | `assign` / `replan`, one job each | `cli.py` |

The assignment engine is a deterministic, replayable "plan" phase; the
analysis is an embarrassingly parallel "predict" phase. Re-planning re-runs
only `plan + predict` and diffs against the previous plan — the same shape as
a turn re-entering its loop after a mid-flight change.

## Model

- **Person** — `role`, `skills` (role → proficiency 1–5), `capacity` (max
  concurrent projects), `leave` (date ranges).
- **Project** — `required` (role → min proficiency), `duration_weeks`,
  `deadline`, `priority`, optional `depends_on`.
- **Assignment** — a squad: one person per required role, plus a start week.

Hard constraints (never violated): role match + proficiency ≥ required, no
booking during leave, concurrent load ≤ capacity, deadline, dependencies.

The objective is realized by one domain rule — **least-qualified-acceptable**:
a slot is filled by the *least* senior person who still meets the level, so
seniors stay free for senior-only work. (Assigning the *most* senior person
first is the trap that makes seniors drown and juniors idle.)

## Usage

### Web UI

```bash
cd squadflow
python3 web.py            # → http://localhost:8091
```

Or Docker:

```bash
docker compose up -d --build   # → http://<host>:8091
```

The UI shows a live plan: a week-by-week timeline per person, utilization
bars, unstaffable/bottleneck alerts, and controls to change the picture
(add leave, slip a deadline, add a person or project) — every change re-plans
immediately.

### CLI

```bash
cd squadflow

# assign incoming projects and print the plan
python3 cli.py --today 2026-10-19 assign \
  --people data/people.json --projects data/projects.json

# plans change: someone goes on leave, a deadline slips → re-plan + diff
python3 cli.py --today 2026-10-19 replan \
  --people data/people.json --projects data/projects.json \
  --leave eng1=2026-10-19:2026-10-30 \
  --slip  p4=2026-11-27
```

The report shows four things the team currently works out by hand:

1. **Assignments** — project → start week → squad.
2. **Unstaffable** — projects that cannot be staffed, and *which role* blocked them.
3. **Utilization** — per person: booked vs. available weeks, with `CRUNCH` /
   `IDLE` flags.
4. **Bottlenecks** — the (role, level) combinations with unmet demand.

## Sample run (what it surfaces)

On `data/`, the engine spreads work so nobody is idle and flags one
irreducible bottleneck:

```
BOTTLENECKS (roles with unmet demand)
  qa >= 3: no available person (crunch / leave / wrong skill)
```

That single line is the "crunch nobody saw coming": three projects need
level-3 QA, but the team has exactly one person who can do it. The tool cannot
manufacture senior QA capacity — but it says precisely where to act (hire,
upskill, or defer p1/p7), instead of discovering it after deadlines slip.

## Requirements

Python 3.10+ standard library only — no dependencies.
