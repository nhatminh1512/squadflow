# Squadflow

Squadflow assigns a squad to each project and shows what breaks when someone is on leave or a deadline moves. A squad is one person per required role: PM, analyst, engineer, and QA.

The planner keeps hard limits: the person matches the role and meets the required skill level, is not on leave, stays within capacity, meets the deadline, and waits for dependencies. It prefers the least senior person who still qualifies, so senior people stay free for work that needs them.

The web app solves with [OR-Tools](https://developers.google.com/optimization) CP-SAT. Without that library it falls back to a greedy scheduler, and the plan is different.

## Deploy

From the repository root:

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/python web.py
```

The last command is how you start the app. It listens on `0.0.0.0:8091` and prints:

```text
squadflow UI → http://localhost:8091
```

Open that URL. The sample roster is `data/people.json` and the sample projects are `data/projects.json`. The planning date in the web app is 19 October 2026.

Use `.venv/bin/python`. System `python3` does not have OR-Tools, so the server warns and plans with the greedy fallback.

### Docker

```bash
docker compose up -d --build
```

The container publishes the same port: `http://localhost:8091`.

## What the UI shows

- Plan health, weekly allocation, project assignments, and team load.
- Why each person was picked.
- Changes: ask a what-if, paste a document to split into work packages, or edit leave, deadlines, people, and projects by hand.
- Re-optimize, then review the before/after table before you approve.

Long documents are read in parts. A part that fails is listed in red with its
reason and a **Read this part again** button; successful parts come from the
cache on a retry. Dependencies are taken only from ordering sentences in the
document — the model's guesses appear as dashed **AI guess** chips and are not
sent to the solver unless you tick them.

What-if text, document import, and the AI summary need `DEEPSEEK_API_KEY`. Assignment, load, and the rest of the plan do not.

## CLI

```bash
.venv/bin/python cli.py --today 2026-10-19 assign \
  --people data/people.json --projects data/projects.json

.venv/bin/python cli.py --today 2026-10-19 replan \
  --people data/people.json --projects data/projects.json \
  --leave eng1=2026-10-19:2026-10-30 \
  --slip p4=2026-11-27
```

`assign` prints the plan. `replan` applies leave or a slipped deadline and prints the delta. The report lists assignments, projects that cannot be staffed (and which role blocked them), utilization (`CRUNCH` / `IDLE`), and bottlenecks.

On the sample data the bottleneck is senior QA: several projects need QA at level 3 or above, and only Ivy qualifies.

## Check

```bash
.venv/bin/python test_regressions.py
```

## Requirements

- Python 3.10+
- `ortools` from `requirements.txt`, installed in `.venv`
