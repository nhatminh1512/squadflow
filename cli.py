"""CLI — the tool surface (Hermes's 'tools registry': small, discoverable
subcommands rather than one god command).

  python cli.py assign --people data/people.json --projects data/projects.json
  python cli.py replan --leave engineer1=2026-11-02:2026-11-13 \
                       --slip p3=2026-12-15 ...
"""
from __future__ import annotations

import argparse
from datetime import date

import pipeline
from models import Person, Project
from report import render


def _iso(s: str) -> date:
    return date.fromisoformat(s)


def _apply_leave(people: list[Person], spec: str) -> None:
    pid, rng = spec.split("=", 1)
    a, b = rng.split(":")
    person = next(p for p in people if p.id == pid)
    person.leave.append((_iso(a), _iso(b)))


def _apply_slip(projects: list[Project], spec: str) -> None:
    pid, d = spec.split("=", 1)
    proj = next(p for p in projects if p.id == pid)
    proj.deadline = _iso(d)


def _assignment_key(plan):
    return {a.project_id: (a.start_week, dict(sorted(a.squad.items()))) for a in plan.assignments}


def cmd_assign(args) -> None:
    people = pipeline.load_people(args.people)
    projects = pipeline.load_projects(args.projects)
    plan, pstats, risks = pipeline.run(people, projects, args.today)
    print(render(plan, people, pstats, risks, projects))


def cmd_replan(args) -> None:
    people = pipeline.load_people(args.people)
    projects = pipeline.load_projects(args.projects)

    before, _, _ = pipeline.run(people, projects, args.today)

    for spec in args.leave or []:
        _apply_leave(people, spec)
    for spec in args.slip or []:
        _apply_slip(projects, spec)

    after, pstats, risks = pipeline.run(people, projects, args.today)
    print(render(after, people, pstats, risks, projects))

    # ---- delta: what changed vs the original plan ----
    kb, ka = _assignment_key(before), _assignment_key(after)
    print("CHANGES FROM ORIGINAL PLAN")
    print("-" * 78)
    changed = False
    for pid in sorted(set(kb) | set(ka)):
        if pid not in kb:
            print(f"  {pid:<16} NEW -> wk{ka[pid][0]}")
            changed = True
        elif pid not in ka:
            print(f"  {pid:<16} DROPPED (was wk{kb[pid][0]})")
            changed = True
        elif kb[pid][0] != ka[pid][0]:
            print(f"  {pid:<16} wk{kb[pid][0]} -> wk{ka[pid][0]}")
            changed = True
        elif kb[pid][1] != ka[pid][1]:
            print(f"  {pid:<16} re-staffed (wk{kb[pid][0]} unchanged)")
            changed = True
    if not changed:
        print("  (no assignments changed)")


def main() -> None:
    ap = argparse.ArgumentParser(prog="squadflow", description="Project-to-squad assignment engine")
    ap.add_argument("--today", type=_iso, default=date.today(), help="planning horizon start (default: today)")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p_assign = sub.add_parser("assign", help="assign incoming projects and print the plan")
    p_assign.add_argument("--people", required=True)
    p_assign.add_argument("--projects", required=True)
    p_assign.set_defaults(fn=cmd_assign)

    p_replan = sub.add_parser("replan", help="apply a change (leave / slip) and re-plan, showing the delta")
    p_replan.add_argument("--people", required=True)
    p_replan.add_argument("--projects", required=True)
    p_replan.add_argument("--leave", action="append", metavar="PID=YYYY-MM-DD:YYYY-MM-DD")
    p_replan.add_argument("--slip", action="append", metavar="PID=YYYY-MM-DD")
    p_replan.set_defaults(fn=cmd_replan)

    args = ap.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
