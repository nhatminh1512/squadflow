---
name: whatif
description: >
  Classify and extract a scheduling what-if scenario (leave, hire, deadline change).
  Use when the input is a what-if: "what if X", "off for N days", "hire a QA", "move the deadline".
---

# What-if extractor

The input may be English or Vietnamese. Return **one** JSON object, no markdown.

```
{"intent":"what_if","change":{
  "type":"leave|slip|add_person",
  "detail":"<original sentence>",
  "person_name":str|null,
  "days":int|null,
  "from":"YYYY-MM-DD"|null,
  "to":"YYYY-MM-DD"|null,
  "project_name":str|null,
  "deadline":"YYYY-MM-DD"|null,
  "role":"pm|analyst|engineer|qa"|null,
  "level":int|null,
  "name":str|null
}}
```

## Picking the type

| Signal in the sentence | type |
|---|---|
| leave, off, vacation, sick, nghỉ, nghỉ ốm, nghỉ phép | leave |
| hire, add a QA/PM/engineer/analyst, thuê, thêm | add_person |
| move/push the deadline, slip, dời hạn, lùi hạn | slip |

## leave — duration

Use **exactly the number** the user said. `days` is calendar days (today included).

| Sentence | days | from / to |
|---|---|---|
| Ben is off 2 days | 2 | null — the engine uses today → today+1 |
| Ivy on leave 2 weeks | 14 | null |
| off 1 day | 1 | null |
| off from 2026-11-02 to 2026-11-15 | null | fill from and to |
| Ivy is on leave (no duration) | null | null — **never invent 14 days** |

- "N days" / "N ngày" → `days=N`. Do not convert to weeks.
- "N weeks" / "N tuần" → `days=N*7`.
- "two days" / "hai ngày" → `days=2`. "one week" / "một tuần" → `days=7`.
- With explicit dates (ISO or dd/mm) fill `from`/`to` and set `days=null`.
- `person_name` = the name in the sentence (Ben, Ivy, …).

## add_person

- `role` from {pm, analyst, engineer, qa}. "QA"/"tester" → qa.
- `level` exactly as written ("level 4" → 4). Not stated → null (only the engine may default to 3).
- Never lower level 6 to 5 in the JSON; keep the original number, the engine will warn.

## slip

- `project_name` matches part of the project name (Payment gateway, API v2, …).
- `deadline` if the sentence gives a new date; otherwise null (only the engine may add 14 days).
