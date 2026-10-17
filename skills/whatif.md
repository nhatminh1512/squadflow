---
name: whatif
description: >
  Phân loại và trích kịch bản giả định xếp lịch (nghỉ, thuê người, dời hạn).
  Dùng khi đầu vào là what-if: "nếu X thì sao", "nghỉ N ngày", "thuê QA", "dời deadline".
---

# What-if extractor

Trả **một** JSON, không markdown.

```
{"intent":"what_if","change":{
  "type":"leave|slip|add_person",
  "detail":"<câu gốc>",
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

## Chọn type

| Tín hiệu trong câu | type |
|---|---|
| nghỉ, nghỉ ốm, nghỉ phép, leave, off, vacation | leave |
| thuê, hire, thêm QA/PM/engineer/analyst | add_person |
| dời hạn, dời deadline, lùi hạn, slip | slip |

## leave — thời lượng

Đọc **đúng con số** người dùng nói. `days` là số ngày lịch (tính cả hôm nay).

| Câu | days | from / to |
|---|---|---|
| Ben nghỉ 2 ngày | 2 | null — engine lấy hôm nay → hôm nay+1 |
| Ivy nghỉ 2 tuần | 14 | null |
| nghỉ 1 ngày | 1 | null |
| nghỉ từ 2026-11-02 đến 2026-11-15 | null | điền from và to |
| Ivy nghỉ (không nêu thời lượng) | null | null — **cấm bịa 14 ngày** |

- "N ngày" / "N day(s)" → `days=N`. Không đổi thành tuần.
- "N tuần" / "N week(s)" → `days=N*7`.
- "hai ngày" → `days=2`. "một tuần" → `days=7`.
- Có mốc ngày cụ thể (ISO hoặc dd/mm) thì điền `from`/`to`, `days=null`.
- `person_name` = tên trong câu (Ben, Ivy, …).

## add_person

- `role` từ {pm, analyst, engineer, qa}. "QA"/"tester" → qa.
- `level` đúng số trong câu ("level 4" → 4). Không nêu → null (engine mới được gán 3).
- Cấm hạ level 6 xuống 5 trong JSON; giữ số gốc, engine sẽ cảnh báo.

## slip

- `project_name` khớp phần tên dự án (Payment gateway, API v2, …).
- `deadline` nếu câu có ngày mới; không có thì null (engine mới được +14 ngày).
