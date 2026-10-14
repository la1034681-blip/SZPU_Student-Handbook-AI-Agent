# -*- coding: utf-8 -*-
"""解析质量体检：找出过切分、缺号、定位异常的文件。
用法: python report_quality.py <clauses.json> <解析问题汇总.json>
"""
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

clauses = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
prob = json.loads(Path(sys.argv[2]).read_text(encoding="utf-8"))
rep = {r["file"]: r for r in prob["files_report"]}

print("文件  环  章  条数  最大条号  重复/虚增  页码区间  标题")
print("-" * 96)
bad = []
for r in prob["files_report"]:
    if "error" in r:
        print(f"{r['file']}  !! {r['error']}  {r['title'][:30]}")
        continue
    n_art, seq_end = r["articles"], r.get("seq_end") or 0
    dup = n_art - seq_end if seq_end else n_art
    flag = ""
    if dup > max(3, seq_end * 0.1):
        flag = " <== 过切分"
        bad.append(r["file"])
    print(f"{r['file']}  {r['chapters']:>2}  {r['articles']:>4}  {seq_end:>6}  "
          f"{dup:>8}  {r['pages']:>9}  {r['title'][:26]}{flag}")
print("-" * 96)
print(f"合计条级条款 {len(clauses)}")

print("\n条号分布 top10:", Counter(c["article_no"] for c in clauses
                                 if c["article_no"]).most_common(10))
print("无条号条款:", prob["articles_without_number"])

print("\n过切分嫌疑文件:", bad)
print("\n各文件条号区间:")
for r in prob["files_report"]:
    if r.get("seq_start") is not None:
        print(f"  {r['file']} {r['title'][:24]:<26} {r['seq_start']:>3} -> {r['seq_end']:>3}"
              f"  breaks={r['continuity_breaks']}")

print("\n样张（每文件首条，最多 12 条）:")
seen = set()
for c in clauses:
    if c["file_id"] in seen:
        continue
    seen.add(c["file_id"])
    if len(seen) > 12:
        break
    print(f"  [{c['clause_id']}] p{c['page_start']} {c['article_label'][:14]} | "
          f"{c['text'][len(c['article_label']):][:70]}")
