# -*- coding: utf-8 -*-
"""解析结果核查：正文是否丢失、页码是否对得上、表格是否可用。
用法: python verify_parse.py <clauses.json> <chunks.jsonl> <tables.json> [clause_id ...]
"""
import json
import sys
from collections import Counter
from pathlib import Path

clauses = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
chunks = [json.loads(l) for l in Path(sys.argv[2]).read_text(encoding="utf-8").splitlines()]
tables = json.loads(Path(sys.argv[3]).read_text(encoding="utf-8"))
want = sys.argv[4:]

arts = [c for c in clauses if c["level"] == "article"]
print(f"条级 {len(arts)}  附件 {len(clauses) - len(arts)}  块 {len(chunks)}  表 {len(tables)}")
print("块类型:", Counter(c["chunk_type"] for c in chunks))

empty = [c for c in arts if len(c["text"]) <= len(c["article_label"]) + 2]
print(f"\n正文疑似缺失（text 仅等于标签行）: {len(empty)}")
for c in empty[:15]:
    print(f"  {c['clause_id']} p{c['page_start']} {c['article_label'][:30]!r}")

lens = sorted((c["char_len"], c["clause_id"]) for c in arts)
print(f"\n条款字数 中位 {lens[len(lens)//2][0]}  最短 {lens[0]}  最长 {lens[-1]}")
print("最短 8 条:", lens[:8])

print("\n表格样张:")
for t in tables[:3]:
    print(f"  {t['table_id']} p{t['page']} {len(t['rows'])}行 x "
          f"{max(len(r) for r in t['rows'])}列 src={t['source']}")
    for r in t["rows"][:4]:
        print("     " + " | ".join(r))

for cid in want:
    hit = [c for c in clauses if c["clause_id"] == cid]
    print(f"\n===== {cid} =====")
    if not hit:
        print("  未找到")
        continue
    c = hit[0]
    print(f"  文件 {c['file_title']} | {c['chapter']} | p{c['page_start']} | {c['article_label']}")
    print(f"  正文: {c['text'][:300]}")
    for cl in c["clauses"]:
        print(f"    款 {cl['label']} {cl['text'][:80]}")
