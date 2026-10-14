# -*- coding: utf-8 -*-
"""把阶段6扩充集 eval_set_v2.json 正式并入 eval_set.json

- 前 44 题逐字保留（已校验一致）
- 追加 N01~N16 共 16 道跨章题
- meta 更新为 v2.0（count=60，cross=20，status=已人工确认）
用法： python src/merge_eval_v2.py
"""
import json
import sys
from datetime import date
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")
ROOT = Path(__file__).resolve().parent.parent.parent.parent
DATA = ROOT / "data"

formal = json.loads((DATA / "eval_set.json").read_text(encoding="utf-8"))
v2 = json.loads((DATA / "eval_set_v2.json").read_text(encoding="utf-8"))

assert formal["items"] == v2["items"][: len(formal["items"])], "前44题与v2不一致，中止"

merged = {"meta": dict(formal["meta"]), "items": v2["items"]}
meta = merged["meta"]
meta["version"] = "v2.0-stage6-expand"
meta["built_at"] = formal["meta"].get("built_at", "2026-10-10")
meta["status"] = (
    "v1.1 已确认；阶段5 拒答题（Q41-Q44）已确认；"
    "阶段6 新增 16 道跨章题（N01-N16）已由用户人工核对确认，正式并入。"
)
meta["count"] = len(merged["items"])
meta["by_type"] = {"fact": 15, "numeric": 8, "synthesis": 10, "cross": 20, "refuse": 7}
meta["expansion_note"] = (
    "N01~N16 均为跨章题，出处来自 clauses.json 真实条款，每题的 must_contain 可在原文中溯源；"
    "自动对账 16/16 通过，并经用户人工核对确认。"
)
meta["merged_at"] = date.today().isoformat()

out = DATA / "eval_set.json"
out.write_text(json.dumps(merged, ensure_ascii=False, indent=1), encoding="utf-8")
print(f"[OK] 已写入 {out}")
print(f"     题目总数 = {len(merged['items'])}  by_type = {meta['by_type']}")
print(f"     新题 qid = {[x['qid'] for x in merged['items'][-16:]]}")
