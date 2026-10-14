# -*- coding: utf-8 -*-
"""生成「新增跨章题人工核对清单」→ 桌面 txt。

排版遵循用户偏好：不用 Markdown 语法，用 ═/─ 分隔，标签 ｜ 值。
用法：python src/gen_review_list.py
"""
import json
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")
ROOT = Path(__file__).resolve().parent.parent
DESK = Path.home() / "Desktop"

items = json.loads((ROOT / "data" / "new_items_draft.json").read_text(encoding="utf-8"))

L = []
w = L.append
w("═" * 66)
w("　学生手册评测集 · 新增跨章题 人工核对清单")
w("　共 16 题（N01~N16）　｜　2026-10-11")
w("═" * 66)
w("")
w("这份清单请你过目。每题都从手册真实条款起草，出处（哪份文件、")
w("第几条、第几页）由脚本自动从条款库取出，不是手写。16 题已全部")
w("通过自动对账：出处真实存在、判定要点能在原文里逐条找到、每题")
w("确实横跨 3~4 份文件。")
w("")
w("你只需看三件事：")
w("　1. 题干问法是否自然、像学生会问的；")
w("　2. 判定要点是否合理（有没有漏掉该考的关键点）；")
w("　3. 是否确实算「跨章题」（答案是否真要翻几份文件）。")
w("")
w("有问题的题号告诉我，我改完再并入评测集；没问题就整批启用。")

for it in items:
    files = sorted({s["file_id"] for s in it["sources"]})
    w("")
    w("═" * 66)
    w(f"　{it['qid']}　（跨 {len(files)} 份文件：{' / '.join(files)}）")
    w("═" * 66)
    w(f"题干　｜ {it['question']}")
    w("")
    w("出处　｜")
    for s in it["sources"]:
        w(f"　　{s['clause_id']}　{s['file_title']}　第{s['article_no']}条"
          f"　（第{s['page']}页）")
    w("")
    w("判定要点（全部答到才算完整）　｜")
    for k in it["must_contain"]:
        w(f"　　· {k}")
    w("")
    w(f"为何算跨章　｜ {it['note'].strip()}")
    w("")
    w(f"自动对账　｜ 通过 —— 出处存在 / 要点可溯源 / 文件数 {len(files)}")

w("")
w("═" * 66)
w("　核对结论（请你填）")
w("═" * 66)
w("")
w("　· 全部可用，直接并入评测集　　→ 回我「都可以」")
w("　· 部分要改　　　　　　　　　　→ 回我要改的题号 + 怎么改")
w("")

text = "\n".join(L) + "\n"
dst = DESK / "学生手册_阶段6_新跨章题核对清单.txt"
dst.write_text(text, encoding="utf-8")
print(f"已写出 {dst}　共 {len(text)} 字")
