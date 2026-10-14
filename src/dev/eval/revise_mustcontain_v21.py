# -*- coding: utf-8 -*-
"""阶段6 · 判分口径修订（v2.0 → v2.1）

只改一处：N08 的 must_contain 里混入了题干没问的「处分类别五种」。
题干问的是"申请学位条件 / 不能授予的情形 / 受过留校察看能否拿学位"，
从未问"违纪处分分哪几种"。原要点里的「警告」「开除学籍」即来自该段
背景信息，属"题外必答点"，会造成答对主干也拿不到满分。

处置：
  - 删除 '警告'、'开除学籍'（题外背景）
  - '作结业处理' 收窄过死（模型写"办理结业"即失配）→ 放宽为 '结业'
  - 原 7 项保留在 must_contain_strict 字段，供双口径复算，留痕可回溯
其余 5 道稳定 0.5 分题（N04/N05/N09/N11/N16）经逐条核对，要点均可由
题干推出，属真实漏答，**不动**。

用法： python src/revise_mustcontain_v21.py
"""
import json
import sys
from datetime import date
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")
DATA = Path(__file__).resolve().parent.parent.parent.parent / "data"

NEW_N08 = ["留校察看及以上处分", "代写、剽窃、伪造", "修满规定学分",
           "结业", "结业后两年内"]

p = DATA / "eval_set.json"
d = json.loads(p.read_text(encoding="utf-8"))

changed = False
for it in d["items"]:
    if it["qid"] != "N08":
        continue
    old = list(it["must_contain"])
    if old == NEW_N08:
        print("[skip] N08 已是修订后口径")
        continue
    it["must_contain_strict"] = old          # 留痕：原严格口径
    it["must_contain"] = NEW_N08
    it["note"] = (it.get("note", "").rstrip()
                  + " 【v2.1口径修订】删除题干未问及的『警告/开除学籍』"
                    "（源自处分类别五种，属背景信息），『作结业处理』放宽为"
                    "『结业』；原要点见 must_contain_strict。")
    changed = True
    print(f"[N08] {old} -> {NEW_N08}")

if changed:
    d["meta"]["version"] = "v2.1-stage6-judge-revised"
    d["meta"]["revision"] = (
        "v2.1：修订 N08 判分要点，剔除题干未问及的题外必答点"
        "（警告/开除学籍），『作结业处理』放宽为『结业』；"
        "原要点留存于 must_contain_strict 以便双口径复算。"
    )
    d["meta"]["revision_at"] = date.today().isoformat()
    p.write_text(json.dumps(d, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"[OK] 已写入 {p}  version={d['meta']['version']}")
else:
    print("[no-op] 无改动")
