# -*- coding: utf-8 -*-
"""看看"模型自述证据不足"该不该无条件采信。

背景：换 GLM-4-Flash 后 Q06（哪些情况不允许转专业？）被误拒 ——
模型说"证据只讲不予申请，没讲不允许"，其实是一条规定的两种说法。
而 Q38~Q44 那 7 道是真该拒的题。

想法：用**检索分**做二次复核。已知作答题重排分中位 0.976、拒答题 0.017，
如果 Q06 的检索分明显高于 7 道拒答题，就能安全地驳回这种保守误判。

本脚本只做观测，不改任何逻辑：
  对每道题单独跑一次检索，记录 top1 重排分，再看模型实际判没判拒答。
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
import _paths  # noqa: E402 把 core/retrieval/infra/cli 也挂上搜索路径

from qa_chain import build_llm, load_env_file      # noqa: E402
from retrievers import HandbookRetriever           # noqa: E402
from agent_qa import AgentQAPipeline               # noqa: E402

load_env_file()

ROOT = Path(__file__).resolve().parent.parent.parent.parent
items = {i["qid"]: i for i in
         json.load((ROOT / "data" / "eval_set.json").open(encoding="utf-8"))["items"]}

r = HandbookRetriever(ROOT / "data", use_rerank=True, verbose=False)
pipe = AgentQAPipeline(r, llm=build_llm(), verbose=False)

CASES = ["Q06"] + [f"Q{n}" for n in range(38, 45)]
print(f"{'题号':<6}{'top1分':<9}{'证据数':<8}{'模型拒答':<10}问题")
print("-" * 78)
rows = []
for qid in CASES:
    q = items[qid]["question"]
    evs = r.search(q, top_k=6) if qid == "Q06" else pipe._merge([q])
    top1 = max((e.score or 0.0) for e in evs) if evs else 0.0
    res = pipe.ask(q)
    rows.append({"qid": qid, "top1": round(top1, 4),
                 "n_ev": len(evs), "refused": bool(res.refused),
                 "reason": (res.refuse_reason or "")[:70]})
    print(f"{qid:<6}{top1:<9.4f}{len(evs):<8}{str(bool(res.refused)):<10}{q[:34]}")

json.dump(rows, (ROOT / "data" / "refuse_confirm_probe.json").open(
    "w", encoding="utf-8"), ensure_ascii=False, indent=1)
print("\n结果已写入 data/refuse_confirm_probe.json")
