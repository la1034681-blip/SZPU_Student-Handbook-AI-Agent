# -*- coding: utf-8 -*-
"""阶段 6 · 第 2 步消融：Q36 在「关自检 / 开自检」下各跑多次

要回答的问题
------------
第 2 步全量跑完，44 题里只有 Q36 从 0.5 涨到 1.0，且它是唯一被自检
检出缺口的题。但生成本身有随机性 —— 必须排除"这次只是碰巧写全了"。

做法
----
固定同一道题（Q36「转专业的整个流程包括哪些环节？」），做两组对照：

  A 组  规划=开，自检=关   （第 1 步的状态）
  B 组  规划=开，自检=开   （第 2 步的状态）

各跑 N 次，检查答案里有没有答到 gold 的关键限定信息：
  「每学年办理一次」与「第一学年末」
（这是 Q36 判 0.5 的直接原因 —— 漏了办理频次与时间。）

若 A 组稳定漏、B 组稳定补上 → 自检的贡献确凿；
若 A 组有时也写全 → 提升不能全归功于自检。

用法
----
    python src/diag_reflect_ablation.py --repeat 3
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
import _paths  # noqa: E402 把 core/retrieval/infra/cli 也挂上搜索路径

from agent_qa import AgentQAPipeline      # noqa: E402
from qa_chain import build_llm            # noqa: E402
from retrievers import HandbookRetriever  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent.parent.parent
QID = "Q36"
# Q36 判 0.5 的直接原因：漏了这两个限定信息（见 eval_set.must_contain）
KEYS = ["每学年", "第一学年末"]


def hit(ans: str):
    return [k for k in KEYS if k in (ans or "")]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(ROOT / "data"))
    ap.add_argument("--repeat", type=int, default=3)
    a = ap.parse_args()

    ev = json.load(open(Path(a.data) / "eval_set.json", encoding="utf-8"))
    item = next(i for i in ev["items"] if i["qid"] == QID)
    q = item["question"]
    print(f"题：{q}")
    print(f"关键限定信息：{KEYS}\n")

    r = HandbookRetriever(a.data, use_rerank=True, verbose=False)
    llm = build_llm()

    summary = {}
    for tag, enable in (("A 关自检", False), ("B 开自检", True)):
        pipe = AgentQAPipeline(r, llm=llm, verbose=False, enable_reflect=enable)
        rows = []
        for k in range(a.repeat):
            res = pipe.ask(q)
            h = hit(res.answer)
            rows.append({"k": k + 1, "hit": h, "rounds": res.reflect_rounds,
                         "lat": round(res.latency, 1),
                         "len": len(res.answer or "")})
            print(f"  {tag} 第{k + 1}次：命中 {len(h)}/{len(KEYS)} {h} ｜ "
                  f"补检 {res.reflect_rounds} 轮 ｜ {res.latency:.1f}s")
        both = sum(1 for x in rows if len(x["hit"]) == len(KEYS))
        summary[tag] = {"rows": rows, "full_hit": both, "n": a.repeat}
        print(f"  → 全中 {both}/{a.repeat}\n")

    print("=" * 60)
    for tag, s in summary.items():
        print(f"{tag}：答到全部关键限定信息 {s['full_hit']}/{s['n']} 次")
    out = ROOT / "data" / "diag_reflect_ablation.json"
    json.dump({"qid": QID, "keys": KEYS, "summary": summary},
              open(out, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print(f"\n明细已存：{out}")


if __name__ == "__main__":
    main()
