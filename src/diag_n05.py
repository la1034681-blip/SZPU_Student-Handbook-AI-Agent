# -*- coding: utf-8 -*-
"""N05 专项诊断：拆解后的检索证据里，到底有没有"退宿"那一条（F38-A015）。

用户反馈 N05 会摇摆（有时整题拒答）。先分清是"证据没进来"还是"进来但不认"。
    python src/diag_n05.py            # 默认带规划
    python src/diag_n05.py --no-planner
"""
import argparse
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, str(Path(__file__).resolve().parent))

from agent_qa import AgentQAPipeline          # noqa: E402
from qa_chain import build_llm, load_env_file  # noqa: E402
from retrievers import HandbookRetriever       # noqa: E402

Q = "哪些情况会被退学处理？退学后还能拿到什么证书或证明？退学后宿舍怎么退？"
GOLD = {"F03-A038", "F03-A041", "F03-A048", "F04-A039", "F38-A015"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-planner", action="store_true")
    a = ap.parse_args()

    r = HandbookRetriever("data", use_rerank=True, verbose=False)
    load_env_file()
    llm = build_llm()
    pipe = AgentQAPipeline(r, llm=llm, top_k=6, enable_planner=not a.no_planner,
                           enable_reflect=True, enable_gate=True, verbose=False)

    print("=" * 70)
    print("规划：", "关" if a.no_planner else "开")
    res = pipe.ask(Q)
    print("plan:", res.plan)
    print(f"\n证据 {len(res.evidences)} 条：")
    got = set()
    for e in res.evidences:
        mark = "★" if e.clause_id in GOLD else " "
        got.add(e.clause_id)
        print(f"  {mark} [{e.eid}] {e.clause_id:<12} "
              f"{(e.cite or '')[:44]}")
    print(f"\ngold 命中 {len(GOLD & got)}/{len(GOLD)}：{sorted(GOLD & got)}")
    print("缺：", sorted(GOLD - got))
    print(f"\n拒答={res.refused}  原因={res.refuse_reason}")
    print(f"自检轮数={res.reflect_rounds}  检索次数={res.hops}")
    print("\n答案：", (res.answer or "")[:600])
    v = res.verify or {}
    sup = v.get("support") or {}
    print(f"\n校验：总体={v.get('passed')} 支持性={sup.get('passed')} "
          f"support={sup.get('n_support')} hard_no={sup.get('n_hard_no')}")
    for d in (sup.get("verdicts") or [])[:40]:
        if d.get("verdict") != "support":
            print(f"   [{d['verdict']}] {d['claim'][:52]}  ← {d.get('reason','')[:40]}")


if __name__ == "__main__":
    main()
