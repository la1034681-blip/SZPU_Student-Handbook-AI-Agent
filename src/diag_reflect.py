# -*- coding: utf-8 -*-
"""阶段 6 · 第 2 步诊断：自检循环的检出率与补检成功率

它回答三个问题
--------------
1. 自检查得出缺口吗？        → 缺口检出率
2. 查出缺口后补检有用吗？    → 补检成功率（带回新证据 / 重答后答案变化）
3. 会不会空转 / 误判？       → 轮数分布、缺口清单，供人工核对

用法
----
    python src/diag_reflect.py                 # 跑默认代表题集
    python src/diag_reflect.py --qids Q36,Q24  # 指定题
    python src/diag_reflect.py --all           # 跑全部非拒答题
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, str(Path(__file__).resolve().parent))

from agent_qa import AgentQAPipeline      # noqa: E402
from qa_chain import build_llm            # noqa: E402
from retrievers import HandbookRetriever  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_QIDS = ["Q36", "Q24", "Q34", "Q35", "Q37", "Q10", "Q16", "Q30"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(ROOT / "data"))
    ap.add_argument("--qids", default="")
    ap.add_argument("--all", action="store_true")
    a = ap.parse_args()

    ev = json.load(open(Path(a.data) / "eval_set.json", encoding="utf-8"))
    items = ev["items"]

    if a.all:
        # 全部非拒答题（拒答题走 Refuse 分支，自检会被跳过）
        qids = [i["qid"] for i in items if i["type"] != "refuse"]
    elif a.qids:
        qids = [x.strip() for x in a.qids.split(",") if x.strip()]
    else:
        qids = [q for q in DEFAULT_QIDS if any(i["qid"] == q for i in items)]

    qmap = {i["qid"]: i for i in items}

    r = HandbookRetriever(a.data, use_rerank=True, verbose=False)
    pipe = AgentQAPipeline(r, llm=build_llm(), verbose=False)

    print("=" * 78)
    print(f"自检循环诊断：{len(qids)} 道题")
    print("=" * 78)

    n_gap_detected = 0     # 检出缺口的题数
    n_loop_entered = 0     # 进入补检循环的题数
    n_grew = 0             # 补检带回新证据的题数
    n_answer_changed = 0   # 终答与首答不同的题数
    rows = []

    for qid in qids:
        it = qmap[qid]
        q = it["question"]
        res = pipe.ask(q)

        gaps_all = []
        for lg in res.reflect_log:
            gaps_all += lg.get("gaps") or []
        detected = bool(gaps_all) and not all(
            lg.get("covered") for lg in res.reflect_log)
        entered = res.reflect_rounds > 0
        changed = (res.answer or "").strip() != (res.first_answer or "").strip()

        if detected:
            n_gap_detected += 1
        if entered:
            n_loop_entered += 1
        if res.n_evidence > 0 and entered:
            pass
        if changed:
            n_answer_changed += 1

        print(f"\n── {qid} [{it['type']}] {'【复合】' if res.plan.get('need_split') else '【简单】'}")
        print(f"   问：{q[:60]}")
        print(f"   首答 {len(res.first_answer or '')} 字 ｜ 终答 {len(res.answer or '')} 字 ｜ "
              f"补检 {res.reflect_rounds} 轮 ｜ 证据 {res.n_evidence} 条 ｜ 延迟 {res.latency:.1f}s")
        if res.plan.get("need_split"):
            print(f"   拆解：{' / '.join(res.plan.get('sub_questions') or [])}")
        for i, lg in enumerate(res.reflect_log, 1):
            gaps = lg.get("gaps") or []
            tag = ("通过" if lg.get("covered") else
                   ("手册无此内容" if lg.get("hopeless") else f"{len(gaps)} 处缺口"))
            print(f"   自检第{i}轮：{tag}")
            for g in gaps:
                print(f"        · {g}")
        if changed:
            print(f"   ★ 终答有变（自检起了作用）")
        rows.append({"qid": qid, "type": it["type"], "detected": detected,
                     "rounds": res.reflect_rounds, "changed": changed,
                     "n_ev": res.n_evidence, "latency": round(res.latency, 2)})

    print("\n" + "=" * 78)
    print("汇总")
    print("=" * 78)
    n = len(rows)
    print(f"  题数                {n}")
    print(f"  检出缺口            {n_gap_detected}  ({n_gap_detected / n * 100:.1f}%)   ← 缺口检出率")
    print(f"  进入补检循环        {n_loop_entered}")
    print(f"  终答发生改变        {n_answer_changed}   ← 自检实质起作用")
    print(f"  平均延迟            {sum(r_['latency'] for r_ in rows) / n:.2f}s")
    print(f"  平均证据条数        {sum(r_['n_ev'] for r_ in rows) / n:.1f}")

    out = ROOT / "data" / "diag_reflect.json"
    json.dump({"rows": rows}, open(out, "w", encoding="utf-8"),
              ensure_ascii=False, indent=2)
    print(f"\n明细已存：{out}")


if __name__ == "__main__":
    main()
