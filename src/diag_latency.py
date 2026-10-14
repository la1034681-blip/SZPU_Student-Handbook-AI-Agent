# -*- coding: utf-8 -*-
"""延迟分解诊断：把一次 ask() 拆成「规划 / 首轮检索 / 生成校验 / 自检 / 补检 / 重生成」计时。

用法：
    python src/diag_latency.py N04,Q36,N06,Q24,N10
（先 warmup 把模型冷启动吃掉，再看真实分段耗时。）
"""
from __future__ import annotations

import json
import sys
import time

sys.stdout.reconfigure(encoding="utf-8")

from agent_qa import AgentQAPipeline   # noqa: E402
from qa_chain import build_llm         # noqa: E402
from retrievers import HandbookRetriever  # noqa: E402


def main():
    qids = (sys.argv[1] if len(sys.argv) > 1 else "N04,Q36,N06,Q24,N10").split(",")
    ev = {it["qid"]: it for it in
          json.load(open("data/eval_set.json", encoding="utf-8"))["items"]}

    r = HandbookRetriever("data", use_rerank=True, verbose=False)
    llm = build_llm()
    pipe = AgentQAPipeline(r, llm=llm, verbose=False)

    # ---------------- warmup：吃掉模型冷启动 ----------------
    t = time.time(); r.search("旷课多少学时会被处分", top_k=6)
    print(f"[warmup] 检索 {time.time()-t:.2f}s", flush=True)
    t = time.time(); llm.invoke("请以 json 输出 {\"ok\": 1}")
    print(f"[warmup] LLM  {time.time()-t:.2f}s", flush=True)
    t = time.time(); r.search("奖学金", top_k=6)
    print(f"[warmup] 检索(热) {time.time()-t:.2f}s", flush=True)

    print()
    print(f"{'题号':<5}{'规划':>7}{'首轮检索':>10}{'生成+校验':>11}"
          f"{'自检':>7}{'补检':>7}{'重生成':>8}{'合计':>8}  说明")
    rows = []
    for qid in qids:
        q = ev[qid]["question"]
        # --- 规划 ---
        t0 = time.time(); plan = pipe.plan_only(q); t_plan = time.time() - t0
        need = bool(plan.get("need_split"))
        queries = ([q] + plan["sub_questions"]) if need else [q]
        # --- 首轮检索（合并）---
        t0 = time.time(); evs = pipe._merge(queries); t_ret = time.time() - t0
        # --- 生成 + 校验 ---
        t0 = time.time(); res = pipe.base.answer_from_evidences(q, evs)
        t_gen = time.time() - t0
        t_ref = t_remerge = t_regen = 0.0
        # --- 自检回环 ---
        if pipe.enable_reflect and not res.refused:
            t0 = time.time(); a = pipe.reflector.assess(q, res.answer, res.evidences)
            t_ref = time.time() - t0
            gaps = a.get("gaps") or []
            if not a.get("covered") and not a.get("hopeless") and gaps:
                t0 = time.time(); new_evs = pipe._merge_into(res.evidences, gaps)
                t_remerge = time.time() - t0
                t0 = time.time()
                res = pipe.base.answer_from_evidences(q, new_evs, hints=gaps)
                t_regen = time.time() - t0
        total = t_plan + t_ret + t_gen + t_ref + t_remerge + t_regen
        note = f"split={need} nq={len(queries)} nev={len(evs)}"
        if t_regen:
            note += "  [自检重答]"
        print(f"{qid:<5}{t_plan:>6.1f}s{t_ret:>9.1f}s{t_gen:>10.1f}s"
              f"{t_ref:>6.1f}s{t_remerge:>6.1f}s{t_regen:>7.1f}s{total:>7.1f}s  {note}",
              flush=True)
        rows.append(dict(qid=qid, plan=t_plan, ret=t_ret, gen=t_gen,
                         remerge=t_remerge, regen=t_regen, total=total))

    print()
    if rows:
        n = len(rows)
        print("=== 均值 ===")
        for k in ["plan", "ret", "gen", "remerge", "regen", "total"]:
            print(f"  {k:<9}{sum(x[k] for x in rows)/n:>7.2f}s")


if __name__ == "__main__":
    main()
