# -*- coding: utf-8 -*-
"""阶段 6 · 第 1 步验收：拆解质量 + 拆解带来的召回增益。

量三件事
--------
1. 拆解触发率 —— 44 题里有多少被判为复合题（诊断用过程指标）。
2. 拆解质量   —— 逐题打印"拆/不拆 + 子问题"，供人工核对
                 （拆得对不对、有没有把简单题误拆）。
3. 召回增益   —— 对 cross/synthesis 题，比较
                 「单查询检索」vs「拆解后多查询合并检索」
                 的 gold 召回，重点看第 0 步仍漏的两条：
                 Q34 缺 F06-A023、Q35 缺 F12-A003 是否被召回。

用法
----
    python src/diag_planner.py
"""
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.stdout.reconfigure(encoding="utf-8")

from agent_qa import AgentQAPipeline          # noqa: E402
from qa_chain import build_llm                # noqa: E402
from retrievers import HandbookRetriever      # noqa: E402

TYPES = ("cross", "synthesis")


def main():
    d = json.load((ROOT / "data" / "eval_set.json").open(encoding="utf-8"))
    items = d["items"]

    r = HandbookRetriever(str(ROOT / "data"), use_rerank=True, verbose=False)
    llm = build_llm()
    ag = AgentQAPipeline(r, llm=llm, verbose=False)

    scored = [i for i in items if i.get("sources")]
    plans = {}
    for it in items:
        plans[it["qid"]] = ag.plan_only(it["question"])

    # ---------- 1) 触发率 ----------
    n_split = sum(1 for p in plans.values() if p["need_split"])
    print("=" * 76)
    print(f"拆解触发率：{n_split}/{len(items)} = {n_split/len(items):.1%}")
    by_t = Counter((it["type"], plans[it["qid"]]["need_split"]) for it in items)
    for t in sorted({i["type"] for i in items}):
        s = by_t.get((t, True), 0)
        ns = by_t.get((t, False), 0)
        print(f"  {t:<10} 拆 {s:>2} ／ 不拆 {ns:>2}")

    # ---------- 2) 逐题拆解明细（人工核对用） ----------
    print("\n" + "=" * 76)
    print("逐题拆解明细（★ = cross/synthesis，重点看）")
    for it in items:
        p = plans[it["qid"]]
        star = "★" if it["type"] in TYPES else " "
        tag = "拆" if p["need_split"] else "不拆"
        print(f"{star}{it['qid']} [{tag:>2}] {it['type']:<10} {it['question'][:40]}")
        for s in p["sub_questions"]:
            print(f"          ├ {s}")

    # ---------- 3) 召回增益（对比单查询 vs 多查询合并） ----------
    print("\n" + "=" * 76)
    print("召回增益：单查询 vs 拆解后多查询合并（cross / synthesis）")
    print(f"{'qid':<5}{'type':<11}{'gold':>5}{'单查询':>8}{'合并':>7}  漏召回→")
    rows, base_tot, merge_tot, gold_tot = [], [0, 0], [0, 0], 0
    for it in items:
        if it["type"] not in TYPES or not it.get("sources"):
            continue
        g = {s["clause_id"] for s in it["sources"]}
        base = {e.clause_id for e in r.search(it["question"], top_k=6)}
        p = plans[it["qid"]]
        if p["need_split"]:
            merged = {e.clause_id for e in
                      ag._merge([it["question"]] + p["sub_questions"])}
        else:
            merged = base
        hb, hm = len(g & base), len(g & merged)
        newly = sorted((g & merged) - base)
        still = sorted(g - merged)
        rows.append({"qid": it["qid"], "type": it["type"], "gold_n": len(g),
                     "base_hit": hb, "merge_hit": hm,
                     "newly": newly, "still_miss": still})
        base_tot[0] += hb; base_tot[1] += len(g)
        merge_tot[0] += hm; merge_tot[1] += len(g)
        mark = "  ★新召回" if newly else ""
        print(f"{it['qid']:<5}{it['type']:<11}{len(g):>5}{hb:>8}{hm:>7}"
              f"  仍缺 {still}{mark}")

    print("-" * 76)
    print(f"合计召回  单查询 {base_tot[0]}/{base_tot[1]} = "
          f"{base_tot[0]/base_tot[1]:.1%}  →  合并 "
          f"{merge_tot[0]}/{merge_tot[1]} = {merge_tot[0]/merge_tot[1]:.1%}")

    # ---------- 重点：两条真跨文件漏项 ----------
    print("\n重点漏项核验")
    for qid, cid in (("Q34", "F06-A023"), ("Q35", "F12-A003")):
        row = next((x for x in rows if x["qid"] == qid), None)
        if not row:
            continue
        ok = cid in row["newly"]
        print(f"  {qid} 的 {cid} {'✅ 已被拆解召回' if ok else '❌ 仍未召回'}")

    out = ROOT / "data" / "diag_planner.json"
    out.write_text(json.dumps({
        "n_split": n_split, "n_total": len(items),
        "plans": plans, "rows": rows,
        "base_recall": base_tot[0] / base_tot[1],
        "merge_recall": merge_tot[0] / merge_tot[1],
    }, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\n已写 {out}")


if __name__ == "__main__":
    main()
