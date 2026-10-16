# -*- coding: utf-8 -*-
"""诊断：跨章/归纳题的 gold 条款，在「检索层」被召回多少。

为什么需要这个脚本
------------------
方案 A（智能体化）第 1 步是加「二跳检索」，前提必须是
**检索层确实漏了 gold 条款**。动手改代码之前先量一遍，避免做一个
收益为零的改动。

两个层级要分清：
    检索层召回 = search() 返回的证据里含哪些 gold（不管模型有没有引用）
    引用层召回 = 模型最终引用到的 gold（= gen_results.json 的 cited_clauses）

评测脚本里的「出处命中率」用的是弱口径 bool(gold & cited) ——
命中任意一条就算过，看不出「4 条只召回 2 条」这种部分遗漏。
本脚本按严格口径统计逐题召回比例。

用法
----
    python src/diag_gold_recall.py
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(ROOT / "src"))
import _paths  # noqa: E402 把 core/retrieval/infra/cli 也挂上搜索路径
sys.stdout.reconfigure(encoding="utf-8")

from retrievers import HandbookRetriever  # noqa: E402

TYPES = ("cross", "synthesis")


def main():
    d = json.load((ROOT / "data" / "eval_set.json").open(encoding="utf-8"))
    items = [i for i in d["items"] if i["type"] in TYPES and i.get("sources")]

    r = HandbookRetriever(str(ROOT / "data"), use_rerank=True, verbose=True)
    r.build()
    r._ensure_models()

    print("\n" + "=" * 78)
    rows = []
    for it in items:
        gs = {s["clause_id"] for s in it["sources"]}
        evs = r.search(it["question"], top_k=6)
        got = {e.clause_id for e in evs}
        hit = gs & got
        miss = sorted(gs - got)
        rows.append((it["qid"], it["type"], len(gs), len(hit), miss))
        ratio = f"{len(hit) / len(gs):.0%}"
        flag = "" if not miss else "   ← 有遗漏"
        print(f"{it['qid']} {it['type']:<10} gold {len(gs):2d} 条 → "
              f"检索层召回 {len(hit):2d}/{len(gs)} ({ratio}){flag}")
        if miss:
            print(f"      未召回：{miss}")
    print("=" * 78)

    tot_g = sum(x[2] for x in rows)
    tot_h = sum(x[3] for x in rows)
    print(f"合计：gold {tot_g} 条，检索层召回 {tot_h} 条（{tot_h / tot_g:.1%}）")
    for t in TYPES:
        sub = [x for x in rows if x[1] == t]
        if not sub:
            continue
        g = sum(x[2] for x in sub)
        h = sum(x[3] for x in sub)
        perfect = sum(1 for x in sub if not x[4])
        print(f"  {t:<10} n={len(sub):2d}  召回 {h}/{g} = {h / g:.1%}  "
              f"全召回题数 {perfect}/{len(sub)}")

    out = ROOT / "data" / "diag_gold_recall.json"
    out.write_text(json.dumps(
        [{"qid": a, "type": b, "gold_n": c, "hit_n": d, "miss": e}
         for a, b, c, d, e in rows], ensure_ascii=False, indent=1),
        encoding="utf-8")
    print(f"\n已写 {out}")


if __name__ == "__main__":
    main()
