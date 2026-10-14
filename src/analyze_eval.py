# -*- coding: utf-8 -*-
"""阶段 3 辅助：逐题差异分析。

回答三个问题：
 1. 哪些题所有策略都没召回（真·检索难点）？
 2. 加重排后 top-1 变好 / 变差的分别是哪些题？
 3. 拒答题最像答案的条款是什么（干扰项是否被顶上来）？

用法: python src/analyze_eval.py data
"""
import json
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

KS = (1, 3, 5, 10)


def first_hit(seq, gold):
    for i, c in enumerate(seq):
        if c in gold:
            return i + 1
    return None


def main():
    d = Path(sys.argv[1] if len(sys.argv) > 1 else "data")
    res = json.load((d / "eval_results.json").open(encoding="utf-8"))
    items = res["items"]
    strategies = res["meta"]["strategies"]
    gold_map = {}
    src_map = {}
    ev = json.load((d / "eval_set.json").open(encoding="utf-8"))
    for it in ev["items"]:
        gold_map[it["qid"]] = it.get("sources", [])

    print("=" * 90)
    print("① 逐题名次表（数字=首次命中名次；- = 前10未命中）")
    print(f"{'题':<5}{'型':<11}{'正确答案':<32}" + "".join(f"{s:>9}" for s in strategies))
    print("-" * 90)
    miss_all = []
    for r in items:
        if r["refuse"]:
            continue
        gold = set(r["golden"])
        cells = []
        ok = False
        for s in strategies:
            h = first_hit(r["ranks"].get(s, []), gold)
            cells.append("-" if h is None else str(h))
            ok = ok or h is not None
        g = ",".join(r["golden"])
        print(f"{r['qid']:<5}{r['type']:<11}{g:<32}" + "".join(f"{c:>9}" for c in cells))
        if not ok:
            miss_all.append(r["qid"])
    print(f"\n全策略均未命中（前10）：{miss_all or '无'}")

    print("\n" + "=" * 90)
    print("② 纯向量 vs 加重排 的 top-1 分歧")
    print("-" * 90)
    for r in items:
        if r["refuse"]:
            continue
        t_dense = r["top1"].get("dense", "")
        t_rr = r["top1"].get("rerank", "")
        if t_dense != t_rr:
            gold = set(r["golden"])
            hd = first_hit(r["ranks"].get("dense", []), gold)
            hr = first_hit(r["ranks"].get("rerank", []), gold)
            flag = "✔改善" if (hr or 99) < (hd or 99) else ("✘变差" if (hr or 99) > (hd or 99) else " 平")
            print(f"{r['qid']} {flag}  命中名次 dense={hd} rerank={hr}")
            print(f"    题：{r['question']}")
            print(f"    dense top1 ：{t_dense}")
            print(f"    rerank top1：{t_rr}")

    print("\n" + "=" * 90)
    print("③ 拒答题的 top-1（看干扰项是否被顶到第一）")
    print("-" * 90)
    for r in items:
        if not r["refuse"]:
            continue
        print(f"{r['qid']}  {r['question']}")
        for s in strategies:
            print(f"    {s:<8} {r['top1'].get(s,'')}")


if __name__ == "__main__":
    main()
