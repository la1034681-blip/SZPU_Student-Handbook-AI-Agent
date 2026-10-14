# -*- coding: utf-8 -*-
"""阶段 6 · 第 3 步：多组消融结果并排对比。

用法
----
    python src/compare_multi.py "基线=data/abl_base.json" "只拆解=data/abl_split.json" \
        "只自检=data/abl_reflect.json" "全开=data/abl_all.json"

输出
----
1. 主指标矩阵（组 × 指标）
2. 分题型正确率矩阵
3. 各组判分不一致的题（定位差异来源）
4. 自检 / 拒答前置的过程统计
"""
import json
import sys
from collections import defaultdict
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")
ROOT = Path(__file__).resolve().parent.parent.parent.parent

KEYS = ["答案正确率", "完全正确率", "要点覆盖率",
        "引用存在率", "引用支持率", "出处命中率", "误拒率", "拒答正确率",
        "拆解触发率", "自检触发率", "前置拒答数", "平均延迟"]
TYPES = ["fact", "numeric", "synthesis", "cross"]


def load(spec):
    label, _, path = spec.partition("=")
    p = Path(path)
    if not p.is_absolute():
        p = ROOT / p
    d = json.load(p.open(encoding="utf-8"))
    return label or path, d["summary"], {i["qid"]: i for i in d["items"]}, d["items"]


def fmt(v):
    if v is None:
        return "—"
    if isinstance(v, float):
        return f"{v:.4f}"
    return str(v)


def by_type(items):
    agg = defaultdict(list)
    for it in items:
        if it.get("judge_score") is not None:
            agg[it["type"]].append(it["judge_score"])
    return {t: sum(v) / len(v) for t, v in agg.items()}


def main():
    specs = sys.argv[1:]
    if not specs:
        print("用法：python src/compare_multi.py \"标签=路径\" ...")
        return
    groups = [load(s) for s in specs]
    labels = [g[0] for g in groups]

    wcol = max(10, max(len(l) for l in labels) + 2)

    def row(name, vals, width=12):
        return f"{name:<14}" + "".join(f"{fmt(v):>{width}}" for v in vals)

    print("=" * (14 + 12 * len(groups)))
    print("主指标")
    print(row("指标", labels, 12))
    print("-" * (14 + 12 * len(groups)))
    for k in KEYS:
        print(row(k, [g[1].get(k) for g in groups]))

    print("\n分题型答案正确率")
    print(row("题型", labels, 12))
    print("-" * (14 + 12 * len(groups)))
    bts = [by_type(g[3]) for g in groups]
    for t in TYPES:
        print(row(t, [b.get(t) for b in bts]))

    # 逐题判分差异
    print("\n各组判分不一致的题")
    allq = sorted(set().union(*[set(g[2]) for g in groups]))
    diffs = 0
    for qid in allq:
        seq = [g[2].get(qid, {}).get("judge_score") for g in groups]
        uniq = {s for s in seq if s is not None}
        if len(uniq) > 1:
            diffs += 1
            t = next(g[2][qid]["type"] for g in groups if qid in g[2])
            print(f"  {qid:<5}[{t:<10}] " +
                  "  ".join(f"{l}={fmt(s)}" for l, s in zip(labels, seq)))
    if not diffs:
        print("  （无差异）")

    # 过程统计
    print("\n过程统计")
    for label, _s, _i, items in groups:
        trig = [x for x in items if x.get("reflect_rounds")]
        gates = [x for x in items if x.get("gate_refused")]
        splits = [x for x in items if x.get("need_split")]
        print(f"  [{label}] 拆解 {len(splits)} ｜ 自检补检 {len(trig)} "
              f"｜ 前置拒答 {len(gates)} {[x['qid'] for x in gates]}")
        if trig:
            for x in trig:
                print(f"        {x['qid']} 补检{x['reflect_rounds']}轮 "
                      f"gaps={x.get('reflect_gaps')}")


if __name__ == "__main__":
    main()
