# -*- coding: utf-8 -*-
"""阶段 6 · 第 3 步：稳定性结果汇总（多配置并排）。

读 stability_test.py 产出的 json（含 per_round / flip_rates / agent_flip / _runs），
输出：跨配置的指标均值与极差、判分翻转率、Agent 组件跨轮波动、分题型跨轮准确率、
以及逐题分数矩阵（跨配置不一致的题）。

用法：
    python src/compare_stability.py "全开=data/stab_all.json" "只自检=data/stab_reflect.json"
"""
import json
import sys
from collections import defaultdict
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")
ROOT = Path(__file__).resolve().parent.parent.parent.parent

METRIC_KEYS = ["答案正确率", "引用支持率", "出处命中率", "误拒率", "拒答正确率"]
TYPES = ["fact", "numeric", "synthesis", "cross"]


def load(spec):
    label, _, path = spec.partition("=")
    p = Path(path)
    if not p.is_absolute():
        p = ROOT / p
    return label or path, json.load(p.open(encoding="utf-8"))


def load_types():
    d = json.load((ROOT / "data" / "eval_set.json").open(encoding="utf-8"))
    return {i["qid"]: i["type"] for i in d["items"]}


def cross_by_round(d, qtype):
    """逐轮分题型答案正确率（拒答题在 round_metrics 里已剔除，这里只算可判分题）。"""
    n = d["meta"]["rounds"]
    out = defaultdict(list)
    for k in range(n):
        acc = defaultdict(list)
        for qid, rec in d["_runs"][k].items():
            t = qtype.get(qid)
            if t is None:
                continue
            acc[t].append(rec.get("score") if not rec.get("refused") else 0.0)
        for t, ss in acc.items():
            ss = [s for s in ss if s is not None]
            if ss:
                out[t].append(round(sum(ss) / len(ss), 4))
    return out


def main():
    specs = sys.argv[1:]
    groups = [load(s) for s in specs]
    labels = [g[0] for g in groups]
    qtype = load_types()

    print("=" * (24 + 18 * len(groups)))
    print("稳定性汇总（各配置独立跑 N 轮）")
    for label, d in groups:
        print(f"  [{label}] 轮数={d['meta']['rounds']} 题数={d['meta']['n_items']} "
              f"耗时={d['meta']['elapsed']}s")
    print("-" * (24 + 18 * len(groups)))
    print(f"{'指标(均值±极差)':<18}" + "".join(f"{l:>18}" for l in labels))
    for k in METRIC_KEYS:
        cells = []
        for _l, d in groups:
            v = (d.get("metrics") or {}).get(k)
            cells.append("—" if not v else f"{v['mean']:.3f}±{v['range']:.3f}")
        print(f"{k:<18}" + "".join(f"{c:>18}" for c in cells))

    print("\n分题型答案正确率（各轮值）")
    print(f"{'题型':<12}" + "".join(f"{l:>22}" for l in labels))
    for t in TYPES:
        cells = []
        for _l, d in groups:
            vals = cross_by_round(d, qtype).get(t, [])
            cells.append("—" if not vals else ",".join(f"{v:.3f}" for v in vals))
        print(f"{t:<12}" + "".join(f"{c:>22}" for c in cells))

    print("\n判分翻转率（基线：复核开 4.6% / 复核关 6.8%）")
    print(f"{'口径':<16}" + "".join(f"{l:>12}" for l in labels))
    allkeys = set()
    for _l, d in groups:
        allkeys |= set((d.get("flip_rates") or {}).keys())
    for k in sorted(allkeys):
        print(f"{k:<16}" + "".join(
            f"{(d.get('flip_rates') or {}).get(k, 0):>12.3f}" for _l, d in groups))

    print("\nAgent 组件跨轮波动")
    for label, d in groups:
        af = d.get("agent_flip") or {}
        sp = [q for q, _ in (af.get("need_split_flip") or [])]
        rf = [q for q, _ in (af.get("reflect_rounds_flip") or [])]
        gt = [q for q, _ in (af.get("gate_flip") or [])]
        print(f"  [{label}] 拆/不拆不一致 {len(sp)} 题 {sp} ｜ "
              f"自检轮数不一致 {len(rf)} 题 {rf} ｜ 前置拒答不一致 {len(gt)} 题 {gt}")

    print("\n逐题分数矩阵（跨配置不一致的题：列出各轮判分序列）")
    runs = []
    for label, d in groups:
        n = d["meta"]["rounds"]
        per_q = {qid: [d["_runs"][k][qid].get("score") for k in range(n)]
                 for qid in d["_runs"][0]}
        runs.append((label, per_q))
    allq = sorted(set().union(*[set(pq) for _l, pq in runs]))
    shown = 0
    for qid in allq:
        vals = set()
        for _l, pq in runs:
            vals |= {s for s in (pq.get(qid) or []) if s is not None}
        if len(vals) > 1:
            shown += 1
            print(f"  {qid} [{qtype.get(qid)}]: " + " ｜ ".join(
                f"{l}={pq.get(qid)}" for l, pq in runs))
    if not shown:
        print("  （无跨配置差异）")


if __name__ == "__main__":
    main()
