# -*- coding: utf-8 -*-
"""阶段6 · 最终全量确认（60题）多轮汇总

读取同配置的多次全量结果，输出：
  1) 各轮总指标表 + 均值/极差
  2) 分题型（答案正确率 / 出处命中率）逐轮与均值
  3) 逐题稳定性：每题各轮得分、均值、是否翻转
  4) 误拒题清单（各轮）

用法: python src/summarize_final60.py
"""
import json
import sys
from collections import defaultdict
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")
DATA = Path(__file__).resolve().parent.parent / "data"

RUNS = [
    ("10-11 19:08", "abl2_all.json"),
    ("10-12 13:25", "gen_results_final60.json"),
    ("10-12 r2",    "gen_results_final60_r2.json"),
    # r3 因 DeepSeek 402 余额耗尽，全部拒答，作废（已隔离到 .trash_tmp/）
]

KEYS = ["答案正确率", "已作答正确率", "完全正确率", "出处命中率",
        "引用支持率", "误拒率", "拒答正确率", "要点覆盖率", "平均延迟"]


def load(name):
    p = DATA / name
    if not p.exists():
        return None
    return json.loads(p.read_text(encoding="utf-8"))


def main():
    loaded = [(tag, load(f)) for tag, f in RUNS]
    loaded = [(t, d) for t, d in loaded if d]
    n_runs = len(loaded)
    print(f"共 {n_runs} 轮：{', '.join(t for t, _ in loaded)}\n")

    # ---------- ① 总指标 ----------
    print("=" * 96)
    print("① 总指标（各轮）")
    header = f"{'指标':<12}" + "".join(f"{t:>14}" for t, _ in loaded) + f"{'均值':>10}"
    print(header)
    print("-" * 96)
    for k in KEYS:
        vals = [d["summary"].get(k) for _, d in loaded]
        nums = [v for v in vals if isinstance(v, (int, float))]
        mean = sum(nums) / len(nums) if nums else None
        line = f"{k:<12}" + "".join(
            f"{(f'{v:.4f}' if isinstance(v, float) else str(v)):>14}" for v in vals)
        line += f"{(f'{mean:.4f}' if mean is not None else '—'):>10}"
        print(line)
    print("=" * 96)

    # ---------- ② 分题型 ----------
    print("\n② 分题型（答案正确率 / 出处命中率）逐轮")
    types = ["fact", "numeric", "synthesis", "cross"]
    # 各轮各题型
    per = {t: {} for t in types}
    for tag, d in loaded:
        buckets = defaultdict(list)
        for r in d["items"]:
            if r.get("refuse_expected"):
                continue
            buckets[r["type"]].append(r)
        for t in types:
            rs = buckets.get(t, [])
            if not rs:
                per[t][tag] = (None, None)
                continue
            acc = sum(x["judge_score"] for x in rs) / len(rs)
            cov = sum(1 for x in rs if x["gold_covered"]) / len(rs)
            per[t][tag] = (acc, cov)
    for t in types:
        cells = []
        for tag, _ in loaded:
            a, c = per[t][tag]
            cells.append("—" if a is None else f"{a:.3f}/{c:.3f}")
        accs = [per[t][tag][0] for tag, _ in loaded if per[t][tag][0] is not None]
        m = sum(accs) / len(accs) if accs else None
        print(f"  {t:<11}" + "".join(f"{x:>16}" for x in cells)
              + f"   均值acc={m:.3f}" if m is not None else "")

    # ---------- ③ 逐题稳定性 ----------
    print("\n③ 逐题稳定性（各轮得分；★=有翻转）")
    qids = [r["qid"] for r in loaded[0][1]["items"]]
    maps = [{r["qid"]: r for r in d["items"]} for _, d in loaded]
    labels = [t for t, _ in loaded]
    print(f"{'题':<5}{'型':<10}" + "".join(f"{l:>13}" for l in labels) + f"{'均值':>8}  ")
    flips = []
    for q in qids:
        scores = []
        typ = ""
        for m in maps:
            r = m.get(q, {})
            typ = r.get("type", typ)
            if r.get("refuse_expected"):
                scores.append(None)
                continue
            s = r.get("judge_score")
            if r.get("refused"):
                s = 0.0
            scores.append(s)
        nn = [s for s in scores if s is not None]
        uniq = set(nn)
        star = "★" if len(uniq) > 1 else ""
        if star:
            flips.append(q)
        mean = sum(nn) / len(nn) if nn else 0.0
        cells = "".join(f"{('—' if s is None else f'{s:.2f}'):>13}" for s in scores)
        print(f"{q:<5}{typ:<10}{cells}{mean:>8.3f}  {star}")
    print(f"\n有翻转的题（{len(flips)}/{len(qids)}）：{', '.join(flips)}")

    # ---------- ④ 误拒 ----------
    print("\n④ 各轮误拒题（非拒答题却拒答）")
    for tag, d in loaded:
        bad = [r["qid"] for r in d["items"]
               if not r.get("refuse_expected") and r.get("refused")]
        print(f"  {tag:<14} 误拒 {len(bad)} 题：{', '.join(bad) if bad else '无'}")


if __name__ == "__main__":
    main()
