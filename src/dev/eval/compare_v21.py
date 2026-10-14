# -*- coding: utf-8 -*-
"""阶段6 · 修订前后对比（v2.0 严格口径 → v2.1 放宽口径）

对比两组：
  BEFORE（v2.0，3 轮有效）:
      abl2_all.json, gen_results_final60.json, gen_results_final60_r2.json
  AFTER（v2.1，2 轮）:
      gen_v21_r1.json, gen_v21_r2.json

输出：
  ① 总指标对比（组均值）
  ② 分题型对比（重点看 cross）
  ③ 逐题：修订前后均分变化（只看发生变化的题）
  ④ 误拒题清单对比

用法: python src/compare_v21.py
"""
import json
import sys
from collections import defaultdict
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")
DATA = Path(__file__).resolve().parent.parent.parent.parent / "data"

BEFORE = [
    ("v2.0 #1", "abl2_all.json"),
    ("v2.0 #2", "gen_results_final60.json"),
    ("v2.0 #3", "gen_results_final60_r2.json"),
]
AFTER = [
    ("v2.1 #1", "gen_v21_r1.json"),
    ("v2.1 #2", "gen_v21_r2.json"),
]

KEYS = ["答案正确率", "已作答正确率", "出处命中率", "引用支持率",
        "误拒率", "拒答正确率", "要点覆盖率", "平均延迟"]


def load(name):
    p = DATA / name
    if not p.exists():
        return None
    return json.loads(p.read_text(encoding="utf-8"))


def group_mean(loaded, key):
    vals = [d["summary"].get(key) for _, d in loaded]
    nums = [v for v in vals if isinstance(v, (int, float))]
    return sum(nums) / len(nums) if nums else None


def type_stats(d, typ):
    rs = [r for r in d["items"] if r["type"] == typ and not r.get("refuse_expected")]
    if not rs:
        return None, None
    acc = sum(x["judge_score"] for x in rs) / len(rs)
    cov = sum(1 for x in rs if x["gold_covered"]) / len(rs)
    return acc, cov


def main():
    before = [(t, d) for t, f in BEFORE if (d := load(f))]
    after = [(t, d) for t, f in AFTER if (d := load(f))]
    if not after:
        print("!! 尚未找到 v2.1 结果文件（gen_v21_r1/r2.json），请先完成重跑。")
        return
    print(f"BEFORE 有效轮次：{len(before)}  {', '.join(t for t, _ in before)}")
    print(f"AFTER  有效轮次：{len(after)}  {', '.join(t for t, _ in after)}\n")

    # ---------- ① 总指标 ----------
    print("=" * 74)
    print("① 总指标（组均值）")
    print(f"{'指标':<12}{'BEFORE(v2.0)':>16}{'AFTER(v2.1)':>16}{'变化':>16}")
    print("-" * 74)
    for k in KEYS:
        b = group_mean(before, k)
        a = group_mean(after, k)
        if b is None or a is None:
            continue
        diff = a - b
        unit = "s" if k == "平均延迟" else ""
        arrow = "↑" if diff > 1e-9 else ("↓" if diff < -1e-9 else "=")
        print(f"{k:<12}{b:>16.4f}{a:>16.4f}{f'{arrow} {diff:+.4f}{unit}':>16}")
    print("=" * 74)

    # ---------- ② 分题型 ----------
    print("\n② 分题型（答案正确率 / 出处命中率）组均值")
    print(f"{'题型':<12}{'BEFORE':>20}{'AFTER':>20}")
    print("-" * 74)
    for typ in ["fact", "numeric", "synthesis", "cross"]:
        def agg(group):
            accs, covs = [], []
            for _, d in group:
                a, c = type_stats(d, typ)
                if a is not None:
                    accs.append(a)
                    covs.append(c)
            return (sum(accs) / len(accs), sum(covs) / len(covs)) if accs else (None, None)
        ba, bc = agg(before)
        aa, ac = agg(after)
        bs = "—" if ba is None else f"{ba:.3f}/{bc:.3f}"
        as_ = "—" if aa is None else f"{aa:.3f}/{ac:.3f}"
        print(f"{typ:<12}{bs:>20}{as_:>20}")
    print("-" * 74)

    # ---------- ③ 逐题变化 ----------
    print("\n③ 逐题：修订前后均分变化（仅列有变化的题）")
    qids = [r["qid"] for r in after[0][1]["items"]]
    bmaps = [{r["qid"]: r for r in d["items"]} for _, d in before]
    amaps = [{r["qid"]: r for r in d["items"]} for _, d in after]

    def qmean(maps, q):
        ss = []
        for m in maps:
            r = m.get(q, {})
            if r.get("refuse_expected"):
                return None
            s = 0.0 if r.get("refused") else r.get("judge_score")
            if s is not None:
                ss.append(s)
        return sum(ss) / len(ss) if ss else None

    print(f"{'题':<5}{'型':<10}{'BEFORE':>10}{'AFTER':>10}{'变化':>10}  备注")
    changed = 0
    for q in qids:
        b = qmean(bmaps, q)
        a = qmean(amaps, q)
        if b is None or a is None:
            continue
        if abs(a - b) < 1e-9:
            continue
        changed += 1
        note = ""
        # 标出误拒解除/新增
        b_ref = any(m.get(q, {}).get("refused") and not m.get(q, {}).get("refuse_expected") for m in bmaps)
        a_ref = any(m.get(q, {}).get("refused") and not m.get(q, {}).get("refuse_expected") for m in amaps)
        if b_ref and not a_ref:
            note = "误杀解除"
        elif a_ref and not b_ref:
            note = "!! 新增误拒"
        print(f"{q:<5}{'':<10}{b:>10.3f}{a:>10.3f}{a-b:>+10.3f}  {note}")
    if changed == 0:
        print("  （无变化）")

    # ---------- ④ 误拒清单 ----------
    print("\n④ 误拒题清单（非拒答题却拒答）")
    for tag, d in before:
        bad = [r["qid"] for r in d["items"]
               if not r.get("refuse_expected") and r.get("refused")]
        print(f"  {tag:<10} 误拒 {len(bad)} 题：{', '.join(bad) if bad else '无'}")
    print("  " + "-" * 40)
    for tag, d in after:
        bad = [r["qid"] for r in d["items"]
               if not r.get("refuse_expected") and r.get("refused")]
        print(f"  {tag:<10} 误拒 {len(bad)} 题：{', '.join(bad) if bad else '无'}")

    # ---------- ⑤ 拒答正确率复核 ----------
    print("\n⑤ 该拒答的题（refuse_expected）是否仍拒答")
    for tag, d in before + after:
        rs = [r for r in d["items"] if r.get("refuse_expected")]
        ok = sum(1 for r in rs if r.get("refused"))
        print(f"  {tag:<10} {ok}/{len(rs)} 正确拒答")


if __name__ == "__main__":
    main()
