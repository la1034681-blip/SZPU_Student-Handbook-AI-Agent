# -*- coding: utf-8 -*-
"""阶段 6 · 第 0 步验收：新旧两轮生成结果逐项对比。

用途
----
第 0 步只改检索层的三个旋钮（兄弟扩展放大），必须回答一个问题：
    召回上去了，但「答案正确率 / 引用合规 / 延迟」有没有被稀释？

本脚本把两轮结果（基线 gen_results.json 与 新配置 gen_results_step0.json）
按同一套指标并排，并逐题列出「分数发生变化」的题，避免只看总量掩盖波动。

用法
----
    python src/compare_step0.py <基线.json> <新配置.json>
"""
import json
import sys
from collections import defaultdict
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

ROOT = Path(__file__).resolve().parent.parent
KEYS = ("引用存在率", "引用支持率", "出处命中率", "误拒率", "拒答正确率",
        "答案正确率", "已作答正确率", "完全正确率", "要点覆盖率",
        "拆解触发率", "自检触发率", "平均延迟")


def load(p):
    p = Path(p)
    if not p.is_absolute():
        p = ROOT / p
    d = json.load(p.open(encoding="utf-8"))
    return d["summary"], {i["qid"]: i for i in d["items"]}, d["items"]


def fmt(v):
    if v is None:
        return "—"
    if isinstance(v, float):
        return f"{v:.4f}"
    return str(v)


def mean(xs):
    xs = [x for x in xs if x is not None]
    return sum(xs) / len(xs) if xs else None


def main():
    base_p = sys.argv[1] if len(sys.argv) > 1 else ".trash_tmp/gen_results_基线v1.1.json"
    new_p = sys.argv[2] if len(sys.argv) > 2 else "data/gen_results_step0.json"
    sb, ib, items_b = load(base_p)
    sn, i_, items_n = load(new_p)

    print("=" * 74)
    print(f"基线：{base_p}\n新配置：{new_p}")
    print("=" * 74)
    print(f"{'指标':<14}{'基线':>12}{'新配置':>12}{'差值':>12}")
    print("-" * 74)
    for k in KEYS:
        vb, vn = sb.get(k), sn.get(k)
        diff = ""
        if isinstance(vb, (int, float)) and isinstance(vn, (int, float)):
            dv = vn - vb
            diff = f"{dv:+.4f}"
        print(f"{k:<14}{fmt(vb):>12}{fmt(vn):>12}{diff:>12}")

    # ---- 分题型 ----
    def by_type(items):
        agg = defaultdict(list)
        for it in items:
            if it.get("judge_score") is not None:
                agg[it["type"]].append(it["judge_score"])
        return {t: mean(v) for t, v in agg.items()}

    bt_b, bt_n = by_type(items_b), by_type(items_n)
    print("\n分题型答案正确率")
    print(f"{'题型':<12}{'基线':>12}{'新配置':>12}{'差值':>12}")
    for t in sorted(set(bt_b) | set(bt_n)):
        vb, vn = bt_b.get(t), bt_n.get(t)
        diff = f"{vn - vb:+.4f}" if vb is not None and vn is not None else ""
        print(f"{t:<12}{fmt(vb):>12}{fmt(vn):>12}{diff:>12}")

    # ---- 逐题变化 ----
    print("\n逐题变化（分数或拒答状态发生改变的题）")
    changed = 0
    for qid in sorted(set(ib) | set(i_)):
        b, n = ib.get(qid), i_.get(qid)
        if b is None or n is None:
            print(f"  {qid}  仅一侧存在")
            continue
        sbv, snv = b.get("judge_score"), n.get("judge_score")
        st_b = "拒答" if b.get("refused") else "作答"
        st_n = "拒答" if n.get("refused") else "作答"
        if sbv != snv or st_b != st_n:
            changed += 1
            print(f"  {qid} {b['type']:<10} 分数 {sbv} → {snv} ｜ "
                  f"{st_b}→{st_n}")
            if n.get("judge_missing"):
                print(f"        新缺失要点：{n['judge_missing']}")
    if not changed:
        print("  （无）")

    # ---- 延迟分布 ----
    lat_b = [it["latency"] for it in items_b]
    lat_n = [it["latency"] for it in items_n]
    print(f"\n延迟：基线 平均{mean(lat_b):.2f}s 最大{max(lat_b):.2f}s ｜ "
          f"新配置 平均{mean(lat_n):.2f}s 最大{max(lat_n):.2f}s")
    over = [it for it in items_n if it["latency"] > 12]
    print(f"超 12 秒的题：{len(over)} 道" +
          ("" if not over else "  " + ", ".join(
              f"{it['qid']}({it['latency']}s)" for it in over)))

    # ---- 证据条数 ----
    n_ev = mean([len(it.get("cited_clauses") or []) for it in items_n])
    print(f"\n提示：第0步只改检索参数，证据条数上升是预期内；"
          f"关键是上面「答案正确率 / 支持率」没有被稀释。")

    # ---- 自检循环统计（若本轮带 Agent 自检） ----
    if any("reflect_rounds" in it for it in items_n):
        answered = [it for it in items_n if not it.get("refused")]
        trig = [it for it in items_n if it.get("reflect_rounds")]
        withgaps = [it for it in items_n if it.get("reflect_gaps")]
        print("\n自检循环")
        print(f"  已作答题 {len(answered)} 道 ｜ 检出过缺口的题 {len(withgaps)} 道 "
              f"｜ 进入补检循环的题 {len(trig)} 道")
        for it in trig:
            print(f"    {it['qid']:<5}{it['type']:<11} 补检 {it['reflect_rounds']} 轮"
                  f"  gaps={it.get('reflect_gaps')}")
        # 补检与答案变化的关系
        chg = []
        for qid in sorted(set(ib) | set(i_)):
            b, n = ib.get(qid), i_.get(qid)
            if b and n and b.get("answer") != n.get("answer") and n.get("reflect_rounds"):
                chg.append(qid)
        print(f"  补检后答案发生变化的题：{chg or '（无）'}")


if __name__ == "__main__":
    main()
