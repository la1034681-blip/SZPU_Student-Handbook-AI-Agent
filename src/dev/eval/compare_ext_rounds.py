# -*- coding: utf-8 -*-
"""阶段 6 · 扩充集：跨章题 3 轮 × 两配置（拆解开/关）汇总。

读取 data/ext_{all,np}_r{1,2,3}.json，输出：
  · 每配置每轮的跨章题正确率 / 完全正确率 / 出处命中率 → 均值±标准差
  · 逐题 3 轮得分对照（看每道题在两配置下的稳定性）
用法：python src/compare_ext_rounds.py
"""
import json
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")
ROOT = Path(__file__).resolve().parent.parent.parent.parent


def load(tag, r):
    p = ROOT / "data" / f"ext_{tag}_r{r}.json"
    if not p.exists():
        return None
    return json.loads(p.read_text(encoding="utf-8"))


def mean(xs):
    xs = [x for x in xs if x is not None]
    return sum(xs) / len(xs) if xs else float("nan")


def std(xs):
    xs = [x for x in xs if x is not None]
    if len(xs) < 2:
        return 0.0
    m = mean(xs)
    return (sum((x - m) ** 2 for x in xs) / (len(xs) - 1)) ** 0.5


def full_cross(items):
    """完全正确率：判分 == 1.0 的题占比。"""
    rs = [it for it in items if it["type"] == "cross"]
    ok = sum(1 for it in rs if it.get("judge_score") == 1.0)
    return ok / len(rs) if rs else float("nan")


def acc_cross(items):
    """答案正确率：判分均分（部分正确按 0.5 计），与评测脚本口径一致。"""
    rs = [it for it in items if it["type"] == "cross"]
    return mean([it.get("judge_score") for it in rs])


def cover_cross(items):
    rs = [it for it in items if it["type"] == "cross"]
    return mean([1.0 if it.get("gold_covered") else 0.0 for it in rs])


def main():
    configs = [("all", "拆解开（全开）"), ("np", "拆解关（只自检+门控）")]
    print("=" * 78)
    print("扩充集 · 跨章题 20 道 · 3 轮重复对照")
    print("=" * 78)

    per = {}
    for tag, label in configs:
        rows = []
        for r in (1, 2, 3):
            d = load(tag, r)
            if d is None:
                continue
            rows.append(d["items"])
        if not rows:
            print(f"\n[{label}] 无数据")
            continue
        accs = [acc_cross(x) for x in rows]
        fulls = [full_cross(x) for x in rows]
        covs = [cover_cross(x) for x in rows]
        per[tag] = rows
        print(f"\n[{label}]  轮数={len(rows)}")
        print(f"    答案正确率(含部分) : " +
              "  ".join(f"{a:.3f}" for a in accs) +
              f"   → 均值 {mean(accs):.3f} ± {std(accs):.3f}")
        print(f"    完全正确率(满分)   : " +
              "  ".join(f"{f:.3f}" for f in fulls) +
              f"   → 均值 {mean(fulls):.3f} ± {std(fulls):.3f}")
        print(f"    出处命中率         : " +
              "  ".join(f"{c:.3f}" for c in covs) +
              f"   → 均值 {mean(covs):.3f} ± {std(covs):.3f}")

    if "all" not in per or "np" not in per:
        print("\n（数据不完整，等两配置齐了再跑）")
        return

    # 逐题 3 轮得分对照
    qids = [it["qid"] for it in per["all"][0]]
    print("\n" + "=" * 78)
    print("逐题 3 轮得分对照（拆解开 | 拆解关）  ●=1.0  ◐=0.5  ○=0")
    print("=" * 78)
    print(f"{'qid':<6}{'拆解开(3轮)':<22}{'拆解关(3轮)':<22}差异")
    n_all_win = n_np_win = n_tie = 0
    for q in qids:
        sa, sb = [], []
        for r in per["all"]:
            m = {x["qid"]: x for x in r}
            sa.append((m[q].get("judge_score") or 0.0))
        for r in per["np"]:
            m = {x["qid"]: x for x in r}
            sb.append((m[q].get("judge_score") or 0.0))

        def glyph(v):
            return "●" if v >= 0.99 else ("◐" if v >= 0.49 else "○")
        ga = "".join(glyph(v) for v in sa)
        gb = "".join(glyph(v) for v in sb)
        ma, mb = mean(sa), mean(sb)
        if ma > mb + 1e-9:
            diff = f"拆解开 +{ma - mb:.2f}"; n_all_win += 1
        elif mb > ma + 1e-9:
            diff = f"拆解关 +{mb - ma:.2f}"; n_np_win += 1
        else:
            diff = "打平"; n_tie += 1
        print(f"{q:<6}{ga:<22}{gb:<22}{diff}")

    print("\n小计：拆解开占优 {} 题 ｜ 拆解关占优 {} 题 ｜ 打平 {} 题".format(
        n_all_win, n_np_win, n_tie))


if __name__ == "__main__":
    main()
