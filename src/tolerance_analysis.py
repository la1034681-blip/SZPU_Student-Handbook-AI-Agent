# -*- coding: utf-8 -*-
"""阶段 3 补充：评测口径敏感性分析（严格口径 vs 平行口径）。

阶段 3 跑完发现一个评测集问题：学生手册里「本科 / 专科」是两套并行文件
（F03/F04 学籍、F13/F14 转专业、F07/F08 成绩），同一件事往往两条几乎同义的规定。
部分题目 golden 只登记了一套，检索命中另一套时被判「未命中」，低估了真实检索能力。

本脚本把这类题目的 golden 扩展为「平行口径」后重算指标，
用于分离两个量：**检索本身的能力** 与 **评测口径造成的扣分**。

⚠ 平行条款由人工逐条比对文本后确定，见 PARALLEL 表。

【状态】2026-10-10 用户已确认，4 处平行条款**已正式写入 eval_set.json**（v1.1），
  因此现在直接跑 evaluate_retrieval.py 得到的就是「平行口径」的结果。
  本脚本保留作方法记录：它当初用来量化「口径造成的扣分」有多大
  （重排 R@1 从 0.676 提到 0.757、MRR 从 0.833 提到 0.874）。
  PARALLEL 的追加是幂等的，重复跑不会改变已有结果。

用法: python src/tolerance_analysis.py data
"""
import json
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

KS = (1, 3, 5, 10)

# 题号 -> 追加的平行口径条款（本科↔专科同义；已比对原文确认主题一致）
PARALLEL = {
    "Q01": ["F03-A021"],   # 本科·首次修读课程不及格可补考，与专科 F04-A016 同义
    "Q05": ["F14-A006"],   # 专科·在校期间原则上只允许办理一次转专业
    "Q06": ["F14-A004"],   # 专科转专业办法·不予申请转专业（与 F04-A026 同主题）
    "Q33": ["F03-A026"],   # 本科·不予申请转专业（与 F13-A005 逐字一致）
}


def agg(rows):
    n = len(rows) or 1
    out = {f"recall@{k}": round(sum(1 for h, _ in rows if k in h) / n, 4) for k in KS}
    out["mrr@10"] = round(sum(rr for _, rr in rows) / n, 4)
    return out


def first_hit(seq, golden):
    for i, c in enumerate(seq[:10]):
        if c in golden:
            return i + 1
    return None


def main():
    d = Path(sys.argv[1] if len(sys.argv) > 1 else "data")
    res = json.load((d / "eval_results.json").open(encoding="utf-8"))
    strategies = res["meta"]["strategies"]
    items = [r for r in res["items"] if not r["refuse"]]

    print("=" * 72)
    print("严格口径 vs 平行口径（追加本科/专科同义条款后）")
    print("=" * 72)
    print(f"{'策略':<10}{'严格 R@1':>10}{'平行 R@1':>10}{'严格 R@3':>10}"
          f"{'平行 R@3':>10}{'严格 MRR':>10}{'平行 MRR':>10}")
    print("-" * 72)
    out = {}
    for s in strategies:
        strict, loose = [], []
        for r in items:
            gold_s = set(r["golden"])
            gold_l = gold_s | set(PARALLEL.get(r["qid"], []))
            seq = r["ranks"].get(s, [])
            for gold, bucket in ((gold_s, strict), (gold_l, loose)):
                h = first_hit(seq, gold)
                hits = {k for k in KS if h and h <= k}
                bucket.append((hits, 1.0 / h if h else 0.0))
        a, b = agg(strict), agg(loose)
        out[s] = {"strict": a, "loose": b}
        print(f"{s:<10}{a['recall@1']:>10.3f}{b['recall@1']:>10.3f}"
              f"{a['recall@3']:>10.3f}{b['recall@3']:>10.3f}"
              f"{a['mrr@10']:>10.3f}{b['mrr@10']:>10.3f}")
    print("-" * 72)

    print("\n逐题：哪些题因「平行口径」名次发生变化")
    for r in items:
        if r["qid"] not in PARALLEL:
            continue
        gold_s = set(r["golden"])
        gold_l = gold_s | set(PARALLEL[r["qid"]])
        for s in strategies:
            seq = r["ranks"].get(s, [])
            hs, hl = first_hit(seq, gold_s), first_hit(seq, gold_l)
            if hs != hl:
                print(f"  {r['qid']} [{s}] golden {sorted(gold_s)} +{PARALLEL[r['qid']]}"
                      f"  →  名次 {hs} → {hl}")

    json.dump(out, (d / "tolerance_analysis.json").open("w", encoding="utf-8"),
              ensure_ascii=False, indent=1)
    print(f"\n已写入 {d/'tolerance_analysis.json'}")


if __name__ == "__main__":
    main()
