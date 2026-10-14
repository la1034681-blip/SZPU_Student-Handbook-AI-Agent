# -*- coding: utf-8 -*-
"""标定"免费大模型兜底"的触发带。

兜底不能瞎叫：叫早了白烧额度、叫晚了没效果、叫错了会把该拒答的题答错。
所以先把 60 题的三个信号拉出来和实测得分对齐：

    rerank   top1 证据的重排分（"这条到底相不相关"的行家打分）
    concl    程序化给出的结论句类型（能判 可以/不可以，还是只能"见下文"）
    split    是否被拆成多子问题（跨章题 —— 程序化的主要短板）

对齐全量评测的 judge_score，看哪一档真的在掉分，触发阈值就从那切。
"""
import json
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, str(Path(__file__).resolve().parent))

from retrievers import HandbookRetriever                      # noqa: E402
from procedural_qa import (ProceduralPipeline, generate, HEAD_RE,  # noqa: E402
                           split_sentences, make_conclusion)

ROOT = Path(__file__).resolve().parent.parent
D = ROOT / "data"


def band_of(score):
    for lo, hi in ((0.0, 0.15), (0.15, 0.5), (0.5, 0.85), (0.85, 0.95), (0.95, 2.0)):
        if lo <= score < hi:
            return f"[{lo:.2f},{hi:.2f})"
    return "?"


def main():
    items = json.load((D / "eval_set.json").open(encoding="utf-8"))["items"]
    graded = {r["qid"]: r for r in
              json.load((D / "proc_full_judge.json").open(encoding="utf-8"))["items"]}

    r = HandbookRetriever(D, use_rerank=True, verbose=False)
    pipe = ProceduralPipeline(r, top_k=6, verbose=False)

    rows = []
    for it in items:
        q = it["question"]
        subs = pipe.split_subquestions(q)
        evs = []
        if subs:
            seen = set()
            for qq in [q] + subs:
                for e in r.search(qq, top_k=6)[:3]:
                    if e.clause_id not in seen:
                        seen.add(e.clause_id)
                        evs.append(e)
            for i, e in enumerate(evs, 1):
                e.eid = f"E{i}"
        else:
            evs = r.search(q, top_k=6)
        rerank = max((float(getattr(e, "score", 0.0) or 0.0) for e in evs), default=0.0)
        out = generate(q, evs, max_evidence=6 if subs else 4)
        if not evs:
            concl = "拒答"
        else:
            first = split_sentences(HEAD_RE.sub("", (evs[0].text or "").strip()))[0] \
                if evs else ""
            concl = make_conclusion(q, first)
        weak_generic = ("见下文" in concl or "手册规定如下" in concl or "如下。" in concl)
        jr = graded.get(it["qid"], {})
        rows.append({
            "qid": it["qid"], "type": it["type"],
            "refuse_expected": bool(it.get("refuse_expected")),
            "rerank": round(rerank, 3), "split": len(subs),
            "concl": concl, "weak": weak_generic,
            "score": jr.get("judge_score"), "verdict": jr.get("judge_verdict"),
        })

    # ---------------- 按重排分分档 ----------------
    from collections import defaultdict
    bins = defaultdict(list)
    for x in rows:
        bins[band_of(x["rerank"])].append(x)
    print("=" * 78)
    print(f"{'重排分档':<14}{'题数':>5}{'其中应拒答':>10}{'平均得分':>10}{'完全正确':>10}")
    print("-" * 78)
    for k in sorted(bins, key=lambda s: float(s[1:].split(",")[0])):
        g = [x for x in bins[k] if not x["refuse_expected"]]
        ref = [x for x in bins[k] if x["refuse_expected"]]
        sc = [x["score"] for x in g if x["score"] is not None]
        full = sum(1 for x in g if x["verdict"] == "correct")
        ms = f"{sum(sc)/len(sc):.3f}" if sc else "—"
        fr = f"{full}/{len(g)}" if g else "—"
        print(f"{k:<14}{len(bins[k]):>5}{len(ref):>10}{ms:>10}{fr:>10}")

    # ---------------- 应作答题里"该叫兜底"的候选 ----------------
    print("\n应作答但得分 < 1.0 的题（这些就是兜底的目标）：")
    weak = sorted([x for x in rows
                   if not x["refuse_expected"] and (x["score"] or 0) < 1.0
                   and x["verdict"] != "correct"],
                  key=lambda x: x["score"] or 0)
    for x in weak:
        print(f"  {x['qid']} {x['type']:<10} 重排{x['rerank']:.3f} 拆{x['split']} "
              f"得分{x['score']:.2f} {'弱结论' if x['weak'] else '有结论'}  "
              f"{x['concl'][:16]}")

    # ---------------- 各触发规则覆盖多少、平均分多少 ----------------
    print("\n触发规则对比（只统计应作答题）")
    cand = [x for x in rows if not x["refuse_expected"]]
    rules = {
        "A 只兜底重排0.15~0.85": lambda x: 0.15 <= x["rerank"] < 0.85,
        "B 只兜底弱结论": lambda x: x["weak"],
        "C 只兜底跨章(拆>=1)": lambda x: x["split"] >= 1,
        "D 弱结论 或 重排<0.85": lambda x: x["weak"] or x["rerank"] < 0.85,
        "E 弱结论 或 跨章": lambda x: x["weak"] or x["split"] >= 1,
        "F 弱结论 或 跨章 或 重排<0.7": lambda x: x["weak"] or x["split"] >= 1
        or x["rerank"] < 0.7,
    }
    base = sum(x["score"] or 0 for x in cand) / len(cand)
    print(f"{'规则':<30}{'触发数':>7}{'触发题均分':>12}{'未触发均分':>12}")
    print("-" * 78)
    print(f"{'（不触发=纯程序化基线）':<30}{0:>7}{'—':>12}{base:>12.3f}")
    for name, fn in rules.items():
        hit = [x for x in cand if fn(x)]
        miss = [x for x in cand if not fn(x)]
        hm = sum(x["score"] or 0 for x in hit) / len(hit) if hit else 0
        mm = sum(x["score"] or 0 for x in miss) / len(miss) if miss else 0
        print(f"{name:<30}{len(hit):>7}{hm:>12.3f}{mm:>12.3f}")

    (D / "fallback_band.json").write_text(
        json.dumps(rows, ensure_ascii=False, indent=1), encoding="utf-8")
    print("\n明细已写入 data/fallback_band.json")


if __name__ == "__main__":
    main()
