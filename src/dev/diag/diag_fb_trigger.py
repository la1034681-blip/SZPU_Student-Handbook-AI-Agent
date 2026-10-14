# -*- coding: utf-8 -*-
"""用两次实测结果反推：兜底到底该在哪里触发。

手里有两份逐题得分：
    proc_full_judge.json   纯程序化（记为 s0）
    proc_fb_judge.json     全量兜底（FB_MODE=weak，记为 s1）

只要某道题在候选规则下"会触发兜底"，它的得分就等于 s1，否则等于 s0 ——
于是可以在**不真跑**的前提下，把各种触发规则的整体正确率算出来，
挑出真正划算的那一条（这叫反事实推算，省掉十几次全量评测）。

标定的重点信号是 focus_cov：问题里的实义字**在程序化实际摘录的那几条条款里**
被覆盖了多少。覆盖率低 = 检索没错、但摘要没摘对条 = 该叫大模型；
覆盖率高 = 条款已经摘对了、只是结论句写得不好看 = 别叫，原文本身更全。
"""
import json
import sys
from collections import defaultdict
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from retrievers import HandbookRetriever                       # noqa: E402
from procedural_qa import (ProceduralPipeline, generate,       # noqa: E402
                           STRONG_KINDS, coverage)

ROOT = Path(__file__).resolve().parent.parent.parent.parent
D = ROOT / "data"


def main():
    items = json.load((D / "eval_set.json").open(encoding="utf-8"))["items"]
    s0 = {r["qid"]: r for r in
          json.load((D / "proc_full_judge.json").open(encoding="utf-8"))["items"]}
    s1 = {r["qid"]: r for r in
          json.load((D / "proc_fb_judge.json").open(encoding="utf-8"))["items"]}

    r = HandbookRetriever(D, use_rerank=True, verbose=False)
    pipe = ProceduralPipeline(r, top_k=6, verbose=False)

    rows = []
    for it in items:
        q = it["question"]
        subs = pipe.split_subquestions(q)
        evs, seen = [], set()
        if subs:
            for qq in [q] + subs:
                for e in r.search(qq, top_k=6)[:3]:
                    if e.clause_id not in seen:
                        seen.add(e.clause_id)
                        evs.append(e)
            for i, e in enumerate(evs, 1):
                e.eid = f"E{i}"
        else:
            evs = r.search(q, top_k=6)
        out = generate(q, evs, max_evidence=6 if subs else 4)
        if out["insufficient"]:
            continue
        # 程序化实际摘录了哪几条 → 把它们拼起来，量问题焦点被覆盖了多少
        kept_ids = set(out["evidence_order"])
        kept_txt = " ".join(e.text or "" for e in evs if e.clause_id in kept_ids)
        focus_cov = coverage(q, kept_txt)
        rows.append({"qid": it["qid"], "type": it["type"],
                     "weak": bool(out["weak"]),
                     "kind": out["conclusion_kind"],
                     "rerank": max((float(getattr(e, "score", 0.0) or 0.0)
                                    for e in evs), default=0.0),
                     "focus_cov": focus_cov,
                     "s0": (s0.get(it["qid"]) or {}).get("judge_score"),
                     "s1": (s1.get(it["qid"]) or {}).get("judge_score"),
                     "ran_fb": (s1.get(it["qid"]) or {}).get("plan_mode") == "procedural→llm"})

    graded = [x for x in rows if x["s0"] is not None]
    base = sum(x["s0"] for x in graded) / len(graded)
    print(f"应作答且程序化已作答 {len(graded)} 题，纯程序化均分 {base:.4f}\n")

    print("焦点覆盖率 vs 实测表现（★=兜底后变好 / ▽=变差）")
    print(f"{'qid':<6}{'类型':<11}{'成色':<12}{'焦点覆盖':>8}{'s0':>6}{'s1':>6}   趋势")
    for x in sorted(graded, key=lambda x: x["focus_cov"]):
        mark = "★" if (x["s1"] or 0) > (x["s0"] or 0) else \
               ("▽" if (x["s1"] or 0) < (x["s0"] or 0) else "·")
        s1v = f"{x['s1']:.1f}" if x["ran_fb"] else "  —"
        print(f"{x['qid']:<6}{x['type']:<11}{x['kind']:<12}{x['focus_cov']:>8.2f}"
              f"{x['s0']:>6.1f}{s1v:>6}   {mark}")

    # ---------------- 反事实推算各种触发规则 ----------------
    def evaluate(name, fn):
        """按规则决定每题走 s0 还是 s1，算出整体正确率和触发规模。"""
        trig, score, up, dn, miss = [], 0.0, 0, 0, 0
        for x in graded:
            if fn(x):
                if not x["ran_fb"]:
                    miss += 1              # 该触发但那次没跑兜底，无 s1 可用
                    score += x["s0"]
                else:
                    trig.append(x["qid"])
                    score += x["s1"]
                    up += 1 if x["s1"] > x["s0"] else 0
                    dn += 1 if x["s1"] < x["s0"] else 0
            else:
                score += x["s0"]
        acc = score / len(graded)
        print(f"{name:<34}{len(trig):>6}{acc:>10.4f}{acc-base:>9.4f}"
              f"{up:>7}{dn:>6}{('  (缺s1:' + str(miss) + ')') if miss else ''}")

    print(f"\n{'触发规则':<34}{'触发数':>6}{'正确率':>10}{'增益':>9}{'变好':>7}{'变差':>6}")
    print("-" * 82)
    print(f"{'（不触发 = 纯程序化基线）':<34}{0:>6}{base:>10.4f}{0.0:>9.4f}")
    evaluate("全部弱结论（当前 FB_MODE=weak）", lambda x: x["weak"])
    for t in (0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90):
        evaluate(f"弱结论 且 焦点覆盖<{t:.2f}", lambda x, t=t: x["weak"] and x["focus_cov"] < t)
    for t in (0.50, 0.60, 0.70, 0.80):
        evaluate(f"弱结论 且 覆盖<{t:.2f} 且 非numeric",
                 lambda x, t=t: x["weak"] and x["focus_cov"] < t and x["type"] != "numeric")
    evaluate("仅 numeric 且 焦点覆盖<0.6",
             lambda x: x["weak"] and x["focus_cov"] < 0.6 and x["type"] == "numeric")
    evaluate("成色为占位句 且 焦点覆盖<0.7",
             lambda x: x["kind"] not in STRONG_KINDS and x["focus_cov"] < 0.7)

    (D / "fb_trigger_diag.json").write_text(
        json.dumps(rows, ensure_ascii=False, indent=1), encoding="utf-8")
    print("\n明细已写入 data/fb_trigger_diag.json")


if __name__ == "__main__":
    main()
