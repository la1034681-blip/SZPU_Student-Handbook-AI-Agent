# -*- coding: utf-8 -*-
"""阶段 5 · 用改进后的阅卷标准重算已有结果的正确率（不重新生成答案）

动机
----
阅卷器的判分口径升级了三处：
  ① 补入「并行口径」规则 —— 手册里本科/专科两套规定并存，答出哪一套都算对；
  ② 明确「多写不扣分、少写才扣分」—— 答得比标准答案更全不算错；
  ③ 把本次检索到的证据一并交给阅卷员 —— 避免把「手册其他条款里的真实数字」
     误判成编造（阶段 5 实测：Q31 的「15 日内作出书面复查结论」被误判为编造）。

答案本身没有变，没必要重新花时间调模型生成。本脚本只重跑「评分」这一层：
证据重新检索一次（检索是确定性的，同一问题结果一致），然后逐题重判。

用法
----
    python src/rescore.py data --file gen_results.json
    python src/rescore.py data --file stability_results.json
    python src/rescore.py data --file gen_results.json --dry     # 只对比，不写回
"""
import argparse
import json
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from judge_answers import judge                       # noqa: E402
from qa_chain import build_llm                        # noqa: E402
from retrievers import HandbookRetriever              # noqa: E402


def _sum_gen(rows):
    graded = [r for r in rows if not r["refuse_expected"]]
    refus = [r for r in rows if r["refuse_expected"]]
    answered = [r for r in graded if not r["refused"]]

    def rate(n, d):
        return round(n / d, 4) if d else None

    def mean(v):
        v = [x for x in v if x is not None]
        return round(sum(v) / len(v), 4) if v else None

    return {
        "n": len(rows), "n_graded": len(graded), "n_refuse": len(refus),
        "引用存在率": rate(sum(1 for r in graded if r["exist_pass"]), len(graded)),
        "引用支持率": rate(sum(1 for r in graded if r["support_pass"]), len(graded)),
        "出处命中率": rate(sum(1 for r in graded if r["gold_covered"]), len(graded)),
        "误拒率": rate(sum(1 for r in graded if r["refused"]), len(graded)),
        "拒答正确率": rate(sum(1 for r in refus if r["refused"]), len(refus)),
        "答案正确率": mean([r["judge_score"] for r in graded]),
        "已作答正确率": mean([r["judge_score"] for r in answered]),
        "完全正确率": rate(sum(1 for r in graded
                            if r.get("judge_verdict") == "correct"), len(graded)),
        "要点覆盖率": mean([r.get("judge_coverage") for r in answered]),
        "平均延迟": round(sum(r["latency"] for r in rows) / len(rows), 2),
    }


def _metrics_stab(per_round, items):
    """与 stability_test.round_metrics 同口径（此处内联，避免跨模块耦合）。"""
    from stability_test import round_metrics
    return [round_metrics(per, items) for per in per_round]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("data_dir", nargs="?", default="data")
    ap.add_argument("--file", required=True, help="要重算的结果文件（在 data 目录下）")
    ap.add_argument("--out", default="", help="输出文件名，默认覆盖原文件")
    ap.add_argument("--top-k", type=int, default=6, help="必须与生成时一致")
    ap.add_argument("--no-rerank", action="store_true")
    ap.add_argument("--model", default=None)
    ap.add_argument("--dry", action="store_true", help="只打印对比，不写回")
    ap.add_argument("--rebuild-only", action="store_true",
                    help="跳过重判，仅从 _runs 重建派生统计（unstable/flip_rates）")
    ap.add_argument("--qids", default="",
                    help="只重判这些题（逗号分隔）；其余题沿用原判分。"
                         "用于口径局部修正后低成本更新")
    a = ap.parse_args()

    only = {q.strip() for q in a.qids.split(",") if q.strip()}
    if only:
        print(f"限定重判范围：{sorted(only)}")

    d = Path(a.data_dir)
    if not d.is_absolute():
        d = Path(__file__).resolve().parent.parent.parent.parent / d
    ev = json.load((d / "eval_set.json").open(encoding="utf-8"))
    items = {it["qid"]: it for it in ev["items"]}

    path = d / a.file
    data = json.loads(path.read_text(encoding="utf-8"))
    print(f"重算 {path.name}（top_k={a.top_k}，重排={'关' if a.no_rerank else '开'}）")

    llm = None if a.rebuild_only else build_llm(model=a.model)
    retr = (None if a.rebuild_only
            else HandbookRetriever(d, use_rerank=not a.no_rerank, verbose=False))
    cache = {}

    def evs_of(qid):
        if qid not in cache:
            cache[qid] = retr.search(items[qid]["question"], top_k=a.top_k)
        return cache[qid]

    changes = []

    if "items" in data:                       # ---------- 生成层结果 ----------
        for r in data["items"]:
            it = items.get(r["qid"])
            if not it or it.get("refuse_expected"):
                continue
            if only and r["qid"] not in only:
                continue
            old = r.get("judge_score")
            if r["refused"]:
                r.update({"judge_verdict": "refused", "judge_score": 0.0,
                          "judge_coverage": None, "judge_missing": [],
                          "judge_numbers_missing": [], "judge_bad_numbers": [],
                          "judge_reason": "系统拒答，未给出答案"})
                changes.append((r["qid"], old, 0.0, "refused"))
                continue
            jr = judge(it["question"], it.get("answer", ""), r["answer"],
                       it.get("must_contain"), it.get("note", ""),
                       llm=llm, evidences=evs_of(r["qid"]))
            r.update({"judge_verdict": jr["verdict"], "judge_score": jr["score"],
                      "judge_coverage": jr["coverage"],
                      "judge_missing": jr["missing_points"],
                      "judge_numbers_missing": jr["numbers_missing"],
                      "judge_bad_numbers": jr["bad_numbers"],
                      "judge_reason": jr["reason"]})
            changes.append((r["qid"], old, jr["score"], jr["verdict"]))
        data["summary"] = _sum_gen(data["items"])
        key = "答案正确率"

    elif "_runs" in data:                     # ---------- 稳定性结果 ----------
        if a.rebuild_only:
            print("  --rebuild-only：跳过重判，仅从 _runs 重建派生统计")
        for per in data["_runs"]:
            for qid, rec in per.items():
                it = items.get(qid)
                if not it or it.get("refuse_expected") or rec.get("refused"):
                    rec["score"] = None
                    rec["judge_verdict"] = "refused" if rec.get("refused") else None
                    continue
                if a.rebuild_only:
                    continue
                if only and qid not in only:
                    continue
                old = rec.get("score")
                jr = judge(it["question"], it.get("answer", ""), rec["answer"],
                           it.get("must_contain"), it.get("note", ""),
                           llm=llm, evidences=evs_of(qid))
                rec["score"] = jr["score"]
                rec["judge_verdict"] = jr["verdict"]
                changes.append((qid, old, jr["score"], jr["verdict"]))

        ids = sorted({q for per in data["_runs"] for q in per})
        items_list = [items[q] for q in ids if q in items]
        per_round = _metrics_stab(data["_runs"], items_list)
        metrics = {}
        for k in per_round[0]:
            vals = [pr[k] for pr in per_round if pr[k] is not None]
            metrics[k] = None if not vals else {
                "mean": round(sum(vals) / len(vals), 4),
                "min": min(vals), "max": max(vals),
                "range": round(max(vals) - min(vals), 4),
                "per_round": [pr[k] for pr in per_round]}
        data["metrics"] = metrics
        data["per_round"] = per_round
        # 关键：判分口径变了，unstable / flip_rates 必须一并重建，
        # 否则会留下「_runs 已是新分、unstable 还写旧分」的陈旧数据。
        from stability_test import compare_runs
        cmp = compare_runs(data["_runs"], items_list)
        data["unstable"] = cmp["unstable"]
        data["stable_items"] = cmp["stable_items"]
        data["flip_rates"] = cmp["flip_rates"]
        key = None
    else:
        raise SystemExit("无法识别的结果文件：既没有 items 也没有 _runs")

    # ---------------- 对比 ----------------
    print("\n逐题新旧对比（只列有变化的）")
    n_up = n_down = 0
    for qid, old, new, verdict in changes:
        if old is None and new is None:
            continue
        if old == new:
            continue
        mark = "↑" if (new or 0) > (old or 0) else "↓"
        n_up += mark == "↑"
        n_down += mark == "↓"
        print(f"  {qid:<5}{mark} {old} → {new}  ({verdict})")
    print(f"\n合计：升 {n_up} 题，降 {n_down} 题，共重判 {len(changes)} 次")

    if "items" in data:
        print(f"\n答案正确率：{data['summary']['答案正确率']}")
    elif key is None and data.get("metrics", {}).get("答案正确率"):
        m = data["metrics"]["答案正确率"]
        print(f"\n答案正确率（跨轮均值）：{m['mean']}  极差 {m['range']}")

    if a.dry:
        print("\n--dry 已指定，未写回文件。")
        return
    out = a.out or a.file
    (d / out).write_text(json.dumps(data, ensure_ascii=False, indent=1),
                         encoding="utf-8")
    print(f"\n已写回 {d / out}")


if __name__ == "__main__":
    main()
