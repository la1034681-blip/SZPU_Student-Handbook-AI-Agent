# -*- coding: utf-8 -*-
"""阶段 5 · 稳定性实验：同一套题连跑 N 轮，量化"波动"

为什么要做
----------
阶段 4 撞见一个真问题：同一道题（国家助学贷款额度）第一轮被判「证据不支持」
而拒答，第二轮同样的代码却通过了。temperature=0 只能压低随机性，
**不能消除服务端输出的非确定性**（同为 OpenAI 兼容接口的通病）。

工程上光知道"有波动"没用，得回答两个问题：
  ① 波动有多大？  ② 我做的改动有没有把它压下来？

本脚本对同一套题连跑 N 轮（默认 3），逐题比对跨轮结论：

  拒答翻转率  同一题有的轮拒答、有的轮作答
  出处翻转率  跨轮引到的条款集合不一致
  答案翻转率  跨轮答案文本不一致（最严格）
  校验翻转率  verify_pass 跨轮不一致

并给出各指标跨轮的均值 / 极差，以及逐题的不稳定清单。

用法
----
    python src/stability_test.py data --rounds 3
    python src/stability_test.py data --rounds 3 --no-recheck --out stab_strict.json
"""
import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from judge_answers import judge as judge_answer     # noqa: E402
from qa_chain import QAPipeline, build_llm          # noqa: E402
from retrievers import HandbookRetriever            # noqa: E402


def one_round(items, pipe, judge_llm, tag=""):
    """跑一轮，返回 {qid: record}。"""
    per = {}
    for i, it in enumerate(items, 1):
        res = pipe.ask(it["question"])
        rec = {
            "question": it["question"],
            "refused": res.refused,
            "verify_pass": bool((res.verify or {}).get("passed")),
            "cited": sorted({e.clause_id for e in res.evidences
                             if e.eid in res.citations}),
            "answer": res.answer,
            "latency": round(res.latency, 2),
            "score": None, "judge_verdict": None,
            # 阶段 6：Agent 过程信息（用于量化组件自身在跨轮之间是否稳定）
            "need_split": bool((getattr(res, "plan", {}) or {}).get("need_split")),
            "reflect_rounds": getattr(res, "reflect_rounds", 0),
            "gate_refused": "拒答前置" in (res.refuse_reason or ""),
        }
        if not it.get("refuse_expected") and not res.refused and judge_llm:
            jr = judge_answer(it["question"], it.get("answer", ""), res.answer,
                              it.get("must_contain"), it.get("note", ""),
                              llm=judge_llm, qtype=it["type"],
                              evidences=res.evidences)
            rec["score"] = jr["score"]
            rec["judge_verdict"] = jr["verdict"]
        per[it["qid"]] = rec
        print(f"  [{tag}] {i:>2}/{len(items)} {it['qid']} "
              f"{'拒答' if res.refused else '作答'} {res.latency:>5.1f}s",
              flush=True)
    return per


def round_metrics(per, items):
    graded = [it for it in items if not it.get("refuse_expected")]
    refus = [it for it in items if it.get("refuse_expected")]

    def rate(n, d):
        return round(n / d, 4) if d else None

    hit, refused_g, scores, verif = 0, 0, [], 0
    for it in graded:
        rec = per[it["qid"]]
        gold = {s["clause_id"] for s in it.get("sources", [])}
        if rec["verify_pass"]:
            verif += 1
        if rec["refused"]:
            refused_g += 1
            scores.append(0.0)
        else:
            if gold & set(rec["cited"]):
                hit += 1
            scores.append(rec["score"] if rec["score"] is not None else 0.0)
    return {
        "拒答正确率": rate(sum(1 for it in refus if per[it["qid"]]["refused"]),
                        len(refus)),
        "误拒率": rate(refused_g, len(graded)),
        "出处命中率": rate(hit, len(graded)),
        "引用支持率": rate(verif, len(graded)),
        "答案正确率": round(sum(scores) / len(scores), 4) if scores else None,
    }


def compare_runs(runs, items):
    """从 _runs 复算逐题稳定性：unstable 清单 / flip_rates / stable_items。

    独立于判分口径：只要 _runs 里的 score / answer / refused / cited /
    verify_pass 被更新过（例如 rescore.py 重判之后），调用本函数即可
    刷新全部派生统计，避免出现「_runs 已是新分、unstable 还留旧分」的
    陈旧数据。

    额外给出两个分层标记，用来回答「波动来自生成还是来自评分」：
      gen_drift   答案文本变了且判分也变 —— 生成波动传导到了判分
      judge_drift 答案文本没变、判分却变 —— 阅卷器自身不稳定
    """
    per_qid = defaultdict(list)
    for per in runs:
        for qid, rec in per.items():
            per_qid[qid].append(rec)

    unstable, stable_n = [], 0
    for qid, recs in per_qid.items():
        refuse_seq = [r_["refused"] for r_ in recs]
        cited_sets = {tuple(r_["cited"]) for r_ in recs}
        answers = {r_["answer"] for r_ in recs}
        verifies = {r_["verify_pass"] for r_ in recs}
        scores = [r_["score"] for r_ in recs if r_["score"] is not None]
        score_var = len(set(scores)) > 1 if scores else False
        ans_var = len(answers) > 1
        rec = {
            "qid": qid, "question": recs[0].get("question", ""),
            "refuse_flip": len(set(refuse_seq)) > 1,
            "cited_flip": len(cited_sets) > 1,
            "answer_flip": ans_var,
            "verify_flip": len(verifies) > 1,
            "refuse_seq": refuse_seq,
            "n_cited_variants": len(cited_sets),
            "score_range": (round(max(scores) - min(scores), 4) if scores else None),
            "score_seq": scores or None,
            "judge_drift": bool(score_var and not ans_var),
            "gen_drift": bool(score_var and ans_var),
        }
        if (rec["refuse_flip"] or rec["cited_flip"] or rec["answer_flip"]
                or rec["verify_flip"] or score_var):
            unstable.append(rec)
        else:
            stable_n += 1

    n = len(per_qid) or 1
    flip_rates = {
        "拒答翻转率": round(sum(1 for x in unstable if x["refuse_flip"]) / n, 4),
        "出处翻转率": round(sum(1 for x in unstable if x["cited_flip"]) / n, 4),
        "答案翻转率": round(sum(1 for x in unstable if x["answer_flip"]) / n, 4),
        "校验翻转率": round(sum(1 for x in unstable if x["verify_flip"]) / n, 4),
        "判分翻转率": round(sum(1 for x in unstable
                            if (x["score_range"] or 0) > 0) / n, 4),
        "任一翻转率": round(len(unstable) / n, 4),
    }
    return {"unstable": unstable, "stable_items": stable_n,
            "flip_rates": flip_rates}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("data_dir", nargs="?", default="data")
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--qids", default="")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--top-k", type=int, default=6)
    ap.add_argument("--no-recheck", action="store_true",
                    help="关闭断言级复核（对照消融）")
    ap.add_argument("--no-judge", action="store_true")
    # 阶段 6：Agent 组件开关（默认全开）；--agent 走 Agent 链路
    ap.add_argument("--agent", action="store_true")
    ap.add_argument("--no-planner", action="store_true")
    ap.add_argument("--no-reflect", action="store_true")
    ap.add_argument("--no-gate", action="store_true")
    ap.add_argument("--model", default=None)
    ap.add_argument("--out", default="stability_results.json")
    a = ap.parse_args()

    d = Path(a.data_dir)
    if not d.is_absolute():
        d = Path(__file__).resolve().parent.parent.parent.parent / d
    items = json.load((d / "eval_set.json").open(encoding="utf-8"))["items"]
    if a.qids:
        want = {q.strip() for q in a.qids.split(",")}
        items = [it for it in items if it["qid"] in want]
    if a.limit:
        items = items[:a.limit]

    cfg = "复核开" if not a.no_recheck else "复核关"
    print(f"稳定性实验：{len(items)} 题 x {a.rounds} 轮  配置={cfg}")

    r = HandbookRetriever(d, use_rerank=True, verbose=False)
    llm = build_llm(model=a.model)
    if a.agent:
        from agent_qa import AgentQAPipeline
        pipe = AgentQAPipeline(r, llm=llm, top_k=a.top_k, llm_verify=True,
                               verify_recheck=not a.no_recheck,
                               enable_planner=not a.no_planner,
                               enable_reflect=not a.no_reflect,
                               enable_gate=not a.no_gate, verbose=False)
    else:
        pipe = QAPipeline(r, llm=llm, top_k=a.top_k, llm_verify=True,
                          verify_recheck=not a.no_recheck, verbose=False)
    judge_llm = None if a.no_judge else llm

    t0 = time.time()
    runs = []
    for k in range(1, a.rounds + 1):
        print(f"\n--- 第 {k}/{a.rounds} 轮 ---")
        runs.append(one_round(items, pipe, judge_llm, tag=f"R{k}"))

    # ---------------- 逐题跨轮比对 ----------------
    cmp = compare_runs(runs, items)
    unstable = cmp["unstable"]
    stable_n = cmp["stable_items"]
    flip_rates = cmp["flip_rates"]
    n = len(items)

    # ---------------- 指标跨轮 ----------------
    per_round = [round_metrics(per, items) for per in runs]
    metrics = {}
    for k in per_round[0]:
        vals = [pr[k] for pr in per_round if pr[k] is not None]
        if not vals:
            metrics[k] = None
            continue
        metrics[k] = {"mean": round(sum(vals) / len(vals), 4),
                      "min": min(vals), "max": max(vals),
                      "range": round(max(vals) - min(vals), 4),
                      "per_round": [pr[k] for pr in per_round]}

    print("\n" + "=" * 78)
    print(f"稳定性结果（{a.rounds} 轮 / 配置={'复核关' if a.no_recheck else '复核开'}）")
    print("-" * 78)
    print(f"{'翻转率':<14}{'值':>10}")
    for k, v in flip_rates.items():
        print(f"{k:<14}{v:>10.3f}")
    print(f"\n{'指标':<14}{'均值':>10}{'极差':>10}   各轮")
    for k, v in metrics.items():
        if v is None:
            print(f"{k:<14}{'—':>10}")
            continue
        print(f"{k:<14}{v['mean']:>10.3f}{v['range']:>10.3f}   {v['per_round']}")
    print("=" * 78)

    print(f"\n不稳定题（{len(unstable)} 题）：")
    for x in unstable:
        tags = "".join(t for t, f in [("拒答", x["refuse_flip"]),
                                      ("出处", x["cited_flip"]),
                                      ("答案", x["answer_flip"]),
                                      ("校验", x["verify_flip"]),
                                      ("判分", (x["score_range"] or 0) > 0)] if f)
        print(f"  {x['qid']:<5}[{tags}] 拒答序列={x['refuse_seq']} "
              f"判分序列={x['score_seq']} "
              f"出处变体={x['n_cited_variants']} 分数极差={x['score_range']}")
    moved = [x for x in unstable if (x["score_range"] or 0) > 0]
    jd = [x["qid"] for x in moved if x["judge_drift"]]
    gd = [x["qid"] for x in moved if x["gen_drift"]]
    print(f"  其中真正影响判分的 {len(moved)} 题；"
          f"生成波动传导 {len(gd)} 题 {gd}；阅卷器自身波动 {len(jd)} 题 {jd}")

    # ---------------- 阶段 6：Agent 组件自身的跨轮波动 ----------------
    agent_flip = {}
    if any("need_split" in runs[0][it["qid"]] for it in items):
        def _flip(field):
            res_ = []
            for it in items:
                seq = [runs[k][it["qid"]].get(field) for k in range(len(runs))]
                if len(set(seq)) > 1:
                    res_.append((it["qid"], seq))
            return res_

        sp, rf, gt = _flip("need_split"), _flip("reflect_rounds"), _flip("gate_refused")
        agent_flip = {"need_split_flip": sp, "reflect_rounds_flip": rf,
                      "gate_flip": gt}
        print(f"\nAgent 组件跨轮波动（{len(runs)} 轮）")
        print(f"  规划判定（拆/不拆）跨轮不一致：{len(sp)} 题")
        for qid, seq in sp:
            print(f"      {qid:<5} need_split 序列={seq}")
        print(f"  自检轮数跨轮不一致：{len(rf)} 题")
        for qid, seq in rf:
            print(f"      {qid:<5} reflect_rounds 序列={seq}")
        print(f"  前置拒答跨轮不一致：{len(gt)} 题 {[q for q, _ in gt]}")

    out = {
        "meta": {"built_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                 "rounds": a.rounds, "n_items": n,
                 "recheck": not a.no_recheck, "judge": not a.no_judge,
                 "top_k": a.top_k, "elapsed": round(time.time() - t0, 1)},
        "flip_rates": flip_rates,
        "metrics": metrics,
        "per_round": per_round,
        "stable_items": stable_n,
        "unstable": unstable,
        "agent_flip": agent_flip,
        "_runs": runs,
    }
    (d / a.out).write_text(json.dumps(out, ensure_ascii=False, indent=1),
                           encoding="utf-8")
    print(f"\n结果已写入 {d / a.out}")


if __name__ == "__main__":
    main()
