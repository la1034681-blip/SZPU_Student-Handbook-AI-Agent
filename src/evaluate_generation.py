# -*- coding: utf-8 -*-
"""阶段 4 · 生成层验收：端到端问答评测

阶段 3 只量「能不能翻到」，这一层量「翻到之后说得对不对、该不该说」。

四项指标
--------
  引用存在率   一级校验通过率 —— 答案里的引用编号是否真实存在（防伪造引用）
  引用支持率   二级校验通过率 —— 被引证据是否真的支持答案（防超纲发挥）
  出处命中率   非拒答题中，被引用的证据里是否含标注的标准条款
  拒答正确率   该拒答的拒了没（漏答）＋ 不该拒答的没误拒（误拒）

用法
----
    python src/evaluate_generation.py data                    # 全跑
    python src/evaluate_generation.py data --limit 12         # 只跑前 12 题
    python src/evaluate_generation.py data --qids Q01,Q38,Q39
"""
import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, str(Path(__file__).resolve().parent))

from judge_answers import judge as judge_answer     # noqa: E402
from qa_chain import QAPipeline, build_llm          # noqa: E402
from retrievers import HandbookRetriever            # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("data_dir", nargs="?", default="data")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--qids", default="")
    ap.add_argument("--top-k", type=int, default=6)
    ap.add_argument("--no-verify", action="store_true", help="关闭二级支持性校验（消融用）")
    ap.add_argument("--no-rerank", action="store_true")
    ap.add_argument("--no-recheck", action="store_true", help="关闭断言级复核（消融用）")
    ap.add_argument("--no-judge", action="store_true", help="不做答案正确率评分")
    ap.add_argument("--agent", action="store_true",
                    help="走 Agent 路径（规划+多路检索），不加则走原 QAPipeline")
    ap.add_argument("--model", default=None)
    ap.add_argument("--procedural", action="store_true",
                    help="走纯程序化抽取式通道（不需要任何大模型）")
    ap.add_argument("--fallback", action="store_true",
                    help="程序化通道开启免费大模型兜底（需 .env 配好 FREE_LLM_API_KEY）")
    ap.add_argument("--fb-mode", default=None, choices=["off", "weak", "cross", "all"],
                    help="兜底触发口径：weak=只兜底结论不实在的（默认）/ cross=加跨章 / all=除拒答外全兜底")
    ap.add_argument("--min-score", type=float, default=0.15,
                    help="程序化通道的拒答阈值（top1 重排分，默认 0.15）")
    # 阶段 6：组件开关，用于消融（默认全开）
    ap.add_argument("--no-planner", action="store_true", help="关掉问题拆解")
    ap.add_argument("--no-reflect", action="store_true", help="关掉自检循环")
    ap.add_argument("--no-gate", action="store_true", help="关掉拒答前置")
    ap.add_argument("--max-merged", type=int, default=28,
                    help="合并后证据条数上限（Agent 路径），用于诊断证据稀释")
    ap.add_argument("--judge-backend", default=None,
                    help="裁判模型后端（默认跟随被测通道）。正式对比评测建议显式指定 "
                         "deepseek —— 不能拿被测模型给自己判分，尺子会跟着松。")
    ap.add_argument("--out", default="gen_results.json")
    ap.add_argument("--eval-set", default="eval_set.json",
                    help="评测集文件名（相对 data_dir），用于扩充集的对比实验")
    a = ap.parse_args()

    d = Path(a.data_dir)
    if not d.is_absolute():
        d = Path(__file__).resolve().parent.parent / d
    items = json.load((d / a.eval_set).open(encoding="utf-8"))["items"]
    if a.qids:
        want = {q.strip() for q in a.qids.split(",")}
        items = [it for it in items if it["qid"] in want]
    if a.limit:
        items = items[:a.limit]
    print(f"评测 {len(items)} 题  二级校验={'开' if not a.no_verify else '关'}  "
          f"重排={'关' if a.no_rerank else '开'}")

    r = HandbookRetriever(d, use_rerank=not a.no_rerank, verbose=False)
    llm = None
    if a.procedural:
        from procedural_qa import ProceduralPipeline, make_free_fallback
        fb = None
        if a.fallback:
            fb = make_free_fallback(verbose=False)
            if fb is None:
                print("⚠ 兜底未启用（未配置 FREE_LLM_API_KEY 或 FREE_LLM_ENABLED=0）")
        pipe = ProceduralPipeline(r, top_k=a.top_k, min_score=a.min_score,
                                  fallback=fb, fb_mode=a.fb_mode, verbose=False)
        print(f"通道=程序化抽取式  拒答阈值={a.min_score}  "
              f"免费兜底={'开（' + (a.fb_mode or 'weak') + '）' if fb else '关'}")
    elif a.agent:
        llm = build_llm(model=a.model)
        from agent_qa import AgentQAPipeline
        pipe = AgentQAPipeline(r, llm=llm, top_k=a.top_k,
                               llm_verify=not a.no_verify,
                               verify_recheck=not a.no_recheck,
                               enable_planner=not a.no_planner,
                               enable_reflect=not a.no_reflect,
                               enable_gate=not a.no_gate,
                               max_merged=a.max_merged, verbose=False)
    else:
        llm = build_llm(model=a.model)
        pipe = QAPipeline(r, llm=llm, top_k=a.top_k, llm_verify=not a.no_verify,
                          verify_recheck=not a.no_recheck, verbose=False)
    # 裁判独立于被测通道：显式给了 --judge-backend 就用它，
    # 否则退回"与被测同后端"。混用同一模型判自己会系统性偏高，别这么干。
    if a.no_judge:
        judge_llm = None
    elif a.judge_backend:
        judge_llm = build_llm(model=a.model, backend=a.judge_backend)
        print(f"裁判后端 = {a.judge_backend}（与被测通道分离）")
    else:
        judge_llm = llm or build_llm(model=a.model)

    rows, t_all = [], time.time()
    for i, it in enumerate(items, 1):
        res = pipe.ask(it["question"])
        gold = {s["clause_id"] for s in it.get("sources", [])}
        cited = {e.clause_id for e in res.evidences if e.eid in res.citations}
        # 注意：拒答的题不能算「出处命中」—— 模型在拒答前往往已经给出了
        # citations 字段，若直接拿 gold∩cited 判定，会把拒答题也算成命中，
        # 让命中率虚高（这一版修的就是这个口径 bug）。
        if not gold:
            covered = None
        elif res.refused:
            covered = False
        else:
            covered = bool(gold & cited)
        # ---------- 答案正确率评分（阶段 5） ----------
        jr = None
        if not it.get("refuse_expected"):
            if res.refused:
                # 误拒 = 没有给出答案，站在"答得对不对"的角度就是 0 分
                jr = {"verdict": "refused", "score": 0.0, "coverage": None,
                      "numbers_missing": [], "bad_numbers": [],
                      "missing_points": [], "reason": "系统拒答，未给出答案",
                      "rule_only": judge_llm is None, "downgraded": False}
            else:
                jr = judge_answer(it["question"], it.get("answer", ""), res.answer,
                                  it.get("must_contain"), it.get("note", ""),
                                  llm=judge_llm, qtype=it["type"],
                                  evidences=res.evidences)

        row = {
            "qid": it["qid"], "type": it["type"], "question": it["question"],
            "refuse_expected": bool(it.get("refuse_expected")),
            "refuse_type": it.get("refuse_type"),
            "refused": res.refused, "refuse_reason": res.refuse_reason,
            "answer": res.answer, "citations": res.citations,
            "gold": sorted(gold), "cited_clauses": sorted(cited),
            "gold_covered": covered,
            "exist_pass": (res.verify.get("existence") or {}).get("passed"),
            "support_pass": (res.verify.get("support") or {}).get("passed"),
            "verify_pass": (res.verify or {}).get("passed"),
            "verify_reasons": (res.verify or {}).get("reasons"),
            "judge_verdict": (jr or {}).get("verdict"),
            "judge_score": (jr or {}).get("score"),
            "judge_coverage": (jr or {}).get("coverage"),
            "judge_missing": (jr or {}).get("missing_points"),
            "judge_numbers_missing": (jr or {}).get("numbers_missing"),
            "judge_bad_numbers": (jr or {}).get("bad_numbers"),
            "judge_reason": (jr or {}).get("reason"),
            "need_split": bool((getattr(res, "plan", {}) or {}).get("need_split")),
            "sub_questions": (getattr(res, "plan", {}) or {}).get("sub_questions"),
            "plan_mode": (getattr(res, "plan", {}) or {}).get("mode"),
            "fb_reason": (getattr(res, "plan", {}) or {}).get("fallback_reason"),
            "conclusion_kind": (getattr(res, "plan", {}) or {}).get("conclusion_kind"),
            "hops": getattr(res, "hops", 1),
            "n_evidence": len(res.evidences),
            "reflect_rounds": getattr(res, "reflect_rounds", 0),
            "reflect_gaps": [g for lg in (getattr(res, "reflect_log", []) or [])
                             for g in (lg.get("gaps") or [])],
            "gate_refused": "拒答前置" in (res.refuse_reason or ""),
            "latency": round(res.latency, 2),
        }
        rows.append(row)
        tag = "拒答" if res.refused else ("命中" if row["gold_covered"] else "未命中")
        print(f"  [{i:>2}/{len(items)}] {it['qid']} {it['type']:<10} "
              f"{tag:<4} {res.latency:>5.1f}s  {it['question'][:28]}")

    # ---------- 汇总 ----------
    graded = [r_ for r_ in rows if not r_["refuse_expected"]]
    refus = [r_ for r_ in rows if r_["refuse_expected"]]
    answered = [r_ for r_ in graded if not r_["refused"]]   # 真正给出答案的

    def rate(num, den):
        return round(num / den, 4) if den else None

    def mean(vals):
        vals = [v for v in vals if v is not None]
        return round(sum(vals) / len(vals), 4) if vals else None

    def fmt(v):
        """None（该子集不含此类题目）显示为「—」，其余浮点保留三位。"""
        if v is None:
            return "—"
        return f"{v:.3f}" if isinstance(v, float) else str(v)

    stat = {
        "n": len(rows), "n_graded": len(graded), "n_refuse": len(refus),
        "引用存在率": rate(sum(1 for r_ in graded if r_["exist_pass"]), len(graded)),
        "引用支持率": rate(sum(1 for r_ in graded if r_["support_pass"]), len(graded)),
        "出处命中率": rate(sum(1 for r_ in graded if r_["gold_covered"]), len(graded)),
        "误拒率": rate(sum(1 for r_ in graded if r_["refused"]), len(graded)),
        "拒答正确率": rate(sum(1 for r_ in refus if r_["refused"]), len(refus)),
        # ---- 阶段 5 新增：正确性维度 ----
        "答案正确率": mean([r_["judge_score"] for r_ in graded]),
        "已作答正确率": mean([r_["judge_score"] for r_ in answered]),
        "完全正确率": rate(sum(1 for r_ in graded if r_["judge_verdict"] == "correct"),
                        len(graded)),
        "要点覆盖率": mean([r_["judge_coverage"] for r_ in answered]),
        "拆解触发率": rate(sum(1 for r_ in rows if r_["need_split"]), len(rows)),
        "兜底触发率": rate(sum(1 for r_ in rows
                            if r_.get("plan_mode") == "procedural→llm"), len(rows)),
        "自检触发率": rate(sum(1 for r_ in rows if r_.get("reflect_rounds")), len(rows)),
        "前置拒答数": sum(1 for r_ in rows if r_.get("gate_refused")),
        "平均延迟": round(sum(r_["latency"] for r_ in rows) / len(rows), 2),
    }
    by_type = defaultdict(list)
    for r_ in graded:
        by_type[r_["type"]].append(r_)

    # 先落盘，再打印 —— 打印环节即便出错也不会丢失本轮实测数据
    out = {"meta": {"built_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                    "top_k": a.top_k, "llm_verify": not a.no_verify,
                    "recheck": not a.no_recheck, "judge": not a.no_judge,
                    "rerank": not a.no_rerank, "agent": a.agent,
                    "elapsed": round(time.time() - t_all, 1)},
           "summary": stat, "items": rows}
    (d / a.out).write_text(json.dumps(out, ensure_ascii=False, indent=1),
                           encoding="utf-8")

    print("\n" + "=" * 76)
    print(f"{'指标':<14}{'值':>10}")
    print("-" * 76)
    for k in ("引用存在率", "引用支持率", "出处命中率", "误拒率", "拒答正确率",
              "答案正确率", "已作答正确率", "完全正确率", "要点覆盖率",
              "拆解触发率", "兜底触发率", "平均延迟"):
        v = stat[k]
        print(f"{k:<14}{fmt(v):>10}")
    print("=" * 76)

    print("\n分题型（答案正确率 / 出处命中率）")
    for t, rs in sorted(by_type.items()):
        print(f"  {t:<12}"
              f"{fmt(mean([r_['judge_score'] for r_ in rs]))} / "
              f"{fmt(rate(sum(1 for r_ in rs if r_['gold_covered']), len(rs)))}")

    # ---- 拒答分类（R1 手册完全无此内容 / R2 有相关条款但不回答问题） ----
    by_refuse = defaultdict(list)
    for r_ in refus:
        by_refuse[r_.get("refuse_type") or "未分类"].append(r_)
    print("\n拒答分类正确率")
    for t, rs in sorted(by_refuse.items()):
        print(f"  {t:<16}{rate(sum(1 for r_ in rs if r_['refused']), len(rs))}"
              f"  （{sum(1 for r_ in rs if r_['refused'])}/{len(rs)}）")

    print("\n拒答题明细")
    for r_ in refus:
        print(f"  {r_['qid']} {'✅拒答' if r_['refused'] else '❌未拒答'}  "
              f"{r_['question'][:30]}")
        if not r_["refused"]:
            print(f"       模型答：{r_['answer'][:80]}")

    print(f"\n结果已写入 {d / a.out}")


if __name__ == "__main__":
    main()
