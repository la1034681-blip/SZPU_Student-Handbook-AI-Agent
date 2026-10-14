# -*- coding: utf-8 -*-
"""探针：把程序化的低分题交给免费大模型（GLM-4-Flash）用**同一批证据**重答，
看分数能不能救回来。

这是"值不值得上兜底"的前置实验 —— 只用同一批证据，所以能严格判断
提升是来自"会写结论"，还是仅仅来自"换了模型"。

用法：
    python src/test_fallback_probe.py Q16,Q17,Q24,Q37,N03,N05
"""
import json
import sys
import time
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from judge_answers import judge as judge_answer                  # noqa: E402
from qa_chain import QAPipeline, build_llm                        # noqa: E402
from retrievers import HandbookRetriever                          # noqa: E402
from procedural_qa import ProceduralPipeline                      # noqa: E402

ROOT = Path(__file__).resolve().parent.parent.parent
D = ROOT / "data"


def main():
    qids = (sys.argv[1] if len(sys.argv) > 1 else
            "Q16,Q17,Q24,Q37,N03,N05,Q14,Q34").split(",")
    qids = [q.strip() for q in qids if q.strip()]

    items = {it["qid"]: it for it in
             json.load((D / "eval_set.json").open(encoding="utf-8"))["items"]}
    base = {r["qid"]: r for r in
            json.load((D / "proc_full_judge.json").open(encoding="utf-8"))["items"]}

    r = HandbookRetriever(D, use_rerank=True, verbose=False)
    pipe = ProceduralPipeline(r, top_k=6, verbose=False)
    free = build_llm(backend="free", timeout=120)
    gen = QAPipeline(None, llm=free, llm_verify=False, verbose=False)
    judge_llm = build_llm(model=None)          # 与基线同一把尺子（DeepSeek）

    rows, scores_p, scores_l = [], [], []
    for qid in qids:
        it = items[qid]
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

        t0 = time.time()
        try:
            res = gen.answer_from_evidences(q, evs)
            ans, err = res.answer, None
        except Exception as e:                     # noqa: BLE001
            ans, err = "", str(e)
        dt = time.time() - t0

        jr = judge_answer(q, it.get("answer", ""), ans, it.get("must_contain"),
                          it.get("note", ""), llm=judge_llm, qtype=it["type"],
                          evidences=evs) if ans else None
        sc_l = (jr or {}).get("score")
        sc_p = base.get(qid, {}).get("judge_score")
        if sc_p is not None:
            scores_p.append(sc_p)
        if sc_l is not None:
            scores_l.append(sc_l)
        rows.append({"qid": qid, "type": it["type"], "question": q,
                     "proc_score": sc_p, "llm_score": sc_l,
                     "llm_verdict": (jr or {}).get("verdict"),
                     "llm_reason": (jr or {}).get("reason"),
                     "llm_answer": ans, "err": err, "latency": round(dt, 1),
                     "refused": bool(getattr(res, "refused", False))})
        print(f"[{qid}] {it['type']:<10} 程序化 {sc_p} → 免费模型 {sc_l} "
              f"（{dt:.1f}s）{(jr or {}).get('verdict')}")
        print(f"      {q[:52]}")
        print(f"      模型答：{(ans or err or '')[:150]}".replace("\n", " "))
        print()

    n = len(rows)
    print("=" * 70)
    print(f"{'题数':<8}{n}")
    print(f"{'程序化均分':<12}{(sum(scores_p)/len(scores_p) if scores_p else 0):.3f}")
    print(f"{'免费模型均分':<12}{(sum(scores_l)/len(scores_l) if scores_l else 0):.3f}")
    print(f"{'平均耗时':<12}{sum(x['latency'] for x in rows)/n:.1f}s")
    (D / "fallback_probe.json").write_text(
        json.dumps(rows, ensure_ascii=False, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
