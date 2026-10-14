# -*- coding: utf-8 -*-
"""诊断工具：逐题打印生成与校验的完整明细，用于排查拒答/误判。

    python src/diag_verify.py "问题1" "问题2" ...
    python src/diag_verify.py --qids Q19,Q22,Q26      # 从评测集取题
"""
import argparse
import json
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, str(Path(__file__).resolve().parent))

from qa_chain import QAPipeline, build_llm, load_env_file   # noqa: E402
from retrievers import HandbookRetriever                     # noqa: E402

ROOT = Path(__file__).resolve().parent.parent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("questions", nargs="*")
    ap.add_argument("--qids", default="")
    ap.add_argument("--top-k", type=int, default=6)
    a = ap.parse_args()

    qs = list(a.questions)
    if a.qids:
        items = json.load((ROOT / "data" / "eval_set.json").open(encoding="utf-8"))["items"]
        want = [q.strip() for q in a.qids.split(",")]
        by = {it["qid"]: it for it in items}
        qs += [by[q]["question"] for q in want if q in by]
    if not qs:
        ap.error("给几个问题，或用 --qids 指定")

    load_env_file()
    r = HandbookRetriever(ROOT / "data", use_rerank=True, verbose=False)
    pipe = QAPipeline(r, llm=build_llm(), top_k=a.top_k, verbose=False)

    for q in qs:
        res = pipe.ask(q)
        print("=" * 78)
        print(f"问：{q}")
        print(f"拒答={res.refused}  耗时={res.latency:.1f}s  {res.refuse_reason}")
        print(f"答：{res.answer[:500]}")
        print("证据：")
        for e in res.evidences:
            print(f"   [{e.eid}] {e.clause_id}  {e.cite}（p{e.page}）")
        print(f"引用：{res.citations}")
        v = res.verify or {}
        ex = v.get("existence") or {}
        sup = v.get("support") or {}
        print(f"一级存在性：{ex.get('passed')}  未知编号{ex.get('unknown_marked')} "
              f"无引用={ex.get('no_citation')}")
        print(f"二级支持性：{sup.get('passed')}  support={sup.get('n_support')} "
              f"partial={sup.get('n_partial')} unsupported={sup.get('n_unsupported')}")
        for d in sup.get("verdicts", []):
            print(f"   [{d['verdict']}] {d['claim'][:80]}  ←  {d['reason'][:60]}")
        print()


if __name__ == "__main__":
    main()
