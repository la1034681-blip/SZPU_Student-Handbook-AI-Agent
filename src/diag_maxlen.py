# -*- coding: utf-8 -*-
"""max_length 对比：rerank 截断长度对耗时与召回的影响（fp16）。"""
from __future__ import annotations

import json
import statistics
import sys
import time

sys.stdout.reconfigure(encoding="utf-8")

from retrievers import BM25_WEIGHT, DENSE_WEIGHT, HandbookRetriever  # noqa: E402


def main():
    r = HandbookRetriever("data", use_rerank=True, verbose=False)
    r.build(); r._ensure_models()
    items = json.load(open("data/eval_set.json", encoding="utf-8"))["items"]
    q0 = items[0]["question"]
    r.search(q0, top_k=8)

    dense = r._dense(q0, r.pool); bm25 = r._bm25(q0, r.pool)
    fused = r._rrf([[c for c, _ in dense], [c for c, _ in bm25]], [DENSE_WEIGHT, BM25_WEIGHT])
    ids = [c for c, _ in fused]
    pairs = [(q0, r.cmap[c]["embed_text"]) for c in ids[:r.rerank_pool]]

    for ml in [256, 384, 512, 768, 1024]:
        r.reranker.max_length = ml
        try:
            r.reranker.model.max_length = ml
        except Exception:
            pass
        # rerank 耗时（min of 3）
        ms = min([(lambda t0: (r.reranker.predict(pairs, batch_size=16, show_progress_bar=False),
                               time.time() - t0)[1])(time.time()) for _ in range(3)]) * 1000
        # 全量召回
        hit = tot = 0
        for it in items:
            evs = r.search(it["question"], top_k=8)
            got = set(e.clause_id for e in evs)
            gold = set(s["clause_id"] for s in it.get("sources", []))
            hit += len(gold & got); tot += len(gold)
        print(f"  max_length={ml:<5} rerank={ms:>6.0f}ms  60题召回={hit}/{tot}={hit/tot:.4f}",
              flush=True)


if __name__ == "__main__":
    main()
