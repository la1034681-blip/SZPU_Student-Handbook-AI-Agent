# -*- coding: utf-8 -*-
"""单次检索分解：dense / bm25 / rrf / 直达 / rerank / 全流程 各占多少毫秒。

用法： python src/diag_search_breakdown.py
"""
from __future__ import annotations

import statistics
import sys
import time

sys.stdout.reconfigure(encoding="utf-8")

from retrievers import BM25_WEIGHT, DENSE_WEIGHT, HandbookRetriever  # noqa: E402


def main():
    r = HandbookRetriever("data", use_rerank=True, verbose=False)
    r.build(); r._ensure_models()
    print(f"device={getattr(r,'_dev','?')}  pool={r.pool}  rerank_pool={r.rerank_pool}  "
          f"expand={r.expand}  n_chunks={len(r.chunks)}")

    q = "旷课多少学时会被处分、分别是什么处分？迟到早退算不算旷课？"
    for _ in range(3):
        r.search(q, top_k=6)                      # warmup

    agg = {}
    N = 5
    for _ in range(N):
        t = time.time(); dense = r._dense(q, r.pool); d1 = time.time() - t
        t = time.time(); bm25 = r._bm25(q, r.pool); d2 = time.time() - t
        t = time.time()
        r._rrf([[c for c, _ in dense], [c for c, _ in bm25]], [DENSE_WEIGHT, BM25_WEIGHT])
        d3 = time.time() - t
        t = time.time(); r._direct_hits(q); d4 = time.time() - t
        cand = [c for c, _ in r._rrf([[c for c, _ in dense], [c for c, _ in bm25]],
                                     [DENSE_WEIGHT, BM25_WEIGHT])][:r.rerank_pool]
        pairs = [(q, r.cmap[c]["embed_text"]) for c in cand]
        t = time.time(); r.reranker.predict(pairs, batch_size=16, show_progress_bar=False)
        d5 = time.time() - t
        t = time.time(); r.search(q, top_k=6); d6 = time.time() - t
        for k, v in dict(dense=d1, bm25=d2, rrf=d3, direct=d4, rerank=d5, full=d6).items():
            agg.setdefault(k, []).append(v)

    print(f"\n=== 单次检索各部分（{N} 次均值，毫秒）===")
    for k, v in agg.items():
        print(f"  {k:<9}{statistics.mean(v)*1000:>9.1f} ms")
    print(f"\n  全流程(d6) - (dense+bm25+rrf+direct+rerank) = "
          f"{(statistics.mean(agg['full'])-sum(statistics.mean(agg[x]) for x in ['dense','bm25','rrf','direct','rerank']))*1000:.1f} ms"
          f"  ← 兄弟扩展/去重/证据构造的净开销")


if __name__ == "__main__":
    main()
