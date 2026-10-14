# -*- coding: utf-8 -*-
"""rerank 精简调参（实时输出）。"""
from __future__ import annotations

import sys
import time

sys.stdout.reconfigure(encoding="utf-8")

from retrievers import BM25_WEIGHT, DENSE_WEIGHT, HandbookRetriever  # noqa: E402


def main():
    r = HandbookRetriever("data", use_rerank=True, verbose=False)
    r.build(); r._ensure_models()
    print(f"device={r._dev}", flush=True)

    q = "旷课多少学时会被处分、分别是什么处分？迟到早退算不算旷课？"
    r.search(q, top_k=6)
    dense = r._dense(q, r.pool); bm25 = r._bm25(q, r.pool)
    fused = r._rrf([[c for c, _ in dense], [c for c, _ in bm25]], [DENSE_WEIGHT, BM25_WEIGHT])
    ids = [c for c, _ in fused]

    def bench(pairs, bs, reps=2):
        ts = []
        for _ in range(reps):
            t = time.time()
            r.reranker.predict(pairs, batch_size=bs, show_progress_bar=False)
            ts.append(time.time() - t)
        return min(ts) * 1000

    p50 = [(q, r.cmap[c]["embed_text"]) for c in ids[:50]]
    print("\n=== batch_size（候选50）===", flush=True)
    for bs in [16, 32, 64]:
        print(f"  bs={bs:<4}{bench(p50, bs):>9.1f} ms", flush=True)

    print("\n=== 候选数（bs=64）===", flush=True)
    for n in [12, 24, 50]:
        p = [(q, r.cmap[c]["embed_text"]) for c in ids[:n]]
        print(f"  n={n:<4}{bench(p, 64):>9.1f} ms", flush=True)


if __name__ == "__main__":
    main()
