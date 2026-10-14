# -*- coding: utf-8 -*-
"""rerank 调参：不同 batch_size / max_length / 候选数下的耗时。

batch_size 只影响并行度、**不改变打分结果**（零质量风险）；
max_length 和候选数会改变结果（需另测质量）。
"""
from __future__ import annotations

import statistics
import sys
import time

import numpy as np

sys.stdout.reconfigure(encoding="utf-8")

from retrievers import BM25_WEIGHT, DENSE_WEIGHT, HandbookRetriever  # noqa: E402


def main():
    r = HandbookRetriever("data", use_rerank=True, verbose=False)
    r.build(); r._ensure_models()
    print(f"device={r._dev}")

    q = "旷课多少学时会被处分、分别是什么处分？迟到早退算不算旷课？"
    r.search(q, top_k=6)

    dense = r._dense(q, r.pool); bm25 = r._bm25(q, r.pool)
    fused = r._rrf([[c for c, _ in dense], [c for c, _ in bm25]], [DENSE_WEIGHT, BM25_WEIGHT])
    ids = [c for c, _ in fused]

    def bench(pairs, bs, ml, reps=3):
        ts = []
        for _ in range(reps):
            t = time.time()
            try:
                r.reranker.predict(pairs, batch_size=bs, show_progress_bar=False)
            except TypeError:
                r.reranker.predict(pairs, batch_size=bs, show_progress_bar=False)
            ts.append(time.time() - t)
        return statistics.mean(ts) * 1000

    print("\n=== A. batch_size（候选 50，仅影响速度）===")
    pairs50 = [(q, r.cmap[c]["embed_text"]) for c in ids[:50]]
    base = None
    for bs in [8, 16, 32, 50, 64]:
        ms = bench(pairs50, bs, 1024)
        print(f"  batch_size={bs:<4}{ms:>9.1f} ms")

    print("\n=== B. 候选数（batch 固定 64）===")
    for n in [12, 18, 24, 30, 40, 50]:
        pairs = [(q, r.cmap[c]["embed_text"]) for c in ids[:n]]
        print(f"  candidates={n:<4}{bench(pairs, 64, 1024):>9.1f} ms")

    print("\n=== C. batch_size=64 时调 max_length ===")
    for ml in [256, 384, 512, 768, 1024]:
        r.reranker.max_length = ml
        try:
            r.reranker.model.max_length = ml
        except Exception:
            pass
        print(f"  max_length={ml:<5}{bench(pairs50, 64, ml):>9.1f} ms")


if __name__ == "__main__":
    main()
