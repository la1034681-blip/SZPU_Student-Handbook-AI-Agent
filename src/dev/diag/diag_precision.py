# -*- coding: utf-8 -*-
"""精度对比：fp16 vs fp32 的检索耗时 + 召回一致性。

用法： python src/diag_precision.py fp16
       python src/diag_precision.py fp32
输出： .trash_tmp/precision_<dtype>.json
"""
from __future__ import annotations

import json
import statistics
import sys
import time

sys.stdout.reconfigure(encoding="utf-8")

from retrievers import BM25_WEIGHT, DENSE_WEIGHT, HandbookRetriever  # noqa: E402


def main():
    tag = sys.argv[1] if len(sys.argv) > 1 else "fp16"
    import torch
    dtype = torch.float16 if tag == "fp16" else torch.float32

    r = HandbookRetriever("data", use_rerank=True, dtype=dtype, verbose=False)
    r.build(); r._ensure_models()
    print(f"dtype={tag} device={r._dev}", flush=True)

    items = json.load(open("data/eval_set.json", encoding="utf-8"))["items"]
    q0 = items[0]["question"]
    r.search(q0, top_k=8)                                  # warmup

    # --- 单次检索耗时（rerank 为主）---
    dense = r._dense(q0, r.pool); bm25 = r._bm25(q0, r.pool)
    fused = r._rrf([[c for c, _ in dense], [c for c, _ in bm25]], [DENSE_WEIGHT, BM25_WEIGHT])
    ids = [c for c, _ in fused]
    pairs = [(q0, r.cmap[c]["embed_text"]) for c in ids[:r.rerank_pool]]
    rr_ms = min([(lambda t0: (r.reranker.predict(pairs, batch_size=16, show_progress_bar=False),
                             time.time() - t0)[1])(time.time()) for _ in range(3)]) * 1000

    full_ms = statistics.mean([
        (lambda t0: (r.search(q0, top_k=6), time.time() - t0)[1])(time.time())
        for _ in range(3)]) * 1000

    # --- 全量检索：记录 top-8 与 gold 召回 ---
    t0 = time.time()
    per_q, hit_tot, gold_tot = {}, 0, 0
    for it in items:
        evs = r.search(it["question"], top_k=8)
        got = [e.clause_id for e in evs]
        gold = [s["clause_id"] for s in it.get("sources", [])]
        gset = set(gold)
        hit = len(gset & set(got))
        hit_tot += hit
        gold_tot += len(gset)
        per_q[it["qid"]] = {"got": got, "gold": gold, "hit": hit}
    dur = time.time() - t0

    out = {"dtype": tag, "rerank_ms": rr_ms, "full_ms": full_ms,
           "recall_all": hit_tot / max(gold_tot, 1),
           "hit_tot": hit_tot, "gold_tot": gold_tot,
           "search_total_s": dur, "per_q": per_q}
    json.dump(out, open(f".trash_tmp/precision_{tag}.json", "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)

    print(f"rerank({len(pairs)} pairs): {rr_ms:.0f} ms", flush=True)
    print(f"full search(单查询)     : {full_ms:.0f} ms", flush=True)
    print(f"60题 gold 召回率        : {hit_tot}/{gold_tot} = {hit_tot/gold_tot:.4f}", flush=True)
    print(f"60题检索总耗时          : {dur:.1f}s", flush=True)


if __name__ == "__main__":
    main()
