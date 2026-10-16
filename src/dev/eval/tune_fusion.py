# -*- coding: utf-8 -*-
"""阶段 3 补充实验：RRF 融合权重扫描。

动机：等权 RRF（dense 与 bm25 各记 1 票）在本语料上 R@1 反而低于纯向量 ——
BM25 较弱，其 top-1 会把 dense 排对但 BM25 排得靠后的条款挤下去。
本脚本给 dense 路加权重 w，扫一组 w 找融合最优解，量化"融合需要调权"。

score(d) = w/(k+rank_dense) + 1/(k+rank_bm25)

用法: python src/tune_fusion.py data
"""
import json
import sys
import time
from pathlib import Path

# 路径自举：src 与 core/retrieval/infra/cli 挂上搜索路径（见 src/_paths.py）
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))   # src/
import _paths  # noqa: E402,F401

import numpy as np

sys.stdout.reconfigure(encoding="utf-8")

from bm25 import BM25
from evaluate_retrieval import (KS, dedup_clause, first_hit, agg,
                                load_chunks, load_embedder)

WEIGHTS = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0, 8.0]
POOL, RRF_K, TOPK = 100, 60, 10


def rrf_weighted(pairs, k=RRF_K):
    score = {}
    for lst, w in pairs:
        if w == 0:
            continue
        for rank, cid in enumerate(lst, start=1):
            score[cid] = score.get(cid, 0.0) + w / (k + rank)
    return sorted(score.items(), key=lambda x: -x[1])


def main():
    d = Path(sys.argv[1] if len(sys.argv) > 1 else "data")
    chunks, cmap = load_chunks(d)
    items = [it for it in json.load((d / "eval_set.json").open(encoding="utf-8"))["items"]
             if not it.get("refuse_expected")]

    print("构建索引 ...", flush=True)
    bm = BM25().fit([(c["chunk_id"], c["embed_text"]) for c in chunks])
    vec = np.load(d / "vectors.npy").astype(np.float32)
    ids = json.load((d / "vector_ids.json").open(encoding="utf-8"))
    model = load_embedder()
    E = model.encode([it["question"] for it in items], normalize_embeddings=True,
                     convert_to_numpy=True, batch_size=16,
                     show_progress_bar=False).astype(np.float32)

    # 预计算两路候选（chunk_id 序）
    dense_seq, bm25_seq = [], []
    for it, q in zip(items, E):
        sims = vec @ q
        order = np.argsort(-sims)[:POOL]
        dense_seq.append([ids[int(i)] for i in order])
        bm25_seq.append([c for c, _ in bm.search(it["question"], k=POOL)])

    golds = [set(s["clause_id"] for s in it.get("sources", [])) for it in items]

    print(f"\n{'dense权重':>10}{'R@1':>9}{'R@3':>9}{'R@5':>9}{'R@10':>9}{'MRR@10':>10}")
    print("-" * 58)
    results = {}
    for w in WEIGHTS:
        rows = []
        for i, it in enumerate(items):
            fused = rrf_weighted([(dense_seq[i], w), (bm25_seq[i], 1.0)])
            seq = dedup_clause([cmap[c]["clause_id"] for c, _ in fused])[:TOPK]
            h = first_hit(seq, golds[i])
            hits = {k for k in KS if h and h <= k}
            rows.append((hits, 1.0 / h if h else 0.0))
        o = agg(rows)
        results[w] = o
        tag = "  ← 纯BM25" if w == 0 else ("  ← 等权（当前阶段3基线）" if w == 1.0 else "")
        print(f"{w:>10.1f}{o['recall@1']:>9.3f}{o['recall@3']:>9.3f}"
              f"{o['recall@5']:>9.3f}{o['recall@10']:>9.3f}{o['mrr@10']:>10.3f}{tag}")
    print("-" * 58)
    best = max(results.items(), key=lambda kv: (kv[1]["mrr@10"], kv[1]["recall@1"]))
    print(f"按 MRR@10 最优：dense 权重 = {best[0]}  "
          f"(R@1={best[1]['recall@1']}  MRR@10={best[1]['mrr@10']})")
    json.dump({str(k): v for k, v in results.items()},
              (d / "fusion_weights.json").open("w", encoding="utf-8"),
              ensure_ascii=False, indent=1)
    print(f"已写入 {d/'fusion_weights.json'}")


if __name__ == "__main__":
    main()
