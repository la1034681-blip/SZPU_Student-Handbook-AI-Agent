# -*- coding: utf-8 -*-
"""阶段 3：检索策略消融评测。

对比四组检索配置，量化「混合检索 + 重排」相对纯向量基线的增益：

  1. bm25    纯 BM25（字符 bigram + 条款编号加权）
  2. dense   纯向量（bge-m3，1024 维，余弦）
  3. hybrid  dense + bm25 经 RRF 融合
  4. rerank  hybrid 候选再经 bge-reranker-v2-m3（cross-encoder）重排

评测口径
--------
- **按条款去重**：同一 clause_id 只保留最好名次。因为「条块」与「款块」共享
  clause_id（款块的 clause_id 继承父条），不去重会让款多的文件虚高。
- Recall@K：正确条款出现在前 K 名（K = 1/3/5/10）
- MRR@10：首个正确条款名次的倒数（未命中记 0）
- 拒答题（Q38–Q40）不计入召回指标 —— 其正确行为是「不给出具体答案」，
  属生成阶段（阶段 4）职责，此处仅列出 top-1 供观察。

用法
----
  python src/evaluate_retrieval.py data                       # 四组全跑
  python src/evaluate_retrieval.py data --strategies dense,bm25
  python src/evaluate_retrieval.py data --pool 100 --rerank-pool 50
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.stdout.reconfigure(encoding="utf-8")

KS = (1, 3, 5, 10)
RRF_K = 60            # RRF 平滑常数（业界常用 60）
LOCAL_EMBED = "models/bge-m3"
LOCAL_RERANK = "models/bge-reranker-v2-m3"


# --------------------------------------------------------------------------
# 数据加载
# --------------------------------------------------------------------------
def load_chunks(d):
    rows = [json.loads(l) for l in (d / "index_chunks.jsonl").open(encoding="utf-8")]
    return rows, {r["chunk_id"]: r for r in rows}


def load_embedder():
    import torch
    from sentence_transformers import SentenceTransformer
    src = LOCAL_EMBED if Path(LOCAL_EMBED, "config.json").exists() else "BAAI/bge-m3"
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    t0 = time.time()
    model = SentenceTransformer(src, device=dev)
    try:
        model.max_seq_length = 8192
    except Exception:
        pass
    print(f"  bge-m3 就绪 {time.time()-t0:.1f}s [{dev}]", flush=True)
    return model


def load_reranker():
    import torch
    from sentence_transformers import CrossEncoder
    if not Path(LOCAL_RERANK, "config.json").exists():
        return None
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    t0 = time.time()
    # max_length=1024：中文长条款（最长 3140 字）在此截断，兼顾显存与覆盖。
    # fp16：RTX 5060 只有 8GB，fp32 下重排显存打满会拖慢近 10 倍（2026-10-14 实测）。
    kw = {}
    if str(dev).startswith("cuda"):
        kw["model_kwargs"] = {"torch_dtype": torch.float16}
    m = CrossEncoder(LOCAL_RERANK, device=dev, max_length=1024, **kw)
    print(f"  bge-reranker-v2-m3 就绪 {time.time()-t0:.1f}s [{dev}/"
          f"{'fp16' if kw else 'fp32'}]", flush=True)
    return m


# --------------------------------------------------------------------------
# 检索实现
# --------------------------------------------------------------------------
def rrf_fuse(lists, weights=None, k=RRF_K):
    """Reciprocal Rank Fusion：只用名次，不用原始分，天然免去跨路归一化。

    weights 可给每路不同票权 —— 单路明显偏弱时，等权融合会把噪音带进头部
    （见 tune_fusion.py 的扫描结果）。
    """
    ws = weights or [1.0] * len(lists)
    score = {}
    for lst, w in zip(lists, ws):
        for rank, cid in enumerate(lst, start=1):
            score[cid] = score.get(cid, 0.0) + w / (k + rank)
    return sorted(score.items(), key=lambda x: -x[1])


def dedup_clause(seq):
    """按 clause_id 去重，保留首次出现顺序。"""
    seen, out = set(), []
    for c in seq:
        if c is None or c in seen:
            continue
        seen.add(c)
        out.append(c)
    return out


# --------------------------------------------------------------------------
# 指标
# --------------------------------------------------------------------------
def first_hit(seq, golden, cap=10):
    for i, c in enumerate(seq[:cap]):
        if c in golden:
            return i + 1
    return None


def agg(rows):
    """rows: list of (recall_hits:set, rr:float)"""
    n = len(rows) or 1
    out = {f"recall@{k}": round(sum(1 for h, _ in rows if k in h) / n, 4) for k in KS}
    out["mrr@10"] = round(sum(rr for _, rr in rows) / n, 4)
    out["n"] = len(rows)
    return out


# --------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("data_dir", nargs="?", default="data")
    ap.add_argument("--strategies", default="bm25,dense,hybrid,rerank")
    ap.add_argument("--pool", type=int, default=100, help="每路召回的候选数")
    ap.add_argument("--rerank-pool", type=int, default=50, help="进入重排的候选数")
    ap.add_argument("--topk", type=int, default=10, help="最终保留名次数")
    ap.add_argument("--dense-weight", type=float, default=1.0,
                    help="RRF 中 dense 路的票权（BM25 固定 1.0）；等权=1.0")
    ap.add_argument("--out", default="eval_results.json", help="结果输出文件名")
    a = ap.parse_args()

    d = Path(a.data_dir)
    strategies = [s.strip() for s in a.strategies.split(",") if s.strip()]
    chunks, cmap = load_chunks(d)
    items = json.load((d / "eval_set.json").open(encoding="utf-8"))["items"]
    graded = [it for it in items if not it.get("refuse_expected")]
    refused = [it for it in items if it.get("refuse_expected")]
    print(f"检索块 {len(chunks)}  评测题 {len(items)}（可判分 {len(graded)} / 拒答 {len(refused)}）")

    need_dense = any(s in ("dense", "hybrid", "rerank") for s in strategies)
    need_bm25 = any(s in ("bm25", "hybrid", "rerank") for s in strategies)

    # ---------- BM25 索引 ----------
    bm = None
    if need_bm25:
        from bm25 import BM25
        t0 = time.time()
        # 两路输入必须一致，否则对比不成立：dense 编码的是 embed_text（含
        # 文件标题 + 章上下文），BM25 若只吃 text，就等于白丢「专科/本科」
        # 「学籍管理规定」这类文件级判别信号，会被不公平地削弱。
        bm = BM25().fit([(c["chunk_id"], c["embed_text"]) for c in chunks])
        print(f"  BM25 索引 {bm.N} 篇 / 词表 {len(bm.df)} / {time.time()-t0:.1f}s", flush=True)

    # ---------- 向量 ----------
    vec = ids = row_of = None
    if need_dense:
        vec = np.load(d / "vectors.npy").astype(np.float32)
        ids = json.load((d / "vector_ids.json").open(encoding="utf-8"))
        row_of = {c: i for i, c in enumerate(ids)}
        print(f"  向量矩阵 {vec.shape}", flush=True)

    # ---------- 查询编码（批量一次） ----------
    qvec = {}
    if need_dense:
        model = load_embedder()
        t0 = time.time()
        qs = [it["question"] for it in items]
        E = model.encode(qs, normalize_embeddings=True, convert_to_numpy=True,
                         batch_size=16, show_progress_bar=False).astype(np.float32)
        for it, e in zip(items, E):
            qvec[it["qid"]] = e
        print(f"  编码 {len(qs)} 问句 {time.time()-t0:.1f}s", flush=True)

    reranker = load_reranker() if "rerank" in strategies else None
    if "rerank" in strategies and reranker is None:
        print("  ⚠ 未找到 models/bge-reranker-v2-m3，跳过 rerank 组")

    # ---------- 逐题检索 ----------
    per_item = []
    t_all = time.time()
    for it in items:
        qid = it["qid"]
        rec = {"qid": qid, "type": it["type"], "question": it["question"],
               "golden": [s["clause_id"] for s in it.get("sources", [])],
               "refuse": bool(it.get("refuse_expected")), "ranks": {}, "top1": {}}

        # 两路基础召回
        dense_res = bm25_res = None
        if need_dense:
            q = qvec[qid]
            sims = vec @ q
            order = np.argsort(-sims)[:a.pool]
            dense_res = [(ids[int(i)], float(sims[int(i)])) for i in order]
        if need_bm25:
            bm25_res = bm.search(it["question"], k=a.pool)

        # 各策略结果（chunk_id 序列）
        if "dense" in strategies and dense_res is not None:
            seq = [c for c, _ in dense_res]
            rec["ranks"]["dense"] = dedup_clause([cmap[c]["clause_id"] for c in seq])[:a.topk]
            rec["top1"]["dense"] = cmap[seq[0]]["cite"]
        if "bm25" in strategies and bm25_res is not None:
            seq = [c for c, _ in bm25_res]
            rec["ranks"]["bm25"] = dedup_clause([cmap[c]["clause_id"] for c in seq])[:a.topk]
            rec["top1"]["bm25"] = cmap[seq[0]]["cite"] if seq else ""
        if "hybrid" in strategies and dense_res is not None and bm25_res is not None:
            fused = rrf_fuse([[c for c, _ in dense_res], [c for c, _ in bm25_res]],
                             [a.dense_weight, 1.0])
            seq = [c for c, _ in fused]
            rec["ranks"]["hybrid"] = dedup_clause([cmap[c]["clause_id"] for c in seq])[:a.topk]
            rec["top1"]["hybrid"] = cmap[seq[0]]["cite"]
        if "rerank" in strategies and reranker is not None and dense_res is not None and bm25_res is not None:
            fused = rrf_fuse([[c for c, _ in dense_res], [c for c, _ in bm25_res]],
                             [a.dense_weight, 1.0])
            cand = [c for c, _ in fused][:a.rerank_pool]
            pairs = [(it["question"], cmap[c]["embed_text"]) for c in cand]
            sc = reranker.predict(pairs, batch_size=16, show_progress_bar=False)
            order = np.argsort(-np.asarray(sc))
            seq = [cand[int(i)] for i in order]
            rec["ranks"]["rerank"] = dedup_clause([cmap[c]["clause_id"] for c in seq])[:a.topk]
            rec["top1"]["rerank"] = cmap[seq[0]]["cite"]

        per_item.append(rec)
    print(f"检索完成 {time.time()-t_all:.1f}s")

    # ---------- 汇总 ----------
    summary = {}
    for s in strategies:
        if s == "rerank" and reranker is None:
            continue
        rows_all, by_type = [], {}
        for rec in per_item:
            if rec["refuse"]:
                continue
            seq = rec["ranks"].get(s, [])
            gold = set(rec["golden"])
            h = first_hit(seq, gold)
            hits = {k for k in KS if h and h <= k}
            rr = 1.0 / h if h else 0.0
            rows_all.append((hits, rr))
            by_type.setdefault(rec["type"], []).append((hits, rr))
        summary[s] = {"overall": agg(rows_all),
                      "by_type": {t: agg(v) for t, v in sorted(by_type.items())}}

    # ---------- 输出 ----------
    print("\n" + "=" * 78)
    print(f"{'策略':<10}{'R@1':>8}{'R@3':>8}{'R@5':>8}{'R@10':>8}{'MRR@10':>9}")
    print("-" * 78)
    for s in strategies:
        if s not in summary:
            continue
        o = summary[s]["overall"]
        print(f"{s:<10}{o['recall@1']:>8.3f}{o['recall@3']:>8.3f}"
              f"{o['recall@5']:>8.3f}{o['recall@10']:>8.3f}{o['mrr@10']:>9.3f}")
    print("=" * 78)

    print("\n分题型 Recall@5 / MRR@10")
    types = ["fact", "numeric", "synthesis", "cross"]
    print(f"{'题型':<12}" + "".join(f"{s:>18}" for s in strategies if s in summary))
    for t in types:
        line = f"{t:<12}"
        for s in strategies:
            if s not in summary:
                continue
            bt = summary[s]["by_type"].get(t)
            line += f"{(str(bt['recall@5'])+' / '+str(bt['mrr@10'])) if bt else '-':>18}"
        print(line)

    out = {
        "meta": {
            "built_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "data": str(d),
            "strategies": [s for s in strategies if s in summary],
            "pool": a.pool, "rerank_pool": a.rerank_pool, "rrf_k": RRF_K,
            "dense_weight": a.dense_weight, "bm25_weight": 1.0,
            "dedup": "by clause_id",
            "embed": "bge-m3 (1024d, cosine)",
            "rerank": "bge-reranker-v2-m3 (cross-encoder, max_length=1024, on embed_text)",
            "graded_items": len(graded), "refuse_items": len(refused),
        },
        "summary": summary,
        "items": per_item,
    }
    (d / a.out).write_text(
        json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\n结果已写入 {d/a.out}")


if __name__ == "__main__":
    main()
