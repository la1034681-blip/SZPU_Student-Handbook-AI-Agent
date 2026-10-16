# -*- coding: utf-8 -*-
"""阶段 2 验收：向量检索闭环是否真的通了。

逐条对应工程计划里阶段 2 的验收标准：
 1. 程序重启后无需重新向量化  → 由 embed_local.py 的缓存指纹保证（本脚本复跑即验证）
 2. 能够查询条款原文、版本与页码 → 检索结果须带 text / cite / page
 3. 条款编号可直达（不依赖模型） → by_clause_no 精确命中

用法: python src/verify_stage2.py data
"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
import _paths  # noqa: E402 把 core/retrieval/infra/cli 也挂上搜索路径
from store import ChunkStore   # noqa: E402

QUERIES = [
    "补考有什么规定",
    "考试作弊怎么处理",
    "国家励志奖学金一年多少钱",
    "课程绩点怎么计算",
    "学生宿舍能不能使用大功率电器",
    "转专业需要满足什么条件",
    "考试违规会怎么处理",
    "学业预警分几个等级",
]


def main(data_dir="data"):
    st = ChunkStore(data_dir)
    info = st.stats()
    print("=" * 70)
    print(f"库内块 {info['chunks']}  类型 {info['by_type']}  文件 {info['files']}")
    print(f"向量 {info['vectors']} × {info['vector_dim']}")
    print("=" * 70)

    # —— 验收 3：条款编号精确直达（不依赖向量）——
    print("\n[验收3] 条款编号直达（无需向量）")
    for fid, no in (("F15", 10), ("F06", 1), ("F28", 21)):
        r = st.by_clause_no(fid, no)
        print(f"  {fid} 第{no}条 → " + (r["cite"] if r else "未命中"))
        if r:
            print(f"      p{r['page_start']}-{r['page_end']}  {r['text'][:60]}")

    # —— 验收 1/2：向量检索 ——
    import os
    os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
    from sentence_transformers import SentenceTransformer
    from embed_local import resolve_model_path, MODEL_NAME
    import torch

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    src = resolve_model_path()
    print(f"\n[验收2] 向量检索（设备 {dev}，模型 {src}，取 Top-3）")
    model = SentenceTransformer(src, device=dev)
    for q in QUERIES:
        t0 = time.time()
        qv = model.encode([q], normalize_embeddings=True, convert_to_numpy=True)[0]
        hits = st.search(qv, k=3)
        ms = (time.time() - t0) * 1000
        print(f"\n  问：{q}   （{ms:.0f} ms）")
        for h in hits:
            print(f"    {h['score']:.4f}  {h['cite']}")
            print(f"            p{h['page_start']}-{h['page_end']}  {h['text'][:64]}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "data")
