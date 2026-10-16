# -*- coding: utf-8 -*-
"""阶段 2 第二步：用本地 bge-m3 向量化全部检索块。

要点
----
- 模型 BAAI/bge-m3：中文事实标准，8192 token 上限（最长块 3140 字，无需二次切分），
  官方说明**不需要**查询指令前缀（区别于 bge-large-zh）。
- 向量归一化后存盘，余弦相似度即点积，检索端无需重复归一。
- 增量保护：把「模型名 + embed_text 指纹」写入 embed_model.json。
  指纹不变且向量文件行数匹配时直接跳过编码 —— 这就是验收项
  「程序重启后无需重新向量化」的实现方式。
- 换 embedding 模型 = 全量重嵌入，因此模型名必须落库。

用法
----
  python src/retrieval/embed_local.py data                  # 全量（1510 块）
  python src/retrieval/embed_local.py data --limit 20       # 只跑前 20 块，验证链路
  python src/retrieval/embed_local.py data --force          # 忽略缓存强制重算
"""
import argparse
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")  # 国内下载镜像

MODEL_NAME = "BAAI/bge-m3"           # 元数据中记录的模型标识（换模型必须改这里）
LOCAL_MODEL_DIR = "models/bge-m3"    # 本地权重目录（存在则离线加载，避免联网）


def resolve_model_path(explicit=None):
    """本地目录优先：离线可跑、且不必重复下载 2.3GB。"""
    if explicit and Path(explicit, "config.json").exists():
        return explicit
    if Path(LOCAL_MODEL_DIR, "config.json").exists():
        return LOCAL_MODEL_DIR
    return MODEL_NAME


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("data_dir", nargs="?", default="data")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    d = Path(a.data_dir)

    rows = [json.loads(l) for l in (d / "index_chunks.jsonl").open(encoding="utf-8")]
    if a.limit:
        rows = rows[:a.limit]
    meta = json.load((d / "index_meta.json").open(encoding="utf-8"))
    fingerprint = meta["embed_text_sha1"]
    ids = [r["chunk_id"] for r in rows]

    # ---------- 缓存命中判断 ----------
    mpath, vpath, ipath = d / "embed_model.json", d / "vectors.npy", d / "vector_ids.json"
    if not a.force and mpath.exists() and vpath.exists() and ipath.exists():
        old = json.load(mpath.open(encoding="utf-8"))
        same = (old.get("model") == MODEL_NAME
                and old.get("embed_text_sha1") == fingerprint
                and old.get("chunks") == len(rows))
        if same:
            print(f"向量已是最新（模型 {old['model']}，{old['chunks']} 块，"
                  f"{old['dim']} 维，{old['built_at']}），跳过编码。")
            print("如需强制重算请加 --force")
            return

    import numpy as np
    import torch
    from sentence_transformers import SentenceTransformer

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"设备 {dev}"
          + (f" · {torch.cuda.get_device_name(0)}" if dev == "cuda" else ""))
    src_path = resolve_model_path()
    kind = "本地目录" if src_path == LOCAL_MODEL_DIR else "在线（将联网下载）"
    print(f"加载模型 {MODEL_NAME}  [{kind}: {src_path}] ...")
    t0 = time.time()
    model = SentenceTransformer(src_path, device=dev)
    try:
        model.max_seq_length = 8192          # bge-m3 上限
    except Exception:
        pass
    print(f"模型就绪 {time.time() - t0:.1f}s  维度 {model.get_sentence_embedding_dimension()}")

    texts = [r["embed_text"] for r in rows]
    print(f"编码 {len(texts)} 块（batch={a.batch}）...")
    t0 = time.time()
    vec = model.encode(texts, batch_size=a.batch, normalize_embeddings=True,
                       convert_to_numpy=True, show_progress_bar=True)
    vec = np.asarray(vec, dtype=np.float32)
    dt = time.time() - t0
    print(f"完成 {dt:.1f}s  {len(texts)/max(dt,1e-6):.1f} 块/秒  形状 {vec.shape}")

    np.save(vpath, vec)
    json.dump(ids, ipath.open("w", encoding="utf-8"), ensure_ascii=False)
    info = {
        "model": MODEL_NAME,
        "dim": int(vec.shape[1]),
        "chunks": int(vec.shape[0]),
        "embed_text_sha1": fingerprint,
        "normalized": True,
        "device": dev,
        "built_at": datetime.now().isoformat(timespec="seconds"),
        "seconds": round(dt, 1),
        "note": "向量已 L2 归一化，余弦相似度=点积。换模型需 --force 全量重算。",
    }
    json.dump(info, mpath.open("w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(f"已写入 {vpath} / {ipath} / {mpath}")


if __name__ == "__main__":
    main()
