# -*- coding: utf-8 -*-
"""阶段 2 一键构建：把「切块产物」变成可检索的库。

顺序（每一步都可单独重跑）：
  1. index_chunks.jsonl  → SQLite（元数据 + 原文 + 引用标签）
  2. 用本地 bge-m3 向量化 → vectors.npy + vector_ids.json
  3. 回填 vector_row，使 SQLite 行与向量矩阵行对齐

之所以要第 3 步：向量矩阵是稠密数组，只能按行号取；而检索命中的是 chunk_id。
两者必须显式对齐，否则会「检索到 A 却返回 B 的原文」——这是 RAG 里最隐蔽的一类错误，
所以这里把它做成构建期的强制步骤，并在最后做一致性校验。

用法:
  python src/dev/build/build_stage2.py data            # 全量
  python src/dev/build/build_stage2.py data --limit 20 # 小样验证链路
  python src/dev/build/build_stage2.py data --force    # 强制重算向量
"""
import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
import _paths  # noqa: E402 把 core/retrieval/infra/cli 也挂上搜索路径
from store import ChunkStore  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("data_dir", nargs="?", default="data")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--skip-embed", action="store_true", help="只重建 SQLite，不动向量")
    a = ap.parse_args()
    d = Path(a.data_dir)

    t0 = time.time()
    print("=" * 66)
    print("[1/3] 导入 SQLite（元数据 + 原文 + 引用标签）")
    st = ChunkStore(a.data_dir)
    n = st.import_chunks()
    print(f"      写入 {n} 块 -> {st.db_path}")

    if a.skip_embed:
        print("[2/3] 跳过向量化")
    else:
        print("[2/3] 本地 bge-m3 向量化")
        cmd = [sys.executable, str(Path(__file__).parent.parent.parent / "embed_local.py"),
               a.data_dir, "--batch", str(a.batch)]
        if a.limit:
            cmd += ["--limit", str(a.limit)]
        if a.force:
            cmd += ["--force"]
        r = subprocess.run(cmd)
        if r.returncode != 0:
            print("      向量化失败，退出码", r.returncode)
            return r.returncode

    print("[3/3] 回填向量行号并校验一致性")
    ids = json.load((d / "vector_ids.json").open(encoding="utf-8"))
    st.set_vector_rows(ids)

    # ---- 强制一致性校验：行号 → chunk_id → 文本，三者必须自洽 ----
    import numpy as np
    vec = np.load(d / "vectors.npy")
    bad = []
    with st.connect() as con:
        for i, cid in enumerate(ids):
            row = con.execute("SELECT chunk_id, vector_row FROM chunks WHERE chunk_id=?",
                              (cid,)).fetchone()
            if row is None or row["vector_row"] != i:
                bad.append((i, cid, None if row is None else row["vector_row"]))
    print(f"      向量 {vec.shape}  映射 {len(ids)} 条")
    if bad:
        print(f"      [警告] {len(bad)} 条映射不一致，示例 {bad[:5]}")
    else:
        print("      一致性校验通过：每行向量都能对回唯一的 chunk_id")

    s = st.stats()
    print("=" * 66)
    print(f"完成 {time.time() - t0:.1f}s | 块 {s['chunks']} | 类型 {s['by_type']} "
          f"| 文件 {s['files']} | 向量 {s['vectors']}x{s['vector_dim']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
