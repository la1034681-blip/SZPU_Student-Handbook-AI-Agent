# -*- coding: utf-8 -*-
"""阶段 2 存储层：条款元数据 + 向量。

存储选型说明（重要）
--------------------
本机没有 PostgreSQL、也没有 Docker。因此阶段 2 先采用
「SQLite（元数据与原文） + npy（向量） + numpy 精确检索」跑通完整闭环，
并把接口做成可替换的，部署阶段（阶段 6）再切到 PostgreSQL + pgvector。

这样安排不是降级，理由有三：
 1. 当前规模只有 1510 个检索块。1510×1024 的 float32 矩阵仅约 6 MB，
    numpy 点积做精确检索在毫秒级；而工程计划里本就写明
    「< 10 万向量先不要建 HNSW，精确检索更准」——现在建 ANN 反而损失精度。
 2. 元数据与向量分离存储，上层检索逻辑不感知后端，换 pgvector 只改这一个文件。
 3. SQLite 单文件、免安装、可随项目走，先保证「程序重启后无需重新向量化」这条验收。

表结构
------
chunks        每个检索块的正文、引用标签、页码、章条元数据、向量行号
clause_index  「文件ID|条号」→ chunk_id，供条款编号精确直达
meta          模型名、embed_text 指纹、构建时间（用于判断是否需要重建）
"""
import json
import sqlite3
from pathlib import Path

import numpy as np

SCHEMA = """
CREATE TABLE IF NOT EXISTS chunks (
  chunk_id      TEXT PRIMARY KEY,
  chunk_type    TEXT NOT NULL,
  parent_id     TEXT,
  clause_id     TEXT,
  file_id       TEXT,
  file_title    TEXT,
  part          TEXT,
  chapter       TEXT,
  article_no    INTEGER,
  article_label TEXT,
  cite          TEXT,
  page_start    INTEGER,
  page_end      INTEGER,
  text          TEXT NOT NULL,
  embed_text    TEXT NOT NULL,
  char_len      INTEGER,
  vector_row    INTEGER
);
CREATE INDEX IF NOT EXISTS idx_type   ON chunks(chunk_type);
CREATE INDEX IF NOT EXISTS idx_file   ON chunks(file_id);
CREATE INDEX IF NOT EXISTS idx_clause ON chunks(clause_id);
CREATE INDEX IF NOT EXISTS idx_pages  ON chunks(page_start, page_end);

CREATE TABLE IF NOT EXISTS clause_index (
  key      TEXT PRIMARY KEY,
  chunk_id TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS meta (
  key   TEXT PRIMARY KEY,
  value TEXT
);
"""

FIELDS = ["chunk_id", "chunk_type", "parent_id", "clause_id", "file_id", "file_title",
          "part", "chapter", "article_no", "article_label", "cite", "page_start",
          "page_end", "text", "embed_text", "char_len", "vector_row"]


class ChunkStore:
    def __init__(self, data_dir="data"):
        self.dir = Path(data_dir)
        self.db_path = self.dir / "handbook.db"
        self.vec_path = self.dir / "vectors.npy"
        self.ids_path = self.dir / "vector_ids.json"
        self.model_path = self.dir / "embed_model.json"
        self._vec = None
        self._ids = None
        self._row = None

    # ---------- 连接与建表 ----------
    def connect(self):
        con = sqlite3.connect(self.db_path)
        con.row_factory = sqlite3.Row
        return con

    def init_schema(self):
        with self.connect() as con:
            con.executescript(SCHEMA)

    # ---------- 导入 ----------
    def import_chunks(self, jsonl="index_chunks.jsonl", meta="index_meta.json",
                      clause_index="clause_index.json"):
        rows = [json.loads(l) for l in
                (self.dir / jsonl).open(encoding="utf-8")]
        with self.connect() as con:
            con.executescript(SCHEMA)
            con.execute("DELETE FROM chunks")
            con.execute("DELETE FROM clause_index")
            con.executemany(
                f"INSERT INTO chunks ({','.join(FIELDS)}) "
                f"VALUES ({','.join('?' * len(FIELDS))})",
                [tuple(r.get(f) for f in FIELDS) for r in rows])
            ci = json.load((self.dir / clause_index).open(encoding="utf-8"))
            con.executemany("INSERT OR REPLACE INTO clause_index VALUES (?,?)",
                            list(ci.items()))
            m = json.load((self.dir / meta).open(encoding="utf-8"))
            con.executemany("INSERT OR REPLACE INTO meta VALUES (?,?)",
                            [(k, json.dumps(v, ensure_ascii=False))
                             for k, v in m.items()])
        return len(rows)

    def set_vector_rows(self, ids):
        """按 ids 的顺序写入 vector_row（向量矩阵的行号）。"""
        with self.connect() as con:
            con.executemany("UPDATE chunks SET vector_row=? WHERE chunk_id=?",
                            [(i, cid) for i, cid in enumerate(ids)])

    # ---------- 向量 ----------
    def load_vectors(self):
        self._vec = np.load(self.vec_path).astype(np.float32)
        self._ids = json.load(self.ids_path.open(encoding="utf-8"))
        self._row = {cid: i for i, cid in enumerate(self._ids)}
        return self._vec

    @property
    def vectors(self):
        if self._vec is None:
            self.load_vectors()
        return self._vec

    def search(self, qvec, k=10, chunk_types=None, exclude_types=None):
        """精确检索（余弦相似度；向量已归一化，点积即余弦）。"""
        q = np.asarray(qvec, dtype=np.float32).reshape(-1)
        q /= (np.linalg.norm(q) + 1e-12)
        sims = self.vectors @ q
        order = np.argsort(-sims)
        out = []
        with self.connect() as con:
            for i in order:
                cid = self._ids[int(i)]
                r = con.execute("SELECT * FROM chunks WHERE chunk_id=?", (cid,)).fetchone()
                if r is None:
                    continue
                if chunk_types and r["chunk_type"] not in chunk_types:
                    continue
                if exclude_types and r["chunk_type"] in exclude_types:
                    continue
                out.append({"chunk_id": cid, "score": float(sims[int(i)]),
                            **{f: r[f] for f in FIELDS if f != "vector_row"}})
                if len(out) >= k:
                    break
        return out

    # ---------- 查询 ----------
    def get(self, chunk_id):
        with self.connect() as con:
            r = con.execute("SELECT * FROM chunks WHERE chunk_id=?", (chunk_id,)).fetchone()
            return dict(r) if r else None

    def by_clause_no(self, file_id, article_no):
        """条款编号精确直达：如 by_clause_no('F15', 10) → F15-A010。"""
        with self.connect() as con:
            r = con.execute("SELECT chunk_id FROM clause_index WHERE key=?",
                            (f"{file_id}|{article_no}",)).fetchone()
            return self.get(r["chunk_id"]) if r else None

    def get_meta(self):
        with self.connect() as con:
            return {k: json.loads(v) for k, v in
                    con.execute("SELECT key,value FROM meta").fetchall()}

    def stats(self):
        with self.connect() as con:
            n = con.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
            types = dict(con.execute(
                "SELECT chunk_type, COUNT(*) FROM chunks GROUP BY chunk_type").fetchall())
            files = con.execute("SELECT COUNT(DISTINCT file_id) FROM chunks").fetchone()[0]
        return {"chunks": n, "by_type": types, "files": files,
                "vectors": None if not self.vec_path.exists() else int(self.vectors.shape[0]),
                "vector_dim": None if not self.vec_path.exists() else int(self.vectors.shape[1])}


if __name__ == "__main__":
    import sys
    s = ChunkStore(sys.argv[1] if len(sys.argv) > 1 else "data")
    n = s.import_chunks()
    print(f"导入 {n} 个块 -> {s.db_path}")
    print("统计:", s.stats())
    print()
    print("示例 · 条款编号直达:", (s.by_clause_no("F15", 10) or {}).get("cite"))
