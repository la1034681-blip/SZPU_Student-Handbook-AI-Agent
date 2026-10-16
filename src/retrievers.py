# -*- coding: utf-8 -*-
"""阶段 4 · 统一检索接口（供生成层调用）

把阶段 3 验证过的最优组合封装成一个类：

    dense(bge-m3) ┐
                  ├─ RRF 融合 ─ bge-reranker-v2-m3 重排 ─ 条款级去重 ─ 证据
    BM25(jieba)   ┘

在阶段 3 基础上补两件工程化处理：

  1. **条款号直达**：提问里出现「第X条 / 《某办法》第X条」时，用元数据精确
     匹配，把这些条款提到重排候选池最前面。学生手册的提问里这类占比不低，
     而纯语义检索未必能把「第九条」排到第一 —— BM25 会把它当普通词，
     向量则完全看不见数字。
  2. **父子块扩展**：命中的若是「款」块，取出它所属「条」的完整原文。
     答案要引的是整条，不是孤立的一款。

一个重要事实（阶段 1/2 踩出来的）：**公式修复只落在检索块上**，
clauses.json 里 F08-A002 的公式仍是残缺的。所以证据原文一律取 chunk 层的
article 块文本，不能用条款主表。

证据粒度 = 条款（clause_id）。同一条命中多个块时只保留最好名次。

对外只有一个入口：
    HandbookRetriever.search(query, top_k) -> list[Evidence]
"""
from __future__ import annotations

import json
import os
import re
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

sys.stdout.reconfigure(encoding="utf-8")

# ------------------------------------------------------------------ 常量
RRF_K = 60                     # RRF 平滑常数（阶段 3 定了 60，不调）
DENSE_WEIGHT = 1.0             # 等权即可 —— 阶段 3 结论：加了重排后调权无收益
BM25_WEIGHT = 1.0
MAX_EVIDENCE_CHARS = 1800      # 单条证据原文上限，超长截断，防止 prompt 爆掉
DIRECT_BOOST_CAP = 20          # 条款号直达最多前置多少块

# ---- 兄弟条款扩展三参数（阶段 6 第 0 步）--------------------------------
# 依据 src/diag_expand_tune.py 的参数扫描：由 (3,2,4) 放大到 (8,6,16) 后，
#   cross gold 召回 72.2% → 88.9%，synthesis 86.7% → 100%，
#   代价是平均证据条数 9.2 → 15.3（+66%），prompt 变长、生成有被稀释的风险。
# 故第 0 步不能只看召回，必须端到端复测「正确率/延迟」是否退化；
# 若退化则退一档到 (8,4,12)。改这三个常量即可回退，无需改逻辑。
SIBLING_PER_FILE = 8           # 每份文件最多补多少条相邻条款
SIBLING_MAX_DIST = 6           # 相邻判定：与命中条款的条号差 <= 此值
SIBLING_MAX_EXTRA = 16         # 兄弟扩展总共最多补多少条

ROOT = Path(__file__).resolve().parent.parent
LOCAL_EMBED = ROOT / "models" / "bge-m3"
LOCAL_RERANK = ROOT / "models" / "bge-reranker-v2-m3"


def _load_env_quietly():
    """读项目根 .env（只填「尚不存在」的变量），读不到就算了。

    刻意**不** import qa_chain 里的同名函数 —— qa_chain 反过来会 import 本模块，
    顶层互相引用会成环。这里重复十几行读取逻辑，换来彻底没有循环风险。
    """
    p = ROOT / ".env"
    try:
        if not p.exists():
            return
        for line in p.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, v = line.split("=", 1)
            k, v = k.strip(), v.strip().strip('"').strip("'")
            if k and k not in os.environ:
                os.environ[k] = v
    except Exception:                                    # noqa: BLE001
        pass


# ------------------------------------------------------------------ 工具
_CN = {"零": 0, "〇": 0, "一": 1, "二": 2, "三": 3, "四": 4,
       "五": 5, "六": 6, "七": 7, "八": 8, "九": 9}
_UNIT = {"十": 10, "百": 100, "千": 1000}


def cn2int(s: str):
    """中文数字 → 整数：'十'=10、'二十三'=23、'一百零五'=105；已是阿拉伯数字则直取。"""
    s = s.strip()
    if not s:
        return None
    if s.isdigit():
        return int(s)
    sec = num = 0
    for ch in s:
        if ch in _CN:
            num = _CN[ch]
        elif ch in _UNIT:
            sec += (num or 1) * _UNIT[ch]
            num = 0
        else:
            return None
    return sec + num


RE_ART_REF = re.compile(r"第\s*([0-9一二三四五六七八九十百零〇]{1,6})\s*条")
RE_FILE_REF = re.compile(r"《([^》]{2,40})》")


# ------------------------------------------------------------------ 数据结构
@dataclass
class Evidence:
    """一条证据 = 一个条款。eid 是给 LLM 看的编号（E1、E2…）。"""
    eid: str
    clause_id: str
    chunk_id: str
    cite: str                      # 完整引用串：《文件》 第X章 第X条
    file_id: str
    file_title: str
    article_label: str
    page: str
    text: str
    score: float
    detail: dict = field(default_factory=dict)

    def render(self) -> str:
        """渲染成喂给 LLM 的证据块。"""
        return (f"[{self.eid}] 出处：{self.cite}（PDF 第 {self.page} 页）\n"
                f"{self.text}")


# ------------------------------------------------------------------ 检索器
class HandbookRetriever:
    def __init__(self, data_dir="data", use_rerank=True, expand=True,
                 pool=100, rerank_pool=50, device=None, dtype=None,
                 backend=None, verbose=True):
        self.d = Path(data_dir)
        if not self.d.is_absolute():
            self.d = ROOT / self.d
        self.use_rerank = use_rerank
        self.expand = expand
        self.pool = pool
        self.rerank_pool = rerank_pool
        self.verbose = verbose
        self._device = device
        self._dtype = dtype          # None = 自动（CUDA 用 fp16，CPU 用 fp32）
        self.backend = self._pick_backend(backend)
        self._ready = False
        # 降级记录（2026-10-16）：向量编码 / 重排这两路都依赖外部接口，
        # 接口一旦不可用（额度用完、限流、断网），原来会**直接抛异常**，
        # 于是一道明明有答案的题也会被整条链路判成「答不出来」。现在改为
        # **记一笔、继续用关键词路（BM25）检索** —— 质量略降，但照样有货。
        # 每条一句话的人话说明，页面上要明说，不许静默降级（见 app.py）。
        self.degraded = []

    # -------------------------------------------------------------- 后端选择
    @staticmethod
    def _pick_backend(explicit=None) -> str:
        """决定「问题侧编码」走本机模型还是在线接口。

        优先级：显式传参 > `.env` 的 `EMBED_BACKEND` > **自动嗅探**。
        自动嗅探的依据是本地模型在不在（`models/bge-m3/config.json` 是否存在）：
        本机装着模型 → local；部署到 Streamlit Cloud（模型不随仓库上传）→ api。
        这样换环境不必改配置，也不会因为忘了设开关而直接跑不起来。

        注意区分：**文档侧**向量（data/vectors.npy）无论哪条路都是复用的，
        这里切换的只是「每次提问现场算问题向量 + 重排」那两步。
        """
        if explicit:
            return str(explicit).strip().lower()
        v = (os.environ.get("EMBED_BACKEND") or "").strip().lower()
        if not v:
            _load_env_quietly()
            v = (os.environ.get("EMBED_BACKEND") or "").strip().lower()
        if v in ("local", "api"):
            return v
        return "local" if (LOCAL_EMBED / "config.json").exists() else "api"

    # -------------------------------------------------------------- 降级
    def _mark_degraded(self, kind: str, err: Exception) -> str:
        """记下「哪一路用不了」，但不中断检索。

        检索本来是「向量 + 关键词」双路，缺了一路还有一路。所以某一路报错时
        正确的反应是**降级继续**，而不是把整道题判死 —— 2026-10-16 踩过：
        线上在线编码的额度用完（HTTP 402），向量路每次提问都抛异常，导致
        **每一道题都在检索这一步失败**，学生看到的却是「本手册中没有对应内容」。
        """
        note = f"{kind}不可用（{type(err).__name__}: {str(err)[:160]}）"
        if note not in self.degraded:
            self.degraded.append(note)
            self._log(f"[降级] {note} —— 本次改用剩下的路继续检索")
        return note

    # -------------------------------------------------------------- 加载
    def _log(self, msg):
        if self.verbose:
            print("  " + msg, flush=True)

    def build(self):
        """加载索引（不加载模型，模型在首次 search 时惰性加载）。"""
        if self._ready:
            return
        t0 = time.time()
        self.chunks = [json.loads(l) for l in
                       (self.d / "index_chunks.jsonl").open(encoding="utf-8")]
        self.cmap = {c["chunk_id"]: c for c in self.chunks}
        self.clauses = {c["clause_id"]: c for c in
                        json.load((self.d / "clauses.json").open(encoding="utf-8"))}
        self.vec = np.load(self.d / "vectors.npy").astype(np.float32)
        self.vids = json.load((self.d / "vector_ids.json").open(encoding="utf-8"))

        # 预建「条款 → 该条所有块」索引，父子扩展要用
        self.blocks_of = {}
        for c in self.chunks:
            cl = c.get("clause_id") or c["chunk_id"]
            self.blocks_of.setdefault(cl, []).append(c)

        self._ready = True
        self._log(f"索引 {len(self.chunks)} 块 / 向量 {self.vec.shape} / "
                  f"{time.time()-t0:.1f}s")

    def _ensure_models(self):
        if getattr(self, "_models_ready", False):
            return

        from bm25 import BM25
        t0 = time.time()
        # 与向量路吃同一份 embed_text：否则 BM25 白丢「专科/本科」这类
        # 文件级判别信号，对比会不公平（阶段 3 修正过这一点）。
        # BM25 是纯 Python 实现，与编码后端无关，两条路都要建。
        self.bm = BM25().fit([(c["chunk_id"], c["embed_text"]) for c in self.chunks])
        self._log(f"BM25 {self.bm.N} 篇 / 词表 {len(self.bm.df)} / {time.time()-t0:.1f}s")

        # ---------------- 在线后端：一行 torch 都不 import ----------------
        # 部署到 Streamlit Cloud 走的就是这条：那边没显卡、内存上限 2.7 GB，
        # 本机这套模型（共 4.4 GB）根本放不下。
        # ⚠ 构造本身也可能失败（密钥没配）→ 同样只降级、不抛，见 _mark_degraded。
        if self.backend == "api":
            self.embedder = None
            self.reranker = None
            try:
                from embed_api import ApiEmbedder, ApiReranker
                t0 = time.time()
                self.embedder = ApiEmbedder()
                self.reranker = ApiReranker() if self.use_rerank else None
                self._dev = "api"
                self._dtype_eff = None
                self._log(f"在线编码就绪 {time.time()-t0:.1f}s "
                          f"[{self.embedder.model}]")
            except Exception as e:                          # noqa: BLE001
                self._dev, self._dtype_eff = "api", None
                self._mark_degraded("在线编码", e)
            self._models_ready = True
            return

        # ---------------- 本机后端（原逻辑，未改动） ----------------
        self.embedder, self.reranker = None, None
        try:
            import torch
            dev = self._device or ("cuda" if torch.cuda.is_available() else "cpu")
            self._dev = dev
            # 精度：CUDA 上默认 **fp16**。
            # 本机 8GB 显存要同时驻留 bge-m3 + bge-reranker-v2-m3（各 ~568M），
            # fp32 下实测已逼近上限（7.7/8.1 GB），batch 调大反而更慢；
            # 半精度让显存减半、矩阵算力翻倍。重排只需相对次序，fp16 对 top-k
            # 排序影响极小（见 diag_precision 对比：召回集合一致）。
            dtype = self._dtype
            if dtype is None:
                dtype = torch.float16 if str(dev).startswith("cuda") else torch.float32
            self._dtype_eff = dtype
            mkw = {"torch_dtype": dtype}

            from sentence_transformers import SentenceTransformer
            src = (str(LOCAL_EMBED) if (LOCAL_EMBED / "config.json").exists()
                   else "BAAI/bge-m3")
            t0 = time.time()
            self.embedder = SentenceTransformer(src, device=dev, model_kwargs=mkw)
            try:
                self.embedder.max_seq_length = 8192
            except Exception:
                pass
            self._log(f"bge-m3 就绪 {time.time()-t0:.1f}s [{dev}/{dtype}]")

            if self.use_rerank and (LOCAL_RERANK / "config.json").exists():
                from sentence_transformers import CrossEncoder
                t0 = time.time()
                self.reranker = CrossEncoder(str(LOCAL_RERANK), device=dev,
                                             max_length=1024, model_kwargs=mkw)
                self._log(f"bge-reranker-v2-m3 就绪 {time.time()-t0:.1f}s "
                          f"[{dev}/{dtype}]")
        except Exception as e:                              # noqa: BLE001
            self._mark_degraded("本机向量模型", e)
        self._models_ready = True

    # -------------------------------------------------------------- 两路召回
    def _dense(self, query, k):
        e = self.embedder.encode([query], normalize_embeddings=True,
                                 convert_to_numpy=True).astype(np.float32)[0]
        sims = self.vec @ e
        order = np.argsort(-sims)[:k]
        return [(self.vids[int(i)], float(sims[int(i)])) for i in order]

    def _bm25(self, query, k):
        return self.bm.search(query, k=k)

    @staticmethod
    def _rrf(lists, weights):
        score = {}
        for lst, w in zip(lists, weights):
            for rank, cid in enumerate(lst, start=1):
                score[cid] = score.get(cid, 0.0) + w / (RRF_K + rank)
        return sorted(score.items(), key=lambda x: -x[1])

    # -------------------------------------------------------------- 条款号直达
    def _direct_hits(self, query):
        """识别「第X条」并按文件名（若给出）缩小范围，返回匹配的 chunk_id。"""
        m = RE_ART_REF.search(query)
        if not m:
            return []
        no = cn2int(m.group(1))
        if no is None or no < 1:
            return []
        fm = RE_FILE_REF.search(query)
        hint = fm.group(1) if fm else None

        out = []
        for cl, c in self.clauses.items():
            if c.get("article_no") != no:
                continue
            if hint and hint not in (c.get("file_title") or ""):
                continue
            # 只要该条的 article 块（chunk_id == clause_id）
            if cl in self.cmap:
                out.append(cl)
        return out[:DIRECT_BOOST_CAP]

    # -------------------------------------------------------------- 证据构造
    def _make_evidence(self, chunk, score, idx, detail=None):
        cl = chunk.get("clause_id") or chunk["chunk_id"]
        # 父子块扩展：一律取该条的 article 块原文（含公式修复后的文本）
        src = self.cmap.get(cl, chunk)
        text = src.get("text") or chunk.get("text") or ""
        if len(text) > MAX_EVIDENCE_CHARS:
            text = text[:MAX_EVIDENCE_CHARS] + "……（原文较长，此处截断）"
        page = src.get("page") or src.get("page_start") or ""
        return Evidence(
            eid=f"E{idx}",
            clause_id=cl,
            chunk_id=chunk["chunk_id"],
            cite=src.get("cite") or chunk.get("cite") or "",
            file_id=src.get("file_id") or chunk.get("file_id") or "",
            file_title=src.get("file_title") or "",
            article_label=src.get("article_label") or "",
            page=str(page),
            text=text,
            score=round(float(score), 4),
            detail=detail or {},
        )

    # -------------------------------------------------------------- 兄弟条款扩展
    def _sibling_expand(self, evidences, base_k, max_extra=SIBLING_MAX_EXTRA,
                        per_file=SIBLING_PER_FILE):
        """同一份文件贡献了 ≥2 条证据时，把该文件里与命中条款相邻
        （条号相差 ≤ SIBLING_MAX_DIST）的条款补进来。

        为什么需要：一个主题常分散在同一份办法的相邻几条上。例如
        《国家奖助学金评审管理办法》里，国家奖学金、励志奖学金、助学金
        分列在第三条、第五条、第七条 —— 跨这三类的归纳型提问，纯靠向量
        召回很容易只命中一两条，「三类各自多少钱」就答不全（Q37 实测）。

        三参数（per_file / max_dist / max_extra）是阶段 6 第 0 步的调优点，
        取值见文件头部 SIBLING_* 常量；调用方要想临时覆盖可显式传参。
        """
        from collections import Counter
        cnt = Counter(e.file_id for e in evidences if e.file_id)
        multi = {f for f, n in cnt.items() if n >= 2}
        if not multi:
            return evidences

        used = {e.clause_id for e in evidences}
        extra = []
        for f in multi:
            blocks = [c for c in self.chunks
                      if c.get("file_id") == f and c["chunk_type"] == "article"
                      and c.get("article_no") is not None]
            hit = [c["article_no"] for c in blocks if c["chunk_id"] in used]
            if not hit:
                continue
            cand = []
            for c in blocks:
                if c["chunk_id"] in used:
                    continue
                dist = min(abs(c["article_no"] - h) for h in hit)
                if dist <= SIBLING_MAX_DIST:
                    cand.append((dist, c))
            cand.sort(key=lambda x: x[0])
            extra.extend(c for _, c in cand[:per_file])

        out = list(evidences)
        for c in extra:
            if len(out) >= base_k + max_extra:
                break
            out.append(self._make_evidence(
                c, 0.0, len(out) + 1, detail={"expand": "sibling"}))
        return out

    # -------------------------------------------------------------- 主入口
    def search(self, query, top_k=8, debug=False):
        self.build()
        self._ensure_models()
        t0 = time.time()

        # 向量路：接口挂了就跳过（别让一道有答案的题因为外部接口而整题失败）。
        # 记过一次之后就把 embedder 置空 —— 否则每一路问句都要白等一次失败。
        dense_res = []
        if getattr(self, "embedder", None) is not None:
            try:
                dense_res = self._dense(query, self.pool)
            except Exception as e:                          # noqa: BLE001
                self._mark_degraded("向量检索", e)
                self.embedder = None
        bm25_res = self._bm25(query, self.pool)

        if dense_res:
            fused = self._rrf([[c for c, _ in dense_res], [c for c, _ in bm25_res]],
                              [DENSE_WEIGHT, BM25_WEIGHT])
            fused_ids = [c for c, _ in fused]
            fscore = dict(fused)
        else:
            # 降级：只剩关键词这一路，就用 BM25 的**原始分**当相关分
            # （0 = 一个字都没对上），这样下面「拒答前置」的门槛仍然有意义 ——
            # 换成 RRF 分（恒为 1/61 量级）会让阈值判断完全失真。
            fused_ids = [c for c, _ in bm25_res]
            fscore = {c: float(s) for c, s in bm25_res}

        # 条款号直达 → 前置
        direct = self._direct_hits(query)
        if direct:
            dset = set(direct)
            fused_ids = direct + [c for c in fused_ids if c not in dset]

        cand = fused_ids[:self.rerank_pool]
        scores = {c: fscore.get(c, 0.0) for c in cand}

        if self.reranker is not None and cand:
            try:
                pairs = [(query, self.cmap[c]["embed_text"]) for c in cand]
                sc = self.reranker.predict(pairs, batch_size=16,
                                           show_progress_bar=False)
                order = np.argsort(-np.asarray(sc))
                scores = {cand[int(i)]: float(sc[int(i)]) for i in order}
                cand = [cand[int(i)] for i in order]
            except Exception as e:                          # noqa: BLE001
                # 重排也走外部接口，同样可能挂。挂了就按融合顺序出结果 ——
                # 位置就是「关键词 + 向量」的共识排序，不是瞎排。
                self._mark_degraded("重排", e)
                self.reranker = None

        # 条款级去重 → 证据
        seen, evidences = set(), []
        dmap = {c: True for c in direct}
        for cid in cand:
            cl = self.cmap[cid].get("clause_id") or cid
            if cl in seen:
                continue
            seen.add(cl)
            evidences.append(self._make_evidence(
                self.cmap[cid], scores.get(cid, 0.0), len(evidences) + 1,
                detail={"direct": bool(dmap.get(cid)),
                        "rrf": round(fscore.get(cid, 0.0), 6),
                        "rerank": round(scores.get(cid, 0.0), 4)}))
            if len(evidences) >= top_k:
                break

        if self.expand:
            evidences = self._sibling_expand(evidences, top_k)

        if debug:
            self._log(f"检索 {time.time()-t0:.2f}s  "
                      f"直达命中 {len(direct)} 条  证据 {len(evidences)} 条")
        return evidences

    # -------------------------------------------------------------- 条款号反查
    def by_clause_id(self, clause_id):
        """按条款号直达原文（阶段 2 验收项：能按 clause_id 反查页码与原文）。"""
        self.build()
        c = self.cmap.get(clause_id)
        if not c:
            return None
        return self._make_evidence(c, 1.0, 1)


if __name__ == "__main__":
    r = HandbookRetriever("data")
    q = sys.argv[1] if len(sys.argv) > 1 else "考核不及格可以补考吗"
    for ev in r.search(q, top_k=5, debug=True):
        print("-" * 70)
        print(ev.render()[:400])
