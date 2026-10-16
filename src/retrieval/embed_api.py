# -*- coding: utf-8 -*-
"""在线编码与重排（模力方舟 moark / 硅基流动 SiliconFlow 等在线接口）
—— 把「问题侧推理」搬出本机显卡。

为什么需要它
------------
本机跑的是 bge-m3 + bge-reranker-v2-m3（各 2.2 GB，吃显卡）。
Streamlit Community Cloud 免费档**没有 GPU、内存上限 2.7 GB**，这两个模型
放都放不下。所以要部署上去，必须把「每次提问现场跑的那两步」换成调接口。

⚠ 注意区分两件事（很容易混）：
   · **文档侧**向量（data/vectors.npy，1510 段）是**离线算好**的 → 直接复用，不用动；
   · **问题侧**每次提问都得现场编码 → 这一步原来在本机 GPU 上跑，现在改成调接口。
  所以不是"把向量库搬走"，只是把现场那一步外包出去。

设计原则：**接口对齐 sentence-transformers**，让 retrievers.py 只换对象、不改算法。
    ApiEmbedder.encode(list[str])   ≈ SentenceTransformer.encode
    ApiReranker.predict(list[pair]) ≈ CrossEncoder.predict

两个硬约束（改动前务必先读）
--------------------------
1. **模型必须是 bge-m3（1024 维）**。库里存的向量就是它算的，换任何别的
   embedding 模型，维度或语义空间对不上，整个向量库作废。
2. **模型 ID 怎么写，看服务商，别凭记忆**：硅基流动要带组织前缀 `BAAI/bge-m3`
   （且必须是非 `Pro/` 那档，否则按 ¥0.07/百万 tokens 计费）；模力方舟**不带前缀**，
   写 `bge-m3`。写错 ID 轻则 400/404，硅基那边误带 `Pro/` 还会开始扣钱。

归一化：本机 encode 时用了 `normalize_embeddings=True`，所以接口返回后
我们**自己补一次 L2 归一化**，保证与库里向量同一尺度（否则余弦退化成内积）。
"""
from __future__ import annotations

import os
import sys
import time

import numpy as np

sys.stdout.reconfigure(encoding="utf-8")

DEFAULT_BASE = "https://api.siliconflow.cn/v1"
# ⚠ 上面三个 DEFAULT_* 只在环境变量一个都没配时兜底。**当前线上/本地实际用的是
#   模力方舟**（`EMBED_API_BASE=https://api.moark.com/v1`、模型名不带前缀），
#   真实取值一律以 .env / Streamlit Secrets 为准，别照 DEFAULT 猜。
DEFAULT_EMBED_MODEL = "BAAI/bge-m3"
DEFAULT_RERANK_MODEL = "BAAI/bge-reranker-v2-m3"


class ApiError(RuntimeError):
    """接口调用失败 —— 调用方据此决定是退本地模型还是直接报错。"""


def _friendly(status: int, body: str) -> str:
    """把 HTTP 错误翻译成能照着做的事，而不是甩一串 JSON 给用户。"""
    head = f"HTTP {status}"
    if status == 401:
        return (f"{head}：密钥无效或没填 —— 检查 EMBED_API_KEY"
                "（到所用服务商的控制台 → API 密钥页重新复制一把）")
    if status == 402:
        return f"{head}：额度不足 —— 确认用的是免费的 `BAAI/...` 而不是 `Pro/BAAI/...`"
    if status == 429:
        return f"{head}：触发限流 —— 免费档是固定速率上限，稍后重试"
    if status in (400, 422):
        return f"{head}：请求被拒（常见原因：模型 ID 写错、输入过长）—— {body[:160]}"
    return f"{head}：{body[:200]}"


class _Client:
    """极薄的 HTTP 客户端：只管发请求 + 重试，不懂业务。"""

    def __init__(self, base_url: str | None = None, api_key: str | None = None,
                 timeout: float | None = None, retries: int = 2):
        self.base = (base_url or os.environ.get("EMBED_API_BASE")
                     or DEFAULT_BASE).rstrip("/")
        self.key = (api_key or os.environ.get("EMBED_API_KEY") or "").strip()
        self.timeout = float(timeout or os.environ.get("EMBED_API_TIMEOUT") or 30)
        self.retries = int(retries)
        if not self.key:
            raise ApiError(
                "缺少 EMBED_API_KEY —— 在线编码需要服务商的密钥。\n"
                "  本地：写在项目根 .env\n"
                "  线上：写在 Streamlit Cloud → Settings → Secrets（TOML 格式）")

    def post(self, path: str, payload: dict) -> dict:
        import requests
        url = f"{self.base}{path}"
        headers = {"Authorization": f"Bearer {self.key}",
                   "Content-Type": "application/json"}
        last = ""
        for attempt in range(self.retries + 1):
            try:
                r = requests.post(url, json=payload, headers=headers,
                                  timeout=self.timeout)
            except Exception as e:                       # noqa: BLE001
                last = f"网络异常：{e}"
                if attempt < self.retries:
                    time.sleep(0.8 * (attempt + 1))
                    continue
                raise ApiError(f"连不上编码接口（{self.base}）：{e}") from e

            if r.status_code < 400:
                try:
                    return r.json()
                except Exception as e:                   # noqa: BLE001
                    raise ApiError(f"{path} 返回的不是合法 JSON：{r.text[:160]}") from e

            last = _friendly(r.status_code, r.text)
            # 只有限流与 5xx 值得重试；401/402/400 重试多少次都一样
            if r.status_code == 429 or r.status_code >= 500:
                if attempt < self.retries:
                    time.sleep(1.2 * (attempt + 1))
                    continue
            raise ApiError(last)
        raise ApiError(last)


class ApiEmbedder:
    """在线文本向量化。用法与 SentenceTransformer.encode 对齐。"""

    def __init__(self, model: str | None = None, base_url: str | None = None,
                 api_key: str | None = None, batch: int | None = None):
        self.c = _Client(base_url, api_key)
        self.model = (model or os.environ.get("EMBED_API_MODEL")
                      or DEFAULT_EMBED_MODEL)
        self.batch = int(batch or os.environ.get("EMBED_API_BATCH") or 32)

    def encode(self, texts, normalize_embeddings: bool = True,
               convert_to_numpy: bool = True, **kw):
        """一条或一批文本 → (n, 1024) float32；单条输入返回 (1024,)。"""
        single = isinstance(texts, str)
        items = [texts] if single else list(texts)
        if not items:
            return np.zeros((0, 1024), dtype=np.float32)

        rows: list = []
        for i in range(0, len(items), self.batch):
            part = items[i:i + self.batch]
            j = self.c.post("/embeddings", {
                "model": self.model,
                "input": part,
                "encoding_format": "float",
            })
            data = sorted(j.get("data") or [], key=lambda d: d.get("index", 0))
            if len(data) != len(part):
                raise ApiError(f"向量条数对不上：要 {len(part)} 条，回 {len(data)} 条")
            rows.extend(d.get("embedding") for d in data)

        arr = np.asarray(rows, dtype=np.float32)
        if normalize_embeddings:
            # 与本地 normalize_embeddings=True 保持同一尺度：内积 = 余弦
            norm = np.linalg.norm(arr, axis=1, keepdims=True)
            arr = arr / np.clip(norm, 1e-12, None)
        return arr[0] if single else arr


class ApiReranker:
    """在线重排。用法与 CrossEncoder.predict 对齐（入参是 (query, doc) 对）。

    注意：接口给的是 0~1 的相关度，本机 CrossEncoder 给的是 logits，**量纲不同**。
    下游只用相对次序（argsort），所以不影响排序；但界面上「相关度」那个数字
    线上与本地不可直接比较。
    """

    def __init__(self, model: str | None = None, base_url: str | None = None,
                 api_key: str | None = None):
        self.c = _Client(base_url, api_key)
        self.model = (model or os.environ.get("EMBED_API_RERANK_MODEL")
                      or DEFAULT_RERANK_MODEL)

    def _one(self, query: str, docs: list) -> list:
        out = [0.0] * len(docs)
        j = self.c.post("/rerank", {
            "model": self.model,
            "query": query,
            "documents": docs,
            "top_n": len(docs),
            "return_documents": False,
        })
        for item in (j.get("results") or []):
            idx = int(item.get("index", -1))
            if 0 <= idx < len(docs):
                out[idx] = float(item.get("relevance_score") or 0.0)
        return out

    def predict(self, pairs, batch_size: int = 16, show_progress_bar: bool = False,
                **kw) -> list:
        pairs = list(pairs)
        if not pairs:
            return []

        queries = {p[0] for p in pairs}
        if len(queries) > 1:
            # 罕见：多问题混合。逐条调，保证正确性优先。
            return [self._one(q, [d])[0] for q, d in pairs]

        query = pairs[0][0]
        docs = [p[1] for p in pairs]
        step = max(int(batch_size or 16), 1)
        scores: list = []
        for i in range(0, len(docs), step):
            part = docs[i:i + step]
            scores.extend(self._one(query, part))
        if len(scores) != len(docs):
            raise ApiError(f"重排条数对不上：要 {len(docs)}，回 {len(scores)}")
        return scores


# ------------------------------------------------------------------ 自测
def _selftest():
    q = sys.argv[1] if len(sys.argv) > 1 else "必修课考试不及格可以补考吗"
    docs = [
        "学生必修课考核不及格，可以参加补考。补考成绩按及格或不及格记载。",
        "学校食堂开放时间为早上六点半至晚上九点。",
        "国家奖学金每人每年八千元。",
    ]
    print(f"问题：{q}")
    t0 = time.time()
    e = ApiEmbedder()
    v = e.encode([q])
    print(f"① 编码 OK  模型={e.model}  维度={v.shape}  耗时={time.time()-t0:.2f}s")
    print(f"   已归一化（模长）={float(np.linalg.norm(v[0])):.6f}（应≈1.0）")

    t0 = time.time()
    r = ApiReranker()
    sc = r.predict([(q, d) for d in docs])
    print(f"② 重排 OK  模型={r.model}  耗时={time.time()-t0:.2f}s")
    for s, d in sorted(zip(sc, docs), key=lambda x: -x[0]):
        print(f"   {s:.4f}  {d[:28]}…")
    print("   排序正确（补考那条应当最高）：",
          max(range(len(sc)), key=lambda i: sc[i]) == 0)


if __name__ == "__main__":
    _selftest()
