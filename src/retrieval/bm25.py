# -*- coding: utf-8 -*-
"""BM25 检索（自研实现，中文按字符 bigram 切分）。

为什么要自研而不是调库：
- 工程计划里明确把「混合检索 + 消融实验」列为项目亮点，检索算法必须自己可控，
  否则说不出「加深 BM25 后 Recall@5 提升多少」；
- 规章文本的核心检索信号是**术语与条款编号**（「补考」「第41号令」），
  这类精确匹配恰恰是向量检索的弱项、BM25 的强项。

中文切分策略：字符 bigram + 英文/数字整体 token。
不用分词词典，避免"教务处/教务/处"这类切分歧义；对法规文本实测足够。
"""
import json
import math
import re
from collections import Counter, defaultdict

RE_CN = re.compile(r"[\u4e00-\u9fff]+")
RE_ALNUM = re.compile(r"[A-Za-z]+|\d+(?:\.\d+)?")
RE_ARTNO = re.compile(r"第\s*([一二三四五六七八九十百零〇\d]{1,4})\s*条")


def tokenize(text):
    """中文 bigram + 英文/数字 token。"""
    toks = []
    if not text:
        return toks
    for seg in RE_CN.findall(text):
        if len(seg) == 1:
            toks.append(seg)
        else:
            toks.extend(seg[i:i + 2] for i in range(len(seg) - 1))
    toks.extend(RE_ALNUM.findall(text))
    return toks


class BM25:
    def __init__(self, k1=1.5, b=0.75, artno_boost=3.0):
        self.k1, self.b, self.artno_boost = k1, b, artno_boost
        self.ids = []
        self.tfs = []           # 每篇的 term -> tf
        self.lens = []          # 每篇 token 数
        self.df = Counter()     # term -> 文档频次
        self.avgdl = 0.0
        self.N = 0
        self.artnos = []        # 每篇出现的条号（用于条款编号加权）

    def fit(self, docs):
        """docs: iterable of (doc_id, text)"""
        self.ids, self.tfs, self.lens, self.df, self.artnos = [], [], [], Counter(), []
        for cid, text in docs:
            toks = tokenize(text)
            tf = Counter(toks)
            self.ids.append(cid)
            self.tfs.append(tf)
            self.lens.append(len(toks))
            self.artnos.append(set(RE_ARTNO.findall(text or "")))
            for t in tf:
                self.df[t] += 1
        self.N = len(self.ids)
        self.avgdl = (sum(self.lens) / self.N) if self.N else 0.0
        return self

    def _idf(self, term):
        n = self.df.get(term, 0)
        if n == 0:
            return 0.0
        return math.log(1 + (self.N - n + 0.5) / (n + 0.5))

    def search(self, query, k=10):
        """返回 [(doc_id, score)]，score 已按条号加权。"""
        q_toks = Counter(tokenize(query))
        if not q_toks or not self.N:
            return []
        q_artnos = set(RE_ARTNO.findall(query))
        scores = defaultdict(float)
        for term, qn in q_toks.items():
            idf = self._idf(term)
            if idf <= 0:
                continue
            for i, tf in enumerate(self.tfs):
                f = tf.get(term)
                if not f:
                    continue
                denom = f + self.k1 * (1 - self.b + self.b * self.lens[i] / (self.avgdl or 1))
                scores[i] += idf * (f * (self.k1 + 1)) / denom * qn
        # 条款编号精确命中加权：用户问「第41号令」时，含该条号的条款应显著上浮
        if q_artnos:
            for i in scores:
                if q_artnos & self.artnos[i]:
                    scores[i] *= self.artno_boost
        top = sorted(scores.items(), key=lambda x: -x[1])[:k]
        return [(self.ids[i], s) for i, s in top]


if __name__ == "__main__":
    import sys
    from pathlib import Path
    data = Path(sys.argv[1] if len(sys.argv) > 1 else "data")
    docs = [(r["chunk_id"], r["text"]) for r in
            (json.loads(l) for l in (data / "index_chunks.jsonl").open(encoding="utf-8"))]
    b = BM25().fit(docs)
    print(f"索引 {b.N} 篇，平均长度 {b.avgdl:.1f} token，词表 {len(b.df)}")
    for q in ["补考", "考试作弊怎么处理", "国家励志奖学金", "课程绩点怎么计算",
              "第41号令", "宿舍大功率电器"]:
        print(f"\n问：{q}")
        for cid, s in b.search(q, k=3):
            print(f"   {s:8.3f}  {cid}")
