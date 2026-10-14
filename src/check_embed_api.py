# -*- coding: utf-8 -*-
"""一致性验证：在线编码接口 与 本机模型，到底是不是同一回事。

为什么非验不可
--------------
`data/vectors.npy` 是本机 bge-m3（fp16、normalize_embeddings=True）算出来的。
线上把「问题侧编码」换成接口之后，**问题向量**必须落在同一个语义空间、
同一个尺度上 —— 否则「库里向量 · 问题向量」的相似度会整体漂移，
检索结果跟着变。

这种漂移很隐蔽：页面不报错、答案也长得像模像样，只是悄悄变差。
所以上线前必须拿证据说话。

验三件事
--------
① **向量一致性**：同一批问题，本机与接口各编码一次，求余弦相似度。
   同一模型 + 同样归一化 → 应当 ≥ 0.99（留一点精度差余量）。
② **重排排序一致性**：同一批候选文档，两边重排后比较顺序。
   ⚠ 分数**不能直接比数值** —— 本机 CrossEncoder 出的是 logits，
   接口给的是 0~1 相关度，量纲不同；只比排序（Spearman + top-5 重合）。
③ **端到端**：同一道题分别用 local / api 检索，看拿到的条款集合重合度。
   这一条是最终判据：前两条再好，端到端不一致也白搭。

用法（本机需有显卡 + 模型，且 .env 里配了 EMBED_API_KEY）
    python src/check_embed_api.py          # 默认 12 题
    python src/check_embed_api.py 30       # 自定义题数（最多 60）
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.stdout.reconfigure(encoding="utf-8")

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from qa_chain import load_env_file          # noqa: E402
from retrievers import HandbookRetriever    # noqa: E402

COS_MIN = 0.99          # 向量余弦的合格线
ORDER_MIN = 0.80        # top-5 重合率的合格线
E2E_MIN = 0.80          # 端到端条款重合率的合格线


def load_questions(n: int) -> list:
    d = json.load((ROOT / "data" / "eval_set.json").open(encoding="utf-8"))
    items = d.get("items") if isinstance(d, dict) else d
    out = []
    for it in items or []:
        q = it.get("question") or it.get("q") or it.get("query")
        if q:
            out.append(str(q).strip())
        if len(out) >= n:
            break
    return out


def spearman(a, b) -> float:
    """秩相关系数：只关心两边排出来的名次像不像，与分数大小无关。"""
    ra = np.argsort(np.argsort(a))
    rb = np.argsort(np.argsort(b))
    if np.std(ra) == 0 or np.std(rb) == 0:
        return 0.0
    return float(np.corrcoef(ra, rb)[0, 1])


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 12
    load_env_file()
    qs = load_questions(n)
    if not qs:
        print("没读到题目，检查 data/eval_set.json")
        return
    print(f"参与验证的题目：{len(qs)} 道\n")

    print("── 加载两条后端 ──")
    local = HandbookRetriever("data", backend="local", verbose=True)
    local.build()
    local._ensure_models()
    api = HandbookRetriever("data", backend="api", verbose=True)
    api.build()
    api._ensure_models()
    print()

    # ---------------------------------------------------------- ① 向量一致性
    print("── ① 向量一致性 ──")
    lv = np.asarray(local.embedder.encode(qs, normalize_embeddings=True),
                    dtype=np.float32)
    av = np.asarray(api.embedder.encode(qs, normalize_embeddings=True),
                    dtype=np.float32)
    if lv.shape != av.shape:
        print(f"   ✗ 维度就不一样：本机 {lv.shape} vs 在线 {av.shape}")
        print("     说明两边不是同一个模型 —— 换接口模型前先停手")
        return
    cos = (lv * av).sum(axis=1)
    print(f"   向量维度 {lv.shape[1]}（应为 1024）")
    print(f"   余弦  最低 {cos.min():.5f}   中位 {np.median(cos):.5f}   "
          f"最高 {cos.max():.5f}")
    bad = [i for i, c in enumerate(cos) if c < COS_MIN]
    print(f"   低于 {COS_MIN} 的题：{len(bad)}/{len(qs)}")
    for i in bad[:5]:
        print(f"     · cos={cos[i]:.4f}  {qs[i][:38]}")

    # ---------------------------------------------------------- ② 重排排序
    print("\n── ② 重排排序一致性 ──")
    q0 = qs[0]
    fused = local._rrf([[c for c, _ in local._dense(q0, 30)],
                        [c for c, _ in local._bm25(q0, 30)]], [1.0, 1.0])
    cand = [local.cmap[c]["embed_text"] for c, _ in fused[:20]]
    pairs = [(q0, d) for d in cand]
    ls = np.asarray(local.reranker.predict(pairs), dtype=float)
    as_ = np.asarray(api.reranker.predict(pairs), dtype=float)
    rho = spearman(ls, as_)
    lo, ao = np.argsort(-ls), np.argsort(-as_)
    top1_same = int(lo[0]) == int(ao[0])
    k = 5
    overlap = len(set(lo[:k].tolist()) & set(ao[:k].tolist())) / k
    print(f"   候选 {len(cand)} 条 | 名次相关（Spearman）{rho:.4f}")
    print(f"   top-1 是否同一条：{top1_same}")
    print(f"   top-{k} 重合率：{overlap:.2f}")
    print(f"   （分数不可比：本机 {ls.min():.2f}~{ls.max():.2f} / "
          f"在线 {as_.min():.2f}~{as_.max():.2f}）")

    # ---------------------------------------------------------- ③ 端到端
    print("\n── ③ 端到端检索一致性（前 6 条条款重合率）──")
    rates = []
    for q in qs:
        a = {e.clause_id for e in local.search(q, top_k=6)}
        b = {e.clause_id for e in api.search(q, top_k=6)}
        r = len(a & b) / max(len(a), 1)
        rates.append(r)
        flag = " " if r >= E2E_MIN else "!"
        print(f"  {flag} {r*100:5.1f}%   {q[:36]}")
    avg = float(np.mean(rates))
    print(f"   平均重合率：{avg*100:.1f}%")

    # ---------------------------------------------------------- 结论
    print("\n" + "=" * 62)
    ok_vec = float(np.median(cos)) >= COS_MIN
    ok_ord = overlap >= ORDER_MIN and rho >= ORDER_MIN
    ok_e2e = avg >= E2E_MIN
    print(f"① 向量一致性  {'通过' if ok_vec else '不通过'}"
          f"（余弦中位 {np.median(cos):.5f}，线 {COS_MIN}）")
    print(f"② 排序一致性  {'通过' if ok_ord else '不通过'}"
          f"（top-5 重合 {overlap:.2f} / 名次相关 {rho:.4f}，线 {ORDER_MIN}）")
    print(f"③ 端到端      {'通过' if ok_e2e else '不通过'}"
          f"（条款重合 {avg*100:.1f}%，线 {E2E_MIN*100:.0f}%）")
    print("-" * 62)
    if ok_vec and ok_ord and ok_e2e:
        print("结论：三条全过 → 换在线接口不影响检索，可以上线。")
    else:
        print("结论：有项目没过 → 先别上线，把不通过的项贴出来一起看。")
        print("      最常见的原因：模型 ID 写错（必须是不带 Pro/ 的 "
              "BAAI/bge-m3）、或接口侧没做 L2 归一化。")
    print("=" * 62)


if __name__ == "__main__":
    main()
