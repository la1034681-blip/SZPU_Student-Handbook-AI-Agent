# -*- coding: utf-8 -*-
"""阶段 6 · 第 3 步诊断：拒答前置的阈值能不能定？

背景
----
组件四「拒答前置」想做的事：生成之前先看一眼检索回来的证据 ——
如果连最高分的证据都很不沾边（rerank 分极低），那就直接判 R1 拒答，
省掉「生成 + 校验」两次 LLM 调用。

但这里有个硬约束：**误拒率必须保持 0%**。
也就是说，阈值只能用来拦「明显不沾边」的，绝不能误伤「能答」的题。

所以要回答一个问题：
    7 道拒答题 与 37 道正常题，在「top-1 rerank 分」上分得开吗？

分得开 → 取两者之间的一段空档当阈值（宁可漏拦，不可误拦）；
分不开 → 如实说明，放弃这个机制（不能为了省调用而破坏守门指标）。

用法
----
    python src/dev/diag/diag_gate.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
import _paths  # noqa: E402 把 core/retrieval/infra/cli 也挂上搜索路径

from retrievers import HandbookRetriever  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent.parent.parent


def main():
    data = ROOT / "data"
    ev = json.load(open(data / "eval_set.json", encoding="utf-8"))
    items = ev["items"]

    # ⚠ 必须先把 .env 读进来（2026-10-17 踩到）：不读密钥时，`EMBED_BACKEND=api`
    #   会因为「密钥没填」而构造失败 → 静默降级成纯关键词（BM25）。那时脚本照样
    #   出分数、照样给「可分/不可分」的结论，但量到的根本不是线上那套**重排分**
    #   （BM25 原始分能到 20+，重排分是 0~1），结论完全不可比。
    from qa_chain import load_env_file
    load_env_file()

    r = HandbookRetriever(str(data), use_rerank=True, verbose=False)

    rows = []
    for it in items:
        # 用与正式链路一致的 top_k；只看"主检索"的最高分
        evs = r.search(it["question"], top_k=6)
        top1 = max((e.score for e in evs), default=None)
        rows.append({"qid": it["qid"], "type": it["type"],
                     "refuse_expected": bool(it.get("refuse_expected")),
                     "top1": top1})

    ref = [x for x in rows if x["refuse_expected"]]
    nor = [x for x in rows if not x["refuse_expected"]]

    # 先把「这次量的是哪条路」印出来 —— 降级状态下量出来的分布没有意义。
    print("=" * 72)
    print(f"编码后端 = {r.backend}　重排器 = {type(r.reranker).__name__}")
    if r.degraded:
        print("⚠ 检索发生了降级 —— 下面这些分数**不是**线上那套重排分：")
        for d in r.degraded:
            print(f"   - {d}")
    else:
        print("无降级，量到的就是线上同款分数")
    print("=" * 72)

    print("=" * 72)
    print("逐题 top-1 rerank 分（降序）")
    print("=" * 72)
    print("\n【拒答题 7 道】")
    for x in sorted(ref, key=lambda z: -(z["top1"] or -99)):
        print(f"  {x['qid']}  top1={x['top1']:>8.4f}")
    print("\n【正常题 37 道】—— 只看最低的 12 道（决定阈值下界）")
    for x in sorted(nor, key=lambda z: (z["top1"] if z["top1"] is not None else 99))[:12]:
        print(f"  {x['qid']} [{x['type']:<10}] top1={x['top1']:>8.4f}")

    ref_max = max(x["top1"] for x in ref if x["top1"] is not None)
    nor_min = min(x["top1"] for x in nor if x["top1"] is not None)
    print("\n" + "=" * 72)
    print(f"拒答题里的最高分（refuse_max） = {ref_max:.4f}")
    print(f"正常题里的最低分（normal_min） = {nor_min:.4f}")
    if nor_min > ref_max:
        print(f"→ 两类可分！空档 [{ref_max:.4f}, {nor_min:.4f}]，"
              f"阈值可取中值 {(ref_max + nor_min) / 2:.4f}")
    else:
        print("→ 两类**交叠**，无法用一个分数干净分开。"
              "必须取极度保守的阈值（低于所有正常题），"
              "或放弃该机制。")

    out = data / "diag_gate.json"
    json.dump({"rows": rows, "refuse_max": ref_max, "normal_min": nor_min},
              open(out, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print(f"\n明细已存：{out}")


if __name__ == "__main__":
    main()
