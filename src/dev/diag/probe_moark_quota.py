# -*- coding: utf-8 -*-
"""摸清模力方舟「免费体验访问令牌」的每日调用上限。

背景（2026-10-16 查官方 FAQ）：
  Q6  选择「免费体验访问令牌」即可享受免费体验，每位用户每日拥有 100 次免费调用次数。
  Q17 撞限时返回 400：「已达到最大当日免费 API 使用次数，请购买资源后继续使用 API。」
                「Today the free API access limit exceeded.」
  官方还写明：购买任意金额的全模型资源包后创建令牌，即可调用免费 API 服务
  （意思是：付费后不再受这 100 次的限制）。

本脚本做两件事：
  1) 连打 embeddings，直到撞限，报出「第几次开始被拦」＋原文错误信息；
  2) 撞限后，再单独试一次 rerank / chat，判断**额度是向量/重排共用一个池子，
     还是各自独立计数**。

用法：
    python src/dev/diag/probe_moark_quota.py <令牌> [最多打几次]

⚠ 副作用：会把该令牌当日的免费额度用尽（次日 0 点重置）。只在准备切换到
   模力方舟前做一次即可。
"""
from __future__ import annotations

import json
import sys
import time

import requests

BASE = "https://api.moark.com/v1"


def _one_embed(key: str) -> tuple[int, dict | str]:
    """单次向量调用，返回 (HTTP 状态码, 响应体)。"""
    try:
        r = requests.post(
            f"{BASE}/embeddings",
            headers={"Authorization": f"Bearer {key}"},
            json={"model": "bge-m3", "input": "补考成绩怎么算"},
            timeout=30,
        )
    except Exception as e:  # 网络类异常
        return -1, f"{type(e).__name__}: {e}"
    try:
        return r.status_code, r.json()
    except Exception:
        return r.status_code, r.text[:300]


def _short(body: dict | str) -> str:
    if isinstance(body, dict):
        msg = body.get("message") or body.get("error") or body
        return str(msg)[:200]
    return str(body)[:200]


def _one_rerank(key: str) -> tuple[int, dict | str]:
    try:
        r = requests.post(
            f"{BASE}/rerank",
            headers={"Authorization": f"Bearer {key}"},
            json={
                "model": "bge-reranker-v2-m3",
                "query": "补考",
                "documents": ["必修课不及格可参加补考", "宿舍熄灯时间"],
                "top_n": 2,
            },
            timeout=30,
        )
    except Exception as e:
        return -1, f"{type(e).__name__}: {e}"
    try:
        return r.status_code, r.json()
    except Exception:
        return r.status_code, r.text[:300]


def _one_chat(key: str) -> tuple[int, dict | str]:
    try:
        r = requests.post(
            f"{BASE}/chat/completions",
            headers={"Authorization": f"Bearer {key}"},
            json={
                "model": "Qwen3-8B",
                "messages": [{"role": "user", "content": "你好"}],
                "max_tokens": 8,
            },
            timeout=60,
        )
    except Exception as e:
        return -1, f"{type(e).__name__}: {e}"
    try:
        return r.status_code, r.json()
    except Exception:
        return r.status_code, r.text[:300]


def main() -> int:
    if len(sys.argv) < 2:
        print("用法：python src/dev/diag/probe_moark_quota.py <令牌> [最多打几次]")
        return 2
    key = sys.argv[1].strip()
    limit = int(sys.argv[2]) if len(sys.argv) > 2 else 130

    print("=" * 66)
    print("① 连续调用向量接口，直到平台开始拒绝")
    print("=" * 66)
    hits: dict[int, int] = {}
    first_block_at = None
    block_body = None
    t0 = time.time()
    for i in range(1, limit + 1):
        code, body = _one_embed(key)
        hits[code] = hits.get(code, 0) + 1
        if code != 200:
            first_block_at = i
            block_body = body
            print(f"  第 {i} 次 ✗ HTTP {code}：{_short(body)}")
            break
        if i % 10 == 0:
            print(f"  … 已成功 {i} 次（累计 {time.time()-t0:.0f}s）")
    if first_block_at is None:
        print(f"  ✓ 连续 {limit} 次全部成功，没撞到上限（说明今日额度 > {limit}，"
              f"或该令牌不是免费体验档）")
    else:
        print(f"  → 结论：第 {first_block_at} 次开始被拒")
        print(f"    原文：{_short(block_body)}")

    print()
    print("=" * 66)
    print("② 撞限后，重排接口还能不能用（判断是否同一个额度池）")
    print("=" * 66)
    code, body = _one_rerank(key)
    print(f"  重排 HTTP {code}：{_short(body) if code != 200 else '正常返回'}")

    print()
    print("=" * 66)
    print("③ 撞限后，对话模型还能不能用")
    print("=" * 66)
    code, body = _one_chat(key)
    print(f"  对话 HTTP {code}：{_short(body) if code != 200 else '正常返回'}")

    print()
    print("=" * 66)
    print("④ 资源包余额")
    print("=" * 66)
    try:
        r = requests.get(
            f"{BASE}/tokens/packages/balance",
            headers={"Authorization": f"Bearer {key}"},
            timeout=20,
        )
        print("  " + json.dumps(r.json(), ensure_ascii=False))
    except Exception as e:
        print(f"  查不到：{type(e).__name__}: {e}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
