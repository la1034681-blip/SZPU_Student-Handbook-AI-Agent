# -*- coding: utf-8 -*-
"""验模力方舟（Gitee AI）的令牌，以及它的向量/重排模型能不能顶上。

    python src/dev/diag/check_moark_key.py BGD3xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
    python src/dev/diag/check_moark_key.py          # 不给就用 .env 里的 EMBED_API_KEY

为什么要有这个（2026-10-16）：硅基流动的钥匙连续三把都回「平台不认识」，而线上
检索编码（向量 + 重排）非走外部接口不可。模力方舟是备选：Gitee 账号登录即有令牌，
向量与重排模型免费，模型名**不带** `BAAI/` 前缀，bge-m3 同为 **1024 维**
⇒ 与现有手册向量库通用，**不用重算手册**，只需改 4 个配置项。

本脚本按顺序验四件事：
  ① 令牌认不认      —— 列模型清单
  ② 向量模型在不在  —— 必须看到 bge-m3，且算出来是 1024 维
  ③ 重排模型在不在  —— 必须看到 bge-reranker-v2-m3，且返回格式与硅基一致
  ④ 资源包余额      —— 0 也能跑（说明免费档生效）；但据此提醒每日次数上限的风险
只读，不写任何配置。
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))
sys.stdout.reconfigure(encoding="utf-8")

BASE = "https://api.moark.com/v1"
EMB = "bge-m3"
RRK = "bge-reranker-v2-m3"


def main() -> int:
    import requests

    key = (sys.argv[1] if len(sys.argv) > 1 else "").strip()
    if not key:
        from qa_chain import load_env_file
        import os
        load_env_file()
        key = (os.environ.get("EMBED_API_KEY") or "").strip()
    if not key:
        print("没有令牌可验：命令行给一个，或先在 .env 里配 EMBED_API_KEY")
        return 2

    print(f"令牌：{key[:8]}…{key[-4:]}（共 {len(key)} 字符）")
    h = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}

    print("\n① 令牌认不认（列模型清单）")
    try:
        r = requests.get(f"{BASE}/models", timeout=30, headers=h)
        d = r.json()
    except Exception as e:                                    # noqa: BLE001
        print(f"   ✗ 连不上：{type(e).__name__}: {e}")
        return 1
    ids = [m.get("id") for m in (d.get("data") or [])]
    if not ids:
        print(f"   ✗ 被拒：{d}")
        return 1
    print(f"   ✓ 令牌有效，平台返回 {len(ids)} 个模型")
    for m in (EMB, RRK):
        print(f"   {'✓' if m in ids else '✗'} 清单里有 {m}")

    print("\n② 向量模型能不能算（维度必须是 1024）")
    r = requests.post(f"{BASE}/embeddings", timeout=40, headers=h,
                      json={"model": EMB, "input": "补考成绩怎么算"})
    e = r.json()
    vec = ((e.get("data") or [{}])[0] or {}).get("embedding")
    if not vec:
        print(f"   ✗ {e}")
        return 1
    dim = len(vec)
    print(f"   ✓ 通，{dim} 维" + ("（与现有手册向量库一致）" if dim == 1024
                                  else "（⚠ 不是 1024 维，不能直接替换）"))

    print("\n③ 重排模型能不能排（返回格式要与硅基一致）")
    r = requests.post(f"{BASE}/rerank", timeout=40, headers=h,
                      json={"model": RRK, "query": "补考",
                            "documents": ["必修课不及格可参加补考", "宿舍熄灯时间"],
                            "top_n": 2})
    rr = r.json()
    res = rr.get("results") or []
    if not res:
        print(f"   ✗ {rr}")
        return 1
    top = res[0]
    ok_shape = "index" in top and "relevance_score" in top
    print(f"   {'✓' if ok_shape else '✗'} 通，最相关那条 = 第 {top.get('index')} 条"
          f"（分数 {top.get('relevance_score')}）")

    print("\n④ 资源包余额（0 也能跑＝免费档生效）")
    try:
        b = requests.get(f"{BASE}/tokens/packages/balance", timeout=20,
                         headers=h).json()
        print(f"   余额 {b.get('balance')} / 总量 {b.get('total_amount')}"
              f" / 已用 {b.get('used_amount')}")
        print("   ⚠ 余额为 0 却能调通 ⇒ 向量与重排走的是免费档，可以放心用。")
        print("     2026-10-16 实测（probe_moark_quota.py）：向量连打 450+ 次、重排 100 次、")
        print("     对话 130 次，全部零限流，used_amount 恒为 0 ⇒ 官方 FAQ 写的「每日 100 次」")
        print("     实际卡不到向量与重排。但那条规矩仍在官方文档上，随时可能收紧——")
        print("     真撞限只会降级、页面出「本次检索打了折扣」的黄条，不会崩。")
    except Exception as e:                                    # noqa: BLE001
        print(f"   （没查到：{type(e).__name__}: {e}）")

    print("\n结论：可以把检索编码搬到模力方舟。线上 Secrets 把这 5 行换成：")
    print(f'EMBED_BACKEND = "api"\nEMBED_API_BASE = "{BASE}"\n'
          f'EMBED_API_MODEL = "{EMB}"\nEMBED_API_RERANK_MODEL = "{RRK}"\n'
          f'EMBED_API_KEY = "{key}"')
    print("\n⚠ 本地 .env 用的是 EMBED_BACKEND=local（走本机模型），"
          "和线上是两份独立配置，改一份不同步另一份。")
    return 0 if dim == 1024 and ok_shape else 1


if __name__ == "__main__":
    sys.exit(main())
