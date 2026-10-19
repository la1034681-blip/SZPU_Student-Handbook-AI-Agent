# -*- coding: utf-8 -*-
"""验 Cloudflare Workers AI 的令牌，以及它的向量/重排模型能不能顶上。

    python src/dev/diag/check_cf_ai.py <TOKEN> <ACCOUNT_ID>
    python src/dev/diag/check_cf_ai.py <TOKEN>       # 试着自动找 Account ID

为什么要有这个（2026-10-20）：模力方舟（api.moark.com）从 Streamlit Cloud 出去
走不通——向量是 Read timed out、重排是 ConnectTimeout，而本地同一把钥匙 8 次全通
⇒ 不是密钥也不是代码，是美国到国内那条链路的问题。Workers AI 走全球边缘节点，
不绕回国内，且 `@cf/baai/bge-m3` 与现用模型同款 **1024 维** ⇒ 手册向量库不用重算。

本脚本按顺序验五件事：
  ① 令牌认不认        —— /user/tokens/verify
  ② Account ID 对不对  —— 列 AI 模型清单，能列出就是对的
  ③ 向量模型在不在    —— @cf/baai/bge-m3，必须 1024 维
  ④ 重排模型在不在    —— 先试 bge-reranker-base，再试 v2-m3，看排序对不对
  ⑤ 免费额度          —— Workers AI 没有余额接口，只说明换算口径
只读，不写任何配置。
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))
import _paths  # noqa: E402
sys.stdout.reconfigure(encoding="utf-8")

API = "https://api.cloudflare.com/client/v4"
EMB = "@cf/baai/bge-m3"
RRK_CANDIDATES = ["@cf/baai/bge-reranker-base", "@cf/baai/bge-reranker-v2-m3"]


def _mask(k: str) -> str:
    return f"{k[:10]}…{k[-4:]}（共 {len(k)} 字符）" if len(k) > 16 else k


def _post(url: str, key: str, payload: dict, timeout: int = 40):
    import requests
    t = time.time()
    r = requests.post(url, headers={"Authorization": f"Bearer {key}",
                                    "Content-Type": "application/json"},
                      json=payload, timeout=timeout)
    return r, time.time() - t


def main() -> int:
    import requests

    args = [a.strip() for a in sys.argv[1:]]
    key = args[0] if args else ""
    account = args[1] if len(args) > 1 else ""
    if not key:
        print("用法：check_cf_ai.py <TOKEN> [ACCOUNT_ID]")
        return 2

    print(f"令牌：{_mask(key)}")
    h = {"Authorization": f"Bearer {key}"}

    print("\n① 令牌认不认")
    try:
        d = requests.get(f"{API}/user/tokens/verify", headers=h, timeout=25).json()
    except Exception as e:                                    # noqa: BLE001
        print(f"   ✗ 连不上：{type(e).__name__}: {e}")
        return 1
    st = ((d.get("result") or {}).get("status") or "").lower()
    if not d.get("success") or st != "active":
        print(f"   ✗ 被拒：{json.dumps(d, ensure_ascii=False)[:300]}")
        return 1
    print(f"   ✓ 有效且激活（token id {((d.get('result') or {}).get('id') or '')[:12]}…）")

    print("\n② Account ID")
    if not account:
        for path, label in (("/accounts", "accounts"), ("/memberships", "memberships")):
            try:
                j = requests.get(f"{API}{path}", headers=h, timeout=25).json()
            except Exception as e:                            # noqa: BLE001
                print(f"   {label} 查不动：{type(e).__name__}")
                continue
            rows = j.get("result") or []
            if isinstance(rows, list) and rows:
                for row in rows:
                    acc = row.get("account") if label == "memberships" else row
                    if acc and acc.get("id"):
                        print(f"   ✓ 找到：{acc.get('id')}  ({acc.get('name')})")
                        account = acc.get("id")
                        break
            if account:
                break
        if not account:
            print("   ✗ 这把令牌没有「读账号」权限，查不到。")
            print("     ⇒ 去 dash.cloudflare.com/ 点进任意账号后看**地址栏**：")
            print("       https://dash.cloudflare.com/<这一串32位就是>/...")
            return 3
    print(f"   使用：{account}")

    print("\n③ 能不能列出 AI 模型（顺带证明 Account ID 对）")
    try:
        j = requests.get(f"{API}/accounts/{account}/ai/models/search?per_page=50",
                         headers=h, timeout=30).json()
        names = [m.get("name") for m in (j.get("result") or [])]
        if names:
            print(f"   ✓ 列出 {len(names)} 个，向量模型"
                  f"{'在' if EMB in names else '不在'}清单里")
        else:
            print(f"   （列出为空，可能权限只到 run：{str(j)[:150]}）")
    except Exception as e:                                    # noqa: BLE001
        print(f"   （跳过：{type(e).__name__}）")

    dim = 0
    print(f"\n④ 向量 {EMB}（必须 1024 维，否则手册向量库得重算）")
    try:
        r, dt = _post(f"{API}/accounts/{account}/ai/run/{EMB}", key,
                      {"text": ["补考成绩怎么算"]})
        j = r.json()
        res = j.get("result") or {}
        data = res.get("data") or []
        dim = len(data[0]) if data else 0
        if not dim:
            print(f"   ✗ HTTP {r.status_code} {json.dumps(j, ensure_ascii=False)[:250]}")
        else:
            print(f"   ✓ 通，{dim} 维，{dt:.2f}s"
                  + ("（与现有手册向量库一致）" if dim == 1024
                     else "（⚠ 不是 1024 维，不能直接替换）"))
    except Exception as e:                                    # noqa: BLE001
        print(f"   ✗ {type(e).__name__}: {e}")

    rrk = ""
    print("\n⑤ 重排（哪一版能排，且排序对不对）")
    docs = [{"text": "必修课不及格可以参加补考。"},
            {"text": "学生公寓大门开放时间为每天 6:00 至 23:30。"},
            {"text": "补考成绩按卷面成绩的 60% 记入。"}]
    for name in RRK_CANDIDATES:
        try:
            r, dt = _post(f"{API}/accounts/{account}/ai/run/{name}", key,
                          {"query": "补考成绩怎么算", "contexts": docs})
            j = r.json()
            res = (j.get("result") or {})
            rows = res.get("response") or res.get("results") or []
            if not rows:
                print(f"   ✗ {name}：HTTP {r.status_code} "
                      f"{json.dumps(j, ensure_ascii=False)[:180]}")
                continue
            best = sorted(rows, key=lambda x: -(x.get("score") or 0))[0]
            ok = int(best.get("id", -1)) in (0, 2)
            rrk = name
            print(f"   ✓ {name} 可用，{dt:.2f}s；最相关那条 = 第 {best.get('id')} 条"
                  f"（{best.get('score')}）{'—— 排序正确' if ok else '—— ⚠ 排序存疑'}")
            break
        except Exception as e:                                # noqa: BLE001
            print(f"   ✗ {name}：{type(e).__name__}: {e}")

    print("\n⑥ 免费额度口径")
    print("   Workers AI 免费档：每天 10,000 Neurons（不绑卡）。bge-m3 约 1,075")
    print("   Neurons / 百万 token ⇒ 每天几百万 token 量级；我们一题只要几百 token，")
    print("   学生问几千题也用不完。真撞顶只降级、不崩。")

    if dim == 1024 and rrk:
        print("\n结论：可以用。线上 Secrets 这样填（4 行）：")
        print(f'EMBED_BACKEND = "api"\n'
              f'EMBED_API_BASE = "{API}/accounts/{account}/ai"\n'
              f'EMBED_API_MODEL = "{EMB}"\n'
              f'EMBED_API_RERANK_MODEL = "{rrk}"\n'
              f'EMBED_API_KEY = "{key}"')
        return 0
    print("\n结论：暂不可用，见上面 ✗ 的那几步。")
    return 1


if __name__ == "__main__":
    sys.exit(main())
