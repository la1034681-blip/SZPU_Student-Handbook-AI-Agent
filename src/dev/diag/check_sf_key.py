# -*- coding: utf-8 -*-
"""验一把硅基流动钥匙到底能不能用（不写任何配置，只做判定）。

    python src/dev/diag/check_sf_key.py sk-xxxxxxxx
    python src/dev/diag/check_sf_key.py            # 不给就用 .env 里那把

为什么要单独有个脚本（2026-10-16）：用户换钥匙时最容易**复制漏一个字符**，
症状就是一句英文 `Token is invalid`；而「钥匙无效」「账号欠费」「模型写错」
三种毛病的处理方式完全不同，必须一次分清。本脚本按顺序验四件事：
  ① 钥匙本身在不在（列模型清单）—— 无效会回 code 30014
  ② 账号状态（嵌入接口）—— 欠费会回 code 30001
  ③ 重排接口通不通
  ④ 嵌入向量能不能真的算出来（返回维度，应该 1024）
每条都给「人话结论」，最后打印一句总结 + 该干什么。
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))
import _paths  # noqa: E402 把 core/retrieval/infra/cli 也挂上搜索路径
sys.stdout.reconfigure(encoding="utf-8")

BASE = "https://api.siliconflow.cn/v1"
EMB = "BAAI/bge-m3"
RRK = "BAAI/bge-reranker-v2-m3"


def _post(path, key, payload):
    import requests
    return requests.post(f"{BASE}{path}", json=payload, timeout=40,
                         headers={"Authorization": f"Bearer {key}",
                                  "Content-Type": "application/json"})


def main() -> int:
    import requests
    key = (sys.argv[1] if len(sys.argv) > 1 else "").strip()
    if not key:
        from qa_chain import load_env_file
        import os
        load_env_file()
        key = (os.environ.get("EMBED_API_KEY") or "").strip()
    if not key:
        print("没有钥匙可验：命令行给一把，或先在 .env 里配 EMBED_API_KEY")
        return 2

    print(f"钥匙：{key[:8]}…{key[-4:]}（共 {len(key)} 字符）")
    if len(key) != 51 or not key.startswith("sk-"):
        print("⚠ 长度/开头不像硅基流动的钥匙（正常是 sk- + 48 位 = 51 字符）"
              "—— 先怀疑复制不全。")

    print("\n① 钥匙有没有被平台认下来（列模型清单）")
    try:
        r = requests.get(f"{BASE}/models", timeout=30,
                         headers={"Authorization": f"Bearer {key}"})
        d = r.json()
    except Exception as e:                                   # noqa: BLE001
        print(f"   ✗ 连不上：{type(e).__name__}: {e}")
        return 1
    # ⚠ 出错时平台回的是 {"code":30014,"data":null,...} —— "data" 这个键**在**，
    # 只是值是 None，所以必须判「有没有内容」，不能判「键在不在」（踩过）。
    if not d.get("data"):
        print(f"   ✗ 被拒：{d}")
        print("   → 若 message 是 'Token is invalid'（code 30014）：钥匙本身不成立，"
              "去控制台点「复制」重抄一次（别手打、别从截图抄），"
              "并确认登录的是同一个站点（cloud.siliconflow.cn 国内 / .com 国际）。")
        return 1
    ids = [m.get("id") for m in d["data"]]
    for m in (EMB, RRK):
        print(f"   {'✓' if m in ids else '✗'} 模型清单里有 {m}")
    print(f"   ✓ 钥匙有效，平台共返回 {len(ids)} 个模型")

    print("\n② 账号状态（算一个向量）")
    r = _post("/embeddings", key, {"model": EMB, "input": "补考成绩怎么算"})
    e = r.json()
    if not e.get("data"):                     # 同上：出错时 data 键在、值是 None
        print(f"   ✗ {e}")
        print("   → 若 message 含 balance（code 30001）：这是**账号**问题（欠费拦截/"
              "余额为 0），跟模型免费与否无关；去控制台看余额与代金券，充值让余额 > 0。")
        return 1
    vec = e["data"][0]["embedding"]
    print(f"   ✓ 通，返回 {len(vec)} 维向量")

    print("\n③ 重排接口")
    r = _post("/rerank", key, {"model": RRK, "query": "补考",
                               "documents": ["必修课不及格可参加补考", "宿舍熄灯时间"],
                               "top_n": 2})
    rr = r.json()
    if not rr.get("results"):                 # 同上：出错时 results 键在、值是 None
        print(f"   ✗ {rr}")
        return 1
    top = rr["results"][0]
    print(f"   ✓ 通，最相关那条 = 第 {top.get('index')} 条"
          f"（分数 {top.get('relevance_score')}）")

    print("\n结论：这把钥匙可用，两个免费模型都正常。"
          "\n下一步：写进项目根 .env 的 EMBED_API_KEY（本地）"
          "＋ Streamlit Cloud → Settings → Secrets 里的 EMBED_API_KEY（线上）。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
