# -*- coding: utf-8 -*-
"""探测临时记忆表：走应用将来用的那条路径（RPC），并验证「直接读表读不到」。

    python src/dev/diag/probe_memory_table.py

三件事：
  ① 存 → 读回来，内容一致（含中文与嵌套 JSON）
  ② 直接 SELECT 表 —— 应该被拒（证明没有公开读，别人拿不到学生会话）
  ③ 清 → 再读为空；顺带试一下 sweep
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))
import _paths  # noqa: E402 把 core/retrieval/infra/cli 也挂上搜索路径
sys.stdout.reconfigure(encoding="utf-8")

import requests                                     # noqa: E402
from cloudstore import get_store                    # noqa: E402
from qa_chain import load_env_file                  # noqa: E402
from memory import (SessionMemory, clear_memory,    # noqa: E402
                    load_memory, save_memory, sweep_memory)

load_env_file()
st = get_store()
print("云库:", {"enabled": st.enabled, "endpoint": st.endpoint,
               "has_key": bool(st.key)})

MEM = "probe00000000000000000000000000ff"
payload = [("user", "旷课多少学时会被处分？"),
           ("assistant", "10-19 学时警告[E1]，50 学时以上开除学籍[E2]。")]
mem = SessionMemory(mem_id=MEM, client_id="probe-client")
for role, text in payload:
    mem.add(role, text)

print("\n① 存 → 读回（走应用真正用的 save_memory / load_memory）")
print("   save 返回:", save_memory(mem))
back = load_memory(MEM)
print("   load 到记忆:", back.brief() if back else None)
print("   内容一致:", bool(back) and back.turns == mem.turns)
print("   摘要一致:", bool(back) and back.summary == mem.summary)
if not back or back.turns != mem.turns:
    print("   实得:", str(getattr(back, "turns", None))[:200])

print("\n② 直接 SELECT 表（应被拒 → 说明别人读不到会话内容）")
try:
    r = requests.get(f"{st.endpoint}/.cloud/database/rest/qa_memory",
                     headers={"x-wb-webapp-access-key": st.key,
                              "Content-Type": "application/json"},
                     params={"select": "mem_id", "limit": "5"}, timeout=10)
    body = r.text[:200].replace("\n", " ")
    rows = None
    if r.status_code < 400:
        try:
            rows = r.json()
        except Exception:                                # noqa: BLE001
            rows = "解析不了"
    # 两种结果都算「安全」：明确拒绝(4xx)，或 200 但一行都看不到
    safe = r.status_code >= 400 or rows == []
    print(f"   HTTP {r.status_code} → {'安全 ✓' if safe else '泄露了 ✗'}")
    print("   响应:", body)
    if isinstance(rows, list) and rows:
        print("   ✗ 竟然读到了", len(rows), "行！")
except Exception as e:                                   # noqa: BLE001
    print("   请求异常（也算挡住了）:", type(e).__name__, str(e)[:120])

print("\n②b 直接往表里写（应被拒 → 只能走函数）")
try:
    r = requests.post(f"{st.endpoint}/.cloud/database/rest/qa_memory",
                      headers={"x-wb-webapp-access-key": st.key,
                               "Content-Type": "application/json"},
                      json={"mem_id": "hack0000000000000000000000000000ff"},
                      timeout=10)
    print(f"   HTTP {r.status_code} → {'已拒绝 ✓' if r.status_code >= 400 else '竟然写进去了 ✗'}")
    print("   响应:", r.text[:160].replace("\n", " "))
except Exception as e:                                   # noqa: BLE001
    print("   请求异常（也算挡住了）:", type(e).__name__, str(e)[:120])

print("\n③ 清 → 再读为空")
print("   clear 返回:", clear_memory(MEM))
print("   清完 load:", load_memory(MEM))
print("   sweep(闲置 60 分钟，返回删了几行):", sweep_memory(60))
