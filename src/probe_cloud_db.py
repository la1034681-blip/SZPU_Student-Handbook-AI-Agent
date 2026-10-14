# -*- coding: utf-8 -*-
"""云服务数据库连通性探针（Python 侧，模拟 Streamlit 服务端调用）。

目的：验证「非浏览器客户端」能否访问 WorkBuddy 云服务的 PostgREST 数据面。
判读：
  · 返回 Postgres 错误码（42501/42P01/23505…）→ 网关放行，只是权限问题 → 方案可行
  · 返回 HTTP 401 且 code=MISSING_CREDENTIALS 等 → 网关直接拦截 → 方案不可行
"""
import json
import sys
import uuid

import requests

sys.stdout.reconfigure(encoding="utf-8")

ENDPOINT = "https://szpu-handbook-qa.app.workbuddy.host"
PUBLISHABLE_KEY = "wbpk_5LyhZ0F5RWsUb899resz7e_VIoZzX64rPlPlexNlB5dPbg4fHrvRzsP"
BASE = f"{ENDPOINT}/.cloud/database/rest"
HEADERS = {
    "x-wb-webapp-access-key": PUBLISHABLE_KEY,
    "Content-Type": "application/json",
}

row = {
    "id": str(uuid.uuid4()),
    "session_id": "probe",
    "question": "探针测试：退学后宿舍怎么退？",
    "refused": False,
    "latency": 1.23,
    "n_evidence": 5,
    "cited_clauses": "F38-A015",
    "answer": "probe",
}

print("=" * 62)
print("① INSERT 探针  →", f"{BASE}/qa_log")
try:
    r = requests.post(f"{BASE}/qa_log", headers=HEADERS, json=row, timeout=30)
    print("   HTTP", r.status_code)
    print("   body:", r.text[:400])
except Exception as e:  # noqa: BLE001
    print("   请求异常:", e)

print()
print("② SELECT 探针  →", f"{BASE}/qa_log?select=id&limit=1")
try:
    r = requests.get(f"{BASE}/qa_log?select=id&limit=1", headers=HEADERS, timeout=30)
    print("   HTTP", r.status_code)
    print("   body:", r.text[:400])
except Exception as e:  # noqa: BLE001
    print("   请求异常:", e)

print()
print("③ 无鉴权头对照（应被拒）")
try:
    r = requests.get(f"{BASE}/qa_log?select=id&limit=1", timeout=30)
    print("   HTTP", r.status_code, "| body:", r.text[:200])
except Exception as e:  # noqa: BLE001
    print("   请求异常:", e)
