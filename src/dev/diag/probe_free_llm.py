# -*- coding: utf-8 -*-
"""一次性探测：智谱免费档此刻的限流状态（不落盘，测完即弃）。"""
from __future__ import annotations

import json
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

# 本脚本在 src/dev/diag/：往上三层是 src/（放模块），四层是项目根（找 .env）
SRC = Path(__file__).resolve().parent.parent.parent
ROOT = SRC.parent
sys.path.insert(0, str(SRC))
from qa_chain import load_env_file                      # noqa: E402
load_env_file()

import os                                        # noqa: E402
BASE = os.environ["FREE_LLM_BASE_URL"].rstrip("/")
KEY = os.environ["FREE_LLM_API_KEY"]
MODEL = os.environ["FREE_LLM_MODEL"]


def once(tag: str) -> bool:
    body = json.dumps({
        "model": MODEL,
        "messages": [{"role": "user", "content": "说「好」一个字"}],
        "max_tokens": 8,
    }).encode("utf-8")
    req = urllib.request.Request(
        f"{BASE}/chat/completions", data=body,
        headers={"Content-Type": "application/json",
                 "Authorization": f"Bearer {KEY}"})
    t0 = time.time()
    try:
        with urllib.request.urlopen(req, timeout=30) as r:
            d = json.loads(r.read().decode("utf-8"))
        txt = str(d["choices"][0]["message"].get("content") or "").strip()
        print(f"  {tag:22s} ✓ {time.time()-t0:5.2f}s  model={d.get('model')}  回复={txt[:12]!r}")
        return True
    except urllib.error.HTTPError as e:
        raw = e.read().decode("utf-8", "replace")[:220]
        print(f"  {tag:22s} ✗ HTTP {e.code}  {raw}")
        return False
    except Exception as e:                                # noqa: BLE001
        print(f"  {tag:22s} ✗ {type(e).__name__}: {e}")
        return False


print(f"模型 {MODEL}｜探 3 次，间隔 3 秒")
ok = 0
for i in range(3):
    if once(f"第 {i+1} 次"):
        ok += 1
    if i < 2:
        time.sleep(3)
print(f"\n结果：{ok}/3 成功")
