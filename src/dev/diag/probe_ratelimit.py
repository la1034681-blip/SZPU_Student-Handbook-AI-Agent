# -*- coding: utf-8 -*-
"""探一探智谱 GLM-4-Flash 免费档的真实限流边界。

网上说法互相打架（有说 1 并发、有说 30 并发、有说完全不限），
与其信别人，不如自己打一轮：
  1) 串行连发 20 次，看单线程会不会被 429；
  2) 并发 5 / 并发 15 各打一轮，看抖动阈值落在哪；
  3) 记录每次的响应耗时，顺便看看免费档的速度稳不稳。
"""
from __future__ import annotations

import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor

import requests

sys.stdout.reconfigure(encoding="utf-8")

KEY = "e584837fa9c64fe4b7a389cddd2091b5.gnvBeaPl6077zoqy"
URL = "https://open.bigmodel.cn/api/paas/v4/chat/completions"
HDR = {"Authorization": "Bearer " + KEY, "Content-Type": "application/json"}
BODY = {"model": "glm-4-flash", "temperature": 0.0,
        "messages": [{"role": "user", "content": "回复两个字：收到"}]}


def one(i: int) -> dict:
    t = time.time()
    try:
        r = requests.post(URL, headers=HDR, json=BODY, timeout=60)
        return {"i": i, "code": r.status_code, "dt": round(time.time() - t, 2),
                "body": r.text[:120].replace("\n", " ")}
    except Exception as e:                                   # noqa: BLE001
        return {"i": i, "code": "ERR", "dt": round(time.time() - t, 2),
                "body": str(e)[:120]}


def summary(tag: str, rs: list[dict]):
    ok = [r for r in rs if r["code"] == 200]
    lim = [r for r in rs if r["code"] == 429]
    bad = [r for r in rs if r["code"] not in (200, 429)]
    dts = [r["dt"] for r in ok]
    print(f"\n【{tag}】共 {len(rs)} 发 → 成功 {len(ok)}｜429 限流 {len(lim)}｜其他异常 {len(bad)}")
    if dts:
        dts.sort()
        print(f"        成功请求耗时：最快 {dts[0]}s｜中位 {dts[len(dts)//2]}s｜最慢 {dts[-1]}s")
    for r in bad:
        print(f"        异常样本 [{r['i']}] {r['code']} {r['body']}")
    return len(ok), len(lim)


print("=" * 68)
print("GLM-4-Flash 免费档限流探测")
print("=" * 68)

print("\n[1/3] 串行连发 20 次（模拟慢慢问）…")
t0 = time.time()
res = [one(i) for i in range(20)]
print(f"      单线程总耗时 {round(time.time()-t0, 1)}s")
summary("串行 20 发", res)

for tag, n in (("并发 5", 5), ("并发 15", 15)):
    print(f"\n[2/3] {tag} 同时打 {n} 发 …")
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=n) as ex:
        res = list(ex.map(one, range(n)))
    print(f"      总耗时 {round(time.time()-t0, 1)}s")
    summary(f"{tag}（{n} 发）", res)

print("\n[3/3] 并发 15 连打两轮（看第二轮会不会被限）…")
with ThreadPoolExecutor(max_workers=15) as ex:
    r1 = list(ex.map(one, range(15)))
with ThreadPoolExecutor(max_workers=15) as ex:
    r2 = list(ex.map(one, range(15, 30)))
summary("第 1 轮（15 发）", r1)
summary("第 2 轮（15 发）", r2)
print("\n完成。")
