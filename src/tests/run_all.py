# -*- coding: utf-8 -*-
"""跑全套离线测试并汇总（2026-10-15 新增）。

为什么要这个：`test_*.py` 有 20+ 个，逐个手敲太累，而且中途被中断就看不出
跑到哪了。这个 runner 把每个脚本的退出码与最后一行结论写进一张表，
跑完一次性给结论 —— 用户说「其他地方有没有问题」时跑它。

只跑离线测试（*_e2e.py 要真调模型，一道题 30~60s，单独跑）。
用法：python src/tests/run_all.py
"""
from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
PY = sys.executable
TESTS = sorted((ROOT / "src" / "tests").glob("test_*.py"))
SKIP = {"test_followup_e2e.py", "test_kuangke_e2e.py", "test_renumber_e2e.py"}

rows = []
t_all = time.time()
for f in TESTS:
    if f.name in SKIP:
        rows.append((f.name, None, "跳过（要真调模型，单独跑）", 0))
        print(f"—— {f.name} 跳过", flush=True)
        continue
    t0 = time.time()
    p = subprocess.run([PY, str(f)], cwd=str(ROOT),
                       capture_output=True, text=True, encoding="utf-8",
                       errors="replace", timeout=900)
    dt = time.time() - t0
    tail = [ln.strip() for ln in (p.stdout or "").splitlines()
            if ln.strip() and "Loading weights" not in ln][-6:]
    verdict = next((ln for ln in reversed(tail)
                    if "全部通过" in ln or "失败" in ln or "有 " in ln), tail[-1] if tail else "")
    rows.append((f.name, p.returncode, verdict[:80], dt))
    print(f"{'OK  ' if p.returncode == 0 else 'FAIL'} {f.name:<28}"
          f"{dt:>6.1f}s  {verdict[:60]}", flush=True)

print("\n================ 汇总 ================", flush=True)
bad = [r for r in rows if r[1] is not None and r[1] != 0]
for name, rc, verdict, dt in rows:
    mark = "跳过" if rc is None else ("通过" if rc == 0 else "失败")
    print(f"  {mark}  {name:<28}{dt:>6.1f}s  {verdict}", flush=True)
print(f"\n共 {len(rows)} 个脚本，失败 {len(bad)} 个，"
      f"总耗时 {time.time() - t_all:.0f} 秒", flush=True)
for name, rc, verdict, dt in bad:
    print(f"  ✗ {name}：{verdict}", flush=True)
sys.exit(1 if bad else 0)
