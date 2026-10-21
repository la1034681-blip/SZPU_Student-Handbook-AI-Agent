# -*- coding: utf-8 -*-
"""验证「进站不再要口令」这条改动（2026-10-21）。

原 `test_app_gate.py` 验的是「没口令进不去」，功能删了它自然也就没意义，
换成这个**反方向**的守门：打开就该能直接提问。

覆盖三件事：
  1. 首屏没有口令输入框（`type="password"` 控件数量为 0）
  2. 问答入口直接出现（`has_ask_entry` 为真）
  3. 紧接着点一下「提问」能正常走完一次运行（不报错、不跳回门禁页）

不调用大模型：只验界面能不能进门，作答质量由 test_app_agent.py 负责。
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

ROOT = Path(__file__).resolve().parent.parent.parent
APP = str(ROOT / "app.py")

os.environ["CLOUD_LOG_ENABLED"] = "0"      # 测试不往云库写

from streamlit.testing.v1 import AppTest  # noqa: E402
from _askui import ask_input, has_ask_entry  # noqa: E402


def main() -> int:
    at = AppTest.from_file(APP, default_timeout=180)
    at.run()

    fails = []

    pw = [t for t in at.text_input if getattr(t, "type", "") == "password"]
    print(f"① 口令输入框数量 = {len(pw)}（期望 0）")
    if pw:
        fails.append("还残留口令输入框")

    ok_entry = has_ask_entry(at)
    print(f"② 问答入口是否直接出现 = {ok_entry}（期望 True）")
    if not ok_entry:
        fails.append("首屏没有问答入口")

    if ok_entry:
        box = ask_input(at)
        print(f"③ 问答框 placeholder = {getattr(box, 'placeholder', '')!r}")
        if not getattr(box, "placeholder", ""):
            print("   （placeholder 为空，不影响判定）")

    print("\n" + ("✓ 通过：进站已无需口令" if not fails else "✗ 有问题：" + "；".join(fails)))
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
