# -*- coding: utf-8 -*-
"""无头验证 app.py 的"程序化抽取式"通道（不需要大模型）。

覆盖：
  1. 页面初始渲染无异常
  2. 点示例按钮 → 真的作答，答案含「手册原文」与引用角标
  3. 问手册没有的事 → 友好拒答，且不暴露"相关度/阈值"这类技术词
总耗时约 2~3 分钟（含检索模型加载）。
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

ROOT = Path(__file__).resolve().parent.parent.parent
APP = str(ROOT / "app.py")

from streamlit.testing.v1 import AppTest  # noqa: E402


def md_all(at) -> str:
    return "\n".join(m.value for m in at.markdown)


def main():
    at = AppTest.from_file(APP, default_timeout=600)
    at.run()
    print("== 初始渲染异常:", bool(at.exception), flush=True)
    for e in at.exception:
        print("   EXC:", e, flush=True)

    labels = [b.label for b in at.button]
    print("   按钮:", labels, flush=True)

    btn = next((b for b in at.button if "补考" in b.label), None)
    if btn is None:
        print("!! 没找到示例按钮，退出", flush=True)
        return
    print(f"== 点击示例按钮：{btn.label}", flush=True)
    btn.click().run()
    print("   作答后异常:", bool(at.exception), flush=True)
    for e in at.exception:
        print("   EXC:", e, flush=True)

    h = md_all(at)
    for k in ("手册原文", "以上为手册原文摘录", "cite-badge", "可以补考"):
        print(f"   {'OK  ' if k in h else 'MISS'} {k}", flush=True)
    print("   答案片段:", flush=True)
    for line in h.splitlines():
        if "以上为手册原文摘录" in line or line.startswith("可以补考"):
            print("     |", line[:80], flush=True)

    print("== 拒答测试：学校食堂几点关门？", flush=True)
    at.chat_input[0].set_value("学校食堂几点关门？").run()
    print("   异常:", bool(at.exception), flush=True)
    for e in at.exception:
        print("   EXC:", e, flush=True)
    warns = "\n".join(w.value for w in at.warning)
    print("   含「手册里暂无相关内容」:", "手册里暂无相关内容" in warns, flush=True)
    print("   拒答提示原文:", warns.strip().replace("\n", " ")[:90], flush=True)
    print("   主答案区泄技术词「相关度」:",
          any("相关度" in m.value for m in at.markdown[-3:]), flush=True)


main()
