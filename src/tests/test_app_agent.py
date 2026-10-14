# -*- coding: utf-8 -*-
"""无头验证 app.py 在 QA_MODE=agent（智能体链路）下能正常跑。

覆盖：
  1. 初始渲染无异常
  2. 点示例按钮 → 真的作答（答案非空、带引用角标）
  3. 「智能体内部过程」面板出现（规划 / 拆解 / 自检信息）
  4. 问手册里没有的事 → 友好拒答，且不把「相关度」「阈值」这类技术词
     暴露在主答案区
本脚本会真实调用大模型，一题约 5~10 秒，总耗时 2~3 分钟。
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
    at = AppTest.from_file(APP, default_timeout=900)
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
    for k in ("手册原文", "智能体", "依据", "cite-badge"):
        print(f"   {'OK  ' if k in h else '----'} 含「{k}」", flush=True)
    print("   答案片段:", flush=True)
    for line in h.splitlines():
        s = line.strip()
        if len(s) > 8 and ("可以" in s or "不予" in s or "规定" in s):
            print("     |", s[:88], flush=True)
            break

    print("== 拒答测试：学校食堂几点关门？", flush=True)
    at.chat_input[0].set_value("学校食堂几点关门？").run()
    print("   异常:", bool(at.exception), flush=True)
    for e in at.exception:
        print("   EXC:", e, flush=True)
    print("   warning 元素数:", len(at.warning), flush=True)
    for i, w in enumerate(at.warning):
        print(f"   [{i}]", w.value[:160].replace("\n", " "), flush=True)
    warns = "\n".join(w.value for w in at.warning)
    print("   含「手册里暂无相关内容」:", "手册里暂无相关内容" in warns, flush=True)
    print("   拒答提示:", warns.strip().replace("\n", " ")[:90], flush=True)
    print("   主答案区泄技术词「相关度」:",
          any("相关度" in m.value for m in at.markdown[-3:]), flush=True)


main()
