# -*- coding: utf-8 -*-
"""无头验证：证据角标是否渲染为可点击锚点 + 证据卡片是否带 id。"""
from __future__ import annotations

import os

# 跳过访问门禁：本脚本只验内容/角标/过程面板，不测口令
# （fail-safe 之后 ACCESS_CODE 为空会被挡住，测试必须显式声明「有意关」）
os.environ["ACCESS_CODE"] = "off"
os.environ["CLOUD_LOG_ENABLED"] = "0"

import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

from streamlit.testing.v1 import AppTest  # noqa: E402
from _askui import ask                       # noqa: E402

APP = str(Path(__file__).resolve().parent.parent.parent / "app.py")


def main():
    at = AppTest.from_file(APP, default_timeout=300)
    at.run()
    print("初始异常:", bool(at.exception), flush=True)
    ask(at, "考试作弊被查到后会怎么处理？")
    print("提问后异常:", bool(at.exception), flush=True)
    for e in at.exception:
        print("EXC:", e, flush=True)

    md = "\n".join(m.value for m in at.markdown)
    n_badge = md.count('class="cite-badge" href="#ev-')
    n_card = md.count('<details class="ev" id="ev-')
    print(f"可点击角标数: {n_badge}", flush=True)
    print(f"证据卡片数  : {n_card}", flush=True)
    print("角标 href 是否指向存在的卡片 id:",
          all(f'id="ev-{e}"' in md for e in
              __import__("re").findall(r'href="#ev-(E\d+)"', md)), flush=True)
    print("未替换的裸角标（应只剩有效范围内）:", flush=True)
    import re
    valid = set(re.findall(r'href="#ev-(E\d+)"', md))
    print("  valid eids:", sorted(valid), flush=True)

    # 拒答分支：独立实例，不应有任何角标
    at2 = AppTest.from_file(APP, default_timeout=300)
    at2.run()
    ask(at2, "学校食堂几点关门？")
    md2 = "\n".join(m.value for m in at2.markdown)
    print("拒答分支角标数（应为0）:", md2.count('class="cite-badge"'), flush=True)
    print("拒答分支证据卡片数（应为0）:", md2.count('<details class="ev"'), flush=True)
    print("拒答分支异常:", bool(at2.exception), flush=True)


if __name__ == "__main__":
    main()
