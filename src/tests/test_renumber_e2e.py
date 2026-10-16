# -*- coding: utf-8 -*-
"""端到端确认：页面上角标必须连号 E1、E2、E3…（用户截图那个 case）。

截图里问「被处分怎么办」，答复中出现的是 [E2]、[E6]、[E21] —— 跳号。
现在应按「答复里首次出现的先后」重编成 E1、E2、E3，卡片同步换号。
走真页面（AppTest），会调大模型，一次就好。

跑法：`python src/tests/test_renumber_e2e.py`
"""
from __future__ import annotations

import os
import re
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")
os.environ.setdefault("ACCESS_CODE", "off")
os.environ.setdefault("CLOUD_LOG_ENABLED", "0")

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / "src"))
import _paths  # noqa: E402 把 core/retrieval/infra/cli 也挂上搜索路径
sys.path.insert(0, str(ROOT / "src" / "tests"))

from streamlit.testing.v1 import AppTest          # noqa: E402
from _askui import ask                            # noqa: E402

at = AppTest.from_file(str(ROOT / "app.py"), default_timeout=300)
at.session_state["authed"] = True
at.run()
if at.exception:
    print("✗ 初始渲染就炸了：", [e.value for e in at.exception])
    sys.exit(1)

ask(at, "被处分怎么办")
at.run()
if at.exception:
    print("✗ 提问后异常：", [e.value for e in at.exception])
    sys.exit(1)

md = "\n".join(m.value for m in at.markdown)
badges = re.findall(r'href="#ev-(E\d+)"', md)
cards = re.findall(r'<details class="ev" id="ev-(E\d+)"', md)
nums = sorted(int(e[1:]) for e in badges)

print("答复里的角标（点击锚点）：", badges)
print("手册原文卡片顺序      ：", cards)
uniq = list(dict.fromkeys(badges))          # 按首次出现先后去重
print("角标编号去重升序      ：", sorted(set(nums)))
print("是否连号 E1..Ek       ：", sorted(set(nums)) == list(range(1, len(set(nums)) + 1)))
print("卡片是否按号升序      ：",
      [int(e[1:]) for e in cards] == sorted(int(e[1:]) for e in cards))
# 同一编号可以出现多次（同一条款被多句引用），判「跳号」要去重后看首次出现顺序
print("首次出现顺序          ：", uniq)
print("答复里是否还有跳号残留：", uniq != [f"E{i}" for i in range(1, len(uniq) + 1)])

ok = (bool(badges)
      and uniq == [f"E{i}" for i in range(1, len(uniq) + 1)]
      and [int(e[1:]) for e in cards] == sorted(int(e[1:]) for e in cards))
print()
print("✓ 通过：角标已连号" if ok else "✗ 未通过，见上面各项")
sys.exit(0 if ok else 1)
