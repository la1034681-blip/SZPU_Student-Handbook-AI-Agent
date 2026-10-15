# -*- coding: utf-8 -*-
"""端到端确认：页面上点「旷课」示例，答案应正常给出（不再误拒）。

对应 2026-10-15 用户截图那个 case。走真页面（AppTest），会调大模型，
所以一次就好 —— 免费档连着调容易撞 429。

跑法：`python src/tests/test_kuangke_e2e.py`
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")
os.environ.setdefault("ACCESS_CODE", "off")      # 跳过门禁，直取问答页

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "src" / "tests"))

from streamlit.testing.v1 import AppTest          # noqa: E402
from _askui import click_sample                   # noqa: E402

at = AppTest.from_file(str(ROOT / "app.py"), default_timeout=300)
at.session_state["authed"] = True

at.run()
if at.exception:
    print("✗ 初始渲染就炸了：", [e.value for e in at.exception])
    sys.exit(1)

print("点击示例：旷课多少学时会被处分……")
click_sample(at, "旷课")
at.run()

if at.exception:
    print("✗ 提问后异常：", [e.value for e in at.exception])
    sys.exit(1)

texts = [m.value for m in at.markdown]
joined = "\n".join(texts)
msgs = list(at.chat_message)

print("对话消息数：", len(msgs))
print("出现拒答话术「暂时答不了」：", "暂时答不了" in joined)
print("出现「本手册中没有对应内容」：", "本手册中没有对应内容" in joined)
print("出现程序化兜底提示：", "程序化兜底" in joined)
print("答案里含可点击角标 cite-badge：", "cite-badge" in joined)
print("答案里含「旷课」：", "旷课" in joined)
print("答案里含处分档次（留校察看/开除学籍）：",
      ("留校察看" in joined) or ("开除学籍" in joined))
caps = " ".join(c.value for c in at.caption)
print("内部过程面板记了「角标回填」：", "角标回填" in caps)
print("（回填只在模型忘打角标时出现，不是每次都有 —— 上一行仅供参考）")

ok = (not at.exception) and len(msgs) >= 2 and "暂时答不了" not in joined \
     and "cite-badge" in joined
print()
print("✓ 通过：这题在页面上已正常作答" if ok else "✗ 仍未通过，见上面各项")
sys.exit(0 if ok else 1)
