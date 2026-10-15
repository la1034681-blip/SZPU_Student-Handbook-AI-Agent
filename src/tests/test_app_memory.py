# -*- coding: utf-8 -*-
"""无头验证：多轮记忆接进页面后，该出现的东西都出现了（2026-10-15 新增）。

守住四件事
----------
① 会话号进了地址栏 ?m=      —— 刷新后凭它从云库把上文读回来
② session_state 里有记忆对象 —— 页面记得住「这次会话聊过什么」
③ 「清空对话」把记忆一起清掉 —— 按下去就是「忘掉这些」，不能只清气泡
④ 「系统自检」页能看到记忆状态与手动清理入口

不碰网络（CLOUD_LOG_ENABLED=0）、不调模型，所以很快、也没副作用。
记忆的**内容**逻辑由 test_memory.py 单测；真实追问效果由 test_followup_e2e.py 验。

跑法：python src/tests/test_app_memory.py
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

# 必须在 import app 之前：关掉所有云库出口（记忆的云库副本也随之关掉）。
# 地址栏 ?m= 与页面内的记忆对象都在本地，照样能测。
os.environ["ACCESS_CODE"] = "off"
os.environ["CLOUD_LOG_ENABLED"] = "0"

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "src" / "tests"))

from streamlit.testing.v1 import AppTest          # noqa: E402
from _askui import has_ask_entry                  # noqa: E402

_fail = 0


def ck(name: str, cond: bool, extra: str = ""):
    global _fail
    if not cond:
        _fail += 1
    print(f"  {'✓' if cond else '✗'} {name}" + (f"  —— {extra}" if extra and not cond else ""))


def qp(at, key: str) -> str:
    """读地址栏参数。AppTest 的 query_params 值可能是列表，统一取第一个。"""
    try:
        raw = at.query_params.get(key)
    except Exception:                                     # noqa: BLE001
        return ""
    if isinstance(raw, (list, tuple)):
        raw = raw[0] if raw else ""
    return (raw or "").strip()


def state(at, key: str):
    return at.session_state.filtered_state.get(key)


def click(at, key: str) -> bool:
    for b in at.button:
        if b.key == key:
            b.click()
            return True
    return False


def md_all(at) -> str:
    return "\n".join(m.value for m in at.markdown)


def cap_all(at) -> str:
    return "\n".join(c.value for c in at.caption)


def main() -> int:
    at = AppTest.from_file(str(ROOT / "app.py"), default_timeout=300)
    at.session_state["authed"] = True
    at.run()

    print("① 一进页面就有会话号，并写进地址栏")
    ck("页面没炸", not at.exception, str([e.value for e in at.exception])[:300])
    ck("已进入问答页", has_ask_entry(at))
    mid = qp(at, "m")
    ck("地址栏出现 ?m=", bool(mid), f"实际 {mid!r}")
    ck("会话号是 32 位十六进制",
       len(mid) == 32 and all(c in "0123456789abcdef" for c in mid), mid)

    print("\n② 页面里存着记忆对象")
    mem = state(at, "mem")
    ck("session_state 里有 mem", mem is not None)
    ck("刚进来时记忆是空的",
       mem is not None and getattr(mem, "turns", None) == [] and not getattr(mem, "summary", ""))

    print("\n③ 「清空对话」把记忆一起清掉")
    # 手动灌两轮进记忆 —— 只是为了验「清空」这个动作，不必真调模型
    mem.add_user("旷课多少学时会被处分？")
    mem.add_assistant("10-19 学时警告，50 学时以上开除学籍。")
    mem.add_user("那迟到早退算不算旷课？")
    mem.add_assistant("迟到早退按旷课计。")
    ck("灌进去 4 条", len(mem.turns) == 4)
    at.run()                                   # 重画：让页面用上这份记忆
    ck("重画后记忆还在", len(state(at, "mem").turns) == 4)
    hit = click(at, "clear_chat")
    at.run()
    ck("点到了「清空对话」", hit)
    ck("对话消息清空", state(at, "messages") == [])
    ck("记忆也清空（不是只清气泡）", state(at, "mem").turns == [],
       f"实际还剩 {len(state(at, 'mem').turns)} 条")
    ck("摘要也清空", state(at, "mem").summary == "")

    print("\n④ 「系统自检」页看得到记忆状态")
    click(at, "open_diag")
    at.run()
    body, caps = md_all(at), cap_all(at)
    ck("自检页没炸", not at.exception, str([e.value for e in at.exception])[:300])
    ck("有「会话记忆」一节", "会话记忆" in body)
    ck("说清了是临时记忆、不做持久保存", "临时" in body and "持久" in body)
    ck("有手动清理按钮", any(b.key == "diag_mem_sweep" for b in at.button))
    ck("提到闲置多久会清掉", "闲置" in caps and "分钟" in caps)

    print("\n" + ("全部通过 ✓" if _fail == 0 else f"有 {_fail} 项失败 ✗"))
    return 0 if _fail == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
