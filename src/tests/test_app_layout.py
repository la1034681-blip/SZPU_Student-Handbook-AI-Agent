# -*- coding: utf-8 -*-
"""版式回归：问答气泡必须落在「问答」区里，不能掉到留言墙下面。

起因（2026-10-15 用户反馈，两张截图）：
  · 点示例问题后，问句和答案出现在**页面最底部**（留言墙、页脚下面）；
  · 一进页面只看得见示例按钮，看不见输入框在哪。

两条都源于同一个结构问题：输入框用了 `st.chat_input`（Streamlit 把它固定在
视口最底部），而作答函数写在脚本末尾 —— Streamlit 自上而下执行脚本，气泡就
渲染在末尾。现在输入框与作答都收进「问答」区，本脚本就把这条位置约定钉住，
以后谁把代码挪回去都会立刻报错，而不是等学生看见。

判定方式：看主区域元素树里 `chat_message` 与「留言墙文本框 / 页脚」的先后顺序，
外加一条用户指定版式：**示例问题在上、输入框在下**（2026-10-15 原话
「在问答的示例下有个输入问问题的输入框」）。
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))     # 为了 import _askui

os.environ["ACCESS_CODE"] = "off"          # 跳过门禁，专测版式
os.environ["CLOUD_LOG_ENABLED"] = "0"

from streamlit.testing.v1 import AppTest      # noqa: E402
from _askui import ask, has_ask_entry, click_sample   # noqa: E402

APP = str(ROOT / "app.py")


def _main(at):
    return list(at.main)


def index_of_first(at, pred) -> int:
    for i, e in enumerate(at.main):
        if pred(e):
            return i
    return -1


def idx_chat(at) -> int:
    return index_of_first(at, lambda e: type(e).__name__ == "ChatMessage")


def idx_suggestion(at) -> int:
    return index_of_first(at, lambda e: type(e).__name__ == "TextArea")


def idx_foot(at) -> int:
    return index_of_first(at, lambda e: type(e).__name__ == "Markdown"
                         and "class=\"foot\"" in str(getattr(e, "value", "")))


def idx_ask_form(at) -> int:
    """问答区那个输入框控件的位置。

    ⚠ 别拿 Markdown 里的样式类名当判据（2026-10-15 踩过）：
    页面顶部的 CSS 样式块里也有这个类名，会第一个被匹配到，
    于是量出来「输入框在索引 1」——比示例按钮还靠前，纯属误报。
    """
    return index_of_first(at, lambda e: type(e).__name__ == "TextInput"
                         and getattr(e, "form_id", "") == "ask_form")


def idx_sample(at, keyword: str) -> int:
    return index_of_first(at, lambda e: type(e).__name__ == "Button"
                         and keyword in str(getattr(e, "label", "")))


fails: list = []


def check(label: str, ok: bool, extra: str = "") -> None:
    print(f"   {'✓' if ok else '✗'} {label} {extra}", flush=True)
    if not ok:
        fails.append(label)


def main():
    at = AppTest.from_file(APP, default_timeout=900)
    at.run()

    print("① 首屏：有输入框，且它在页面中部（不是被藏到最底）", flush=True)
    check("无异常", not at.exception, [str(e) for e in at.exception][:1])
    check("能看到问答输入框", has_ask_entry(at))
    f, s = idx_ask_form(at), idx_suggestion(at)
    check("输入框在留言墙之上", 0 <= f < s, f"ask_form={f} 留言墙={s}")
    print(f"   元素顺序: 输入框={f} 留言墙={s}", flush=True)

    # 用户 2026-10-15 明确要求：示例问题在**上**、输入框在**下**
    # （原话「在问答的示例下有个输入问问题的输入框」）
    s_btn = idx_sample(at, "补考")
    check("示例按钮在输入框之上", 0 <= s_btn < f, f"示例={s_btn} 输入框={f}")
    print(f"   示例按钮={s_btn} 输入框={f}", flush=True)

    print("\n② 点示例问题 → 气泡必须落在留言墙/页脚之上", flush=True)
    click_sample(at, "国家奖学金")
    check("无异常", not at.exception, [str(e) for e in at.exception][:1])
    c, s2, ft = idx_chat(at), idx_suggestion(at), idx_foot(at)
    check("出现了问答气泡", c >= 0, f"chat={c}")
    check("气泡在留言墙之上", 0 <= c < s2, f"chat={c} 留言墙={s2}")
    check("气泡在页脚之上", 0 <= c < ft, f"chat={c} 页脚={ft}")
    msgs = at.session_state["messages"]
    check("会话里已记下这一问一答",
          [m["role"] for m in msgs] == ["user", "assistant"],
          [m["role"] for m in msgs])
    check("问答区下方仍保留输入框（可继续追问）", has_ask_entry(at))

    print("\n③ 再手打一句 → 与示例同一个位置（仍在留言墙之上）", flush=True)
    ask(at, "学校食堂几点关门？")
    check("无异常", not at.exception, [str(e) for e in at.exception][:1])
    c2, s3 = idx_chat(at), idx_suggestion(at)
    check("两条用户气泡都在留言墙之上", 0 <= c2 < s3, f"chat={c2} 留言墙={s3}")
    roles = [m["role"] for m in at.session_state["messages"]]
    check("四轮消息齐全", roles == ["user", "assistant"] * 2, roles)

    print("\n④ 清空对话后回到空态", flush=True)
    for b in at.button:
        if b.label.strip() == "清空对话":
            b.click().run()
            break
    check("无异常", not at.exception, [str(e) for e in at.exception][:1])
    check("消息已清空", at.session_state["messages"] == [])
    check("示例按钮回来了", any("补考" in b.label for b in at.button))
    check("输入框还在", has_ask_entry(at))

    print("\n" + ("全部通过" if not fails else f"失败 {len(fails)} 项：{fails}"),
          flush=True)
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())