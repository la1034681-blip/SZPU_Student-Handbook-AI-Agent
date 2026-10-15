# -*- coding: utf-8 -*-
"""测试公用小工具：定位问答区输入框并提交问题。

为什么需要它（2026-10-15）：页面的提问入口从 `st.chat_input`（天生贴在视口
最底部，学生一进来看不见）换成了问答区里的表单 `ask_form`。于是所有拿
`at.chat_input[0]` 当「问答入口」探针的测试都失效了。

这里提供两个统一动作，测试脚本只管调：
    has_ask_entry(at)  页面是否出现了问答输入框（门禁探针）
    ask(at, "问题…")    在输入框里打字并点「提 问」，返回跑完的 AppTest

判定依据是 `form_id == "ask_form"` 而不是下标 —— 门禁页也有 text_input，
按下标猜迟早认错页。
"""
from __future__ import annotations

ASK_FORM = "ask_form"
ASK_BTN = "提 问"


def ask_input(at):
    """返回问答区的那个输入框控件；找不到返回 None。"""
    for t in at.text_input:
        if getattr(t, "form_id", "") == ASK_FORM:
            return t
    return None


def ask_btn(at):
    for b in at.button:
        if b.label.strip() == ASK_BTN:
            return b
    return None


def has_ask_entry(at) -> bool:
    """页面上有没有问答输入框 —— 用来判断「是否已通过口令进门」。"""
    return ask_input(at) is not None


def ask(at, text: str):
    """在输入框里打字并提交，随后返回跑完一遍的 AppTest。"""
    ti, btn = ask_input(at), ask_btn(at)
    if ti is None:
        raise RuntimeError(
            "没找到问答输入框（form_id=ask_form）。"
            f"当前 text_input 的 form_id={[getattr(t,'form_id','') for t in at.text_input]}")
    if btn is None:
        raise RuntimeError("没找到「提 问」按钮")
    ti.set_value(text)
    btn.click().run()
    return at


def click_sample(at, keyword: str):
    """点一个示例问题按钮（按关键字匹配）。"""
    for b in at.button:
        if keyword in b.label:
            b.click().run()
            return at
    raise RuntimeError(f"没找到含「{keyword}」的示例按钮："
                       f"{[b.label for b in at.button]}")