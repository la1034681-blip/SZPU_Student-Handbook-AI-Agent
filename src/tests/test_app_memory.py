# -*- coding: utf-8 -*-
"""无头验证：多轮记忆的页面接线（2026-10-15 定稿版）。

现在的行为（用户 2026-10-15 最终确认的口径）
--------------------------------------------
· 同一页里连着问  → 记得住，追问接得上
· 刷新页面 / 关掉标签页 → 记忆就没了（**不写服务器，所以天然如此**）
· 点「清空对话」  → 立刻清掉
· **没有**「闲置 60 分钟自动清」那套 —— 用户要求去掉；
  也不再把记忆存到云库（存了就得靠超时去猜着清，那正是被去掉的东西）

守住七件事
----------
① session_state 里有记忆对象，一进来是空的
② 「清空对话」把记忆一起清掉（不是只清气泡）
③ **一个字节都不出浏览器**：整页走一遍，save/load/clear_memory 一次都没被调
④ 刷新（新建 AppTest 实例＝新的浏览会话）＝ 一张白纸
⑤ 地址栏里不再有 `?m=`（会话号机制已随云库副本一起去掉）
⑥ 「系统自检」页看得到记忆状态，且**没有**「清掉闲置记忆」按钮、不再提「闲置 N 分钟」
⑦ 全程走完，仍然一个字节都没出浏览器

不碰网络（CLOUD_LOG_ENABLED=0）、不调模型，所以很快、也没副作用。
记忆的**内容**逻辑由 test_memory.py 单测；真实追问效果由 test_followup_e2e.py 验。

跑法：python src/tests/test_app_memory.py
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

# 必须在 import app 之前：关掉所有云库出口。记忆本来就不走云库，
# 这里再把出口关掉，等于「就算它偷偷想写也写不成」。
os.environ["ACCESS_CODE"] = "off"
os.environ["CLOUD_LOG_ENABLED"] = "0"

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / "src"))
import _paths  # noqa: E402 把 core/retrieval/infra/cli 也挂上搜索路径
sys.path.insert(0, str(ROOT / "src" / "tests"))

import memory as memmod                           # noqa: E402
from streamlit.testing.v1 import AppTest          # noqa: E402
from _askui import has_ask_entry                  # noqa: E402

_fail = 0


def ck(name: str, cond: bool, extra: str = ""):
    global _fail
    if not cond:
        _fail += 1
    print(f"  {'✓' if cond else '✗'} {name}" + (f"  —— {extra}" if extra and not cond else ""))


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
    # 一整页走下来，云库那三个函数一次都不该被叫到
    calls = []
    keep = {n: getattr(memmod, n)
            for n in ("save_memory", "load_memory", "clear_memory")}
    for n in keep:
        setattr(memmod, n, (lambda name: lambda *a, **k: calls.append(name))(n))
    try:
        at = AppTest.from_file(str(ROOT / "app.py"), default_timeout=300)
        at.session_state["authed"] = True
        at.run()

        print("① 一进页面就有记忆对象，且是空的")
        ck("页面没炸", not at.exception, str([e.value for e in at.exception])[:300])
        ck("已进入问答页", has_ask_entry(at))
        mem = state(at, "mem")
        ck("session_state 里有 mem", mem is not None)
        ck("刚进来时记忆是空的",
           mem is not None and getattr(mem, "turns", None) == []
           and not getattr(mem, "summary", ""))

        print("\n② 「清空对话」把记忆一起清掉")
        # 手动灌两轮进记忆 —— 只验「清空」这个动作，不必真调模型
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

        print("\n③ 一个字节都不出浏览器（云库那三个函数没被叫过）")
        ck("save / load / clear 都没被调用", calls == [], f"实际被调：{calls}")

        print("\n④ 刷新（新的浏览会话）＝一张白纸")
        at2 = AppTest.from_file(str(ROOT / "app.py"), default_timeout=300)
        at2.session_state["authed"] = True
        at2.run()
        ck("刷新后没炸", not at2.exception, str([e.value for e in at2.exception])[:300])
        ck("记忆是空的", state(at2, "mem").turns == [])

        print("\n⑤ 地址栏里不再有 ?m=")
        ck("没有 m 这个参数", not at2.query_params.get("m"),
           f"实际 {at2.query_params.get('m')!r}")

        print("\n⑥ 自检页已不上墙（2026-10-16），但记忆那节代码保留着")
        # 自检页入口撤掉后，原来的「点按钮看自检页」走不通了。改为两条更硬的断言：
        #   ① 页面上确实没有自检入口（不能让学生看到）
        #   ② 记忆那一节的代码原样在 render_diag 里（「保留代码」不是空话）
        ck("页面上没有自检入口",
           not any(b.key == "open_diag" for b in at2.button)
           and not any("系统自检" in (b.label or "") for b in at2.button))
        ck("不存在「清掉闲置记忆」按钮",
           not any(b.key == "diag_mem_sweep" for b in at2.button))

        import inspect
        import importlib.util as _ilu
        _spec = _ilu.spec_from_file_location("_app_mem_diag", str(ROOT / "app.py"))
        _m = _ilu.module_from_spec(_spec)
        _spec.loader.exec_module(_m)
        _src = inspect.getsource(_m.render_diag)
        ck("自检页里「会话记忆」一节保留着", "会话记忆" in _src)
        ck("「刷新/关掉就没了」的说法保留着",
           "刷新页面" in _src and "关掉标签页" in _src)
        ck("「不往服务器写」的说法保留着", "不往服务器写" in _src)
        ck("没有「闲置 N 分钟」那套说法", "闲置" not in _src, _src[:200])

        print("\n⑦ 整个流程走完，仍然一个字节都没出浏览器")
        ck("save / load / clear 依旧没被调用", calls == [], f"实际被调：{calls}")
    finally:
        for n, f in keep.items():
            setattr(memmod, n, f)

    print("\n" + ("全部通过 ✓" if _fail == 0 else f"有 {_fail} 项失败 ✗"))
    return 0 if _fail == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
