# -*- coding: utf-8 -*-
"""验证两个修复：示例按钮能作答、拒答话术讲人话。"""
import pathlib
import sys

sys.stdout.reconfigure(encoding="utf-8")
ROOT = pathlib.Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / "src"))

from streamlit.testing.v1 import AppTest  # noqa: E402


def pick_button(at, kw):
    for i, b in enumerate(at.button):
        if kw in (b.label or ""):
            return i
    return None


def case(name, keyword, expect_answer: bool):
    print("=" * 70, flush=True)
    print(f"[{name}] 点击示例按钮含「{keyword}」", flush=True)
    at = AppTest.from_file(str(ROOT / "app.py"), default_timeout=600)
    at.run()
    print("  初始异常:", bool(at.exception), flush=True)
    print("  按钮标签:", [b.label for b in at.button], flush=True)

    idx = pick_button(at, keyword)
    if idx is None:
        print("  !! 没找到按钮", flush=True)
        return
    at.button[idx].click().run()
    print("  点击后异常:", bool(at.exception), flush=True)
    for e in at.exception:
        print("  EXC:", e, flush=True)

    warns = [w.value for w in at.warning]
    mds = [m.value for m in at.markdown]
    alltext = "\n".join(warns + mds)
    print("  chat_message 数(应>=2):", len(at.chat_message), flush=True)
    print("  warning:", [w[:70] for w in warns], flush=True)
    if expect_answer:
        print("  是否给出了答案(非拒答):", bool(mds) and not warns, flush=True)
    else:
        print("  含『本手册中没有对应内容』:", "本手册中没有对应内容" in alltext, flush=True)
        print("  是否泄露『阈值/相关分』:", ("阈值" in alltext) or ("相关分" in alltext), flush=True)


if __name__ == "__main__":
    case("示例按钮·正常作答", "国家奖学金", expect_answer=True)
    case("示例按钮·应拒答", "食堂", expect_answer=False)
    print("ALL DONE", flush=True)
