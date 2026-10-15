# -*- coding: utf-8 -*-
"""作答通道的「不许静默」测试。

起因（2026-10-15 用户提问）：点示例后答案是**程序化直接作答**（「结论成色：
可以（肯定）」），没调大模型 —— 他以为是智能体链路坏了。

两件事都要钉住：
 ① 通道默认值不能再是「悄悄退回程序化」。原来 `get_pipeline()` 写的是
    `os.environ.get("QA_MODE") or "procedural"`：配置一没读到（.env 丢了、
    线上 Secrets 漏了这行），程序化通道照答不误、答得也挺像样，页面上
    毫无异常 —— 这正是本项目最忌讳的「静默降级」。现在默认是主线 agent，
    且页面明说当前通道。
 ② 通道必须**写在页面上**：学生/管理员一眼能看出这一答有没有经过大模型。

用 AppTest 走真实渲染，顺带 import app 模块直接单测 `_qa_mode()` 的语义。
"""
from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))     # 为了 import _askui

os.environ["ACCESS_CODE"] = "off"          # 跳过门禁，本脚本只测通道
os.environ["CLOUD_LOG_ENABLED"] = "0"

from streamlit.testing.v1 import AppTest      # noqa: E402

APP = str(ROOT / "app.py")
fails: list = []


def check(label: str, ok: bool, extra: str = "") -> None:
    print(f"   {'✓' if ok else '✗'} {label} {extra}", flush=True)
    if not ok:
        fails.append(label)


def load_appmod():
    """把 app.py 当模块加载（不跑 main()），只为了单测纯函数。"""
    spec = importlib.util.spec_from_file_location("appmod_for_test", APP)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def md_all(at) -> str:
    return "\n".join(m.value for m in at.markdown)


def caps_all(at) -> str:
    return "\n".join(c.value for c in at.caption)


def main():
    mod = load_appmod()

    print("① _qa_mode() 的语义：没配 → 走主线（agent）且标记为「猜的」", flush=True)
    saved = os.environ.get("QA_MODE")
    try:
        os.environ.pop("QA_MODE", None)
        m, guessed = mod._qa_mode()
        check("未配置时 → agent", m == "agent", f"得到 {m!r}")
        check("未配置时 → 标记为猜的（页面会提示）", guessed is True)

        os.environ["QA_MODE"] = "agent"
        check("显式 agent → (agent, False)", mod._qa_mode() == ("agent", False),
              str(mod._qa_mode()))

        os.environ["QA_MODE"] = "PROCEDURAL "          # 大小写/空格都该认
        check("显式 procedural → (procedural, False)",
              mod._qa_mode() == ("procedural", False), str(mod._qa_mode()))

        os.environ["QA_MODE"] = "garbage"              # 写错了 → 别静默照用
        m2, g2 = mod._qa_mode()
        check("写错的值 → 回落主线并标记", m2 == "agent" and g2 is True,
              f"得到 {(m2, g2)}")
    finally:
        if saved is None:
            os.environ.pop("QA_MODE", None)
        else:
            os.environ["QA_MODE"] = saved

    print("\n② 页面必须明说当前通道", flush=True)
    at = AppTest.from_file(APP, default_timeout=900)
    at.run()
    check("无异常", not at.exception, [str(e) for e in at.exception][:1])
    caps = caps_all(at)
    check("页面上出现「作答通道：」", "作答通道：" in caps, caps[:120].replace("\n", " "))
    check("写明是智能体链路", "智能体链路" in caps)
    check("写明大脑是哪家模型", "GLM-4-Flash" in caps or "DeepSeek" in caps)

    print("\n③ 切到 procedural → 页面如实改成「程序化直接作答」", flush=True)
    os.environ["QA_MODE"] = "procedural"
    at2 = AppTest.from_file(APP, default_timeout=900)
    at2.run()
    check("无异常", not at2.exception, [str(e) for e in at2.exception][:1])
    caps2 = caps_all(at2)
    check("出现「程序化直接作答（不调大模型）」",
          "程序化直接作答（不调大模型）" in caps2,
          caps2[:120].replace("\n", " "))
    os.environ["QA_MODE"] = "agent"

    print("\n④ 系统自检页也要报通道", flush=True)
    at3 = AppTest.from_file(APP, default_timeout=900)
    at3.run()
    for b in at3.button:
        if b.label.strip() == "系统自检":
            b.click().run()
            break
    check("无异常", not at3.exception, [str(e) for e in at3.exception][:1])
    h = md_all(at3)
    check("自检页出现「作答通道」", "作答通道" in h)
    check("自检页写明链路名", "智能体链路" in h or "程序化" in h)

    print("\n" + ("全部通过" if not fails else f"失败 {len(fails)} 项：{fails}"),
          flush=True)
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())