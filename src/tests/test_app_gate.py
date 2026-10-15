# -*- coding: utf-8 -*-
"""无头验证访问保护：口令门禁 + 进门之后仍能正常作答。

覆盖：
  1. 未输入口令 → 只出现登录页，**没有**问答输入框（ask_form 表单不存在）
  2. 口令输错 → 报错、仍停登录页
  3. 口令输对 → 进入主界面，问答入口出现
  4. 正常提问能作答（真实调用模型）
  5. 紧接着再问一句 → 依旧作答，不会被拦
     （提问限流已于 2026-10-15 临时下线，本脚本随之去掉限流用例；
       限流版脚本见 .trash_tmp/限流下线-20261015/test_app_gate.py.bak）
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

ROOT = Path(__file__).resolve().parent.parent.parent
APP = str(ROOT / "app.py")

# 必须在 import app 之前设好：load_env_file 只填「尚未存在」的变量，不会被 .env 覆盖
os.environ["ACCESS_CODE"] = "SZPU666"
os.environ["CLOUD_LOG_ENABLED"] = "0"      # 测试不往云库写，免得污染日志

from streamlit.testing.v1 import AppTest  # noqa: E402
from _askui import ask, has_ask_entry      # noqa: E402


def md_all(at) -> str:
    return "\n".join(m.value for m in at.markdown)


def show_exc(at, tag: str) -> None:
    print(f"   [{tag}] 异常={bool(at.exception)}", flush=True)
    for e in at.exception:
        print("      EXC:", e, flush=True)


def main():
    at = AppTest.from_file(APP, default_timeout=900)
    at.run()

    print("① 初始页面（应停在门禁）", flush=True)
    show_exc(at, "初始")
    print("   有口令输入框:", len(at.text_input) > 0, flush=True)
    print("   有问答输入框:", has_ask_entry(at),
          "（应为 False —— 没进门就不给问答入口）", flush=True)
    print("   按钮:", [b.label for b in at.button], flush=True)
    print("   含「访问口令」字样:", "访问口令" in md_all(at), flush=True)

    print("\n② 输错口令", flush=True)
    at.text_input[0].set_value("SZPU123")
    at.button[0].click().run()
    show_exc(at, "错误口令")
    print("   error:", [e.value[:60] for e in at.error], flush=True)
    print("   仍在门禁页:", not has_ask_entry(at), flush=True)

    print("\n③ 输入正确口令 SZPU666", flush=True)
    at.text_input[0].set_value("SZPU666")
    at.button[0].click().run()
    show_exc(at, "正确口令")
    print("   已进入主界面（有问答入口）:", has_ask_entry(at), flush=True)
    print("   页面上出现「校园文化」:", "校园文化" in md_all(at), flush=True)

    print("\n④ 正常提问一次（真实调用模型）", flush=True)
    btn = next((b for b in at.button if "国奖" in b.label or "奖学金" in b.label), None)
    if btn is None:
        print("   !! 没找到示例按钮，改用输入框", flush=True)
        ask(at, "国家奖学金一年多少钱？")
    else:
        btn.click().run()
    show_exc(at, "作答")
    h = md_all(at)
    print("   答案里含「依据」:", "依据" in h or "手册原文" in h, flush=True)
    print("   页面出现 warning 数:", len(at.warning), flush=True)

    print("\n⑤ 紧接着再问一句（限流已下线，应照常作答）", flush=True)
    ask(at, "旷课多少学时会被处分？")
    show_exc(at, "连续提问")
    msgs = at.session_state["messages"]
    roles = [m["role"] for m in msgs]
    print("   会话消息序列:", roles, "（应为 user/assistant/user/assistant —— 第二问没被丢）",
          flush=True)
    hit_rate = [w.value[:60].replace("\n", " ") for w in at.warning
                if ("问得有点快" in w.value or "额度" in w.value
                    or "人有点多" in w.value)]
    print("   限流类提示条数:", len(hit_rate), "（应为 0）", hit_rate, flush=True)
    print("   页面已无「再问一次」按钮:",
          all("再问一次" not in b.label for b in at.button), flush=True)

    print("\n⑥ 口令没配（留空）→ 必须挡住，不能门户大开", flush=True)
    # 这条是 2026-10-15 线上事故的直接教训：Secrets 没生效 → ACCESS_CODE 为空 →
    # 旧逻辑视为「不设门禁」，学生直接就能用，页面上还看不出任何异常。
    os.environ["ACCESS_CODE"] = ""
    at.run()
    show_exc(at, "口令为空")
    errs = " ".join(e.value for e in at.error)
    print("   出现「还没配访问口令」提示:", "还没配访问口令" in errs, flush=True)
    print("   没有问答入口:", not has_ask_entry(at), flush=True)
    print("   没放行（仍在门禁）:", (not has_ask_entry(at)) and "还没配访问口令" in errs,
          flush=True)

    print("\n⑦ 口令显式设成 off → 这才是有意关闭门禁", flush=True)
    os.environ["ACCESS_CODE"] = "off"
    at.run()
    show_exc(at, "口令 off")
    print("   放行（有问答入口）:", has_ask_entry(at), flush=True)
    os.environ["ACCESS_CODE"] = "SZPU666"          # 复原，别影响别的脚本


main()
