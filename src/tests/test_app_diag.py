# -*- coding: utf-8 -*-
"""无头验证「配置缺失告警」与「系统自检」页（2026-10-15 新增）。

起因 —— 线上部署后同时冒出来两个看着毫不相干的症状：
  ① 提问报「缺少 EMBED_API_KEY」 ② 留言报「没提交上，可能网络不通」
根因其实是同一个：Streamlit 后台 Secrets 没生效，而 `load_secrets()` 当时把异常
**静默吞掉**了，于是配置全空却照常运行 —— 错报在用到的那一刻才零散地冒出来。
这个脚本守住两件事：
  1. 配置缺了就显眼地说出来，并点名缺了哪几项（不再靠猜）
  2. 有个地方能把「哪项没配、链路通不通」一次摊开看（页脚「系统自检」）
不触碰真实网络：连通性按钮不点就不会发任何请求，所以跑得很快、也无副作用。

跑法：python src/tests/test_app_diag.py
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

ROOT = Path(__file__).resolve().parent.parent.parent
APP = str(ROOT / "app.py")

# 必须在 import app 之前：load_env_file 只填「尚不存在」的变量，不会被 .env 覆盖
os.environ["ACCESS_CODE"] = "SZPU666"
os.environ["CLOUD_LOG_ENABLED"] = "0"      # 测试不往云库写，免得污染日志

from streamlit.testing.v1 import AppTest  # noqa: E402


def md_all(at) -> str:
    return "\n".join(m.value for m in at.markdown)


def warn_all(at) -> str:
    return "\n".join(w.value for w in at.warning)


def show_exc(at, tag: str) -> None:
    print(f"   [{tag}] 异常={bool(at.exception)}", flush=True)
    for e in at.exception:
        print("      EXC:", e, flush=True)


def click(at, key: str) -> bool:
    """按 key 找一个按钮点下去（AppTest 的 button 索引不稳，按 key 最可靠）。"""
    for b in at.button:
        if b.key == key:
            b.click()
            return True
    return False


def main():
    ok = True
    at = AppTest.from_file(APP, default_timeout=900)
    at.run()

    print("① 先用口令进门", flush=True)
    at.text_input[0].set_value("SZPU666")
    at.button[0].click().run()             # 表单要真的点提交，只 set_value 不生效
    show_exc(at, "进门")
    entered = len(at.chat_input) > 0
    print("   已进入主界面:", entered, flush=True)
    ok &= entered

    print("\n② 配置齐全时不该打扰用户", flush=True)
    w = warn_all(at)
    quiet = "后台配置不完整" not in w
    print("   页面上没有「后台配置不完整」告警:", quiet, flush=True)
    ok &= quiet

    print("\n③ 人为掏空三项配置 → 应显眼报出来并点名", flush=True)
    # 注意：空字符串也算「变量已存在」，所以 .env 不会再把它们填回来
    os.environ["EMBED_API_KEY"] = ""
    os.environ["CLOUD_ENDPOINT"] = ""
    os.environ["CLOUD_PUBLISHABLE_KEY"] = ""
    at.run()
    show_exc(at, "缺配置")
    w = warn_all(at)
    named = all(k in w for k in ("EMBED_API_KEY", "CLOUD_ENDPOINT", "CLOUD_PUBLISHABLE_KEY"))
    alerted = "后台配置不完整" in w
    print("   出现告警:", alerted, flush=True)
    print("   点到了缺的那三项:", named, flush=True)
    print("   告警原文:", (w[:120] + "…") if len(w) > 120 else w or "（无）", flush=True)
    ok &= alerted and named

    print("\n④ 打开页脚「系统自检」页", flush=True)
    hit = click(at, "open_diag")
    at.run()
    show_exc(at, "自检页")
    body = md_all(at)
    caps = "\n".join(c.value for c in at.caption)
    has_title = "系统自检" in body
    has_secrets_note = "Secrets 读取情况" in body
    listed = all(k in body for k in ("ACCESS_CODE", "ADMIN_CODE", "EMBED_API_KEY",
                                     "CLOUD_ENDPOINT", "CLOUD_PUBLISHABLE_KEY",
                                     "FREE_LLM_API_KEY"))
    counted = "关键配置" in body
    print("   点到了按钮:", hit, flush=True)
    print("   页面标题「系统自检」:", has_title, flush=True)
    print("   有「Secrets 读取情况」一行:", has_secrets_note, flush=True)
    print("   六项配置逐条列出:", listed, flush=True)
    print("   有「N / 6 项已就位」计数:", counted, flush=True)
    print("   说明文字里提到免费档网络风险:", "网络问题" in caps or "境外" in caps, flush=True)
    ok &= hit and has_title and has_secrets_note and listed and counted

    print("\n⑤ 自检页不该自己发网络请求（按钮没点）", flush=True)
    no_net = "云库不通" not in body and "在线编码：" not in body
    print("   未见任何连通性结果:", no_net, flush=True)
    ok &= no_net

    # 复原，免得影响同进程里后续可能的操作
    os.environ["EMBED_API_KEY"] = "sk-tljrrknwcqklummbyvitxjyiijvfkhmdwfpetdeaujgbxpjc"
    os.environ["CLOUD_ENDPOINT"] = "https://szpu-handbook-qa.app.workbuddy.host"
    os.environ["CLOUD_PUBLISHABLE_KEY"] = "wbpk_5LyhZ0F5RWsUb899resz7e_VIoZzX64rPlPlexNlB5dPbg4fHrvRzsP"

    print("\n" + ("全部通过" if ok else "有失败项，见上"), flush=True)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
