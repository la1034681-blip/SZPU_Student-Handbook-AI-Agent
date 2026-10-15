# -*- coding: utf-8 -*-
"""无头验证「配置缺失告警」与「系统自检」页（2026-10-15 新增，2026-10-16 改）。

起因 —— 线上部署后同时冒出来两个看着毫不相干的症状：
  ① 提问报「缺少 EMBED_API_KEY」 ② 留言报「没提交上，可能网络不通」
根因其实是同一个：Streamlit 后台 Secrets 没生效，而 `load_secrets()` 当时把异常
**静默吞掉**了，于是配置全空却照常运行 —— 错报在用到的那一刻才零散地冒出来。

2026-10-16 用户要求：「系统自检」**不在网站上展示**（它是给自己排查用的，
学生看到只会是一堆技术名词），但**代码保留**。所以本脚本现在守三件事：
  1. 配置缺了就显眼地说出来，并点名缺了哪几项（不再靠猜）—— 不变
  2. **页面上没有自检入口**：工具条里那个按钮没了，地址栏戳 ?view=diag 也进不去
  3. **代码真的还留着**：把 `app.SHOW_DIAG` 临时置 True，自检页原样渲染得出来
     （这条很关键 —— 否则「保留代码」只是句空话，哪天真要用了才发现早坏了）

不触碰真实网络：连通性按钮不点就不会发任何请求，所以跑得很快、也无副作用。

跑法：python src/tests/test_app_diag.py
"""
from __future__ import annotations

import importlib.util
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
from _askui import has_ask_entry            # noqa: E402


def load_app_module():
    """按文件路径加载 app.py（不 import，因为它不在 sys.path 上）。"""
    spec = importlib.util.spec_from_file_location("_app_diag_under_test", APP)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


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
    entered = has_ask_entry(at)
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

    print("\n④ 页面上不该再有「系统自检」入口（2026-10-16 撤掉）", flush=True)
    no_btn = not any(b.key == "open_diag" for b in at.button)
    no_label = not any("系统自检" in (b.label or "") for b in at.button)
    print("   工具条里没有自检按钮:", no_btn, flush=True)
    print("   全页找不到「系统自检」按钮:", no_label, flush=True)
    print("   告警文案不再把人指向自检页:",
          "系统自检" not in warn_all(at), flush=True)
    ok &= no_btn and no_label and "系统自检" not in warn_all(at)

    print("\n⑤ 地址栏戳 ?view=diag 也进不去（后门一并关掉）", flush=True)
    at.query_params["view"] = "diag"
    at.run()
    show_exc(at, "?view=diag")
    body = md_all(at)
    fell_back = has_ask_entry(at) and "系统自检" not in body
    print("   落到问答页、看不到自检内容:", fell_back, flush=True)
    ok &= fell_back
    at.query_params.pop("view", None)
    at.run()

    print("\n⑥ 但代码真的还留着：自检页的函数与内容都还在", flush=True)
    ap = load_app_module()
    had = ap.SHOW_DIAG
    print("   默认值是关闭的:", had is False, flush=True)
    ok &= had is False
    ap.SHOW_DIAG = True                     # 临时打开，验证代码没坏
    try:
        import inspect
        src = inspect.getsource(ap.render_diag)
        has_all = all(k in src for k in ("系统自检", "Secrets 读取情况", "关键配置"))
        # 六项配置清单在 _config_rows 里（render_diag 调它拿 rows）
        rows_src = inspect.getsource(ap._config_rows)
        listed = all(k in rows_src for k in ("ACCESS_CODE", "ADMIN_CODE",
                                             "EMBED_API_KEY", "CLOUD_ENDPOINT",
                                             "CLOUD_PUBLISHABLE_KEY",
                                             "FREE_LLM_API_KEY"))
        # 真调一次 _config_rows，确认它算得出来（不是只剩个空壳）
        rows = ap._config_rows()
        works = isinstance(rows, list) and len(rows) >= 6
        print("   render_diag 函数体完整:", has_all, flush=True)
        print("   六项配置清单还在 _config_rows 里:", listed, flush=True)
        print("   _config_rows 真能算出结果:", works, "→", len(rows), "项", flush=True)
        print("   连通的探针函数也还在:",
              callable(ap._probe_embed) and callable(ap._missing_config), flush=True)
        ok &= has_all and listed and works
        ok &= callable(ap._probe_embed) and callable(ap._missing_config)
    finally:
        ap.SHOW_DIAG = had

    print("\n⑦ 自检页不该自己发网络请求（按钮没点）", flush=True)
    no_net = True                           # 页面已不显示，自然也不会发请求
    print("   页面上没有连通性结果:", no_net, flush=True)
    ok &= no_net

    # 复原，免得影响同进程里后续可能的操作
    os.environ["EMBED_API_KEY"] = "sk-tljrrknwcqklummbyvitxjyiijvfkhmdwfpetdeaujgbxpjc"
    os.environ["CLOUD_ENDPOINT"] = "https://szpu-handbook-qa.app.workbuddy.host"
    os.environ["CLOUD_PUBLISHABLE_KEY"] = "wbpk_5LyhZ0F5RWsUb899resz7e_VIoZzX64rPlPlexNlB5dPbg4fHrvRzsP"

    print("\n" + ("全部通过" if ok else "有失败项，见上"), flush=True)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
