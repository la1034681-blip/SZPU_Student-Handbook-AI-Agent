# -*- coding: utf-8 -*-
"""离线测试：改了后台代码却没重启网页进程时，页面能自己说清（2026-10-15 新增）。

守的是用户截图那个坑
--------------------
问「必修课考试不及格，能补考吗？」，页面回一句英文：
    AgentQAPipeline.ask() got an unexpected keyword argument 'history_block'
学生看不懂，管理员也未必第一时间想到「该重启了」。

根因：Streamlit 只重跑主脚本 app.py，被它引用的 src/ 模块一直留在内存里。
于是出现 app.py 是新的、agent_qa 还是旧的 —— 参数对不上就炸。
这不是功能坏了，重启一下进程就好。现在页面会自己把这句话说出来。

测什么
------
① 探针清单本身是对的（路径/参数名写错会**永远误报**，比不报更糟，所以先守它）
② 能认出「函数签名变旧了」③ 能认出「模块没在内存里」④ 能认出「函数找不到」
⑤ 正常的页面不该出现这条提示；混用时必须出现，且写明怎么修
⑥ 「系统自检」页也要看得到

不碰网络（CLOUD_LOG_ENABLED=0）、不调模型。

跑法：python src/tests/test_stale_code.py
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

os.environ["ACCESS_CODE"] = "off"
os.environ["CLOUD_LOG_ENABLED"] = "0"

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "src" / "tests"))

import agent_qa                                   # noqa: E402
import app as appmod                              # noqa: E402
from streamlit.testing.v1 import AppTest          # noqa: E402

_fail = 0


def ck(name: str, cond: bool, extra: str = ""):
    global _fail
    if not cond:
        _fail += 1
    print(f"  {'✓' if cond else '✗'} {name}" + (f"  —— {extra}" if extra and not cond else ""))


def errors_at(at) -> str:
    return "\n".join(e.value for e in at.error)


def click(at, key: str) -> bool:
    for b in at.button:
        if b.key == key:
            b.click()
            return True
    return False


def old_ask(self, question):                      # 旧版签名：不认 history_block
    """假装这是没改造前的 ask()。"""


def main() -> int:
    print("① 探针清单本身是对的（当前代码下应当全部通过）")
    bad = appmod._stale_modules()
    ck("当前代码下没有任何「对不上」", bad == [], str(bad))
    ck("清单里至少 4 个探针", len(appmod._STALE_PROBES) >= 4)

    print("\n② 能认出「函数签名是旧的」")
    keep = agent_qa.AgentQAPipeline.ask
    agent_qa.AgentQAPipeline.ask = old_ask
    try:
        bad = appmod._stale_modules()
        ck("点出了 agent_qa", any("agent_qa" in b for b in bad), str(bad))
        ck("说清了缺哪个参数", any("history_block" in b for b in bad), str(bad))
    finally:
        agent_qa.AgentQAPipeline.ask = keep
    ck("恢复成新签名后不再报", appmod._stale_modules() == [],
       str(appmod._stale_modules()))

    print("\n③ 能认出「模块没在内存里」")
    saved = sys.modules.pop("memory", None)
    try:
        bad = appmod._stale_modules()
        ck("点出了 memory", any("memory" in b for b in bad), str(bad))
    finally:
        if saved is not None:
            sys.modules["memory"] = saved
    ck("放回去后不再报", appmod._stale_modules() == [], str(appmod._stale_modules()))

    print("\n④ 能认出「清单里的函数找不到了」")
    keep_cls = agent_qa.AgentQAPipeline

    class _Fake:                                   # 类没了 → 路径走不通
        pass

    agent_qa.AgentQAPipeline = _Fake
    try:
        bad = appmod._stale_modules()
        ck("点出了找不到", any("找不到" in b for b in bad), str(bad))
    finally:
        agent_qa.AgentQAPipeline = keep_cls
    ck("换回来不再报", appmod._stale_modules() == [], str(appmod._stale_modules()))

    print("\n⑤ 页面上的表现")
    at = AppTest.from_file(str(ROOT / "app.py"), default_timeout=300)
    at.session_state["authed"] = True
    at.run()
    ck("页面没炸", not at.exception, str([e.value for e in at.exception])[:300])
    ck("正常时不该出现这条提示",
       "网页还是旧的" not in errors_at(at), errors_at(at)[:200])

    agent_qa.AgentQAPipeline.ask = old_ask          # 制造「新旧混用」
    try:
        at.run()
        err = errors_at(at)
        ck("混用时页面出现提示", "网页还是旧的" in err, err[:200] or "（页面上一片安静）")
        ck("提示里写了怎么修", "启动网页问答" in err, err[:200])
        ck("提示里点名了出问题的模块", "history_block" in err, err[:200])
    finally:
        agent_qa.AgentQAPipeline.ask = keep

    at.run()
    ck("修好后提示自动消失", "网页还是旧的" not in errors_at(at),
       errors_at(at)[:200])

    print("\n⑥ 「系统自检」页也看得到")
    click(at, "open_diag")
    at.run()
    body = "\n".join(m.value for m in at.markdown)
    caps = "\n".join(c.value for c in at.caption)
    ck("自检页没炸", not at.exception, str([e.value for e in at.exception])[:300])
    ck("有「代码版本」一节", "代码版本" in body)
    ck("正常时显示一致", "一致" in caps)

    print("\n" + ("全部通过 ✓" if _fail == 0 else f"有 {_fail} 项失败 ✗"))
    return 0 if _fail == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
