# -*- coding: utf-8 -*-
"""离线测试：留言墙「读不到」和「真的没有」必须说不同的话（2026-10-15 新增）。

起因
----
用户报「留言板一开始没显示之前的留言」。查下来是：
`select()` 设计上失败返回 None（好让调用方区分「读到 0 条」与「没读到」），
但 `list_suggestions()` 写了句 `return rows or []` —— **None 被抹成空列表**，
页面于是照着「空」渲染，谎报「墙上还空着」。云库一超时就是这个症状。

守住
----
① 读不到（None）→ 页面明说「暂时读不到」+ 给一个「重新加载」按钮，
   **不许**出现「墙上还空着」
② 真的没有（[]）→ 才显示「墙上还空着」
③ 读到了 → 显示条数与内容
④ 提交成功后立刻能看到刚提交的那条（缓存被清掉）

全离线：把 `cloudstore.get_store` 换成假的，不发任何请求。

跑法：python src/tests/test_suggest_wall.py
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

os.environ["ACCESS_CODE"] = "off"
os.environ["CLOUD_LOG_ENABLED"] = "0"

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / "src"))
import _paths  # noqa: E402 把 core/retrieval/infra/cli 也挂上搜索路径
sys.path.insert(0, str(ROOT / "src" / "tests"))

import cloudstore                                   # noqa: E402
from cloudstore import CloudStore                  # noqa: E402
from streamlit.runtime.caching import cache_data    # noqa: E402
from streamlit.testing.v1 import AppTest           # noqa: E402

_fail = 0


def ck(name: str, cond: bool, extra: str = ""):
    global _fail
    if not cond:
        _fail += 1
    print(f"  {'✓' if cond else '✗'} {name}" + (f"  —— {extra}" if extra and not cond else ""))


class FakeStore:
    """假云库：rows 为 None 表示「读不到」，为 [] 表示「真的没有」。"""

    def __init__(self, rows):
        self.rows = rows
        self.added = []

    @property
    def enabled(self):
        return True

    def list_suggestions(self, limit: int = 60, timeout: int = 4):
        return None if self.rows is None else list(self.rows)

    def add_suggestion(self, client_id: str, content: str):
        self.added.append(content)
        if self.rows is None:
            self.rows = []
        self.rows.insert(0, {"id": "x", "content": content,
                             "created_at": "2026-10-15T23:50:00+08:00"})
        return True, "收到了，谢谢"


def body_text(at) -> str:
    return "\n".join([m.value for m in at.markdown]
                     + [c.value for c in at.caption]
                     + [w.value for w in at.warning]
                     + [s.value for s in at.success])


def open_page(rows):
    """用假云库开一次页面，返回 (AppTest, FakeStore)。"""
    cache_data.clear()                              # 清掉 _load_suggestions 的缓存
    fake = FakeStore(rows)
    cloudstore.get_store = lambda: fake             # noqa: PLW0642
    at = AppTest.from_file(str(ROOT / "app.py"), default_timeout=300)
    at.session_state["authed"] = True
    at.run()
    return at, fake


def main() -> int:
    keep = cloudstore.get_store
    try:
        print("① 读不到（云库超时/没配）—— 不能谎称「墙上还空着」")
        at, _ = open_page(None)
        txt = body_text(at)
        ck("页面没炸", not at.exception, str([e.value for e in at.exception])[:300])
        ck("明说「暂时读不到」", "暂时读不到" in txt, txt[-200:])
        ck("没有谎称「墙上还空着」", "墙上还空着" not in txt, txt[-200:])
        ck("给了「重新加载」按钮", any(b.key == "sugg_retry" for b in at.button))

        print("\n② 真的没有留言 —— 这时候才可以说「还空着」")
        at, _ = open_page([])
        txt = body_text(at)
        ck("页面没炸", not at.exception, str([e.value for e in at.exception])[:300])
        ck("显示「墙上还空着」", "墙上还空着" in txt, txt[-200:])
        ck("没有误报「暂时读不到」", "暂时读不到" not in txt)
        ck("没有多出重试按钮", not any(b.key == "sugg_retry" for b in at.button))

        print("\n③ 读到了 —— 显示条数与内容")
        at, _ = open_page([{"id": "a", "content": "补考能不能重考",
                            "created_at": "2026-10-15T10:00:00+08:00"}])
        txt = body_text(at)
        ck("显示「墙上已有 1 条留言」", "墙上已有 1 条留言" in txt, txt[-200:])
        ck("留言内容出现在页面上", "补考能不能重考" in txt)
        ck("没有误报「还空着」", "墙上还空着" not in txt)

        print("\n④ 提交后立刻能看到刚写的那条")
        at, fake = open_page([{"id": "a", "content": "老留言",
                               "created_at": "2026-10-15T10:00:00+08:00"}])
        ta = at.text_area[0]
        ta.set_value("这条是新写的")
        for b in at.button:
            if b.key and "提交建议" in (b.label or ""):
                b.click()
                break
        at.run()
        txt = body_text(at)
        ck("提示提交成功", "提交成功" in txt, txt[-200:])
        ck("新留言立刻出现", "这条是新写的" in txt, txt[-250:])
        ck("假云库确实收到了", "这条是新写的" in fake.added)
        ck("条数变成 2", "墙上已有 2 条留言" in txt, txt[-200:])

        print("\n⑤ 单元级：list_suggestions 把 None 原样透出")
        real = CloudStore.__new__(CloudStore)         # 不跑 __init__，只借方法
        real.sugg_table = "qa_suggestion"             # __init__ 平时会设好的那个表名
        real.select = lambda *a, **k: None           # noqa: E731
        ck("读不到 → 返回 None（不是 []）",
           CloudStore.list_suggestions(real) is None)
        real.select = lambda *a, **k: []             # noqa: E731
        ck("读到空表 → 返回 []",
           CloudStore.list_suggestions(real) == [])
        real.select = lambda *a, **k: [{"id": "1"}]   # noqa: E731
        ck("读到数据 → 原样返回",
           CloudStore.list_suggestions(real) == [{"id": "1"}])
    finally:
        cloudstore.get_store = keep

    print("\n" + ("全部通过 ✓" if _fail == 0 else f"有 {_fail} 项失败 ✗"))
    return 0 if _fail == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
