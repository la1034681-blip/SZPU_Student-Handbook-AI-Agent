# -*- coding: utf-8 -*-
"""浏览器目检脚本的共用件：拿到「真正承载应用内容」的那个对象。

为什么需要（2026-10-17 踩到，白追半天）
--------------------------------------
**Streamlit Community Cloud 会把应用放进一个 iframe**，路径形如 `<域名>/~/+/`；
主文档里只剩约 48 个字符的壳。于是 `page.locator("input")` 恒为 0、
`page.inner_text("body")` 恒为空 —— 极容易被误读成「页面很慢／根本没渲染出来」，
**而截图里页面明明完好**（实测线上首屏只要 **9.2 秒**，一点都不慢）。
本地 `localhost:8501` 没有这一层，所以这个坑只有在第一次对 streamlit.app
跑的时候才会现形。

用法
----
    pg.goto(URL, wait_until="domcontentloaded")
    view = content_view(pg)      # 之后所有 DOM 操作都用 view，别再用 pg
    view.locator(...)  view.inner_text("body")  view.evaluate(...)
    pg.screenshot(...)           # 截图仍用 page（它抓的是整页合成结果）

⚠ **每次 goto / reload / 切视图之后都要重新取一次** —— iframe 会被换成新对象。
⚠ `keyboard` 是 page 级的，进了 iframe 就按不到里面的元素；改用元素自己的
  `press()`：`view.locator('input').first.press("Enter")`。
"""
from __future__ import annotations


def content_view(pg, limit=90):
    """返回承载应用内容的 frame（线上 Cloud 是 iframe），找不到就退回主文档。

    判据：这个对象里出现了「口令框」或「问答输入框」—— 即首屏真的画出来了。
    """
    for _ in range(limit * 2):
        for fr in pg.frames:                 # iframe 优先（线上）
            if fr == pg.main_frame:
                continue
            try:
                if (fr.locator('input[type="password"]').count()
                        or fr.locator('[data-testid="stTextInput"] input').count()):
                    return fr
            except Exception:                # frame 可能正被替换，忽略这一轮
                pass
        try:                                 # 主文档（本地）
            if (pg.locator('input[type="password"]').count()
                    or pg.locator('[data-testid="stTextInput"] input').count()):
                return pg.main_frame
        except Exception:
            pass
        pg.wait_for_timeout(500)
    return pg.main_frame
