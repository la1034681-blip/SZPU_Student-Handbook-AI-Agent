# -*- coding: utf-8 -*-
"""版式目检：真浏览器里走一遍「进门 → 点示例 → 看答案落在哪」。

为什么不用 agent-browser：它的 Chromium 要从 googleapis 下，这台机器连不上。
改用 playwright 的 `channel="chrome"`，直接驱动本机已装的 Chrome，不额外下载。

用法：
    python src/dev/diag/shot_layout.py [地址]
默认 http://localhost:8502/ —— 端口由项目根 `_pick_port.py` 决定；
本机还装着别的 Streamlit 项目占着 8501，所以别把端口写死。

产出：截图落在 .trash_tmp/layout_shots/ 下，供人工核对位置。
"""
from __future__ import annotations

import sys
from pathlib import Path

from playwright.sync_api import sync_playwright

URL = sys.argv[1] if len(sys.argv) > 1 else "http://localhost:8502/"
OUT = Path(__file__).resolve().parents[3] / ".trash_tmp" / "layout_shots"
OUT.mkdir(parents=True, exist_ok=True)


def main():
    with sync_playwright() as p:
        br = p.chromium.launch(channel="chrome", headless=True,
                               args=["--disable-gpu"])
        pg = br.new_page(viewport={"width": 1080, "height": 900})
        pg.goto(URL, wait_until="domcontentloaded")
        pg.wait_for_timeout(6000)

        # 门禁
        if pg.locator('input[type="password"]').count():
            pg.fill('input[type="password"]', "SZPU666")
            pg.get_by_role("button", name="进 入").click()
            pg.wait_for_timeout(8000)

        pg.screenshot(path=str(OUT / "01_first_screen.png"), full_page=True)
        print("① 首屏已存", flush=True)

        def top(sel):
            loc = pg.locator(sel)
            return loc.first.bounding_box()["y"] if loc.count() else -1

        # 关键检查：输入框必须在示例按钮**下面**（用户 2026-10-15 指定）
        y_sample = top('button:has-text("必修课考试不及格")')
        y_input = top('[data-testid="stTextInput"] input')
        y_wall = top("textarea")
        print(f"   示例按钮 y = {y_sample}", flush=True)
        print(f"   输入框   y = {y_input}", flush=True)
        print(f"   留言框   y = {y_wall}", flush=True)
        print("   示例在输入框之上:", 0 < y_sample < y_input, flush=True)
        print("   输入框在留言墙之上:", 0 < y_input < y_wall, flush=True)

        # 点第一个示例
        pg.get_by_role("button", name="必修课考试不及格").click()
        pg.wait_for_timeout(3000)
        pg.screenshot(path=str(OUT / "02_loading.png"), full_page=True)
        print("② 点击后（加载中）已存", flush=True)

        # 等答案：出现「手册原文」或拒答话术为止，最多 240 秒
        for _ in range(240):
            body = pg.inner_text("body")
            if "手册原文" in body or "手册里暂无相关内容" in body:
                break
            pg.wait_for_timeout(1000)
        pg.wait_for_timeout(2500)
        pg.screenshot(path=str(OUT / "03_answered.png"), full_page=True)
        print("③ 出答案后已存", flush=True)

        y_msg = top('[data-testid="stChatMessage"]')
        y_wall2 = top("textarea")
        y_sample2 = top('button:has-text("必修课考试不及格")')
        print(f"   问答气泡 y = {y_msg}", flush=True)
        print(f"   留言框   y = {y_wall2}", flush=True)
        print("   气泡在留言墙之上:", 0 < y_msg < y_wall2, flush=True)
        print("   答完后示例按钮已收起:", y_sample2 < 0, flush=True)
        br.close()
    print("输出目录:", OUT, flush=True)


if __name__ == "__main__":
    main()