# -*- coding: utf-8 -*-
"""验证两个用户反馈 bug 是否修好（2026-10-17）。

  bug①  点第二轮问答的 [E1] 角标，跳到的却是**第一轮**的手册原文。
        根因：每轮的锚点 id 都叫 `ev-E1` / `ev-E2`，浏览器只认页面上第一个
        同 id 的元素 ⇒ 第二轮角标把人送回第一轮。改法：每轮带自己的前缀
        （`ev-t3-E1`），见 `app.linkify_citations` / `app.render_assistant`。
  bug②  每一轮问答的等待提示都说「启动后首次需加载模型，约 30 秒」，
        而本机模型其实只在**第一题**加载，之后每题都是秒级。
        改法：提示语除了看后端，还要看模型是否真的还没进内存
        （`app._loading_tip` + `app._local_models_ready`）。

判据（写死在脚本里）：
  ① 页面里所有 `ev-*` 锚点 id **唯一**；
     第二轮问答的角标 href 必须带自己的轮次前缀，与第一轮**不同**；
     点下去之后 location.hash 落在第二轮那个 id 上，且该卡片确实滚进视口。
  ② 第一题的等待提示**应**出现「加载模型」；
     第二题的等待提示**不应**再出现。

用法：
    python src/dev/diag/check_citation_anchor.py [地址]
默认 http://localhost:8501/

⚠ 必须在「服务刚启动、模型一次都还没加载过」的状态下跑，
  否则第一题那条「应提示加载模型」抓不到（模型已在内存里）。
"""
from __future__ import annotations

import sys
import time

from playwright.sync_api import sync_playwright

URL = sys.argv[1] if len(sys.argv) > 1 else "http://localhost:8501/"
CODE = "SZPU666"
PROBLEMS: list[str] = []


def ck(name: str, cond: bool, extra: str = ""):
    tag = "✓" if cond else "✗"
    if not cond:
        PROBLEMS.append(name)
    print(f"   {tag} {name}" + (f"　—— {extra}" if extra else ""), flush=True)


def wait_first_screen(pg, limit=90) -> str:
    """等首屏真画出来，返回 "gate" / "app" / "unknown"。

    Streamlit 是异步渲染，goto 一返回 DOM 还是空的 —— 这时数「有没有口令框」
    必然数到 0，会误判成「已经在里面」。所以必须等到两者之一出现再判断。
    """
    for _ in range(limit):
        try:
            if "试试这样问" in pg.inner_text("body"):
                return "app"
            if pg.locator('input[type="password"]').count():
                return "gate"
        except Exception:                                   # noqa: BLE001
            pass
        pg.wait_for_timeout(1000)
    return "unknown"


def pass_gate(pg) -> bool:
    state = wait_first_screen(pg)
    if state == "app":
        return True
    if state != "gate":
        return False
    for _ in range(3):
        box = pg.locator('input[type="password"]').first
        box.fill(CODE)
        if box.input_value() != CODE:
            pg.wait_for_timeout(800)
            continue
        pg.get_by_role("button", name="进 入").click()
        for _ in range(120):
            pg.wait_for_timeout(500)
            if not pg.locator('input[type="password"]').count():
                return True
    return False


def ask(pg, q: str, want_rounds: int, timeout: int = 420):
    """问一题，等第 want_rounds 轮答案出来。

    返回 (是否答完, 等待期间出现过的提示语集合)。
    提示语是在 spinner 里一闪而过的，所以边等边抓，别等完了再找。
    """
    pg.locator('[data-testid="stTextInput"] input').first.fill(q)
    pg.keyboard.press("Enter")
    tips: set[str] = set()
    t0 = time.time()
    while time.time() - t0 < timeout:
        try:
            body = pg.inner_text("body")
        except Exception:                                   # noqa: BLE001
            body = ""
        for line in body.splitlines():
            if "正在检索" in line:
                tips.add(line.strip())
        if body.count("耗时") >= want_rounds:
            return True, tips
        pg.wait_for_timeout(300)
    return False, tips


def main() -> int:
    with sync_playwright() as p:
        br = p.chromium.launch(channel="chrome", headless=True, args=["--disable-gpu"])
        pg = br.new_page(viewport={"width": 1280, "height": 900})
        pg.goto(URL, wait_until="domcontentloaded")

        print("① 进门")
        if not pass_gate(pg):
            print("   ✗ 没进得去，后面无法验证")
            br.close()
            return 1
        ck("进得去", True)

        print("\n② 第一题（模型应还没加载）")
        t0 = time.time()
        ok1, tips1 = ask(pg, "必修课考试不及格怎么办", 1)
        ck("答出来了", ok1, f"用了 {time.time()-t0:.1f} 秒")
        tip1 = "；".join(sorted(tips1)) or "（没抓到提示语）"
        print(f"      等待提示：{tip1}")
        ck("第一题提示里说了要加载模型（这题确实要等）",
           any("加载模型" in t for t in tips1), tip1)

        print("\n③ 第二题（模型已在内存，不该再喊加载）")
        t0 = time.time()
        ok2, tips2 = ask(pg, "那它算不算迟到早退", 2)
        ck("答出来了", ok2, f"用了 {time.time()-t0:.1f} 秒")
        tip2 = "；".join(sorted(tips2)) or "（没抓到提示语）"
        print(f"      等待提示：{tip2}")
        ck("第二题不再提「首次需加载模型」",
           not any("加载模型" in t for t in tips2), tip2)

        print("\n④ 两轮问答的角标锚点")
        ids = pg.eval_on_selector_all("[id^='ev-']", "els => els.map(e => e.id)")
        hrefs = pg.eval_on_selector_all(
            "a.cite-badge", "els => els.map(e => e.getAttribute('href'))")
        print(f"      锚点 id：{ids}")
        print(f"      角标 href：{hrefs}")
        ck("页面上有角标可点", len(hrefs) > 0, f"只有 {len(hrefs)} 个")
        ck("所有锚点 id 唯一（这是 bug① 的根子）",
           len(ids) == len(set(ids)), f"{len(ids)} 个 id、{len(set(ids))} 个不重复")
        # 两轮的前缀必须不同
        pres = {h.lstrip("#").split("-")[1] for h in hrefs if h and h.startswith("#ev-")}
        ck("两轮问答的角标前缀不同", len(pres) >= 2, f"前缀只有 {pres}")

        print("\n⑤ 点第二轮的角标（bug① 的核心判据）")
        last = pg.locator("a.cite-badge").last
        target = last.get_attribute("href")
        last.click()
        pg.wait_for_timeout(1000)
        h = pg.evaluate("location.hash")
        ck("点下去地址栏落到该轮的锚点", h == target, f"想要 {target}，实际 {h}")

        # 目标卡片是不是真滚进了视口
        rect = pg.evaluate(
            """(id) => { const el = document.getElementById(id);
                 if (!el) return null;
                 const r = el.getBoundingClientRect();
                 return {id: el.id, top: Math.round(r.top),
                         bottom: Math.round(r.bottom), vh: window.innerHeight}; }""",
            h.lstrip("#"))
        print(f"      目标卡片位置：{rect}")
        ck("目标卡片在视口里（说明真跳过去了）",
           bool(rect) and rect["top"] < rect["vh"] and rect["bottom"] > 0,
           str(rect))

        # 顺带确认：这个 id 属于**第二轮**，不是第一轮那个同号锚点
        ck("跳的是第二轮自己的卡片，不是第一轮的",
           bool(rect) and rect["id"] == target.lstrip("#"), str(rect))

        br.close()

    print("\n———— 结论 ————")
    if PROBLEMS:
        print("发现问题：")
        for x in PROBLEMS:
            print(f"   ✗ {x}")
        return 1
    print("两个 bug 都已修好 ✓")
    return 0


if __name__ == "__main__":
    sys.exit(main())
