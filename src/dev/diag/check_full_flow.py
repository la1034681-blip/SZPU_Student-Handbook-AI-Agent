# -*- coding: utf-8 -*-
"""全流程目检：用户说「全流程最后跑一次检查」时跑这个。

走一遍学生真正会碰的完整路径，每一步都记时间：

    进门（口令）→ 首屏几秒开 → 留言墙是不是**第一次就显示**已有留言
    → 问一道题（几秒出答案）→ 追问一句（记忆接不接得上）
    → 刷新页面（记忆按设计清空、留言还在）
    → 提交一条留言（当场能不能看到）
    → 使用统计页 → 系统自检页（有没有红字）

判据全在脚本里，坏在哪一步会直接打印出来，不用人眼找。

为什么不用 agent-browser：它的 Chromium 要从 googleapis 下，这台机器连不上。
改用 playwright 的 `channel="chrome"`，直接驱动本机已装的 Chrome。

⚠ 三个踩过的坑，脚本里已经处理，别改掉：
  ① 刷新页面会连「已通过口令」一起丢掉 → 刷新后必须重新过门禁，
     否则 main() 在门禁处就 return，后面全白等（曾误判成「功能坏了」）。
  ② 等答案要看「耗时 x 秒」出现，别用固定 sleep（免费模型慢）。
  ③ 别默认用户跑的是本地 bat —— 也可能是线上 streamlit.app，端口由参数给。

用法：
    python src/dev/diag/check_full_flow.py [地址]
默认 http://localhost:8501/
"""
from __future__ import annotations

import re
import sys
import time
from pathlib import Path

from playwright.sync_api import sync_playwright

URL = sys.argv[1] if len(sys.argv) > 1 else "http://localhost:8501/"
OUT = Path(__file__).resolve().parents[3] / ".trash_tmp" / "layout_shots"
OUT.mkdir(parents=True, exist_ok=True)

CODE = "SZPU666"
PROBLEMS: list[str] = []
STEP_TIMES: list[tuple[str, float]] = []


def ck(name: str, cond: bool, extra: str = ""):
    tag = "✓" if cond else "✗"
    if not cond:
        PROBLEMS.append(f"{name}　{extra}")
    print(f"   {tag} {name}" + (f"　—— {extra}" if extra else ""), flush=True)


def wait_first_screen(pg, limit=90) -> str:
    """等首屏真的画出来，返回 "gate"（要口令）/ "app"（已经在里面）/ "unknown"。

    ⚠ 这个函数是本脚本最容易骗人的地方（2026-10-16 半夜踩到，白等 7 分钟）：
    Streamlit 是**异步渲染**，`goto` 一返回时 DOM 还是空的。这时候去数
    「有没有口令框」必然数到 0 —— 于是老版本的 pass_gate 认定「不用口令」，
    直接放行，后面每一句检查都做在**登录页**上：
    留言墙「没显示」、问答「答不出」……全是不存在的故障。
    所以必须先等到「要么口令框、要么问答入口」出现，再判断该走哪条路。
    """
    for _ in range(limit * 2):
        if pg.locator('input[type="password"]').count():
            return "gate"
        if pg.locator('[data-testid="stTextInput"] input').count():
            return "app"
        pg.wait_for_timeout(500)
    return "unknown"


def pass_gate(pg) -> bool:
    """门口那道口令。已经在里面就直接返回 True。"""
    state = wait_first_screen(pg)
    if state == "app":
        return True
    if state != "gate":
        return False

    for _ in range(3):                    # 提交没生效就重来一次（Streamlit 偶发丢事件）
        box = pg.locator('input[type="password"]').first
        box.fill(CODE)
        if box.input_value() != CODE:     # 值没进去就别白点
            pg.wait_for_timeout(800)
            continue
        pg.get_by_role("button", name="进 入").click()
        for _ in range(120):
            pg.wait_for_timeout(500)
            if not pg.locator('input[type="password"]').count():
                return True
    return False


def wait_ready(pg, limit=180) -> bool:
    """等一次脚本运行结束（Streamlit 的 running 标记消失）。

    首屏尚未渲染时它也可能「一开始就不存在」，所以调用前先确认页面已经画出来了。
    """
    pg.wait_for_timeout(1500)
    for _ in range(limit * 2):
        if pg.locator('[data-testid="stStatusWidget"]').count() == 0:
            return True
        pg.wait_for_timeout(500)
    return False


def step(name: str, t0: float):
    dt = time.time() - t0
    STEP_TIMES.append((name, dt))
    print(f"   ⏱ {name}：{dt:.1f} 秒", flush=True)


def main():
    with sync_playwright() as p:
        br = p.chromium.launch(channel="chrome", headless=True,
                               args=["--disable-gpu"])
        pg = br.new_page(viewport={"width": 1080, "height": 1000})

        # ---------------------------------------------------------- ① 进门
        print("\n① 进门（输口令）", flush=True)
        t0 = time.time()
        pg.goto(URL, wait_until="domcontentloaded")
        ok = pass_gate(pg)
        ck("进得去", ok)
        wait_ready(pg)
        step("开门", t0)
        body = pg.inner_text("body")
        ck("没有红字报错", "出错了" not in body and "Traceback" not in body)
        # 防止「其实还在门口、后面全白测」（见 wait_first_screen 的注释）
        inside = (pg.locator('input[type="password"]').count() == 0
                  and "问答" in body)
        ck("确实站在问答页上（不是在门口）", inside,
           "口令没输进去或页面没跳转——后面几项的结论都不作数")
        if not inside:
            pg.screenshot(path=str(OUT / "f0_stuck_at_gate.png"), full_page=True)
            br.close()
            print("\n没进到问答页，后面的检查没意义，先停下。截图：", OUT, flush=True)
            return 1

        # ---------------------------------------------- ② 留言墙第一次就显示
        print("\n② 留言墙（用户报的就是这条：一开始没显示之前的留言）", flush=True)
        wall = pg.inner_text("body")
        m = re.search(r"墙上已有\s*(\d+)\s*条留言", wall)
        ck("首屏就显示已有留言条数", bool(m),
           "首屏里找不到「墙上已有 N 条留言」—— 慢或没显示就是这条坏了")
        if m:
            print(f"   墙上现在有 {m.group(1)} 条留言", flush=True)
        ck("没有谎报「还空着」", "墙上还空着" not in wall)
        ck("没有误报「暂时读不到」", "暂时读不到" not in wall)
        pg.screenshot(path=str(OUT / "f1_first_screen.png"), full_page=True)

        # ---------------------------------------------------------- ③ 问一道题
        print("\n③ 问一道题", flush=True)
        t0 = time.time()
        pg.locator('[data-testid="stTextInput"] input').first.fill("必修课考试不及格怎么办")
        pg.keyboard.press("Enter")
        answered = False
        for _ in range(300):
            pg.wait_for_timeout(1000)
            b = pg.inner_text("body")
            if "耗时" in b and ("手册原文" in b or "本手册中没有对应内容" in b):
                answered = True
                break
        ck("答出来了", answered, "等了 300 秒还没出答案")
        step("第一问", t0)
        b = pg.inner_text("body")
        ck("答复里有手册原文可看", "手册原文" in b)
        pg.screenshot(path=str(OUT / "f2_answered.png"), full_page=True)

        # ---------------------------------------------------------- ④ 追问
        print("\n④ 追问一句（同一页里记忆接不接得上）", flush=True)
        t0 = time.time()
        pg.locator('[data-testid="stTextInput"] input').first.fill("那它算不算迟到早退")
        pg.keyboard.press("Enter")
        done2 = False
        for _ in range(300):
            pg.wait_for_timeout(1000)
            b = pg.inner_text("body")
            if b.count("耗时") >= 2 and ("手册原文" in b or "本手册中没有对应内容" in b):
                done2 = True
                break
        ck("追问也答了", done2)
        step("追问", t0)
        b = pg.inner_text("body")
        ck("页面上有两轮问答", b.count("耗时") >= 2, f"页面只看到 {b.count('耗时')} 轮")
        pg.screenshot(path=str(OUT / "f3_followup.png"), full_page=True)

        # --------------------------------------- ⑤ 刷新：记忆清空、留言还在
        print("\n⑤ 刷新页面（按你的要求：刷新＝清掉上一次会话的临时记忆）", flush=True)
        t0 = time.time()
        pg.reload(wait_until="domcontentloaded")
        pass_gate(pg)              # 坑①：刷新会连口令状态一起丢
        wait_ready(pg)
        step("刷新+重新进门", t0)
        b = pg.inner_text("body")
        ck("问答记录已清空", b.count("耗时") == 0, f"刷新后还剩 {b.count('耗时')} 轮")
        ck("上一轮答案没残留", "必修课考试不及格怎么办" not in b)
        m2 = re.search(r"墙上已有\s*(\d+)\s*条留言", b)
        ck("留言墙刷新后照样显示", bool(m2))
        pg.screenshot(path=str(OUT / "f4_after_reload.png"), full_page=True)

        # ---------------------------------------------------------- ⑥ 提交留言
        print("\n⑥ 提交一条留言（当场能不能看到）", flush=True)
        t0 = time.time()
        ta = pg.locator("textarea").first
        ta.fill("全流程检查留的这条：刷新后还能提交并立刻显示")
        btns = pg.get_by_role("button")
        hit = False
        for i in range(btns.count()):
            if "提交" in (btns.nth(i).inner_text() or ""):
                btns.nth(i).click()
                hit = True
                break
        ck("找得到提交按钮", hit)
        for _ in range(120):
            pg.wait_for_timeout(500)
            b = pg.inner_text("body")
            if "全流程检查留的这条" in b and b.count("全流程检查留的这条") >= 1:
                break
        step("提交留言", t0)
        ck("刚写的那条立刻出现在墙上", "全流程检查留的这条" in b)
        pg.screenshot(path=str(OUT / "f5_suggest.png"), full_page=True)

        # ---------------------------------------------------------- ⑦ 系统自检
        print("\n⑦ 系统自检页（?view=diag）", flush=True)
        t0 = time.time()
        pg.goto(URL.rstrip("/") + "/?view=diag", wait_until="domcontentloaded")
        pass_gate(pg)
        wait_ready(pg)
        step("自检页", t0)
        d = pg.inner_text("body")
        ck("自检页没有红字", "出错了" not in d)
        ck("不再提「闲置 N 分钟」", "闲置" not in d)
        ck("没有「清掉闲置记忆」按钮", "清掉闲置" not in d)
        ck("明说记忆只在这一页活着", "这一页" in d or "刷新" in d)
        pg.screenshot(path=str(OUT / "f6_diag.png"), full_page=True)

        br.close()

    print("\n———— 时间小结 ————", flush=True)
    for n, dt in STEP_TIMES:
        print(f"   {n:<14}{dt:>6.1f} 秒", flush=True)
    if PROBLEMS:
        print(f"\n有 {len(PROBLEMS)} 处不对：", flush=True)
        for x in PROBLEMS:
            print("   ✗ " + x, flush=True)
    else:
        print("\n全流程没有发现问题 ✓", flush=True)
    print("截图目录:", OUT, flush=True)
    return 1 if PROBLEMS else 0


if __name__ == "__main__":
    sys.exit(main())
