# -*- coding: utf-8 -*-
"""门禁专项目检：门口挨个试坏输入，看页面会不会崩（2026-10-16 新增）。

起因
----
用户看到页面上报 `TypeError: comparing strings with non-ASCII characters
is not supported`，担心是改代码改坏了。实际是老代码就有的坑：
`hmac.compare_digest()` 在 Python 3 里只吃 ASCII，学生用中文输入法
打一个汉字进口令框就会抛异常 → 整个页面报错、谁都进不来。

这个脚本专门守门口：空口令、汉字、emoji、错口令、对口令挨个试，
**每一步都看页面上有没有 `.stException`（就是那行红字报错）**。

用法：
    python src/dev/diag/gate_probe.py [地址]
默认 http://localhost:8502/
"""
from __future__ import annotations

import sys
from pathlib import Path

from playwright.sync_api import sync_playwright

URL = sys.argv[1] if len(sys.argv) > 1 else "http://localhost:8502/"
CODE = "SZPU666"
ADMIN = "9925"
OUT = Path(__file__).resolve().parents[3] / ".trash_tmp" / "layout_shots"
OUT.mkdir(parents=True, exist_ok=True)
PROBLEMS: list[str] = []


def ck(name: str, cond: bool, extra: str = ""):
    if not cond:
        PROBLEMS.append(name)
    print(f"   {'✓' if cond else '✗'} {name}" + (f"　—— {extra}" if extra else ""),
          flush=True)


def at_gate(pg) -> bool:
    """还在门口（有口令框）。"""
    return pg.locator('input[type="password"]').count() > 0


def page_crashed(pg) -> str:
    """页面上有没有那行红字报错。返回报错文本（没有则空串）。"""
    if pg.locator(".stException").count() == 0:
        return ""
    return (pg.locator(".stException").first.inner_text() or "")[:200]


def try_code(pg, value: str, label: str):
    """在门口输一个值提交，返回 (是否崩, 崩的内容, 是否进去了)。"""
    box = pg.locator('input[type="password"]').first
    box.fill(value)
    pg.get_by_role("button", name="进 入").click()
    for _ in range(40):                 # 最多等 20 秒
        pg.wait_for_timeout(500)
        if not at_gate(pg):
            break
    crash = page_crashed(pg)
    inside = not at_gate(pg) and "问答" in pg.inner_text("body")
    print(f"   [{label}] 崩={bool(crash)} 进去了={inside}"
          + (f"　崩的内容：{crash}" if crash else ""), flush=True)
    return bool(crash), crash, inside


def main():
    with sync_playwright() as p:
        br = p.chromium.launch(channel="chrome", headless=True,
                               args=["--disable-gpu"])

        # ---- 第 0 步：先证明「正常情况能进得去」----
        # 这一步必须**单独开一页**：下面那串坏输入会连错好几次，触发
        # 「尝试次数过多」的 60 秒锁定，而锁定期间**连正确口令也不放行**
        # （这是门口的有意设计，不是 bug）。混在一页里测，正确口令就会被
        # 锁定挡住，脚本自己误报「进不去」（2026-10-16 踩过）。
        print("⓪ 正常情况：输对口令应该进得去", flush=True)
        pg0 = br.new_page(viewport={"width": 1080, "height": 900})
        pg0.goto(URL, wait_until="domcontentloaded")
        for _ in range(60):
            if at_gate(pg0):
                break
            pg0.wait_for_timeout(500)
        ck("门口正常显示（有口令框）", at_gate(pg0))
        pg0.locator('input[type="password"]').first.fill(CODE)
        pg0.get_by_role("button", name="进 入").click()
        inside0 = False
        for _ in range(60):
            pg0.wait_for_timeout(500)
            if not at_gate(pg0):
                inside0 = True
                break
        ck("正确口令进得去", inside0 and "问答" in pg0.inner_text("body"),
           "正确口令都被挡住了 —— 门口功能本身坏了")
        ck("进去后页面没崩", not page_crashed(pg0))
        pg0.screenshot(path=str(OUT / "g0_inside.png"), full_page=True)

        # ---- 顺路查「使用统计」看板：进门后点进去，输密钥，看能不能出数 ----
        # ⚠ 两件事别搞混（这个脚本第一版就错在这）：
        #   ① 门禁页和看板页**都有** type=password 的输入框，得靠 placeholder 区分
        #      （口令框＝「请输入访问口令」，密钥框＝「请输入查看密钥」）；
        #   ② 直接 goto ?view=stats 等于**新开一次页面加载**，上个页面的登录状态
        #      没了 → 会先回到门禁页。所以得：先过口令 → 再看是不是在问密钥 → 再输密钥。
        print("\n⓪b 使用统计页（要密钥 9925）", flush=True)
        pg0.goto(URL.rstrip("/") + "/?view=stats", wait_until="domcontentloaded")

        def box_by_ph(pg, ph: str):
            loc = pg.locator(f'input[placeholder="{ph}"]')
            return loc.first if loc.count() else None

        for _ in range(60):                       # 等首屏画出来
            if pg0.locator('input[type="password"]').count():
                break
            pg0.wait_for_timeout(500)
        got_ph = None
        b = box_by_ph(pg0, "请输入访问口令")
        if b:                                    # 被拦在门禁 → 先过口令
            b.fill(CODE)
            pg0.get_by_role("button", name="进 入").click()
            for _ in range(60):
                pg0.wait_for_timeout(500)
                if not box_by_ph(pg0, "请输入访问口令"):
                    break
        b = box_by_ph(pg0, "请输入查看密钥")
        ck("统计页确实在问查看密钥（而不是还在门口）", b is not None,
           "没看到「请输入查看密钥」—— 可能还在门禁页")
        if b:
            b.fill(ADMIN)
            pg0.get_by_role("button", name="进 入").click()
            for _ in range(60):
                pg0.wait_for_timeout(500)
                if not box_by_ph(pg0, "请输入查看密钥"):
                    break
            got_ph = "查看密钥"
        body0 = pg0.inner_text("body")
        ck("统计页没崩", not page_crashed(pg0))
        ck("密钥认了（没再问密钥、也没说密钥不对）",
           got_ph == "查看密钥" and "密钥不对" not in body0, body0[-200:])
        # 判据要具体：看板要么给出数字，要么明说取不到；不能拿「次」这种字瞎蒙
        has_num = ("提问" in body0 and ("次" in body0 or "条" in body0))
        ck("看板有内容（有统计数字，或明说取不到）",
           has_num or "取不到统计" in body0, body0[-300:])
        pg0.screenshot(path=str(OUT / "g0b_stats.png"), full_page=True)
        pg0.close()

        # ---- 剩下的：在**另一页**上折腾坏输入，只看会不会崩 ----
        pg = br.new_page(viewport={"width": 1080, "height": 900})
        pg.goto(URL, wait_until="domcontentloaded")
        for _ in range(60):             # 等首屏画出来再动手（Streamlit 异步渲染）
            if at_gate(pg):
                break
            pg.wait_for_timeout(500)
        ck("新页也在门口", at_gate(pg))

        print("\n① 输汉字（用户切中文输入法时的真实场景）", flush=True)
        crash, _, inside = try_code(pg, "口令", "汉字「口令」")
        ck("页面没崩", not crash, crash)
        ck("也没放进去", not inside)
        pg.screenshot(path=str(OUT / "g1_chinese.png"), full_page=True)

        print("\n② 汉字夹在正确口令里", flush=True)
        crash, _, inside = try_code(pg, "SZPU666中", "混合")
        ck("页面没崩", not crash, crash)
        ck("也没放进去", not inside)

        print("\n③ emoji", flush=True)
        crash, _, inside = try_code(pg, "🎉🎉", "emoji")
        ck("页面没崩", not crash, crash)
        ck("也没放进去", not inside)

        print("\n④ 空口令", flush=True)
        crash, _, inside = try_code(pg, "", "空")
        ck("页面没崩", not crash, crash)
        ck("也没放进去", not inside)

        print("\n⑤ 普通的错口令（英文数字）", flush=True)
        crash, _, inside = try_code(pg, "wrong123", "错口令")
        ck("页面没崩", not crash, crash)
        ck("也没放进去", not inside)
        ck("连错后页面照样只是提示，没红字", not page_crashed(pg))

        br.close()

    if PROBLEMS:
        print(f"\n门口有 {len(PROBLEMS)} 处问题：{PROBLEMS}", flush=True)
    else:
        print("\n门口一切正常：坏输入只是「不对」，绝不崩页面 ✓", flush=True)
    return 1 if PROBLEMS else 0


if __name__ == "__main__":
    sys.exit(main())
