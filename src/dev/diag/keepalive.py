# -*- coding: utf-8 -*-
"""保活：定时打开线上页面，别让 Streamlit Community Cloud 把应用睡过去。

为什么要它
----------
Community Cloud 的应用在**长时间没人访问**时会休眠；学生隔半天再打开，
先要等它被唤醒（几十秒的「连不上/正在启动」）。每 5 小时碰它一下就够了。

为什么不是「curl 一下地址」
--------------------------
Cloud 把应用放在 iframe 里跑，真正建立会话的是浏览器加载后那串
**websocket 连接**。只拿回一个 HTML 壳，几乎不被算作「有人在用」。
所以这里用**真浏览器**打开 → 等首屏渲染出来（说明 websocket 已连上、
应用已在跑）→ 再挂 75 秒，最后关掉。

用法
----
    python src/dev/diag/keepalive.py                      # 线上地址 + 保持 75 秒
    python src/dev/diag/keepalive.py <地址> <保持秒数>
    KEEPALIVE_HOLD=120 python src/dev/diag/keepalive.py    # 也可用环境变量

退出码：0 ＝ 页面真的跑起来了（保活成功）；1 ＝ 没起来（附原因）。
每次运行都往 `logs/keepalive.log` 追一行，便于事后看「哪段时间没保上」。
"""
from __future__ import annotations

import sys
import time
from datetime import datetime
from pathlib import Path

from playwright.sync_api import sync_playwright

from _browser import content_view      # 穿透线上 Cloud 的 iframe，见该模块 docstring

URL = sys.argv[1] if len(sys.argv) > 1 else \
    "https://szpustudent-handbook-ai-agent.streamlit.app/"
HOLD = int(sys.argv[2]) if len(sys.argv) > 2 else 75
ROOT = Path(__file__).resolve().parents[3]
LOG = ROOT / "logs" / "keepalive.log"

# 首屏最多等多久（秒）。120 秒不够用（2026-10-18 踩到）：那天刚改完 Secrets，
# Cloud 正在重建，两次都卡在 130 秒（＝120 秒等满 + 打开页面的时间）报 FAIL，
# 于是被误判成「网站坏了、要登录」。实测正常首屏 9~20 秒，冷唤醒 40 秒上下，
# 但**重建期间**会超过两分钟 —— 给到 240 秒，宁可多等也别误报。
FIRST_LIMIT = 240


def write_log(msg: str) -> None:
    LOG.parent.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with LOG.open("a", encoding="utf-8") as f:
        f.write(f"{stamp}  {msg}\n")


def alive(view) -> bool:
    """应用还在跑吗 —— 有输入框（口令框或问答框）就算在。"""
    try:
        return bool(view.locator('input[type="password"]').count()
                    or view.locator('[data-testid="stTextInput"] input').count())
    except Exception:                  # frame 正被替换
        return False


def attempt(br, limit=FIRST_LIMIT):
    """开一次页面并等首屏画出来。返回 (view, page, 耗时秒, 错误说明)。"""
    pg = br.new_page(viewport={"width": 1080, "height": 900})
    t0 = time.time()
    try:
        pg.goto(URL, wait_until="domcontentloaded", timeout=90_000)
    except Exception as e:                       # 网络类
        return None, pg, time.time() - t0, \
            f"打开失败（{type(e).__name__}: {str(e)[:120]}）"
    view = content_view(pg, limit=limit)
    first = time.time() - t0
    if not alive(view):
        return None, pg, first, f"等满 {limit}s 仍未渲染出输入框"
    return view, pg, first, ""


def main() -> int:
    print(f"保活地址：{URL}", flush=True)
    with sync_playwright() as p:
        br = p.chromium.launch(channel="chrome", headless=True,
                               args=["--disable-gpu"])
        t0 = time.time()
        view, pg, first, err = attempt(br)
        if not view:
            # 重建 / 冷唤醒期间会短暂不可用，隔 20 秒再看一眼（只重试一次）。
            # 不重试的话，一次赶巧就写进一条 FAIL，事后排查要多花一晚上。
            (ROOT / ".trash_tmp").mkdir(exist_ok=True)
            try:
                pg.screenshot(path=str(ROOT / ".trash_tmp" / "keepalive_fail.png"),
                              full_page=True)
                pg.close()
            except Exception:                    # noqa: BLE001
                pass
            print(f"  第一次没起来（{err}），20 秒后重试一次……", flush=True)
            write_log(f"RETRY {err}")
            time.sleep(20)
            view, pg, first, err = attempt(br)
        if not view:
            try:
                pg.screenshot(path=str(ROOT / ".trash_tmp" / "keepalive_fail.png"),
                              full_page=True)
            except Exception:                    # noqa: BLE001
                pass
            br.close()
            msg = f"{err}（已重试一次）"
            print(f"✗ {msg}", flush=True)
            write_log(f"FAIL {msg}")
            return 1
        print(f"  首屏渲染完成：{first:.1f}s", flush=True)

        # 挂着不动。中间轻推两下（滚动一下），让会话保持有交互。
        for i in range(HOLD):
            pg.wait_for_timeout(1000)
            if i and i % 20 == 0:
                try:
                    view.evaluate("window.scrollBy(0, 60)")
                except Exception:
                    pass
        still = alive(content_view(pg))
        total = time.time() - t0
        br.close()
        if not still:
            msg = f"挂了 {HOLD}s 后页面丢了（首屏 {first:.1f}s，总 {total:.1f}s）"
            print(f"✗ {msg}", flush=True)
            write_log(f"FAIL {msg}")
            return 1
        print(f"✓ OK　首屏 {first:.1f}s，保持 {HOLD}s，总 {total:.1f}s", flush=True)
        write_log(f"OK 首屏 {first:.1f}s 保持 {HOLD}s 总 {total:.1f}s")
        return 0


if __name__ == "__main__":
    sys.exit(main())
