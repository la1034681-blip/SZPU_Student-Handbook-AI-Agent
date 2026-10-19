# -*- coding: utf-8 -*-
"""线上把降级黄条背后的「技术细节」抓出来 —— 到底哪个接口、报了什么错。

为什么要有这个（2026-10-19）
------------------------------
页面上的黄条只说「智能匹配那一环没能连上」，具体报错藏在
「技术细节（给管理员）」那个折叠块里。不把它抽出来，就只能猜：
是密钥错（401）、是额度用完（402）、是地址写错，还是网络层读超时？
四种的修法完全不同 —— 401 换钥匙、402 换模型或充值、读超时换服务商。

用法：
    python src/dev/diag/probe_degrade_reason.py [地址]
默认线上地址；传 local 跑本机 8501。口令从 .env 的 ACCESS_CODE 读。
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

from playwright.sync_api import sync_playwright

from _browser import content_view

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "src"))
try:
    import _paths  # noqa: E402,F401
    from qa_chain import load_env_file  # noqa: E402
    load_env_file()
except Exception:  # noqa: BLE001
    pass

CODE = (os.environ.get("ACCESS_CODE") or "").strip()
if not CODE:
    print("缺访问口令：.env 里写 ACCESS_CODE=xxx 再跑。")
    sys.exit(1)

DEFAULT_URL = "https://szpustudent-handbook-ai-agent.streamlit.app/"
# 用法：脚本 [地址|local] [问题1] [问题2] …
# 不给问题就用下面这两道默认题（一道会试重排、一道不会）。
_args = sys.argv[1:]
URL = DEFAULT_URL
_qs: list[str] = []
for _a in _args:
    if _a.startswith("http"):
        URL = _a
    elif _a == "local":
        URL = "http://localhost:8501/"
    else:
        _qs.append(_a)

QUESTIONS = _qs or ["补考怎么算成绩", "旷课多少学时会被处分"]


def last_answer(view) -> str:
    msgs = view.locator('[data-testid="stChatMessage"]')
    if msgs.count() == 0:
        return ""
    try:
        return msgs.nth(msgs.count() - 1).inner_text()
    except Exception:
        return ""


def main() -> int:
    print(f"\n地址：{URL}\n", flush=True)
    with sync_playwright() as p:
        br = p.chromium.launch(channel="chrome", headless=True,
                              args=["--disable-gpu"])
        pg = br.new_page(viewport={"width": 1080, "height": 1000})
        pg.goto(URL, wait_until="domcontentloaded")
        view = content_view(pg, limit=180)
        print("✓ 首屏出来了", flush=True)

        # 过口令门
        pw = view.locator('input[type="password"]')
        if pw.count():
            pw.first.fill(CODE)
            pw.first.press("Enter")
            for _ in range(120):
                view.wait_for_timeout(500)
                if not view.locator('input[type="password"]').count():
                    break
            print("✓ 进站", flush=True)

        box = view.locator('[data-testid="stTextInput"] input').first
        for i, q in enumerate(QUESTIONS, 1):
            prev = last_answer(view)
            print(f"\n=== 第 {i} 题：{q} ===", flush=True)
            t0 = time.time()
            box.fill(q)
            box.press("Enter")
            for _ in range(300):
                view.wait_for_timeout(1000)
                txt = last_answer(view)
                if txt and txt != prev and "耗时" in txt:
                    break
            print(f"耗时 {time.time() - t0:.1f}s", flush=True)

            whole = last_answer(view)
            # 先把答复正文抄一段出来 —— 官网兜底有没有命中、拒答话术对不对，
            # 都只能从正文里看出来，光看黄条是不够的。
            print("──── 答复正文 ────", flush=True)
            print(whole[:900], flush=True)
            print("──────────────────", flush=True)

            if "打了折扣" not in whole:
                print("（本题没有降级黄条 —— 编码接口这次是通的）", flush=True)
                continue

            # 点开「技术细节（给管理员）」，把里面的字抄出来
            found = False
            for det in view.locator("details").all():
                try:
                    head = det.inner_text()[:40]
                except Exception:
                    continue
                if "技术细节" in head:
                    try:
                        det.locator("summary").first.click()
                        view.wait_for_timeout(600)
                    except Exception:
                        pass
                    print("──── 技术细节 ────", flush=True)
                    print(det.inner_text().strip(), flush=True)
                    print("──────────────────", flush=True)
                    found = True
                    break
            if not found:
                print("（黄条在，但没找到「技术细节」折叠块）", flush=True)
                print(whole[:800], flush=True)

        br.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
