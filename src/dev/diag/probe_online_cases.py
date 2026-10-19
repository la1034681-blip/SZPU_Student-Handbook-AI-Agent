# -*- coding: utf-8 -*-
"""线上批量抽检：连问几道真题，逐题看「答没答、降没降级、拒得对不对」。

为什么要有这个（2026-10-17）
------------------------------
线上 Secrets 从硅基流动换成模力方舟之后，光验证「钥匙能连上」不够 ——
要确认**页面上的那条黄条真的不见了**。黄条＝`plan["degraded_retrieval"]`，
只在向量检索/重排调用失败时才出现。所以抽检要看三件事：

    ① 每题都出得来答复          —— 不是卡住、不是白页
    ② 没有「本次检索打了折扣」  —— 说明新的编码接口真的顶上了
    ③ 该拒的拒、该答的答        —— 拒答话术统一为「本手册中没有对应内容」；
                                  技术故障走「服务暂时出了点小状况」（两者不一样）

和 `check_full_flow.py` 的分工：那个走「进门→留言→刷新→自检」的全流程，
只问 1 题；这个不看流程，只看**多道题的质量**。两个都跑才叫完整。

用法：
    python src/dev/diag/probe_online_cases.py [地址]
默认线上地址。传 `local` 就跑本机 8501。

⚠ 别把它当成评测：这里只看「页面表现」，不看答案对不对（那要 `evaluate_generation.py`）。
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

from playwright.sync_api import sync_playwright

from _browser import content_view   # 穿透线上 Cloud 的 iframe，见该模块 docstring

# 访问口令**不许写死在这个文件里**（2026-10-19 修）：仓库是公开的，写死等于把
# 口令贴到互联网上。改成从环境变量读，本地由项目根 `.env` 提供（该文件已被
# .gitignore 排除）。缺了就明确报错退出，别拿空口令去点「进入」然后误判成"进不去"。
sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "src"))
try:
    import _paths                     # noqa: E402,F401
    from qa_chain import load_env_file  # noqa: E402
    load_env_file()
except Exception:                     # noqa: BLE001  # 环境不完整也不该让脚本直接崩
    pass

CODE = (os.environ.get("ACCESS_CODE") or "").strip()
if not CODE:
    print("缺访问口令：在项目根 .env 里写 ACCESS_CODE=xxx（或 set ACCESS_CODE=xxx）再跑。")
    sys.exit(1)

DEFAULT_URL = "https://szpustudent-handbook-ai-agent.streamlit.app/"
arg = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_URL
URL = "http://localhost:8501/" if arg == "local" else arg

OUT = Path(__file__).resolve().parents[3] / ".trash_tmp" / "layout_shots"
OUT.mkdir(parents=True, exist_ok=True)

# 抽检题：覆盖「补考 / 处分 / 宿舍 / 学业预警 / 学位」五类，
# 最后加一道手册里**没有**的，专测拒答话术。
CASES = [
    ("Q01 补考", "必修课考试不及格，能补考吗？补考成绩怎么算？"),
    ("Q04 旷课", "旷课太多会有什么后果？"),
    ("Q07 宿舍电器", "宿舍里能用电磁炉、电饭锅吗？"),
    ("Q09 学业预警", "学业预警分几个等级？"),
    ("Q10 学位", "拿学士学位需要满足什么条件？"),
    ("N01 手册外", "学校食堂几点开门？"),
    # ⚠ 这道是 2026-10-17 线上抽检抓到的**真错答**：手册里没有图书馆开放时间，
    #   但「学生公寓大门开放时间为每天 6:00 至 23:30」这条句式太像，模型直接
    #   把「图书馆」代进句式，答成「图书馆 6:00–23:30」。它的 top1 重排分 0.0347，
    #   刚好越过 0.03 的拒答前置阈值 → 必须留在探针里，别再让它溜过去。
    ("N02 图书馆", "图书馆几点开门、几点关门？"),
]

# 可选第二个参数：只跑哪几道（按 tag 里的关键词匹配），冷却后补测单题用。
#   python src/dev/diag/probe_online_cases.py <地址> N01
PICK = [s.strip().lower() for s in (sys.argv[2].split(",") if len(sys.argv) > 2 else [])
        if s.strip()]
if PICK:
    CASES = [c for c in CASES if any(p in c[0].lower() for p in PICK)]

RESULTS: list[dict] = []


def ck(name: str, cond: bool, extra: str = ""):
    print(f"   {'✓' if cond else '✗'} {name}" + (f"　—— {extra}" if extra else ""),
          flush=True)


def pass_gate(view) -> bool:
    if not view.locator('input[type="password"]').count():
        return bool(view.locator('[data-testid="stTextInput"] input').count())
    for _ in range(3):
        box = view.locator('input[type="password"]').first
        box.fill(CODE)
        if box.input_value() != CODE:
            view.wait_for_timeout(800)
            continue
        view.get_by_role("button", name="进 入").click()
        for _ in range(120):
            view.wait_for_timeout(500)
            if not view.locator('input[type="password"]').count():
                return True
    return False


def wait_ready(view, limit=120):
    """等一次脚本运行结束（running 标记消失），Streamlit 是异步的。"""
    view.wait_for_timeout(1500)
    for _ in range(limit * 2):
        if view.locator('[data-testid="stStatusWidget"]').count() == 0:
            return True
        view.wait_for_timeout(500)
    return False


def last_answer(view) -> str:
    """取最后一条助手消息的正文。"""
    msgs = view.locator('[data-testid="stChatMessage"]')
    if msgs.count() == 0:
        return ""
    try:
        return msgs.nth(msgs.count() - 1).inner_text()
    except Exception:
        return ""


def ask(view, q: str, timeout_s: int = 300) -> tuple[str, float, bool]:
    """问一句，等**新的**答复出现，返回 (答复正文, 耗时秒, 是否真的拿到了新答复)。

    ⚠ 判据必须是「正文变了」，不能只看「消息条数变多」——2026-10-17 踩过：
      免费模型被连问逼到限流时，一题可能要 3~4 分钟，脚本等到 240 秒超时后
      回退去读「最后一条消息」，读到的却是**上一题**的正文（连「耗时 135.6 秒」
      都一模一样），于是把超时误判成「答好了」——最会骗人的一种假通过。
    """
    prev = last_answer(view)
    t0 = time.time()
    box = view.locator('[data-testid="stTextInput"] input').first
    box.fill(q)
    box.press("Enter")
    for _ in range(timeout_s):
        view.wait_for_timeout(1000)
        txt = last_answer(view)
        if txt and txt != prev and ("耗时" in txt or "服务暂时" in txt
                                    or "本手册中没有对应内容" in txt):
            return txt, time.time() - t0, True
    return last_answer(view), time.time() - t0, False


def main() -> int:
    print(f"\n抽检地址：{URL}\n", flush=True)
    with sync_playwright() as p:
        br = p.chromium.launch(channel="chrome", headless=True,
                               args=["--disable-gpu"])
        pg = br.new_page(viewport={"width": 1080, "height": 1000})

        print("① 进门", flush=True)
        t0 = time.time()
        pg.goto(URL, wait_until="domcontentloaded")
        view = content_view(pg, limit=180)      # Cloud 冷启动可能要几十秒
        ck("门口画出来了", view is not pg.main_frame or
           view.locator('input').count() > 0, f"用了 {time.time()-t0:.1f} 秒")
        ok = pass_gate(view)
        ck("进得去", ok)
        wait_ready(view)
        view = content_view(pg)
        body = view.inner_text("body")
        ck("没有红字报错", "出错了" not in body and "Traceback" not in body)
        inside = (view.locator('input[type="password"]').count() == 0
                  and "问答" in body)
        ck("确实站在问答页上", inside)
        if not inside:
            pg.screenshot(path=str(OUT / "cases_stuck.png"), full_page=True)
            br.close()
            print("\n没进到问答页，后面没意义。截图：", OUT, flush=True)
            return 1
        print(f"   进门用时 {time.time()-t0:.1f} 秒\n", flush=True)

        print(f"② 连问 {len(CASES)} 道", flush=True)
        for tag, q in CASES:
            txt, dt, ok = ask(view, q)
            degraded = ok and "打了折扣" in txt
            refused = ok and "本手册中没有对应内容" in txt
            tech = ok and "服务暂时出了点小状况" in txt
            has_src = ok and "手册原文" in txt
            ck(f"{tag}　{dt:.1f}s", ok,
               "" if ok else "超时：等了这么久还没出答复（大概率是免费模型被限流）")
            if ok:
                flags = []
                if has_src:
                    flags.append("有原文")
                if refused:
                    flags.append("拒答（手册外）")
                if degraded:
                    flags.append("★降级黄条")
                if tech:
                    flags.append("★★技术故障话术")
                print(f"       {'、'.join(flags) if flags else '答复无原文卡片'}",
                      flush=True)
            RESULTS.append({"tag": tag, "q": q, "dt": dt, "ok": ok,
                            "degraded": degraded, "refused": refused,
                            "tech": tech, "has_src": has_src,
                            "text": txt if ok else ""})
            pg.screenshot(path=str(OUT / f"cases_{tag.split()[0]}.png"),
                          full_page=True)

        # ---------------------------------------------------------- 汇总
        print("\n③ 汇总", flush=True)
        n = len(RESULTS)
        n_ok = sum(r["ok"] for r in RESULTS)
        n_deg = sum(r["degraded"] for r in RESULTS)
        n_tech = sum(r["tech"] for r in RESULTS)
        n_src = sum(r["has_src"] for r in RESULTS)
        # 负例＝tag 以 N 开头的（手册里根本没这回事）；其余都算真题。
        negs = [r for r in RESULTS if r["tag"].startswith("N")]
        reals = [r for r in RESULTS if not r["tag"].startswith("N")]
        n_ref = sum(r["refused"] for r in negs)
        n_real = sum(1 for r in reals if r["ok"])
        ck(f"{n} 题都答出来了（{n_ok}/{n}）", n_ok == n)
        ck(f"没有降级黄条（{n_deg} 次）", n_deg == 0,
           "还有黄条 ⇒ 线上 Secrets 没生效或接口又挂了")
        ck(f"没有技术故障话术（{n_tech} 次）", n_tech == 0)
        ck(f"真题都给了手册原文（{n_src}/{len(reals)}）", n_src >= n_real,
           "真题不给原文 ⇒ 可能被误拒")
        if negs:
            ck(f"手册外的问题都拒答（{n_ref}/{len(negs)}）", n_ref == len(negs),
               "答出来就是错答 —— 见 N02 图书馆那条的注释")
        tt = [r["dt"] for r in RESULTS]
        if tt:
            print(f"   单题耗时：{min(tt):.1f} ~ {max(tt):.1f} 秒，"
                  f"平均 {sum(tt)/len(tt):.1f} 秒", flush=True)

        log = OUT / "online_cases.json"
        # 与上一次的结果**合并**（同 tag 覆盖），别整份覆盖 ——
        # 否则冷却后拿 `PICK` 补测 1 道，就会把上一次 6 道的正文全冲掉。
        old = []
        if log.exists():
            try:
                old = json.loads(log.read_text(encoding="utf-8"))
            except Exception:
                old = []
        seen = {r["tag"]: r for r in old if isinstance(r, dict) and "tag" in r}
        for r in RESULTS:
            seen[r["tag"]] = r
        log.write_text(json.dumps(list(seen.values()), ensure_ascii=False, indent=2),
                       encoding="utf-8")
        print(f"   逐题正文存到：{log}（累计 {len(seen)} 条）", flush=True)
        br.close()

    bad = (n_ok < n or n_deg or n_tech or n_src < n_real
           or (negs and n_ref < len(negs)))
    print("\n" + ("有问题，逐条看上面 ✗" if bad else "全项通过"), flush=True)
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
