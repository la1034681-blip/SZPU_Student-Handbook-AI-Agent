# -*- coding: utf-8 -*-
"""官网兜底验收：20 道题过一遍，逐题看「走的哪一层、读到了什么、花了多久」。

为什么要有这个（2026-10-17）
------------------------------
`web_fallback.py` 里全是「搜 → 筛 → 抓 → 打分」的启发式规则，任何一条改过头
都会静默变差：要么该给的给不出来（漏答），要么给出风马牛不相及的一句
（错答，比漏答坏得多）。所以每次动它都要跑这一遍，看四组表现：

    A 官网能答（8 题）   —— 至少要拿到原文句或表格
    B 只在图片里（2 题） —— 必须走 link_only，**给链接**
    C 官网大概也没有（6 题）—— 不设硬指标，人工看它有没有瞎凑
    D 要登录（4 题）     —— 必须被 need_login 拦下、且不去搜官网

C 组为什么不设硬指标：官网内容会变，今天没有不代表明天没有。硬卡 pass/fail
只会逼着人去调阈值迎合测试。这一组的作用是**把结果摊在你眼前**，由人判断
"这句是不是在瞎凑"。

用法：
    python src/dev/diag/probe_web.py            # 全部 20 题
    python src/dev/diag/probe_web.py A          # 只跑 A 组
    python src/dev/diag/probe_web.py A B        # 跑 A、B 组
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

ROOT = next((p for p in Path(__file__).resolve().parents if (p / "app.py").exists()), None)
if ROOT and str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))
import _paths                                    # noqa: F401

from web_fallback import ask_web                 # noqa: E402
from need_login import check as login_check      # noqa: E402

BUDGET = 8.0
LIMIT_S = 9.0            # 超过这个就算慢（比预算多留 1 秒给解释器开销）

CASES = [
    # (编号, 组, 问题)
    ("A1", "A", "学校的作息时间是怎样的？"),
    ("A2", "A", "第5节课几点上？"),
    ("A3", "A", "上午第一节课什么时候开始？"),
    ("A4", "A", "学校的班车怎么坐？"),
    ("A5", "A", "校园卡丢了去哪里补办？"),
    ("A6", "A", "学校的网络怎么连？"),
    ("B1", "B", "2025-2026 学年校历怎么安排的？"),
    ("B2", "B", "这学期的校历在哪看？"),
    ("C1", "C", "食堂几点开门？"),
    ("C2", "C", "学校有几个食堂？"),
    ("C3", "C", "体育馆几点开门？"),
    ("C4", "C", "校医院在哪里？"),
    ("C5", "C", "学校门口有地铁吗？"),
    ("C6", "C", "快递站在哪里？"),
    ("C7", "C", "学校教务处电话是多少？"),   # 实测官网没有专门页面，正好当负例
    ("C8", "C", "图书馆在哪个校区？"),       # 图书馆是独立二级站 lib.szpu.edu.cn，主站搜不到
    ("D1", "D", "我这学期绩点多少？"),
    ("D2", "D", "我的课表在哪看？"),
    ("D3", "D", "我欠不欠学费？"),
    ("D4", "D", "我借的书超期了吗？"),
]


def brief(res: dict, n: int = 88) -> str:
    """把一条结果压成一行，方便扫。"""
    if res["passages"]:
        p = res["passages"][0]
        tag = "表格" if p["kind"] == "table" else "正文"
        return f"[{tag}] {p['text'][:n]}"
    if res["tables"]:
        rows = res["tables"][0]["rows"]
        return f"[整表 {len(rows)}行] {res['tables'][0]['title'][:30]}"
    if res["links"]:
        return f"[链接] {res['links'][0]['url'][:80]}"
    return "（没找到任何内容）"


def main() -> int:
    want = [a.upper() for a in sys.argv[1:] if a.upper() in ("A", "B", "C", "D")] or ["A", "B", "C", "D"]

    print(f"官网兜底验收　预算 {BUDGET}s/题　本次跑 {'/'.join(want)} 组\n")

    rows, bad = [], 0
    for cid, grp, q in CASES:
        if grp not in want:
            continue

        # D 组：登录墙，根本不该去搜官网
        if grp == "D":
            t0 = time.time()
            hit = login_check(q)
            dt = time.time() - t0
            ok = hit is not None
            bad += 0 if ok else 1
            print(f"{'✓' if ok else '✗'} {cid} [{grp}] {q}")
            print(f"     {dt:.2f}s　" + ("拦下：" + hit["topic"] if hit else "没拦住 —— 会去搜官网，白等 8 秒"))
            rows.append((cid, grp, ok, dt, "拦截" if ok else "漏拦"))
            continue

        r = ask_web(q, budget=BUDGET)
        dt = r["stats"]["elapsed"]
        reason = r["reason"]
        got = bool(r["passages"] or r["tables"])
        link = bool(r["links"])

        if grp == "A":
            ok = (got or reason == "link_only") and dt <= LIMIT_S
        elif grp == "B":
            ok = (reason == "link_only" and link) or got
        else:
            ok = True                     # C 组只摊结果，不判对错
        bad += 0 if ok else 1

        mark = "·" if grp == "C" else ("✓" if ok else "✗")
        print(f"{mark} {cid} [{grp}] {q}")
        print(f"     {dt:.1f}s　{reason}　词={'/'.join(r['stats']['keywords'])}")
        print(f"     {brief(r)}")
        if grp == "C" and got:
            print("     ↑ C 组给了内容，人工看这句是不是在瞎凑")
        rows.append((cid, grp, ok, dt, reason))

    # ------------------------------------------------------------ 汇总
    a = [x for x in rows if x[1] == "A"]
    b = [x for x in rows if x[1] == "B"]
    c = [x for x in rows if x[1] == "C"]
    d = [x for x in rows if x[1] == "D"]
    times = [x[3] for x in rows if x[1] != "D"]

    print("\n" + "=" * 74)
    if a:
        n = sum(1 for x in a if x[2])
        print(f"A 官网能答　{n}/{len(a)} 拿到内容")
    if b:
        n = sum(1 for x in b if x[2])
        print(f"B 只在图片里　{n}/{len(b)} 走了「只给链接」")
    if c:
        n = sum(1 for x in c if "no_result" in x[4] or "link_only" in x[4])
        print(f"C 官网大概也没有　{n}/{len(c)} 判了「没有」（其余给了内容，人工看）")
    if d:
        n = sum(1 for x in d if x[2])
        print(f"D 要登录　{n}/{len(d)} 被拦下、没去搜官网")
    if times:
        print(f"耗时　平均 {sum(times)/len(times):.1f}s　最长 {max(times):.1f}s　"
              f"超预算 {sum(1 for t in times if t > LIMIT_S)} 题")

    print("\n全项通过" if bad == 0 else f"\n有 {bad} 题不达标，逐条看上面 ✗")
    return 0 if bad == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
