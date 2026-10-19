# -*- coding: utf-8 -*-
"""守门：主体一致闸装对了没有（2026-10-18）。

它只回答三个问题：

    A. 该拦的拦了吗？  —— 问图书馆/食堂/快递/超市…，证据里一个字都没提到，
                          必须判"答非所问"，别再拿《学生公寓》第十九条去套。
    B. 该放的放了吗？  —— 有主体支撑的（宿舍、公寓大门、图书馆赔偿）必须放行；
                          证据分够高的（勤工助学条款 0.4+）连闸门都不该启动。
    C. 60 题无一声误伤？ —— 这是硬红线。闸一（证据分 <0.08 才启动）的取值
                          就是按 60 题正常题最低分 0.1118 算出来的，
                          这里把它当断言再验一遍，改了阈值就会红。

不调大模型、不连网，只跑检索 + 判据，几十秒出结果。

用法：
    python src/dev/diag/check_subject_guard.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")
SRC = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(SRC))
import _paths  # noqa: E402,F401

from qa_chain import load_env_file  # noqa: E402
from retrievers import HandbookRetriever  # noqa: E402
from subject_guard import WEAK_TOP, check  # noqa: E402

ROOT = SRC.parent
FAIL = []


def ck(name: str, cond: bool, extra: str = ""):
    print(f"   {'✓' if cond else '✗'} {name}" + (f"　—— {extra}" if extra else ""),
          flush=True)
    if not cond:
        FAIL.append(name)


# ---- A：必须拦（问句主体在手册里根本没规定） ----
MUST_BLOCK = [
    "图书馆几点开门、几点关门？",
    "图书馆开放时间",
    "学校食堂几点开门？",
    "食堂的营业时间",
    "快递站在哪、几点关门？",
    "超市晚上几点关门",
    "体育馆要预约吗",
]

# ---- B：必须放行（主体在证据里真有，或者证据分足够高） ----
MUST_PASS = [
    "学生公寓大门几点关？",
    "宿舍里能用电磁炉、电饭锅吗？",
    "宿舍可以养宠物吗",
    "图书馆的书弄丢了怎么赔偿",
    "在食堂勤工助学有工时限制吗",
]


def main() -> int:
    load_env_file()
    r = HandbookRetriever(str(ROOT / "data"), use_rerank=True, verbose=False)
    print(f"\n编码后端 = {r.backend}　重排 = {r.use_rerank}　"
          f"降级 = {r.degraded or '无'}")
    print(f"弱证据线 WEAK_TOP = {WEAK_TOP}\n")

    def run(q, k=6):
        evs = r.search(q, top_k=k)
        top1 = max((e.score or 0.0) for e in evs) if evs else 0.0
        return evs, top1, check(q, evs)

    print("A. 该拦的（主体在证据里完全没有）", flush=True)
    for q in MUST_BLOCK:
        evs, top1, hit = run(q)
        ck(f"{q[:22]:<24} top1={top1:.4f}", hit is not None,
           (hit["reason"] if hit else "★没拦住 —— 会被模型拿去答非所问"))

    print("\nB. 该放的（主体有支撑 / 证据够强）", flush=True)
    for q in MUST_PASS:
        evs, top1, hit = run(q)
        ck(f"{q[:22]:<24} top1={top1:.4f}", hit is None,
           (f"★被误拦：{hit}" if hit else "放行"))

    print("\nC. 60 题全量回归（硬红线：一道真题都不许被拦）", flush=True)
    items = json.load(open(ROOT / "data" / "eval_set.json",
                           encoding="utf-8"))["items"]
    gated, blocked = [], []
    for it in items:
        q = it["question"]
        evs = r.search(q, top_k=6)
        top1 = max((e.score or 0.0) for e in evs) if evs else 0.0
        hit = check(q, evs)
        if top1 < WEAK_TOP:
            gated.append((it["qid"], top1, bool(it.get("refuse_expected"))))
        if hit:
            blocked.append((it["qid"], top1, hit["subject"], hit["reason"],
                            bool(it.get("refuse_expected"))))
    print(f"   60 题里证据分低于 {WEAK_TOP} 的（闸门启动面）：{len(gated)} 道")
    for qid, t, ref in gated:
        print(f"      {qid} top1={t:.4f}　{'该拒（负例题）' if ref else '★真题'}")
    # 红线不是「一道都不拦」，而是「**真题**一道都不许拦」——
    # 手册里真没有的题被拦下来（还省了两次模型调用）是收益，不是损失。
    wrong = [b for b in blocked if not b[4]]
    ck(f"真题零误拦（被拦 {len(blocked)} 道，其中真题 {len(wrong)} 道）", not wrong)
    for qid, t, subj, reason, ref in blocked:
        print(f"      {'✓' if ref else '★'} {qid} top1={t:.4f} 主体={subj}"
              f"　{'本来就是该拒的题' if ref else '真题被误拦！'}")
    print("   说明：60 题凡是落到闸门启动面的，都已经被 0.03 的拒答前置先拦下了，\n"
          "        所以本闸在评测集上**零增量**——它补的是 0.03~0.08 那段漏网区间\n"
          "        （「图书馆几点开门」top1=0.0349 正是从这儿漏过去的）。", flush=True)

    print("\n" + ("有问题，看上面 ✗" if FAIL else "全项通过"), flush=True)
    return 1 if FAIL else 0


if __name__ == "__main__":
    sys.exit(main())
