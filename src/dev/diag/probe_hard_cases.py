# -*- coding: utf-8 -*-
"""刁钻问法全流程抽检（2026-10-19）。

目的：**故意往坏里问**，看链路还稳不稳。分五组，每组都有明确判据 ——
不是"看起来还行"，而是能判红绿：

    A 换话题（带上文）      不能沾上一题的内容
    B 真追问（带上文）      必须靠上文接得上
    C 该拒的（手册真没有）   必须拒答，可接官网，但**不许拿别的条款硬套**
    D 该答的（口语 / 错字）  必须答出来，且答案里有手册原词
    E 个人数据             0 秒给引导话术，不许检索出内容

用法（要调模型，约 10~14 分钟）：
    python src/dev/diag/probe_hard_cases.py
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")
SRC = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(SRC))
import _paths  # noqa: E402,F401

from agent_qa import AgentQAPipeline          # noqa: E402
from qa_chain import build_llm, load_env_file  # noqa: E402
from retrievers import HandbookRetriever      # noqa: E402

FAIL = []


def ck(name: str, cond: bool, extra: str = ""):
    print(f"   {'✓' if cond else '✗'} {name}" + (f"　—— {extra}" if extra else ""),
          flush=True)
    if not cond:
        FAIL.append(name)


def brief(res, n: int = 70) -> str:
    a = (res.answer or "").replace("\n", " ").strip()
    if res.refused:
        tag = "拒答" + ("（官网）" if res.web else "")
    elif res.web_login:
        tag = "登录引导"
    else:
        tag = "作答"
    return f"{tag}｜{a[:n]}"


# ---- A：换话题（前一句一个主题，后一句完全不搭界）----
SWITCH = [
    ("国家奖学金怎么申请？", "宿舍能用吹风机吗？", ["奖学金"]),
    ("旷课多少学时会被处分？", "图书馆几点开门？", ["处分", "旷课"]),
]

# ---- B：真追问（第二句单看读不懂）----
FOLLOW = [
    ("学生公寓能不能用电磁炉？", "那吹风机呢？", ["违章电器", "吹风"]),
    ("补考成绩怎么算？", "那重修呢？", ["重修"]),
    ("旷课多少学时会被处分？", "那迟到早退算不算旷课？", ["迟到", "早退"]),
]

# ---- C：手册里真没有 → 必须拒（可以走官网，但不许套别的条款）----
MUST_REFUSE = [
    ("学校wifi密码是多少", ["公寓", "违章", "处分", "奖学金"]),
    ("图书馆几点开门？", ["6:00", "23:30", "公寓大门"]),
    ("食堂几点开门？", ["6:00", "23:30", "公寓大门"]),
    ("学校有几个校门？", ["公寓大门", "违章"]),
]

# ---- D：口语 / 错字 / 语气词 → 必须答得出来 ----
MUST_ANSWER = [
    ("翘课几次会被开除？", ["旷课", "开除", "处分"]),
    ("打小抄会怎么样？", ["作弊", "处分"]),
    ("宿舍能用电磁炉吗？", ["违章电器", "电磁炉"]),
    ("宿舍能用吹风机嘛？", ["违章电器"]),
]

# ---- E：问自己的数据 → 0 秒引导，不许检索 ----
MUST_LOGIN = ["我挂了几科？", "我这学期绩点多少？"]


def hist_of(q1: str, a1: str) -> str:
    return ("【之前的对话】\n"
            f"学生：{q1}\n"
            f"助手：{(a1 or '').replace(chr(10), ' ')[:600]}\n"
            "（以上是背景；请优先回答本次【问题】。若本次问题里的"
            "「它/这个/那」指代不明，按上面的对话理解。）")


def main() -> int:
    print("载入检索器与模型…", flush=True)
    load_env_file()
    r = HandbookRetriever("data", use_rerank=True, verbose=False)
    pipe = AgentQAPipeline(r, llm=build_llm(), verbose=False)
    t_all = time.time()

    print("\n[A] 换话题：不许沾上一题", flush=True)
    for i, (q1, q2, bad) in enumerate(SWITCH, 1):
        r1 = pipe.ask(q1)
        time.sleep(3)
        r2 = pipe.ask(q2, history_block=hist_of(q1, r1.answer), prev_user=q1)
        ans = r2.answer or ""
        polluted = [w for w in bad if w in ans]
        # A 组只测一件事：有没有沾上一题。这一题手册该答还是该拒，
        # 不归 A 管（吹风机手册确实有第二十九条管着，答出来才是对的）。
        ck(f"A{i} 「{q2}」", not polluted,
           ("沾到：" + "、".join(polluted)) if polluted else brief(r2))
        time.sleep(3)

    print("\n[B] 真追问：必须接得上", flush=True)
    for i, (q1, q2, good) in enumerate(FOLLOW, 1):
        r1 = pipe.ask(q1)
        time.sleep(3)
        r2 = pipe.ask(q2, history_block=hist_of(q1, r1.answer), prev_user=q1)
        ans = r2.answer or ""
        hit = [w for w in good if w in ans]
        ck(f"B{i} 「{q2}」", (not r2.refused) and bool(hit),
           ("命中：" + "、".join(hit)) if hit else brief(r2, 60))
        time.sleep(3)

    print("\n[C] 手册真没有：必须拒，不许套别的条款", flush=True)
    for i, (q, bad) in enumerate(MUST_REFUSE, 1):
        res = pipe.ask(q)
        ans = res.answer or ""
        polluted = [w for w in bad if w in ans]
        ck(f"C{i} 「{q}」", res.refused and not polluted,
           ("套了：" + "、".join(polluted)) if polluted else brief(res))
        time.sleep(3)

    print("\n[D] 口语 / 错字：必须答得出来", flush=True)
    for i, (q, good) in enumerate(MUST_ANSWER, 1):
        res = pipe.ask(q)
        ans = res.answer or ""
        hit = [w for w in good if w in ans]
        ck(f"D{i} 「{q}」", (not res.refused) and bool(hit),
           ("命中：" + "、".join(hit)) if hit else brief(res, 60))
        time.sleep(3)

    print("\n[E] 问自己的数据：0 秒引导，不许检索", flush=True)
    for i, q in enumerate(MUST_LOGIN, 1):
        t0 = time.time()
        res = pipe.ask(q)
        dt = time.time() - t0
        ck(f"E{i} 「{q}」", bool(res.web_login) and dt < 2.0,
           f"{dt:.2f}s ｜ {brief(res, 50)}")

    print("\n" + "=" * 62)
    print(f"总耗时 {time.time() - t_all:.0f}s")
    if FAIL:
        print(f"✗ 未通过 {len(FAIL)} 项：" + "；".join(FAIL))
        return 1
    print("✓ 全流程通过：换话题不带偏、追问接得上、该拒的拒、该答的答")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
