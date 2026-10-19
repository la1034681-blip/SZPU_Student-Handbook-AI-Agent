# -*- coding: utf-8 -*-
"""守门：换了新话题，不许被上一题带偏（2026-10-19）。

起因（线上真实撞到，本地稳定复现）
--------------------------------
学生先问「旷课多少学时会被处分」，紧接着问「学校的班车怎么坐」。
第一次答出来的是一整页**处分细则**；同一句单独问，却正常拒答。

根子在「追问兜底」那一路：planner 没补出指代时，会把「上一句 + 本句」拼成
一路去检索，靠上一句的词补「它 / 那」。可学生**根本没在追问**，只是换了
话题 —— 上一句的「旷课 / 处分 / 学时」把《违纪处分办法》顶到 0.986 分，
闸门放行，模型照抄。空上文时同一句只有 0.0008 分。

修法：只认承接词（它 / 那 / 呢 / 还有 / 刚才…）才当追问；换了话题就只用
原句检索。本脚本回答三个问题：

    A. 换话题的题，还答成上一题的内容吗？  —— 5 组两轮，端到端。
    B. 真追问还接得上吗？                  —— 3 组两轮，端到端。
    C. 没有上文时判据一律不出手？          —— 纯函数，瞬间出结果
                                              （这是「60 题不受影响」的根据）。

用法（要调模型，约 6~8 分钟，撞 429 会自动重试）：
    python src/dev/diag/check_followup_guard.py
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")
SRC = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(SRC))
import _paths  # noqa: E402,F401

from agent_qa import AgentQAPipeline, _is_followup  # noqa: E402
from qa_chain import build_llm, load_env_file      # noqa: E402
from retrievers import HandbookRetriever           # noqa: E402

FAIL = []


def ck(name: str, cond: bool, extra: str = ""):
    print(f"   {'✓' if cond else '✗'} {name}" + (f"　—— {extra}" if extra else ""),
          flush=True)
    if not cond:
        FAIL.append(name)


# ---- A：换话题（前一句一个主题，后一句完全不搭界；后一句手册里本来没有）----
NEW_TOPIC = [
    ("旷课多少学时会被处分、分别是什么处分？迟到早退算不算旷课？",
     "学校的班车怎么坐？", ["处分", "旷课", "警告"]),
    ("国家奖学金怎么申请？", "图书馆几点开门？", ["奖学金"]),
    ("学生公寓能不能用电磁炉？", "学校食堂几点开门？", ["电磁炉"]),
    ("补考成绩怎么算？", "学校有几个校门？", ["补考"]),
    ("转专业要什么条件？", "校医院在哪？", ["转专业"]),
]

# ---- B：真追问（第二句自己读不懂，必须靠上一句）----
FOLLOWUP = [
    ("旷课多少学时会被处分？", "那迟到早退算不算旷课？", ["迟到", "早退"]),
    ("补考成绩怎么算？", "那重修呢？", ["重修"]),
    # ⚠ 这里原本用「那吹风机呢？」—— 手册全文「吹风」0 次、清单里只有
    # 电热棒/电水壶/电热毯/电热锅/电饭锅/微波炉/电磁炉…，没有吹风机。
    # 所以它拒答是**对的**（宁可拒，也别拿电磁炉那条去套）。要让它答得上，
    # 得先给 term_alias 加「吹风 → 违章电器」，属另一件事，别混在这里。
    ("学生公寓能不能用电磁炉？", "那电饭锅呢？", ["电饭锅", "电饭"]),
]


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

    # ---------- C：无上文时判据恒不出手（60 题行为不变的保证）----------
    print("\n[C] 没有上文 → 一律当新话题（行为与改动前完全一致）", flush=True)
    samples = ["学校的班车怎么坐？", "那迟到早退算不算旷课？",
               "它需要什么条件？", "重修呢", "宿舍几点关门？"]
    ck("5 个样本在无上文时全部判「不是追问」",
       all(not _is_followup(q, "") for q in samples))
    ck("有上文时「那迟到早退算不算旷课？」判为追问",
       _is_followup("那迟到早退算不算旷课？", "旷课多少学时会被处分？"))
    ck("有上文时「学校的班车怎么坐？」判为换话题",
       not _is_followup("学校的班车怎么坐？", "旷课多少学时会被处分？"))

    # ---------- A：换话题不许被带偏 ----------
    print("\n[A] 换新话题：不许答成上一题的内容", flush=True)
    for i, (q1, q2, bad) in enumerate(NEW_TOPIC, 1):
        r1 = pipe.ask(q1)
        hist = hist_of(q1, r1.answer)
        time.sleep(3)
        r2 = pipe.ask(q2, history_block=hist, prev_user=q1)
        ans = r2.answer or ""
        polluted = [w for w in bad if w in ans]
        ok = (r2.refused or bool(r2.web)) and not polluted
        ev = "拒答" if r2.refused else ("官网" if r2.web else "作答")
        ck(f"A{i} 「{q2}」→ {ev}，未沾上一题",
           ok, ("沾到了：" + "、".join(polluted)) if polluted else "")
        print(f"       上一句：{q1[:28]}…", flush=True)
        time.sleep(3)

    # ---------- B：真追问要接得上 ----------
    print("\n[B] 真追问：还得靠上文答得出来", flush=True)
    for i, (q1, q2, good) in enumerate(FOLLOWUP, 1):
        r1 = pipe.ask(q1)
        hist = hist_of(q1, r1.answer)
        time.sleep(3)
        r2 = pipe.ask(q2, history_block=hist, prev_user=q1)
        ans = r2.answer or ""
        hit = [w for w in good if w in ans]
        ok = (not r2.refused) and bool(hit)
        ck(f"B{i} 「{q2}」→ {'答上了' if ok else '没答上'}", ok,
           ("命中：" + "、".join(hit)) if hit else
           (r2.refuse_reason or "答复为空")[:60])
        time.sleep(3)

    print("\n" + "=" * 60)
    if FAIL:
        print(f"✗ 未通过 {len(FAIL)} 项：" + "；".join(FAIL))
        return 1
    print("✓ 全部通过：换话题不带偏，真追问接得上")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
