# -*- coding: utf-8 -*-
"""角标回填 —— 纯离线，不需要模型。

背景（2026-10-15 用户截图实证）
-----------------------------
问「旷课多少学时会被处分、分别是什么处分？迟到早退算不算旷课？」，
检索分 0.99、答案与手册逐字对得上，学生收到的却是「抱歉，这个问题
我暂时答不了」。原因不是检索错、也不是模型错，而是：

  GLM-4-Flash 把出处老老实实填进了 `citations` 字段（E1/E3/E4/E5），
  正文却**一个 [Ex] 角标都没打** → 一级校验的硬规矩「有实质内容却一处
  引用都没标 → 不放行」（防「全篇不标」蒙混过关）命中 → 整题拒答。

即**尺子量不出东西，被当成学生作弊** —— 本项目最忌讳的那类误杀。
修法是 `qa_chain.backfill_citations`：把「已申报且确实在本次证据里」的
编号补到答案末尾（句末标点前），原文一字不改，再走两级校验。

对照实验见 `src/dev/diag/exp_badge_backfill.py`（末尾补一行 vs 逐句各挂
编号：后者会多出一条没被判定的断言、留漏洞，故取前者）。

跑法：`python src/tests/test_badge_backfill.py`
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / "src"))

from qa_chain import backfill_citations             # noqa: E402
from verify_citations import parse_citations, verify  # noqa: E402
from retrievers import Evidence                     # noqa: E402

fails: list = []


def check(name: str, got, want) -> None:
    ok = got == want
    print(f"  {'✓' if ok else '✗'} {name}")
    if not ok:
        fails.append(name)
        print(f"      期望 {want!r}\n      实得 {got!r}")


def ev(eid, clause="F06-A017", text="旷课10-19学时给予警告处分。"):
    return Evidence(eid=eid, clause_id=clause, chunk_id=f"K-{clause}",
                    cite="《深圳职业技术大学学生违纪处分办法》 第06条",
                    file_id="F06", file_title="违纪处分办法",
                    article_label="第6条", page="31",
                    text=text, score=0.99)


Q_ANS = ("旷课10-19学时给予警告处分，20-29学时给予严重警告处分。"
         "迟到早退累计三次按旷课一学时计。")

EVS = [ev("E1"), ev("E3"), ev("E4"), ev("E5")]

print("① 正文无角标 + 申报合法 → 回填在句末标点之前")
text, hit = backfill_citations(Q_ANS, ["E1", "E3", "E4", "E5"], EVS)
check("触发了回填", hit, True)
check("角标插在最后一个句号前", text.endswith("按旷课一学时计[E1][E3][E4][E5]。"), True)
check("编号去重保序", parse_citations(text), ["E1", "E3", "E4", "E5"])
check("原文一字未改（去掉角标后与原文相同）",
      text.replace("[E1][E3][E4][E5]", ""), Q_ANS)

print("② 正文自己标了角标 → 绝不插手")
already = "旷课10-19学时给予警告处分[E1]。"
check("不触发", backfill_citations(already, ["E1"], EVS)[1], False)
check("文本原样", backfill_citations(already, ["E1"], EVS)[0], already)

print("③ 申报里混了不存在的编号 → 只回填合法的")
text, hit = backfill_citations(Q_ANS, ["E1", "E9", "E3"], EVS)
check("触发了回填", hit, True)
check("只补合法的两个", parse_citations(text), ["E1", "E3"])

print("④ 申报的全是编的 → 不圆场（伪造引用仍要被一级校验拦下）")
check("不触发", backfill_citations(Q_ANS, ["E9", "E8"], EVS)[1], False)
check("文本原样", backfill_citations(Q_ANS, ["E9"], EVS)[0], Q_ANS)

print("⑤ 没有任何申报 → 不凭空造出处")
check("不触发", backfill_citations(Q_ANS, [], EVS)[1], False)
check("不触发（None）", backfill_citations(Q_ANS, None, EVS)[1], False)

print("⑥ 末尾没有句末标点 → 角标直接接在末尾")
text, hit = backfill_citations("迟到早退累计三次按旷课一学时计", ["E1"], EVS)
check("触发了回填", hit, True)
check("角标在末尾", text, "迟到早退累计三次按旷课一学时计[E1]")

print("⑦ 空答案 → 不折腾")
check("不触发", backfill_citations("", ["E1"], EVS)[1], False)

print("⑧ 端到端（不发模型）：回填前后一级校验翻转")
before = verify(Q_ANS, ["E1", "E3", "E4", "E5"], EVS, llm=None)
check("回填前：一级校验不放行", before["existence"]["passed"], False)
check("回填前的原因就是「通篇无出处」",
      before["existence"]["no_citation"], True)
after_text, _ = backfill_citations(Q_ANS, ["E1", "E3", "E4", "E5"], EVS)
after = verify(after_text, ["E1", "E3", "E4", "E5"], EVS, llm=None)
check("回填后：一级校验放行", after["existence"]["passed"], True)
check("回填后不再是「通篇无出处」", after["existence"]["no_citation"], False)
check("整体 passed（二级无 llm 时跳过，不阻断）", after["passed"], True)

print()
if fails:
    print(f"✗ 有 {len(fails)} 项没通过：{fails}")
    sys.exit(1)
print("✓ 全部通过")
