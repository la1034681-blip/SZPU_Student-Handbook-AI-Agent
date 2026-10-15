# -*- coding: utf-8 -*-
"""实验：模型申报了出处、正文却没打 [Ex] 角标时，怎么补救最稳。

对照三种处理，看引用校验能不能从 refused 翻成 passed：
  A. 原样（现状）—— 预期：一级校验「通篇无出处」直接毙掉
  B. 末尾补一行「（依据：[E1][E3][E4][E5]）」—— 整段成为一条断言
  C. 逐句各挂同一组编号 —— 每句各自成为一条断言

跑法：
    python src/dev/diag/exp_badge_backfill.py
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))
sys.stdout.reconfigure(encoding="utf-8")

import re                                          # noqa: E402
from qa_chain import build_llm, load_env_file      # noqa: E402
from retrievers import HandbookRetriever           # noqa: E402
from verify_citations import verify, parse_citations  # noqa: E402

load_env_file()

Q = "旷课多少学时会被处分、分别是什么处分？迟到早退算不算旷课？"

# 截图/诊断里模型真实给出的答案与申报引用（正文一处角标都没有）
ANSWER = ("旷课10-19学时给予警告处分，20-29学时给予严重警告处分，"
          "30-39学时给予记过处分，40-49学时给予留校察看处分，"
          "50学时以上给予开除学籍处分。"
          "迟到早退累计三次按旷课一学时计。")
DECLARED = ["E1", "E3", "E4", "E5"]

r = HandbookRetriever(str(ROOT / "data"), use_rerank=True, verbose=False)
evs = r.search(Q, top_k=6)
evmap = {e.eid: e for e in evs}
print("证据：", [(e.eid, e.clause_id, round(e.score, 3)) for e in evs])
print("申报的编号是否都在证据里：",
      [d for d in DECLARED if d in evmap])


def per_sentence(text, eids):
    """逐句各挂同一组编号（句末标点后插入）。"""
    marks = "".join(f"[{e}]" for e in eids)
    parts = re.split(r"(?<=[。；])", text)
    return "".join((p + marks) if p.strip() and not p.endswith("\n") else p
                   for p in parts)


CANDIDATES = {
    "A 原样（现状）": ANSWER,
    "B 末尾补一行依据": ANSWER + "\n\n（依据：" + "".join(f"[{e}]" for e in DECLARED) + "）",
    "C 逐句各挂编号": per_sentence(ANSWER, DECLARED),
}

llm = build_llm()
for name, text in CANDIDATES.items():
    res = verify(text, parse_citations(text) or DECLARED, evs, llm=llm)
    ex, sup = res["existence"], res["support"]
    print("=" * 72)
    print(f"[{name}]  passed={res['passed']}")
    print(f"  一级 exists={ex['passed']}  marked={ex['marked']}  "
          f"no_citation={ex.get('no_citation')}")
    print(f"  断言数={res['n_claims']}  support={sup.get('n_support')} "
          f"partial={sup.get('n_partial')} no={sup.get('n_no')} "
          f"hard_no={sup.get('n_hard_no')} skipped={sup.get('skipped')}")
    for d in (sup.get("verdicts") or []):
        print(f"    ·#{d['id']} {d['verdict']:8s} {d['eids']} "
              f"「{d['claim'][:26]}」")
    print("  reasons:", res["reasons"])
