# -*- coding: utf-8 -*-
"""单题拒答诊断：把一道题在智能体链路里跑一遍，打印每一环的中间结果。

    python src/dev/diag/diag_refuse_case.py "旷课多少学时会被处分……"

用途：页面出现「校验未通过」式拒答时，定位到底是
  · 拒答前置（检索分太低）
  · 模型自述证据不足（拒答复核有没有救回来）
  · 引用校验判 passed=False（挑的哪条理由）
哪一环掉链子 —— 只看页面上的「抱歉」是看不出来的。
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))
sys.stdout.reconfigure(encoding="utf-8")

from agent_qa import AgentQAPipeline              # noqa: E402
from qa_chain import build_llm, load_env_file     # noqa: E402
from retrievers import HandbookRetriever          # noqa: E402

load_env_file()

q = (sys.argv[1] if len(sys.argv) > 1
     else "旷课多少学时会被处分、分别是什么处分？迟到早退算不算旷课？")

r = HandbookRetriever(str(ROOT / "data"), use_rerank=True, verbose=False)
pipe = AgentQAPipeline(r, llm=build_llm(), verbose=True)
res = pipe.ask(q)

print("=" * 72)
print("问题：", res.question)
print("拆解：", json.dumps(res.plan, ensure_ascii=False))
print("检索次数：", res.hops, "｜ 证据条数：", res.n_evidence)
print("-" * 72)
print("证据（前 8 条，按链路给出的顺序）：")
for e in (res.evidences or [])[:8]:
    print(f"  [{e.eid}] score={e.score:.4f} {e.clause_id} "
          f"{e.title[:34] if getattr(e, 'title', '') else ''}")
print("-" * 72)
print("refused :", res.refused)
print("reason  :", res.refuse_reason)
print("insufficient:", getattr(res, "insufficient", None),
      "｜ model_error:", getattr(res, "model_error", ""))
print("refuse_confirm:", json.dumps(getattr(res, "refuse_confirm", None),
                                   ensure_ascii=False))
print("-" * 72)
print("答案：")
print(res.answer or "(空)")
print("引用：", res.citations)
print("-" * 72)
v = getattr(res, "verify", None) or {}
print("校验 passed :", v.get("passed"), "｜ degraded:", v.get("degraded"))
print("校验 reasons:", json.dumps(v.get("reasons"), ensure_ascii=False))
print("校验 verdicts:")
for item in (v.get("verdicts") or []):
    print("   ", json.dumps(item, ensure_ascii=False))
print("校验 error  :", v.get("error", ""))
print("=" * 72)
