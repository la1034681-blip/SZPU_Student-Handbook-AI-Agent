# -*- coding: utf-8 -*-
"""追问诊断：验证「多轮记忆」是否真的起作用。

    python src/dev/diag/diag_followup.py
    python src/dev/diag/diag_followup.py "那它算不算迟到早退？"

跑法说明：脚本先造一段「上一轮」的历史（旷课处分），再把指定的**追问**
丢进链路。看三件事：
  ① planner 有没有把追问补全（plan.standalone 应该不再含「它」）
  ② 检索拿的是补全后的问题（证据应该扣住「迟到早退」）
  ③ 最终答案是不是在回答追问
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))
sys.stdout.reconfigure(encoding="utf-8")

from agent_qa import AgentQAPipeline              # noqa: E402
from memory import SessionMemory, new_mem_id      # noqa: E402
from qa_chain import build_llm, load_env_file     # noqa: E402
from retrievers import HandbookRetriever          # noqa: E402

load_env_file()

follow_up = (sys.argv[1] if len(sys.argv) > 1
             else "那它算不算迟到早退？")

mem = SessionMemory(mem_id=new_mem_id(), client_id="diag")
mem.add_user("旷课多少学时会被处分？")
mem.add_assistant("旷课10-19学时给予警告处分，20-29学时给予严重警告处分，"
                  "30-39学时给予记过处分，40-49学时给予留校察看处分，"
                  "50学时以上给予开除学籍处分。")
hist = mem.history_block()

print("=" * 72)
print("历史块：")
print(hist)
print("-" * 72)
print("本次追问：", follow_up)

r = HandbookRetriever(str(ROOT / "data"), use_rerank=True, verbose=False)
pipe = AgentQAPipeline(r, llm=build_llm(), verbose=True)

print("\n—— 不带上文（对照：模型不知道「它」指什么）——")
res0 = pipe.ask(follow_up)
print("   拆解：", json.dumps(res0.plan, ensure_ascii=False)[:220])
print("   拒答：", res0.refused, "｜", res0.refuse_reason[:80])
print("   答案：", (res0.answer or "")[:160])

print("\n—— 带上文（多轮记忆）——")
prev = mem.previous_user_question(follow_up)
print("   上一轮学生问句：", prev)
res = pipe.ask(follow_up, history_block=hist, prev_user=prev)
print("   拆解：", json.dumps(res.plan, ensure_ascii=False)[:220])
print("   standalone：", (res.plan or {}).get("standalone"))
print("   检索次数：", res.hops, "｜ 证据条数：", res.n_evidence)
for e in (res.evidences or [])[:5]:
    print(f"     [{e.eid}] {e.score:.4f} {e.clause_id}")
print("   拒答：", res.refused, "｜", res.refuse_reason[:80])
print("   答案：", (res.answer or "")[:300])
print("=" * 72)
