# -*- coding: utf-8 -*-
"""阶段 6 · 第 1 步：问题拆解（Planner）

作用
----
判断一道提问是「简单题」还是「复合题」。复合题（一句话里问了好几件事，
或答案要横跨多份文件）拆成 2~4 个子问题，逐个去检索，避免"一锅炖"漏条件。

为什么需要它（本项目实测依据）
------------------------------
第 0 步调参数后，跨章题 gold 召回从 72.2% 升到 88.9%，但剩下的 2 条漏项
（Q34 缺 F06《考勤管理规定》、Q35 缺 F12《学业预警管理办法》）属于
**整份文件没进候选**——「多翻两页」救不了，因为那本书根本没被拿下来。
拆解走的是另一条路：把"从补考到重修经历哪些环节"拆成
["补考怎么安排", "重修什么条件下要修", "旷课会怎样"]，
让"旷课"这个子问题单独去检索，才能把 F06 拉进候选。

四条硬约束（省钱省时，方案 3.1 约定）
------------------------------------
1. 默认不拆 —— 只有判定为复合/跨章题才拆，简单题直接走原路径。
2. 只拆一层 —— 子问题不再继续拆，防止无限套娃。
3. 子问题上限 4 个，超过截断。
4. 只用「问题本身」判，不看检索结果（先规划后行动）。

对外接口
--------
    p = Planner(llm)
    plan = p.plan("从补考到重修会经历哪些环节？")
    # -> {"need_split": True, "sub_questions": [...], "levels": "未指明"}
"""
from __future__ import annotations

import sys

sys.stdout.reconfigure(encoding="utf-8")

MAX_SUB = 4          # 子问题数量上限
MIN_SUB = 2          # 少于 2 个就不算拆解


PLANNER_SYSTEM = """你是学生手册问答系统的「问题规划器」。你的唯一任务：
判断用户这一句提问是否需要拆成多个子问题、分别去检索。

【需要拆解】—— 满足任一条即算复合题：
1. 一句话里并列了 2 个及以上**不同主题/不同事务**（用「和 / 以及 / 分别 / 各 /
   有什么区别」连接）。例：「考试违纪和作弊有什么区别？分别怎么处理？」
2. 问「流程 / 环节 / 经过 / 从 A 到 B 会经历什么」这类链路问题。
3. 问「（从轻到重 / 逐级 / 一步步）会有哪些处理、后果、处分」这类**递进后果**
   问题 —— 这类事情常由多份不同文件分别规定（例如「学籍处理」与「学业预警」
   就是两套并行口径），必须拆开各自去查。
  例：「如果学分一直修不够，从轻到重学校会怎么处理？」

【不要拆解】—— 满足任一条即算简单题：
1. 单一事实 / 单一数值 / 单一条件 / 名单类问题。
2. **同一件事的连续追问**，不要因为用了两个问号就拆。
   例：「必修课不及格，能补考吗？补考成绩怎么算？」—— 两个问号问的都是
   「补考」这一件事，属同一主题，**不拆**。
3. 只涉及一份文件的单一主题。
4. 手册里可能根本没写的内容（拒答类）。

【拆解规则】
- 子问题 2~4 个，每个都要能独立拿去检索，用简短问句表达；
- 子问题应换「不同的检索视角」——能带回**不同文件或不同条款**，
  而不是把一个主题换个措辞说两遍；
- 只拆一层，子问题内部不要再嵌套；
- 子问题合起来要覆盖原问题的每个组成部分，不要漏、也不要脑补原问题没有的条件；
- 子问题不要和原问题完全一样。

【输出格式】
只输出一个 JSON 对象，不要任何其他文字、不要代码围栏，含三个键：
  need_split    布尔值，需要拆解为 true
  sub_questions 字符串数组，拆出来的子问题；need_split 为 false 时给空数组 []
  levels        字符串，取 "本科" / "专科" / "两者" / "未指明" 之一"""

PLANNER_USER = """【问题】{question}"""


class Planner:
    def __init__(self, llm=None, verbose=False):
        self.llm = llm
        self.verbose = verbose
        self._chain = None

    def _build_chain(self, llm):
        from langchain_core.output_parsers import JsonOutputParser
        from langchain_core.prompts import ChatPromptTemplate
        prompt = ChatPromptTemplate.from_messages([
            ("system", PLANNER_SYSTEM),
            ("human", PLANNER_USER),
        ])
        return prompt | llm | JsonOutputParser()

    def plan(self, question: str, llm=None) -> dict:
        llm = llm or self.llm
        if llm is None:
            return self._fallback("未接入 LLM，默认不拆")
        if self._chain is None or getattr(self, "_llm_ref", None) is not llm:
            self._chain = self._build_chain(llm)
            self._llm_ref = llm
        try:
            raw = self._chain.invoke({"question": question})
        except Exception as e:                      # noqa: BLE001
            # 规划失败不能让整条链路挂掉：退回"不拆"，走原来的路径
            return self._fallback(f"规划失败，退回不拆：{str(e)[:80]}")
        return self._sanitize(question, raw)

    @staticmethod
    def _fallback(reason: str) -> dict:
        return {"need_split": False, "sub_questions": [], "levels": "未指明",
                "reason": reason}

    @staticmethod
    def _sanitize(question: str, raw) -> dict:
        if not isinstance(raw, dict):
            return {"need_split": False, "sub_questions": [], "levels": "未指明",
                    "reason": "规划器未返回 JSON 对象"}
        need = bool(raw.get("need_split"))
        subs = raw.get("sub_questions") or []
        if isinstance(subs, str):
            subs = [subs]
        # 清洗：去空白、去尾标点、去空项
        cleaned = []
        for s in subs:
            t = str(s).strip().strip("？?。.、 ")
            if t:
                cleaned.append(t)
        # 去重 + 去掉与原问题完全相同者（保留首次出现顺序）
        qkey = question.strip().rstrip("？?。. ")
        out, seen = [], set()
        for t in cleaned:
            if t in seen or t == qkey:
                continue
            seen.add(t)
            out.append(t)
        # 硬约束：上限 4；不足 2 个视为未拆解
        if len(out) > MAX_SUB:
            out = out[:MAX_SUB]
        if len(out) < MIN_SUB:
            need = False
        if not need:
            out = []
        levels = str(raw.get("levels") or "未指明")
        return {"need_split": need, "sub_questions": out, "levels": levels}


if __name__ == "__main__":
    import json as _json
    from qa_chain import build_llm
    q = sys.argv[1] if len(sys.argv) > 1 else "一门必修课不及格之后，从补考到重修会经历哪些环节？"
    p = Planner(build_llm())
    print(_json.dumps(p.plan(q), ensure_ascii=False, indent=2))
