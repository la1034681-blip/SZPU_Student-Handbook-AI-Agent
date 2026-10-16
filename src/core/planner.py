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
    # -> {"need_split": True, "sub_questions": [...], "levels": "未指明",
    #     "standalone": "…"}
    # 给了 history_block 时，standalone 是「把追问补全成可独立理解的问题」，
    # 由检索与生成共用 —— 见 src/infra/memory.py 的说明。
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

【有历史对话时】—— 如果下面给了【之前的对话】：
- 先看本次问题有没有**省略或指代**（「它 / 他 / 这个 / 那个 / 那……呢 /
  还有呢 / 那怎么算 / 为什么」）。有的话，把本次问题补成一句**单独拿出来
  也看得懂**的完整问题，写进 standalone 字段。
- standalone **只替换指代词、补全省略的主语或事项**，其余照原样保留。
  绝对不要把上一轮问句的内容搬进来，也不要把两个问题拼成一句。
  例：上文在讲「旷课处分」，本次问「那它算不算迟到早退？」
      → standalone 应写「迟到早退算不算旷课？」
      **不要**写成「旷课多少学时算迟到早退」—— 那是把上一轮的问题混进来了。
- 本次问题本身已经完整、不依赖上文时，standalone 就照抄本次问题原句。
- 没有给历史对话时，standalone 一律照抄本次问题原句。

【输出格式】
只输出一个 JSON 对象，不要任何其他文字、不要代码围栏，含四个键：
  need_split    布尔值，需要拆解为 true
  sub_questions 字符串数组，拆出来的子问题；need_split 为 false 时给空数组 []
  levels        字符串，取 "本科" / "专科" / "两者" / "未指明" 之一
  standalone    字符串，把本次问题补全成可独立理解的完整问句（见上）"""

PLANNER_USER = """{history_block}【问题】{question}"""


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

    def plan(self, question: str, llm=None, history_block: str = "") -> dict:
        """规划。给了 history_block 时，顺带产出 standalone（补全指代的完整问题）。

        为什么把「追问改写」放在这里做：规划本来就是一道必跑的模型调用，
        让它**顺手**输出 standalone，等于零额外成本。单独再调一次模型去改写，
        在免费档上要多撞一次 429 风险，还要多等几秒。
        """
        llm = llm or self.llm
        if llm is None:
            return self._fallback("未接入 LLM，默认不拆", question)
        if self._chain is None or getattr(self, "_llm_ref", None) is not llm:
            self._chain = self._build_chain(llm)
            self._llm_ref = llm
        try:
            raw = self._chain.invoke({"question": question,
                                      "history_block": history_block or ""})
        except Exception as e:                      # noqa: BLE001
            # 规划失败不能让整条链路挂掉：退回"不拆"，走原来的路径
            return self._fallback(f"规划失败，退回不拆：{str(e)[:80]}", question)
        return self._sanitize(question, raw, bool(history_block))

    @staticmethod
    def _fallback(reason: str, question: str = "") -> dict:
        return {"need_split": False, "sub_questions": [], "levels": "未指明",
                "standalone": question, "reason": reason}

    @staticmethod
    def _sanitize(question: str, raw, allow_standalone: bool = False) -> dict:
        if not isinstance(raw, dict):
            return {"need_split": False, "sub_questions": [], "levels": "未指明",
                    "standalone": question, "reason": "规划器未返回 JSON 对象"}
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
        # standalone：只有「这次给了历史」且模型确实改写了，才认它的改写；
        # 否则一律用原问题 —— 免得没有历史时被模型顺手改题意。
        standalone = str(raw.get("standalone") or "").strip().strip("？?。.、 ")
        if not (allow_standalone and standalone):
            standalone = question
        standalone = standalone[:200]
        return {"need_split": need, "sub_questions": out, "levels": levels,
                "standalone": standalone}


if __name__ == "__main__":
    import json as _json
    from qa_chain import build_llm
    q = sys.argv[1] if len(sys.argv) > 1 else "一门必修课不及格之后，从补考到重修会经历哪些环节？"
    p = Planner(build_llm())
    print(_json.dumps(p.plan(q), ensure_ascii=False, indent=2))
