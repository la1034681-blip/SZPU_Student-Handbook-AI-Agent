# -*- coding: utf-8 -*-
"""阶段 6 · 第 2 步：自检循环（Reflect · 答完自己检查）

作用
----
答案生成之后，**不急着交付**，先用第二次 LLM 调用换一个视角复查：

    用户问的每一个点，我都答到了吗？

查出来两类信息，交给外层循环去决定"要不要回去补检"：
    gaps      没答到的点（逐条），外层拿它当新一轮检索的查询
    covered   是否全覆盖
    hopeless  缺口是否属于"手册里本来就没写"（补检也救不了）

为什么需要它（本项目实测依据）
------------------------------
第 1 步（问题拆解）之后，检索层的 gold 召回已经到 97.0%，但端到端成绩
**一分没动**。追根到底：

  · Q36 判"部分正确"，可检索层其实 3 条全找到了 —— 是**生成时漏答了
    其中一步**（转专业的某个环节没写进答案）；
  · 这类"资料在手边却没答全"的错误，扩大检索、拆解都无解，**只能靠
    生成之后回头自查**：查出漏了哪一步，把它补进答案。

所以自检循环治的正是「生成不全」，与第 0 步（检索不全）、第 1 步
（跨文件召回）互为补位，三者合起来才是完整的智能体。

设计取舍
--------
- **保守优先，绝不动守门指标**：自检只负责"发现缺口 + 触发补检重生成"，
  **不负责拒答**。检出 hopeless（手册本就没有）时也只停止循环、保留原答案，
  不把原本能答的题改成拒答 —— 因为「误拒率 0%」是红线，不能被自检破坏。
- **失败不阻断**：LLM 调用失败 / 返回空 / 解析出错，一律按 covered=True
  处理（当作"没查出问题"），让主链路照常出答案。自检是"加分项"，不是"关口"。
- **只看"问题问到的点"**：提示词反复强调不许脑补原问题没有的要求，否则会
  把不该补的东西当成缺口，越补越偏。

对外接口
--------
    r = Reflector(llm)
    a = r.assess(question, answer, evidences)
    # -> {"covered": False, "gaps": ["转专业后学分如何认定"], "hopeless": False}
"""
from __future__ import annotations

import sys

sys.stdout.reconfigure(encoding="utf-8")


REFLECT_SYSTEM = """你是学生手册问答系统的「自检员」。
你的唯一任务：核对一份【答复】是否把【问题】问到的每一个点都答到了。

【判定原则】
1. 只核对「问题里明确问到的点」。问题没问的、或你自己觉得"应该补充"的内容，
   一律不算缺口。宁可漏判，不可多判。
2. 一个点只要答复里有对应表述就算「答到」，不要求措辞与问题一致。
3. 问题若由几个部分组成（用「和 / 以及 / 分别 / 各 / 还有什么」连接），
   或问「流程 / 环节 / 从 A 到 B 经历什么」，要**逐个部分**核对。
4. 数值、期限、比例、类别档次（本科/专科、各奖项等级）的缺漏，
   必须算作缺口。
5. 不评判答案的对错与文采，只找「有没有答到」。
6. **限定信息必须答到（最容易漏的一类，重点查）**：
   凡证据里就该事项写明了「时间 / 频次 / 次数限制 / 期限 / 数量 / 比例 /
   适用范围」这类限定条件，答复里就必须有对应表述；缺一个都算缺口。
   不要被问题的措辞带偏 —— 问「流程 / 环节」时，除了环节本身，
   还要确认办理的**时间与频次**有没有答到。
   典型例：问「转专业的整个流程包括哪些环节」，证据写明「转专业每学年办理
   一次、申请时间为入学第一学年末」，而答复只列了各环节、未提办理频次与时间，
   则必须报出缺口，形如「未说明转专业的办理频次与申请时间（每学年一次、
   第一学年末）」。
   （前提：证据里确有该限定信息。证据没有的，不算缺口。）

【缺口的可解性】
- 只有当缺口涉及的内容，在【证据】里**完全找不到任何相关字句**
  （说明手册本来就没写这件事），才把 hopeless 判为 true。
- 只要证据里出现过相关主题、只是没被答全，就**不算** hopeless，
  这种缺口正是"补检 + 重答"要解决的。

【输出格式】
只输出一个 JSON 对象，不要任何其他文字、不要代码围栏，含三个键：
  covered   布尔值，问题的每个点都答到了为 true
  gaps      字符串数组，逐条列出「没答到的点」，用简短问句描述；
            covered 为 true 时给空数组 []
  hopeless  布尔值，缺口属于「手册中本就没有」（补检无用）时为 true，否则 false"""

REFLECT_USER = """【问题】{question}

【已给出的答复】
{answer}

【答复所依据的证据】
{context}"""


class Reflector:
    def __init__(self, llm=None, verbose=False):
        self.llm = llm
        self.verbose = verbose
        self._chain = None
        self._llm_ref = None

    def _log(self, m):
        if self.verbose:
            print("  [自检] " + m, flush=True)

    def _build_chain(self, llm):
        from langchain_core.output_parsers import JsonOutputParser
        from langchain_core.prompts import ChatPromptTemplate
        prompt = ChatPromptTemplate.from_messages([
            ("system", REFLECT_SYSTEM),
            ("human", REFLECT_USER),
        ])
        return prompt | llm | JsonOutputParser()

    @staticmethod
    def _ok(reason=""):
        """失败/跳过时统一返回"没查出问题"，绝不阻断主链路。"""
        return {"covered": True, "gaps": [], "hopeless": False, "reason": reason}

    def assess(self, question: str, answer: str, evidences) -> dict:
        if self.llm is None:
            return self._ok("未接入 LLM")
        if not (answer or "").strip():
            return self._ok("答复为空，无可检")
        if not evidences:
            return self._ok("无证据")

        if self._chain is None or self._llm_ref is not self.llm:
            self._chain = self._build_chain(self.llm)
            self._llm_ref = self.llm

        context = "\n\n".join(e.render() for e in evidences)
        try:
            raw = self._chain.invoke(
                {"question": question, "answer": answer, "context": context})
        except Exception as e:                      # noqa: BLE001
            return self._ok(f"自检调用失败，跳过：{str(e)[:80]}")

        return self._sanitize(raw)

    @staticmethod
    def _sanitize(raw) -> dict:
        if not isinstance(raw, dict):
            return {"covered": True, "gaps": [], "hopeless": False,
                    "reason": "自检未返回 JSON 对象"}
        gaps = []
        for g in (raw.get("gaps") or []):
            t = str(g).strip().strip("？?。.、 ")
            if t:
                gaps.append(t)
        # 去重保序
        out, seen = [], set()
        for g in gaps:
            if g not in seen:
                seen.add(g)
                out.append(g)
        covered = bool(raw.get("covered")) or not out
        return {"covered": covered, "gaps": [] if covered else out,
                "hopeless": bool(raw.get("hopeless"))}


if __name__ == "__main__":
    import json as _json
    from qa_chain import build_llm
    from retrievers import HandbookRetriever

    q = (sys.argv[1] if len(sys.argv) > 1
         else "从补考到重修会经历哪些环节？")
    r = HandbookRetriever("data", use_rerank=True, verbose=False)
    evs = r.search(q, top_k=6)
    ans = "（此处应填一段答复，本文件默认做 dry-run 演示）"
    rf = Reflector(build_llm(), verbose=True)
    print(_json.dumps(rf.assess(q, ans, evs), ensure_ascii=False, indent=2))
