# -*- coding: utf-8 -*-
"""阶段 6 · Agent 问答链路（规划 → 多路检索 → 生成 → 自检回环）

它跟原来的 qa_chain.QAPipeline 有什么区别
------------------------------------------
原链路是一条直线：
    提问 → 检索一次 → 生成 → 校验

本链路在两头各加了一个"大脑"，中间多了一个"能回去重来的环"：
    提问 → 【规划：要不要拆？】（默认开启，见下）
         → 拆了就多路检索、合并去重 / 不拆就检索一次
         → 【拒答前置：证据够不够沾边？】（够不上 → 直接判 R1，省两次调用）
         → 生成 + 校验
         → 【自检：问题的每个点都答到了吗？】
              └─ 有缺口 → 带着缺口回去补检 → 重新生成 → 再自检（最多 2 轮）

所以它是**增量**的：
  · 简单题：不拆解、检索一次；但同样过一遍自检（多半一次即判"无缺口"，不进循环）；
  · 复合/跨章题：才走拆解（默认开启）；两者的自检回环机制一致。
这正是方案 3.3「自检循环」的落地；拆解是否默认开启，见下方 enable_planner 处的常量注释
（阶段 6 第 4 步扩充评测集后，已由"关"改回"开"）。

【阶段 6 第 4 步（评测集扩充后）的默认配置：规划开 / 自检开 / 门控开】
--------------------------------------------------------------------------
把跨章题从 4 道扩到 20 道后，重跑"拆解开 vs 拆解关"两配置（20 道跨章
题各 3 轮 + 全量 60 题 1 轮），结论**较原先反了过来**：
  · 全量 60 题：答案正确率 89.6% → 93.4%，出处命中率 94.3% → 100%，
    误拒率 5.7% → 0%（平均延迟 5.1s → 7.8s，仍在 12s 红线内）；
  · 跨章题 3 轮均值：答案正确率 75.8% → 80.8%，出处命中率 85.0% → 98.3%
    （后者基线零方差，信号最干净）。
原先判"拆解净收益中性偏负"，是因为旧评测集只有 4 道跨章题、且都已满分
——样本太小，拆解的"补漏"作用无处发挥。扩充后它的价值才显出来。
故默认改回 **enable_planner=True**。
遗留：拆解会把证据顶到 max_merged 上限，个别题（Q34/Q35/N15）出现稀释、
偶发掉分；实测把上限放开到 45 或收紧到 18 都更差，28 已是较好档，
后续需在"证据排序/精简"上优化，而非调上限。

三步各自治什么（本项目实测依据）
--------------------------------
  第 0 步 · 检索调参　→ 治"同文件相邻条款被名额挤掉"（Q24）
  第 1 步 · 问题拆解　→ 治"整份文件没进候选"（Q34/Q35 的跨文件漏项）← 召回有效，端到端无效
  第 2 步 · 自检回环　→ 治"资料到手却没答全"（Q36 生成漏答）← 真正涨分的一步

三条硬约束（防死循环，方案 3.3 约定）
------------------------------------
1. 最多 2 轮补检，第 3 轮无论结果如何都收工；
2. 缺口与上一轮完全相同 → 立刻停（补检没带来新料，再转也是空转）；
3. 补检没带回任何新证据 → 也停（不靠"重复重答"刷分）。

对外接口
--------
    pipe = AgentQAPipeline(retriever, llm)
    res = pipe.ask("从补考到重修会经历哪些环节？")
    res.plan            拆解结果
    res.hops            本题实际检索次数（1=未拆）
    res.n_evidence      最终送给模型的证据条数
    res.reflect_rounds  自检触发的补检轮数（0=没补）
    res.reflect_log     每轮自检的缺口清单
"""
from __future__ import annotations

import sys
import time
from concurrent.futures import ThreadPoolExecutor

sys.stdout.reconfigure(encoding="utf-8")

from planner import Planner              # noqa: E402
from qa_chain import QAPipeline, QAAnswer  # noqa: E402
from reflect import Reflector            # noqa: E402


def _norm_q(text: str) -> str:
    """问题比对用的归一化：去首尾空白与尾部标点。

    为什么要归一化再用：planner 会把 standalone 的尾巴（？。等）削掉，
    所以「那要多久呢？？」与「那要多久呢」必须被认成同一句 —— 否则
    「模型没改写 → 走确定性兜底」这个判断会失效（2026-10-15 实测踩到：
    兜底检索路没被加上，模糊追问仍然搜不到东西）。
    """
    return (text or "").strip().strip("？?。.、，,！! ").strip()


def _same_q(a: str, b: str) -> bool:
    return _norm_q(a) == _norm_q(b) and bool(_norm_q(a))


class AgentQAPipeline:
    def __init__(self, retriever, llm=None, top_k=6, max_merged=28,
                 planner=None, reflect=None, enable_planner=True,
                 enable_reflect=True, max_reflect_rounds=2,
                 enable_gate=True, gate_min_top1=0.03,
                 parallel_search=False, search_workers=5,
                 verbose=True, **qa_kwargs):
        self.retriever = retriever
        self.llm = llm
        self.top_k = top_k
        # 合并后证据上限。注意：**必须 ≥ 单查询自身的证据条数**
        # （base_k + 兄弟扩展上限 = 6 + 16 = 22），否则合并会
        # "比单查询还少"——把原始问题已经拿到的证据截掉，等于拆解反而丢证据。
        # 这里给 28 留出子问题新增证据的空间，并由 _merge 兜底保证不截原始集。
        self.max_merged = max_merged
        # ✅ 默认开启（阶段 6 第 4 步"扩充评测集"后结论反转，依据见 compare_ext_rounds）：
        #   · 旧评测集只有 4 道跨章题且都已满分，拆解的"补漏"作用无处发挥，
        #     曾据此误判为"净收益中性偏负"（当时的全关+全开打平，仅是样本太小）；
        #   · 跨章题扩到 20 道后重跑：全量 60 题答案 89.6%→93.4%、出处 94.3%→100%、
        #     误拒 5.7%→0%；跨章 3 轮出处 85.0%→98.3%（基线零方差，信号最干净）。
        #   · 教训：**样本太小会让好组件背黑锅** —— 判定组件价值前先确认样本够不够。
        #   遗留：个别题（Q34/Q35/N15）仍会因证据稀释 / 口径混同偶发掉分，
        #   需在"证据排序/精简"上优化（上限放开到 45 或收紧到 18 都更差，28 已是较好档）。
        self.enable_planner = enable_planner
        self.enable_reflect = enable_reflect
        self.max_reflect_rounds = max_reflect_rounds
        # 组件四 · 拒答前置：证据最高分低于 gate_min_top1 就直接判 R1，省两次调用。
        # 阈值依据见 src/diag_gate.py：44 题里拒答题最高 0.1104、正常题最低 0.1124，
        # 两档空档仅 0.002 —— 取"中值"必然过拟合，任何一道正常题抖动就会被误拒。
        # 所以这里取远低于正常题最低分的 0.03（余量 0.08），只为拦"最不沾边"那一档，
        # 宁可漏拦、绝不误拦（误拒率 0% 是红线）。
        self.enable_gate = enable_gate
        self.gate_min_top1 = gate_min_top1
        # 性能：多路（原问题 + 各子问题）检索理论上可并发，但**实测默认关**。
        # 本机检索走 GPU（RTX 5060，torch.cuda 可用），并发反而抢 GPU 排队，
        # 实测首轮检索零提速、自检补检还从 1~4s 涨到 12s+（见 diag_latency 对比）。
        # 保留开关，供纯 CPU / 多卡环境按需开启；默认串行即最优。
        self.parallel_search = parallel_search
        self.search_workers = search_workers
        self.planner = planner or Planner(llm=llm)
        self.reflector = reflect or Reflector(llm=llm, verbose=False)
        self.verbose = verbose
        # 生成 + 校验完全复用原链路（阶段 6 抽出的 answer_from_evidences）
        self.base = QAPipeline(retriever, llm=llm, top_k=top_k,
                               verbose=verbose, **qa_kwargs)

    # ------------------------------------------------------------------ 检索合并
    # 多路检索的「每路条数上限」（2026-10-14 修 N05 时新增）
    # ------------------------------------------------------------------
    # 背景：retriever.search 每次返回的不止 top_k 条 —— 它先取 top_k 个种子，
    # 再做「同文件兄弟条款扩展」，最多再补 SIBLING_MAX_EXTRA(16) 条，即单路
    # 最多可返回 22 条。多路合并时若不做每路截断，第 1 路就能把 max_merged
    # （28）的额度吃光。
    # 实测 N05（退学情形 / 证书 / 宿舍怎么退 三问合一）：前 3 路全是学籍类
    # 条款，共占满 28 条，第 4 路「退学后宿舍怎么退」的命中条款 F38-A015
    # （该路排名第 1）被整体挤出上下文 → 模型判"第三问无依据"→ 整题拒答。
    # 注意：这不是检索能力问题 —— 单独查「退学后宿舍怎么退」时 F38-A015 稳排第 1。
    #
    # 故按路限额：主干路（原问题）放宽到 top_k + MAIN_EXTRA，保住"种子 +
    # 少量兄弟扩展"的既有收益（阶段 6 第 0 步调参的成果，治 Q37 那种
    # 同文件相邻条款漏项）；子问题路只保留 top_k 个种子，它们的职责是
    # 「跨文件补漏」，跨文件本来就没有兄弟条款可扩。
    MAIN_EXTRA = 4

    def _search_all(self, queries, caps=None):
        """对多路查询检索，返回与 queries 等长、**同序**的结果列表。

        caps：与 queries 等长的「每路最多保留条数」；None / 元素为 None 表示不截断。
              不传则按 top_k 截断（并发说明见下）。

        并发说明：首个查询同步执行（顺带把索引/模型加载做掉），其余并发。
        这样既避开多线程首次初始化的竞态，又拿到并发提速。
        """
        if not queries:
            return []
        if caps is None:
            caps = [self.top_k] * len(queries)

        def _one(q, cap):
            evs = self.retriever.search(q, top_k=self.top_k)
            return evs[:cap] if cap else evs

        first = _one(queries[0], caps[0])
        if len(queries) == 1 or not self.parallel_search:
            return [first] + [_one(q, c)
                              for q, c in zip(queries[1:], caps[1:])]
        rest = list(zip(queries[1:], caps[1:]))
        with ThreadPoolExecutor(max_workers=min(len(rest), self.search_workers)) as ex:
            rest_res = list(ex.map(lambda qc: _one(*qc), rest))
        return [first] + rest_res

    def _collect(self, queries, merged, order, caps=None):
        """把 queries 的检索结果并入 merged/order（同 clause_id 留高分那条）。"""
        for result in self._search_all(queries, caps):
            for e in result:
                if e.clause_id not in merged:
                    merged[e.clause_id] = e
                    order.append(e.clause_id)
                elif e.score > merged[e.clause_id].score:
                    merged[e.clause_id] = e

    @staticmethod
    def _finalize(merged, order, cap):
        """按顺序取出证据、重编号（引用必须能对上 E1..En）。"""
        evs = [merged[c] for c in order][:cap]
        for i, e in enumerate(evs, 1):
            e.eid = f"E{i}"
        return evs

    def _merge(self, queries):
        """首轮多路检索 + 合并去重。

        queries[0] 约定为「原始问题」。它的证据优先保留（且放宽到
        top_k + MAIN_EXTRA 条，保住兄弟扩展的收益）；其余子问题路
        各限额 top_k 条 —— 避免任何一路吃掉全部额度而挤掉别的子问题的
        命中条款（N05 的教训，见 MAIN_EXTRA 处注释）。
        """
        caps = ([self.top_k + self.MAIN_EXTRA] + [self.top_k] * (len(queries) - 1)
                if len(queries) > 1 else None)
        merged, order, n_orig = {}, [], 0
        for i, result in enumerate(self._search_all(queries, caps)):
            for e in result:
                if e.clause_id not in merged:
                    merged[e.clause_id] = e
                    order.append(e.clause_id)
                    if i == 0:
                        n_orig += 1
                elif e.score > merged[e.clause_id].score:
                    merged[e.clause_id] = e
        return self._finalize(merged, order, max(self.max_merged, n_orig))

    def _merge_into(self, existing, queries):
        """自检回环：把新检索到的证据**并入**已有证据池（已有在前，新证据追加）。

        已有证据永不截断（cap 至少等于已有条数），保证"补检只会加料、不会掉料"。
        补检查询都是针对**具体缺口**的短问句，每路只取 top_k 个种子即可。
        """
        merged, order = {}, []
        for e in existing:
            if e.clause_id not in merged:
                merged[e.clause_id] = e
                order.append(e.clause_id)
        n_have = len(order)
        self._collect(queries, merged, order, [self.top_k] * len(queries))
        return self._finalize(merged, order, max(self.max_merged, n_have))

    # ------------------------------------------------------------------ 规划
    def plan_only(self, question, history_block=""):
        if not self.enable_planner:
            return {"need_split": False, "sub_questions": [], "levels": "未指明",
                    "standalone": question, "reason": "planner 关闭"}
        return self.planner.plan(question, self.llm, history_block=history_block)

    # ------------------------------------------------------------------ 主链路
    def ask(self, question: str, history_block: str = "",
            prev_user: str = ""):
        """主导航：一切都由大模型驱动；只有模型自己坏了才交给程序化兜底。

        `history_block`（可选）：本次会话之前的对话摘要 + 最近几轮原文，
        由 `src/memory.py` 拼好。它有两个用处 ——
          ① 规划器据此产出 standalone（把「它/这个」补成完整问题），检索拿
             完整语义去搜，追问题才搜得到东西；
          ② 生成时作为背景喂进提示词，让回答接得上上文（但它**不是证据**，
             见 qa_chain.SYSTEM_PROMPT 第 8 条）。
        `prev_user`（可选）：上一轮学生问的那句原文。小模型不一定每次都肯
        把追问改写干净，所以再留一条**确定性**兜底 —— 见 `_ask` 里的注释。
        两个都不传 = 原来的「每题独立」行为，完全向后兼容。
        """
        try:
            res = self._ask(question, history_block, prev_user)
        except Exception as e:                          # noqa: BLE001
            if self.verbose:
                print(f"[兜底] 大模型不可用，转程序化兜底："
                      f"{type(e).__name__}: {str(e)[:120]}", flush=True)
            return self._rescue(question, e)

        # 链路没抛异常、但结果是「模型调用失败」式的拒答 —— 那也不算答了，
        # 同样转兜底（这条是主路径：429 重试烧完、网络断、密钥失效都走这里）。
        if getattr(res, "model_error", ""):
            if self.verbose:
                print(f"[兜底] 模型调用失败，转程序化兜底："
                      f"{res.model_error[:120]}", flush=True)
            return self._rescue(question, RuntimeError(res.model_error),
                                evs=res.evidences)
        return res

    def _rescue(self, question: str, err: Exception, evs=None):
        """大模型用不了时的保底作答：照抄手册原文，不留白。"""
        t0 = time.time()
        note = f"{type(err).__name__}: {str(err)[:200]}"
        res = QAAnswer(question=question)
        res.plan = {"fallback": "procedural", "fallback_error": note}
        if evs is None:
            try:
                evs = self.retriever.search(question, top_k=self.top_k)
            except Exception as e2:                     # noqa: BLE001
                res.refused = True
                res.refuse_reason = f"大模型不可用（{note}），检索也失败（{e2}）"
                res.plan["fallback_ok"] = False
                res.latency = time.time() - t0
                return res

        res.evidences = evs
        res.hops = 1
        res.n_evidence = len(evs)
        try:
            from procedural_qa import generate, _verify as _pverify
            out = generate(question, evs, max_evidence=3)
        except Exception as e3:                         # noqa: BLE001
            out = None
            res.plan["fallback_error"] += f" ｜ 兜底本身也失败：{e3}"
        if out and not out["insufficient"]:
            res.answer = out["answer"]
            res.citations = out["citations"]
            res.verify = _pverify(res.answer, res.citations, evs)
            res.plan["fallback_ok"] = True
        else:
            res.refused = True
            res.refuse_reason = "手册里没有找到与问题相关的条款"
            res.plan["fallback_ok"] = True
        res.latency = time.time() - t0
        return res

    def _ask(self, question: str, history_block: str = "", prev_user: str = ""):
        t0 = time.time()
        plan = self.plan_only(question, history_block)
        need_split = bool(plan.get("need_split"))
        # 检索用「补全指代后的问题」：带历史时它是 standalone，没历史时就是原问题
        # （planner 保证 standalone 至少等于原问题）。生成仍然用原问句，这样
        # 回答的是学生真正问的那句，语气与承接也对得上。
        root = (plan.get("standalone") or question).strip() or question
        queries = ([root] + plan["sub_questions"] if need_split else [root])
        # 追问兜底（确定性，不指望小模型）：本次是追问、而模型又**没**把指代
        # 补出来（standalone 与原句一样）时，再补一路「上一句 + 本句」——
        # 靠上一个话题的词把「它/那」补上。例：「那要多久呢？」单独拿去检索
        # 什么都没；拼成「重修要什么条件 那要多久呢」就能搜到重修条款。
        if (prev_user and history_block
                and _same_q(root, question) and prev_user not in queries):
            queries = queries + [f"{prev_user} {question}"]
        # 口语词换手册词（2026-10-16）：学生说「体测」，手册写「体质测试」，
        # 字面一个字都不重合 → BM25 零命中、向量也拉不起来，最高分只有 0.05，
        # 最后被引用校验判「证据只是擦边」整题拒答。这里多走几路「换过词的
        # 同义问句」（追加/前缀/替换三种写法），合并时按条款取最高分，
        # 所以只会多找到东西、不会弄丢原来的。命中的词见 src/term_alias.py。
        # 没命中任何口语词时 term_alias.variants 返回空，等于完全没有开销。
        _alias_used = []
        try:
            from term_alias import variants as _alias_variants
            _extra = [v for v in _alias_variants(root) if v not in queries]
            if _extra:
                _alias_used = list(_extra)
        except Exception:                                   # noqa: BLE001
            _extra = []                                     # 别名表坏了不能拖垮主路
        if _extra:
            queries = queries + _extra
        if _alias_used:
            # 记进 plan 供内部过程面板展示（用 update，别整个覆盖 plan）
            plan = dict(plan)
            plan["alias_queries"] = _alias_used
        evs = self._merge(queries)

        # ---------- 拒答前置（组件四） ----------
        # 生成之前先看一眼：连最高分的证据都很不沾边 → 直接判 R1 拒答，
        # 省掉「生成 + 校验」两次 LLM 调用。后面 ②生成中 / ③生成后 两道
        # 拒答路径原样保留，本步只是把"最明显那一档"提前拦掉。
        if self.enable_gate and evs:
            top1 = max((e.score or 0.0) for e in evs)
            if top1 < self.gate_min_top1:
                res = QAAnswer(question=question)
                res.evidences = evs
                res.refused = True
                res.refuse_reason = (f"拒答前置：最高证据相关分 {top1:.4f} "
                                     f"低于阈值 {self.gate_min_top1}")
                res.plan = plan
                res.hops = len(queries)
                res.n_evidence = len(evs)
                res.latency = time.time() - t0
                return res

        res = self.base.answer_from_evidences(question, evs,
                                              history_block=history_block)

        # ---------- 自检回环 ----------
        # 门控说明：**不按 need_split 限定**。第 1 步发现，唯一还失分的 Q36
        # 是"简单题"（不拆解），它的病是"资料到手却漏答"——正是自检要治的。
        # 所以凡"已作答"的题都过一遍自检；简单题通常 1 次即判无缺口，不进循环，
        # 只多一次短调用（代价见验收报告的延迟一栏）。
        log, rounds = [], 0
        first_answer = res.answer
        if self.enable_reflect and not res.refused:
            prev_gaps = None
            for _ in range(self.max_reflect_rounds):
                a = self.reflector.assess(question, res.answer, res.evidences)
                log.append(a)
                gaps = a.get("gaps") or []
                if a.get("covered") or a.get("hopeless") or not gaps:
                    break                         # 无缺口 / 非检索可解 → 收工
                if prev_gaps is not None and gaps == prev_gaps:
                    break                         # 缺口与上轮相同 → 空转，停
                prev_gaps = gaps
                new_evs = self._merge_into(res.evidences, gaps)
                grew = len(new_evs) > len(res.evidences)
                if not grew:
                    # 补检没带回新证据：本轮仍"带着缺口重答"一次（治"资料在
                    # 手却没答全"），但靠缺口重复规则在下一轮收住，不空转。
                    pass
                res = self.base.answer_from_evidences(question, new_evs,
                                                     hints=gaps,
                                                     history_block=history_block)
                rounds += 1
                if res.refused:
                    break

        res.plan = plan
        res.hops = len(queries)
        res.n_evidence = len(res.evidences)
        res.reflect_rounds = rounds
        res.reflect_log = log
        res.first_answer = first_answer
        res.latency = time.time() - t0
        return res


if __name__ == "__main__":
    from qa_chain import build_llm
    from retrievers import HandbookRetriever
    q = sys.argv[1] if len(sys.argv) > 1 else "一门必修课不及格之后，从补考到重修会经历哪些环节？"
    r = HandbookRetriever("data", use_rerank=True, verbose=False)
    pipe = AgentQAPipeline(r, llm=build_llm(), verbose=False)
    out = pipe.ask(q)
    print("拆解：", out.plan)
    print("检索次数：", out.hops, "｜ 证据条数：", out.n_evidence,
          "｜ 自检补检轮数：", out.reflect_rounds)
    for i, a in enumerate(out.reflect_log, 1):
        print(f"  第{i}轮自检：covered={a.get('covered')} hopeless={a.get('hopeless')} "
              f"gaps={a.get('gaps')}")
    print(out.pretty())
