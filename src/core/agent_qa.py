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

import os
import re
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


# 「接着上一句问」的承接词。只认这些字面信号，不按句子长短猜。
_FOLLOWUP_RE = re.compile(
    r"(它|它们|这个|那个|这些|那些|上述|刚才|上面|前面|还有|另外|那么|那|呢|的话|接着|继续)")


def _is_followup(question: str, prev_user: str = "") -> bool:
    """这句是「接着上一句问」，还是另起一个新话题？

    为什么必须判（2026-10-19 线上撞到，本地稳定复现）
    --------------------------------------------------
    下面有一段「追问兜底」：planner 没补出指代时，把「上一句 + 本句」拼成一路
    去检索，靠上一句的词补「它 / 那」——本意是好的。但学生**换话题**时
    （先问「旷课多少学时会被处分」，接着问「学校的班车怎么坐」）这句也照样被
    拼进去：上一句的「旷课 / 处分 / 学时」把《违纪处分办法》顶到 0.986 分，
    模型于是把处分条款照抄出来当答案 —— 学生问班车，得到的是一整页处分细则。
    空上文时同一句只有 0.0008 分、正常拒答，差别全在那一句拼接上。

    ⚠ 只认承接词，**不按句子长短猜**：把「宿舍几点关门」这种短句误判成追问，
      等于把新话题又拼回旧话题，那是最坏的一种错 —— 答得一本正经、引用还是真的。
    """
    if not prev_user:
        return False
    return bool(_FOLLOWUP_RE.search(question or ""))


class AgentQAPipeline:
    def __init__(self, retriever, llm=None, top_k=6, max_merged=28,
                 planner=None, reflect=None, enable_planner=True,
                 enable_reflect=True, max_reflect_rounds=2,
                 enable_gate=True, gate_min_top1=0.03,
                 enable_subject=True,
                 parallel_search=False, search_workers=5,
                 enable_web=True, verbose=True, **qa_kwargs):
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
        # 阈值依据见 src/dev/diag/diag_gate.py：44 题里拒答题最高 0.1104、正常题最低 0.1124，
        # 两档空档仅 0.002 —— 取"中值"必然过拟合，任何一道正常题抖动就会被误拒。
        # 所以这里取远低于正常题最低分的 0.03（余量 0.08），只为拦"最不沾边"那一档，
        # 宁可漏拦、绝不误拦（误拒率 0% 是红线）。
        self.enable_gate = enable_gate
        self.gate_min_top1 = gate_min_top1
        # 主体一致闸（2026-10-18）：证据分很低、且问句里的场所（图书馆/食堂/快递…）
        # 在证据里一个字都没出现 → 判"答非所问"，直接拒答、不调模型。
        # 治的是「问图书馆几点开门，答成学生公寓大门 6:00–23:30」那一类错答。
        # 判据、实测数据与零误伤论证见 src/retrieval/subject_guard.py 文件头。
        # 关掉的办法：环境变量 SUBJECT_GUARD=off，或构造时传 enable_subject=False。
        self.enable_subject = enable_subject
        # 官网兜底（2026-10-17）：手册判「没有」之后，再去学校官网找一趟。
        # 硬原则见 src/retrieval/web_fallback.py 文件头 —— 手册优先、来源单标、
        # 同样只许照抄。它只在这一个位置生效：**手册已经答不出来**时；
        # 手册能答的题一个字都不掺官网内容。
        # 关掉的办法：环境变量 WEB_FALLBACK=off（不用改代码，也不用重新部署）。
        self.enable_web = enable_web and (
            (os.environ.get("WEB_FALLBACK") or "").strip().lower()
            not in ("off", "0", "false", "no"))
        self.web_budget = 8.0       # 官网兜底的总预算秒数（超时就照旧拒答）
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
        由 `src/infra/memory.py` 拼好。它有两个用处 ——
          ① 规划器据此产出 standalone（把「它/这个」补成完整问题），检索拿
             完整语义去搜，追问题才搜得到东西；
          ② 生成时作为背景喂进提示词，让回答接得上上文（但它**不是证据**，
             见 qa_chain.SYSTEM_PROMPT 第 8 条）。
        `prev_user`（可选）：上一轮学生问的那句原文。小模型不一定每次都肯
        把追问改写干净，所以再留一条**确定性**兜底 —— 见 `_ask` 里的注释。
        两个都不传 = 原来的「每题独立」行为，完全向后兼容。
        """
        # ---------- 个人数据：**提前**分流（2026-10-17） ----------
        # 为什么必须放在最前面，而不是等手册判拒之后再分流：这类问题手册里
        # 本来就不该有，但检索分并不低 —— 实测问「我这学期绩点多少」，
        # 手册《学分绩点计算办法》能匹配到 0.666 分，免费模型真的会拿"绩点
        # 怎么算"去答，学生看到的是一段规则，而不是自己的分。他要的是
        # 「我没有这个数，得你自己去系统查」。所以命中就**不检索、不调模型**，
        # 0 秒给引导话术（判据与话术见 src/retrieval/need_login.py，三道闸
        # 故意收紧 —— 误把「奖学金评选条件」挡成个人数据，代价更大）。
        if self.enable_web:
            hit = self._login_guard(question)
            if hit:
                res = QAAnswer(question=question)
                res.refused = True
                res.web_login = hit
                res.refuse_reason = f"个人数据（{hit['topic']}），需登录学校系统查看"
                res.plan = {"personal_data": hit["topic"],
                            "reason": "命中个人数据主题，提前分流，未检索手册"}
                res.latency = 0.0
                if self.verbose:
                    print(f"[个人数据] 提前分流：{hit['topic']}", flush=True)
                return res

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

        # ---------- 官网兜底（2026-10-17） ----------
        # 只有走到这里才说明「整条链路正常跑完、结论是手册里没有」——
        # 这才是去官网找一趟的时机。**技术故障不走这条**（上面已经 return 给
        # _rescue 了）：那是系统坏了，不是手册没规定，拿官网内容顶上会把故障
        # 掩盖成「手册里没有，但官网有」，管理员就再也看不出后台瘫了。
        if res.refused and self.enable_web:
            res.web, res.web_login = self._web_lookup(question)
        return res

    def _login_guard(self, question: str):
        """问的是「我自己的数据」→ 返回引导话术；否则 None。

        判据和话术都在 `src/retrieval/need_login.py`（三道闸：得自称、得是个人
        数据主题、不能是公共规则），这里只负责「提前拦」这个动作，不复制词表 ——
        词表只留一处，将来加词不用改两遍。
        """
        try:
            from need_login import check as _check
            return _check(question)
        except Exception:                                   # noqa: BLE001
            return None                                     # 词表坏了不能拖垮主路

    def _subject_guard(self, question: str, evs):
        """问的场所是不是证据里真有 → 没有就返回说明字典；否则 None。

        判据全在 `src/retrieval/subject_guard.py`（两道闸 + 词表），这里只做
        「防御式调用」：闸门自己写错了，宁可放行也不能让整题崩掉。
        """
        try:
            from subject_guard import check as _check
            return _check(question, evs)
        except Exception:                                   # noqa: BLE001
            return None

    def _web_lookup(self, question: str):
        """手册判拒之后，去学校官网找一趟。返回 (官网素材, 空)。

        三条纪律（与 src/retrieval/web_fallback.py 文件头一致）：
          ① 只在**手册已判拒**之后调用 —— 手册能答的题，官网一个字都不掺；
          ② 异常一律吞掉：官网挂了就只能拒答，绝不能让它把主链路拖塌；
          ③ 个人数据在 `ask()` 开头就已经分流掉了（`_login_guard`），走到这里
             说明确实该去公开网页找 —— 万一前置漏判，让官网再试一次也比直接
             拒答好（官网本来就查不到别人的成绩）。
        """
        t0 = time.time()
        try:
            from web_fallback import ask_web, rewrites as _rewrites
            # 先拿原话搜；搜不到就把学生用词换成官网用词再搜一次。
            # 为什么不能只塞关键词（2026-10-20 实测）：官网那条链路后面还有一道
            # 「这句得覆盖问题里三成的词」，问题本身仍是「校车」，时刻表里没有
            # 「校」字 → 搜到了也全被筛掉。问「环校巴士」原来就是这样静默失败的。
            tries = [question] + _rewrites(question)
            deadline = time.time() + self.web_budget * 1.6
            web = None
            for qv in tries:
                left = deadline - time.time()
                if left < 3 and web is not None:
                    break
                w = ask_web(qv, budget=max(3.0, min(self.web_budget, left)))
                w["query_used"] = qv
                if web is None or (not web.get("found") and w.get("found")):
                    web = w
                if w.get("found") or w.get("links"):
                    web = w
                    break
        except Exception as e:                              # noqa: BLE001
            # 留一条 reason 让页面与日志都看得出「是官网这条路坏了」，
            # 而不是把它当成「官网也没有」——两者对学生是同一句拒答，
            # 对排查是天壤之别。
            return ({"found": False, "reason": "error", "passages": [],
                     "tables": [], "links": [],
                     "error": f"{type(e).__name__}: {str(e)[:160]}",
                     "fetched_at": time.strftime("%Y-%m-%d %H:%M"),
                     "stats": {"elapsed": round(time.time() - t0, 1)}}, {})

        if self.verbose:
            used = web.get("query_used") or question
            print(f"[官网兜底] {web.get('reason')} ｜ 命中 {len(web.get('passages') or [])} 段"
                  f" / 表 {len(web.get('tables') or [])} 张"
                  f" ｜ 搜的是「{used}」"
                  f" ｜ {web.get('stats', {}).get('elapsed', 0)}s", flush=True)
        return web, {}

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
        # 兜底这一路可能也是降级检索（向量接口不可用 → 只用关键词），留痕。
        _deg = list(getattr(self.retriever, "degraded", []) or [])
        if _deg:
            res.plan["degraded_retrieval"] = _deg
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
            # ⚠ 这里**不能**照抄「手册里没有」——本次根本没走成智能体链路
            # （大模型或检索挂了一环），而程序化兜底又只会照抄原文、不会改写
            # 措辞，所以「它没找到」≠「手册里没有」。说成前者会让学生以为
            # 手册里就是没规定，其实过几分钟再问就能答（2026-10-16 线上
            # 编码接口额度用完，整站每题都栽在这一步，页面却谎称「手册里没有」）。
            res.refuse_reason = f"大模型不可用（{note}），程序化兜底也没找到可用条款"
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
        # 换了新话题就别让上文碰到检索（2026-10-19，缘由见 _is_followup 的注释）：
        # 不是追问时，连 planner 的补全句也不认 —— 小模型被吩咐过「别把上一轮搬
        # 进来」，但它不一定听；原句本身语义完整，直接用原句检索最安全。
        _fu = _is_followup(question, prev_user)
        if not _fu:
            root = question
        queries = ([root] + plan["sub_questions"] if need_split else [root])
        # 追问兜底（确定性，不指望小模型）：本次是追问、而模型又**没**把指代
        # 补出来（standalone 与原句一样）时，再补一路「上一句 + 本句」——
        # 靠上一个话题的词把「它/那」补上。例：「那要多久呢？」单独拿去检索
        # 什么都没；拼成「重修要什么条件 那要多久呢」就能搜到重修条款。
        if (_fu and prev_user and history_block
                and _same_q(root, question) and prev_user not in queries):
            queries = queries + [f"{prev_user} {question}"]
        # 追问时「原句」也单独搜一路（2026-10-19）。planner 的补全会**加戏**：
        # 问「那吹风机呢？」它补成「那吹风机可以用来吹头发吗」——「公寓」这个
        # 场景词丢了，"吹头发"又把相关性稀释掉，最高分只有 0.021，整题被判拒答；
        # 而原句「那吹风机呢？」单独搜是 0.91。合并按条款取最高分，多一路只增不减。
        if _fu and not _same_q(root, question):
            queries.append(question)
        # 口语词换手册词（2026-10-16）：学生说「体测」，手册写「体质测试」，
        # 字面一个字都不重合 → BM25 零命中、向量也拉不起来，最高分只有 0.05，
        # 最后被引用校验判「证据只是擦边」整题拒答。这里多走几路「换过词的
        # 同义问句」（追加/前缀/替换三种写法），合并时按条款取最高分，
        # 所以只会多找到东西、不会弄丢原来的。命中的词见 src/retrieval/term_alias.py。
        # 没命中任何口语词时 term_alias.variants 返回空，等于完全没有开销。
        _alias_used = []
        try:
            from term_alias import variants as _alias_variants
            # 对「补全句」和「原句」都换一轮词：追问时两者都在 queries 里，
            # 只换补全句的话，原句里那个具体词（吹风机…）就白瞎了。
            _srcs = [root] if _same_q(root, question) else [root, question]
            _extra = []
            for _s in _srcs:
                for v in _alias_variants(_s):
                    if v not in queries and v not in _extra:
                        _extra.append(v)
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

        # 检索降级留痕（2026-10-16）：向量编码 / 重排走的是外部接口，接口一旦
        # 不可用，检索会**自动降级**成只用关键词匹配（见 retrievers._mark_degraded），
        # 而不是把整道题判死。这里把降级原因带出来，页面上要明说 ——
        # 隐性的质量下降比明确的失败更坏：学生会以为「手册里就那么写的」。
        _deg = list(getattr(self.retriever, "degraded", []) or [])
        if _deg:
            plan = dict(plan)
            plan["degraded_retrieval"] = _deg

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

        # ---------- 主体一致闸（2026-10-18） ----------
        # 上面那道闸只看"分低不低"，这道看"问的和答的是不是同一个地方"。
        # 起因：手册里没有图书馆开放时间，但《学生公寓管理办法》第十九条
        # 「学生公寓大门开放时间为每天 6:00 至 23:30」句式太像，模型直接把
        # 「图书馆」代进句式答了出来 —— 引用是真的，主体是假的。
        # 判据见 src/retrieval/subject_guard.py：**证据分 <0.08 且主体词在证据里
        # 只占不到两成**才拦；0.08 这条线在 60 题所有正常题（最低 0.1118）之下，
        # 所以题只要有一点像样证据，本闸根本不启动 —— 零误伤是算出来的。
        # 拦下 = 判"手册里没有"，`ask()` 随后自然会去官网兜一次。
        if self.enable_subject:
            subj = self._subject_guard(question, evs)
            if subj:
                res = QAAnswer(question=question)
                res.evidences = evs
                res.refused = True
                res.refuse_reason = subj["reason"]
                plan = dict(plan)
                plan["subject_mismatch"] = {
                    k: subj[k] for k in ("subject", "top", "support", "ratio")}
                res.plan = plan
                res.hops = len(queries)
                res.n_evidence = len(evs)
                res.latency = time.time() - t0
                if self.verbose:
                    print(f"[主体一致] {subj['reason']}", flush=True)
                return res

        # 降级时把 degraded 交给生成层：那边会关掉「拒答复核」。
        # 理由见 qa_chain 里那段注释 —— 降级后分数是 BM25 原始分，量纲不同、
        # 与真题/弃答题的重叠严重，拿它去驳模型只会逼出一个硬凑的答案。
        res = self.base.answer_from_evidences(question, evs,
                                              history_block=history_block,
                                              degraded=bool(_deg))

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
                                                     history_block=history_block,
                                                     degraded=bool(_deg))
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
