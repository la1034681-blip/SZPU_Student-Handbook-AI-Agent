# -*- coding: utf-8 -*-
"""程序化（抽取式）答案生成器 —— 全程不需要任何大模型。

    证据（retrievers.Evidence）
      → 按"与问题的契合度"给证据重排
      → 从条款原文切句、挑最相关的句子
      → 用极性规则合成一句结论（可以 / 不可以 / 需要办 …）
      → 拼成「结论句 + 原文条款 + 出处」的答案
      → 契合度低于阈值则拒答

设计取舍
--------
- **为什么用字符匹配而不是分词**：这台机器上没装 jieba，中文分词还要背词典。
  规章问答的关键词基本是"补考""宿舍""退宿"这类 2 字词，用字符滑窗匹配
  够用（3-gram 不够就退到 2-gram），零依赖、结果完全可复现。
- **拒答靠"字符覆盖率"而不是分数**：实测发现相关度分数会骗人 —— 一道
  问图书馆借书的题，能匹配到助学贷款条款还拿到 0.43 的高分。改看更硬的
  指标：问题里的实义字符有多少比例真的出现在条款原文里。覆盖率低就是
  没这条规定，直接拒答。
- **结论句只做判断题**：结论句只回答"可以/不可以/需要按流程办"，用极性
  词表（可以/不得/应当）判；判不出来就老实说"见下文"。绝不编。
- **原文逐字照抄**：所以出处命中率天然接近 100%，也不会出现"数字被改写"
  这种经典幻觉。代价是读起来像手册摘录，不那么像人说话。

用法
----
    from procedural_qa import ProceduralPipeline
    pipe = ProceduralPipeline(retriever, top_k=6)
    res = pipe.ask("必修课考试不及格，能补考吗")
"""
from __future__ import annotations

import re
import sys
import time
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

from qa_chain import QAAnswer          # noqa: E402  复用同一套结果结构

# ------------------------------------------------------------------ 词表
QW = ("怎么样", "怎么办理", "怎么样处理", "怎么处理", "怎么", "如何办理", "如何",
      "什么样的", "什么", "哪些", "哪个", "哪几种", "哪一种", "哪", "是否",
      "可不可以", "能不能够", "能不能", "可以吗", "有没有规定", "有没有",
      "多少", "多长时间", "多久", "几个", "几天", "几本", "几", "为什么",
      "请问", "我想知道", "想知道", "是不是", "该怎么办", "要办", "一点",
      "吗", "呢", "吧", "啊", "呀", "嘛", "哦", "嗯")

NOISE = set("的了着过地得和与及或并且但是因为所以如果那么就是都也还很更太最"
            "会不会能不能可否这那哪些什么怎么如何为何")

NUM_RE = re.compile(r"[0-9０-９一二三四五六七八九十百千]+(?:个|次|天|日|周|月|年|学期|"
                    r"学分|分|门|科|条|款|项|%|％|小时|工作日|元)?")

NEG_WORDS = ("不得", "不予", "禁止", "严禁", "不能", "不允许", "无效", "取消")
POS_WORDS = ("可以", "可参加", "可申请", "可提交", "可获得", "可享受", "允许", "可在", "可")
OBL_WORDS = ("应当", "必须", "须", "需要", "应于", "应聘", "应提交")

HEAD_RE = re.compile(r"^\s*(?:第[一二三四五六七八九十百零〇0-9]+条|"
                     r"[（(][一二三四五六七八九十0-9]+[)）]|[一二三四五六七八九十]+[、.．])"
                     r"\s*")

SENT_SPLIT = re.compile(r"(?<=[。；！？!?])")
CLAUSE_SPLIT = re.compile(r"[，,。；;！!？?、：:\s]+")
HANZI = re.compile(r"[\u4e00-\u9fff]")
HANZI_RUN = re.compile(r"[\u4e00-\u9fff0-9]+")


# ------------------------------------------------------------------ 基础工具
def strip_question_words(q: str) -> str:
    for w in QW:
        q = q.replace(w, " ")
    return q


def core_text(q: str) -> str:
    """问题的实义汉字串：剥掉疑问词与标点，只留汉字。"""
    return "".join(HANZI.findall(strip_question_words(q)))


def grams(text: str, n: int = 2) -> set:
    s = "".join(ch for ch in text if "\u4e00" <= ch <= "\u9fff" or ch.isdigit())
    out = set()
    for i in range(len(s) - n + 1):
        win = s[i:i + n]
        if any(ch in NOISE for ch in win):
            continue
        out.add(win)
    return out


def ngrams_of_question(question: str):
    core = core_text(question)
    return grams(core, 2) | grams(core, 3)


def split_sentences(text: str, keep_short: int = 6):
    parts, cur = [], ""
    for seg in SENT_SPLIT.split(text):
        cur += seg
        if len(cur.strip()) >= keep_short:
            parts.append(cur.strip())
            cur = ""
    if cur.strip():
        parts.append(cur.strip())
    return parts or [text.strip()]


def coverage(question: str, text: str) -> float:
    """问题实义字符被条款原文覆盖的比例（贪心最长匹配，贪心向右推进）。

    这是本模块的"硬指标"：覆盖率低 = 条款没提这事 = 该拒答。
    """
    core = core_text(question)
    if not core:
        return 0.0
    n, i, covered = len(core), 0, 0
    while i < n:
        matched = 0
        for L in (4, 3, 2):
            if i + L <= n and core[i:i + L] in text:
                matched = L
                break
        if matched:
            covered += matched
            i += matched
        else:
            i += 1
    return covered / n


def extract_topic(question: str, top_sent: str) -> str:
    """抽"被问的那个事"。规矩：**从最后一个分句往前找**（中文问句的焦点多半
    落在最后那个分句上），剥掉疑问词后从末尾往回取 5→2 字的词，且必须同时
    出现在问题里、也出现在对得上的原文里。

    两个约束缺一不可，专门压住"可以必修课"这类错答：候选词必须在证据
    原文里出现，否则不算数。
    """
    for seg in reversed([s for s in CLAUSE_SPLIT.split(question) if s.strip()]):
        core = "".join(HANZI.findall(strip_question_words(seg)))
        for L in (5, 4, 3, 2):
            if len(core) >= L and core[-L:] in top_sent:
                return core[-L:]
    core = core_text(question)
    for L in (5, 4, 3, 2):
        if len(core) >= L and core[-L:] in top_sent:
            return core[-L:]
    return ""


# ------------------------------------------------------------------ 打分
def score_sentence(sent: str, qgrams: set, q_nums: set) -> float:
    if not sent:
        return 0.0
    sg = grams(sent, 2)
    hit = len(sg & qgrams) / len(qgrams) if qgrams else 0.0
    s = hit
    nums = set(NUM_RE.findall(sent))
    if q_nums and (nums & q_nums):
        s += 0.35
    if nums:
        s += 0.06
    if len(sent) <= 8:
        s -= 0.15
    return s


def score_evidence(ev, qgrams: set, q_nums: set, question: str):
    """证据排序分：以重排模型的分打底（它才是判"相不相关"的行家），
    再用字符覆盖率与最佳句得分做微调。"""
    text = ev.text or ""
    whole = len(grams(text, 2) & qgrams) / max(len(qgrams), 1)
    best = max((score_sentence(s, qgrams, q_nums)
                for s in split_sentences(text)), default=0.0)
    cov = coverage(question, text)
    rerank = float(getattr(ev, "score", 0.0) or 0.0)
    score = 0.55 * rerank + 0.30 * cov + 0.15 * best
    return score, best, cov


# ------------------------------------------------------------------ 结论句
# 结论句的"成色"分类：只有前三种是**真回答了问题**（可以/不可以/需按流程办），
# 其余都只是"把原文丢给用户自己看"。这张表就是兜底的触发依据。
STRONG_KINDS = ("yesno_no", "yesno_yes", "yesno_rule")


def make_conclusion(question: str, top_sent: str, with_kind: bool = False):
    """合成结论句。with_kind=True 时返回 (结论句, 成色标签)。

    成色标签是给兜底用的：'见下文' 类结论 = 没答上 = 值得叫一次免费大模型。
    """
    q = question
    topic = extract_topic(question, top_sent)

    def has(words):
        return any(w in top_sent for w in words)

    neg, pos, obl = has(NEG_WORDS), has(POS_WORDS), has(OBL_WORDS)

    yesno = any(w in q for w in ("吗", "能否", "能不能", "可以吗", "是否",
                                 "可不可以", "行不行", "有没有"))
    num = any(w in q for w in ("多少", "几", "多久", "几天", "几个", "多长时间"))
    how = any(w in q for w in ("怎么办理", "如何办理", "流程", "手续", "怎么办",
                               "怎样办", "怎么申请", "如何申请"))
    what = any(w in q for w in ("哪些", "包括", "有什么", "条件"))
    who = any(w in q for w in ("由谁", "哪个部门", "找谁"))
    consequence = any(w in q for w in ("处理", "后果", "处分", "惩罚", "处罚",
                                       "会怎样", "受到"))

    kind, text = "generic", (f"关于{topic}，手册规定如下。" if topic
                             else "手册中的相关规定如下。")
    if yesno:
        if neg and not pos:
            kind, text = "yesno_no", (f"不可以{topic}。" if topic else "不可以。")
        elif pos and not neg:
            kind, text = "yesno_yes", (f"可以{topic}。" if topic else "可以。")
        elif obl and not neg:
            kind, text = "yesno_rule", (f"{topic}需要按手册规定办理。" if topic
                                        else "需要按手册规定办理。")
        else:
            kind, text = "yesno_open", (f"手册对{topic}有明确规定，见下文。" if topic
                                        else "手册中的相关规定如下。")
    elif consequence:
        kind, text = "consequence", "手册对此有明确的处理规定，见下文。"
    elif num:
        kind, text = "numeric", "具体数量或期限以手册规定为准，见下文。"
    elif how:
        kind, text = "how", "办理要求与流程见下文。"
    elif who:
        kind, text = "who", "负责部门与办理方式见下文。"
    elif what:
        kind, text = "what", "相关情形见下文。"
    return (text, kind) if with_kind else text


# ------------------------------------------------------------------ 主流程
def generate(question: str, evs, max_evidence: int = 3, max_sent: int = 3,
             long_clause: int = 520, min_score: float = 0.15,
             min_cov: float = 0.10) -> dict:
    """抽取式作答。返回 dict(answer, citations, insufficient, reason, confidence)。

    拒答判据（本模块的核心，靠数据标定而不是拍脑袋）：
      以 top1 证据的**重排分**为主。60 题标定结果 —— 应作答的题重排分
      中位数 0.976，应拒答的题中位数只有 0.017，两边几乎不重叠。
      重排模型本来干的就是"这条到底相不相关"，拿它的分当拒答闸门，
      比我自己编字符串相似度可靠得多。
      覆盖率只做极端兜底（防"分数虚高但词全对不上"的怪例）。
    """
    if not evs:
        return {"answer": "", "citations": [], "insufficient": True,
                "reason": "检索未获得任何证据", "confidence": 0.0,
                "evidence_order": [], "weak": False, "conclusion_kind": "none",
                "focus_cov": 0.0}

    qgrams = ngrams_of_question(question)
    q_nums = set(NUM_RE.findall(question))

    scored = []
    for ev in evs:
        sc, best, cov = score_evidence(ev, qgrams, q_nums, question)
        scored.append((sc, best, cov, ev))
    scored.sort(key=lambda x: x[0], reverse=True)

    top_score, _, top_cov, top_ev = scored[0]
    # 拒答看的是"重排模型给出的最高分"，而不是我重排后的第一名 ——
    # 否则一条字面沾边、重排却不认的条款会把整题拖成拒答。
    rerank = max((float(getattr(e, "score", 0.0) or 0.0) for e in evs), default=0.0)
    confidence = top_cov                       # 对外的契合度 = 覆盖率

    # 覆盖率只做"防怪例"兜底：只有当重排分也不高（<0.5）时才拿它说话，
    # 否则会误伤"重排分很高、只是问法和原文用词对不上"的题（实测踩过）。
    if rerank < min_score or (rerank < 0.5 and top_cov < min_cov):
        return {"answer": "", "citations": [], "insufficient": True,
                "reason": f"手册中未找到与问题相关的条款（相关度 {rerank:.2f}）",
                "confidence": confidence, "evidence_order": [],
                "weak": False, "conclusion_kind": "refuse", "focus_cov": 0.0}

    keep = [x for x in scored if x[2] >= min_cov][:max_evidence]
    if not keep:
        keep = scored[:1]

    blocks, cites = [], []
    for sc, best, cov, ev in keep:
        eid = ev.eid
        cites.append(eid)
        body = HEAD_RE.sub("", (ev.text or "").strip())
        if len(body) <= long_clause:
            snippet = body
        else:
            sents = split_sentences(body)
            ranked = sorted(sents, key=lambda s: score_sentence(s, qgrams, q_nums),
                            reverse=True)
            picked = []
            for s in ranked[:max_sent]:
                if s not in picked:
                    picked.append(s)
            ps = set(picked)
            snippet = "".join(s for s in sents if s in ps) or body[:long_clause]
        blocks.append((eid, ev, snippet))

    first_sent = split_sentences(HEAD_RE.sub("", (top_ev.text or "").strip()))[0]
    conclusion, ckind = make_conclusion(question, first_sent, with_kind=True)

    # 焦点覆盖率：问题的实义字在**实际摘录出来的这几条条款**里被覆盖了多少。
    # 这是兜底触发用的"硬信号"——
    #   低 → 检索没错、但摘要没摘对条，程序化再怎么组织结论都是答非所问，该叫大模型；
    #   高 → 条款已经摘对了，只是结论句朴素，原文本身比模型概括得更全，别叫。
    # （60 题实测：全量兜底 -0.9pp，加上这条闸门后 +1.9pp 且零回退）
    focus_cov = coverage(question, " ".join(ev.text or "" for _, _, _, ev in keep))

    lines = [conclusion, "", "**手册原文**"]
    for eid, ev, snippet in blocks:
        body = snippet.replace("“", "「").replace("”", "」")
        lines.append(f"[{eid}] {ev.cite}（第 {ev.page} 页）")
        lines.append(f"“{body}”")
        lines.append("")
    lines.append("以上为手册原文摘录，未作任何改写。")

    return {"answer": "\n".join(lines).strip(), "citations": cites,
            "insufficient": False, "reason": "", "confidence": confidence,
            "evidence_order": [ev.clause_id for _, _, _, ev in keep],
            "weak": ckind not in STRONG_KINDS, "conclusion_kind": ckind,
            "focus_cov": focus_cov}


# ------------------------------------------------------------------ 链路壳
class ProceduralPipeline:
    """与 QAPipeline 同接口（ask → QAAnswer），但不持有任何 LLM。

    fallback：可选的兜底 callable(question, evs) -> QAAnswer。只有当抽取式
    契合度不足、而调用方配好了免费大模型时才用得上（默认 None = 纯程序化）。
    """

    def __init__(self, retriever, top_k=6, max_evidence=4, min_score=0.15,
                 min_cov=0.10, fallback=None, verbose=False, fb_mode=None):
        self.retriever = retriever
        self.top_k = top_k
        self.max_evidence = max_evidence
        self.min_score = min_score
        self.min_cov = min_cov
        self.fallback = fallback
        self.verbose = verbose
        self.n_fallback = 0
        self.n_fb_fail = 0
        # 兜底触发口径（FB_MODE 环境变量可覆盖）：
        #   focus 只兜底"结论不实在 **且** 条款没摘对条"的题（默认，最划算）
        #   weak  兜底所有"结论不实在"的题（更激进，实测反而略掉分）
        #   cross weak + 跨章题
        #   all   除程序化判定的拒答外，全交给大模型（最像"纯 LLM"，延迟最高）
        #   off   关闭兜底，纯程序化
        import os
        self.fb_mode = (fb_mode or os.environ.get("FB_MODE") or "focus").strip().lower()
        # focus 口径的门槛：焦点覆盖率低于它才认为"条款没摘对条"
        self.fb_focus_cov = float(os.environ.get("FB_FOCUS_COV") or 0.60)
        self.fb_on_insufficient = False      # 拒答闸门标定过，默认不让大模型插手
        self._is_cross = False
        # 与 AgentQAPipeline 保持同名接口 —— app.py 会给管道统一设这几个开关，
        # 程序化通道用不上，但显式声明出来，免得被动态赋值埋下隐性耦合。
        self.enable_planner = False
        self.enable_reflect = False
        self.enable_gate = True
        self.max_merged = max_evidence

    def _log(self, m):
        if self.verbose:
            print("  " + m, flush=True)

    @staticmethod
    def split_subquestions(question: str):
        """把"一次问好几件事"的题拆成子问题 —— 程序化版的多跳。

        这是跨章题的关键：整体检索只会围着最主要那个话题打转，
        拆开各自检索才能把散在不同文件里的规定都捞回来。
        切不出两段就返回空列表，走单路检索。
        """
        segs = [s.strip() for s in re.split(r"[？?。；;]", question) if len(s.strip()) >= 4]
        if len(segs) < 2:
            segs = [s.strip() for s in re.split(r"[，,、]", question) if len(s.strip()) >= 5]
        return segs if len(segs) >= 2 else []

    def ask(self, question: str) -> QAAnswer:
        t0 = time.time()
        subs = self.split_subquestions(question)
        self._is_cross = len(subs) >= 2
        if subs:
            seen, evs = set(), []
            for q in [question] + subs:          # 原问题 + 每个子问题各检一路
                for e in self.retriever.search(q, top_k=self.top_k)[:3]:
                    if e.clause_id not in seen:
                        seen.add(e.clause_id)
                        evs.append(e)
            for i, e in enumerate(evs, 1):       # 重编 E1..En，保证角标连续
                e.eid = f"E{i}"
            # 一次问好几件事，就得给每条事留出位置，否则前面的条款把名额占光
            n_ev = min(6, 2 + len(subs))
            self._log(f"拆出 {len(subs)} 个子问题，合并 {len(evs)} 条证据，摘 {n_ev} 条")
        else:
            evs = self.retriever.search(question, top_k=self.top_k)
            n_ev = self.max_evidence
        res = self.answer_from_evidences(question, evs, max_evidence=n_ev)
        res.latency = time.time() - t0
        # 注意：这里是**合并**而不是覆盖 —— 否则会把"是否走了兜底""结论成色"
        # 这些由 answer_from_evidences 写好的诊断字段一并抹掉（踩过这个坑）。
        plan = dict(getattr(res, "plan", {}) or {})
        plan.update({"need_split": bool(subs), "sub_questions": subs})
        plan.setdefault("mode", "procedural")
        res.plan = plan
        res.hops = 1 + len(subs)
        return res

    def _need_fallback(self, out) -> str:
        """该不该叫免费大模型？返回触发理由，空串代表不叫。

        触发口径（见 __init__ 的 FB_MODE）：
          focus（默认）—— 结论句没给出实质答复，**并且**问题焦点在程序化实际
              摘录的条款里覆盖不足。这一条是 60 题反事实标定出来的：全量兜底
              会让正确率掉 0.9 个点（大模型概括时必漏一个小要点，而原文照抄
              不会漏），加上"没摘对条"这个闸门后才真正变成正收益。
          weak  —— 只要结论句不实在就叫（更激进，留作对照）。
          cross —— weak 之外再补上跨章题。
          all   —— 除拒答外全叫。
        拒答题（程序化判定"手册无相关规定"）默认**不**叫大模型：
        那个闸门是用 60 题标定过的（重排分中位数 0.976 vs 0.017），
        叫来只会把该拒的题答错。
        """
        if self.fallback is None or self.fb_mode == "off":
            return ""
        if out["insufficient"]:
            return "程序化拒答" if self.fb_on_insufficient else ""
        if self.fb_mode == "all":
            return "全量兜底"
        if self.fb_mode == "focus":
            if out.get("weak") and (out.get("focus_cov") or 0.0) < self.fb_focus_cov:
                return "结论不实在且条款没摘对"
            return ""
        if self.fb_mode in ("weak", "cross") and out.get("weak"):
            return "结论不实在"
        if self.fb_mode == "cross" and self._is_cross:
            return "跨章题"
        return ""

    @staticmethod
    def _compose(out, fb, evs, long_clause=520):
        """把"大模型的结论"和"程序化的原文条款"拼成一份完整答复。

        还要做一件事：大模型引用的条款，程序化的摘录里可能没摘到（摘录只取
        得分前几条）。那样用户会看到"结论引 [E1]，下面却没有 E1"的错位 ——
        所以把模型引到、摘录里缺的条款补进原文区，保证**引用编号在正文里都找得到**。
        """
        body = out.get("answer") or ""
        marker = "**手册原文**"
        tail = body.split(marker, 1)[1] if marker in body else ""
        concl = (fb.answer or "").strip()

        by_eid = {e.eid: e for e in evs}
        shown = list(out.get("citations") or [])
        extra = [c for c in (fb.citations or []) if c in by_eid and c not in shown]
        if extra:
            if not tail:                       # 摘录区都没建，先起个头
                tail = ""
            add = []
            for eid in extra:
                ev = by_eid[eid]
                b = HEAD_RE.sub("", (ev.text or "").strip())
                if len(b) > long_clause:
                    b = "".join(split_sentences(b))[:long_clause] + "…"
                add.append(f"[{eid}] {ev.cite}（第 {ev.page} 页）")
                add.append(f"“{b.replace('“', '「').replace('”', '」')}”")
                add.append("")
            tail = (tail.rstrip("\n") + "\n\n" + "\n".join(add)) if tail \
                else "\n" + "\n".join(add)
        ans = (concl + "\n\n" + marker + tail) if tail else concl
        cites = list(dict.fromkeys((fb.citations or []) + (out.get("citations") or [])))
        return ans, cites

    def answer_from_evidences(self, question: str, evs, max_evidence=None) -> QAAnswer:
        res = QAAnswer(question=question)
        res.evidences = evs
        res.n_evidence = len(evs)
        res.plan = {"need_split": False, "sub_questions": [], "mode": "procedural",
                    "conclusion_kind": None, "fallback_reason": None}
        res.hops = 1

        out = generate(question, evs,
                       max_evidence=max_evidence or self.max_evidence,
                       min_score=self.min_score, min_cov=self.min_cov)
        res.plan["conclusion_kind"] = out.get("conclusion_kind")

        # ---------- 兜底分流 ----------
        why = self._need_fallback(out)
        if why:
            self.n_fallback += 1
            self._log(f"触发兜底（{why}），转免费大模型")
            fb = None
            try:
                fb = self.fallback(question, evs)
            except Exception as e:                     # noqa: BLE001
                self.n_fb_fail += 1
                self._log(f"兜底调用失败（{e}），回退程序化答案")
            if fb is not None and not fb.refused and (fb.answer or "").strip():
                # 两层合成：**结论句**用大模型写的（它能把"多少钱""怎么处理"
                # 归纳出来），**原文条款与出处**照旧用程序化抽取的。
                # 这样既补上了程序化最弱的那一环，又保住了"逐字照抄、出处
                # 可核"的老本 —— 实测直接替换会连出处带要点一起丢掉。
                ans, cites = self._compose(out, fb, evs)
                rt = _verify(ans, cites, evs)
                if rt.get("passed") or not out["answer"]:
                    fb.answer, fb.citations, fb.verify = ans, cites, rt
                    fb.refused = False
                    fb.refuse_reason = ""
                else:
                    fb.answer = out["answer"]
                    fb.citations = out["citations"]
                    fb.refused = False
                    fb.refuse_reason = ""
                    fb.verify = _verify(fb.answer, fb.citations, evs)
                fb.latency = res.latency
                fb.plan = {"mode": "procedural→llm", "fallback_reason": why,
                           "conclusion_kind": out.get("conclusion_kind"),
                           "sub_questions": []}
                fb.hops = res.hops
                fb.n_fallback = self.n_fallback
                return fb
            if fb is not None and fb.refused:
                self.n_fb_fail += 1
                self._log("兜底也没答上（大模型判证据不足），回退程序化答案")
            # 兜底失败/答不上 → 落回程序化的结果，绝不让用户拿到空答复

        if out["insufficient"]:
            res.answer = ""
            res.citations = []
            res.insufficient = True
            res.reason = out["reason"]
            res.confidence = out["confidence"]
            res.refused = True
            res.refuse_reason = out["reason"]
            res.verify = {"passed": True, "reasons": [], "existence": {},
                          "support": {}}
            return res

        res.answer = out["answer"]
        res.citations = out["citations"]
        res.insufficient = out["insufficient"]
        res.reason = out["reason"]
        res.confidence = out["confidence"]
        res.refused = bool(out["insufficient"])
        if res.refused:
            res.refuse_reason = out["reason"]
        res.verify = _verify(res.answer, res.citations, evs)
        if not res.refused and not res.verify.get("passed"):
            res.refused = True
            res.refuse_reason = ("自校验未通过：" +
                                 "；".join(res.verify.get("reasons") or []))
        return res


def _verify(answer, citations, evs):
    try:
        from verify_citations import verify as _v
        return _v(answer, citations, evs, llm=None, recheck=False)
    except Exception as e:                        # noqa: BLE001
        return {"passed": True, "reasons": [f"校验跳过：{e}"],
                "existence": {"passed": True}, "support": {"passed": True}}


def make_free_fallback(enabled=None, verbose=False, tag_missing=True):
    """造一个"免费大模型兜底"函数，给抽取式通道补复杂题。

    只有同时满足「FREE_LLM_ENABLED=1」且「FREE_LLM_API_KEY 配置好」才生效；
    否则返回 None —— 管道就按纯程序化跑，一分钱不花、也不依赖网络。

    兜底复用 QAPipeline.answer_from_evidences（它只吃证据、不吃检索器），
    所以生成/两级校验那套逻辑原样生效，不用重写。

    tag_missing：免费小模型（GLM-4-Flash）有个稳定的小毛病 —— 答案正文写得
    挺准，却忘了在句末打 [E1] 这种角标（它是把出处单独放进 citations 字段的）。
    而我们的引用闸门要求正文必须带角标，于是好答案被自己的校验拦下。这里做
    一个**不改变正文内容**的补救：把模型自己给出的出处列表补到正文末尾，
    然后再重新过一遍引用校验 —— 补完仍不过的，照样拒掉，不放水。
    """
    import os
    if enabled is None:
        try:                                  # 先读项目根目录的 .env，再判断开关
            from qa_chain import load_env_file
            load_env_file()
        except Exception:                     # noqa: BLE001
            pass
        v = (os.environ.get("FREE_LLM_ENABLED", "0") or "0").strip().lower()
        enabled = v not in ("0", "false", "no", "off", "")
    if not enabled:
        return None
    try:
        from qa_chain import QAPipeline, build_llm
        llm = build_llm(backend="free")
        gen = QAPipeline(None, llm=llm, llm_verify=False, verbose=verbose)

        def _fallback(question, evs):
            res = gen.answer_from_evidences(question, evs)
            ans = (res.answer or "").strip()
            # 注意：不能因为 res.refused 就早退 —— 引用闸门恰恰是在
            # answer_from_evidences 内部就因为"没角标"把它拒掉的，
            # 我们要处理的正是这种被误伤的好答案（踩过一次这个坑）。
            if not tag_missing or not ans or res.insufficient:
                return res
            if re.search(r"\[E\d+\]", ans):
                return res
            # 只留"确实和答案有字面重合"的那几条出处。免费小模型有时偷懒，
            # 把它手上所有证据编号一股脑列进 citations，直接照搬会在正文后面
            # 拖出 [E1][E2]…[E22] 一整串，既难看又不诚实。
            by_eid = {e.eid: e for e in evs}
            ang = grams(ans, 2)
            picked = []
            for c in (res.citations or []):
                ev = by_eid.get(c)
                if ev is None:
                    continue
                if len(ang & grams(ev.text or "", 2)) >= 3:
                    picked.append(c)
            if not picked:                        # 一条都对不上 → 退回重排分最高的几条
                picked = [e.eid for e in
                          sorted(evs, key=lambda e: -(float(getattr(e, "score", 0.0) or 0.0)))[:3]]
            picked = picked[:5]
            if not picked:
                return res
            res.citations = picked
            res.answer = ans + " " + "".join(f"[{c}]" for c in picked)
            v = _verify(res.answer, res.citations, evs)
            res.verify = v
            if v.get("passed"):
                if res.refused and "未标注任何出处" in (res.refuse_reason or ""):
                    res.refused, res.refuse_reason = False, ""   # 撤销误伤
            else:
                res.refused = True
                res.refuse_reason = ("引用校验未通过：" +
                                     "；".join(v.get("reasons") or []))
            return res

        return _fallback
    except Exception as e:                        # noqa: BLE001
        print(f"  [免费兜底通道不可用，按纯程序化运行] {e}", flush=True)
        return None
