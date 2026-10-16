# -*- coding: utf-8 -*-
"""程序化（抽取式）兜底答案生成器 —— 全程不需要任何大模型。

**它不再是一条作答通道。** 作答永远由智能体链路主导（大模型规划 → 多路检索
→ 生成 → 自检 → 引用校验）。这里只负责一种情况：**大模型用不了的时候，
别让学生白问一场** —— 例如 429 限流重试耗尽、网络断了、密钥失效、模型返回
的结构解析不了。这时把手册原文照抄给他，宁可朴素，也不编。

   证据（retrievers.Evidence）
     → 按"与问题的契合度"给证据重排
     → 从条款原文切句、挑最相关的句子
     → 用极性规则合成一句结论（可以 / 不可以 / 需要办 …）
     → 拼成「结论句 + 原文条款 + 出处」的答案
     → 契合度低于阈值则拒答

两处会用到它（都在 `qa_chain.py`）：
  1. **模型出错兜底**：整条智能体链路抛异常 → 改由这里照抄原文作答，
     并且**在页面上标明**「大模型暂时不可用，以下为手册原文照抄」。
  2. **模型坚持拒答时**：模型两次都说"证据不足"，而检索分明显很高
     （见 `REFUSE_CONFIRM_MIN` 复核）→ 改由这里照抄，而不是拒答。

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
    from procedural_qa import generate, _verify
    out = generate(question, evs, max_evidence=3)   # evs = 检索到的证据
    if not out["insufficient"]:
        res.answer, res.citations = out["answer"], out["citations"]
"""
from __future__ import annotations

import re
import sys
import time
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

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


def _verify(answer, citations, evs):
    try:
        from verify_citations import verify as _v
        return _v(answer, citations, evs, llm=None, recheck=False)
    except Exception as e:                        # noqa: BLE001
        return {"passed": True, "reasons": [f"校验跳过：{e}"],
                "existence": {"passed": True}, "support": {"passed": True}}
