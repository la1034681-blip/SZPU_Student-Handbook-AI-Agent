# -*- coding: utf-8 -*-
"""阶段 4 · 引用校验与拒答（本项目核心自研件之二）

LLM 会犯错，尤其是引用。生产环境里"答案看着对、出处是编的"最危险，
所以生成之后必须过两道闸：

  一级 · 存在性 —— 纯规则，不发 LLM
      答案里出现的每个 [Ex] 标记，是否都在本次实际检索到的证据里？
      LLM 常见错误：引用一个根本不存在的编号，或干脆不标引用。
      这一级不通过 = 伪造引用，直接拒答，没有商量。

  二级 · 支持性 —— 需要语义判断
      把答案拆成「断言」，逐条配对它所引的证据，问：这条证据真的支持这句话吗？
      判 support / partial / no。全部判 no = 证据不足，拒答。

拒答触发（任一命中）
  · 检索无任何证据
  · 一级校验失败（伪造引用 / 关键断言没有引用）
  · 二级校验中，支撑答案的证据全部不成立
  · LLM 自述 insufficient

对外入口：
    verify(answer_text, citations, evidences, llm=None) -> dict
"""
from __future__ import annotations

import json
import re
import sys
from dataclasses import dataclass, field

sys.stdout.reconfigure(encoding="utf-8")

RE_CITE = re.compile(r"\[E(\d+)\]")
RE_CITE_GROUP = re.compile(r"(?:\[E\d+\])+")   # 连续引用标记，如 [E1][E3]
RE_SENT_SPLIT = re.compile(r"(?<=[。；！？\n])")


# ------------------------------------------------------------------ 断言拆解
@dataclass
class Claim:
    idx: int
    text: str                 # 去掉引用标记后的纯断言文本
    eids: list = field(default_factory=list)   # 该断言引用的证据编号


def parse_citations(text: str) -> list:
    """抓出文本里所有 [Ex] 引用编号（去重、保序）。"""
    seen, out = set(), []
    for m in RE_CITE.finditer(text or ""):
        e = f"E{int(m.group(1))}"
        if e not in seen:
            seen.add(e)
            out.append(e)
    return out


def split_claims(answer: str) -> list:
    """把答案切成「断言 + 其引用」单元。

    切法是 **以引用标记组为分隔**：上一处引用之后、到这一处引用（含）之间的
    全部文字，连同这组引用标记，构成一条断言。

    为什么不用句号切：模型最自然的写法是「一段讲一个结论，段尾标引用」。
    按句号切会把段落里除末句之外的全部内容判成「没有出处」，纯属误报
    （这一版修的就是这个问题）。
    """
    claims = []
    last = 0
    for m in RE_CITE_GROUP.finditer(answer or ""):
        seg = answer[last:m.end()]
        last = m.end()
        eids = []
        for mm in RE_CITE.finditer(seg):
            e = f"E{int(mm.group(1))}"
            if e not in eids:
                eids.append(e)
        clean = RE_CITE.sub("", seg).strip(" 　\n")
        if clean and eids:
            claims.append(Claim(idx=len(claims) + 1, text=clean[:400], eids=eids))
    return claims


# ------------------------------------------------------------------ 一级：存在性
def check_existence(answer: str, citations, known_eids) -> dict:
    """纯规则校验。known_eids 是本次检索真实给出的证据编号集合。"""
    known = set(known_eids)
    marked = parse_citations(answer)                 # 答案正文里标了的
    declared = [str(c).upper() for c in (citations or [])]  # 结构化字段里申报的

    fake_marked = [e for e in marked if e not in known]
    fake_declared = [e for e in declared if e not in known]

    # 整段没有任何引用 → 该段无法溯源（按段统计，不按句，避免误报）
    uncited = []
    for para in re.split(r"\n+", answer or ""):
        clean = RE_CITE.sub("", para).strip(" 　")
        if len(clean) >= 20 and not RE_CITE.search(para):
            uncited.append(clean[:40])

    # 有实质内容却一处引用都没标 → 同样不可放行（否则"全篇不标"反而能蒙混过关）
    body = RE_CITE.sub("", answer or "").strip()
    no_citation = (not marked) and len(body) >= 20

    ok = (not fake_marked) and (not fake_declared) and (not no_citation)
    return {
        "passed": ok,
        "marked": marked,
        "declared": declared,
        "unknown_marked": fake_marked,      # 正文引用了不存在的编号
        "unknown_declared": fake_declared,  # 申报了不存在的编号
        "no_citation": no_citation,         # 通篇没有标注任何出处
        "uncited_paragraphs": uncited,      # 有个别段落没标出处（提示用，不单独拒答）
        "marked_not_declared": [e for e in marked if e not in declared],
    }


# ------------------------------------------------------------------ 二级：支持性
# 「元陈述」：模型在正文里声明「这段出自哪份文件 / 属于哪套口径」。
# 提示词已明令禁止，但模型偶尔仍会写；这类句子证据原文里当然找不到，
# 若让它参与支持性校验，会把整段正确回答否决掉（Q26 实测）。
# 这里用规则兜底：识别出来即豁免，不计入否决。
META_PATTERNS = [
    r"出自《", r"见[于]?第[一二三四五六七八九十百零\d]+条", r"来自《", r"依据《",
    r"(本条|该条|上述规定|上述内容|以上规定).{0,10}(出自|见于|载于|来自)",
    r"(上述|以上).{0,12}(本科|专科).{0,8}(口径|规定|标准|要求)",
]
RE_META = re.compile("|".join(META_PATTERNS))


def is_ignorable_claim(text: str) -> bool:
    """可忽略的断言：元陈述，或过短且不含数字的连接句。

    含数字的断言永远不可忽略 —— 数字必须被证据支持。
    """
    t = (text or "").strip()
    if not t:
        return True
    if re.search(r"\d", t):
        return False
    if RE_META.search(t):
        return True
    return len(t) < 10


VERIFY_PROMPT = """你是严格的引用校验员。下面给你若干条「断言」和它们各自引用的「证据原文」。
逐条判断：证据原文是否真的支持该断言？

判据（三选一，不许含糊）：
- support：证据直接、明确地支持该断言（含合理的同义改写、由原文可直接推出的结论）
- partial：证据只支持断言的一部分，或必须额外假设才能推出来
- no：证据与断言无关，或证据与断言相互矛盾

特别注意：
- 断言若把「本科」的规定当成全校通用，而证据只覆盖专科，判 partial。
- 断言若出现证据里没有的数字、期限、比例，判 no。
- 证据块开头的「出自 xxx」是它的真实出处，若断言只是在说明出处，
  能对上就判 support。

只输出 JSON，不要任何解释性文字：
{"verdicts":[{"id":1,"verdict":"support","reason":"不超过20字的理由"}]}"""


def _build_verify_body(claims, evmap) -> str:
    """拼校验提示词正文。证据块带上完整出处（cite），
    否则断言的「出处说明」类句子会被判成无依据。"""
    lines = []
    for c in claims:
        parts = []
        for e in c.eids:
            if e in evmap:
                cite = getattr(evmap[e], "cite", "") or ""
                parts.append(f"  {e}（出处：{cite}）：{evmap[e].text}")
            else:
                parts.append(f"  {e}：（不存在）")
        lines.append(f"断言 {c.idx}：{c.text}\n引用的证据：\n" + "\n".join(parts) + "\n")
    return "\n".join(lines)


def _ask_verdicts(claims, evmap, llm):
    """调一次 LLM，返回 ({claim_idx: {"verdict","reason"}}, error)。"""
    from langchain_core.messages import HumanMessage, SystemMessage

    body = _build_verify_body(claims, evmap)
    try:
        resp = llm.invoke([SystemMessage(content=VERIFY_PROMPT),
                           HumanMessage(content=body)])
        raw = resp.content if hasattr(resp, "content") else str(resp)
        data = json.loads(_extract_json(raw))
        verdicts = data.get("verdicts", [])
    except Exception as e:                                    # noqa: BLE001
        return None, str(e)[:200]

    out = {}
    for v in verdicts:
        if isinstance(v, dict):
            out[int(v.get("id", 0))] = {
                "verdict": str(v.get("verdict", "unknown")).lower(),
                "reason": str(v.get("reason", ""))[:60]}
    return out, None


_RANK = {"support": 2, "partial": 1, "no": 0, "unknown": 0}

# 二级校验放行阈值：允许最多 N 条「实质断言」被判 no 仍算通过。
# 背景（阶段6 定稿）：原条件是"零条 hard_no"，即一条无依据就整题拒答。
# 跨章题答案长（常 10+ 条断言），LLM 判 no 本身有抖动，单条误判就会
# 误杀整题（实测 N10/N15 检索完全正确却被拒）。放宽到 ≤1 条：既消掉
# 单条噪声，又保留"两条以上无依据即拒"的防编造能力。
# 改回严格档只需把常量设为 0。
SUPPORT_MAX_HARD_NO = 1


def check_support(claims, evmap, llm, recheck=True) -> dict:
    """二级校验。

    阶段 5 新增两处**降低波动**的设计：

      1. 断言级复核（recheck）：首轮没判 support 的断言，单独再判一次；
         两次中只要有一次 support 即取 support，两次都不行才认。只复核少数
         断言，成本增量很小，但能消掉"同一输入两次判定不同"的单次抖动。
      2. 元陈述豁免：判 no 的断言若是元陈述 / 短连接句，不计入否决
         （区分 hard_no 与 ignored_no）。

    通过条件（阶段6 定稿放宽）：
      ① 至少一条断言被证据明确支持；② 实质断言中判 no 的不超过
         SUPPORT_MAX_HARD_NO 条（默认 1）。原先是"零条"，跨章长答案
         易被单条抖动误杀，故放宽 1 条，同时保留至少一条"无依据"即拒
         的防编造能力。
    """
    if not claims:
        return {"passed": True, "skipped": True, "reason": "无可校验断言",
                "verdicts": []}
    if llm is None:
        return {"passed": True, "skipped": True, "reason": "未提供 LLM，跳过二级校验",
                "verdicts": []}

    first, err = _ask_verdicts(claims, evmap, llm)
    if first is None:
        # 校验器自身出错时不能"默默放行" —— 标记未通过，交由上层拒答
        return {"passed": False, "skipped": False, "error": err, "verdicts": []}

    # ---- 断言级复核：只对首轮非 support 的断言复判 ----
    second, n_recheck = {}, 0
    if recheck:
        pending = [c for c in claims
                   if first.get(c.idx, {}).get("verdict") != "support"]
        if pending:
            second, _ = _ask_verdicts(pending, evmap, llm)
            second = second or {}
            n_recheck = len(second)

    detail = []
    for c in claims:
        v1 = first.get(c.idx, {"verdict": "unknown", "reason": ""})
        v2 = second.get(c.idx)
        cand = [v1.get("verdict", "unknown")] + ([v2["verdict"]] if v2 else [])
        final = max(cand, key=lambda x: _RANK.get(x, 0))
        detail.append({
            "id": c.idx, "claim": c.text[:60], "eids": c.eids,
            "verdict": final,
            "first": v1.get("verdict"),
            "recheck": (v2 or {}).get("verdict"),
            "flipped": bool(v2 and v2["verdict"] != v1.get("verdict")),
            "ignorable": is_ignorable_claim(c.text),
            "reason": (v2 or v1).get("reason", ""),
        })

    supported = [d for d in detail if d["verdict"] == "support"]
    partial = [d for d in detail if d["verdict"] == "partial"]
    nos = [d for d in detail if d["verdict"] == "no"]
    hard_no = [d for d in nos if not d["ignorable"]]
    ignored_no = [d for d in nos if d["ignorable"]]

    passed = (len(supported) >= 1) and (len(hard_no) <= SUPPORT_MAX_HARD_NO)
    return {"passed": passed, "skipped": False,
            "n_support": len(supported), "n_partial": len(partial),
            "n_no": len(nos), "n_hard_no": len(hard_no),
            "n_ignored_no": len(ignored_no),
            "n_recheck": n_recheck,
            "n_flipped": sum(1 for d in detail if d["flipped"]),
            "verdicts": detail}


def _extract_json(s: str) -> str:
    """从可能带 ```json 围栏或前后废话的回复里抠出 JSON 主体。"""
    s = (s or "").strip()
    s = re.sub(r"^```(?:json)?\s*", "", s)
    s = re.sub(r"\s*```$", "", s)
    i, j = s.find("{"), s.rfind("}")
    if i != -1 and j != -1 and j > i:
        return s[i:j + 1]
    return s


# ------------------------------------------------------------------ 汇总
def verify(answer: str, citations, evidences, llm=None, recheck=True) -> dict:
    """两级校验总入口。evidences: list[Evidence]。

    recheck=False 可关闭断言级复核（消融实验用，对照波动率）。
    """
    evmap = {e.eid: e for e in evidences}
    ex = check_existence(answer, citations, evmap.keys())
    claims = split_claims(answer)
    sup = check_support(claims, evmap, llm, recheck=recheck)

    passed = ex["passed"] and sup["passed"]
    reasons = []
    if not ex["passed"]:
        if ex["unknown_marked"]:
            reasons.append(f"正文引用了不存在的证据编号 {ex['unknown_marked']}")
        if ex["unknown_declared"]:
            reasons.append(f"申报了不存在的证据编号 {ex['unknown_declared']}")
        if ex.get("no_citation"):
            reasons.append("答案通篇未标注任何出处")
    if not sup["passed"]:
        if sup.get("error"):
            reasons.append(f"支持性校验失败：{sup['error']}")
        elif sup.get("n_hard_no"):
            reasons.append(f"有 {sup['n_hard_no']} 条实质断言不被所引证据支持")
        else:
            reasons.append("没有一条断言被证据明确支持（证据只是擦边）")

    return {
        "passed": passed,
        "existence": ex,
        "support": sup,
        "n_claims": len(claims),
        "n_citations": len(evmap),
        "reasons": reasons,
    }


# ------------------------------------------------------------------ 自测
if __name__ == "__main__":
    from retrievers import HandbookRetriever
    r = HandbookRetriever("data", use_rerank=False, verbose=False)
    evs = r.search("考核不及格可以补考吗", top_k=4)
    print("证据：", [(e.eid, e.clause_id) for e in evs])
    good = "考核不及格者可以参加一次补考[E1]。实践类课程需申请重修[E2]。"
    bad = "考核不及格者可以参加三次补考[E9]。"
    for name, txt in [("正常答案", good), ("伪造引用", bad)]:
        res = verify(txt, parse_citations(txt), evs, llm=None)
        print(f"\n[{name}] passed={res['passed']}")
        print("  一级：", res["existence"]["passed"],
              "| 未知编号", res["existence"]["unknown_marked"])
        print("  原因：", res["reasons"])
