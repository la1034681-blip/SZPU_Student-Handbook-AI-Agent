# -*- coding: utf-8 -*-
"""阶段 5 · 答案正确率自动评分（Correctness Judge）

为什么单独做这一层
------------------
阶段 4 量的是「引用合不合规」：存在率 / 支持率 / 出处命中率。这三项全过，
只说明「答案的每一句都有出处且出处在本次证据里」，**不说明答得对**。
一篇引用 100% 合规、却答非所问的答复，在阶段 4 是满分。

本模块补上"答得对不对"这一维，用评测集里的标准答案作参照。

两层评分
--------
  一、规则层 · 要点覆盖率（零成本、结果完全确定，可复现）
      评测集每题都有 must_contain（必须出现的关键要点）。统计系统答案覆盖几项。
      局限：纯字面比对 ——「不予补考」也会命中「补考」。
      所以它只做**下限约束**，不下最终结论。

  二、语义层 · LLM 判定（temperature = 0 + JSON 结构化输出）
      把「问题 + 标准答案 + 判定要点 + 待评答案」交给模型，
      判 correct / partial / incorrect，并列出遗漏要点与可疑数字。

  三、数值铁律（在两者之上）
      must_contain 里的数字，若待评答案里找不到对应数值，无论 LLM 怎么判，
      一律降级（correct → partial）。规章问答里答错一个金额/期限，
      比答漏一句话严重得多，不能交给语义判断。

对外入口
--------
    judge(question, gold, system, must_contain, note, llm, qtype) -> dict
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

# 路径自举：src 与 core/retrieval/infra/cli 挂上搜索路径（见 src/_paths.py）
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))   # src/
import _paths  # noqa: E402,F401

sys.stdout.reconfigure(encoding="utf-8")

JUDGE_PROMPT = """你是严格的阅卷员。请对照【标准答案】，评判【待评答案】的正确程度。

判分只看两件事：
  ① 标准答案的核心结论，待评答案覆盖了没有；
  ② 待评答案有没有与标准答案矛盾，或凭空编造。

必须三选一：
- correct：核心结论与关键数据全部正确。
- partial：遗漏了标准答案的重要结论，或表述含混、数值有偏差。
- incorrect：核心结论错误、答非所问，或出现了【证据】和【标准答案】都查不到的具体数字/事实。

判分要点（逐条读完再判）：
1. 只比实质内容。措辞、句式、详略不同都不算错，同义改写算对。
2. **多写不扣分**：待评答案若比标准答案详尽，只要新增内容不与标准答案冲突，
   且能在【证据】里找到依据，就不扣分，也不因此从 correct 降到 partial。
3. **少写才扣分**：把【判定要点】逐条拎出来，在待评答案里找对应内容（同义表达算覆盖）。
   只要有一条要点未被覆盖，就不能判 correct（最多 partial）；漏两条以上判 incorrect。
   输出时必须在 covered / missing 里把要点逐条列全，不得省略。
4. **并行口径**：手册里同一事项常有「本科」与「专科」两套并行规定，标准答案可能只列
   其中一套。待评答案答出另一套（内容属实）、或把两套都答出来，同样判对，不得因此扣分。
5. 数字、期限、比例、金额必须与标准答案一致；不一致至少判 partial，核心数字错判 incorrect。
6. **编造只认「查不到」**：只有出现的具体数字/事实在【证据】和【标准答案】里都找不到，
   才算编造。证据里确有依据的细节不算编造。
7. 若待评答案给出了标准答案未规定、且证据也不支持的结论（例如断言两项奖学金不能同时
   获得），判 incorrect。
8. **判定要点是唯一依据**：【补充说明】只用于理解题干背景与口径差异，不能从它里面
   自行追加要点。凡【判定要点】未列的内容（例如另一套并行口径的细则），未覆盖一律
   不扣分；反之，【判定要点】列出的必须逐条核对。两者若冲突，一律以【判定要点】为准。

只输出 JSON，不要任何解释性文字：
{"verdict":"correct","covered":["要点A"],"missing":["要点B"],
 "bad_numbers":["编造或错误的具体数字"],"reason":"不超过40字的理由"}"""


# ------------------------------------------------------------------ 数字工具
_CN_DIGIT = str.maketrans("０１２３４５６７８９", "0123456789")
_CN_NUM = {"零": 0, "〇": 0, "一": 1, "二": 2, "三": 3, "四": 4,
           "五": 5, "六": 6, "七": 7, "八": 8, "九": 9}
_CN_UNIT = {"十": 10, "百": 100, "千": 1000}
RE_NUM = re.compile(r"\d+(?:\.\d+)?")


def norm_text(s: str) -> str:
    """全角数字 → 半角，去空白，便于字面比对。"""
    return (s or "").translate(_CN_DIGIT).replace(" ", "").replace("\u3000", "")


def int2cn(n: int) -> str:
    """整数 → 中文数字。

    覆盖规章里会出现的量级；万位以上用「万」表示（10000 → 一万、20000 → 二万）。
    仅供字面比对，不追求「一百零八万零三」这种完整读法。
    """
    if n <= 0:
        return str(n)
    d = "零一二三四五六七八九"
    if n < 10:
        return d[n]
    if n < 20:
        return "十" + (d[n % 10] if n % 10 else "")
    if n < 100:
        return d[n // 10] + "十" + (d[n % 10] if n % 10 else "")
    if n < 1000:
        s = d[n // 100] + "百"
        r = n % 100
        if not r:
            return s
        if r < 10:
            return s + "零" + d[r]
        if r < 20:                          # 112 → 一百一十二（不是「一百十二」）
            return s + "一十" + (d[r % 10] if r % 10 else "")
        return s + int2cn(r)
    if n < 10000:
        s = d[n // 1000] + "千"
        r = n % 1000
        if not r:
            return s
        if r < 100:
            return s + "零" + int2cn(r)
        return s + int2cn(r)
    w, r = divmod(n, 10000)
    s = int2cn(w) + "万"
    if not r:
        return s
    return s + ("零" if r < 1000 else "") + int2cn(r)


def number_variants(tok: str):
    """一个数字的各种可能写法：'10000' → {10000, '10000', '1万', '一万'}。"""
    out = {tok}
    if not tok.isdigit():
        return out
    try:
        n = int(tok)
        cn = int2cn(n)
        out.add(cn)
        if "二" in cn:                     # 口语常写作「两万」「两千」
            out.add(cn.replace("二", "两"))
    except Exception:                      # noqa: BLE001
        return out
    if n % 10000 == 0 and n >= 10000:
        out.add(f"{n // 10000}万")
        out.add(f"{int2cn(n // 10000)}万")
    if n % 1000 == 0 and n >= 1000:
        out.add(f"{n // 1000}千")
    return out


def num_in_text(tok: str, text: str) -> bool:
    """待评答案里是否出现了这个数值（容错 §全角、万/千、中文数字）。"""
    t = norm_text(text)
    for v in number_variants(tok):
        if norm_text(v) in t:
            return True
    return False


def numbers_of(items) -> list:
    out = []
    for it in items or []:
        out.extend(RE_NUM.findall(str(it)))
    return out


# ------------------------------------------------------------------ 评分
def coverage_rule(system: str, must_contain) -> dict:
    """规则层：要点覆盖率 + 数字漏配。完全确定性，不依赖任何模型。"""
    must = [str(m) for m in (must_contain or [])]
    t = norm_text(system)
    hits = [m for m in must if norm_text(m) in t]
    misses = [m for m in must if m not in hits]
    nums_missing = [n for n in numbers_of(must) if not num_in_text(n, system)]
    return {
        "coverage": round(len(hits) / len(must), 4) if must else None,
        "hits": hits, "misses": misses, "numbers_missing": nums_missing,
    }


def _extract_json(s: str) -> str:
    s = (s or "").strip()
    s = re.sub(r"^```(?:json)?\s*", "", s)
    s = re.sub(r"\s*```$", "", s)
    i, j = s.find("{"), s.rfind("}")
    if i != -1 and j != -1 and j > i:
        return s[i:j + 1]
    return s


def _render_evidences(evidences, total_limit=12000) -> str:
    """把本次检索到的证据渲染成文本，供阅卷员判断「是不是编造」。

    不传证据时，阅卷员会把「标准答案里没有的数字」一律当成编造 —— 但那些数字
    可能来自手册的其他条款（阶段 5 实测：Q31 的「15 日内作出书面复查结论」
    被判成编造，实际是手册真实规定）。所以证据必须一并给出。
    """
    if not evidences:
        return ""
    if isinstance(evidences, str):
        return evidences[:total_limit]
    out = []
    for e in evidences:
        out.append(e.render() if hasattr(e, "render") else str(e))
    return "\n\n".join(out)[:total_limit]


def judge_llm(question, gold, system, must_contain, note, llm,
              evidences=None) -> dict:
    """语义层：LLM 判 correct / partial / incorrect。"""
    from langchain_core.messages import HumanMessage, SystemMessage

    must = "、".join(str(m) for m in (must_contain or [])) or "（无）"
    ev = _render_evidences(evidences)
    human = (f"【问题】{question}\n\n"
             f"【标准答案】{gold}\n\n"
             f"【判定要点】{must}\n\n"
             f"【补充说明】{note or '（无）'}\n\n"
             f"【证据】（本次检索到的条款原文，用于判断「是否编造」）\n"
             f"{ev or '（未提供）'}\n\n"
             f"【待评答案】{system}")
    try:
        resp = llm.invoke([SystemMessage(content=JUDGE_PROMPT),
                           HumanMessage(content=human)])
        raw = resp.content if hasattr(resp, "content") else str(resp)
        data = json.loads(_extract_json(raw))
    except Exception as e:                                    # noqa: BLE001
        return {"verdict": "unknown", "error": str(e)[:200],
                "covered": [], "missing": [], "bad_numbers": [], "reason": ""}

    v = str(data.get("verdict", "unknown")).lower().strip()
    if v not in ("correct", "partial", "incorrect"):
        v = "unknown"
    return {"verdict": v,
            "covered": [str(x) for x in (data.get("covered") or [])],
            "missing": [str(x) for x in (data.get("missing") or [])],
            "bad_numbers": [str(x) for x in (data.get("bad_numbers") or [])],
            "reason": str(data.get("reason") or "")[:120]}


SCORE_OF = {"correct": 1.0, "partial": 0.5, "incorrect": 0.0, "unknown": None}


def judge(question, gold, system, must_contain=None, note="", llm=None,
          qtype="", evidences=None) -> dict:
    """总入口。returns dict：verdict / score / coverage / ... 全部中间量。

    evidences：本次检索到的证据（list[Evidence] 或已渲染的 str），交给阅卷员
    判断「标准答案里没有的数字」是不是编造。强烈建议传。
    llm=None 时只跑规则层，verdict 由覆盖率粗推（供 dry-run / 省成本场景）。
    """
    system = system or ""
    if not system.strip():
        return {"verdict": "incorrect", "score": 0.0, "coverage": 0.0,
                "hits": [], "misses": [str(m) for m in (must_contain or [])],
                "numbers_missing": numbers_of(must_contain),
                "bad_numbers": [], "reason": "答案为空", "rule_only": llm is None,
                "downgraded": False}

    rule = coverage_rule(system, must_contain)

    if llm is None:
        c = rule["coverage"]
        if c is None:
            v = "unknown"
        elif c >= 0.8:
            v = "correct"
        elif c >= 0.5:
            v = "partial"
        else:
            v = "incorrect"
        sem = {"verdict": v, "covered": rule["hits"], "missing": rule["misses"],
               "bad_numbers": [], "reason": "仅规则层（未提供 LLM）"}
    else:
        sem = judge_llm(question, gold, system, must_contain, note, llm,
                        evidences=evidences)

    verdict = sem["verdict"]
    downgraded = False
    rule_note = ""
    # --- 数值铁律：数字漏配 ⇒ 至少降到 partial ---
    if rule["numbers_missing"] and verdict == "correct":
        verdict = "partial"
        downgraded = True
        rule_note = f"漏配数字 {rule['numbers_missing']}"
    # --- 要点兜底：判定要点字面命中率低于 0.8 ⇒ 阅卷员很可能漏读了要点，不允许判 correct ---
    #   阈值取 0.8 而非 1.0，是为了容忍同义改写造成的字面假阴性（例如「取消成绩」写成「成绩无效」）。
    #   实测：Q36 只命中 4/6 项（漏「每学年办理一次」「第一学年末」），旧口径被判 correct，
    #   属于放水；加这条兜底后降为 partial。
    cov = rule["coverage"]
    if verdict == "correct" and cov is not None and cov < 0.8:
        verdict = "partial"
        downgraded = True
        rule_note = f"判定要点字面仅命中 {cov:.0%}，存在遗漏"

    score = SCORE_OF.get(verdict)
    return {
        "verdict": verdict,
        "score": score,
        "coverage": rule["coverage"],
        "hits": rule["hits"],
        "misses": rule["misses"],
        "numbers_missing": rule["numbers_missing"],
        "bad_numbers": sem.get("bad_numbers", []),
        "missing_points": sem.get("missing", []),
        "reason": (sem.get("reason", "") + (f"；{rule_note}" if rule_note else ""))[:160],
        "rule_only": llm is None,
        "downgraded": downgraded,
    }


# ------------------------------------------------------------------ 自测
if __name__ == "__main__":
    from qa_chain import build_llm

    q = "必修课考试不及格，能补考吗？补考成绩怎么算？"
    gold = ("必修课程、限选课程考核不及格者，可在下学期开学参加一次补考。"
            "补考卷面成绩连同修课时的平时成绩一起进行综合评定，考核合格即获得"
            "该门课程的学分，课程成绩标注“补考”后记入成绩单。")
    must = ["一次补考", "平时成绩", "综合评定", "记入成绩单"]
    cases = [
        ("完全正确", gold),
        ("漏了成绩单", "考核不及格可以参加一次补考，补考成绩由卷面成绩和平时成绩综合评定。"),
        ("数字错", "考核不及格可以参加三次补考，成绩记入成绩单。"),
    ]
    try:
        llm = build_llm()
    except Exception as e:                                    # noqa: BLE001
        print("无可用 LLM，仅演示规则层：", e)
        llm = None
    for name, sysans in cases:
        r = judge(q, gold, sysans, must, llm=llm)
        print(f"\n[{name}] {r['verdict']}  score={r['score']}  "
              f"coverage={r['coverage']}  漏配数字={r['numbers_missing']}")
        print("   理由：", r["reason"])
