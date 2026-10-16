# -*- coding: utf-8 -*-
"""校验器的 JSON 解析与「校验器自身故障」处理 —— 纯离线，不需要模型。

背景（2026-10-15 实测踩到）：
  问「从补考到重修会经历哪些环节？」，检索完全正确，却被整题拒答，
  原因是校验器的 JSON 解析报 `Extra data`。
  两个独立缺陷叠在一起造成这次误杀：

    ① `_extract_json` 用「第一个 { 到最后一个 }」截取。模型只要把对象写了
       两遍、或在对象后面补一句带花括号的解释，截出来就是 `{...}{...}`
       这种非法串。
    ② 更严重的是**解析失败被当成「证据不足」**。`_ask_verdicts` 返回
       `(None, err)` 时，`check_support` 判 `passed: False`，上层据此拒答 ——
       等于「尺子坏了」被当成「学生作弊」。注释里写的「不能默默放行」
       是对的（不能当成通过就悄悄放过），但正确做法是**降级放行并留痕**，
       因为一级校验（引用存在性、纯规则）已经过了，那才是防伪造的主闸。

跑法：`python src/tests/test_verify_json.py`
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / "src"))
import _paths  # noqa: E402 把 core/retrieval/infra/cli 也挂上搜索路径

from verify_citations import _extract_json           # noqa: E402
from verify_citations import check_support, split_claims, parse_citations  # noqa: E402
from retrievers import Evidence                     # noqa: E402

fails: list = []


def check(name: str, got, want) -> None:
    ok = got == want
    print(f"  {'✓' if ok else '✗'} {name}")
    if not ok:
        print(f"      期望 {want!r}，实得 {got!r}")
        fails.append(name)


def try_parse(s: str) -> str:
    """把 _extract_json 的输出真的喂给 json.loads，返回状态标签。"""
    body = _extract_json(s)
    if "{" not in s:
        return "无JSON(预期)"
    try:
        obj = json.loads(body)
    except Exception as e:                                # noqa: BLE001
        return f"解析失败:{type(e).__name__}"
    return "OK" if isinstance(obj, dict) and "verdicts" in obj else "缺verdicts"


# ============================================================ ① 解析健壮性
print("① _extract_json：模型各种不规矩输出都要能抠出对象", flush=True)

check("正常裸 JSON",
      try_parse('{"verdicts":[{"id":1,"verdict":"support"}]}'), "OK")

check("带 ```json 围栏",
      try_parse('```json\n{"verdicts":[{"id":1,"verdict":"no"}]}\n```'), "OK")

check("对象写两遍（本次真凶）",
      try_parse('{"verdicts":[{"id":1,"verdict":"support"}]}\n'
                '{"verdicts":[{"id":2}]}'), "OK")

check("对象后跟解释且解释里带花括号",
      try_parse('{"verdicts":[{"id":1,"verdict":"partial"}]}\n说明：补充 {备注}'), "OK")

check("前面有废话",
      try_parse('好的，我的判断如下：\n{"verdicts":[{"id":1,"verdict":"support"}]}'), "OK")

check("嵌套对象与数组",
      try_parse('{"verdicts":[{"id":1,"verdict":"support"}],"meta":{"a":[1,2]}}'), "OK")

check("完全没有 JSON（不该崩）",
      try_parse("我认为证据不足。"), "无JSON(预期)")

check("空串", try_parse(""), "无JSON(预期)")

# 只吃第一个对象时，取到的是**第一个**（与模型首个判断一致，不是后面的）
first = json.loads(_extract_json('{"verdicts":[{"id":1,"verdict":"support"}]}\n'
                                 '{"verdicts":[{"id":1,"verdict":"no"}]}'))
check("取第一个对象而非最后一个",
      first["verdicts"][0]["verdict"], "support")


# ====================================== ② 校验器故障不得等同于「证据不足」
class _BrokenLLM:
    """模拟校验器自己坏掉：invoke 直接抛异常（限流/网络/解析都归这条）。"""

    def __init__(self, bad_json: bool = False):
        self.bad_json = bad_json
        self.calls = 0

    def invoke(self, msgs):
        self.calls += 1
        text = ("{这不是合法 JSON" if self.bad_json else "网络异常")
        return type("R", (), {"content": text})()


# 用真实的 Evidence 类，不用 dict —— evmap 的值是**带属性**的对象
# （`_build_verify_body` 里写的是 `evmap[e].text`，属性访问）。
evmap = {"E1": Evidence(
    eid="E1", clause_id="c1", chunk_id="k1",
    cite="《某某规定》 第二十一条", file_id="f1", file_title="某某规定",
    article_label="第二十一条", page="36",
    text="学生首次修读的课程考核不及格者，可在下学期开学初参加一次补考。",
    score=0.98)}
claims = split_claims("补考安排在下学期开学初。[E1]")

print("\n② 校验器自身故障：必须降级放行 + 留痕，不能当成「证据不足」", flush=True)

r = check_support(claims, evmap, _BrokenLLM(), recheck=False)
check("invoke 抛异常 → 不再直接判拒答", r["passed"], True)
check("标记为 degraded（页面/日志能看出这次没真校验）", r.get("degraded"), True)
check("留痕 error 字段非空", bool(r.get("error")), True)
check("verdicts 留空，不伪造判定", r["verdicts"], [])

r2 = check_support(claims, evmap, _BrokenLLM(bad_json=True), recheck=False)
check("返回非法 JSON → 同样降级放行", r2["passed"], True)
check("同样标记 degraded", r2.get("degraded"), True)

# 关键对照：模型**明确判 no** 时，防编造的主闸必须还在
class _NoLLM:
    def invoke(self, msgs):
        return type("R", (), {"content":
                              '{"verdicts":[{"id":1,"verdict":"no","reason":"证据里没有"}]}'})()


r3 = check_support(claims, evmap, _NoLLM(), recheck=False)
check("真判 no 仍会拦截（防编造主闸未被削弱）", r3["passed"], False)
check("真判 no 不标 degraded", bool(r3.get("degraded")), False)


# ==================================================== ③ 断言拆解没被带坏
print("\n③ 断言拆解（顺手回归，确认解析修复没牵连它）", flush=True)
a = split_claims("结论一句。[E1]\n- 要点一。[E1][E3]\n- 要点二。[E2]")
check("拆出 3 条断言", len(a), 3)
check("角标去重保序", parse_citations("见 [E2] 与 [E1] 还有 [E2]"), ["E2", "E1"])


# ============================================================ 汇总
print("\n" + "=" * 58)
if fails:
    print(f"✗ 失败 {len(fails)} 项：")
    for f in fails:
        print("   -", f)
    raise SystemExit(1)
print("✓ 全部通过")
