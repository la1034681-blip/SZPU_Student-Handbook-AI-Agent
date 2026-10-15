# -*- coding: utf-8 -*-
"""角标重编号 —— 纯离线，不需要模型。

背景（2026-10-15 用户要求）
-------------------------
「E1、E2 必须每个问题都按 E1… 的数字顺序往后排，对应到条款。」

链路的证据编号是按「多路检索的合并顺序」给的 E1..En，模型只挑其中几条引用，
于是页面上会出现 [E2]、[E6]、[E21] 这种跳号 —— 学生看不出引了几条依据、
也看不出先后。

`app.renumber_citations()` 在渲染前按**答复里首次出现的先后**重编：
引到的第一条就是 E1、第二条 E2…… 下面「手册原文」的卡片同步换号并按
E1、E2、E3… 排好，点角标照样跳得到。只动编号，不改一个字。

跑法：`python src/tests/test_renumber.py`
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

import app                                            # noqa: E402

fails: list = []


def check(name: str, got, want) -> None:
    ok = got == want
    print(f"  {'✓' if ok else '✗'} {name}")
    if not ok:
        fails.append(name)
        print(f"      期望 {want!r}\n      实得 {got!r}")


def ev(eid, score=0.9, cite="《某规定》 第X条"):
    return {"eid": eid, "cite": cite, "page": "10", "score": score,
            "file_title": "某规定", "text": f"{eid} 的原文"}


def meta(answer, citations, eids, refused=False):
    return {"answer": answer, "refused": refused, "citations": list(citations),
            "evidences": [ev(e, score=0.9 - i * 0.01) for i, e in enumerate(eids)]}


def strip_cites(s):
    return re.sub(r"\[E\d+\]", "", s or "")


print("① 跳号 → 收紧成 E1、E2（按答复里首次出现的先后）")
m = meta("第一条依据是这样[E2]。第二条依据是那样[E6]。",
         ["E2", "E6"], ["E1", "E2", "E6", "E7"])
out = app.renumber_citations(m)
check("答复里的编号变成 E1/E2", out["answer"],
      "第一条依据是这样[E1]。第二条依据是那样[E2]。")
check("citations 也换成新号", out["citations"], ["E1", "E2"])
check("卡片按 E1、E2 排好（证据顺序同步重排）",
      [e["eid"] for e in out["evidences"]], ["E1", "E2", "E3", "E4"])
check("被引条款的原文跟着走（E2 的内容跟到了新 E1）",
      out["evidences"][0]["text"], "E2 的原文")

print("② 只动编号，一个字不改")
m = meta("旷课10-19学时给予警告处分。迟到早退累计三次按旷课一学时计[E5]。",
         ["E5"], ["E1", "E5"])
out = app.renumber_citations(m)
check("去掉角标后与原答复逐字相同",
      strip_cites(out["answer"]).strip(), strip_cites(m["answer"]).strip())

print("③ 同一编号在答复里出现多次 → 仍是同一个新号")
m = meta("前半句[E3]。后半句又提到[E3]。", ["E3"], ["E3"])
out = app.renumber_citations(m)
check("两处都变成 E1", out["answer"], "前半句[E1]。后半句又提到[E1]。")
check("编号不重复", out["citations"], ["E1"])

print("④ 本来就连续 → 原样返回（不做无谓改动）")
m = meta("先说这个[E1]，再说那个[E2]。", ["E1", "E2"], ["E1", "E2", "E3"])
out = app.renumber_citations(m)
check("答复不变", out["answer"], m["answer"])
check("citations 不变", out["citations"], ["E1", "E2"])

print("⑤ 拒答 → 不动（拒答不展示答复与卡片）")
m = meta("", [], ["E1"], refused=True)
out = app.renumber_citations(m)
check("refused 原样", out["refused"], True)
check("evidences 未被重排", [e["eid"] for e in out["evidences"]], ["E1"])

print("⑥ 答复里出现证据之外的编号 → 不误映射，原样留着")
m = meta("这条是编的[E9]。这条是真的[E2]。", ["E2"], ["E2", "E3"])
out = app.renumber_citations(m)
check("E9 不被改号", "[E9]" in out["answer"], True)
check("E2 → E1", "[E1]" in out["answer"], True)

print("⑦ 没有引用 / 没有证据 → 安全返回")
m = meta("没有角标的答复。", [], ["E1"])
check("无引用时不炸", app.renumber_citations(m)["answer"], "没有角标的答复。")
m2 = meta("有角标但没证据[E1]。", ["E1"], [])
check("无证据时不炸", app.renumber_citations(m2)["answer"], "有角标但没证据[E1]。")

print("⑧ 重编号后，页面要用的「合法角标集合」仍覆盖答复里的每一个编号")
m = meta("甲[E2]。乙[E6]。丙[E21]。", ["E2", "E6", "E21"],
         ["E1", "E2", "E5", "E6", "E21"])
out = app.renumber_citations(m)
valid = {e["eid"] for e in out["evidences"]}
badges = set(re.findall(r"\[(E\d+)\]", out["answer"]))
check("答复里的编号全部有对应证据", sorted(badges - valid), [])
check("编号恰好是 E1、E2、E3",
      sorted(badges, key=lambda s: int(s[1:])), ["E1", "E2", "E3"])
check("citations 与卡片一致", out["citations"], ["E1", "E2", "E3"])

print()
if fails:
    print(f"✗ 有 {len(fails)} 项没通过：{fails}")
    sys.exit(1)
print("✓ 全部通过")
