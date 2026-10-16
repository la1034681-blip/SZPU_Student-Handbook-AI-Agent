# -*- coding: utf-8 -*-
"""离线测试：口语词换手册词（2026-10-16 新增）。

起因
----
学生问「体测成绩对于奖学金是否是硬性要求」被拒答。查下来是**词对不上**：
手册里根本没有「体测」二字（实测 0 次），手册写的是「体质测试」。
BM25 按字面匹配，一个字对不上就是零命中 → 最高相关分只有 0.05 →
最后的引用校验判「证据只是擦边」→ 整题拒答。

守住三件事
----------
① **命中要准**：体测/挂科/翘课 这些口语词要能认出来，并给出手册写法
② **不许误伤**：口语词撞进正确写法时要放过
   （「拿奖学金」不能被当成「拿奖」、「专升本」不能被当成「升本」）
③ **照手册说的不折腾**：本来就问「旷课」「奖学金」时，一条多余的路都不加

外加一条端到端：真跑一遍 `AgentQAPipeline._ask`（用假检索器，不调模型，
不联网），确认①多走了几路、②plan 里留了痕。

全离线，秒级。跑法：python src/tests/test_term_alias.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / "src"))
import _paths  # noqa: E402 把 core/retrieval/infra/cli 也挂上搜索路径

import term_alias as ta                        # noqa: E402
from retrievers import Evidence                # noqa: E402

_fail = 0


def ck(name: str, cond: bool, extra: str = ""):
    global _fail
    if not cond:
        _fail += 1
    print(f"  {'✓' if cond else '✗'} {name}"
          + (f"　—— {extra}" if extra and not cond else ""), flush=True)


# ------------------------------------------------------------------ ① 命中
print("① 口语词要认得出来")
ck("体测 → 体质测试", dict(ta.hits("体测成绩对奖学金有影响吗")).get("体测") == ("体质测试",))
ck("挂科 → 不及格/重修", dict(ta.hits("我这学期挂科了")).get("挂科") == ("不及格", "重修"))
ck("翘课 → 旷课", dict(ta.hits("翘课多少节会被处分")).get("翘课") == ("旷课",))
ck("打小抄 → 考试作弊", dict(ta.hits("打小抄会怎么样")).get("打小抄") == ("考试作弊",))
ck("蹲级 → 留级/降级", dict(ta.hits("蹲级了怎么办")).get("蹲级") == ("留级", "降级"))
ck("打工 → 勤工助学", dict(ta.hits("想打工可以吗")).get("打工") == ("勤工助学",))
ck("一句里两个口语词都认出来",
   len(ta.hits("翘课挂科了咋办")) == 2, str(ta.hits("翘课挂科了咋办")))

# ------------------------------------------------------------------ ② 不误伤
print("\n② 不许把好端端的词改坏（这是写表时当场踩到的坑）")
ck("「拿奖学金」不算命中（表里已删「拿奖」）",
   ta.hits("我这学期挂科了还能拿奖学金吗") == [("挂科", ("不及格", "重修"))],
   str(ta.hits("我这学期挂科了还能拿奖学金吗")))
ck("「专升本」不算命中（表里已删「升本」）",
   ta.hits("想专升本怎么办") == [], str(ta.hits("想专升本怎么办")))
ck("「国家励志奖学金」不算命中",
   ta.hits("国家励志奖学金条件") == [])
ck("替换后「奖学金」三个字完好",
   "奖学金" in ta.variants("挂科了还能拿奖学金吗")[-1],
   str(ta.variants("挂科了还能拿奖学金吗")))
ck("替换后不出现「获奖学金」这种怪词",
   all("获奖学金" not in v for v in ta.variants("挂科了还能拿奖学金吗")))

# ------------------------------------------------------------------ ③ 照手册说的不折腾
print("\n③ 本来就用的手册词 → 一条多余检索都不加（零开销）")
for q in ("旷课多少学时会被处分", "奖学金一年多少钱", "补考什么时候考"):
    ck(f"「{q}」不产生扩展", ta.variants(q) == [], str(ta.variants(q)))
    ck(f"「{q}」不给换词提示", ta.hint_for(q) == "")

# ------------------------------------------------------------------ variants
print("\n④ 换词问句的三种写法")
vs = ta.variants("体测成绩对于奖学金是否是硬性要求")
ck("给了 3 路", len(vs) == 3, str(vs))
ck("① 追加：原句 + 手册词", vs[0] == "体测成绩对于奖学金是否是硬性要求 体质测试")
ck("② 前缀：手册词 + 原句", vs[1].startswith("体质测试 "))
ck("③ 替换：口语词就地换掉", vs[2] == "体质测试成绩对于奖学金是否是硬性要求")
ck("三路互不相同", len(set(vs)) == 3)
ck("不跟原句重复", all(v != "体测成绩对于奖学金是否是硬性要求" for v in vs))

# ------------------------------------------------------------------ hint
print("\n⑤ 查不到时的换说法提示")
h = ta.hint_for("体测成绩对于奖学金是否是硬性要求")
ck("提示里点出手册写法", "体质测试" in h, h)
ck("提示里带上学生的说法", "体测" in h)
ck("没有口语词 → 不给提示", ta.hint_for("学校有几个食堂") == "")

# ------------------------------------------------------------------ ⑥ 端到端
print("\n⑥ 端到端：链路真的多走几路，并留痕")


class FakeRetriever:
    """只记下被问了什么，返回固定证据 —— 不加载模型、不发请求。"""

    def __init__(self):
        self.queries = []

    def search(self, q, top_k=8):
        self.queries.append(q)
        # 造一条高分的假证据，免得被拒答前置拦掉
        return [Evidence(eid="E1", clause_id="F20-A009", chunk_id="F20-A009",
                         cite="《深圳职业技术大学学生综合素质评价办法》第九条",
                         file_id="F20", file_title="学生综合素质评价办法",
                         article_label="第九条", page="129",
                         text="（假证据，测试用）", score=0.9)]


from agent_qa import AgentQAPipeline              # noqa: E402

fr = FakeRetriever()
pipe = AgentQAPipeline(fr, llm=None, verbose=False)
pipe.enable_planner = False                       # 不调模型，standalone 就是原句
pipe.enable_reflect = False
res = pipe.ask("体测成绩对于奖学金是否是硬性要求")

alias_q = (res.plan or {}).get("alias_queries") or []
ck("链路读了 4 路（原句 + 3 路换词）", len(fr.queries) == 4, str(fr.queries))
ck("plan 里记了 alias_queries", len(alias_q) == 3, str(alias_q))
ck("换词那几路确实带上了手册词",
   all("体质测试" in q for q in alias_q), str(alias_q))
ck("原句仍然排在第 1 路（顺序不能乱）",
   fr.queries[0] == "体测成绩对于奖学金是否是硬性要求", fr.queries[0])

fr2 = FakeRetriever()
pipe2 = AgentQAPipeline(fr2, llm=None, verbose=False)
pipe2.enable_planner = False
pipe2.enable_reflect = False
res2 = pipe2.ask("旷课多少学时会被处分")
ck("照手册说的只走 1 路（不多花时间）", len(fr2.queries) == 1, str(fr2.queries))
ck("没命中就不记 alias_queries", not (res2.plan or {}).get("alias_queries"))

print("\n" + ("全部通过 ✓" if _fail == 0 else f"有 {_fail} 项失败 ✗"))
sys.exit(1 if _fail else 0)
