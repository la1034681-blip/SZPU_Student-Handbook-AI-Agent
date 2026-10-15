# -*- coding: utf-8 -*-
"""离线单测：会话记忆模块 src/memory.py（2026-10-15 新增多轮记忆）。

测什么
------
① 会话号          随机、够长、不重复
② 写入与读取      空消息不进；「上一轮那句」取法正确（会跳过本次自己）
③ 历史块          空记忆返回空串；拼装顺序对；超预算从**最旧**开始丢，最新一定在
④ 超预算判定      小记忆不触发、大记忆触发
⑤ 摘要            够短不摘要；模型不可用 / 报错 / 返回空 → **一条都不删**（不漏信息）
⑥ 序列化          存取往返一致；形状不对就返回 None（调用方当没有记忆）
⑦ 云库返回兼容    不同网关把 count 包成不同形状，都要认得出
⑧ 云库关闭时的容错 四个云库函数全部安静失败、不抛异常（不能因为记忆坏了耽误提问）

为什么第 ⑤ 条盯得这么紧：摘要是**有损**的。模型一旦抽风，宁可不压缩、多占点
上下文预算，也绝不能把学生刚说过的话悄悄弄丢 —— 那属于「静默降级」。

跑法：python src/tests/test_memory.py
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

# 必须在导入 memory 之前：这一条就能把整个云库出口关掉（CloudStore 读它），
# 于是本脚本不碰一次网络，跑得飞快。
os.environ["CLOUD_LOG_ENABLED"] = "0"

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / "src"))

from memory import (MAX_HISTORY_CHARS, MAX_TURNS_STORED,  # noqa: E402
                    MSG_TRUNC, SUMMARY_CHARS, SessionMemory, _as_count,
                    clear_memory, load_memory, new_mem_id, save_memory,
                    summarize_older, sweep_memory)

_fail = 0


def ck(name: str, cond: bool, extra: str = ""):
    global _fail
    mark = "✓" if cond else "✗"
    if not cond:
        _fail += 1
    print(f"  {mark} {name}" + (f"  —— {extra}" if extra and not cond else ""))


class FakeLLM:
    """假的模型：只回一句固定话。用来隔离「摘要」这一步的逻辑。"""

    def __init__(self, text="（摘要）学生问过 A，答过 B。", boom=False):
        self.text, self.boom = text, boom
        self.calls = 0

    def invoke(self, messages):
        self.calls += 1
        if self.boom:
            raise RuntimeError("模型炸了")
        return type("R", (), {"content": self.text})()


def mem_with(n_pairs: int, text="内容") -> SessionMemory:
    m = SessionMemory(mem_id=new_mem_id(), client_id="t")
    for i in range(n_pairs):
        m.add_user(f"{text}问{i}")
        m.add_assistant(f"{text}答{i}")
    return m


def main() -> int:
    print("① 会话号")
    a, b = new_mem_id(), new_mem_id()
    ck("32 位、全是十六进制", len(a) == 32 and all(c in "0123456789abcdef" for c in a), a)
    ck("两次不同", a != b)

    print("\n② 写入与读取")
    m = SessionMemory(mem_id=new_mem_id())
    m.add_user("   ")          # 空白：不该进
    m.add("user", "")
    ck("空白消息不进记忆", m.turns == [])
    m.add_user("旷课多少学时会被处分？")
    m.add_assistant("10-19 学时警告。")
    m.add_user("那迟到早退算不算旷课？")
    ck("记了 3 条", len(m.turns) == 3)
    ck("上一轮那句 = 旷课那题",
       m.previous_user_question("那迟到早退算不算旷课？") == "旷课多少学时会被处分？")
    ck("不传参时 = 最近一条提问",
       m.previous_user_question() == "那迟到早退算不算旷课？")
    # 关键：当前这句已经在记忆里时，要跳过它再往上找；上面没有更早的提问了 → 空串
    m2 = SessionMemory(mem_id=new_mem_id())
    m2.add_user("Q1")
    m2.add_assistant("A1")
    ck("当前问题已入记忆、上面没别的提问 → 空串",
       m2.previous_user_question("Q1") == "")
    m2.add_user("Q2")
    m2.add_assistant("A2")
    ck("当前=Q2 且已入记忆 → 往上取到 Q1",
       m2.previous_user_question("Q2") == "Q1")
    ck("当前=Q3（还没入记忆）→ 取到 Q2",
       m2.previous_user_question("Q3") == "Q2")
    m.reset()
    ck("reset 清空", m.turns == [] and m.summary == "")

    print("\n③ 历史块")
    ck("空记忆 → 空串", SessionMemory(mem_id="x").history_block() == "")
    only_sum = SessionMemory(mem_id="x", summary="学生问过补考。")
    ck("只有摘要也能出块", "更早内容的摘要" in only_sum.history_block())
    m = SessionMemory(mem_id="x")
    m.add_user("必修课不及格能补考吗？")
    m.add_assistant("可以，补考安排在下学期初。")
    hb = m.history_block()
    ck("有【之前的对话】抬头", "【之前的对话】" in hb)
    ck("学生与助手都在", "学生：" in hb and "助手：" in hb)
    ck("旧的在前、新的在后",
       hb.index("学生：必修课不及格能补考吗？") < hb.index("助手：可以，补考安排在下学期初。"))
    ck("末尾叮嘱指代按上文理解", "指代不明" in hb)
    big = mem_with(30, text="很长很长的内容" * 8)
    hb2 = big.history_block()
    ck("超预算时总量被压住",
       len(hb2) < MAX_HISTORY_CHARS + 400, f"实际 {len(hb2)}")
    ck("最新的那条一定还在", "内容问29" in hb2)
    ck("最旧的已被丢出", "内容问0" not in hb2)
    one = SessionMemory(mem_id="x")
    one.add_user("很长的问" * 200)
    line = [l for l in one.history_block().split("\n") if l.startswith("学生：")][0]
    ck(f"单条超长截到 {MSG_TRUNC} 字左右",
       line.endswith("…") and len(line) <= MSG_TRUNC + len("学生：") + 1,
       f"实际 {len(line)} 字")

    print("\n④ 超预算判定")
    ck("小记忆不触发摘要", not mem_with(2).over_budget())
    ck("大记忆触发摘要", mem_with(30, text="很长很长的内容" * 8).over_budget())

    print("\n⑤ 摘要（有损的一步，守得最紧）")
    # mem_with(10) = 10 问 10 答 = 20 条；摘要后应只剩最近 keep=6 条
    m = mem_with(10)
    llm = FakeLLM()
    ck("确实做了摘要", summarize_older(m, llm, keep=6) is True)
    ck("只留最近 6 条", len(m.turns) == 6, f"实际 {len(m.turns)}")
    ck("留下的就是最末那 6 条", m.turns[-1]["content"] == "内容答9")
    ck("被压掉的确实进了摘要", m.summary.startswith("（摘要）"))
    ck("调了 1 次模型", llm.calls == 1)
    # 下面三种「模型不给力」的情形都必须**一条不删**（mem_with(5) = 10 条）
    m = mem_with(5)
    ck("模型不可用 → 不摘要", summarize_older(m, None, keep=6) is False)
    ck("模型不可用 → 一条不删", len(m.turns) == 10, f"实际 {len(m.turns)}")
    m = mem_with(5)
    ck("模型报错 → 不摘要", summarize_older(m, FakeLLM(boom=True), keep=6) is False)
    ck("模型报错 → 一条不删（宁超预算不漏信息）", len(m.turns) == 10, f"实际 {len(m.turns)}")
    m = mem_with(5)
    ck("模型返回空 → 不摘要", summarize_older(m, FakeLLM(text="   "), keep=6) is False)
    ck("模型返回空 → 一条不删", len(m.turns) == 10, f"实际 {len(m.turns)}")
    m = mem_with(5)
    summarize_older(m, FakeLLM(text="废" * 500), keep=6)
    ck(f"摘要被截到 {SUMMARY_CHARS} 字", len(m.summary) == SUMMARY_CHARS)
    m = mem_with(3)          # 6 条，本来就没到 keep
    ck("轮次本来就不多 → 不动", summarize_older(m, FakeLLM(), keep=6) is False
       and len(m.turns) == 6)

    print("\n⑥ 序列化")
    m = mem_with(3)
    m.summary = "旧摘要"
    back = SessionMemory.from_dict("id1", m.to_dict(), "cid")
    ck("往返一致", back is not None and back.turns == m.turns and back.summary == "旧摘要")
    ck("形状不对 → None",
       SessionMemory.from_dict("x", None) is None
       and SessionMemory.from_dict("x", "字符串") is None
       and SessionMemory.from_dict("x", {"turns": "不是表"}) is None)
    dirty = {"summary": 5, "turns": [
        {"role": "user", "content": "好问题"},
        {"role": "boss", "content": "非法角色"},
        {"role": "assistant", "content": 123},
    ]}
    clean = SessionMemory.from_dict("x", dirty)
    ck("脏数据被过滤，只留合法一条",
       clean is not None and clean.turns == [{"role": "user", "content": "好问题"}])
    ck("summary 非字符串 → 退成空串", clean.summary == "")
    many = mem_with(80)
    ck(f"云库里最多存 {MAX_TURNS_STORED} 条",
       len(many.to_dict()["turns"]) == MAX_TURNS_STORED)
    ck("brief 有会话号与轮次", "会话" in many.brief() and "轮次" in many.brief())

    print("\n⑦ 云库返回兼容（不同网关包的形状不一样）")
    ck("整数", _as_count(3) == 3)
    ck("[3]", _as_count([3]) == 3)
    ck("[{'count':3}]", _as_count([{"count": 3}]) == 3)
    ck("{'n':3}", _as_count({"n": 3}) == 3)
    ck("None → None（不知道，就不假装清了 0 行）", _as_count(None) is None)
    ck("空表 → None", _as_count([]) is None)
    ck("True（bool 不是行数）→ None", _as_count(True) is None)
    ck("认不出 → None", _as_count("随便什么东西") is None)

    print("\n⑧ 云库关掉时全部安静失败（记忆坏了也不能耽误提问）")
    mid = new_mem_id()
    m = mem_with(1)
    m.mem_id = mid
    ck("save 返回 False 不抛", save_memory(m) is False)
    ck("load 返回 None 不抛", load_memory(mid) is None)
    ck("clear 返回 False 不抛", clear_memory(mid) is False)
    ck("sweep 返回 None 不抛", sweep_memory(60) is None)
    ck("空 mem_id 也安全",
       save_memory(SessionMemory(mem_id="")) is False
       and load_memory("") is None and clear_memory("") is False)

    print("\n" + ("全部通过 ✓" if _fail == 0 else f"有 {_fail} 项失败 ✗"))
    return 0 if _fail == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
