# -*- coding: utf-8 -*-
"""端到端确认：追问能接上上文作答（2026-10-15 新增多轮记忆）。

为什么必须走真页面
------------------
要验的是「学生连着问两句，第二句里的『它／那』能被正确理解」——
这中间穿过 planner（改写指代）→ 多路检索（拿完整语义去搜）→ 生成（带背景）
三道关，任何一道断了都答不上来。只测模块函数看不出这个。

测法
----
第一句点「旷课」示例；第二句手打「那要多久呢？」——「那」指上一句的处分。
追问单独拿去检索是搜不到东西的（实测最高相关分 0.02，会被拒答前置拦掉），
带上文之后才答得出来。所以「第二句能正常作答」本身就是多轮记忆生效的证据。

会真调大模型（两轮），免费档连跑容易撞 429，一次就好。
跑法：python src/tests/test_followup_e2e.py
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

os.environ.setdefault("ACCESS_CODE", "off")       # 跳过门禁，直取问答页
os.environ.setdefault("CLOUD_LOG_ENABLED", "0")   # 不往云库写，免得污染日志

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / "src"))
import _paths  # noqa: E402 把 core/retrieval/infra/cli 也挂上搜索路径
sys.path.insert(0, str(ROOT / "src" / "tests"))

from streamlit.testing.v1 import AppTest          # noqa: E402
from _askui import ask, click_sample              # noqa: E402

Q1_HIT = "旷课"
Q2 = "那要多久呢？"

_fail = 0


def ck(name: str, cond: bool, extra: str = ""):
    global _fail
    if not cond:
        _fail += 1
    print(f"  {'✓' if cond else '✗'} {name}" + (f"  —— {extra}" if extra and not cond else ""))


def snap(at) -> dict:
    """把当前页面上能看的东西抓成一份快照。"""
    return {
        "md": "\n".join(m.value for m in at.markdown),
        "cap": "\n".join(c.value for c in at.caption),
        "warn": "\n".join(w.value for w in at.warning),
        "n_msg": len(list(at.chat_message)),
    }


def main() -> int:
    at = AppTest.from_file(str(ROOT / "app.py"), default_timeout=600)
    at.session_state["authed"] = True
    at.run()
    if at.exception:
        print("✗ 初始渲染就炸了：", [e.value for e in at.exception][:2])
        return 1

    print("第一轮：点示例「旷课多少学时会被处分……」")
    click_sample(at, Q1_HIT)
    at.run()
    if at.exception:
        print("✗ 第一轮异常：", [e.value for e in at.exception][:2])
        return 1
    s1 = snap(at)
    ck("第一轮拿到了答案", s1["n_msg"] >= 2)
    ck("第一轮不是拒答", "本手册中没有对应内容" not in s1["warn"])

    mem = at.session_state.filtered_state.get("mem")
    ck("第一轮已进记忆（2 条）", mem is not None and len(mem.turns) == 2,
       f"实际 {len(getattr(mem, 'turns', []))} 条")

    print(f"\n第二轮（追问）：{Q2}")
    time.sleep(5)                     # 免费档缓一下，别撞 429
    ask(at, Q2)
    at.run()
    if at.exception:
        print("✗ 第二轮异常：", [e.value for e in at.exception][:2])
        return 1
    s2 = snap(at)
    joined = s2["md"] + "\n" + s2["cap"]
    print("  页面摘录：")
    for line in joined.split("\n"):
        if "多轮记忆" in line or "检索实际用的完整问句" in line:
            print("   ", line[:110])

    print()
    ck("第二轮也拿到了答案", s2["n_msg"] >= 4)
    ck("★ 追问没有被拒答（★ 这条就是「记忆生效」的证据）",
       "本手册中没有对应内容" not in s2["warn"], s2["warn"][:120])
    ck("★ 追问答案里出现上一轮的处分数字（10/50 学时）",
       ("10" in s2["md"] and "50" in s2["md"]) or "开除学籍" in s2["md"])
    ck("内部过程面板记了「多轮记忆」", "多轮记忆" in s2["cap"])
    ck("面板里能看到检索实际用的完整问句", "检索实际用的完整问句" in s2["cap"])
    ck("记忆长到 4 条（两轮各一问一答）",
       len(at.session_state.filtered_state["mem"].turns) == 4,
       f"实际 {len(at.session_state.filtered_state['mem'].turns)} 条")
    ck("地址栏里不再有 ?m=（会话号机制已随云库副本一起去掉）",
       not at.query_params.get("m"))

    print("\n" + ("全部通过 ✓" if _fail == 0 else f"有 {_fail} 项失败 ✗"))
    return 0 if _fail == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
