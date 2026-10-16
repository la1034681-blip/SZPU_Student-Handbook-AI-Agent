# -*- coding: utf-8 -*-
"""兜底链路测试：**主动权在模型，模型坏了才轮到程序化**。

背景（2026-10-15 用户要求）：「只留主动权在 LLM，其他删除，但保留兜底的程序
和内容 —— 比如模型出问题了有程序化兜底」。于是：
  · 「程序化作答通道」这层壳（QA_MODE 开关、ProceduralPipeline、可切换的
    过程面板）全部删除；
  · `src/core/procedural_qa.py` 保留，但身份变成**兜底生成器**，两处会被调用：
      ① `AgentQAPipeline.ask()` 抛异常时 → `_rescue()` 照抄原文作答；
      ② 模型两次坚持拒答时 → 同样改照抄（零幻觉保底）。

本脚本三组验证：
  ① 单测 rescue：模型必炸 → 必须给出「照抄原文」的答案（不是报错、不是空白）
  ② 兜底程序本身没被删坏：`procedural_qa.generate` 仍能出带出处的答案
  ③ 端到端：把 API Key 改成错的 → 页面上必须出现「程序化兜底」提示
     （这条最贵也最值得：它验的正是「模型出问题学生不会白问一场」）

⚠ 第 ③ 组不发真实模型调用（Key 是错的，请求会被拒），但会真跑检索。
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / "src"))
import _paths  # noqa: E402 把 core/retrieval/infra/cli 也挂上搜索路径
sys.path.insert(0, str(Path(__file__).resolve().parent))

os.environ["ACCESS_CODE"] = "off"
os.environ["CLOUD_LOG_ENABLED"] = "0"

from retrievers import Evidence                        # noqa: E402
from langchain_core.runnables import RunnableLambda    # noqa: E402

fails: list = []


def check(label: str, ok: bool, extra: str = "") -> None:
    print(f"   {'✓' if ok else '✗'} {label} {extra}", flush=True)
    if not ok:
        fails.append(label)


def make_evs() -> list:
    """真实 Evidence 对象（别自己造 dict —— .text 是属性访问，喂 dict 会炸）。"""
    text = ("第十条 学生课程考核不及格，可以参加补考。补考成绩合格者，"
            "成绩按及格记载；补考不合格的，应当重修该课程。"
            "第十五条 学生旷课累计达到10学时，给予警告处分；"
            "累计达到20学时，给予严重警告处分。")
    return [Evidence(eid=f"E{i+1}", clause_id=f"C{i+1}", chunk_id=f"K{i+1}",
                     cite=f"《深圳职业技术大学专科学生学籍管理规定》 第{10+i}条",
                     file_id="F1", file_title="学籍管理规定",
                     article_label=f"第{10+i}条", page=str(20 + i),
                     text=text, score=0.97 - i * 0.01)
            for i in range(3)]


class _FakeRetriever:
    """只实现 search —— 足够跑通兜底路径，不用加载真的检索模型。"""

    def __init__(self, evs):
        self._evs = evs

    def search(self, q, top_k=6, **kw):
        return list(self._evs)[:top_k]


class _BrokenRetriever:
    def search(self, q, top_k=6, **kw):
        raise RuntimeError("向量库文件被删了")


def _boom(_):
    raise RuntimeError("429 您的使用量已超出频率限制，将于 18:20 重置")


def main():
    from agent_qa import AgentQAPipeline

    print("① 模型炸了 → 必须由程序化兜底照抄原文（不是报错、不是空白）", flush=True)
    bad_llm = RunnableLambda(_boom)          # 合法 Runnable：链能建成，调用时炸
    pipe = AgentQAPipeline(_FakeRetriever(make_evs()), llm=bad_llm,
                           top_k=6, verbose=False)
    res = pipe.ask("必修课考试不及格，能补考吗？")
    plan = res.plan or {}
    check("标明了这是兜底", plan.get("fallback") == "procedural", str(plan)[:100])
    check("兜底执行成功", plan.get("fallback_ok") is True)
    check("记下了原始错误", "429" in str(plan.get("fallback_error") or ""),
          str(plan.get("fallback_error"))[:80])
    check("没被当成拒答", res.refused is False, res.refuse_reason)
    check("答案里有原文", "补考" in res.answer, res.answer[:60].replace("\n", " "))
    check("答案带引用角标", "[E1]" in res.answer)
    check("citations 非空", bool(res.citations), str(res.citations))
    check("引用校验通过", bool((res.verify or {}).get("passed")),
          str((res.verify or {}).get("reasons"))[:80])

    print("\n② 连检索都炸了 → 只能老实拒答，不许编", flush=True)
    pipe2 = AgentQAPipeline(_BrokenRetriever(), llm=bad_llm, verbose=False)
    res2 = pipe2.ask("学校食堂几点关门？")
    check("拒答", res2.refused is True)
    check("拒答理由说明是大模型不可用",
          "大模型不可用" in (res2.refuse_reason or ""), res2.refuse_reason[:80])
    check("答案为空（没编）", not (res2.answer or "").strip())

    print("\n③ 兜底程序本身没坏：procedural_qa.generate 仍能出答案", flush=True)
    from procedural_qa import generate, _verify
    out = generate("必修课考试不及格，能补考吗？", make_evs(), max_evidence=2)
    check("不判为证据不足", out.get("insufficient") is False, str(out.get("reason")))
    check("带出处", bool(out.get("citations")), str(out.get("citations")))
    check("原文照抄（含「补考成绩合格」）", "补考成绩合格" in out.get("answer", ""))
    check("照抄结果过引用校验",
          bool(_verify(out["answer"], out["citations"], make_evs()).get("passed")))

    print("\n④ 通道开关确实没了：程序化不再是可选项", flush=True)
    app_src = (ROOT / "app.py").read_text(encoding="utf-8")
    check("app.py 不再读 QA_MODE", 'os.environ.get("QA_MODE")' not in app_src)
    check("app.py 不再有程序化过程面板", "_render_process_proc" not in app_src)
    import procedural_qa
    check("procedural_qa 里没有通道壳 ProceduralPipeline",
          not hasattr(procedural_qa, "ProceduralPipeline"))
    check("procedural_qa 里没有 make_free_fallback",
          not hasattr(procedural_qa, "make_free_fallback"))

    print("\n⑤ 端到端：Key 故意配错 → 页面上必须出现兜底提示", flush=True)
    os.environ["FREE_LLM_API_KEY"] = "sk-this-key-is-deliberately-wrong"
    from streamlit.testing.v1 import AppTest
    from _askui import ask
    at = AppTest.from_file(str(ROOT / "app.py"), default_timeout=1200).run()
    check("页面能起来（模型坏了不该白屏）", not at.exception,
          [str(e) for e in at.exception][:1])
    ask(at, "必修课考试不及格，能补考吗？")
    info = "\n".join(i.value for i in at.info)
    md = "\n".join(m.value for m in at.markdown)
    body = info + "\n" + md
    check("出现「程序化兜底」提示", "程序化兜底" in body or "没答上来" in body,
          body[:150].replace("\n", " "))
    check("给出了答案（不是空）", "补考" in md, md[:120].replace("\n", " "))
    roles = [m["role"] for m in at.session_state["messages"]]
    check("会话记录完整（一问一答）", roles == ["user", "assistant"], roles)

    print("\n" + ("全部通过" if not fails else f"失败 {len(fails)} 项：{fails}"),
          flush=True)
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
