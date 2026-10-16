# -*- coding: utf-8 -*-
"""检索降级测试：外部接口挂了，**不许把整道题判死**。

背景（2026-10-16 线上事故，用户贴图）：
  学生问「必修课考试不及格，能补考吗？补考成绩怎么算？」—— 手册里写得明明白白
  （第十六条、第二十一条），页面却回「本手册中没有对应内容」，耗时 **0.7 秒**。
  0.7 秒这个数字就是线索：正常作答要 40~60 秒，这么快说明大模型根本没被叫起来。

  真凶是**检索的第一步就抛异常**：向量检索与重排走的是外部接口（硅基流动），
  那个账号 **额度用完**，接口回 HTTP 402：
      {"code":30001,"message":"Sorry, your account balance is insufficient"}
  于是：检索抛错 → 兜底路径再搜一次、又抛错 → 老实拒答。每一步本身都没做错，
  错的是**两件配套的事**：
    ① 缺了一路就该降级继续（还有 BM25 关键词路，它不依赖任何外部接口），
       而不是让整道题跟着一起死；
    ② 事故原因被页面说成了「手册里没有对应内容」，还建议学生「换个说法再问」——
       后台明明已经瘫了，学生换一百个说法也没用。

本脚本六组：
  ① 向量路挂 → 仍出证据，且相关分退回 BM25 原始分（拒答门槛才有意义）
  ② 重排挂 → 顺序退回融合序，仍出证据
  ③ 都正常 → **一条降级记录都不许有**（不然就是误报）
  ④ 降级原因能传到流水线的 plan 里（页面靠它显示提示）
  ⑤ 端到端：把在线编码打成必然失败 → 页面上必须出现「打了折扣」提示，
     并且**照样给出答案**（不再是「手册里没有对应内容」）
  ⑥ 技术故障不许说成「手册里没有」（friendly_refuse 的分支）
全离线：向量路用假编码器顶替，不发一个网络请求。
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

import numpy as np                                  # noqa: E402
from bm25 import BM25                               # noqa: E402
from retrievers import HandbookRetriever            # noqa: E402

fails: list = []

Q = "必修课考试不及格，能补考吗？补考成绩怎么算？"


def check(label: str, ok: bool, extra: str = "") -> None:
    print(f"   {'✓' if ok else '✗'} {label} {extra}", flush=True)
    if not ok:
        fails.append(label)


class _BoomEmbedder:
    """一编码就抛 —— 模拟「在线编码接口额度用完 / 断网」。"""

    model = "stub"

    def encode(self, *a, **kw):
        raise RuntimeError("HTTP 402：额度不足 —— 模拟硅基流动余额不足")


class _OkEmbedder:
    """返回一个固定向量：够让 `_dense` 跑完（内容的对错不是本测试要验的）。"""

    model = "stub"

    def encode(self, texts, **kw):
        n = len(texts) if isinstance(texts, (list, tuple)) else 1
        return np.ones((n, 1024), dtype=np.float32) / 32.0


class _BoomReranker:
    def predict(self, *a, **kw):
        raise RuntimeError("HTTP 402：额度不足 —— 模拟重排接口挂了")


class _OkReranker:
    def predict(self, pairs, **kw):
        return np.linspace(1.0, 0.0, len(pairs), dtype=np.float32)


def _offline_retriever(embedder, reranker=None) -> HandbookRetriever:
    """造一个**不加载任何真模型**的检索器：索引是真的，模型是假的。"""
    r = HandbookRetriever(str(ROOT / "data"), use_rerank=True,
                          backend="local", verbose=False)
    r.build()                                  # 只读 data/ 里的索引与向量文件
    r.bm = BM25().fit([(c["chunk_id"], c["embed_text"]) for c in r.chunks])
    r.embedder, r.reranker = embedder, reranker
    r._models_ready = True                     # 别去真的加载 bge-m3 / 走在线接口
    r._dev, r._dtype_eff = "stub", None
    return r


def main() -> int:
    print("① 向量路挂了 → 照样出证据（只剩关键词路）", flush=True)
    r1 = _offline_retriever(_BoomEmbedder())
    evs = r1.search(Q, top_k=6)
    check("检索没抛异常、有结果", bool(evs), f"{len(evs)} 条")
    check("记下了降级原因", any("向量检索" in d for d in r1.degraded),
          str(r1.degraded)[:90])
    check("降级原因里带得出「额度」这类可排查的话",
          any("402" in d for d in r1.degraded), str(r1.degraded)[:90])
    check("相关分是 BM25 原始分（0 = 一个字都没对上，门槛才有意义）",
          max(e.score for e in evs) > 1.0, f"top1={max(e.score for e in evs):.3f}")
    check("向量路废了一次就不再重试（省掉每题一次白等）",
          r1.embedder is None)
    evs2 = r1.search(Q, top_k=6)
    check("第二次检索也照常出结果", bool(evs2), f"{len(evs2)} 条")
    check("降级记录不重复堆", len(r1.degraded) == 1, str(r1.degraded)[:90])

    print("\n② 重排挂了 → 顺序退回融合序，仍出证据", flush=True)
    r2 = _offline_retriever(_OkEmbedder(), _BoomReranker())
    evs3 = r2.search(Q, top_k=6)
    check("有结果", bool(evs3), f"{len(evs3)} 条")
    check("记下了「重排」降级", any("重排" in d for d in r2.degraded),
          str(r2.degraded)[:90])
    check("没有误记「向量检索」降级",
          not any("向量检索" in d for d in r2.degraded), str(r2.degraded)[:90])

    print("\n③ 两路都正常 → 一条降级记录都不许有（防误报）", flush=True)
    r3 = _offline_retriever(_OkEmbedder(), _OkReranker())
    _ = r3.search(Q, top_k=6)
    check("没有降级记录", r3.degraded == [], str(r3.degraded)[:90])

    print("\n④ 降级原因要能传到流水线的 plan 里（页面靠它提示）", flush=True)
    from agent_qa import AgentQAPipeline
    pipe = AgentQAPipeline(_offline_retriever(_BoomEmbedder()), llm=None,
                           enable_planner=False, enable_reflect=False,
                           verbose=False)
    plan = pipe.plan_only(Q)
    check("planner 关掉时也拿得到 plan", isinstance(plan, dict))
    from types import SimpleNamespace
    evs4 = pipe._merge([Q])
    check("_merge 在降级下仍能合并出证据", bool(evs4), f"{len(evs4)} 条")
    deg = list(getattr(pipe.retriever, "degraded", []) or [])
    check("流水线能读到降级原因（app.py 就是从这里取的）", bool(deg), str(deg)[:90])
    _ = SimpleNamespace()

    print("\n⑤ 端到端：在线编码必然失败 → 页面必须明说「打了折扣」，"
          "并且照样给答案", flush=True)
    # 把在线编码的两条路都打成必失败：不发网络请求，立刻抛。
    import embed_api
    embed_api.ApiEmbedder.encode = _BoomEmbedder.encode
    embed_api.ApiReranker.predict = _BoomReranker.predict
    os.environ["EMBED_BACKEND"] = "api"
    os.environ["EMBED_API_KEY"] = "sk-offline-stub-not-used"
    os.environ["FREE_LLM_API_KEY"] = "sk-this-key-is-deliberately-wrong"

    from streamlit.testing.v1 import AppTest
    from _askui import ask
    at = AppTest.from_file(str(ROOT / "app.py"), default_timeout=1200).run()
    check("页面能起来（接口挂了不该白屏）", not at.exception,
          [str(e) for e in at.exception][:1])
    ask(at, Q)
    body = "\n".join(i.value for i in at.info) + "\n" + \
        "\n".join(w.value for w in at.warning) + "\n" + \
        "\n".join(m.value for m in at.markdown)
    check("页面上出现了「打了折扣」的说明", "打了折扣" in body,
          body[:160].replace("\n", " "))
    check("页面/过程面板里有降级记录", "降级" in body or "向量检索" in body,
          body[:160].replace("\n", " "))
    check("**没有**再把技术故障说成「本手册中没有对应内容」",
          "本手册中没有对应内容" not in body, body[:160].replace("\n", " "))
    roles = [m["role"] for m in at.session_state["messages"]]
    check("会话记录完整（一问一答）", roles == ["user", "assistant"], roles)

    print("\n⑥ 技术故障的分支：说人话，且不推给「换个说法」", flush=True)
    import importlib.util
    spec = importlib.util.spec_from_file_location("_app_mod", ROOT / "app.py")
    ap = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ap)                            # 只抠函数用，不跑页面
    t, tip = ap.friendly_refuse("大模型不可用（RuntimeError: HTTP 402：额度不足），"
                                "检索也失败（RuntimeError: …）",
                                question="挂科怎么办")
    check("标题是技术故障那套", t == "服务暂时出了点小状况", t)
    check("说明里点明不是「手册里没有」", "不是手册里没有规定" in tip, tip[:80])
    check("技术故障不贴「换个说法再问」的口语词提示",
          "手册里的写法" not in tip, tip[-80:])
    t2, tip2 = ap.friendly_refuse("引用校验未通过：断言无法被证据支持",
                                  question="挂科怎么办")
    check("真·查不到时仍照旧提示换手册用词", "手册里的写法" in tip2, tip2[-90:])

    print("\n" + ("全部通过" if not fails else f"失败 {len(fails)} 项：{fails}"),
          flush=True)
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
