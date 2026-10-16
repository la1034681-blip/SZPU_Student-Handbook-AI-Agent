# -*- coding: utf-8 -*-
"""线上条件的单题样例：强制走「在线编码」后端，用 .env 里那把钥匙，问一题。

    python src/dev/diag/sample_ask.py
    python src/dev/diag/sample_ask.py "挂科怎么办"

为什么要有这个（2026-10-16）：线上 Secrets 写的是 EMBED_BACKEND = "api"，
而本地 .env 写的是 "local"（本机有显卡，用本地模型更快）。也就是说
**在本地直接跑，永远跑不到线上那条路**。本脚本在导入任何项目模块之前
把 EMBED_BACKEND 改成 "api"，从而在本地复现线上条件：
  ① 问题向量与重排都走硅基流动的在线接口
  ② 钥匙一旦不被平台认/账号欠费，就会触发降级
  ③ 打印降级记录 —— 这就是学生页面上那条黄条的来源
它只读，不改任何配置，也不写云库。
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]

# ⚠ 必须在导入项目模块之前设置：配置是在模块导入时读进内存的。
# 默认复现线上（api）；想对照"满血版"就跑 SAMPLE_EMBED_BACKEND=local。
os.environ["EMBED_BACKEND"] = os.environ.get("SAMPLE_EMBED_BACKEND", "api")

sys.path.insert(0, str(ROOT / "src"))
sys.stdout.reconfigure(encoding="utf-8")

from qa_chain import build_llm, load_env_file          # noqa: E402
from retrievers import HandbookRetriever               # noqa: E402
from agent_qa import AgentQAPipeline                   # noqa: E402

DEFAULT_Q = "必修课考试不及格，能补考吗？补考成绩怎么算？"


def main() -> int:
    q = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_Q
    load_env_file()

    print("=" * 72)
    print("【线上条件复现】检索编码后端 =", os.environ.get("EMBED_BACKEND"))
    key = (os.environ.get("EMBED_API_KEY") or "").strip()
    print(f"本次使用的钥匙 = {key[:8]}…{key[-4:]}（{len(key)} 字符）")
    print(f"问题 = {q}")
    print("=" * 72)

    t0 = time.time()
    r = HandbookRetriever(str(ROOT / "data"), use_rerank=True, verbose=False)
    print(f"① 检索器就绪（{time.time() - t0:.1f}s）")

    pipe = AgentQAPipeline(r, llm=build_llm(), top_k=6, enable_planner=True,
                           enable_reflect=True, enable_gate=True, verbose=False)

    t1 = time.time()
    res = pipe.ask(q)
    cost = time.time() - t1

    print(f"\n② 作答完成，耗时 {cost:.1f}s（正常满血约 40~60s；秒回＝前面就炸了）")

    deg = list(getattr(r, "degraded", []) or [])
    plan = res.plan or {}
    deg_plan = list(plan.get("degraded_retrieval") or [])
    print(f"\n③ 降级记录：检索器上 {len(deg)} 条 / 本次答复里 {len(deg_plan)} 条")
    for d in deg:
        print(f"   - {d}")
    print(f"   向量编码器是否已弃用 = {r.embedder is None}"
          f"   重排器是否已弃用 = {r.reranker is None}")

    print(f"\n④ 证据 {len(res.evidences)} 条")
    for e in res.evidences:
        print(f"   [{e.eid}] {e.clause_id:<12} {(e.cite or '')[:46]}")

    print(f"\n⑤ 学生看到的结果")
    print(f"   拒答 = {res.refused}   原因 = {res.refuse_reason}")
    if res.refused:
        try:
            sys.path.insert(0, str(ROOT))
            from app import friendly_refuse                # noqa: E402
            title, body = friendly_refuse(res.refuse_reason or "", q)
            print(f"   页面标题：{title}")
            print(f"   页面正文：{body}")
        except Exception as e:                             # noqa: BLE001
            print(f"   （取页面话术失败：{type(e).__name__}: {e}）")
    else:
        ans = (res.answer or "").strip()
        print("   " + ans[:400].replace("\n", "\n   "))

    v = res.verify or {}
    print(f"\n⑥ 自检：通过 = {v.get('passed')}  降级放行 = {v.get('degraded')}")
    print(f"   自检轮数 = {res.reflect_rounds}   检索次数 = {res.hops}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
