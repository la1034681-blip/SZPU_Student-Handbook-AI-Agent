# -*- coding: utf-8 -*-
"""守卫：官网兜底接进主链路之后，四条路各走对了没有（2026-10-17）。

不问「答案好不好」，只问**分流对不对** —— 因为错的代价不对称：

  ① 手册能答的题，跑去查官网        → 不许（手册优先，一字都不许掺）
  ② 手册没有、官网有的题            → 要摊出官网原文（W1…）
  ③ 手册没有、官网也没有的题        → 老实拒答，不许瞎凑
  ④ 个人数据（我的绩点/课表）        → 直接给登录引导，**0 秒**，不搜官网

用法：python src/dev/diag/check_web_link.py
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

ROOT = next((p for p in Path(__file__).resolve().parents if (p / "app.py").exists()), None)
if ROOT is None:
    raise SystemExit("定位不到项目根目录（往上找不到 app.py）")
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "src" / "dev"))
import _paths  # noqa: F401,E402  把 core/retrieval/infra 挂上搜索路径

from qa_chain import build_llm, load_env_file      # noqa: E402
from retrievers import HandbookRetriever           # noqa: E402
from agent_qa import AgentQAPipeline               # noqa: E402

# (编号, 期望, 问题)
#   期望 = manual  手册能答，官网不该被碰
#   期望 = web     手册没有、官网有，要摊出原文
#   期望 = refuse  哪儿都没有，老实拒
#   期望 = login   个人数据，0 秒给引导
CASES = [
    ("M1", "manual", "旷课太多会有什么后果？"),
    ("M2", "manual", "拿学士学位需要满足什么条件？"),
    # ⚠ 「学校的作息时间是怎样的」**不**放 web 组：手册里确有作息纪律条款
    # （违纪处分第二十九条晚归、公寓管理办法第二十条熄灯），只是没有作息表 ——
    # 属于「手册能答一部分」，按手册优先原则不该去查官网。
    ("W1", "web", "学校的班车怎么坐？"),
    ("W2", "web", "第5节课几点上？"),
    ("R1", "refuse", "学校有几个食堂？"),
    ("L1", "login", "我这学期绩点多少？"),
]

_fail = 0


def ck(label: str, ok: bool, why: str = "") -> None:
    global _fail
    print(("  ✓ " if ok else "  ✗ ") + label + (f"　← {why}" if not ok and why else ""),
          flush=True)
    if not ok:
        _fail += 1


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    load_env_file()
    r = HandbookRetriever(str(ROOT / "data"), use_rerank=True, verbose=False)
    pipe = AgentQAPipeline(r, llm=build_llm(), verbose=True)

    print(f"后端 {r.backend} ｜ 官网兜底 {'开' if pipe.enable_web else '关'}", flush=True)

    rows = []
    for tag, want, q in CASES:
        t0 = time.time()
        res = pipe.ask(q)
        dt = time.time() - t0
        web = getattr(res, "web", {}) or {}
        login = getattr(res, "web_login", {}) or {}
        got = ("login" if login else
               "web" if web.get("found") else
               "refuse" if res.refused else "manual")
        print("=" * 74, flush=True)
        print(f"{tag} 期望 {want} → 实际 {got} ｜ {dt:.1f}s ｜ refused={res.refused}"
              f" ｜ 检索分 {max((e.score or 0) for e in res.evidences) if res.evidences else 0:.4f}",
              flush=True)
        print(f"   问：{q}", flush=True)
        if web:
            print(f"   官网：found={web.get('found')} reason={web.get('reason')} "
                  f"段{len(web.get('passages') or [])} 表{len(web.get('tables') or [])} "
                  f"链接{len(web.get('links') or [])} "
                  f"{web.get('stats', {}).get('elapsed', 0)}s", flush=True)
            for p in (web.get("passages") or [])[:2]:
                print(f"      · {p['text'][:88]}", flush=True)
        if login:
            print(f"   登录引导：{login.get('topic')}", flush=True)
        rows.append((tag, want, got, dt, bool(web), bool(login)))
        ck(f"{tag} 分流正确（期望 {want}）", got == want, f"实际 {got}")

    print("\n" + "-" * 74, flush=True)
    # 关键不变量：手册能答的题，一律不许带官网痕迹
    leak = [t for t, w, g, *_ in rows if w == "manual" and g != "manual"]
    ck("手册能答的题没有去碰官网（手册优先）", not leak, f"漏了 {leak}", )
    # 登录墙必须快（不搜官网 = 8 秒省下来）
    slow = [t for t, w, g, dt, *_ in rows if w == "login" and dt > 3]
    ck("登录墙题在 3 秒内分流（没去搜官网）", not slow, f"{slow} 超时")
    print(("\n全部通过" if not _fail else f"\n{_fail} 项不通过"), flush=True)
    return 1 if _fail else 0


if __name__ == "__main__":
    sys.exit(main())
