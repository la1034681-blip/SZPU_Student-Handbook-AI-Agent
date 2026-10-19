# -*- coding: utf-8 -*-
"""无头验证「建议墙 + 使用统计看板 + 匿名编号 + 使用计数落库」。

覆盖：
  1. 进门后地址栏带上匿名编号 ?u=（刷新也不变，统计数据才认得是同一个人）
  2. 建议墙常驻在页面上
  3. 提交一条建议 → 提示成功 → 内容真的出现在墙上（读写云库）
  4. 点「使用统计」入口 → 出现密钥框（学生到此为止）
  5. 密钥输错 → 报错，看不到任何数字
  6. 密钥 TESTADMIN → 进入看板，出现统计数字
  7. 真实提问一次 → 云库 qa_usage 的计数 +1

测试会往云库写一条带【自测】标记的留言，跑完由外部 SQL 清掉（页面没有删除入口）。
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

ROOT = Path(__file__).resolve().parent.parent.parent
APP = str(ROOT / "app.py")
sys.path.insert(0, str(ROOT / "src"))
import _paths  # noqa: E402 把 core/retrieval/infra/cli 也挂上搜索路径

# 必须在 import app 之前：load_env_file 只填「尚未存在」的变量
os.environ["ACCESS_CODE"] = "TESTCODE"
os.environ["ADMIN_CODE"] = "TESTADMIN"
os.environ["CLOUD_LOG_ENABLED"] = "1"

from streamlit.testing.v1 import AppTest          # noqa: E402
from _askui import ask, has_ask_entry            # noqa: E402


def md_all(at) -> str:
    return "\n".join(m.value for m in at.markdown)


def show_exc(at, tag: str) -> None:
    if at.exception:
        print(f"   [{tag}] 异常:")
        for e in at.exception:
            print("      ", str(e)[:300], flush=True)


def btn(at, frag: str):
    return next((b for b in at.button if frag in b.label), None)


def main():
    from qa_chain import load_env_file
    load_env_file()
    from cloudstore import get_store
    store = get_store()
    before = store.stats() or {}
    print("起始统计:", before, flush=True)

    at = AppTest.from_file(APP, default_timeout=900)
    at.run()

    print("\n① 通过口令进入", flush=True)
    show_exc(at, "初始")
    at.text_input[0].set_value("TESTCODE")
    at.button[0].click().run()
    show_exc(at, "进入")
    print("   已进主界面:", has_ask_entry(at), flush=True)

    print("\n② 匿名编号 ?u=", flush=True)
    try:
        qp = dict(at.query_params)
    except Exception as e:                                # noqa: BLE001
        qp = {}
        print("   读 query_params 失败:", e, flush=True)
    uid = qp.get("u") or ""
    if isinstance(uid, list):
        uid = uid[0] if uid else ""
    print("   query_params:", qp, flush=True)
    print("   编号形态正确:", bool(uid) and len(str(uid)) == 12, flush=True)

    print("\n③ 建议墙在页面上", flush=True)
    h = md_all(at)
    print("   有「我想说两句」:", "我想说两句" in h, flush=True)
    print("   有提交按钮:", btn(at, "提交建议") is not None, flush=True)

    print("\n④ 提交一条建议", flush=True)
    marker = "【自测】这是一条自动化测试留言，跑完会被删掉"
    if at.text_area:
        at.text_area[0].set_value(marker)
    else:
        print("   !! 找不到输入框", flush=True)
    b = btn(at, "提交建议")
    if b is not None:
        b.click().run()
        show_exc(at, "提交")
        succ = [s.value for s in at.success]
        print("   success 提示:", succ, flush=True)
        h2 = md_all(at)
        print("   墙上出现该留言:", marker[:12] in h2, flush=True)
    else:
        print("   !! 找不到提交按钮", flush=True)

    print("\n⑤ 点「使用统计」入口", flush=True)
    b = btn(at, "使用统计")
    print("   入口按钮存在:", b is not None, flush=True)
    if b is not None:
        b.click().run()
        show_exc(at, "看板入口")
        print("   出现密钥框:", len(at.text_input) > 0, flush=True)
        print("   还没看到数字:", "累计提问" not in md_all(at), flush=True)

    print("\n⑥ 密钥输错", flush=True)
    if at.text_input:
        at.text_input[0].set_value("1234")
        bb = btn(at, "进 入") or (at.button[0] if at.button else None)
        if bb is not None:
            bb.click().run()
            show_exc(at, "错误密钥")
            print("   error:", [e.value[:40] for e in at.error], flush=True)
            print("   仍看不到数字:", "累计提问" not in md_all(at), flush=True)

    print("\n⑦ 密钥 TESTADMIN 进看板", flush=True)
    if at.text_input:
        at.text_input[0].set_value("TESTADMIN")
        bb = btn(at, "进 入") or (at.button[0] if at.button else None)
        if bb is not None:
            bb.click().run()
            show_exc(at, "正确密钥")
            hm = md_all(at)
            print("   出现统计卡片:", "累计提问" in hm, flush=True)
            print("   出现趋势面板:", "最近 14 天" in hm, flush=True)
            for kw in ("累计用过的人", "今天提问", "墙上留言"):
                print(f"     含「{kw}」:", kw in hm, flush=True)

    print("\n⑧ 回问答页，真实提问一次（验证使用计数落库）", flush=True)
    bb = btn(at, "回到问答")
    if bb is not None:
        bb.click().run()
        show_exc(at, "返回")
    if has_ask_entry(at):
        ask(at, "国家奖学金一年多少钱？")
        show_exc(at, "作答")
        print("   出现答案:", ("手册原文" in md_all(at) or "依据" in md_all(at)),
              flush=True)

    import time
    time.sleep(2.0)
    after = store.stats() or {}
    print("\n⑨ 使用计数是否 +1", flush=True)
    print("   前:", before.get("total"), "→ 后:", after.get("total"), flush=True)
    print("   留言数:", before.get("sug"), "→", after.get("sug"), flush=True)
    print("   本次测试用的编号:", uid, flush=True)


main()
