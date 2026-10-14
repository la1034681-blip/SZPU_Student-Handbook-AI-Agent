# -*- coding: utf-8 -*-
"""验证「提问 → 访问日志」整条链路真的通：本地 access.log + 云库 qa_log。

与 test_app_gate.py 的区别：本脚本**不关**云库写入，用的是 .env 里的真实配置，
所以末尾能直接看到云库写入成功/失败数。
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

ROOT = Path(__file__).resolve().parent.parent.parent
APP = str(ROOT / "app.py")
sys.path.insert(0, str(ROOT / "src"))

os.environ["ACCESS_CODE"] = "SZPU666"

from streamlit.testing.v1 import AppTest  # noqa: E402
from oplog import get_logger              # noqa: E402


def main():
    at = AppTest.from_file(APP, default_timeout=900)
    at.run()

    at.text_input[0].set_value("SZPU666")
    at.button[0].click().run()
    print("进门成功（有问答入口）:", len(at.chat_input) > 0, flush=True)

    print("提问：国家奖学金一年多少钱？", flush=True)
    at.chat_input[0].set_value("国家奖学金一年多少钱？").run()
    print("异常:", bool(at.exception), flush=True)
    for e in at.exception:
        print("  EXC:", e, flush=True)

    # 等后台写日志的线程把队列排空
    lg = get_logger()
    for _ in range(60):
        s = lg.stats()
        if s["queued"] == 0 and (s["ok"] or s["fail"]):
            break
        time.sleep(0.5)
    print("日志统计:", lg.stats(), flush=True)

    p = ROOT / "logs" / "access.log"
    print("\n本地 logs/access.log 末尾 3 行:", flush=True)
    if p.exists():
        for line in p.read_text(encoding="utf-8").strip().splitlines()[-3:]:
            print("  ", line, flush=True)
    else:
        print("   （文件不存在）", flush=True)


main()
