# -*- coding: utf-8 -*-
"""离线测试：改完后台代码，刷新网页就能生效（2026-10-15 新增）。

守的是什么
----------
Streamlit 只重新执行主脚本 app.py，被 import 的 src/ 模块留在内存里。
所以「改完代码 → 刷新网页」本该是没用的，页面继续按旧代码跑 —— 用户两次
栽在这里（第二次原话：「刷新了都」）。`src/hot_reload.refresh()` 就是补这个：
每次页面加载先把内存与磁盘对齐，该换的换掉。

于是有三件事必须永远成立
------------------------
① 第一次运行只登记、不乱重载（此时 import 到的本来就是最新代码）
② 文件改过 → 认出来，并且**新代码真的生效**（不是只报个名字）
③ 对齐之后再调，安安静静不动手（不然每次刷新都重载，白白变慢）

为了不碰项目里的真实模块，测试临时造一个探针文件放在 src/ 下，
用完删掉；`hot_reload.ORDER` 也只在测试期间改，结束时还原。

跑法：python src/tests/test_hot_reload.py
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / "src"))
import _paths  # noqa: E402 把 core/retrieval/infra/cli 也挂上搜索路径

import hot_reload                                    # noqa: E402

PROBE = ROOT / "src" / "_hotreload_probe.py"
_fail = 0


def ck(name: str, cond: bool, extra: str = ""):
    global _fail
    if not cond:
        _fail += 1
    print(f"  {'✓' if cond else '✗'} {name}" + (f"  —— {extra}" if extra and not cond else ""))


def write_probe(value: int) -> None:
    PROBE.write_text(f"# -*- coding: utf-8 -*-\nVALUE = {value}\n", encoding="utf-8")


def main() -> int:
    keep_order, keep_seen = hot_reload.ORDER, dict(hot_reload._SEEN)
    try:
        write_probe(1)
        old = hot_reload.ORDER
        hot_reload.ORDER = ("_hotreload_probe",)
        hot_reload._SEEN.clear()

        mod = __import__("_hotreload_probe")
        ck("探针模块初始值是 1", mod.VALUE == 1, str(mod.VALUE))

        print("① 第一次：只登记，不动手")
        ck("返回空", hot_reload.refresh() == [])
        ck("值没变", sys.modules["_hotreload_probe"].VALUE == 1)

        print("② 文件改过：认出来，且新代码真的生效")
        time.sleep(0.6)                      # 判据是 mtime 差超过 0.5 秒
        write_probe(2)
        changed = hot_reload.refresh()
        ck("认出了探针模块", "_hotreload_probe" in changed, str(changed))
        ck("新代码真的生效（VALUE 变成 2）",
           sys.modules["_hotreload_probe"].VALUE == 2,
           str(getattr(sys.modules["_hotreload_probe"], "VALUE", None)))

        print("③ 对齐之后：安静")
        ck("返回空", hot_reload.refresh() == [])

        print("④ 又改一次，还能认出来（不是只认第一次）")
        time.sleep(0.6)
        write_probe(3)
        ck("第二次改动也认出来了", "_hotreload_probe" in hot_reload.refresh())
        ck("VALUE 变成 3", sys.modules["_hotreload_probe"].VALUE == 3)
        ck("留了历史记录（自检页要显示）", len(hot_reload.HISTORY) >= 2)

        print("⑤ 真实清单健全")
        ck("清单里有 agent_qa / qa_chain / memory",
           {"agent_qa", "qa_chain", "memory"} <= set(keep_order))
        ck("被引用的排在引用者前面（依赖顺序）",
           keep_order.index("qa_chain") < keep_order.index("agent_qa"))
        ck("清单里的文件都存在",
           # 用 hot_reload 自己的定位函数，别写死 src/<模块>.py ——
           # 2026-10-17 模块按分类搬进了 core/retrieval/infra/cli 子目录。
           all(hot_reload._file_of(n).exists() for n in keep_order))
        ck("WHERE 登记了 ORDER 里的每个模块（新加模块别忘了登记目录）",
           all(n in hot_reload.WHERE for n in keep_order))
    finally:
        hot_reload.ORDER = keep_order
        hot_reload._SEEN.clear()
        hot_reload._SEEN.update(keep_seen)
        sys.modules.pop("_hotreload_probe", None)
        try:
            PROBE.unlink()
        except OSError:
            pass

    ck("探针文件已清理", not PROBE.exists())
    ck("ORDER 已还原", hot_reload.ORDER == keep_order)

    print("\n" + ("全部通过 ✓" if _fail == 0 else f"有 {_fail} 项失败 ✗"))
    return 0 if _fail == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
