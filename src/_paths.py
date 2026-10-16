# -*- coding: utf-8 -*-
"""把 src/ 下各分类子目录挂上模块搜索路径（入口统一接入点）。

为什么需要它
------------
本项目里的模块之间用**平铺模块名**互相引用，例如 `from retrievers import
HandbookRetriever`。以前所有代码都平铺在 src/ 下，入口只要把 src/ 挂上
sys.path 就够了：

    sys.path.insert(0, str(ROOT / "src"))
    from qa_chain import build_llm

2026-10-17 按职责把代码分进了四个子目录：

    src/core/        问答核心（提问 → 找条款 → 写答案 → 校验）
    src/retrieval/   检索与编码（找条款、算向量、口语词换算）
    src/infra/       支撑设施（日志、会话记忆、云库、热重载）
    src/cli/         命令行入口

此时只挂 src/ 已经找不到这些模块了 —— 还得把上面四个目录也挂上。与其在每个
入口重复四行，不如收在这里一处维护；入口在原来的 sys.path.insert 之后加一行：

    import _paths      # noqa: F401   把 core/retrieval/infra/cli 也挂上

注意
----
· 只挂**已存在的**目录，缺了不报错（免得删掉某个分类时整页起不来）。
· 幂等：Python 的 import 本身就只执行一次。
· 新加一个分类目录，只要往 CATEGORIES 里补一个名字。
"""
from __future__ import annotations

import sys
from pathlib import Path

SRC = Path(__file__).resolve().parent

# 分类子目录名 —— 新增一类就补一个
CATEGORIES = ("core", "retrieval", "infra", "cli")


def register() -> list[str]:
    """把 src/ 与各分类子目录挂上搜索路径，返回这次真正插进去的目录。"""
    added = []
    for d in (SRC, *(SRC / c for c in CATEGORIES)):
        s = str(d)
        if d.is_dir() and s not in sys.path:
            sys.path.insert(0, s)      # 插到最前，保证优先于同名第三方包
            added.append(d.name or "src")
    return added


register()
