# -*- coding: utf-8 -*-
"""让「刷新网页」也能用上新代码 —— 内存里的旧模块就地重新加载。

为什么需要它
------------
Streamlit 有个机制：只重新执行主脚本（app.py），被它 import 的其它文件
（本项目就是 src/ 里的那些）**一直留在内存里**，文件改了也不会重新读。

于是出现最难解释的一类故障 —— 改完代码、刷新网页，页面报：

    AgentQAPipeline.ask() got an unexpected keyword argument 'history_block'

新的一半（app.py）和旧的一半（agent_qa）对不上。学生看不懂，
我们也得先问一句「重启过没有」——2026-10-15 用户连着两次栽在这里
（第二次的原话：「刷新了都」）。

本模块在每次执行的最开头把内存与磁盘对齐：发现内存里是旧的，就地重新加载。
于是**刷新一下网页就生效**，不必去找那个黑窗口。

怎么发现「内存里是旧的」——两条线索，缺一不可
--------------------------------------------
1. **文件时间对不上**：本模块记着「上次看到这些文件时它们是什么时候改的」，
   下次执行再读一遍，对不上就说明文件刚被改过，而内存里还是上一版。
2. **接口对不上**（`PROBES`）：直接问「你这个函数认不认这个参数」。

   为什么非要第二条：如果**进程启动时加载的就已经是旧代码**（改完代码才发现
   网页没关，进程一直开着），第 1 条线索是失效的 —— 我们登记文件时间的那一刻，
   磁盘上早就是新文件了，再怎么比对都「一致」。只有直接查接口才看得出内存里
   装的是旧货。用户的实际情况正是这一种。

用它的三条规矩
--------------
1. **必须在 app.py 的 import 之前调用 `refresh()`** —— 只有先对齐，
   后面那句 `from memory import ...` 才会拿到新对象；写在 import 之后就晚了
   （Python 的 import 是幂等的，直接从内存取，不会重新读文件）。
2. 重载顺序即依赖顺序（见 `ORDER`）：被引用的先重载，引用者随后重载才能拿到新对象。
3. 调用方要顺手清掉 `st.cache_resource` 里可能攥着旧类实例的缓存
   （见 app.py 里紧随其后的 `get_pipeline.clear()`）。
"""
from __future__ import annotations

import importlib
import inspect
import os
import sys
import time
from pathlib import Path

# 本项目会被热重载的模块，**顺序＝依赖顺序**：被引用的排前面。
# 只列运行时模块；dev/、tests/ 里的不参与。
ORDER = (
    "bm25",
    "term_alias",
    "subject_guard",
    "need_login",
    "store",
    "verify_citations",
    "embed_local",
    "embed_api",
    "retrievers",
    "web_fallback",
    "qa_chain",
    "reflect",
    "planner",
    "procedural_qa",
    "cloudstore",
    "oplog",
    "memory",
    "agent_qa",
)

# 每个模块住在哪个分类子目录下（2026-10-17 按职责重组目录后新增）。
# 模块名本身不变（代码里仍是 `from retrievers import ...`），只是文件位置变了，
# 所以按名字找文件这一步要来这里查一次。见 src/_paths.py 的目录约定。
WHERE = {
    "bm25": "retrieval",
    "term_alias": "retrieval",
    "embed_local": "retrieval",
    "embed_api": "retrieval",
    "retrievers": "retrieval",
    "web_fallback": "retrieval",
    "need_login": "retrieval",
    "subject_guard": "retrieval",
    "qa_chain": "core",
    "planner": "core",
    "reflect": "core",
    "procedural_qa": "core",
    "verify_citations": "core",
    "agent_qa": "core",
    "hot_reload": "infra",
    "oplog": "infra",
    "cloudstore": "infra",
    "store": "infra",
    "memory": "infra",
    "ask": "cli",
}

# 关键接口探针：（模块名，类.函数路径，该函数必须认得的参数名）。
# 每改一次会牵动别处的接口，就往这里加一条 —— 这是「新旧混用」最可靠的照妖镜。
PROBES = (
    ("agent_qa", "AgentQAPipeline.ask", "history_block"),
    ("qa_chain", "QAPipeline.answer_from_evidences", "history_block"),
    ("planner", "Planner.plan", "history_block"),
    ("memory", "SessionMemory.history_block", None),
)

# 探针这条路最多试几次。试够了还不对，说明重载解决不了（比如真的不兼容），
# 就别每次刷新都重载一遍白费时间 —— 交给页面上的红字提示用户重启进程。
MAX_PROBE_TRIES = 3

_SEEN: dict = {}        # 模块名 → 上次看到的文件修改时间（进程内一直保留）
_probe_tries = 0        # 已经因为「探针对不上」重载过几次
HISTORY: list = []      # 本进程里历次自动重载的记录，供「系统自检」页显示


def src_dir() -> Path:
    """src/ 目录（本文件住在 src/infra/ 下，往上退一级）。"""
    return Path(__file__).resolve().parent.parent


def _file_of(name: str) -> Path:
    """模块名 → 源文件。位置查 WHERE，查不到就按老规矩当平级文件找。"""
    return src_dir() / WHERE.get(name, "") / f"{name}.py"


def _probe(name: str, dotted: str, param):
    """查一个接口。返回 None＝正常，否则返回给人看的说明。"""
    obj = sys.modules.get(name)
    if obj is None:
        return f"{name}（进程里没加载）"
    for part in dotted.split("."):
        obj = getattr(obj, part, None)
        if obj is None:
            return f"{name}.{dotted}（找不到这个函数）"
    if not param:
        return None
    try:
        if param not in inspect.signature(obj).parameters:
            return f"{name}.{dotted}() 不认 {param} 这个参数"
    except (TypeError, ValueError):
        return f"{name}.{dotted}() 读不到参数"
    return None


def stale_modules() -> list:
    """内存里的模块是不是旧版 —— 对不上的逐条说明；空列表＝一致。"""
    out = []
    for name, dotted, param in PROBES:
        msg = _probe(name, dotted, param)
        if msg:
            out.append(msg)
    return out


def _by_probe() -> list:
    """探针口径下的「旧模块」，只算**已经加载进内存**的。

    还没加载的不算 —— 等会儿 import 时自然是新的，不必重载。
    """
    return [name for name, dotted, param in PROBES
            if name in sys.modules and _probe(name, dotted, param)]


def _by_mtime() -> list:
    """文件时间口径下的「旧模块」，顺带更新登记。"""
    changed = []
    for name in ORDER:
        p = _file_of(name)
        if not p.exists():
            continue
        try:
            now = os.path.getmtime(p)
        except OSError:
            continue
        before = _SEEN.get(name)
        # 只在「上一轮记过、这一轮对不上」时才认定改过；第一次只登记
        if before is not None and abs(now - before) > 0.5 and name in sys.modules:
            changed.append(name)
        _SEEN[name] = now
    return changed


def _reload_all() -> None:
    """按依赖顺序重载所有已加载的本项目模块。

    有一个变了就连带重载全部：模块之间互相引用，只换一半照样新旧混用
    —— 那正是要修的病。
    """
    for name in ORDER:
        mod = sys.modules.get(name)
        if mod is None:
            continue
        try:
            importlib.reload(mod)
        except Exception as exc:                              # noqa: BLE001
            # 单个模块重载失败不该拖垮整页：留痕，让自检页看得见
            HISTORY.append({"at": time.time(), "modules": [name],
                            "note": f"重载失败：{type(exc).__name__}"})


def refresh() -> list:
    """把内存里的项目模块与磁盘对齐。

    返回这一次真的被重新加载的模块名；空列表＝一切已是最新（绝大多数时候如此）。
    """
    global _probe_tries

    changed = _by_mtime()
    if not changed:
        hit = _by_probe()
        if not hit:
            return []
        if _probe_tries >= MAX_PROBE_TRIES:
            return []          # 试够了还是不对，别再耗 —— 页面红字会接手
        _probe_tries += 1
        changed = hit

    _reload_all()
    HISTORY.append({"at": time.time(), "modules": list(changed)})
    return changed
