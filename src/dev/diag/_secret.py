# -*- coding: utf-8 -*-
"""诊断脚本用的口令读取 —— 只从项目根 `.env` 读，脚本里**不许写死**。

为什么单独抽出来（2026-10-19）
------------------------------
这个仓库是**公开**的。诊断脚本里写死访问口令，等于把门禁贴在互联网上；
写死看板密钥同理。第 17 条踩坑就是这么来的
（`probe_online_cases.py` 里写死过，改成读 .env 之后又被另一行覆盖回去，
说明"改一半"比"没改"更危险）。

所以：口令只留 `.env` 一处，所有脚本统一从这里取。缺了就明确报错退出，
不要拿空口令去点「进入」然后误判成"进不去"。

用法：
    from _secret import code, admin
    CODE = code()        # 缺了会直接 sys.exit 并说明怎么补
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]      # 项目根（往上三层到含 app.py 的目录）


def _read(key: str) -> str:
    """从项目根 .env 取一个值；环境变量已存在则优先（便于临时覆盖）。"""
    import os
    v = (os.environ.get(key) or "").strip()
    if v:
        return v
    env = ROOT / ".env"
    if not env.exists():
        return ""
    for line in env.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, _, val = line.partition("=")
        if k.strip() == key:
            return val.strip().strip('"').strip("'")
    return ""


def code() -> str:
    """访问口令。**2026-10-21 起网站已取消进站口令**，这里保留只为兼容老脚本。

    取不到就返回空串（不再 sys.exit）：所有调用的脚本都是先看页面上有没有
    口令框，没有就直接进站 —— 空串正好让那段逻辑自然跳过。
    """
    return _read("ACCESS_CODE")


def admin() -> str:
    """看板密钥。读不到同样退出（看板测试没有它跑不下去）。"""
    v = _read("ADMIN_CODE")
    if not v:
        sys.exit(f"缺看板密钥：在 {ROOT / '.env'} 里写 ADMIN_CODE=xxx 再跑。")
    return v
