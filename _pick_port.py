# -*- coding: utf-8 -*-
"""给「启动网页问答.bat」找一个能用的端口，打印出来给 bat 读。

为什么需要它（2026-10-15 踩到）：这台机器上还有另一个 Streamlit 项目
（桌面上的「兴智杯」）也用了 8501。谁先启动谁占住，后启动的那个会因为
端口冲突当场报错退出 —— 而浏览器已经打开了 8501，于是你看到的是**别人的
页面**，或者一个**旧进程留下的残页**（点按钮毫无反应）。

判定「被占用」用**实际连一下**，不用 bind 试探（2026-10-15 连踩两次）：
Windows 上 bind 同一个端口**会成功**，哪怕端口正被别人听着 —— 试过
`0.0.0.0:8501` 和 `::` 两种写法都能绑上，于是脚本假报「8501 空闲」。
改成 connect：连得上 = 有人在这上面服务 = 这个端口不能用。
"""
from __future__ import annotations

import socket
import sys

PREFERRED = 8501
TRIES = 20            # 往后试多少个端口，免得无限往上撞


def is_free(port: int) -> bool:
    """能连上就说明有服务在跑，那就不算空闲。"""
    for host in ("127.0.0.1", "::1"):
        try:
            with socket.create_connection((host, port), timeout=0.4):
                return False
        except OSError:
            continue          # 连不上：这个地址族上没人；换下一个试
    return True


def pick(preferred: int = PREFERRED) -> int:
    for port in range(preferred, preferred + TRIES):
        if is_free(port):
            return port
    raise SystemExit(
        f"端口 {preferred}~{preferred + TRIES - 1} 全被占用，请先关掉别的服务")


if __name__ == "__main__":
    print(pick())
    sys.stdout.flush()
