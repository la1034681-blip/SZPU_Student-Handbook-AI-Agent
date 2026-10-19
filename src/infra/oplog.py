# -*- coding: utf-8 -*-
"""访问日志 —— 把每次提问异步写到 WorkBuddy 云库，同时留一份本地安全日志。

为什么要写
----------
给学生多人使用时，必须能回答三个问题：谁在问、问什么、有没有异常。
本地日志能回答「有没有异常」，云端日志能回答「问什么」（换台机器也能查）。

设计要点
--------
1. **绝不阻塞作答**：写入放后台单线程队列，用户答完就走，不等网络。
2. **绝不因日志失败影响用户**：任何异常都吞掉，只在本地安全日志里留一行。
3. **默认开启、可一键关闭**：`LOG_ENABLED=0` 即完全静默（离线演示时用）。
4. **只上传必要字段**：问题、是否拒答、耗时、条数、引用的条款号、答案。
   不采集任何身份信息（只用一个随机会话号，重启浏览器即换新）。

表结构（云库，见 .env 的 CLOUD_ENDPOINT / CLOUD_PUBLISHABLE_KEY）
- `qa_log`：id / session_id / question / refused / latency / n_evidence /
  cited_clauses / answer，日期由数据库自动填。**只授 INSERT** —— 里面有学生
  提问原文，不给读权限。
- `qa_usage`：id / client_id / session_id / created_at。一次提问一行，
  不含内容。**公开读 + 追加写**，供统计看板使用。
  （提问限流已于 2026-10-15 临时下线；这张表仍照常记录，看板数字不受影响，
   将来恢复限流时它还是现成的回填数据源。）
"""
from __future__ import annotations

import json
import os
import queue
import threading
import time
import uuid
from pathlib import Path

import requests

# 项目根目录：从本文件往上找，第一个含 app.py 的目录就是（模块已按分类住进
# src/infra/ 等子目录，不能再写死往上两级）。见 src/_paths.py 的目录约定。
_ROOT = next((d for d in Path(__file__).resolve().parents if (d / "app.py").exists()),
             Path(__file__).resolve().parents[2])
_LOCAL_LOG = _ROOT / "logs" / "access.log"
_SENTINEL = object()


def _flag(name: str, default: str = "1") -> bool:
    v = (os.environ.get(name, default) or default).strip().lower()
    return v not in ("0", "false", "no", "off", "")


class OpLogger:
    """后台单线程写日志器：调用方只管 `log()`，永不等待。"""

    def __init__(self):
        self.enabled = _flag("LOG_ENABLED", "1")
        self.endpoint = (os.environ.get("CLOUD_ENDPOINT")
                         or "https://szpu-handbook-qa.app.workbuddy.host").rstrip("/")
        self.key = (os.environ.get("CLOUD_PUBLISHABLE_KEY")
                    or "wbpk_5LyhZ0F5RWsUb899resz7e_VIoZzX64rPlPlexNlB5dPbg4fHrvRzsP")
        self.table = os.environ.get("CLOUD_LOG_TABLE") or "qa_log"
        # 使用计数表：只记「谁在什么时候问了一次」，不含问题内容。
        # 主要给统计看板供数（限流下线后不再用于回填）。
        self.usage_table = os.environ.get("CLOUD_USAGE_TABLE") or "qa_usage"
        self.remote = _flag("CLOUD_LOG_ENABLED", "1")
        self._q: queue.Queue = queue.Queue(maxsize=500)
        self._n_ok = 0
        self._n_fail = 0
        self._worker: threading.Thread | None = None
        self._lock = threading.Lock()

    # ------------------------------------------------------------ 内部
    def _ensure_worker(self):
        if self._worker and self._worker.is_alive():
            return
        with self._lock:
            if self._worker and self._worker.is_alive():
                return
            t = threading.Thread(target=self._run, name="oplog", daemon=True)
            t.start()
            self._worker = t

    def _local(self, line: str):
        try:
            _LOCAL_LOG.parent.mkdir(parents=True, exist_ok=True)
            with _LOCAL_LOG.open("a", encoding="utf-8") as f:
                f.write(line + "\n")
        except Exception:                                        # noqa: BLE001
            pass

    def _post(self, row: dict, table: str = ""):
        url = f"{self.endpoint}/.cloud/database/rest/{table or self.table}"
        try:
            r = requests.post(
                url,
                headers={"x-wb-webapp-access-key": self.key,
                         "Content-Type": "application/json"},
                json=row, timeout=15)
            if r.status_code in (200, 201, 204):
                self._n_ok += 1
            else:
                self._n_fail += 1
                self._local(f"[cloud {r.status_code}] {r.text[:160]}")
        except Exception as e:                                   # noqa: BLE001
            self._n_fail += 1
            self._local(f"[cloud err] {e}")

    def _run(self):
        while True:
            row = self._q.get()
            if row is _SENTINEL:
                return
            try:
                if self.remote:
                    table = row.pop("_table", "")
                    self._post(row, table)
            except Exception as e:                               # noqa: BLE001
                self._local(f"[worker err] {e}")

    # ------------------------------------------------------------ 对外
    def log(self, session_id: str, question: str, answer: str = "",
            refused: bool = False, latency: float = 0.0, n_evidence: int = 0,
            cited: list | str | None = None, web: str = ""):
        """记一条提问。任何情况下都不会抛异常、不会阻塞。

        `web`（2026-10-17 新增）：官网兜底的命中情况，只写**本地日志**，
        不进云库行 —— 云库那张表是固定结构，加列会让写入整体失败，而这条
        信息眼下只在排查与对账时用得上。取值形如 "hit:2"（命中 2 段）、
        "link"（只给链接）、"none"（官网也没有）、""（没走这趟）。
        ⚠ 「官网命中」必须能和「手册答对」分开看 —— 混在一起，手册的误拒
        就会被官网的命中掩盖掉。
        """
        if not self.enabled:
            return
        ts = time.strftime("%Y-%m-%d %H:%M:%S")
        if isinstance(cited, (list, tuple)):
            cited = ",".join(str(c) for c in cited)
        cited = cited or ""
        # 本地安全日志：一行一条，纯文本好查
        self._local(f"{ts}\t{session_id}\trefused={int(bool(refused))}\t"
                    + (f"web={web}\t" if web else "")
                    + f"{latency:.2f}s\t{question[:80]}")
        if not self.remote:
            return
        row = {
            "id": str(uuid.uuid4()),
            "session_id": session_id[:64],
            "question": (question or "")[:2000],
            # 答案留前 4000 字：既够复盘，也不至于把库撑爆
            "answer": (answer or "")[:4000],
            "refused": bool(refused),
            "latency": round(float(latency or 0.0), 2),
            "n_evidence": int(n_evidence or 0),
            "cited_clauses": cited[:2000],
        }
        try:
            self._q.put_nowait(row)
            self._ensure_worker()
        except queue.Full:                                       # 队列满 → 丢日志，不丢服务
            self._local(f"{ts}\t[queue full, dropped]")

    def usage(self, client_id: str, session_id: str = ""):
        """记一条「使用计数」：谁、什么时候问了一次。**不含任何问题内容**。

        为什么单独一张表而不是从 qa_log 派生：qa_log 里全是学生的问题原文，
        为了统计而给它开公开读权限，等于把提问内容也摊开了。这张表只有随机
        编号和时间，就算读权限被拿到也挖不出隐私 —— 而看板需要的恰恰就是它。

        失败静默、不阻塞：跟日志一个待遇，统计少一行无所谓，用户多等一秒不行。
        """
        if not self.enabled or not self.remote:
            return
        row = {
            "_table": self.usage_table,
            "id": str(uuid.uuid4()),
            "client_id": (client_id or "anon")[:64],
            "session_id": (session_id or "")[:64],
        }
        try:
            self._q.put_nowait(row)
            self._ensure_worker()
        except queue.Full:
            pass

    def stats(self) -> dict:
        return {"enabled": self.enabled, "remote": self.remote,
                "queued": self._q.qsize(), "ok": self._n_ok, "fail": self._n_fail}


_LOGGER: OpLogger | None = None
_LOCK = threading.Lock()


def get_logger() -> OpLogger:
    global _LOGGER
    if _LOGGER is None:
        with _LOCK:
            if _LOGGER is None:
                _LOGGER = OpLogger()
    return _LOGGER


if __name__ == "__main__":
    import sys
    sys.stdout.reconfigure(encoding="utf-8")
    from qa_chain import load_env_file
    load_env_file()
    lg = get_logger()
    print("配置:", lg.stats())
    lg.log("selftest", "自测：国奖多少钱？", answer="国家奖学金每生每年 8000 元 [E1]",
           refused=False, latency=1.23, n_evidence=3, cited=["F27-A003", "F27-A004"])
    for _ in range(40):                       # 等后台线程发完
        if lg.stats()["queued"] == 0 and (lg.stats()["ok"] or lg.stats()["fail"]):
            break
        time.sleep(0.25)
    print("写入后:", lg.stats())
    print("本地日志尾部:")
    print("\n".join(_LOCAL_LOG.read_text(encoding="utf-8").strip().splitlines()[-3:])
          if _LOCAL_LOG.exists() else "(无)")
