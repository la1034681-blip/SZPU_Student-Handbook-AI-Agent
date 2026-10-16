# -*- coding: utf-8 -*-
"""云库交互数据 —— 建议留言墙 + 使用统计看板。

与 oplog 的分工
---------------
oplog 是「单向、异步、失败静默」的日志出口；这里是「要等结果」的交互数据：
读留言墙、提交建议、拉统计。失败必须让调用方看得见，所以两者不共用代码 ——
把「静默」和「必须报错」揉进一个模块，迟早互相打架。

用到的表（WorkBuddy 云库，见 .env 的 CLOUD_ENDPOINT / CLOUD_PUBLISHABLE_KEY）
    qa_usage       一次提问一行：client_id / session_id / created_at，**不含内容**
    qa_suggestion  留言墙：client_id / content / created_at
两张表都是 RLS「公开读 + 追加写」，不给 UPDATE / DELETE。要删留言得由管理员
在云库后台做 —— 页面上不留删除入口，免得学生之间互相删。

为什么不把行拉回来本地算
-----------------------
统计走云库里的 `qa_stats()` 聚合函数（一次请求返回全部数字）。数据涨到几万
行，看板还是一次请求、几十毫秒；若把行拉回来在 Python 里算，行数一多页面
就会肉眼可见地卡。
"""
from __future__ import annotations

import os
import uuid
from datetime import datetime

import requests


def _flag(name: str, default: str = "1") -> bool:
    v = (os.environ.get(name, default) or default).strip().lower()
    return v not in ("0", "false", "no", "off", "")


class CloudStore:
    """云库的读写门面。任何方法都**不抛异常**，失败用返回值表达。"""

    def __init__(self):
        self.endpoint = (os.environ.get("CLOUD_ENDPOINT")
                         or "https://szpu-handbook-qa.app.workbuddy.host").rstrip("/")
        self.key = os.environ.get("CLOUD_PUBLISHABLE_KEY") or ""
        self.usage_table = os.environ.get("CLOUD_USAGE_TABLE") or "qa_usage"
        self.sugg_table = os.environ.get("CLOUD_SUGGEST_TABLE") or "qa_suggestion"
        self.enabled = _flag("CLOUD_LOG_ENABLED", "1")

    # ------------------------------------------------------------ 底层
    def _url(self, path: str) -> str:
        return f"{self.endpoint}/.cloud/database/rest/{path.lstrip('/')}"

    def _headers(self) -> dict:
        return {"x-wb-webapp-access-key": self.key,
                "Content-Type": "application/json"}

    def select(self, table: str, select: str = "*", order: str = "",
               extra: dict | None = None, limit: int = 50, timeout: int = 8):
        """读一张表。失败返回 None —— 让调用方区分「读到了 0 条」和「没读到」。"""
        if not self.enabled or not self.key:
            return None
        params = {"select": select, "limit": str(limit)}
        if order:
            params["order"] = order
        if extra:
            params.update(extra)
        try:
            r = requests.get(self._url(table), headers=self._headers(),
                             params=params, timeout=timeout)
            if r.status_code in (200, 206):
                return r.json()
        except Exception:                                        # noqa: BLE001
            pass
        return None

    def insert(self, table: str, row: dict, timeout: int = 8) -> bool:
        if not self.enabled or not self.key:
            return False
        try:
            r = requests.post(self._url(table), headers=self._headers(),
                              json=row, timeout=timeout)
            return r.status_code in (200, 201, 204)
        except Exception:                                        # noqa: BLE001
            return False

    def rpc(self, name: str, payload: dict | None = None, timeout: int = 10):
        if not self.enabled or not self.key:
            return None
        try:
            r = requests.post(self._url(f"rpc/{name}"), headers=self._headers(),
                              json=payload or {}, timeout=timeout)
            if r.status_code in (200, 201):
                return r.json()
        except Exception:                                        # noqa: BLE001
            pass
        return None

    def rpc_ok(self, name: str, payload: dict | None = None,
               timeout: int = 10) -> bool:
        """调 RPC，只问「成没成功」。

        为什么不复用 rpc()：返回 void 的函数**没有响应体**，`r.json()` 会直接
        抛异常 → rpc() 返回 None → 调用方把「写成功了」误判成「失败」
        （2026-10-15 查临时记忆表时踩到：qa_mem_save 明明写进去了，
        save_memory 却报 False）。这里只看 HTTP 状态码，不看响应体。
        """
        if not self.enabled or not self.key:
            return False
        try:
            r = requests.post(self._url(f"rpc/{name}"), headers=self._headers(),
                              json=payload or {}, timeout=timeout)
            return 200 <= r.status_code < 300
        except Exception:                                        # noqa: BLE001
            return False

    # ------------------------------------------------------------ 自检
    def probe(self) -> dict:
        """连通性探测 —— 专给「系统自检」页用，**故意把失败原因拆开说清**。

        正常的读写方法都遵循「失败就静默 / 只给一句人话」，那是为了不打扰学生；
        但运维排查时那种含蓄反而要命：地址错了、密钥没配、还是网络不通，
        从一句「没提交上，可能网络不通」里根本分不出来。这里把地址、有没有密钥、
        HTTP 状态码、异常类型、耗时全部返回，且不抛异常。
        """
        import time
        info = {"endpoint": self.endpoint, "has_key": bool(self.key),
                "enabled": self.enabled, "status": None, "ms": None,
                "err": "", "ok": False}
        if not self.enabled:
            info["err"] = "CLOUD_LOG_ENABLED=0 —— 云库功能被开关关掉了"
            return info
        if not self.key:
            info["err"] = ("没有 CLOUD_PUBLISHABLE_KEY —— 线上请在 Streamlit 后台 "
                           "Settings → Secrets 里配（不是 Environment variables）")
            return info
        t0 = time.time()
        try:
            r = requests.get(self._url(self.sugg_table), headers=self._headers(),
                             params={"select": "id", "limit": "1"}, timeout=12)
            info["status"] = r.status_code
            info["ms"] = int((time.time() - t0) * 1000)
            if r.status_code in (200, 206):
                info["ok"] = True
            else:
                info["err"] = f"HTTP {r.status_code} —— {r.text[:180]}"
        except Exception as e:                                   # noqa: BLE001
            info["ms"] = int((time.time() - t0) * 1000)
            info["err"] = f"{type(e).__name__}: {str(e)[:180]}"
        return info

    # ------------------------------------------------------------ 留言墙
    def add_suggestion(self, client_id: str, content: str) -> tuple[bool, str]:
        """提交一条建议。返回 (成功?, 给用户看的一句话)。"""
        text = (content or "").strip()
        if len(text) < 2:
            return False, "再多写几个字吧"
        if len(text) > 500:
            return False, "500 字以内就够了，精简一下"
        ok = self.insert(self.sugg_table, {
            "id": str(uuid.uuid4()),
            "client_id": (client_id or "anon")[:64],
            "content": text,
        }, timeout=6)
        if ok:
            return True, "收到了，谢谢"
        return False, "没提交上，可能网络不通，稍后再试一次"

    def list_suggestions(self, limit: int = 60, timeout: int = 4):
        """读留言墙（新→旧）。

        **读不到返回 None，真的没有才返回空列表** —— 这两件事在页面上必须
        说不同的话：读不到要说「暂时读不到」，说成「墙上还空着」等于
        骗人（2026-10-15 用户报「一开始没显示之前的留言」查出来的就是它：
        云库一超时，None 被 `or []` 抹成空列表，页面照着「空」渲染）。

        timeout 默认给 4 秒而不是 select 的 8 秒：留言墙不是关键功能，
        宁可快速降级也别让整页干等 —— 它排在页面最后，卡住最影响体感。
        """
        rows = self.select(self.sugg_table,
                           select="id,content,created_at",
                           order="created_at.desc", limit=limit,
                           timeout=timeout)
        return None if rows is None else list(rows)

    # ------------------------------------------------------------ 统计
    def stats(self):
        """看板用的全部数字，一次拿回。失败返回 None。"""
        return self.rpc("qa_stats")

    def usage_since(self, since_iso: str, limit: int = 20000) -> list:
        """取某时刻之后的使用记录（只两列），升序。

        现在没有调用方 —— 提问限流已于 2026-10-15 下线。保留它，是为了限流
        将来恢复时（`limiter.restore()` 需要这批数据）不用重写一遍。
        """
        rows = self.select(self.usage_table,
                           select="client_id,created_at",
                           order="created_at.asc",
                           extra={"created_at": f"gte.{since_iso}"},
                           limit=limit, timeout=12)
        return rows or []

    def usage_records(self, since_iso: str, limit: int = 20000) -> list:
        """→ [(client_id, 时间戳秒), ...]，按时间升序；解析不了的行直接跳过。"""
        out = []
        for r in self.usage_since(since_iso, limit):
            cid = r.get("client_id")
            raw = r.get("created_at")
            if not cid or not raw:
                continue
            try:
                ts = datetime.fromisoformat(
                    str(raw).replace("Z", "+00:00")).timestamp()
            except ValueError:
                continue
            out.append((cid, ts))
        return out

    @staticmethod
    def local_midnight_iso() -> str:
        """本地时区今天 0 点的 ISO 字符串，用作「今日」查询的起点。"""
        now = datetime.now().astimezone()
        return now.replace(hour=0, minute=0, second=0,
                           microsecond=0).isoformat()


# ------------------------------------------------------------------ 进程单例
_STORE: CloudStore | None = None
_LOCK = None


def get_store() -> CloudStore:
    """取全进程唯一的云库客户端。"""
    global _STORE, _LOCK
    if _LOCK is None:
        import threading
        _LOCK = threading.Lock()
    if _STORE is None:
        with _LOCK:
            if _STORE is None:
                _STORE = CloudStore()
    return _STORE


if __name__ == "__main__":
    import sys
    sys.stdout.reconfigure(encoding="utf-8")
    from qa_chain import load_env_file
    load_env_file()
    st_ = get_store()
    print("配置:", {"endpoint": st_.endpoint, "enabled": st_.enabled,
                  "usage": st_.usage_table, "sugg": st_.sugg_table})
    print("统计:", st_.stats())
    rows = st_.list_suggestions(5)
    if rows is None:
        print("留言墙：读不到（云库超时或没配 —— 注意不是「没有留言」）")
    else:
        print(f"留言墙（{len(rows)} 条）:")
        for r in rows:
            print("  ", r.get("created_at", "")[:16], (r.get("content") or "")[:50])
