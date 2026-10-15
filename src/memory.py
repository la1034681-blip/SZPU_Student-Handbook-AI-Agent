# -*- coding: utf-8 -*-
"""会话记忆 —— 让追问能接着上文答，但只做**临时**记忆，退出即清空。

为什么需要它
------------
链路本身是「每题独立」的：`qa_chain.USER_TEMPLATE` 里只有【问题】【证据】
两个槽位，没有任何历史。学生追问一句「那它算不算迟到早退」时，模型不知道
「它」指什么，检索也会拿这半截话去搜 —— 必然答不上。

三层结构
--------
1. 轮次 turns        最近几轮问答原文，直接拼进提示词（`history_block()`）
2. 滚动摘要 summary  更早的轮次被压成一段摘要。**超预算时才做，多一次模型调用**
3. 追问改写          由 planner 顺带产出 standalone 问题（**零额外调用**），
                     让检索拿到完整语义 —— 见 `planner.py` 的 `standalone`

存在哪（两层，互相兜底）
------------------------
- 主存放：`st.session_state` —— 浏览器标签页内，关掉就没了
- 副本：  云库 `qa_memory` —— **临时**表，闲置超时由 `qa_mem_sweep` 清掉。
  该表不授权给匿名角色（`REVOKE ALL ... FROM anon, authenticated`），
  只能通过 `qa_mem_save/load/clear/sweep` 四个 SECURITY DEFINER 函数读写，
  且函数必须带上会话自己的随机 id —— **别人拿不到你的会话内容**。
  验证脚本：`src/dev/diag/probe_memory_table.py`（含「直接读表被拒」一项）。

为什么不把整段历史一直留着
--------------------------
上下文越长越贵、越慢，且模型容易被前面的内容带偏。所以给一个字符预算：
超了就把最旧的几轮压成摘要。**摘要失败就不删** —— 宁可超一点预算，
也不静默丢信息（`history_block()` 本身会按预算从新到旧截断）。

接在哪（页面侧，见 app.py）
---------------------------
    memory_id()     会话号 → 地址栏 ?m=，刷新后仍认得出是同一次会话
    get_memory()    取记忆对象；本页没有就从云库读回（读不到就新建一个空的）
    ask_and_render() 作答前 history_block()/previous_user_question() 一起交给
                    `AgentQAPipeline.ask(history_block=…, prev_user=…)`；
                    作答后 `_remember()` 写入这一轮、超预算则 summarize_older()、
                    再 save_memory() 存一份到云库
    「清空对话」      mem.reset() + clear_memory() 一起清
    「系统自检」页   显示记忆状态，另有「清掉闲置记忆」按钮
    main() 启动时    _sweep_idle() 清掉闲置会话（近似「用户已经关页面走了」）
"""
from __future__ import annotations

import json
import secrets
import sys
from dataclasses import dataclass, field

sys.stdout.reconfigure(encoding="utf-8")


# ------------------------------------------------------------------ 预算
MAX_HISTORY_CHARS = 1200     # 历史块字符预算（摘要 + 最近几轮）
KEEP_MESSAGES = 6            # 摘要后至少保留最近 6 条（3 轮）
MSG_TRUNC = 400              # 单条消息进提示词时截到多少字
MAX_MESSAGES = 40            # 硬上限，防无限增长（超了直接压摘要）
SUMMARY_CHARS = 200          # 摘要长度上限（提示词里约定，这里只做兜底截断）
MAX_TURNS_STORED = 60        # 云库里最多存多少条，防单行过大


def new_mem_id() -> str:
    """会话记忆的随机 id（32 位十六进制）。它就是访问云库那行的唯一凭据。"""
    return secrets.token_hex(16)


# ------------------------------------------------------------------ 记忆对象
@dataclass
class SessionMemory:
    mem_id: str
    client_id: str = "anon"
    summary: str = ""
    turns: list = field(default_factory=list)   # [{"role","content"}, ...]

    # ---------------------------------------------------------- 写
    def add(self, role: str, content: str) -> None:
        text = (content or "").strip()
        if not text:
            return
        self.turns.append({"role": role, "content": text})

    def add_user(self, text: str) -> None:
        self.add("user", text)

    def add_assistant(self, text: str) -> None:
        self.add("assistant", text)

    def reset(self) -> None:
        self.summary = ""
        self.turns = []

    # ---------------------------------------------------------- 读
    def recent(self, n: int = KEEP_MESSAGES) -> list:
        return self.turns[-n:] if n > 0 else list(self.turns)

    def previous_user_question(self, current: str = "") -> str:
        """最近一次**上一轮**的学生提问（供追问检索兜底用）。

        会跳过与 current 相同的那一条 —— 万一调用方先把本次问题写进了记忆
        再来取，也不会把本次问题当成「上一轮」。
        """
        cur = (current or "").strip()
        skipped = False
        for m in reversed(self.turns):
            if m.get("role") != "user":
                continue
            t = (m.get("content") or "").strip()
            if not t:
                continue
            if not skipped and cur and t == cur:
                skipped = True
                continue
            return t
        return ""

    @staticmethod
    def _line(m: dict) -> str:
        who = "学生" if m.get("role") == "user" else "助手"
        text = (m.get("content") or "").replace("\n", " ").strip()
        if len(text) > MSG_TRUNC:
            text = text[:MSG_TRUNC] + "…"
        return f"{who}：{text}"

    def history_block(self, budget: int = MAX_HISTORY_CHARS) -> str:
        """拼给提示词的历史块。没历史返回空串（调用方据此跳过注入）。

        从**最新往回**装，装到预算为止 —— 保证「最近说的」一定在，
        更早的靠 summary 兜着。
        """
        if not self.turns and not self.summary:
            return ""
        lines: list = []
        used = 0
        for m in reversed(self.turns):
            line = self._line(m)
            if used + len(line) > budget:
                break
            lines.append(line)
            used += len(line) + 1
        lines.reverse()
        parts = ["【之前的对话】"]
        if self.summary:
            parts.append(f"（更早内容的摘要）{self.summary}")
        if not lines:
            parts.append("（更早的对话已省略）")
        parts.extend(lines)
        parts.append("（以上是背景；请优先回答本次【问题】。若本次问题里的"
                     "「它/这个/那」指代不明，按上面的对话理解。）")
        return "\n".join(parts)

    def over_budget(self, budget: int = MAX_HISTORY_CHARS) -> bool:
        """该做摘要了吗。"""
        if len(self.turns) > MAX_MESSAGES:
            return True
        return len(self.history_block(budget)) > budget

    def pop_oldest(self, n: int) -> list:
        """摘掉最旧的 n 条并返回它们（供摘要）。不动 summary。"""
        n = max(0, min(n, len(self.turns) - 1))
        out, self.turns = self.turns[:n], self.turns[n:]
        return out

    # ---------------------------------------------------------- 序列化
    def to_dict(self) -> dict:
        return {"summary": self.summary,
                "turns": self.turns[-MAX_TURNS_STORED:]}

    @classmethod
    def from_dict(cls, mem_id: str, data, client_id: str = "anon"):
        """从云库读回来的 JSON 复原。形状不对就返回 None（调用方当没有记忆）。"""
        if not isinstance(data, dict):
            return None
        turns = data.get("turns")
        if not isinstance(turns, list):
            return None
        clean = []
        for m in turns:
            if (isinstance(m, dict) and m.get("role") in ("user", "assistant")
                    and isinstance(m.get("content"), str)):
                clean.append({"role": m["role"], "content": m["content"]})
        # 只认字符串摘要。云库那一列是 jsonb，万一被别处写成数字/对象，
        # 与其 str() 成一个 "5" 这种垃圾摘要喂给模型，不如当没有。
        summary = data.get("summary")
        return cls(mem_id=mem_id, client_id=client_id,
                   summary=summary if isinstance(summary, str) else "",
                   turns=clean)

    def brief(self) -> str:
        return (f"会话 {self.mem_id[:8]}… ｜ 轮次 {len(self.turns)} 条 ｜ "
                f"摘要 {len(self.summary)} 字")


# ------------------------------------------------------------------ 摘要
SUMMARY_SYSTEM = """你在维护一段多轮问答（学生问大学学生手册）的记忆摘要。
把对话压成简短要点，且**只能**保留对话里真实出现过的内容：
  · 学生问过哪些主题
  · 已经给出的关键数字、期限、结论
  · 还没解决或学生明显还要继续问的点
不要补充任何对话里没有的信息，不要写建议或客套话。"""

SUMMARY_USER = """【已有的更早摘要】
{prev}

【需要并入的新对话】
{dialog}

把这两部分合并成**一段**新的摘要，{limit} 字以内，直接输出摘要正文："""


def summarize_older(mem: SessionMemory, llm, keep: int = KEEP_MESSAGES) -> bool:
    """把最旧的几轮压进 mem.summary，返回是否真的改动了。

    三条原则：
      · 只压不删到「最近 keep 条」为止 —— 最近的对话永远原文保留；
      · 模型不可用 / 调用失败 → **一条都不删**，宁可超预算（不漏信息）；
      · 摘要本身也截断（防止模型写长了把预算吃光）。
    """
    if llm is None or len(mem.turns) <= keep:
        return False
    n = len(mem.turns) - keep
    old = mem.turns[:n]
    dialog = "\n".join(SessionMemory._line(m) for m in old)
    prompt = SUMMARY_USER.format(prev=mem.summary or "（无）",
                                 dialog=dialog, limit=SUMMARY_CHARS)
    try:
        from langchain_core.messages import (HumanMessage,  # noqa: E402
                                             SystemMessage)
        resp = llm.invoke([SystemMessage(content=SUMMARY_SYSTEM),
                           HumanMessage(content=prompt)])
        raw = resp.content if hasattr(resp, "content") else str(resp)
        text = (raw or "").strip()
    except Exception:                                        # noqa: BLE001
        return False
    if not text:
        return False
    mem.summary = text[:SUMMARY_CHARS]
    mem.turns = mem.turns[n:]
    return True


# ------------------------------------------------------------------ 云库副本
def _store():
    from cloudstore import get_store
    return get_store()


def save_memory(mem: SessionMemory) -> bool:
    """把记忆存进云库临时表。失败返回 False（不影响作答 —— 页面仍用内存那份）。

    注意：`qa_mem_save` 返回 void、**没有响应体**，所以只能用 `rpc_ok`
    （看 HTTP 状态码），不能用 `rpc`（解析 JSON 会把成功判成失败）。
    """
    if mem is None or not mem.mem_id:
        return False
    try:
        return _store().rpc_ok("qa_mem_save", {
            "p_id": mem.mem_id,
            "p_client": (mem.client_id or "anon")[:64],
            "p_data": mem.to_dict(),
            "p_turns": len(mem.turns) // 2,
        })
    except Exception:                                        # noqa: BLE001
        return False


def load_memory(mem_id: str, client_id: str = "anon"):
    """从云库取回这次会话的记忆。取不到返回 None。"""
    if not mem_id:
        return None
    try:
        rows = _store().rpc("qa_mem_load", {"p_id": mem_id})
    except Exception:                                        # noqa: BLE001
        return None
    if not isinstance(rows, list) or not rows:
        return None
    row = rows[0] if isinstance(rows[0], dict) else {}
    data = row.get("data")
    if isinstance(data, str):                                # 有的网关把 jsonb 当字符串
        try:
            data = json.loads(data)
        except Exception:                                    # noqa: BLE001
            return None
    return SessionMemory.from_dict(mem_id, data, client_id)


def clear_memory(mem_id: str) -> bool:
    """清掉这次会话在云库里的记忆（用户点「清空对话」或会话结束）。"""
    if not mem_id:
        return False
    try:
        return _store().rpc("qa_mem_clear", {"p_id": mem_id}) is not None
    except Exception:                                        # noqa: BLE001
        return False


def _as_count(out):
    """从 RPC 返回里挤出「删了几行」。

    不同网关把同一个 `SELECT count(*)::int` 包成不同形状 —— 见过 `[3]`、
    `[{"count": 3}]`、`{"count": 3}`。这里逐个认，认不出返回 None
    （宁可说「不知道」，也不要瞎报一个 0 假装清理过了）。
    """
    if isinstance(out, bool) or out is None:
        return None
    if isinstance(out, int):
        return out
    if isinstance(out, list):
        return _as_count(out[0]) if out else None
    if isinstance(out, dict):
        for v in out.values():
            if isinstance(v, int) and not isinstance(v, bool):
                return v
    return None


def sweep_memory(minutes: int = 60):
    """清掉闲置超过 minutes 分钟的会话（视为用户已经关页面走了）。

    为什么靠超时：浏览器关标签页时服务器收不到通知（HTTP 本来就是无连接的），
    没有任何可靠的「用户退出了」事件。所以用「多久没人动过」来近似。
    返回删掉的行数；云库不可用返回 None。
    """
    try:
        out = _store().rpc("qa_mem_sweep", {"p_minutes": int(minutes)})
    except Exception:                                        # noqa: BLE001
        return None
    return _as_count(out)


if __name__ == "__main__":
    from qa_chain import load_env_file
    load_env_file()
    m = SessionMemory(mem_id=new_mem_id(), client_id="selftest")
    m.add_user("旷课多少学时会被处分？")
    m.add_assistant("10-19 学时警告，50 学时以上开除学籍。")
    m.add_user("那迟到早退算不算旷课？")
    print("历史块：")
    print(m.history_block())
    print("\n超预算？", m.over_budget())
    print("存云库：", save_memory(m))
    back = load_memory(m.mem_id)
    print("读回来：", back.brief() if back else None)
    print("读完一致：", bool(back) and back.turns == m.turns)
    print("清掉：", clear_memory(m.mem_id), "｜ 再读：", load_memory(m.mem_id))
