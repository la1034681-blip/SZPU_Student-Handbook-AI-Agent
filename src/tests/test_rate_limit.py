# -*- coding: utf-8 -*-
"""免费档 429 的处理 —— 纯离线，不发任何网络请求。

背景（2026-10-15）：
  智谱 GLM-4-Flash 是「永久免费、**不限量但限速**」。一道题要调模型好几次
  （规划 + 生成 + 校验 + 复核），几个学生同时用就可能撞上
  「429 您的使用量已超出频率限制，将在 XX:XX 重置」。

  原实现没有任何 429 处理：异常直接冒到页面，学生看到的是天书，
  还会以为系统坏了。本测试守住两件事：
    ① 429 能被识别（且不会误伤其他异常）
    ② 429 被翻译成人话，其他异常照旧上抛（不能把真故障也糊过去）

跑法：`python src/tests/test_rate_limit.py`
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

SRC = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(SRC))
import _paths  # noqa: E402 把 core/retrieval/infra/cli 也挂上搜索路径

from qa_chain import _env_int, _wrap_429, is_rate_limited   # noqa: E402

fails: list = []


def check(name: str, got, want) -> None:
    ok = got == want
    print(f"  {'✓' if ok else '✗'} {name}")
    if not ok:
        print(f"      期望 {want!r}，实得 {got!r}")
        fails.append(name)


# ================================================== ① 导入自检
print("① 模块可导入、关键符号在位", flush=True)

import qa_chain                                     # noqa: E402

check("is_rate_limited 存在", hasattr(qa_chain, "is_rate_limited"), True)
check("_wrap_429 存在", hasattr(qa_chain, "_wrap_429"), True)


# ============================================ ② 异常分类本身（翻译在 ④ 验）
print("\n② 限流异常该被认出、真故障不该被误判（翻译行为在 ④ 用真实类验）", flush=True)

check("429 状态码",
      is_rate_limited(Exception("Error code: 429 - {'error': {'code': '1302'}}")), True)
check("限流中文原文",
      is_rate_limited(Exception("429 您的使用量已超出频率限制，将在 12:04:44 重置")), True)
check("请求过于频繁",
      is_rate_limited(Exception("请求过于频繁，请稍后再试")), True)
check("英文 Too Many Requests",
      is_rate_limited(Exception("Too Many Requests")), True)
check("RateLimitError 类名",
      is_rate_limited(type("RateLimitError", (Exception,), {})("boom")), True)

print("   —— 反向：这些是**真故障**，不能被当成限流 ——")
check("JSON 解析错", is_rate_limited(Exception("Extra data: line 1")), False)
check("网络超时", is_rate_limited(TimeoutError("timed out")), False)
check("密钥无效", is_rate_limited(Exception("401 Unauthorized")), False)
check("空消息", is_rate_limited(Exception("")), False)


# ======================================================== ③ 配置项容错
print("\n③ _env_int：配置写错不能崩", flush=True)
import os                                        # noqa: E402

os.environ.pop("FREE_LLM_MAX_RETRIES", None)
check("未配置 → 默认 3", _env_int("FREE_LLM_MAX_RETRIES", 3), 3)
os.environ["FREE_LLM_MAX_RETRIES"] = "5"
check("正常值", _env_int("FREE_LLM_MAX_RETRIES", 3), 5)
os.environ["FREE_LLM_MAX_RETRIES"] = "abc"
check("写错 → 退回默认而非崩溃", _env_int("FREE_LLM_MAX_RETRIES", 3), 3)
os.environ["FREE_LLM_MAX_RETRIES"] = ""
check("空值 → 默认", _env_int("FREE_LLM_MAX_RETRIES", 3), 3)
os.environ.pop("FREE_LLM_MAX_RETRIES", None)


# ==================================================== ④ build_llm 真接上
print("\n④ build_llm(free)：包装后必须仍是合法 Runnable", flush=True)
from langchain_core.runnables import Runnable              # noqa: E402
from langchain_core.language_models.chat_models import BaseChatModel  # noqa: E402
from langchain_core.output_parsers import JsonOutputParser  # noqa: E402
from langchain_core.prompts import ChatPromptTemplate        # noqa: E402
from qa_chain import build_llm                              # noqa: E402

llm = build_llm(backend="free")

# 这三条是回归重点：第一版把客户端套成裸代理类，结果整条链炸在
# 「Expected a Runnable, callable or dict」——chain 只收 Runnable。
check("仍是 Runnable", isinstance(llm, Runnable), True)
check("仍是 BaseChatModel", isinstance(llm, BaseChatModel), True)
check("能被拼进 chain（第一版的致命伤）",
      (ChatPromptTemplate.from_messages([("user", "{q}")])
       | llm | JsonOutputParser()) is not None, True)
check("模型名透传", llm.model_name, "glm-4-flash")
check("重试次数已开", getattr(llm, "max_retries", None), 3)
check("temperature 透传", llm.temperature, 0.0)
check("有 with_retry 方法", hasattr(llm, "with_retry"), True)

# 429 的翻译路径：**打真实包装类**的实例（不造子类，免得测了个假东西）。
# 做法是 monkeypatch 它的父类 ChatOpenAI.invoke，抛一个 429 看包装层怎么翻译。
from langchain_openai import ChatOpenAI as _Raw                      # noqa: E402

_real_invoke = _Raw.invoke
try:
    def _boom(self, *a, **k):
        raise RuntimeError("429 您的使用量已超出频率限制，将在 12:04:44 重置")

    _Raw.invoke = _boom
    try:
        llm.invoke("x")
        check("模拟 429 被翻译", False, True)
    except RuntimeError as e:
        check("模拟 429 → 说人话", "限流" in str(e), True)
        check("原始错误保留", "429" in str(e), True)

    def _boom_other(self, *a, **k):
        raise RuntimeError("Extra data: line 1 column 5")

    _Raw.invoke = _boom_other
    try:
        llm.invoke("x")
        check("真故障被翻译（不该）", False, True)
    except RuntimeError as e:
        check("真故障不被糊成限流", "限流" in str(e), False)
finally:
    _Raw.invoke = _real_invoke


print("\n" + "=" * 58)
if fails:
    print(f"✗ 失败 {len(fails)} 项：")
    for f in fails:
        print("   -", f)
    raise SystemExit(1)
print("✓ 全部通过")
