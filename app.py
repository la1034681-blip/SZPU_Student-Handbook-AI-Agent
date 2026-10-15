# -*- coding: utf-8 -*-
"""学生手册智能问答系统 · 服务化界面（阶段 7 / 第 2 版：去侧边栏 + 深职大风格）

启动（在项目根目录执行）：
    C:\\Users\\lijiahui\\miniconda3\\envs\\agent\\python.exe -m streamlit run app.py

设计说明
--------
· **无侧边栏**：智能体组件（问题拆解 / 自检循环 / 拒答前置）**固定默认全开**，
  页面上不再暴露开关，界面只留主页面一条干净主线。
· **视觉**：参考深圳职业技术大学的校园格调 —— 校徽以红色为主调（中心三角形取自
  深圳市花「簕杜鹃」），校园是欧式学院风（朱红坡顶 + 米白立柱），校训
  「德业并进，自强不息」。故整页采用「深职红 + 米白 + 暖金」主色，克制不花哨。
  页头使用**学校官方校徽**（assets/szpu_logo.png，白底圆盘承载于红色横幅之上）。
· 功能全部保留：答案带出处、`[E1]` 可点击角标、内部过程面板、友好拒答话术。

性能（阶段 7 优化）
------------------
· 检索模型（bge-m3 + bge-reranker）在 CUDA 上以 fp16 推理，重排提速 ~3.3 倍，
  60 题检索总耗时 83.8s → 27.5s，召回完全不变。
· 每题作答普遍 3~12 秒；仅「触发自检重答」的复杂题会到 15 秒左右。
"""
from __future__ import annotations

import base64
import hmac
import html as _html
import os
import re
import sys
import time
import uuid
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from qa_chain import build_llm, load_env_file          # noqa: E402
from retrievers import HandbookRetriever                # noqa: E402
from agent_qa import AgentQAPipeline                    # noqa: E402
from oplog import get_logger                           # noqa: E402
from cloudstore import get_store                       # noqa: E402

DATA_DIR = str(ROOT / "data")

st.set_page_config(page_title="学生手册智能问答 ｜ 深圳职业技术大学",
                   page_icon="🔴", layout="wide",
                   initial_sidebar_state="collapsed")

# ==================================================================== 全局样式
_STYLE = """
<style>
/* 隐藏侧边栏（含折叠把手），只保留主页面 */
[data-testid="stSidebar"],
[data-testid="stSidebarCollapsedControl"],
[data-testid="collapsedControl"]{ display:none !important; }

:root{
  --szpu-red:#B01B22;        /* 深职校徽红：正红偏朱 */
  --szpu-red-deep:#7E0F14;   /* 渐变暗端 */
  --szpu-red-soft:#FBEDEA;   /* 浅红底 */
  --szpu-gold:#C8A15A;       /* 暖金点缀（呼应朱红坡顶的暖调） */
  --szpu-cream:#FBF8F3;      /* 米白底（呼应米白立柱） */
  --szpu-ink:#24211E;        /* 正文深色 */
  --szpu-gray:#6E6A66;       /* 次级文字 */
  --szpu-line:#EFE6DA;       /* 分隔线 */
}

html, body, [class*="css"]{
  font-family:"PingFang SC","Microsoft YaHei","Hiragino Sans GB",
              "Source Han Sans SC",system-ui,-apple-system,sans-serif;
}
.stApp{
  background:
    radial-gradient(1100px 460px at 50% -240px, #FBEDEA 0%, rgba(251,237,234,0) 72%),
    var(--szpu-cream);
}
.block-container{ padding-top:1.4rem; padding-bottom:6.5rem; max-width:1060px; }
#MainMenu, footer, header[data-testid="stHeader"]{ visibility:hidden; height:0; }

/* ---------------- Hero：校徽意象 + 校名 + 校训 ---------------- */
.hero{
  position:relative; overflow:hidden;
  background:linear-gradient(135deg,#B01B22 0%,#96141A 52%,#7E0F14 100%);
  border-radius:20px; padding:26px 32px; margin-bottom:14px;
  display:flex; align-items:center; gap:26px;
  box-shadow:0 12px 32px rgba(140,18,24,.22);
}
.hero::after{
  content:""; position:absolute; right:-70px; top:-90px; width:300px; height:300px;
  background:radial-gradient(circle at 32% 32%, rgba(255,255,255,.18), rgba(255,255,255,0) 66%);
  border-radius:50%; pointer-events:none;
}
.hero::before{
  content:""; position:absolute; left:0; bottom:0; width:100%; height:4px;
  background:linear-gradient(90deg,var(--szpu-gold),rgba(200,161,90,.25)); opacity:.85;
}
/* 官方校徽：白底圆盘，浮在红色横幅上 */
.crest{ flex:0 0 auto; }
.crest img{
  width:98px; height:98px; display:block; border-radius:50%;
  box-shadow:0 7px 20px rgba(0,0,0,.26), 0 0 0 3px rgba(255,255,255,.30);
}
.hero-uni{ color:#FFD9C7; font-size:.86rem; letter-spacing:5px; margin-bottom:2px; }
.hero-title{ color:#fff; font-size:2.02rem; font-weight:800; letter-spacing:2px;
             line-height:1.16; margin:0 0 7px; }
.hero-sub{ color:rgba(255,255,255,.86); font-size:.94rem; }
.hero-motto{ margin-top:11px; color:#F3D9A8; font-size:.95rem; font-weight:600;
             letter-spacing:4px; }

/* ---------------- 能力标签 ---------------- */
.chiprow{ display:flex; flex-wrap:wrap; gap:8px; margin:0 0 16px; }
.chip{ background:#fff; border:1px solid var(--szpu-line); color:var(--szpu-gray);
       border-radius:999px; padding:5px 13px; font-size:.81rem; white-space:nowrap; }
.chip b{ color:var(--szpu-red); font-weight:700; }

/* ---------------- 校园文化带（校训/校风/教风/学风/深职精神） ---------------- */
.culture{
  display:grid; grid-template-columns:repeat(5,1fr);
  background:#fff; border:1px solid var(--szpu-line); border-radius:14px;
  overflow:hidden; margin:0 0 18px; box-shadow:0 1px 3px rgba(36,33,30,.04);
}
.culture .item{ padding:13px 16px; border-right:1px solid var(--szpu-line); }
.culture .item:last-child{ border-right:none; }
.culture .k{ color:var(--szpu-red); font-size:.76rem; font-weight:700;
             letter-spacing:2px; margin-bottom:5px; }
.culture .v{ color:var(--szpu-ink); font-size:.9rem; font-weight:600;
             letter-spacing:.6px; line-height:1.45; }
@media (max-width:900px){
  .culture{ grid-template-columns:repeat(2,1fr); }
  .culture .item:nth-child(2n){ border-right:none; }
}

/* ---------------- 分区小标题 ---------------- */
.sec{ display:flex; align-items:center; gap:9px; margin:6px 0 10px; }
.sec-bar{ width:4px; height:16px; border-radius:2px; background:var(--szpu-red); }
.sec-t{ font-size:1.0rem; font-weight:700; color:var(--szpu-ink); letter-spacing:.5px; }

/* ---------------- 示例问题按钮（卡片感） ---------------- */
.stButton > button{
  width:100%; text-align:left; border-radius:12px;
  border:1px solid var(--szpu-line); background:#fff; color:var(--szpu-ink);
  padding:.72rem .95rem; font-size:.93rem; line-height:1.5; font-weight:500;
  box-shadow:0 1px 2px rgba(36,33,30,.04); transition:all .16s ease;
}
.stButton > button:hover{
  border-color:var(--szpu-red); color:var(--szpu-red);
  box-shadow:0 5px 16px rgba(176,27,34,.13); transform:translateY(-1px);
}
.stButton > button:active{ transform:translateY(0); }

/* ---------------- 对话气泡 ---------------- */
[data-testid="stChatMessage"]{
  background:#fff; border:1px solid var(--szpu-line); border-radius:14px;
  padding:.4rem .8rem; box-shadow:0 1px 3px rgba(36,33,30,.045); margin-bottom:10px;
}
[data-testid="stChatMessage"] p{ color:var(--szpu-ink); }

/* ---------------- 输入框 ---------------- */
[data-testid="stChatInput"]{
  border-radius:14px; border:1px solid var(--szpu-line); background:#fff;
}
[data-testid="stChatInput"]:focus-within{
  border-color:var(--szpu-red); box-shadow:0 0 0 3px rgba(176,27,34,.10);
}
/* ---------------- 提问表单（问答区内的输入框 + 提问按钮） ----------------
   两个坑（2026-10-15 都踩过）：
   ① 用 :has() 精确命中「装着提问输入框的那个表单」，而不是套一层 <div> ——
      Streamlit 每个 st.markdown 都是独立容器，套 div 包不住后面的表单，
      样式根本不会生效。门禁页和留言墙也是 st.form，不限定就会一起被染色。
   ② Streamlit 1.62 对 st.form 下了 `border:none !important; padding:0 !important`，
      所以这两条必须也带 !important 才盖得住（背景/圆角不用）。 */
[data-testid="stForm"]:has(input[aria-label="你的问题"]){
  border:1px solid var(--szpu-line) !important;
  border-radius:14px !important;
  background:#fff !important;
  padding:10px 12px !important;
  box-shadow:0 1px 3px rgba(36,33,30,.05);
}
[data-testid="stForm"]:has(input[aria-label="你的问题"]):focus-within{
  border-color:var(--szpu-red) !important;
  box-shadow:0 0 0 3px rgba(176,27,34,.10);
}
[data-testid="stForm"]:has(input[aria-label="你的问题"]) input{ font-size:.95rem; }
[data-testid="stForm"]:has(input[aria-label="你的问题"]) [data-testid="stVerticalBlock"]{ gap:.1rem; }

/* ---------------- 展开面板（内部过程） ---------------- */
[data-testid="stExpander"]{
  border:1px solid var(--szpu-line); border-radius:12px; background:#fff;
}
[data-testid="stExpander"] summary{ color:var(--szpu-gray); font-size:.9rem; }

/* ---------------- 引用角标 + 证据卡片 ---------------- */
.cite-badge{
  display:inline-block; padding:0 7px; margin:0 2px; border-radius:7px;
  background:var(--szpu-red-soft); color:var(--szpu-red) !important;
  font-size:.8em; line-height:1.65; font-weight:700; text-decoration:none !important;
  border:1px solid #F0CFCF; vertical-align:baseline; cursor:pointer;
}
.cite-badge:hover{ background:var(--szpu-red); color:#fff !important; border-color:var(--szpu-red); }
details.ev{
  border:1px solid var(--szpu-line); border-left:3px solid #E7C3A0; border-radius:10px;
  margin:8px 0; padding:6px 13px 9px; background:#FDFBF7;
}
details.ev:target{
  border-color:var(--szpu-red); border-left-color:var(--szpu-red); background:#FDF4F4;
  box-shadow:0 0 0 2px rgba(176,27,34,.14);
}
details.ev summary{ cursor:pointer; font-weight:600; color:#3A342F; }
.ev-meta{ color:var(--szpu-gray); font-size:.83em; margin:6px 0 4px; }
.ev-text{ white-space:pre-wrap; line-height:1.72; font-size:.93em; color:#413B36; }

/* ---------------- 访问口令门禁 ---------------- */
.gate-tip{ text-align:center; color:var(--szpu-gray); font-size:.93rem; line-height:1.75; }
.gate-tip b{ color:var(--szpu-red); }
.gate-rule{ text-align:center; color:var(--szpu-gray); font-size:.8rem;
            line-height:1.9; margin-top:4px; }
[data-testid="stForm"]{ border:none !important; padding:0 !important; }

/* ---------------- 建议墙 ---------------- */
.wall-n{ color:var(--szpu-gray); font-size:.83rem; margin:12px 0 8px; }
.wall-item{
  background:#fff; border:1px solid var(--szpu-line);
  border-left:3px solid var(--szpu-gold);
  border-radius:10px; padding:11px 14px 24px; margin:7px 0;
  position:relative; color:#3A342F; font-size:.92rem; line-height:1.72;
  white-space:pre-wrap; word-break:break-word;
}
.wall-t{ position:absolute; right:12px; bottom:6px; color:#A39C94; font-size:.75rem; }

/* ---------------- 使用统计看板 ---------------- */
.stats{ display:grid; grid-template-columns:repeat(5,1fr); gap:10px; margin:12px 0 6px; }
.stat{ background:#fff; border:1px solid var(--szpu-line); border-radius:12px;
       padding:12px 13px 13px; text-align:center; }
.stat-v{ color:var(--szpu-red); font-size:1.58rem; font-weight:800; line-height:1.15; }
.stat-v span{ font-size:.75rem; font-weight:600; color:var(--szpu-gray); margin-left:3px; }
.stat-k{ color:var(--szpu-gray); font-size:.78rem; margin-top:5px; }
.panel-t{ color:#3A342F; font-weight:700; font-size:.9rem; margin:17px 0 8px; }
.bars{ background:#fff; border:1px solid var(--szpu-line); border-radius:12px;
       padding:12px 15px; }
.bar-row{ display:grid; grid-template-columns:48px 1fr auto; align-items:center;
          gap:10px; margin:5px 0; }
.bar-d{ color:var(--szpu-gray); font-size:.78rem; }
.bar-tr{ display:block; height:11px; border-radius:3px; background:#F4EEE4; }
.bar-b{ display:block; height:11px; border-radius:3px;
        background:linear-gradient(90deg,#B01B22,#D05A4E); }
.bar-n{ color:var(--szpu-gray); font-size:.76rem; white-space:nowrap; }
table.tbl{ width:100%; border-collapse:collapse; background:#fff; font-size:.85rem;
           border:1px solid var(--szpu-line); border-radius:12px;
           overflow:hidden; }
table.tbl th{ text-align:left; color:var(--szpu-gray); font-weight:600;
              font-size:.78rem; background:#FDFBF7; padding:8px 13px;
              border-bottom:1px solid var(--szpu-line); }
table.tbl td{ padding:7px 13px; border-bottom:1px solid #F6F1E9; color:#3A342F; }
table.tbl tr:last-child td{ border-bottom:none; }
@media (max-width:720px){
  .stats{ grid-template-columns:repeat(2,1fr); }
  .culture{ grid-template-columns:repeat(2,1fr); }
}

/* ---------------- 页脚 ---------------- */
.foot{
  margin-top:26px; padding-top:14px; border-top:1px solid var(--szpu-line);
  color:var(--szpu-gray); font-size:.79rem; line-height:1.85; text-align:center;
}
.foot b{ color:var(--szpu-red); }
</style>
"""

# 学校官方校徽（assets/szpu_logo.png → base64 内联，页面自包含、无外部依赖）
_LOGO_PATH = ROOT / "assets" / "szpu_logo.png"


def _load_logo() -> str:
    if not _LOGO_PATH.exists():
        return ('<div style="width:98px;height:98px;border-radius:50%;background:#fff;'
                'display:flex;align-items:center;justify-content:center;color:#B01B22;'
                'font-weight:700;font-size:12px;text-align:center;line-height:1.3;">'
                '校徽<br>缺失</div>')
    b64 = base64.b64encode(_LOGO_PATH.read_bytes()).decode("ascii")
    return (f'<img src="data:image/png;base64,{b64}" '
            f'alt="深圳职业技术大学校徽" title="深圳职业技术大学校徽"/>')


_EMBLEM = _load_logo()

_HERO = """
<div class="hero">
  <div class="crest">__EMBLEM__</div>
  <div>
    <div class="hero-uni">深圳职业技术大学</div>
    <div class="hero-title">学生手册智能问答</div>
    <div class="hero-sub">《学生手册（2025 年修订）》　·　智能体检索增强问答</div>
    <div class="hero-motto">德业并进　·　自强不息</div>
  </div>
</div>
""".replace("__EMBLEM__", _EMBLEM)

_CHIPS = """
<div class="chiprow">
  <span class="chip">智能体 RAG 检索</span>
  <span class="chip">答案可溯源 · 条款 + PDF 页码</span>
  <span class="chip">60 题评测 <b>94.3%</b></span>
  <span class="chip">平均作答 <b>6.5 秒</b></span>
  <span class="chip">没依据就 <b>如实拒答</b></span>
</div>
"""

_CULTURE = """
<div class="sec"><span class="sec-bar"></span><span class="sec-t">校园文化</span></div>
<div class="culture">
  <div class="item"><div class="k">校训</div><div class="v">德业并进，自强不息</div></div>
  <div class="item"><div class="k">校风</div><div class="v">崇德立业，创新卓越</div></div>
  <div class="item"><div class="k">教风</div><div class="v">崇教厚德，理实并重</div></div>
  <div class="item"><div class="k">学风</div><div class="v">勤学善思，明辨笃行</div></div>
  <div class="item"><div class="k">深职精神</div><div class="v">守正创新，勇立潮头</div></div>
</div>
"""

_FOOT = """
<div class="foot">
  本系统只依据《深圳职业技术大学学生手册（2025 年修订）》原文作答，不臆测、不编造；
  手册中查不到的内容会明确告知。<br>
  设访问口令保护，仅供校内师生学习使用。<br>
  <b>深圳职业技术大学</b>　·　学生手册智能问答
</div>
"""

# ==================================================================== 固定配置
# 组件固定默认全开（用户要求：默认全开，界面不暴露开关）
OPT = dict(planner=True, reflect=True, gate=True, top_k=6, max_merged=28)

# ==================================================================== 密钥来源
_SECRETS_NOTE = "（尚未读取）"      # 读 Secrets 的结果，供「系统自检」页显示
_SECRETS_DONE = False               # 同一轮只读一次，避免被重复调用覆盖掉结果


def load_secrets():
    """把 Streamlit Secrets 灌进环境变量 —— 线上与本地共用同一套读取代码。

    为什么必须放在 `load_env_file()` **之前**：`load_env_file` 只在「变量尚不存在」
    时才写入，所以先灌 Secrets 就等于「线上以 Secrets 为准、本地以 .env 为准」，
    两份配置互不打架。本地没有 secrets.toml 时这里静默跳过，不影响现在的用法。

    线上怎么配（Streamlit Cloud → 你的 app → Settings → Secrets，TOML 格式）：

        ACCESS_CODE = "SZPU666"
        ADMIN_CODE = "9925"
        FREE_LLM_API_KEY = "…"
        EMBED_BACKEND = "api"
        EMBED_API_KEY = "…"
        EMBED_API_BASE = "https://api.siliconflow.cn/v1"

    注意：在 Streamlit Cloud 的 Settings → Environment variables 里设的变量
    不会自动进 st.secrets，密钥请统一写在 Secrets 里。

    ⚠ 2026-10-15 教训：原来这里 `except Exception: return 0` 把异常**静默吞掉**，
    结果 Secrets 格式写错（或压根没配）时程序照常启动，却在真正用到的那一刻
    冒出一堆看起来毫不相干的零散错误（提问说"缺少 EMBED_API_KEY"、留言说
    "没提交上"），排查方向全靠猜。现在把结果记进 `_SECRETS_NOTE`，
    由「系统自检」页摊开给人看 —— 失败要留下痕迹。
    """
    global _SECRETS_NOTE, _SECRETS_DONE
    if _SECRETS_DONE:
        return 0
    _SECRETS_DONE = True
    try:
        items = dict(st.secrets)
    except Exception as e:                    # 没有 secrets.toml / 格式错 / 非 Streamlit 环境
        _SECRETS_NOTE = f"没读到 —— {type(e).__name__}: {str(e)[:140]}"
        return 0
    n = 0
    for k, v in items.items():
        if isinstance(v, (str, int, float)) and str(k) not in os.environ:
            os.environ[str(k)] = str(v)
            n += 1
    _SECRETS_NOTE = f"读到 {len(items)} 项（新写入环境变量 {n} 项）"
    return n


# ==================================================================== 匿名编号
# 为什么非要有它：Streamlit 刷新一次页面就是一次新会话，session_state 归零。
# 把它放进地址栏的 ?u=，由浏览器替我们记住，刷新 / 收藏 / 后退都在 ——
# 这样「同一个浏览器」这件事在刷新后仍然成立，使用统计才认得出来是谁。
# 它是一串随机十六进制，不含任何身份信息。
# （注：提问限流已按需求临时下线，编号目前只服务于使用统计。）
_CLIENT_RE = re.compile(r"^[0-9a-f]{8,32}$")


def client_id() -> str:
    """取（必要时创建）本浏览器的匿名编号。"""
    raw = st.query_params.get("u")
    if isinstance(raw, (list, tuple)):
        raw = raw[0] if raw else ""
    uid = (raw or "").strip()
    if not _CLIENT_RE.match(uid):
        uid = uuid.uuid4().hex[:12]
        st.query_params["u"] = uid
    st.session_state["client_id"] = uid
    return uid


def admin_code() -> str:
    """看板密钥（.env 的 ADMIN_CODE，默认 9925）—— 只在服务端比对，不外发。"""
    return (os.environ.get("ADMIN_CODE") or "9925").strip()


# ==================================================================== 访问门禁
# 口令写在 .env 的 ACCESS_CODE（默认 SZPU666）。留空 = 不设门禁（本地自测用）。
_GATE_FAIL_MAX = 5        # 连续输错几次开始短时锁定
_GATE_LOCK_SEC = 60       # 锁定秒数


def _gate_code() -> str:
    return (os.environ.get("ACCESS_CODE") or "").strip()


def _render_gate():
    """未通过口令时**只**显示这一页：不加载模型、不渲染任何问答入口。

    这样即便有人扫到这个网址，也拿不到任何手册内容，更不会触发
    半分钟的模型加载把机器拖垮。
    """
    st.markdown(_STYLE, unsafe_allow_html=True)
    st.markdown(_HERO, unsafe_allow_html=True)
    st.markdown(
        "<div class='sec' style='justify-content:center;margin:16px 0 6px'>"
        "<span class='sec-bar'></span><span class='sec-t'>访问口令</span></div>"
        "<div class='gate-tip'>本系统面向在校师生开放，请先输入访问口令<br>"
        "口令由老师发放，<b>请勿外传</b></div>",
        unsafe_allow_html=True)
    st.write("")

    _, mid, _ = st.columns([1, 1.15, 1])
    with mid:
        with st.form("gate_form", clear_on_submit=True):
            pw = st.text_input("访问口令", type="password",
                               label_visibility="collapsed",
                               placeholder="请输入访问口令")
            submit = st.form_submit_button("进 入", use_container_width=True)

        now = time.time()
        lock_until = float(st.session_state.get("gate_lock") or 0)
        if submit:
            if now < lock_until:                      # 锁定中，连正确口令也不放行
                st.error(f"尝试次数过多，请 {int(lock_until - now) + 1} 秒后再试")
            elif hmac.compare_digest(pw.strip(), _gate_code()):
                st.session_state["authed"] = True
                st.session_state["gate_fails"] = 0
                st.session_state.pop("gate_lock", None)
                st.rerun()
            else:
                fails = int(st.session_state.get("gate_fails") or 0) + 1
                if fails >= _GATE_FAIL_MAX:
                    st.session_state["gate_fails"] = 0
                    st.session_state["gate_lock"] = now + _GATE_LOCK_SEC
                    st.error(f"口令连续输错 {_GATE_FAIL_MAX} 次，"
                             f"请 {_GATE_LOCK_SEC} 秒后再试")
                else:
                    st.session_state["gate_fails"] = fails
                    st.error(f"口令不对，请再试一次"
                             f"（还可尝试 {_GATE_FAIL_MAX - fails} 次）")
        elif now < lock_until:
            st.error(f"尝试次数过多，请 {int(lock_until - now) + 1} 秒后再试")

        st.markdown(
            "<div class='gate-rule'>本系统只依据《学生手册（2025 年修订）》原文作答</div>",
            unsafe_allow_html=True)


_GATE_OPEN_VALUES = ("off", "none", "0", "false", "open")


def _render_gate_unconfigured():
    """口令**没配好**时挡在这里，明说这是配置问题，而不是假装门禁不存在。"""
    st.markdown(_STYLE, unsafe_allow_html=True)
    st.markdown(_HERO, unsafe_allow_html=True)
    st.error(
        "**后台还没配访问口令，暂时不能开放使用**\n\n"
        "管理员请到 Streamlit 后台 **Settings → Secrets** 补上 `ACCESS_CODE`"
        "（TOML 格式：等号两边留空格、值用英文双引号），保存后会自动重新部署。\n\n"
        "在配好之前这里会一直挡着 —— 这是故意的：宁可挡住，也不能因为漏配就门户大开。")
    st.caption("本地调试若想有意不设门禁，把 ACCESS_CODE 设成 off —— "
               "与「漏配导致为空」区分开。")


def _gate_ok() -> bool:
    """门禁放行判定；未通过就把登录页渲染掉、结束本次脚本。

    三种情况（2026-10-15 改，起因见下）：
      · 口令已配            → 正常走门禁
      · 口令**留空**        → 视为「没配好」，**拒绝放行**（fail-safe）
      · 口令是 off/none/0…  → 有意关闭门禁（本地调试用）

    为什么「留空」不再等于「不设门禁」：线上 Secrets 没生效时 ACCESS_CODE 就是空的，
    按旧逻辑等于**门户大开** —— 手册内容随便拿、免费 GLM 额度随便烧，而且页面上
    一点异常都看不出来（学生看到的就是个正常可用的系统）。宁可挡住并说明原因，
    也绝不默认放行。本地 .env 里配了 SZPU666，所以日常使用不受影响。
    """
    code = _gate_code()
    if code.lower() in _GATE_OPEN_VALUES:
        return True
    if not code:
        _render_gate_unconfigured()
        return False
    if st.session_state.get("authed"):
        return True
    _render_gate()
    return False


_CITE_RE = re.compile(r"\[(E\d+)\]")


def format_answer(text: str) -> str:
    """把模型答复整理成「一句结论 + 分点」，**只插换行，一个字都不改**。

    为什么要在显示前做这一下（2026-10-15 用户反馈「答案不该是一大段」）：
    提示词里已经写明要分点，但免费小模型（GLM-4-Flash）对排版指令的服从度有限，
    仍常把要点全挤成一大段 —— 而它的**内容**是对的。所以在渲染这一步做纯排版
    整理：按句末标点断成几条，第一条留作结论，其余每条独立成行。
    因为只插换行、不动任何文字，**不影响引用校验**，日志里存的也仍是模型原话。
    已经自己分了行、或者本来就短（<80 字）的答复原样放行，尊重模型排好的版。
    """
    t = (text or "").strip()
    if not t or "\n" in t or len(t) < 80:
        return t
    parts = [s.strip() for s in re.split(r"(?<=[。！？])", t) if s.strip()]
    # 句末标点后紧跟的 [E1][E2] 会被切成分外的碎片，并回上一句，别让它单独成行
    merged = []
    for p in parts:
        if merged and not _CITE_RE.sub("", p).strip():
            merged[-1] += p
        else:
            merged.append(p)
    if len(merged) < 2:
        return t
    return merged[0] + "\n\n" + "\n".join("- " + p for p in merged[1:])


def linkify_citations(text: str, valid_eids) -> str:
    """把正文里的 [E1] 换成可点击角标，点击滚到对应证据卡片。"""
    valid = set(valid_eids)

    def repl(m):
        eid = m.group(1)
        if eid in valid:
            return (f'<a class="cite-badge" href="#ev-{eid}" '
                    f'title="跳到证据 {eid} 原文">[{eid}]</a>')
        return m.group(0)

    return _CITE_RE.sub(repl, text)


def friendly_refuse(reason: str):
    """把技术性的拒答原因翻译成用户看得懂的话。

    返回 (标题, 说明)。原则：面向用户的回答区只讲人话，
    "最高相关分低于阈值 0.03" 这类细节不出现（挪到「内部过程」面板）。
    """
    r = reason or ""
    if "校验未通过" in r:
        return ("抱歉，这个问题我暂时答不了",
                "手册里有相关信息，但我没法确认答案完全有依据。"
                "为了不给你一个不确定的结论，就先不作答了。")
    if "模型调用失败" in r or "未接入" in r:
        return ("服务暂时出了点小状况", "请稍等片刻，再试一次。")
    # 其余（检索无证据 / 拒答前置 / 模型判定证据不足）统一按「没找到」处理
    return ("手册里暂无相关内容",
            "这个问题在《学生手册（2025 年修订）》里没有找到可以依据的规定，"
            "我就不猜测了。你可以换个问法，或者问手册里的事项"
            "（学籍、考试、奖助学金、宿舍、违纪处分等）。")


# ==================================================================== 资源加载
QA_MODE_DEFAULT = "agent"      # 没配 QA_MODE 时走哪条 —— 主线是智能体链路


def _qa_mode() -> tuple:
    """返回 (通道, 是否靠默认值猜的)。

    为什么单独抽出来（2026-10-15 踩到）：`get_pipeline()` 里原来写的是
    `os.environ.get("QA_MODE") or "procedural"` —— 配置一旦没读到（.env 丢了、
    线上 Secrets 漏了这一行），就**悄悄**退回程序化通道：能答、答得也不差，
    只是不再调大模型。页面上看不出任何异常，最容易让人以为「智能体链路坏了」。
    按本项目一贯的规矩（配置缺失必须留痕），这里把「猜的」这件事记下来，
    由页面明说。读法是唯一定义，`get_pipeline` 与页头提示都调它。
    """
    raw = (os.environ.get("QA_MODE") or "").strip().lower()
    if raw in ("agent", "procedural"):
        return raw, False
    return QA_MODE_DEFAULT, True


@st.cache_resource(show_spinner=False)
def get_pipeline():
    """加载检索器 + 生成通道（全进程只做一次）。

    QA_MODE=agent（默认）走**智能体链路**：大模型规划 → 多路检索 → 生成 →
    自检回补 → 引用校验，每题都会调大模型（免费 GLM-4-Flash）。
    QA_MODE=procedural 走**纯程序化抽取式**：直接摘手册原文作答，不调大模型。

    检索的「问题侧编码」会在本机模型与在线接口之间自动选择（见
    `retrievers.HandbookRetriever._pick_backend`）：本机有 models/ 就用本机，
    没有（例如部署到 Streamlit Cloud）就调在线接口，不需要改配置。
    """
    load_secrets()
    load_env_file()
    retriever = HandbookRetriever(DATA_DIR, use_rerank=True, verbose=False)
    if _qa_mode()[0] != "agent":
        from procedural_qa import ProceduralPipeline, make_free_fallback
        # fallback 没配就返回 None，等于纯程序化；配了才在难题上走免费 API
        return ProceduralPipeline(retriever, top_k=6,
                                  fallback=make_free_fallback(), verbose=False)
    llm = build_llm()
    return AgentQAPipeline(retriever, llm=llm, verbose=False)


def _channel_label() -> str:
    """给页面用的中文通道名 —— 学生/管理员一眼能看出「刚才谁答的」。"""
    mode, guessed = _qa_mode()
    if mode != "agent":
        return "程序化直接作答（不调大模型）" + ("　⚠ QA_MODE 未配置，用的默认值"
                                        if guessed else "")
    backend = (os.environ.get("LLM_BACKEND") or "free").strip().lower()
    brain = {"free": "免费 GLM-4-Flash", "deepseek": "DeepSeek",
             "local": "本机模型"}.get(backend, backend)
    return ("智能体链路（规划 → 多路检索 → 生成 → 自检 → 引用校验）"
            f"，大脑 {brain}" + ("　⚠ QA_MODE 未配置，用的默认值" if guessed else ""))


def serialize(res) -> dict:
    """把 QAAnswer 压成可放进 session_state 的普通 dict。"""
    return {
        "answer": res.answer,
        "refused": bool(res.refused),
        "refuse_reason": res.refuse_reason or res.reason or "",
        "citations": list(res.citations or []),
        "evidences": [{
            "eid": e.eid, "cite": e.cite, "page": e.page,
            "score": e.score,
            "file_title": getattr(e, "file_title", ""),
            "text": e.text,
        } for e in res.evidences],
        "latency": res.latency,
        "plan": res.plan or {},
        "hops": res.hops,
        "n_evidence": res.n_evidence or len(res.evidences),
        "reflect_rounds": res.reflect_rounds,
        "reflect_log": res.reflect_log or [],
        "verify": res.verify or {},
    }


# ==================================================================== 渲染
def _render_process_proc(meta: dict, plan: dict):
    """程序化通道的过程面板（与智能体链路的口径完全不同，单独渲染）。"""
    subs = plan.get("sub_questions") or []
    used_fb = str(plan.get("mode") or "") == "procedural→llm"
    kind_cn = {"yesno_yes": "可以（肯定）", "yesno_no": "不可以（否定）",
               "yesno_rule": "需按流程办", "numeric": "数量/期限",
               "how": "办理流程", "who": "负责部门", "what": "情形列举",
               "consequence": "处理后果", "yesno_open": "有规定但需展开",
               "generic": "需展开", "refuse": "无相关规定"}.get(
                   plan.get("conclusion_kind") or "", "—")
    head = (f"内部过程 ｜ 作答通道：{'程序化 + 免费大模型兜底' if used_fb else '纯程序化'}"
            f" ｜ 检索 {meta['hops']} 次 ｜ 原文 {meta['n_evidence']} 条")

    with st.expander(head, expanded=False):
        c1, c2, c3 = st.columns(3)
        c1.metric("检索次数", meta["hops"])
        c2.metric("原文条数", meta["n_evidence"])
        c3.metric("耗时", f"{meta['latency']:.1f}s")

        st.markdown("**① 问题拆解**")
        if subs:
            st.write("判定为一次问了多件事，拆成以下几个子问题分别检索：")
            for i, s in enumerate(subs, 1):
                st.write(f"{i}. {s}")
        else:
            st.write("判定为单一问题，直接用原问题检索。")

        st.markdown("**② 作答通道**")
        if meta["refused"]:
            st.write("检索到的条款相关度不足，按「手册中无相关规定」处理，不给出答案。")
            st.caption(f"技术细节：{meta['refuse_reason']}")
        elif used_fb:
            st.write(f"程序化给出的结论句不够实在（成色：{kind_cn}），"
                     "于是把这批原文交给免费大模型重新归纳结论；"
                     "**原文条款与出处仍然是程序化逐字摘录的**，未改写。")
        else:
            st.write(f"程序化直接作答（结论成色：{kind_cn}），未调用任何大模型。")

        st.markdown("**③ 引用校验**")
        v = meta.get("verify") or {}
        if meta["refused"]:
            st.write("—")
        elif v.get("passed"):
            st.write("通过：答案里的每个 [Ex] 角标都能在原文里找到对应条款。")
        else:
            st.write("未通过：" + "；".join(v.get("reasons") or []))

        st.markdown("**④ 为什么答案里带原文**")
        st.write("数字、期限、比例全部逐字照抄手册，不做任何改写 —— "
                 "这是「零幻觉」的代价，读起来会像手册摘录。")


def render_process(meta: dict):
    """答案下方的「内部过程」面板。"""
    plan = meta.get("plan") or {}
    if str(plan.get("mode") or "").startswith("procedural"):
        _render_process_proc(meta, plan)
        return
    need_split = bool(plan.get("need_split"))
    subs = plan.get("sub_questions") or []
    if meta["refused"]:
        head = "内部过程 ｜ 已拒答（技术细节）"
    else:
        # 明写「智能体」：不然学生看不出这一答到底有没有经过大模型
        head = (f"内部过程 ｜ 作答通道：智能体"
                f" ｜ 拆解{'：是' if need_split else '：否'}"
                f" ｜ 检索 {meta['hops']} 次 ｜ 证据 {meta['n_evidence']} 条"
                f" ｜ 自检补检 {meta['reflect_rounds']} 轮")

    with st.expander(head, expanded=False):
        c1, c2, c3 = st.columns(3)
        c1.metric("检索次数", meta["hops"])
        c2.metric("证据条数", meta["n_evidence"])
        c3.metric("耗时", f"{meta['latency']:.1f}s")

        st.markdown("**① 问题拆解**")
        if need_split and subs:
            st.write("判定为复合/跨章题，拆成以下几个子问题：")
            for i, s in enumerate(subs, 1):
                st.write(f"{i}. {s}")
        else:
            st.write("判定为简单题，不拆解，直接用原问题检索。"
                     + (f"（{plan.get('reason')}）" if plan.get("reason") else ""))

        st.markdown("**② 检索与生成**")
        if meta["refused"]:
            st.write("未生成答案，已按「无可靠依据」处理。")
            st.caption(f"技术细节：{meta['refuse_reason']}")
        else:
            st.write(f"多路检索后合并去重，最终送 {meta['n_evidence']} 条证据给模型作答。")

        st.markdown("**③ 自检循环**")
        log = meta.get("reflect_log") or []
        if not log:
            st.write("未触发自检（或组件已关闭）。")
        for i, a in enumerate(log, 1):
            gaps = a.get("gaps") or []
            verdict = ("无缺口" if a.get("covered")
                       else ("非检索可解，停止" if a.get("hopeless")
                             else f"发现缺口：{'；'.join(gaps)}"))
            st.write(f"第 {i} 轮：{verdict}")

        st.markdown("**④ 引用校验**")
        v = meta.get("verify") or {}
        if meta["refused"]:
            st.write("—")
        elif v.get("passed"):
            st.success("通过：引用的条款真实存在，且支持答案里的结论。")
        else:
            st.error("未通过：存在无法被证据支持的断言（系统已据此降级）。")


def render_assistant(meta: dict):
    if meta["refused"]:
        title, tip = friendly_refuse(meta["refuse_reason"])
        st.warning(f"**{title}**\n\n{tip}")
    else:
        valid = {e["eid"] for e in meta["evidences"]}
        st.markdown(linkify_citations(format_answer(meta["answer"]) or "（空）", valid),
                    unsafe_allow_html=True)

    emap = {e["eid"]: e for e in meta["evidences"]}
    cits = meta["citations"] or []
    if cits and not meta["refused"]:
        st.markdown("**手册原文**（点答复里的 <span class='cite-badge'>[E2]</span> "
                    "角标可直接跳到对应条款）：",
                    unsafe_allow_html=True)
        cards = []
        for eid in cits:
            e = emap.get(eid)
            if not e:
                continue
            cite = _html.escape(str(e.get("cite") or ""))
            text = _html.escape(str(e.get("text") or ""))
            page = e.get("page")
            score = e.get("score") or 0.0
            # `open`：条款默认就是摊开的 —— 答复里只说结论，完整规定在这里看，
            # 点角标滚下来要能直接读全文，不用再点一次（2026-10-15 按用户反馈）。
            cards.append(
                f'<details class="ev" id="ev-{eid}" open>'
                f'<summary>[{eid}] {cite} ｜ PDF 第 {page} 页</summary>'
                f'<div class="ev-meta">相关度 {score:.4f}</div>'
                f'<div class="ev-text">{text}</div>'
                f'</details>')
        st.markdown("".join(cards), unsafe_allow_html=True)

    render_process(meta)
    st.caption(f"耗时 {meta['latency']:.1f} 秒")


# ==================================================================== 作答
def _loading_tip() -> str:
    """等待提示语跟着「问题侧编码」的后端走。

    本机跑模型时要等两个模型进显存（**实测 2026-10-15：模型构造 11.3s，
    整轮首次 search 31.6s，第二次 0.7s**），得提前打招呼，否则像卡死；
    线上是调在线接口（每题多两三秒），根本没有模型可加载 —— 照搬那句话会让
    学生白白等满。判据与 `retrievers._pick_backend` 完全一致：
    显式开关 > 本机模型在不在。
    """
    bk = (os.environ.get("EMBED_BACKEND") or "").strip().lower()
    if bk not in ("local", "api"):
        bk = "local" if (ROOT / "models" / "bge-m3" / "config.json").exists() else "api"
    return ("正在检索手册并作答…（启动后首次需加载模型，约 30 秒）" if bk == "local"
            else "正在检索手册并作答…")


def ask_and_render(prompt: str) -> bool:
    """统一作答入口：示例按钮与手动输入都走这里。

    提问频率限制（限流）已按需求临时下线 —— 这里现在是「来者即答」的直通路径。
    限流代码整体备份在 .trash_tmp/限流下线-20261015/，需要时拿回来接上即可。
    """
    sid = st.session_state.get("sid") or "anon"
    cid = st.session_state.get("client_id") or client_id()

    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    with st.chat_message("assistant"):
        with st.spinner(_loading_tip()):
            try:
                pipe = get_pipeline()
                pipe.enable_planner = OPT["planner"]
                pipe.enable_reflect = OPT["reflect"]
                pipe.enable_gate = OPT["gate"]
                pipe.top_k = OPT["top_k"]
                pipe.max_merged = OPT["max_merged"]
                res = pipe.ask(prompt)
                meta = serialize(res)
            except Exception as e:                      # noqa: BLE001
                st.error(f"出错了：{e}")
                st.session_state.messages.pop()          # 回滚这条提问
                st.stop()
        st.session_state.messages.append(
            {"role": "assistant", "content": meta["answer"], "meta": meta})

    # 留痕：后台异步写云库，失败也不影响用户（详见 src/oplog.py）
    # 两条：qa_log 存完整问答（只授 INSERT），qa_usage 只存「谁何时问了一次」
    # （公开读，供统计看板）。
    lg = get_logger()
    lg.log(sid, prompt, answer=meta["answer"],
           refused=meta["refused"], latency=meta["latency"],
           n_evidence=meta["n_evidence"], cited=meta["citations"])
    lg.usage(cid, sid)
    return True


# ==================================================================== 建议墙
@st.cache_data(ttl=45, show_spinner=False)
def _load_suggestions(limit: int = 60):
    """读留言墙。挂 45 秒缓存：刷新页面不必每次都打云库。"""
    return get_store().list_suggestions(limit)


def _nice_time(raw) -> str:
    s = str(raw or "")
    return s[:16].replace("T", " ") if s else ""


def render_suggestions():
    st.markdown(
        "<div class='sec'><span class='sec-bar'></span>"
        "<span class='sec-t'>我想说两句</span></div>",
        unsafe_allow_html=True)
    st.caption("对手册问答、对手册内容本身、对学校的服务，有什么想说的都可以写下来。"
               "留言会一直留在这面墙上，别人也看得到 —— 所以请好好说话。")

    if st.session_state.pop("sugg_done", None):
        st.success("提交成功，谢谢你的建议！")

    with st.form("sugg_form", clear_on_submit=True):
        txt = st.text_area("建议内容", max_chars=500, height=96,
                           label_visibility="collapsed",
                           placeholder="写点什么…（500 字以内）")
        if st.form_submit_button("提交建议"):
            ok, msg = get_store().add_suggestion(
                st.session_state.get("client_id") or "anon", txt)
            if ok:
                _load_suggestions.clear()          # 清缓存，让新留言立刻出现
                st.session_state["sugg_done"] = True
                st.rerun()
            else:
                st.warning(msg)

    rows = _load_suggestions()
    if not rows:
        st.caption("墙上还空着，你可以是第一个留言的人。")
        return
    st.markdown(f"<div class='wall-n'>墙上已有 {len(rows)} 条留言</div>",
                unsafe_allow_html=True)
    for r in rows:
        st.markdown(
            "<div class='wall-item'>"
            f"{_html.escape(str(r.get('content') or ''))}"
            f"<div class='wall-t'>{_nice_time(r.get('created_at'))}</div></div>",
            unsafe_allow_html=True)


# ==================================================================== 使用看板
def render_admin():
    st.markdown(
        "<div class='sec'><span class='sec-bar'></span>"
        "<span class='sec-t'>使用统计</span></div>",
        unsafe_allow_html=True)
    st.caption("这块只有老师能看 —— 学生点进来只会看到这个要密钥的框。")

    if not st.session_state.get("admin_ok"):
        _, mid, _ = st.columns([1, 1.15, 1])
        with mid:
            with st.form("admin_form", clear_on_submit=True):
                pw = st.text_input("查看密钥", type="password",
                                   label_visibility="collapsed",
                                   placeholder="请输入查看密钥")
                if st.form_submit_button("进 入", use_container_width=True):
                    if hmac.compare_digest(pw.strip(), admin_code()):
                        st.session_state["admin_ok"] = True
                        st.rerun()
                    else:
                        st.error("密钥不对")
        return

    data = get_store().stats()
    if not data:
        st.info("暂时取不到统计数据（可能是网络不通）。过一会儿刷新看看。")
        return

    cards = [("累计用过的人", data.get("clients_all"), "人"),
             ("累计提问", data.get("total"), "次"),
             ("今天用过的人", data.get("clients_today"), "人"),
             ("今天提问", data.get("today"), "次"),
             ("墙上留言", data.get("sug"), "条")]
    st.markdown(
        "<div class='stats'>" + "".join(
            f"<div class='stat'><div class='stat-v'>{int(v or 0)}"
            f"<span>{u}</span></div><div class='stat-k'>{k}</div></div>"
            for k, v, u in cards) + "</div>",
        unsafe_allow_html=True)

    daily = data.get("daily") or []
    if daily:
        mx = max([int(d.get("n") or 0) for d in daily] + [1])
        st.markdown(
            "<div class='panel-t'>最近 14 天</div><div class='bars'>" + "".join(
                f"<div class='bar-row'><span class='bar-d'>{_html.escape(str(d.get('day') or ''))}</span>"
                f"<span class='bar-tr'><span class='bar-b' style='width:"
                f"{max(int(d.get('n') or 0) / mx * 100, 2):.0f}%'></span></span>"
                f"<span class='bar-n'>{int(d.get('n') or 0)} 次 · {int(d.get('u') or 0)} 人</span>"
                "</div>" for d in daily) + "</div>",
            unsafe_allow_html=True)
    else:
        st.caption("最近 14 天还没有使用记录。")

    top = data.get("top_clients") or []
    if top:
        st.markdown(
            "<div class='panel-t'>谁用得最多</div>"
            "<table class='tbl'><thead><tr><th>匿名编号</th><th>提问数</th>"
            "<th>最后一次</th></tr></thead><tbody>" + "".join(
                f"<tr><td>{_html.escape(str(t.get('client_id') or ''))}</td>"
                f"<td>{int(t.get('n') or 0)}</td>"
                f"<td>{_html.escape(str(t.get('last') or ''))}</td></tr>"
                for t in top) + "</tbody></table>",
            unsafe_allow_html=True)
        st.caption("编号是浏览器随机生成的，只用来区分「是不是同一个人」，不含身份信息。")


# ==================================================================== 系统自检
def _config_rows() -> list:
    """关键配置到没到位。纯内存判断，不发网络请求，随便调。"""
    def has(k: str) -> bool:
        return bool((os.environ.get(k) or "").strip())

    bk = (os.environ.get("EMBED_BACKEND") or "").strip().lower()
    if bk not in ("local", "api"):
        bk = "local" if (ROOT / "models" / "bge-m3" / "config.json").exists() else "api"
    return [
        ("ACCESS_CODE", has("ACCESS_CODE"), "访问口令；没配＝不设门禁，谁都能进"),
        ("ADMIN_CODE", has("ADMIN_CODE"), "统计看板密钥；没配则打不开"),
        ("EMBED_API_KEY", has("EMBED_API_KEY"),
         f"在线编码密钥（当前编码后端 = {bk}）；后端是 api 时缺它直接没法提问"),
        ("CLOUD_ENDPOINT", has("CLOUD_ENDPOINT"), "云库地址；缺了留言墙读不到、日志不落库"),
        ("CLOUD_PUBLISHABLE_KEY", has("CLOUD_PUBLISHABLE_KEY"), "云库密钥；同上"),
        ("FREE_LLM_API_KEY", has("FREE_LLM_API_KEY"), "大脑（GLM-4-Flash）密钥；缺了没法作答"),
    ]


def _missing_config() -> list:
    return [name for name, ok, _ in _config_rows() if not ok]


def _render_config_warning():
    """配置不全时，把「缺哪几项」明明白白摆在页面上。

    为什么非要有它：Secrets 一没配好，程序不会崩，只会在真正用到的那一刻冒一个
    零散错误（提问说"缺少 EMBED_API_KEY"、留言说"没提交上"）—— 两件事看着毫不
    相干，排查方向全靠猜。这里一次说清，并指向「系统自检」页。
    """
    miss = _missing_config()
    if not miss:
        return
    st.warning(
        "**后台配置不完整**（缺 " + str(len(miss)) + " 项："
        + "、".join(f"`{m}`" for m in miss) + "）\n\n"
        "提问或留言可能因此失败。管理员请点页面底部「系统自检」查看详情。")


def _probe_embed() -> str:
    """真调一次在线编码接口，看通不通、多快。"""
    import time
    try:
        from embed_api import ApiEmbedder
        t0 = time.time()
        v = ApiEmbedder().encode(["连通性测试"], normalize_embeddings=True)
        return (f"通了 —— {int((time.time() - t0) * 1000)} ms，"
                f"返回形状 {getattr(v, 'shape', None)}")
    except Exception as e:                                       # noqa: BLE001
        return f"失败 —— {type(e).__name__}: {str(e)[:200]}"


def render_diag():
    """系统自检：配置状态 + 两条外部链路，只看「有没有配好、通不通」，不显示密钥内容。"""
    st.markdown(
        "<div class='sec'><span class='sec-bar'></span>"
        "<span class='sec-t'>系统自检</span></div>",
        unsafe_allow_html=True)
    st.caption("只报「配没配、通不通」，不显示任何密钥内容。"
               "本地看 .env，线上看 Streamlit 后台 Secrets。")

    st.markdown(f"**Secrets 读取情况**：{_html.escape(str(_SECRETS_NOTE))}")
    if _SECRETS_NOTE.startswith("没读到") and not (ROOT / ".env").exists():
        st.caption("线上出现「没读到」通常意味着 Secrets 没保存，或 TOML 写法有误 —— "
                   "注意分隔线、中文标点、全角引号都不是合法 TOML。")

    # 作答通道：这一条最容易被忽略 —— 程序化通道也能答得挺像样，只是不调大模型，
    # 光看答案是看不出来的，所以必须写在自检页最显眼处。
    _mode, _guessed = _qa_mode()
    st.write("")
    st.markdown(f"**作答通道**：{'智能体链路' if _mode == 'agent' else '程序化（不调大模型）'}"
                + ("　⚠ **QA_MODE 未配置**，用的是默认值 —— 线上请在 Secrets 里"
                   "补一行 `QA_MODE = \"agent\"`" if _guessed else ""))
    st.caption(_channel_label())

    st.write("")
    rows = _config_rows()
    ok_n = sum(1 for _, ok, _ in rows if ok)
    st.markdown(f"**关键配置**：{ok_n} / {len(rows)} 项已就位")
    for name, ok, hint in rows:
        c1, c2 = st.columns([2.0, 4.0])
        with c1:
            st.markdown(("✓ " if ok else "✗ ") + f"`{name}`")
        with c2:
            st.caption(("已配置" if ok else "**未配置**") + " —— " + hint)

    st.write("")
    st.markdown("**连通性测试**（点一下才发请求，平时不打扰）")
    c1, c2 = st.columns(2)
    with c1:
        if st.button("测云库", use_container_width=True, key="diag_cloud"):
            st.session_state["diag_cloud_r"] = get_store().probe()
    with c2:
        if st.button("测在线编码", use_container_width=True, key="diag_embed"):
            st.session_state["diag_embed_r"] = _probe_embed()

    r1 = st.session_state.get("diag_cloud_r")
    if r1:
        head = "云库通了" if r1.get("ok") else "云库不通"
        st.markdown(f"**{head}**（耗时 {r1.get('ms')} ms，HTTP {r1.get('status')}）")
        st.code(f"地址：{r1.get('endpoint')}\n"
                f"密钥：{'已配' if r1.get('has_key') else '未配'}\n"
                f"开关：{'开' if r1.get('enabled') else '关'}\n"
                f"问题：{r1.get('err') or '（无）'}", language=None)
        if not r1.get("ok"):
            st.caption("若地址与密钥都对、只是超时或连不上 —— 那多半是网络问题。"
                       "线上部署在境外，访问国内云库可能不稳，可换个时间再测。")

    r2 = st.session_state.get("diag_embed_r")
    if r2:
        st.markdown("**在线编码**：" + r2)


# ==================================================================== 主流程
def main():
    load_secrets()                        # 线上：先灌 Secrets（有则以它为准）
    load_env_file()                       # 本地：.env 兜底（已存在的变量不覆盖）
    if not _gate_ok():                    # 未通过口令 → 只显示登录页
        return

    st.markdown(_STYLE, unsafe_allow_html=True)
    st.markdown(_HERO, unsafe_allow_html=True)
    st.markdown(_CHIPS, unsafe_allow_html=True)

    client_id()                           # 匿名编号：写进地址栏，供使用统计认人

    # 看板视图：学生点开也只是一页要密钥的框，看不到任何数据
    raw_view = (st.session_state.get("view")
                or st.query_params.get("view") or "chat")
    if isinstance(raw_view, (list, tuple)):      # URL 里同名参数可能取到列表
        raw_view = raw_view[0] if raw_view else "chat"
    if str(raw_view) in ("stats", "diag"):
        if str(raw_view) == "stats":
            render_admin()
        else:
            render_diag()
        st.markdown(_FOOT, unsafe_allow_html=True)
        if st.button("← 回到问答"):
            st.session_state["view"] = "chat"
            st.query_params.pop("view", None)
            st.rerun()
        return

    _render_config_warning()      # 配置缺了什么，先说清，别让人去猜（2026-10-15 踩过）

    st.markdown(_CULTURE, unsafe_allow_html=True)

    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "sid" not in st.session_state:      # 随机会话号：只用于日志，不含身份信息
        st.session_state.sid = uuid.uuid4().hex[:12]

    # ---------------------------------------------------------------- 问答区
    st.markdown(
        "<div class='sec'><span class='sec-bar'></span>"
        "<span class='sec-t'>问答</span></div>",
        unsafe_allow_html=True)

    # 当前作答通道写在明面上（2026-10-15 用户问「为什么点示例是程序化作答、
    # 没调大模型」—— 根因是他看的是旧进程的页面；但通道本来就该一眼可见，
    # 否则「配置没读到」和「链路坏了」在学生眼里长得一模一样）。
    st.caption("作答通道：" + _channel_label())

    # 先渲染历史问答，再处理本次提问 —— 于是新问题的气泡永远接在旧答案下面，
    # 位置固定在「问答」这一栏里。
    # 为什么必须这么排（2026-10-15 用户反馈「点示例后答案跑到页面最下面」）：
    # Streamlit 从上往下执行脚本，作答函数写在脚本末尾，它的气泡就渲染在末尾
    # —— 页面上半截的问答区空着，学生看到的却是「答案掉到留言墙下面去了」。
    # 现在输入框与作答都收进问答区，示例按钮也走同一条通道（pending_q），
    # 无论点示例还是打字，问题和答案永远出现在同一个位置。
    for m in st.session_state.messages:
        with st.chat_message(m["role"]):
            if m["role"] == "user":
                st.markdown(m["content"])
            else:
                render_assistant(m["meta"])

    pending = st.session_state.pop("pending_q", None)
    if pending:
        ask_and_render(pending)        # 就地渲染：接在上一条答案下面
        st.rerun()                     # 再整页重画一次，历史与新答案同格式

    # 输入区放在示例问题**下面**（2026-10-15 用户指定：「问答的示例下要有个
    # 输入问问题的输入框」）。原来用 st.chat_input —— 它天生贴在视口最底部，
    # 学生一进来只看得见示例按钮，看不见输入框在哪，所以换成普通表单。
    # 示例问题：点击后与手动输入走同一条作答链路（真正调用模型）。
    clicked = None
    if not st.session_state.messages:
        st.markdown(
            "<div class='sec'><span class='sec-bar'></span>"
            "<span class='sec-t'>试试这样问</span></div>",
            unsafe_allow_html=True)
        samples = [
            "必修课考试不及格，能补考吗？补考成绩怎么算？",
            "国家奖学金一年多少钱？",
            "旷课多少学时会被处分、分别是什么处分？迟到早退算不算旷课？",
            "学校食堂几点关门？",
        ]
        cols = st.columns(2)
        for i, q in enumerate(samples):
            if cols[i % 2].button(q, key=f"s{i}", use_container_width=True):
                clicked = q

    if clicked:                      # 示例 → 也走 pending 通道，位置与打字一致
        st.session_state["pending_q"] = clicked
        st.rerun()

    st.markdown(
        "<div class='sec'><span class='sec-bar'></span>"
        "<span class='sec-t'>输入你的问题</span></div>",
        unsafe_allow_html=True)
    with st.form("ask_form", clear_on_submit=True):
        c_in, c_btn = st.columns([5.2, 1.0])
        with c_in:
            q_text = st.text_input(
                "你的问题", label_visibility="collapsed",
                placeholder="在这里输入问题，例如：奖学金怎么申请？")
        with c_btn:
            go = st.form_submit_button("提 问", use_container_width=True)
    if go and (q_text or "").strip():
        st.session_state["pending_q"] = q_text.strip()
        st.rerun()

    # 工具条：紧跟问答区。清空对话管的就是上面这块对话区，放在这儿最顺手；
    # 使用统计 / 系统自检 也一并收进来，页面底部只留留言墙与页脚。
    st.write("")
    b1, b2, b3 = st.columns(3)
    with b1:
        if st.button("清空对话", key="clear_chat", use_container_width=True):
            st.session_state.messages = []
            st.rerun()
    with b2:
        if st.button("使用统计", key="open_admin", use_container_width=True):
            st.session_state["view"] = "stats"
            st.query_params["view"] = "stats"
            st.rerun()
    with b3:
        if st.button("系统自检", key="open_diag", use_container_width=True):
            st.session_state["view"] = "diag"
            st.query_params["view"] = "diag"
            st.rerun()

    # 建议墙：常驻在页面下方，刷新、换台机器进来都还在（内容存在云库）
    # 分区小标题由 render_suggestions() 自己带，这里不再重复画一遍。
    st.write("")
    render_suggestions()

    st.markdown(_FOOT, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
