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
    """
    try:
        items = dict(st.secrets)
    except Exception:                    # 没有 secrets.toml / 非 Streamlit 环境
        return 0
    n = 0
    for k, v in items.items():
        if isinstance(v, (str, int, float)) and str(k) not in os.environ:
            os.environ[str(k)] = str(v)
            n += 1
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


def _gate_ok() -> bool:
    """门禁放行判定；未通过就直接把登录页渲染掉、结束本次脚本。"""
    if not _gate_code():                 # 没配口令 → 不设门禁
        return True
    if st.session_state.get("authed"):
        return True
    _render_gate()
    return False


_CITE_RE = re.compile(r"\[(E\d+)\]")


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
@st.cache_resource(show_spinner=False)
def get_pipeline():
    """加载检索器 + 生成通道（全进程只做一次）。

    QA_MODE=procedural（默认）走**纯程序化抽取式**：直接摘手册原文作答，
    不需要大模型、不花钱、不吃显卡，毫秒级出结果。
    QA_MODE=agent 走原来的智能体链路（规划+多跳+校验），需要用大模型，收费。

    检索的「问题侧编码」会在本机模型与在线接口之间自动选择（见
    `retrievers.HandbookRetriever._pick_backend`）：本机有 models/ 就用本机，
    没有（例如部署到 Streamlit Cloud）就调在线接口，不需要改配置。
    """
    load_secrets()
    load_env_file()
    retriever = HandbookRetriever(DATA_DIR, use_rerank=True, verbose=False)
    mode = (os.environ.get("QA_MODE") or "procedural").strip().lower()
    if mode != "agent":
        from procedural_qa import ProceduralPipeline, make_free_fallback
        # fallback 没配就返回 None，等于纯程序化；配了才在难题上走免费 API
        return ProceduralPipeline(retriever, top_k=6,
                                  fallback=make_free_fallback(), verbose=False)
    llm = build_llm()
    return AgentQAPipeline(retriever, llm=llm, verbose=False)


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
        head = (f"内部过程 ｜ 拆解{'：是' if need_split else '：否'}"
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
        st.markdown(linkify_citations(meta["answer"] or "（空）", valid),
                    unsafe_allow_html=True)

    emap = {e["eid"]: e for e in meta["evidences"]}
    cits = meta["citations"] or []
    if cits and not meta["refused"]:
        st.markdown("**依据（点答案里的 <span class='cite-badge'>[E1]</span> 角标可跳到对应条款）：**",
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
            cards.append(
                f'<details class="ev" id="ev-{eid}">'
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
    if str(raw_view) == "stats":
        render_admin()
        st.markdown(_FOOT, unsafe_allow_html=True)
        if st.button("← 回到问答"):
            st.session_state["view"] = "chat"
            st.query_params.pop("view", None)
            st.rerun()
        return

    st.markdown(_CULTURE, unsafe_allow_html=True)

    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "sid" not in st.session_state:      # 随机会话号：只用于日志，不含身份信息
        st.session_state.sid = uuid.uuid4().hex[:12]

    # 小工具条：状态提示 + 清空对话
    c1, c2 = st.columns([7, 1.2])
    with c1:
        st.markdown(
            "<div class='sec'><span class='sec-bar'></span>"
            "<span class='sec-t'>问答</span></div>",
            unsafe_allow_html=True)
    with c2:
        if st.button("清空对话", use_container_width=True):
            st.session_state.messages = []
            st.rerun()

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

    for m in st.session_state.messages:
        with st.chat_message(m["role"]):
            if m["role"] == "user":
                st.markdown(m["content"])
            else:
                render_assistant(m["meta"])

    # 建议墙：常驻在问答下方，刷新、换台机器进来都还在（内容存在云库）
    st.write("")
    render_suggestions()

    # 角落里的小入口：学生点开也只是一页「要密钥」的提示
    st.write("")
    _, c2 = st.columns([6.4, 1.6])
    with c2:
        if st.button("使用统计", key="open_admin", use_container_width=True):
            st.session_state["view"] = "stats"
            st.query_params["view"] = "stats"
            st.rerun()

    st.markdown(_FOOT, unsafe_allow_html=True)

    prompt = st.chat_input("输入你的问题，例如：奖学金怎么申请？") or clicked
    if prompt:
        if ask_and_render(prompt):
            st.rerun()


if __name__ == "__main__":
    main()
