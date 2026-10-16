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

# 先把内存里的模块与磁盘对齐，**再** import —— 于是「改完后台代码，刷新一下网页」
# 就生效，不必去找那个启动服务的黑窗口。为什么必须写在这两行前面：Python 的
# import 是幂等的，模块已加载就直接从内存取，不会重新读文件；只有先对齐，
# 下面这句 `from qa_chain import ...` 才拿得到新对象。详见 src/hot_reload.py。
import hot_reload                                       # noqa: E402
_RELOADED = hot_reload.refresh()        # 本次真的换掉了哪几个模块（平时是空的）

from qa_chain import build_llm, load_env_file          # noqa: E402
from retrievers import HandbookRetriever                # noqa: E402
from agent_qa import AgentQAPipeline                    # noqa: E402
from oplog import get_logger                           # noqa: E402
from cloudstore import get_store                       # noqa: E402
from memory import SessionMemory, new_mem_id, summarize_older   # noqa: E402

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


# ==================================================================== 会话记忆
# 只做**临时**记忆：让追问能接着上文答（「那它算不算迟到早退」这种）。
#
# 记忆**只存在这一页的内存里**（`st.session_state`），不往服务器写任何东西 ——
# 所以刷新页面、关掉标签页，它自己就没了，不需要谁去清理。
#
# 为什么刻意不落云库（2026-10-15 用户要求）：他要的是「在页面上问答期间记得住，
# 一关掉或刷新就清空」。而浏览器关标签页时服务器**收不到任何通知**，写进云库
# 反而会留下没有主的残留行，只能靠「闲置多久」去猜着清 —— 那是一套多余的近似。
# 干脆不写：关闭即清、刷新即清，天然成立，也不用再猜。
#
# 三层结构（最近几轮原文 / 更早的压成摘要 / 追问改写）见 src/memory.py 的说明。


def get_memory() -> SessionMemory:
    """拿本次会话的记忆对象；这一页还没有，就新建一个空的。

    「一次会话」＝这一页从打开到刷新/关闭。刷新或关掉标签页，Streamlit 会把
    `st.session_state` 整个丢掉，记忆随之消失 —— 这正是不做持久记忆的意思。
    同一页里接着提问**不会**走到新建这一步（那时 mem 还在），所以
    「那要多久呢？」这种追问照样接得上：记忆只活在「这一页、这一轮」里。
    """
    mem = st.session_state.get("mem")
    if isinstance(mem, SessionMemory):
        return mem
    # 后台代码热重载后（见文件开头 hot_reload），SessionMemory 这个类换成了新对象，
    # session_state 里留着的是旧类的实例 —— isinstance 认不出它，但里面的对话还在。
    # 搬过来，别丢。
    if mem is not None and hasattr(mem, "to_dict"):
        try:
            mem = SessionMemory.from_dict(mem.to_dict())
            st.session_state["mem"] = mem
            return mem
        except Exception:                                    # noqa: BLE001
            pass
    cid = st.session_state.get("client_id") or client_id()
    mem = SessionMemory(mem_id=new_mem_id(), client_id=cid)
    st.session_state["mem"] = mem
    return mem


def admin_code() -> str:
    """看板密钥（.env 的 ADMIN_CODE，默认 9925）—— 只在服务端比对，不外发。"""
    return (os.environ.get("ADMIN_CODE") or "9925").strip()


# ==================================================================== 访问门禁
# 口令写在 .env 的 ACCESS_CODE（默认 SZPU666）。留空 = 不设门禁（本地自测用）。
_GATE_FAIL_MAX = 5        # 连续输错几次开始短时锁定
_GATE_LOCK_SEC = 60       # 锁定秒数

# 「系统自检」页（配置状态 + 云库/编码接口连通性）—— 2026-10-16 用户要求
# **不在网站上展示**：它是给自己排查用的，学生看到只会是一堆技术名词。
# 代码与页面内容原样保留，改成 True 即可完整恢复（工具条那个入口按钮也留了
# 注释版）。关掉时连地址栏 ?view=diag 这条后门一起关，免得「撤了等于没撤」。
SHOW_DIAG = False


def _gate_code() -> str:
    return (os.environ.get("ACCESS_CODE") or "").strip()


def _same_secret(got: str, want: str) -> bool:
    """口令比对，**输什么都只是「不对」，绝不崩页面**。

    为什么不能直接用 `hmac.compare_digest(a, b)`：
    Python 3 里它只接受 **ASCII** 的 str（或 bytes）。学生切到中文输入法
    随手打一个汉字进去，它就抛
    `TypeError: comparing strings with non-ASCII characters is not supported`
    —— 整个页面直接报错、谁也进不来（2026-10-15 夜目检时撞到）。

    所以先统一编码成 utf-8 的 bytes 再比：比较强度一样（都是定长时间比较），
    但汉字、emoji、随便什么字符都只会让结果变成 False。
    """
    if not want:
        return False
    try:
        return hmac.compare_digest(str(got).strip().encode("utf-8"),
                                   want.strip().encode("utf-8"))
    except Exception:
        return False


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
            elif _same_secret(pw, _gate_code()):
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


def friendly_refuse(reason: str, question: str = ""):
    """把技术性的拒答原因翻译成用户看得懂的话。

    返回 (标题, 说明)。原则：面向用户的回答区只讲人话，
    "最高相关分低于阈值 0.03" 这类细节不出现（挪到「内部过程」面板）。

    2026-10-15 用户要求统一口径：凡是「答不上来」，一律直说
    「本手册中没有对应内容」，**不再出现**「我暂时答不了 / 拿不准 /
    为了不给不确定的结论就不作答」这类模棱两可、听着像推脱的话。
    所有「查不到」的原因（检索无证据 / 拒答前置 / 模型自称证据不足 /
    引用校验未通过）共用这一套话术；真正的技术原因仍然完整写在
    「内部过程」面板里，排查不受影响。

    2026-10-16 补：给了 `question` 且问句里有口语词（体测/挂科/翘课…）时，
    额外告诉学生手册里的写法 —— 干巴巴一句「查不到」最让人束手无策，
    而这类查不到**多半就是词没对上**（见 src/term_alias.py）。

    ⚠ 2026-10-16 再补（当日线上事故）：**必须先把「技术故障」和「手册里没有」
    分开**。原来只认「模型调用失败 / 未接入」两个词，可是「大模型或检索接口
    不可用」这条路写的是「大模型不可用（…），检索也失败（…）」，一个都没命中
    → 于是接口挂掉的时候，页面用一句「本手册中没有对应内容」把**所有题**都说成
    「手册里没规定」，还贴心地建议学生「换个说法再问」—— 学生照着换一百个说法
    也没用，管理端则完全看不出后台已经瘫了。技术故障一律走上面那条分支，
    并且**不加**换说法提示（那不是措辞问题）。
    """
    r = reason or ""
    if ("模型调用失败" in r or "未接入" in r or "大模型不可用" in r
            or "检索也失败" in r or "超时" in r or "timeout" in r.lower()
            or "ApiError" in r or "HTTP 4" in r or "HTTP 5" in r
            or "ConnectionError" in r or "请求失败" in r):
        return ("服务暂时出了点小状况",
                "这次没能连上后台（多半是接口额度用完、被限流，或网络抖了一下），"
                "不是手册里没有规定。请过一两分钟再问一次；"
                "要是一直这样，请找管理员看一下后台配置（技术细节在下方"
                "「内部过程」里）。")
    tip = ("这个问题在《学生手册（2025 年修订）》里没有找到可以依据的规定。"
           "你可以换个问法，或者改问手册里的事项"
           "（学籍、考试、奖助学金、宿舍、违纪处分等）。")
    if question:
        try:
            from term_alias import hint_for
            extra = hint_for(question)
        except Exception:                                   # noqa: BLE001
            extra = ""                                      # 别名表坏了不影响拒答话术
        if extra:
            tip = tip + "\n\n" + extra
    return ("本手册中没有对应内容", tip)


# ==================================================================== 资源加载
@st.cache_resource(show_spinner=False)
def get_pipeline():
    """加载检索器 + 智能体链路（全进程只做一次）。

    作答通道只有一条：**智能体链路**
    —— 大模型规划 → 多路检索 → 生成 → 自检回补 → 引用校验，
    每题都调大模型（免费 GLM-4-Flash）。

    （原来还有一条「程序化抽取式」通道，用 QA_MODE 环境变量切换。2026-10-15
     按需求把「可切换」这层壳去掉了：QA_MODE 开关、页面上的通道小字、
     程序化过程面板都已删除。但程序化实现本身**保留**，退居为模型失效时的
     **兜底安全网** —— 大模型炸或调用失败时，`AgentQAPipeline._rescue()` 会
     让 `src/procedural_qa.py` 照抄手册原文作答，并在页面上明说「本次是兜底」。
     相关测试见 `src/tests/test_fallback.py`。）

    检索的「问题侧编码」会在本机模型与在线接口之间自动选择（见
    `retrievers.HandbookRetriever._pick_backend`）：本机有 models/ 就用本机，
    没有（例如部署到 Streamlit Cloud）就调在线接口，不需要改配置。
    """
    load_secrets()
    load_env_file()
    retriever = HandbookRetriever(DATA_DIR, use_rerank=True, verbose=False)
    llm = build_llm()
    return AgentQAPipeline(retriever, llm=llm, verbose=False)


if _RELOADED:
    # 刚才发生过热重载（见文件开头 hot_reload.refresh()）。缓存里那个流水线对象
    # 可能是用旧类造的，攥着不放就又回到「新旧混用」——清掉，稍后按新代码重建。
    get_pipeline.clear()


def renumber_citations(meta: dict) -> dict:
    """把答复里的角标重编成 E1、E2、E3… 连续编号（2026-10-15 用户要求）。

    背景：链路给证据编号是按「多路检索的合并顺序」E1..En，模型却只挑其中
    几条来引用，于是学生看到的是 [E2]、[E6]、[E21] 这种跳号 —— 既看不出
    一共引了几条依据，也看不出先后。

    这里按**答复里首次出现的先后**重新编号：引到的第一条条款就是 E1，
    第二条 E2，依此类推；下面「手册原文」的卡片同步换成同一套编号，
    并按 E1、E2、E3… 排好，点角标照样跳得到。

    只动编号，不改一个字；没被引用的证据排到被引用的后面（不显示成卡片）。
    本来就已经是 E1..Ek 连续时原样返回，不做无谓改动。
    """
    if meta.get("refused"):
        return meta
    evs = meta.get("evidences") or []
    answer = meta.get("answer") or ""
    if not evs or not answer:
        return meta

    known = {e["eid"] for e in evs}
    cited: list = []
    for m in _CITE_RE.finditer(answer):
        eid = m.group(1)
        if eid in known and eid not in cited:
            cited.append(eid)
    if not cited:
        return meta

    order = cited + [e["eid"] for e in evs if e["eid"] not in cited]
    mapping = {old: f"E{i}" for i, old in enumerate(order, 1)}
    if all(old == new for old, new in mapping.items()):
        return meta                                   # 本来就连号，不动

    meta["answer"] = _CITE_RE.sub(
        lambda m: f"[{mapping.get(m.group(1), m.group(1))}]", answer)
    # 卡片按新编号升序排，保证页面上就是 E1、E2、E3…
    meta["citations"] = sorted(
        {mapping[c] for c in (meta.get("citations") or []) if c in mapping},
        key=lambda s: int(s[1:]))
    emap = {e["eid"]: e for e in evs}
    new_evs = []
    for old in order:
        e = dict(emap[old])
        e["eid"] = mapping[old]
        new_evs.append(e)
    meta["evidences"] = new_evs
    return meta


def serialize(res) -> dict:
    """把 QAAnswer 压成可放进 session_state 的普通 dict。"""
    return renumber_citations({
        "answer": res.answer,
        "question": res.question,
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
        "badge_backfill": bool(getattr(res, "badge_backfill", False)),
    })


# ==================================================================== 渲染
def render_process(meta: dict):
    """答案下方的「内部过程」面板。"""
    plan = meta.get("plan") or {}
    need_split = bool(plan.get("need_split"))
    subs = plan.get("sub_questions") or []
    if meta["refused"]:
        head = "内部过程 ｜ 已拒答（技术细节）"
    elif plan.get("fallback") == "procedural":
        head = (f"内部过程 ｜ **程序化兜底**（大模型不可用）"
                f" ｜ 检索 {meta['hops']} 次 ｜ 证据 {meta['n_evidence']} 条")
    else:
        head = (f"内部过程 ｜ 拆解{'：是' if need_split else '：否'}"
                f" ｜ 检索 {meta['hops']} 次 ｜ 证据 {meta['n_evidence']} 条"
                f" ｜ 自检补检 {meta['reflect_rounds']} 轮")

    with st.expander(head, expanded=False):
        if plan.get("fallback") == "procedural":
            st.write("这条答复**不是大模型给的** —— 智能体链路在某一环出错了"
                     "（规划 / 生成 / 自检 / 引用校验都依赖大模型），"
                     "按「不能让学生白问一场」的原则改由程序化抽取式照抄手册原文。"
                     "原文与出处都是逐字摘录，未作改写，不会编造。")
            st.caption("技术细节：" + str(plan.get("fallback_error") or ""))
            st.write("")

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

        # 多轮记忆留痕（2026-10-15）：带上文了没有、检索实际用的是哪句，
        # 一眼能看见 —— 追问答不好时，先看这里就知道是「没接上文」还是别的毛病。
        if meta.get("history_used"):
            st.caption(f"多轮记忆：本次带了上文 {meta.get('history_n')} 条作背景"
                       f"（上文只是背景，不是证据）；追问里的「它／这个」按上文理解。")
            if plan.get("standalone"):
                st.caption(f"检索实际用的完整问句：{plan['standalone']}")

        # 口语词换手册词留痕（2026-10-16）：学生说「体测」手册写「体质测试」，
        # 字面对不上就搜不到，所以系统会**额外走几路换过词的问句**。这里把它
        # 显示出来 —— 口语题答得好不好，先看这几路有没有生成。
        alias_q = plan.get("alias_queries") or []
        if alias_q:
            st.caption(f"口语词换手册词：问句里有日常说法在手册里对不上号，"
                       f"已额外按手册用词搜了 {len(alias_q)} 路 —— "
                       + "；".join(alias_q))

        st.markdown("**② 检索与生成**")
        deg = plan.get("degraded_retrieval") or []
        if deg:
            st.warning("检索降级（本应是「向量 + 关键词」双路，实际只有一路）："
                       + "；".join(str(x) for x in deg))
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
        if meta.get("badge_backfill"):
            # 不留痕不行（2026-10-15）：这条不是「答得对不对」的问题，
            # 而是「答案的文字被动过没有」—— 动过就得说清楚。
            st.caption("角标回填：模型把出处写进了 citations 字段，正文却没打 [Ex]；"
                       "系统按**已申报的**编号补上角标（原文一字未改），再重跑校验。")
        if meta["refused"]:
            st.write("—")
        elif v.get("passed"):
            st.success("通过：引用的条款真实存在，且支持答案里的结论。")
        else:
            st.error("未通过：存在无法被证据支持的断言（系统已据此降级）。")


def render_assistant(meta: dict):
    # 兜底作答要把话说明白（2026-10-15）：大模型挂了才走程序化照抄，
    # 页面上必须讲清楚这不是模型答的，否则就是「静默降级」。
    plan = meta.get("plan") or {}
    if plan.get("fallback") == "procedural" and not meta["refused"]:
        st.info("**大模型这次没答上来**（多半是免费额度被限流或网络抖动），"
                "已自动改用**程序化兜底**：下面是把手册里最相关的原文**照抄**给你，"
                "一字未改、出处带页码，所以读起来会像手册摘录。"
                "稍等一两分钟再问一次，就能恢复模型的正常作答。")

    # 检索降级也要说出来（2026-10-16 线上事故）：向量检索与重排走的是外部
    # 接口，接口一挂（例如额度用完），原来整道题会直接失败；现在改为只用
    # 关键词匹配继续答。**这是质量打了折扣，不是失败** —— 隐性降质比明着
    # 报错更坏：学生会以为「手册里就是这么规定的」。
    deg = plan.get("degraded_retrieval") or []
    if deg:
        st.warning("**本次检索打了折扣**：智能匹配那一环没能连上，"
                   "只用了关键词匹配。答案仍然逐字来自手册原文，"
                   "但可能漏掉「换了种说法写」的条款 —— 换个问法再问一次，"
                   "通常就能找到了。")
        with st.expander("技术细节（给管理员）", expanded=False):
            for d in deg:
                st.caption("· " + str(d))

    if meta["refused"]:
        title, tip = friendly_refuse(meta["refuse_reason"],
                                     question=meta.get("question") or "")
        st.warning(f"**{title}**\n\n{tip}")
    else:
        valid = {e["eid"] for e in meta["evidences"]}
        st.markdown(linkify_citations(format_answer(meta["answer"]) or "（空）", valid),
                    unsafe_allow_html=True)

    emap = {e["eid"]: e for e in meta["evidences"]}
    cits = meta["citations"] or []
    if cits and not meta["refused"]:
        st.markdown("**手册原文**（点答复里的 "
                    f"<span class='cite-badge'>[{cits[0]}]</span> "
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


def _remember(mem: SessionMemory, prompt: str, meta: dict) -> None:
    """把这一轮写进**这一页的**会话记忆：超预算就先把最旧几轮压成摘要。

    只写内存（`st.session_state`），不落云库 —— 所以刷新/关掉页面它就跟着没了，
    不需要任何清理动作（2026-10-15 用户要求）。
    记忆是**加分项** —— 任何一步失败都只影响「追问顺不顺」，不影响本次作答，
    所以这里一律吞掉异常、不打扰学生（也不静默改结果，只是记忆少一点）。
    """
    try:
        mem.add_user(prompt)
        if meta.get("refused"):
            # 拒答也记一笔，但只留一句短标记 —— 免得把「本手册中没有对应内容」
            # 这种话当成上文带进下一轮，反而干扰模型对下一句的理解。
            mem.add_assistant("（本手册中没有对应内容）")
        else:
            mem.add_assistant(meta.get("answer") or "")
        if mem.over_budget():
            # 只在超预算时才多花一次模型调用；模型不可用则一条不删（宁可超预算）
            summarize_older(mem, getattr(get_pipeline(), "llm", None))
    except Exception:                                       # noqa: BLE001
        pass


def ask_and_render(prompt: str) -> bool:
    """统一作答入口：示例按钮与手动输入都走这里。

    提问频率限制（限流）已按需求临时下线 —— 这里现在是「来者即答」的直通路径。
    限流代码整体备份在 .trash_tmp/限流下线-20261015/，需要时拿回来接上即可。

    多轮记忆（2026-10-15 新增）：本次提问**之前**的对话会作为背景一起送给模型，
    追问里的「它／这个」才接得上；做法与原理见 src/memory.py。
    """
    sid = st.session_state.get("sid") or "anon"
    cid = st.session_state.get("client_id") or client_id()
    mem = get_memory()
    hist = mem.history_block()                    # 本次提问之前的上文（可能为空）
    prev_q = mem.previous_user_question(prompt)   # 上一轮那句原文：追问检索的兜底

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
                res = pipe.ask(prompt, history_block=hist, prev_user=prev_q)
                meta = serialize(res)
                meta["history_used"] = bool(hist)
                meta["history_n"] = len(mem.turns)
            except Exception as e:                      # noqa: BLE001
                st.error(f"出错了：{e}")
                st.session_state.messages.pop()          # 回滚这条提问
                st.stop()
        st.session_state.messages.append(
            {"role": "assistant", "content": meta["answer"], "meta": meta})

    _remember(mem, prompt, meta)          # 本轮进记忆（失败不影响上面的作答）

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
@st.cache_data(ttl=180, show_spinner=False)
def _load_suggestions(limit: int = 60):
    """读留言墙。挂 3 分钟缓存。

    缓存从 45 秒放长到 3 分钟：留言墙不是关键功能，却排在页面**最后** ——
    它一卡，整页就卡着「正在运行」。原来 45 秒一过期，每个访客一进来都得
    重打一次云库（线上访问国内云库本就慢，实测本机也要 300ms+）。

    **返回 None ＝ 读不到**（云库没配 / 超时），**[] ＝ 真的还没有留言**。
    这两者在页面上必须说不同的话，见 render_suggestions。
    """
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
    if rows is None:
        # 「读不到」和「墙上没留言」必须分开说。以前两者被 `or []` 抹成同一个，
        # 于是云库一超时就谎报「墙上还空着」—— 学生以为之前的留言被删了
        # （2026-10-15 用户报「一开始没显示之前的留言」，查出来的就是这里）。
        st.warning("留言墙暂时读不到（云库连不上或超时）。不影响你提问，"
                   "留言照常能提交。")
        if st.button("重新加载留言", key="sugg_retry"):
            _load_suggestions.clear()
            st.rerun()
        return
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
                    if _same_secret(pw, admin_code()):
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


# 关键接口探针（判据与说明都在 src/hot_reload.py，那边一处维护）。
_STALE_PROBES = hot_reload.PROBES


def _stale_modules() -> list:
    """内存里的 src/ 模块是不是「旧版」——文件改了，但页面还在按旧代码跑。

    平时这条不该有输出：页面每次加载都会先自动对齐一次模块（见文件开头
    `hot_reload.refresh()`）。它冒出来，说明自动对齐也没辙了 —— 页面会顺势
    提示「重启一下」。判据见 src/hot_reload.py。
    """
    return hot_reload.stale_modules()


def _render_stale_warning():
    """自动对齐之后仍对不上时，才把「重启一下」明说出来，别让人对着一句英文猜。"""
    bad = _stale_modules()
    if not bad:
        return
    st.error(
        "**后台代码已更新，但跑着的网页还是旧的**\n\n"
        + "\n".join(f"- {b}" for b in bad) + "\n\n"
        "页面已经试过自动换新（见 src/hot_reload.py），还是没对上。"
        "把启动网页时那个黑色窗口关掉，重新双击桌面的「启动网页问答.bat」。")


def _render_config_warning():
    """配置不全时，把「缺哪几项」明明白白摆在页面上。

    为什么非要有它：Secrets 一没配好，程序不会崩，只会在真正用到的那一刻冒一个
    零散错误（提问说"缺少 EMBED_API_KEY"、留言说"没提交上"）—— 两件事看着毫不
    相干，排查方向全靠猜。这里一次说清。

    ⚠ 2026-10-16：原来这里让管理员「点页面底部『系统自检』查看详情」，但那个
    入口已按用户要求从页面上撤掉了 —— 文案必须同步改，否则是把人指到一个
    不存在的地方去。现在改为让管理员看启动网页的黑窗口（诊断详情都打印在那里）。
    """
    miss = _missing_config()
    if not miss:
        return
    st.warning(
        "**后台配置不完整**（缺 " + str(len(miss)) + " 项："
        + "、".join(f"`{m}`" for m in miss) + "）\n\n"
        "提问或留言可能因此失败。管理员请补齐配置后重启服务，"
        "具体缺什么看启动网页时那个黑窗口里的输出。")


def _render_degraded_warning():
    """检索降级了没有？降了就在页面上明说，别等学生一道道题去撞。

    降级记录挂在检索器实例上（进程级），所以**上一次提问踩到的坑，这次
    一进页面就能看见** —— 不用再问一道题才知道后台的编码接口已经挂了。
    判据见 `retrievers._mark_degraded`：向量检索 / 重排走外部接口，
    接口不可用（额度、限流、断网）时只剩关键词路。
    """
    try:
        deg = list(getattr(get_pipeline().retriever, "degraded", []) or [])
    except Exception:                                       # noqa: BLE001
        return                                              # 取不到就不说，别因此报错
    if not deg:
        return
    st.warning(
        "**检索正在降级运行**（本应是「向量 + 关键词」双路，现在只剩关键词）：\n\n"
        + "\n".join(f"- {d}" for d in deg)
        + "\n\n答案仍取自手册原文，但「换了说法写」的条款容易漏。"
        "要恢复完整检索：本地跑就在 .env 里配 `EMBED_BACKEND=\"local\"`；"
        "线上跑就检查在线编码密钥与额度（Streamlit 后台 Secrets 里的 "
        "`EMBED_API_KEY`）。")


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

    st.write("")
    st.markdown("**代码版本**（后台代码 vs 跑着的网页进程）")
    _stale = _stale_modules()
    if _stale:
        st.markdown("✗ **对不上** —— 网页进程还是旧代码，需要重启")
        for b in _stale:
            st.caption("· " + b)
        st.caption("正常情况不会看到这一条：页面每次加载都会自动把旧模块换成新的"
                   "（见 src/hot_reload.py），刷新一下网页就生效。"
                   "若它反复出现，就把启动网页的那个黑色窗口关掉，"
                   "重新双击「启动网页问答.bat」。")
    else:
        st.caption("✓ 一致（内存里的模块与磁盘上的代码对得上）。")
    if _RELOADED:
        st.caption(f"本次加载时自动换掉了 {len(_RELOADED)} 个旧模块："
                   + "、".join(_RELOADED) + " —— 这就是「改完代码刷新页面即生效」。")
    elif hot_reload.HISTORY:
        _last = hot_reload.HISTORY[-1]
        st.caption("这个进程里自动重新加载过 "
                   + str(len(hot_reload.HISTORY)) + " 次（最近一次换了 "
                   + str(len(_last.get("modules") or [])) + " 个模块）。")

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
    st.markdown("**会话记忆**（只在这一页里活着）")
    mem = st.session_state.get("mem")
    if isinstance(mem, SessionMemory):
        c1, c2, c3 = st.columns(3)
        c1.metric("已记轮次", len(mem.turns))
        c2.metric("摘要字数", len(mem.summary))
        c3.metric("历史块字数", len(mem.history_block()))
        st.caption("记忆只活在「这一页、这一轮」里：**刷新页面 / 关掉标签页就没了**，"
                   "同一页里接着追问才记得住；点「清空对话」可以立刻清掉。"
                   "它**不往服务器写、也不做持久保存** —— 没有定时清理这类动作，"
                   "也不需要。")
    else:
        st.caption("本次会话还没有记忆（还没提问过）。")

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
    get_memory()                          # 会话记忆：只活在这一次会话里，刷新/关闭即没

    # 看板视图：学生点开也只是一页要密钥的框，看不到任何数据
    raw_view = (st.session_state.get("view")
                or st.query_params.get("view") or "chat")
    if isinstance(raw_view, (list, tuple)):      # URL 里同名参数可能取到列表
        raw_view = raw_view[0] if raw_view else "chat"
    # 「系统自检」已在页面上撤掉（2026-10-16 用户要求），连地址栏 ?view=diag
    # 这条后门也一并关掉 —— 否则它照样能被访问到，等于没撤干净。
    # 代码、页面内容、诊断逻辑**一个字都没删**：把下面 SHOW_DIAG 改成 True
    # 即可原样恢复（工具条那个按钮也留了注释版，见 main() 末尾）。
    if not SHOW_DIAG and str(raw_view) == "diag":
        raw_view = "chat"
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
    _render_degraded_warning()    # 检索降级了没有，一进页面就说（2026-10-16 线上踩过）
    _render_stale_warning()       # 代码更新了却没重启进程 → 直接说清（2026-10-15 踩过）

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

    # 顺手告诉学生「可以接着追问」（2026-10-15 加多轮记忆后才有这句话）。
    # 不写这句，学生多半以为一次只能问一个独立问题，追问能力就白做了。
    # 「等它答完再问」不是客套：Streamlit 在脚本运行中收到新交互会**打断当前
    # 运行** —— 上一题答到一半时提交下一题，上一题会被取消、也不会进记忆
    # （2026-10-15 目检时踩到，一度误判成「记忆没生效」）。
    st.caption("可以接着追问 —— 比如问完「旷课怎么处分」，再问「那要多久呢？」，"
               "系统记得住你这次会话问过什么。等它答完再问下一句就好。")

    # 工具条：紧跟问答区。清空对话管的就是上面这块对话区，放在这儿最顺手。
    #
    # 「系统自检」按钮**已从页面上撤掉**（2026-10-16 用户要求）：
    #   · 它是给自己排查用的（看配置全不全、云库/编码接口通不通），
    #     压根不是给学生看的 —— 学生点进去只会看到一堆技术名词。
    #   · `render_diag()` 与 `?view=diag` 的渲染路径**代码全部保留**，
    #     自检内容一个字没删；要临时用，把下面这行注释恢复即可，
    #     或者直接访问 `?view=diag`（main() 里那条分支也保留着）。
    st.write("")
    b1, b2 = st.columns(2)
    with b1:
        if st.button("清空对话", key="clear_chat", use_container_width=True):
            st.session_state.messages = []
            mem = get_memory()
            mem.reset()               # 记忆也清掉，否则追问还会用到已经「忘掉」的上文
            st.session_state["mem"] = mem
            st.rerun()
    with b2:
        if st.button("使用统计", key="open_admin", use_container_width=True):
            st.session_state["view"] = "stats"
            st.query_params["view"] = "stats"
            st.rerun()
    # 下面这段是原「系统自检」入口，保留备查（需要时取消注释即可恢复）：
    # with b3:
    #     if st.button("系统自检", key="open_diag", use_container_width=True):
    #         st.session_state["view"] = "diag"
    #         st.query_params["view"] = "diag"
    #         st.rerun()

    # 建议墙：常驻在页面下方，刷新、换台机器进来都还在（内容存在云库）
    # 分区小标题由 render_suggestions() 自己带，这里不再重复画一遍。
    st.write("")
    render_suggestions()

    st.markdown(_FOOT, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
