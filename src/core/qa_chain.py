# -*- coding: utf-8 -*-
"""阶段 4 · 生成层：LangChain LCEL 问答链路

    证据（retrievers.Evidence）
      → 证据上下文（每条带 [E1] 编号 + 条款号 + 页码）
      → ChatPromptTemplate（硬约束：只依据证据作答 / 逐句标注引用 / 证据不足须拒答）
      → ChatOpenAI（LLM_BACKEND 选 deepseek 走云端 / local 走本机模型，temperature = 0）
      → 结构化 JSON（answer / citations / insufficient / reason）
      → verify_citations 两级校验
      → 放行 or 拒答

设计取舍
--------
- **为什么强制 JSON 输出**：拒答与否必须是一个可程序判断的字段，不能靠
  在自然语言里找"未找到"三个字。结构化输出把「模型说什么」和「程序怎么判」
  彻底分开。
- **为什么还要两道校验**：JSON 保证了格式，但保证不了内容真实。引用编号
  可以编、结论可以超出原文。校验层是最后一道闸，也是本项目最想讲的部分。
- **为什么 temperature = 0**：规章问答不要创造性，要可复现。

用法
----
    from qa_chain import QAPipeline
    pipe = QAPipeline(retriever, llm)
    res = pipe.ask("考核不及格可以补考吗")
"""
from __future__ import annotations

import json
import os
import re
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

from verify_citations import parse_citations, verify  # noqa: E402

# 项目根目录：从本文件往上找，第一个含 app.py 的目录就是。不写死往上两级，
# 是因为模块已按分类住进 src/core/、src/retrieval/ 等子目录（见 src/_paths.py）。
ROOT = next((d for d in Path(__file__).resolve().parents if (d / "app.py").exists()),
            Path(__file__).resolve().parents[2])
DEFAULT_BASE_URL = "https://api.deepseek.com/v1"
DEFAULT_MODEL = "deepseek-chat"


def load_env_file(path=None):
    """轻量读取项目根 .env（不引入 python-dotenv）。已存在的环境变量优先。"""
    p = Path(path) if path else ROOT / ".env"
    if not p.exists():
        return False
    for line in p.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        k, v = k.strip(), v.strip().strip('"').strip("'")
        if k and k not in os.environ:
            os.environ[k] = v
    return True


# ------------------------------------------------------------------ Prompt
SYSTEM_PROMPT = """你是深圳职业技术大学《学生手册（2025 年修订）》的问答助手。

【必须遵守】
1. 只能依据下方【证据】回答。禁止使用证据之外的任何知识，禁止推测，禁止"一般情况下"。
2. 答案里每一个具体结论后面都要紧跟它依据的证据编号，形如 [E1]；一句话依据多条就写 [E1][E3]。
3. 若证据不足以回答（手册里没有相关规定，或证据只是字面沾边但答不了这个问题），
   必须把 insufficient 设为 true，并在 answer 里说明"手册中未找到相关规定"。绝不编造。
4. 证据若同时存在"本科口径"与"专科口径"，必须分别说明，不得混为一谈。
5. 证据中与问题相关的**每一个**数量、上限、期限、条件都要答到，不得只挑其中一项。
   同一事项分若干类别的（本科/专科、免修/置换/替代、各奖项档次），逐类说明。
   例：证据同时给出「免修或置换课程累计不得超过12学分」与「替代任选课累计不得超过
   6学分」时，两个上限都必须答出，只答一个即为遗漏。
6. 不要在正文里写任何「来源说明」或「口径归属说明」，例如"上述规定出自《某某办法》"、
   "本条规定见于第X条"、"上述为专科口径的规定"。出处由系统在答复之外单独列出。
   本科与专科确有差异时，直接用「本科：…」「专科：…」分述即可，不要额外加说明句。
7. 【答复怎么排版 —— 先给结论，再分点】
   answer 字段写成「一句结论 + 若干要点」，**不要整段照抄条款原文**：
   · 第一行：一句**核心结论**，直接回答这个问题的要点（不写条款号）；
   · 之后每个要点**独占一行**，用「- 」开头，一句话说清，行末紧跟 [Ex]；
   · 该答的要点按第 5 条**一个都不能少** —— 宁可多分一条，也不要为了短而漏项；
   · 除这些行以外不要再写别的句子。原文里的程序性描述、兜底条款、重复表述
     （如"具体按学校某某规定执行""本办法自发布之日起施行"）不要写进来；
   · 数字、期限、比例、金额必须与证据**完全一致**，一个字都不能改、不能省、不能约；
   · 【重要】手册原文由系统在答复下方另行完整展示，你**不需要**保留原文措辞。

   【排版范例】格式照下面这样，内容只是举例、不要照搬：
     正确的 answer：
       必修课首次不及格可以补考一次。[E1]
       - 补考成绩由卷面成绩与平时成绩综合评定，合格即获得学分。[E1]
       - 成绩标注「补考」记入成绩单；补考仍不及格的可申请重修。[E1]
     要避免的写法（整段照抄原文）：
       除通识教育选修课程外，学生首次修读的课程考核不及格者，可在下学期开学初
       参加一次补考（部分不适合补考的实践类、过程性考核的课程则需申请参加重修）……[E1]

8. 【关于【之前的对话】】如果这题带了【之前的对话】，它**不是证据**，只是
   背景 —— 只用来搞清「学生这次在问什么」（把「它 / 这个 / 那」还原成具体
   事项）。答案的依据**仍然只有下方【证据】**：对话里出现过的说法、数字、
   结论都不能当作依据写进答案，更不能照抄。若【证据】里没有本次问题所需
   的规定，照第 3 条处理（insufficient），不要拿上文凑。

9. 【问「是不是 / 算不算 / 可不可以 / 有没有硬性要求」这类是非题时】
   先把结论行**改写成证据能直接读出来的正面事实**，再去回答，规则是：
   · 证据里有**直接**回答该是非的条款 → 直接答，引该条款。
   · 没有直接规定，只有**相关**条款 → 结论行一律写成
     「A（手册里的词）会计入 / 影响 B」这种正面表述，
     **绝对不许**在结论里写「不是硬性要求」「没有要求」「不算」这类
     **论无**的话 —— 手册里没有这些原话，写出来就是无依据的臆断。
     然后用第 7 条的分点，把相关条款一条条列出来。
     例：问「体测是不是奖学金的硬性要求」，证据只有「体质测试是综合素质
     评价中体育素养的一项」和「综合素质评价结果是各项奖学金评选的依据」：
       ✅ 正确：
            体测成绩会计入综合素质评价，影响奖学金评选。[E3][E4]
            - 体质测试属于综合素质评价中的体育素养模块。[E3]
            - 综合素质评价结果是学校各项奖学金评选的依据和参考。[E4]
       ❌ 错误（手册没有这句话，会被判无依据）：
            体测成绩不是奖学金的硬性要求。[E3][E4]
   · 一条相关规定都没有 → 按第 3 条判 insufficient。

【输出格式】
只输出一个 JSON 对象，不要任何其他文字，不要代码围栏。JSON 含四个键：
  answer       字符串，带 [Ex] 标注的答复正文
  citations    字符串数组，答案实际用到的全部证据编号
  insufficient 布尔值，证据不足以回答时为 true
  reason       字符串，仅在 insufficient 为 true 时写明原因，否则留空"""

USER_TEMPLATE = """{history_block}【问题】{question}

【证据】
{context}{hints_block}"""

# 无历史时的占位（空串）—— 模板里保留槽位，但内容为空，等价于没有这一节。
HISTORY_NONE = ""

# 阶段 6 第 2 步：自检回环把「上一版漏答的点」回灌给生成，让它带着缺口重答
HINTS_BLOCK = """
【你上一版答复被检查出遗漏了下面这些要点，请在本版中一并答到】
{hints}
（注意：以上不是新证据，只是提醒清单；若证据中确实没有对应内容，不必强答。）"""


# ------------------------------------------------------------------ LLM
# 本地后端默认参数：llama-server / Ollama / LM Studio 都提供 OpenAI 兼容接口，
# 所以本地这条路和云端那条用的是同一个 ChatOpenAI 客户端，只是 base_url 不同。
DEFAULT_LOCAL_BASE = "http://127.0.0.1:8080/v1"
DEFAULT_LOCAL_MODEL = "intern-s1-mini"

# 免费兜底后端默认参数：智谱 GLM-4-Flash（注册即可用、不收费）。
# 只用来给"程序化抽取式答不好"的复杂题补一段通顺答案，日常题走不到这里。
DEFAULT_FREE_BASE = "https://open.bigmodel.cn/api/paas/v4"
DEFAULT_FREE_MODEL = "glm-4-flash"

_FALSE = ("0", "false", "False", "no", "off")


def llm_backend():
    """读 LLM_BACKEND：deepseek（默认，走云端 API）/ local（本机模型）/ free（免费 API）。

    放在环境变量而不是写死，是为了不改一行代码就能来回切：
    本地调试/演示用 local 不花钱，跑正式评测用 deepseek 保证质量。
    """
    return (os.environ.get("LLM_BACKEND") or "deepseek").strip().lower()


def is_rate_limited(exc: BaseException) -> bool:
    """判断异常是不是「免费档限流」（429 / 请求过于频繁）。

    为什么单独判：免费模型是「永久免费、**不限量但限速**」，限流属于
    设计的一部分，不是故障。把它跟真故障混在一起，页面上会出现
    「模型调用失败：429…」这种天书；分开之后才能对该退避的退避、
    对该报错的报错。
    """
    s = f"{type(exc).__name__}: {exc}".lower()
    return ("429" in s or "rate" in s and "limit" in s
            or "too many request" in s
            or "请求过于频繁" in s
            or "使用量已超出频率限制" in s
            or "超出频率限制" in s)


def _wrap_429(kw: dict):
    """构造一个「把 429 说人话」的免费档 LLM 客户端。

    实现方式是**继承 ChatOpenAI** 而不是拿它当属性套一层 —— 这点很关键，
    第一版写成了 `class _Client: def __getattr__...` 的代理，结果整条链炸在
    「Expected a Runnable, callable or dict」：LangChain 的 chain 只接受
    `Runnable`，而裸代理类不是 `BaseChatModel` 的子类，注册不上。
    子类化则天生就是合法 Runnable，属性、配置、结构化输出全部照常。

    为什么要两层（`max_retries` + 这一层）：
      · `max_retries` 负责**真的重试**，在 SDK 层重发，连接与会话都由它管；
      · 这一层负责**兜底的兜底** —— 万一重试全用尽仍被限流，别把
        「429 您的使用量已超出频率限制，将在 12:04:44 重置」原样甩到页面上。
        那句话对学生毫无意义，而且看起来像系统坏了。
    """
    from langchain_openai import ChatOpenAI

    class _Friendly429LLM(ChatOpenAI):
        """免费档专用：只覆盖 invoke 的异常翻译，行为与父类一致。"""

        def invoke(self, *a, **kw2):
            try:
                return super().invoke(*a, **kw2)
            except Exception as e:                        # noqa: BLE001
                if not is_rate_limited(e):
                    raise
                raise RuntimeError(
                    "在线大脑现在响应的人太多（免费额度限流），"
                    "稍等一两分钟再问一次就好。"
                    f"原始错误：{str(e)[:120]}") from e

    return _Friendly429LLM(**kw)


def _env_int(name: str, default: int) -> int:
    """读一个整数型环境变量，非法值退回默认（不让配置写错就崩）。"""
    try:
        return int(str(os.environ.get(name, "") or "").strip() or default)
    except (TypeError, ValueError):
        return default


def build_llm(model=None, temperature=0.0, api_key=None, base_url=None,
              use_json_mode=True, timeout=90, backend=None):
    """构造聊天模型客户端，由 backend 参数或 LLM_BACKEND 决定走哪条路。

    deepseek  走 DeepSeek 云端 API，密钥从 DEEPSEEK_API_KEY 取（计费）
    local     走本机 OpenAI 兼容服务（llama-server 等），不需要密钥、不产生费用
    free      走免费额度的云端 API（默认智谱 GLM-4-Flash），给程序化通道做兜底

    三条路都返回 ChatOpenAI，所以下游的 planner / verify / reflect 完全无感。
    """
    backend = (backend or llm_backend()).strip().lower()
    from langchain_openai import ChatOpenAI

    load_env_file()

    if backend == "free":
        base = (base_url or os.environ.get("FREE_LLM_BASE_URL")
                or DEFAULT_FREE_BASE)
        key = (api_key or os.environ.get("FREE_LLM_API_KEY") or "").strip()
        if not key or "你的" in key or "填" in key or len(key) < 12:
            raise RuntimeError(
                "FREE_LLM_API_KEY 缺失或仍是占位符。免费兜底通道需要一个密钥，"
                "推荐智谱 GLM-4-Flash（免费）：到 https://open.bigmodel.cn 注册，"
                "在「API 密钥」新建一个，填进项目根目录 .env 的 FREE_LLM_API_KEY。")
        kw = dict(model=model or os.environ.get("FREE_LLM_MODEL") or DEFAULT_FREE_MODEL,
                  temperature=temperature, api_key=key, base_url=base,
                  max_tokens=2000, timeout=timeout)
        # 免费档「不限量但限速」，429 是常态而非故障。langchain 自带
        # max_retries（默认 None = 不重试），在这里打开：它会在 SDK 层重发
        # 请求，比我们在外面包一层更稳（连接、会话、流式都它管）。
        # 3 次 + 指数退避（约 2/4/8 秒）足以跨过限流窗口。
        kw["max_retries"] = _env_int("FREE_LLM_MAX_RETRIES", 3)
        if use_json_mode:
            kw["model_kwargs"] = {"response_format": {"type": "json_object"}}
        return _wrap_429(kw)

    if backend == "local":
        base = (base_url or os.environ.get("LOCAL_LLM_BASE_URL")
                or DEFAULT_LOCAL_BASE)
        key = (api_key or os.environ.get("LOCAL_LLM_API_KEY")
               or "sk-local").strip()          # 本地服务不校验，占位即可
        mdl = model or os.environ.get("LOCAL_LLM_MODEL") or DEFAULT_LOCAL_MODEL
        kw = dict(model=mdl, temperature=temperature, api_key=key,
                  base_url=base, max_tokens=2000, timeout=timeout)
        if use_json_mode and os.environ.get("LOCAL_LLM_JSON_MODE", "1") not in _FALSE:
            kw["model_kwargs"] = {"response_format": {"type": "json_object"}}
        # 推理型模型（Intern-S1-mini 带思维链）会把答案前先写一大段思考，
        # 既能撑爆 max_tokens 又拖慢速度，问答场景建议关掉
        if os.environ.get("LOCAL_LLM_THINKING", "0") in _FALSE:
            kw["extra_body"] = {"chat_template_kwargs": {"enable_thinking": False}}
        return ChatOpenAI(**kw)

    key = (api_key or os.environ.get("DEEPSEEK_API_KEY") or "").strip()
    base = base_url or os.environ.get("DEEPSEEK_BASE_URL") or DEFAULT_BASE_URL
    # 占位符/半成品要挡在前面，否则错误会以 401 的形式绕一大圈才暴露
    if not key or "你的" in key or "填" in key or len(key) < 20:
        raise RuntimeError(
            "DEEPSEEK_API_KEY 缺失或仍是占位符。请到 https://platform.deepseek.com "
            "创建密钥，填进项目根目录的 .env 文件（照 .env.example 改），或设为环境变量。")
    kw = dict(model=model or os.environ.get("DEEPSEEK_MODEL") or DEFAULT_MODEL,
              temperature=temperature, api_key=key, base_url=base,
              max_tokens=2000, timeout=timeout)
    if use_json_mode:
        kw["model_kwargs"] = {"response_format": {"type": "json_object"}}
    return ChatOpenAI(**kw)


# ------------------------------------------------------------------ 引用编号规范化
def normalize_citations(items) -> list:
    """把模型给的引用编号收拾干净：'[E1]'、' e1 '、'E1,E2'、'[E1][E3]' → ['E1','E2'…]。

    为什么需要（2026-10-15 实测踩到）
    --------------------------------
    GLM 有时把方括号一起塞进字符串：返回 `citations: ["[E1]", "[E2]", "[E6]"]`。
    原来的 `str(c).upper()` 于是得到 `"[E1]"` —— 一级校验拿它跟证据编号 `E1`
    比对必然对不上，判成「申报了不存在的证据编号」，**一条完全正确的回答被
    整题拒答**。角标回填也救不了：回填只认合法编号，带括号的一个都不合法，
    于是它一个都不补 —— 拒答照旧。

    这里统一抽成 `E<数字>`。抽不出数字的（例如模型写了 'E'）原样留下，
    让它照常被一级校验拦下 —— 规范化只负责去掉**格式噪声**，不替伪造圆场。
    """
    out, seen = [], set()
    for c in (items or []):
        found = re.findall(r"[Ee]\s*(\d+)", str(c))
        if found:
            cand = [f"E{int(n)}" for n in found]
        else:
            cand = [re.sub(r"[^A-Za-z0-9]", "", str(c)).upper()]
        for eid in cand:
            if eid and eid not in seen:
                seen.add(eid)
                out.append(eid)
    return out


# ------------------------------------------------------------------ 角标回填
def backfill_citations(answer: str, citations, evs):
    """角标回填：模型**申报了出处、正文却一处 [Ex] 都没打**时，替它补上。

    为什么需要这一步（2026-10-15 用户截图实证）
    -------------------------------------------
    免费小模型（GLM-4-Flash）非常常见的毛病：`citations` 字段老老实实填了
    E1/E3/E4/E5，正文却一个 `[Ex]` 角标都不打。而一级校验（`check_existence`）
    有条硬规矩「有实质内容却一处引用都没标 → 不放行」（防「全篇不标」蒙混），
    于是**一条与手册逐字对得上的正确回答被整题拒掉** —— 检索分 0.99，
    答案完全正确，学生看到的却是「抱歉，这个问题我暂时答不了」。

    这正是本项目最忌讳的那类误杀：尺子量不出东西，被当成学生作弊。

    做法
    ----
    把**已申报、且确实存在于本次证据**里的编号拼成一组角标，插到答案末尾
    （有句末标点就插在它之前，符合中文「句末标注出处」的习惯），
    **原文一个字不改**，然后照常走两级校验（即「角标回填 + 重校验」）。

    只回填合法编号：申报了不存在的编号属于伪造引用，那种情形本来就该被
    一级校验拦下，这里绝不替它圆场。

    返回 (文本, 是否回填过)。
    """
    text = (answer or "").strip()
    if not text:
        return text, False
    if parse_citations(text):          # 正文自己标了角标 → 不动它
        return text, False

    known = {e.eid for e in evs}
    declared, seen = [], set()
    for c in (citations or []):
        e = str(c).upper()
        if e in known and e not in seen:
            seen.add(e)
            declared.append(e)
    if not declared:                   # 没申报、或申报的全是编的 → 不圆场
        return text, False

    marks = "".join(f"[{e}]" for e in declared)
    m = re.search(r"[。！？；]\s*$", text)
    return ((text[:m.start()] + marks + text[m.start():]) if m
            else (text + marks)), True


# ------------------------------------------------------------------ 结果
@dataclass
class QAAnswer:
    question: str
    answer: str = ""
    citations: list = field(default_factory=list)
    insufficient: bool = False
    reason: str = ""
    refused: bool = False
    refuse_reason: str = ""
    model_error: str = ""                      # 非空 = 这次是模型本身用不了（不是"手册没规定"）
    badge_backfill: bool = False               # 这次替模型补过 [Ex] 角标（见 backfill_citations）
    evidences: list = field(default_factory=list)
    verify: dict = field(default_factory=dict)
    latency: float = 0.0
    prompt_preview: str = ""
    confidence: float = 0.0                 # 程序化通道的契合度（LLM 通道不填）
    # ---- 阶段 6：Agent 化的过程信息（原 ask() 不填，保持默认即可） ----
    plan: dict = field(default_factory=dict)   # planner 输出
    hops: int = 1                              # 本题实际检索次数
    n_evidence: int = 0                        # 送给模型的证据条数
    reflect_rounds: int = 0                    # 自检触发的补检轮数（0=没补）
    reflect_log: list = field(default_factory=list)  # 每轮自检结果（缺口清单等）
    first_answer: str = ""                     # 自检前的首答快照（验收对比用）
    refuse_confirm: dict = field(default_factory=dict)  # 拒答复核记录（驳回/维持）
    # ---- 官网兜底（2026-10-17 新增） ----
    # 手册判「没有」之后才去官网找的那一份素材，形状见
    # src/retrieval/web_fallback.ask_web；空 dict = 没走这一趟。
    # ⚠ 它**不是** evidence：不进 citations、不进「手册答对率」，
    #   页面另起一张「官网来源」卡单独标（W1/W2…），页面顶部明说不是学生手册。
    web: dict = field(default_factory=dict)
    # 命中「得登录才能查」时的引导话术（src/retrieval/need_login.check），
    # 命中就**根本不去搜官网**，省 8 秒。
    web_login: dict = field(default_factory=dict)

    def pretty(self) -> str:
        lines = [f"问题：{self.question}", ""]
        if self.refused:
            lines.append(f"■ 拒答：{self.refuse_reason}")
            if self.answer and self.answer != REFUSE_TEXT:
                lines.append(f"  （模型原话：{self.answer}）")
        else:
            lines.append("■ 答复：")
            lines.append(self.answer)
            lines.append("")
            lines.append("■ 依据：")
            emap = {e.eid: e for e in self.evidences}
            for e in self.citations or []:
                ev = emap.get(e)
                if ev:
                    lines.append(f"  [{e}] {ev.cite}（PDF 第 {ev.page} 页）")
        lines.append("")
        lines.append(f"■ 耗时 {self.latency:.2f}s ｜ 证据 {len(self.evidences)} 条 ｜ "
                     f"引用 {len(self.citations)} 处 ｜ "
                     f"校验 {'通过' if self.verify.get('passed') else '未通过'}")
        return "\n".join(lines)


REFUSE_TEXT = "手册中未找到相关规定。"


# ------------------------------------------------------------------ 链路
class QAPipeline:
    def __init__(self, retriever, llm=None, top_k=6, llm_verify=True, verbose=True,
                 verify_recheck=True, gen_retry=1, refuse_confirm_min=None):
        self.retriever = retriever
        self.llm = llm
        self.top_k = top_k
        self.llm_verify = llm_verify
        self.verbose = verbose
        self.verify_recheck = verify_recheck   # 阶段5：断言级复核，降低判定波动
        self.gen_retry = gen_retry             # 阶段5：生成失败重试次数
        # 拒答复核阈值（REFUSE_CONFIRM_MIN 可覆盖，设 0 关闭复核）。
        # 动机：换成免费小模型后它会"保守误判"——证据明明在手却说没依据。
        # 标定依据见 src/probe_refuse_confirm.py：被误拒的 Q06 top1=0.987，
        # 而 7 道真该拒的题 top1 全在 0.0009~0.1101，两档差一个数量级，
        # 取 0.5 作界两边各留 0.39 的余量，不可能互相串档。
        if refuse_confirm_min is None:
            try:
                refuse_confirm_min = float(
                    os.environ.get("REFUSE_CONFIRM_MIN") or 0.5)
            except ValueError:
                refuse_confirm_min = 0.5
        self.refuse_confirm_min = refuse_confirm_min
        self._chain = None

    def _log(self, m):
        if self.verbose:
            print("  " + m, flush=True)

    def _build_chain(self):
        """LCEL：prompt | llm | JsonOutputParser。惰性构建，便于 dry-run 时不建链。"""
        if self._chain is not None:
            return self._chain
        from langchain_core.output_parsers import JsonOutputParser
        from langchain_core.prompts import ChatPromptTemplate

        prompt = ChatPromptTemplate.from_messages([
            ("system", SYSTEM_PROMPT),
            ("human", USER_TEMPLATE),
        ])
        self._chain = prompt | self.llm | JsonOutputParser()
        return self._chain

    @staticmethod
    def build_context(evidences) -> str:
        return "\n\n".join(e.render() for e in evidences)

    def _retry_with_confirm(self, question, context, top1, history_block=""):
        """带「证据已确认」的提示重答一次；仍判不足或调用失败 → 返回 None。

        为什么值得多花这一次调用：免费小模型判「证据不足」多半不是真不足，
        而是措辞保守 —— 把「不予申请」认不出等价于「不允许」。
        在提示里点破这一点，它通常就能正常作答，
        且给出的结论句质量远高于让程序机械照抄条款。
        """
        note = ("【系统复核】已确认上面列出的条款确实包含本题所问的规定"
                f"（最高相关度 {top1:.2f}）。请直接依据条款给出结论式的答复，"
                "不要以「证据不足」为由拒答。若条款用了不同措辞"
                "（例如「不予申请」即等价于「不允许」），请视为同一规定。")
        try:
            raw = self._build_chain().invoke(
                {"question": question, "context": context, "hints_block": note,
                 "history_block": history_block or HISTORY_NONE})
        except Exception as e:                      # noqa: BLE001
            self._log(f"复核重答调用失败：{str(e)[:80]}")
            return None
        if (isinstance(raw, dict) and not raw.get("insufficient")
                and (raw.get("answer") or "").strip()):
            return raw
        return None

    def ask(self, question: str, history_block: str = "") -> QAAnswer:
        t0 = time.time()
        evs = self.retriever.search(question, top_k=self.top_k)
        res = self.answer_from_evidences(question, evs,
                                         history_block=history_block)
        res.latency = time.time() - t0
        return res

    def answer_from_evidences(self, question: str, evs, hints=None,
                              history_block: str = "") -> QAAnswer:
        """生成 + 两级校验：给定证据产出答案（**不含检索**）。

        阶段 6 抽出：供 Agent 循环在"合并了多路检索的证据"之后复用，
        免得 Agent 那边把生成/校验再写一遍。原 ask() 行为不变，
        只是变成 `search() -> answer_from_evidences()` 两步。

        hints：阶段 6 第 2 步新增。自检查出"上一版漏答的点"时，把清单
        回灌给生成（不是新证据，只是提醒），让它带着缺口重答一遍 ——
        专门治 Q36 那种"资料在手边、生成时却没答全"的毛病。

        注意：本方法不写 res.latency —— 由调用方统一计时，
        这样「规划 + 检索 + 生成」能合成一个端到端耗时。
        """
        res = QAAnswer(question=question)
        res.evidences = evs
        if not evs:
            res.refused, res.refuse_reason = True, "检索未获得任何证据"
            return res

        context = self.build_context(evs)
        hints_block = (HINTS_BLOCK.format(hints="\n".join(f"- {h}" for h in hints))
                       if hints else "")
        hist = history_block or HISTORY_NONE
        res.prompt_preview = system_user_preview(question, context, hints_block,
                                                 hist)

        if self.llm is None:                      # dry-run：只到证据构造
            res.refused = True
            res.refuse_reason = "dry-run：未接入 LLM"
            return res

        # ---------- 生成（失败重试：压掉偶发的解析/网络抖动） ----------
        raw, last_err = None, ""
        for attempt in range(self.gen_retry + 1):
            try:
                raw = self._build_chain().invoke(
                    {"question": question, "context": context,
                     "hints_block": hints_block, "history_block": hist})
                if isinstance(raw, dict) and (raw.get("answer") or "").strip():
                    break
                last_err, raw = "模型返回内容为空", None
            except Exception as e:                # noqa: BLE001
                last_err, raw = str(e)[:160], None
            if raw is None and attempt < self.gen_retry:
                self._log(f"生成未成功，重试 {attempt + 1}/{self.gen_retry}："
                          f"{last_err[:60]}")
        if raw is None:
            # 标出来：这不是「手册里没规定」式拒答，而是**模型本身用不了**
            # （429 重试烧完 / 网络断 / 密钥失效 / 返回空）。上层据此转程序化
            # 兜底照抄原文，而不是把这个拒答丢给学生（2026-10-15 加）。
            res.model_error = last_err
            res.refused, res.refuse_reason = True, f"模型调用失败：{last_err}"
            return res
        if not isinstance(raw, dict):
            raw = {"answer": str(raw)}

        res.answer = (raw.get("answer") or "").strip()
        res.citations = normalize_citations(raw.get("citations"))
        if not res.citations:                     # 模型没给列表，就从正文里捞
            res.citations = parse_citations(res.answer)
        res.insufficient = bool(raw.get("insufficient"))
        res.reason = raw.get("reason") or ""
        # ---------- 角标回填（2026-10-15） ----------
        # GLM 常见「citations 填对了、正文忘打 [Ex]」→ 一级校验「通篇无出处」
        # 会把一句正确回答整题毙掉。这里替它把合法编号补到答案末尾，
        # 原文一字不改，随后照常走两级校验。见 backfill_citations 的注释。
        # 注意：模型自称「证据不足」时**不**回填 —— 那时正文是说明为什么答不了，
        # 给它挂角标毫无意义，还会污染「模型原话」那一栏。
        if not res.insufficient:
            res.answer, res.badge_backfill = backfill_citations(
                res.answer, res.citations, evs)
            if res.badge_backfill:
                self._log(f"模型正文未打角标，已按申报的 citations 回填："
                          f"{res.citations}")

        # ---------- 模型自述证据不足 ----------
        if res.insufficient:
            # 拒答复核：不让"证据明显在手、模型却嘴硬说没有"这种情况直接变成拒答。
            # 判据用检索分而不是模型自述 —— 检索分是客观打分，模型自述是主观判断，
            # 弱模型最容易在这主观一步上保守过头（Q06 就是这么被误拒的）。
            top1 = max((e.score or 0.0) for e in evs) if evs else 0.0
            if evs and self.refuse_confirm_min > 0 and top1 >= self.refuse_confirm_min:
                self._log(f"模型称证据不足，但 top1={top1:.3f} ≥ "
                          f"{self.refuse_confirm_min}，启动拒答复核")
                # 第一手：把「证据确实有」说破，让模型重答一次。
                fixed = self._retry_with_confirm(question, context, top1, hist)
                if fixed is not None:
                    res.answer = (fixed.get("answer") or "").strip()
                    res.citations = normalize_citations(fixed.get("citations"))
                    if not res.citations:
                        res.citations = parse_citations(res.answer)
                    # 复核重答同样可能忘打角标 —— 一并回填，免得刚救回来的
                    # 答案又被一级校验的「通篇无出处」毙掉。
                    res.answer, res.badge_backfill = backfill_citations(
                        res.answer, res.citations, evs)
                    res.insufficient = False
                    res.reason = ""
                    res.refused = False
                    res.refuse_reason = ""
                    res.verify = verify(res.answer, res.citations, evs,
                                        llm=self.llm if self.llm_verify else None,
                                        recheck=self.verify_recheck)
                    if not res.verify["passed"]:
                        res.refused = True
                        res.refuse_reason = ("复核重答未通过引用校验：" +
                                             "；".join(res.verify["reasons"]))
                    res.refuse_confirm = {"top1": round(top1, 4),
                                          "action": "驳回拒答，模型重答"}
                    return res
                # 第二手：模型还是不松口，就改成照抄原文（零幻觉保底）
                try:
                    from procedural_qa import generate, _verify as _pverify
                    out = generate(question, evs, max_evidence=3)
                except Exception as e:              # noqa: BLE001
                    self._log(f"拒答复核异常，维持原判：{e}")
                    out = None
                if out and not out["insufficient"]:
                    res.answer = out["answer"]
                    res.citations = out["citations"]
                    res.insufficient = False
                    res.reason = ""
                    res.refused = False
                    res.refuse_reason = ""
                    res.verify = _pverify(res.answer, res.citations, evs)
                    res.refuse_confirm = {"top1": round(top1, 4),
                                          "action": "驳回拒答，改照抄原文"}
                    return res
            res.refused = True
            res.refuse_reason = res.reason or "模型判定证据不足以回答"
            res.refuse_confirm = {"top1": round(top1, 4), "action": "维持拒答"}
            return res

        # ---------- 两级校验 ----------
        v = verify(res.answer, res.citations, evs,
                   llm=self.llm if self.llm_verify else None,
                   recheck=self.verify_recheck)
        res.verify = v
        if not v["passed"]:
            res.refused = True
            res.refuse_reason = "引用校验未通过：" + "；".join(v["reasons"])
        return res


def system_user_preview(question, context, hints_block="", history_block=""):
    return (f"[system]\n{SYSTEM_PROMPT}\n\n[human]\n"
            + USER_TEMPLATE.format(question=question, context=context,
                                   hints_block=hints_block,
                                   history_block=history_block
                                   or HISTORY_NONE))


if __name__ == "__main__":
    from retrievers import HandbookRetriever
    r = HandbookRetriever("data", use_rerank=True)
    pipe = QAPipeline(r, llm=None, top_k=5)      # 无 key 时跑 dry-run
    print(pipe.ask(sys.argv[1] if len(sys.argv) > 1 else
                   "考核不及格可以补考吗").pretty())
