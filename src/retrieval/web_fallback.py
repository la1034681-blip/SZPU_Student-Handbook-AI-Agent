# -*- coding: utf-8 -*-
"""官网兜底：手册查不到时，去学校官网找「可以照抄的原话」（2026-10-17 新增）

为什么需要它
------------
手册只有 974 条、42 份制度。学生问「食堂几点开门」「这学期校历怎么排」，
手册里根本没有 —— 以前只能回一句「本手册中没有对应内容」。可这些答案
学校官网上就写着。

所以补一条**只在手册判拒答之后才走**的支路：去官网搜、抓回原文，让学生
至少能拿到一句能核对的话 + 一个能点开的链接。

三条红线（改这个文件前先读一遍）
--------------------------------
1. **手册优先**。这个模块只负责「手册之外」；调用方必须先确认手册答不了，
   才准调 `ask_web()`。这里不做任何"官网优先"的判断。
2. **只许照抄**。返回的每一句话都必须能在抓回来的原文里**逐字找到**
   （`passages[].text` 是原文切片，不是生成的）。做不到就返回空，让上层
   照旧拒答 —— 宁可少答，不许编。
3. **只抓自家域名**。所有 URL 的域名必须是 `szpu.edu.cn` 或其子域，否则丢。

它怎么工作（四道工序）
----------------------
    ① 站内搜索   先 GET 首页拿 cookie，再 POST 搜索接口（不带 cookie 只回空壳）
    ② 筛候选页   只留 info/xxx/yyy.htm 单篇正文页，按标题词重合度重排
    ③ 抓正文     定位 v_news_content 容器 → 剥标签 → 抽 HTML 表格
                 正文太短就下附件（PDF），用 find_tables 认表格
    ④ 切句打分   句子 + 表格行一起用 BM25 打分，留最高的几句

★ 表格不摊成纯文本
--------------------
实测（2026-10-17，作息时间表 PDF）：连续提取文字得到的是
「教学楼 时间 上午 下午 晚上…」这种流水乱序；换成表格识别，直接拿到
7 行 × 6 列，单元格里完整保留「第1节：8:30—9:15／第2节：9:20—10:05」。
所以这里**先认表格**，表格按行拼成「上午｜第1节：8:30—9:15；第2节…」
这样的行语句去参与打分；同时把二维数组原样放进 `tables`，交给页面上
画成真表格（学生看表比看我描述准）。

★ 图片版 PDF：只给链接
------------------------
有些附件是影印图片（实测：2025-2026 学年校历 PDF，456KB、1 页、
抽不出一个字）。这种不硬上 OCR（几十上百兆依赖、免费容器扛不住），
也不对学生说"我识别不出"——直接返回 `reason="link_only"` 和链接，
让上层把链接原样给出去。

注意
----
· 总预算 `budget` 秒，到点就收手，返回已经拿到的部分（哪怕只有链接）。
· 限速：两次请求之间至少 `MIN_INTERVAL` 秒，对官网客气点。
· 缓存 24 小时（内存字典）—— 同一个问题一天内不重复抓。
· `pymupdf` 是可选依赖：没装就自动降级成「只给链接」，不会崩。
"""
from __future__ import annotations

import base64
import html as html_mod
import json
import re
import time
import urllib.parse
import urllib.request
import http.cookiejar
from datetime import datetime

try:                                    # 只为了抽 PDF，缺了也能跑
    import pymupdf
except Exception:                       # pragma: no cover
    pymupdf = None

SITE = "https://www.szpu.edu.cn"
SEARCH_URL = f"{SITE}/sousuojieguo.jsp?wbtreeid=1001"
ALLOW_SUFFIX = "szpu.edu.cn"
UA = "Mozilla/5.0 (compatible; SZPU-handbook-bot/1.0; +handbook assistant)"
MIN_INTERVAL = 0.5          # 两次请求的最小间隔（秒）
CACHE_TTL = 24 * 3600       # 同一问题缓存多久
MAX_KEYWORDS = 4            # 最多试几组关键词（一组一拍，多了会顶穿预算）
MAX_PAGES = 2               # 最多抓几篇正文
MAX_PDF = 1                 # 最多下几个附件
MIN_BODY = 200              # 正文少于这么多字，才去下附件

# 这几种词会污染检索，搜索前先摘掉
_DROP = ("请问", "问一下", "想问", "帮我", "查询", "一下", "呢", "吗", "啊", "吧",
         "的", "是", "有", "在", "和", "与", "怎么", "怎样", "如何", "什么", "多少",
         "几点", "哪里", "哪个", "可以", "能不能", "是否")

# 「学校」这种泛词在官网标题里几乎不出现，留着会把关键词带偏
# （「学校的作息时间」→ 去掉泛词才是「作息时间」）
_GENERIC = ("我们学校", "咱们学校", "咱学校", "学校", "我校", "贵校")

# 正文里的噪声（导航、版权、搜索框残留）
_NOISE = ("Copyright", "版权所有", "备案号", "ICP备", "请输入搜索", "群团组织及其他单位",
          "当前栏目", "关闭", "打印本页", "分享到")

# 「学生怎么说」→「官网上该搜什么」
# 为什么需要：学生问「第5节课几点上」，官网上那页叫《校园作息时间》——
# 一个字都对不上，搜出来全是「思政第一课」这类新闻。跟手册那边的
# term_alias 是同一个道理，只是换到官网场景。
WEB_TERMS = (
    (("上课", "下课", "课间", "作息", "几点上", "第几节"), ("作息时间",)),
    (("节课", "一节课", "两节课", "三节课"), ("作息时间",)),
    (("上午", "下午", "晚上", "一早"), ("作息时间",)),
    (("开学", "放假", "寒暑假", "校历", "几号"), ("校历",)),
    # 2026-10-20 补「环校巴士／校车／交通车」：学生嘴上叫「环校巴士」，官网文件里
    # 写的是「校园交通车」「班车」。原来只认「班车/校车/通勤/发车」，于是问
    # 「环校巴士」时关键词退化成「环校」「巴士」，站内搜索一条都搜不到 → 拒答。
    # 实测：搜「校园交通车」能抓到《通勤车及校园交通车各线路时间表》含真表格。
    (("班车", "校车", "通勤", "发车", "环校巴士", "环校", "巴士", "交通车"),
     ("校园交通车", "班车")),
    (("图书馆", "借书", "还书", "开馆", "闭馆"), ("图书馆",)),
    (("食堂", "饭堂", "吃饭", "就餐", "打饭"), ("食堂", "就餐")),
    (("电话", "联系", "找谁", "咨询"), ("联系电话",)),
    (("宿舍", "公寓", "寝室"), ("学生公寓",)),
    (("wifi", "wi-fi", "上网", "网络"), ("网络服务",)),
    (("校园卡", "一卡通", "饭卡"), ("校园卡",)),
)


def _web_terms(question: str) -> list[str]:
    """按问题内容，给出官网上真正会这么写的词。"""
    low = (question or "").lower()
    out = []
    for keys, words in WEB_TERMS:
        if any(k in low for k in keys):        # 命中任一同义词就够了
            out.extend(words)
    return out


def rewrites(question: str) -> list[str]:
    """把问句里的学生用词**就地换成官网用词**，返回改写后的问句。

    为什么光给关键词不够（2026-10-20 实测）：
        只往搜索词里塞「校园交通车」，可以搜到页面；但**筛句那一步**还有一道
        「这句得覆盖问题里至少三成的词」，而问题本身仍是「校车」两个字 ——
        时刻表里一个「校」都没有，覆盖率 0 → 抓到了也全被丢掉，照样 no_result。
        换成「校园交通车怎么坐」再跑一遍，整条链路都按官网的用词走，立刻有表有句子。

    所以：先拿原话搜，搜不到就用改写句再搜一次。
    """
    q = " ".join((question or "").split())
    out: list[str] = []
    for keys, words in WEB_TERMS:
        for k in keys:
            if k and k in q:
                if words:
                    cand = " ".join(q.replace(k, words[0]).split())
                    if cand and cand != q and cand not in out:
                        out.append(cand)
                break                            # 同一行只换一次，别叠加
    return out[:2]


_CACHE: dict[str, tuple[float, dict]] = {}


# ------------------------------------------------------------------ 小工具
def _clean(s: str) -> str:
    """去标签、解实体、压空白。"""
    s = re.sub(r"(?is)<(script|style|noscript).*?</\1>", " ", s or "")
    s = re.sub(r"(?s)<[^>]+>", " ", s)
    s = html_mod.unescape(s)
    s = s.replace("\u3000", " ").replace("\xa0", " ")
    return re.sub(r"[ \t\r\f\v]+", " ", s).strip()


def _lines(s: str) -> list[str]:
    """压成"一行一句"，并丢掉噪声行。"""
    out = []
    for x in s.splitlines():
        x = x.strip()
        if len(x) < 4:
            continue
        if any(n in x for n in _NOISE):
            continue
        out.append(x)
    return out


def _split_sentences(text: str) -> list[str]:
    """按中文句读切句。

    下限 9 个字：再短的多半不是句子，而是署名、栏目名、编辑信息这类碎片
    （实测抓到过「（图书馆 吴思）」——那是新闻稿的作者署名，拿它当答案
    比不答更糟）。
    """
    parts = re.split(r"(?<=[。；！？])|\n", text or "")
    out = []
    for p in parts:
        p = p.strip(" ·、|")
        if 9 <= len(p) <= 300:
            out.append(p)
    return out


def _strip_question(q: str) -> str:
    s = q or ""
    for w in _GENERIC:
        s = s.replace(w, "")
    for w in _DROP:
        s = s.replace(w, "")
    return re.sub(r"[^\w\u4e00-\u9fff]", "", s) or q or ""


def _bigrams(s: str) -> set[str]:
    """中文切 bigram，数字/英文当整串。

    为什么不能一股脑按字符切：
    · `2025-2026` 和 `2022～2023` 会共享 `20`、`02` 这些碎片，于是
      「2025-2026 校历」能匹配到《2022～2023学年信息公开报告》；
    · 但 `第5节课` 里的 `5` 又必须留着，不然「第5节」和「第6节」就分不开了。
    所以：中文与数字混合的 bigram 保留，**纯数字的碎片丢掉**，数字串整体算一个词。
    """
    s = re.sub(r"[^\u4e00-\u9fffA-Za-z0-9]", "", s or "")
    out: set[str] = set()
    for i in range(len(s) - 1):
        bg = s[i:i + 2]
        if bg[0].isdigit() and bg[1].isdigit():
            continue                      # 纯数字碎片
        out.add(bg)
    out.update(re.findall(r"[A-Za-z]{2,}|\d{2,}", s))
    return out


def _coverage(text: str, q_bigrams: set[str]) -> float:
    """这句话覆盖了问题里几成的词。用来挡「碰巧有一个词对上」的噪声。"""
    if not q_bigrams:
        return 0.0
    return len(q_bigrams & _bigrams(text)) / len(q_bigrams)


def _ok_host(url: str) -> bool:
    """白名单：只认自家域名。"""
    try:
        host = urllib.parse.urlparse(url).netloc.split(":")[0].lower()
    except Exception:
        return False
    return host == ALLOW_SUFFIX or host.endswith("." + ALLOW_SUFFIX)


def _abs(url: str) -> str:
    """相对路径补全成绝对地址（官网结果页给的是「无前导斜杠」的相对路径）。"""
    if not url:
        return ""
    url = url.strip()
    if url.startswith("//"):
        return "https:" + url
    if url.startswith("http"):
        return url
    if url.startswith("/"):
        return SITE + url
    return SITE + "/" + url


# ------------------------------------------------------------------ 会话
class _Session:
    """带 cookie 的轻会话：先 GET 一次首页，后面所有请求都带着它。

    为什么非要这一步：站内搜索**不带 cookie 也回 HTTP 200**，但正文是
    40KB 的空壳，一条结果都没有（2026-10-17 实测栽在这里）。
    """

    def __init__(self, timeout: float = 6.0):
        self.timeout = timeout
        self.cj = http.cookiejar.CookieJar()
        self.op = urllib.request.build_opener(urllib.request.HTTPCookieProcessor(self.cj))
        self.warmed = False
        self._last = 0.0
        self.requests = 0

    def _wait(self):
        gap = time.time() - self._last
        if gap < MIN_INTERVAL:
            time.sleep(MIN_INTERVAL - gap)
        self._last = time.time()

    def _open(self, url, data=None):
        self._wait()
        req = urllib.request.Request(url, data=data, headers={"User-Agent": UA})
        self.requests += 1
        with self.op.open(req, timeout=self.timeout) as r:
            raw = r.read()
            return raw, r.headers.get_content_charset() or "utf-8"

    def warm(self):
        if not self.warmed:
            self._open(SITE + "/")
            self.warmed = True

    def get_text(self, url) -> str:
        self.warm()
        raw, enc = self._open(url)
        return raw.decode(enc, "ignore")

    def post_text(self, url, fields: dict) -> str:
        self.warm()
        body = urllib.parse.urlencode(fields).encode("utf-8")
        raw, enc = self._open(url, body)
        return raw.decode(enc, "ignore")

    def get_bytes(self, url) -> bytes:
        self.warm()
        raw, _ = self._open(url)
        return raw


# ------------------------------------------------------------------ ① 搜索
def keywords(question: str, hints=None) -> list[str]:
    """把学生原话拆成几组「官网能搜到东西」的关键词。

    为什么不能整句搜：「图书馆开放时间」→ 0 条、「食堂几点开门」→ 0 条；
    拆成「图书馆」→ 15 条、「食堂」→ 10 条（2026-10-17 实测）。
    官网的检索是按词切分的，句子越长越搜不到。

    顺序也讲究：**先试"官网会这么写"的词**（`WEB_TERMS`），再试原话，
    最后才是切出来的短词。因为「第5节课几点上」这种，只有换成「作息时间」
    才搜得到 —— 官网那一页的标题里压根没有"课"字。
    """
    # 第一优先：明确给的提示 + 学生腔换成官网词
    head: list[str] = []
    for h in list(hints or []) + _web_terms(question):
        h = (h or "").strip()
        if len(h) >= 2 and h not in head:
            head.append(h)

    # 第二优先：去掉虚词的原话，再从最长片段里切短词
    base = _strip_question(question)
    tail: list[str] = []
    if base:
        tail.append(base)
        cn = re.findall(r"[\u4e00-\u9fff]+", base)
        if cn:
            longest = max(cn, key=len)
            for n in (2, 3, 4):
                if len(longest) > n:
                    tail.append(longest[:n])
                    tail.append(longest[-n:])

    # 短词先试（召回高），但排在映射词后面
    tail = [x for x in dict.fromkeys(tail) if len(x) >= 2 and x not in head]
    tail.sort(key=len)
    return (head + tail)[:MAX_KEYWORDS]


RE_ITEM = re.compile(
    r'<a[^>]+href="((?:\.\./)*info/[^"]+?\.(?:htm|html))"[^>]*class="[^"]*a flex[^"]*"[^>]*>(.*?)</a>',
    re.S,
)


def search_site(sess: _Session, kw: str) -> list[dict]:
    """搜一次，返回 [{url, title}]。空列表＝这次没结果。"""
    key = base64.b64encode(kw.encode("utf-8")).decode()
    h = sess.post_text(SEARCH_URL, {
        "lucenenewssearchkey": key,
        "_lucenesearchtype": "1",
        "searchScope": "0",
    })
    out, seen = [], set()
    for url, block in RE_ITEM.findall(h):
        title = _clean(block)
        title = re.sub(r"\s+", "", title)
        if not title or len(title) < 4:
            continue
        u = _abs(url)
        if u in seen:
            continue
        seen.add(u)
        out.append({"url": u, "title": title})
    # 兜底：有些结果块没有 class="a flex"，只认 info/ 路径也能凑
    if not out:
        for url in re.findall(r'href="((?:\.\./)*info/[^"]+?\.htm)"', h):
            u = _abs(url)
            if u not in seen:
                seen.add(u)
                out.append({"url": u, "title": ""})
    return out


def rank_candidates(items: list[dict], question: str, kws=None) -> list[dict]:
    """按标题与问题的词重合度重排。

    为什么不信官网自己的顺序：搜「食堂」第一条是《深职大举行2026届毕业生
    毕业典礼》（2026-10-17 实测）。它的排序基本是"最新发布优先"。

    `kws` 也要算进来：搜索时用的是换过的词（「上午第一节课」→ 搜「作息时间」），
    只拿原问题去比标题，那页叫《校园作息时间》、一个词都对不上，会被误丢掉。
    """
    qb = _bigrams(_strip_question(question)) | _bigrams(" ".join(kws or []))
    strong, weak = [], []
    for it in items:
        title = it["title"]
        tb = _bigrams(title)
        overlap = len(qb & tb)
        info_page = any(k in title for k in ("指南", "服务", "联系", "一览", "须知",
                                             "办法", "问答", "交通", "时刻", "地址", "办公"))
        if overlap <= 0:
            # 标题一个词都不沾：只有"信息型标题"才给一次机会
            # （搜「班车」，那页叫《校园交通指南》；搜「电话」，那页叫《各部门联系方式》）
            if info_page:
                weak.append((1.0, it))
            continue
        score = overlap
        if any(k in title for k in ("作息", "校历", "时间", "电话", "开放", "服务", "指南")):
            score += 0.5
        if len(title) > 28:
            score -= 0.5          # 长标题多半是新闻稿，不是信息页
        strong.append((score, it))
    strong.sort(key=lambda x: -x[0])
    weak.sort(key=lambda x: -x[0])
    # 带上 strong 标记：靠"信息型标题"补位进来的页面可信度低一档，
    # 只许取它的正文，不许拿它的表格（实测：问「教务处电话」，
    # 补位页面是《技能培训项目一览表》，整张表全是课程设置）。
    out = [dict(it, strong=True) for _, it in strong]
    out += [dict(it, strong=False) for _, it in weak]
    return out[:MAX_PAGES]


_PLACE_WORDS = ("校区", "园区", "楼", "室", "地址", "位于", "坐落", "号",
                "东区", "西区", "南区", "北区", "门口")
_TIME_RE = re.compile(r"\d{1,2}[:：]\s*\d{2}|\d{1,2}\s*点")


def _focus_ok(question: str, text: str) -> bool:
    """答案得对上"问题问的那一类东西"。

    问「在哪个校区」，答案里就得有校区/楼/地址；问「几点」，就得有具体时间。
    不然会出现这种情况：问「图书馆在哪个校区」，抓到一句「他把生态文明理念
    融入图书馆资源建设」——字面上有「图书馆」，但根本没回答"在哪儿"。
    """
    q = question or ""
    if len(text) < 25:
        return True        # 短句多半是页面里的小标题，不该被"焦点"规则卡掉
    if re.search(r"哪个校区|哪里|在哪|什么地方|地址", q):
        if not any(w in text for w in _PLACE_WORDS):
            return False
    # 「开放时间 / 营业时间 / 开门 / 关门」也算问时刻（2026-10-18 补）。
    # 起因：问「图书馆开放时间」，抓回一条《校园开放日活动成功举办》——里面有
    # 「图书馆」也有「开放**日**」，字面都对得上，可它一个字没回答几点开门。
    # 原来只认「几点/什么时候/时间表」，这一问法恰好漏在网眼下面。
    if re.search(r"几点|什么时候|多少点|时间表|开放时间|营业时间|开门|关门|作息", q):
        if not _TIME_RE.search(text):
            return False
    if re.search(r"电话|联系方式|找谁", q):
        if not re.search(r"\d{7,}|内线|电话", text):
            return False
    return True


# ------------------------------------------------------------------ ③ 页面
RE_TABLE = re.compile(r"(?is)<table[^>]*>(.*?)</table>")
RE_TR = re.compile(r"(?is)<tr[^>]*>(.*?)</tr>")
RE_TD = re.compile(r"(?is)<t[hd][^>]*>(.*?)</t[hd]>")
RE_ATTACH = re.compile(
    r'href="([^"]*(?:download\.jsp\?[^"]*|\.pdf|\.docx?|\.xlsx?)[^"]*)"[^>]*>(.*?)</a>', re.I | re.S)
RE_HREF = re.compile(r'href="([^"]+)"', re.I)
RE_EMBED_PDF = re.compile(r'showVsbpdfIframe\(\s*"([^"]+\.pdf)"')
RE_PUBLISHED = re.compile(r"(?:发布|时间|日期)[：: ]*\s*(20\d{2}[-/年]\d{1,2}[-/月]\d{1,2})")


def parse_tables(html: str) -> list[list[list[str]]]:
    """把关联网页里的 <table> 抽成二维数组（官网很少用，但通知里偶尔有）。"""
    out = []
    for tb in RE_TABLE.findall(html):
        rows = []
        for tr in RE_TR.findall(tb):
            cells = [_clean(td) for td in RE_TD.findall(tr)]
            if any(cells):
                rows.append(cells)
        if len(rows) >= 2 and max(len(r) for r in rows) >= 2:
            out.append(rows)
    return out


def _body_html(html: str) -> tuple[str, bool]:
    """取正文容器；取不到就退回全文。返回 (html片段, 是否命中容器)。"""
    m = re.search(r'(?is)<div[^>]+class="[^"]*\bv_news_content\b[^"]*"[^>]*>(.*?)'
                  r'(?:<div[^>]+id="div_vote_id"|</div>\s*<div[^>]+class="[^"]*share)', html)
    if m:
        return m.group(1), True
    m = re.search(r'(?is)<div[^>]+class="[^"]*\bv_news_content\b[^"]*"[^>]*>(.*)', html)
    if m:
        return m.group(1)[:60000], True
    return html, False


def fetch_page(sess: _Session, url: str) -> dict:
    """抓一页：正文、表格、附件、内嵌 PDF、发布日期。"""
    h = sess.get_text(url)
    title = _clean(re.search(r"(?is)<title>(.*?)</title>", h).group(1)) if re.search(
        r"(?is)<title>(.*?)</title>", h) else ""
    title = re.sub(r"-深圳职业技术大学$|-深圳职业技术学院$", "", title).strip()

    body, container = _body_html(h)

    tables = parse_tables(body)
    # 表格从正文文本里剔掉，免得同一份内容既算表格又算句子
    plain = RE_TABLE.sub(" ", body)
    text = "\n".join(_lines(_clean(plain)))

    # 附件：`download.jsp` 是 CMS 的统一下载入口（不知道类型）；也有直接给 .pdf 的
    pdfs, seen_href = [], set()
    for href, label in RE_ATTACH.findall(body):
        u = _abs(href)
        if u in seen_href:
            continue
        seen_href.add(u)
        if href.lower().endswith((".pdf", ".doc", ".docx", ".xls", ".xlsx")):
            pdfs.append({"url": u, "label": _clean(label)[:60]})
    for href in RE_HREF.findall(body):
        if "download.jsp" in href and _abs(href) not in seen_href:
            seen_href.add(_abs(href))
            pdfs.append({"url": _abs(href), "label": ""})
    # 内嵌预览（Webplus 用 showVsbpdfIframe 把 PDF 直接嵌在正文里）
    for href in RE_EMBED_PDF.findall(body):
        if _abs(href) not in seen_href:
            seen_href.add(_abs(href))
            pdfs.append({"url": _abs(href), "label": "内嵌 PDF"})

    pub = RE_PUBLISHED.search(_clean(h))
    return {
        "url": url,
        "title": title,
        "published": (pub.group(1).replace("年", "-").replace("月", "-").replace("日", "")
                      if pub else ""),
        "text": text,
        "tables": tables,
        "attachments": pdfs[:2],
        "container": container,
    }


# ------------------------------------------------------------------ ④ PDF
def read_pdf(data: bytes) -> dict:
    """读 PDF：优先认表格，再取文字。

    返回 {text, tables, image_only, pages}
    · image_only=True 表示"一个字都抽不出来"（影印图片版）—— 这时不要
      硬凑内容，交给上层只给链接。
    """
    if pymupdf is None:
        return {"text": "", "tables": [], "image_only": False, "pages": 0,
                "error": "没有 pymupdf"}
    try:
        doc = pymupdf.open(stream=data, filetype="pdf")
    except Exception as e:
        return {"text": "", "tables": [], "image_only": False, "pages": 0,
                "error": f"打不开：{e}"}

    texts, tables = [], []
    for page in doc:
        try:
            texts.append(page.get_text())
        except Exception:
            pass
        try:
            for t in page.find_tables().tables:
                rows = []
                for row in t.extract():
                    cells = [(c or "").strip() for c in row]
                    if any(cells):
                        rows.append(cells)
                if len(rows) >= 2 and max(len(r) for r in rows) >= 2:
                    tables.append(rows)
        except Exception:
            pass                       # 认不出表不影响拿文字
    text = "\n".join(texts)
    return {
        "text": text,
        "tables": tables,
        "pages": doc.page_count,
        "image_only": len(text.strip()) < 20 and not tables,
        "error": "",
    }


def table_facts(rows: list[list[str]]) -> list[str]:
    """把二维表按行拼成「行标签｜内容；内容」的句子，好参与检索打分。

    为什么不直接摊平：摊平会把「上午」「下午」这种行标签丢掉，于是
    「第5节几点」就不知道是下午——而表格的意义恰恰在这层对应关系。
    """
    facts = []
    for row in rows:
        cells = [c for c in row if c]
        if not cells:
            continue
        cells = [re.sub(r"\s+", " ", c) for c in cells]
        head, rest = cells[0], cells[1:]
        body = "；".join(rest) if rest else ""
        line = f"{head}｜{body}" if body else head
        if 4 <= len(line) <= 400:
            facts.append(line)
    return facts


# ------------------------------------------------------------------ ⑥ 打分
def rank_texts(texts: list[tuple[str, str]], question: str, k: int = 3) -> list[tuple[str, float]]:
    """用项目现成的 BM25 给句子打分。texts=[(原文, 文本)]，返回 [(原文, 分数)]。"""
    from bm25 import BM25

    if not texts:
        return []
    bm = BM25()
    bm.fit([(str(i), t) for i, (_, t) in enumerate(texts)])
    q = _strip_question(question)
    hits = bm.search(q, k=max(k * 3, 8))
    return [(texts[int(i)][0], s) for i, s in hits if s > 0][:k]


def _info_boost(text: str, titles: list[str]) -> float:
    """给"有干货的句子"加权，给"标题党"降权。

    为什么需要：BM25 会按长度做归一化，短句天然占便宜。实测问「作息时间」
    时，排在第一位的是页面标题「深圳职业技术大学作息时间表（试行）」——
    字面最像、信息量为零；真正要的「第1节：8:30—9:15」反而排在后面。
    """
    b = 1.0
    if "｜" in text:                 # 表格行，通常是干货
        b += 0.35
    if re.search(r"\d", text):       # 含数字：时间、电话、日期、节次
        b += 0.15
    if len(text) < 25 and not re.search(r"[0-9。；：:！？]", text):
        b -= 0.5                     # 光秃秃一小句，多半是标题行
    for t in titles:
        if len(t) > 6 and (text == t or (text in t and len(text) > len(t) * 0.6)):
            b -= 0.45                # 跟页面标题几乎一样的，基本是标题行
            break
    return b


# ------------------------------------------------------------------ 对外
def ask_web(question: str, hints=None, budget: float = 8.0) -> dict:
    """去官网找一个问题的答案素材。

    参数
    ----
    question : 学生原话
    hints    : 额外要试的关键词（比如换词后的说法），可空
    budget   : 总预算秒数，到点收手

    返回
    ----
    {
      found   : 有没有拿到能用的素材（passages 或 tables 非空）
      reason  : ok / link_only / no_result / timeout / error
      passages: [{text, url, title, published, score, kind}]   # text 是原文切片
      tables  : [{url, title, rows}]                            # 给页面渲染真表格
      links   : [{url, title}]                                  # 只给链接的情形
      stats   : {keywords, candidates, fetched, elapsed, requests}
      fetched_at
    }
    """
    t0 = time.time()
    q = (question or "").strip()

    if q in _CACHE and time.time() - _CACHE[q][0] < CACHE_TTL:
        hit = json.loads(json.dumps(_CACHE[q][1]))
        hit["stats"]["cached"] = True
        return hit

    res = {
        "found": False, "reason": "no_result",
        "passages": [], "tables": [], "links": [],
        "stats": {"keywords": [], "candidates": 0, "fetched": 0,
                  "elapsed": 0.0, "requests": 0, "cached": False},
        "fetched_at": datetime.now().strftime("%Y-%m-%d %H:%M"),
    }

    def _left() -> float:
        return budget - (time.time() - t0)

    sess = _Session(timeout=min(6.0, max(2.0, budget / 2)))

    try:
        # ---------------------------------------------- ① 搜索（多组词）
        kws = keywords(q, hints)
        res["stats"]["keywords"] = kws
        pool: dict[str, dict] = {}
        for kw in kws:
            if _left() < 1.5:
                res["reason"] = "timeout"
                break
            for it in search_site(sess, kw):
                it["kw"] = kw
                pool.setdefault(it["url"], it)
            if rank_candidates(list(pool.values()), q, kws):
                break                       # 有一组能出候选就够了

        if not pool:
            res["stats"]["elapsed"] = round(time.time() - t0, 1)
            res["stats"]["requests"] = sess.requests
            _CACHE[q] = (time.time(), res)
            return res

        # ---------------------------------------------- ② 筛候选
        cands = rank_candidates(list(pool.values()), q, kws)[:MAX_PAGES]
        res["stats"]["candidates"] = len(pool)
        if not cands:
            res["stats"]["elapsed"] = round(time.time() - t0, 1)
            res["stats"]["requests"] = sess.requests
            _CACHE[q] = (time.time(), res)
            return res

        # ---------------------------------------------- ③ 抓正文
        pages, link_only = [], []
        for it in cands:
            if _left() < 2.0:
                res["reason"] = "timeout"
                break
            try:
                p = fetch_page(sess, it["url"])
            except Exception:
                continue
            p["title"] = p["title"] or it["title"]
            p["strong"] = it.get("strong", True)
            res["stats"]["fetched"] += 1

            # 正文太少／有附件 → 下附件
            if (len(p["text"]) < MIN_BODY or not p["text"]) and p["attachments"]:
                if _left() > 2.5:
                    try:
                        att = p["attachments"][0]
                        raw = sess.get_bytes(att["url"])
                        pdf = read_pdf(raw)
                        if pdf["tables"]:
                            p["tables"].extend(pdf["tables"])
                        if pdf["text"]:
                            p["text"] = (p["text"] + "\n" + pdf["text"]).strip()
                        elif pdf["image_only"]:
                            link_only.append({"url": att["url"],
                                              "title": f"{p['title']}（附件）"})
                    except Exception:
                        pass

            # 页面自己也没料，但内嵌了 PDF → 只给链接
            if len(p["text"]) < MIN_BODY and not p["tables"]:
                if p["attachments"]:
                    link_only.append({"url": _abs(p["attachments"][0]["url"]),
                                      "title": f"{p['title']}（附件）"})
                else:
                    link_only.append({"url": p["url"], "title": p["title"]})
            pages.append(p)

        # ---------------------------------------------- ④ 切句打分
        q_bi = _bigrams(_strip_question(q))
        body_texts: list[tuple[str, str]] = []      # (交给上层的原文, 参与打分的文本)
        table_rows: list[tuple] = []                # (命中数, 表内顺序, 句子, 页面)

        seq = 0
        for p in pages:
            for s in _split_sentences(p["text"]):
                body_texts.append((s, s))
            for rows in p["tables"]:
                if not p.get("strong", True):
                    continue          # 补位页面不取表格，见 rank_candidates 的注释
                rows = [[re.sub(r"\s+", " ", c).strip() for c in row] for row in rows]
                facts = table_facts(rows)
                # 整张表跟问题一个词都不沾就别摆出来（否则「教务处电话」会配上一张
                # 培训通知的表，看着像答了、其实毫无关系）
                if not any(_focus_ok(q, f) and (q_bi & _bigrams(f)) for f in facts):
                    continue
                res["tables"].append({"url": p["url"], "title": p["title"], "rows": rows})
                for fact in facts:
                    table_rows.append((len(q_bi & _bigrams(fact)), seq, fact, p))
                    seq += 1

        # 表格行不交给 BM25：它按长度归一化，越短的行分越高，于是「晚上｜第11、12节」
        # 总压过「上午｜第1节…」这一整行（实测三次都是这样）。表格是结构化的，
        # 直接按"命中问题几个词"排，命中一样就保持表格原顺序（上午在前）。
        table_rows.sort(key=lambda x: (-x[0], x[1]))

        picked = rank_texts(body_texts, q, k=8)
        titles = [p["title"] for p in pages if p.get("title")]
        picked = sorted(picked, key=lambda x: -(x[1] * _info_boost(x[0], titles)))

        table_blob = " ".join(t for _, _, t, _ in table_rows)
        seen: set[str] = set()

        def _push(text, score, page, kind):
            seen.add(text)
            res["passages"].append({
                "text": text, "score": round(float(score), 3),
                "cover": round(_coverage(text, q_bi), 2),   # 覆盖问题几成的词，上层可据此取舍
                "url": page.get("url", ""), "title": page.get("title", ""),
                "published": page.get("published", ""), "kind": kind,
            })

        # 表格行优先（最多 2 条），正文补 1 条
        for _hit, _seq, fact, p in table_rows:
            if len(res["passages"]) >= 2:
                break
            if fact not in seen:
                _push(fact, _hit, p, "table")

        for text, score in picked:
            if len(res["passages"]) >= 3:
                break
            if text in seen or text in table_blob:
                continue          # 表格已经把这段说完，正文别再说一遍
            # 正文句子必须覆盖问题里至少三成的词，否则就是"碰巧撞上一个词"的噪声
            # （实测：问校历，抓到《信息公开报告》里「学校历来重视…」——就撞了「校历」）
            if _coverage(text, q_bi) < 0.34:
                continue
            if not _focus_ok(q, text):
                continue
            _push(text, score, pages[0] if pages else {}, "body")

        # 每条素材归属到它真正来自的页面（卡片要标对出处）
        for ps in res["passages"]:
            for p in pages:
                blob = "\n".join(" ".join(r) for rows in p["tables"] for r in rows)
                if ps["text"] in p["text"] or ps["text"] in blob:
                    ps["url"], ps["title"] = p["url"], p["title"]
                    ps["published"] = p.get("published", "")
                    break

        res["found"] = bool(res["passages"] or res["tables"])
        res["links"] = [] if res["found"] else link_only[:2]
        if res["found"]:
            res["reason"] = "ok"
        elif res["links"]:
            res["reason"] = "link_only"       # 只给链接，不说"识别不出"
        elif res["reason"] != "timeout":
            res["reason"] = "no_result"

    except Exception as e:                    # 官网挂了也不许把整题判死
        res["reason"] = "error"
        res["stats"]["error"] = f"{type(e).__name__}: {e}"

    res["stats"]["elapsed"] = round(time.time() - t0, 1)
    res["stats"]["requests"] = sess.requests
    if res["reason"] == "ok" or res["passages"]:
        _CACHE[q] = (time.time(), res)
    return res


def render_hint(res: dict) -> str:
    """把结果压成一句"给学生的话"。上层可以原样贴，也可以只取链接。"""
    if res.get("reason") == "link_only" and res.get("links"):
        return "这个信息在官网这份文件里，点开就能看：" + res["links"][0]["url"]
    if not res.get("found"):
        return ""
    first = res["passages"][0]
    return f"官网原文（{first['title']}）：{first['text']}"


# ------------------------------------------------------------------ 单跑
if __name__ == "__main__":
    import sys
    from pathlib import Path

    sys.stdout.reconfigure(encoding="utf-8")
    if len(sys.argv) < 2:
        print('用法：python src/retrieval/web_fallback.py "食堂几点开门" [关键词1 关键词2]')
        sys.exit(2)

    root = next((p for p in Path(__file__).resolve().parents if (p / "app.py").exists()), None)
    if root and str(root / "src") not in sys.path:
        sys.path.insert(0, str(root / "src"))
    import _paths                          # noqa: F401  把 retrieval/ 等挂上路径

    q = sys.argv[1]
    hint = sys.argv[2:] or None
    out = ask_web(q, hints=hint)
    print(json.dumps(out, ensure_ascii=False, indent=2)[:3000])
    print("\n—— 给学生的话 ——")
    print(render_hint(out) or "（没有可给的内容）")
