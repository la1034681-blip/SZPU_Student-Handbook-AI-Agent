# -*- coding: utf-8 -*-
"""文本规范化：把 PDF 解析出的原始文本整理成可入库、可检索的文本。

为什么必须在切块入库前做：
- 「第41 号令」这类数字后空格会把 BM25 的「41号令」切碎，造成检索漏词；
- Symbol 字体残留的私用区字符（U+E000-F8FF）是乱码，进向量等于噪声。

规则（只做形式整理，不改写语义、不补全内容）：
 1. 清除私用区字符（Σ、×、项目符号等的字体映射残留）
 2. 修复「数字+空格+中文」「中文+空格+数字」的断词
 3. 中文语境下半角标点转全角（; : , ! ?），保护 URL 与数字千分位
 4. 全角化「内含中文的半角括号」
 5. 合并多余空白、去掉中文括号内侧空格
 6. 保留解析器加入的结构标记「［表格］」
"""
import re

PUA = re.compile(r"[\ue000-\uf8ff]")
URL = re.compile(r"(?:https?://|www\.)[^\s，。；）)】、;,，；：]+")
CJK = r"\u4e00-\u9fff"

# 已知的 Symbol 字体映射残留 → 正确字符
PUA_MAP = {
    "\uf0e5": "∑", "\uf0e1": "∑", "\uf0f1": "∑",   # summation
    "\uf0b4": "×", "\uf0b7": "·", "\uf0b1": "±",
    "\uf0a3": "≤", "\uf0b3": "≥", "\uf0b9": "≠",
    "\uf0d6": "√", "\uf0b5": "∞", "\uf0ae": "→",
}
# 列表项目符号类残留，直接删除（保留后续文字）
PUA_DROP = {"\uf06a", "\uf06b", "\uf06c", "\uf0a7", "\uf0d8", "\uf0fc", "\uf0a8"}


def protect_urls(s):
    store = []

    def rep(m):
        store.append(m.group(0))
        return "\x00%d\x00" % (len(store) - 1)

    return URL.sub(rep, s), store


def restore_urls(s, store):
    for i, u in enumerate(store):
        s = s.replace("\x00%d\x00" % i, u)
    return s


def normalize(s):
    if not s:
        return ""
    s, store = protect_urls(s)

    # 1) 私用区字符
    s = "".join(PUA_MAP.get(ch, "" if ch in PUA_DROP else "") if PUA.match(ch) else ch
                for ch in s)
    s = PUA.sub("", s)

    # 2) 数字/中文之间的断词空格
    s = re.sub(r"(?<=[%s])[ \t]+(?=[%s])" % (CJK, CJK), "", s)
    s = re.sub(r"(?<=\d)[ \t]+(?=[%s])" % CJK, "", s)
    s = re.sub(r"(?<=[%s])[ \t]+(?=\d)" % CJK, "", s)
    s = re.sub(r"(?<=\d)[ \t]+(?=[，。、；：）】%])", "", s)

    # 3) 全角化内含中文的半角括号
    s = re.sub(r"\(([^()]*[%s][^()]*)\)" % CJK, r"（\1）", s)
    s = re.sub(r"\[([^\[\]]*[%s][^\[\]]*)\]" % CJK, r"［\1］", s)

    # 4) 括号内侧空格
    s = re.sub(r"([（【［])[ \t]+", r"\1", s)
    s = re.sub(r"[ \t]+([）】］])", r"\1", s)

    # 5) 半角标点转全角
    s = re.sub(r"[ \t]*;[ \t]*", "；", s)
    s = re.sub(r"(?<!\d)[ \t]*,[ \t]*(?!\d)", "，", s)
    s = re.sub(r"(?<=[%s])[ \t]*:[ \t]*" % CJK, "：", s)
    s = re.sub(r"(?<=[%s])!" % CJK, "！", s)
    s = re.sub(r"(?<=[%s])\?" % CJK, "？", s)

    # 6) 空白合并
    s = re.sub(r"[ \t\u3000]{2,}", " ", s)
    s = re.sub(r"[ \t]+(?=[，。、；：）】］])", "", s)

    # 7) 编号与后文之间补一个空格。
    #    第 2 步会把「第一条  学生考勤时间」中的间隔一并删掉，读起来会粘成
    #    「第一条学生考勤时间」，既影响可读性也让分词粘连，这里统一补回。
    #    必须放在空白合并之后，否则又会被第 2 步删掉。
    s = re.sub(r"^(第\s*[一二三四五六七八九十百零〇\d]{1,6}\s*条)(?=[^\s])", r"\1 ", s)
    s = re.sub(r"^(第\s*[一二三四五六七八九十百零〇\d]{1,4}\s*[章节节])\s*", r"\1 ", s)
    s = re.sub(r"^(\d{1,2}\s*[.、])(?=[^\s])", r"\1 ", s)
    s = re.sub(r"^([（(]\s*[一二三四五六七八九十\d]{1,3}\s*[)）])(?=[^\s])", r"\1 ", s)
    return restore_urls(s, store).strip()


# —— 破损公式检测：只认「运算符号连排」这一种特征 ——
#    被拍平的分式会出现 ΣΣ×Σ 这类符号堆叠；而正常公式是「Σ(课程学分×课程绩点)」，
#    符号之间隔着变量名，不会连排。用这个特征可以避免把修好的公式误报成破损。
BROKEN_FORMULA = re.compile(r"[∑×÷±≤≥]{2,}")


def flag_broken_formula(s):
    """返回文本中疑似被 PDF 拍平的公式片段（仅用于统计与标记，不改写）。"""
    return BROKEN_FORMULA.findall(s)


if __name__ == "__main__":
    tests = [
        "创建于1993 年的深圳高等职业技术学院",
        "第41 号令《普通高等学校学生管理规定》",
        "考生违规记录应当由2 名以上监考员签字确认",
        "学生可申请免听其重修课程的部分学时;但必须征得任课教师同意",
        "GPA 的计算方法为:课程成绩为 A+、A、A-",
        "学校中文网址为http://www.szpu.edu.cn;英文网址为http://english.szpu.edu.cn。",
        "（ http://xjxl.chsi.com.cn/）上进行注册",
        "平均课程绩点\uf0e5\uf0e1×\uf0f1课程学分绩点课程学分",
        "申请发明专利1 项。  \uf06b自主知识产权1 个。",
        "所修课程成绩在1,000 元以上(含)",
    ]
    for t in tests:
        print(repr(t), "->", repr(normalize(t)))
