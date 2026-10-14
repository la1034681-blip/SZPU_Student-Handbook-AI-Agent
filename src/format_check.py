# -*- coding: utf-8 -*-
"""阶段1 格式体检：查错位、乱序、异常字符、切分粒度异常。"""
import json
import re
import sys
from collections import Counter

data_dir = sys.argv[1] if len(sys.argv) > 1 else "data"
cl = json.load(open(f"{data_dir}/clauses.json", encoding="utf-8"))
arts = [c for c in cl if c["level"] == "article"]
out = []


def p(s=""):
    out.append(str(s))


p("=" * 60)
p(f"条款 {len(arts)} 条")

# 1) 数字后残留空格（影响 BM25 分词）
n_sp = [c for c in arts if re.search(r"\d\s+[\u4e00-\u9fff]", c["text"])]
p(f"\n[1] 含「数字+空格+中文」的条款: {len(n_sp)}")
for c in n_sp[:5]:
    m = re.search(r".{6}\d\s+[\u4e00-\u9fff].{6}", c["text"])
    p(f"    {c['clause_id']}: ...{m.group(0)}...")

# 2) 切分粒度异常
short = sorted([c for c in arts if c["char_len"] < 25], key=lambda x: x["char_len"])
long_ = sorted([c for c in arts if c["char_len"] > 1500], key=lambda x: -x["char_len"])
p(f"\n[2] 超短条款(<25字): {len(short)}")
for c in short[:8]:
    p(f"    {c['clause_id']} {c['char_len']}字 p{c['page_start']} {c['article_label'][:26]!r}")
p(f"    超长条款(>1500字): {len(long_)}")
for c in long_[:8]:
    p(f"    {c['clause_id']} {c['char_len']}字 p{c['page_start']}-{c['page_end']} {c['article_label'][:26]!r}")

# 3) 公式碎片：单字符/2字符成行的密集出现（分式被拆）
frag = []
for c in arts:
    lines = re.findall(r"(?m)^.{0,2}$", c["text"])
    if len(lines) >= 5:
        frag.append(c)
p(f"\n[3] 疑似公式/分式被拆行污染的条款: {len(frag)}")
for c in frag[:6]:
    p(f"    {c['clause_id']} {c['char_len']}字 p{c['page_start']} {c['article_label'][:30]!r}")

# 4) 异常字符
OK = re.compile(r"[\u4e00-\u9fff\u3000-\u303f\uff01-\uff5e\uffe0-\uffe5"
                r"A-Za-z0-9\s,.。、；：！？（）()《》“”‘’\"'【】\-—–…·/％%+×÷≤≥＝=＜＞<>°〔〕［］]")
bad = Counter()
bad_where = {}
for c in arts:
    for ch in c["text"]:
        if not OK.match(ch):
            bad[ch] += 1
            bad_where.setdefault(ch, c["clause_id"])
p(f"\n[4] 异常字符: {dict(bad)}")
for ch, n in bad.most_common(12):
    p(f"    {ch!r} U+{ord(ch):04X} x{n}  首次出现于 {bad_where[ch]}")

# 5) 条级与款级一致性：条级 text 是否覆盖其所有款
miss = []
for c in arts:
    for u in c["clauses"]:
        t = u["text"]
        if t and t[:20] not in c["text"]:
            miss.append((c["clause_id"], u["clause_id"], t[:24]))
p(f"\n[5] 款内容未包含在条正文中: {len(miss)}")
for m in miss[:8]:
    p(f"    {m}")

# 6) 空条款 / 只有标签的条款
only_label = [c for c in arts if c["text"].strip() == c["article_label"].strip()]
p(f"\n[6] 正文仅等于标签行的条款: {len(only_label)}")
for c in only_label[:10]:
    p(f"    {c['clause_id']} p{c['page_start']} {c['article_label'][:34]!r}")

# 7) 无条号容器与无章文件
ax = [c for c in cl if c["article_no"] is None and c["level"] == "article"]
p(f"\n[7] 无条号容器 {len(ax)} 条:")
for c in ax:
    p(f"    {c['clause_id']} p{c['page_start']} {c['char_len']}字 {c['article_label']} | {c['text'][:40]!r}")

open(f"{data_dir}/_fmt.txt", "w", encoding="utf-8").write("\n".join(out))
print("ok ->", f"{data_dir}/_fmt.txt")
