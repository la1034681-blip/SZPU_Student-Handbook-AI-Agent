# -*- coding: utf-8 -*-
"""验证：条编号是否在章内重新计数（决定条款主键设计）。"""
import re
import sys
from pathlib import Path

import pymupdf

pdf = Path(sys.argv[1])
doc = pymupdf.open(pdf)
seq = []
for i, page in enumerate(doc):
    t = page.get_text("text")
    for m in re.finditer(r"第\s*([一二三四五六七八九十百零〇\d]{1,6})\s*(章|节|条)", t):
        seq.append((i + 1, m.group(2), m.group(1)))

zhang = [(p, n) for p, k, n in seq if k == "章"]
tiao = [(p, n) for p, k, n in seq if k == "条"]

print(f"章标记出现 {len(zhang)} 次：")
print("  " + "  ".join(f"p{p}:第{n}章" for p, n in zhang[:20]))
print(f"条标记出现 {len(tiao)} 次，前 40 个（页, 条号）：")
print("  " + "  ".join(f"p{p}·{n}" for p, n in tiao[:40]))

cn = "零一二三四五六七八九十百"
def zh2int(s):
    if s.isdigit():
        return int(s)
    d = {c: i for i, c in enumerate(cn)}
    if s == "十":
        return 10
    if "十" in s:
        a, _, b = s.partition("十")
        return (d.get(a, 1) if a else 1) * 10 + (d.get(b, 0) if b else 0)
    return d.get(s, -1)

nums = [zh2int(n) for _, n in tiao]
desc = sum(1 for a, b in zip(nums, nums[1:]) if b < a)
print(f"条号出现回落次数（说明重新计数）: {desc} / {max(len(nums)-1,1)}")
