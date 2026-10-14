# -*- coding: utf-8 -*-
"""导出 PDF 全文（带页标记），供校准结构解析规则。
用法: python dump_fulltext.py <pdf> <out.txt>
"""
import sys
from pathlib import Path

import pymupdf


def main():
    pdf = Path(sys.argv[1])
    out = Path(sys.argv[2])
    doc = pymupdf.open(pdf)
    buf = []
    for i, page in enumerate(doc):
        buf.append(f"\n<<< PAGE {i + 1} >>>\n")
        buf.append(page.get_text("text"))
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("".join(buf), encoding="utf-8")
    print(f"pages={len(doc)} -> {out}")


if __name__ == "__main__":
    main()
