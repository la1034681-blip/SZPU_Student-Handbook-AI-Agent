# -*- coding: utf-8 -*-
"""
版面体检：对 PDF 做逐页体检，产出可核验的解析质量证据。
用法: python pdf_layout_audit.py <pdf> <out.json>
"""
import collections
import json
import re
import sys
from pathlib import Path

import fitz  # PyMuPDF


def audit(pdf_path: Path, out_path: Path) -> dict:
    doc = fitz.open(pdf_path)
    pages = []

    for i, page in enumerate(doc):
        rec = {"page": i + 1}
        try:
            W, H = page.rect.width, page.rect.height
            rec["w"], rec["h"] = round(W, 1), round(H, 1)

            text = page.get_text("text")
            rec["chars"] = len(re.sub(r"\s", "", text))

            blocks = page.get_text("blocks")
            tb = [b for b in blocks if b[6] == 0 and str(b[4]).strip()]
            rec["blocks"] = len(tb)

            cx = [((b[0] + b[2]) / 2.0) for b in tb]
            left = sum(1 for c in cx if c < W * 0.45)
            right = sum(1 for c in cx if c > W * 0.55)
            mid = sum(1 for c in cx if W * 0.35 <= c <= W * 0.65)
            rec["blk_left"], rec["blk_right"], rec["blk_mid"] = left, right, mid

            two_col = False
            if left >= 3 and right >= 3 and len(tb) >= 6:
                lx1 = max(b[2] for b, c in zip(tb, cx) if c < W * 0.5)
                rx0 = min(b[0] for b, c in zip(tb, cx) if c >= W * 0.5)
                gap = rx0 - lx1
                rec["col_gap"] = round(gap, 1)
                two_col = gap > W * 0.025 and mid <= max(2, len(tb) * 0.15)
            rec["two_col"] = two_col

            rec["images"] = len(page.get_images(full=True))
            try:
                rec["tables"] = len(page.find_tables().tables)
            except Exception:
                rec["tables"] = -1
            rec["drawings"] = len(page.get_drawings())

            sizes = collections.Counter()
            d = page.get_text("dict")
            for blk in d.get("blocks", []):
                for line in blk.get("lines", []):
                    for span in line.get("spans", []):
                        t = span.get("text", "")
                        if t.strip():
                            sizes[round(span.get("size", 0), 1)] += len(t)
            rec["top_sizes"] = sizes.most_common(5)

            if tb:
                rec["first_blk"] = re.sub(r"\s+", " ", str(tb[0][4]))[:40]
                rec["last_blk"] = re.sub(r"\s+", " ", str(tb[-1][4]))[:40]
        except Exception as e:
            rec["error"] = repr(e)
        pages.append(rec)

    full = "\n".join(p.get_text("text") for p in doc)

    def cnt(pat):
        m = re.findall(pat, full, re.M)
        return {"total": len(m), "unique": len(set(m))}

    toc_pages = [
        p["page"] for p in pages
        if full and p.get("chars", 0) > 0
        and len(re.findall(r"[\.·…]{3,}", doc[p["page"] - 1].get_text("text"))) >= 5
    ]

    firsts = collections.Counter(p.get("first_blk", "") for p in pages if p.get("first_blk"))
    lasts = collections.Counter(p.get("last_blk", "") for p in pages if p.get("last_blk"))
    n = len(pages)
    heads = [(k, v) for k, v in firsts.most_common(6) if v >= max(3, n * 0.2)]
    feet = [(k, v) for k, v in lasts.most_common(6) if v >= max(3, n * 0.2)]

    result = {
        "file": pdf_path.name,
        "pages_total": n,
        "chars_total": sum(p.get("chars", 0) for p in pages),
        "avg_chars_per_page": round(sum(p.get("chars", 0) for p in pages) / max(n, 1), 1),
        "structure": {
            "章": cnt(r"第\s*[一二三四五六七八九十百零〇\d]{1,4}\s*章"),
            "节": cnt(r"第\s*[一二三四五六七八九十百零〇\d]{1,4}\s*节"),
            "条": cnt(r"第\s*[一二三四五六七八九十百零〇\d]{1,6}\s*条"),
            "款_顿号": cnt(r"^\s*[一二三四五六七八九十]{1,3}、"),
            "款_括号": cnt(r"^\s*[（(][一二三四五六七八九十]{1,3}[)）]"),
            "款_阿拉伯": cnt(r"^\s*\d{1,2}[.、]"),
        },
        "toc_pages": toc_pages,
        "suspect_scanned_or_blank": [p["page"] for p in pages if p.get("chars", 0) < 50],
        "low_text_pages": [p["page"] for p in pages if 50 <= p.get("chars", 0) < 200],
        "two_col_pages": [p["page"] for p in pages if p.get("two_col")],
        "table_pages": [p["page"] for p in pages if p.get("tables", 0) > 0],
        "image_pages": [p["page"] for p in pages if p.get("images", 0) > 0],
        "repeated_headers": heads,
        "repeated_footers": feet,
        "size_histogram": collections.Counter(
            s for p in pages for s, _ in p.get("top_sizes", [])
        ).most_common(),
        "pages": pages,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, ensure_ascii=False, indent=1), encoding="utf-8")
    return result


def main():
    pdf = Path(sys.argv[1])
    out = Path(sys.argv[2]) if len(sys.argv) > 2 else pdf.with_suffix(".audit.json")
    r = audit(pdf, out)
    print(f"文件: {r['file']}")
    print(f"总页数: {r['pages_total']}")
    print(f"总字数(去空白): {r['chars_total']}   平均每页: {r['avg_chars_per_page']}")
    print(f"结构计数: {r['structure']}")
    print(f"疑似目录页: {r['toc_pages']}")
    print(f"疑似空白/扫描页: {len(r['suspect_scanned_or_blank'])} 页 -> {r['suspect_scanned_or_blank'][:30]}")
    print(f"低文字量页: {len(r['low_text_pages'])} 页 -> {r['low_text_pages'][:30]}")
    print(f"疑似双栏页: {len(r['two_col_pages'])} 页 -> {r['two_col_pages'][:30]}")
    print(f"含表格页: {len(r['table_pages'])} 页 -> {r['table_pages'][:30]}")
    print(f"含图片页: {len(r['image_pages'])} 页 -> {r['image_pages'][:30]}")
    print(f"重复页眉候选: {r['repeated_headers']}")
    print(f"重复页脚候选: {r['repeated_footers']}")
    print(f"正文字号分布: {r['size_histogram']}")
    print(f"明细已写入: {out}")


if __name__ == "__main__":
    main()
