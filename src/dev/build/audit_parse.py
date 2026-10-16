# -*- coding: utf-8 -*-
"""阶段1 解析产物完整性审计。

思路：把 PDF 每一页的每一个"文本行"拿来做归属核对——
只要该行（去空白后）是产出文本池的子串，就认为它被成功回收；
否则列为"丢失行"。同时检查跨页续句、条款覆盖、表格收录。

用法: python src/dev/build/audit_parse.py 手册.pdf data
"""
import json
import re
import sys
from collections import Counter, defaultdict

import pymupdf

norm = lambda s: re.sub(r"[\s\u3000]", "", s or "")
RE_PAGENUM = re.compile(r"^\s*[-—–\s]*\d{1,3}[-—–\s]*$")


def main(pdf_path, data_dir):
    doc = pymupdf.open(pdf_path)
    n_pages = doc.page_count

    clauses = json.load(open(f"{data_dir}/clauses.json", encoding="utf-8"))
    tables = json.load(open(f"{data_dir}/tables.json", encoding="utf-8"))
    files = json.load(open(f"{data_dir}/文件清单.json", encoding="utf-8"))
    problems = json.load(open(f"{data_dir}/解析问题汇总.json", encoding="utf-8"))

    # ---------- 产出文本池 ----------
    # 池 = 条款正文 + 条款的元数据标签(章/节/条标签) + 文件标题 + 表格单元格
    # 章标题、节标题虽不单独成条，但作为 chapter/section 字段随条款保存，
    # 因此必须纳入池，否则会被误判为"丢失"。
    pool_parts = [c["text"] for c in clauses]
    for c in clauses:
        for k in ("file_title", "chapter", "section", "article_label", "part"):
            v = c.get(k)
            if v:
                pool_parts.append(v)
    pool_parts += [f["title"] for f in files]
    pool_parts += [f.get("part") or "" for f in files]
    for t in tables:  # 表格单元格也进池
        for row in t["rows"]:
            pool_parts.append("".join(row))
            pool_parts += list(row)
    pool = norm("".join(pool_parts))

    RE_CH = re.compile(r"^第\s*[一二三四五六七八九十百零〇\d]{1,4}\s*章")
    RE_SEC = re.compile(r"^第\s*[一二三四五六七八九十百零〇\d]{1,4}\s*节")

    # ---------- 1. 逐页逐行归属核对 ----------
    lost = []          # 未回收的行
    page_rec = []      # 每页回收率
    for i in range(1, n_pages + 1):
        raw = doc[i - 1].get_text("text")
        lines = [ln for ln in raw.split("\n")]
        total = hit = 0
        for ln in lines:
            s = norm(ln)
            if not s:
                continue
            if RE_PAGENUM.match(ln):        # 页码行不算
                continue
            total += 1
            if s in pool:
                hit += 1
            else:
                kind = ("章标题" if RE_CH.match(ln.strip())
                        else "节标题" if RE_SEC.match(ln.strip())
                        else "正文")
                lost.append({"page": i, "line": ln.strip(), "chars": len(s),
                             "kind": kind})
        page_rec.append({"page": i, "lines": total, "hit": hit,
                         "rate": round(hit / total * 100, 1) if total else 100.0})

    # ---------- 2. 跨页条款核查 ----------
    arts = [c for c in clauses if c["level"] == "article"]
    multi = [c for c in arts if c["page_start"] != c["page_end"]]
    # 查跨页处是否出现"同一行被拆两半"造成的重复或断裂
    span_chars = 0
    for c in arts:
        span_chars += c["char_len"]

    # ---------- 3. 文件级覆盖 ----------
    file_rec = []
    for f in files:
        fa = [c for c in clauses if c["file_id"] == f["file_id"]
              and c["level"] == "article"]
        got = sum(c["char_len"] for c in fa)
        sp, ep = f["start_page"], f["end_page"]
        if not sp:
            continue
        raw_chars = 0
        for pno in range(sp, min(ep, n_pages) + 1):
            txt = doc[pno - 1].get_text("text")
            for ln in txt.split("\n"):
                s = norm(ln)
                if s and not RE_PAGENUM.match(ln):
                    raw_chars += len(s)
        file_rec.append({"file_id": f["file_id"], "title": f["title"][:24],
                         "pages": f"{sp}-{ep}", "raw": raw_chars, "got": got,
                         "rate": round(got / raw_chars * 100, 1) if raw_chars else 100.0})

    # ---------- 输出 ----------
    print("=" * 66)
    print(f"PDF 物理页 {n_pages}")
    print(f"产出：条款级 {len(clauses)}（条 {len(arts)}）  表 {len(tables)}  文件 {len(files)}")
    print(f"条款文本总字符 {span_chars}")

    bad_pages = [r for r in page_rec if r["rate"] < 100]
    print(f"\n[1] 行级回收：未回收行 {len(lost)} 条，涉及 {len(set(x['page'] for x in lost))} 页")
    print(f"    回收率 <100% 的页：{len(bad_pages)}")
    for r in sorted(bad_pages, key=lambda x: x["rate"])[:25]:
        print(f"      p{r['page']:>3}  {r['hit']}/{r['lines']}  {r['rate']}%")

    cnt = Counter(x["kind"] for x in lost)
    print(f"    分类: {dict(cnt)}")
    body_lost = [x for x in lost if x["kind"] == "正文"]
    print(f"    >>> 真正需要关注的【正文类】丢失行: {len(body_lost)}")

    if body_lost:
        print("\n    正文类丢失行明细:")
        byp = defaultdict(list)
        for x in body_lost:
            byp[x["page"]].append(x)
        for pg in sorted(byp):
            print(f"      --- p{pg} ({len(byp[pg])} 行) ---")
            for x in byp[pg][:8]:
                print(f"        [{x['chars']}字] {x['line'][:76]}")
            if len(byp[pg]) > 8:
                print(f"        ... 另 {len(byp[pg]) - 8} 行")

    print(f"\n[2] 跨页条款：{len(multi)} 条 page_start != page_end")
    multi.sort(key=lambda c: -c["char_len"])
    for c in multi[:8]:
        print(f"      {c['clause_id']}  p{c['page_start']}-{c['page_end']}  {c['char_len']}字  "
              f"{c['article_label'][:22]}")

    r_lo = sorted(file_rec, key=lambda x: x["rate"])
    print(f"\n[3] 文件级字符覆盖（最低 10 份）:")
    for r in r_lo[:10]:
        print(f"      {r['file_id']} p{r['pages']:<9} {r['got']:>6}/{r['raw']:<6} "
              f"{r['rate']:>6}%  {r['title']}")

    print("\n[4] 汇总已知问题:")
    for k, v in problems.items():
        if isinstance(v, list):
            print(f"      {k}: {len(v)}")
        else:
            print(f"      {k}: {v}")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else "data")
