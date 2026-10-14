# -*- coding: utf-8 -*-
"""
阶段 1：PDF 解析与条款结构化
- 目录页解析：拿到「部分 / 文件 / 印刷页码」
- 正文定位：用目录页码 + 全局页码偏移量校验定位每份文件的起始页
- 结构切分：文件 > 章 > 节 > 条 > 款
- 表格页回退：pymupdf 判定表格区域并剔除，pdfplumber 重新抽取表格
输出：data/文件清单.json, data/clauses.json, data/chunks.jsonl,
      data/tables.json, data/解析问题汇总.json
用法: python parse_clauses.py <pdf> <audit.json> <outdir>
"""
import json
import re
import sys
from collections import Counter
from pathlib import Path

import pdfplumber
import pymupdf

CN = "零一二三四五六七八九十百"
RE_PART = re.compile(r"^第([一二三四五六七八九十]+)部分\s*(.*)$")
RE_TOC = re.compile(r"^([一二三四五六七八九十]+)、(.+?)[.…·]{1,}\s*(\d+)$")
RE_CHAPTER = re.compile(r"^第\s*([一二三四五六七八九十百零〇\d]{1,4})\s*章\s*(.*)$")
RE_SECTION = re.compile(r"^第\s*([一二三四五六七八九十百零〇\d]{1,4})\s*节\s*(.*)$")
RE_ARTICLE = re.compile(r"^第\s*([一二三四五六七八九十百零〇\d]{1,6})\s*条\s*(.*)$")
RE_KUAN_PAREN = re.compile(r"^[（(]\s*([一二三四五六七八九十]{1,3}|\d{1,2})\s*[)）]\s*(.*)$")
RE_KUAN_DUN = re.compile(r"^([一二三四五六七八九十]{1,3})\s*、\s*(.*)$")
RE_KUAN_ARABIC = re.compile(r"^(\d{1,2})\s*[.、]\s*(\S.*)$")
RE_PAGENUM = re.compile(r"^\d{1,3}$")
# 附件标题：既可能是「附件1、…」「附件：1.…」，也可能是「附件：学生信息资料更改申请表」
# 这类纯中文标题，因此不能要求冒号后必须是数字。用行长度(<=40字)兜住误匹配风险。
RE_APPENDIX = re.compile(r"^附\s*件\s*[：:]?\s*\S")
APPENDIX_MAX_LEN = 40

# 少数附件表单页的文字在 PDF 里是逐字符倒排的（"班级名称"→"称名级班"），
# pdfplumber 会原样读出乱码，需要用这些倒序特征词识别并反转还原。
REV_MARKERS = ("称名", "号序", "级班", "名姓", "号学", "年学", "期学", "院学",
               "果成", "书证", "请申", "修必", "选任", "别类", "止截", "核审",
               "报申", "导领", "存留", "份一")


def looks_reversed(rows):
    """判断一整张表是否为倒序文字。"""
    tot = hit = 0
    for r in rows:
        for c in r:
            s = norm(c)
            if len(s) < 2:
                continue
            tot += 1
            if any(m in s for m in REV_MARKERS):
                hit += 1
    return tot >= 3 and hit / tot >= 0.3


def unreverse(rows):
    """逐字符反转，还原倒排文字。"""
    return [[(c or "")[::-1] for c in row] for row in rows]

norm = lambda s: re.sub(r"[\s\u3000]", "", s or "")

_PUNCT = str.maketrans({"（": "(", "）": ")", "［": "[", "］": "]",
                        "｛": "{", "｝": "}", "“": "", "”": "",
                        "‘": "", "’": "", "《": "", "》": ""})


def nkey(s):
    """用于标题比对的宽松键：去空白 + 全半角括号统一 + 去引号书名号。"""
    return norm(s).translate(_PUNCT)


def cn2int(s):
    if s is None:
        return None
    s = str(s).strip()
    if s.isdigit():
        return int(s)
    d = {c: i for i, c in enumerate(CN)}
    if "十" not in s and "百" not in s:
        return d.get(s)
    section, num = 0, 0
    for ch in s:
        if ch not in d:
            return None
        v = d[ch]
        if v >= 10:
            section += (num or 1) * v
            num = 0
        else:
            num = v
    return section + num


def parse_toc(doc):
    raw = doc[3].get_text("text") + "\n" + doc[4].get_text("text")
    part, buf, files = None, "", []
    for ln in raw.split("\n"):
        ln = ln.strip()
        if not ln or ln.startswith("目"):
            continue
        m = RE_PART.match(ln)
        if m:
            part = f"第{m.group(1)}部分 {m.group(2).strip()}"
            buf = ""
            continue
        buf += ln
        m2 = RE_TOC.match(buf)
        if m2:
            files.append({"part": part, "seq": m2.group(1),
                          "title": m2.group(2).strip(),
                          "printed_page": int(m2.group(3))})
            buf = ""
    leftovers = buf.strip()
    return files, leftovers


def table_boxes(page):
    try:
        return [pymupdf.Rect(t.bbox) for t in page.find_tables().tables]
    except Exception:
        return []


def block_lines(page, boxes):
    out = []
    for b in sorted(page.get_text("blocks"), key=lambda x: (round(x[1], 1), x[0])):
        if b[6] != 0:
            continue
        txt = str(b[4]).strip()
        if not txt:
            continue
        r = pymupdf.Rect(b[0], b[1], b[2], b[3])
        area = max(r.width * r.height, 1e-6)
        drop = False
        for bx in boxes:
            inter = r & bx
            ia = 0.0 if (inter is None or inter.is_empty) else inter.get_area()
            if ia / area > 0.5:
                drop = True
                break
        if not drop:
            out.extend(x.strip() for x in txt.split("\n") if x.strip())
    return out


def plumber_tables(pdf_path, page_index):
    """pdfplumber 抽表；为空则回退 pymupdf 的 table.extract()。"""
    got = []
    try:
        with pdfplumber.open(pdf_path) as pl:
            for t in pl.pages[page_index].extract_tables() or []:
                rows = [[(c or "").replace("\n", " ").strip() for c in row] for row in t]
                rows = [r for r in rows if any(r)]
                if rows:
                    got.append(rows)
    except Exception:
        pass
    if got:
        return got, "pdfplumber"
    try:
        d = pymupdf.open(pdf_path)
        for t in d[page_index].find_tables().tables:
            rows = [[(c or "").replace("\n", " ").strip() for c in row] for row in t.extract()]
            rows = [r for r in rows if any(r)]
            if rows:
                got.append(rows)
        d.close()
    except Exception:
        pass
    return got, "pymupdf_fallback"


def classify(line, mode):
    """mode: tiao / dun / paren / arabic —— 该文件里哪一层算「条级」。"""
    m = RE_CHAPTER.match(line)
    if m:
        return "chapter", cn2int(m.group(1)), line
    m = RE_SECTION.match(line)
    if m:
        return "section", cn2int(m.group(1)), line
    m = RE_ARTICLE.match(line)
    if m:
        return "article", cn2int(m.group(1)), line
    for tag, rx in (("paren", RE_KUAN_PAREN), ("dun", RE_KUAN_DUN),
                    ("arabic", RE_KUAN_ARABIC)):
        m = rx.match(line)
        if m:
            if tag == mode:
                return "article", cn2int(m.group(1)), line
            prefix = line[:m.start(2)].strip()
            return "clause", prefix, line
    return "body", None, line


def pick_mode(flat, has_article):
    """文件内只把最外层的一种序号提为「条级」，其余降为款。"""
    if has_article:
        return "tiao"
    lines = [t for k, _, t in flat if k == "line"]
    if any(RE_KUAN_DUN.match(t) for t in lines):
        return "dun"
    for t in lines:
        m = RE_KUAN_PAREN.match(t)
        if m and not m.group(1).isdigit():
            return "paren"
    return "arabic"


def main():
    pdf_path = Path(sys.argv[1])
    audit_path = Path(sys.argv[2])
    outdir = Path(sys.argv[3])
    outdir.mkdir(parents=True, exist_ok=True)

    problems = {}
    doc = pymupdf.open(pdf_path)
    n_pages = len(doc)
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    table_pages = set(audit.get("table_pages", []))

    files, toc_leftover = parse_toc(doc)
    problems["toc_unparsed_tail"] = toc_leftover

    page_nkey = [nkey(p.get_text("text")) for p in doc]
    offs = []
    for f in files:
        t = nkey(f["title"])
        f["_cands"] = [i + 1 for i in range(5, n_pages) if t and t in page_nkey[i]]
        offs.extend(p - f["printed_page"] for p in f["_cands"])
    offset = Counter(offs).most_common(1)[0][0] if offs else 0
    problems["page_offset"] = offset
    problems["page_offset_distribution"] = Counter(offs).most_common(8)

    bad_loc = []
    for f in files:
        exp = f["printed_page"] + offset
        f["start_page"] = exp if exp in f["_cands"] else (min(f["_cands"]) or None
                                                          if f["_cands"] else None)
        if f["start_page"] != exp:
            bad_loc.append({"title": f["title"], "expect": exp,
                            "actual": f["start_page"], "cands": f["_cands"][:5]})
        f.pop("_cands", None)
    problems["file_locate_mismatch"] = bad_loc

    for i, f in enumerate(files):
        f["file_id"] = f"F{i + 1:02d}"
        nxt = next((g["start_page"] for g in files[i + 1:] if g.get("start_page")), None)
        f["end_page"] = (nxt - 1) if nxt else n_pages

    clauses, chunks, tables, per_file = [], [], [], []
    reversed_pages = set()

    for f in files:
        fid, title = f["file_id"], f["title"]
        sp, ep = f.get("start_page"), f.get("end_page")
        if not sp:
            per_file.append({"file": fid, "title": title, "error": "起始页未定位"})
            continue

        title_key = nkey(title)
        flat = []
        for pi in range(sp - 1, min(ep, n_pages)):
            page = doc[pi]
            ts, src = [], None
            if pi + 1 in table_pages:
                ts, src = plumber_tables(pdf_path, pi)
                fixed = []
                for rows in ts:
                    if looks_reversed(rows):
                        rows = unreverse(rows)
                        reversed_pages.add(pi + 1)
                    fixed.append(rows)
                ts = fixed
            # 只有表格结构化结果确实可用时，才剔除表格区域内的文本块并用表格替代；
            # 若该页抽不出表（空白表单页 / 只有表头），则保留原始文本行，避免丢字。
            boxes = table_boxes(page) if ts else []
            for ln in block_lines(page, boxes):
                nk = nkey(ln)
                if not nk or RE_PAGENUM.match(ln):
                    continue
                if nk in title_key or nk in ("(试行)", "试行"):
                    continue
                flat.append(("line", pi + 1, ln))
            for k, rows in enumerate(ts):
                flat.append(("table", pi + 1, {"rows": rows, "src": src, "idx": k}))

        has_article = any(k == "line" and RE_ARTICLE.match(t) for k, _, t in flat)
        mode = pick_mode(flat, has_article)

        chapter = section = None
        art, art_lines, kuan = None, [], None
        arts, file_tables = [], []
        chapters, art_nums = [], []
        kuan_counter = 0
        in_appendix, app_lines, ax_counter = False, [], 0

        def close_kuan():
            nonlocal kuan
            if kuan is not None:
                kuan["text"] = "".join(kuan.pop("_buf")).strip()
                if art is not None:
                    art.setdefault("clauses_units", []).append(kuan)
                kuan = None

        def close_art():
            nonlocal art, art_lines
            if art is not None:
                close_kuan()
                art["text"] = "".join(art_lines).strip()
                arts.append(art)
            art, art_lines = None, []

        def ensure_art(pg):
            """没有条号时兜底建一条容器，避免章标题与首条之间的正文被丢弃。"""
            nonlocal art, art_lines, ax_counter, kuan_counter
            if art is None:
                ax_counter += 1
                art = {"file_id": fid, "file_title": title, "part": f["part"],
                       "chapter": chapter, "section": section, "no": None,
                       "ax": ax_counter, "label": f"（无条号·{ax_counter}）",
                       "page": pg, "_pages": {pg}}
                art_lines = []
                kuan_counter = 0

        for kind, pg, payload in flat:
            if in_appendix:
                if kind == "table":
                    app_lines.append((pg, "［表格］" + " ".join(
                        " ".join(r) for r in payload["rows"])))
                else:
                    app_lines.append((pg, payload))
                continue

            if kind == "table":
                trec = {"table_id": f"{fid}-T{len(file_tables) + 1:02d}",
                        "file_id": fid, "page": pg, "chapter": chapter,
                        "section": section,
                        "article_label": art["label"] if art else None,
                        "rows": payload["rows"], "source": payload["src"]}
                file_tables.append(trec)
                if art is not None:
                    art.setdefault("tables", []).append(trec["table_id"])
                    art.setdefault("_pages", set()).add(pg)
                    art_lines.append("［表格］" + " ".join(
                        " ".join(r) for r in payload["rows"]))
                continue

            if RE_APPENDIX.match(payload) and len(nkey(payload)) <= APPENDIX_MAX_LEN:
                close_art()
                in_appendix = True
                app_lines.append((pg, payload))
                continue

            tag, num, raw = classify(payload, mode)
            if tag == "chapter":
                close_art()
                chapter, section = raw, None
                chapters.append({"no": num, "label": raw, "page": pg})
            elif tag == "section":
                close_art()
                section = raw
            elif tag == "article":
                close_art()
                art_nums.append(num)
                kuan_counter = 0
                art = {"file_id": fid, "file_title": title, "part": f["part"],
                       "chapter": chapter, "section": section,
                       "no": num, "label": raw, "page": pg, "_pages": {pg}}
                art_lines = [raw]
            elif tag == "clause":
                ensure_art(pg)
                close_kuan()
                kuan_counter += 1
                kuan = {"clause_no": kuan_counter, "label": raw, "by": tag,
                        "page": pg, "_buf": []}
                art_lines.append(raw)
            else:
                ensure_art(pg)
                if kuan is not None:
                    kuan["_buf"].append(raw)
                art_lines.append(raw)

            if art is not None:
                art.setdefault("_pages", set()).add(pg)

        close_art()
        f["chapters"] = chapters

        for a in arts:
            units = a.get("clauses_units") or []
            pgs = sorted(a.get("_pages") or [a["page"]])
            cid = (f"{fid}-A{a['no']:03d}" if a["no"] is not None
                   else f"{fid}-AX{a.get('ax', 0):02d}")
            rec = {"clause_id": cid, "level": "article",
                   "file_id": fid, "file_title": title, "part": a["part"],
                   "chapter": a["chapter"], "section": a["section"],
                   "article_no": a["no"], "article_label": a["label"],
                   "text": a["text"], "page_start": pgs[0], "page_end": pgs[-1],
                   "table_refs": a.get("tables", []), "clauses": [],
                   "char_len": len(a["text"])}
            for u in units:
                ucid = f"{cid}-C{u['clause_no']:02d}"
                rec["clauses"].append({"clause_id": ucid, "clause_no": u["clause_no"],
                                       "label": u["label"], "by": u["by"],
                                       "text": u["text"], "page": u["page"]})
                if u["text"]:
                    chunks.append({"chunk_id": ucid, "clause_id": cid,
                                   "parent_id": cid, "chunk_type": "clause",
                                   "file_id": fid, "chapter": a["chapter"],
                                   "article_label": a["label"], "page": u["page"],
                                   "text": u["label"] + u["text"]})
            clauses.append(rec)
            chunks.append({"chunk_id": cid, "clause_id": cid, "parent_id": None,
                           "chunk_type": "article", "file_id": fid,
                           "chapter": a["chapter"], "article_label": a["label"],
                           "page": pgs[0], "text": a["text"]})

        if app_lines:
            atext = "".join(t for _, t in app_lines).strip()
            apg = [p for p, _ in app_lines]
            if atext:
                aid = f"{fid}-APP"
                clauses.append({"clause_id": aid, "level": "appendix",
                                "file_id": fid, "file_title": title,
                                "part": f["part"], "chapter": None,
                                "section": None, "article_no": None,
                                "article_label": "附件/表单", "text": atext,
                                "page_start": min(apg), "page_end": max(apg),
                                "table_refs": [], "clauses": [],
                                "char_len": len(atext)})
                chunks.append({"chunk_id": aid, "clause_id": None,
                               "parent_id": None, "chunk_type": "appendix",
                               "file_id": fid, "chapter": None,
                               "article_label": "附件/表单", "page": min(apg),
                               "text": atext})

        for t in file_tables:
            if not t["rows"]:
                continue
            txt = "［表格］" + "；".join("，".join(c for c in r if c) for r in t["rows"])
            chunks.append({"chunk_id": t["table_id"], "clause_id": None,
                           "parent_id": None, "chunk_type": "table",
                           "file_id": fid, "chapter": t["chapter"],
                           "article_label": t["article_label"],
                           "page": t["page"], "text": txt})
        tables.extend(file_tables)

        seq = [n for n in art_nums if n is not None]
        breaks = [b for a, b in zip(seq, seq[1:]) if b != a + 1]
        per_file.append({"file": fid, "title": title, "part": f["part"],
                         "pages": f"{sp}-{ep}", "mode": mode,
                         "chapters": len(chapters),
                         "articles": len(arts), "tables": len(file_tables),
                         "seq_start": seq[0] if seq else None,
                         "seq_end": seq[-1] if seq else None,
                         "continuity_breaks": len(breaks),
                         "has_chapter": bool(chapters)})

    problems["files_report"] = per_file
    problems["files_without_chapter"] = [r["file"] for r in per_file if not r.get("has_chapter")]
    problems["files_with_seq_breaks"] = [
        {"file": r["file"], "title": r["title"], "breaks": r["continuity_breaks"]}
        for r in per_file if r.get("continuity_breaks", 0) > 0]
    app_pages = set()
    for c in clauses:
        if c["level"] == "appendix" and c["page_start"]:
            app_pages.update(range(c["page_start"], c["page_end"] + 1))
    got_pages = {t["page"] for t in tables}
    problems["appendix_page_ranges"] = sorted(app_pages)
    problems["table_pages_without_extract"] = sorted(
        table_pages - got_pages - app_pages)
    problems["tables_text_reversed_pages"] = sorted(reversed_pages)
    problems["articles_without_number"] = sum(
        1 for c in clauses if c["level"] == "article" and c["article_no"] is None)
    problems["appendix_records"] = sum(1 for c in clauses if c["level"] == "appendix")
    problems["total_articles"] = len(clauses)
    problems["total_chunks"] = len(chunks)
    problems["total_tables"] = len(tables)

    (outdir / "文件清单.json").write_text(json.dumps(files, ensure_ascii=False, indent=1), encoding="utf-8")
    (outdir / "clauses.json").write_text(json.dumps(clauses, ensure_ascii=False, indent=1), encoding="utf-8")
    with (outdir / "chunks.jsonl").open("w", encoding="utf-8") as fh:
        for c in chunks:
            fh.write(json.dumps(c, ensure_ascii=False) + "\n")
    (outdir / "tables.json").write_text(json.dumps(tables, ensure_ascii=False, indent=1), encoding="utf-8")
    (outdir / "解析问题汇总.json").write_text(json.dumps(problems, ensure_ascii=False, indent=1), encoding="utf-8")

    print(f"文件数 {len(files)}  页码偏移 {offset}")
    print(f"条级条款 {len(clauses)}  检索块 {len(chunks)}  表格 {len(tables)}")
    print(f"无章结构文件 {len(problems['files_without_chapter'])}  条号不连续 {len(problems['files_with_seq_breaks'])}  定位异常 {len(bad_loc)}")
    print(f"表格页未抽出 {problems['table_pages_without_extract']}")


if __name__ == "__main__":
    main()
