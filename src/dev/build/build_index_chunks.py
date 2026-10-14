# -*- coding: utf-8 -*-
"""阶段 2 第一步：生成入库用的检索块（index_chunks.jsonl）。

在阶段 1 产出的 clauses.json / chunks.jsonl 基础上做四件事：

 1. 规范化文本（normalize）——清理数字断词空格、私用区字符、半角标点
 2. 应用公式重建补丁（fix_formula）——修复被 PDF 拍平的分式
 3. 生成「文件标题 + 章标题」上下文前缀 —— Contextual Retrieval 思路：
    短条款（如「第四十八条 本规定由教务处负责解释。」）单看毫无检索价值，
    补上所属文件与章之后才能被正确召回
 4. 生成引用标签 cite —— 供答案引用与前端展示，形如
    《深圳职业技术大学学生考勤管理规定》第一章 第一条

用法: python src/build_index_chunks.py data
输出: data/index_chunks.jsonl, data/clause_index.json, data/index_meta.json
"""
import hashlib
import json
import re
import sys
from datetime import datetime

from fix_formula import apply_patches
from normalize import normalize, flag_broken_formula

RE_ART_NO = re.compile(r"^(第\s*[一二三四五六七八九十百零〇\d]{1,6}\s*条)")
RE_DUN_NO = re.compile(r"^([一二三四五六七八九十]{1,3}\s*、)")
RE_PAREN_NO = re.compile(r"^([（(]\s*[一二三四五六七八九十\d]{1,3}\s*[)）])")


def make_cite(file_title, chapter, article_label):
    """生成可读的引用标签。"""
    parts = [f"《{file_title}》"]
    if chapter:
        parts.append(chapter)
    lab = (article_label or "").strip()
    if lab and not lab.startswith("（无条号"):
        m = RE_ART_NO.match(lab) or RE_DUN_NO.match(lab) or RE_PAREN_NO.match(lab)
        parts.append(m.group(1).replace("  ", " ") if m else lab[:24])
    return " ".join(parts)


def make_prefix(part, file_title, chapter, article_label, chunk_type):
    """检索用上下文前缀。"""
    head = f"《{file_title}》"
    if part:
        head = f"{part}｜{head}"
    if chapter:
        head += f" {chapter}"
    if chunk_type == "clause" and article_label:
        head += f" {article_label[:20]}"
    elif chunk_type == "appendix":
        head += " 附件/表单"
    return head


def main(data_dir="data"):
    clauses = json.load(open(f"{data_dir}/clauses.json", encoding="utf-8"))
    cmap = {c["clause_id"]: c for c in clauses}

    chunks = [json.loads(l) for l in open(f"{data_dir}/chunks.jsonl", encoding="utf-8")]
    # 表格块与附录块的元数据分别落在 tables.json / clauses.json，需要按 chunk_id 回查
    tmap = {t["table_id"]: t for t in json.load(
        open(f"{data_dir}/tables.json", encoding="utf-8"))}
    fmap, pmap = {}, {}
    for c in clauses:
        fmap.setdefault(c["file_id"], c["file_title"])
        pmap.setdefault(c["file_id"], c.get("part"))

    out, fmt_hits, broken = [], {}, 0
    for ch in chunks:
        # 附录/表格块的 clause_id 为空，需要用自身 chunk_id 回查元数据
        cid = ch.get("clause_id") or ch.get("parent_id")
        if cid not in cmap:
            cid = ch["chunk_id"] if ch["chunk_id"] in cmap else cid
        src = cmap.get(cid, {})
        if not src and ch["chunk_id"] in tmap:
            t = tmap[ch["chunk_id"]]
            src = {"file_id": t["file_id"], "file_title": fmap.get(t["file_id"], ""),
                   "part": pmap.get(t["file_id"]), "chapter": t.get("chapter"),
                   "article_label": t.get("article_label"),
                   "page_start": t.get("page"), "page_end": t.get("page")}
        file_title = src.get("file_title", "")
        part = src.get("part")
        chapter = src.get("chapter")
        label = ch.get("article_label") or src.get("article_label") or ""

        # 1) 规范化
        text = normalize(ch.get("text", ""))

        # 2) 公式补丁（按所属条款 id）
        if cid:
            text, hits = apply_patches(cid, text)
            if hits:
                fmt_hits.setdefault(cid, []).extend(hits)
        if flag_broken_formula(text):
            broken += 1

        # 3) 上下文前缀
        prefix = make_prefix(part, file_title, chapter, label, ch["chunk_type"])
        embed_text = f"{prefix}\n{text}" if text else prefix

        rec = {
            "chunk_id": ch["chunk_id"],
            "chunk_type": ch["chunk_type"],
            "parent_id": ch.get("parent_id"),
            "clause_id": cid,
            "file_id": ch.get("file_id") or src.get("file_id"),
            "file_title": file_title,
            "part": part,
            "chapter": chapter,
            "article_no": src.get("article_no"),
            "article_label": label,
            "cite": make_cite(file_title, chapter, label),
            "page_start": src.get("page_start") or ch.get("page"),
            "page_end": src.get("page_end") or ch.get("page"),
            "page": ch.get("page"),
            "text": text,
            "embed_text": embed_text,
            "char_len": len(text),
            "patched": fmt_hits.get(cid, []),
        }
        out.append(rec)

    # —— 条款编号直达索引：把「文件+条号」映射到块，供精确检索用 ——
    clause_index = {}
    for r in out:
        if r["chunk_type"] != "article" or r["article_no"] is None:
            continue
        key = f"{r['file_id']}|{r['article_no']}"
        clause_index[key] = r["chunk_id"]

    texts = [r["text"] for r in out]
    meta = {
        "built_at": datetime.now().isoformat(timespec="seconds"),
        "source": "data/clauses.json + data/chunks.jsonl",
        "chunks": len(out),
        "by_type": {t: sum(1 for r in out if r["chunk_type"] == t)
                    for t in sorted({r["chunk_type"] for r in out})},
        "total_chars": sum(len(t) for t in texts),
        "max_chars": max((len(t) for t in texts), default=0),
        "empty_chunks": sum(1 for t in texts if not t.strip()),
        "patched_clauses": fmt_hits,
        "residual_broken_formula": broken,
        "embed_text_sha1": hashlib.sha1(
            "\n".join(r["embed_text"] for r in out).encode("utf-8")).hexdigest(),
        "note": ("embed_text 用于向量化；text 为规范化后原文，用于展示与引用。"
                 "embed_text_sha1 用于判断入库存量是否需要重建。"),
    }

    with open(f"{data_dir}/index_chunks.jsonl", "w", encoding="utf-8") as fh:
        for r in out:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    json.dump(clause_index, open(f"{data_dir}/clause_index.json", "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)
    json.dump(meta, open(f"{data_dir}/index_meta.json", "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)

    print(f"检索块 {meta['chunks']}  类型 {meta['by_type']}")
    print(f"总字符 {meta['total_chars']}  最长 {meta['max_chars']}  空块 {meta['empty_chunks']}")
    print(f"公式补丁命中 {list(fmt_hits.keys())}")
    print(f"残留疑似破损公式 {broken}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "data")
