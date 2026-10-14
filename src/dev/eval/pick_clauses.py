# -*- coding: utf-8 -*-
"""按关键词检索条款，用于挑选评测集素材（人工标注用）。

用法：
  python src/pick_clauses.py 补考 重修 作弊
  python src/pick_clauses.py --file F24 奖励标准
  python src/pick_clauses.py --id F24-A003            # 直接按条取
  python src/pick_clauses.py --id F24-A003 --full     # 打印全文
"""
import argparse
import json
import re
import sys
from pathlib import Path

D = Path("data")


def load():
    cl = json.load((D / "clauses.json").open(encoding="utf-8"))
    arts = [c for c in cl if c["level"] == "article"]
    byid = {c["clause_id"]: c for c in arts}
    return arts, byid


def brief(c, width=110):
    t = re.sub(r"\s+", " ", c["text"])
    print("%-11s %-4s p%-7s %4d字 | %s" % (
        c["clause_id"], c["file_id"],
        "%s-%s" % (c["page_start"], c["page_end"]), c["char_len"], t[:width]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("words", nargs="*")
    ap.add_argument("--file", default=None)
    ap.add_argument("--id", default=None)
    ap.add_argument("--full", action="store_true")
    ap.add_argument("--max", type=int, default=25)
    a = ap.parse_args()
    arts, byid = load()

    if a.id:
        c = byid.get(a.id)
        if not c:
            print("未找到", a.id)
            sys.exit(1)
        print("=" * 78)
        print("%s | %s | %s" % (c["clause_id"], c["file_title"], c["article_label"]))
        print("页 %s-%s | %d 字 | 章 %s" % (c["page_start"], c["page_end"],
                                          c["char_len"], c["chapter"]))
        print("=" * 78)
        print(c["text"] if a.full else re.sub(r"\s+", " ", c["text"])[:1500])
        return

    hits = arts
    if a.file:
        hits = [c for c in hits if c["file_id"] == a.file.upper()]
    if a.words:
        pats = [re.compile(re.escape(w)) for w in a.words]
        hits = [c for c in hits if any(p.search(c["text"]) for p in pats)]
    for c in hits[:a.max]:
        brief(c, 240 if a.full else 110)
    print("—— 共 %d 条" % len(hits))


if __name__ == "__main__":
    main()
