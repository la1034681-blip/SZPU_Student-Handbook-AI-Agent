# -*- coding: utf-8 -*-
"""阶段 6 · 评测集扩充：按主题导出条款原文（供出题取原文用）。

用法：
    python src/dump_topic_clauses.py            # 导出预设主题集
    python src/dump_topic_clauses.py 缓考 休学   # 只导出指定主题

输出：data/_mine_dump.txt（每行一条条款：文件/条款号/原文前 300 字）
"""
import json
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")
ROOT = Path(__file__).resolve().parent.parent

DEFAULT_TOPICS = [
    "缓考", "休学", "复学", "保留学籍", "退学", "免修", "置换",
    "毕业", "结业", "学位", "违纪", "申诉", "旷课", "请假", "考勤",
    "家庭经济困难", "助学金", "助学贷款", "勤工助学", "学费", "退费",
    "军训", "公寓", "港澳台", "留学生", "创新创业", "入学资格", "体检",
]

MAXLEN = 300


def main():
    topics = sys.argv[1:] or DEFAULT_TOPICS
    cl = json.load((ROOT / "data" / "clauses.json").open(encoding="utf-8"))
    art = [c for c in cl if c.get("level") == "article"]

    hit = {}
    for c in art:
        txt = c.get("text") or ""
        for t in topics:
            if t in txt:
                hit.setdefault(t, []).append(c)

    lines = []
    for t in topics:
        rows = hit.get(t, [])
        if not rows:
            continue
        byfile = {}
        for c in rows:
            byfile.setdefault((c["file_id"], c["file_title"]), []).append(c)
        lines.append("=" * 78)
        lines.append(f"### 主题：{t}　命中 {len(rows)} 条，横跨 {len(byfile)} 份文件")
        lines.append("=" * 78)
        for (fid, ft), lst in sorted(byfile.items()):
            lines.append(f"\n── [{fid}] {ft}")
            for c in sorted(lst, key=lambda x: (x.get("article_no") or 0)):
                txt = " ".join((c.get("text") or "").split())
                lines.append(f"  {c['clause_id']}（第{c.get('article_no')}条）"
                             f" {txt[:MAXLEN]}")
        lines.append("")

    out = ROOT / "data" / "_mine_dump.txt"
    out.write_text("\n".join(lines), encoding="utf-8")
    print(f"已写出 {out}　共 {sum(len(hit.get(t, [])) for t in topics)} 条条款")
    for t in topics:
        n = len({c['file_id'] for c in hit.get(t, [])})
        print(f"  {t:<10} 条款 {len(hit.get(t, [])):>3}  文件 {n}")


if __name__ == "__main__":
    main()
