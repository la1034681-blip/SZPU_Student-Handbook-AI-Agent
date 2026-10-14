# -*- coding: utf-8 -*-
"""阶段 6 · 评测集扩充：按 clause_id 导出条款全文（出题取原文用）。

用法：
    python src/show_clauses.py            # 导出预设清单
    python src/show_clauses.py F03-A023 F04-A017
"""
import json
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")
ROOT = Path(__file__).resolve().parent.parent

# 出题要用到的条款（按题分组，见下方 GROUPS）
GROUPS = {
    "N01 缓考": ["F03-A023", "F04-A017", "F10-A012"],
    "N02 休学": ["F03-A030", "F03-A032", "F03-A034", "F04-A029", "F41-A018"],
    "N03 复学": ["F03-A035", "F04-A035", "F09-A011"],
    "N04 入伍": ["F04-A033", "F09-A012", "F41-A020"],
    "N05 退学": ["F03-A038", "F03-A041", "F03-A048", "F04-A039", "F38-A015"],
    "N06 免修置换": ["F09-A001", "F09-A002", "F09-A007", "F09-A013",
                     "F03-A013", "F04-A021"],
    "N07 结业换毕业": ["F03-A045", "F03-A046", "F04-A042", "F04-A043", "F10-A003"],
    "N08 学位": ["F16-A003", "F16-A004", "F16-A005", "F03-A047", "F28-A004"],
    "N09 处分申诉": ["F28-A004", "F28-A006", "F28-A016", "F29-A007",
                     "F29-A018", "F29-A021", "F04-A026"],
    "N10 考勤": ["F06-A023", "F06-A015", "F06-A017", "F06-A018",
                 "F28-A025", "F10-A004"],
    "N11 资助": ["F26-A007", "F24-A002", "F24-A007", "F25-A004", "F27-A004"],
    "N12 退费": ["F41-A018", "F41-A019", "F41-A020", "F41-A024",
                 "F38-A015", "F03-A041"],
    "N13 军训": ["F42-A001", "F42-A002", "F42-A003", "F42-A006", "F42-A007",
                 "F30-A008", "F20-A019"],
    "N14 港澳台": ["F30-A006", "F30-A008", "F30-A010", "F30-A011", "F09-A001"],
    "N15 创新创业": ["F17-A003", "F17-A004", "F17-A005", "F09-A011", "F04-A030"],
    "N16 公寓": ["F38-A013", "F38-A015", "F38-A016", "F41-A019", "F28-A038"],
}


def main():
    ids = sys.argv[1:]
    cl = {c["clause_id"]: c for c in
          json.load((ROOT / "data" / "clauses.json").open(encoding="utf-8"))}
    lines = []
    if ids:
        for cid in ids:
            c = cl.get(cid)
            lines.append(f"\n[{cid}] {'存在' if c else '★不存在'}")
            if c:
                lines.append(f"  {c['file_id']} {c['file_title']} 第{c.get('article_no')}条")
                lines.append("  " + " ".join((c.get("text") or "").split())[:900])
    else:
        for g, ids2 in GROUPS.items():
            lines.append("=" * 78)
            lines.append(f"### {g}")
            lines.append("=" * 78)
            for cid in ids2:
                c = cl.get(cid)
                if not c:
                    lines.append(f"  ★ {cid} 不存在")
                    continue
                lines.append(f"\n[{cid}] {c['file_title']} 第{c.get('article_no')}条")
                lines.append("  " + " ".join((c.get("text") or "").split())[:900])
            lines.append("")
    out = ROOT / "data" / "_new_q_clauses.txt"
    out.write_text("\n".join(lines), encoding="utf-8")
    print(f"已写出 {out}")

    # 附带：F42 军训全部条款清单
    print("\n===== F42 军训条款清单 =====")
    for c in sorted([x for x in cl.values() if x["file_id"] == "F42"],
                    key=lambda x: (x.get("article_no") or 0)):
        print(f"  {c['clause_id']} 第{c.get('article_no')}条 "
              f"{' '.join((c.get('text') or '').split())[:70]}")


if __name__ == "__main__":
    main()
