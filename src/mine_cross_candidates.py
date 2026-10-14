# -*- coding: utf-8 -*-
"""阶段 6 · 评测集扩充：从语料里"挖"候选跨章题。

思路
----
一道真正的"跨章题"，其答案要点必然散落在**多份不同文件**的条款里。
所以先按主题词扫一遍语料，统计每个主题横跨了几份文件：

    横跨文件数 >= 3  → 有资格做跨章题候选
    横跨文件数 <  3  → 大概率是单文件题，排除

本脚本只做"候选发现"，不生成题目。输出：
  1. 每个主题词横跨的文件数、涉及条款数
  2. 对候选主题，列出落在各文件里的具体条款（供人工/LLM 撰写题目时取原文）

用法：
    python src/mine_cross_candidates.py              # 打印概览
    python src/mine_cross_candidates.py 学业预警      # 展开某个主题的条款
"""
import json
import sys
from collections import defaultdict
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")
ROOT = Path(__file__).resolve().parent.parent

# 覆盖手册各模块的主题词（用"子串匹配"，够用且可解释）
TOPICS = [
    # 学籍 / 成绩
    "补考", "重修", "缓考", "免修", "学分认定", "置换", "绩点", "学业预警",
    "退学", "转专业", "休学", "复学", "保留学籍", "毕业", "结业", "学位",
    "考核", "成绩", "学分",
    # 考勤 / 纪律
    "请假", "旷课", "考勤", "迟到", "早退", "违纪", "处分", "作弊",
    "考试违规", "申诉", "行为准则",
    # 奖惩 / 资助
    "奖学金", "助学金", "助学贷款", "家庭经济困难", "勤工助学",
    "综合素质", "评选", "推荐", "优秀毕业生", "三好",
    # 其他
    "创新创业", "学费", "收费", "退费", "公寓", "宿舍", "一卡通",
    "军训", "入学资格", "体检", "社团", "网络", "出国", "港澳台",
    "留学生", "考风考纪", "诚信",
]


def load_clauses():
    cl = json.load((ROOT / "data" / "clauses.json").open(encoding="utf-8"))
    return [c for c in cl if c.get("level") == "article"]


def main():
    clauses = load_clauses()
    arg = sys.argv[1] if len(sys.argv) > 1 else None

    index = defaultdict(list)          # topic -> [(file_id, file_title, clause_id, article_no)]
    for c in clauses:
        txt = c.get("text") or ""
        for t in TOPICS:
            if t in txt:
                index[t].append((c["file_id"], c["file_title"],
                                 c["clause_id"], c.get("article_no")))

    if arg:
        rows = index.get(arg, [])
        print(f"主题「{arg}」命中 {len(rows)} 条条款：")
        byfile = defaultdict(list)
        for fid, ft, cid, an in rows:
            byfile[(fid, ft)].append((cid, an))
        for (fid, ft), lst in sorted(byfile.items()):
            print(f"\n  [{fid}] {ft} —— {len(lst)} 条")
            for cid, an in sorted(lst, key=lambda x: (x[1] or 0)):
                print(f"      {cid}（第{an}条）")
        return

    # 概览：按横跨文件数排序
    stats = []
    for t, rows in index.items():
        files = {r[0] for r in rows}
        stats.append((len(files), len(rows), t))
    stats.sort(reverse=True)

    print(f"{'横跨文件':>6}{'条款数':>8}   主题")
    print("-" * 40)
    for nf, nc, t in stats:
        mark = "  ← 候选" if nf >= 3 else ""
        print(f"{nf:>6}{nc:>8}   {t}{mark}")

    cand = [t for nf, _, t in stats if nf >= 3]
    print(f"\n合计 {len(cand)} 个主题横跨 ≥3 份文件：")
    print("  " + "、".join(cand))


if __name__ == "__main__":
    main()
