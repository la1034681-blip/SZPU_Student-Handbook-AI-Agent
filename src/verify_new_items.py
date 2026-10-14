# -*- coding: utf-8 -*-
"""阶段 6 · 评测集扩充：新题自动对账（防编造）。

对 data/new_items_draft.json 逐题校验：
  1) 每个 sources.clause_id 在 clauses.json 中真实存在；
  2) sources 的 file_id / file_title / article_no / page 与 clauses.json 一致；
  3) 每条 must_contain（去空白后）能在 sources 原文里找到 —— 证明要点可溯源；
  4) 去重后 file_id ≥3 —— 确属跨章；
  5) 答案里出现的 "Fxx" 文件代号，都在 sources 覆盖范围内 —— 防引用越界。

用法：python src/verify_new_items.py [draft.json]
退出码：0 全通过；1 存在失败项。
"""
import json
import re
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")
ROOT = Path(__file__).resolve().parent.parent


def norm(s: str) -> str:
    """去掉所有空白（含全角空格），便于宽松包含判断。"""
    return re.sub(r"\s+", "", s or "")


def main():
    draft = Path(sys.argv[1]) if len(sys.argv) > 1 else (ROOT / "data" / "new_items_draft.json")
    items = json.loads(draft.read_text(encoding="utf-8"))
    cl = {c["clause_id"]: c for c in
          json.load((ROOT / "data" / "clauses.json").open(encoding="utf-8"))}

    n_fail = 0
    print("=" * 78)
    print(f"新题对账报告　共 {len(items)} 题　源文件 {draft.name}")
    print("=" * 78)
    for it in items:
        errs = []
        # 1/2 出处存在且元数据一致
        ntext_parts = []
        for s in it["sources"]:
            c = cl.get(s["clause_id"])
            if not c:
                errs.append(f"条款不存在: {s['clause_id']}")
                continue
            ntext_parts.append(norm(c.get("text")))
            if c.get("file_id") != s.get("file_id"):
                errs.append(f"{s['clause_id']} file_id 不符")
            if c.get("file_title") != s.get("file_title"):
                errs.append(f"{s['clause_id']} file_title 不符")
            if c.get("article_no") != s.get("article_no"):
                errs.append(f"{s['clause_id']} article_no 不符")
        big = "".join(ntext_parts)

        # 3 每条要点可溯源
        miss = [k for k in it["must_contain"] if norm(k) not in big]
        if miss:
            errs.append("要点在原文找不到: " + " / ".join(miss))

        # 4 跨章文件数
        files = sorted({s["file_id"] for s in it["sources"]})
        if len(files) < 3:
            errs.append(f"文件数不足3（={len(files)}）")

        # 5 答案里的 Fxx 是否越界
        code_f = set(re.findall(r"F\d{2}", it["answer"]))
        out_of = code_f - set(files)
        if out_of:
            errs.append("答案引用了 sources 之外的文件: " + ",".join(sorted(out_of)))

        status = "PASS" if not errs else "FAIL"
        if errs:
            n_fail += 1
        print(f"\n[{status}] {it['qid']}  文件{len(files)}={files}  要点{len(it['must_contain'])}")
        print(f"    题：{it['question'][:48]}")
        if errs:
            for e in errs:
                print(f"    ✗ {e}")

    print("\n" + "=" * 78)
    print(f"结果：{len(items) - n_fail}/{len(items)} 通过；失败 {n_fail} 题")
    print("=" * 78)
    return 1 if n_fail else 0


if __name__ == "__main__":
    sys.exit(main())
