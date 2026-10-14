# -*- coding: utf-8 -*-
"""诊断：同文件相邻条款为什么没被「兄弟扩展」补进来。

背景
----
diag_hop_potential.py 显示：7 条漏项里 5 条是「同文件、条号距离 1~2」，
而 retrievers.py 的 _sibling_expand() 阈值正是「条号差 ≤ 2」。
理论上应该能补进来，却没补。需要确认到底卡在哪一步：

    A. 未触发   —— 该文件命中数 < 2（代码要求同一文件贡献 ≥2 条才扩展）
    B. 名额不足 —— per_file=3 / max_extra=4 / 上限 base_k+max_extra 被挤占
    C. 排序问题 —— 候选按距离排序后，近的被远的挤出

用法
----
    python src/diag_sibling.py
"""
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.stdout.reconfigure(encoding="utf-8")

from retrievers import HandbookRetriever  # noqa: E402

QIDS = ["Q24", "Q34", "Q35", "Q36", "Q37"]


def main():
    d = json.load((ROOT / "data" / "eval_set.json").open(encoding="utf-8"))
    cl = {c["clause_id"]: c for c in
          json.load((ROOT / "data" / "clauses.json").open(encoding="utf-8"))}
    by_qid = {i["qid"]: i for i in d["items"]}

    r = HandbookRetriever(str(ROOT / "data"), use_rerank=True, verbose=False)
    r.build()
    r._ensure_models()

    for qid in QIDS:
        it = by_qid[qid]
        gold = [s["clause_id"] for s in it["sources"]]

        r.expand = False
        base = r.search(it["question"], top_k=6)
        r.expand = True
        full = r.search(it["question"], top_k=6)

        b_ids = [e.clause_id for e in base]
        f_ids = [e.clause_id for e in full]

        print("=" * 76)
        print(f"{qid}  gold={gold}")
        print(f"  扩展前 {len(b_ids)} 条：{b_ids}")

        cnt = Counter(e.file_id for e in base if e.file_id)
        multi = {f for f, n in cnt.items() if n >= 2}
        print(f"  文件计数：{dict(cnt)}")
        print(f"  触发条件（同文件≥2条）满足的文件：{sorted(multi) or '无'}")

        used = set(b_ids)
        base_k = 6
        for f in sorted(multi):
            blocks = [c for c in r.chunks
                      if c.get("file_id") == f and c["chunk_type"] == "article"
                      and c.get("article_no") is not None]
            hit = [c["article_no"] for c in blocks if c["chunk_id"] in used]
            cand = []
            for c in blocks:
                if c["chunk_id"] in used:
                    continue
                dist = min(abs(c["article_no"] - h) for h in hit)
                if dist <= 2:
                    cand.append((dist, c["clause_id"]))
            cand.sort(key=lambda x: x[0])
            print(f"    {f}: 命中条号={sorted(hit)}")
            print(f"        距离≤2 的候选 {len(cand)} 个：{cand}")
            print(f"        per_file=3 只取：{[c for _, c in cand[:3]]}")

        need = [g for g in gold if g not in f_ids]
        print(f"  扩展后 {len(f_ids)} 条：{f_ids}")
        print(f"  仍未召回：{need}")
        if need:
            for n in need:
                fid = n.split("-")[0]
                print(f"    {n}  所在文件 {fid} 命中数={cnt.get(fid, 0)}  "
                      f"条号={cl.get(n, {}).get('article_no')}")


if __name__ == "__main__":
    main()
