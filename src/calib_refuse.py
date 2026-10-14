# -*- coding: utf-8 -*-
"""拒答阈值标定：对全部题目记录 top1 的检索分与字符覆盖率，看能否把
"该拒答的题"和"该回答的题"分开。用来定 min_conf。

    python src/calib_refuse.py
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.stdout.reconfigure(encoding="utf-8")

from retrievers import HandbookRetriever                      # noqa: E402
from procedural_qa import coverage, core_text                  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
ev = json.load((ROOT / "data" / "eval_set.json").open(encoding="utf-8"))["items"]

r = HandbookRetriever(ROOT / "data", use_rerank=True, verbose=False)

rows = []
for it in ev:
    res = r.search(it["question"], top_k=6)
    if not res:
        rows.append((it["qid"], bool(it.get("refuse_expected")), 0.0, 0.0, 0.0, len(it.get("sources") or [])))
        continue
    top1 = res[0]
    cov = coverage(it["question"], top1.text or "")
    # 前 3 条里最高的覆盖率（真实答案可能不在第一条）
    cov3 = max((coverage(it["question"], e.text or "") for e in res[:3]), default=0.0)
    rows.append((it["qid"], bool(it.get("refuse_expected")), top1.score, cov, cov3,
                 len(it.get("sources") or [])))

ans = [x for x in rows if not x[1]]
ref = [x for x in rows if x[1]]


def show(title, data, key):
    print(f"\n=== {title} ===")
    for qid, isref, sc, c1, c3, ngold in sorted(data, key=lambda x: key(x), reverse=True):
        print(f"  {qid:<5} score={sc:>7.3f}  cov1={c1:.2f}  cov3={c3:.2f}  gold={ngold}")


def stat(data, idx):
    vals = sorted(x[idx] for x in data)
    if not vals:
        return "-"
    n = len(vals)
    return f"min={vals[0]:.3f} p25={vals[n//4]:.3f} p50={vals[n//2]:.3f} p75={vals[3*n//4]:.3f} max={vals[-1]:.3f}"


print(f"题目 {len(rows)}  应作答 {len(ans)}  应拒答 {len(ref)}")
for name, idx in (("top1 检索分", 2), ("cov1 覆盖率", 3), ("cov3 覆盖率", 4)):
    print(f"\n{name}:")
    print("  应作答:", stat(ans, idx))
    print("  应拒答:", stat(ref, idx))

show("应拒答题（按检索分）", ref, lambda x: x[2])
show("应作答题：检索分最低的 12 道", sorted(ans, key=lambda x: x[2])[:12], lambda x: -x[2])
show("应作答题：cov3 最低的 12 道", sorted(ans, key=lambda x: x[4])[:12], lambda x: -x[4])

json.dump([{"qid": q, "refuse": rf, "score": sc, "cov1": c1, "cov3": c3, "n_gold": g}
           for q, rf, sc, c1, c3, g in rows],
          (ROOT / "data" / "calib_refuse.json").open("w", encoding="utf-8"),
          ensure_ascii=False, indent=1)
print("\n明细已写 data/calib_refuse.json")
