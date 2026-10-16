# -*- coding: utf-8 -*-
"""诊断：兄弟扩展的「名额 / 距离 / 上限」三参数扫描。

背景
----
diag_sibling.py 已把 7 条漏项定位到具体原因：
    Q24 缺 A007/A008 —— 候选排在第 7/8 位，被 per_file=3 挤掉（名额不足）
    Q37 缺 A008      —— 候选排第 4 位，同样被挤掉（名额不足）
    Q37 缺 A002      —— 距离 2，排第 7，被挤掉（名额不足）
    Q36 缺 A004      —— 距离 4，超出 dist<=2 阈值（距离太紧）
    Q34 缺 F06-A023  —— 整份文件没召回（跨文件，扩展救不了）
    Q35 缺 F12-A003  —— 整份文件没召回（跨文件，扩展救不了）

也就是说：7 条里 5 条靠「调参数」就能救，不必上任何新架构。
本脚本把 _sibling_expand 的逻辑参数化复现，扫一遍参数网格：

    指标：cross / synthesis 的 gold 召回率、全召回题数、平均证据条数
    「平均证据条数」是噪音代理量 —— 条数越多，prompt 越长、生成被稀释的风险越大。

用法
----
    python src/diag_expand_tune.py
"""
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(ROOT / "src"))
import _paths  # noqa: E402 把 core/retrieval/infra/cli 也挂上搜索路径
sys.stdout.reconfigure(encoding="utf-8")

from retrievers import HandbookRetriever  # noqa: E402

TYPES = ("cross", "synthesis")


def expand_ids(base_ids, chunks, per_file, max_dist, max_extra, base_k=6):
    """复现 _sibling_expand 的挑选逻辑，返回扩后的 clause_id 列表。"""
    by_file = {}
    for c in chunks:
        if c["chunk_type"] == "article" and c.get("article_no") is not None:
            by_file.setdefault(c.get("file_id"), []).append(c)

    fid_of = {}
    for c in chunks:
        cl = c.get("clause_id") or c["chunk_id"]
        fid_of.setdefault(cl, c.get("file_id"))

    cnt = Counter(fid_of.get(i) for i in base_ids if fid_of.get(i))
    multi = {f for f, n in cnt.items() if n >= 2}
    if not multi:
        return list(base_ids)

    used = set(base_ids)
    extra = []
    for f in multi:
        blocks = by_file.get(f, [])
        hit = [c["article_no"] for c in blocks if c["chunk_id"] in used]
        if not hit:
            continue
        cand = []
        for c in blocks:
            if c["chunk_id"] in used:
                continue
            d = min(abs(c["article_no"] - h) for h in hit)
            if d <= max_dist:
                cand.append((d, c["clause_id"]))
        cand.sort(key=lambda x: x[0])
        extra.extend(cid for _, cid in cand[:per_file])

    out = list(base_ids)
    for cid in extra:
        if len(out) >= base_k + max_extra:
            break
        if cid in out:
            continue
        out.append(cid)
    return out


def main():
    d = json.load((ROOT / "data" / "eval_set.json").open(encoding="utf-8"))
    items = [i for i in d["items"] if i["type"] in TYPES and i.get("sources")]

    r = HandbookRetriever(str(ROOT / "data"), use_rerank=True, expand=False,
                          verbose=False)
    r.build()
    r._ensure_models()

    gold = {}
    base = {}
    for it in items:
        gold[it["qid"]] = {s["clause_id"] for s in it["sources"]}
        base[it["qid"]] = [e.clause_id for e in r.search(it["question"], top_k=6)]

    grid = [(3, 2, 4), (5, 2, 6), (5, 4, 8), (8, 4, 12), (8, 6, 16), (12, 6, 24)]

    print(f"{'per_file':>8} {'dist':>5} {'max_extra':>9} │ "
          f"{'cross召回':>10} {'synth召回':>10} {'全召回题':>8} {'平均条数':>8}")
    print("─" * 72)
    rows = []
    for pf, md, me in grid:
        tot = {"cross": [0, 0], "synthesis": [0, 0]}
        perfect = 0
        n_ev = 0
        for it in items:
            g = gold[it["qid"]]
            ids = expand_ids(base[it["qid"]], r.chunks, pf, md, me)
            hit = len(g & set(ids))
            tot[it["type"]][0] += hit
            tot[it["type"]][1] += len(g)
            if hit == len(g):
                perfect += 1
            n_ev += len(ids)
        cr = tot["cross"][0] / tot["cross"][1]
        sr = tot["synthesis"][0] / tot["synthesis"][1]
        avg = n_ev / len(items)
        rows.append((pf, md, me, cr, sr, perfect, avg))
        print(f"{pf:>8} {md:>5} {me:>9} │ {cr:>9.1%} {sr:>10.1%} "
              f"{perfect:>5}/{len(items)} {avg:>8.1f}")

    print("\n注：第一行 (3,2,4) 就是当前线上配置。")
    base_row = rows[0]
    print(f"当前配置 cross 召回 {base_row[3]:.1%}、全召回 {base_row[5]}/{len(items)}")
    best = max(rows, key=lambda x: (x[3] + x[4], -x[6]))
    print(f"召回最优配置 ({best[0]},{best[1]},{best[2]})："
          f"cross {best[3]:.1%}、synth {best[4]:.1%}、"
          f"全召回 {best[5]}/{len(items)}、平均 {best[6]:.1f} 条")

    (ROOT / "data" / "diag_expand_tune.json").write_text(
        json.dumps([{"per_file": a, "max_dist": b, "max_extra": c,
                     "cross_recall": d, "synth_recall": e,
                     "perfect": f, "avg_ev": g}
                    for a, b, c, d, e, f, g in rows],
                   ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\n已写 {ROOT / 'data' / 'diag_expand_tune.json'}")


if __name__ == "__main__":
    main()
