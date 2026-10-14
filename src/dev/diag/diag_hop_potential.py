# -*- coding: utf-8 -*-
"""诊断：漏掉的 gold 条款，各自属于哪一类「可救回」的原因。

背景
----
diag_gold_recall.py 已证明：跨章题 gold 条款只召回 72.2%，4 道题无一全召回。
但「漏了」不等于「二跳能救」。漏因至少有三类，对应三种不同的修法：

    甲、同文件 + 距离近  →  放宽兄弟扩展的名额/距离即可（改 retrievers 常量）
    乙、同文件 + 距离远  →  同上，但要权衡噪音
    丙、跨文件          →  必须靠「顺着正文里的《文件名》引用去追」（真二跳）

因此本脚本对每道有遗漏的题，输出：
    1. 漏掉的 gold 条款，以及它与「已召回同文件条款」的最小条号距离
    2. 已召回证据正文里出现过的所有《…》文件名引用（二跳的线索来源）
    3. 漏掉条款所属文件，是否出现在上述引用里（决定二跳能否救回）

用法
----
    python src/diag_hop_potential.py
"""
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.stdout.reconfigure(encoding="utf-8")

from retrievers import HandbookRetriever, RE_FILE_REF  # noqa: E402

TYPES = ("cross", "synthesis")
DROP = "深圳职业技术大学"          # 手册正文引用文件时通常省略校名


def short_title(t: str) -> str:
    return (t or "").replace(DROP, "").strip()


def main():
    d = json.load((ROOT / "data" / "eval_set.json").open(encoding="utf-8"))
    cl = {c["clause_id"]: c for c in
          json.load((ROOT / "data" / "clauses.json").open(encoding="utf-8"))}
    items = [i for i in d["items"] if i["type"] in TYPES and i.get("sources")]

    r = HandbookRetriever(str(ROOT / "data"), use_rerank=True, verbose=False)
    r.build()
    r._ensure_models()

    stats = {"甲 同文件近": 0, "乙 同文件远": 0, "丙 跨文件": 0}
    lines = []

    for it in items:
        gs = {s["clause_id"] for s in it["sources"]}
        evs = r.search(it["question"], top_k=6)
        got = {e.clause_id for e in evs}
        miss = sorted(gs - got)
        if not miss:
            continue

        text_all = "\n".join(e.text for e in evs)
        refs = sorted(set(RE_FILE_REF.findall(text_all)))
        lines.append("=" * 76)
        lines.append(f"{it['qid']} [{it['type']}] {it['question'][:40]}")
        lines.append(f"  gold {len(gs)} 条，召回 {len(gs & got)} 条，缺 {len(miss)} 条")

        for m in miss:
            fid = m.split("-")[0]
            no = cl.get(m, {}).get("article_no")
            title = cl.get(m, {}).get("file_title", "")
            same_file_hits = [g for g in got if g.startswith(fid + "-")]
            dist = None
            for g in same_file_hits:
                gn = cl.get(g, {}).get("article_no")
                if gn is not None and no is not None:
                    dd = abs(gn - no)
                    dist = dd if dist is None else min(dist, dd)
            if dist is None:
                cls = "丙 跨文件"
            elif dist <= 4:
                cls = "甲 同文件近"
            else:
                cls = "乙 同文件远"
            stats[cls] += 1
            # 二跳线索：正文引用里有没有这份文件
            seg = short_title(title)
            key = seg.replace("（试行）", "").replace("(试行)", "")[:10]
            clue = [x for x in refs if key and key in x]
            lines.append(
                f"    缺 {m} [{cls}] 距同文件最近命中 "
                f"{dist if dist is not None else '—'} ｜ {seg}")
            if clue:
                lines.append(f"       正文引用线索命中：{clue}")

        # 已召回证据里出现过的文件引用（去重，去掉自称）
        cited_files = set()
        for e in evs:
            for x in RE_FILE_REF.findall(e.text):
                if short_title(e.file_title)[:6] not in x:
                    cited_files.add(x)
        if cited_files:
            lines.append(f"  正文引用到的其它文件（{len(cited_files)}）："
                         + "、".join(sorted(cited_files)[:8]))

    print("\n".join(lines))
    print("\n" + "=" * 76)
    print("漏因分类统计：")
    for k, v in stats.items():
        print(f"  {k:<12} {v} 条")
    print("\n说明：")
    print("  甲/乙 两类可由「放宽兄弟扩展」解决（改常量，无需 LLM）")
    print("  丙 类必须靠二跳（顺着《文件名》引用跨文件追）")

    (ROOT / "data" / "diag_hop_potential.txt").write_text(
        "\n".join(lines), encoding="utf-8")
    print(f"\n已写 {ROOT / 'data' / 'diag_hop_potential.txt'}")


if __name__ == "__main__":
    main()
