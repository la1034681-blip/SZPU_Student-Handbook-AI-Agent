# -*- coding: utf-8 -*-
"""程序化通道的快速目检：挑代表题，打印答案、出处与契合度。

    python src/test_procedural.py            # 默认抽 6 道
    python src/test_procedural.py Q01 Q38    # 指定题号
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.stdout.reconfigure(encoding="utf-8")

from retrievers import HandbookRetriever                      # noqa: E402
from procedural_qa import ProceduralPipeline                  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent.parent
ev = json.load((ROOT / "data" / "eval_set.json").open(encoding="utf-8"))["items"]

want = sys.argv[1:]
if want:
    picked = [x for x in ev if x["qid"] in want]
else:
    fact = [x for x in ev if not x.get("refuse_expected")]
    ref = [x for x in ev if x.get("refuse_expected")]
    picked = fact[:4] + ref[:2]

r = HandbookRetriever(ROOT / "data", use_rerank=True, verbose=False)
pipe = ProceduralPipeline(r, top_k=6)

ok = 0
for it in picked:
    res = pipe.ask(it["question"])
    gold = {s["clause_id"] for s in it.get("sources", [])}
    cited = {e.clause_id for e in res.evidences if e.eid in res.citations}
    covered = None if not gold else (False if res.refused else bool(gold & cited))
    if covered:
        ok += 1
    print("=" * 78)
    print(f"[{it['qid']}] {it['type']}  契合度={res.confidence:.2f}  "
          f"拒答={res.refused}  出处命中={covered}  耗时={res.latency:.2f}s")
    print(f"问：{it['question']}")
    if res.refused:
        print(f"→ 拒答：{res.refuse_reason}")
    else:
        print(res.answer)
    print(f"gold={sorted(gold)}  cited={sorted(cited)}")

print("=" * 78)
print(f"出处命中 {ok}/{len(picked)}")
