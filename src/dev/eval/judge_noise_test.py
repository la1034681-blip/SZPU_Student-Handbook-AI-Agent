# -*- coding: utf-8 -*-
"""阅卷器噪声测试：固定同一份答案，重复判分 N 次，看判分是否稳定。

为什么需要它
------------
稳定性实验里，「答案文本三轮不完全相同」几乎必然成立（模型换句话写），
所以「答案翻转率」无法区分下面两种情况：
  ① 生成确实写了不同的内容 → 判分变化是合理的
  ② 生成写的内容等价、只是措辞不同 → 判分变化是阅卷器的噪声

本脚本把变量钉死：**同一道题、同一份答案、同一批证据**，只重复调用阅卷器。
判分若仍然变化，就是纯粹的阅卷器噪声。

用法
----
    python src/judge_noise_test.py data --qids Q24,Q35 --repeat 3
    python src/judge_noise_test.py data --source stability_results.json --round 1 --repeat 3
"""
import argparse
import json
import sys
from collections import Counter
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
import _paths  # noqa: E402 把 core/retrieval/infra/cli 也挂上搜索路径

from judge_answers import judge                       # noqa: E402
from qa_chain import build_llm                        # noqa: E402
from retrievers import HandbookRetriever              # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("data_dir", nargs="?", default="data")
    ap.add_argument("--qids", default="Q24,Q35")
    ap.add_argument("--source", default="stability_results.json",
                    help="取答案的来源结果文件")
    ap.add_argument("--round", type=int, default=1,
                    help="取该来源文件的第几轮答案（从 1 开始）")
    ap.add_argument("--repeat", type=int, default=3, help="同一份答案重复判几次")
    ap.add_argument("--top-k", type=int, default=6)
    ap.add_argument("--model", default=None)
    ap.add_argument("--out", default="judge_noise.json")
    a = ap.parse_args()

    d = Path(a.data_dir)
    if not d.is_absolute():
        d = Path(__file__).resolve().parent.parent.parent.parent / d
    items = {it["qid"]: it for it in
             json.load((d / "eval_set.json").open(encoding="utf-8"))["items"]}
    src = json.loads((d / a.source).read_text(encoding="utf-8"))
    runs = src["_runs"]
    run = runs[a.round - 1]

    llm = build_llm(model=a.model)
    retr = HandbookRetriever(d, use_rerank=True, verbose=False)
    cache = {}

    qids = [q.strip() for q in a.qids.split(",") if q.strip()]
    out = {"meta": {"source": a.source, "round": a.round,
                    "repeat": a.repeat, "qids": qids}, "items": []}
    total_flip = 0

    for qid in qids:
        it = items[qid]
        ans = run[qid]["answer"]
        if qid not in cache:
            cache[qid] = retr.search(it["question"], top_k=a.top_k)
        evs = cache[qid]

        print("=" * 78)
        print(f"{qid}  {it['question']}")
        print(f"  以固定答案重复判分 {a.repeat} 次（答案取自第 {a.round} 轮）")
        verdicts, scores = [], []
        for i in range(1, a.repeat + 1):
            jr = judge(it["question"], it.get("answer", ""), ans,
                       it.get("must_contain"), it.get("note", ""),
                       llm=llm, qtype=it["type"], evidences=evs)
            verdicts.append(jr["verdict"])
            scores.append(jr["score"])
            print(f"    第{i}次：{jr['verdict']:<10} score={jr['score']} "
                  f"覆盖={jr['coverage']}  理由={str(jr.get('reason',''))[:50]}")

        uniq_v = len(set(verdicts))
        uniq_s = len(set(scores))
        flip = uniq_s > 1
        total_flip += flip
        print(f"  → 判分{'不稳定' if flip else '稳定'}："
              f"结论 {dict(Counter(verdicts))} / 分数 {dict(Counter(scores))}")
        out["items"].append({
            "qid": qid, "question": it["question"],
            "verdicts": verdicts, "scores": scores,
            "verdict_variants": uniq_v, "score_variants": uniq_s,
            "unstable": flip,
        })

    print("\n" + "=" * 78)
    print(f"汇总：{len(qids)} 道题中，同一份答案重复判分出现分数变化的 "
          f"{total_flip} 道（{total_flip / len(qids):.1%}）")
    out["summary"] = {"n": len(qids), "unstable": total_flip,
                      "noise_rate": round(total_flip / len(qids), 4)}
    (d / a.out).write_text(json.dumps(out, ensure_ascii=False, indent=1),
                           encoding="utf-8")
    print(f"结果已写入 {d / a.out}")


if __name__ == "__main__":
    main()
