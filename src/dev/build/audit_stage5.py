"""阶段 5 结果独立审计。

目的：不信任已写入 summary 的指标，全部从 items 逐条重算；
     并检查三份结果文件之间、与评测集之间的一致性。
用法：python src/audit_stage5.py data
"""
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")
ROOT = Path(__file__).resolve().parent.parent.parent.parent
D = ROOT / (sys.argv[1] if len(sys.argv) > 1 else "data")

FAIL = []


def check(cond, msg):
    print(("  [OK]  " if cond else "  [FAIL]") + " " + msg)
    if not cond:
        FAIL.append(msg)


def load(name):
    return json.loads((D / name).read_text(encoding="utf-8"))


def rate(num, den):
    return round(num / den, 4) if den else None


print("=" * 78)
print("A. gen_results.json —— 从 items 独立复算 vs summary")
print("=" * 78)
g = load("gen_results.json")
items = g["items"]
sm = g["summary"]
graded = [r for r in items if not r["refuse_expected"]]
refus = [r for r in items if r["refuse_expected"]]

calc = {
    "n": len(items),
    "n_graded": len(graded),
    "n_refuse": len(refus),
    "引用存在率": rate(sum(1 for r in graded if r["exist_pass"]), len(graded)),
    "引用支持率": rate(sum(1 for r in graded if r["support_pass"]), len(graded)),
    "出处命中率": rate(sum(1 for r in graded if r["gold_covered"]), len(graded)),
    "误拒率": rate(sum(1 for r in graded if r["refused"]), len(graded)),
    "拒答正确率": rate(sum(1 for r in refus if r["refused"]), len(refus)),
}
print(f"  {'指标':<12}{'summary':>12}{'独立复算':>12}")
for k, v in calc.items():
    s = sm.get(k)
    same = (s == v)
    print(f"  {k:<12}{str(s):>12}{str(v):>12}{'' if same else '   <<< 不一致'}")
    check(same, f"summary[{k}] 与复算一致")

# 延迟
lat = round(sum(r["latency"] for r in items) / len(items), 2)
print(f"  {'平均延迟':<12}{str(sm.get('平均延迟')):>12}{lat:>12}")
check(abs((sm.get("平均延迟") or 0) - lat) < 0.02, "平均延迟一致")

# 答案正确率：avg of judge_score；完全正确率
scores = [r["judge_score"] for r in graded if r.get("judge_score") is not None]
acc = round(sum(scores) / len(scores), 4) if scores else None
full = rate(sum(1 for s in scores if s >= 1.0), len(scores))
print(f"\n  答案正确率(复算) = {acc}   完全正确率(复算) = {full}")
for key in ("答案正确率", "完全正确率"):
    if key in sm:
        print(f"  summary[{key}] = {sm[key]}")

print("\n" + "=" * 78)
print("B. 字段完整性与自洽")
print("=" * 78)
need = ["qid", "type", "question", "refused", "answer", "citations",
        "gold", "cited_clauses", "gold_covered", "latency"]
missing = [(r["qid"], k) for r in items for k in need if k not in r]
check(not missing, f"所有题目含必备字段（缺失 {missing[:5]}）")
# 拒答题不应计 gold_covered
bad = [r["qid"] for r in refus if r.get("gold_covered") is not None]
check(not bad, f"拒答题 gold_covered 均为 None（异常 {bad}）")
# 已拒答的题不应算作出处命中
bad2 = [r["qid"] for r in graded if r["refused"] and r.get("gold_covered")]
check(not bad2, f"被拒答的可判分题未计入出处命中（异常 {bad2}）")
# 判分为 None 的题应为拒答或未判分
bad3 = [r["qid"] for r in graded if r.get("judge_score") is None]
check(not bad3, f"所有可判分题都有 judge_score（缺 {bad3}）")
# 引用编号必须是 E 开头
bad4 = [r["qid"] for r in items for c in (r.get("citations") or [])
        if not str(c).startswith("E")]
check(not bad4, f"引用编号格式合法（异常 {bad4[:5]}）")

print("\n" + "=" * 78)
print("C. gen_results 与 eval_set 的题目集合一致性")
print("=" * 78)
ev = load("eval_set.json")
evmap = {it["qid"]: it for it in ev["items"]}
check(len(evmap) == len(ev["items"]), "评测集 qid 唯一")
gset = {r["qid"] for r in items}
check(gset == set(evmap), f"题号集合一致（仅评测集有 {sorted(set(evmap)-gset)}，仅结果有 {sorted(gset-set(evmap))}）")
rt = Counter(r["type"] for r in items)
et = Counter(it["type"] for it in ev["items"])
check(rt == et, f"题型分布一致 结果{dict(rt)} 评测集{dict(et)}")

print("\n" + "=" * 78)
print("D. gold 出处真实性（回查 clauses.json）")
print("=" * 78)
cmap = {c["clause_id"]: c for c in load("clauses.json")}
badg = []
for r in items:
    if r["refuse_expected"]:
        continue
    for cid in r.get("gold") or []:
        if cid not in cmap:
            badg.append((r["qid"], cid))
check(not badg, f"gold 条款 ID 全部存在于语料（异常 {badg[:5]}）")
badc = [(r["qid"], c) for r in items for c in (r.get("cited_clauses") or [])
        if c not in cmap]
check(not badc, f"引用条款 ID 全部存在于语料（异常 {badc[:5]}）")

print("\n" + "=" * 78)
print("E. 稳定性文件 —— 从 _runs 原始数据独立复算")
print("=" * 78)
for fn in ("stability_results.json", "stability_no_recheck.json"):
    if not (D / fn).exists():
        print(f"  {fn} 不存在，跳过")
        continue
    x = load(fn)
    rounds = x["meta"]["rounds"]
    runs = x["_runs"]
    n = x["meta"]["n_items"]
    print(f"\n  --- {fn} （复核={'开' if x['meta']['recheck'] else '关'}，"
          f"{rounds} 轮，{n} 题）---")
    check(len(runs) == rounds, f"{fn}: _runs 轮数 = meta.rounds")

    # ① 逐题跨轮比对（复刻 stability_test 的定义）
    per_qid = defaultdict(list)
    for per in runs:
        for qid, rec in per.items():
            per_qid[qid].append(rec)
    check(len(per_qid) == n, f"{fn}: _runs 覆盖全部 {n} 题")

    fr_calc = {"拒答翻转率": 0, "出处翻转率": 0, "答案翻转率": 0,
               "校验翻转率": 0, "判分翻转率": 0, "任一翻转率": 0}
    unstable_calc = set()
    for qid, recs in per_qid.items():
        rf = len({r_["refused"] for r_ in recs}) > 1
        cf = len({tuple(r_["cited"]) for r_ in recs}) > 1
        af = len({r_["answer"] for r_ in recs}) > 1
        vf = len({r_["verify_pass"] for r_ in recs}) > 1
        sv = len({r_["score"] for r_ in recs if r_["score"] is not None}) > 1
        fr_calc["拒答翻转率"] += rf
        fr_calc["出处翻转率"] += cf
        fr_calc["答案翻转率"] += af
        fr_calc["校验翻转率"] += vf
        fr_calc["判分翻转率"] += sv
        if rf or cf or af or vf or sv:
            unstable_calc.add(qid)
    for k in fr_calc:
        fr_calc[k] = round(fr_calc[k] / n, 4)
    fr_calc["任一翻转率"] = round(len(unstable_calc) / n, 4)

    print(f"     {'翻转率':<12}{'文件值':>10}{'复算值':>10}")
    for k in fr_calc:
        same = x["flip_rates"].get(k) == fr_calc[k]
        print(f"     {k:<12}{x['flip_rates'].get(k):>10}{fr_calc[k]:>10}"
              f"{'' if same else '   <<< 不一致'}")
        check(same, f"{fn}: flip_rates[{k}] 可复现")

    un_file = {u["qid"] for u in x["unstable"]}
    check(un_file == unstable_calc,
          f"{fn}: unstable 名单一致（差集 {sorted(un_file ^ unstable_calc)[:5]}）")
    check(x["stable_items"] == n - len(unstable_calc),
          f"{fn}: stable_items 计数一致")

    # unstable 里每题的 score_range 必须与 _runs 现算的一致。
    # 这一项专门防「rescore 重判后 _runs 更新、unstable 留旧值」的陈旧数据。
    bad_rng = []
    for u in x["unstable"]:
        sc = [runs[k][u["qid"]]["score"] for k in range(rounds)]
        valid = [s for s in sc if s is not None]
        rng = round(max(valid) - min(valid), 4) if valid else None
        if rng != u.get("score_range"):
            bad_rng.append((u["qid"], u.get("score_range"), rng))
    check(not bad_rng,
          f"{fn}: unstable.score_range 与 _runs 同步（陈旧 {bad_rng[:3]}）")

    # ② 逐轮指标独立复算
    ev_d = load("eval_set.json")["items"]
    evmap2 = {it["qid"]: it for it in ev_d}
    grad = [q for q in per_qid if not evmap2[q].get("refuse_expected")]
    refu = [q for q in per_qid if evmap2[q].get("refuse_expected")]

    def rnd_rate(a, b):
        return round(a / b, 4) if b else None

    per_round_calc = []
    for k in range(rounds):
        hit = refused_g = verif = 0
        scores = []
        for qid in grad:
            rec = runs[k][qid]
            gold = {s["clause_id"] for s in evmap2[qid].get("sources", [])}
            if rec["verify_pass"]:
                verif += 1
            if rec["refused"]:
                refused_g += 1
                scores.append(0.0)
            else:
                if gold & set(rec["cited"]):
                    hit += 1
                scores.append(rec["score"] if rec["score"] is not None else 0.0)
        per_round_calc.append({
            "拒答正确率": rnd_rate(sum(1 for q in refu if runs[k][q]["refused"]),
                                len(refu)),
            "误拒率": rnd_rate(refused_g, len(grad)),
            "出处命中率": rnd_rate(hit, len(grad)),
            "引用支持率": rnd_rate(verif, len(grad)),
            "答案正确率": (round(sum(scores) / len(scores), 4) if scores else None),
        })
    print(f"     {'指标':<12}{'文件mean':>10}{'文件range':>10}"
          f"{'复算mean':>10}{'复算range':>10}")
    for key in per_round_calc[0]:
        vals = [pr[key] for pr in per_round_calc if pr[key] is not None]
        cm = round(sum(vals) / len(vals), 4) if vals else None
        cr = round(max(vals) - min(vals), 4) if vals else None
        fm, fr_ = x["metrics"][key]["mean"], x["metrics"][key]["range"]
        ok = (fm == cm and fr_ == cr)
        print(f"     {key:<12}{fm:>10}{fr_:>10}{cm:>10}{cr:>10}"
              f"{'' if ok else '  <<<'}")
        check(ok, f"{fn}: metrics[{key}] 可复现")

    # ③ 摘要：哪些“翻转”其实不影响结论
    cited_only = sum(1 for u in x["unstable"]
                     if u["cited_flip"] and not u["refuse_flip"]
                     and not u["verify_flip"] and u["score_range"] == 0.0)
    print(f"     [解读] 仅引用集合变化、分数零偏差的题：{cited_only}/{len(x['unstable'])}")

print("\n" + "=" * 78)
print("F. 不稳定题分层：措辞变化 vs 真正影响判分的变化")
print("=" * 78)
print("  说明：『答案翻转率 60%+』大多只是模型换句话写，结论没变。")
print("        真正要盯的是 score_range>0 —— 同一题跨轮判分不一致。\n")
for fn in ("stability_results.json", "stability_no_recheck.json"):
    if not (D / fn).exists():
        continue
    x = load(fn)
    un = x["unstable"]
    score_moved = [u for u in un if (u.get("score_range") or 0) > 0]
    text_only = [u for u in un if u["answer_flip"] and not u["refuse_flip"]
                 and not u["cited_flip"] and not u["verify_flip"]
                 and (u.get("score_range") or 0) == 0]
    tag = "复核开" if x["meta"]["recheck"] else "复核关"
    print(f"  [{tag}] 不稳定 {len(un)} 题 = 判分有变 {len(score_moved)} + "
          f"纯措辞变 {len(text_only)} + 其他 {len(un)-len(score_moved)-len(text_only)}")
    if score_moved:
        print("      判分有变的题：",
              ", ".join(f"{u['qid']}(±{u['score_range']})" for u in score_moved))
    print()

print("\n" + "=" * 78)
print("G. 非满分题（最终口径）")
print("=" * 78)
for r in sorted(graded, key=lambda r: (r.get("judge_score") or 0)):
    s = r.get("judge_score")
    if s is None or s >= 1.0:
        continue
    print(f"  {r['qid']} {r['type']:<10} score={s} verdict={r.get('judge_verdict')} "
          f"覆盖率={r.get('judge_coverage')}")
    print(f"      问：{r['question'][:50]}")
    print(f"      漏：{r.get('judge_missing')}")

print("\n" + "=" * 78)
if FAIL:
    print(f"审计发现 {len(FAIL)} 处不一致：")
    for m in FAIL:
        print("   - " + m)
else:
    print("审计通过：所有指标可复现，字段自洽，引用可回查。")
print("=" * 78)
