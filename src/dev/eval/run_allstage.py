# -*- coding: utf-8 -*-
"""全流程一次性体检（All-Stage Run）

从「环境 → 数据资产 → 结构化 → 索引 → 检索 → 生成 → 服务」整条链路跑一遍，
每个环节独立计时、判定状态（OK / WARN / FAIL），并把本轮指标与历史基线对照，
标出「正常」或「异常波动」。

用法
----
    python src/run_allstage.py                # 全流程（含 60 题端到端，约 15~20 分钟）
    python src/run_allstage.py --skip-gen     # 跳过最慢的生成评测
    python src/run_allstage.py --limit 8      # 生成评测只跑前 8 题（冒烟）
"""
import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

ROOT = Path(__file__).resolve().parent.parent.parent.parent
PY = sys.executable
LOG_DIR = ROOT / "logs"
DATA = ROOT / "data"

# --------------------------------------------------------------------------
# 历史基线（上一轮定稿实测，用于判断"是否异常波动"）
# --------------------------------------------------------------------------
BASE_GEN = {                      # 2026-10-14 修复 N05 后的定稿轮
    "答案正确率": 0.9623,         # 来源 data/allstage_gen_fixed.json
    "出处命中率": 1.0,
    "误拒率": 0.0,
    "拒答正确率": 1.0,
    "平均延迟": 6.47,
}
BASE_RETRIEVAL = None             # 运行时从 data/eval_results.json 读
TOL = {                           # 波动阈值（超出即 WARN）
    "答案正确率": 0.05,           # 掉 5 个百分点
    "出处命中率": 0.02,
    "误拒率": 0.02,               # 误拒上升 2 个百分点
    "拒答正确率": 0.02,
    "平均延迟": 0.40,             # 延迟涨 40%
}

results = []                      # 每个环节的记录


def hr(title):
    print("\n" + "=" * 78)
    print(f"  {title}")
    print("=" * 78, flush=True)


def md5_head(p, n=8):
    h = hashlib.md5()
    with open(p, "rb") as f:
        for blk in iter(lambda: f.read(1 << 20), b""):
            h.update(blk)
    return h.hexdigest()[:n]


def _is_synthetic(cid):
    """无条号的合成块号：F0X-AX01（卷首/总则）、F0X-APP（附件/表单）。
    这类块参与检索但不进条号索引，属设计如此。"""
    if not cid:
        return False
    return ("-AX" in cid) or cid.endswith("-APP")


def rec(stage, name, status, elapsed, detail="", metrics=None, anomalies=None):
    r = {"stage": stage, "name": name, "status": status,
         "elapsed": round(elapsed, 2), "detail": detail,
         "metrics": metrics or {}, "anomalies": anomalies or []}
    results.append(r)
    flag = {"OK": "[OK]  ", "WARN": "[WARN]", "FAIL": "[FAIL]", "SKIP": "[SKIP]"}[status]
    print(f"  {flag} {name}  {elapsed:6.1f}s  {detail}", flush=True)
    return r


def run_sub(cmd, log_path):
    """跑子进程，stdout/stderr 实时落日志文件。"""
    log_path.parent.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ, PYTHONIOENCODING="utf-8", PYTHONUTF8="1")
    t0 = time.time()
    with open(log_path, "w", encoding="utf-8") as lf:
        p = subprocess.run(cmd, cwd=str(ROOT), stdout=lf, stderr=subprocess.STDOUT,
                           env=env, text=True)
    return p.returncode, time.time() - t0, log_path


# --------------------------------------------------------------------------
# S0 环境与依赖
# --------------------------------------------------------------------------
def stage0_env():
    hr("S0 · 环境与依赖体检")
    t0 = time.time()
    detail, metrics, anomalies = [], {}, []
    try:
        import torch
        metrics["python"] = sys.version.split()[0]
        metrics["torch"] = torch.__version__
        metrics["cuda"] = bool(torch.cuda.is_available())
        if torch.cuda.is_available():
            metrics["gpu"] = torch.cuda.get_device_name(0)
            total = torch.cuda.get_device_properties(0).total_memory / 1024**3
            metrics["显存GB"] = round(total, 1)
        else:
            anomalies.append("CUDA 不可用，将退化为 CPU（延迟会显著上升）")
        for m in ("sentence_transformers", "langchain", "streamlit", "numpy"):
            try:
                metrics[m] = getattr(__import__(m), "__version__", "?")
            except Exception as e:
                anomalies.append(f"{m} 导入失败：{e}")
        detail.append(f"py{sys.version.split()[0]} torch{torch.__version__} "
                      f"CUDA={metrics['cuda']}")
    except Exception as e:
        return rec("S0", "环境与依赖", "FAIL", time.time() - t0,
                   f"torch 导入失败：{e}", metrics, [str(e)])

    # API key 是否就位（只看存在性，不回显）
    env_file = ROOT / ".env"
    has_key = False
    if env_file.exists():
        for line in env_file.read_text(encoding="utf-8").splitlines():
            if line.strip().startswith("DEEPSEEK_API_KEY=") and \
               len(line.split("=", 1)[1].strip()) > 10:
                has_key = True
    metrics["api_key"] = has_key
    if not has_key:
        anomalies.append("未检测到 DEEPSEEK_API_KEY，生成层将无法调用模型")
    detail.append(f"API密钥={'已配置' if has_key else '缺失'}")
    status = "OK" if (metrics.get("cuda") and has_key and not anomalies) else "WARN"
    return rec("S0", "环境与依赖", status, time.time() - t0,
               " ｜ ".join(detail), metrics, anomalies)


# --------------------------------------------------------------------------
# S1 数据资产
# --------------------------------------------------------------------------
def stage1_assets():
    hr("S1 · 数据资产盘点（PDF → 条款 → 索引）")
    t0 = time.time()
    detail, metrics, anomalies = [], {}, []
    files = {
        "PDF原文件": ROOT / "附件1. 学生手册(2025年修订)(1).pdf",
        "结构化条款": DATA / "clauses.json",
        "条款索引": DATA / "clause_index.json",
        "检索分块": DATA / "index_chunks.jsonl",
        "稠密向量": DATA / "vectors.npy",
        "评测集": DATA / "eval_set.json",
    }
    for name, p in files.items():
        if not p.exists():
            anomalies.append(f"缺失：{name} ({p.name})")
            metrics[name] = None
            continue
        metrics[name] = f"{p.stat().st_size/1024:.0f}KB md5:{md5_head(p)}"
    # 计数
    try:
        n_chunks = sum(1 for _ in (DATA / "index_chunks.jsonl").open(encoding="utf-8"))
        metrics["分块数"] = n_chunks
        ci = json.load((DATA / "clause_index.json").open(encoding="utf-8"))
        # clause_index 结构：{"文件ID|条号": "chunk_id"}（条级索引，非 clause_id 集合）
        metrics["条款索引键数"] = len(ci) if isinstance(ci, (list, dict)) else "?"
        ev = json.load((DATA / "eval_set.json").open(encoding="utf-8"))
        items = ev["items"]
        metrics["评测题数"] = len(items)
        metrics["题型分布"] = {}
        for it in items:
            metrics["题型分布"][it["type"]] = metrics["题型分布"].get(it["type"], 0) + 1
        detail.append(f"条款索引{metrics.get('条款索引键数')} 分块{n_chunks} 题{len(items)}")
    except Exception as e:
        anomalies.append(f"计数失败：{e}")

    # 模型权重
    for m in ("bge-m3", "bge-reranker-v2-m3"):
        p = ROOT / "models" / m
        ok = (p / "config.json").exists()
        metrics[f"模型/{m}"] = "就位" if ok else "缺失"
        if not ok:
            anomalies.append(f"本地模型缺失：{m}")

    status = "OK" if not anomalies else "FAIL"
    return rec("S1", "数据资产", status, time.time() - t0,
               " ｜ ".join(detail), metrics, anomalies)


# --------------------------------------------------------------------------
# S2 结构与索引一致性
# --------------------------------------------------------------------------
def stage2_consistency():
    hr("S2 · 结构化与索引一致性校验")
    t0 = time.time()
    detail, metrics, anomalies = [], {}, []
    try:
        import numpy as np
        chunks = [json.loads(l) for l in
                  (DATA / "index_chunks.jsonl").open(encoding="utf-8")]
        vecs = np.load(DATA / "vectors.npy")
        ci = json.load((DATA / "clause_index.json").open(encoding="utf-8"))

        metrics["分块数"] = len(chunks)
        metrics["向量shape"] = list(vecs.shape)
        metrics["向量维度"] = vecs.shape[1]
        # 1) 向量数 == 分块数
        if vecs.shape[0] != len(chunks):
            anomalies.append(f"向量数({vecs.shape[0]}) ≠ 分块数({len(chunks)})")
        # 2) 维度
        if vecs.shape[1] != 1024:
            anomalies.append(f"向量维度异常：{vecs.shape[1]}（应为 1024）")
        # 3) NaN / 零向量
        nan = int(np.isnan(vecs).any(axis=1).sum())
        zero = int((np.abs(vecs).sum(axis=1) < 1e-6).sum())
        metrics["NaN向量"] = nan
        metrics["零向量"] = zero
        if nan:
            anomalies.append(f"{nan} 条向量含 NaN")
        if zero:
            anomalies.append(f"{zero} 条零向量（检索会失效）")
        # 4) 空文本
        empty = sum(1 for c in chunks if not (c.get("embed_text") or "").strip())
        metrics["空文本块"] = empty
        if empty:
            anomalies.append(f"{empty} 条分块文本为空")
        # 5) 条款覆盖：chunk 的 clause_id 必须出现在条款索引的值集合里
        #    （clause_index 是 {"文件ID|条号": "chunk_id"}）
        #    例外：-AX**（卷首/总则等无条号内容）与 -APP（附件/表单）是合成的块号，
        #    本就不进条号索引，单列统计、不计为异常。
        flat = set()
        for v in (ci.values() if isinstance(ci, dict) else ci):
            if isinstance(v, list):
                flat.update(v)
            else:
                flat.add(v)
        synthetic = [c for c in chunks if _is_synthetic(c.get("clause_id"))]
        checkable = [c for c in chunks if not _is_synthetic(c.get("clause_id"))]
        miss = [c.get("clause_id") for c in checkable
                if c.get("clause_id") and c["clause_id"] not in flat]
        metrics["条款索引条目"] = len(flat)
        metrics["无条号块(AX/APP)"] = len(synthetic)
        metrics["孤儿分块"] = len(miss)
        if miss:
            anomalies.append(f"{len(miss)} 条分块的 clause_id 不在条款索引中")
        # 6) 分块主键唯一性（clause_id 有重复属正常：款块继承父条编号）
        ids = [c.get("chunk_id") for c in chunks]
        dup = len(ids) - len(set(ids))
        metrics["重复chunk_id"] = dup
        if dup:
            anomalies.append(f"{dup} 个重复 chunk_id（主键应唯一）")
        metrics["共享clause_id的分块"] = len(ids) - len({c.get("clause_id") for c in chunks})
        detail.append(f"向量{vecs.shape} 孤儿{len(miss)} 空块{empty} NaN{nan}")
    except Exception as e:
        return rec("S2", "结构一致性", "FAIL", time.time() - t0,
                   f"校验异常：{e}", metrics, [str(e)])
    status = "OK" if not anomalies else ("WARN" if len(anomalies) <= 1 else "FAIL")
    return rec("S2", "结构一致性", status, time.time() - t0,
               " ｜ ".join(detail), metrics, anomalies)


# --------------------------------------------------------------------------
# S3 检索层（四策略消融）
# --------------------------------------------------------------------------
def stage3_retrieval(reuse=False):
    hr("S3 · 检索层（BM25 / 向量 / 混合 / 重排 四策略）")
    out_json = DATA / "allstage_ret.json"
    reused = reuse and out_json.exists()
    if reused:
        el, detail_note = 0.0, "（复用上一轮结果）"
    else:
        detail_note = ""
        code, el, log = run_sub(
            [PY, "src/evaluate_retrieval.py", "data", "--out", "allstage_ret.json"],
            LOG_DIR / "allstage_ret.log")
        if code != 0 or not out_json.exists():
            return rec("S3", "检索层评测", "FAIL", el,
                       f"子进程退出码 {code}，见 logs/allstage_ret.log", {},
                       [f"exit={code}"])
    detail, metrics, anomalies = [detail_note] if detail_note else [], {}, []
    out = json.load(out_json.open(encoding="utf-8"))
    sm = out["summary"]
    for s in ("bm25", "dense", "hybrid", "rerank"):
        if s in sm:
            o = sm[s]["overall"]
            metrics[s] = {k: round(o[k], 4) for k in
                          ("recall@1", "recall@5", "recall@10", "mrr@10")}
    if "rerank" in sm and "hybrid" in sm:
        d = sm["rerank"]["overall"]["recall@5"] - sm["hybrid"]["overall"]["recall@5"]
        metrics["重排相对混合 R@5 增益"] = round(d, 4)
        detail.append(f"重排R@5={sm['rerank']['overall']['recall@5']:.3f} "
                      f"(混合{sm['hybrid']['overall']['recall@5']:.3f})")
    # 与历史基线对比
    base_p = DATA / "eval_results.json"
    if base_p.exists():
        try:
            b = json.load(base_p.open(encoding="utf-8"))["summary"]
            for s in ("dense", "hybrid", "rerank"):
                if s in b and s in sm:
                    dd = sm[s]["overall"]["recall@5"] - b[s]["overall"]["recall@5"]
                    metrics[f"基线Δ_{s}"] = round(dd, 4)
                    if dd < -0.03:
                        anomalies.append(f"{s} R@5 较基线跌 {abs(dd):.3f}")
        except Exception as e:
            anomalies.append(f"基线对比失败：{e}")
    if "rerank" in sm and sm["rerank"]["overall"]["recall@5"] < 0.85:
        anomalies.append("重排后 R@5 低于 0.85，检索质量下滑")
    status = "OK" if not anomalies else "WARN"
    return rec("S3", "检索层评测", status, el, " ｜ ".join(detail), metrics, anomalies)


# --------------------------------------------------------------------------
# S4 生成层（Agent 全链路 60 题）
# --------------------------------------------------------------------------
def stage4_generation(limit=0, smoke=False, reuse=False):
    hr("S4 · 生成层（规划 + 多路检索 + 生成校验 + 自检回环）")
    out_name = "allstage_gen.json" if not smoke else "allstage_smoke.json"
    out_json = DATA / out_name
    reused = reuse and out_json.exists() and not limit
    detail, metrics, anomalies = [], {}, []
    if reused:
        el, log_name = 0.0, "allstage_gen.log"
        detail.append("（复用上一轮结果）")
    else:
        cmd = [PY, "src/evaluate_generation.py", "data", "--agent",
               "--out", out_name]
        if limit:
            cmd += ["--limit", str(limit)]
        log_name = "allstage_gen.log" if not smoke else "allstage_smoke.log"
        code, el, log = run_sub(cmd, LOG_DIR / log_name)
        if code != 0 or not out_json.exists():
            return rec("S4", "生成层评测", "FAIL", el,
                       f"子进程退出码 {code}，见 logs/{log_name}", metrics,
                       [f"exit={code}"])
    out = json.load(out_json.open(encoding="utf-8"))
    sm, items = out["summary"], out["items"]
    for k in ("n", "答案正确率", "已作答正确率", "完全正确率", "要点覆盖率",
              "出处命中率", "误拒率", "拒答正确率", "平均延迟",
              "拆解触发率", "自检触发率", "前置拒答数"):
        metrics[k] = sm[k]
    # 延迟分位
    lat = sorted(x["latency"] for x in items)
    n = len(lat)
    metrics["延迟min"] = lat[0]
    metrics["延迟p50"] = lat[n // 2]
    metrics["延迟p90"] = lat[int(n * 0.9)]
    metrics["延迟max"] = lat[-1]
    metrics["超12s题数"] = sum(1 for x in items if x["latency"] > 12)
    metrics["用例总耗时s"] = out["meta"]["elapsed"]

    # ---- 基线对比（仅全量 60 题时做，冒烟不对比） ----
    if not limit and not smoke:
        for k, bv in BASE_GEN.items():
            v = metrics.get(k)
            if v is None:
                continue
            delta = v - bv
            tol = TOL.get(k, 0.05)
            metrics[f"Δ{k}"] = round(delta, 4)
            bad = (delta < -tol) if k != "误拒率" and k != "平均延迟" else \
                  (delta > tol)
            if bad:
                anomalies.append(f"{k} 较基线{bv:.3f}→{v:.3f}（Δ{delta:+.3f}）")
        for x in items:
            if x["latency"] > 25:
                anomalies.append(f"{x['qid']} 延迟 {x['latency']}s 异常偏高")
    detail.append(f"答案{metrics['答案正确率']:.3f} 出处{metrics['出处命中率']:.3f} "
                  f"误拒{metrics['误拒率']:.3f} p50={metrics['延迟p50']}s")
    status = "OK" if not anomalies else "WARN"
    return rec("S4", "生成层评测", status, el, " ｜ ".join(detail), metrics, anomalies)


# --------------------------------------------------------------------------
# S5 服务层（网页界面无头自检）
# --------------------------------------------------------------------------
def stage5_service():
    hr("S5 · 服务层（Streamlit 网页界面无头自检）")
    code, el, log = run_sub([PY, "src/test_app_v2.py"], LOG_DIR / "allstage_app.log")
    detail, metrics, anomalies = [], {}, []
    txt = log.read_text(encoding="utf-8", errors="replace") if log.exists() else ""
    ok = ("零异常" in txt or "全部通过" in txt or "PASS" in txt.upper()) and code == 0
    metrics["退出码"] = code
    metrics["日志行数"] = len(txt.splitlines())
    if code != 0:
        anomalies.append(f"无头自检退出码 {code}")
    detail.append(f"退出码{code} 日志{metrics['日志行数']}行")
    status = "OK" if not anomalies else "WARN"
    return rec("S5", "服务层界面自检", status, el, " ｜ ".join(detail),
               metrics, anomalies)


# --------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0, help="生成评测只跑前 N 题")
    ap.add_argument("--smoke", action="store_true", help="冒烟模式（不对比基线）")
    ap.add_argument("--skip-gen", action="store_true")
    ap.add_argument("--skip-service", action="store_true")
    ap.add_argument("--reuse", action="store_true",
                    help="复用已有的 allstage_ret.json / allstage_gen.json，不重跑子进程")
    ap.add_argument("--quick", action="store_true",
                    help="只跑 S0/S1/S2 三个快速环节（秒级）")
    a = ap.parse_args()

    t_all = time.time()
    print("全流程体检开始", time.strftime("%Y-%m-%d %H:%M:%S"), flush=True)
    stage0_env()
    stage1_assets()
    stage2_consistency()
    if a.quick:
        total = time.time() - t_all
        n_ok = sum(1 for r in results if r["status"] == "OK")
        n_warn = sum(1 for r in results if r["status"] == "WARN")
        n_fail = sum(1 for r in results if r["status"] == "FAIL")
        hr("快速复核汇总")
        print(f"  环节 {len(results)}：OK {n_ok} ｜ WARN {n_warn} ｜ FAIL {n_fail}"
              f" ｜ 总耗时 {total:.1f}s")
        for r in results:
            print(f"    {r['status']:<5} {r['name']:<14} {r['elapsed']:>7.1f}s  {r['detail']}")
            for an in r["anomalies"]:
                print(f"          ! {an}")
        out = {"meta": {"built_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                        "total_elapsed": round(total, 1), "mode": "quick",
                        "counts": {"OK": n_ok, "WARN": n_warn, "FAIL": n_fail}},
               "stages": results}
        (DATA / "allstage_quick.json").write_text(
            json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
        print(f"\n记录已写入 {DATA/'allstage_quick.json'}\n", flush=True)
        return
    stage3_retrieval(reuse=a.reuse)
    if not a.skip_gen:
        stage4_generation(limit=a.limit, smoke=a.smoke, reuse=a.reuse)
    else:
        rec("S4", "生成层评测", "SKIP", 0.0, "本次跳过")
    if not a.skip_service:
        stage5_service()
    else:
        rec("S5", "服务层界面自检", "SKIP", 0.0, "本次跳过")

    total = time.time() - t_all
    n_ok = sum(1 for r in results if r["status"] == "OK")
    n_warn = sum(1 for r in results if r["status"] == "WARN")
    n_fail = sum(1 for r in results if r["status"] == "FAIL")
    n_skip = sum(1 for r in results if r["status"] == "SKIP")

    hr("全流程汇总")
    print(f"  环节 {len(results)} 个：OK {n_ok} ｜ WARN {n_warn} ｜ "
          f"FAIL {n_fail} ｜ SKIP {n_skip}")
    print(f"  总耗时 {total:.1f}s ({total/60:.1f} 分钟)")
    for r in results:
        print(f"    {r['status']:<5} {r['name']:<14} {r['elapsed']:>7.1f}s")
        for an in r["anomalies"]:
            print(f"          ! {an}")
    print("=" * 78, flush=True)

    stamp = time.strftime("%Y%m%d-%H%M%S")
    out = {"meta": {"built_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                    "total_elapsed": round(total, 1),
                    "counts": {"OK": n_ok, "WARN": n_warn, "FAIL": n_fail,
                               "SKIP": n_skip}},
           "stages": results}
    p = DATA / f"allstage_{stamp}.json"
    p.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\n记录已写入 {p}\n")


if __name__ == "__main__":
    main()
