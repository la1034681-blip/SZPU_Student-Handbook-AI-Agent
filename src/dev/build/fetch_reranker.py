# -*- coding: utf-8 -*-
"""从 ModelScope 下载 BAAI/bge-reranker-v2-m3（阶段 3 重排模型，本地离线部署）。

与 fetch_bge_m3.py 同源逻辑，区别只有两点：
- 模型改为 bge-reranker-v2-m3（XLM-RoBERTa 架构的 cross-encoder）；
- 权重使用 model.safetensors（该仓库只提供 safetensors，无 pytorch_model.bin）。

cross-encoder 不需要 pooling / sentence_bert_config，故白名单只保留
config + tokenizer + 权重。约 2.19 GB。

用法：
    python src/dev/build/fetch_reranker.py [目标目录，默认 models/bge-reranker-v2-m3]
"""
import json
import os
import ssl
import sys
import time
import urllib.parse
import urllib.request

ssl._create_default_https_context = ssl._create_unverified_context

MODEL = "BAAI/bge-reranker-v2-m3"
REVISION = "master"
API = f"https://www.modelscope.cn/api/v1/models/{MODEL}/repo/files?Revision={REVISION}&Root="
RAW = f"https://www.modelscope.cn/models/{MODEL}/resolve/{REVISION}/"

# CrossEncoder 加载必需：模型结构 + 分词器 + 权重
NEED = [
    "config.json",
    "configuration.json",
    "model.safetensors",
    "sentencepiece.bpe.model",
    "special_tokens_map.json",
    "tokenizer.json",
    "tokenizer_config.json",
]
UA = {"User-Agent": "Mozilla/5.0"}


def log(msg, fh=None):
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    if fh:
        fh.write(line + "\n")
        fh.flush()


def list_files():
    req = urllib.request.Request(API, headers=UA)
    data = json.loads(urllib.request.urlopen(req, timeout=30).read().decode("utf-8"))
    return [f["Path"] for f in data.get("Data", {}).get("Files", [])]


def download(rel, dest_dir, fh):
    dest = os.path.join(dest_dir, rel.replace("/", os.sep))
    os.makedirs(os.path.dirname(dest) or dest_dir, exist_ok=True)
    part = dest + ".part"
    url = RAW + urllib.parse.quote(rel)

    req = urllib.request.Request(url, headers=UA, method="HEAD")
    try:
        total = int(urllib.request.urlopen(req, timeout=30).headers.get("Content-Length", 0))
    except Exception:
        total = 0
    if os.path.exists(dest) and total and os.path.getsize(dest) == total:
        log(f"  跳过（已存在） {rel}  {total/2**20:.1f} MB", fh)
        return dest

    done = os.path.getsize(part) if os.path.exists(part) else 0
    headers = dict(UA)
    if done:
        headers["Range"] = f"bytes={done}-"
    req = urllib.request.Request(url, headers=headers)
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=60) as r, open(part, "ab") as out:
        while True:
            b = r.read(1024 * 1024)
            if not b:
                break
            out.write(b)
            done += len(b)
            if total and time.time() - t0 > 2:
                pct = done * 100.0 / total
                spd = done / max(time.time() - t0, 1e-6) / 2**20
                log(f"  {rel}  {pct:5.1f}%  {done/2**20:6.1f}/{total/2**20:.1f} MB  {spd:5.1f} MB/s", fh)
                t0 = time.time()
    os.replace(part, dest)
    log(f"  完成 {rel}  {os.path.getsize(dest)/2**20:.1f} MB", fh)
    return dest


def main():
    dest_dir = sys.argv[1] if len(sys.argv) > 1 else "models/bge-reranker-v2-m3"
    os.makedirs(dest_dir, exist_ok=True)
    logpath = os.path.join(os.path.dirname(dest_dir) or ".", "_download_reranker.log")
    with open(logpath, "w", encoding="utf-8") as fh:
        log(f"目标目录: {os.path.abspath(dest_dir)}", fh)
        try:
            remote = set(list_files())
            log(f"远端文件数: {len(remote)}", fh)
        except Exception as e:
            log(f"列目录失败({str(e)[:80]})，改用白名单直接下载", fh)
            remote = set()

        targets = [f for f in NEED if (not remote or f in remote)]
        if remote and not targets:
            targets = list(remote)
        for rel in targets:
            try:
                download(rel, dest_dir, fh)
            except Exception as e:
                log(f"  失败 {rel}: {str(e)[:120]}", fh)
        log("全部完成", fh)
    print("DOWNLOAD_DONE")


if __name__ == "__main__":
    main()
