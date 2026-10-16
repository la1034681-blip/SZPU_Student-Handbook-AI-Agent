# -*- coding: utf-8 -*-
"""从 ModelScope 下载 BAAI/bge-m3 模型文件（本地部署用，不走 HuggingFace）。

设计要点：
- 只下载 sentence-transformers 加载所需的白名单文件，跳过 onnx/ 与示例图片；
- 支持断点续传（.part 文件），网络中断可重跑；
- 实时打印进度，写入 models/_download.log 便于外部观察。

用法：
    python src/dev/build/fetch_bge_m3.py [目标目录，默认 models/bge-m3]
"""
import json
import os
import ssl
import sys
import time
import urllib.request

ssl._create_default_https_context = ssl._create_unverified_context

MODEL = "BAAI/bge-m3"
REVISION = "master"
API = f"https://www.modelscope.cn/api/v1/models/{MODEL}/repo/files?Revision={REVISION}&Root="
RAW = f"https://www.modelscope.cn/models/{MODEL}/resolve/{REVISION}/"

# sentence-transformers 加载 bge-m3 必需的文件（跳过 onnx/、imgs/、示例）
NEED = [
    "config.json",
    "config_sentence_transformers.json",
    "sentence_bert_config.json",
    "modules.json",
    "1_Pooling/config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "sentencepiece.bpe.model",
    "model.safetensors",
    "pytorch_model.bin",
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
    files = data.get("Data", {}).get("Files", [])
    return [f["Path"] for f in files]


def download(rel, dest_dir, fh):
    dest = os.path.join(dest_dir, rel.replace("/", os.sep))
    os.makedirs(os.path.dirname(dest) or dest_dir, exist_ok=True)
    part = dest + ".part"
    url = RAW + urllib.parse.quote(rel)

    # 已完整下载则跳过（按远端大小比对）
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
    dest_dir = sys.argv[1] if len(sys.argv) > 1 else "models/bge-m3"
    os.makedirs(dest_dir, exist_ok=True)
    logpath = os.path.join(os.path.dirname(dest_dir) or ".", "_download.log")
    with open(logpath, "w", encoding="utf-8") as fh:
        log(f"目标目录: {os.path.abspath(dest_dir)}", fh)
        try:
            remote = set(list_files())
            log(f"远端文件数: {len(remote)}", fh)
        except Exception as e:
            log(f"列目录失败({str(e)[:80]})，改用白名单直接下载", fh)
            remote = set()

        if "model.safetensors" in remote:
            targets = [f for f in NEED if f == "model.safetensors" or f != "pytorch_model.bin"]
            log("使用 model.safetensors（跳过 pytorch_model.bin）", fh)
        else:
            targets = [f for f in NEED if f != "model.safetensors"]
            log("使用 pytorch_model.bin（无 safetensors）", fh)

        if remote:
            targets = [f for f in targets if f in remote] or list(remote)
        for rel in targets:
            try:
                download(rel, dest_dir, fh)
            except Exception as e:
                log(f"  失败 {rel}: {str(e)[:120]}", fh)
        log("全部完成", fh)
    print("DOWNLOAD_DONE")


if __name__ == "__main__":
    import urllib.parse  # noqa: E402
    main()
