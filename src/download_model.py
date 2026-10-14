# -*- coding: utf-8 -*-
"""Intern-S1-mini GGUF 下载器（多源 + 断点续传 + 自动重试 + 进度日志）

用法：
    python src/download_model.py <目标目录> [档位]
档位：q8（默认，官方 Q8_0，走国内 ModelScope 最快）/ q4 / q5（社区 imatrix，走 HF 镜像）
"""
import os
import sys
import time
import urllib.request

# 两个源：modelscope 国内快、hf-mirror 国内可通但慢
SOURCES = {
    "ms": "https://modelscope.cn/models/{repo}/resolve/master/{rel}",
    "hf": "https://hf-mirror.com/{repo}/resolve/main/{rel}",
}

# 档位 -> [(源, 仓库, 文件相对路径), ...]  最后一个通常是视觉投影文件
VARIANTS = {
    "q8": [
        ("ms", "Shanghai_AI_Laboratory/Intern-S1-mini-GGUF", "Q8_0/Intern-S1-mini-Q8_0.gguf"),
        ("ms", "Shanghai_AI_Laboratory/Intern-S1-mini-GGUF", "Q8_0/mmproj-Intern-S1-mini-Q8_0.gguf"),
    ],
    "q4": [
        ("hf", "mradermacher/Intern-S1-mini-i1-GGUF", "Intern-S1-mini.i1-Q4_K_M.gguf"),
        ("ms", "Shanghai_AI_Laboratory/Intern-S1-mini-GGUF", "f16/mmproj-Intern-S1-mini-f16.gguf"),
    ],
    "q5": [
        ("hf", "mradermacher/Intern-S1-mini-i1-GGUF", "Intern-S1-mini.i1-Q5_K_M.gguf"),
        ("ms", "Shanghai_AI_Laboratory/Intern-S1-mini-GGUF", "f16/mmproj-Intern-S1-mini-f16.gguf"),
    ],
}


def human(n):
    if n >= 1024 ** 3:
        return "%.2f GB" % (n / 1024 ** 3)
    return "%.1f MB" % (n / 1024 / 1024)


def fetch(src, repo, rel, dest_dir, max_try=40):
    """下载单个文件，支持断点续传；返回是否成功。"""
    os.makedirs(dest_dir, exist_ok=True)
    out = os.path.join(dest_dir, os.path.basename(rel))
    url = SOURCES[src].format(repo=repo, rel=rel)
    name = os.path.basename(rel)

    for attempt in range(1, max_try + 1):
        have = os.path.getsize(out) if os.path.exists(out) else 0
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            if have:
                req.add_header("Range", "bytes=%d-" % have)
            t0 = time.time()
            with urllib.request.urlopen(req, timeout=60) as r:
                total = have + int(r.headers.get("Content-Length") or 0)
                mode = "ab" if (have and r.status == 206) else "wb"
                if mode == "wb":
                    have = 0
                done = have
                last = time.time()
                with open(out, mode) as f:
                    while True:
                        chunk = r.read(524288)
                        if not chunk:
                            break
                        f.write(chunk)
                        done += len(chunk)
                        now = time.time()
                        if now - last > 20:
                            sp = done / 1024 / 1024 / (now - t0)
                            pct = 100.0 * done / total if total else 0
                            print("  %s  %s/%s  %.1f%%  %.2f MB/s"
                                  % (name, human(done), human(total), pct, sp), flush=True)
                            last = now
            size = os.path.getsize(out)
            if total and size >= total:
                print("  [完成] %s  %s" % (name, human(size)), flush=True)
                return True
            print("  [续传] %s 已 %s，继续第 %d 次" % (name, human(size), attempt), flush=True)
        except Exception as e:
            print("  [重试 %d] %s %s，5 秒后继续" % (attempt, type(e).__name__, e), flush=True)
            time.sleep(5)
    print("  [失败] %s 重试 %d 次仍未完成" % (rel, max_try), flush=True)
    return False


def main():
    dest = sys.argv[1] if len(sys.argv) > 1 else os.path.join(
        os.path.expanduser("~"), "Desktop", "Intern-S1-mini-GGUF")
    key = (sys.argv[2] if len(sys.argv) > 2 else "q8").lower()
    files = VARIANTS[key]
    print("目标目录: %s" % dest, flush=True)
    print("档位: %s，共 %d 个文件" % (key, len(files)), flush=True)
    t0 = time.time()
    results = [fetch(*meta, dest) for meta in files]
    print("总计耗时 %.1f 分钟   结果: %s"
          % ((time.time() - t0) / 60,
             "、".join("%s=%s" % (os.path.basename(f[2]), "OK" if r else "FAIL")
                       for f, r in zip(files, results))), flush=True)


if __name__ == "__main__":
    main()
