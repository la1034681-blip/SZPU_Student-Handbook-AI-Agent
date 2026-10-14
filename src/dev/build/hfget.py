# -*- coding: utf-8 -*-
"""多线程分段下载器

国内模型站（ModelScope）单连接会被限速到 ~1 MB/s，但支持 Range 分段并发，
实测 8 线程聚合可达 30-40 MB/s，8GB 文件几分钟下完。

用法：
    python src/hfget.py <url> <保存路径> [线程数]
"""
import os
import sys
import time
import threading
import urllib.request

UA = {"User-Agent": "Mozilla/5.0"}


def human(n):
    if n >= 1024 ** 3:
        return "%.2f/%.2f GB" % (n / 1024 ** 3, 0)  # 占位，实际在 report 里拼
    return "%.1f MB" % (n / 1024 / 1024)


def probe_size(url):
    """用 Range: bytes=0-0 探测总大小（跟随 302 跳转）。"""
    req = urllib.request.Request(url, headers=dict(UA, Range="bytes=0-0"))
    with urllib.request.urlopen(req, timeout=60) as r:
        cr = r.headers.get("Content-Range")
        if cr and "/" in cr:
            return int(cr.rsplit("/", 1)[-1])
        return int(r.headers.get("Content-Length") or 0)


def download(url, out, threads=12, seg_mb=16, deadline_s=5400):
    total = probe_size(url)
    if total <= 0:
        raise RuntimeError("探测不到文件大小，可能该源不支持分段")
    if os.path.exists(out) and os.path.getsize(out) == total:
        print("  已存在完整文件，跳过", flush=True)
        return True

    os.makedirs(os.path.dirname(os.path.abspath(out)) or ".", exist_ok=True)
    with open(out, "wb") as f:
        f.truncate(total)

    seg = seg_mb * 1024 * 1024
    todo = [(s, min(s + seg - 1, total - 1)) for s in range(0, total, seg)]
    lock = threading.Lock()
    progress = [0]
    t0 = time.time()
    last = [t0]
    name = os.path.basename(out)

    def report():
        now = time.time()
        if now - last[0] >= 15:
            got = progress[0]
            print("  %s  %.2f/%.2f GB  %.1f%%  %.2f MB/s"
                  % (name, got / 1024 ** 3, total / 1024 ** 3,
                     100.0 * got / total, got / 1048576 / (now - t0)), flush=True)
            last[0] = now

    def worker():
        while True:
            with lock:
                if not todo:
                    return
                s, e = todo.pop()
            if time.time() - t0 > deadline_s:
                return
            need = e - s + 1
            try:
                req = urllib.request.Request(url, headers=dict(
                    UA, Range="bytes=%d-%d" % (s, e)))
                with urllib.request.urlopen(req, timeout=120) as r, \
                        open(out, "r+b") as f:
                    f.seek(s)
                    got = 0
                    while got < need:
                        b = r.read(1 << 20)
                        if not b:
                            break
                        f.write(b)
                        got += len(b)
                        with lock:
                            progress[0] += len(b)
                            report()
                if got < need:                    # 连接被截断，把没下完的尾巴放回队列
                    with lock:
                        todo.append((s + got, e))
            except Exception as ex:
                print("  [重试] %s %s" % (type(ex).__name__, str(ex)[:60]), flush=True)
                time.sleep(2)
                with lock:
                    todo.append((s, e))

    ts = [threading.Thread(target=worker, daemon=True) for _ in range(threads)]
    for t in ts:
        t.start()
    for t in ts:
        t.join()

    ok = progress[0] >= total and os.path.getsize(out) == total
    print("  [%s] %s  %.1f 分钟" % ("完成" if ok else "未完成", name,
                                    (time.time() - t0) / 60), flush=True)
    return ok


if __name__ == "__main__":
    url = sys.argv[1]
    out = sys.argv[2]
    th = int(sys.argv[3]) if len(sys.argv) > 3 else 12
    print("下载 %s\n -> %s（%d 线程）" % (url, out, th), flush=True)
    sys.exit(0 if download(url, out, threads=th) else 1)
