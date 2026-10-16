# -*- coding: utf-8 -*-
"""换硅基流动的钥匙：先验通，再写本地 .env，最后给出线上要粘的那一行。

    python src/dev/diag/set_sf_key.py sk-xxxxxxxx

为什么把「验」和「写」焊在一起（2026-10-16 教训）：
用户连续给了两把钥匙，平台都回 `30014 Token is invalid`（这串字符在平台档案里
根本不存在）。如果直接写进配置，本地和线上会**同时**再坏一次，而且坏掉的症状
还是「查不到手册内容」，看不出到底是钥匙不存在、账号欠费、还是代码有问题。
所以本脚本的规矩是：三关全过才允许动文件；只要有一关不过，原样退出，
**一个字节都不改**，并告诉你卡在哪一关。

三关（顺序即因果）：
  ① 认不认这把钥匙  —— 列得出模型清单才算过（30014 卡这里）
  ② 账号给不给服务  —— 能算出一个向量才算过（30001 卡这里，账号余额为 0）
  ③ 重排接口通不通  —— 能排出名次才算过
三关都过 = 真正的「可用钥匙」，这时候再改配置才有意义。
"""
from __future__ import annotations

import re
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.stdout.reconfigure(encoding="utf-8")

from check_sf_key import BASE, EMB, RRK, _post            # noqa: E402

ENV = ROOT / ".env"


def _clean(raw: str) -> str:
    """复制粘贴最常见的两种脏：前后空格、被换行截成两半。"""
    return re.sub(r"\s+", "", (raw or "").strip())


def verify(key: str) -> tuple[bool, str]:
    """跑三关。返回 (是否全过, 卡在哪一关/全过的说明)。"""
    import requests

    # ① 钥匙在不在
    try:
        r = requests.get(f"{BASE}/models", timeout=30,
                         headers={"Authorization": f"Bearer {key}"})
        d = r.json()
    except Exception as e:                                # noqa: BLE001
        return False, f"第①关连不上：{type(e).__name__}: {e}"
    # ⚠ 平台出错时回 {"code":30014,"data":null} —— "data" 键**在**、值是 None，
    #   所以只能判「有没有内容」，不能判「键在不在」（踩过两次）。
    if not d.get("data"):
        code, msg = d.get("code"), d.get("message")
        if code == 30014:
            return False, ("第①关没过：平台不认识这串字符（30014 Token is invalid）。"
                           "→ 去控制台点「复制」重抄一次，别手打、别从截图抄；"
                           "并确认登录的站点和钥匙所属站点一致。")
        return False, f"第①关没过：code={code} message={msg}"
    got = len(d["data"])
    for m in (EMB, RRK):
        if m not in [x.get("id") for x in d["data"]]:
            return False, f"第①关过了，但模型清单里缺 {m} —— 换一把钥匙或核对账号。"

    # ② 账号给不给服务（这一步才是"欠费"的照妖镜）
    r = _post("/embeddings", key, {"model": EMB, "input": "补考成绩怎么算"})
    e = r.json()
    if not e.get("data"):
        code, msg = e.get("code"), e.get("message")
        if code == 30001 or "balance" in str(msg).lower():
            return False, ("第①关过了（钥匙是真的），第②关没过：账号余额不足"
                           "（30001）。→ 这是**账号**问题，跟模型免不免费无关；"
                           "给这个账号充一点钱让余额 > 0，服务会自动恢复，"
                           "钥匙不用换。")
        return False, f"第②关没过：code={code} message={msg}"
    dim = len(e["data"][0]["embedding"])
    if dim != 1024:
        return False, f"第②关算出的向量是 {dim} 维，本该 1024 维 —— 别用这把改配置。"

    # ③ 重排
    r = _post("/rerank", key, {"model": RRK, "query": "补考",
                               "documents": ["必修课不及格可参加补考", "宿舍熄灯时间"],
                               "top_n": 2})
    rr = r.json()
    if not rr.get("results"):
        return False, f"第③关没过：{rr}"
    return True, f"三关全过（模型 {got} 个、向量 {dim} 维、重排正常）"


def write_env(key: str) -> Path:
    """只替换 EMBED_API_KEY 那一行，其余原样；先备份。"""
    text = ENV.read_text(encoding="utf-8")
    if not re.search(r"(?m)^EMBED_API_KEY=", text):
        raise SystemExit(f"⚠ {ENV} 里没有 EMBED_API_KEY 这一行，拒绝盲改。")
    bak = ENV.with_name(f".env.bak.{datetime.now():%Y%m%d-%H%M%S}")
    bak.write_text(text, encoding="utf-8")
    new = re.sub(r"(?m)^EMBED_API_KEY=.*$", f"EMBED_API_KEY={key}", text)
    ENV.write_text(new, encoding="utf-8")
    return bak


def main() -> int:
    args = [a for a in sys.argv[1:]]
    force = False
    if "--force" in args:
        force = True
        args = [a for a in args if a != "--force"]
    raw = " ".join(args)
    key = _clean(raw)
    if not key:
        print("用法：python src/dev/diag/set_sf_key.py sk-xxxxxxxx [--force]")
        print("  --force：明知三关没过也照写（默认拒绝）。只用于「我想亲眼看一次线上什么样」"
              "这类实验，**不会**把坏钥匙变成好钥匙。")
        return 2

    print(f"待验钥匙：{key[:8]}…{key[-4:]}（共 {len(key)} 字符）")
    if len(key) != 51 or not key.startswith("sk-"):
        print("⚠ 形态就不对（正常是 sk- + 48 位 = 51 字符）—— 多半是复制不全，"
              "但既然你给了，我还是照实打一遍。")

    ok, why = verify(key)
    print(("✓ " if ok else "✗ ") + why)
    if not ok and not force:
        print("\n结论：**没有改动任何文件**（.env 与线上 Secrets 都保持原样）。")
        print("先把钥匙这条搞定，再谈改配置 —— 改配置不能修好一把坏钥匙。")
        return 1
    if not ok and force:
        print("\n⚠ --force 已指定：明知这关没过，仍按你的要求写入。")
        print("  请记住：写进去不会让它变好。线上表现会是「能答，但检索打了折扣」"
              "（页面会有黄条），精度低于正常。要恢复满血，仍需一把三关全过的钥匙。")

    bak = write_env(key)
    print(f"\n✓ 已写入 {ENV}")
    print(f"  旧文件备份在 {bak.name}（改坏了直接换回来）")

    print("\n—— 线上 Streamlit Cloud 要做的（只有一步）——")
    print("打开 Manage app → Settings → Secrets，把 **EMBED_API_KEY 这一行**")
    print("整行替换成下面这行，其他行一个字都别碰，然后保存：\n")
    print(f'EMBED_API_KEY = "{key}"')
    print("\n保存后它自己会重启，等 1~2 分钟；卡住就 Manage app → ⋮ → Reboot。")
    print("⚠ 本地 .env 与线上 Secrets 是两份独立配置，改一份不会同步另一份 ——"
          "两处都要换，否则就是「改了没生效」。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
