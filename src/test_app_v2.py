# -*- coding: utf-8 -*-
"""无头验证新版 app.py（去侧边栏 + 深职大风格主页面 + 默认全开）。

覆盖：
  1. 初始渲染无异常；主页面 Hero/校训/标签都在；无侧边栏组件
  2. 点示例按钮 → 真正作答（出现 assistant 回答）
  3. 拒答分支 → 出现「手册里暂无相关内容」，且无"阈值/相关分"技术词
总耗时约 2~4 分钟（含模型加载）。
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

ROOT = Path(__file__).resolve().parent.parent
APP = str(ROOT / "app.py")

from streamlit.testing.v1 import AppTest  # noqa: E402


def main():
    # ---------- 1. 初始渲染 ----------
    at = AppTest.from_file(APP, default_timeout=300)
    at.run()
    print("== 初始渲染异常:", bool(at.exception), flush=True)
    for e in at.exception:
        print("   EXC:", e, flush=True)

    html_all = "\n".join(m.value for m in at.markdown)
    print("== 主页面关键元素 ==", flush=True)
    for key in ["学生手册智能问答", "深圳职业技术大学", "德业并进", "自强不息",
                "hero", "hero-title", "cite-badge", "chiprow"]:
        print(f"   {'OK ' if key in html_all else 'MISS'} {key}", flush=True)
    print("== 校园文化带（五条）==", flush=True)
    for key in ["校园文化", "校训", "校风", "教风", "学风", "深职精神",
                "崇德立业，创新卓越", "崇教厚德，理实并重",
                "勤学善思，明辨笃行", "守正创新，勇立潮头"]:
        print(f"   {'OK ' if key in html_all else 'MISS'} {key}", flush=True)

    print("== 官方校徽嵌入 ==", flush=True)
    print("   含 crest 容器:", "class=\"crest\"" in html_all, flush=True)
    print("   含 base64 校徽:", "data:image/png;base64," in html_all, flush=True)
    print("   含'缺失'占位:", "校徽<br>缺失" in html_all, flush=True)

    print("== 侧边栏检查（应无可交互元素）==", flush=True)
    print("   sidebar toggle 数:", len(at.toggle), flush=True)
    print("   sidebar slider 数:", len(at.slider), flush=True)
    print("   按钮标签:", [b.label for b in at.button], flush=True)

    # ---------- 2. 示例按钮作答 ----------
    at2 = AppTest.from_file(APP, default_timeout=300)
    at2.run()
    # 注意：button[0] 是「清空对话」，示例题从 button[1] 开始
    sample_btn = next(b for b in at2.button if "补考" in b.label)
    print("   将点击示例按钮:", sample_btn.label, flush=True)
    sample_btn.click().run()
    print("== 示例按钮作答异常:", bool(at2.exception), flush=True)
    for e in at2.exception:
        print("   EXC:", e, flush=True)
    print("   对话消息数（应>=2：1问1答）:", len(at2.chat_message), flush=True)
    md2 = "\n".join(m.value for m in at2.markdown)
    print("   回答里是否含'依据/补考/条款':",
          any(k in md2 for k in ["依据", "补考", "条款", "PDF"]), flush=True)

    # ---------- 3. 拒答分支 ----------
    at3 = AppTest.from_file(APP, default_timeout=300)
    at3.run()
    at3.chat_input[0].set_value("学校食堂几点关门？").run()
    print("== 拒答分支异常:", bool(at3.exception), flush=True)
    warns = [w.value for w in at3.warning]
    joined = "\n".join(warns)
    print("   提示条数:", len(warns), flush=True)
    print("   含'手册里暂无相关内容':", "手册里暂无相关内容" in joined, flush=True)
    print("   泄露技术词(阈值/相关分):",
          ("阈值" in joined or "相关分" in joined), flush=True)
    for w in warns:
        print("   WARN:", w[:90].replace(chr(10), " "), flush=True)

    print("== DONE ==", flush=True)


if __name__ == "__main__":
    main()
