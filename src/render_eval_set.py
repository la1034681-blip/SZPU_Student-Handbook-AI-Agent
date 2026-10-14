# -*- coding: utf-8 -*-
"""把 data/eval_set.json 渲染成可读的桌面 txt，并校验出处真实性。

两件事：
 1. 校验：每条题的每个 sources.clause_id 必须真实存在于 data/clauses.json，
    且 article_no / page 与库中记录一致。任何对不上就报错退出 ——
    防止标注集里出现"看起来合理但库里根本没有"的出处。
 2. 渲染：输出纯文本版（无 Markdown 语法），供人工逐条核对。

用法: python src/render_eval_set.py
"""
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
D = ROOT / "data"
OUT = Path.home() / "Desktop" / "学生手册RAG_评测集_40题_待确认.txt"

TYPE_CN = {"fact": "事实题", "numeric": "数值题", "synthesis": "归纳题",
           "cross": "跨章题", "refuse": "拒答题"}
SEP = "=" * 72
THIN = "-" * 72


def load_clauses():
    cl = json.load((D / "clauses.json").open(encoding="utf-8"))
    return {c["clause_id"]: c for c in cl}


def validate(ev, by):
    errs = []
    for it in ev["items"]:
        for s in it.get("sources", []):
            cid = s["clause_id"]
            c = by.get(cid)
            if c is None:
                errs.append(f"{it['qid']}: 出处 {cid} 在 clauses.json 中不存在")
                continue
            if s.get("article_no") is not None and c.get("article_no") != s["article_no"]:
                errs.append(f"{it['qid']}: {cid} 条号不符 "
                            f"（标注 {s['article_no']} / 实为 {c.get('article_no')}）")
            pg = f"{c['page_start']}-{c['page_end']}" if c["page_start"] != c["page_end"] \
                else str(c["page_start"])
            if s.get("page") != pg:
                errs.append(f"{it['qid']}: {cid} 页码不符（标注 {s['page']} / 实为 {pg}）")
            if s.get("file_title") and s["file_title"] != c["file_title"]:
                errs.append(f"{it['qid']}: {cid} 文件名不符 "
                            f"（标注 {s['file_title']} / 实为 {c['file_title']}）")
    return errs


CN = "零一二三四五六七八九"


def cn_no(n):
    """阿拉伯数字条号 → 中文（1→一，16→十六，57→五十七）。"""
    if n is None:
        return None
    if n < 10:
        return CN[n]
    if n < 20:
        return "十" + (CN[n % 10] if n % 10 else "")
    return CN[n // 10] + "十" + (CN[n % 10] if n % 10 else "")


def fmt_src(s):
    c = BY[s["clause_id"]]
    no = s.get("article_no")
    tag = f"第{cn_no(no)}条" if no else "（无条号）"
    return (f"{s['clause_id']}  《{c['file_title']}》{tag}"
            f"  p{c['page_start']}" + (f"-{c['page_end']}" if c['page_end'] != c['page_start'] else ""))


def render(ev):
    L = []
    A = L.append
    A("深圳职业技术大学学生手册 RAG · 评测集（40 题）")
    A("状态：草稿，待你逐条核对确认")
    A(f"生成时间：{ev['meta']['built_at']}")
    A("")
    A(SEP)
    A("这是什么 / 怎么用")
    A(SEP)
    A("这是给系统出的「带标准答案的考卷」。每题的标准答案都由我从手册原文摘出，")
    A("出处能对回具体条款和页码，不是凭印象写的。你要做的是三件事：")
    A("")
    A("  1. 逐条看「标准答案」对不对 —— 有错就直接在文档里改")
    A("  2. 看「出处」是不是找对了条款 —— 这是最容易出错的地方")
    A("  3. 确认没问题的，在后面标个 OK；有疑问的标个 ？")
    A("")
    A("确认完之后我会：跑一遍系统答这 40 题，机器答案对着这份标准答案逐条比，")
    A("算出正确率。之后做检索优化时，就用同一份题反复测，得出对比表。")
    A("")
    A("题目配比：")
    A(f"  事实题 15   问一个具体规定，答案在单条条款内")
    A(f"  数值题  8   问金额/比例/数量，考数字精度")
    A(f"  归纳题 10   需要把一条或几条里的多个要点完整列全")
    A(f"  跨章题  4   一个答案要跨多份文件、多个条款拼起来")
    A(f"  拒答题  3   手册里根本没有的问题，考系统会不会编")
    A("")
    A("关于最后那一类，说明一下：这三题没有正确答案，考点恰恰是「不回答」。")
    A("手册里查不到的东西，系统应该老实说查不到。如果它编出一个像模像样的时间")
    A("或数量，那就是幻觉，就算听起来再合理也算错。")
    A("")

    for t, cn in (("fact", "事实题"), ("numeric", "数值题"),
                  ("synthesis", "归纳题"), ("cross", "跨章题"), ("refuse", "拒答题")):
        items = [x for x in ev["items"] if x["type"] == t]
        A("")
        A(SEP)
        A(f"{cn}（{len(items)} 题）")
        A(SEP)
        for it in items:
            A("")
            A(f"【{it['qid']}】{it['question']}")
            A(THIN)
            A("标准答案：")
            for line in re.split(r"(?<=。)", it["answer"]):
                line = line.strip()
                if line:
                    A("    " + line)
            if it.get("must_contain"):
                A("")
                A("判定要点（答案里必须出现这些）：")
                A("    " + " / ".join(it["must_contain"]))
            A("")
            if it.get("sources"):
                A("出处：")
                for s in it["sources"]:
                    A("    " + fmt_src(s))
            else:
                A("出处：无（手册中没有对应规定，这正是本题的考点）")
            if it.get("note"):
                A("")
                A("标注注意：" + it["note"])
            A("")
            A("核对结果：[   ]  OK    [   ]  有疑问    [   ]  答案要改")
            A("")

    A("")
    A(SEP)
    A("确认清单（回给我这几条就行）")
    A(SEP)
    A("1. 哪些题的「标准答案」需要改？（说题号 + 改成什么）")
    A("2. 哪些题的「出处」引错了条款？")
    A("3. 拒答题（Q38-Q40）你认可吗？如果你觉得该换成别的问题，告诉我题目。")
    A("4. 整体确认后，我就进入阶段 3，先用这份题跑纯向量基线。")
    A("")
    return "\n".join(L)


if __name__ == "__main__":
    ev = json.load((D / "eval_set.json").open(encoding="utf-8"))
    BY = load_clauses()
    errs = validate(ev, BY)
    n = len(ev["items"])
    print(f"题目数 {n}  配比 "
          + " ".join(f"{TYPE_CN[k]}={sum(1 for x in ev['items'] if x['type'] == k)}"
                     for k in TYPE_CN))
    if errs:
        print(f"\n[校验失败] {len(errs)} 处出处对不上：")
        for e in errs:
            print("   ", e)
        sys.exit(1)
    print("出处校验通过：全部引用均真实存在，条号与页码一致。")
    OUT.write_text(render(ev), encoding="utf-8")
    print(f"已写出 {OUT}")
