# -*- coding: utf-8 -*-
"""阶段 4 · 命令行问答入口

    python src/cli/ask.py "考核不及格可以补考吗"          # 正常问答
    python src/cli/ask.py --dry-run "……"                  # 不调 LLM，只看检索到的证据与拼好的 prompt
    python src/cli/ask.py -i                              # 交互模式
    python src/cli/ask.py --show-prompt "……"              # 显示送进模型的完整 prompt

密钥：在项目根建 .env（照 .env.example 填 DEEPSEEK_API_KEY），或设同名环境变量。
"""
import argparse
import json
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))   # src/
import _paths  # noqa: E402,F401

from retrievers import HandbookRetriever          # noqa: E402
from qa_chain import QAPipeline, build_llm, load_env_file   # noqa: E402


def make_pipeline(a):
    r = HandbookRetriever("data", use_rerank=not a.no_rerank, verbose=a.verbose)
    llm = None
    if not a.dry_run:
        load_env_file()
        llm = build_llm(model=a.model)
    return QAPipeline(r, llm=llm, top_k=a.top_k,
                      llm_verify=not a.no_verify, verbose=a.verbose)


def one(a, pipe, q):
    res = pipe.ask(q)
    if a.json:
        print(json.dumps({
            "question": res.question, "answer": res.answer,
            "citations": res.citations, "refused": res.refused,
            "refuse_reason": res.refuse_reason, "insufficient": res.insufficient,
            "latency": round(res.latency, 2),
            "evidences": [{"eid": e.eid, "clause_id": e.clause_id,
                           "cite": e.cite, "page": e.page, "score": e.score}
                          for e in res.evidences],
            "verify": res.verify,
        }, ensure_ascii=False, indent=1))
        return
    if a.show_prompt and res.prompt_preview:
        print("=" * 78)
        print(res.prompt_preview[:4000])
        print("=" * 78)
        print()
    print(res.pretty())
    if a.debug and res.verify:
        print("\n--- 校验明细 ---")
        print(json.dumps(res.verify, ensure_ascii=False, indent=1)[:3000])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("question", nargs="*", help="要问的问题")
    ap.add_argument("-i", "--interactive", action="store_true", help="交互模式")
    ap.add_argument("--dry-run", action="store_true", help="不调 LLM，只走检索与证据构造")
    ap.add_argument("--show-prompt", action="store_true", help="打印送进模型的 prompt")
    ap.add_argument("--json", action="store_true", help="以 JSON 输出结果")
    ap.add_argument("--debug", action="store_true", help="打印校验明细")
    ap.add_argument("--no-rerank", action="store_true", help="关闭 cross-encoder 重排")
    ap.add_argument("--no-verify", action="store_true", help="关闭二级支持性校验")
    ap.add_argument("--top-k", type=int, default=6, help="送给模型的证据条数")
    ap.add_argument("--model", default=None, help="覆盖模型名（默认 deepseek-chat）")
    ap.add_argument("--quiet", dest="verbose", action="store_false", help="少打印")
    a = ap.parse_args()

    pipe = make_pipeline(a)

    if a.interactive:
        print("输入问题进行检索问答（q 退出）")
        while True:
            try:
                q = input("\n问> ").strip()
            except (EOFError, KeyboardInterrupt):
                break
            if q.lower() in ("q", "quit", "exit"):
                break
            if q:
                one(a, pipe, q)
        return

    q = " ".join(a.question).strip()
    if not q:
        ap.error("请给出问题，例如：python src/cli/ask.py \"考核不及格可以补考吗\"")
    one(a, pipe, q)


if __name__ == "__main__":
    main()
