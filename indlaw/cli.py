"""Command line: build an index, ask a question, evaluate against a gold set."""

from __future__ import annotations

import argparse
import json

from .answer import answer
from .evaluate import evaluate, load_gold
from .generate import answer_rag, generator_from_env
from .index import Index


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="indlaw")
    sub = p.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build", help="index a folder of .txt judgments")
    b.add_argument("folder")
    b.add_argument("--out", default="index.json")
    b.add_argument("--dense", action="store_true", help="also embed passages locally (needs numpy and fastembed)")
    a = sub.add_parser("ask", help="ask a question")
    a.add_argument("question")
    a.add_argument("--index", default="index.json")
    a.add_argument("--no-llm", action="store_true", help="print extractive quotes instead of a generated answer")
    e = sub.add_parser("eval", help="retrieval metrics against a gold set")
    e.add_argument("gold")
    e.add_argument("--index", default="index.json")
    args = p.parse_args(argv)

    if args.cmd == "build":
        idx = Index.from_folder(args.folder)
        if args.dense:
            idx.add_dense()
        idx.save(args.out)
        print(f"{len(idx.chunks)} chunks from {len({c.doc_id for c in idx.chunks})} documents -> {args.out}")
    elif args.cmd == "ask":
        index = Index.load(args.index)
        generator = None if args.no_llm else generator_from_env()
        if generator is None:
            print("[extractive mode: quotes from the judgments, no language model]")
            print(answer(index, args.question).render())
        else:
            result = answer_rag(index, args.question, generator)
            print(f"[{result.mode}]\n{result.text}")
            for n, chunk in enumerate(result.sources, start=1):
                print(f"  [{n}] {chunk.title}, {chunk.label}")
            if result.uncited_sentences or result.bad_citations:
                print(f"  warning: {len(result.uncited_sentences)} uncited sentence(s), "
                      f"invalid citation numbers {result.bad_citations}")
    else:
        print(json.dumps(evaluate(Index.load(args.index), load_gold(args.gold)), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
