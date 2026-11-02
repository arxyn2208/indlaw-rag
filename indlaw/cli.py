"""Command line: build an index, ask a question, evaluate against a gold set."""

from __future__ import annotations

import argparse
import json
import sys

from .answer import answer
from .evaluate import evaluate, load_gold
from .generate import GenerationError, answer_rag, generator_from_env
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
    m = sub.add_parser("map", help="list the legal provisions each judgment mentions, with the paragraph and sentence")
    m.add_argument("folder")
    m.add_argument("--dot", help="also write Graphviz source to this file")
    args = p.parse_args(argv)

    if args.cmd == "build":
        idx = Index.from_folder(args.folder)
        if args.dense:
            idx.add_dense()
        idx.save(args.out)
        print(f"{len(idx.chunks)} chunks from {len({c.doc_id for c in idx.chunks})} documents -> {args.out}")
    elif args.cmd == "map":
        from .kmap import build_map, to_dot

        edges = build_map(Index.from_folder(args.folder).chunks)
        for e in edges:
            print(f"{e.title} | {e.label} | {e.provision}\n    {e.evidence}")
        print(f"{len(edges)} mentions found (pattern matches, not interpretation)")
        if args.dot:
            with open(args.dot, "w", encoding="utf-8") as f:
                f.write(to_dot(edges))
    elif args.cmd == "ask":
        index = Index.load(args.index)
        generator = None if args.no_llm else generator_from_env()
        if generator is None:
            print("[extractive mode: quotes from the judgments, no language model]")
            print(answer(index, args.question).render())
        else:
            try:
                result = answer_rag(index, args.question, generator)
            except GenerationError as err:
                print(f"error: {err}", file=sys.stderr)
                return 1
            print(f"[{result.mode}]\n{result.text}")
            for n, chunk in enumerate(result.sources, start=1):
                print(f"  [{n}] {chunk.title}, {chunk.label}")
            if result.removed:
                print(f"  removed {len(result.removed)} statement(s) that failed the quotation check")
    else:
        print(json.dumps(evaluate(Index.load(args.index), load_gold(args.gold)), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
