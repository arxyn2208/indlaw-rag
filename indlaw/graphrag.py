"""Graph-assisted retrieval (experimental) and an unsourced baseline, for a same-question comparison.

GraphRAG here means one thing: after the usual search, the provisions named in the top passages are looked
up in the provision map, and a few paragraphs from OTHER cases that name the same provision are added to the
context. Each added passage is tagged with the provision that linked it. Links are literal mentions found by
pattern, so a link says two paragraphs name the same provision, not that they agree. This is not shown to be
better than plain RAG.
"""

from __future__ import annotations

from dataclasses import replace

from .answer import coverage
from .index import Index
from .kmap import find_provisions
from .text import Chunk, tokens

MAX_ADDED = 4
PER_PROVISION = 2

_CACHE: dict[int, dict[str, list[Chunk]]] = {}


def provision_index(index: Index) -> dict[str, list[Chunk]]:
    """provision -> paragraphs that name it, over the whole corpus (not the filtered map view)."""
    key = id(index)
    if key not in _CACHE:
        table: dict[str, list[Chunk]] = {}
        for c in index.chunks:
            for prov in {p for p, _ in find_provisions(c.text) if "not named" not in p}:
                table.setdefault(prov, []).append(c)
        _CACHE[key] = table
    return _CACHE[key]


def graph_expand(index: Index, hits: list[Chunk], question: str, max_added: int = MAX_ADDED,
                 per_provision: int = PER_PROVISION) -> list[Chunk]:
    """Paragraphs from other cases that name a provision named in the hits, tagged with that provision."""
    table = provision_index(index)
    hit_ids = {id(c) for c in hits}
    hit_docs = {c.doc_id for c in hits}
    terms = set(tokens(question))
    added: list[Chunk] = []
    seen = set(hit_ids)
    provisions: list[str] = []
    for c in hits:
        for prov, _ in find_provisions(c.text):
            if "not named" not in prov and prov not in provisions:
                provisions.append(prov)
    for prov in provisions:
        pool = [c for c in table.get(prov, []) if id(c) not in seen and c.doc_id not in hit_docs]
        pool.sort(key=lambda c: coverage(terms, c), reverse=True)
        for c in pool[:per_provision]:
            if len(added) >= max_added:
                break
            seen.add(id(c))
            added.append(replace(c, label=f"{c.label}, added through {prov}"))
    return added


LLM_ONLY_PROMPT = ("Answer the question about Indian law in at most 3 short sentences. You are given no source "
                   "documents. If you are not sure, say so.\nQuestion: {question}")


def answer_llm_only(question: str, generator) -> str:
    """The same model with no passages. Nothing in this answer can be checked against a judgment."""
    return generator.complete(LLM_ONLY_PROMPT.format(question=question)).strip()
