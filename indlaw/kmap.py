"""A provenance knowledge map: cases, the legal provisions they mention, and the paragraph of each mention.

Every edge is a literal mention found by pattern in one paragraph, and carries that sentence as evidence.
It records that a paragraph mentions a provision. It does not say the case interprets, applies or follows it.
Patterns are simple, so mentions can be missed or mis-split; nothing here is a measured accuracy.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from .text import Chunk

ARTICLE = re.compile(r"\bArticles?\s+(\d{1,3}[A-Z]?(?:\s*(?:,|and|&)\s*\d{1,3}[A-Z]?)*)")
NUMBER = re.compile(r"\d{1,3}[A-Z]?")
SECTION = re.compile(
    r"\bSections?\s+(\d{1,4}[A-Z]?)(?:\s*\(\s*\d+\s*\))?"
    r"(?:\s+(?:of|under)\s+the\s+((?:[A-Z][A-Za-z]*\s+){0,6}(?:Code\s+of\s+(?:Criminal|Civil)\s+Procedure|Act|Code)(?:,?\s+(?:1[89]|20)\d\d)?)|\s+(IPC|CrPC|CPC|NDPS Act))"
)
SENTENCE = re.compile(r"(?<=[.;!?])\s+")


@dataclass(frozen=True)
class Edge:
    doc_id: str
    title: str
    label: str  # paragraph label, e.g. "para 3"
    provision: str
    evidence: str  # the sentence that contains the mention


def find_provisions(text: str) -> list[tuple[str, str]]:
    """``(provision, sentence)`` for each literal mention in the text."""
    found: list[tuple[str, str]] = []
    for sent in SENTENCE.split(text):
        for m in ARTICLE.finditer(sent):
            for num in NUMBER.findall(m.group(1)):
                found.append((f"Article {num}", sent.strip()))
        for m in SECTION.finditer(sent):
            act = (m.group(2) or m.group(3) or "").strip().rstrip(",")
            found.append((f"Section {m.group(1)}, {act}", sent.strip()))
    return found


def build_map(chunks: list[Chunk]) -> list[Edge]:
    seen: set[tuple[str, str, str]] = set()
    edges: list[Edge] = []
    for c in chunks:
        for prov, sent in find_provisions(c.text):
            key = (c.doc_id, c.label, prov)
            if key not in seen:
                seen.add(key)
                edges.append(Edge(c.doc_id, c.title, c.label, prov, sent))
    return edges


def to_dot(edges: list[Edge], top: int | None = None) -> str:
    """Graphviz source: cases left, provisions right, one edge per case and provision, labelled with its paragraph (or the paragraph count). ``top`` keeps the N most mentioned provisions."""
    if top:
        from collections import Counter

        keep = {p for p, _ in Counter(e.provision for e in edges).most_common(top)}
        edges = [e for e in edges if e.provision in keep]

    def q(s: str) -> str:
        return '"' + s.replace("\\", "\\\\").replace('"', '\\"') + '"'

    lines = ["digraph kmap {", "  rankdir=LR;", "  node [fontname=Helvetica];"]
    cases = sorted({(e.doc_id, e.title) for e in edges})
    provs = sorted({e.provision for e in edges})
    for doc, title in cases:
        lines.append(f"  {q('c:' + doc)} [label={q(title)}, shape=box, style=filled, fillcolor=\"#e8eef7\"];")
    for p in provs:
        lines.append(f"  {q('p:' + p)} [label={q(p)}, shape=ellipse, style=filled, fillcolor=\"#f7efe0\"];")
    grouped: dict[tuple[str, str], list[str]] = {}
    for e in edges:
        grouped.setdefault((e.doc_id, e.provision), []).append(e.label)
    for (doc, prov), labels in grouped.items():
        text = labels[0] if len(labels) == 1 else f"{len(labels)} paragraphs"
        lines.append(f"  {q('c:' + doc)} -> {q('p:' + prov)} [label={q(text)}, fontsize=10];")
    lines.append("}")
    return "\n".join(lines)
