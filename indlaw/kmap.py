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
            act = re.sub(r"\s+", " ", (m.group(2) or m.group(3) or "")).strip().rstrip(",")
            if act in ("Act", "Code"):
                # "the Act" alone does not say which Act: keep it separate from named Acts
                found.append((f"Section {m.group(1)} (Act not named in the sentence)", sent.strip()))
            else:
                found.append((f"Section {m.group(1)}, {act}", sent.strip()))
    return found


def build_map(chunks: list[Chunk]) -> list[Edge]:
    seen: set[tuple[str, str, str]] = set()
    edges: list[Edge] = []
    for c in chunks:
        for prov, sent in find_provisions(c.text):
            if prov.endswith("(Act not named in the sentence)"):
                # "the Act" differs from case to case, so it must not link two cases
                prov = f"{prov} - {c.doc_id.removesuffix('.txt')}"
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


VIS_URL = "https://unpkg.com/vis-network@9.1.9/standalone/umd/vis-network.min.js"


def to_vis_html(edges: list[Edge], height: int = 620) -> str:
    """A self-contained interactive page (vis-network from a CDN): drag, zoom and pan.

    vis-network 9.1.9 is pinned (Apache-2.0 OR MIT). One node per case and per provision, one edge per case-provision pair. Hovering an edge shows how many
    paragraphs mention it. Nodes and edges come only from the literal mentions in ``edges``.
    """
    import json

    cases: dict[str, str] = {}
    pairs: dict[tuple[str, str], list[str]] = {}
    for e in edges:
        cases[e.doc_id] = e.title
        pairs.setdefault((e.doc_id, e.provision), []).append(e.label)
    provs = sorted({p for _, p in pairs})
    nodes = [{"id": "c:" + d, "label": (t[:38] + "...") if len(t) > 40 else t, "title": t, "shape": "box",
              "color": "#e8eef8", "font": {"size": 12}} for d, t in cases.items()]
    nodes += [{"id": "p:" + p, "label": p, "title": p, "shape": "ellipse", "color": "#f7efe0",
               "value": sum(len(v) for (_, pp), v in pairs.items() if pp == p)} for p in provs]
    links = [{"from": "c:" + d, "to": "p:" + p, "title": f"{len(v)} paragraph(s): " + ", ".join(v[:6]) + (" ..." if len(v) > 6 else ""),
              "value": len(v)} for (d, p), v in pairs.items()]
    data = json.dumps({"nodes": nodes, "edges": links}).replace("<", "\\u003c").replace(">", "\\u003e").replace("&", "\\u0026").replace("\u2028", "\\u2028").replace("\u2029", "\\u2029")
    return (f'<div id="g" style="height:{height}px;border:1px solid #ddd"></div>'
            f'<script src="{VIS_URL}"></script><script>const d={data};'
            'new vis.Network(document.getElementById("g"),{nodes:new vis.DataSet(d.nodes),edges:new vis.DataSet(d.edges)},'
            '{physics:{stabilization:{iterations:200}},interaction:{hover:true,zoomView:true,dragView:true,navigationButtons:true},'
            'edges:{color:{color:"#9aa5b5"},smooth:false},nodes:{scaling:{min:12,max:40}}});</script>')
