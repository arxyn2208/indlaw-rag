"""Build a corpus index from a folder of .txt judgments, and search it."""

from __future__ import annotations

import json
import os
from dataclasses import asdict

from .bm25 import BM25
from .dense import DenseIndex, rrf
from .text import Chunk, make_chunks, tokens


def read_corpus(folder: str) -> list[tuple[str, str, str]]:
    """``(doc_id, title, raw_text)`` for every .txt file. Title is the first non-empty line."""
    docs = []
    for name in sorted(os.listdir(folder)):
        if not name.lower().endswith(".txt"):
            continue
        with open(os.path.join(folder, name), encoding="utf-8", errors="replace") as handle:
            raw = handle.read()
        first = next((ln.strip() for ln in raw.splitlines() if ln.strip()), name)
        title = first.split(":", 1)[1].strip() if first.upper().startswith("CASE TITLE:") else first
        docs.append((name, title[:160], raw))
    return docs


class Index:
    def __init__(self, chunks: list[Chunk], dense: DenseIndex | None = None):
        self.chunks = chunks
        self.dense = dense
        self.bm25 = BM25([tokens(c.title + " " + c.text) for c in chunks])

    @classmethod
    def from_folder(cls, folder: str, max_words: int = 220) -> Index:
        chunks: list[Chunk] = []
        for doc_id, title, raw in read_corpus(folder):
            chunks.extend(make_chunks(doc_id, title, raw, max_words))
        if not chunks:
            raise ValueError(f"no .txt judgments found in {folder!r}")
        return cls(chunks)

    def add_dense(self) -> None:
        """Embed every chunk locally. Needs numpy and fastembed; no network call after the first model download."""
        self.dense = DenseIndex.build([c.title + ". " + c.text for c in self.chunks])

    def save(self, path: str) -> None:
        with open(path, "w", encoding="utf-8") as handle:
            json.dump([asdict(c) for c in self.chunks], handle)
        if self.dense is not None:
            self.dense.save(path + ".npz")

    @classmethod
    def load(cls, path: str) -> Index:
        with open(path, encoding="utf-8") as handle:
            chunks = [Chunk(**row) for row in json.load(handle)]
        dense = DenseIndex.load(path + ".npz") if os.path.exists(path + ".npz") else None
        return cls(chunks, dense)

    def search(self, query: str, k: int = 5, mode: str = "auto") -> list[tuple[Chunk, float]]:
        """``mode``: "bm25", "hybrid" (BM25 + dense, rank-fused) or "auto" (hybrid when dense vectors exist)."""
        lexical = self.bm25.top(tokens(query), max(k * 4, 20))
        if mode == "bm25" or (mode == "auto" and self.dense is None):
            return [(self.chunks[i], s) for i, s in lexical[:k]]
        if self.dense is None:
            raise ValueError("hybrid search needs dense vectors: run `indlaw build --dense`")
        semantic = self.dense.top(query, max(k * 4, 20))
        fused = rrf([[i for i, _ in lexical], [i for i, _ in semantic]])[:k]
        lex_score = dict(lexical)
        return [(self.chunks[i], lex_score.get(i, 0.0)) for i in fused]

    def search_docs(self, query: str, k: int = 5, mode: str = "auto") -> list[str]:
        """Best chunk per document, in rank order. Document-level ranking for evaluation."""
        seen: list[str] = []
        for chunk, _ in self.search(query, k=max(k * 8, 40), mode=mode):
            if chunk.doc_id not in seen:
                seen.append(chunk.doc_id)
            if len(seen) == k:
                break
        return seen
