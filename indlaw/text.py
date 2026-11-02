"""Tokenising and splitting judgment text."""

from __future__ import annotations

import re
from dataclasses import dataclass

STOPWORDS = frozenset(
    ["a", "an", "the", "of", "to", "in", "on", "at", "by", "for", "from", "with", "and", "or", "nor", "but", "if", "then", "than", "that", "this", "these", "those", "is", "are", "was", "were", "be", "been", "being", "it", "its", "as", "not", "no", "do", "does", "did", "has", "have", "had", "which", "who", "whom", "whose", "what", "when", "where", "why", "how", "can", "could", "may", "might", "shall", "should", "will", "would", "such", "any", "all", "each", "other", "also", "into", "over", "under", "upon", "against", "between", "about", "their", "there", "they", "he", "she", "his", "her", "him", "them"]
)

_WORD = re.compile(r"[a-z0-9]+(?:'[a-z]+)?")
_PARA = re.compile(r"^\s*(\d{1,3})\s*[.)]\s+(?=\S)", re.MULTILINE)


def tokens(text: str, keep_stopwords: bool = False) -> list[str]:
    found = _WORD.findall(text.lower())
    return found if keep_stopwords else [w for w in found if w not in STOPWORDS]


@dataclass(frozen=True)
class Chunk:
    doc_id: str
    title: str
    label: str  # paragraph label such as "para 12" or "part 3"
    text: str


def split_paragraphs(raw: str) -> list[tuple[str, str]]:
    """Return ``(label, text)`` pairs.

    Judgments number their paragraphs. When the text has at least three numbered
    paragraphs, split on those numbers so a citation can say "para 12". Otherwise
    fall back to blank-line blocks, labelled "part N".
    """
    marks = list(_PARA.finditer(raw))
    if len(marks) >= 3:
        pieces = []
        for i, mark in enumerate(marks):
            end = marks[i + 1].start() if i + 1 < len(marks) else len(raw)
            body = " ".join(raw[mark.end():end].split())
            if body:
                pieces.append((f"para {mark.group(1)}", body))
        return pieces
    blocks = [" ".join(b.split()) for b in re.split(r"\n\s*\n", raw)]
    return [(f"part {i}", b) for i, b in enumerate((b for b in blocks if b), start=1)]


def make_chunks(doc_id: str, title: str, raw: str, max_words: int = 220) -> list[Chunk]:
    """Chunks that never cross a paragraph boundary. Long paragraphs are cut by sentence."""
    chunks: list[Chunk] = []
    for label, body in split_paragraphs(raw):
        words = body.split()
        if len(words) <= max_words:
            chunks.append(Chunk(doc_id, title, label, body))
            continue
        sentences = re.split(r"(?<=[.;?!])\s+", body)
        current: list[str] = []
        count = 0
        part = 1
        for sentence in sentences:
            n = len(sentence.split())
            if current and count + n > max_words:
                chunks.append(Chunk(doc_id, title, f"{label}({part})", " ".join(current)))
                current, count, part = [], 0, part + 1
            current.append(sentence)
            count += n
        if current:
            suffix = f"({part})" if part > 1 else ""
            chunks.append(Chunk(doc_id, title, f"{label}{suffix}", " ".join(current)))
    return chunks
