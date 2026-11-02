"""Extractive, cited answers with a transparent abstention rule.

No language model is used. The answer is the sentences from the retrieved
paragraphs that best match the question, each tagged with where it came from.
The system refuses when too few of the question's content words appear in the
best retrieved chunk, rather than showing weakly related text as an answer.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from .index import Index
from .text import Chunk, tokens

MIN_COVERAGE = 0.5  # share of question content words the best chunk must contain


@dataclass
class Answer:
    abstained: bool
    reason: str
    sentences: list[tuple[str, Chunk]] = field(default_factory=list)

    def render(self) -> str:
        if self.abstained:
            return f"No answer: {self.reason}"
        return "\n".join(f"- {s} [{c.title}, {c.label}]" for s, c in self.sentences)


def coverage(question_terms: set[str], chunk: Chunk) -> float:
    if not question_terms:
        return 0.0
    return len(question_terms & set(tokens(chunk.text + " " + chunk.title))) / len(question_terms)


def answer(index: Index, question: str, k: int = 4, max_sentences: int = 4,
           min_coverage: float = MIN_COVERAGE) -> Answer:
    terms = set(tokens(question))
    hits = index.search(question, k=k)
    if not hits:
        return Answer(True, "nothing in the corpus matches the question.")
    best = max(coverage(terms, c) for c, _ in hits)
    if best < min_coverage:
        return Answer(True, f"the closest passages cover only {best:.0%} of the question's key terms "
                            f"(threshold {min_coverage:.0%}).")
    scored = []
    for rank, (chunk, _) in enumerate(hits):
        for sentence in re.split(r"(?<=[.;?!])\s+", chunk.text):
            overlap = len(terms & set(tokens(sentence)))
            if overlap and len(sentence.split()) >= 6:
                scored.append((overlap, -rank, sentence.strip(), chunk))
    scored.sort(key=lambda row: (-row[0], -row[1]))
    return Answer(False, "", [(s, c) for _, _, s, c in scored[:max_sentences]])
