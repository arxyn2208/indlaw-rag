"""Okapi BM25 over pre-tokenised documents."""

from __future__ import annotations

import math
from collections import Counter


class BM25:
    def __init__(self, docs: list[list[str]], k1: float = 1.5, b: float = 0.75):
        self.k1, self.b = k1, b
        self.tf = [Counter(d) for d in docs]
        self.len = [len(d) for d in docs]
        self.avg = (sum(self.len) / len(docs)) if docs else 0.0
        df: Counter = Counter()
        for counts in self.tf:
            df.update(counts.keys())
        n = len(docs)
        self.idf = {t: math.log(1 + (n - f + 0.5) / (f + 0.5)) for t, f in df.items()}

    def score(self, query: list[str]) -> list[float]:
        out = [0.0] * len(self.tf)
        for term in set(query):
            idf = self.idf.get(term)
            if idf is None:
                continue
            for i, counts in enumerate(self.tf):
                f = counts.get(term)
                if f:
                    norm = 1 - self.b + self.b * self.len[i] / (self.avg or 1.0)
                    out[i] += idf * f * (self.k1 + 1) / (f + self.k1 * norm)
        return out

    def top(self, query: list[str], k: int) -> list[tuple[int, float]]:
        scores = self.score(query)
        order = sorted(range(len(scores)), key=lambda i: -scores[i])[:k]
        return [(i, scores[i]) for i in order if scores[i] > 0]
