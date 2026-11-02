"""Document-level retrieval metrics against a gold set you provide.

Gold file: JSON list of {"question": str, "relevant": [doc_id, ...]} (answerable)
and {"question": str, "relevant": []} (out of corpus, used for abstention checks).
"""

from __future__ import annotations

import json
import math
import random

from .answer import answer
from .index import Index


def recall_at(ranked: list[str], relevant: set[str], k: int) -> float:
    return len(set(ranked[:k]) & relevant) / len(relevant)


def reciprocal_rank(ranked: list[str], relevant: set[str]) -> float:
    for i, doc in enumerate(ranked, start=1):
        if doc in relevant:
            return 1 / i
    return 0.0


def ndcg_at(ranked: list[str], relevant: set[str], k: int) -> float:
    dcg = sum(1 / math.log2(i + 1) for i, d in enumerate(ranked[:k], start=1) if d in relevant)
    ideal = sum(1 / math.log2(i + 1) for i in range(1, min(len(relevant), k) + 1))
    return dcg / ideal if ideal else 0.0


def bootstrap_ci(values: list[float], n: int = 2000, seed: int = 0) -> tuple[float, float]:
    rng = random.Random(seed)
    means = sorted(sum(rng.choices(values, k=len(values))) / len(values) for _ in range(n))
    return means[int(0.025 * n)], means[int(0.975 * n) - 1]


def evaluate(index: Index, gold: list[dict], k: int = 5) -> dict:
    scored = [g for g in gold if g["relevant"]]
    unanswerable = [g for g in gold if not g["relevant"]]
    r, mrr, ndcg = [], [], []
    for g in scored:
        ranked = index.search_docs(g["question"], k=10)
        rel = set(g["relevant"])
        r.append(recall_at(ranked, rel, k))
        mrr.append(reciprocal_rank(ranked, rel))
        ndcg.append(ndcg_at(ranked, rel, 10))
    out: dict = {"answerable": len(scored), "out_of_corpus": len(unanswerable)}
    if scored:
        out[f"recall@{k}"] = sum(r) / len(r)
        out[f"recall@{k}_ci95"] = bootstrap_ci(r)
        out["mrr"] = sum(mrr) / len(mrr)
        out["ndcg@10"] = sum(ndcg) / len(ndcg)
        out["wrongly_refused"] = sum(answer(index, g["question"]).abstained for g in scored)
    if unanswerable:
        out["correctly_refused"] = sum(answer(index, g["question"]).abstained for g in unanswerable)
    return out


def load_gold(path: str) -> list[dict]:
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)
