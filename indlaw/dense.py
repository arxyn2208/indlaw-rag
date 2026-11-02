"""Optional semantic retrieval with a local ONNX embedder (fastembed). Needs numpy and fastembed."""

from __future__ import annotations

MODEL = "BAAI/bge-small-en-v1.5"


class DenseIndex:
    def __init__(self, vectors, model: str = MODEL):
        self.vectors = vectors  # (n, d) float32, L2-normalised
        self.model = model
        self._embedder = None

    @staticmethod
    def _load_embedder(model: str):
        from fastembed import TextEmbedding  # imported late: optional dependency

        return TextEmbedding(model_name=model)

    @classmethod
    def build(cls, texts: list[str], model: str = MODEL, batch_size: int = 16) -> DenseIndex:
        import numpy as np

        embedder = cls._load_embedder(model)
        rows = []
        for start in range(0, len(texts), batch_size):
            rows.extend(embedder.embed(texts[start:start + batch_size]))
        matrix = np.asarray(rows, dtype=np.float32)
        matrix /= np.linalg.norm(matrix, axis=1, keepdims=True) + 1e-12
        out = cls(matrix, model)
        out._embedder = embedder
        return out

    def save(self, path: str) -> None:
        import numpy as np

        np.savez_compressed(path, vectors=self.vectors.astype("float16"), model=self.model)

    @classmethod
    def load(cls, path: str) -> DenseIndex:
        import numpy as np

        data = np.load(path)
        return cls(data["vectors"].astype("float32"), str(data["model"]))

    def top(self, query: str, k: int) -> list[tuple[int, float]]:
        import numpy as np

        if self._embedder is None:
            self._embedder = self._load_embedder(self.model)
        q = np.asarray(next(iter(self._embedder.embed([query]))), dtype=np.float32)
        q /= np.linalg.norm(q) + 1e-12
        scores = self.vectors @ q
        order = np.argsort(-scores)[:k]
        return [(int(i), float(scores[i])) for i in order]


def rrf(rankings: list[list[int]], k: int = 60) -> list[int]:
    """Reciprocal rank fusion of several ranked lists of ids."""
    score: dict[int, float] = {}
    for ranking in rankings:
        for rank, item in enumerate(ranking, start=1):
            score[item] = score.get(item, 0.0) + 1.0 / (k + rank)
    return sorted(score, key=lambda i: -score[i])
