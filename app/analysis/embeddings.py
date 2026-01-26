from __future__ import annotations

from functools import lru_cache
from typing import List

import numpy as np

@lru_cache(maxsize=4)
def _get_sentence_transformer(model_name: str):
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(model_name)


class EmbeddingProvider:
    """Local-only embeddings using Sentence-Transformers.

    Usage:
        provider = EmbeddingProvider(model_name="all-MiniLM-L6-v2")
        embs = provider.embed_texts(["hello", "world"])
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model_name = model_name
        self.model = _get_sentence_transformer(model_name)

    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        if not texts:
            return []
        embs = self.model.encode(
            texts,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
        return [e.tolist() for e in embs]

def cosine_similarity(a: List[float], b: List[float]) -> float:
    a = np.array(a, dtype=float)
    b = np.array(b, dtype=float)
    if np.all(a == 0) or np.all(b == 0):
        return 0.0
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))