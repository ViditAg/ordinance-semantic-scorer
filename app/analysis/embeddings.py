from typing import List, Optional
import numpy as np

class EmbeddingProvider:
    """
    Provides embeddings using either a local Sentence-Transformers model
    or the OpenAI embeddings API.

    Usage:
        provider = EmbeddingProvider(backend="local", model_name="all-MiniLM-L6-v2")
        embs = provider.embed_texts(["hello", "world"])
    """
    def __init__(self, backend: str = "local", model_name: str = "all-MiniLM-L6-v2", openai_api_key: Optional[str] = None):
        self.backend = backend
        if backend == "local":
            from sentence_transformers import SentenceTransformer
            self.model = SentenceTransformer(model_name)
        elif backend == "openai":
            import openai
            if openai_api_key:
                openai.api_key = openai_api_key
            self.openai = openai
            # choose a reasonable embedding model available at time of writing
            self.openai_model = "text-embedding-3-small"
        else:
            raise ValueError("backend must be 'local' or 'openai'")

    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        if self.backend == "local":
            embs = self.model.encode(texts, show_progress_bar=False, convert_to_numpy=True)
            return [e.tolist() for e in embs]
        else:
            # OpenAI batching (simple)
            results = []
            BATCH = 16
            for i in range(0, len(texts), BATCH):
                batch = texts[i:i+BATCH]
                resp = self.openai.Embedding.create(model=self.openai_model, input=batch)
                for r in resp["data"]:
                    results.append(r["embedding"])
            return results

def cosine_similarity(a: List[float], b: List[float]) -> float:
    a = np.array(a, dtype=float)
    b = np.array(b, dtype=float)
    if np.all(a == 0) or np.all(b == 0):
        return 0.0
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))