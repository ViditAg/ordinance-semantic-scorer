from typing import List, Dict, Any
import numpy as np
from .embeddings import cosine_similarity

class OrdinanceScorer:
    """
    Score an ordinance document using semantic similarities.

    criteria: list of dicts, each with keys:
      - title
      - description
      - weight (optional, default 1.0)
      - short (optional short summary)
    """
    def __init__(self, criteria: List[Dict[str, Any]]):
        self.criteria = criteria
        # normalize weights
        weights = [c.get("weight", 1.0) for c in criteria]
        total = sum(weights) or 1.0
        self.weights = [w / total for w in weights]

    def score(self, doc_chunks: List[str], doc_embeddings: List[List[float]],
              crit_embeddings: List[List[float]], top_k: int = 1) -> Dict[str, Any]:
        """
        For each criterion, compute the max cosine similarity across document chunks,
        scale to 0-100, and collect top matching excerpts.
        """
        doc_embs = doc_embeddings
        crit_embs = crit_embeddings

        # Precompute numpy array for speed
        import numpy as np
        D = np.array(doc_embs)
        C = np.array(crit_embs)

        criteria_results = []
        per_scores = []
        for idx, crit in enumerate(self.criteria):
            c_emb = C[idx]
            # compute cosine similarities to all chunks
            sims = self._cosine_similarities_array(c_emb, D)
            # get top_k indices
            top_idx = sims.argsort()[::-1][:top_k]
            top_excerpts = [doc_chunks[i] for i in top_idx]
            top_scores = sims[top_idx].tolist()
            # convert best similarity to 0-100 (assumes similarity in [-1,1])
            best_sim = float(np.max(sims)) if len(sims) > 0 else 0.0
            score = self._sim_to_score(best_sim)
            per_scores.append(score)
            criteria_results.append({
                "title": crit.get("title", f"Criterion {idx+1}"),
                "short": crit.get("short", crit.get("description", "")[:200]),
                "score": score,
                "raw_similarity": best_sim,
                "top_excerpts": top_excerpts,
                "top_scores": top_scores,
                "weight": crit.get("weight", 1.0)
            })

        # Weighted overall score using original weights (normalized)
        overall = 0.0
        for w, s in zip(self.weights, per_scores):
            overall += w * s
        return {
            "overall_score": overall,
            "criteria_results": criteria_results
        }

    @staticmethod
    def _cosine_similarities_array(vec, matrix):
        import numpy as np
        v = np.array(vec)
        M = np.array(matrix)
        # handle zero vectors
        v_norm = np.linalg.norm(v)
        M_norm = np.linalg.norm(M, axis=1)
        denom = v_norm * M_norm
        # avoid division by zero
        denom[denom == 0] = 1e-10
        sims = (M @ v) / denom
        return sims

    @staticmethod
    def _sim_to_score(sim: float) -> float:
        # map cosine similarity [-1,1] to [0,100], but cap negative to 0
        scaled = max(sim, 0.0) * 100.0
        return float(round(scaled, 2))