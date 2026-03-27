"""
Semantic scoring for ordinance text against weighted criteria.

This module ties together:

* **Embeddings** — local Sentence-Transformers models (cached per model name).
* **Scoring** — per-criterion max cosine similarity between criterion vectors
  and document-chunk vectors, mapped to 0–100 and combined with normalized
  weights.

Typical usage::

    scorer = OrdinanceScorer(criteria, model_name='all-MiniLM-L6-v2')
    doc_embs = scorer.embed_texts(chunks)
    crit_embs = scorer.embed_texts([c['description'] for c in criteria])
    result = scorer.score(chunks, doc_embs, crit_embs, top_k=2)
"""

# imports
from __future__ import annotations
from functools import lru_cache
from typing import Any, Dict, List
import numpy as np

# function to load and cache a SentenceTransformer model
@lru_cache(maxsize=4)
def _get_sentence_transformer(model_name: str):
    """
    Load (or reuse) a SentenceTransformer by name.

    The global cache keeps at most a few models in memory so switching model
    names in one process does not reload the same weights repeatedly.

    Args:
        model_name: Hugging Face / Sentence-Transformers model id.

    Returns:
        A loaded ``SentenceTransformer`` instance.
    """
    # Import inside the function so importing this module does not require
    # sentence_transformers unless embeddings are actually used.
    from sentence_transformers import SentenceTransformer

    # return the SentenceTransformer instance
    return SentenceTransformer(model_name)


class OrdinanceScorer:
    """
    Embed ordinance and criterion text locally, then score by semantic similarity.

    Each criterion is a ``dict`` with at least ``title`` and ``description``.
    Optional keys: ``weight`` (defaults to ``1.0``), ``short`` (UI summary).

    The overall score is the weighted average of per-criterion scores (0–100),
    where weights are renormalized to sum to 1.

    The underlying model is loaded lazily on first call to :meth:`embed_texts`,
    so constructing this object without embedding does not download or load
    Sentence-Transformers weights (useful for tests that only call :meth:`score`
    with precomputed vectors).
    """

    def __init__(
        self,
        criteria: List[Dict[str, Any]],
        model_name: str = "all-MiniLM-L6-v2",
    ) -> None:
        """
        Initialize the OrdinanceScorer.
        Args:
            criteria: Evaluation rubric; one entry per scored criterion.
            model_name: Sentence-Transformers model used by :meth:`embed_texts`.
        """
        # set the criteria
        self.criteria = criteria
        # set the model name
        self.model_name = model_name
        # set the model to None
        self._model = None  # filled on first access via :attr:`model`

        # get the weights
        ## normalize weights so the overall score is a convex combination
        weights = [c.get("weight", 1.0) for c in criteria]
        ## sum the weights and divide by the total to get the weights
        total = sum(weights) or 1.0
        ## divide the weights by the total to get the weights
        self.weights = [w / total for w in weights]

    
    @property
    def model(self):
        """
        Lazily instantiated SentenceTransformer for :meth:`embed_texts`.

        Access looks like a simple attribute (``self.model``) but runs the
        loader only once per instance thanks to ``self._model`` caching.
        Returns:
            The SentenceTransformer instance.
        """
        # if the model is not loaded, load it
        if self._model is None:
            # load the model
            self._model = _get_sentence_transformer(self.model_name)
        # return the model
        return self._model

    
    def embed_texts(
        self,
        texts: List[str],
    ) -> List[List[float]]:
        """
        Encode strings into L2-normalized embedding vectors.

        Args:
            texts: Batch of strings to embed. Empty list returns ``[]``.

        Returns:
            List of embeddings (each embedding is a list of floats).
        """
        # if the texts are empty, return an empty list
        if not texts:
            return []
        
        # encode the texts
        embs = self.model.encode(
            texts,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
        # return the embeddings
        return [e.tolist() for e in embs]

    @staticmethod
    def _cosine_similarities_array(
        vec: List[float],
        matrix: List[List[float]],
    ) -> List[float]:
        """
        Cosine similarity between one vector and each row of a matrix.

        Args:
            vec: Single embedding (1D sequence of floats).
            matrix: 2D array whose rows are chunk embeddings.

        Returns:
            1D NumPy array of similarities, one per row of *matrix*.
        """
        # convert the vector and matrix to numpy arrays
        v = np.array(vec)
        # convert the matrix to a numpy array
        M = np.array(matrix)
        # get the norm of the vector
        v_norm = np.linalg.norm(v)
        M_norm = np.linalg.norm(M, axis=1)
        denom = v_norm * M_norm
        # Avoid NaNs when a row or vec is zero-length in L2 norm.
        denom[denom == 0] = 1e-10
        # get the similarities
        sims = (M @ v) / denom
        # return the similarities
        return sims

    @staticmethod
    def _sim_to_score(
        sim: float,
    ) -> float:
        """
        Map cosine similarity to a 0–100 score for one criterion.

        Negative cosine is clipped to 0 before scaling so the score stays
        in a simple interpretable range for stakeholders.

        Args:
            sim: Cosine similarity in ``[-1, 1]``.

        Returns:
            Score rounded to two decimal places.
        """
        # scale the similarity
        scaled = max(sim, 0.0) * 100.0
        # return the scaled similarity rounded to two decimal places
        return float(round(scaled, 2))
    
    
    def score(
        self,
        doc_chunks: List[str],
        doc_embeddings: List[List[float]],
        crit_embeddings: List[List[float]],
        top_k: int = 1,
    ) -> Dict[str, Any]:
        """
        Score pre-embedded document chunks against pre-embedded criteria.

        For each criterion, finds the maximum cosine similarity over all
        chunks, maps that value from ``[-1, 1]`` to ``[0, 100]`` (negative
        similarities become 0 before scaling), and records the top-*k* chunks.

        Args:
            doc_chunks: Text segments parallel to *doc_embeddings*.
            doc_embeddings: One vector per chunk (same order as *doc_chunks*).
            crit_embeddings: One vector per criterion (same order as
                ``self.criteria``).
            top_k: How many highest-similarity excerpts to keep per criterion.

        Returns:
            Dict with:

            * ``overall_score`` — weighted mean of per-criterion 0–100 scores.
            * ``criteria_results`` — list of dicts with title, short, score,
              raw_similarity, excerpts, etc.
        """
        # Stack rows for vectorized dot products against all chunks at once.
        D = np.array(doc_embeddings)
        C = np.array(crit_embeddings)

        # initialize the criteria results
        criteria_results = []
        # initialize the per scores
        per_scores = []
        # iterate through the criteria
        for idx, crit in enumerate(self.criteria):
            # get the criterion embedding
            c_emb = C[idx]
            sims = self._cosine_similarities_array(c_emb, D)
            # Highest similarity first: argsort ascending then reverse slice.
            top_idx = sims.argsort()[::-1][:top_k]
            # get the top excerpts
            top_excerpts = [doc_chunks[i] for i in top_idx]
            # get the top scores
            top_scores = sims[top_idx].tolist()
            # get the best similarity
            best_sim = float(np.max(sims)) if len(sims) > 0 else 0.0
            # get the score
            score = self._sim_to_score(best_sim)
            # add the score to the per scores
            per_scores.append(score)
            
            # add the criteria result to the criteria results
            criteria_results.append(
                {
                    "title": crit.get("title", f"Criterion {idx+1}"),
                    "short": crit.get("short", crit.get("description", "")[:200]),
                    "score": score,
                    "raw_similarity": best_sim,
                    "top_excerpts": top_excerpts,
                    "top_scores": top_scores,
                    "weight": crit.get("weight", 1.0),
                }
            )
        # initialize the overall score
        overall = 0.0
        # iterate through the weights and per scores
        for w, s in zip(self.weights, per_scores):
            # add the weighted score to the overall score
            overall += w * s

        # return the overall score and criteria results
        return {
            "overall_score": overall,
            "criteria_results": criteria_results,
        }

    
