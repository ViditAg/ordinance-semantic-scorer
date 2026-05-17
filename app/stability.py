"""
Chunk hyperparameter sweep for semantic-score stability analysis.

Given fixed criteria and embedding model, varies ``chunk_size`` and ``overlap``,
re-chunks the same document text, re-embeds chunks, and scores each setting.
Criterion **probe** embeddings are computed once per sweep (they do not depend on chunking).

Used by ``scripts/calibrate.py`` (local HTML/CSV benchmarking) and importable for
custom notebooks. Helps find settings that balance high scores against local
robustness (avoiding razor-thin peaks).
"""

from __future__ import annotations

import math
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from app.scorer import OrdinanceScorer
from app.utils import chunk_text


def valid_chunk_pairs(
    chunk_sizes: Sequence[int],
    overlaps: Sequence[int],
) -> List[Tuple[int, int]]:
    """
    Cartesian product of sizes and overlaps, dropping invalid ``overlap >= size``.
    """
    pairs: List[Tuple[int, int]] = []
    for cs in chunk_sizes:
        for ov in overlaps:
            if ov < cs:
                pairs.append((cs, ov))
    return pairs


def _grid_indices(
    chunk_size: int,
    overlap: int,
    sizes: Sequence[int],
    overlaps: Sequence[int],
) -> Tuple[int, int]:
    return sizes.index(chunk_size), overlaps.index(overlap)


def annotate_neighbor_stability(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    For each evaluated ``(chunk_size, overlap)``, compute the sample standard
    deviation of ``overall_score`` over existing Moore neighbors on the grid
    implied by unique sizes and overlaps present in ``rows`` (missing parameter
    combinations are skipped). Stored as ``neighbor_score_stdev`` (lower ⇒
    smoother local landscape around that setting).
    """
    sizes = sorted({int(r["chunk_size"]) for r in rows})
    ovs = sorted({int(r["overlap"]) for r in rows})
    score_map: Dict[Tuple[int, int], float] = {
        (int(r["chunk_size"]), int(r["overlap"])): float(r["overall_score"]) for r in rows
    }

    out: List[Dict[str, Any]] = []
    for r in rows:
        cs, ov = int(r["chunk_size"]), int(r["overlap"])
        si, oi = _grid_indices(cs, ov, sizes, ovs)
        neigh_scores: List[float] = []
        for dsi in (-1, 0, 1):
            for doi in (-1, 0, 1):
                nsi, noi = si + dsi, oi + doi
                if 0 <= nsi < len(sizes) and 0 <= noi < len(ovs):
                    key = (sizes[nsi], ovs[noi])
                    if key in score_map:
                        neigh_scores.append(score_map[key])
        if len(neigh_scores) >= 2:
            mean = sum(neigh_scores) / len(neigh_scores)
            var = sum((s - mean) ** 2 for s in neigh_scores) / (len(neigh_scores) - 1)
            stdev = math.sqrt(var)
        else:
            stdev = 0.0
        rr = dict(r)
        rr["neighbor_score_stdev"] = round(stdev, 4)
        out.append(rr)
    return out


def composite_rank_score(
    overall: float,
    neighbor_stdev: float,
    stability_weight: float,
) -> float:
    """
    Higher is better: reward high ``overall``, penalize volatile neighborhoods.

    ``stability_weight`` scales how much local score jitter should count against
    a setting (typical range 0–2).
    """
    return overall - stability_weight * neighbor_stdev


def run_chunk_sweep(
    raw_text: str,
    criteria: List[Dict[str, Any]],
    model_name: str,
    chunk_sizes: Sequence[int],
    overlaps: Sequence[int],
    top_k: int = 1,
    progress: Optional[Callable[[int, int], None]] = None,
) -> List[Dict[str, Any]]:
    """
    Run scoring for every valid ``(chunk_size, overlap)`` pair.

    Criterion embeddings are encoded once; document chunks are re-embedded for
    each pair (the expensive part).

    Args:
        raw_text: Full ordinance text.
        criteria: Rubric list passed to ``OrdinanceScorer``.
        model_name: Sentence-Transformers checkpoint id.
        chunk_sizes: Candidate chunk sizes (characters).
        overlaps: Candidate overlaps (characters); pairs with ``overlap >= size`` skipped.
        top_k: Passed through to ``OrdinanceScorer.score``.
        progress: Optional ``progress(done, total)`` callback for UI progress bars.

    Returns:
        List of dicts with keys including ``chunk_size``, ``overlap``,
        ``num_chunks``, ``overall_score``, ``criteria_score_stdev`` (spread of
        per-criterion scores for that run).
    """
    pairs = valid_chunk_pairs(chunk_sizes, overlaps)
    total = len(pairs)
    scorer = OrdinanceScorer(criteria=criteria, model_name=model_name)
    crit_probe_embeddings = scorer.embed_criteria_probes()

    rows: List[Dict[str, Any]] = []
    for i, (cs, ov) in enumerate(pairs):
        chunks = chunk_text(raw_text, chunk_size=cs, overlap=ov)
        if not chunks:
            rows.append(
                {
                    "chunk_size": cs,
                    "overlap": ov,
                    "num_chunks": 0,
                    "overall_score": 0.0,
                    "criteria_score_stdev": 0.0,
                }
            )
            if progress:
                progress(i + 1, total)
            continue

        doc_embeddings = scorer.embed_texts(chunks)
        results = scorer.score(
            doc_chunks=chunks,
            doc_embeddings=doc_embeddings,
            crit_probe_embeddings=crit_probe_embeddings,
            top_k=top_k,
        )
        per = [float(r["score"]) for r in results["criteria_results"]]
        mean = sum(per) / len(per)
        var = sum((s - mean) ** 2 for s in per) / (len(per) - 1) if len(per) > 1 else 0.0
        crit_stdev = math.sqrt(var)

        rows.append(
            {
                "chunk_size": cs,
                "overlap": ov,
                "num_chunks": len(chunks),
                "overall_score": float(results["overall_score"]),
                "criteria_score_stdev": round(crit_stdev, 4),
            }
        )
        if progress:
            progress(i + 1, total)

    return annotate_neighbor_stability(rows)
