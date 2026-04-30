#!/usr/bin/env python3
"""
Local chunk-hyperparameter calibration / benchmarking (not Streamlit).

Runs ``app.stability.run_chunk_sweep`` on one PDF or plain-text file, then writes:

* ``summary.csv`` — full grid with composite score
* ``meta.json`` — run configuration for audit
* ``figures/*.png`` — heatmap, histogram, scatter
* ``report.html`` — self-contained narrative + embedded plots (base64)

Usage (from repository root, with optional calibration extras installed)::

    pip install -r requirements.txt -r requirements-calibration.txt
    python scripts/calibrate.py --pdf sample/your.pdf --out reports/run1

Defaults for the sweep grid match ``app.chunking_presets`` sweep seeds unless
you pass ``--chunk-sizes`` / ``--overlaps`` as comma-separated lists.

**Multiple embedding models** — pass ``--models id1,id2`` (comma-separated
Sentence-Transformers checkpoint ids). Each model gets its own subdirectory
under ``--out`` (named from the id, with ``/`` replaced). A single model still
writes files directly into ``--out`` (backwards compatible).
"""

from __future__ import annotations

import argparse
import base64
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

# Repo root on sys.path when executed as ``python scripts/calibrate.py``
_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from app.chunking_presets import (  # noqa: E402
    SWEEP_CHUNK_SIZE_MAX_DEFAULT,
    SWEEP_CHUNK_SIZE_MIN_DEFAULT,
    SWEEP_CHUNK_SIZE_STEP_DEFAULT,
    SWEEP_OVERLAP_MAX_DEFAULT,
    SWEEP_OVERLAP_MIN_DEFAULT,
    SWEEP_OVERLAP_STEP_DEFAULT,
)
from app.defaults import DEFAULT_SENTENCE_TRANSFORMER_MODEL  # noqa: E402
from app.stability import (  # noqa: E402
    composite_rank_score,
    run_chunk_sweep,
    valid_chunk_pairs,
)
from app.utils import extract_text_from_pdf  # noqa: E402


def _parse_int_list(s: str) -> List[int]:
    return [int(x.strip()) for x in s.split(",") if x.strip()]


def _default_chunk_sizes() -> List[int]:
    return list(
        range(
            SWEEP_CHUNK_SIZE_MIN_DEFAULT,
            SWEEP_CHUNK_SIZE_MAX_DEFAULT + 1,
            SWEEP_CHUNK_SIZE_STEP_DEFAULT,
        )
    )


def _default_overlaps() -> List[int]:
    return list(
        range(
            SWEEP_OVERLAP_MIN_DEFAULT,
            SWEEP_OVERLAP_MAX_DEFAULT + 1,
            SWEEP_OVERLAP_STEP_DEFAULT,
        )
    )


def _load_text(*, pdf: Path | None, text_file: Path | None) -> str:
    if pdf and text_file:
        raise SystemExit("Pass only one of --pdf or --text.")
    if pdf:
        with pdf.open("rb") as fh:
            return extract_text_from_pdf(fh)
    if text_file:
        return text_file.read_text(encoding="utf-8")
    raise SystemExit("Provide --pdf or --text.")


def _save_plot_b64(fig: plt.Figure, path: Path) -> str:
    fig.savefig(path, format="png", dpi=120, bbox_inches="tight")
    plt.close(fig)
    return base64.b64encode(path.read_bytes()).decode("ascii")


def _plot_heatmap(df: pd.DataFrame) -> plt.Figure:
    pivot = df.pivot_table(
        index="overlap",
        columns="chunk_size",
        values="overall_score",
        aggfunc="mean",
    )
    fig, ax = plt.subplots(figsize=(8, 5))
    arr = pivot.to_numpy(dtype=float)
    im = ax.imshow(
        arr,
        aspect="auto",
        origin="lower",
        cmap="viridis",
    )
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns.astype(int), rotation=45, ha="right")
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index.astype(int))
    ax.set_xlabel("Chunk size (chars)")
    ax.set_ylabel("Overlap (chars)")
    ax.set_title("Overall semantic score (higher = brighter)")
    fig.colorbar(im, ax=ax, label="Overall score")
    fig.tight_layout()
    return fig


def _plot_hist(df: pd.DataFrame) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(df["overall_score"], bins=min(20, max(5, len(df) // 2)), color="steelblue", edgecolor="white")
    ax.set_xlabel("Overall score")
    ax.set_ylabel("Count (grid cells)")
    ax.set_title("Distribution of overall scores across the grid")
    ax.axvline(df["overall_score"].mean(), color="orange", linestyle="--", label="Mean")
    ax.legend()
    fig.tight_layout()
    return fig


def _model_slug(model_id: str) -> str:
    safe = model_id.replace("/", "_").replace(":", "_").replace(" ", "_")
    return safe[:160] if len(safe) > 160 else safe


def _resolve_model_names(*, model: Optional[str], models: Optional[str]) -> List[str]:
    if models and models.strip():
        return [m.strip() for m in models.split(",") if m.strip()]
    if model and model.strip():
        return [model.strip()]
    return [DEFAULT_SENTENCE_TRANSFORMER_MODEL]


def _run_one_model_calibration(
    *,
    out: Path,
    model_name: str,
    raw: str,
    criteria: List[Dict[str, Any]],
    crit_path: Path,
    chunk_sizes: List[int],
    overlaps: List[int],
    pairs_count: int,
    pdf: Path | None,
    text_file: Path | None,
    stability_weight: float,
    top_k: int,
) -> None:
    out.mkdir(parents=True, exist_ok=True)
    fig_dir = out / "figures"
    fig_dir.mkdir(exist_ok=True)

    print(f"Model: {model_name}")
    print(f"Grid: {pairs_count} pairs · chars in doc: {len(raw)}")

    def prog(i: int, tot: int) -> None:
        pct = 100 * i / tot if tot else 0
        print(f"\r  [{_model_slug(model_name)}] {i}/{tot} ({pct:.0f}%)", end="", flush=True)

    rows = run_chunk_sweep(
        raw_text=raw,
        criteria=criteria,
        model_name=model_name,
        chunk_sizes=chunk_sizes,
        overlaps=overlaps,
        top_k=top_k,
        progress=prog,
    )
    print()

    for r in rows:
        r["composite"] = round(
            composite_rank_score(
                float(r["overall_score"]),
                float(r["neighbor_score_stdev"]),
                stability_weight,
            ),
            4,
        )

    df = pd.DataFrame(rows).sort_values("composite", ascending=False)

    csv_path = out / "summary.csv"
    df.to_csv(csv_path, index=False)

    meta: Dict[str, Any] = {
        "timestamp": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "model_name": model_name,
        "criteria_path": str(crit_path.relative_to(_REPO)),
        "pdf": str(pdf) if pdf else None,
        "text_file": str(text_file) if text_file else None,
        "chunk_sizes": chunk_sizes,
        "overlaps": overlaps,
        "num_pairs": pairs_count,
        "stability_weight": stability_weight,
        "top_k": top_k,
        "chars_in_document": len(raw),
    }
    (out / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    heat_path = fig_dir / "heatmap.png"
    hist_path = fig_dir / "score_histogram.png"
    scat_path = fig_dir / "scatter_size_vs_score.png"
    b64_heat = _save_plot_b64(_plot_heatmap(df), heat_path)
    b64_hist = _save_plot_b64(_plot_hist(df), hist_path)
    b64_scat = _save_plot_b64(_plot_scatter(df), scat_path)

    best = df.iloc[0]
    worst = df.iloc[-1]
    mean_score = float(df["overall_score"].mean())
    std_score = float(df["overall_score"].std(ddof=0))

    top_html = df.head(15).to_html(index=False, float_format=lambda x: f"{x:.4f}")

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8"/>
  <title>Calibration report</title>
  <style>
    body {{ font-family: system-ui, sans-serif; max-width: 960px; margin: 2rem auto; padding: 0 1rem; }}
    h1 {{ color: #1a1a2e; }}
    .muted {{ color: #555; font-size: 0.95rem; }}
    img {{ max-width: 100%; height: auto; border: 1px solid #ddd; margin: 1rem 0; }}
    table.data {{ border-collapse: collapse; width: 100%; font-size: 0.85rem; }}
    table.data th, table.data td {{ border: 1px solid #ccc; padding: 0.35rem 0.5rem; }}
  </style>
</head>
<body>
  <h1>Chunk hyperparameter calibration</h1>
  <p class="muted">Generated {meta["timestamp"]} · model <code>{meta["model_name"]}</code></p>
  <p>Document length: <strong>{meta["chars_in_document"]}</strong> characters ·
     grid: <strong>{meta["num_pairs"]}</strong> valid (chunk_size, overlap) pairs.</p>
  <p>Overall score on grid — min <strong>{df["overall_score"].min():.2f}</strong>,
     max <strong>{df["overall_score"].max():.2f}</strong>,
     mean <strong>{mean_score:.2f}</strong>, stdev <strong>{std_score:.2f}</strong>.</p>
  <p><strong>Top composite</strong> (weight {stability_weight} on neighbor stdev):
     chunk_size={int(best["chunk_size"])}, overlap={int(best["overlap"])},
     overall={float(best["overall_score"]):.2f}, composite={float(best["composite"]):.4f}.</p>
  <p><strong>Lowest composite:</strong> chunk_size={int(worst["chunk_size"])},
     overlap={int(worst["overlap"])}, overall={float(worst["overall_score"]):.2f}.</p>

  <h2>Heatmap</h2>
  <p class="muted">Row = overlap, column = chunk size. Missing cells = invalid overlap ≥ size.</p>
  <img alt="heatmap" src="data:image/png;base64,{b64_heat}"/>

  <h2>Score distribution</h2>
  <img alt="histogram" src="data:image/png;base64,{b64_hist}"/>

  <h2>Chunk size vs score</h2>
  <img alt="scatter" src="data:image/png;base64,{b64_scat}"/>

  <h2>Top 15 by composite</h2>
  {top_html}

  <h2>Files</h2>
  <ul>
    <li><code>summary.csv</code> — full results</li>
    <li><code>meta.json</code> — run configuration</li>
    <li><code>figures/</code> — same plots as PNG</li>
  </ul>
</body>
</html>
"""
    report_path = out / "report.html"
    report_path.write_text(html, encoding="utf-8")

    print(f"Wrote: {csv_path}")
    print(f"Wrote: {out / 'meta.json'}")
    print(f"Wrote: {report_path}")
    print(f"Wrote: {fig_dir}/*.png")


def _plot_scatter(df: pd.DataFrame) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(7, 4))
    sc = ax.scatter(
        df["chunk_size"],
        df["overall_score"],
        c=df["overlap"],
        cmap="coolwarm",
        s=60,
        alpha=0.85,
        edgecolors="k",
        linewidths=0.3,
    )
    ax.set_xlabel("Chunk size (chars)")
    ax.set_ylabel("Overall score")
    ax.set_title("Score vs chunk size (color = overlap)")
    fig.colorbar(sc, ax=ax, label="Overlap")
    fig.tight_layout()
    return fig


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--pdf", type=Path, help="Path to ordinance PDF")
    p.add_argument("--text", type=Path, help="Path to UTF-8 plain text (alternative to PDF)")
    p.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Output directory (default: calibration_reports/<UTC timestamp>)",
    )
    p.add_argument(
        "--chunk-sizes",
        type=str,
        default=None,
        help=f"Comma-separated sizes (default: range from presets, "
        f"{SWEEP_CHUNK_SIZE_MIN_DEFAULT}–{SWEEP_CHUNK_SIZE_MAX_DEFAULT} step "
        f"{SWEEP_CHUNK_SIZE_STEP_DEFAULT})",
    )
    p.add_argument(
        "--overlaps",
        type=str,
        default=None,
        help=f"Comma-separated overlaps (default: range from presets)",
    )
    p.add_argument("--max-pairs", type=int, default=72, help="Abort if valid grid exceeds this")
    p.add_argument("--stability-weight", type=float, default=1.0, help="Composite score penalty weight")
    p.add_argument("--top-k", type=int, default=1, help="Top excerpts per criterion (passed to scorer)")
    p.add_argument(
        "--model",
        type=str,
        default=None,
        help="Single Sentence-Transformers model id (default: app.defaults.DEFAULT_SENTENCE_TRANSFORMER_MODEL)",
    )
    p.add_argument(
        "--models",
        type=str,
        default=None,
        help="Comma-separated model ids; when set, overrides --model and writes each run to a subfolder",
    )
    args = p.parse_args()

    out = args.out
    if out is None:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        out = _REPO / "calibration_reports" / stamp
    out.mkdir(parents=True, exist_ok=True)

    chunk_sizes = (
        _parse_int_list(args.chunk_sizes)
        if args.chunk_sizes
        else _default_chunk_sizes()
    )
    overlaps = _parse_int_list(args.overlaps) if args.overlaps else _default_overlaps()
    pairs = valid_chunk_pairs(chunk_sizes, overlaps)
    if not pairs:
        raise SystemExit("No valid (chunk_size, overlap) pairs (need overlap < chunk size).")
    if len(pairs) > args.max_pairs:
        raise SystemExit(
            f"Grid has {len(pairs)} pairs; max is {args.max_pairs}. "
            "Narrow ranges or increase --max-pairs."
        )

    raw = _load_text(pdf=args.pdf, text_file=args.text).strip()
    if not raw:
        raise SystemExit("No document text extracted or read.")

    crit_path = _REPO / "app" / "criteria.json"
    criteria = json.loads(crit_path.read_text(encoding="utf-8"))["criteria"]

    model_names = _resolve_model_names(model=args.model, models=args.models)
    if len(model_names) > 1:
        (out / "models_run.json").write_text(
            json.dumps({"models": model_names}, indent=2),
            encoding="utf-8",
        )

    for idx, model_name in enumerate(model_names, start=1):
        target_out = out if len(model_names) == 1 else (out / _model_slug(model_name))
        print(f"\n=== Calibration {idx}/{len(model_names)} → {target_out} ===\n")
        _run_one_model_calibration(
            out=target_out,
            model_name=model_name,
            raw=raw,
            criteria=criteria,
            crit_path=crit_path,
            chunk_sizes=chunk_sizes,
            overlaps=overlaps,
            pairs_count=len(pairs),
            pdf=args.pdf,
            text_file=args.text,
            stability_weight=args.stability_weight,
            top_k=args.top_k,
        )


if __name__ == "__main__":
    main()
