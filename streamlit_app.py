"""
Streamlit front-end for the Dark Sky Ordinance Semantic Scorer.

**What this app does**

1. Loads weighted scoring criteria from ``app/criteria.json`` (titles, descriptions,
   relative weights used in the final aggregate score).
2. Accepts a user-uploaded PDF ordinance, extracts plain text, and splits it into
   overlapping chunks using **fixed defaults** in ``app/chunking_presets.py``.
3. Embeds each chunk and each criterion description with a **fixed** Sentence-Transformers
   model id from ``app.defaults`` (no model picker in the UI).
4. Calls ``OrdinanceScorer.score`` to compute per-criterion semantic similarity,
   surfaces the best-matching excerpts, and offers a JSON download for archival.

**How to run**

From the repository root (with dependencies installed)::

    streamlit run streamlit_app.py

**Layout**

* **Sidebar** — ``top_k`` excerpts per criterion only.
* **Main area** — upload, optional text preview, run button, metrics, per-row results,
  download button.

Orchestration uses ``app.application.scoring_service.OrdinanceScoringService`` with
``app.adapters`` chunking; the matrix engine lives in ``app.scorer``; PDF helpers in
``app.utils``. Fixed model and chunking live in ``app.defaults`` / ``app.chunking_presets``.
"""

from __future__ import annotations

import datetime as dt
import json
from pathlib import Path

import streamlit as st

from app.adapters.chunking_char import FixedCharacterChunker
from app.application.scoring_service import OrdinanceScoringService
from app.defaults import (
    DEFAULT_CHUNK_OVERLAP,
    DEFAULT_CHUNK_SIZE,
    DEFAULT_SENTENCE_TRANSFORMER_MODEL,
)
from app.domain.models import ScoringRequest
from app.utils import extract_text_from_pdf

_APP_ROOT = Path(__file__).resolve().parent

CHUNK_SIZE = DEFAULT_CHUNK_SIZE
CHUNK_OVERLAP = DEFAULT_CHUNK_OVERLAP
MODEL_NAME = DEFAULT_SENTENCE_TRANSFORMER_MODEL

# -----------------------------------------------------------------------------
# Page chrome — title, layout, and introductory copy for first-time users.
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Dark Sky Ordinance Semantic Scorer",
    layout="wide",
)

st.title("Dark Sky Ordinance Semantic Scorer")
st.markdown(
    "Upload a dark sky or outdoor lighting ordinance PDF, run semantic analysis "
    "against dark sky criteria, and get a score breakdown."
)

# -----------------------------------------------------------------------------
# Sidebar
# -----------------------------------------------------------------------------
st.sidebar.header("Options")

top_k = st.sidebar.number_input(
    "Top excerpts per criterion",
    min_value=1,
    max_value=5,
    value=1,
    help="Increase to see alternative supporting passages in the results table.",
)

st.sidebar.caption(
    f"**Fixed model:** `{MODEL_NAME}` (`app/defaults.py`).  \n"
    f"**Fixed segmentation:** {CHUNK_SIZE} chars, overlap {CHUNK_OVERLAP} (`app/chunking_presets.py`)."
)

# -----------------------------------------------------------------------------
# Criteria rubric — static JSON shipped with the app (not user-editable in UI).
# -----------------------------------------------------------------------------
uploaded_file = st.file_uploader("Upload ordinance PDF", type=["pdf"])

criteria_path = _APP_ROOT / "app" / "criteria.json"
with open(criteria_path, "r", encoding="utf-8") as fh:
    criteria = json.load(fh)

st.sidebar.markdown(
    f"Loaded {len(criteria['criteria'])} criteria from app/criteria.json"
)

# -----------------------------------------------------------------------------
# Main pipeline — only runs after the user supplies a PDF.
# -----------------------------------------------------------------------------
if uploaded_file is not None:
    with st.spinner("Extracting text from PDF..."):
        raw_text = extract_text_from_pdf(uploaded_file)

    if not raw_text.strip():
        st.error("No text found in PDF.")
        st.stop()

    st.info(f"Extracted ~{len(raw_text)} characters.")

    if st.checkbox("Show text preview"):
        st.text_area("Document text (preview)", value=raw_text[:10000], height=300)

    chunker = FixedCharacterChunker(CHUNK_SIZE, CHUNK_OVERLAP)
    chunks = chunker.chunk(raw_text)
    st.success(
        f"Split into {len(chunks)} chunks (fixed: {CHUNK_SIZE} chars, overlap {CHUNK_OVERLAP})."
    )

    scoring_service = OrdinanceScoringService(
        criteria=criteria["criteria"],
        model_name=MODEL_NAME,
        chunker=chunker,
    )

    if st.button("Run semantic scoring"):
        with st.spinner("Embedding document and criteria..."):
            result = scoring_service.score_chunks(
                chunks,
                ScoringRequest(top_k=int(top_k)),
            )

        results = result.score_payload

        st.metric("Overall ordinance score", f"{results['overall_score']:.1f} / 100")

        st.subheader("Per-criterion scores")
        header_cols = st.columns([2, 1, 3])
        header_cols[0].write("Criterion")
        header_cols[1].write("Score")
        header_cols[2].write("Top excerpt")

        for r in results["criteria_results"]:
            row_cols = st.columns([2, 1, 3])
            row_cols[0].markdown(f"**{r['title']}**\n\n{r['short']}")
            row_cols[1].metric("", f"{r['score']:.1f}")
            nl = "\n"
            excerpt = "\n\n---\n\n".join(
                [f"> {e.replace(nl, ' ')}" for e in r["top_excerpts"]]
            )
            row_cols[2].write(excerpt)

        report = {
            "meta": {
                "timestamp": dt.datetime.now(dt.timezone.utc)
                .isoformat()
                .replace("+00:00", "Z"),
                "overall_score": results["overall_score"],
                "num_chunks": len(chunks),
                "chunk_size": CHUNK_SIZE,
                "chunk_overlap": CHUNK_OVERLAP,
                "model_name": MODEL_NAME,
            },
            "criteria_results": results["criteria_results"],
        }
        st.download_button(
            "Download JSON report",
            data=json.dumps(report, indent=2),
            file_name="ordinance_score.json",
            mime="application/json",
        )
