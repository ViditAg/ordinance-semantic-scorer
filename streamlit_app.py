"""
Streamlit front-end for the Dark Sky Ordinance Semantic Scorer.

**What this app does**

1. Loads weighted scoring criteria from ``app/criteria.json`` (titles, atomic
   ``probes``, relative weights used in the final aggregate score).
2. Accepts a user-uploaded PDF ordinance, extracts plain text, and splits it into
   overlapping chunks using **fixed defaults** in ``app/chunking_presets.py``.
3. Embeds each chunk and each criterion probe with a **fixed** Sentence-Transformers
   model id from ``app.defaults`` (no model picker in the UI).
4. Calls ``OrdinanceScorer.score`` to compute per-criterion semantic similarity,
   surfaces the best-matching excerpts, and offers a JSON download for archival.

**How to run**

From the repository root (with dependencies installed)::

    streamlit run streamlit_app.py

**Layout**

* **Sidebar** — ``top_k`` excerpts per criterion only.
* **Main area** — upload, optional text preview, run button, metrics, per-row results
  (criterion title plus full embedded probe text, score, top excerpt text), JSON
  download plus optional **server-side save** in the sidebar
  (local runs only; browsers cannot pick a download directory for ``st.download_button``).

Orchestration uses ``app.application.scoring_service.OrdinanceScoringService`` with
``app.adapters`` chunking; the matrix engine lives in ``app.scorer``; PDF helpers in
``app.utils``. Fixed model and chunking live in ``app.defaults`` / ``app.chunking_presets``.
"""

# importing python standard libraries
from __future__ import annotations

import datetime as dt
import html
import json
from pathlib import Path
from typing import Dict, List, Optional

# importing third-party libraries
import streamlit as st

# importing local libraries
from app.adapters.chunking_char import FixedCharacterChunker
from app.application.scoring_service import OrdinanceScoringService
from app.defaults import (
    DEFAULT_CHUNK_OVERLAP,
    DEFAULT_CHUNK_SIZE,
    DEFAULT_SENTENCE_TRANSFORMER_MODEL,
)
from app.domain.models import ScoringRequest
from app.scorer import criterion_probe_texts, criterion_short_preview
from app.utils import extract_text_from_pdf

# defining constants
## application root
_APP_ROOT = Path(__file__).resolve().parent
## chunk size
CHUNK_SIZE = DEFAULT_CHUNK_SIZE
## chunk overlap
CHUNK_OVERLAP = DEFAULT_CHUNK_OVERLAP
## model name
MODEL_NAME = DEFAULT_SENTENCE_TRANSFORMER_MODEL

def _criterion_by_title(criteria_list: List[Dict]) -> Dict[str, Dict]:
    """Map title -> full criterion dict for lookups (last wins if duplicate titles)."""
    return {c["title"]: c for c in criteria_list}


def _criterion_probes_html(title: str, probes: List[str]) -> str:
    """Scrollable HTML: title plus bullet list of atomic probes (embedded for scoring)."""
    t = html.escape((title or "").strip() or "Untitled")
    usable = [p.strip() for p in probes if (p or "").strip()]
    if not usable:
        body = "<em>No probes in criteria.json.</em>"
    else:
        lis = "".join(f"<li>{html.escape(p)}</li>" for p in usable)
        body = (
            f"<ul style='margin:0.35em 0 0 1.1em;padding:0;'>{lis}</ul>"
        )
    return (
        '<div class="criterion-full-panel">'
        f"<strong>{t}</strong><br/>"
        f'<span class="criterion-full-body">{body}</span>'
        "</div>"
    )


def _format_excerpts_plain(excerpts: List[str]) -> str:
    """Join top excerpts for monospace display (no HTML / markdown interpretation)."""
    parts: List[str] = []
    for raw in excerpts:
        t = (raw or "").strip()
        parts.append(t if t else "(empty)")
    if not parts:
        return "(no excerpts)"
    if len(parts) == 1:
        return parts[0]
    sep = "\n\n---\n\n"
    return sep.join(parts)


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

## get top k excerpts per criterion
top_k = st.sidebar.number_input(
    "Top excerpts per criterion",
    min_value=1,
    max_value=5,
    value=1,
    help="Increase to see alternative supporting passages in the results table.",
)

## optional: write JSON on the machine running Streamlit (local use).
## Browser "Download" cannot choose a folder; that is controlled by the browser.
save_json_folder = st.sidebar.text_input(
    "Save JSON to folder (optional)",
    value="",
    placeholder="e.g. /Users/you/Documents/ordinance_reports",
    help=(
        "If set, after scoring the app writes the same JSON to this folder on the "
        "computer running Streamlit (creates the folder if needed). "
        "Does not apply to hosted cloud Streamlit unless the server path is writable."
    ),
)
save_json_filename = st.sidebar.text_input(
    "JSON file name",
    value="ordinance_score.json",
    help="Basename only (no slashes). Used with the folder above and for the download button.",
)

## show fixed model and chunking parameters as a caption
st.sidebar.caption(
    f"**Fixed model:** `{MODEL_NAME}` (`app/defaults.py`).  \n"
    f"**Fixed segmentation:** {CHUNK_SIZE} chars, overlap {CHUNK_OVERLAP} (`app/chunking_presets.py`)."
)

# -----------------------------------------------------------------------------
# Criteria rubric — static JSON shipped with the app (not user-editable in UI).
# -----------------------------------------------------------------------------

## upload ordinance PDF
uploaded_file = st.file_uploader(
    "Upload ordinance PDF",
    type=["pdf"],
    help="Upload a dark sky or outdoor lighting ordinance PDF to run semantic analysis against dark sky criteria and get a score breakdown.",
)

# load criteria from JSON file
## criteria path is fixed at app/criteria.json for consistency
criteria_path = _APP_ROOT / "app" / "criteria.json"
with open(criteria_path, "r", encoding="utf-8") as fh:
    criteria = json.load(fh)
## show number of criteria loaded as a caption
st.sidebar.markdown(
    f"Loaded {len(criteria['criteria'])} criteria from app/criteria.json"
)

## optionally show criteria in the app for transparency/review
show_criteria = st.sidebar.checkbox(
    "Show criteria.json",
    value=False,
    help="Display the scoring rubric loaded from app/criteria.json.",
)

if show_criteria:
    st.subheader("Scoring rubric (criteria.json)")
    with st.expander("Criteria summary", expanded=False):
        summary_rows = [
            {
                "serial_number": c.get("serial_number"),
                "pillar": c.get("pillar"),
                "title": c.get("title"),
                "weight": c.get("weight"),
                "probes_preview": criterion_short_preview(c, max_len=140),
            }
            for c in criteria["criteria"]
        ]
        st.dataframe(summary_rows, use_container_width=True, hide_index=True)
    with st.expander("Raw criteria.json", expanded=False):
        st.json(criteria)

# -----------------------------------------------------------------------------
# Main pipeline — only runs after the user supplies a PDF.
# -----------------------------------------------------------------------------

## only run if a PDF is uploaded
if uploaded_file is not None:
    ## extract text from PDF
    with st.spinner("Extracting text from PDF..."):
        raw_text = extract_text_from_pdf(uploaded_file)

    ## if no text is found, show an error and stop
    if not raw_text.strip():
        st.error("No text found in PDF.")
        st.stop()

    st.info(f"Extracted ~{len(raw_text)} characters.")

    ## show text preview if checkbox is checked
    if st.checkbox("Show text preview"):
        st.text_area(
            "Document text (preview)",
            value=raw_text[:10000],
            height=300,
        )
    ## chunk text using fixed character chunker
    chunker = FixedCharacterChunker(
        chunk_size=CHUNK_SIZE,
        overlap=CHUNK_OVERLAP,
    )
    chunks = chunker.chunk(
        text=raw_text,
    )
    ## show number of chunks as a success message
    st.success(
        f"Split into {len(chunks)} chunks (fixed: {CHUNK_SIZE} chars, overlap {CHUNK_OVERLAP})."
    )

    ## initialize scoring service
    scoring_service = OrdinanceScoringService(
        criteria=criteria["criteria"],
        model_name=MODEL_NAME,
        chunker=chunker,
    )

    ## run semantic scoring if button is clicked
    if st.button("Run semantic scoring"):
        ## embed document and criteria
        with st.spinner("Embedding document and rubric probes..."):
            result = scoring_service.score_chunks(
                chunks,
                ScoringRequest(top_k=int(top_k)),
            )

        ## get results
        results = result.score_payload
        crit_lookup = _criterion_by_title(criteria["criteria"])

        ## create report dictionary (same payload for disk + download)
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
        report_json = json.dumps(report, indent=2)

        name_raw = (save_json_filename or "").strip() or "ordinance_score.json"
        folder_raw = (save_json_folder or "").strip()

        ## headline row: overall score + JSON export at the top
        col_metric, col_export = st.columns([2, 1])
        with col_metric:
            st.metric(
                "Overall ordinance score",
                value=f"{results['overall_score']:.1f}",
            )
        with col_export:
            st.markdown("**JSON export**")
            if "/" in name_raw or "\\" in name_raw:
                st.warning(
                    "JSON file name must be a basename only (no path). "
                    "Use “Save JSON to folder” for the directory."
                )
            else:
                if folder_raw:
                    try:
                        dest_dir = Path(folder_raw).expanduser().resolve()
                        dest_dir.mkdir(parents=True, exist_ok=True)
                        out_path = dest_dir / Path(name_raw).name
                        out_path.write_text(report_json, encoding="utf-8")
                        st.caption(f"Saved to `{out_path}`")
                    except OSError as exc:
                        st.warning(
                            f"Could not save JSON to disk ({exc!s}). "
                            "Use the download button, or check folder permissions / path."
                        )
                st.download_button(
                    "Download JSON report",
                    data=report_json,
                    file_name=Path(name_raw).name,
                    mime="application/json",
                    type="primary",
                    use_container_width=True,
                )

        st.markdown(
            """
<style>
  div.criterion-full-panel {
    margin: 0.35em 0;
    max-height: 16rem;
    overflow-y: auto;
    padding: 0.45em 0.65em 0.45em 0.75em;
    border-left: 3px solid #43a047;
    border-radius: 0.2rem;
    font-size: 0.88rem;
    line-height: 1.45;
    color: var(--text-color, inherit);
  }
  span.criterion-full-body {
    display: inline-block;
    margin-top: 0.35em;
  }
</style>
""",
            unsafe_allow_html=True,
        )

        ## show per-criterion scores
        st.subheader("Per-criterion scores")
        st.caption(
            "Each **score** averages 0–100 matches from the **atomic probes** listed for "
            "that criterion. Excerpts rank chunks by the best cosine match to any probe."
        )
        header_cols = st.columns([2.45, 0.65, 2.9])
        header_cols[0].write("Criterion (atomic probes)")
        header_cols[1].write("Score")
        header_cols[2].write("Top excerpt(s)")

        prev_pillar: Optional[str] = None
        for r in results["criteria_results"]:
            pillar = r.get("pillar")
            if pillar and pillar != prev_pillar:
                st.markdown(f"#### {pillar}")
                prev_pillar = pillar
            row_cols = st.columns([2.45, 0.65, 2.9])
            crit_full = crit_lookup.get(r["title"])
            title = r["title"]
            probes_list: List[str] = criterion_probe_texts(crit_full) if crit_full else []
            row_cols[0].markdown(
                _criterion_probes_html(title, probes_list),
                unsafe_allow_html=True,
            )
            row_cols[1].metric(
                "Score",
                f"{r['score']:.1f}",
                label_visibility="hidden",
            )
            row_cols[2].text(_format_excerpts_plain(r["top_excerpts"]))