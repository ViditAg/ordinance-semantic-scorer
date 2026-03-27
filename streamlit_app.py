"""Streamlit app for the Dark Sky Ordinance Semantic Scorer."""

import json
from pathlib import Path

import streamlit as st

from app.ordinance_workflow import (
    DEFAULT_CHUNK_OVERLAP,
    DEFAULT_CHUNK_SIZE,
    DEFAULT_CRITERIA_PATH,
    DEFAULT_MODEL_NAME,
    build_score_report,
    extract_text_and_chunk,
    load_criteria_bundle,
    run_scoring,
)

st.set_page_config(
    page_title="Dark Sky Ordinance Semantic Scorer",
    layout="wide",
)

st.title("Dark Sky Ordinance Semantic Scorer")
st.markdown(
    "Upload a dark sky or outdoor lighting ordinance PDF, "
    "run semantic analysis against dark sky criteria, "
    "and get a score breakdown."
)

st.sidebar.header("Options")
st.sidebar.caption(
    f"Model `{DEFAULT_MODEL_NAME}`, chunks {DEFAULT_CHUNK_SIZE} chars, "
    f"overlap {DEFAULT_CHUNK_OVERLAP} (fixed for consistent scores)."
)

top_k = st.sidebar.number_input(
    "Top [1-5] excerpts per criterion",
    min_value=1,
    max_value=5,
    value=1,
    help="How many highest-similarity text spans to show per criterion (1-5)",
)

uploaded_file = st.file_uploader(
    "Upload ordinance PDF",
    type=["pdf"],
    help="Upload a dark sky or outdoor lighting ordinance PDF",
)

criteria_path = Path(DEFAULT_CRITERIA_PATH)
criteria, criteria_list = load_criteria_bundle(criteria_path)

st.sidebar.caption(
    f"{len(criteria_list)} criteria loaded from `{criteria_path}`",
)

with st.expander(
    f"Scoring rubric ({len(criteria_list)} criteria)",
    expanded=False,
):
    st.markdown(
        "Each score compares your ordinance text to the **description** below "
        "(semantic match). Weights affect the overall score."
    )
    for i, crit in enumerate(criteria_list):
        num = crit.get("serial_number", "")
        title = crit.get("title", "Untitled")
        w = crit.get("weight", 1.0)
        st.markdown(f"**{num}. {title}**  ·  weight **{w}**")
        st.caption(crit.get("short", ""))
        with st.expander(f"Full description: {title}", expanded=False):
            st.write(crit.get("description", ""))
        if i < len(criteria_list) - 1:
            st.divider()
    st.download_button(
        "Download criteria JSON",
        data=json.dumps(criteria, indent=2, ensure_ascii=False),
        file_name="criteria.json",
        mime="application/json",
        key="download_criteria_json",
    )

if uploaded_file is not None:
    with st.spinner("Extracting text from PDF..."):
        raw_text, chunks = extract_text_and_chunk(uploaded_file)
    if not raw_text.strip():
        st.error("No text found in PDF.")
        st.stop()

    st.info(f"Extracted ~{len(raw_text)} characters.")

    if st.checkbox("Show text preview"):
        st.text_area("Document text (preview)", value=raw_text[:10000], height=300)

    st.success(
        f"Split into {len(chunks)} chunks "
        f"(size {DEFAULT_CHUNK_SIZE}, overlap {DEFAULT_CHUNK_OVERLAP})"
    )

    if st.button("Run semantic scoring"):
        with st.spinner("Embedding document and criteria..."):
            results = run_scoring(chunks, criteria_list, top_k)

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
            excerpt = "\n\n---\n\n".join(
                [f"> {e.replace(chr(10), ' ')}" for e in r["top_excerpts"]]
            )
            row_cols[2].write(excerpt)

        report = build_score_report(
            results,
            num_chunks=len(chunks),
            top_k=top_k,
        )
        st.download_button(
            "Download JSON report",
            data=json.dumps(report, indent=2),
            file_name="ordinance_score.json",
            mime="application/json",
        )
