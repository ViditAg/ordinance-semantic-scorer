import streamlit as st
import json
from app.utils.pdf_parser import extract_text_from_pdf
from app.utils.text_splitter import chunk_text
from app.analysis.embeddings import EmbeddingProvider
from app.analysis.scorer import OrdinanceScorer
from pathlib import Path
import datetime

st.set_page_config(page_title="Ordinance Semantic Scorer", layout="wide")

st.title("City Ordinance Semantic Scorer")
st.markdown("Upload a city ordinance PDF, run semantic analysis, and get a score breakdown.")

# Sidebar options
st.sidebar.header("Options")
backend = st.sidebar.selectbox("Embedding backend", ["local (sentence-transformers)", "openai"]) 
chunk_size = st.sidebar.slider("Chunk size (chars)", min_value=500, max_value=5000, value=2000, step=100)
chunk_overlap = st.sidebar.slider("Chunk overlap (chars)", min_value=50, max_value=1000, value=200, step=50)
top_k = st.sidebar.number_input("Top excerpts per criterion", min_value=1, max_value=5, value=1)

uploaded_file = st.file_uploader("Upload ordinance PDF", type=["pdf"]) 

criteria_path = Path("app/data/criteria.json")
with open(criteria_path, "r", encoding="utf-8") as fh:
    criteria = json.load(fh)

st.sidebar.markdown(f"Loaded {len(criteria['criteria'])} criteria from app/data/criteria.json")

if uploaded_file is not None:
    with st.spinner("Extracting text from PDF..."):
        raw_text = extract_text_from_pdf(uploaded_file)
    if not raw_text.strip():
        st.error("No text found in PDF.")
        st.stop()

    st.info(f"Extracted ~{len(raw_text)} characters.")

    # Show small preview
    if st.checkbox("Show text preview"):
        st.text_area("Document text (preview)", value=raw_text[:10000], height=300)

    # Chunk
    chunks = chunk_text(raw_text, chunk_size=chunk_size, overlap=chunk_overlap)
    st.success(f"Split into {len(chunks)} chunks (size ~{chunk_size})")

    # Embeddings provider
    if backend.startswith("openai"):
        api_key = st.sidebar.text_input("OpenAI API key (optional)", type="password")
        provider = EmbeddingProvider(backend="openai", openai_api_key=api_key or None)
    else:
        model_name = st.sidebar.text_input("Sentence-Transformers model", value="all-MiniLM-L6-v2")
        provider = EmbeddingProvider(backend="local", model_name=model_name)

    if st.button("Run semantic scoring"):
        with st.spinner("Embedding document and criteria..."):
            doc_embeddings = provider.embed_texts(chunks)
            crit_texts = [c["description"] for c in criteria["criteria"]]
            crit_embeddings = provider.embed_texts(crit_texts)

        scorer = OrdinanceScorer(criteria=criteria["criteria"])
        results = scorer.score(doc_chunks=chunks, doc_embeddings=doc_embeddings,
                               crit_embeddings=crit_embeddings, top_k=top_k)

        # Display overall score
        st.metric("Overall ordinance score", f"{results['overall_score']:.1f} / 100")

        # Display table of criteria
        st.subheader("Per-criterion scores")
        header_cols = st.columns([2, 1, 3])
        header_cols[0].write("Criterion")
        header_cols[1].write("Score")
        header_cols[2].write("Top excerpt")

        for r in results["criteria_results"]:
            row_cols = st.columns([2, 1, 3])
            row_cols[0].markdown(f"**{r['title']}**\n\n{r['short']}")
            row_cols[1].metric("", f"{r['score']:.1f}")
            excerpt = "\n\n---\n\n".join([f"> {e.replace('\\n', ' ')}" for e in r["top_excerpts"]])
            row_cols[2].write(excerpt)

        # Downloadable JSON report
        report = {
            "meta": {
                "timestamp": datetime.datetime.utcnow().isoformat() + "Z",
                "overall_score": results["overall_score"],
                "num_chunks": len(chunks)
            },
            "criteria_results": results["criteria_results"]
        }
        st.download_button("Download JSON report", data=json.dumps(report, indent=2), file_name="ordinance_score.json", mime="application/json")
