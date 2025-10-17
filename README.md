# City Ordinance Semantic Scorer (Streamlit)

This repository contains a Streamlit app that lets you upload a city ordinance PDF, parses the text, runs semantic analysis against a checklist of ordinance quality criteria, and returns a per-criterion score and overall score.

Features
- Upload a PDF and extract text (pdfplumber)
- Chunking/smart splitting for semantic embeddings
- Two embedding backends:
  - Local (Sentence-Transformers) — default, private, no API key
  - OpenAI embeddings — optional (requires `OPENAI_API_KEY`)
- Scoring engine that compares document chunks to criteria and returns:
  - Per-criterion score (0–100)
  - Top matching excerpt(s) for each criterion
  - Weighted overall score
- Downloadable JSON report

Quick start (local)
1. Create and activate a Python 3.9+ venv:
   python -m venv .venv
   source .venv/bin/activate  # macOS / Linux
   .venv\Scripts\activate     # Windows

2. Install dependencies:
   pip install -r requirements.txt

3. Run the app:
   streamlit run streamlit_app.py

Optional: OpenAI
- To use OpenAI embeddings instead of a local model, set:
  export OPENAI_API_KEY="sk-..."
  and pick "OpenAI" in the app sidebar.

Files of interest
- streamlit_app.py — Streamlit UI and orchestration
- app/utils/pdf_parser.py — PDF -> text
- app/utils/text_splitter.py — chunking
- app/analysis/embeddings.py — embedding provider (local or OpenAI)
- app/analysis/scorer.py — scoring logic
- app/data/criteria.json — default ordinance evaluation criteria

Design notes
- The scoring method is semantic similarity: for each criterion we compute the highest cosine similarity between its description and any document chunk, normalize to 0–100, apply criterion weights, and combine.
- The code is modular so you can replace or extend criteria, change embedding models, or swap parsing logic.

License
- MIT