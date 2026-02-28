# Dark Sky Ordinance Semantic Scorer (Streamlit)

This repository contains a Streamlit app that lets you upload a dark sky or outdoor lighting ordinance PDF, parses the text, runs semantic analysis against dark sky criteria, and returns a per-criterion score and overall score.

**📖 For detailed documentation, see [EXPLAINER.md](EXPLAINER.md)**

Features
- Upload a PDF and extract text (pdfplumber)
- Chunking/smart splitting for semantic embeddings
- Local embeddings (Sentence-Transformers) — runs fully offline after the model is downloaded
- Scoring engine that compares document chunks to criteria and returns:
  - Per-criterion score (0–100)
  - Top matching excerpt(s) for each criterion
  - Weighted overall score
- Downloadable JSON report

Quick start (local)
1. Create and activate a Python 3.9+ venv (the helper script does both steps):
   bash create_venv.sh
   source .venv-ordinance-semantic-scorer/bin/activate

2. Install all dependencies (app + dev/test):
   pip install -r requirements.txt

3. Run the app:
   streamlit run streamlit_app.py

Running tests
- Run the full test suite:
   pytest
- Unit tests only:
   pytest tests/unit/
- Integration tests only:
   pytest tests/integration/
- Stop on first failure:
   pytest -x

Files of interest
- streamlit_app.py — Streamlit UI and orchestration
- app/utils/pdf_parser.py — PDF -> text
- app/utils/text_splitter.py — chunking
- app/analysis/embeddings.py — local embedding provider (Sentence-Transformers)
- app/analysis/scorer.py — scoring logic
- app/data/criteria.json — dark sky ordinance evaluation criteria

Design notes
- The scoring method is semantic similarity: for each criterion we compute the highest cosine similarity between its description and any document chunk, normalize to 0–100, apply criterion weights, and combine.
- The code is modular so you can replace or extend criteria, change embedding models, or swap parsing logic.
- Uses SentenceTransformer models locally - no external API calls required. Models are downloaded and cached automatically on first use.

Documentation
- **[EXPLAINER.md](EXPLAINER.md)** - Comprehensive guide covering:
  - How the tool works (technical details)
  - Scoring methodology
  - All 29 dark sky criteria explained
  - Usage instructions (web and CLI)
  - Interpreting results
  - Best practices
  - Troubleshooting
  - Advanced customization

License
- MIT