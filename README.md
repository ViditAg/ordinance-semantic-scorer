# DarkSky Ordinance Semantic Scorer

This repository contains a python-based Streamlit app that lets you,
- Upload a dark sky or outdoor lighting ordinance PDF
- Parses the text and split text into phrases for semantic embeddings
- Runs semantic analysis (locally) against dark sky criteria
   - Local embeddings: Sentence-Transformers python library runs fully offline after the model is downloaded
- Returns a per-criterion score (0–100), Top matching excerpt(s) for each criterion and overall score.
   - Downloadable JSON report

**📖 For detailed documentation on how the tool works, see [EXPLAINER.md](EXPLAINER.md)**
 
# Usage Guide

Quick start (local)
1. Create and activate a venv. Use **Python 3.9** (any patch except **3.9.7**), **3.10+**, or **3.11+** — see Streamlit’s `Requires-Python` on PyPI. The helper script creates the venv and installs from the lockfile:
   bash create_venv.sh
   source .venv-ordinance-semantic-scorer/bin/activate

2. Dependencies are **fully pinned** in `requirements.txt` (including transitive packages) so installs stay reproducible. Reinstall with:
   pip install -r requirements.txt
   The lock was produced on **Python 3.9.6 / macOS**; if `pip install` fails on another OS (for example a different `torch` wheel), create a fresh venv there, install the same six top-level packages at the versions listed in the file header comment in `requirements.txt`, then run `pip freeze` and replace the lock for that platform.

3. Run the app:
   streamlit run streamlit_app.py

### Model and chunking (fixed in code)

The Streamlit UI does **not** expose the embedding model, chunk size, or overlap. Every run uses **`DEFAULT_SENTENCE_TRANSFORMER_MODEL`** in `app/defaults.py` and **`DEFAULT_CHUNK_SIZE`** / **`DEFAULT_CHUNK_OVERLAP`** from `app/chunking_presets.py` (re-exported through `defaults` for a single import surface). Edit those modules and redeploy to change behavior. For grid-style chunk experiments, call `app.stability.run_chunk_sweep` from a script or notebook.

### Architecture (hexagonal layout)

The repo separates **ports** (interfaces), **application** (orchestration), **adapters** (I/O and chunk policy), and the **engine** (`OrdinanceScorer`):

| Layer | Location | Role |
|--------|------------|------|
| Domain | `app/domain/` | `TextSource` and `Chunker` protocols (`ports.py`); `ScoringRequest` / `ScoringResult` (`models.py`). |
| Application | `app/application/` | `OrdinanceScoringService` loads text (optional), chunks via an injected `Chunker`, then calls `OrdinanceScorer.embed_texts` and `score`. |
| Adapters | `app/adapters/` | `PlainTextSource`, `PdfTextSource` (`text_plain.py`, `text_pdf.py`); `FixedCharacterChunker` wraps `app.utils.chunk_text`. |
| Engine | `app/scorer.py` | Sentence-Transformers embeddings and cosine / weighting math (unchanged contract). |

`streamlit_app.py` extracts the PDF once with `extract_text_from_pdf`, builds chunks with `FixedCharacterChunker`, and runs **`OrdinanceScoringService.score_chunks`** on “Run semantic scoring” so the same chunk list used in the UI is scored without a second PDF parse. Other entry points (CLI, notebooks) can use **`score_document(PdfTextSource(...), ScoringRequest(...))`** instead.

### Local calibration (benchmarking)

Run a sweep on one PDF or text file, then open an HTML report with plots (heatmap, histogram, scatter) plus CSV:

```bash
pip install -r requirements.txt -r requirements-calibration.txt
python scripts/calibrate.py --pdf path/to/ordinance.pdf
```

Outputs go to `calibration_reports/<UTC timestamp>/` by default (`report.html`, `summary.csv`, `meta.json`, `figures/*.png`). Use `--out ./my_run` to pick a folder. Override the grid with `--chunk-sizes 1000,1500,2000` and `--overlaps 50,100,150`. See `python scripts/calibrate.py --help`.

Running tests
- Run the full test suite (excludes ``slow`` markers — real embedding benchmarks):
   pytest -m "not slow"
- HTML coverage + JUnit locally:
   pytest -m "not slow" --cov=app --cov-report=html:htmlcov --junitxml=test-reports/junit.xml
- Unit tests only:
   pytest tests/unit/
- Integration tests only:
   pytest tests/integration/
- Stop on first failure:
   pytest -x

**Continuous integration:** Pushes and PRs to ``main`` run ``.github/workflows/ci.yml`` (pytest + coverage + JUnit). Download the **test-reports-…** artifact from the Actions run for XML and HTML coverage.

**Calibration (manual / workflow):** See ``docs/AGENTIC_DEVELOPMENT.md`` for multi-model ``scripts/calibrate.py --models …`` and the optional GitHub **Calibration report** workflow.

**Agentic workflow:** Step-by-step guide for benchmarks, PDF corpus, and determinism: [docs/AGENTIC_DEVELOPMENT.md](docs/AGENTIC_DEVELOPMENT.md).

Files of interest
- streamlit_app.py — Streamlit UI; wires `FixedCharacterChunker` + `OrdinanceScoringService`
- app/domain/ — port protocols and scoring request/result types
- app/application/scoring_service.py — `OrdinanceScoringService` (document → chunks → embed → score)
- app/adapters/ — concrete `TextSource` / `Chunker` implementations for PDF, plain text, and char windows
- app/utils.py — PDF extraction and text chunking (used by adapters and `app.stability`)
- app/scorer.py — local embeddings (Sentence-Transformers) and scoring logic
- app/stability.py — optional grid sweep of chunk size/overlap vs overall score (stability)
- app/defaults.py — fixed Sentence-Transformers model id (+ chunk constants re-export)
- app/chunking_presets.py — chunk size/overlap defaults and optional preset helpers for scripts/tests
- app/criteria.json — dark sky ordinance evaluation criteria
- scripts/calibrate.py — local chunk grid sweep → HTML report + plots (optional `requirements-calibration.txt`)

Design notes
- The scoring method is semantic similarity: for each criterion we compute the highest cosine similarity between its description and any document chunk, normalize to 0–100, apply criterion weights, and combine.
- The code is modular so you can replace or extend criteria, change embedding models, or swap parsing logic.
- Hexagonal boundaries (`TextSource`, `Chunker`, `OrdinanceScoringService`) keep UI and scripts thin and make new sources (e.g. DOCX) or chunk strategies pluggable without changing the scorer.
- Uses SentenceTransformer models locally - no external API calls required. Models are downloaded and cached automatically on first use.


# Best Practices
## For Ordinance Authors

1. **Be Specific**: Use clear, detailed language rather than vague statements
2. **Use Technical Terms**: Include proper terminology (e.g., "full cutoff", "BUG rating")
3. **Provide Examples**: Include examples of compliant and non-compliant lighting
4. **Define Terms**: Have a definitions section for technical terms
5. **Be Comprehensive**: Address all 29 criteria for best scores

## For Evaluators
1. **Review Evidence**: Always check the evidence excerpts, not just scores
2. **Context Matters**: Low scores may indicate missing sections, not poor quality
3. **Compare Multiple Ordinances**: Use scores to compare different ordinances
4. **Iterate**: Use results to identify areas for improvement
5. **Manual Review**: Use tool as a guide, but always do manual review

## For Developers
1. **Customize Criteria**: Edit `app/criteria.json` to add/modify criteria
2. **Adjust Weights**: Change weights based on local priorities
3. **Experiment with Models**: Try different SentenceTransformer models
4. **Tune Chunking**: Adjust chunk size/overlap for your document types
5. **Extend Functionality**: Add new features (e.g., comparison mode, trend analysis)

# Limitations and Considerations
## What the Tool Does Well
- Identifies if criteria are addressed in the ordinance
- Finds relevant sections automatically
- Provides quantitative scores for comparison
- Handles various writing styles and formats
- Works with any PDF ordinance document

## What the Tool Cannot Do
- Evaluate legal quality or enforceability
- Check for contradictions or conflicts
- Verify compliance with local laws
- Assess political feasibility
- Replace expert legal review

## Important Notes
- Semantic similarity is not perfect: The tool may miss some matches or find false positives
- Context matters: A low score doesn't always mean the ordinance is bad
- Language variations: Different wording for the same concept may score differently
- Model limitations: Smaller models may miss technical nuances
- PDF quality: Poorly scanned PDFs may have extraction errors


# License
- [MIT](LICENSE)