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

### Local calibration (benchmarking)

Run a sweep on one PDF or text file, then open an HTML report with plots (heatmap, histogram, scatter) plus CSV:

```bash
pip install -r requirements.txt -r requirements-calibration.txt
python scripts/calibrate.py --pdf path/to/ordinance.pdf
```

Outputs go to `calibration_reports/<UTC timestamp>/` by default (`report.html`, `summary.csv`, `meta.json`, `figures/*.png`). Use `--out ./my_run` to pick a folder. Override the grid with `--chunk-sizes 1000,1500,2000` and `--overlaps 50,100,150`. See `python scripts/calibrate.py --help`.

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
- app/utils.py — PDF extraction and text chunking
- app/scorer.py — local embeddings (Sentence-Transformers) and scoring logic
- app/stability.py — optional grid sweep of chunk size/overlap vs overall score (stability)
- app/defaults.py — fixed Sentence-Transformers model id (+ chunk constants re-export)
- app/chunking_presets.py — chunk size/overlap defaults and optional preset helpers for scripts/tests
- app/criteria.json — dark sky ordinance evaluation criteria
- scripts/calibrate.py — local chunk grid sweep → HTML report + plots (optional `requirements-calibration.txt`)

Design notes
- The scoring method is semantic similarity: for each criterion we compute the highest cosine similarity between its description and any document chunk, normalize to 0–100, apply criterion weights, and combine.
- The code is modular so you can replace or extend criteria, change embedding models, or swap parsing logic.
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