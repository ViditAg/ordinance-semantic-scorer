# DarkSky Ordinance Semantic Scorer

Python Streamlit app: upload an outdoor lighting / dark-sky ordinance PDF, extract text, chunk it, embed with [Sentence-Transformers](https://huggingface.co/sentence-transformers) locally, and score against `app/criteria.json` (per-criterion 0–100, weighted overall, top evidence excerpts, JSON export).

**Everything below is in this file** (single documentation entry point).

---

## 1. How to learn (step by step)

Do these in order the first time you touch the repo.

1. **Skim this README** through section 3 (how it works) so you know the pipeline and scoring rule.
2. **Open the rubric** [`app/criteria.json`](app/criteria.json): each row has `title`, `probes` (short phrases embedded for matching), `weight`, optional `pillar` (UI grouping only).
3. **Read the engine** [`app/scorer.py`](app/scorer.py) docstring and `score()`: max cosine per probe over chunks, mean across probes per criterion, weighted overall.
4. **Trace the app path** [`streamlit_app.py`](streamlit_app.py) → [`app/application/scoring_service.py`](app/application/scoring_service.py) → `OrdinanceScorer`; note [`app/utils.py`](app/utils.py) for PDF extraction and character chunking.
5. **Run the test suite** (section 2.3) and open one unit file (e.g. `tests/unit/test_scorer.py`) to see expected scoring behavior with mocked embeddings.
6. **Optional deep dive:** read [`app/chunking_presets.py`](app/chunking_presets.py) for `DEFAULT_CHUNK_*` vs `SWEEP_*` (calibration grid), and section 5 on evidence-level limits.

---

## 2. How to perform (step by step)

### 2.1 Environment and run the app

1. Use **Python 3.9** (not **3.9.7**), **3.10+**, or **3.11+** (see Streamlit’s `Requires-Python` on PyPI).
2. Create a venv and install (helper script if you use it):

   ```bash
   bash create_venv.sh
   source .venv-ordinance-semantic-scorer/bin/activate
   pip install -r requirements.txt
   ```

   Dependencies are pinned in `requirements.txt`. If install fails on another OS (e.g. `torch` wheel), create a fresh venv there, install the top-level versions from the file header comment, then `pip freeze` and refresh the lock for that platform.

3. Start the UI:

   ```bash
   streamlit run streamlit_app.py
   ```

4. In the app: upload a PDF → set **Top Excerpts** if needed → **Run semantic scoring** → review overall score, per-criterion scores, and **evidence excerpts** → download JSON if useful.

### 2.2 Model and chunking (fixed in code)

The UI does **not** change the embedding model or chunk hyperparameters. They live in:

- [`app/defaults.py`](app/defaults.py) — `DEFAULT_SENTENCE_TRANSFORMER_MODEL` (currently `sentence-transformers/all-mpnet-base-v2`; scores are **not** comparable to older MiniLM runs).
- [`app/chunking_presets.py`](app/chunking_presets.py) — `DEFAULT_CHUNK_SIZE` / `DEFAULT_CHUNK_OVERLAP` (re-exported through `defaults` for a single import).

Edit those modules and redeploy to change behavior. For programmatic grid experiments use `app.stability.run_chunk_sweep` or `scripts/calibrate.py`.

### 2.3 Tests (local)

```bash
pytest -m "not slow"                    # default CI-equivalent (skip real embedding benchmark)
pytest tests/unit/
pytest tests/integration/
pytest -m "not slow" --cov=app --cov-report=html:htmlcov --junitxml=test-reports/junit.xml
pytest -x                               # stop on first failure
```

**CI:** pushes/PRs to `main` run `.github/workflows/ci.yml` and upload **test-reports** + coverage artifacts.

### 2.4 Benchmark PDFs and optional real-model floors

- Corpus list: [`tests/fixtures/ordinances/manifest.json`](tests/fixtures/ordinances/manifest.json). Committed samples: `sample/Flagstaff.pdf`, `Sisters.pdf`, `Bend.pdf`, `Deschutes.pdf`.
- Fast integration tests: [`tests/integration/test_ordinance_corpus.py`](tests/integration/test_ordinance_corpus.py) — real extraction + mocked embeddings (deterministic).
- **Slow** real-model regression (downloads weights; not in default CI):

  ```bash
  export ORDINANCE_RUN_SCORING_BENCHMARK=1
  pytest tests/integration/test_ordinance_corpus.py -m slow -v --tb=short
  ```

- Ground-truth spot checks (substring + manifest integrity): [`tests/fixtures/ground_truth/labels.json`](tests/fixtures/ground_truth/labels.json) and `tests/integration/test_ground_truth_labels.py`.

**Scanned PDFs:** only embedded text is read (`pdfplumber`); image-only PDFs fail extraction checks — use a text-layer PDF or `scripts/calibrate.py --text` with a `.txt` extract.

### 2.5 Calibration (chunk grid, optional multi-model)

```bash
pip install -r requirements.txt -r requirements-calibration.txt
python scripts/calibrate.py --pdf sample/Flagstaff.pdf
```

Default output: `calibration_reports/<UTC>/` with `report.html`, `summary.csv`, `meta.json`, `figures/*.png`. Without `--chunk-sizes` / `--overlaps`, the script uses `SWEEP_*` from `app/chunking_presets.py` (chunk sizes **500–1500** step 250, overlaps **50–150** step 25 — includes production-style **750**). Every overlap must be **strictly less than** every chunk size; `--max-pairs` defaults to `72`.

Multi-model example:

```bash
python scripts/calibrate.py \
  --pdf sample/Flagstaff.pdf \
  --out calibration_reports/compare_models \
  --models "all-MiniLM-L6-v2,sentence-transformers/all-mpnet-base-v2,sentence-transformers/all-MiniLM-L12-v2" \
  --max-pairs 72
```

**GitHub:** Actions → **Calibration report** → run workflow (PDF path relative to repo, models, chunk sizes, overlaps). Artifact contains the bundle.

### 2.6 New ordinance PDF in the benchmark set

1. Add the file under a repo path (e.g. `sample/` or `sample/ordinances/`).
2. Add or update an entry in `tests/fixtures/ordinances/manifest.json` (`path`, `min_chars`, `keywords`, optional `benchmark_min_overall_score`).
3. Run `pytest tests/integration/test_ordinance_corpus.py` (and slow benchmarks if you use floors).

---

## 3. How it works

### 3.1 Architecture (hexagonal)

| Layer | Location | Role |
|--------|------------|------|
| Domain | `app/domain/` | `TextSource`, `Chunker` (`ports.py`); `ScoringRequest` / `ScoringResult` (`models.py`). |
| Application | `app/application/` | `OrdinanceScoringService` chunks (via injected `Chunker`), `embed_texts`, `score`. |
| Adapters | `app/adapters/` | `PlainTextSource`, `PdfTextSource`; `FixedCharacterChunker` → `app.utils.chunk_text`. |
| Engine | `app/scorer.py` | Sentence-Transformers embeddings, cosine similarity, weighting. |

`streamlit_app.py` extracts the PDF once, chunks with `FixedCharacterChunker`, calls `OrdinanceScoringService.score_chunks` so the UI uses one chunk list.

### 3.2 Data flow

```
PDF → extract_text_from_pdf → chunk_text → embed chunks + embed criterion probes
     → per probe: max cosine over chunks → mean probe scores per criterion
     → weighted overall + top-k excerpts per criterion
```

### 3.3 Scoring detail

- Each criterion has one or more **probes** (short strings in `criteria.json`; if missing, the `title` is embedded once).
- For each probe, take the **maximum** cosine similarity across **all** document chunks, map to 0–100 with `max(sim, 0) * 100`.
- **Criterion score** = mean of those probe scores. **Overall** = globally renormalized weights from `criteria.json` ( `pillar` is UI-only).

**Evidence:** top‑k chunks for a criterion are ranked by the best match to **any** probe for that criterion.

### 3.4 Rubric shape

**17** scored criteria (five **pillars** in the UI): Purpose; Applicability; Features of light (shielding, trespass, CCT, uplight, after-hours, controls, budgets); Type of lighted areas (classes, streets, greenhouses, signage, parking/holiday/sports); Regulatory (permits, enforcement). If you change the count, update `EXPECTED_CRITERIA_COUNT` in `tests/integration/test_pipeline.py`.

### 3.5 Interpreting per-criterion bands (similarity-based, not legal advice)

| Range | Rough read |
|-------|------------|
| 90–100 | Strong match to probe themes |
| 70–89 | Adequate topical coverage |
| 50–69 | Partial |
| 30–49 | Weak |
| 0–29 | Little or no match |

Always read **excerpts**; similarity is not legal sufficiency.

### 3.6 Default model note

`all-mpnet-base-v2` is a strong general sentence encoder; overall scores tend to run higher than small MiniLM checkpoints on the same text. Do not compare absolute numbers across different models.

---

## 4. Evaluation and retrieval strategy

The pipeline asks *“is there some chunk that looks like this probe?”* It does **not** prove correct legal obligations, negation, or exceptions.

**Common issues:** false-positive excerpts, high scores from vague topical overlap, weak handling of legal nuance on general-purpose embeddings.

**What helps:** manifest + committed PDFs under `tests/fixtures/ordinances/manifest.json`; calibration sweep aligned with production (`SWEEP_*` includes 750); `tests/fixtures/ground_truth/labels.json` for v0 integrity (substring + ids), not full Recall@K yet.

**What aggregate scores are bad at:** proving the **right** paragraph was retrieved. Prefer extending labels with gold spans or chunk indices and measuring hit rate @K against `top_excerpts`, plus small gold vs hard-negative probe sets (negation, numeric limits). Optional next steps: other bi-encoders, cross-encoder rerank on top‑K — report deltas vs `DEFAULT_SENTENCE_TRANSFORMER_MODEL`.

**Determinism (golden runs):** lock rubric revision, model + `requirements.txt`, `DEFAULT_CHUNK_*`, and document set; use slow benchmarks or release-only jobs for score floors, not every PR.

---

## 5. Agent / automation cheat sheet

| Goal | Action |
|------|--------|
| Fix CI | Reproduce with `pytest -m "not slow"`; patch until green. |
| New PDF | Add file; update manifest; run corpus integration tests. |
| Bad scores / empty text | Check `app/extraction_quality.py` metrics; suspect scan → text PDF or `--text`. |
| Model comparison | `calibrate.py --models A,B,C`; compare excerpts and `composite`, not only raw score. |
| Lock behavior | Set defaults from one chosen (model, chunk_size, overlap); tune manifest floors from slow run. |

---

## 6. Files of interest

- `streamlit_app.py` — UI
- `app/application/scoring_service.py` — orchestration
- `app/scorer.py` — embeddings + scoring
- `app/utils.py` — PDF + chunking
- `app/stability.py` — chunk sweep
- `app/defaults.py`, `app/chunking_presets.py` — model + chunk policy
- `app/criteria.json` — rubric
- `scripts/calibrate.py` — calibration reports

Design: modular criteria and chunk policy; hexagonal boundaries keep the scorer independent of Streamlit.

---

## 7. Appendix: retrieval quality review checklist (concise)

When improving match quality, inspect in order:

1. **Embeddings** — model id, `normalize_embeddings=True`, truncation, query/document conventions if you swap models.
2. **Geometry** — cosine / dot product on unit vectors; score compression (`app/utils.py`).
3. **Chunking** — size, overlap, mixed-topic dilution; align experiments with `DEFAULT_CHUNK_*`.
4. **Aggregation** — max over chunks per probe rewards any single hit; consider whether product needs rerank or stricter evidence rules.
5. **Domain** — legal negation, numbers, headings; consider hybrid (sparse + dense) or cross-encoder rerank on candidate chunks.
6. **ANN** — this codebase uses brute-force cosine over all chunks per run; if you add FAISS etc., validate recall vs exact neighbors.

---

## 8. Best practices

**Ordinance authors:** be specific; use technical terms (e.g. full cutoff, BUG); define terms; cover rubric themes for stronger scores.

**Evaluators:** trust excerpts over numbers alone; compare ordinances with the **same** model and chunk settings; manually review important decisions.

**Developers:** version rubric changes; adjust weights deliberately; tune chunking for your PDFs; extend tests when changing scoring contracts.

---

## 9. Limitations

**Does well:** surfaces related sections, comparable scores, local/offline runs, varied wording.

**Does not:** legal enforceability, contradiction checks, jurisdiction-specific compliance, or replace expert review. Poor PDFs → bad extraction. Semantic similarity can miss matches or show false positives.

---

## 10. License

[MIT](LICENSE)
