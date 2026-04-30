# Agentic development guide

Use this document when driving the repo with an AI coding agent (Cursor, Copilot, etc.). It ties together **CI**, **tests**, **calibration**, **benchmark PDFs**, and a path toward **deterministic** scoring.

## 1. Day-to-day loop

1. **Branch** off `main` for any feature or fix.
2. **Run tests locally** before pushing (see **Test reports** below). Fix failures the agent surfaces; re-run until green.
3. **Open a PR** into `main`. GitHub Actions runs the same suite (minus `slow` tests) and uploads **JUnit XML** + **coverage** as workflow artifacts.
4. **Merge** only when CI is green unless you intentionally bypass (not recommended).

## 2. Test reports (local and CI)

**Local HTML coverage + terminal summary:**

```bash
cd /path/to/ordinance-semantic-scorer
source .venv-ordinance-semantic-scorer/bin/activate   # or your venv
pip install pytest-cov   # if not already installed
pytest tests/ \
  -m "not slow" \
  --cov=app \
  --cov-report=html:htmlcov \
  --cov-report=term-missing \
  --junitxml=test-reports/junit.xml
open htmlcov/index.html   # macOS
```

**CI:** On push/PR to `main`, workflow `.github/workflows/ci.yml` runs `pytest -m "not slow"` with `--cov` and uploads `test-reports/` + `htmlcov/` as the artifact **test-reports-&lt;run id&gt;**.

**Prompting an agent:** *“Run pytest with coverage and junit; open failures; fix until green.”*

## 3. Optional benchmark PDFs (four jurisdictions)

1. Add PDFs under `sample/ordinances/` using the names in `sample/ordinances/README.md`, or edit `tests/fixtures/ordinances/manifest.json` paths to match your files.
2. **Fast checks (default CI):** `tests/integration/test_ordinance_corpus.py` — for each manifest entry whose file exists, runs **real PDF extraction**, **quality heuristics** (`app/extraction_quality.py`), **keyword** checks, then **full scoring with a mocked embedding model** (no download, deterministic).
3. **If PDF is not parsed properly:** extraction tests fail with a clear message (short text, low letter ratio, missing keywords). That usually means **image-only scan** — this project does not OCR; replace with a text-layer PDF or pre-extract text to `.txt` and use `scripts/calibrate.py --text`.

**Prompting an agent:** *“Run integration tests for the ordinance corpus; if Flagstaff PDF fails extraction, inspect with pdfplumber and suggest OCR or a new source PDF.”*

## 4. Real-model regression (slow, manual)

When you are ready to assert **Flagstaff / Sisters should score high** (or any manifest `benchmark_min_overall_score`):

```bash
export ORDINANCE_RUN_SCORING_BENCHMARK=1
pytest tests/integration/test_ordinance_corpus.py -m slow -v --tb=short
```

This downloads Sentence-Transformers weights on first run and is **not** part of default CI.

**Prompting an agent:** *“With ORDINANCE_RUN_SCORING_BENCHMARK=1, run slow corpus tests; if Sisters fails the floor, print overall_score and suggest updating manifest thresholds or criteria wording.”*

## 5. Calibration reports (chunk grid, one or many models)

**Local (single model, default grid):**

```bash
pip install -r requirements.txt -r requirements-calibration.txt
python scripts/calibrate.py --pdf sample/ordinances/flagstaff.pdf
```

**Multiple Sentence-Transformers models** (each model → subdirectory under `--out`):

```bash
python scripts/calibrate.py \
  --pdf sample/ordinances/flagstaff.pdf \
  --out calibration_reports/compare_models \
  --models "all-MiniLM-L6-v2,sentence-transformers/all-mpnet-base-v2" \
  --chunk-sizes "1000,1500,2000" \
  --overlaps "50,100" \
  --max-pairs 200
```

Compare `report.html` / `summary.csv` per model folder. Pick a model + chunk pair that balances score and stability (composite in CSV).

**CI artifact (optional):** GitHub → Actions → **Calibration report** → *Run workflow*. Supply PDF path relative to repo and comma-separated models. Download the uploaded artifact.

**Prompting an agent:** *“Add a second model to calibrate.py output; run a small grid on Sisters.pdf; summarize which model gives higher mean overall_score.”*

## 6. Toward deterministic “golden” behaviour

Determinism requires freezing **four** things:

| Lever | What to lock |
|--------|----------------|
| Rubric | Versioned `app/criteria.json` (git tag or checksum in report meta). |
| Model | One `DEFAULT_SENTENCE_TRANSFORMER_MODEL` + pinned `sentence-transformers` / `torch` in `requirements.txt`. |
| Segmentation | `DEFAULT_CHUNK_SIZE` / `DEFAULT_CHUNK_OVERLAP` (or one chosen pair from calibration). |
| Documents | Fixed PDF set + optional `benchmark_min_overall_score` in manifest; run slow tests in release pipeline only. |

Workflow: run calibration → pick (model, chunk_size, overlap) → set defaults → tighten manifest floors from observed scores → enable `ORDINANCE_RUN_SCORING_BENCHMARK` on a **release** workflow or weekly schedule, not every PR.

**Prompting an agent:** *“Document the chosen golden hyperparameters in README; add meta.criteria_sha256 to calibration meta.json.”* (Implement sha when you want the agent to add it.)

## 7. Agent task cheat sheet

| Goal | One-line agent instruction |
|------|-----------------------------|
| Fix CI | “Read GitHub Actions logs for the failed job; reproduce with pytest; patch.” |
| New ordinance | “Add PDF under sample/ordinances; align manifest path; run corpus integration tests.” |
| Poor PDF score | “Compare extraction_quality metrics and top excerpts; check for scan/OCR.” |
| Model comparison | “Run calibrate.py --models A,B on the same PDF; tabulate mean overall_score from summary.csv.” |
| Determinism | “Pin one model and chunk preset; propose manifest benchmark floors from last slow run.” |

## 8. Files added for this workflow

- `.github/workflows/ci.yml` — test + coverage artifacts on `main`.
- `.github/workflows/calibration.yml` — manual calibration bundle upload.
- `app/extraction_quality.py` — extraction sanity checks.
- `tests/fixtures/ordinances/manifest.json` — corpus list + optional score floors.
- `tests/integration/test_ordinance_corpus.py` — extraction + mocked scoring (+ slow real model).
