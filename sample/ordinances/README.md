# Ordinance PDF corpus (local benchmarks)

Place full-text PDFs here so integration tests and calibration can find them:

| Suggested filename | Jurisdiction (from comparison sheet) |
|--------------------|--------------------------------------|
| `flagstaff.pdf` | City of Flagstaff |
| `sisters.pdf` | City of Sisters |
| `bend.pdf` | City of Bend |
| `deschutes_county.pdf` | Deschutes County |

Paths and expectations live in `tests/fixtures/ordinances/manifest.json`. Rename your files to match, or edit the manifest to match your filenames.

**CI:** PDFs are optional. Tests that need them skip until files exist.

**Scanned PDFs:** `pdfplumber` only reads embedded text; image-only scans yield empty extraction. The integration suite flags that via `app.extraction_quality.assess_extraction_quality`.
