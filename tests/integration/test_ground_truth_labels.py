"""
Integrity checks for ``tests/fixtures/ground_truth/labels.json``.

These rows are **not** full semantic-retrieval ground truth (that would need
char spans or chunk ids and human adjudication). They anchor:

* manifest document ids resolve to files on disk;
* ``criterion_serial_number`` values exist in ``app/criteria.json``;
* ``must_contain_substrings`` appear in the extracted PDF text (text-layer sanity).

Extend the file as you add adjudicated labels for Recall@K or reranker training.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from app.utils import extract_text_from_pdf

ROOT = Path(__file__).resolve().parents[2]
MANIFEST_PATH = ROOT / "tests" / "fixtures" / "ordinances" / "manifest.json"
LABELS_PATH = ROOT / "tests" / "fixtures" / "ground_truth" / "labels.json"
CRITERIA_PATH = ROOT / "app" / "criteria.json"


def _load_manifest_docs():
    data = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    return {d["id"]: d for d in data["documents"]}


def _load_criteria_serials():
    data = json.loads(CRITERIA_PATH.read_text(encoding="utf-8"))
    return {int(c["serial_number"]) for c in data["criteria"]}


class TestGroundTruthLabelsFile:
    def test_labels_json_parseable(self):
        data = json.loads(LABELS_PATH.read_text(encoding="utf-8"))
        assert data["version"] == 1
        assert isinstance(data["labels"], list)
        assert len(data["labels"]) >= 1

    def test_each_label_references_manifest_and_criteria(self):
        manifest = _load_manifest_docs()
        serials = _load_criteria_serials()
        data = json.loads(LABELS_PATH.read_text(encoding="utf-8"))
        for row in data["labels"]:
            assert row["document_id"] in manifest, f"unknown document_id {row['document_id']!r}"
            sn = int(row["criterion_serial_number"])
            assert sn in serials, f"unknown criterion_serial_number {sn}"
            subs = row["must_contain_substrings"]
            assert isinstance(subs, list) and subs, "must_contain_substrings must be non-empty list"
            for s in subs:
                assert isinstance(s, str) and s.strip(), f"bad substring entry: {s!r}"

    @pytest.mark.parametrize("row", json.loads(LABELS_PATH.read_text(encoding="utf-8"))["labels"])
    def test_substrings_present_when_pdf_exists(self, row):
        manifest = _load_manifest_docs()
        entry = manifest[row["document_id"]]
        path = ROOT / entry["path"]
        if not path.exists():
            pytest.skip(f"PDF not present: {path}")

        raw = extract_text_from_pdf(path.open("rb"))
        lower = raw.lower()
        for needle in row["must_contain_substrings"]:
            assert needle.lower() in lower, (
                f"{row['document_id']}: expected substring {needle!r} "
                f"for criterion {row['criterion_serial_number']} not found in extract"
            )
