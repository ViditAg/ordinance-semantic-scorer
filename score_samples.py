from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from app.analysis.embeddings import EmbeddingProvider
from app.analysis.scorer import OrdinanceScorer
from app.utils.pdf_parser import extract_text_from_pdf
from app.utils.text_splitter import chunk_text


def _load_text(path: Path) -> str:
    if path.suffix.lower() == ".pdf":
        with path.open("rb") as fh:
            return extract_text_from_pdf(fh)
    if path.suffix.lower() in {".txt", ".md"}:
        return path.read_text(encoding="utf-8", errors="replace")
    raise ValueError(f"Unsupported file type: {path.suffix}")


def _load_criteria(criteria_path: Path) -> list[dict[str, Any]]:
    data = json.loads(criteria_path.read_text(encoding="utf-8"))
    if isinstance(data, dict) and "criteria" in data:
        return data["criteria"]
    raise ValueError("criteria file must be a JSON object with a top-level 'criteria' list")


def main() -> int:
    parser = argparse.ArgumentParser(description="Score all sample ordinances locally (Sentence-Transformers only).")
    parser.add_argument("--sample-dir", default="sample", help="Directory containing sample files")
    parser.add_argument("--criteria", default="app/data/criteria.json", help="Path to criteria JSON")
    parser.add_argument("--model", default="all-MiniLM-L6-v2", help="Sentence-Transformers model name")
    parser.add_argument("--chunk-size", type=int, default=2000)
    parser.add_argument("--chunk-overlap", type=int, default=200)
    parser.add_argument("--top-k", type=int, default=1)
    parser.add_argument("--out-dir", default="out", help="Write per-document JSON reports here")
    args = parser.parse_args()

    sample_dir = Path(args.sample_dir)
    criteria_path = Path(args.criteria)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    criteria = _load_criteria(criteria_path)
    scorer = OrdinanceScorer(criteria=criteria)
    provider = EmbeddingProvider(model_name=args.model)

    crit_texts = [c.get("description", "") for c in criteria]
    crit_embeddings = provider.embed_texts(crit_texts)

    files = sorted(
        [
            p
            for p in sample_dir.rglob("*")
            if p.is_file() and p.suffix.lower() in {".pdf", ".txt", ".md"}
        ]
    )
    if not files:
        raise SystemExit(f"No supported sample files found under: {sample_dir}")

    for path in files:
        raw_text = _load_text(path)
        if not raw_text.strip():
            print(f"[skip] {path} (no text)")
            continue

        chunks = chunk_text(raw_text, chunk_size=args.chunk_size, overlap=args.chunk_overlap)
        doc_embeddings = provider.embed_texts(chunks)

        results = scorer.score(
            doc_chunks=chunks,
            doc_embeddings=doc_embeddings,
            crit_embeddings=crit_embeddings,
            top_k=args.top_k,
        )

        report = {
            "meta": {
                "file": str(path),
                "model": args.model,
                "num_chunks": len(chunks),
                "overall_score": results["overall_score"],
            },
            "criteria_results": results["criteria_results"],
        }
        out_path = out_dir / f"{path.stem}_score.json"
        out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"{path.name}: {results['overall_score']:.2f} / 100  ->  {out_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
