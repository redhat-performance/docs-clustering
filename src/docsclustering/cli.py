"""Command-line interface for docs-clustering-cli."""

import argparse
import sys
from pathlib import Path

from docsclustering.loaders import FILE_TYPE, load_docs
from docsclustering.report import default_threshold, print_report, write_matrix
from docsclustering.similarity import sim_st, sim_tfidf


def build_parser() -> argparse.ArgumentParser:
    """Construct the CLI argument parser."""
    ap = argparse.ArgumentParser(
        description="Group similar documents by cosine similarity."
    )
    ap.add_argument("--data-dir", type=Path, default=Path("data"))
    ap.add_argument("--method", choices=["st", "tfidf"], default="st")
    ap.add_argument("--model", default="sentence-transformers/all-MiniLM-L6-v2")
    ap.add_argument("--threshold", type=float, default=None)
    ap.add_argument("--top-k", type=int, default=None)
    ap.add_argument("--out", type=Path, default=Path("similarity_matrix.csv"))
    return ap


def main(argv=None) -> None:
    """Run the clustering pipeline end to end."""
    a = build_parser().parse_args(argv)

    names, texts = load_docs(a.data_dir)
    if not names:
        sys.exit(f"No {FILE_TYPE} files in {a.data_dir}")

    S = sim_tfidf(texts) if a.method == "tfidf" else sim_st(texts, a.model)
    thr = a.threshold if a.threshold is not None else default_threshold(a.method)

    write_matrix(a.out, names, S)
    model_label = a.model if a.method == "st" else "-"
    print_report(
        names, S, method=a.method, model_label=model_label, threshold=thr, top_k=a.top_k
    )


if __name__ == "__main__":
    main()
