"""Command-line interface for docs-clustering-cli."""

import argparse
import sys
from pathlib import Path

from docsclustering.loaders import FILE_TYPE, load_docs, load_docs_json
from docsclustering.report import default_threshold, print_report, write_matrix
from docsclustering.similarity import sim_multiset, sim_setjacc, sim_st, sim_tfidf


def build_parser() -> argparse.ArgumentParser:
    """Construct the CLI argument parser."""
    ap = argparse.ArgumentParser(
        description="Group similar documents by cosine similarity."
    )
    group = ap.add_mutually_exclusive_group()
    group.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data"),
        help="Directory to scan for " + FILE_TYPE + " files (default: %(default)s)",
    )
    group.add_argument(
        "--data-json",
        type=Path,
        default=None,
        help="JSON file mapping document IDs to text; exclusive with --data-dir",
    )
    ap.add_argument(
        "--method",
        choices=["tfidf", "st", "setjacc", "multiset"],
        default="tfidf",
        help="Similarity method: tfidf, st (sentence-transformers), setjacc (binary token Jaccard), or multiset (count-aware Jaccard) (default: %(default)s)",
    )
    ap.add_argument(
        "--model",
        default="sentence-transformers/all-MiniLM-L6-v2",
        help="Sentence-transformer model name, st only (default: %(default)s)",
    )
    ap.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="Minimum similarity for clustering (default: 0.6 for st, 0.3 for tfidf)",
    )
    ap.add_argument(
        "--top-k",
        type=int,
        default=None,
        help="Limit per-file ranking rows (default: all)",
    )
    ap.add_argument(
        "--out",
        type=Path,
        default=Path("similarity_matrix.csv"),
        help="CSV output path (default: %(default)s)",
    )
    return ap


def main(argv=None) -> None:
    """Run the clustering pipeline end to end."""
    a = build_parser().parse_args(argv)

    if a.data_json is not None:
        names, texts = load_docs_json(a.data_json)
    else:
        names, texts = load_docs(a.data_dir)
    if not names:
        if a.data_json is not None:
            sys.exit(f"No documents in {a.data_json}")
        sys.exit(f"No {FILE_TYPE} files in {a.data_dir}")

    if a.method == "tfidf":
        S = sim_tfidf(texts)
    elif a.method == "multiset":
        S = sim_multiset(texts)
    elif a.method == "setjacc":
        S = sim_setjacc(texts)
    else:
        S = sim_st(texts, a.model)
    thr = a.threshold if a.threshold is not None else default_threshold(a.method)

    write_matrix(a.out, names, S)
    model_label = a.model if a.method == "st" else "-"
    print_report(
        names, S, method=a.method, model_label=model_label, threshold=thr, top_k=a.top_k
    )


if __name__ == "__main__":
    main()
