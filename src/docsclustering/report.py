"""Output: structured JSON report file plus console report."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Sequence

from docsclustering.clustering import clusters


def collect_report(
    names: Sequence[str],
    S,
    *,
    method: str,
    model_label: str,
    threshold: float,
    top_k: int | None = None,
) -> dict:
    """Build the structured report: rankings, ranked pairs, and clusters.

    Returns a JSON-serializable dict consumed by both the console report and
    the optional ``--out`` JSON file, so other tools can consume the exact
    same data that is printed to stdout.
    """
    rankings = {}
    for i, name in enumerate(names):
        order = sorted((j for j in range(len(names)) if j != i), key=lambda j: -S[i][j])
        if top_k:
            order = order[:top_k]
        rankings[name] = [[names[j], round(float(S[i][j]), 4)] for j in order]

    pairs = [
        [names[i], names[j], round(float(S[i][j]), 4)]
        for i in range(len(names))
        for j in range(i + 1, len(names))
    ]
    pairs.sort(key=lambda p: p[2], reverse=True)

    return {
        "method": method,
        "model": model_label,
        "threshold": round(float(threshold), 4),
        "rankings": rankings,
        "pairs": pairs,
        "clusters": clusters(names, S, threshold),
    }


def write_json_report(path: Path, data: dict) -> None:
    """Dump structured report data to ``path`` as indented UTF-8 JSON."""
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
        f.write("\n")


def print_report(data: dict) -> None:
    """Print the human-readable report to stdout (no file side effects)."""
    print(
        f"# method={data['method']} model={data['model']} threshold={data['threshold']}"
    )

    print("\n## Per-file rankings (most similar first)")
    for name, row in data["rankings"].items():
        print(f"\n{name}:")
        for other, sim in row:
            print(f"  {sim:.4f}  {other}")

    print("\n## Ranked pairs")
    for a, b, sim in data["pairs"]:
        print(f"  {sim:.4f}  {a} <-> {b}")

    print(f"\n## Clusters (sim >= {data['threshold']})")
    for g in data["clusters"]:
        print("  " + ", ".join(g))


def default_threshold(method: str) -> float:
    """Return the fallback similarity threshold for a method."""
    return 0.3 if method in ("tfidf", "multiset", "setjacc") else 0.6
