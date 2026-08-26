"""Output: similarity matrix file plus console report."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

from docsclustering.clustering import clusters


def write_matrix(path: Path, names: Sequence[str], S) -> None:
    """Write a CSV similarity matrix with a leading header row of names."""
    with path.open("w") as f:
        f.write("," + ",".join(names) + "\n")
        for i, row in enumerate(S):
            f.write(names[i] + "," + ",".join(f"{v:.4f}" for v in row) + "\n")


def print_report(
    names: Sequence[str],
    S,
    *,
    method: str,
    model_label: str,
    threshold: float,
    top_k: int | None,
) -> None:
    """Print ranked neighbors, ranked pairs, and clusters to stdout."""
    print(f"# method={method} model={model_label} threshold={threshold}")

    print("\n## Per-file rankings (most similar first)")
    for i, name in enumerate(names):
        order = sorted((j for j in range(len(names)) if j != i), key=lambda j: -S[i][j])
        if top_k:
            order = order[:top_k]
        print(f"\n{name}:")
        for j in order:
            print(f"  {S[i][j]:.4f}  {names[j]}")

    print("\n## Ranked pairs")
    pairs = sorted(
        (
            (S[i][j], names[i], names[j])
            for i in range(len(names))
            for j in range(i + 1, len(names))
        ),
        reverse=True,
    )
    for s, x, y in pairs:
        print(f"  {s:.4f}  {x} <-> {y}")

    print(f"\n## Clusters (sim >= {threshold})")
    for g in clusters(names, S, threshold):
        print("  " + ", ".join(g))


def default_threshold(method: str) -> float:
    """Return the fallback similarity threshold for a method."""
    return 0.3 if method in ("tfidf", "multiset", "setjacc") else 0.6
