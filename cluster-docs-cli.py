#!/usr/bin/env python3
"""Group similar documents by embedding or TF-IDF cosine similarity."""
import argparse
import re
import sys
from pathlib import Path

FILE_TYPE = "*.log"


def normalize(text: str) -> str:
    lines = text.splitlines()
    if lines and lines[0].lstrip().startswith("[") and "investigate.py" in lines[0]:
        lines = lines[1:]
    text = "\n".join(lines)
    text = re.sub(r"https?://\S+", " ", text)
    text = re.sub(
        r"\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}:\d{2}[.,]?\d*(?:[+-]\d{2}:?\d{2})?", " ", text
    )
    text = re.sub(r"[IEWF]\d{4} \d{2}:\d{2}:\d{2}\.\d+", " ", text)
    text = re.sub(r"\b[0-9a-f]{7,}\b", " ", text)
    text = re.sub(r"\b\d{5,}\b", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def load_docs(data_dir: Path):
    paths = sorted(data_dir.glob(FILE_TYPE))
    names = [p.name for p in paths]
    texts = [normalize(p.read_text(encoding="utf-8", errors="replace")) for p in paths]
    return names, texts


def sim_st(texts, model_name):
    from sentence_transformers import SentenceTransformer, util

    model = SentenceTransformer(model_name)
    emb = model.encode(
        texts,
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=False,
    )
    return util.cos_sim(emb, emb).numpy()


def sim_tfidf(texts):
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity

    X = TfidfVectorizer(sublinear_tf=True, ngram_range=(1, 2), min_df=1).fit_transform(texts)
    return cosine_similarity(X)


def clusters(names, S, threshold):
    n = len(names)
    parent = list(range(n))

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    for i in range(n):
        for j in range(i + 1, n):
            if S[i][j] >= threshold:
                parent[find(i)] = find(j)
    groups = {}
    for i in range(n):
        groups.setdefault(find(i), []).append(names[i])
    return [sorted(g) for g in groups.values()]


def main():
    ap = argparse.ArgumentParser(description="Group similar documents by cosine similarity.")
    ap.add_argument("--data-dir", type=Path, default=Path("data"))
    ap.add_argument("--method", choices=["st", "tfidf"], default="st")
    ap.add_argument("--model", default="sentence-transformers/all-MiniLM-L6-v2")
    ap.add_argument("--threshold", type=float, default=None)
    ap.add_argument("--top-k", type=int, default=None)
    ap.add_argument("--out", type=Path, default=Path("similarity_matrix.csv"))
    a = ap.parse_args()

    names, texts = load_docs(a.data_dir)
    if not names:
        sys.exit(f"No {FILE_TYPE} files in {a.data_dir}")

    S = sim_tfidf(texts) if a.method == "tfidf" else sim_st(texts, a.model)
    thr = a.threshold if a.threshold is not None else (0.3 if a.method == "tfidf" else 0.6)

    with a.out.open("w") as f:
        f.write("," + ",".join(names) + "\n")
        for i, row in enumerate(S):
            f.write(names[i] + "," + ",".join(f"{v:.4f}" for v in row) + "\n")

    model_label = a.model if a.method == "st" else "-"
    print(f"# method={a.method} model={model_label} threshold={thr}")

    print("\n## Per-file rankings (most similar first)")
    for i, name in enumerate(names):
        order = sorted((j for j in range(len(names)) if j != i), key=lambda j: -S[i][j])
        if a.top_k:
            order = order[: a.top_k]
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

    print(f"\n## Clusters (sim >= {thr})")
    for g in clusters(names, S, thr):
        print("  " + ", ".join(g))


if __name__ == "__main__":
    main()
