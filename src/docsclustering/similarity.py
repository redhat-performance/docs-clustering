"""Cosine similarity matrices between documents."""

from collections import Counter
from typing import List

import numpy as np


def sim_st(texts: List[str], model_name: str):
    """Return a cosine similarity matrix from sentence-transformer embeddings.

    Dependencies are imported lazily: sentence-transformers (and its torch
    stack) is heavy and only needed when this method is actually chosen.
    """
    from sentence_transformers import SentenceTransformer, util

    model = SentenceTransformer(model_name)
    emb = model.encode(
        texts,
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=False,
    )
    return util.cos_sim(emb, emb).numpy()


def sim_tfidf(texts: List[str]):
    """Return a TF-IDF cosine similarity matrix with sublinear term weighting.

    sklearn is imported lazily so the script can run even where the full ML
    stack is unavailable.
    """
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity

    X = TfidfVectorizer(sublinear_tf=True, ngram_range=(1, 2), min_df=1).fit_transform(
        texts
    )
    return cosine_similarity(X)


def sim_multiset(texts: List[str]):
    """Return a count-aware multiset Jaccard similarity matrix.

    Token counts matter: a term repeated N times in one doc vs once in
    another lowers their overlap (sum of min counts over sum of max counts), so
    a repetition mismatch between docs counts against similarity.  Works on any
    text; for ordinary prose it behaves like plain Jaccard.
    """
    counters = [Counter(t.split()) for t in texts]
    n = len(counters)
    S = np.eye(n)
    for i in range(n):
        for j in range(i + 1, n):
            ci, cj = counters[i], counters[j]
            num = 0.0
            den = 0.0
            for t in set(ci) | set(cj):
                a, b = ci.get(t, 0), cj.get(t, 0)
                num += min(a, b)
                den += max(a, b)
            S[i, j] = S[j, i] = num / den if den else 0.0
    return S
