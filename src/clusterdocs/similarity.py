"""Cosine similarity matrices between documents."""

from typing import List


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

    X = TfidfVectorizer(sublinear_tf=True, ngram_range=(1, 2), min_df=1).fit_transform(texts)
    return cosine_similarity(X)
