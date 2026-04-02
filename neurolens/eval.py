"""Eval module: RSA-based comparison of AI model embeddings to brain predictions."""

from __future__ import annotations

import numpy as np
from scipy.stats import spearmanr

from neurolens.cache import CacheManager


def compute_pairwise_similarity_matrix(vectors: list[np.ndarray]) -> np.ndarray:
    """Compute pairwise cosine similarity matrix for a list of vectors.
    Returns np.ndarray of shape (n, n).
    """
    mat = np.stack(vectors)
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1.0, norms)
    mat_normed = mat / norms
    return mat_normed @ mat_normed.T


def compute_rsa_score(
    sim_matrix_a: np.ndarray,
    sim_matrix_b: np.ndarray,
) -> float:
    """Compute RSA score: Spearman correlation between upper triangles.
    Returns float: Spearman correlation coefficient.
    """
    n = sim_matrix_a.shape[0]
    idx = np.triu_indices(n, k=1)
    vec_a = sim_matrix_a[idx]
    vec_b = sim_matrix_b[idx]
    corr, _ = spearmanr(vec_a, vec_b)
    return float(corr)


def compute_model_brain_alignment(
    cache: CacheManager,
    model_name: str,
    stimulus_ids: list[str],
) -> float:
    """Compute overall brain alignment score for a model using RSA.
    Returns float: RSA alignment score in [-1, 1].
    """
    embeddings = []
    brain_vecs = []
    for sid in stimulus_ids:
        emb = cache.load_embedding(sid, model_name)
        preds = cache.load_brain_preds(sid)
        if emb is None or preds is None:
            continue
        embeddings.append(emb.numpy())
        brain_vecs.append(preds.mean(axis=0))

    if len(embeddings) < 3:
        return 0.0

    emb_sim = compute_pairwise_similarity_matrix(embeddings)
    brain_sim = compute_pairwise_similarity_matrix(brain_vecs)
    return compute_rsa_score(emb_sim, brain_sim)


def compute_all_model_alignments(
    cache: CacheManager,
    stimulus_ids: list[str],
) -> dict[str, float]:
    """Compute brain alignment scores for all available models.
    Returns dict mapping model_name to RSA score, sorted descending.
    """
    models = cache.available_models()
    scores = {}
    for model_name in models:
        scores[model_name] = compute_model_brain_alignment(cache, model_name, stimulus_ids)
    return dict(sorted(scores.items(), key=lambda x: x[1], reverse=True))
