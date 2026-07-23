import tempfile
from pathlib import Path

import numpy as np
import torch

from neurolens.cache import CacheManager
from neurolens.eval import (
    compute_rsa_score,
    compute_pairwise_similarity_matrix,
    compute_model_brain_alignment,
)


def _setup(tmp: Path) -> tuple[CacheManager, list[str]]:
    preds_dir = tmp / "brain_preds"
    preds_dir.mkdir()

    ids = []
    for i in range(10):
        sid = f"clip_{i:03d}"
        ids.append(sid)
        np.savez(preds_dir / f"{sid}.npz", preds=np.random.randn(3, 20484))

    for model_name in ["vjepa2", "clip", "whisper"]:
        emb_dir = tmp / "embeddings" / model_name
        emb_dir.mkdir(parents=True)
        for sid in ids:
            torch.save(torch.randn(256), emb_dir / f"{sid}.pt")

    return CacheManager(tmp), ids


def test_compute_pairwise_similarity_matrix():
    vecs = [np.random.randn(100) for _ in range(5)]
    mat = compute_pairwise_similarity_matrix(vecs)
    assert mat.shape == (5, 5)
    np.testing.assert_allclose(np.diag(mat), 1.0, atol=1e-6)
    np.testing.assert_allclose(mat, mat.T, atol=1e-6)


def test_compute_rsa_score():
    mat = np.random.randn(5, 5)
    mat = (mat + mat.T) / 2
    score = compute_rsa_score(mat, mat)
    assert abs(score - 1.0) < 1e-6


def test_compute_rsa_score_uncorrelated():
    np.random.seed(42)
    mat_a = np.random.randn(10, 10)
    mat_a = (mat_a + mat_a.T) / 2
    mat_b = np.random.randn(10, 10)
    mat_b = (mat_b + mat_b.T) / 2
    score = compute_rsa_score(mat_a, mat_b)
    assert abs(score) < 0.5


def test_compute_model_brain_alignment():
    with tempfile.TemporaryDirectory() as tmp:
        cm, ids = _setup(Path(tmp))
        score = compute_model_brain_alignment(cm, "vjepa2", ids)
        assert isinstance(score, float)
        assert -1.0 <= score <= 1.0
