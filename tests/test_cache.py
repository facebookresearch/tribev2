import json
import tempfile
from pathlib import Path

import numpy as np
import torch

from neurolens.cache import CacheManager


def _setup_cache(tmp: Path) -> Path:
    """Create a minimal cache structure."""
    # Brain predictions
    preds_dir = tmp / "brain_preds"
    preds_dir.mkdir()
    np.savez(preds_dir / "clip_001.npz", preds=np.random.randn(5, 20484))

    # ROI summaries
    roi_dir = tmp / "roi_summaries"
    roi_dir.mkdir()
    (roi_dir / "clip_001.json").write_text(
        json.dumps({"Visual Cortex": 0.5, "Auditory Cortex": 0.3})
    )

    # Embeddings
    for model_name in ["vjepa2", "clip"]:
        emb_dir = tmp / "embeddings" / model_name
        emb_dir.mkdir(parents=True)
        torch.save(torch.randn(256), emb_dir / "clip_001.pt")

    return tmp


def test_load_brain_preds():
    with tempfile.TemporaryDirectory() as tmp:
        cache_dir = _setup_cache(Path(tmp))
        cm = CacheManager(cache_dir)
        preds = cm.load_brain_preds("clip_001")
        assert preds.shape == (5, 20484)


def test_load_brain_preds_missing():
    with tempfile.TemporaryDirectory() as tmp:
        cache_dir = _setup_cache(Path(tmp))
        cm = CacheManager(cache_dir)
        assert cm.load_brain_preds("nonexistent") is None


def test_load_roi_summary():
    with tempfile.TemporaryDirectory() as tmp:
        cache_dir = _setup_cache(Path(tmp))
        cm = CacheManager(cache_dir)
        roi = cm.load_roi_summary("clip_001")
        assert roi["Visual Cortex"] == 0.5


def test_load_embedding():
    with tempfile.TemporaryDirectory() as tmp:
        cache_dir = _setup_cache(Path(tmp))
        cm = CacheManager(cache_dir)
        emb = cm.load_embedding("clip_001", "vjepa2")
        assert emb.shape == (256,)


def test_load_embedding_missing_model():
    with tempfile.TemporaryDirectory() as tmp:
        cache_dir = _setup_cache(Path(tmp))
        cm = CacheManager(cache_dir)
        assert cm.load_embedding("clip_001", "nonexistent") is None


def test_available_models():
    with tempfile.TemporaryDirectory() as tmp:
        cache_dir = _setup_cache(Path(tmp))
        cm = CacheManager(cache_dir)
        models = cm.available_models()
        assert set(models) == {"vjepa2", "clip"}
