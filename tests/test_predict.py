import json
import tempfile
from pathlib import Path

import numpy as np

from neurolens.cache import CacheManager
from neurolens.predict import get_prediction_at_time, get_top_rois, get_modality_contribution


def _setup(tmp: Path) -> CacheManager:
    preds_dir = tmp / "brain_preds"
    preds_dir.mkdir()
    # 5 timesteps, 20484 vertices
    preds = np.random.randn(5, 20484)
    np.savez(preds_dir / "clip_001.npz", preds=preds)

    # Also save per-modality predictions
    for mod in ["video", "audio", "text", "combined"]:
        np.savez(preds_dir / f"clip_001__{mod}.npz", preds=preds * np.random.rand())

    roi_dir = tmp / "roi_summaries"
    roi_dir.mkdir()
    (roi_dir / "clip_001.json").write_text(
        json.dumps({"Visual Cortex": 0.9, "Auditory Cortex": 0.3, "Language Areas": 0.1})
    )
    return CacheManager(tmp)


def test_get_prediction_at_time():
    with tempfile.TemporaryDirectory() as tmp:
        cm = _setup(Path(tmp))
        data = get_prediction_at_time(cm, "clip_001", time_idx=2)
        assert data.shape == (20484,)


def test_get_prediction_at_time_clamps():
    with tempfile.TemporaryDirectory() as tmp:
        cm = _setup(Path(tmp))
        data = get_prediction_at_time(cm, "clip_001", time_idx=999)
        assert data.shape == (20484,)  # Should clamp to last timestep


def test_get_top_rois():
    with tempfile.TemporaryDirectory() as tmp:
        cm = _setup(Path(tmp))
        top = get_top_rois(cm, "clip_001", k=2)
        assert len(top) == 2
        assert top[0][0] == "Visual Cortex"  # Highest activation first


def test_get_modality_contribution():
    with tempfile.TemporaryDirectory() as tmp:
        cm = _setup(Path(tmp))
        data = get_modality_contribution(cm, "clip_001", modality="video", time_idx=0)
        assert data.shape == (20484,)


def test_get_modality_contribution_missing():
    with tempfile.TemporaryDirectory() as tmp:
        cm = _setup(Path(tmp))
        data = get_modality_contribution(cm, "clip_001", modality="nonexistent", time_idx=0)
        assert data is None
