# tests/test_integration.py
"""End-to-end integration test with a mock cache."""

import json
import tempfile
from pathlib import Path

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")

from neurolens.cache import CacheManager
from neurolens.stimulus import StimulusLibrary
from neurolens.predict import get_prediction_at_time, get_top_rois
from neurolens.match import find_similar_stimuli, find_contrast_stimuli
from neurolens.eval import compute_all_model_alignments
from neurolens.viz import plot_brain_surface, make_radar_chart


def _build_mock_cache(tmp: Path) -> Path:
    """Build a complete mock cache for integration testing."""
    stimuli = [
        {"id": f"clip_{i:03d}", "name": f"Clip {i}", "category": cat,
         "media_type": "video", "duration_sec": 10.0}
        for i, cat in enumerate(["Speech", "Music", "Silence + Visuals",
                                  "Emotional", "Multimodal-rich"])
    ]
    (tmp / "stimuli").mkdir()
    (tmp / "stimuli" / "metadata.json").write_text(json.dumps(stimuli))

    (tmp / "brain_preds").mkdir()
    (tmp / "roi_summaries").mkdir()
    for s in stimuli:
        preds = np.random.randn(5, 20484).astype(np.float32)
        np.savez(tmp / "brain_preds" / f"{s['id']}.npz", preds=preds)
        roi = {"Visual Cortex": float(np.random.rand()),
               "Auditory Cortex": float(np.random.rand()),
               "Language Areas": float(np.random.rand())}
        (tmp / "roi_summaries" / f"{s['id']}.json").write_text(json.dumps(roi))

    for model in ["vjepa2", "clip", "whisper"]:
        (tmp / "embeddings" / model).mkdir(parents=True)
        for s in stimuli:
            torch.save(torch.randn(256), tmp / "embeddings" / model / f"{s['id']}.pt")

    return tmp


def test_full_pipeline():
    with tempfile.TemporaryDirectory() as tmp:
        cache_dir = _build_mock_cache(Path(tmp))
        cache = CacheManager(cache_dir)
        library = StimulusLibrary(cache_dir)

        # 1. Predict
        data = get_prediction_at_time(cache, "clip_000", time_idx=2)
        assert data.shape == (20484,)
        top = get_top_rois(cache, "clip_000", k=3)
        assert len(top) == 3

        # 2. Match
        target = data
        results = find_similar_stimuli(cache, target, library.ids(), top_k=3)
        assert len(results) == 3
        contrast = find_contrast_stimuli(
            cache, library.ids(), "Visual Cortex", "Auditory Cortex", top_k=3
        )
        assert len(contrast) == 3

        # 3. Eval
        scores = compute_all_model_alignments(cache, library.ids())
        assert len(scores) == 3
        assert all(isinstance(v, float) for v in scores.values())

        # 4. Visualization
        fig = plot_brain_surface(data, views=["left"])
        assert fig is not None
        import matplotlib.pyplot as plt
        plt.close(fig)

        radar_data = {"Clip 0": cache.load_roi_summary("clip_000")}
        fig2 = make_radar_chart(radar_data)
        assert fig2 is not None
        plt.close(fig2)
