import json
import tempfile
from pathlib import Path

import numpy as np

from neurolens.cache import CacheManager
from neurolens.match import (
    find_similar_stimuli,
    build_target_from_regions,
    find_contrast_stimuli,
)


def _setup(tmp: Path) -> tuple[CacheManager, list[str]]:
    preds_dir = tmp / "brain_preds"
    preds_dir.mkdir()
    roi_dir = tmp / "roi_summaries"
    roi_dir.mkdir()

    ids = []
    for i in range(5):
        sid = f"clip_{i:03d}"
        ids.append(sid)
        preds = np.random.randn(3, 20484)
        np.savez(preds_dir / f"{sid}.npz", preds=preds)
        (roi_dir / f"{sid}.json").write_text(
            json.dumps({"Visual Cortex": float(np.random.rand()),
                         "Auditory Cortex": float(np.random.rand())})
        )
    return CacheManager(tmp), ids


def test_find_similar_stimuli():
    with tempfile.TemporaryDirectory() as tmp:
        cm, ids = _setup(Path(tmp))
        target = cm.load_brain_preds("clip_000")[0]
        results = find_similar_stimuli(cm, target, ids, top_k=3)
        assert len(results) == 3
        assert all(isinstance(r[0], str) for r in results)
        assert all(isinstance(r[1], float) for r in results)
        assert results[0][1] >= results[1][1] >= results[2][1]


def test_find_similar_includes_self():
    with tempfile.TemporaryDirectory() as tmp:
        cm, ids = _setup(Path(tmp))
        # Use mean-averaged prediction as target to match default aggregation
        target = cm.load_brain_preds("clip_000").mean(axis=0)
        results = find_similar_stimuli(cm, target, ids, top_k=5)
        assert results[0][0] == "clip_000"
        assert results[0][1] > 0.99


def test_build_target_from_regions():
    target = build_target_from_regions(
        {"Visual Cortex": 1.0, "Auditory Cortex": 0.0}
    )
    assert target.shape == (20484,)


def test_find_contrast_stimuli():
    with tempfile.TemporaryDirectory() as tmp:
        cm, ids = _setup(Path(tmp))
        results = find_contrast_stimuli(
            cm, ids,
            maximize_roi="Visual Cortex",
            minimize_roi="Auditory Cortex",
            top_k=3,
        )
        assert len(results) == 3
        assert results[0][1] >= results[1][1]
