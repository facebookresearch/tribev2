"""Predict module: load and slice brain predictions from cache."""

from __future__ import annotations

import numpy as np

from neurolens.cache import CacheManager


def get_prediction_at_time(
    cache: CacheManager,
    stimulus_id: str,
    time_idx: int,
) -> np.ndarray:
    """Return brain prediction at a specific timestep.
    time_idx is clamped to valid range.
    Returns np.ndarray of shape (n_vertices,)
    """
    preds = cache.load_brain_preds(stimulus_id)
    time_idx = min(time_idx, preds.shape[0] - 1)
    time_idx = max(time_idx, 0)
    return preds[time_idx]


def get_num_timesteps(cache: CacheManager, stimulus_id: str) -> int:
    """Return total number of timesteps for a stimulus."""
    preds = cache.load_brain_preds(stimulus_id)
    return preds.shape[0]


def get_top_rois(
    cache: CacheManager,
    stimulus_id: str,
    k: int = 5,
) -> list[tuple[str, float]]:
    """Return top-k ROI groups by mean activation, sorted descending.
    Returns list of (roi_name, mean_value) tuples.
    """
    roi_summary = cache.load_roi_summary(stimulus_id)
    sorted_rois = sorted(roi_summary.items(), key=lambda x: x[1], reverse=True)
    return sorted_rois[:k]


def get_modality_contribution(
    cache: CacheManager,
    stimulus_id: str,
    modality: str,
    time_idx: int,
) -> np.ndarray | None:
    """Return brain prediction for a specific modality at a timestep.
    Per-modality predictions are stored as {stimulus_id}__{modality}.npz.
    Returns None if the modality file doesn't exist.
    """
    mod_id = f"{stimulus_id}__{modality}"
    preds = cache.load_brain_preds(mod_id)
    if preds is None:
        return None
    time_idx = min(time_idx, preds.shape[0] - 1)
    time_idx = max(time_idx, 0)
    return preds[time_idx]
