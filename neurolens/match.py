"""Match module: find stimuli matching target brain activation patterns."""

from __future__ import annotations

import numpy as np

from neurolens.cache import CacheManager
from neurolens.roi import ROI_GROUPS


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Compute cosine similarity between two 1D arrays."""
    dot = np.dot(a, b)
    norm = np.linalg.norm(a) * np.linalg.norm(b)
    if norm == 0:
        return 0.0
    return float(dot / norm)


def find_similar_stimuli(
    cache: CacheManager,
    target: np.ndarray,
    stimulus_ids: list[str],
    top_k: int = 5,
    time_aggregation: str = "mean",
) -> list[tuple[str, float]]:
    """Find stimuli whose brain predictions are most similar to a target pattern.

    Parameters
    ----------
    time_aggregation : str
        How to reduce the time dimension of cached predictions before comparing.
        ``"first"`` uses the first timestep (default), ``"mean"`` averages all timesteps.

    Returns
    -------
    list of (stimulus_id, similarity_score) sorted descending.
    """
    scores = []
    for sid in stimulus_ids:
        preds = cache.load_brain_preds(sid)
        if preds is None:
            continue
        if time_aggregation == "mean":
            avg_pred = preds.mean(axis=0)
        else:
            avg_pred = preds[0]
        sim = _cosine_similarity(target, avg_pred)
        scores.append((sid, sim))
    scores.sort(key=lambda x: x[1], reverse=True)
    return scores[:top_k]


def build_target_from_regions(
    region_intensities: dict[str, float],
    mesh: str = "fsaverage5",
    n_vertices: int = 20484,
) -> np.ndarray:
    """Build a synthetic target activation vector from ROI group intensities.
    Returns np.ndarray of shape (n_vertices,).
    """
    from tribev2.utils import get_hcp_roi_indices

    target = np.zeros(n_vertices)
    for group_name, intensity in region_intensities.items():
        if group_name not in ROI_GROUPS:
            continue
        for region in ROI_GROUPS[group_name]:
            try:
                indices = get_hcp_roi_indices(region, hemi="both", mesh=mesh)
                target[indices] = intensity
            except ValueError:
                continue
    return target


def find_contrast_stimuli(
    cache: CacheManager,
    stimulus_ids: list[str],
    maximize_roi: str,
    minimize_roi: str,
    top_k: int = 5,
) -> list[tuple[str, float]]:
    """Find stimuli that maximize one ROI while minimizing another.
    Returns list of (stimulus_id, contrast_score) sorted descending.
    """
    scores = []
    for sid in stimulus_ids:
        roi_summary = cache.load_roi_summary(sid)
        if roi_summary is None:
            continue
        max_val = roi_summary.get(maximize_roi, 0.0)
        min_val = roi_summary.get(minimize_roi, 0.0)
        contrast = max_val - min_val
        scores.append((sid, contrast))
    scores.sort(key=lambda x: x[1], reverse=True)
    return scores[:top_k]
