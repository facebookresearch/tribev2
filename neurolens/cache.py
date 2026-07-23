"""CacheManager: load pre-computed brain predictions, ROI summaries, and embeddings."""

import json
from pathlib import Path

import numpy as np
import torch


class CacheManager:
    """Loads cached data from the NeuroLens cache directory.

    Expected layout::

        cache_dir/
        ├── brain_preds/{stimulus_id}.npz    (key: "preds")
        ├── roi_summaries/{stimulus_id}.json
        └── embeddings/{model_name}/{stimulus_id}.pt
    """

    def __init__(self, cache_dir: str | Path) -> None:
        self.cache_dir = Path(cache_dir)

    def load_brain_preds(self, stimulus_id: str) -> np.ndarray | None:
        """Load brain predictions array of shape (n_timesteps, n_vertices).

        Returns None if the file doesn't exist.
        """
        path = self.cache_dir / "brain_preds" / f"{stimulus_id}.npz"
        if not path.exists():
            return None
        return np.load(path)["preds"]

    def load_roi_summary(self, stimulus_id: str) -> dict[str, float] | None:
        """Load per-ROI-group mean activations.

        Returns None if the file doesn't exist.
        """
        path = self.cache_dir / "roi_summaries" / f"{stimulus_id}.json"
        if not path.exists():
            return None
        return json.loads(path.read_text())

    def load_embedding(self, stimulus_id: str, model_name: str) -> torch.Tensor | None:
        """Load a model embedding tensor.

        Returns None if the file doesn't exist.
        """
        path = self.cache_dir / "embeddings" / model_name / f"{stimulus_id}.pt"
        if not path.exists():
            return None
        return torch.load(path, map_location="cpu", weights_only=True)

    def available_models(self) -> list[str]:
        """Return sorted list of model names that have cached embeddings."""
        emb_dir = self.cache_dir / "embeddings"
        if not emb_dir.exists():
            return []
        return sorted(d.name for d in emb_dir.iterdir() if d.is_dir())

    def all_brain_pred_ids(self) -> list[str]:
        """Return stimulus ids that have cached brain predictions."""
        preds_dir = self.cache_dir / "brain_preds"
        if not preds_dir.exists():
            return []
        return sorted(p.stem for p in preds_dir.glob("*.npz"))
