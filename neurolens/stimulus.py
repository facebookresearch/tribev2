"""Stimulus library: metadata loading and lookup."""

import json
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Stimulus:
    id: str
    name: str
    category: str
    media_type: str  # "video", "audio", or "text"
    duration_sec: float


class StimulusLibrary:
    """Loads and queries stimulus metadata from a cache directory.

    Expects ``<cache_dir>/stimuli/metadata.json`` — a JSON array of objects
    with keys: id, name, category, media_type, duration_sec.
    """

    def __init__(self, cache_dir: str | Path) -> None:
        self.cache_dir = Path(cache_dir)
        meta_path = self.cache_dir / "stimuli" / "metadata.json"
        raw = json.loads(meta_path.read_text())
        self._stimuli = [Stimulus(**item) for item in raw]
        self._by_id = {s.id: s for s in self._stimuli}

    def __len__(self) -> int:
        return len(self._stimuli)

    def get(self, stimulus_id: str) -> Stimulus | None:
        """Return a Stimulus by id, or None if not found."""
        return self._by_id.get(stimulus_id)

    def all(self) -> list[Stimulus]:
        """Return all stimuli."""
        return list(self._stimuli)

    def filter_by_category(self, category: str) -> list[Stimulus]:
        """Return stimuli matching the given category."""
        return [s for s in self._stimuli if s.category == category]

    def categories(self) -> list[str]:
        """Return sorted list of unique categories."""
        return sorted(set(s.category for s in self._stimuli))

    def ids(self) -> list[str]:
        """Return list of all stimulus ids."""
        return [s.id for s in self._stimuli]

    def dropdown_options(self) -> list[tuple[str, str]]:
        """Return (display_label, id) pairs for ipywidgets Dropdown."""
        return [(f"{s.name} [{s.category}]", s.id) for s in self._stimuli]
