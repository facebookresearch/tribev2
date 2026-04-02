import json
import tempfile
from pathlib import Path

from neurolens.stimulus import Stimulus, StimulusLibrary


def _make_metadata(tmp: Path) -> Path:
    stimuli = [
        {
            "id": "clip_001",
            "name": "Nature timelapse",
            "category": "Silence + Visuals",
            "media_type": "video",
            "duration_sec": 10.0,
        },
        {
            "id": "clip_002",
            "name": "TED talk excerpt",
            "category": "Speech",
            "media_type": "video",
            "duration_sec": 12.0,
        },
        {
            "id": "clip_003",
            "name": "Classical music",
            "category": "Music",
            "media_type": "audio",
            "duration_sec": 15.0,
        },
    ]
    meta_path = tmp / "stimuli" / "metadata.json"
    meta_path.parent.mkdir(parents=True)
    meta_path.write_text(json.dumps(stimuli))
    return tmp


def test_stimulus_dataclass():
    s = Stimulus(
        id="clip_001",
        name="Nature timelapse",
        category="Silence + Visuals",
        media_type="video",
        duration_sec=10.0,
    )
    assert s.id == "clip_001"
    assert s.category == "Silence + Visuals"


def test_library_load():
    with tempfile.TemporaryDirectory() as tmp:
        cache_dir = _make_metadata(Path(tmp))
        lib = StimulusLibrary(cache_dir)
        assert len(lib) == 3
        assert lib.get("clip_001").name == "Nature timelapse"


def test_library_filter_by_category():
    with tempfile.TemporaryDirectory() as tmp:
        cache_dir = _make_metadata(Path(tmp))
        lib = StimulusLibrary(cache_dir)
        music = lib.filter_by_category("Music")
        assert len(music) == 1
        assert music[0].id == "clip_003"


def test_library_categories():
    with tempfile.TemporaryDirectory() as tmp:
        cache_dir = _make_metadata(Path(tmp))
        lib = StimulusLibrary(cache_dir)
        cats = lib.categories()
        assert set(cats) == {"Silence + Visuals", "Speech", "Music"}


def test_library_get_missing_returns_none():
    with tempfile.TemporaryDirectory() as tmp:
        cache_dir = _make_metadata(Path(tmp))
        lib = StimulusLibrary(cache_dir)
        assert lib.get("nonexistent") is None
