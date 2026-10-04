import pandas as pd

from tribev2.eventstransforms import ExtractWordsFromAudio


def test_each_chunk_gets_only_its_own_words(tmp_path):
    # A cached transcript next to the WAV file, so WhisperX is never run.
    wav = tmp_path / "audio.wav"
    pd.DataFrame(
        {
            "text": ["a", "b", "c"],
            "start": [1.0, 61.0, 110.0],
            "duration": [0.3] * 3,
            "sequence_id": [0, 1, 2],
            "sentence": ["a", "b", "c"],
        }
    ).to_csv(wav.with_suffix(".tsv"), sep="\t", index=False)
    # A 119 s clip after ChunkEvents: two audio events sharing one file.
    events = pd.DataFrame(
        {
            "type": ["Audio", "Audio"],
            "filepath": [str(wav)] * 2,
            "start": [0.0, 60.0],
            "offset": [0.0, 60.0],
            "duration": [60.0, 59.0],
            "frequency": [16000, 16000],
            "timeline": ["clip", "clip"],
        }
    )
    out = ExtractWordsFromAudio()._run(events)
    words = out[out.type == "Word"].sort_values("start")
    assert words.text.tolist() == ["a", "b", "c"]  # each word once
    assert words.start.tolist() == [1.0, 61.0, 110.0]  # at its true time
