"""
TRIBE v2 model wrapper — ZeroGPU compatible.

Three patches required for ZeroGPU environment:

1. LAZY IMPORT: x_transformers initializes CUDA at class-definition time.
   All tribev2 imports must happen inside @spaces.GPU decorated functions.

2. DATALOADER num_workers=0: ZeroGPU runs @spaces.GPU functions inside
   daemon processes. Daemon processes CANNOT spawn children. PyTorch's
   DataLoader with num_workers>0 spawns worker processes → AssertionError.
   Fix: monkey-patch DataLoader to force num_workers=0 (single-process
   loading, identical results, no child processes).

3. WHISPER IN-PROCESS: tribev2 calls `uvx whisperx` as a subprocess for
   word-level transcription. uvx is unavailable, and subprocesses would
   also hit the daemon restriction. Fix: use faster-whisper directly in
   Python (same Whisper model, proper word timestamps, no subprocess).
"""
import os
import tempfile
import logging
import numpy as np
from pathlib import Path
from typing import Optional

try:
    import spaces
    _ZEROGPU = True
except ImportError:
    _ZEROGPU = False
    class spaces:
        @staticmethod
        def GPU(fn=None, duration=60, size="large"):
            if fn is not None:
                return fn
            def decorator(f):
                return f
            return decorator

logger = logging.getLogger(__name__)

_model = None
_patched = False
_whisper_model = None


def _patch_dataloader():
    """Force all DataLoaders to use num_workers=0 (no child processes)."""
    import torch.utils.data
    _orig_init = torch.utils.data.DataLoader.__init__

    def _patched_init(self, *args, **kwargs):
        kwargs["num_workers"] = 0
        kwargs.pop("prefetch_factor", None)
        kwargs["persistent_workers"] = False
        _orig_init(self, *args, **kwargs)

    torch.utils.data.DataLoader.__init__ = _patched_init
    logger.info("Patched DataLoader: num_workers=0 (ZeroGPU daemon-safe)")


def _patch_whisperx():
    """Replace uvx whisperx subprocess with in-process faster-whisper."""
    import pandas as pd
    from tribev2.eventstransforms import ExtractWordsFromAudio

    def _faster_whisper_transcript(wav_filename: Path, language: str) -> pd.DataFrame:
        global _whisper_model
        import torch
        from faster_whisper import WhisperModel

        wav_filename = Path(wav_filename)

        if _whisper_model is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
            compute_type = "float16" if device == "cuda" else "int8"
            _whisper_model = WhisperModel("base", device=device, compute_type=compute_type)
            logger.info("Loaded faster-whisper 'base' model on %s (%s)", device, compute_type)

        lang_codes = {"english": "en", "french": "fr", "spanish": "es", "dutch": "nl", "chinese": "zh"}
        lang = lang_codes.get(language, "en")

        segments, info = _whisper_model.transcribe(
            str(wav_filename),
            word_timestamps=True,
            language=lang,
        )

        words = []
        for seg_idx, segment in enumerate(segments):
            sentence = segment.text.strip().replace('"', "")
            if segment.words:
                for w in segment.words:
                    words.append({
                        "text": w.word.strip().replace('"', ""),
                        "start": w.start,
                        "duration": w.end - w.start,
                        "sequence_id": seg_idx,
                        "sentence": sentence,
                    })

        logger.info("Whisper transcribed %d words from %.1fs audio", len(words), info.duration)
        if words:
            return pd.DataFrame(words)
        return pd.DataFrame(columns=["text", "start", "duration", "sequence_id", "sentence"])

    ExtractWordsFromAudio._get_transcript_from_audio = staticmethod(_faster_whisper_transcript)
    logger.info("Patched WhisperX: using faster-whisper in-process (real Whisper, no uvx)")


def _apply_patches():
    """Apply all ZeroGPU patches once."""
    global _patched
    if _patched:
        return
    _patch_dataloader()
    _patch_whisperx()
    _patched = True


def _get_or_load_model():
    """Load model inside GPU context. Called only within @spaces.GPU functions."""
    global _model
    if _model is not None:
        return _model

    _apply_patches()

    logger.info("Loading TRIBE v2 from facebook/tribev2 ...")
    from tribev2 import TribeModel

    cache_dir = Path(os.environ.get("TRIBE_CACHE", "./cache"))
    cache_dir.mkdir(parents=True, exist_ok=True)

    _model = TribeModel.from_pretrained(
        "facebook/tribev2",
        cache_folder=str(cache_dir),
    )
    logger.info("TRIBE v2 loaded successfully.")
    return _model


def is_ready() -> bool:
    return True


def get_error() -> Optional[str]:
    return None


def warm_up():
    logger.info("ZeroGPU mode: TRIBE v2 will load on first inference request.")


@spaces.GPU(duration=120)
def predict_text(text: str, max_chars: int = 3000) -> np.ndarray:
    """
    Run TRIBE v2 on text → spatial activation map (n_vertices,).
    GPU is held for the duration of this call only.
    """
    model = _get_or_load_model()
    text = text[:max_chars].strip()
    if not text:
        raise ValueError("Empty text input.")

    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".txt", delete=False, encoding="utf-8"
    ) as f:
        f.write(text)
        tmp_path = f.name
    try:
        df = model.get_events_dataframe(text_path=tmp_path)
        preds, _ = model.predict(events=df)
        return preds.mean(axis=0)
    finally:
        os.unlink(tmp_path)


@spaces.GPU(duration=240)
def predict_texts_batch(texts: list[str], max_chars: int = 500) -> list[np.ndarray]:
    """
    Score a batch of texts in a single GPU allocation.
    Used by CTR predictor — all N titles share one H200 grab.
    """
    model = _get_or_load_model()
    return _predict_texts_with_model(model, texts=texts, max_chars=max_chars)


@spaces.GPU(duration=90)
def predict_texts_quick(texts: list[str], max_chars: int = 500) -> list[np.ndarray]:
    """
    Short-budget TRIBE batch scoring for mini tools.
    """
    model = _get_or_load_model()
    return _predict_texts_with_model(model, texts=texts, max_chars=max_chars)


def _predict_texts_with_model(model, texts: list[str], max_chars: int = 500) -> list[np.ndarray]:
    results = []
    for text in texts:
        text = text[:max_chars].strip()
        if not text:
            results.append(None)
            continue
        try:
            with tempfile.NamedTemporaryFile(
                mode="w", suffix=".txt", delete=False, encoding="utf-8"
            ) as f:
                f.write(text)
                tmp_path = f.name
            try:
                df = model.get_events_dataframe(text_path=tmp_path)
                preds, _ = model.predict(events=df)
                results.append(preds.mean(axis=0))
            finally:
                os.unlink(tmp_path)
        except Exception as exc:
            logger.warning("Batch item failed '%s...': %s", text[:40], exc)
            results.append(None)
    return results


@spaces.GPU(duration=240)
def predict_images_batch(
    image_paths: list[str],
    image_duration_sec: float = 4.0,
    fps: int = 8,
) -> list[np.ndarray]:
    """
    Score a batch of static images using TRIBE v2's official video pathway.

    TRIBE v2 public API accepts text/audio/video paths. For still screenshots,
    we convert each image into a short silent mp4 and infer via video_path.
    """
    model = _get_or_load_model()
    return _predict_images_with_model(
        model,
        image_paths=image_paths,
        image_duration_sec=image_duration_sec,
        fps=fps,
    )


@spaces.GPU(duration=120)
def predict_images_quick(
    image_paths: list[str],
    image_duration_sec: float = 1.5,
    fps: int = 6,
) -> list[np.ndarray]:
    """
    Short-budget image scoring for mini tools.
    """
    model = _get_or_load_model()
    return _predict_images_with_model(
        model,
        image_paths=image_paths,
        image_duration_sec=image_duration_sec,
        fps=fps,
    )


def _predict_images_with_model(
    model,
    image_paths: list[str],
    image_duration_sec: float = 4.0,
    fps: int = 8,
) -> list[np.ndarray]:
    import contextlib
    import uuid
    from moviepy import ImageClip

    results = []

    duration = max(1.0, float(image_duration_sec))
    fps = max(2, int(fps))

    for image_path in image_paths:
        if not image_path:
            results.append(None)
            continue

        tmp_video = Path(tempfile.gettempdir()) / f"tribe_img_{uuid.uuid4().hex}.mp4"
        try:
            clip = ImageClip(str(image_path), duration=duration)
            try:
                with open(os.devnull, "w") as devnull, contextlib.redirect_stdout(
                    devnull
                ), contextlib.redirect_stderr(devnull):
                    clip.write_videofile(
                        str(tmp_video),
                        codec="libx264",
                        audio=False,
                        fps=fps,
                    )
            finally:
                clip.close()

            df = model.get_events_dataframe(video_path=str(tmp_video))
            preds, _ = model.predict(events=df)
            results.append(preds.mean(axis=0))
        except Exception as exc:
            logger.warning(
                "Image batch item failed '%s...': %s",
                str(image_path)[:60],
                exc,
            )
            results.append(None)
        finally:
            try:
                tmp_video.unlink(missing_ok=True)
            except Exception:
                pass

    return results
