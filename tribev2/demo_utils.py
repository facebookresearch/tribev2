# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""TribeModel for inference and utilities for building event DataFrames."""

import logging
import typing as tp
from pathlib import Path

import numpy as np
import pandas as pd
import pydantic
import requests
import torch
import yaml
from einops import rearrange
from exca import ConfDict, TaskInfra
from tqdm import tqdm

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
if not logger.handlers:
    _handler = logging.StreamHandler()
    _handler.setFormatter(logging.Formatter("%(levelname)s - %(message)s"))
    logger.addHandler(_handler)
from neuralset.events.transforms import (
    AddContextToWords,
    AddSentenceToWords,
    AddText,
    ChunkEvents,
    ExtractAudioFromVideo,
    RemoveMissing,
)
from neuralset.events.utils import standardize_events

from tribev2.eventstransforms import ExtractWordsFromAudio
from tribev2.main import TribeExperiment

VALID_SUFFIXES: dict[str, set[str]] = {
    "text_path": {".txt"},
    "audio_path": {".wav", ".mp3", ".flac", ".ogg"},
    "video_path": {".mp4", ".avi", ".mkv", ".mov", ".webm"},
}


def download_file(url: str, path: str | Path) -> Path:
    """Download a file from *url* and save it to *path*.

    Raises ``requests.HTTPError`` on non-2xx responses.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with requests.get(url, stream=True, timeout=30) as r:
        r.raise_for_status()
        with open(path, "wb") as f:
            for chunk in r.iter_content(chunk_size=128 * 1024):
                if chunk:
                    f.write(chunk)
    logger.info(f"Downloaded {url} -> {path}")
    return path


def get_audio_and_text_events(
    events: pd.DataFrame, audio_only: bool = False
) -> pd.DataFrame:
    """Run the audio/video-to-text pipeline on an events DataFrame.

    Extracts audio from video, chunks long clips, transcribes words, and
    attaches sentence/context annotations.  Set *audio_only* to ``True``
    to skip the transcription and text stages.
    """
    transforms = [
        ExtractAudioFromVideo(),
        ChunkEvents(event_type_to_chunk="Audio", max_duration=60, min_duration=30),
        ChunkEvents(event_type_to_chunk="Video", max_duration=60, min_duration=30),
    ]
    if not audio_only:
        transforms.extend(
            [
                ExtractWordsFromAudio(),
                AddText(),
                AddSentenceToWords(max_unmatched_ratio=0.05),
                AddContextToWords(
                    sentence_only=False, max_context_len=1024, split_field=""
                ),
                RemoveMissing(),
            ]
        )
    events = standardize_events(events)
    for transform in transforms:
        events = transform(events)
    return standardize_events(events)


class TextToEvents(pydantic.BaseModel):
    """Convert raw text to an events DataFrame via text-to-speech + transcription.

    The text is synthesised to audio with gTTS, then processed through
    :func:`get_audio_and_text_events` to obtain word-level events.
    """

    text: str
    infra: TaskInfra = TaskInfra()

    def model_post_init(self, __context: tp.Any) -> None:
        if self.infra.folder is None:
            raise ValueError("A folder must be specified to save the audio file.")

    @infra.apply()
    def get_events(self) -> pd.DataFrame:
        from gtts import gTTS
        from langdetect import detect

        audio_path = Path(self.infra.uid_folder(create=True)) / "audio.mp3"
        lang = detect(self.text)
        tts = gTTS(self.text, lang=lang)
        tts.save(str(audio_path))
        logger.info(f"Wrote TTS audio to {audio_path}")

        audio_event = {
            "type": "Audio",
            "filepath": str(audio_path),
            "start": 0,
            "timeline": "default",
            "subject": "default",
        }
        return get_audio_and_text_events(pd.DataFrame([audio_event]))


class TribeModel(TribeExperiment):
    """High-level inference wrapper around :class:`TribeExperiment`.

    Provides a simple ``from_pretrained`` / ``predict`` interface for
    generating fMRI-like brain-activity predictions from text, audio,
    or video inputs.

    Typical usage::

        model = TribeModel.from_pretrained("facebook/tribev2")
        events = model.get_events_dataframe(video_path="clip.mp4")
        preds, segments = model.predict(events)
    """

    cache_folder: str = "./cache"
    remove_empty_segments: bool = True

    @classmethod
    def from_pretrained(
        cls,
        checkpoint_dir: str | Path,
        checkpoint_name: str = "best.ckpt",
        cache_folder: str | Path = None,
        cluster: str = None,
        device: str = "auto",
        config_update: dict | None = None,
    ) -> "TribeModel":
        """Load a trained model from a checkpoint directory or HuggingFace Hub repo.

        ``checkpoint_dir`` can be either a local path containing
        ``config.yaml`` and ``<checkpoint_name>``, or a HuggingFace Hub
        repo id (e.g. ``"facebook/tribev2"``).

        Parameters
        ----------
        checkpoint_dir:
            Local directory or HuggingFace Hub repo id that contains
            ``config.yaml`` and the checkpoint file.
        checkpoint_name:
            Filename of the checkpoint inside *checkpoint_dir*.
        cache_folder:
            Directory used to cache extracted features. Created if it
            does not exist.  Defaults to ``"./cache"`` when ``None``.
        cluster:
            Cluster backend forwarded to feature-extractor infra
            (``"auto"`` by default).
        device:
            Torch device. ``"auto"`` selects CUDA > MPS > CPU.
            Note: feature extractors (text, audio, video) always run on CPU
            regardless of the specified device.
        config_update:
            Optional dictionary of config overrides applied after the
            YAML config is loaded.

        Returns
        -------
        TribeModel
            A ready-to-use model instance with weights loaded in eval mode.
        """
        if cache_folder is not None:
            Path(cache_folder).mkdir(parents=True, exist_ok=True)
        if device == "auto":
            if torch.cuda.is_available():
                device = "cuda"
            elif torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"
        checkpoint_dir = Path(checkpoint_dir)
        if checkpoint_dir.exists():
            config_path = checkpoint_dir / "config.yaml"
            ckpt_path = checkpoint_dir / checkpoint_name
        else:
            from huggingface_hub import hf_hub_download

            repo_id = str(checkpoint_dir)
            config_path = hf_hub_download(repo_id, "config.yaml")
            ckpt_path = hf_hub_download(repo_id, checkpoint_name)
        with open(config_path, "r") as f:
            config = ConfDict(yaml.load(f, Loader=yaml.UnsafeLoader))
        if device in ("cpu", "mps"):  # mps not supported by neuralset extractors
            # Override all extractor devices to cpu when cuda is unavailable
            for modality in ["text", "audio"]:
                config[f"data.{modality}_feature.device"] = "cpu"
            config["data.image_feature.image.device"] = "cpu"
            config["data.video_feature.image.device"] = "cpu"
        for modality in ["text", "audio", "video"]:
            config[f"data.{modality}_feature.infra.folder"] = cache_folder
            config[f"data.{modality}_feature.infra.cluster"] = cluster

        for param in [
            "infra.workdir",
            "data.study.infra_timelines",
            "data.neuro.infra",
            "data.image_feature.infra",
        ]:
            config.pop(param)
        config["data.study.path"] = "."
        config["average_subjects"] = True
        config["checkpoint_path"] = str(config["infra.folder"]) + f"/{checkpoint_name}"
        config["cache_folder"] = (
            str(cache_folder) if cache_folder is not None else "./cache"
        )
        if config_update is not None:
            config.update(config_update)
        xp = cls(**config)

        logger.info(f"Loading model from {ckpt_path}")
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True, mmap=True)
        build_args = ckpt["model_build_args"]
        state_dict = {
            k.removeprefix("model."): v for k, v in ckpt["state_dict"].items()
        }
        del ckpt

        model = xp.brain_model_config.build(**build_args)
        model.load_state_dict(state_dict, strict=True, assign=True)
        del state_dict
        model.to(device)
        model.eval()
        xp._model = model
        return xp

    def get_events_dataframe(
        self,
        text_path: str | None = None,
        audio_path: str | None = None,
        video_path: str | None = None,
    ) -> pd.DataFrame:
        """Build an events DataFrame from exactly one input source.

        Parameters
        ----------
        text_path:
            Path to a ``.txt`` file. The text is converted to speech, then
            transcribed back to produce word-level events.
        audio_path:
            Path to an audio file (``.wav``, ``.mp3``, ``.flac``, ``.ogg``).
        video_path:
            Path to a video file (``.mp4``, ``.avi``, ``.mkv``, ``.mov``,
            ``.webm``).

        Returns
        -------
        pd.DataFrame
            Standardised events DataFrame with columns such as ``type``,
            ``filepath``, ``start``, ``duration``, ``timeline``, and
            ``subject``.

        Raises
        ------
        ValueError
            If zero or more than one path is provided, or if the file
            extension does not match the expected suffixes.
        FileNotFoundError
            If the specified file does not exist.
        """
        provided = {
            name: value
            for name, value in [
                ("text_path", text_path),
                ("audio_path", audio_path),
                ("video_path", video_path),
            ]
            if value is not None
        }
        if len(provided) != 1:
            raise ValueError(
                f"Exactly one of text_path, audio_path, video_path must be "
                f"provided, got: {list(provided.keys()) or 'none'}"
            )

        name, value = next(iter(provided.items()))
        path = Path(value)
        suffix = path.suffix.lower()
        if suffix not in VALID_SUFFIXES[name]:
            raise ValueError(
                f"{name} must end with one of {sorted(VALID_SUFFIXES[name])}, "
                f"got '{suffix}'"
            )
        if not path.is_file():
            raise FileNotFoundError(f"{name} does not exist: {path}")

        if text_path is not None:
            text = path.read_text(encoding="utf-8")
            if not text.strip():
                raise ValueError(f"Text file is empty: {path}")
            return TextToEvents(
                text=text,
                infra={"folder": self.cache_folder, "mode": "retry"},
            ).get_events()

        event_type = "Audio" if audio_path is not None else "Video"
        event = {
            "type": event_type,
            "filepath": str(path),
            "start": 0,
            "timeline": "default",
            "subject": "default",
        }
        return get_audio_and_text_events(pd.DataFrame([event]))

    def predict(
        self, events: pd.DataFrame, verbose: bool = True
    ) -> tuple[np.ndarray, list]:
        """Run inference on an events DataFrame and return per-TR predictions.

        Each batch is split into segments of length ``data.TR``.  When
        ``remove_empty_segments`` is ``True`` (the default), segments that
        contain no events are discarded.

        Parameters
        ----------
        events:
            Events DataFrame, typically produced by
            :meth:`get_events_dataframe`.
        verbose:
            If ``True`` (default), display a ``tqdm`` progress bar.

        Returns
        -------
        preds : np.ndarray
            Array of shape ``(n_kept_segments, n_vertices)`` with the
            predicted brain activity.
        all_segments : list
            Corresponding segment objects aligned with *preds*.

        Raises
        ------
        RuntimeError
            If the model has not been loaded via :meth:`from_pretrained`.
        """
        if self._model is None:
            raise RuntimeError(
                "TribeModel must be instantiated via the .from_pretrained method"
            )
        model = self._model
        loader = self.data.get_loaders(events=events, split_to_build="all")["all"]

        preds, all_segments = [], []
        n_samples, n_kept = 0, 0
        with torch.inference_mode():
            for batch in tqdm(loader, disable=not verbose):
                batch = batch.to(model.device)
                batch_segments = []
                for segment in batch.segments:
                    for t in np.arange(0, segment.duration - 1e-2, self.data.TR):
                        batch_segments.append(
                            segment.copy(offset=t, duration=self.data.TR)
                        )
                if self.remove_empty_segments:
                    keep = np.array([len(s.ns_events) > 0 for s in batch_segments])
                else:
                    keep = np.ones(len(batch_segments), dtype=bool)
                n_kept += keep.sum()
                n_samples += len(batch_segments)
                batch_segments = [s for i, s in enumerate(batch_segments) if keep[i]]
                y_pred = model(batch).detach().cpu().numpy()
                y_pred = rearrange(y_pred, "b d t -> (b t) d")[keep]
                preds.append(y_pred)
                all_segments.extend(batch_segments)
        preds = np.concatenate(preds)
        if len(all_segments) != preds.shape[0]:
            raise ValueError(
                f"Number of samples: {preds.shape[0]} != {len(all_segments)}"
            )
        logger.info(
            "Predicted %d / %d segments (%.1f%% kept)",
            n_kept,
            n_samples,
            100.0 * n_kept / max(n_samples, 1),
        )
        return preds, all_segments
