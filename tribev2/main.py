# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Data loading and experiment configuration for inference."""

import gc
import logging
import typing as tp
from pathlib import Path

import neuralset as ns
import numpy as np
import pandas as pd
import pydantic
import torch
from exca import ConfDict, TaskInfra
from neuralset.events.etypes import EventTypesHelper
from neuralset.events.utils import standardize_events
from neuraltrain.models import BaseModelConfig
from neuraltrain.models.common import SubjectLayers
from neuraltrain.utils import BaseExperiment
from torch.utils.data import DataLoader

from .eventstransforms import *  # register custom events transforms in neuralset
from .model import *  # register custom models in neuraltrain
from .utils import (
    MultiStudyLoader,
    set_study_in_average_subject_mode,
)
from .utils_fmri import *  # register TribeSurfaceProjector

# Configure logger
LOGGER = logging.getLogger(__name__)
_handler = logging.StreamHandler()
_formatter = logging.Formatter("[%(asctime)s %(levelname)s] %(message)s", "%H:%M:%S")
_handler.setFormatter(_formatter)
if not LOGGER.handlers:
    LOGGER.addHandler(_handler)
LOGGER.setLevel(logging.INFO)


def _free_extractor_model(extractor: ns.extractors.BaseExtractor) -> None:
    """Delete cached GPU model from an extractor after its features are cached."""
    targets = [extractor]
    if hasattr(extractor, "image"):
        targets.append(extractor.image)
    for target in targets:
        for attr in ("_model",):
            obj = getattr(target, attr, None)
            if isinstance(obj, torch.nn.Module):
                try:
                    delattr(target, attr)
                except Exception:
                    pass
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


class Data(pydantic.BaseModel):
    """Handles configuration and creation of DataLoaders from dataset and extractors."""

    model_config = pydantic.ConfigDict(extra="forbid")

    study: MultiStudyLoader
    # features
    neuro: ns.extractors.BaseExtractor
    text_feature: ns.extractors.BaseExtractor | None = None
    image_feature: ns.extractors.BaseExtractor | None = None
    audio_feature: ns.extractors.BaseExtractor | None = None
    video_feature: ns.extractors.BaseExtractor | None = None
    subject_id: ns.extractors.LabelEncoder = ns.extractors.LabelEncoder(
        event_field="subject", allow_missing=True, aggregation="first"
    )
    frequency: float | None = None
    features_to_use: list[
        tp.Literal["text", "audio", "video", "image", "context", "flow", "music"]
    ]
    features_to_mask: list[
        tp.Literal["text", "audio", "video", "image", "context", "flow", "music"]
    ] = []
    n_layers_to_use: int | None = None
    layers_to_use: list[float] | None = None
    layer_aggregation: tp.Literal["group_mean", "mean"] | None = "group_mean"
    # Dataset
    duration_trs: int = 40
    overlap_trs_train: int = 0
    overlap_trs_val: int | None = None
    batch_size: int = 64
    num_workers: int | None = None
    shuffle_train: bool = True
    shuffle_val: bool = False
    stride_drop_incomplete: bool = False
    split_segments_by_time: bool = False

    def model_post_init(self, __context):
        super().model_post_init(__context)
        layers_to_use = None
        if self.n_layers_to_use is not None or self.layers_to_use is not None:
            assert not (
                self.n_layers_to_use is not None and self.layers_to_use is not None
            ), "Only one of n_layers_to_use or layers_to_use can be specified"
            if self.n_layers_to_use is not None:
                layers_to_use = np.linspace(0, 1, self.n_layers_to_use).tolist()
            else:
                layers_to_use = self.layers_to_use
        for modality in self.features_to_use:
            extractor = getattr(self, f"{modality}_feature")
            if hasattr(extractor, "layers"):
                setattr(extractor, "layer_aggregation", self.layer_aggregation)
                if layers_to_use is not None:
                    setattr(extractor, "layers", layers_to_use)
            if hasattr(extractor, "image") and hasattr(extractor.image, "layers"):
                setattr(extractor.image, "layer_aggregation", self.layer_aggregation)
                if layers_to_use is not None:
                    setattr(extractor.image, "layers", layers_to_use)
        if self.frequency is not None:
            for modality in self.features_to_use:
                extractor = getattr(self, f"{modality}_feature")
                if hasattr(extractor, "frequency"):
                    setattr(extractor, "frequency", self.frequency)

    @property
    def TR(self) -> float:
        return 1 / self.neuro.frequency

    def get_events(self) -> pd.DataFrame:
        events = self.study.run()
        events = events[events.type != "Sentence"]

        cols = ["index", "subject", "timeline"]
        event_summary = (
            events.reset_index().groupby(["study", "split", "type"])[cols].nunique()
        )
        LOGGER.info("Event summary: \n%s", event_summary)
        return events

    def get_loaders(
        self,
        events: pd.DataFrame | None = None,
        split_to_build: tp.Literal["train", "val", "all"] | None = None,
    ) -> tuple[dict[str, DataLoader], int]:

        if events is None:
            events = self.get_events()
        else:
            events = standardize_events(events)

        extractors = {}
        for modality in self.features_to_use:
            extractors[modality] = getattr(self, f"{modality}_feature")
        if "Fmri" in events.type.unique():
            extractors["fmri"] = self.neuro
        dummy_events = []
        for timeline_name, timeline in events.groupby("timeline"):
            if "split" in timeline.columns:
                splits = timeline.split.dropna().unique()
                assert (
                    len(splits) == 1
                ), f"Timeline {timeline_name} has multiple splits: {splits}"
                split = splits[0]
            else:
                split = "all"
            dummy_event = {
                "type": "CategoricalEvent",
                "timeline": timeline_name,
                "start": timeline.start.min(),
                "duration": timeline.stop.max() - timeline.start.min(),
                "split": split,
                "subject": timeline.subject.unique()[0],
            }
            dummy_events.append(dummy_event)
        events = pd.concat([events, pd.DataFrame(dummy_events)])
        events = standardize_events(events)

        extractors["subject_id"] = self.subject_id

        features_to_remove = set()
        for extractor_name, extractor in extractors.items():
            event_types = EventTypesHelper(extractor.event_types).names
            if not any(
                [event_type in events.type.unique() for event_type in event_types]
            ):
                features_to_remove.add(extractor_name)
        for extractor_name in features_to_remove:
            del extractors[extractor_name]
            LOGGER.warning(
                "Removing extractor %s as there are no corresponding events",
                extractor_name,
            )

        for name, extractor in extractors.items():
            LOGGER.info("Preparing extractor: %s", name)
            extractor.prepare(events)
            _free_extractor_model(extractor)

        # Prepare dataloaders
        loaders = {}
        if split_to_build is None:
            splits = ["train", "val"]
        else:
            splits = [split_to_build]
        for split in splits:
            LOGGER.info("Building dataloader for split %s", split)
            if split == "all" or self.split_segments_by_time:
                split_sel = [True] * len(events)
                shuffle = False
                overlap_trs = self.overlap_trs_train
            else:
                split_sel = events.split == split
                if split not in events.split.unique():
                    shuffle = False
                else:
                    shuffle = (
                        self.shuffle_train if split == "train" else self.shuffle_val
                    )
                if split == "val":
                    overlap_trs = self.overlap_trs_val or self.overlap_trs_train
                else:
                    overlap_trs = self.overlap_trs_train

            sel = np.array(split_sel)
            segments = ns.segments.list_segments(
                events[sel],
                triggers=events[sel].type == "CategoricalEvent",
                stride=(self.duration_trs - overlap_trs) * self.TR,
                duration=self.duration_trs * self.TR,
                stride_drop_incomplete=self.stride_drop_incomplete,
            )
            if len(segments) == 0:
                LOGGER.warning("No events found for split %s", split)
                continue
            dataset = ns.dataloader.SegmentDataset(
                extractors=extractors,
                segments=segments,
                remove_incomplete_segments=False,
            )
            dataloader = dataset.build_dataloader(
                shuffle=shuffle,
                num_workers=self.num_workers,
                batch_size=self.batch_size,
            )
            loaders[split] = dataloader

        return loaders


class TribeExperiment(BaseExperiment):
    """Experiment configuration for TRIBE v2 inference."""

    model_config = pydantic.ConfigDict(extra="forbid")

    data: Data
    seed: int | None = 33
    brain_model_config: BaseModelConfig
    # Hardware
    accelerator: str = "gpu"
    # Eval
    average_subjects: bool = False
    checkpoint_path: str | None = None
    load_checkpoint: bool = True

    # Internal
    _model: tp.Any = None

    infra: TaskInfra = TaskInfra(version="1")

    def model_post_init(self, __context: tp.Any) -> None:
        super().model_post_init(__context)

        if (
            not (self.checkpoint_path and self.load_checkpoint)
        ):
            study_summary = self.data.study.study_summary()
            self.data.subject_id.predefined_mapping = {
                subject: i for i, subject in enumerate(study_summary.subject.unique())
            }
            self.brain_model_config.subject_layers.n_subjects = (
                study_summary.subject.nunique()
            )
            if isinstance(self.brain_model_config.projector, SubjectLayers):
                self.brain_model_config.projector.n_subjects = (
                    study_summary.subject.nunique()
                )

        if self.average_subjects:
            study_name = self.data.study.names
            self.brain_model_config.subject_layers.average_subjects = True
            self.brain_model_config.subject_layers.n_subjects = 0
            if isinstance(self.brain_model_config.projector, SubjectLayers):
                self.brain_model_config.projector.average_subjects = True
            self.data.neuro.aggregation = "mean"
            self.data.subject_id.predefined_mapping = None
            if isinstance(study_name, str):
                LOGGER.debug(f"Setting study {study_name} in average subject mode")
                trigger_type = (
                    "Video" if study_name in ["Wen2017", "Allen2022Bold"] else "Audio"
                )
                self.data.study = set_study_in_average_subject_mode(
                    self.data.study, trigger_type=trigger_type, trigger_field="filepath"
                )
