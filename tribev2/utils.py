# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import typing as tp
from collections import Counter, OrderedDict, defaultdict
from functools import lru_cache
from pathlib import Path

import exca
import mne
import neuralset as ns
import numpy as np
import pandas as pd
from neuralset.events.study import Chain, Study
from neuralset.events.transforms import EventsBuilder, EventsTransform
from neuralset.extractors.neuro import FSAVERAGE_SIZES

from tribev2.eventstransforms import RemoveDuplicates

FMRI_SPACES = {
    "Algonauts2025Bold": "MNI152NLIN2009C_ASYM_RES_01",
    "Wen2017": "MNI152NLIN6_ASYM_RES_01",
    "Lahner2024Bold": "MNI152NLIN2009C_ASYM_RES_01",
    "Lebel2023Bold": "MNI152NLIN2009C_ASYM_RES_01",
    "Vanessen2023": "MNI152NLIN6_ASYM_RES_01",
    "Aliko2020": "MNICOLIN27",
    "Li2022": "MNICOLIN27",
    "Nastase2020": "MNI152NLIN2009C_ASYM_RES_01",
}
RECORDING_DURATIONS = {
    "Algonauts2025Bold/sub-01": 66.4,
    "Algonauts2025Bold/sub-02": 66.4,
    "Algonauts2025Bold/sub-03": 66.4,
    "Algonauts2025Bold/sub-04": 0,
    "Algonauts2025Bold/sub-05": 66.4,
    "Algonauts2025Bold/sub-06": 0,
    "Lahner2024Bold/1": 6.2,
    "Lahner2024Bold/10": 6.2,
    "Lahner2024Bold/2": 6.2,
    "Lahner2024Bold/3": 6.2,
    "Lahner2024Bold/4": 6.2,
    "Lahner2024Bold/5": 6.2,
    "Lahner2024Bold/6": 6.2,
    "Lahner2024Bold/7": 6.2,
    "Lahner2024Bold/8": 6.2,
    "Lahner2024Bold/9": 6.2,
    "Lebel2023Bold/UTS01": 17.9,
    "Lebel2023Bold/UTS02": 18.1,
    "Lebel2023Bold/UTS03": 18.1,
    "Lebel2023Bold/UTS04": 6.2,
    "Lebel2023Bold/UTS05": 6.4,
    "Lebel2023Bold/UTS06": 6.4,
    "Lebel2023Bold/UTS07": 6.4,
    "Lebel2023Bold/UTS08": 6.4,
    "Wen2017/subject1": 11.7,
    "Wen2017/subject2": 11.7,
    "Wen2017/subject3": 11.7,
}


class MultiStudyLoader(EventsBuilder):
    """Config for loading multiple studies.
    Note that the query and enhancers are shared across all studies.
    For example, setting timeline_index == 0 will select the first timeline of each study.
    """

    names: str | list[str]
    path: str | Path
    transforms: list[EventsTransform] | OrderedDict[str, EventsTransform] | None = None
    query: str | None = None
    studies_to_include: list[str] | None = None
    infra_timelines: exca.MapInfra = exca.MapInfra(cluster="processpool", max_jobs=None)

    def model_post_init(self, log__: tp.Any) -> None:
        super().model_post_init(log__)
        if self.studies_to_include is not None:
            for name in self.studies_to_include:
                if name not in self.names:
                    raise ValueError(f"Study {name} not found in {self.names}")
        self.get_studies()  # run this so that studies are registered (in case _run is cached)

    @infra_timelines.apply(item_uid=str)
    def dummy(self, items: tp.Iterable[str]) -> tp.Iterator[None]:
        for item in items:
            yield None

    def get_studies(self) -> dict[str, Chain]:
        studies = {}
        if isinstance(self.names, str):
            names = [self.names]
        else:
            names = self.names
        for name in names:
            studies[name] = Study(
                name=name,
                path=self.path,
                query=self.query,
                infra_timelines=self.infra_timelines,
            )
        return studies

    def study_summary(self, apply_query: bool = True) -> pd.DataFrame:
        summaries = []
        for name, study in self.get_studies().items():
            if (
                apply_query
                and self.studies_to_include is not None
                and name not in self.studies_to_include
            ):
                continue
            summary = study.study_summary(apply_query=apply_query)
            summary.loc[:, "study"] = name
            summaries.append(summary)
        return pd.concat(summaries, ignore_index=True)

    def _run(self) -> pd.DataFrame:
        dfs = []
        for name, study in self.get_studies().items():
            if (
                self.studies_to_include is not None
                and name not in self.studies_to_include
            ):
                continue
            chain = Chain(steps={"study": study, **OrderedDict(self.transforms)})
            df = chain.run()
            df.loc[:, "study"] = name
            dfs.append(df)
        out = pd.concat(dfs, ignore_index=True)
        return out


def split_segments_by_time(
    segments: list[ns.segments.Segment], val_ratio: float, split: str
) -> list[ns.segments.Segment]:
    timeline_segments = defaultdict(list)
    return_segments = []
    for segment in segments:
        if len(segment.ns_events) == 0:
            continue
        timeline = segment.ns_events[0].timeline
        timeline_segments[timeline].append(segment)
    for timeline, segments in timeline_segments.items():
        start = min(segment.start for segment in segments)
        stop = max(segment.stop for segment in segments)
        split_time = start + (stop - start) * val_ratio
        for segment in segments:
            if split == "val" and segment.start < split_time:
                return_segments.append(segment)
            elif split == "train" and segment.start >= split_time:
                return_segments.append(segment)
    return return_segments


def assign_fmri_space(events: pd.DataFrame, space: str | None = None) -> pd.DataFrame:
    assert events.study.nunique() == 1, "Only one study can be assigned at a time"
    study_name = events.study.unique()[0]
    if study_name not in FMRI_SPACES:
        raise ValueError(f"Study {study_name} not found in FMRI_SPACES")
    default_space = FMRI_SPACES[study_name]
    assigned_space = space or default_space
    events.loc[events.type == "Fmri", "space"] = assigned_space
    return events


def set_study_in_average_subject_mode(
    study: EventsBuilder, trigger_type: str, trigger_field: str = "filepath"
) -> EventsBuilder:
    study.transforms["alignevents"] = ns.events.transforms.AlignEvents(
        trigger_type=trigger_type, trigger_field=trigger_field, types_to_align="Event"
    )
    study.transforms["removeduplicates"] = RemoveDuplicates(
        subset=["start", "stop", "filepath", "type"]
    )
    for key in ["chunksounds", "chunkvideos"]:
        study.transforms.move_to_end(key)
    return study


def get_subject_weights(
    subject_id_mapping: dict[str, int],
    weigh_by: tp.Literal[
        "n_subjects", "speech", "video", "recording_time"
    ] = "n_subjects",
) -> dict[str, float]:
    subject_weights = []
    if weigh_by in ["speech", "video"]:
        for subject in subject_id_mapping:
            if weigh_by == "speech":
                weight = int(subject.startswith("Lebel"))
            elif weigh_by == "video":
                weight = int(subject.startswith("Algonauts"))
            subject_weights.append(float(weight))
    elif weigh_by == "recording_time":
        for subject in subject_id_mapping:
            if subject not in RECORDING_DURATIONS:
                raise ValueError(f"Subject {subject} not found in RECORDING_DURATIONS")
            subject_weights.append(float(RECORDING_DURATIONS[subject]))
    elif weigh_by == "n_subjects":
        num_subjects_per_study = Counter(
            [k.split("/")[0] for k in subject_id_mapping.keys()]
        )
        for subject in subject_id_mapping:
            weight = 1 / num_subjects_per_study[subject.split("/")[0]]
            subject_weights.append(float(weight))
    else:
        raise ValueError(f"Invalid weight type: {weigh_by}")
    return subject_weights


@lru_cache
def get_hcp_labels(mesh="fsaverage5", combine=False, hemi="both"):
    """
    Get the HCP labels for the fsaverage subject.
    """
    if hemi in ["right", "left"]:
        subjects_dir = Path(mne.datasets.sample.data_path()) / "subjects"
        mne.datasets.fetch_hcp_mmp_parcellation(
            subjects_dir=subjects_dir, accept=True, verbose=True, combine=combine
        )
        name = "HCPMMP1_combined" if combine else "HCPMMP1"
        with ns.utils.ignore_all():
            labels = mne.read_labels_from_annot(
                "fsaverage", name, hemi="both", subjects_dir=subjects_dir
            )
        label_to_vertices = {}
        for label in labels:
            name, vertices = label.name, np.array(label.vertices)
            if not combine:
                name = name[2:]
            name = name.replace("_ROI", "")  # .replace(" Cortex", "")
            if (hemi == "right" and "-lh" in name) or (
                hemi == "left" and "-rh" in name
            ):
                continue
            name = name.replace("-rh", "").replace("-lh", "")
            label_to_vertices[name] = np.array(vertices)
        assert sum(len(v) for v in label_to_vertices.values()) == 163842
        expected_size = FSAVERAGE_SIZES[mesh]
        index_offset = expected_size if hemi == "right" else 0
        label_to_vertices = {
            k: v[v < expected_size] + index_offset for k, v in label_to_vertices.items()
        }
        assert sum(len(v) for v in label_to_vertices.values()) == expected_size
        return label_to_vertices
    else:
        assert hemi == "both", f"Invalid hemisphere: {hemi}"
        left, right = get_hcp_labels(
            mesh=mesh, combine=combine, hemi="left"
        ), get_hcp_labels(mesh=mesh, combine=combine, hemi="right")
        label_to_vertices = {
            k: np.concatenate([left[k], right[k]]) for k in left.keys()
        }
        return label_to_vertices


def get_hcp_vertex_labels(mesh="fsaverage5", combine=False):
    labels = get_hcp_labels(mesh, combine)
    out = [""] * FSAVERAGE_SIZES[mesh] * 2
    for label, vertices in labels.items():
        for vertex in vertices:
            out[int(vertex)] = label
    return out


def get_hcp_roi_indices(rois: str | list[str], hemi="both", mesh="fsaverage5"):
    labels = get_hcp_labels(mesh=mesh, combine=False, hemi=hemi)
    if isinstance(rois, str):
        rois = [rois]
    selected_labels = []
    for roi in rois:
        if roi[-1] == "*":
            sel = [label for label in labels.keys() if label.startswith(roi[:-1])]
        elif roi[0] == "*":
            sel = [label for label in labels.keys() if label.endswith(roi[1:])]
        else:
            sel = [label for label in labels.keys() if label == roi]
        if not sel:
            raise ValueError(f"ROI {roi} not found in HCP labels")
        selected_labels.extend(sel)
    vertex_indices = np.concatenate([labels[label] for label in selected_labels])
    return vertex_indices


def summarize_by_roi(data: np.ndarray, hemi="both", mesh="fsaverage5"):
    assert data.ndim == 1, "Data must be 1D"
    if hemi in ["left", "right", "both"]:
        labels = get_hcp_labels(mesh=mesh, combine=False, hemi=hemi)
        out = np.array(
            [
                data[get_hcp_roi_indices(roi, hemi=hemi, mesh=mesh)].mean()
                for roi in labels.keys()
            ]
        )
    elif hemi == "both_separate":
        out = np.concatenate(
            [
                summarize_by_roi(data, hemi="left", mesh=mesh),
                summarize_by_roi(data, hemi="right", mesh=mesh),
            ]
        )
    else:
        raise ValueError(f"Invalid hemisphere: {hemi}")
    return out


def get_topk_rois(data: np.ndarray, hemi="both", mesh="fsaverage5", k=10) -> list[str]:
    values = summarize_by_roi(data, hemi=hemi, mesh=mesh)
    if hemi == "both_separate":
        left_labels = get_hcp_labels(mesh=mesh, combine=False, hemi="left").keys()
        right_labels = get_hcp_labels(mesh=mesh, combine=False, hemi="right").keys()
        labels = [f"{l}-lh" for l in left_labels] + [f"{l}-rh" for l in right_labels]
    else:
        labels = list(get_hcp_labels(mesh=mesh, combine=False, hemi=hemi).keys())
    top_k = np.argsort(values)[::-1][:k]
    return np.array(labels)[top_k]
