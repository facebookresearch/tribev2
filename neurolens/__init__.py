"""NeuroLens: Interactive neuroscience playground built on TRIBE v2."""

from neurolens.cache import CacheManager
from neurolens.stimulus import Stimulus, StimulusLibrary
from neurolens.predict import get_prediction_at_time, get_num_timesteps, get_top_rois
from neurolens.match import find_similar_stimuli, build_target_from_regions, find_contrast_stimuli
from neurolens.eval import compute_all_model_alignments, compute_model_brain_alignment
from neurolens.roi import ROI_GROUPS, get_roi_group_names, summarize_by_roi_group
from neurolens.viz import plot_brain_surface, make_radar_chart

__all__ = [
    "CacheManager",
    "Stimulus",
    "StimulusLibrary",
    "get_prediction_at_time",
    "get_num_timesteps",
    "get_top_rois",
    "find_similar_stimuli",
    "build_target_from_regions",
    "find_contrast_stimuli",
    "compute_all_model_alignments",
    "compute_model_brain_alignment",
    "ROI_GROUPS",
    "get_roi_group_names",
    "summarize_by_roi_group",
    "plot_brain_surface",
    "make_radar_chart",
]
