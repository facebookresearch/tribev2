# neuroLoop — State of the Union

**Date**: 2026-04-13
**Branch**: main (6cbf42d)
**Remote**: https://github.com/markuel/neuroLoop.git
**Upstream**: https://github.com/facebookresearch/tribev2.git

---

## What This Repo Is

A fork of Meta's **TRIBE v2** — a foundation model that predicts fMRI brain responses to naturalistic stimuli (video, audio, text) — plus a new **neuroLoop SDK** for mapping those predictions to named brain regions.

**License**: CC-BY-NC 4.0 (non-commercial only)

---

## Repository Structure

```
neuroLoop/                     Brain region analysis SDK
├── __init__.py                   Lazy export of BrainAtlas
├── atlas.py                      BrainAtlas class (261 lines)
└── regions.py                    HCP-MMP1 grouping tables (167 lines)

tribev2/                       TRIBE v2 inference engine (from Meta)
├── model.py                      FmriEncoder: Transformer multimodal→fMRI (234 lines)
├── main.py                       Data + TribeExperiment: config & data loading (240 lines)
├── demo_utils.py                 TribeModel: inference API (392 lines)
├── eventstransforms.py           Event processing pipeline (273 lines)
├── utils.py                      HCP labels, multi-study loading (210 lines)
├── utils_fmri.py                 Surface projection (248 lines)
└── plotting/
    ├── base.py                   BasePlotBrain abstract class (497 lines)
    ├── cortical.py               Nilearn backend (311 lines)
    ├── cortical_pv.py            PyVista backend (280 lines)
    ├── subcortical.py            Subcortical visualization (311 lines)
    └── utils.py                  Colormaps, normalization, layout (563 lines)

tribe_demo.ipynb               Colab notebook: load model → predict → visualize
pyproject.toml                 Package config (Python >=3.11)
```

**Removed** (training-only, not needed for inference):
- `tribev2/studies/` — dataset definitions (Algonauts2025, Lahner2024, Lebel2023, Wen2017)
- `tribev2/grids/` — Slurm grid search configs
- `tribev2/pl_module.py` — PyTorch Lightning training module

---

## Model Architecture

```
Multimodal inputs (text / audio / video)
  │
  ├─ Text:  Llama-3.2-3B (layers 0–1.0)
  ├─ Audio: Wav2VecBert (layers 0.75–1.0)
  └─ Video: V-JEPA2 ViT-G (layers 0.75–1.0)
        │
        ▼
  Per-modality projectors → Combiner MLP
        │
        ▼
  Temporal smoothing (learnable Gaussian 1D conv)
        │
        ▼
  Positional embeddings (learnable, max 1024 tokens)
        │
        ▼
  Transformer encoder (x_transformers, hidden=1152)
        │
        ▼
  Low-rank bottleneck (2048) → Subject-specific head
        │
        ▼
  Output: (batch, n_vertices, n_timesteps) on fsaverage5
```

**Key numbers**:
- Output: 20,484 vertices (10,242 per hemisphere) on fsaverage5
- Hemodynamic offset: 5 seconds into the past
- Training: MSE loss, Pearson correlation as primary metric
- 15 epochs, batch size 8, Adam lr=1e-4 + OneCycleLR

---

## Training Datasets (used by pretrained model, not included in repo)

The pretrained model on HuggingFace was trained on these datasets:

| Study | Stimuli | Subjects | TR (sec) |
|-------|---------|----------|----------|
| Algonauts2025 | Friends + 4 films | 4 | 1.49 |
| Lahner2024 | 3-sec video clips | 10 | 1.75 |
| Lebel2023 | Spoken stories | 8 | 2.0 |
| Wen2017 | Naturalistic video | 3 | 2.0 |

---

## neuroLoop SDK

Maps the model's raw 20,484-vertex output to named HCP-MMP1 brain regions.

### API

```python
from neuroLoop import BrainAtlas

atlas = BrainAtlas()  # mesh="fsaverage5", hemi="both"

# Discover
atlas.list_groups("coarse")                    # 7 Yeo networks
atlas.list_groups("fine")                      # 22 Glasser networks
atlas.list_regions("Visual", level="coarse")   # regions in Visual network

# Extract timeseries from model.predict() output
atlas.region_timeseries(preds, "V1")           # (n_timesteps,) for one region
atlas.all_region_timeseries(preds)             # DataFrame: timesteps × 360 regions
atlas.group_timeseries(preds, "Visual")        # (n_timesteps,) averaged over network
atlas.all_group_timeseries(preds, "coarse")    # DataFrame: timesteps × 7 networks

# Export
atlas.to_csv(preds, "output.csv")
```

### Grouping Levels

**Coarse — 7 networks** (Yeo 2011 spatial overlap):

| Network | Regions | Role |
|---------|---------|------|
| Visual | 32 | Seeing |
| Somatomotor | 28 | Movement, touch, basic auditory |
| Dorsal Attention | 23 | Directed spatial attention |
| Ventral Attention | 24 | Reorienting, salience |
| Limbic | 11 | Emotion, memory, reward |
| Frontoparietal | 26 | Executive control |
| Default | 36 | Self-reflection, mind wandering |

**Fine — 22 networks** (Glasser 2016):

| Network | # | Network | # |
|---------|---|---------|---|
| Primary Visual | 1 | Insular & Frontal Opercular | 13 |
| Early Visual | 3 | Medial Temporal | 7 |
| Dorsal Stream Visual | 6 | Lateral Temporal | 9 |
| Ventral Stream Visual | 7 | Temporo-Parieto-Occipital Jcn | 5 |
| MT+ Complex | 9 | Superior Parietal | 10 |
| Somatosensory & Motor | 5 | Inferior Parietal | 10 |
| Paracentral & Mid-Cingulate | 8 | Posterior Cingulate | 14 |
| Premotor | 7 | Anterior Cingulate & Med. PFC | 15 |
| Posterior Opercular | 6 | Orbital & Polar Frontal | 11 |
| Early Auditory | 5 | Inferior Frontal | 8 |
| Auditory Association | 8 | Dorsolateral Prefrontal | 13 |

All 180 regions verified against MNE-Python's HCP-MMP1 parcellation. No duplicates.

---

## Fork Changes from Upstream

| Commit | What changed |
|--------|-------------|
| `bcd5c23` | Added `grids/configs.py` with mini_config (Qwen3-0.6B, VJEPA2-ViTL) and base_config. Grid scripts default to mini_config. |
| `fb1eac6` | Python 3.11 compat fix + exca dependency conflict resolution |
| `6cbf42d` | Added neuroLoop SDK (atlas.py, regions.py) |

No architectural changes to the upstream TRIBE v2 model.

---

## Dependencies

### Core
- torch >=2.5.1,<2.7
- numpy ==2.2.6
- neuralset ==0.0.2 / neuraltrain ==0.0.2
- x_transformers ==1.27.20
- transformers, einops, huggingface_hub

### Media
- moviepy >=2.2.1, soundfile, julius, gtts, langdetect, spacy

### Optional: plotting
- nibabel, matplotlib, nilearn, pyvista, seaborn, colorcet, scipy, scikit-image

---

## Entry Points

```python
# Inference
from tribev2 import TribeModel
model = TribeModel.from_pretrained("facebook/tribev2")
preds, segments = model.predict(model.get_events_dataframe(video_path="clip.mp4"))

# Region analysis
from neuroLoop import BrainAtlas
df = BrainAtlas().to_dataframe(preds)

# Demo notebook
# jupyter notebook tribe_demo.ipynb
```

---

## Known Limitations

- **No test suite** — only `test_run.py` for integration smoke tests
- **Hardcoded values** — recording durations, fMRI spaces, split attributes all in dicts
- **Language** — speech transcription hardcoded to English
- **Environment** — DATAPATH/SAVEPATH env vars required for training, no validation
- **Average subject only** — inference produces predictions for the "average" subject
- **50-70% segment retention** — empty segments filtered during prediction
- **Non-commercial license** — CC-BY-NC 4.0 restricts commercial use
