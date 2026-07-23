# NeuroLens — Getting Started Guide

A step-by-step guide to testing, running, and extending NeuroLens.

---

## Table of Contents

1. [Prerequisites](#1-prerequisites)
2. [Local Setup](#2-local-setup)
3. [Running Tests](#3-running-tests)
4. [Generating the Cache (Precompute)](#4-generating-the-cache-precompute)
5. [Running the Interactive Notebook](#5-running-the-interactive-notebook)
6. [Project Structure](#6-project-structure)
7. [Next Steps & Roadmap](#7-next-steps--roadmap)
8. [Troubleshooting](#8-troubleshooting)

---

## 1. Prerequisites

| Requirement | Version | Notes |
|-------------|---------|-------|
| Python | 3.10+ | Tested on 3.11 |
| uv | Latest | Package manager (`pip install uv` or `curl -LsSf https://astral.sh/uv/install.sh \| sh`) |
| Git | Any | For version control |
| Google Colab account | Free tier | For GPU-based precompute step |
| HuggingFace account | Free | Needed for LLaMA 3.2 access (gated model) |

**Hardware:**
- **Precompute notebook:** Requires GPU (Colab free tier T4 is sufficient)
- **Main notebook:** CPU only (all heavy computation is pre-cached)
- **Tests:** CPU only, ~12 seconds for full suite

---

## 2. Local Setup

### 2.1 Clone and enter the repo

```bash
cd /opt/CodeRepo/tribev2   # or wherever your clone lives
```

### 2.2 Create a virtual environment

```bash
uv venv .venv --python 3.11
source .venv/bin/activate
```

### 2.3 Install dependencies

```bash
# Core tribev2 + plotting dependencies
uv pip install -e ".[plotting]"

# Additional NeuroLens dependencies
uv pip install plotly ipywidgets scipy
```

### 2.4 Verify the installation

```bash
python -c "import neurolens; print('NeuroLens OK')"
python -c "from tribev2 import TribeModel; print('TRIBE v2 OK')"
```

Both should print OK without errors.

---

## 3. Running Tests

### 3.1 Run the full test suite

```bash
python -m pytest tests/ -v
```

Expected output: **31 passed** in ~12 seconds.

### 3.2 Run tests by module

```bash
# Individual module tests
python -m pytest tests/test_roi.py -v          # ROI utilities (3 tests)
python -m pytest tests/test_stimulus.py -v     # Stimulus library (5 tests)
python -m pytest tests/test_cache.py -v        # Cache manager (6 tests)
python -m pytest tests/test_viz.py -v          # Visualization (3 tests)
python -m pytest tests/test_predict.py -v      # Predict module (5 tests)
python -m pytest tests/test_match.py -v        # Match module (4 tests)
python -m pytest tests/test_eval.py -v         # Eval module (4 tests)
python -m pytest tests/test_integration.py -v  # End-to-end pipeline (1 test)
```

### 3.3 What the tests cover

| Test File | Tests | What it verifies |
|-----------|-------|-----------------|
| `test_roi.py` | 3 | ROI groups are defined, names returned correctly, summarization works on fsaverage5 |
| `test_stimulus.py` | 5 | Stimulus dataclass, library loading from JSON, filtering, categories, missing ID handling |
| `test_cache.py` | 6 | Loading `.npz` brain preds, `.json` ROI summaries, `.pt` embeddings, missing file returns None |
| `test_viz.py` | 3 | Brain surface plot returns figure, radar charts work for single and comparison modes |
| `test_predict.py` | 5 | Time-sliced predictions, index clamping, top ROIs, modality contributions |
| `test_match.py` | 4 | Cosine similarity search, self-similarity > 0.99, synthetic target from regions, contrast mode |
| `test_eval.py` | 4 | Pairwise similarity matrix, RSA score computation, model-brain alignment |
| `test_integration.py` | 1 | Full pipeline: cache -> predict -> match -> eval -> visualize |

### 3.4 Known warnings

You may see this warning during tests that call `build_target_from_regions`:

```
UserWarning: LabelEncoder: event_types has not been set...
```

This comes from `neuralset` (TRIBE v2 dependency) and is harmless. It does not affect functionality.

### 3.5 First run note

The first run of `test_roi.py` or `test_match.py::test_build_target_from_regions` downloads MNE sample data (~1.65 GB) to `~/mne_data/`. Subsequent runs use the cached data and are fast.

---

## 4. Generating the Cache (Precompute)

This is where you process your stimulus library through TRIBE v2 and comparison models. **Run this once on a GPU.**

### 4.1 Prepare your stimuli

Collect 50-80 short video/audio clips (5-15 seconds each). Good sources for CC-licensed content:

| Source | Type | License | URL |
|--------|------|---------|-----|
| Pexels | Video | Free (Pexels License) | https://www.pexels.com/videos/ |
| Pixabay | Video/Audio | Pixabay License | https://pixabay.com/ |
| LibriVox | Audiobook | Public Domain | https://librivox.org/ |
| Freesound | Audio | CC | https://freesound.org/ |

**Recommended category distribution:**

| Category | Count | Examples |
|----------|-------|---------|
| Speech | 8-10 | TED talks, podcast clips, audiobook excerpts |
| Music | 8-10 | Classical, hip-hop, ambient, vocals-only |
| Silence + Visuals | 8-10 | Nature timelapse, abstract art, faces |
| Emotional | 8-10 | Horror, comedy, heartwarming scenes |
| Multimodal-rich | 8-10 | Movie scenes with dialogue + action + music |
| Text-only | 5-8 | Narrated stories, poetry readings |

### 4.2 Upload to Colab

1. Open `neurolens_precompute.ipynb` in Google Colab
2. Set runtime to **GPU** (Runtime > Change runtime type > T4 GPU)
3. Upload your stimulus files to a `stimuli/` folder in Colab

### 4.3 Edit the stimulus list

In Cell 2, update the `STIMULI` list with your actual files:

```python
STIMULI = [
    {"id": "clip_001", "name": "Nature timelapse", "category": "Silence + Visuals",
     "media_type": "video", "duration_sec": 10.0, "path": "stimuli/nature.mp4"},
    {"id": "clip_002", "name": "Beethoven Moonlight", "category": "Music",
     "media_type": "audio", "duration_sec": 15.0, "path": "stimuli/moonlight.wav"},
    # ... add all your stimuli
]
```

**ID naming convention:** Use `clip_001`, `clip_002`, etc. for consistency.

### 4.4 Authenticate with HuggingFace

TRIBE v2 uses LLaMA 3.2 (gated model). Before running Cell 3:

1. Go to https://huggingface.co/meta-llama/Llama-3.2-3B and accept the license
2. Create an access token at https://huggingface.co/settings/tokens (read access)
3. In Colab, run:
```python
!huggingface-cli login
# Paste your token when prompted
```

### 4.5 Run all cells

Run cells 1-6 in order. Expected timing on Colab T4:

| Cell | Operation | Time estimate |
|------|-----------|--------------|
| 1 | Install dependencies | 2-3 minutes |
| 2 | Define stimuli + save metadata | Instant |
| 3 | TRIBE v2 predictions | 1-3 min per stimulus (depends on length) |
| 4 | CLIP embeddings | ~2 sec per video stimulus |
| 5 | Whisper embeddings | ~3 sec per stimulus with audio |
| 6 | GPT-2 embeddings | ~1 sec per stimulus |

For 50 stimuli, expect ~2-3 hours total for Cell 3 (the bottleneck).

### 4.6 Download the cache

After all cells complete, download the `neurolens_cache/` folder:

```python
# Option A: Zip and download
!zip -r neurolens_cache.zip neurolens_cache/
# Then use Colab's file browser to download

# Option B: Upload to Google Drive
from google.colab import drive
drive.mount('/content/drive')
!cp -r neurolens_cache/ /content/drive/MyDrive/neurolens_cache/

# Option C: Upload to HuggingFace Hub (recommended for sharing)
!huggingface-cli upload your-username/neurolens-cache neurolens_cache/
```

### 4.7 Cache structure reference

After precompute, your cache should look like:

```
neurolens_cache/
├── stimuli/
│   └── metadata.json              # Stimulus metadata (auto-generated)
├── brain_preds/
│   ├── clip_001.npz               # Shape: (n_timesteps, 20484)
│   ├── clip_002.npz
│   └── ...
├── roi_summaries/
│   ├── clip_001.json              # {"Visual Cortex": 0.42, ...}
│   ├── clip_002.json
│   └── ...
└── embeddings/
    ├── clip/
    │   ├── clip_001.pt            # CLIP ViT-B-32 image embedding
    │   └── ...
    ├── whisper/
    │   ├── clip_001.pt            # Whisper-base encoder embedding
    │   └── ...
    └── gpt2/
        ├── clip_001.pt            # GPT-2 text embedding
        └── ...
```

---

## 5. Running the Interactive Notebook

### 5.1 Place the cache

Copy `neurolens_cache/` into the project root (same level as `neurolens.ipynb`):

```bash
# If downloaded locally
cp -r ~/Downloads/neurolens_cache/ /opt/CodeRepo/tribev2/neurolens_cache/

# If on Google Drive (in Colab)
!cp -r /content/drive/MyDrive/neurolens_cache/ neurolens_cache/
```

### 5.2 Launch the notebook

**Option A: Local Jupyter**
```bash
source .venv/bin/activate
uv pip install jupyterlab
jupyter lab neurolens.ipynb
```

**Option B: Google Colab**
1. Upload `neurolens.ipynb` to Colab
2. Upload `neurolens_cache/` folder
3. Upload the `neurolens/` package folder
4. Set runtime to CPU (GPU not needed)
5. Run all cells

### 5.3 Using each module

**Module 1: PREDICT**
- Select a stimulus from the dropdown
- Drag the timestep slider to see how brain activation changes over time
- Select different views (left, right, medial, dorsal) to see different angles
- The top-5 most activated brain regions are shown below the plot

**Module 2: MATCH**
- **Region Picker mode:** Set intensity sliders for different brain regions, click "Find Matches" to find stimuli that best activate those regions
- **More Like This mode:** Select a stimulus, find neurally similar content
- **Contrast mode:** Pick a region to maximize and one to minimize (e.g., "maximize Visual Cortex, minimize Auditory Cortex")

**Module 3: EVAL**
- Click "Run Leaderboard" to see which AI model's representations align most with brain responses
- Select two models from the dropdowns and click "Compare Models" for a head-to-head comparison
- Each model gets a "Brain Report Card" with alignment percentage

---

## 6. Project Structure

```
tribev2/                          # Root repository
├── tribev2/                      # Original TRIBE v2 package (Meta)
│   ├── model.py                  # FmriEncoder neural network
│   ├── demo_utils.py             # TribeModel inference API
│   ├── utils.py                  # HCP ROI utilities, data loading
│   ├── plotting/                 # Brain visualization (nilearn, pyvista)
│   └── ...
│
├── neurolens/                    # NeuroLens package (our code)
│   ├── __init__.py               # Public API exports
│   ├── roi.py                    # Human-friendly ROI groups
│   ├── stimulus.py               # Stimulus metadata management
│   ├── cache.py                  # Cache loading (npz, json, pt)
│   ├── viz.py                    # Brain plots + radar charts
│   ├── predict.py                # Time-sliced brain predictions
│   ├── match.py                  # Neural similarity search
│   └── eval.py                   # RSA model-brain alignment
│
├── tests/                        # 31 tests across 8 files
│   ├── test_roi.py
│   ├── test_stimulus.py
│   ├── test_cache.py
│   ├── test_viz.py
│   ├── test_predict.py
│   ├── test_match.py
│   ├── test_eval.py
│   └── test_integration.py
│
├── neurolens_precompute.ipynb    # GPU notebook: generate cache
├── neurolens.ipynb               # Main interactive notebook (CPU)
│
├── docs/
│   ├── GETTING_STARTED.md        # This file
│   └── superpowers/
│       ├── specs/                # Design spec
│       └── plans/                # Implementation plan
│
└── neurolens_cache/              # Generated cache (not in git)
```

---

## 7. Next Steps & Roadmap

### Immediate (to get it running)

- [ ] **Collect stimuli** — Download 50-80 CC-licensed clips across the 6 categories
- [ ] **Run precompute** — Process clips through TRIBE v2 + comparison models on Colab
- [ ] **Test the experience** — Open the main notebook and interact with all 3 modules
- [ ] **Polish visualizations** — Adjust colormaps, layout, and labels based on real data

### Short-term improvements

- [ ] **Add more comparison models** — Add DINO v2 (already in TRIBE v2), BLIP-2, or newer models to the eval module. Just extract embeddings and save as `.pt` in the cache
- [ ] **Per-ROI eval** — Extend `compute_model_brain_alignment` to return per-ROI-group scores (not just overall). This enables the radar chart comparison in the Eval module
- [ ] **Cache hosting** — Upload the cache to HuggingFace Hub with auto-download in the notebook setup cell
- [ ] **Modality toggle** — The Predict module supports modality contributions (video-only, audio-only, text-only) but the precompute notebook doesn't generate per-modality caches yet. Add per-modality TRIBE v2 runs

### Medium-term extensions

- [ ] **Blog post** — Write up the project as a portfolio piece. Focus on: system design (compute/serving separation), eval methodology (RSA), and the creative "brain-matched content" angle
- [ ] **Streamlit/Gradio web app** — Convert the notebook into a deployed web app for a more polished demo
- [ ] **Subject-specific predictions** — TRIBE v2 supports per-subject predictions. Add a subject selector to the Predict module
- [ ] **Temporal RSA** — Instead of time-averaging brain predictions, compute RSA at each timestep to see how alignment changes over time

### Portfolio positioning

When presenting this project in interviews or applications:

1. **System design** — "I designed a compute/serving split where GPU-heavy inference is pre-computed and the interactive layer runs on CPU. This is the same pattern used in production recommendation systems."
2. **Evaluation methodology** — "I implemented Representational Similarity Analysis (RSA) to benchmark how closely AI models' internal representations match biological neural responses. This is directly transferable to LLM evaluation."
3. **Multimodal AI** — "I worked hands-on with 7 SOTA models (LLaMA 3.2, V-JEPA2, Wav2Vec-BERT, DINOv2, CLIP, Whisper, GPT-2) in a unified pipeline."

---

## 8. Troubleshooting

### `ModuleNotFoundError: No module named 'neuralset'`

The `neuralset` and `neuraltrain` packages are custom Meta libraries bundled with TRIBE v2. Install with:

```bash
uv pip install -e "."
```

### `ModuleNotFoundError: No module named 'nilearn'`

Install plotting dependencies:

```bash
uv pip install -e ".[plotting]"
```

### MNE data download hangs or fails

The first call to `build_target_from_regions` or `summarize_by_roi_group` downloads MNE sample data (~1.65 GB). If it fails:

```bash
python -c "import mne; mne.datasets.sample.data_path()"
```

This downloads to `~/mne_data/`. Ensure you have disk space and internet access.

### `RuntimeError: TribeModel must be instantiated via .from_pretrained`

You're trying to use `TribeModel` directly. Always use:

```python
model = TribeModel.from_pretrained("facebook/tribev2", cache_folder="./cache")
```

### Colab runs out of memory during precompute

TRIBE v2 + LLaMA 3.2 needs ~12 GB VRAM. On Colab free tier:
- Use T4 GPU (16 GB VRAM)
- Process stimuli one at a time (the notebook already does this)
- If still OOM, reduce stimulus duration to < 10 seconds

### `FileNotFoundError: metadata.json`

The `neurolens_cache/stimuli/metadata.json` file is missing. Either:
- You haven't run the precompute notebook yet
- The cache directory is in the wrong location (must be at project root)

### Tests pass locally but fail in CI

The `test_roi.py` and `test_match.py::test_build_target_from_regions` tests require MNE sample data. In CI, either:
- Pre-cache MNE data in the CI image
- Mark these tests with `@pytest.mark.slow` and skip in CI
