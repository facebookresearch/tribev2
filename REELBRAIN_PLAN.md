# ReelBrain: Neural Scoring Engine for Instagram Reels

## Context

Pranav owns multiple Instagram channels (100k+ followers each) and has access to retention/engagement data. From analyzing 4 reels through TRIBE v2 brain scans, we discovered a clear pattern: **reels that work have strong, consistent brain activation from the first frame with no dead spots**. Reels that flop either start weak or have activation dips.

The goal is to productize this into a scoring tool for Pranav's clients — upload a reel, get a neural engagement score + actionable feedback before posting.

---

## Architecture

```
Client uploads reel (.mp4)
        |
   [ ReelBrain API — FastAPI on GPU server ]
        |
   [ TRIBE v2 inference ]
   V-JEPA2 + LLaMA 3.2 + Wav2Vec
        |
   (timesteps, ~20K vertices) raw predictions
        |
   [ Feature Extractor ]
   - ROI averages (visual cortex, reward, attention, emotion)
   - Temporal stats (first 3s intensity, consistency, dips, ramp)
   - Audio coverage analysis
        |
   [ Scorer ]
   - Weighted formula → 0-100 score
   - Threshold: SHIP IT / FIX IT / KILL IT
        |
   [ Report Generator ]
   - Score + grade
   - Timestep-by-timestep breakdown
   - Specific fix suggestions ("weak at 2-4s, add visual cut")
   - Brain heatmap image
        |
   Client gets report (JSON + images)
```

---

## Build Phases

### Phase 1: Calibration Dataset (Week 1-2)
**You do this — not code.**

1. Collect 50-100 reels from your channels with known performance data:
   - Views, likes, shares, saves, completion rate, retention curve
   - Label each: HIT (top 25% performance) / OK (middle 50%) / FLOP (bottom 25%)
2. Run each through TRIBE v2, save raw predictions as `.npy` files
3. This becomes your ground truth for tuning the scoring formula

### Phase 2: Feature Extraction Pipeline (Week 2)
**Build `reelbrain/features.py`**

Extract these features from TRIBE v2 output per reel:

**Temporal Features:**
| Feature | What | Why |
|---------|------|-----|
| `hook_score` | Mean activation intensity in first 3 seconds | First impression decides scroll vs watch |
| `sustain_score` | Min activation across all timesteps | Any dip = viewer drops off |
| `consistency` | 1 - (std / mean) of activation over time | Stable engagement throughout |
| `ramp_direction` | Slope of activation over time | Front-loaded = good, back-loaded = too late |
| `dead_spots` | Count of timesteps below threshold | Hard drops kill reels |
| `peak_intensity` | Max activation value | How high does engagement go |
| `audio_coverage` | % of timesteps with non-silent audio | Silence = scroll |

**Cortical ROI Features (using HCP Glasser parcellation, `tribev2/utils.py`):**

Paper reports these are the most reliably predicted regions (Pearson R values from Figure 8):

| Feature | HCP ROIs | Pearson R | Signal |
|---------|----------|-----------|--------|
| `visual_early` | V1, V2, V3, V4 | 0.40-0.50 | Are they visually stimulated? (strongest signal) |
| `visual_ventral` | VVC, PHA, FFC | 0.25-0.35 | Face/object/place recognition |
| `visual_motion` | MT, V3A, V6, V7 | 0.30-0.40 | Motion processing — fast cuts, action, camera movement |
| `auditory_early` | A1, MBelt, LBelt, PBelt | 0.20-0.30 | Raw audio stimulation |
| `auditory_association` | A4, A5, STSv, STSdp | 0.20-0.30 | Speech/language processing |
| `language_network` | 45 (Broca), STS, TE1a | 0.10-0.15 | Narrative comprehension |
| `multisensory_tpj` | TPOJ1, TPOJ2, TPOJ3 | 0.15-0.25 | Audio-visual integration (up to 50% multimodal gain!) |
| `default_mode` | PGi, TPJ | 0.10-0.20 | Deep engagement / "mind wandering" suppression |

**Subcortical ROI Features (Harvard-Oxford atlas, 8,802 voxels across 8 regions):**

| Feature | Region | Signal |
|---------|--------|--------|
| `emotional_arousal` | Amygdala | Fear, excitement, surprise — emotional punch |
| `memory_formation` | Hippocampus | Memorability — will they remember this reel? |
| `reward_dopamine` | Accumbens (Nucleus) | "I want more" — dopamine/reward circuit |
| `attention_gate` | Thalamus | Attention gating — is the brain paying attention? |
| `action_urge` | Putamen | Motor preparation — urge to act (like/share/comment) |

**Multimodal Integration Feature (from paper Figure 7):**

| Feature | What | Why |
|---------|------|-----|
| `multimodal_score` | Activation at TPO junction when all 3 modalities present | Paper shows up to 50% encoding gain here — reels using text + audio + video together activate this zone most |

**Total: ~20 features per reel**

### Phase 3: Scoring Formula (Week 2-3)
**Build `reelbrain/scorer.py`**

Two approaches, start simple:

**v1 — Weighted formula (ship this first):**
```python
# Temporal scores (how the brain responds over time)
temporal_score = (
    hook_score * 0.35 +        # first 3s matter most (note: 5s hemodynamic delay already handled by model)
    sustain_score * 0.25 +     # no drop-offs
    consistency * 0.15 +       # stable engagement
    dead_spots_penalty * 0.25  # penalize hard dips
)

# Cortical scores (which brain areas light up)
cortical_score = (
    visual_early * 0.20 +      # V1-V4, strongest signal (R=0.40-0.50)
    visual_motion * 0.15 +     # MT complex, motion detection
    auditory_assoc * 0.15 +    # speech/audio processing
    multisensory_tpj * 0.20 +  # multimodal integration (biggest engagement predictor)
    language_network * 0.10 +  # narrative comprehension
    default_mode * 0.20        # deep engagement (suppressed = good)
)

# Subcortical scores (emotional & reward circuits)
subcortical_score = (
    reward_dopamine * 0.30 +   # accumbens — "I want more"
    emotional_arousal * 0.25 + # amygdala — emotional punch
    memory_formation * 0.20 +  # hippocampus — memorability
    attention_gate * 0.15 +    # thalamus — attention
    action_urge * 0.10         # putamen — urge to act
)

# Combined score
score = (temporal_score * 0.40 + cortical_score * 0.35 + subcortical_score * 0.25) * 100
```

Tune weights using your 50-100 labeled reels — run correlation between each feature and actual performance, then weight accordingly.

**v2 — Trained model (after you have 200+ reels):**
- Simple logistic regression or XGBoost
- Input: 12 features, Output: HIT/OK/FLOP probability
- Train on your labeled dataset, cross-validate

### Phase 4: Report Generator (Week 3)
**Build `reelbrain/report.py`**

For each reel, output:
```
REELBRAIN SCORE: 78/100 — SHIP IT

Hook (0-3s):     ████████░░  82  Strong open
Sustain:         ██████░░░░  61  Dip at 5-7s
Audio:           █████████░  94  Good coverage
Visual:          ████████░░  79  Solid
Emotion:         ██████░░░░  63  Moderate

FIXES:
- Weak spot at 5-7s: add a visual transition or text overlay
- Emotional peak is late — move your strongest moment earlier

[brain_heatmap.png attached]
```

### Phase 5: API (Week 3-4)
**Build `reelbrain/api.py`**

See full API section below.

### Phase 6: Client Dashboard (Week 4+)
**Optional — only if clients want it**

Simple web UI: drag-drop reel, see score + report + brain animation.

---

## Where to Run

### GPU Requirements

TRIBE v2 inference loads ~6B params across 3 models:
- V-JEPA2 (ViT-Giant) — needs ~8GB VRAM
- LLaMA 3.2-3B — needs ~6GB VRAM
- Wav2Vec-BERT 2.0 — needs ~2GB VRAM

**Minimum: 24GB VRAM GPU (e.g., RTX 3090, RTX 4090, A10G)**
Models load sequentially (extract features one modality at a time), so they don't all need to fit simultaneously.

### Hosting Options (Cheapest to Most Scalable)

#### Option A: Single GPU Server (Start Here)
**Best for: 0-50 reels/day**

| Provider | GPU | Cost | Notes |
|----------|-----|------|-------|
| **RunPod** | RTX 4090 | ~$0.40/hr | On-demand, pay per hour |
| **Vast.ai** | RTX 3090 | ~$0.20/hr | Cheapest, community GPUs |
| **Lambda Labs** | A10G | ~$0.60/hr | More reliable |
| **Your own RTX 4090** | — | One-time $1,600 | Best if running 24/7 |

Setup:
```bash
# SSH into GPU server
ssh user@gpu-server

# Clone repo + install
git clone <your-repo>
cd reelbrain
pip install -e ".[inference]"

# Download TRIBE v2 weights (one time)
python -c "from tribev2 import TribeModel; TribeModel.from_pretrained('facebook/tribev2')"

# Start API
uvicorn reelbrain.api:app --host 0.0.0.0 --port 8000
```

#### Option B: Serverless GPU (Scale to Demand)
**Best for: 50-500 reels/day, bursty traffic**

| Provider | How It Works | Cost |
|----------|-------------|------|
| **Modal** | Deploy as a function, GPU spins up on request | ~$0.001/sec of GPU |
| **Replicate** | Wrap as a Cog model, get API endpoint | ~$0.0023/sec (A40) |
| **RunPod Serverless** | Auto-scaling GPU workers | ~$0.00040/sec (3090) |

Modal example:
```python
# deploy.py
import modal

app = modal.App("reelbrain")
image = modal.Image.debian_slim().pip_install("tribev2", "fastapi", ...)

@app.cls(gpu="A10G", image=image, container_idle_timeout=300)
class ReelBrainWorker:
    @modal.enter()
    def load_model(self):
        from tribev2 import TribeModel
        self.model = TribeModel.from_pretrained("facebook/tribev2", cache_folder="/cache")

    @modal.method()
    def score(self, video_bytes: bytes) -> dict:
        # save bytes to temp file, run inference, return score
        ...

@app.function()
@modal.web_endpoint(method="POST")
def score_reel(video: bytes):
    worker = ReelBrainWorker()
    return worker.score.remote(video)
```

#### Option C: Cloud GPU Instance (Persistent)
**Best for: 500+ reels/day, need uptime guarantees**

| Provider | Instance | GPU | Cost/month |
|----------|----------|-----|-----------|
| **AWS** | g5.xlarge | A10G (24GB) | ~$750/mo |
| **GCP** | g2-standard-4 | L4 (24GB) | ~$550/mo |
| **Azure** | NC4as_T4_v3 | T4 (16GB) | ~$400/mo (tight on VRAM) |

Use Docker for deployment:
```dockerfile
FROM pytorch/pytorch:2.5.1-cuda12.1-cudnn9-runtime
WORKDIR /app
COPY . .
RUN pip install -e ".[inference]"
RUN python -c "from tribev2 import TribeModel; TribeModel.from_pretrained('facebook/tribev2', cache_folder='/app/cache')"
EXPOSE 8000
CMD ["uvicorn", "reelbrain.api:app", "--host", "0.0.0.0", "--port", "8000"]
```

### Recommendation

```
Phase 1-3 (building + calibrating):  Your local GPU or Vast.ai ($5-10/day)
Phase 4-5 (first clients):           RunPod on-demand ($0.40/hr, turn on when needed)
Phase 6+ (scaling):                   Modal serverless (auto-scales, pay per reel)
```

---

## How to Create the API

### Full API Design

```
reelbrain/
  api.py            # FastAPI app
  worker.py         # Background GPU inference queue
  features.py       # Feature extraction from TRIBE v2 output
  scorer.py         # Score computation
  report.py         # Report generation
  config.py         # Weights, thresholds, ROI definitions
  storage.py        # Save/load results (local disk or S3)
```

### API Endpoints

```
POST   /api/v1/score          Upload MP4, returns job_id (async)
GET    /api/v1/score/{job_id} Poll for results
GET    /api/v1/report/{job_id} Full report with brain images
GET    /api/v1/health         Server + GPU status
POST   /api/v1/batch          Upload multiple reels
GET    /api/v1/history        List past scores for a client
```

### API Flow (Async — Because GPU Inference Takes 1-3 Minutes)

```
Client                          API Server                      GPU Worker
  |                                |                                |
  |-- POST /score (mp4 file) ---->|                                |
  |<-- 202 { job_id: "abc123" } --|                                |
  |                                |-- queue job ------------------>|
  |                                |                                |-- load video
  |                                |                                |-- run TRIBE v2
  |                                |                                |-- extract features
  |                                |                                |-- compute score
  |-- GET /score/abc123 --------->|                                |
  |<-- 200 { status: "processing"}|                                |
  |                                |<-- job done -------------------|
  |-- GET /score/abc123 --------->|                                |
  |<-- 200 { score: 78, ... } ----|                                |
```

### API Code Structure

**`api.py`** — FastAPI application:
```python
from fastapi import FastAPI, UploadFile, HTTPException
from fastapi.responses import FileResponse
import uuid, asyncio
from reelbrain.worker import InferenceQueue
from reelbrain.storage import ResultStore

app = FastAPI(title="ReelBrain", version="0.1.0")
queue = InferenceQueue()
store = ResultStore(path="./results")

@app.post("/api/v1/score")
async def score_reel(file: UploadFile):
    """Upload a reel MP4 and get a job ID back."""
    if not file.filename.endswith(".mp4"):
        raise HTTPException(400, "Only .mp4 files accepted")

    job_id = str(uuid.uuid4())[:8]
    video_path = store.save_upload(job_id, file)
    queue.enqueue(job_id, video_path)

    return {"job_id": job_id, "status": "queued"}

@app.get("/api/v1/score/{job_id}")
async def get_score(job_id: str):
    """Poll for scoring results."""
    result = store.get_result(job_id)
    if result is None:
        raise HTTPException(404, "Job not found")
    return result

@app.get("/api/v1/report/{job_id}")
async def get_report(job_id: str):
    """Get full report with brain heatmap."""
    report = store.get_report(job_id)
    if report is None:
        raise HTTPException(404, "Report not found")
    return report

@app.get("/api/v1/report/{job_id}/heatmap")
async def get_heatmap(job_id: str):
    """Get brain heatmap image."""
    path = store.get_heatmap_path(job_id)
    if path is None:
        raise HTTPException(404, "Heatmap not found")
    return FileResponse(path, media_type="image/png")

@app.get("/api/v1/health")
async def health():
    return {
        "status": "ok",
        "gpu_available": torch.cuda.is_available(),
        "queue_size": queue.size(),
        "model_loaded": queue.model_loaded,
    }
```

**`worker.py`** — Background GPU inference:
```python
import threading, queue, torch, numpy as np
from tribev2 import TribeModel
from reelbrain.features import extract_features
from reelbrain.scorer import compute_score
from reelbrain.report import generate_report
from reelbrain.storage import ResultStore

class InferenceQueue:
    def __init__(self):
        self._queue = queue.Queue()
        self._model = None
        self._thread = threading.Thread(target=self._worker, daemon=True)
        self._thread.start()

    @property
    def model_loaded(self):
        return self._model is not None

    def size(self):
        return self._queue.qsize()

    def enqueue(self, job_id: str, video_path: str):
        self._queue.put((job_id, video_path))

    def _load_model(self):
        if self._model is None:
            self._model = TribeModel.from_pretrained(
                "facebook/tribev2", cache_folder="./cache"
            )

    def _worker(self):
        self._load_model()
        store = ResultStore(path="./results")

        while True:
            job_id, video_path = self._queue.get()
            store.update_status(job_id, "processing")

            try:
                # Step 1: Get events from video
                events = self._model.get_events_dataframe(video_path=video_path)

                # Step 2: Run TRIBE v2 inference
                preds, segments = self._model.predict(events)

                # Step 3: Extract features
                features = extract_features(preds, segments, video_path)

                # Step 4: Compute score
                score, breakdown = compute_score(features)

                # Step 5: Generate report + brain heatmap
                report = generate_report(score, breakdown, features, preds)

                # Step 6: Save everything
                store.save_result(job_id, {
                    "status": "done",
                    "score": score,
                    "grade": "SHIP IT" if score >= 70 else "FIX IT" if score >= 40 else "KILL IT",
                    "breakdown": breakdown,
                    "features": features,
                    "report": report,
                })
                np.save(store.path / job_id / "preds.npy", preds)

            except Exception as e:
                store.save_result(job_id, {
                    "status": "error",
                    "error": str(e),
                })
```

### Client Usage

**cURL:**
```bash
# Upload reel
curl -X POST http://your-server:8000/api/v1/score \
  -F "file=@my_reel.mp4"
# → {"job_id": "a1b2c3d4", "status": "queued"}

# Poll for result (wait 1-3 minutes)
curl http://your-server:8000/api/v1/score/a1b2c3d4
# → {"status": "done", "score": 78, "grade": "SHIP IT", ...}

# Get brain heatmap
curl http://your-server:8000/api/v1/report/a1b2c3d4/heatmap -o brain.png
```

**Python SDK (for clients):**
```python
import requests, time

class ReelBrain:
    def __init__(self, api_url):
        self.url = api_url

    def score(self, video_path: str, timeout: int = 300) -> dict:
        # Upload
        with open(video_path, "rb") as f:
            r = requests.post(f"{self.url}/api/v1/score", files={"file": f})
        job_id = r.json()["job_id"]

        # Poll
        for _ in range(timeout // 5):
            r = requests.get(f"{self.url}/api/v1/score/{job_id}")
            result = r.json()
            if result["status"] in ("done", "error"):
                return result
            time.sleep(5)

        return {"status": "timeout"}

# Usage
rb = ReelBrain("http://your-server:8000")
result = rb.score("my_reel.mp4")
print(f"Score: {result['score']}/100 — {result['grade']}")
```

---

## Key Paper Findings That Power This Product

Source: "A foundation model of vision, audition, and language for in-silico neuroscience" (d'Ascoli et al., 2026)

### Why TRIBE v2 predictions are trustworthy
- **1st place** out of 263 teams in Algonauts 2025 brain prediction competition
- Trained on **1,000+ hours of fMRI** across **720 subjects**
- Predicts **20,484 cortical vertices** + **8,802 subcortical voxels**
- Zero-shot group prediction **R ≈ 0.4** on HCP 7T dataset (better than most individual subjects)
- Log-linear scaling — **performance ceiling not yet reached**, will improve with more data

### Modality map (Figure 7 — critical for scoring)
The paper maps which modalities drive which brain regions:
- **Video alone** → occipital + parietal cortex (visual processing)
- **Audio alone** → temporal cortex (auditory processing)
- **Text alone** → prefrontal lobe + language areas (semantic/narrative)
- **Text + Audio** → superior temporal lobe (speech comprehension)
- **Video + Audio** → ventral/dorsal visual cortex + hippocampus (memory!)
- **All three combined** → TPO junction gets **50% encoding boost** over best single modality

**Implication for reels**: Content that combines visual action + speech/music + on-screen text will activate the broadest brain response. This is the "multimodal engagement multiplier."

### The 5 ICA components (Figure 6)
TRIBE v2's internal representations decompose into 5 interpretable networks:
1. **Primary auditory cortex** — raw sound processing
2. **Language network** — speech comprehension
3. **Motion detection area** — visual movement
4. **Default mode network** — internal thought (suppressed during engagement!)
5. **Visual system** — visual processing

These 5 networks can each be scored independently for a richer analysis.

### In-silico experiments work (Figures 4-5)
The paper validates that TRIBE v2 recovers known neuroscience findings:
- **Fusiform Face Area (FFA)** lights up for faces
- **Parahippocampal Place Area (PPA)** lights up for places
- **Extrastriate Body Area (EBA)** lights up for bodies
- **Visual Word Form Area (VWFA)** lights up for text
- **Broca's area (45)** lights up for complex sentences
- **TPJ** lights up for emotional content

This means: if a reel shows faces → we can verify FFA activation. If it has text overlays → we can check VWFA. The predictions are grounded in real neuroscience.

---

## Files to Reuse from TRIBE v2

| What | File | Function |
|------|------|----------|
| Run inference | `tribev2/demo_utils.py` | `TribeModel.from_pretrained()`, `.predict()` |
| ROI extraction | `tribev2/utils.py` | `get_hcp_roi_indices()`, `summarize_by_roi()` |
| Brain heatmaps | `tribev2/plotting/` | `PlotBrainNilearn.plot_surf()` |
| Surface mesh | `tribev2/utils_fmri.py` | `TribeSurfaceProjector` |

---

## Key Decisions

1. **No LLM in the scoring loop** — deterministic formula is faster, cheaper, and explainable
2. **Start with weighted formula** — upgrade to ML model after 200+ labeled reels
3. **Async API** — GPU inference takes 1-3 min per reel, can't block the request
4. **Ship the API before the dashboard** — clients can integrate or use cURL/Python SDK

---

## Verification Plan

1. Run scorer on your 4 existing brain scans — confirm the 3 hits score high, 1 flop scores low
2. Run on 50 labeled reels — check correlation between score and actual engagement
3. Tune weights until score correlates >0.5 with completion rate
4. Give 5 clients access, have them score reels before posting for 2 weeks
5. Compare scored vs unscored reel performance

---

## Scaling Path

```
Now:         Weighted formula, 50 labeled reels, RunPod on-demand
100 reels:   Tune weights with real correlation data
200 reels:   Train XGBoost classifier (HIT/OK/FLOP)
500 reels:   Per-niche models (tech vs entertainment vs education)
1000 reels:  Fine-tune TRIBE v2 on short-form vertical video
5000 reels:  Custom brain model trained specifically on reels
```

---

## Cost Estimate Per Reel

| Component | Cost |
|-----------|------|
| GPU inference (1-3 min on A10G) | $0.01-0.03 |
| Storage (preds + video) | ~$0.001 |
| API hosting (shared) | ~$0.001 |
| **Total per reel** | **~$0.03-0.05** |

At $5-10 per reel to clients, that's **99% gross margin**.
