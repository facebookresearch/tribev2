# neuroLoop Dashboard — Design Spec

**Date**: 2026-04-13
**Status**: Approved

---

## Goal

A browser-based dashboard that lets a user upload video/audio/text, run TRIBE v2 inference on a Lambda Cloud GPU, and view the results as a synchronized 3D brain visualization with ranked region activation scores.

## Architecture

```
┌─────────────┐     ┌──────────────────┐     ┌─────────┐
│  React App  │────▶│  FastAPI Server   │────▶│  AWS S3  │
│  (Browser)  │◀────│  (Lambda Cloud)   │◀────│         │
│             │     │  GPU + TRIBE v2   │     │         │
└─────────────┘     └──────────────────┘     └─────────┘
```

- **Frontend**: React + Vite, served statically or from the same Lambda instance
- **Backend**: FastAPI running on the Lambda Cloud GPU instance (same machine as the model)
- **Storage**: AWS S3 for uploaded media, prediction results, and cached mesh geometry

## Data Flow

1. **Upload**: User drops a video/audio file or pastes text. Frontend uploads the file to S3 via a presigned URL from the backend.
2. **Process**: Frontend sends the S3 key to `POST /api/predict`. Backend downloads the file, runs TRIBE v2 inference (produces `(n_timesteps, 20484)` predictions), runs `BrainAtlas.all_region_timeseries()` for region scores, and saves both arrays + metadata to S3.
3. **View**: Frontend fetches prediction results from S3. Loads fsaverage5 mesh geometry once (cached). Colors mesh vertices per timestep using Three.js. Syncs brain visualization with video playback via a shared timeline.

## Layout

Side-by-side + bottom panel layout:

```
┌──────────────────────┬──────────────────────┐
│                      │                      │
│   Video Playback     │   3D Brain Viewer    │
│                      │   (Three.js)         │
│                      │                      │
├─────────────────────────────────────────────┤
│  ◀ ▶  ━━━━━━━━━━●━━━━━━━━━━  0:18 / 0:52   │
├─────────────────────────────────────────────┤
│  TOP ACTIVATED REGIONS @ t=18s              │
│  V1 (Primary Visual)    ████████████  0.92  │
│  MT (Motion)            ██████████▒   0.87  │
│  V2 (Early Visual)      █████████▒▒   0.81  │
│  A1 (Auditory)          ████████▒▒▒   0.73  │
│  STSdp (Language)       ███████▒▒▒▒   0.68  │
└─────────────────────────────────────────────┘
```

### Panels

1. **Top-left: Video/Text playback** — HTML5 `<video>` for video/audio, or scrolling text display for text input. Synced to the shared timeline.

2. **Top-right: 3D brain viewer** — Three.js scene with the fsaverage5 cortical mesh. Vertex colors updated each timestep from prediction data. User can rotate/zoom (OrbitControls). Activation values mapped to a colorscale (cold → hot).

3. **Middle: Timeline scrubber** — Shared timeline that controls both video playback and brain activation timestep. Play/pause button. Draggable scrubber. Current time display.

4. **Bottom: Region scores** — Ranked list of top activated HCP-MMP1 regions at the current timestep. Each row shows region name, functional group label (from neuroLoop), a bar showing relative activation, and the numeric score. Updates live as the user scrubs the timeline. Sorted by activation value, descending.

### Top bar

- **neuroLoop** logo/title on the left
- **Upload Video** and **Paste Text** buttons on the right
- Upload triggers a file picker dialog; text opens a modal with a textarea

## API Endpoints

All endpoints served by FastAPI on the Lambda Cloud GPU instance.

### `POST /api/upload`

Returns a presigned S3 URL for the client to upload media directly to S3.

**Request**: `{ "filename": "clip.mp4", "content_type": "video/mp4" }`
**Response**: `{ "upload_url": "https://s3...", "s3_key": "uploads/abc123/clip.mp4" }`

### `POST /api/predict`

Starts a prediction job. Runs TRIBE v2 on the uploaded media.

**Request**: `{ "s3_key": "uploads/abc123/clip.mp4", "input_type": "video" }`
**Response**: `{ "job_id": "job_xyz" }`

Backend processing:
1. Download media from S3
2. Run `model.get_events_dataframe()` and `model.predict()`
3. Run `BrainAtlas().all_region_timeseries(preds)` and `BrainAtlas().all_group_timeseries(preds, "coarse")`
4. Save to S3:
   - `results/{job_id}/preds.npy` — raw vertex predictions `(n_timesteps, 20484)`
   - `results/{job_id}/regions.json` — region timeseries as `{ region_name: [values] }`
   - `results/{job_id}/meta.json` — input filename, duration, n_timesteps, timestamp

### `GET /api/results/{job_id}`

Poll job status and get result URLs.

**Response (processing)**: `{ "status": "processing", "progress": 0.4 }`
**Response (done)**: `{ "status": "done", "preds_url": "https://s3...", "regions_url": "https://s3...", "meta": { ... } }`

### `GET /api/mesh`

Returns the fsaverage5 mesh geometry. Cached — only needs to be fetched once per session.

**Response**: `{ "vertices": [[x,y,z], ...], "faces": [[i,j,k], ...], "n_vertices": 20484 }`

Vertices and faces are extracted from nilearn's `fetch_surf_fsaverage("fsaverage5")` and served as JSON (or a binary buffer for performance).

### `GET /api/runs`

List past prediction runs stored in S3.

**Response**: `{ "runs": [{ "job_id": "...", "filename": "clip.mp4", "timestamp": "...", "n_timesteps": 52 }, ...] }`

## Tech Stack

### Frontend

| Package | Purpose |
|---------|---------|
| React 19 | UI framework |
| Vite | Build tool / dev server |
| three / @react-three/fiber | 3D brain rendering |
| @react-three/drei | OrbitControls, helpers |
| Tailwind CSS | Styling |
| Zustand | State management (shared timeline, predictions) |

### Backend

| Package | Purpose |
|---------|---------|
| FastAPI | API server |
| uvicorn | ASGI server |
| tribev2 | TRIBE v2 inference |
| neuroLoop | Region analysis (BrainAtlas) |
| boto3 | AWS S3 client |
| nilearn | Mesh geometry extraction |

### Infrastructure

- **Compute**: Lambda Cloud GPU instance (model inference + API server)
- **Storage**: AWS S3 bucket
- **Frontend hosting**: Served from the same Lambda instance (or any static host)

## File Structure

```
neuroLoop/
├── dashboard/
│   ├── frontend/
│   │   ├── src/
│   │   │   ├── App.jsx                 Main layout
│   │   │   ├── components/
│   │   │   │   ├── BrainViewer.jsx     Three.js brain mesh + coloring
│   │   │   │   ├── VideoPlayer.jsx     HTML5 video synced to timeline
│   │   │   │   ├── TextDisplay.jsx     Scrolling text for text input
│   │   │   │   ├── Timeline.jsx        Shared scrubber + play/pause
│   │   │   │   ├── RegionPanel.jsx     Ranked region activation list
│   │   │   │   ├── UploadModal.jsx     File upload / text paste UI
│   │   │   │   └── TopBar.jsx          Nav bar with upload buttons
│   │   │   ├── stores/
│   │   │   │   └── useStore.js         Zustand store (timeline, preds)
│   │   │   └── utils/
│   │   │       └── colorscale.js       Activation value → vertex color
│   │   ├── package.json
│   │   ├── vite.config.js
│   │   └── tailwind.config.js
│   │
│   └── backend/
│       ├── app/
│       │   ├── main.py                 FastAPI app + endpoints
│       │   ├── predict.py              TRIBE v2 inference wrapper
│       │   ├── mesh.py                 Mesh geometry extraction + caching
│       │   └── s3.py                   S3 upload/download helpers
│       └── requirements.txt
```

## Brain Rendering Details

The 3D brain is rendered using Three.js with react-three-fiber:

1. **Mesh geometry**: Load fsaverage5 vertices (20,484 × 3) and faces from `/api/mesh` once. Create a `BufferGeometry` with position and index attributes.

2. **Vertex coloring**: Store all prediction timesteps in a Float32Array. On each frame (or when the user scrubs), update the `color` attribute of the BufferGeometry by mapping activation values through a colorscale (e.g., viridis or hot). Both hemispheres displayed — left hemisphere vertices 0–10,241, right hemisphere 10,242–20,483.

3. **Controls**: OrbitControls for rotate/zoom/pan. Optional: preset views (left, right, top, medial).

4. **Performance**: The mesh is static geometry with only the color buffer changing per frame. This is fast — just a buffer upload, no geometry rebuild. Should handle 60fps easily.

## Region Panel Details

The bottom panel shows a ranked list from `BrainAtlas.all_region_timeseries()`:

- All 360 region scores are loaded into the frontend store
- At each timestep, sort by activation value descending
- Display top ~10 regions with:
  - Region name (e.g., "V1")
  - Group name from neuroLoop (e.g., "Primary Visual" at fine level)
  - Horizontal bar proportional to the score
  - Numeric score value
- Group name coloring: color-code by coarse Yeo network (Visual = red, Somatomotor = blue, etc.)
- Updates instantly on timeline scrub — no API call needed, all data is client-side

## Error Handling

- **Upload fails**: Show toast with error message, allow retry
- **Prediction fails**: Show error in UI with the failure reason from the backend
- **S3 unreachable**: Backend returns 503, frontend shows connectivity error
- **Model not loaded**: First prediction may take longer (model loading). Backend returns progress updates via polling.
- **Large files**: Set a reasonable upload limit (500MB). Reject with clear message.

## Scope Boundary

**In scope**: Upload media, run inference, view 3D brain + video side by side, see region scores, scrub timeline, list past runs.

**Out of scope** (for now): User auth, multi-user support, batch processing, real-time streaming, export/download of results, comparison of multiple runs, region filtering/search.
