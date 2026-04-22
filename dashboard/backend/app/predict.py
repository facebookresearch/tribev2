# dashboard/backend/app/predict.py
import json
import logging
import tempfile
import threading
import uuid
from pathlib import Path
from datetime import datetime, timezone

import numpy as np
import torch

from . import storage

logger = logging.getLogger(__name__)

# Global job store (in-memory, single instance)
_jobs: dict[str, dict] = {}

# ------------------------------------------------------------------
# Manifest — persists completed jobs across server restarts
# ------------------------------------------------------------------

_MANIFEST_KEY = "manifest.json"
_manifest_lock = threading.Lock()


def _read_manifest_raw() -> list[dict]:
    """Read manifest from storage. Returns empty list if not found."""
    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = str(Path(tmpdir) / "manifest.json")
            storage.download_file(_MANIFEST_KEY, tmp_path)
            return json.loads(Path(tmp_path).read_text())
    except Exception:
        return []


def _append_to_manifest(entry: dict) -> None:
    """Thread-safe append of a completed job entry to the manifest."""
    with _manifest_lock:
        entries = _read_manifest_raw()
        entries.append(entry)
        storage.upload_bytes(
            json.dumps(entries, indent=2).encode(),
            _MANIFEST_KEY,
            "application/json",
        )


def load_manifest() -> None:
    """Populate _jobs with completed jobs from the manifest on startup."""
    entries = _read_manifest_raw()
    for entry in entries:
        job_id = entry["job_id"]
        if job_id not in _jobs:
            _jobs[job_id] = {**entry, "status": "done", "progress": 1.0}
    logger.info("Loaded %d jobs from manifest", len(entries))


# ------------------------------------------------------------------
# Thumbnail extraction
# ------------------------------------------------------------------

def _extract_thumbnail(local_path: str, job_id: str, input_type: str) -> str | None:
    """Extract a JPEG thumbnail from a video file and upload to storage.

    Returns the storage key on success, None for non-video inputs or on failure.
    """
    if input_type != "video":
        return None
    try:
        import subprocess
        with tempfile.TemporaryDirectory() as thumb_dir:
            thumb_path = str(Path(thumb_dir) / "thumbnail.jpg")
            result = subprocess.run(
                [
                    "ffmpeg", "-ss", "1", "-i", local_path,
                    "-vframes", "1", "-q:v", "4", "-y", thumb_path,
                ],
                capture_output=True,
                timeout=30,
            )
            if result.returncode != 0 or not Path(thumb_path).exists():
                logger.warning("ffmpeg thumbnail failed for %s: %s", job_id, result.stderr.decode())
                return None
            key = f"results/{job_id}/thumbnail.jpg"
            storage.upload_bytes(Path(thumb_path).read_bytes(), key, "image/jpeg")
            return key
    except Exception as exc:
        logger.warning("Thumbnail extraction skipped for %s: %s", job_id, exc)
        return None

def get_job(job_id: str) -> dict | None:
    return _jobs.get(job_id)

def list_jobs() -> list[dict]:
    return [
        {
            "job_id": j["job_id"],
            "filename": j["filename"],
            "input_type": j.get("input_type"),
            "timestamp": j["timestamp"],
            "status": j["status"],
            "duration_seconds": j.get("duration_seconds"),
            "n_timesteps": j.get("n_timesteps"),
            "thumbnail_url": (
                storage.presigned_download_url(j["thumbnail_key"])
                if j.get("thumbnail_key") else None
            ),
        }
        for j in sorted(_jobs.values(), key=lambda x: x["timestamp"], reverse=True)
    ]

def start_prediction(s3_key: str, input_type: str) -> str:
    """Start a prediction job in background. Returns job_id."""
    job_id = f"job_{uuid.uuid4().hex[:8]}"
    filename = s3_key.split("/")[-1]
    _jobs[job_id] = {
        "job_id": job_id,
        "s3_key": s3_key,
        "input_type": input_type,
        "filename": filename,
        "status": "processing",
        "progress": 0.0,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    thread = threading.Thread(target=_run_prediction, args=(job_id,), daemon=True)
    thread.start()
    return job_id

def _run_prediction(job_id: str) -> None:
    job = _jobs[job_id]
    try:
        job["progress"] = 0.1

        # Download media from storage
        with tempfile.TemporaryDirectory() as tmpdir:
            local_path = str(Path(tmpdir) / job["filename"])
            storage.download_file(job["s3_key"], local_path)
            job["progress"] = 0.2

            # Load model (lazy — first call is slow)
            model = _get_model()
            job["progress"] = 0.4

            # Build events and predict
            input_type = job["input_type"]
            event_kwarg = {"video": "video_path", "audio": "audio_path", "text": "text_path"}
            if input_type not in event_kwarg:
                raise ValueError(f"Unknown input_type: {input_type}")
            events = model.get_events_dataframe(**{event_kwarg[input_type]: local_path})

            job["progress"] = 0.6

            # Use mixed precision (fp16) on CUDA for ~1.5-2x faster inference
            use_amp = torch.cuda.is_available() and model._model is not None
            if use_amp:
                # Monkey-patch predict to use autocast around the model forward pass
                _orig_forward = model._model.forward
                def _amp_forward(*args, **kwargs):
                    with torch.autocast("cuda", dtype=torch.float16):
                        return _orig_forward(*args, **kwargs)
                model._model.forward = _amp_forward

            preds, segments = model.predict(events=events, verbose=False)

            if use_amp:
                model._model.forward = _orig_forward

            job["progress"] = 0.8

            # Run neuroLoop region analysis
            atlas = _get_atlas()
            region_df = atlas.all_region_timeseries(preds)
            regions_dict = {col: region_df[col].tolist() for col in region_df.columns}

            # Extract segment timestamps for temporal alignment
            segment_times = [
                {"start": float(s.start), "duration": float(s.duration)}
                for s in segments
            ]
            duration_seconds = (
                segment_times[-1]["start"] + segment_times[-1]["duration"]
                if segment_times else 0.0
            )

            # Save results to storage
            prefix = f"results/{job_id}"

            # Prepare all payloads before uploading
            if preds.dtype != np.float32:
                preds = preds.astype(np.float32)
            preds_bytes = preds.tobytes()

            global_vmin, global_vmax = np.percentile(preds, [1, 99]).tolist()

            regions_bytes = json.dumps({"regions": regions_dict}).encode()

            meta = {
                "job_id": job_id,
                "filename": job["filename"],
                "input_type": input_type,
                "n_timesteps": int(preds.shape[0]),
                "n_vertices": int(preds.shape[1]),
                "duration_seconds": duration_seconds,
                "segment_times": segment_times,
                "hemodynamic_lag": 5.0,
                "global_vmin": global_vmin,
                "global_vmax": global_vmax,
                "timestamp": job["timestamp"],
            }
            meta_bytes = json.dumps(meta).encode()

            # Extract thumbnail while local file is still available
            thumbnail_key = _extract_thumbnail(local_path, job_id, input_type)

            # Upload all result files in parallel (significant for S3 mode)
            from concurrent.futures import ThreadPoolExecutor
            with ThreadPoolExecutor(max_workers=3) as pool:
                pool.submit(storage.upload_bytes, preds_bytes, f"{prefix}/preds.bin", "application/octet-stream")
                pool.submit(storage.upload_bytes, regions_bytes, f"{prefix}/regions.json", "application/json")
                pool.submit(storage.upload_bytes, meta_bytes, f"{prefix}/meta.json", "application/json")

            job["n_timesteps"] = meta["n_timesteps"]
            job["meta_cache"] = meta
            job["status"] = "done"
            job["progress"] = 1.0
            job["results_prefix"] = prefix
            job["thumbnail_key"] = thumbnail_key

            _append_to_manifest({
                "job_id": job_id,
                "filename": job["filename"],
                "input_type": input_type,
                "timestamp": job["timestamp"],
                "results_prefix": prefix,
                "n_timesteps": meta["n_timesteps"],
                "duration_seconds": duration_seconds,
                "thumbnail_key": thumbnail_key,
            })

    except Exception as e:
        import traceback
        traceback.print_exc()
        job["status"] = "error"
        job["error"] = str(e)


# ---------------------------------------------------------------------------
# Cached singletons (model, atlas)
# ---------------------------------------------------------------------------

_model_cache = None

def _get_model():
    global _model_cache
    if _model_cache is None:
        from tribev2 import TribeModel
        _model_cache = TribeModel.from_pretrained("facebook/tribev2", cache_folder="./cache")

        # Compile the model for faster inference (PyTorch 2.x)
        if hasattr(torch, "compile"):
            try:
                _model_cache._model = torch.compile(_model_cache._model, mode="reduce-overhead")
                logger.info("torch.compile applied to TRIBE model")
            except Exception as exc:
                logger.warning("torch.compile failed, running uncompiled: %s", exc)

    return _model_cache


_atlas_cache_obj = None

def _get_atlas():
    """Cached BrainAtlas instance — reused across jobs."""
    global _atlas_cache_obj
    if _atlas_cache_obj is None:
        from neuroLoop import BrainAtlas
        _atlas_cache_obj = BrainAtlas()
        # Warm up cached properties so first job doesn't pay the cost
        _ = _atlas_cache_obj.labels
        _ = _atlas_cache_obj._indicator_matrix
    return _atlas_cache_obj


_atlas_data_cache = None

def get_atlas_data() -> dict:
    """Static atlas data (region vertices + group lookups). Cached after first call."""
    global _atlas_data_cache
    if _atlas_data_cache is not None:
        return _atlas_data_cache

    from neuroLoop.regions import FINE_GROUPS, COARSE_GROUPS

    atlas = _get_atlas()
    region_vertices = {
        name: [int(v) for v in verts]
        for name, verts in atlas.labels.items()
    }
    region_to_fine = {}
    for group, members in FINE_GROUPS.items():
        for r in members:
            region_to_fine[r] = group
    region_to_coarse = {}
    for group, members in COARSE_GROUPS.items():
        for r in members:
            region_to_coarse[r] = group

    _atlas_data_cache = {
        "region_vertices": region_vertices,
        "fine_groups": region_to_fine,
        "coarse_groups": region_to_coarse,
    }
    return _atlas_data_cache
