# dashboard/backend/app/predict.py
import json
import tempfile
import uuid
from pathlib import Path
from datetime import datetime, timezone

import numpy as np

from . import storage

# Global job store (in-memory, single instance)
_jobs: dict[str, dict] = {}

def get_job(job_id: str) -> dict | None:
    return _jobs.get(job_id)

def list_jobs() -> list[dict]:
    return [
        {
            "job_id": j["job_id"],
            "filename": j["filename"],
            "timestamp": j["timestamp"],
            "status": j["status"],
            "n_timesteps": j.get("n_timesteps"),
        }
        for j in sorted(_jobs.values(), key=lambda x: x["timestamp"], reverse=True)
    ]

def start_prediction(s3_key: str, input_type: str) -> str:
    """Start a prediction job in background. Returns job_id."""
    import threading

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

        # Download media from S3
        with tempfile.TemporaryDirectory() as tmpdir:
            local_path = str(Path(tmpdir) / job["filename"])
            storage.download_file(job["s3_key"], local_path)
            job["progress"] = 0.2

            # Load model (lazy — first call is slow)
            from tribev2 import TribeModel
            model = _get_model()
            job["progress"] = 0.4

            # Build events and predict
            input_type = job["input_type"]
            if input_type == "video":
                events = model.get_events_dataframe(video_path=local_path)
            elif input_type == "audio":
                events = model.get_events_dataframe(audio_path=local_path)
            elif input_type == "text":
                events = model.get_events_dataframe(text_path=local_path)
            else:
                raise ValueError(f"Unknown input_type: {input_type}")

            job["progress"] = 0.6
            preds, segments = model.predict(events=events, verbose=False)
            job["progress"] = 0.8

            # Run neuroLoop region analysis
            from neuroLoop import BrainAtlas
            atlas = BrainAtlas()
            region_df = atlas.all_region_timeseries(preds)
            regions_dict = {col: region_df[col].tolist() for col in region_df.columns}

            # Build region -> vertex indices mapping (for frontend highlighting)
            region_vertices = {
                name: [int(v) for v in verts]
                for name, verts in atlas.labels.items()
            }

            # Build group lookup for frontend (region_name -> fine group name)
            from neuroLoop.regions import FINE_GROUPS, COARSE_GROUPS
            region_to_fine = {}
            for group, members in FINE_GROUPS.items():
                for r in members:
                    region_to_fine[r] = group
            region_to_coarse = {}
            for group, members in COARSE_GROUPS.items():
                for r in members:
                    region_to_coarse[r] = group

            # Extract segment timestamps for temporal alignment
            segment_times = [
                {"start": float(s.start), "duration": float(s.duration)}
                for s in segments
            ]
            duration_seconds = (
                segment_times[-1]["start"] + segment_times[-1]["duration"]
                if segment_times else 0.0
            )

            # Save results to S3
            prefix = f"results/{job_id}"

            # preds as raw float32 binary (no numpy header to parse)
            preds_f32 = preds.astype(np.float32)
            storage.upload_bytes(
                preds_f32.tobytes(),
                f"{prefix}/preds.bin",
                content_type="application/octet-stream",
            )

            # Compute global min/max for consistent colorscale
            global_vmin = float(np.percentile(preds, 1))
            global_vmax = float(np.percentile(preds, 99))

            # regions + group lookup as JSON
            regions_payload = {
                "regions": regions_dict,
                "fine_groups": region_to_fine,
                "coarse_groups": region_to_coarse,
                "region_vertices": region_vertices,
            }
            storage.upload_bytes(
                json.dumps(regions_payload).encode(),
                f"{prefix}/regions.json",
                content_type="application/json",
            )

            # metadata
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
            storage.upload_bytes(
                json.dumps(meta).encode(),
                f"{prefix}/meta.json",
                content_type="application/json",
            )

            job["n_timesteps"] = meta["n_timesteps"]
            job["meta_cache"] = meta
            job["status"] = "done"
            job["progress"] = 1.0
            job["results_prefix"] = prefix

    except Exception as e:
        import traceback
        traceback.print_exc()
        job["status"] = "error"
        job["error"] = str(e)


_model_cache = None

def _get_model():
    global _model_cache
    if _model_cache is None:
        from tribev2 import TribeModel
        _model_cache = TribeModel.from_pretrained("facebook/tribev2", cache_folder="./cache")
    return _model_cache
