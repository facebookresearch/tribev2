# dashboard/backend/app/main.py
from fastapi import FastAPI, UploadFile, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, Response
from pydantic import BaseModel

from .storage import (
    STORAGE_MODE,
    presigned_upload_url,
    presigned_download_url,
    get_local_file_path,
    LOCAL_DATA_DIR,
)
from .mesh import get_fsaverage5_mesh_binary
from .predict import start_prediction, get_job, list_jobs, get_atlas_data, load_manifest

app = FastAPI(title="neuroLoop API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["X-N-Vertices", "X-N-Faces"],
)


@app.on_event("startup")
def _warmup():
    """Pre-load heavy resources in a background thread so first request is fast."""
    import threading

    def _load():
        from .predict import _get_model, _get_atlas, get_atlas_data
        from .mesh import get_fsaverage5_mesh_binary
        get_fsaverage5_mesh_binary()
        get_atlas_data()
        _get_model()

    load_manifest()
    threading.Thread(target=_load, daemon=True).start()


# ------------------------------------------------------------------
# Upload
# ------------------------------------------------------------------

class UploadRequest(BaseModel):
    filename: str
    content_type: str

@app.post("/api/upload")
def upload(req: UploadRequest):
    return presigned_upload_url(req.filename, req.content_type)


@app.put("/api/upload/file/{key:path}")
async def upload_file_local(key: str, request: Request):
    """Direct file upload for local storage mode."""
    body = await request.body()
    dest = LOCAL_DATA_DIR / key
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_bytes(body)
    return {"status": "ok", "key": key}


# ------------------------------------------------------------------
# Local file serving (only used in local storage mode)
# ------------------------------------------------------------------

@app.get("/api/files/{key:path}")
def serve_file(key: str):
    """Serve a locally stored file. Only active in local mode."""
    path = get_local_file_path(key)
    if not path.exists():
        return {"error": "not found"}, 404
    return FileResponse(str(path))


# ------------------------------------------------------------------
# Download results as zip (local mode convenience)
# ------------------------------------------------------------------

@app.get("/api/download/{job_id}")
def download_results(job_id: str):
    """Zip and serve all results for a job. Works in both modes."""
    import tempfile, zipfile, io
    from .storage import download_file, list_prefix

    prefix = f"results/{job_id}"
    keys = list_prefix(prefix)
    if not keys:
        return {"error": "no results found"}

    if STORAGE_MODE == "local":
        # Zip directly from local files
        zip_buf = io.BytesIO()
        with zipfile.ZipFile(zip_buf, "w", zipfile.ZIP_DEFLATED) as zf:
            for key in keys:
                path = get_local_file_path(key)
                arcname = key.split("/", 2)[-1] if "/" in key else key
                zf.write(str(path), arcname)
        zip_buf.seek(0)

        # Write to temp file for FileResponse
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".zip")
        tmp.write(zip_buf.getvalue())
        tmp.close()
        return FileResponse(
            tmp.name,
            media_type="application/zip",
            filename=f"neuroloop-{job_id}.zip",
        )
    else:
        # S3 mode — download to temp, zip, serve
        zip_buf = io.BytesIO()
        with zipfile.ZipFile(zip_buf, "w", zipfile.ZIP_DEFLATED) as zf:
            for key in keys:
                with tempfile.NamedTemporaryFile() as tmp:
                    download_file(key, tmp.name)
                    arcname = key.split("/", 2)[-1] if "/" in key else key
                    zf.write(tmp.name, arcname)
        zip_buf.seek(0)

        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".zip")
        tmp.write(zip_buf.getvalue())
        tmp.close()
        return FileResponse(
            tmp.name,
            media_type="application/zip",
            filename=f"neuroloop-{job_id}.zip",
        )


# ------------------------------------------------------------------
# Other endpoints
# ------------------------------------------------------------------

@app.get("/api/health")
def health():
    return {"status": "ok", "storage_mode": STORAGE_MODE}

@app.get("/api/mesh")
def mesh():
    buf, n_vertices, n_faces = get_fsaverage5_mesh_binary()
    return Response(
        content=buf,
        media_type="application/octet-stream",
        headers={
            "X-N-Vertices": str(n_vertices),
            "X-N-Faces": str(n_faces),
        },
    )


@app.get("/api/atlas")
def atlas():
    """Static region data (vertex indices, group lookups). Cached client-side."""
    return get_atlas_data()


class PredictRequest(BaseModel):
    s3_key: str
    input_type: str  # "video", "audio", "text"

@app.post("/api/predict")
def predict(req: PredictRequest):
    job_id = start_prediction(req.s3_key, req.input_type)
    return {"job_id": job_id}

@app.get("/api/results/{job_id}")
def results(job_id: str):
    job = get_job(job_id)
    if job is None:
        return {"status": "not_found"}
    if job["status"] == "processing":
        return {"status": "processing", "progress": job["progress"]}
    if job["status"] == "error":
        return {"status": "error", "error": job.get("error", "unknown")}
    # done
    prefix = job["results_prefix"]
    return {
        "status": "done",
        "preds_url": presigned_download_url(f"{prefix}/preds.bin"),
        "regions_url": presigned_download_url(f"{prefix}/regions.json"),
        "meta_url": presigned_download_url(f"{prefix}/meta.json"),
        "meta": job.get("meta_cache", {}),
    }

@app.get("/api/runs")
def runs():
    return {"runs": list_jobs()}
