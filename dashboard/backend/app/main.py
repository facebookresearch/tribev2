# dashboard/backend/app/main.py
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from .s3 import presigned_upload_url, presigned_download_url
from .mesh import get_fsaverage5_mesh
from .predict import start_prediction, get_job, list_jobs

app = FastAPI(title="neuroLoop API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

class UploadRequest(BaseModel):
    filename: str
    content_type: str

@app.post("/api/upload")
def upload(req: UploadRequest):
    return presigned_upload_url(req.filename, req.content_type)

@app.get("/api/health")
def health():
    return {"status": "ok"}

@app.get("/api/mesh")
def mesh():
    return get_fsaverage5_mesh()

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
    meta_url = presigned_download_url(f"{prefix}/meta.json")
    return {
        "status": "done",
        "preds_url": presigned_download_url(f"{prefix}/preds.bin"),
        "regions_url": presigned_download_url(f"{prefix}/regions.json"),
        "meta_url": meta_url,
        "meta": job.get("meta_cache", {}),
    }

@app.get("/api/runs")
def runs():
    return {"runs": list_jobs()}
