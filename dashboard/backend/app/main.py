# dashboard/backend/app/main.py
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from .s3 import presigned_upload_url

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
