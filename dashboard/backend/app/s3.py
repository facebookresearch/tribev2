# dashboard/backend/app/s3.py
import uuid
import boto3
from functools import lru_cache

BUCKET = "neuroloop-data"

@lru_cache
def _client():
    return boto3.client("s3")

def presigned_upload_url(filename: str, content_type: str) -> dict:
    key = f"uploads/{uuid.uuid4().hex[:12]}/{filename}"
    url = _client().generate_presigned_url(
        "put_object",
        Params={"Bucket": BUCKET, "Key": key, "ContentType": content_type},
        ExpiresIn=3600,
    )
    return {"upload_url": url, "s3_key": key}

def presigned_download_url(key: str) -> str:
    return _client().generate_presigned_url(
        "get_object",
        Params={"Bucket": BUCKET, "Key": key},
        ExpiresIn=3600,
    )

def download_file(key: str, local_path: str) -> str:
    _client().download_file(BUCKET, key, local_path)
    return local_path

def upload_file(local_path: str, key: str) -> None:
    _client().upload_file(local_path, BUCKET, key)

def upload_bytes(data: bytes, key: str, content_type: str = "application/octet-stream") -> None:
    _client().put_object(Bucket=BUCKET, Key=key, Body=data, ContentType=content_type)

def list_prefix(prefix: str) -> list[str]:
    resp = _client().list_objects_v2(Bucket=BUCKET, Prefix=prefix)
    return [obj["Key"] for obj in resp.get("Contents", [])]
