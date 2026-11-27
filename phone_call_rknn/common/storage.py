"""对象存储（MinIO）：取流、存违规截图、上传测试视频。"""

from __future__ import annotations

import os
from pathlib import Path
from urllib.parse import quote, urlparse

import cv2
from minio import Minio

MINIO_ENDPOINT = os.getenv("MINIO_ENDPOINT", "http://192.17.110.253:30090")
MINIO_ACCESS_KEY = os.getenv("MINIO_ACCESS_KEY", "root")
MINIO_SECRET_KEY = os.getenv("MINIO_SECRET_KEY", "Litong123!")
VIDEO_BUCKET = os.getenv("MINIO_VIDEO_BUCKET", "detect-violating-videos")
IMAGE_BUCKET = os.getenv("MINIO_IMAGE_BUCKET", "detect-violating-images")


def make_minio_client() -> Minio:
    parsed = urlparse(MINIO_ENDPOINT)
    if parsed.scheme:
        endpoint, secure = parsed.netloc, parsed.scheme == "https"
    else:
        endpoint, secure = MINIO_ENDPOINT, False
    return Minio(endpoint, access_key=MINIO_ACCESS_KEY, secret_key=MINIO_SECRET_KEY, secure=secure)


def public_object_url(bucket: str, object_name: str) -> str:
    return f"{MINIO_ENDPOINT.rstrip('/')}/{bucket}/{quote(object_name)}"


def ensure_bucket(client: Minio, bucket: str) -> None:
    if not client.bucket_exists(bucket):
        client.make_bucket(bucket)


def download_video(client: Minio, task_id: str, dest: Path) -> Path:
    """把 MinIO 上的 taskId 对象拉到本地（调用方负责临时目录）。"""
    client.fget_object(VIDEO_BUCKET, task_id, str(dest))
    return dest


def save_violation_frame(client: Minio, task_id: str, frame_idx: int, frame, work_dir: Path) -> str:
    image_name = f"{Path(task_id).stem}_{frame_idx}.jpg"
    image_path = work_dir / image_name
    if not cv2.imwrite(str(image_path), frame):
        raise RuntimeError(f"Failed to save violation frame: {image_path}")
    object_name = f"{Path(task_id).stem}/{image_name}"
    client.fput_object(IMAGE_BUCKET, object_name, str(image_path), content_type="image/jpeg")
    return public_object_url(IMAGE_BUCKET, object_name)


def upload_video_to_minio(video_path, task_id: str) -> str:
    path = Path(video_path)
    if not path.is_file():
        raise FileNotFoundError(f"Video not found: {path}")
    client = make_minio_client()
    ensure_bucket(client, VIDEO_BUCKET)
    ensure_bucket(client, IMAGE_BUCKET)
    client.fput_object(VIDEO_BUCKET, task_id, str(path), content_type="video/mp4")
    return task_id
