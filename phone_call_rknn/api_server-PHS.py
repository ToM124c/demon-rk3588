

import json
import os
import sys
import tempfile
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import quote, urlparse

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if __package__ in (None, ""):
    sys.path.insert(0, str(PROJECT_ROOT))

import cv2
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from minio import Minio

from phone_call_rknn.accel_pipeline import AccelConfig, create_thread_pools, iter_accel_video_results
from phone_call_rknn.config import DEFAULT_SOURCE


MINIO_ENDPOINT = os.getenv("MINIO_ENDPOINT", "http://192.17.110.253:30090")
MINIO_ACCESS_KEY = os.getenv("MINIO_ACCESS_KEY", "root")
MINIO_SECRET_KEY = os.getenv("MINIO_SECRET_KEY", "Litong123!")
VIDEO_BUCKET = os.getenv("MINIO_VIDEO_BUCKET", "detect-violating-videos")
IMAGE_BUCKET = os.getenv("MINIO_IMAGE_BUCKET", "detect-violating-images")

STAGE = 1
STAGE_NAME = "driver_behavior_detection"
CODING_PHONE_CALL = 1
CODING_NO_SEATBELT = 2
VIOLATION_MIN_FRAMES = int(os.getenv("VIOLATION_MIN_FRAMES", "3"))
VIOLATION_COOLDOWN_SEC = float(os.getenv("VIOLATION_COOLDOWN_SEC", "0.5"))


app = FastAPI(title="Driver Behavior Detection API")
_pool_lock = threading.Lock()
_detect_pool = None
_pose_pool = None
_accel_config = None


def build_accel_config():
    return AccelConfig(
        decode=os.getenv("DECODE", "vpu"),
        decode_fallback=os.getenv("DECODE_FALLBACK", "true").lower() != "false",
        codec=os.getenv("CODEC", "h264"),
        detect_cores=os.getenv("DETECT_CORES", "0,1"),
        pose_cores=os.getenv("POSE_CORES", "2"),
        decode_queue=int(os.getenv("DECODE_QUEUE", "4")),
        encode_queue=int(os.getenv("ENCODE_QUEUE", "2")),
        frame_budget=int(os.getenv("FRAME_BUDGET", "6")),
        no_save_vis=True,
        quiet=True,
    )


def get_thread_pools():
    global _detect_pool, _pose_pool, _accel_config
    if _detect_pool is None:
        _accel_config = build_accel_config()
        _detect_pool, _pose_pool = create_thread_pools(_accel_config)
    return _detect_pool, _pose_pool, _accel_config


def make_minio_client():
    parsed = urlparse(MINIO_ENDPOINT)
    if parsed.scheme:
        endpoint = parsed.netloc
        secure = parsed.scheme == "https"
    else:
        endpoint = MINIO_ENDPOINT
        secure = False
    return Minio(endpoint, access_key=MINIO_ACCESS_KEY, secret_key=MINIO_SECRET_KEY, secure=secure)


def public_object_url(bucket, object_name):
    base = MINIO_ENDPOINT.rstrip("/")
    return f"{base}/{bucket}/{quote(object_name)}"


def ndjson_event(payload):
    return json.dumps(payload, ensure_ascii=False) + "\n"


def now_ms():
    return int(time.time() * 1000)


def format_video_time(seconds):
    total_millis = max(0, int(round(float(seconds) * 1000)))
    hours, remainder = divmod(total_millis, 3600_000)
    minutes, remainder = divmod(remainder, 60_000)
    whole_seconds, millis = divmod(remainder, 1000)
    return f"{hours:02d}:{minutes:02d}:{whole_seconds:02d}.{millis:03d}"


def base_event(status, task_id):
    return {
        "status": status,
        "stage": STAGE,
        "stage_name": STAGE_NAME,
        "taskId": task_id,
        "timestamp": now_ms(),
    }


@app.on_event("shutdown")
def release_pools():
    global _detect_pool, _pose_pool, _accel_config
    if _detect_pool is not None:
        _detect_pool.release()
        _detect_pool = None
    if _pose_pool is not None:
        _pose_pool.release()
        _pose_pool = None
    _accel_config = None


def save_violation_frame(minio_client, task_id, frame_idx, frame, work_dir):
    image_name = f"{Path(task_id).stem}_{frame_idx}.jpg"
    image_path = work_dir / image_name
    if not cv2.imwrite(str(image_path), frame):
        raise RuntimeError(f"Failed to save violation frame: {image_path}")
    object_name = f"{Path(task_id).stem}/{image_name}"
    minio_client.fput_object(IMAGE_BUCKET, object_name, str(image_path), content_type="image/jpeg")
    return public_object_url(IMAGE_BUCKET, object_name)


def update_violation_counts(counts, frame_codings):
    for coding in (CODING_PHONE_CALL, CODING_NO_SEATBELT):
        if coding in frame_codings:
            counts[coding] += 1
        else:
            counts[coding] = 0


def reportable_codings(counts, last_report_at, video_seconds):
    codings = []
    for coding in (CODING_PHONE_CALL, CODING_NO_SEATBELT):
        if counts[coding] < VIOLATION_MIN_FRAMES:
            continue
        last_at = last_report_at.get(coding)
        if last_at is not None and video_seconds - last_at < VIOLATION_COOLDOWN_SEC:
            continue
        codings.append(coding)
    return codings


def frame_codings_from_record(record):
    codings = []
    if record["danger"]:
        codings.append(CODING_PHONE_CALL)
    if not record["driver_seatbelt_on"]:
        codings.append(CODING_NO_SEATBELT)
    return codings


def probe_video_fps(video_path):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    cap.release()
    return fps


def analyze_video(task_id):
    minio_client = make_minio_client()
    with tempfile.TemporaryDirectory(prefix="driver_behavior_") as temp_dir:
        work_dir = Path(temp_dir)
        local_video = work_dir / Path(task_id).name
        minio_client.fget_object(VIDEO_BUCKET, task_id, str(local_video))

        yield ndjson_event(base_event("stage_start", task_id))

        fps = probe_video_fps(local_video)
        completion_seconds = 0.0
        counts = {CODING_PHONE_CALL: 0, CODING_NO_SEATBELT: 0}
        last_report_at = {}

        with _pool_lock:
            detect_pool, pose_pool, accel_config = get_thread_pools()
            for result in iter_accel_video_results(
                local_video,
                accel_config,
                detect_pool,
                pose_pool,
                source_name=task_id,
                writer=None,
            ):
                frame_idx = result["frame_idx"]
                record = result["record"]
                video_seconds = (frame_idx - 1) / fps
                completion_seconds = video_seconds

                update_violation_counts(counts, frame_codings_from_record(record))
                codings = reportable_codings(counts, last_report_at, video_seconds)
                if not codings:
                    continue

                for coding in codings:
                    last_report_at[coding] = video_seconds

                img_url = save_violation_frame(minio_client, task_id, frame_idx, result["frame"], work_dir)
                event = base_event("violation", task_id)
                event.update(
                    {
                        "coding": codings,
                        "img_url": img_url,
                        "wrong_action_time": format_video_time(video_seconds),
                    }
                )
                yield ndjson_event(event)

        completion_time = format_video_time(completion_seconds)
        completed = base_event("completed", task_id)
        completed["stage_completion_time"] = completion_time
        yield ndjson_event(completed)

        all_completed = base_event("all_completed", task_id)
        all_completed["stage_completion_time"] = completion_time
        yield ndjson_event(all_completed)


def stream_task(task_id):
    try:
        yield from analyze_video(task_id)
    except Exception as exc:
        event = base_event("failed", task_id)
        event["message"] = str(exc)
        yield ndjson_event(event)


@app.get("/analyze/{taskId}")
async def stream_progress(taskId: str):
    return StreamingResponse(stream_task(taskId), media_type="application/x-ndjson")


#视频上传minio桶
def ensure_bucket(minio_client, bucket):
    if not minio_client.bucket_exists(bucket):
        minio_client.make_bucket(bucket)
        print(f"Created bucket: {bucket}")


def upload_video_to_minio(video_path, task_id):
    video_path = Path(video_path)
    if not video_path.is_file():
        raise FileNotFoundError(f"Video not found: {video_path}")

    minio_client = make_minio_client()
    ensure_bucket(minio_client, VIDEO_BUCKET)
    ensure_bucket(minio_client, IMAGE_BUCKET)
    minio_client.fput_object(
        VIDEO_BUCKET,
        task_id,
        str(video_path),
        content_type="video/mp4",
    )
    print(f"Uploaded video: {VIDEO_BUCKET}/{task_id} <- {video_path}")
    return task_id


def run_stream_test(task_id):
    print(f"Stream test start: taskId={task_id}")
    event_count = 0
    violation_count = 0
    for line in stream_task(task_id):
        event_count += 1
        event = json.loads(line)
        status = event.get("status")
        if status == "violation":
            violation_count += 1
        print(line, end="")
    print(f"Stream test done: events={event_count}, violations={violation_count}")
    return event_count, violation_count


def run_http_test(base_url, task_id):
    import urllib.error
    import urllib.request

    url = f"{base_url.rstrip('/')}/analyze/{quote(task_id)}"
    print(f"HTTP test start: {url}")
    event_count = 0
    violation_count = 0
    try:
        with urllib.request.urlopen(url, timeout=3600) as response:
            for raw_line in response:
                line = raw_line.decode("utf-8").strip()
                if not line:
                    continue
                event_count += 1
                event = json.loads(line)
                if event.get("status") == "violation":
                    violation_count += 1
                print(line)
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {exc.code}: {body}") from exc

    print(f"HTTP test done: events={event_count}, violations={violation_count}")
    return event_count, violation_count


@dataclass
class CliArgs:
    video: str
    task_id: str
    upload_only: bool
    skip_upload: bool
    http: bool
    base_url: str


def parse_cli_args():
    import argparse

    parser = argparse.ArgumentParser(description="Upload test video to MinIO and test analyze API.")
    parser.add_argument("--video", default=DEFAULT_SOURCE, help="Local video path to upload")
    parser.add_argument("--task-id", default=None, help="MinIO object name / taskId (default: video filename)")
    parser.add_argument("--upload-only", action="store_true", help="Only upload video, do not run analyze test")
    parser.add_argument("--skip-upload", action="store_true", help="Skip upload and only run analyze test")
    parser.add_argument("--http", action="store_true", help="Test via HTTP against a running uvicorn server")
    parser.add_argument("--base-url", default="http://127.0.0.1:8000", help="API base URL for --http mode")
    args = parser.parse_args()
    video_path = Path(args.video)
    task_id = args.task_id or video_path.name
    return CliArgs(
        video=str(video_path),
        task_id=task_id,
        upload_only=args.upload_only,
        skip_upload=args.skip_upload,
        http=args.http,
        base_url=args.base_url,
    )


def main_cli():
    args = parse_cli_args()
    if not args.skip_upload:
        upload_video_to_minio(args.video, args.task_id)
    if args.upload_only:
        return

    if args.http:
        run_http_test(args.base_url, args.task_id)
    else:
        run_stream_test(args.task_id)


if __name__ == "__main__":
    main_cli()
