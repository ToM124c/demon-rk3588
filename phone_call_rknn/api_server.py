import json
import os
import sys
import tempfile
import threading
import time
from dataclasses import dataclass
from importlib import util as importlib_util
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
from phone_call_rknn.visualization import draw_annotations


MINIO_ENDPOINT = os.getenv("MINIO_ENDPOINT", "http://192.17.110.253:30090")
MINIO_ACCESS_KEY = os.getenv("MINIO_ACCESS_KEY", "root")
MINIO_SECRET_KEY = os.getenv("MINIO_SECRET_KEY", "Litong123!")
VIDEO_BUCKET = os.getenv("MINIO_VIDEO_BUCKET", "detect-violating-videos")
IMAGE_BUCKET = os.getenv("MINIO_IMAGE_BUCKET", "detect-violating-images")

STAGE = 1
MODE_DRIVER = "driver"
MODE_ESCORT = "escort"
STAGE_NAME = "driver-behavior_escort_detection"
CODING_PHONE_CALL = 1
CODING_NO_SEATBELT = 2
CODING_SUSPECT_HAND = 3
CODING_LEFT_POLICE_ACTION = 4
CODING_RIGHT_POLICE_ACTION = 5
DRIVER_FRAME_STRIDE = int(os.getenv("DRIVER_FRAME_STRIDE", "15"))
ESCORT_FRAME_STRIDE = int(os.getenv("ESCORT_FRAME_STRIDE", "15"))
# With frame sampling enabled, 1 sampled hit is usually enough; keep cooldown to limit repeat reports.
VIOLATION_MIN_FRAMES = int(os.getenv("VIOLATION_MIN_FRAMES", "5"))
ESCORT_ROOT = Path(os.getenv("ESCORT_ROOT", "/mnt/jl/yolov26-rk3588"))
ESCORT_SCRIPT = Path(os.getenv("ESCORT_SCRIPT", str(ESCORT_ROOT / "inference-police-escort-rknn-hand-assign.py")))

app = FastAPI(title="Driver Behavior Detection API")
_pool_lock = threading.Lock()
_detect_pool = None
_pose_pool = None
_accel_config = None
_escort_lock = threading.Lock()
_escort_module = None
_escort_pose_runner = None
_escort_detect_runner = None


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


def load_escort_module():
    global _escort_module
    if _escort_module is not None:
        return _escort_module
    if not ESCORT_SCRIPT.is_file():
        raise FileNotFoundError(f"Escort script not found: {ESCORT_SCRIPT}")
    sys.path.insert(0, str(ESCORT_ROOT))
    spec = importlib_util.spec_from_file_location("police_escort_inference", ESCORT_SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load escort script: {ESCORT_SCRIPT}")
    module = importlib_util.module_from_spec(spec)
    spec.loader.exec_module(module)
    _escort_module = module
    return _escort_module


@dataclass
class EscortArgs:
    pose_model: str
    detect_model: str
    imgsz: int = 640
    det_conf: float = 0.3
    person_conf: float = 0.25
    kpt_conf: float = 0.30
    iou: float = 0.45
    hand_max_pixel_dist: float = 180.0
    suspect_wrist_pixel_thres: float = 120.0
    contact_pixel_thres: float = 95.0
    pose_input_layout: str = "nhwc"
    detect_input_layout: str = "nhwc"
    no_save_vis: bool = True
    quiet: bool = True
    log_interval: int = 30


def build_escort_args(module):
    return EscortArgs(
        pose_model=os.getenv("ESCORT_POSE_MODEL", module.POSE_MODEL),
        detect_model=os.getenv("ESCORT_DETECT_MODEL", module.DETECT_MODEL),
        imgsz=int(os.getenv("ESCORT_IMGSZ", "640")),
        det_conf=float(os.getenv("ESCORT_DET_CONF", "0.3")),
        person_conf=float(os.getenv("ESCORT_PERSON_CONF", "0.25")),
        kpt_conf=float(os.getenv("ESCORT_KPT_CONF", "0.30")),
        iou=float(os.getenv("ESCORT_IOU", "0.45")),
        hand_max_pixel_dist=float(os.getenv("ESCORT_HAND_MAX_PIXEL_DIST", "180.0")),
        suspect_wrist_pixel_thres=float(os.getenv("ESCORT_SUSPECT_WRIST_PIXEL_THRES", "120.0")),
        contact_pixel_thres=float(os.getenv("ESCORT_CONTACT_PIXEL_THRES", "95.0")),
        pose_input_layout=os.getenv("ESCORT_POSE_INPUT_LAYOUT", "nhwc"),
        detect_input_layout=os.getenv("ESCORT_DETECT_INPUT_LAYOUT", "nhwc"),
        no_save_vis=True,
        quiet=True,
    )


def get_escort_runners():
    global _escort_pose_runner, _escort_detect_runner
    module = load_escort_module()
    args = build_escort_args(module)
    if _escort_pose_runner is None:
        _escort_pose_runner = module.util.RKNNRunner(args.pose_model, args.imgsz, args.pose_input_layout)
    if _escort_detect_runner is None:
        _escort_detect_runner = module.util.RKNNRunner(args.detect_model, args.imgsz, args.detect_input_layout)
    return module, _escort_pose_runner, _escort_detect_runner, args


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


def normalize_mode(mode):
    normalized = (mode or MODE_DRIVER).strip().lower()
    if normalized not in (MODE_DRIVER, MODE_ESCORT):
        raise ValueError(f"Unsupported mode: {mode}. Expected '{MODE_DRIVER}' or '{MODE_ESCORT}'.")
    return normalized


def base_event(status, task_id, mode=MODE_DRIVER):
    normalize_mode(mode)
    return {
        "status": status,
        "stage": STAGE,
        "stage_name": STAGE_NAME,
        "taskId": task_id,
        "timestamp": now_ms(),
    }


@app.on_event("shutdown")
def release_pools():
    global _detect_pool, _pose_pool, _accel_config, _escort_pose_runner, _escort_detect_runner
    if _detect_pool is not None:
        _detect_pool.release()
        _detect_pool = None
    if _pose_pool is not None:
        _pose_pool.release()
        _pose_pool = None
    _accel_config = None
    if _escort_pose_runner is not None:
        _escort_pose_runner.release()
        _escort_pose_runner = None
    if _escort_detect_runner is not None:
        _escort_detect_runner.release()
        _escort_detect_runner = None


def save_violation_frame(minio_client, task_id, frame_idx, frame, work_dir):
    image_name = f"{Path(task_id).stem}_{frame_idx}.jpg"
    image_path = work_dir / image_name
    if not cv2.imwrite(str(image_path), frame):
        raise RuntimeError(f"Failed to save violation frame: {image_path}")
    object_name = f"{Path(task_id).stem}/{image_name}"
    minio_client.fput_object(IMAGE_BUCKET, object_name, str(image_path), content_type="image/jpeg")
    return public_object_url(IMAGE_BUCKET, object_name)


def visualize_driver_frame(frame, record):
    wrists = record["wrists"]
    assignments = {index: {"left": None, "right": None} for index in range(len(wrists))}
    return draw_annotations(
        frame,
        wrists,
        assignments,
        record["hands"],
        record["phones"],
        record["seatbelts"],
        record["danger"],
    )


class ViolationPeriodTracker:
    def __init__(self, candidate_codings, minio_client, task_id, work_dir):
        self.candidate_codings = candidate_codings
        self.minio_client = minio_client
        self.task_id = task_id
        self.work_dir = work_dir
        self.counts = {coding: 0 for coding in candidate_codings}
        self.starts = {}
        self.last_seen = {}
        self.image_urls = {}

    def update(self, frame_codings, video_seconds, frame_idx, frame, visualize_frame=None):
        current_codings = set(frame_codings)
        closed_periods = []
        for coding in self.candidate_codings:
            if coding in current_codings:
                if self.counts[coding] == 0:
                    self.starts[coding] = (
                        video_seconds,
                        frame_idx,
                        frame.copy(),
                        visualize_frame,
                    )
                self.counts[coding] += 1
                self.last_seen[coding] = video_seconds
            else:
                period = self._close(coding)
                if period is not None:
                    closed_periods.append(period)
        return closed_periods

    def finish(self):
        closed_periods = []
        for coding in self.candidate_codings:
            period = self._close(coding)
            if period is not None:
                closed_periods.append(period)
        return closed_periods

    def _close(self, coding):
        period = None
        if self.counts[coding] >= VIOLATION_MIN_FRAMES:
            start_seconds, frame_idx, frame, visualize_frame = self.starts[coding]
            end_seconds = self.last_seen[coding]
            img_url = self.image_urls.get(frame_idx)
            if img_url is None:
                if visualize_frame is not None:
                    frame = visualize_frame(frame)
                img_url = save_violation_frame(
                    self.minio_client, self.task_id, frame_idx, frame, self.work_dir
                )
                self.image_urls[frame_idx] = img_url
            period = {
                "coding": coding,
                "start_time": format_video_time(start_seconds),
                "end_time": format_video_time(end_seconds),
                "img_url": img_url,
            }
        self.counts[coding] = 0
        self.starts.pop(coding, None)
        self.last_seen.pop(coding, None)
        return period


def violation_period_event(task_id, mode, period):
    violation = base_event("violation", task_id, mode)
    violation.update(
        {
            "coding": [period["coding"]],
            "img_url": period["img_url"],
            "wrong_action_time": "null",
            "wrong_action_duration": [
                {
                    "start_time": period["start_time"],
                    "end_time": period["end_time"],
                }
            ],
        }
    )
    return ndjson_event(violation)


def completion_events(task_id, mode, completion_seconds):
    completion_time = format_video_time(completion_seconds)
    completed = base_event("completed", task_id, mode)
    completed["stage_completion_time"] = completion_time
    yield ndjson_event(completed)

    all_completed = base_event("all_completed", task_id, mode)
    all_completed["stage_completion_time"] = completion_time
    yield ndjson_event(all_completed)


def frame_codings_from_record(record):
    codings = []
    if record["danger"]:
        codings.append(CODING_PHONE_CALL)
    if not record["driver_seatbelt_on"]:
        codings.append(CODING_NO_SEATBELT)
    return codings


def escort_codings_from_record(record):
    codings = []
    if not record["suspect_hand_ok"]:
        codings.append(CODING_SUSPECT_HAND)
    if not record["left_police_action_ok"]:
        codings.append(CODING_LEFT_POLICE_ACTION)
    if not record["right_police_action_ok"]:
        codings.append(CODING_RIGHT_POLICE_ACTION)
    return codings


def probe_video_fps(video_path):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    cap.release()
    return fps


def analyze_driver_video(task_id):
    minio_client = make_minio_client()
    with tempfile.TemporaryDirectory(prefix="driver_behavior_") as temp_dir:
        work_dir = Path(temp_dir)
        local_video = work_dir / Path(task_id).name
        minio_client.fget_object(VIDEO_BUCKET, task_id, str(local_video))

        yield ndjson_event(base_event("stage_start", task_id, MODE_DRIVER))

        fps = probe_video_fps(local_video)
        completion_seconds = 0.0
        candidate_codings = (CODING_PHONE_CALL, CODING_NO_SEATBELT)
        tracker = ViolationPeriodTracker(candidate_codings, minio_client, task_id, work_dir)

        with _pool_lock:
            detect_pool, pose_pool, accel_config = get_thread_pools()
            for result in iter_accel_video_results(
                local_video,
                accel_config,
                detect_pool,
                pose_pool,
                source_name=task_id,
                writer=None,
                frame_stride=DRIVER_FRAME_STRIDE,
            ):
                frame_idx = result["frame_idx"]
                record = result["record"]
                video_seconds = (frame_idx - 1) / fps
                completion_seconds = video_seconds

                for period in tracker.update(
                    frame_codings_from_record(record),
                    video_seconds,
                    frame_idx,
                    result["frame"],
                    visualize_frame=lambda frame, record=record: visualize_driver_frame(
                        frame, record
                    ),
                ):
                    yield violation_period_event(task_id, MODE_DRIVER, period)

        for period in tracker.finish():
            yield violation_period_event(task_id, MODE_DRIVER, period)
        yield from completion_events(task_id, MODE_DRIVER, completion_seconds)


def analyze_escort_video(task_id):
    minio_client = make_minio_client()
    with tempfile.TemporaryDirectory(prefix="police_escort_") as temp_dir:
        work_dir = Path(temp_dir)
        local_video = work_dir / Path(task_id).name
        minio_client.fget_object(VIDEO_BUCKET, task_id, str(local_video))

        yield ndjson_event(base_event("stage_start", task_id, MODE_ESCORT))

        fps = probe_video_fps(local_video)
        cap = cv2.VideoCapture(str(local_video))
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video: {local_video}")

        completion_seconds = 0.0
        candidate_codings = (CODING_SUSPECT_HAND, CODING_LEFT_POLICE_ACTION, CODING_RIGHT_POLICE_ACTION)
        tracker = ViolationPeriodTracker(candidate_codings, minio_client, task_id, work_dir)
        frame_idx = 0

        try:
            with _escort_lock:
                module, pose_runner, detect_runner, args = get_escort_runners()
                while True:
                    ret, frame = cap.read()
                    if not ret:
                        break

                    frame_idx += 1
                    video_seconds = (frame_idx - 1) / fps
                    completion_seconds = video_seconds
                    if (frame_idx - 1) % ESCORT_FRAME_STRIDE != 0:
                        continue
                    record, visualization_data, _frame_result = module.run_one_frame(
                        frame,
                        frame_idx,
                        task_id,
                        pose_runner,
                        detect_runner,
                        args,
                        defer_visualization=True,
                    )

                    for period in tracker.update(
                        escort_codings_from_record(record),
                        video_seconds,
                        frame_idx,
                        frame,
                        visualize_frame=lambda frame, data=visualization_data: module.annotate_frame(
                            frame,
                            data["roles"],
                            data["pose_people"],
                            data["hand_dets"],
                            data["frame_result"],
                            show_action_details=False,
                        ),
                    ):
                        yield violation_period_event(task_id, MODE_ESCORT, period)
        finally:
            cap.release()

        for period in tracker.finish():
            yield violation_period_event(task_id, MODE_ESCORT, period)
        yield from completion_events(task_id, MODE_ESCORT, completion_seconds)


def analyze_video(task_id, mode=MODE_DRIVER):
    mode = normalize_mode(mode)
    if mode == MODE_ESCORT:
        yield from analyze_escort_video(task_id)
    else:
        yield from analyze_driver_video(task_id)


def stream_task(task_id, mode=MODE_DRIVER):
    try:
        yield from analyze_video(task_id, mode)
    except Exception as exc:
        event = base_event("failed", task_id, mode if mode in (MODE_DRIVER, MODE_ESCORT) else MODE_DRIVER)
        event["message"] = str(exc)
        yield ndjson_event(event)


@app.get("/analyze/{taskId}")
# async def stream_progress(taskId: str, mode: str = MODE_ESCORT):
async def stream_progress(taskId: str, mode: str = MODE_DRIVER):
    return StreamingResponse(stream_task(taskId, mode), media_type="application/x-ndjson")


# 视频上传minio桶
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


def run_stream_test(task_id, mode):
    print(f"Stream test start: taskId={task_id}, mode={mode}")
    event_count = 0
    violation_count = 0
    for line in stream_task(task_id, mode):
        event_count += 1
        event = json.loads(line)
        status = event.get("status")
        if status == "violation":
            violation_count += 1
        print(line, end="")
    print(f"Stream test done: events={event_count}, violations={violation_count}")
    return event_count, violation_count


def run_http_test(base_url, task_id, mode):
    import urllib.error
    import urllib.request

    url = f"{base_url.rstrip('/')}/analyze/{quote(task_id)}?mode={quote(mode)}"
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
    mode: str
    upload_only: bool
    skip_upload: bool
    http: bool
    base_url: str


def parse_cli_args():
    import argparse

    parser = argparse.ArgumentParser(description="Upload test video to MinIO and test analyze API.")
    parser.add_argument("--video", default=DEFAULT_SOURCE, help="Local video path to upload")
    parser.add_argument("--task-id", default=None, help="MinIO object name / taskId (default: video filename)")
    parser.add_argument( "--mode",choices=(MODE_DRIVER, MODE_ESCORT),default=MODE_DRIVER,
        help="Detection mode: driver phone/seatbelt or police escort compliance",
    )
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
        mode=args.mode,
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
        run_http_test(args.base_url, args.task_id, args.mode)
    else:
        run_stream_test(args.task_id, args.mode)


if __name__ == "__main__":
    main_cli()
