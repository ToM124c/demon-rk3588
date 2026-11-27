"""输入类型识别：图片/视频扩展名、图片列表、视频 fps 探测。"""

from __future__ import annotations

from pathlib import Path

import cv2

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".flv", ".wmv", ".webm"}


def is_video(path: Path) -> bool:
    return Path(path).is_file() and Path(path).suffix.lower() in VIDEO_EXTS


def is_image(path: Path) -> bool:
    return Path(path).is_file() and Path(path).suffix.lower() in IMAGE_EXTS


def image_files(path: Path) -> list[Path]:
    path = Path(path)
    if is_image(path):
        return [path]
    if path.is_dir():
        return sorted(p for p in path.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTS)
    return []


def probe_video_fps(video_path) -> float:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    cap.release()
    return float(fps)


def probe_video_size(video_path) -> tuple[int, int, float]:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 25.0)
    cap.release()
    return width, height, fps
