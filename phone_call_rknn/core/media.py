from ..config import IMAGE_EXTS, VIDEO_EXTS


def is_video(path):
    return path.is_file() and path.suffix.lower() in VIDEO_EXTS


def image_files(path):
    if path.is_file() and path.suffix.lower() in IMAGE_EXTS:
        return [path]
    if path.is_dir():
        return sorted(p for p in path.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTS)
    return []

