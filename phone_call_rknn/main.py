import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.append(str(Path(__file__).resolve().parents[1]))

from phone_call_rknn.cli import parse_args


def main():
    args = parse_args()
    source_path = Path(args.source)
    output_dir = Path(args.output_dir)
    if not source_path.exists():
        raise FileNotFoundError(f"Source path not found: {source_path}")
    output_dir.mkdir(parents=True, exist_ok=True)
    result_path = output_dir / "frame_results.jsonl"

    from phone_call_rknn.media import is_video
    from phone_call_rknn.pipeline import process_images, process_video
    from phone_call_rknn.rknn_runner import RKNNRunner

    pose_runner = RKNNRunner(args.pose_model, args.imgsz, args.pose_input_layout)
    detect_runner = RKNNRunner(args.detect_model, args.imgsz, args.detect_input_layout)
    try:
        with result_path.open("w", encoding="utf-8") as jsonl_file:
            if is_video(source_path):
                process_video(source_path, output_dir, pose_runner, detect_runner, args, jsonl_file)
            else:
                process_images(source_path, output_dir, pose_runner, detect_runner, args, jsonl_file)
        print(f"Frame results saved: {result_path}")
    finally:
        pose_runner.release()
        detect_runner.release()


if __name__ == "__main__":
    main()
