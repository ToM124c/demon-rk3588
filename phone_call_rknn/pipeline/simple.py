import time

import cv2

from ..core.media import image_files
from ..core.postprocess import decode_detect_outputs, decode_pose_outputs
from ..core.preprocess import preprocess
from ..core.records import make_frame_record, write_record
from ..core.rules import closest_wrist_phone_match, evaluate_detect_rule, get_wrist_points
from ..core.timing import add_timing_sums, new_timing_sums, print_overall_speed, round_timing, zero_runner_timing
from ..core.visualize import draw_annotations


def run_one_frame(frame, frame_idx, source_name, pose_runner, detect_runner, args):
    shared_preprocess_ms = 0.0
    pose_timing = zero_runner_timing()
    persons = []
    wrists = []
    assignments = {}

    share_preprocess = (
        pose_runner.imgsz == detect_runner.imgsz
        and pose_runner.input_layout == detect_runner.input_layout
    )
    if share_preprocess:
        preprocess_start = time.perf_counter()
        tensor, det_scale, det_pad_x, det_pad_y = preprocess(frame, detect_runner.imgsz, detect_runner.input_layout)
        shared_preprocess_ms = (time.perf_counter() - preprocess_start) * 1000.0
        det_outputs, detect_timing = detect_runner.infer_tensor(tensor)
    else:
        tensor = None
        det_outputs, det_scale, det_pad_x, det_pad_y, detect_timing = detect_runner.infer(frame)

    detect_postprocess_start = time.perf_counter()
    dets = decode_detect_outputs(det_outputs, det_scale, det_pad_x, det_pad_y, frame.shape, args.det_conf, args.iou)
    hand_dets = [det for det in dets if det["class"] == "hand"]
    phone_dets = [det for det in dets if det["class"] == "phone"]
    seatbelt_dets = [det for det in dets if det["class"] == "seatbelt"]
    danger, matches, decision, need_pose = evaluate_detect_rule(hand_dets, phone_dets, args)
    detect_timing["postprocess_ms"] = (time.perf_counter() - detect_postprocess_start) * 1000.0
    detect_timing["total_runner_ms"] += detect_timing["postprocess_ms"]

    if need_pose:
        if share_preprocess:
            pose_outputs, pose_timing = pose_runner.infer_tensor(tensor)
            pose_scale = det_scale
            pose_pad_x = det_pad_x
            pose_pad_y = det_pad_y
        else:
            pose_outputs, pose_scale, pose_pad_x, pose_pad_y, pose_timing = pose_runner.infer(frame)
        pose_postprocess_start = time.perf_counter()
        persons = decode_pose_outputs(
            pose_outputs, pose_scale, pose_pad_x, pose_pad_y, frame.shape, args.person_conf, args.iou
        )
        wrists = get_wrist_points(persons, args.kpt_conf)
        assignments = {i: {"left": None, "right": None} for i in range(len(wrists))}
        pose_match = closest_wrist_phone_match(wrists, phone_dets)
        if pose_match is not None:
            danger = pose_match["distance_px"] < args.wrist_phone_pixel_thres
            matches.append(pose_match)
            decision = "pose_wrist_phone_hit" if danger else "pose_wrist_phone_miss"
        else:
            decision = "pose_no_valid_wrist"
        pose_timing["postprocess_ms"] = (time.perf_counter() - pose_postprocess_start) * 1000.0
        pose_timing["total_runner_ms"] += pose_timing["postprocess_ms"]

    record = make_frame_record(
        frame_idx, source_name, persons, wrists, hand_dets, phone_dets, seatbelt_dets, danger, matches, decision
    )
    total_preprocess_ms = shared_preprocess_ms + pose_timing["preprocess_ms"] + detect_timing["preprocess_ms"]
    total_postprocess_ms = pose_timing["postprocess_ms"] + detect_timing["postprocess_ms"]
    total_rknn_inference_ms = pose_timing["rknn_inference_ms"] + detect_timing["rknn_inference_ms"]
    total_runner_ms = total_preprocess_ms + total_postprocess_ms + total_rknn_inference_ms
    timing = {
        "pose": pose_timing,
        "detect": detect_timing,
        "total": {
            "shared_preprocess_ms": shared_preprocess_ms,
            "preprocess_ms": total_preprocess_ms,
            "postprocess_ms": total_postprocess_ms,
            "rknn_inference_ms": total_rknn_inference_ms,
            "total_runner_ms": total_runner_ms,
            "runner_fps": 1000.0 / total_runner_ms if total_runner_ms > 0 else 0.0,
        },
    }
    record["timing"] = round_timing(timing)
    vis = draw_annotations(frame, wrists, assignments, hand_dets, phone_dets, seatbelt_dets, danger)
    return record, vis, timing


def process_images(source_path, output_dir, pose_runner, detect_runner, args, jsonl_file):
    paths = image_files(source_path)
    if not paths:
        raise FileNotFoundError(f"No supported image files found: {source_path}")
    vis_dir = output_dir / "vis_images"
    if not args.no_save_vis:
        vis_dir.mkdir(parents=True, exist_ok=True)

    danger_count = 0
    timing_sums = new_timing_sums()
    for idx, path in enumerate(paths, start=1):
        frame = cv2.imread(str(path))
        if frame is None:
            raise RuntimeError(f"Failed to read image: {path}")
        record, vis, timing = run_one_frame(frame, idx, path.name, pose_runner, detect_runner, args)
        add_timing_sums(timing_sums, timing)
        danger_count += int(record["danger"])
        write_record(jsonl_file, record, args.quiet)
        if not args.no_save_vis:
            cv2.imwrite(str(vis_dir / f"{path.stem}_phone_call.jpg"), vis)
    print(f"Images done: total={len(paths)}, danger={danger_count}")
    print_overall_speed("Images", len(paths), timing_sums)


def process_video(source_path, output_dir, pose_runner, detect_runner, args, jsonl_file):
    cap = cv2.VideoCapture(str(source_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {source_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    writer = None
    out_video = output_dir / f"{source_path.stem}_phone_call.mp4"
    if not args.no_save_vis:
        fourcc = cv2.VideoWriter.fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(out_video), fourcc, fps, (width, height))
        if not writer.isOpened():
            cap.release()
            raise RuntimeError(f"Cannot create video writer: {out_video}")

    frame_idx = 0
    danger_count = 0
    timing_sums = new_timing_sums()
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_idx += 1
        record, vis, timing = run_one_frame(frame, frame_idx, source_path.name, pose_runner, detect_runner, args)
        add_timing_sums(timing_sums, timing)
        danger_count += int(record["danger"])
        write_record(jsonl_file, record, args.quiet)
        if writer is not None:
            writer.write(vis)

    cap.release()
    if writer is not None:
        writer.release()
        print(f"Visualization video saved: {out_video}")
    print(f"Video done: total_frames={frame_idx}, danger_frames={danger_count}")
    print_overall_speed("Video", frame_idx, timing_sums)

