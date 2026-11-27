def zero_runner_timing():
    return {
        "preprocess_ms": 0.0,
        "postprocess_ms": 0.0,
        "rknn_inference_ms": 0.0,
        "total_runner_ms": 0.0,
    }


def round_timing(timing):
    return {
        name: {key: round(float(value), 3) for key, value in values.items()}
        for name, values in timing.items()
    }


def print_frame_speed(frame_idx, source_name, timing):
    pose = timing["pose"]
    detect = timing["detect"]
    total = timing["total"]
    print(
        f"Frame speed: source={source_name}, frame={frame_idx}, "
        f"shared_pre={total['shared_preprocess_ms']:.2f}ms, "
        f"pose(pre={pose['preprocess_ms']:.2f}ms,post={pose['postprocess_ms']:.2f}ms,"
        f"rknn={pose['rknn_inference_ms']:.2f}ms,total={pose['total_runner_ms']:.2f}ms), "
        f"detect(pre={detect['preprocess_ms']:.2f}ms,post={detect['postprocess_ms']:.2f}ms,"
        f"rknn={detect['rknn_inference_ms']:.2f}ms,total={detect['total_runner_ms']:.2f}ms), "
        f"total(pre={total['preprocess_ms']:.2f}ms,post={total['postprocess_ms']:.2f}ms,rknn={total['rknn_inference_ms']:.2f}ms,"
        f"runner={total['total_runner_ms']:.2f}ms,fps={total['runner_fps']:.2f})"
    )


def new_timing_sums():
    return {
        "shared_preprocess_ms": 0.0,
        "pose_preprocess_ms": 0.0,
        "pose_postprocess_ms": 0.0,
        "pose_rknn_inference_ms": 0.0,
        "pose_frame_count": 0,
        "detect_preprocess_ms": 0.0,
        "detect_postprocess_ms": 0.0,
        "detect_rknn_inference_ms": 0.0,
        "total_preprocess_ms": 0.0,
        "total_postprocess_ms": 0.0,
        "total_rknn_inference_ms": 0.0,
        "total_runner_ms": 0.0,
    }


def add_timing_sums(timing_sums, timing):
    timing_sums["shared_preprocess_ms"] += timing["total"]["shared_preprocess_ms"]
    timing_sums["pose_preprocess_ms"] += timing["pose"]["preprocess_ms"]
    timing_sums["pose_postprocess_ms"] += timing["pose"]["postprocess_ms"]
    timing_sums["pose_rknn_inference_ms"] += timing["pose"]["rknn_inference_ms"]
    if timing["pose"]["total_runner_ms"] > 0:
        timing_sums["pose_frame_count"] += 1
    timing_sums["detect_preprocess_ms"] += timing["detect"]["preprocess_ms"]
    timing_sums["detect_postprocess_ms"] += timing["detect"]["postprocess_ms"]
    timing_sums["detect_rknn_inference_ms"] += timing["detect"]["rknn_inference_ms"]
    timing_sums["total_preprocess_ms"] += timing["total"]["preprocess_ms"]
    timing_sums["total_postprocess_ms"] += timing["total"]["postprocess_ms"]
    timing_sums["total_rknn_inference_ms"] += timing["total"]["rknn_inference_ms"]
    timing_sums["total_runner_ms"] += timing["total"]["total_runner_ms"]


def print_overall_speed(label, frame_count, timing_sums):
    runner_ms_total = timing_sums["total_runner_ms"]
    if frame_count <= 0 or runner_ms_total <= 0:
        print(f"{label} speed summary: no frames")
        return
    avg_ms = runner_ms_total / frame_count
    fps = 1000.0 * frame_count / runner_ms_total
    pose_frame_count = timing_sums["pose_frame_count"]
    pose_divisor = pose_frame_count if pose_frame_count > 0 else 1
    pose_avg_ms = timing_sums["pose_rknn_inference_ms"] / pose_divisor if pose_frame_count > 0 else 0.0
    print(f"{label} speed summary: frames={frame_count}, avg_total={avg_ms:.2f}ms/frame, fps={fps:.2f}")
    print(f"前处理时间: total={timing_sums['total_preprocess_ms']:.2f}ms, avg={timing_sums['total_preprocess_ms'] / frame_count:.2f}ms/frame")
    print(f"后处理时间: total={timing_sums['total_postprocess_ms']:.2f}ms, avg={timing_sums['total_postprocess_ms'] / frame_count:.2f}ms/frame")
    print(
        f"模型推理时间: total={timing_sums['total_rknn_inference_ms']:.2f}ms, "
        f"avg={timing_sums['total_rknn_inference_ms'] / frame_count:.2f}ms/frame"
    )
    print(
        f"detect模型推理时间: total={timing_sums['detect_rknn_inference_ms']:.2f}ms, "
        f"avg={timing_sums['detect_rknn_inference_ms'] / frame_count:.2f}ms/frame"
    )
    print(
        f"pose模型推理时间: frames={pose_frame_count}, total={timing_sums['pose_rknn_inference_ms']:.2f}ms, "
        f"avg={pose_avg_ms:.2f}ms/pose-frame"
    )



def make_timing(detect_timing, pose_timing, shared_preprocess_ms=0.0):
    """把 detect/pose 两个阶段的耗时合成一条记录（两个模式共用）。"""
    total_pre = shared_preprocess_ms + detect_timing["preprocess_ms"] + pose_timing["preprocess_ms"]
    total_post = detect_timing["postprocess_ms"] + pose_timing["postprocess_ms"]
    total_infer = detect_timing["rknn_inference_ms"] + pose_timing["rknn_inference_ms"]
    total = total_pre + total_post + total_infer
    return {
        "pose": pose_timing,
        "detect": detect_timing,
        "total": {
            "shared_preprocess_ms": shared_preprocess_ms,
            "preprocess_ms": total_pre,
            "postprocess_ms": total_post,
            "rknn_inference_ms": total_infer,
            "total_runner_ms": total,
            "runner_fps": 1000.0 / total if total > 0 else 0.0,
        },
    }

