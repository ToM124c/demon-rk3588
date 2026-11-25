import numpy as np

from .config import (
    DEFAULT_PERSON_SCALE,
    LEFT_SHOULDER_IDX,
    LEFT_WRIST_IDX,
    MIN_PERSON_SCALE,
    RIGHT_SHOULDER_IDX,
    RIGHT_WRIST_IDX,
)
from .geometry import box_pair_iou, is_finite_array, point_dist


def person_scale_from_shoulders(xy, confs, kpt_conf):
    if LEFT_SHOULDER_IDX >= len(xy) or RIGHT_SHOULDER_IDX >= len(xy):
        return DEFAULT_PERSON_SCALE
    if confs[LEFT_SHOULDER_IDX] < kpt_conf or confs[RIGHT_SHOULDER_IDX] < kpt_conf:
        return DEFAULT_PERSON_SCALE
    left_shoulder = xy[LEFT_SHOULDER_IDX]
    right_shoulder = xy[RIGHT_SHOULDER_IDX]
    if not is_finite_array(left_shoulder) or not is_finite_array(right_shoulder):
        return DEFAULT_PERSON_SCALE
    shoulder_width = point_dist(left_shoulder, right_shoulder)
    if not np.isfinite(shoulder_width) or shoulder_width < MIN_PERSON_SCALE:
        return DEFAULT_PERSON_SCALE
    return shoulder_width


def get_wrist_points(persons, kpt_conf):
    wrists = []
    for person_idx, person in enumerate(persons):
        xy = np.asarray(person["xy"], dtype=np.float32)
        confs = np.asarray(person["conf"], dtype=np.float32)
        left = None
        right = None
        if LEFT_WRIST_IDX < len(xy) and confs[LEFT_WRIST_IDX] >= kpt_conf and is_finite_array(xy[LEFT_WRIST_IDX]):
            left = xy[LEFT_WRIST_IDX]
        if RIGHT_WRIST_IDX < len(xy) and confs[RIGHT_WRIST_IDX] >= kpt_conf and is_finite_array(xy[RIGHT_WRIST_IDX]):
            right = xy[RIGHT_WRIST_IDX]
        if left is not None or right is not None:
            wrists.append(
                {
                    "person_idx": person_idx,
                    "left": left,
                    "right": right,
                    "scale": person_scale_from_shoulders(xy, confs, kpt_conf),
                }
            )
    return wrists


def assign_hands_to_wrists(wrists, hand_dets, thres):
    assignments = {i: {"left": None, "right": None} for i in range(len(wrists))}
    candidates = []
    for wrist_idx, wrist in enumerate(wrists):
        scale = wrist.get("scale", DEFAULT_PERSON_SCALE)
        for side in ("left", "right"):
            wrist_point = wrist[side]
            if wrist_point is None:
                continue
            for hand_idx, hand in enumerate(hand_dets):
                dist_norm = point_dist(hand["center"], wrist_point) / scale
                if dist_norm <= thres:
                    candidates.append((dist_norm, wrist_idx, side, hand_idx))

    used_hands = set()
    for dist_norm, wrist_idx, side, hand_idx in sorted(candidates):
        if hand_idx in used_hands or assignments[wrist_idx][side] is not None:
            continue
        assignments[wrist_idx][side] = hand_dets[hand_idx]
        assignments[wrist_idx][side]["wrist_dist_norm"] = float(dist_norm)
        used_hands.add(hand_idx)
    return assignments


def check_phone_on_hands(assignments, phone_dets, wrists, thres):
    matches = []
    for wrist_idx, wrist in enumerate(wrists):
        scale = wrist.get("scale", DEFAULT_PERSON_SCALE)
        for side in ("left", "right"):
            hand = assignments[wrist_idx].get(side)
            if hand is None:
                continue
            for phone in phone_dets:
                dist_norm = point_dist(hand["center"], phone["center"]) / scale
                if dist_norm <= thres:
                    matches.append(
                        {
                            "person_idx": int(wrist["person_idx"]),
                            "side": side,
                            "distance_norm": float(dist_norm),
                            "hand_conf": float(hand["conf"]),
                            "phone_conf": float(phone["conf"]),
                        }
                    )
    return bool(matches), matches


def make_hand_phone_match(mode, hand_idx, phone_idx, hand, phone):
    distance_px = point_dist(hand["center"], phone["center"])
    return {
        "mode": mode,
        "hand_idx": int(hand_idx),
        "phone_idx": int(phone_idx),
        "distance_px": round(float(distance_px), 3),
        "iou": round(box_pair_iou(hand["box"], phone["box"]), 4),
        "hand_conf": round(float(hand["conf"]), 4),
        "phone_conf": round(float(phone["conf"]), 4),
    }


def hand_phone_matches(hand_dets, phone_dets, mode):
    matches = []
    for hand_idx, hand in enumerate(hand_dets):
        for phone_idx, phone in enumerate(phone_dets):
            matches.append(make_hand_phone_match(mode, hand_idx, phone_idx, hand, phone))
    return sorted(matches, key=lambda match: match["distance_px"])


def is_hand_phone_hit(match, args):
    return (
        match is not None
        and match["distance_px"] < args.phone_hand_pixel_thres
        and match["iou"] > args.phone_hand_iou_thres
    )


def evaluate_detect_rule(hand_dets, phone_dets, args):
    if not phone_dets:
        return False, [], "no_phone", False
    if not hand_dets:
        return False, [], "no_hand", False

    if len(hand_dets) >= 2:
        matches = hand_phone_matches(hand_dets, phone_dets, "detect_hand_phone")
        match = next((item for item in matches if is_hand_phone_hit(item, args)), matches[0] if matches else None)
        danger = is_hand_phone_hit(match, args)
        reason = "detect_2plus_hands_phone_hit" if danger else "detect_2plus_hands_phone_miss"
        return danger, [match] if match else [], reason, False

    matches = hand_phone_matches(hand_dets, phone_dets, "detect_single_hand_phone")
    match = next((item for item in matches if is_hand_phone_hit(item, args)), matches[0] if matches else None)
    if is_hand_phone_hit(match, args):
        return True, [match], "detect_single_hand_phone_hit", False
    if match is not None and match["distance_px"] <= args.phone_hand_pixel_thres:
        return False, [match], "detect_single_hand_phone_iou_miss", False

    return False, [match] if match else [], "need_pose_wrist_phone_check", True


def closest_wrist_phone_match(wrists, phone_dets):
    best_match = None
    for wrist_idx, wrist in enumerate(wrists):
        for side in ("left", "right"):
            wrist_point = wrist[side]
            if wrist_point is None:
                continue
            for phone_idx, phone in enumerate(phone_dets):
                distance_px = point_dist(wrist_point, phone["center"])
                match = {
                    "mode": "pose_wrist_phone",
                    "person_idx": int(wrist["person_idx"]),
                    "wrist_idx": int(wrist_idx),
                    "side": side,
                    "phone_idx": int(phone_idx),
                    "distance_px": round(float(distance_px), 3),
                    "phone_conf": round(float(phone["conf"]), 4),
                }
                if best_match is None or match["distance_px"] < best_match["distance_px"]:
                    best_match = match
    return best_match

