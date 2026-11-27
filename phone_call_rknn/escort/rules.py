"""护送合规判定（纯逻辑，无硬件依赖，可单测）。

判定规则：
1. 画面里至少要有 3 个人（按右肩 x 排序 + 置信度/面积筛选）才能分出
   right_police / suspect / left_police 三个角色；
2. 手部检测框按「是否包含手腕点」优先、中心点距离次之，分配给左右手腕；
3. 嫌疑人**不能**有可见的手（双手被控）；
4. 右警察的左手要贴到嫌疑人右肘、左警察的右手要贴到嫌疑人左肘
   （像素距离 ≤ contact_pixel_thres）。
"""

from __future__ import annotations

import numpy as np

from ..core.geometry import (
    box_area,
    box_center,
    is_finite_array,
    point_dist,
    point_in_box,
)
from ..core.records import to_json_box, to_json_point
from .config import (
    KEYPOINT_INDEX,
    LABEL_COMPLIANT,
    LABEL_NON_COMPLIANT,
    ROLE_ORDER,
)


# --------------------------------------------------------------------------
# 关键点
# --------------------------------------------------------------------------
def get_keypoint(person, name, kpt_conf):
    idx = KEYPOINT_INDEX[name]
    xy = np.asarray(person["xy"], dtype=np.float32)
    conf = np.asarray(person["conf"], dtype=np.float32)
    if idx >= len(xy) or idx >= len(conf):
        return None
    if not np.isfinite(conf[idx]) or float(conf[idx]) < kpt_conf:
        return None
    point = xy[idx]
    if not is_finite_array(point):
        return None
    return np.asarray(point, dtype=np.float32)


def extract_pose_people(persons, kpt_conf):
    pose_people = []
    for pose_idx, person in enumerate(persons):
        box = np.asarray(person["box"], dtype=np.float32)
        if box.shape != (4,) or not is_finite_array(box):
            continue

        points = {
            "left_shoulder": get_keypoint(person, "left_shoulder", kpt_conf),
            "right_shoulder": get_keypoint(person, "right_shoulder", kpt_conf),
            "left_elbow": get_keypoint(person, "left_elbow", kpt_conf),
            "right_elbow": get_keypoint(person, "right_elbow", kpt_conf),
            "left_wrist": get_keypoint(person, "left_wrist", kpt_conf),
            "right_wrist": get_keypoint(person, "right_wrist", kpt_conf),
        }
        right_shoulder = points["right_shoulder"]
        if right_shoulder is None:
            continue

        pose_people.append(
            {
                "pose_idx": int(pose_idx),
                "role": None,
                "box": box,
                "center": box_center(box),
                "score": float(person.get("score", 0.0)),
                "points": points,
                "hands": {"left": None, "right": None},
                "right_shoulder_x": float(right_shoulder[0]),
            }
        )
    return pose_people


# --------------------------------------------------------------------------
# 角色 / 手部分配
# --------------------------------------------------------------------------
def assign_roles_by_right_shoulder(pose_people):
    for person in pose_people:
        person["role"] = None
        person["hands"] = {"left": None, "right": None}

    if len(pose_people) < 3:
        return {}

    selected = sorted(pose_people, key=lambda item: (-item["score"], -box_area(item["box"])))[:3]
    selected = sorted(selected, key=lambda item: item["right_shoulder_x"])
    roles = {}
    for role, person in zip(ROLE_ORDER, selected):
        person["role"] = role
        roles[role] = person
    return roles


def hand_assignment_candidates(roles, hand_dets, max_pixel_dist):
    candidates = []
    wrist_slots = []
    for role, person in roles.items():
        for side in ("left", "right"):
            wrist = person["points"].get(f"{side}_wrist")
            if wrist is not None:
                wrist_slots.append((role, side, wrist))

    for hand_idx, hand in enumerate(hand_dets):
        for role, side, wrist in wrist_slots:
            inside = point_in_box(wrist, hand["box"])
            dist_px = point_dist(hand["center"], wrist)
            if not inside and max_pixel_dist > 0 and dist_px > max_pixel_dist:
                continue
            # 包含手腕的框优先，其次按距离
            priority = 0 if inside else 1
            candidates.append(
                {
                    "priority": priority,
                    "distance_px": float(dist_px),
                    "role": role,
                    "side": side,
                    "hand_idx": int(hand_idx),
                    "contains_wrist": bool(inside),
                }
            )
    return candidates


def assign_hands_to_roles(roles, hand_dets, max_pixel_dist):
    for person in roles.values():
        person["hands"] = {"left": None, "right": None}

    candidates = hand_assignment_candidates(roles, hand_dets, max_pixel_dist)
    assigned_hands = set()
    assigned_slots = set()
    assignments = []

    for cand in sorted(candidates, key=lambda item: (item["priority"], item["distance_px"])):
        slot = (cand["role"], cand["side"])
        hand_idx = cand["hand_idx"]
        if slot in assigned_slots or hand_idx in assigned_hands:
            continue
        hand = dict(hand_dets[hand_idx])
        assignment = {
            "hand_idx": hand_idx,
            "role": cand["role"],
            "side": cand["side"],
            "distance_px": round(cand["distance_px"], 2),
            "contains_wrist": cand["contains_wrist"],
        }
        hand["assignment"] = assignment
        roles[cand["role"]]["hands"][cand["side"]] = hand
        assigned_slots.add(slot)
        assigned_hands.add(hand_idx)
        assignments.append(assignment)
    return assignments


# --------------------------------------------------------------------------
# 合规判定
# --------------------------------------------------------------------------
def distance_to_target(hand, target_point):
    if hand is None or target_point is None:
        return None
    return round(point_dist(hand["center"], target_point), 2)


def evaluate_suspect_hands(suspect):
    if suspect is None:
        return {"ok": False, "reason": "suspect_missing", "metrics": {}}

    left_hand = suspect["hands"].get("left")
    right_hand = suspect["hands"].get("right")
    has_hand = left_hand is not None or right_hand is not None
    metrics = {
        "left_hand_found": left_hand is not None,
        "right_hand_found": right_hand is not None,
    }
    ok = not has_hand
    return {
        "ok": ok,
        "reason": "suspect_hands_not_visible" if ok else "suspect_hands_detected",
        "metrics": metrics,
    }


def fixed_hand_contact(police, hand_side, suspect, target_name, contact_pixel_thres):
    if police is None or suspect is None:
        return {
            "ok": False,
            "reason": "role_missing",
            "distance_px": None,
            "hand_side": hand_side,
            "target_name": target_name,
            "contact": None,
        }

    hand = police["hands"].get(hand_side)
    target_point = suspect["points"].get(target_name)
    distance = distance_to_target(hand, target_point)
    if hand is None:
        reason = "hand_missing"
        passed = False
    elif target_point is None:
        reason = "target_missing"
        passed = False
    elif distance is not None and distance <= contact_pixel_thres:
        reason = "action_ok"
        passed = True
    else:
        reason = "distance_too_far"
        passed = False

    contact = {
        "role": police["role"],
        "hand_side": hand_side,
        "target_name": target_name,
        "hand_center": None if hand is None else to_json_point(hand["center"]),
        "target_point": to_json_point(target_point),
        "distance_px": distance,
        "passed": passed,
    }
    return {
        "ok": passed,
        "reason": reason,
        "distance_px": distance,
        "hand_side": hand_side,
        "target_name": target_name,
        "contact": contact,
    }


def evaluate_police_actions(roles, contact_pixel_thres):
    suspect = roles.get("suspect")
    right_police = roles.get("right_police")
    left_police = roles.get("left_police")
    if suspect is None:
        return {
            "ok": False,
            "reason": "suspect_missing",
            "right_police_action": {"ok": False, "reason": "suspect_missing", "distance_px": None},
            "left_police_action": {"ok": False, "reason": "suspect_missing", "distance_px": None},
            "contacts": [],
            "metrics": {},
        }
    if right_police is None or left_police is None:
        missing_reason = "two_police_missing"
        return {
            "ok": False,
            "reason": missing_reason,
            "right_police_action": {
                "ok": False,
                "reason": missing_reason if right_police is None else "peer_missing",
                "distance_px": None,
                "hand_side": "left",
                "target_name": "right_elbow",
            },
            "left_police_action": {
                "ok": False,
                "reason": missing_reason if left_police is None else "peer_missing",
                "distance_px": None,
                "hand_side": "right",
                "target_name": "left_elbow",
            },
            "contacts": [],
            "metrics": {},
        }

    right_result = fixed_hand_contact(right_police, "left", suspect, "right_elbow", contact_pixel_thres)
    left_result = fixed_hand_contact(left_police, "right", suspect, "left_elbow", contact_pixel_thres)
    contacts = [r["contact"] for r in (right_result, left_result) if r["contact"] is not None]

    ok = bool(right_result["ok"] and left_result["ok"])
    metrics = {
        "right_police_left_to_right_elbow_px": right_result["distance_px"],
        "left_police_right_to_left_elbow_px": left_result["distance_px"],
    }
    return {
        "ok": ok,
        "reason": "police_actions_ok" if ok else "police_actions_failed",
        "right_police_action": {
            "ok": right_result["ok"],
            "reason": right_result["reason"],
            "distance_px": right_result["distance_px"],
            "hand_side": right_result["hand_side"],
            "target_name": right_result["target_name"],
        },
        "left_police_action": {
            "ok": left_result["ok"],
            "reason": left_result["reason"],
            "distance_px": left_result["distance_px"],
            "hand_side": left_result["hand_side"],
            "target_name": left_result["target_name"],
        },
        "contacts": contacts,
        "metrics": metrics,
    }


def evaluate_frame(roles, assignments, hand_dets, config):
    role_ok = len(roles) == 3
    suspect_result = evaluate_suspect_hands(roles.get("suspect"))
    police_result = evaluate_police_actions(roles, config.contact_pixel_thres)
    overall_ok = role_ok and suspect_result["ok"] and police_result["ok"]
    reason_parts = []
    if not role_ok:
        reason_parts.append("three_roles_not_found")
    if not suspect_result["ok"]:
        reason_parts.append(suspect_result["reason"])
    if not police_result["ok"]:
        reason_parts.append(police_result["reason"])

    return {
        "label": LABEL_COMPLIANT if overall_ok else LABEL_NON_COMPLIANT,
        "reason": "escort_compliant" if overall_ok else "|".join(reason_parts),
        "role_ok": role_ok,
        "suspect_hand": suspect_result,
        "police_action": police_result,
        "assignments": assignments,
        "hand_count": len(hand_dets),
    }


def make_frame_record(frame_idx, source_name, roles, hand_dets, frame_result, timing):
    return {
        "frame_index": int(frame_idx),
        "source": source_name,
        "label": frame_result["label"],
        "reason": frame_result["reason"],
        "role_ok": bool(frame_result["role_ok"]),
        "hand_count": int(frame_result["hand_count"]),
        "suspect_hand_ok": bool(frame_result["suspect_hand"]["ok"]),
        "suspect_hand_reason": frame_result["suspect_hand"]["reason"],
        "police_action_ok": bool(frame_result["police_action"]["ok"]),
        "police_action_reason": frame_result["police_action"]["reason"],
        "right_police_action_ok": bool(frame_result["police_action"]["right_police_action"]["ok"]),
        "right_police_action_reason": frame_result["police_action"]["right_police_action"]["reason"],
        "left_police_action_ok": bool(frame_result["police_action"]["left_police_action"]["ok"]),
        "left_police_action_reason": frame_result["police_action"]["left_police_action"]["reason"],
        "roles": {
            role: {
                "pose_idx": int(person["pose_idx"]),
                "box": to_json_box(person["box"]),
                "score": round(float(person["score"]), 4),
                "right_shoulder": to_json_point(person["points"].get("right_shoulder")),
                "left_wrist": to_json_point(person["points"].get("left_wrist")),
                "right_wrist": to_json_point(person["points"].get("right_wrist")),
            }
            for role, person in roles.items()
        },
        "hands": [
            {
                "box": to_json_box(det["box"]),
                "center": to_json_point(det["center"]),
                "conf": round(float(det["conf"]), 4),
            }
            for det in hand_dets
        ],
        "assignments": frame_result["assignments"],
        "suspect_hand_metrics": frame_result["suspect_hand"]["metrics"],
        "police_action_metrics": frame_result["police_action"]["metrics"],
        "police_contacts": frame_result["police_action"]["contacts"],
        "timing": timing,
    }
