"""护送模式的绘制：角色框、关键点、手部连线、接触线、中文状态面板。"""

from __future__ import annotations

import cv2
import numpy as np

from ..core.visualize import get_chinese_font
from .config import (
    HAND_SIDE_LABELS,
    LABEL_COMPLIANT,
    ROLE_COLORS,
    ROLE_ORDER,
    SIDE_COLORS,
    TARGET_LABELS,
)


def status_word(ok: bool) -> str:
    return "合规" if ok else "不合规"


def format_distance_text(distance_px) -> str:
    if distance_px is None:
        return "距离未知"
    return f"{distance_px:.0f}px"


def format_suspect_action_detail(suspect_action) -> str:
    reason = suspect_action.get("reason", "")
    if reason == "suspect_hands_not_visible":
        return "未检测到嫌疑人手部"
    if reason == "suspect_hands_detected":
        metrics = suspect_action.get("metrics", {})
        if metrics.get("left_hand_found") and metrics.get("right_hand_found"):
            return "检测到双手"
        if metrics.get("left_hand_found"):
            return "检测到左手"
        if metrics.get("right_hand_found"):
            return "检测到右手"
        return "检测到嫌疑人手部"
    if reason == "suspect_missing":
        return "未识别到嫌疑人"
    return reason


def format_police_action_detail(police_action) -> str:
    hand_side = HAND_SIDE_LABELS.get(police_action.get("hand_side"), "手")
    target_name = TARGET_LABELS.get(police_action.get("target_name"), "目标点")
    distance_text = format_distance_text(police_action.get("distance_px"))
    reason = police_action.get("reason", "")

    if reason == "action_ok":
        return f"{hand_side}控{target_name}，{distance_text}"
    if reason == "hand_missing":
        return f"未分配到{hand_side}"
    if reason == "target_missing":
        return f"缺少嫌疑人{target_name}关键点"
    if reason == "distance_too_far":
        return f"{hand_side}距{target_name}过远，{distance_text}"
    if reason in ("role_missing", "two_police_missing", "peer_missing"):
        return "未识别到该警察"
    if reason == "suspect_missing":
        return "缺少嫌疑人"
    return f"{hand_side}控{target_name}，{distance_text}"


def build_status_texts(frame_result, show_action_details=True):
    suspect_action = frame_result["suspect_hand"]
    police_action = frame_result["police_action"]
    right_police_action = police_action.get(
        "right_police_action",
        {"ok": False, "reason": "role_missing", "distance_px": None, "hand_side": "left", "target_name": "right_elbow"},
    )
    left_police_action = police_action.get(
        "left_police_action",
        {"ok": False, "reason": "role_missing", "distance_px": None, "hand_side": "right", "target_name": "left_elbow"},
    )
    texts = [
        (f"总体判定: {status_word(frame_result['label'] == LABEL_COMPLIANT)}", frame_result["label"] == LABEL_COMPLIANT),
        (f"三人识别: {'成功' if frame_result['role_ok'] else '失败'} | hand数量: {frame_result['hand_count']}", frame_result["role_ok"]),
    ]
    if show_action_details:
        texts.extend(
            [
                (f"嫌疑人动作: {status_word(suspect_action['ok'])} | {format_suspect_action_detail(suspect_action)}", suspect_action["ok"]),
                (f"右警察动作: {status_word(right_police_action['ok'])} | {format_police_action_detail(right_police_action)}", right_police_action["ok"]),
                (f"左警察动作: {status_word(left_police_action['ok'])} | {format_police_action_detail(left_police_action)}", left_police_action["ok"]),
            ]
        )
    else:
        texts.extend(
            [
                (f"嫌疑人动作: {status_word(suspect_action['ok'])}", suspect_action["ok"]),
                (f"右警察动作: {status_word(right_police_action['ok'])}", right_police_action["ok"]),
                (f"左警察动作: {status_word(left_police_action['ok'])}", left_police_action["ok"]),
            ]
        )
    return texts


def draw_status_panel(canvas, text_items):
    font = get_chinese_font(22)
    if font is None:                     # 没有中文字体时退回英文占位
        for idx, (_text, ok) in enumerate(text_items):
            color = (0, 180, 0) if ok else (220, 0, 0)
            cv2.putText(canvas, f"line{idx + 1} {'OK' if ok else 'NG'}", (22, 40 + idx * 28),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2, cv2.LINE_AA)
        return canvas

    from PIL import Image, ImageDraw

    image = Image.fromarray(cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(image)
    for idx, (text, ok) in enumerate(text_items):
        color = (0, 180, 0) if ok else (220, 0, 0)
        draw.text((22, 16 + idx * 28), text, fill=color, font=font, stroke_width=1, stroke_fill=(0, 0, 0))
    canvas[:] = cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR)
    return canvas


def draw_box(frame, box, color, label, thickness=2):
    x1, y1, x2, y2 = [int(round(float(v))) for v in box]
    h, w = frame.shape[:2]
    x1 = max(0, min(x1, w - 1))
    y1 = max(0, min(y1, h - 1))
    x2 = max(0, min(x2, w - 1))
    y2 = max(0, min(y2, h - 1))
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
    cv2.putText(frame, label, (x1, max(20, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2, cv2.LINE_AA)


def draw_point(frame, point, color, label):
    if point is None:
        return
    x, y = int(round(float(point[0]))), int(round(float(point[1])))
    cv2.circle(frame, (x, y), 5, color, -1)
    cv2.putText(frame, label, (x + 6, y - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)


def draw_contact(frame, contact, show_distance=True):
    hand_center = contact.get("hand_center")
    target_point = contact.get("target_point")
    if hand_center is None or target_point is None:
        return
    start = (int(round(hand_center[0])), int(round(hand_center[1])))
    end = (int(round(target_point[0])), int(round(target_point[1])))
    color = (0, 220, 0) if contact.get("passed") else (0, 0, 255)
    cv2.line(frame, start, end, color, 2, cv2.LINE_AA)
    distance = contact.get("distance_px")
    if show_distance and distance is not None:
        mid = ((start[0] + end[0]) // 2, (start[1] + end[1]) // 2)
        cv2.putText(frame, f"{distance:.0f}px", mid, cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)


def annotate_frame(frame, roles, pose_people, hand_dets, frame_result, show_action_details=True):
    canvas = frame.copy()

    for person in pose_people:
        if person.get("role") is None:
            draw_box(canvas, person["box"], (120, 120, 120), f"pose {person['score']:.2f}", thickness=1)

    for role in ROLE_ORDER:
        person = roles.get(role)
        if person is None:
            continue
        color = ROLE_COLORS[role]
        draw_box(canvas, person["box"], color, f"{role} {person['score']:.2f}")
        for name in ("right_shoulder", "left_wrist", "right_wrist", "left_elbow", "right_elbow"):
            draw_point(canvas, person["points"].get(name), color, name)

        for side, hand in person["hands"].items():
            if hand is None:
                continue
            side_color = SIDE_COLORS[side]
            label = f"{role}_{side}_hand"
            if show_action_details:
                label += f" {hand.get('assignment', {}).get('distance_px', 0):.0f}px"
            draw_box(canvas, hand["box"], side_color, label)
            wrist = person["points"].get(f"{side}_wrist")
            if wrist is not None:
                start = tuple(int(round(float(v))) for v in hand["center"])
                end = tuple(int(round(float(v))) for v in wrist)
                cv2.line(canvas, start, end, side_color, 1, cv2.LINE_AA)

    assigned_hand_indices = {item["hand_idx"] for item in frame_result["assignments"]}
    for hand_idx, det in enumerate(hand_dets):
        if hand_idx in assigned_hand_indices:
            continue
        draw_box(canvas, det["box"], (0, 180, 180), f"hand {det['conf']:.2f}", thickness=1)

    for contact in frame_result["police_action"]["contacts"]:
        draw_contact(canvas, contact, show_distance=show_action_details)

    return draw_status_panel(canvas, build_status_texts(frame_result, show_action_details))
