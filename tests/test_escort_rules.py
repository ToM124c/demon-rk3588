"""护送判定逻辑单测（不加载模型、不需要板端 NPU）。

用法：python tests/test_escort_rules.py
"""

from __future__ import annotations

import numpy as np

from _util import check, hr

from phone_call_rknn.escort.config import LABEL_COMPLIANT, LABEL_NON_COMPLIANT, default_escort_config
from phone_call_rknn.escort.rules import (
    assign_hands_to_roles,
    assign_roles_by_right_shoulder,
    evaluate_frame,
    extract_pose_people,
)


def make_person(box, score, right_shoulder_x, keypoints=None):
    """构造一个 pose 输出格式的人体（17x3 关键点）。"""
    xy = np.zeros((17, 2), dtype=np.float32)
    conf = np.zeros((17,), dtype=np.float32)
    defaults = {
        "right_shoulder": (right_shoulder_x, 120),
        "left_shoulder": (right_shoulder_x - 60, 120),
        "right_elbow": (right_shoulder_x + 40, 200),
        "left_elbow": (right_shoulder_x - 100, 200),
        "right_wrist": (right_shoulder_x + 60, 260),
        "left_wrist": (right_shoulder_x - 120, 260),
    }
    for name, point in (keypoints or defaults).items():
        idx = {"left_shoulder": 5, "right_shoulder": 6, "left_elbow": 7,
               "right_elbow": 8, "left_wrist": 9, "right_wrist": 10}[name]
        xy[idx] = point
        conf[idx] = 0.9
    return {"box": np.asarray(box, dtype=np.float32), "score": float(score), "xy": xy, "conf": conf}


def hand_at(center, size=40):
    x, y = center
    box = np.array([x - size / 2, y - size / 2, x + size / 2, y + size / 2], dtype=np.float32)
    return {"box": box, "center": np.array([x, y], dtype=np.float32), "conf": 0.8, "class": "hand"}


def build_scene(hand_near_elbows=False):
    # 右肩 x 排序：left_police < suspect < right_police
    right_police = make_person([900, 80, 1100, 400], 0.9, 1000)
    suspect = make_person([500, 80, 700, 400], 0.85, 600,
                          keypoints={
                              "right_shoulder": (600, 120), "left_shoulder": (540, 120),
                              "right_elbow": (660, 200), "left_elbow": (480, 200),
                              "right_wrist": (680, 300), "left_wrist": (460, 300),
                          })
    left_police = make_person([100, 80, 300, 400], 0.8, 200,
                              keypoints={
                                  "right_shoulder": (200, 120), "left_shoulder": (140, 120),
                                  "right_elbow": (260, 200), "left_elbow": (80, 200),
                                  "right_wrist": (280, 300), "left_wrist": (60, 300),
                              })
    return [right_police, suspect, left_police]


def main() -> int:
    cfg = default_escort_config(det_conf=0.3)
    ok = True

    hr("角色分配（按右肩 x 升序：最左 = right_police，与旧脚本一致）")
    pose_people = extract_pose_people(build_scene(), cfg.kpt_conf)
    roles = assign_roles_by_right_shoulder(pose_people)
    ok &= check(set(roles) == {"left_police", "suspect", "right_police"}, "三个角色齐备")
    ok &= check(roles["right_police"]["right_shoulder_x"] < roles["suspect"]["right_shoulder_x"]
                < roles["left_police"]["right_shoulder_x"], "按 x 升序映射 right/suspect/left")

    hr("手部归属：包含手腕的框优先")
    # left_police 的左手手腕在 (1060, 260)，手框把它包住
    hands = [hand_at((1060, 260)), hand_at((2000, 2000))]   # 第二个远处，不应被分配
    assignments = assign_hands_to_roles(roles, hands, cfg.hand_max_pixel_dist)
    ok &= check(len(assignments) == 1 and assignments[0]["contains_wrist"] is True, "只分配包含手腕的手")
    ok &= check(roles["left_police"]["hands"]["right"] is not None, "手分给了 left_police 的右手腕")

    hr("合规判定")
    # 场景 A：嫌疑人没有可见的手，但警察没摸到肘 -> 不合规
    scene = build_scene()
    pose_people = extract_pose_people(scene, cfg.kpt_conf)
    roles = assign_roles_by_right_shoulder(pose_people)
    frame_result = evaluate_frame(roles, [], [], cfg)
    ok &= check(frame_result["label"] == LABEL_NON_COMPLIANT, "警察未接触肘部 -> 不合规")
    ok &= check("police_actions_failed" in frame_result["reason"], f"原因含 police_actions_failed（{frame_result['reason']}）")

    # 场景 B：两手分别贴到嫌疑人两肘 -> 合规
    scene = build_scene()
    pose_people = extract_pose_people(scene, cfg.kpt_conf)
    roles = assign_roles_by_right_shoulder(pose_people)
    suspect = roles["suspect"]
    hands = [hand_at(tuple(suspect["points"]["right_elbow"])), hand_at(tuple(suspect["points"]["left_elbow"]))]
    # 右警察的左手必须是包含其手腕的框 -> 直接塞进 roles 更直观
    roles["right_police"]["hands"]["left"] = {"box": hands[0]["box"], "center": hands[0]["center"], "conf": 0.8}
    roles["left_police"]["hands"]["right"] = {"box": hands[1]["box"], "center": hands[1]["center"], "conf": 0.8}
    frame_result = evaluate_frame(roles, [], hands, cfg)
    ok &= check(frame_result["label"] == LABEL_COMPLIANT, f"双手控双肘 -> 合规（{frame_result['reason']}）")
    ok &= check(frame_result["suspect_hand"]["ok"], "嫌疑人无手 -> 通过")

    # 场景 C：嫌疑人自己也有手 -> 不合规
    roles["suspect"]["hands"]["left"] = {"box": hands[0]["box"], "center": hands[0]["center"], "conf": 0.5}
    frame_result = evaluate_frame(roles, [], hands, cfg)
    ok &= check(frame_result["label"] == LABEL_NON_COMPLIANT and "suspect_hands_detected" in frame_result["reason"],
                "嫌疑人手部可见 -> 不合规")

    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
