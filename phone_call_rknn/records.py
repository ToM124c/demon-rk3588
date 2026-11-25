import json

import numpy as np


def to_json_box(box):
    return [round(float(v), 2) for v in np.asarray(box, dtype=np.float32).reshape(-1)]


def to_json_point(point):
    if point is None:
        return None
    return [round(float(point[0]), 2), round(float(point[1]), 2)]


def make_frame_record(
    frame_idx,
    source_name,
    persons,
    wrists,
    hand_dets,
    phone_dets,
    seatbelt_dets,
    danger,
    matches,
    decision,
):
    driver_seatbelt_on = bool(seatbelt_dets)
    return {
        "frame_index": int(frame_idx),
        "source": source_name,
        "danger": bool(danger),
        "driver_seatbelt_on": driver_seatbelt_on,
        "decision": decision,
        "person_count": len(persons),
        "wrist_count": len(wrists),
        "hand_count": len(hand_dets),
        "phone_count": len(phone_dets),
        "seatbelt_count": len(seatbelt_dets),
        "wrists": [
            {
                "person_idx": int(w["person_idx"]),
                "left": to_json_point(w["left"]),
                "right": to_json_point(w["right"]),
                "scale": round(float(w["scale"]), 3),
            }
            for w in wrists
        ],
        "hands": [{"box": to_json_box(det["box"]), "conf": round(float(det["conf"]), 4)} for det in hand_dets],
        "phones": [{"box": to_json_box(det["box"]), "conf": round(float(det["conf"]), 4)} for det in phone_dets],
        "seatbelts": [{"box": to_json_box(det["box"]), "conf": round(float(det["conf"]), 4)} for det in seatbelt_dets],
        "matches": matches,
    }


def write_record(jsonl_file, record, quiet):
    line = json.dumps(record, ensure_ascii=False)
    jsonl_file.write(line + "\n")
    jsonl_file.flush()
    if not quiet:
        print(line)

