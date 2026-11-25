import cv2
import numpy as np

try:
    from PIL import Image, ImageDraw, ImageFont
except ImportError:
    Image = None
    ImageDraw = None
    ImageFont = None


CHINESE_FONT_PATHS = (
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.otf",
    "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",
    "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc",
    "/usr/share/fonts/truetype/wqy/wqy-microhei.ttc",
    "/usr/share/fonts/truetype/arphic/uming.ttc",
)
_FONT_CACHE = {}


def get_chinese_font(size):
    if ImageFont is None:
        return None
    if size not in _FONT_CACHE:
        font = None
        for path in CHINESE_FONT_PATHS:
            try:
                font = ImageFont.truetype(path, size)
                break
            except OSError:
                continue
        _FONT_CACHE[size] = font
    return _FONT_CACHE[size]


def draw_chinese_lines(canvas, lines, origin=(40, 40), font_size=32):
    font = get_chinese_font(font_size)
    if Image is None or ImageDraw is None or font is None:
        y = origin[1] + font_size
        for text, color in lines:
            fallback = text.encode("ascii", errors="ignore").decode("ascii") or "status"
            cv2.putText(canvas, fallback, (origin[0], y), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
            y += font_size + 8
        return canvas

    image = Image.fromarray(cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(image)
    x, y = origin
    line_gap = 8
    for text, color in lines:
        rgb = (int(color[2]), int(color[1]), int(color[0]))
        draw.text((x, y), text, font=font, fill=rgb, stroke_width=2, stroke_fill=(0, 0, 0))
        bbox = draw.textbbox((x, y), text, font=font, stroke_width=2)
        y = bbox[3] + line_gap
    return cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR)


def draw_annotations(frame, wrists, assignments, hand_dets, phone_dets, seatbelt_dets, danger):
    canvas = frame.copy()
    for wrist_idx, wrist in enumerate(wrists):
        for side in ("left", "right"):
            point = wrist[side]
            if point is None:
                continue
            color = (0, 255, 0) if side == "left" else (255, 0, 0)
            cv2.circle(canvas, (int(point[0]), int(point[1])), 5, color, -1)
            hand = assignments[wrist_idx].get(side)
            if hand is not None:
                x1, y1, x2, y2 = map(int, hand["box"])
                cv2.rectangle(canvas, (x1, y1), (x2, y2), color, 2)
                cv2.putText(canvas, f"{side}_hand {hand['conf']:.2f}", (x1, max(0, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)

    assigned_ids = {id(assignments[i][side]) for i in assignments for side in ("left", "right") if assignments[i][side] is not None}
    for hand in hand_dets:
        if id(hand) in assigned_ids:
            continue
        x1, y1, x2, y2 = map(int, hand["box"])
        cv2.rectangle(canvas, (x1, y1), (x2, y2), (0, 180, 0), 1)
        cv2.putText(canvas, f"hand {hand['conf']:.2f}", (x1, max(0, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 180, 0), 1)

    for phone in phone_dets:
        x1, y1, x2, y2 = map(int, phone["box"])
        cv2.rectangle(canvas, (x1, y1), (x2, y2), (0, 255, 255), 2)
        cv2.putText(canvas, f"phone {phone['conf']:.2f}", (x1, max(0, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2)

    for seatbelt in seatbelt_dets:
        x1, y1, x2, y2 = map(int, seatbelt["box"])
        cv2.rectangle(canvas, (x1, y1), (x2, y2), (255, 0, 255), 2)
        cv2.putText(
            canvas,
            f"seatbelt {seatbelt['conf']:.2f}",
            (x1, max(0, y1 - 6)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 0, 255),
            2,
        )

    call_color = (0, 0, 255) if danger else (0, 255, 0)
    seatbelt_on = bool(seatbelt_dets)
    seatbelt_color = (0, 255, 0) if seatbelt_on else (0, 0, 255)
    canvas = draw_chinese_lines(
        canvas,
        [
            (f"打电话：{'是' if danger else '否'}", call_color),
            (f"驾驶员安全带：{'已系' if seatbelt_on else '未系'}", seatbelt_color),
        ],
    )
    return canvas

