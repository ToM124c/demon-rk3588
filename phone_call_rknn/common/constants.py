"""API 层的公共常量：阶段号、模式名、模式归一化。"""

from __future__ import annotations

STAGE = 1
STAGE_NAME = "driver-behavior_escort_detection"

MODE_DRIVER = "driver"
MODE_ESCORT = "escort"
MODES = (MODE_DRIVER, MODE_ESCORT)


def normalize_mode(mode: str | None) -> str:
    normalized = (mode or MODE_DRIVER).strip().lower()
    if normalized not in MODES:
        raise ValueError(f"Unsupported mode: {mode}. Expected one of {MODES}.")
    return normalized
