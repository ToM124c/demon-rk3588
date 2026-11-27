"""兼容入口：`python -m phone_call_rknn.main_accel`（= `python -m phone_call_rknn.driver`）。"""

from .driver.cli import main

if __name__ == "__main__":
    raise SystemExit(main())
