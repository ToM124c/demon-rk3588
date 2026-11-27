"""兼容入口：`python -m phone_call_rknn.main`（参考链路，RKNNLite 单线程）。"""

from .driver.cli import main_simple

if __name__ == "__main__":
    raise SystemExit(main_simple())
