"""跨模式共享的基础设施（不依赖 driver/escort 任一业务域）。

- `config`     环境变量解析、项目路径、模型路径解析
- `constants`  stage / mode 常量与归一化
- `events`     NDJSON 事件构造
- `storage`    MinIO 取流/存图
- `tracking`   违规时段聚合
- `media`      输入类型识别、fps 探测
- `serialization` 坐标序列化
- `fonts`      中文字体加载（绘制面板用）
- `timing`     耗时结构（两个模式共用）
"""

from .config import LOCAL_MODEL_DIR, PROJECT_ROOT, env_bool, env_float, env_int, env_str, model_path, resolve_rknn_path
from .constants import MODE_DRIVER, MODE_ESCORT, MODES, STAGE_NAME, normalize_mode

__all__ = [
    "LOCAL_MODEL_DIR",
    "PROJECT_ROOT",
    "env_bool",
    "env_float",
    "env_int",
    "env_str",
    "model_path",
    "resolve_rknn_path",
    "MODE_DRIVER",
    "MODE_ESCORT",
    "MODES",
    "STAGE_NAME",
    "normalize_mode",
]
