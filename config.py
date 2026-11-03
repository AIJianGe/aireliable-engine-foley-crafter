"""FoleyCrafter V2A 引擎配置。

从环境变量 + 命令行参数加载，提供引擎运行所需的全部配置。

v0.2.1 双源真相消除策略：
    - MODEL_SUBDIR 默认读环境变量 AIRELIABLE_MODEL_LAYOUT_BASE
      （由调度层 EngineManager 启动子进程时同步 manifest.models.model_layout.base）
    - 优先级：环境变量 > 硬编码 fallback
"""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from aireliable_engine import EngineConfig as BaseEngineConfig


# ===== 常量 =====

ENGINE_ID = "foleycrafter_a3b2c1"
ENGINE_NAME = "FoleyCrafter"
CAPABILITY = "video-to-audio"
ENGINE_TYPE = "audio"  # 新增音效独立分类（区别于 voice）

# 默认端口（与 config/engines/foleycrafter_a3b2c1.yaml 一致）
# v0.2.9+：engine_id 与目录名统一为下划线（三同名契约）
# 详见 PR/architecture.md §3.26 + PR/development-guide.md §四十
DEFAULT_PORT = 7004

# 模型目录布局（相对 MODELS_ROOT）
# v0.2.1：优先读环境变量 AIRELIABLE_MODEL_LAYOUT_BASE
_DEFAULT_MODEL_SUBDIR = "audio/v2a/foleycrafter/"


def _resolve_model_subdir() -> str:
    """解析模型子目录（v0.2.1：优先读环境变量）。"""
    env_val = os.environ.get("AIRELIABLE_MODEL_LAYOUT_BASE")
    if env_val:
        return env_val.rstrip("/") + "/"
    return _DEFAULT_MODEL_SUBDIR


# 运行时常量
MODEL_SUBDIR = _resolve_model_subdir()

# v0.2.15+ 日志统一到 logs/engines/（项目顶层）
# 旧的 LOG_SUBDIR = "logs/"（相对 data_root）已废弃

# 默认视频长度限制（秒）
# 用户决策 6：默认 10 秒，可通过 --max-video-duration 覆盖
DEFAULT_MAX_VIDEO_DURATION = 10.0

# 默认推理参数（与 inference.py 对齐）
DEFAULT_NUM_INFERENCE_STEPS = 25
DEFAULT_TEMPORAL_SCALE = 0.2
DEFAULT_SEMANTIC_SCALE = 1.0


@dataclass
class EngineConfig(BaseEngineConfig):
    """FoleyCrafter V2A 引擎运行配置。

    通过环境变量 + 命令行参数构造。
    """

    # ===== 基础（身份 / 端口默认值由本引擎覆盖，其余公共字段见基类）=====
    engine_id: str = ENGINE_ID
    engine_name: str = ENGINE_NAME
    capability: str = CAPABILITY
    engine_type: str = ENGINE_TYPE  # audio（新增类型）
    port: int = DEFAULT_PORT

    # ===== 推导路径 =====
    model_dir: Path = field(init=False)
    log_dir: Path = field(init=False)

    # ===== FoleyCrafter 特有配置 =====
    # 视频长度限制（秒），默认 10.0，可通过 --max-video-duration 覆盖
    max_video_duration: float = DEFAULT_MAX_VIDEO_DURATION

    # 默认推理参数
    default_num_inference_steps: int = DEFAULT_NUM_INFERENCE_STEPS
    default_temporal_scale: float = DEFAULT_TEMPORAL_SCALE
    default_semantic_scale: float = DEFAULT_SEMANTIC_SCALE

    def __post_init__(self) -> None:
        """推导子路径。"""
        # 模型相关
        self.model_dir = self.models_root / MODEL_SUBDIR

        # 日志目录（v0.2.15+ 统一到 logs/engines/）
        self.log_dir = self.project_root / "logs" / "engines"

        # 确保运行时目录存在
        self.output_root.mkdir(parents=True, exist_ok=True)
        self.log_dir.mkdir(parents=True, exist_ok=True)

    def to_dict(self) -> dict:
        """转 dict（用于日志/调试）。"""
        return {
            "engine_id": self.engine_id,
            "engine_name": self.engine_name,
            "capability": self.capability,
            "engine_type": self.engine_type,
            "port": self.port,
            "host": self.host,
            "models_root": str(self.models_root),
            "model_dir": str(self.model_dir),
            "output_root": str(self.output_root),
            "log_dir": str(self.log_dir),
            "device": self.device,
            "log_level": self.log_level,
            "max_video_duration": self.max_video_duration,
            "default_num_inference_steps": self.default_num_inference_steps,
            "default_temporal_scale": self.default_temporal_scale,
            "default_semantic_scale": self.default_semantic_scale,
        }


def parse_args() -> EngineConfig:
    """解析命令行参数 + 环境变量，构造 EngineConfig。

    优先级：命令行 > 环境变量 > 默认值
    """
    parser = argparse.ArgumentParser(description="FoleyCrafter V2A 引擎")

    parser.add_argument("--port", type=int, default=DEFAULT_PORT,
                        help=f"HTTP 端口（默认 {DEFAULT_PORT}）")
    parser.add_argument("--host", type=str, default="127.0.0.1",
                        help="绑定地址（默认 127.0.0.1）")
    parser.add_argument("--device", type=str, default="auto",
                        help="推理设备：auto / cpu / cuda:0（默认 auto）")
    parser.add_argument("--log-level", type=str, default="INFO",
                        help="日志级别（默认 INFO）")
    parser.add_argument("--models-root", type=str, default=None,
                        help="模型根目录（默认 models/）")
    parser.add_argument(
        "--max-video-duration",
        type=float,
        default=DEFAULT_MAX_VIDEO_DURATION,
        help=f"视频长度限制（秒，默认 {DEFAULT_MAX_VIDEO_DURATION}）",
    )

    args = parser.parse_args()

    # 环境变量 fallback
    models_root = Path(args.models_root or os.environ.get("AIRELIABLE_MODELS_ROOT", "models"))

    config = EngineConfig(
        port=args.port,
        host=args.host,
        device=args.device,
        log_level=args.log_level,
        models_root=models_root,
        max_video_duration=args.max_video_duration,
    )

    return config
