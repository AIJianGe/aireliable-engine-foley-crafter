"""FoleyCrafter V2A 引擎 HTTP 服务（FastAPI + 原生 WebUI 单端口）。

8 个标准端点 / CORS / 日志端点 / 静态挂载 / 根路径跳转由 SDK 骨架
（``build_engine_app``）提供；本文件只保留引擎特有路由（``/upload``）。

启动：
    uv run --no-sync --project envs/audio-foleycrafter \\
        python -m engines.foleycrafter_a3b2c1.server --port 7004

端点：
    GET  /health         健康检查
    GET  /capabilities   能力声明（可用模型变体）
    POST /load           加载 FoleyCrafter 4 子模型 + IP-Adapter encoder
    POST /unload         卸载模型（释放显存）
    POST /infer          V2A / T2A 推理（SSE 流式进度）
    POST /lora/load      v0.1 返回 501
    POST /lora/unload    v0.1 返回 501
    GET  /output/*       静态文件（音频 / 视频产物）
    POST /upload         上传视频文件（WebUI V2A 模式输入）
    GET  /ui/*           原生 WebUI（FLUX 风格三件套）
    GET  /               307 重定向到 /ui/
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, FastAPI

from aireliable_engine import build_engine_app, mount_upload, run_engine_server

from .config import EngineConfig, parse_args
from .engine import FoleyCrafterEngine

logger = logging.getLogger(__name__)

_WEBUI_DIR = Path(__file__).parent / "webui"


# ===== 全局引擎实例 =====
# 由 SDK 骨架的 startup_hook 绑定，供本文件的路由使用
_engine: Optional[FoleyCrafterEngine] = None
_config: Optional[EngineConfig] = None


def get_engine() -> FoleyCrafterEngine:
    """获取引擎实例。"""
    if _engine is None:
        raise RuntimeError("引擎未初始化（FastAPI lifespan 未执行？）")
    return _engine


def get_config() -> EngineConfig:
    """获取配置。"""
    if _config is None:
        raise RuntimeError("配置未初始化")
    return _config


# ===== Lifespan 钩子（由 SDK 骨架调用）=====

async def _on_startup(engine: FoleyCrafterEngine) -> None:
    """启动钩子：绑定全局实例 + 异步刷新 variant 扫描。"""
    global _engine

    _engine = engine
    # v0.2.11+ 方案 B：lifespan startup 阶段异步刷新扫描（平台 API 已就绪）
    # __init__ 阶段的同步 _refresh_scan 只是 fallback，此处用平台 API 重新检测
    await engine._refresh_scan_async()
    logger.info(
        "引擎实例化完成: state=%s, variants=%d",
        engine.state_machine.state.value,
        len(engine._available_variants),
    )


async def _on_shutdown(engine: FoleyCrafterEngine) -> None:
    """关闭钩子：卸载模型并清空全局引用。"""
    global _engine, _config

    logger.info("===== FoleyCrafter V2A 引擎关闭 =====")
    try:
        if engine.wrapper.is_loaded:
            try:
                await engine._unload_impl()
            except Exception as e:
                logger.warning("关闭时卸载模型失败: %s", e)
    finally:
        _engine = None
        _config = None


# ===== 引擎特有路由 =====

router = APIRouter()


mount_upload(
    router,
    lambda: get_config().output_root / "uploads",
    log_label="视频",
)


# ===== 创建 FastAPI 应用 =====

def create_app(config: EngineConfig | None = None) -> FastAPI:
    """创建 FastAPI 应用。

    Args:
        config: 引擎配置（None 则从命令行解析）

    Returns:
        FastAPI: 配置好的应用实例
    """
    global _config

    if config is None:
        config = parse_args()
    _config = config

    return build_engine_app(
        FoleyCrafterEngine,
        config=config,
        title="FoleyCrafter V2A Engine",
        description=(
            "AIReliable 平台的 FoleyCrafter 视频转音效引擎"
            "（Video-to-Audio / Text-to-Audio）"
        ),
        version="0.1.0",
        static_dirs={"/ui": _WEBUI_DIR, "/output": config.output_root},
        extra_routers=[router],
        startup_hook=_on_startup,
        shutdown_hook=_on_shutdown,
    )


# ===== 入口 =====

def main() -> None:
    """命令行启动入口。"""
    config = parse_args()
    app = create_app(config)
    run_engine_server(app, config=config)


if __name__ == "__main__":
    main()
