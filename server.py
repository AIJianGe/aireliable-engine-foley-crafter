"""FoleyCrafter V2A 引擎 HTTP 服务（FastAPI + Gradio 单端口）。

实现 8 个标准 HTTP 端点 + 挂载 Gradio UI + 挂载输出目录。

启动：
    uv run --no-sync --project envs/audio-foleycrafter \\
        python -m engines.foleycrafter_a3b2c1.server --port 7004

端点：
    GET  /health         健康检查
    GET  /capabilities   能力声明（可用模型变体）
    POST /load           加载 FoleyCrafter 4 子模型 + IP-Adapter encoder
    POST /unload         卸载模型（释放显存）
    POST /infer          V2A / T2A 推理（SSE 流式进度）
    GET  /lora/load      v0.1 返回 501
    POST /lora/unload    v0.1 返回 501
    GET  /output/*       静态文件（音频 / 视频产物）
    POST /upload         上传视频文件（WebUI V2A 模式输入）
    GET  /ui/*           原生 WebUI（FLUX 风格三件套）
    GET  /               307 重定向到 /ui/
"""

from __future__ import annotations

import logging
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, RedirectResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

from aireliable_engine import (
    ErrorResponse,
    HealthResponse,
    InferRequest,
    LoadRequest,
    LoraLoadRequest,
    LoraUnloadRequest,
    register_log_level_endpoint,
    setup_engine_logging,
)

from .config import EngineConfig, parse_args
from .engine import FoleyCrafterEngine

logger = logging.getLogger(__name__)


# ===== 全局引擎实例 =====
# FastAPI 启动时初始化，所有端点共享
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


# ===== Lifespan 生命周期管理 =====

@asynccontextmanager
async def lifespan(app: FastAPI):
    """FastAPI 生命周期管理。

    启动时初始化引擎，关闭时清理资源（卸载模型）。
    """
    global _engine, _config

    # 解析配置（如果尚未从命令行注入）
    if _config is None:
        _config = parse_args()

    # 配置日志
    setup_engine_logging(
        engine_id=_config.engine_id,
        log_level=_config.log_level,
        log_dir=_config.log_dir,
    )

    logger.info("===== FoleyCrafter V2A 引擎启动 =====")
    logger.info("配置: %s", _config.to_dict())

    # 初始化引擎
    _engine = FoleyCrafterEngine(_config)
    # v0.2.11+ 方案 B：lifespan startup 阶段异步刷新扫描（平台 API 已就绪）
    # __init__ 阶段的同步 _refresh_scan 只是 fallback，此处用平台 API 重新检测
    await _engine._refresh_scan_async()
    logger.info(
        "引擎实例化完成: state=%s, variants=%d",
        _engine.state_machine.state.value,
        len(_engine._available_variants),
    )

    yield

    # 关闭清理：卸载模型
    logger.info("===== FoleyCrafter V2A 引擎关闭 =====")
    if _engine is not None and _engine.wrapper.is_loaded:
        try:
            await _engine._unload_impl()
        except Exception as e:
            logger.warning("关闭时卸载模型失败: %s", e)
    _engine = None


# ===== 创建 FastAPI 应用 =====

def create_app(config: EngineConfig | None = None) -> FastAPI:
    """创建 FastAPI 应用。

    Args:
        config: 引擎配置（None 则从命令行解析）

    Returns:
        FastAPI: 配置好的应用实例
    """
    global _config
    if config is not None:
        _config = config

    app = FastAPI(
        title="FoleyCrafter V2A Engine",
        description=(
            "AIReliable 平台的 FoleyCrafter 视频转音效引擎"
            "（Video-to-Audio / Text-to-Audio）"
        ),
        version="0.1.0",
        lifespan=lifespan,
    )

    # CORS（允许主控跨域调用，iframe 嵌入需要）
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],                  # v0.1 简化，生产环境应限制
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # ===== 注册 8 个标准端点 =====
    _register_endpoints(app)

    # 根路径重定向到 WebUI（平台「浏览器打开」打开基础 URL，无此路由会 404）
    @app.get("/", include_in_schema=False)
    async def _root_redirect() -> RedirectResponse:
        return RedirectResponse(url="/ui/", status_code=307)

    # ===== 挂载输出目录（/output/* 提供音频/视频文件下载）=====
    if _config is not None:
        app.mount(
            "/output",
            StaticFiles(directory=str(_config.output_root)),
            name="output",
        )
        logger.info("输出目录挂载: /output -> %s", _config.output_root)

        # ===== 挂载原生 WebUI（FLUX 风格三件套）=====
        _mount_webui(app)

    return app


def _register_endpoints(app: FastAPI) -> None:
    """注册 8 个标准 HTTP 端点。"""

    # 1. /health
    @app.get("/health", response_model=HealthResponse)
    async def health() -> HealthResponse:
        """健康检查。"""
        return await get_engine().health()

    # 2. /capabilities
    @app.get("/capabilities")
    async def capabilities() -> dict:
        """能力声明（可用模型变体 / V2A + T2A 支持情况）。

        注意：手动序列化 variants 以保留 V2AVariantInfo 扩展字段
        （supports_v2a / supports_t2a / default_temporal_scale 等），
        避免 Pydantic 按 base VariantInfo 序列化时丢弃子类字段。
        """
        resp = await get_engine().capabilities()
        data = resp.model_dump()
        data["variants"] = [v.model_dump() for v in resp.variants]
        return data

    # 3. /load
    @app.post("/load")
    async def load(request: LoadRequest) -> dict:
        """加载 FoleyCrafter 4 子模型 + IP-Adapter encoder。"""
        try:
            response = await get_engine().load(request)
            return response.model_dump()
        except FileNotFoundError as e:
            raise HTTPException(status_code=404, detail=str(e))
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))
        except Exception as e:
            logger.exception("加载失败")
            raise HTTPException(status_code=500, detail=str(e))

    # 4. /unload
    @app.post("/unload")
    async def unload() -> dict:
        """卸载模型（释放显存）。"""
        try:
            response = await get_engine().unload()
            return response.model_dump()
        except Exception as e:
            logger.exception("卸载失败")
            raise HTTPException(status_code=500, detail=str(e))

    # 5. /infer（SSE 流式）
    @app.post("/infer")
    async def infer(request: InferRequest):
        """V2A / T2A 推理（SSE 流式返回进度 + 最终音频/视频 URL）。

        请求体参考 InferRequest schema：
            - variant: 模型变体名（如 foleycrafter-v1）
            - inputs: V2AInputs 字段（prompt / temporal_scale / semantic_scale / ...）
            - audios: 视频输入列表（role=video，V2A 模式必填）
            - task_id: 可选任务 ID

        V2A 模式（audios[role=video] 存在）：
            生成 wav + mp4 两个文件
        T2A 模式（仅 prompt）：
            生成 wav 文件
        """
        engine = get_engine()

        # BaseEngine.infer 返回异步迭代器（SSE 帧）
        sse_stream = engine.infer(request)

        return StreamingResponse(
            sse_stream,
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",          # Nginx 不缓冲
            },
        )

    # 6. /status
    @app.get("/status")
    async def status() -> dict:
        """引擎状态查询。"""
        return (await get_engine().status()).model_dump()

    # 7. /lora/load（v0.1 返回 501）
    @app.post("/lora/load")
    async def lora_load(request: LoraLoadRequest) -> dict:
        """加载 LoRA（v0.1 不支持）。"""
        try:
            response = await get_engine().load_lora(request)
            return response.model_dump()
        except HTTPException:
            raise
        except Exception as e:
            logger.exception("LoRA 加载失败")
            raise HTTPException(status_code=500, detail=str(e))

    # 8. /lora/unload（v0.1 返回 501）
    @app.post("/lora/unload")
    async def lora_unload(request: LoraUnloadRequest) -> dict:
        """卸载 LoRA（v0.1 不支持）。"""
        try:
            response = await get_engine().unload_lora(request)
            return response.model_dump()
        except HTTPException:
            raise
        except Exception as e:
            logger.exception("LoRA 卸载失败")
            raise HTTPException(status_code=500, detail=str(e))

    # 错误处理
    @app.exception_handler(HTTPException)
    async def http_exception_handler(request: Request, exc: HTTPException):
        """HTTP 异常统一格式。"""
        return JSONResponse(
            status_code=exc.status_code,
            content=ErrorResponse(error={
                "type": exc.__class__.__name__,
                "message": str(exc.detail),
                "code": exc.status_code,
            }).model_dump(),
        )

    # 9. /upload（WebUI V2A 模式视频上传）
    @app.post("/upload")
    async def upload(file: UploadFile) -> dict:
        """上传视频文件，保存到 output/uploads/，返回引擎侧绝对路径。

        WebUI 前端先调本端点拿 path，再在 /infer 请求体的
        audios[role=video].path 中引用（engine._infer_impl 只认 path，
        AudioInput 的 b64_json 不适用于大视频，走 multipart 分块落盘）。
        """
        config = get_config()
        upload_dir = config.output_root / "uploads"
        upload_dir.mkdir(parents=True, exist_ok=True)

        # 防路径穿越：只取文件名部分；uuid 前缀防重名覆盖
        safe_name = Path(file.filename or "upload.bin").name
        dest = upload_dir / f"{uuid.uuid4().hex[:8]}_{safe_name}"

        size = 0
        with open(dest, "wb") as f:
            while True:
                chunk = await file.read(1024 * 1024)  # 1MB 分块写盘
                if not chunk:
                    break
                f.write(chunk)
                size += len(chunk)

        logger.info("视频上传: %s -> %s (%d bytes)", safe_name, dest, size)
        return {"path": str(dest), "filename": safe_name, "size": size}

    # v0.2.5+ T6.2：日志级别动态调整端点（GET /log_level + POST /log_level）
    register_log_level_endpoint(app)


def _mount_webui(app: FastAPI) -> None:
    """挂载原生 WebUI（FLUX 风格三件套：设置 / 参数 / 结果三区布局）。

    webui/ 目录（index.html / style.css / app.js）存在时挂载到 /ui；
    目录不存在时跳过（仅提供 API 端点）。
    ui.py（Gradio 旧界面）已退役，保留文件仅作历史参考。
    """
    webui_dir = Path(__file__).parent / "webui"
    if webui_dir.is_dir() and (webui_dir / "index.html").exists():
        app.mount("/ui", StaticFiles(directory=str(webui_dir), html=True), name="webui")
        logger.info("WebUI 挂载: /ui -> %s", webui_dir)
    else:
        logger.warning("WebUI 未挂载（webui/index.html 不存在）")


# ===== 入口 =====

# 模块级 app 实例（uvicorn 直接引用）
app: Optional[FastAPI] = None


def main() -> None:
    """命令行启动入口。"""
    global _config, app

    # 解析命令行参数
    _config = parse_args()

    # 配置日志（lifespan 也会配置，这里先配一遍让启动日志可见）
    setup_engine_logging(
        engine_id=_config.engine_id,
        log_level=_config.log_level,
        log_dir=_config.log_dir,
    )

    logger.info("启动 FoleyCrafter V2A 引擎: host=%s, port=%d", _config.host, _config.port)

    # 创建 app
    app = create_app(_config)

    # 用 uvicorn 启动
    import uvicorn
    uvicorn.run(
        app,
        host=_config.host,
        port=_config.port,
        log_level=_config.log_level.lower(),
    )


if __name__ == "__main__":
    main()
