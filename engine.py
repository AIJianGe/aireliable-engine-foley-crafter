"""FoleyCrafter V2A 引擎推理核心。

继承 ``CustomEngine``（全自定义模式 / integrated 范式），实现 7 个
``_xxx_impl`` 方法，封装 FoleyCrafterWrapper 的加载/推理/卸载流程。

8 端点契约：
    - /capabilities    返回扫描到的 FoleyCrafter 模型变体
    - /load            调用 FoleyCrafterWrapper.load() 加载 4 子模型
    - /unload          调用 FoleyCrafterWrapper.unload() 释放显存
    - /infer           解析 V2AInputs → 调用 generate() → 保存音频/视频 → SSE 推送
    - /status          返回引擎状态（已加载变体 / 设备 / 显存）
    - /lora/load       v0.1 不支持，返回 501
    - /lora/unload     v0.1 不支持，返回 501

V2A / T2A 双模式：
    - V2A 模式（audios[role=video] 存在）：处理视频 → 生成音频 → 合成 mp4
    - T2A 模式（仅 prompt）：直接文本生成音频
"""

from __future__ import annotations

import asyncio
import logging
import time
import uuid
from pathlib import Path
from typing import Any, Optional

from pydantic import BaseModel, ConfigDict, Field

from aireliable_engine import (
    CapabilitiesResponse,
    CustomEngine,
    InferRequest,
    LoadResponse,
    LoraLoadResponse,
    LoraUnloadResponse,
    SSEStream,
    StatusResponse,
    UnloadResponse,
    VariantInfo,
)

from .audio_utils import (
    get_audio_duration,
    get_video_duration,
    save_audio,
    save_video_with_audio,
)
from .config import EngineConfig
from .model_scanner import ModelScanner
from .progress_reporter import ProgressReporter
from .schemas import V2AInputs, V2AVariantInfo
from .foleycrafter_wrapper import FoleyCrafterWrapper

logger = logging.getLogger(__name__)


class FoleyCrafterEngine(CustomEngine):
    """FoleyCrafter V2A 引擎。

    全自定义模式（integrated 范式），直接控制模型加载/推理流程，
    不走工作流节点系统。UI 挂载 Gradio（由 server.py 处理）。
    """

    def __init__(self, config: EngineConfig):
        super().__init__(
            engine_id=config.engine_id,
            engine_name=config.engine_name,
            capability=config.capability,
            engine_type=config.engine_type,
        )
        self.config: EngineConfig = config

        # 核心组件
        self.wrapper = FoleyCrafterWrapper(config)
        # v0.2.11+ 方案 B：注入 engine=self，启用平台 API 声明式检测
        # （平台不可达时 fallback 到本地扫描）
        self.scanner = ModelScanner(config.model_dir, engine=self)

        # 启动时扫描模型目录（同步初始化为空，异步扫描在 lifespan 中触发）
        self._available_variants: list = []
        # v0.2.11+ 方案 B：__init__ 阶段平台 API 可能尚未就绪，先用 fallback 本地扫描
        # 保证引擎实例化不失败；后续 lifespan 会在平台 API 可用后再次异步刷新
        self._refresh_scan()

        # v0.1 不支持 LoRA
        self._available_loras: list[str] = []

        # 输出根目录（供 BaseEngine.get_output_path 使用）
        self.output_root = config.output_root

        # 读取 manifest 中的 default_variant（供 BaseEngine.infer auto-load 使用）
        # 优先级：request.variant > self.default_variant > 第一个 available variant
        self.default_variant = self._read_default_variant_from_manifest(
            Path(__file__).parent / "manifest.yaml"
        )

    # ============ 扫描 ============

    def _refresh_scan(self) -> None:
        """刷新模型扫描结果（同步 fallback，用于 __init__ 阶段）。

        v0.2.11+ 方案 B：此方法走 fallback 本地扫描逻辑。
        平台 API 异步扫描请调用 `_refresh_scan_async()`（lifespan startup 阶段调用）。
        """
        logger.info("刷新模型扫描: %s", self.config.model_dir)
        self._available_variants = self.scanner.scan_variants()

    async def _refresh_scan_async(self) -> None:
        """v0.2.11+ 方案 B：异步刷新模型扫描结果（优先平台 API）。

        在 lifespan startup 阶段调用，此时平台 API 已就绪。
        平台不可达时 fallback 到本地扫描。
        """
        logger.info("异步刷新模型扫描（方案 B 平台 API）: %s", self.config.model_dir)
        try:
            self._available_variants = await self.scanner.scan_variants_async()
        except Exception as e:
            logger.warning("异步扫描失败，回退到本地扫描: %s", e)
            self._available_variants = self.scanner.scan_variants()

    # ============ 7 个 _xxx_impl 实现 ============

    async def _capabilities_impl(self) -> CapabilitiesResponse:
        """GET /capabilities：返回可用模型变体。"""
        variants: list[VariantInfo] = []
        for v in self._available_variants:
            vram_mb = getattr(v, "vram_required", 0)
            # vram_required 是字符串（如 "10GB"），转 MB 数值用于兼容
            vram_str = str(vram_mb) if vram_mb else "未知"

            variants.append(V2AVariantInfo(
                name=v.name,
                base_model=v.base_model,
                precision=v.precision,
                loader=v.loader,
                vram_required=vram_str,
                available=True,
                default_steps=v.default_steps,
                default_temporal_scale=v.default_temporal_scale,
                default_semantic_scale=v.default_semantic_scale,
                supports_v2a=v.supports_v2a,
                supports_t2a=v.supports_t2a,
            ))

        return CapabilitiesResponse(
            engine_id=self.engine_id,
            engine_name=self.engine_name,
            capability=self.capability,
            type=self.engine_type,
            variants=variants,
            loras=self._available_loras,
            supported_sizes=[],                      # V2A 不用 size 概念
            default_variant=self.default_variant,
            # v0.4.1+ MCP 化：推理输入 JSON Schema（MCP 专属层动态 tool 数据源）
            input_schema=V2AInputs.model_json_schema(),
        )

    async def _load_impl(self, variant: str) -> LoadResponse:
        """POST /load：加载 FoleyCrafter 4 子模型。

        FoleyCrafterWrapper.load 是同步阻塞调用，用 asyncio.to_thread
        包装避免阻塞事件循环。
        """
        variant_path = self.scanner.resolve_variant_path(variant)
        if variant_path is None:
            raise FileNotFoundError(
                f"variant 不存在: {variant} "
                f"（可用: {[v.name for v in self._available_variants]}）"
            )

        logger.info("开始加载 variant: %s (path=%s)", variant, variant_path)

        start_time = time.time()
        try:
            await asyncio.to_thread(
                self.wrapper.load,
                variant,              # variant 名（存入 _loaded_variant）
                str(variant_path),    # variant 路径（用于加载权重）
                self.config.device,
            )
        except Exception as e:
            logger.exception("FoleyCrafter 模型加载失败: variant=%s", variant)
            raise RuntimeError(f"FoleyCrafter 模型加载失败: {e}") from e

        load_time = time.time() - start_time
        vram_used = self.wrapper.get_vram_used()

        logger.info(
            "加载完成: variant=%s, device=%s, sample_rate=%d, load_time=%.2fs",
            variant, self.wrapper.device, self.wrapper.sample_rate, load_time,
        )

        return LoadResponse(
            variant=variant,
            state="ready",
            vram_used=f"{vram_used}MB",
            load_time=load_time,
        )

    async def _unload_impl(self) -> UnloadResponse:
        """POST /unload：卸载模型，释放显存。"""
        if not self.wrapper.is_loaded:
            return UnloadResponse(state="idle", vram_freed="0MB")

        vram_freed = self.wrapper.get_vram_used()
        await asyncio.to_thread(self.wrapper.unload)

        logger.info("卸载完成: vram_freed=%dMB", vram_freed)
        return UnloadResponse(
            state="idle",
            vram_freed=f"{vram_freed}MB",
        )

    async def _infer_impl(self, request: InferRequest, stream: SSEStream) -> None:
        """POST /infer：V2A / T2A 推理（SSE 流式推送进度）。

        流程：
            1. 解析 V2AInputs（从 request.inputs）
            2. 提取视频输入（V2A 模式从 request.audios[role=video]）
            3. 判断模式（V2A / T2A）
            4. 上报 start / video_decoding / timestamp_detect / image_encoding 帧
            5. 线程池执行 wrapper.generate()（同步阻塞）
            6. 保存音频到 output/audio/{task_id}.wav
            7. V2A 模式：合成带音轨的 mp4 到 output/video/{task_id}.mp4
            8. 上报 completed 帧（含 audio_url + video_url）

        错误处理：内部捕获异常，推送 error 帧后 return，
        不重新抛出（避免 BaseEngine.infer 把状态置为 ERROR）。

        v0.3.4+ 集成 SDK 任务上报：
            - 入口生成 task_id（若 request.task_id 为 None）
            - 调 reporter.report_created 上报到 gateway（status=queued）
            - emit_start 自动触发 fire_report_running
            - emit_completed/emit_error 自动触发 fire_report_succeeded/failed
        """
        # ----- 0. 生成 task_id（v0.3.4+ T7）-----
        # 引擎 /infer 入口统一生成 task_id，前缀 fc_ 便于在任务中心识别
        task_id = request.task_id or f"fc_{uuid.uuid4().hex[:16]}"

        reporter = ProgressReporter(stream, task_id=task_id)

        # ----- 1. 解析输入 -----
        try:
            inputs = V2AInputs(**request.inputs)
        except Exception as e:
            logger.error("V2AInputs 解析失败: %s", e)
            reporter.emit_error(f"输入参数无效: {e}", "invalid_inputs")
            return

        # ----- 2. 提取视频输入 -----
        video_path = inputs.video_path
        for audio in request.audios:
            if not audio.path:
                continue
            if audio.role == "video":
                video_path = audio.path

        # ----- 3. 判断模式 -----
        is_v2a = bool(video_path)
        mode = "v2a" if is_v2a else "t2a"
        save_video = inputs.save_video if inputs.save_video is not None else is_v2a

        # ----- 3.5 v0.3.4+ 上报任务创建到 gateway（status=queued）-----
        # 必须在 emit_start 之前调用，确保任务先入库再标记 running
        await reporter.init_task_reporter()
        await reporter.report_created(
            request_summary={
                "prompt": inputs.prompt[:200] if inputs.prompt else "",
                "mode": mode,
            },
            variant=request.variant or "",
        )

        # ----- 4. 上报开始阶段 -----
        reporter.emit_start(inputs.prompt, mode, task_id=task_id)

        if is_v2a:
            # V2A 模式：上报视频处理阶段
            reporter.emit_video_decoding(0.10, frames=0, duration=0.0)
            reporter.emit_timestamp_detect(0.20)
            reporter.emit_image_encoding(0.35)
        else:
            # T2A 模式：跳过视频处理，直接到音频生成
            pass

        # ----- 5. 线程池执行推理 -----
        def _generate():
            # 进度回调（diffusers 扩散步数）
            def _on_step(step, total_steps):
                progress = 0.40 + 0.45 * (step / max(total_steps, 1))
                reporter.emit_audio_generating(progress, step=step, total_steps=total_steps)

            return self.wrapper.generate(
                prompt=inputs.prompt,
                negative_prompt=inputs.negative_prompt,
                seed=inputs.seed,
                temporal_align=inputs.temporal_align,
                temporal_scale=inputs.temporal_scale,
                semantic_scale=inputs.semantic_scale,
                num_inference_steps=inputs.num_inference_steps,
                video_path=video_path,
                max_video_duration=inputs.max_video_duration,
                progress_callback=_on_step,
            )

        try:
            samples, sample_rate = await asyncio.to_thread(_generate)
        except Exception as e:
            logger.exception("FoleyCrafter 推理失败")
            reporter.emit_error(f"推理失败: {e}", "infer_failed")
            return

        # ----- 6. 保存音频 -----
        reporter.emit_saving(0.90)

        # task_id 在 _infer_impl 入口已生成（v0.3.4+ T7），此处直接复用
        output_filename = f"{task_id}.{inputs.output_format}"
        output_path = self.get_output_path("audio", output_filename)

        try:
            saved_audio_path = await asyncio.to_thread(
                save_audio,
                samples,
                sample_rate,
                output_path,
                inputs.output_format,
            )
        except Exception as e:
            logger.exception("保存音频失败")
            reporter.emit_error(f"保存音频失败: {e}", "save_failed")
            return

        # ----- 7. V2A 模式：合成带音轨的 mp4 -----
        saved_video_path = None
        if save_video and is_v2a:
            video_output_filename = f"{task_id}.mp4"
            video_output_path = self.get_output_path("video", video_output_filename)
            duration = get_audio_duration(samples, sample_rate)

            try:
                saved_video_path = await asyncio.to_thread(
                    save_video_with_audio,
                    Path(video_path),
                    saved_audio_path,
                    video_output_path,
                    duration,
                )
            except Exception as e:
                logger.exception("合成视频失败（音频已保存，仅视频合成失败）")
                # 视频合成失败不阻塞，仍返回音频
                saved_video_path = None

        # ----- 8. 上报完成 -----
        # audio_url 为相对路径（server.py 把 output_root 挂载到 /output）
        audio_url = f"/output/audio/{saved_audio_path.name}"
        # audio_path 为本地绝对路径（用于 Gradio UI 播放，绕过 SSRF 防护）
        audio_path = str(saved_audio_path.resolve())
        duration = get_audio_duration(samples, sample_rate)

        video_url = ""
        video_path_abs = ""
        if saved_video_path is not None:
            video_url = f"/output/video/{saved_video_path.name}"
            video_path_abs = str(saved_video_path.resolve())

        logger.info(
            "V2A 推理完成: task_id=%s, mode=%s, duration=%.2fs, "
            "audio_url=%s, video_url=%s",
            task_id, mode, duration, audio_url, video_url,
        )

        await reporter.emit_completed(
            audio_url=audio_url,
            duration=duration,
            sample_rate=sample_rate,
            mode=mode,
            audio_path=audio_path,
            video_url=video_url,
            video_path=video_path_abs,
        )

    async def _status_impl(self) -> StatusResponse:
        """GET /status：返回引擎状态。"""
        vram_used = self.wrapper.get_vram_used()
        return StatusResponse(
            state=self.state_machine.state.value,
            loaded_variant=self.wrapper.loaded_variant,
            vram_used=f"{vram_used}MB",
            loaded_loras=[],                      # v0.1 不支持 LoRA
            last_used_at=self.state_machine.last_used_at,
            uptime=self.state_machine.uptime,
            extra={
                "device": self.wrapper.device,
                "device_name": self.wrapper.device_name,
                "sample_rate": self.wrapper.sample_rate,
                "is_loaded": self.wrapper.is_loaded,
            },
        )

    async def _load_lora_impl(self, name: str, weight: float) -> LoraLoadResponse:
        """POST /lora/load：v0.1 不支持，返回 501。"""
        from fastapi import HTTPException
        raise HTTPException(
            status_code=501,
            detail="FoleyCrafter V2A 引擎 v0.1 不支持运行时 LoRA 切换",
        )

    async def _unload_lora_impl(self, name: str) -> LoraUnloadResponse:
        """POST /lora/unload：v0.1 不支持，返回 501。"""
        from fastapi import HTTPException
        raise HTTPException(
            status_code=501,
            detail="FoleyCrafter V2A 引擎 v0.1 不支持运行时 LoRA 切换",
        )
