"""SSE 进度上报工具（FoleyCrafter V2A 专用）。

把 FoleyCrafter 的推理过程转换为 AIReliable 进度帧。

V2A 模式进度阶段：
    start              0.00          开始 V2A 推理
    video_decoding     0.05-0.15     视频解码（moviepy 读帧）
    timestamp_detect   0.15-0.25     时间戳检测（time_detector）
    image_encoding     0.25-0.40     CLIP 视觉编码
    audio_generating   0.40-0.85     扩散推理（pipe()，25 步）
    saving             0.85-0.95     保存音频/视频
    completed          1.00          完成

T2A 模式跳过 video_decoding / timestamp_detect / image_encoding 阶段，
直接从 start 跳到 audio_generating。

v0.3.4+ 集成 SDK 任务上报：
    - emit_start → fire_report_running（gateway status=running）
    - emit (通用) → fire_report_progress（gateway 更新进度）
    - emit_completed → fire_report_succeeded（gateway status=succeeded）
    - emit_error → fire_report_failed（gateway status=failed）
    - 引擎 _infer_impl 入口调 report_created（gateway status=queued）
    详见 TASK_CENTER_V034_ANALYSIS.md T6
"""

from __future__ import annotations

import logging
import time
from typing import Any, Optional

from aireliable_engine import (
    SSEStream,
    make_frame,
    make_final_frame,
    make_error_frame,
    PlatformTaskReporter,
    get_platform_task_reporter,
)

logger = logging.getLogger(__name__)


# ===== 进度阶段定义 =====

class ProgressStage:
    """V2A 推理进度阶段。"""

    START = "v2a.start"                        # 0.00
    VIDEO_DECODING = "v2a.video_decoding"      # 0.05 - 0.15
    TIMESTAMP_DETECT = "v2a.timestamp_detect"  # 0.15 - 0.25
    IMAGE_ENCODING = "v2a.image_encoding"      # 0.25 - 0.40
    AUDIO_GENERATING = "v2a.audio_generating"  # 0.40 - 0.85
    SAVING = "v2a.saving"                      # 0.85 - 0.95
    COMPLETED = "v2a.completed"                # 1.00
    ERROR = "v2a.error"


# 进度区间
STAGE_RANGES: dict[str, tuple[float, float]] = {
    ProgressStage.START: (0.00, 0.05),
    ProgressStage.VIDEO_DECODING: (0.05, 0.15),
    ProgressStage.TIMESTAMP_DETECT: (0.15, 0.25),
    ProgressStage.IMAGE_ENCODING: (0.25, 0.40),
    ProgressStage.AUDIO_GENERATING: (0.40, 0.85),
    ProgressStage.SAVING: (0.85, 0.95),
}


class ProgressReporter:
    """V2A 进度上报器。

    封装 SSEStream，提供按阶段构造帧的便捷方法。
    v0.3.4+ 同时通过 SDK PlatformTaskReporter 上报任务状态到 gateway。
    """

    def __init__(self, stream: SSEStream, task_id: str = ""):
        """初始化进度上报器。

        Args:
            stream: SSE 流
            task_id: 任务 ID（v0.3.4+，用于 SDK 上报；为空则不上报）
        """
        self.stream = stream
        self.task_id = task_id
        self._start_time: float = time.time()
        # v0.3.4+ SDK 任务上报器（惰性初始化，需在 async 上下文中调 init_task_reporter）
        self._task_reporter: Optional[PlatformTaskReporter] = None

    @property
    def elapsed(self) -> float:
        """已用时（秒）。"""
        return time.time() - self._start_time

    # ===== v0.3.4+ SDK 任务上报集成 =====

    async def init_task_reporter(self) -> None:
        """v0.3.4+ 异步初始化 SDK 任务上报器（捕获主 loop + 获取单例）。

        必须在 emit_* 方法之前调用（在 _infer_impl 入口处）。
        失败不阻断推理。
        """
        if self._task_reporter is None and self.task_id:
            try:
                self._task_reporter = await get_platform_task_reporter()
            except Exception as e:
                logger.warning("init_task_reporter 失败（不阻断推理）: %s", e)

    async def report_created(
        self,
        request_summary: dict,
        variant: str = "",
    ) -> None:
        """v0.3.4+ 上报任务创建到 gateway（status=queued）。

        Args:
            request_summary: 请求摘要（如 {"prompt": "...", "mode": "v2a"}）
            variant: variant 名
        """
        if self._task_reporter and self.task_id:
            try:
                await self._task_reporter.report_created(
                    task_id=self.task_id,
                    request_summary=request_summary,
                    variant=variant,
                )
            except Exception as e:
                logger.warning("report_created 失败（不阻断推理）: %s", e)

    def _fire_running(self, stage: str) -> None:
        """v0.3.4+ fire-and-forget 上报任务进入 running 状态。"""
        if self._task_reporter and self.task_id:
            self._task_reporter.fire_report_running(self.task_id, stage=stage)

    def _fire_progress(self, progress: float, stage: str) -> None:
        """v0.3.4+ fire-and-forget 上报任务进度。"""
        if self._task_reporter and self.task_id:
            self._task_reporter.fire_report_progress(
                self.task_id, progress=progress, stage=stage,
            )

    def _fire_succeeded(self, result: dict) -> None:
        """v0.3.4+ fire-and-forget 上报任务成功。"""
        if self._task_reporter and self.task_id:
            self._task_reporter.fire_report_succeeded(self.task_id, result=result)

    def _fire_failed(self, error: dict) -> None:
        """v0.3.4+ fire-and-forget 上报任务失败。"""
        if self._task_reporter and self.task_id:
            self._task_reporter.fire_report_failed(self.task_id, error=error)

    # ===== 通用帧 =====

    def emit(self, progress: float, stage: str, **extra: Any) -> None:
        """发送一帧进度（线程安全）。"""
        frame = make_frame(progress, stage, **extra)
        self.stream.put_threadsafe(frame)
        # v0.3.4+ 同步上报进度到 gateway（fire-and-forget，不阻塞 SSE 流）
        self._fire_progress(progress, stage)

    async def emit_async(self, progress: float, stage: str, **extra: Any) -> None:
        """发送一帧进度（协程内调用）。"""
        frame = make_frame(progress, stage, **extra)
        await self.stream.put(frame)
        # v0.3.4+ 同步上报进度到 gateway
        self._fire_progress(progress, stage)

    def emit_error(self, message: str, error_type: str = "v2a_error") -> None:
        """发送错误帧。"""
        frame = make_error_frame(message, error_type)
        self.stream.put_threadsafe(frame)
        # v0.3.4+ 上报任务失败到 gateway
        self._fire_failed({"type": error_type, "message": message})

    # ===== 阶段便捷方法 =====

    def emit_start(self, prompt: str, mode: str, task_id: str = "") -> None:
        """开始推理。

        Args:
            prompt: 文本提示
            mode: 模式（v2a / t2a）
            task_id: 任务 ID（v0.3.4+ 已迁移到 __init__，此处保留兼容）
        """
        self.emit(
            0.0,
            ProgressStage.START,
            message=f"开始 {mode.upper()} 推理",
            prompt=prompt[:100],
            mode=mode,
            task_id=task_id or self.task_id,
        )
        # v0.3.4+ 上报任务进入 running 状态
        self._fire_running(ProgressStage.START)

    def emit_video_decoding(
        self,
        progress: float = 0.10,
        frames: int = 0,
        duration: float = 0.0,
    ) -> None:
        """视频解码中（V2A 模式专用）。"""
        self.emit(
            progress,
            ProgressStage.VIDEO_DECODING,
            message="解码视频帧",
            frames=frames,
            duration=duration,
        )

    def emit_timestamp_detect(self, progress: float = 0.20) -> None:
        """时间戳检测中（V2A 模式专用）。"""
        self.emit(
            progress,
            ProgressStage.TIMESTAMP_DETECT,
            message="检测视频时间戳",
        )

    def emit_image_encoding(self, progress: float = 0.35) -> None:
        """CLIP 视觉编码中（V2A 模式专用）。"""
        self.emit(
            progress,
            ProgressStage.IMAGE_ENCODING,
            message="编码视频视觉特征",
        )

    def emit_audio_generating(
        self,
        progress: float = 0.50,
        step: int = 0,
        total_steps: int = 25,
        **extra: Any,
    ) -> None:
        """扩散推理中。"""
        self.emit(
            progress,
            ProgressStage.AUDIO_GENERATING,
            message=f"生成音频中... ({step}/{total_steps})",
            step=step,
            total_steps=total_steps,
            **extra,
        )

    def emit_saving(self, progress: float = 0.90) -> None:
        """保存中。"""
        self.emit(
            progress,
            ProgressStage.SAVING,
            message="保存音频/视频文件",
        )

    async def emit_completed(
        self,
        audio_url: str,
        duration: float,
        sample_rate: int,
        mode: str,
        audio_path: str = "",
        video_url: str = "",
        video_path: str = "",
    ) -> None:
        """完成。

        Args:
            audio_url: 音频 URL 路径（如 /output/audio/xxx.wav），用于下载链接
            audio_path: 音频本地绝对路径，用于 Gradio UI 播放（绕过 SSRF 防护）
            video_url: 视频 URL 路径（V2A 模式才有）
            video_path: 视频本地绝对路径（V2A 模式才有）
            duration: 音频时长（秒）
            sample_rate: 采样率
            mode: 模式（v2a / t2a）
        """
        timing = {
            "total_seconds": self.elapsed,
            "audio_duration": duration,
            "sample_rate": sample_rate,
        }
        result = {
            "audio_url": audio_url,
            "audio_path": audio_path,
            "duration": duration,
            "sample_rate": sample_rate,
            "mode": mode,
        }
        # V2A 模式额外包含视频信息
        if video_url:
            result["video_url"] = video_url
            result["video_path"] = video_path
        frame = make_final_frame(result, timing)
        await self.stream.put(frame)
        # v0.3.4+ 上报任务成功到 gateway
        self._fire_succeeded(result)

    def emit_completed_threadsafe(
        self,
        audio_url: str,
        duration: float,
        sample_rate: int,
        mode: str,
        audio_path: str = "",
        video_url: str = "",
        video_path: str = "",
    ) -> None:
        """完成（线程安全版本）。"""
        timing = {
            "total_seconds": self.elapsed,
            "audio_duration": duration,
            "sample_rate": sample_rate,
        }
        result = {
            "audio_url": audio_url,
            "audio_path": audio_path,
            "duration": duration,
            "sample_rate": sample_rate,
            "mode": mode,
        }
        if video_url:
            result["video_url"] = video_url
            result["video_path"] = video_path
        frame = make_final_frame(result, timing)
        self.stream.put_threadsafe(frame)
        # v0.3.4+ 上报任务成功到 gateway
        self._fire_succeeded(result)
