"""SSE 进度上报工具（FoleyCrafter V2A / T2A 专用）。

把视频转音频的推理过程转换为 AIReliable 进度帧。

进度阶段：
    start             0.00          开始推理
    video_decoding    0.05 - 0.15   解码视频帧（V2A）
    timestamp_detect  0.15 - 0.25   检测时间戳
    image_encoding    0.25 - 0.40   CLIP 视觉编码
    audio_generating  0.40 - 0.85   扩散生成音频
    saving            0.85 - 0.95   保存音频/视频
    completed         1.00          完成

任务 71 P0-B：通用帧构造 / 任务上报 / 心跳 / 完成帧全部由 SDK
``BaseProgressReporter`` 提供，本文件只保留阶段常量与语义化方法。
"""

from __future__ import annotations

from typing import Any

from aireliable_engine.progress import BaseProgressReporter


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


class ProgressReporter(BaseProgressReporter):
    """V2A 进度上报器（阶段语义 + SDK 通用帧）。"""

    error_type = "v2a_error"

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
        result: dict[str, Any] = {
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
        await super().emit_completed(
            result,
            {
                "total_seconds": self.elapsed,
                "audio_duration": duration,
                "sample_rate": sample_rate,
            },
        )

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
        result: dict[str, Any] = {
            "audio_url": audio_url,
            "audio_path": audio_path,
            "duration": duration,
            "sample_rate": sample_rate,
            "mode": mode,
        }
        if video_url:
            result["video_url"] = video_url
            result["video_path"] = video_path
        super().emit_completed_threadsafe(
            result,
            {
                "total_seconds": self.elapsed,
                "audio_duration": duration,
                "sample_rate": sample_rate,
            },
        )
