"""音频/视频处理工具。

提供音频保存、视频合成（V2A 模式）、时长计算等功能。

V2A 模式输出：
    - wav 文件：纯音频（save_audio）
    - mp4 文件：带音轨的视频（save_video_with_audio，使用 moviepy 合成）
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


def save_audio(
    samples: np.ndarray,
    sample_rate: int,
    output_path: Path,
    format: str = "wav",
) -> Path:
    """保存音频到文件。

    Args:
        samples: 音频采样数据，形状 (samples,) 或 (channels, samples)
        sample_rate: 采样率
        output_path: 输出路径
        format: 音频格式（wav/mp3/flac）

    Returns:
        实际保存路径
    """
    # 确保目录存在
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # 确保数据类型为 float32
    if samples.dtype != np.float32:
        samples = samples.astype(np.float32)

    # 归一化到 [-1.0, 1.0]
    max_val = np.max(np.abs(samples))
    if max_val > 1.0:
        samples = samples / max_val

    try:
        import soundfile as sf

        # 根据格式选择子类型
        if format == "wav":
            subtype = "PCM_16"
        elif format == "flac":
            subtype = "PCM_16"
        elif format == "mp3":
            # soundfile 可能不支持 mp3 写入，fallback 到 wav
            logger.warning("soundfile 不支持 mp3 写入，fallback 到 wav")
            format = "wav"
            output_path = output_path.with_suffix(".wav")
            subtype = "PCM_16"
        else:
            subtype = "PCM_16"

        sf.write(
            str(output_path),
            samples,
            sample_rate,
            subtype=subtype,
        )

        duration = len(samples) / sample_rate
        logger.info(
            "音频已保存: %s, format=%s, sr=%d, duration=%.2fs",
            output_path,
            format,
            sample_rate,
            duration,
        )
        return output_path

    except ImportError:
        raise RuntimeError("soundfile 未安装，无法保存音频")
    except Exception as e:
        raise RuntimeError(f"保存音频失败: {e}") from e


def save_video_with_audio(
    video_path: Path,
    audio_path: Path,
    output_path: Path,
    duration: Optional[float] = None,
) -> Path:
    """合成带音轨的视频（V2A 模式专用）。

    用 moviepy 把生成的音频与原视频合成为新的 mp4 文件。

    Args:
        video_path: 原视频路径（无声）
        audio_path: 生成的音频路径（wav）
        output_path: 输出 mp4 路径
        duration: 视频时长（秒），None 则自动取原视频时长

    Returns:
        实际保存路径
    """
    # 确保输出目录存在
    output_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        from moviepy.editor import AudioFileClip, VideoFileClip

        logger.info(
            "开始合成视频: video=%s, audio=%s, output=%s",
            video_path, audio_path, output_path,
        )

        audio_clip = AudioFileClip(str(audio_path))
        video_clip = VideoFileClip(str(video_path))

        # 截取音频到视频时长
        if duration is None:
            duration = video_clip.duration
        audio_clip = audio_clip.subclip(0, duration)

        # 设置视频音频
        video_clip = video_clip.set_audio(audio_clip)
        video_clip = video_clip.subclip(0, duration)

        # 写入 mp4（libx264 + libmp3lame）
        video_clip.write_videofile(
            str(output_path),
            codec="libx264",
            audio_codec="libmp3lame",
            logger=None,  # 禁用 moviepy 进度条
        )

        # 关闭 clip 释放资源
        video_clip.close()
        audio_clip.close()

        logger.info(
            "视频已合成: %s, duration=%.2fs",
            output_path,
            duration,
        )
        return output_path

    except ImportError:
        raise RuntimeError("moviepy 未安装，无法合成视频")
    except Exception as e:
        raise RuntimeError(f"合成视频失败: {e}") from e


def get_audio_duration(samples: np.ndarray, sample_rate: int) -> float:
    """计算音频时长（秒）。"""
    if samples.ndim == 1:
        return len(samples) / sample_rate
    return samples.shape[-1] / sample_rate


def get_video_duration(video_path: Path) -> float:
    """获取视频时长（秒）。

    Args:
        video_path: 视频文件路径

    Returns:
        视频时长（秒），失败返回 0.0
    """
    try:
        from moviepy.editor import VideoFileClip
        clip = VideoFileClip(str(video_path))
        duration = clip.duration
        clip.close()
        return float(duration)
    except Exception as e:
        logger.warning("获取视频时长失败: %s, error: %s", video_path, e)
        return 0.0


def normalize_audio(samples: np.ndarray) -> np.ndarray:
    """归一化音频到 [-1.0, 1.0]。"""
    max_val = np.max(np.abs(samples))
    if max_val > 0:
        return samples / max_val
    return samples
