"""FoleyCrafter V2A 引擎数据模型。

定义 Pydantic 模型用于 /infer 端点请求解析与 /capabilities 响应。

V2AInputs 设计要点：
    - prompt / negative_prompt 可选（空时仅靠视频语义）
    - video_path 由 audios[role=video] 注入（无需客户端在 inputs 中传）
    - max_video_duration 可覆盖 EngineConfig 默认值
    - save_video 自动判断（V2A 模式 True，T2A 模式 False）
"""

from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, ConfigDict, Field

from aireliable_engine import VariantInfo


# ===== V2A 输入模型 =====

class V2AInputs(BaseModel):
    """FoleyCrafter V2A 推理输入。

    通过 InferRequest.inputs 字段传入。
    """

    model_config = ConfigDict(extra="allow")

    # ===== 文本输入（可选） =====
    prompt: str = Field(
        default="",
        description="正向文本提示（可空，空时仅靠视频语义）",
    )
    negative_prompt: str = Field(
        default="",
        description="负向文本提示（控制不希望出现的音效）",
    )

    # ===== 视频输入（可选，由 audios[role=video] 注入） =====
    # 客户端通常不直接传 video_path，而是通过 InferRequest.audios[role=video] 传入
    # engine._infer_impl 会从中提取并填入此字段
    video_path: str = Field(
        default="",
        description="视频文件绝对路径（由 audios[role=video] 注入，T2A 模式为空）",
    )

    # ===== 推理参数 =====
    seed: int = Field(
        default=42,
        description="随机种子（默认 42）",
    )
    temporal_align: bool = Field(
        default=True,
        description="是否启用时间对齐（temporal_adapter）",
    )
    temporal_scale: float = Field(
        default=0.2,
        ge=0.0,
        le=1.0,
        description="时间对齐强度（默认 0.2）",
    )
    semantic_scale: float = Field(
        default=1.0,
        ge=0.0,
        le=2.0,
        description="语义强度（IP-Adapter scale，默认 1.0）",
    )
    num_inference_steps: int = Field(
        default=25,
        ge=1,
        le=100,
        description="扩散推理步数（默认 25）",
    )
    max_video_duration: float = Field(
        default=10.0,
        gt=0,
        description="视频长度限制（秒，超过则截断）",
    )

    # ===== 输出参数 =====
    output_format: str = Field(
        default="wav",
        description="音频输出格式（wav / mp3）",
    )
    # save_video 由 _infer_impl 自动判断：
    #   - V2A 模式（video_path 非空）：True，输出 wav + mp4
    #   - T2A 模式（video_path 为空）：False，仅输出 wav
    # 客户端可显式覆盖
    save_video: Optional[bool] = Field(
        default=None,
        description="是否合成带音轨的 mp4（None=自动判断）",
    )


# ===== V2A 变体信息（扩展 SDK VariantInfo） =====

class V2AVariantInfo(VariantInfo):
    """V2A 变体能力声明（扩展 SDK VariantInfo）。

    在标准字段之外补充 V2A 特有信息：默认推理步数 / 时间对齐 / 语义强度。
    Pydantic model_dump() 会序列化全部字段，前端可读取这些扩展字段
    渲染 V2A 专属 UI。
    """

    model_config = ConfigDict(extra="allow")

    default_steps: int = Field(default=25, description="默认扩散推理步数")
    default_temporal_scale: float = Field(default=0.2, description="默认时间对齐强度")
    default_semantic_scale: float = Field(default=1.0, description="默认语义强度")
    supports_v2a: bool = Field(default=True, description="是否支持 V2A（视频转音效）")
    supports_t2a: bool = Field(default=True, description="是否支持 T2A（文本转音效）")
