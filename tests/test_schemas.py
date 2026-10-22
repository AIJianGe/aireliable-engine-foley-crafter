"""V2AInputs / V2AVariantInfo 单元测试（T13）。

覆盖：
- V2AInputs 默认值 / 字段约束 / extra='allow' 行为
- V2AVariantInfo 扩展字段（V2A 特有）
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from engines.foleycrafter_a3b2c1.schemas import V2AInputs, V2AVariantInfo


class TestV2AInputs:
    """V2AInputs 测试。"""

    def test_default_values(self):
        """默认值应与分析文档决策一致。"""
        inputs = V2AInputs()
        assert inputs.prompt == ""
        assert inputs.negative_prompt == ""
        assert inputs.video_path == ""
        assert inputs.seed == 42
        assert inputs.temporal_align is True
        assert inputs.temporal_scale == 0.2          # 决策 6 默认
        assert inputs.semantic_scale == 1.0
        assert inputs.num_inference_steps == 25
        assert inputs.max_video_duration == 10.0     # 决策 6 默认 10s
        assert inputs.output_format == "wav"
        assert inputs.save_video is None             # None = 自动判断

    def test_prompt_only_t2a(self):
        """T2A 模式：仅 prompt，无视频路径。"""
        inputs = V2AInputs(prompt="一声枪响")
        assert inputs.prompt == "一声枪响"
        assert inputs.video_path == ""

    def test_v2a_with_video_path(self):
        """V2A 模式：含视频路径。"""
        inputs = V2AInputs(
            prompt="汽车引擎",
            video_path="/path/to/video.mp4",
        )
        assert inputs.video_path == "/path/to/video.mp4"

    def test_max_video_duration_must_be_positive(self):
        """max_video_duration 必须 > 0。"""
        with pytest.raises(ValidationError):
            V2AInputs(max_video_duration=0)
        with pytest.raises(ValidationError):
            V2AInputs(max_video_duration=-1.0)

    def test_temporal_scale_range(self):
        """temporal_scale 应在 [0.0, 1.0] 范围内。"""
        V2AInputs(temporal_scale=0.0)    # 边界值合法
        V2AInputs(temporal_scale=1.0)    # 边界值合法
        with pytest.raises(ValidationError):
            V2AInputs(temporal_scale=-0.1)
        with pytest.raises(ValidationError):
            V2AInputs(temporal_scale=1.1)

    def test_semantic_scale_range(self):
        """semantic_scale 应在 [0.0, 2.0] 范围内。"""
        V2AInputs(semantic_scale=0.0)
        V2AInputs(semantic_scale=2.0)
        with pytest.raises(ValidationError):
            V2AInputs(semantic_scale=-0.1)
        with pytest.raises(ValidationError):
            V2AInputs(semantic_scale=2.1)

    def test_num_inference_steps_range(self):
        """num_inference_steps 应在 [1, 100] 范围内。"""
        V2AInputs(num_inference_steps=1)
        V2AInputs(num_inference_steps=100)
        with pytest.raises(ValidationError):
            V2AInputs(num_inference_steps=0)
        with pytest.raises(ValidationError):
            V2AInputs(num_inference_steps=101)

    def test_extra_fields_allowed(self):
        """extra='allow' 应允许额外字段透传。"""
        inputs = V2AInputs(prompt="test", custom_field="custom_value")
        assert inputs.prompt == "test"
        # 额外字段应保留在 model_extra
        assert inputs.model_extra.get("custom_field") == "custom_value"

    def test_save_video_explicit_override(self):
        """save_video 可显式覆盖（如 T2A 强制生成视频 = False）。"""
        inputs = V2AInputs(prompt="test", save_video=False)
        assert inputs.save_video is False


class TestV2AVariantInfo:
    """V2AVariantInfo 测试。"""

    def test_default_values(self):
        v = V2AVariantInfo(
            name="foleycrafter-v1",
            alias="FoleyCrafter-v1",
            base_model="auffusion-full-no-adapter",
            precision="fp16",
            loader="foleycrafter",
            vram_required="10GB",
        )
        # 默认值
        assert v.default_steps == 25
        assert v.default_temporal_scale == 0.2
        assert v.default_semantic_scale == 1.0
        assert v.supports_v2a is True
        assert v.supports_t2a is True

    def test_custom_values(self):
        v = V2AVariantInfo(
            name="custom",
            alias="Custom",
            base_model="custom-base",
            precision="fp32",
            loader="custom-loader",
            vram_required="20GB",
            default_steps=50,
            default_temporal_scale=0.5,
            default_semantic_scale=1.5,
            supports_v2a=True,
            supports_t2a=False,
        )
        assert v.default_steps == 50
        assert v.supports_t2a is False

    def test_model_dump_includes_v2a_fields(self):
        """model_dump() 应包含 V2A 扩展字段。"""
        v = V2AVariantInfo(
            name="test",
            alias="Test",
            base_model="base",
            precision="fp16",
            loader="loader",
            vram_required="10GB",
        )
        d = v.model_dump()
        assert "default_steps" in d
        assert "default_temporal_scale" in d
        assert "default_semantic_scale" in d
        assert "supports_v2a" in d
        assert "supports_t2a" in d
