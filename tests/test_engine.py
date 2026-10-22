"""FoleyCrafterEngine 单元测试（T13）。

覆盖 7 个 `_xxx_impl` 方法：
- _capabilities_impl：返回扫描到的变体（含 V2A/T2A 支持标记）
- _load_impl：调用 wrapper.load，未找到 variant 抛错
- _unload_impl：未加载时返回 idle
- _infer_impl：V2A / T2A 双模式参数校验、推理成功/失败路径、视频输入解析
- _status_impl：返回设备/采样率等额外信息
- _load_lora_impl / _unload_lora_impl：v0.1 返回 501

Mock FoleyCrafterWrapper 和 ModelScanner 避免真实模型加载。
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Optional
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from fastapi import HTTPException

from aireliable_engine.schemas import AudioInput, InferRequest

from engines.foleycrafter_a3b2c1.config import EngineConfig
from engines.foleycrafter_a3b2c1.schemas import V2AInputs, V2AVariantInfo


# ===== Mock 工厂 =====


def _make_variant(
    name: str = "foleycrafter-v1",
    supports_v2a: bool = True,
    supports_t2a: bool = True,
) -> V2AVariantInfo:
    return V2AVariantInfo(
        name=name,
        alias=name.upper(),
        base_model="auffusion-full-no-adapter",
        precision="fp16",
        loader="foleycrafter",
        vram_required="10GB",
        available=True,
        default_steps=25,
        default_temporal_scale=0.2,
        default_semantic_scale=1.0,
        supports_v2a=supports_v2a,
        supports_t2a=supports_t2a,
    )


def _make_mock_wrapper(
    is_loaded: bool = False,
    loaded_variant: Optional[str] = None,
    device: str = "cpu",
    sample_rate: int = 16000,
    vram_used: int = 0,
    generate_samples: Optional[np.ndarray] = None,
    fail_generate: bool = False,
    fail_load: bool = False,
) -> MagicMock:
    """构造一个 Mock 的 FoleyCrafterWrapper。"""
    wrapper = MagicMock()
    wrapper.is_loaded = is_loaded
    # 默认空字符串（与真实 wrapper 一致，None 仅在显式传入时使用）
    wrapper.loaded_variant = loaded_variant if loaded_variant is not None else ""
    wrapper.device = device
    wrapper.device_name = "mock-cpu"
    wrapper.sample_rate = sample_rate
    wrapper.get_vram_used.return_value = vram_used

    if fail_load:
        def _fail_load(*a, **kw):
            raise RuntimeError("mock load failed")
        wrapper.load.side_effect = _fail_load
    else:
        wrapper.load.return_value = None

    if fail_generate:
        def _fail_gen(*a, **kw):
            raise RuntimeError("mock generate failed")
        wrapper.generate.side_effect = _fail_gen
    else:
        samples = generate_samples if generate_samples is not None else np.zeros(
            sample_rate, dtype=np.float32
        )

        def _generate_with_progress(*a, **kw):
            # 模拟扩散推理的进度回调（25 步），触发 v2a.audio_generating 帧
            progress_callback = kw.get("progress_callback")
            num_steps = kw.get("num_inference_steps", 25)
            if progress_callback is not None:
                for step in range(1, num_steps + 1):
                    progress_callback(step, num_steps)
            return (samples, sample_rate)

        wrapper.generate.side_effect = _generate_with_progress

    wrapper.unload.return_value = None
    return wrapper


# sentinel 用于区分 "未传参" 和 "显式传 None"
_SENTINEL = object()


def _make_mock_scanner(
    variants: Optional[list] = None,
    variant_path: object = _SENTINEL,
) -> MagicMock:
    """构造一个 Mock 的 ModelScanner。

    Args:
        variants: 变体列表，None 表示用默认
        variant_path: resolve_variant_path 的返回值。
                      未传（_SENTINEL）用默认路径，显式传 None 返回 None。
    """
    scanner = MagicMock()
    if variants is None:
        variants = [_make_variant()]
    scanner.scan_variants.return_value = variants
    if variant_path is _SENTINEL:
        variant_path = Path("/fake/foleycrafter-v1")
    scanner.resolve_variant_path.return_value = variant_path
    return scanner


def _fake_save_audio(samples, sr, path, fmt="wav"):
    """模拟 save_audio：直接写空文件，避免依赖 soundfile。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"FAKE_WAV")
    return path


def _fake_get_duration(samples, sr):
    """模拟 get_audio_duration。"""
    return 1.0


def _fake_save_video_with_audio(video_path, audio_path, output_path, duration=None):
    """模拟 save_video_with_audio：直接写空文件，避免依赖 moviepy。"""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_bytes(b"FAKE_MP4")
    return output_path


@pytest.fixture
def mock_audio_utils():
    """mock engine 模块中的 save_audio / get_audio_duration / save_video_with_audio，
    避免依赖 soundfile / moviepy。"""
    with patch(
        "engines.foleycrafter_a3b2c1.engine.save_audio",
        side_effect=_fake_save_audio,
    ), patch(
        "engines.foleycrafter_a3b2c1.engine.get_audio_duration",
        side_effect=_fake_get_duration,
    ), patch(
        "engines.foleycrafter_a3b2c1.engine.save_video_with_audio",
        side_effect=_fake_save_video_with_audio,
    ):
        yield


def _make_engine(
    tmp_path: Path,
    wrapper: Optional[MagicMock] = None,
    scanner: Optional[MagicMock] = None,
) -> "FoleyCrafterEngine":
    """构造一个带 mock 依赖的 FoleyCrafterEngine。

    注意：save_audio / get_audio_duration / save_video_with_audio 的 mock
    由 ``mock_audio_utils`` fixture 提供，需要在测试方法中声明该 fixture。
    """
    from engines.foleycrafter_a3b2c1.engine import FoleyCrafterEngine

    config = EngineConfig(
        models_root=tmp_path / "models",
        data_root=tmp_path / "data",
        project_root=tmp_path,
        device="cpu",
    )

    with patch(
        "engines.foleycrafter_a3b2c1.engine.FoleyCrafterWrapper"
    ) as MockWrapper, patch(
        "engines.foleycrafter_a3b2c1.engine.ModelScanner"
    ) as MockScanner:
        MockWrapper.return_value = wrapper or _make_mock_wrapper()
        MockScanner.return_value = scanner or _make_mock_scanner()
        engine = FoleyCrafterEngine(config)
        # 把 mock 引用挂上去便于断言
        engine.wrapper = MockWrapper.return_value
        engine.scanner = MockScanner.return_value
    return engine


def _parse_sse_frames(frames: list[str]) -> list[dict]:
    """解析 SSE 帧字符串列表为 dict 列表。"""
    out = []
    for f in frames:
        body = f
        if body.startswith("data: "):
            body = body[len("data: "):]
        if body.endswith("\n\n"):
            body = body[:-2]
        out.append(json.loads(body))
    return out


# ===== 测试类 =====


class TestCapabilitiesImpl:
    """_capabilities_impl 测试。"""

    @pytest.mark.asyncio
    async def test_returns_scanned_variants(self, tmp_path: Path):
        v1 = _make_variant(name="foleycrafter-v1")
        scanner = _make_mock_scanner(variants=[v1])
        eng = _make_engine(tmp_path, scanner=scanner)

        r = await eng._capabilities_impl()

        assert r.engine_id == eng.engine_id
        assert r.capability == "video-to-audio"
        assert r.type == "audio"                  # 新增 audio 类型
        assert len(r.variants) == 1
        assert r.variants[0].name == "foleycrafter-v1"
        # V2A 特有字段
        assert r.variants[0].supports_v2a is True
        assert r.variants[0].supports_t2a is True
        assert r.variants[0].default_steps == 25

    @pytest.mark.asyncio
    async def test_empty_variants_when_no_models(self, tmp_path: Path):
        scanner = _make_mock_scanner(variants=[])
        eng = _make_engine(tmp_path, scanner=scanner)

        r = await eng._capabilities_impl()
        assert r.variants == []
        assert r.loras == []


class TestLoadImpl:
    """_load_impl 测试。"""

    @pytest.mark.asyncio
    async def test_load_success(self, tmp_path: Path):
        wrapper = _make_mock_wrapper()
        scanner = _make_mock_scanner(variant_path=Path("/fake/foleycrafter-v1"))
        eng = _make_engine(tmp_path, wrapper=wrapper, scanner=scanner)

        r = await eng._load_impl("foleycrafter-v1")

        wrapper.load.assert_called_once()
        # load 应该被传入 (variant_name, variant_path, device)
        call_args = wrapper.load.call_args
        assert call_args[0][0] == "foleycrafter-v1"
        assert "foleycrafter-v1" in str(call_args[0][1])
        assert r.variant == "foleycrafter-v1"
        assert r.state == "ready"
        assert "MB" in r.vram_used

    @pytest.mark.asyncio
    async def test_load_unknown_variant_raises(self, tmp_path: Path):
        scanner = _make_mock_scanner(variant_path=None)
        eng = _make_engine(tmp_path, scanner=scanner)

        with pytest.raises(FileNotFoundError, match="variant 不存在"):
            await eng._load_impl("unknown")

    @pytest.mark.asyncio
    async def test_load_failure_wraps_error(self, tmp_path: Path):
        wrapper = _make_mock_wrapper(fail_load=True)
        scanner = _make_mock_scanner(variant_path=Path("/fake/foleycrafter-v1"))
        eng = _make_engine(tmp_path, wrapper=wrapper, scanner=scanner)

        with pytest.raises(RuntimeError, match="FoleyCrafter 模型加载失败"):
            await eng._load_impl("foleycrafter-v1")


class TestUnloadImpl:
    """_unload_impl 测试。"""

    @pytest.mark.asyncio
    async def test_unload_when_not_loaded(self, tmp_path: Path):
        wrapper = _make_mock_wrapper(is_loaded=False, vram_used=0)
        eng = _make_engine(tmp_path, wrapper=wrapper)

        r = await eng._unload_impl()

        wrapper.unload.assert_not_called()
        assert r.state == "idle"
        assert "0MB" in r.vram_freed

    @pytest.mark.asyncio
    async def test_unload_when_loaded(self, tmp_path: Path):
        wrapper = _make_mock_wrapper(is_loaded=True, vram_used=2048)
        eng = _make_engine(tmp_path, wrapper=wrapper)

        r = await eng._unload_impl()

        wrapper.unload.assert_called_once()
        assert r.state == "idle"
        assert "2048MB" in r.vram_freed


class TestInferImpl:
    """_infer_impl 测试。"""

    @pytest.mark.asyncio
    async def test_infer_t2a_mode_success(self, tmp_path: Path, mock_audio_utils):
        """T2A 模式：仅 prompt 输入，生成 wav 文件。"""
        wrapper = _make_mock_wrapper(
            is_loaded=True, loaded_variant="foleycrafter-v1",
            generate_samples=np.zeros(16000, dtype=np.float32),
        )
        eng = _make_engine(tmp_path, wrapper=wrapper)
        eng.output_root = tmp_path / "output"

        req = InferRequest(
            variant="foleycrafter-v1",
            inputs={"prompt": "一声枪响"},
            task_id="t2a-task-1",
        )

        frames: list[str] = []
        async for frame in eng.infer(req):
            frames.append(frame)

        parsed = _parse_sse_frames(frames)
        stages = [f.get("stage") for f in parsed]

        # T2A 模式应包含 start / audio_generating / saving / completed
        assert "v2a.start" in stages
        assert "completed" in stages
        # T2A 模式不应包含 video_decoding / timestamp_detect / image_encoding
        assert "v2a.video_decoding" not in stages
        assert "v2a.timestamp_detect" not in stages
        assert "v2a.image_encoding" not in stages

        # 最后一帧含 audio_url，无 video_url（T2A 模式）
        last = parsed[-1]
        assert last["stage"] == "completed"
        assert "/output/audio/t2a-task-1.wav" in last["result"]["audio_url"]
        assert last["result"]["mode"] == "t2a"
        assert "video_url" not in last["result"] or not last["result"].get("video_url")
        wrapper.generate.assert_called_once()

        # 检查 generate 调用参数：video_path 应为空（T2A 模式）
        _, kwargs = wrapper.generate.call_args
        assert kwargs.get("video_path") == ""

    @pytest.mark.asyncio
    async def test_infer_v2a_mode_success(self, tmp_path: Path, mock_audio_utils):
        """V2A 模式：视频输入，生成 wav + mp4 两个文件。"""
        wrapper = _make_mock_wrapper(
            is_loaded=True, loaded_variant="foleycrafter-v1",
            generate_samples=np.zeros(16000, dtype=np.float32),
        )
        eng = _make_engine(tmp_path, wrapper=wrapper)
        eng.output_root = tmp_path / "output"

        req = InferRequest(
            variant="foleycrafter-v1",
            inputs={"prompt": "汽车引擎声"},
            audios=[AudioInput(path="/fake/video.mp4", role="video")],
            task_id="v2a-task-1",
        )

        frames: list[str] = []
        async for frame in eng.infer(req):
            frames.append(frame)

        parsed = _parse_sse_frames(frames)
        stages = [f.get("stage") for f in parsed]

        # V2A 模式应包含全部 7 阶段
        assert "v2a.start" in stages
        assert "v2a.video_decoding" in stages
        assert "v2a.timestamp_detect" in stages
        assert "v2a.image_encoding" in stages
        assert "v2a.audio_generating" in stages
        assert "v2a.saving" in stages
        assert "completed" in stages

        # 最后一帧含 audio_url + video_url
        last = parsed[-1]
        assert last["stage"] == "completed"
        assert "/output/audio/v2a-task-1.wav" in last["result"]["audio_url"]
        assert "/output/video/v2a-task-1.mp4" in last["result"]["video_url"]
        assert last["result"]["mode"] == "v2a"
        # audio_path + video_path 为本地绝对路径
        assert last["result"]["audio_path"]
        assert last["result"]["video_path"]
        wrapper.generate.assert_called_once()

        # 检查 generate 调用参数：video_path 应来自 audios[role=video]
        _, kwargs = wrapper.generate.call_args
        assert kwargs.get("video_path") == "/fake/video.mp4"

    @pytest.mark.asyncio
    async def test_infer_invalid_inputs_emits_error(self, tmp_path: Path):
        """V2AInputs 解析失败时推送 error 帧。"""
        wrapper = _make_mock_wrapper(is_loaded=True)
        eng = _make_engine(tmp_path, wrapper=wrapper)

        # max_video_duration 必须大于 0，传 0 触发校验错误
        req = InferRequest(
            variant="foleycrafter-v1",
            inputs={"prompt": "test", "max_video_duration": 0},
        )

        frames: list[str] = []
        async for frame in eng.infer(req):
            frames.append(frame)

        parsed = _parse_sse_frames(frames)
        error_frames = [f for f in parsed if f.get("stage") == "error"]
        assert len(error_frames) == 1
        assert error_frames[0]["error"]["type"] == "invalid_inputs"
        wrapper.generate.assert_not_called()

    @pytest.mark.asyncio
    async def test_infer_generate_failure_emits_error(self, tmp_path: Path, mock_audio_utils):
        """推理异常时推送 error 帧。"""
        wrapper = _make_mock_wrapper(
            is_loaded=True, fail_generate=True,
        )
        eng = _make_engine(tmp_path, wrapper=wrapper)
        eng.output_root = tmp_path / "output"

        req = InferRequest(
            variant="foleycrafter-v1",
            inputs={"prompt": "test"},
        )

        frames: list[str] = []
        async for frame in eng.infer(req):
            frames.append(frame)

        parsed = _parse_sse_frames(frames)
        error_frames = [f for f in parsed if f.get("stage") == "error"]
        assert len(error_frames) == 1
        assert error_frames[0]["error"]["type"] == "infer_failed"

    @pytest.mark.asyncio
    async def test_infer_v2a_video_synthesis_failure_does_not_block(
        self, tmp_path: Path, mock_audio_utils
    ):
        """V2A 模式下视频合成失败时，音频仍应正常返回。"""
        wrapper = _make_mock_wrapper(
            is_loaded=True, loaded_variant="foleycrafter-v1",
            generate_samples=np.zeros(16000, dtype=np.float32),
        )
        eng = _make_engine(tmp_path, wrapper=wrapper)
        eng.output_root = tmp_path / "output"

        # 让 save_video_with_audio 抛错
        with patch(
            "engines.foleycrafter_a3b2c1.engine.save_video_with_audio",
            side_effect=RuntimeError("mock video synthesis failed"),
        ):
            req = InferRequest(
                variant="foleycrafter-v1",
                inputs={"prompt": "test"},
                audios=[AudioInput(path="/fake/video.mp4", role="video")],
                task_id="v2a-fail-video",
            )

            frames: list[str] = []
            async for frame in eng.infer(req):
                frames.append(frame)

        parsed = _parse_sse_frames(frames)
        last = parsed[-1]
        # 仍应成功完成（音频已保存），但无 video_url
        assert last["stage"] == "completed"
        assert "/output/audio/v2a-fail-video.wav" in last["result"]["audio_url"]
        # video_url 应为空字符串
        assert not last["result"].get("video_url")


class TestStatusImpl:
    """_status_impl 测试。"""

    @pytest.mark.asyncio
    async def test_status_when_idle(self, tmp_path: Path):
        wrapper = _make_mock_wrapper(is_loaded=False, vram_used=0)
        eng = _make_engine(tmp_path, wrapper=wrapper)

        r = await eng._status_impl()

        assert r.state == eng.state_machine.state.value
        assert r.loaded_variant == ""
        assert r.vram_used == "0MB"
        assert r.loaded_loras == []
        assert r.extra["is_loaded"] is False
        assert r.extra["device"] == "cpu"
        assert r.extra["sample_rate"] == 16000

    @pytest.mark.asyncio
    async def test_status_when_ready(self, tmp_path: Path):
        wrapper = _make_mock_wrapper(
            is_loaded=True, loaded_variant="foleycrafter-v1", vram_used=8192,
        )
        eng = _make_engine(tmp_path, wrapper=wrapper)

        r = await eng._status_impl()

        assert r.loaded_variant == "foleycrafter-v1"
        assert "8192MB" in r.vram_used
        assert r.extra["is_loaded"] is True


class TestLoraImpl:
    """_load_lora_impl / _unload_lora_impl 测试（v0.1 返回 501）。"""

    @pytest.mark.asyncio
    async def test_load_lora_returns_501(self, tmp_path: Path):
        eng = _make_engine(tmp_path)
        with pytest.raises(HTTPException) as exc_info:
            await eng._load_lora_impl("any", 1.0)
        assert exc_info.value.status_code == 501
        assert "不支持" in exc_info.value.detail

    @pytest.mark.asyncio
    async def test_unload_lora_returns_501(self, tmp_path: Path):
        eng = _make_engine(tmp_path)
        with pytest.raises(HTTPException) as exc_info:
            await eng._unload_lora_impl("any")
        assert exc_info.value.status_code == 501
        assert "不支持" in exc_info.value.detail
