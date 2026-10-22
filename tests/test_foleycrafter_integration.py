"""FoleyCrafter V2A 引擎 HTTP 端点集成测试（T14）。

通过 FastAPI TestClient 测试 8 个标准端点的完整请求/响应周期，
mock FoleyCrafterWrapper 和 ModelScanner 避免真实模型加载。

覆盖：
- GET  /health         健康检查
- GET  /capabilities   返回变体列表
- POST /load           加载 variant
- POST /unload         卸载模型
- POST /infer          V2A / T2A 推理（SSE 流式）
- GET  /status         状态查询
- POST /lora/load      返回 501
- POST /lora/unload    返回 501
- GET  /output/*       静态文件下载（推理产物）

参考：engines/e_c4a8f2e7b931/tests/test_engine.py（单元测试）
差异：集成测试通过 HTTP 调用，验证端到端行为
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from fastapi.testclient import TestClient

from aireliable_engine.schemas import AudioInput

from engines.foleycrafter_a3b2c1.config import EngineConfig
from engines.foleycrafter_a3b2c1.schemas import V2AVariantInfo


# ===== Mock 工厂 =====


def _make_variant(name: str = "foleycrafter-v1") -> V2AVariantInfo:
    return V2AVariantInfo(
        name=name,
        alias="FoleyCrafter-v1",
        base_model="auffusion-full-no-adapter",
        precision="fp16",
        loader="foleycrafter",
        vram_required="10GB",
        available=True,
        default_steps=25,
        default_temporal_scale=0.2,
        default_semantic_scale=1.0,
        supports_v2a=True,
        supports_t2a=True,
    )


def _make_mock_wrapper(
    is_loaded: bool = False,
    loaded_variant: Optional[str] = None,
    sample_rate: int = 16000,
    vram_used: int = 0,
    generate_samples: Optional[np.ndarray] = None,
) -> MagicMock:
    wrapper = MagicMock()
    wrapper.is_loaded = is_loaded
    # 默认空字符串（与真实 wrapper 一致）
    wrapper.loaded_variant = loaded_variant if loaded_variant is not None else ""
    wrapper.device = "cpu"
    wrapper.device_name = "mock-cpu"
    wrapper.sample_rate = sample_rate
    wrapper.get_vram_used.return_value = vram_used
    wrapper.load.return_value = None
    wrapper.unload.return_value = None
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
    return wrapper


def _make_mock_scanner(variant_path: Optional[Path] = None) -> MagicMock:
    scanner = MagicMock()
    scanner.scan_variants.return_value = [_make_variant()]
    scanner.resolve_variant_path.return_value = variant_path or Path("/fake/v1")
    return scanner


def _fake_save_audio(samples, sr, path, fmt="wav"):
    """模拟 save_audio：直接写空文件。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"FAKE_WAV")
    return path


def _fake_save_video_with_audio(video_path, audio_path, output_path, duration=None):
    """模拟 save_video_with_audio。"""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_bytes(b"FAKE_MP4")
    return output_path


def _fake_get_duration(samples, sr):
    return 1.0


@pytest.fixture
def app_with_mock(tmp_path: Path):
    """构造一个带 mock 依赖的 FastAPI app + TestClient。"""
    wrapper = _make_mock_wrapper()
    scanner = _make_mock_scanner(variant_path=tmp_path / "fake-v1")

    config = EngineConfig(
        models_root=tmp_path / "models",
        data_root=tmp_path / "data",
        project_root=tmp_path,
        device="cpu",
    )

    # mock audio_utils 避免依赖 soundfile / moviepy
    with patch(
        "engines.foleycrafter_a3b2c1.engine.save_audio",
        side_effect=_fake_save_audio,
    ), patch(
        "engines.foleycrafter_a3b2c1.engine.get_audio_duration",
        side_effect=_fake_get_duration,
    ), patch(
        "engines.foleycrafter_a3b2c1.engine.save_video_with_audio",
        side_effect=_fake_save_video_with_audio,
    ), patch(
        "engines.foleycrafter_a3b2c1.engine.FoleyCrafterWrapper",
        return_value=wrapper,
    ), patch(
        "engines.foleycrafter_a3b2c1.engine.ModelScanner",
        return_value=scanner,
    ):
        from engines.foleycrafter_a3b2c1.server import create_app
        app = create_app(config)
        # 把 wrapper / scanner 挂到 engine 上（通过 lifespan 后）
        with TestClient(app) as client:
            # lifespan 启动后 _engine 已初始化
            from engines.foleycrafter_a3b2c1 import server as server_module
            engine = server_module._engine
            engine.wrapper = wrapper
            engine.scanner = scanner
            yield client, wrapper, scanner, engine


# ===== 测试类 =====


class TestHealthEndpoint:
    """GET /health 测试。"""

    def test_health_returns_200(self, app_with_mock):
        client, _, _, _ = app_with_mock
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["engine_id"] == "foleycrafter_a3b2c1"
        assert data["state"] in ("idle", "ready", "loading", "inferring", "error")


class TestCapabilitiesEndpoint:
    """GET /capabilities 测试。"""

    def test_capabilities_returns_variants(self, app_with_mock):
        client, _, _, _ = app_with_mock
        response = client.get("/capabilities")
        assert response.status_code == 200
        data = response.json()

        assert data["engine_id"] == "foleycrafter_a3b2c1"
        assert data["capability"] == "video-to-audio"
        assert data["type"] == "audio"      # 新增类型
        assert len(data["variants"]) == 1
        v = data["variants"][0]
        assert v["name"] == "foleycrafter-v1"
        assert v["supports_v2a"] is True
        assert v["supports_t2a"] is True
        assert v["default_steps"] == 25


class TestLoadEndpoint:
    """POST /load 测试。"""

    def test_load_success(self, app_with_mock):
        client, wrapper, _, _ = app_with_mock
        response = client.post("/load", json={"variant": "foleycrafter-v1"})
        assert response.status_code == 200
        data = response.json()
        assert data["variant"] == "foleycrafter-v1"
        assert data["state"] == "ready"
        wrapper.load.assert_called_once()

    def test_load_unknown_variant_returns_404(self, app_with_mock):
        client, _, scanner, _ = app_with_mock
        scanner.resolve_variant_path.return_value = None
        response = client.post("/load", json={"variant": "unknown"})
        assert response.status_code == 404
        data = response.json()
        assert "variant 不存在" in data["error"]["message"]


class TestUnloadEndpoint:
    """POST /unload 测试。"""

    def test_unload_when_not_loaded(self, app_with_mock):
        client, wrapper, _, _ = app_with_mock
        wrapper.is_loaded = False
        response = client.post("/unload")
        assert response.status_code == 200
        data = response.json()
        assert data["state"] == "idle"


class TestStatusEndpoint:
    """GET /status 测试。"""

    def test_status_returns_engine_info(self, app_with_mock):
        client, wrapper, _, _ = app_with_mock
        wrapper.is_loaded = False
        wrapper.get_vram_used.return_value = 0
        response = client.get("/status")
        assert response.status_code == 200
        data = response.json()
        assert "state" in data
        assert data["vram_used"] == "0MB"
        assert "device" in data["extra"]
        assert "sample_rate" in data["extra"]


class TestInferEndpoint:
    """POST /infer 测试（SSE 流式）。"""

    def test_infer_t2a_returns_sse_stream(self, app_with_mock):
        """T2A 模式：仅 prompt，应返回完整 SSE 流并最终含 audio_url。"""
        client, wrapper, _, _ = app_with_mock
        wrapper.is_loaded = True
        wrapper.loaded_variant = "foleycrafter-v1"

        response = client.post(
            "/infer",
            json={
                "variant": "foleycrafter-v1",
                "inputs": {"prompt": "一声枪响"},
                "task_id": "t2a-integration-1",
            },
        )
        assert response.status_code == 200

        # 解析 SSE 流
        frames = []
        for line in response.iter_lines():
            if line.startswith("data: "):
                frames.append(json.loads(line[6:]))

        stages = [f.get("stage") for f in frames]
        assert "v2a.start" in stages
        assert "completed" in stages

        # T2A 模式不应有视频处理阶段
        assert "v2a.video_decoding" not in stages

        # 最后一帧含 audio_url
        last = frames[-1]
        assert last["stage"] == "completed"
        assert "/output/audio/t2a-integration-1.wav" in last["result"]["audio_url"]
        assert last["result"]["mode"] == "t2a"
        # T2A 模式无 video_url
        assert not last["result"].get("video_url")

    def test_infer_v2a_returns_audio_and_video(self, app_with_mock):
        """V2A 模式：视频输入，应返回 audio_url + video_url。"""
        client, wrapper, _, _ = app_with_mock
        wrapper.is_loaded = True
        wrapper.loaded_variant = "foleycrafter-v1"

        response = client.post(
            "/infer",
            json={
                "variant": "foleycrafter-v1",
                "inputs": {"prompt": "汽车引擎声"},
                "audios": [{"path": "/fake/video.mp4", "role": "video"}],
                "task_id": "v2a-integration-1",
            },
        )
        assert response.status_code == 200

        # 解析 SSE 流
        frames = []
        for line in response.iter_lines():
            if line.startswith("data: "):
                frames.append(json.loads(line[6:]))

        stages = [f.get("stage") for f in frames]
        # V2A 模式应有全部 7 阶段
        assert "v2a.video_decoding" in stages
        assert "v2a.timestamp_detect" in stages
        assert "v2a.image_encoding" in stages
        assert "v2a.audio_generating" in stages
        assert "completed" in stages

        last = frames[-1]
        assert last["stage"] == "completed"
        assert "/output/audio/v2a-integration-1.wav" in last["result"]["audio_url"]
        assert "/output/video/v2a-integration-1.mp4" in last["result"]["video_url"]
        assert last["result"]["mode"] == "v2a"

    def test_infer_invalid_inputs_returns_error_frame(self, app_with_mock):
        """输入参数校验失败时通过 SSE error 帧返回。"""
        client, wrapper, _, _ = app_with_mock
        wrapper.is_loaded = True

        response = client.post(
            "/infer",
            json={
                "variant": "foleycrafter-v1",
                "inputs": {"prompt": "test", "max_video_duration": 0},  # 校验失败
            },
        )
        assert response.status_code == 200  # SSE 流本身返回 200

        frames = []
        for line in response.iter_lines():
            if line.startswith("data: "):
                frames.append(json.loads(line[6:]))

        error_frames = [f for f in frames if f.get("stage") == "error"]
        assert len(error_frames) == 1
        assert error_frames[0]["error"]["type"] == "invalid_inputs"


class TestLoraEndpoints:
    """POST /lora/load + /lora/unload 测试（v0.1 返回 501）。"""

    def test_lora_load_returns_501(self, app_with_mock):
        client, _, _, _ = app_with_mock
        response = client.post("/lora/load", json={"name": "any", "weight": 1.0})
        assert response.status_code == 501
        data = response.json()
        assert "不支持" in data["error"]["message"]

    def test_lora_unload_returns_501(self, app_with_mock):
        client, _, _, _ = app_with_mock
        response = client.post("/lora/unload", json={"name": "any"})
        assert response.status_code == 501
        data = response.json()
        assert "不支持" in data["error"]["message"]


class TestOutputStaticFiles:
    """GET /output/* 静态文件下载测试。"""

    def test_output_audio_downloadable_after_infer(self, app_with_mock, tmp_path: Path):
        """推理后音频文件应可通过 /output/audio/ 下载。"""
        client, wrapper, _, engine = app_with_mock
        wrapper.is_loaded = True
        wrapper.loaded_variant = "foleycrafter-v1"

        # 执行推理
        client.post(
            "/infer",
            json={
                "variant": "foleycrafter-v1",
                "inputs": {"prompt": "test"},
                "task_id": "download-test",
            },
        )

        # 通过 /output/audio/ 下载
        response = client.get("/output/audio/download-test.wav")
        assert response.status_code == 200
        assert response.content == b"FAKE_WAV"


class TestEngineIdAndType:
    """验证 engine_id / engine_type 与决策一致。"""

    def test_engine_id_uses_underscore(self, app_with_mock):
        """engine_id 字符串使用下划线（v0.2.9+ 三同名契约）。"""
        client, _, _, engine = app_with_mock
        response = client.get("/capabilities")
        data = response.json()
        assert data["engine_id"] == "foleycrafter_a3b2c1"   # 下划线

    def test_engine_type_is_audio(self, app_with_mock):
        """engine_type 应为新增的 audio 类型（决策 1）。"""
        client, _, _, _ = app_with_mock
        response = client.get("/capabilities")
        data = response.json()
        assert data["type"] == "audio"   # 新增类型，区别于 voice
