"""ModelScanner 单元测试（T13 + v0.2.11+ 方案 B）。

覆盖：
- scan_variants：完整模型 / 部分模型 / 空目录 / 目录不存在（fallback 本地扫描）
- scan_variants_async：平台 API 返回 / 平台不可达 fallback / engine 未注入
- resolve_variant_path：已知 / 未知变体
- _check_model_files：5 个标志文件的检查逻辑
- check_variant_files：文件状态详情
- _merge_with_manifest：平台数据 + manifest 引擎特有字段合并
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from engines.foleycrafter_a3b2c1.model_scanner import ModelScanner, _KNOWN_VARIANTS
from engines.foleycrafter_a3b2c1.schemas import V2AVariantInfo


# ===== Fixtures =====


@pytest.fixture
def variant_dir(tmp_path: Path) -> Path:
    """创建一个完整的 foleycrafter-v1 变体目录（含 5 个标志文件）。"""
    v1_dir = tmp_path / "models" / "audio" / "v2a" / "foleycrafter" / "v1"
    v1_dir.mkdir(parents=True)

    # 创建 5 个标志文件
    (v1_dir / "temporal_adapter.ckpt").write_bytes(b"fake")
    (v1_dir / "semantic").mkdir()
    (v1_dir / "semantic" / "semantic_adapter.bin").write_bytes(b"fake")
    (v1_dir / "timestamp_detector.pth.tar").write_bytes(b"fake")
    (v1_dir / "auffusion").mkdir()
    (v1_dir / "auffusion" / "model_index.json").write_text("{}")
    (v1_dir / "ip-adapter").mkdir()
    (v1_dir / "ip-adapter" / "config.json").write_text("{}")

    return tmp_path / "models" / "audio" / "v2a" / "foleycrafter"


@pytest.fixture
def partial_variant_dir(tmp_path: Path) -> Path:
    """创建一个不完整的变体目录（缺 ip-adapter）。"""
    v1_dir = tmp_path / "models" / "audio" / "v2a" / "foleycrafter" / "v1"
    v1_dir.mkdir(parents=True)

    (v1_dir / "temporal_adapter.ckpt").write_bytes(b"fake")
    (v1_dir / "semantic").mkdir()
    (v1_dir / "semantic" / "semantic_adapter.bin").write_bytes(b"fake")
    (v1_dir / "timestamp_detector.pth.tar").write_bytes(b"fake")
    (v1_dir / "auffusion").mkdir()
    (v1_dir / "auffusion" / "model_index.json").write_text("{}")
    # 缺 ip-adapter/config.json

    return tmp_path / "models" / "audio" / "v2a" / "foleycrafter"


# ===== 测试类 =====


class TestScanVariants:
    """scan_variants 测试。"""

    def test_scan_finds_complete_variant(self, variant_dir: Path):
        """完整变体目录应被识别为可用。"""
        scanner = ModelScanner(variant_dir)
        variants = scanner.scan_variants()

        assert len(variants) == 1
        v = variants[0]
        assert isinstance(v, V2AVariantInfo)
        assert v.name == "foleycrafter-v1"
        assert v.alias == "FoleyCrafter-v1"
        assert v.base_model == "auffusion-full-no-adapter"
        assert v.precision == "fp16"
        assert v.loader == "foleycrafter"
        assert v.vram_required == "10GB"
        assert v.supports_v2a is True
        assert v.supports_t2a is True
        assert v.default_steps == 25
        assert v.default_temporal_scale == 0.2
        assert v.default_semantic_scale == 1.0

    def test_scan_skips_partial_variant(self, partial_variant_dir: Path):
        """不完整的变体目录应被跳过（不加入可用列表）。"""
        scanner = ModelScanner(partial_variant_dir)
        variants = scanner.scan_variants()
        assert len(variants) == 0    # 缺 ip-adapter，跳过

    def test_scan_returns_empty_when_dir_not_exists(self, tmp_path: Path):
        """目录不存在时应返回空列表。"""
        scanner = ModelScanner(tmp_path / "nonexistent")
        variants = scanner.scan_variants()
        assert variants == []

    def test_scan_returns_empty_when_no_variants(self, tmp_path: Path):
        """目录存在但无变体子目录时应返回空列表。"""
        (tmp_path / "models").mkdir()
        scanner = ModelScanner(tmp_path / "models")
        variants = scanner.scan_variants()
        assert variants == []


class TestResolveVariantPath:
    """resolve_variant_path 测试。"""

    def test_resolve_known_variant(self, variant_dir: Path):
        scanner = ModelScanner(variant_dir)
        path = scanner.resolve_variant_path("foleycrafter-v1")
        assert path is not None
        assert path.name == "v1"
        assert path.exists()

    def test_resolve_unknown_variant_returns_none(self, variant_dir: Path):
        scanner = ModelScanner(variant_dir)
        path = scanner.resolve_variant_path("unknown-variant")
        assert path is None


class TestCheckVariantFiles:
    """check_variant_files 测试。"""

    def test_check_complete_variant(self, variant_dir: Path):
        scanner = ModelScanner(variant_dir)
        result = scanner.check_variant_files("foleycrafter-v1")

        assert result["variant"] == "foleycrafter-v1"
        assert result["all_ready"] is True
        assert len(result["files"]) == 4
        for f in result["files"]:
            assert f["exists"] is True

    def test_check_partial_variant(self, partial_variant_dir: Path):
        scanner = ModelScanner(partial_variant_dir)
        result = scanner.check_variant_files("foleycrafter-v1")

        assert result["all_ready"] is False
        # ip-adapter/config.json 应该不存在
        ip_config = [f for f in result["files"] if "ip-adapter" in f["path"]]
        assert len(ip_config) == 1
        assert ip_config[0]["exists"] is False

    def test_check_unknown_variant(self, variant_dir: Path):
        scanner = ModelScanner(variant_dir)
        result = scanner.check_variant_files("unknown")

        assert result["variant"] == "unknown"
        assert result["all_ready"] is False
        assert result["files"] == []


class TestKnownVariants:
    """_KNOWN_VARIANTS 配置测试。"""

    def test_known_variants_contains_v1(self):
        """确认 _KNOWN_VARIANTS 包含 foleycrafter-v1 定义。"""
        names = [v["name"] for v in _KNOWN_VARIANTS]
        assert "foleycrafter-v1" in names

    def test_v1_has_4_required_files(self):
        """确认 v1 变体声明了 4 个必需文件（v0.2.14.3+ 不再检查大权重文件）。"""
        v1 = next(v for v in _KNOWN_VARIANTS if v["name"] == "foleycrafter-v1")
        assert len(v1["required_files"]) == 4
        assert "temporal_adapter.ckpt" in v1["required_files"]
        assert "timestamp_detector.pth.tar" in v1["required_files"]
        assert "auffusion/model_index.json" in v1["required_files"]
        assert "ip-adapter/config.json" in v1["required_files"]


# ===== v0.2.11+ 方案 B 测试 =====


class TestScanVariantsAsync:
    """scan_variants_async 测试（v0.2.11+ 方案 B）。

    覆盖三种场景：
        1. 平台 API 返回 variant 列表 → 合并 manifest 引擎特有字段
        2. 平台 API 不可达（get_variants 返回 None）→ fallback 到本地扫描
        3. engine 未注入（engine=None）→ 直接走 fallback 本地扫描
    """

    @pytest.mark.asyncio
    async def test_platform_api_returns_variants(self, variant_dir: Path):
        """场景 1：平台 API 返回 variant 列表，应合并 manifest 引擎特有字段。"""
        # 构造 mock engine，get_variants 返回平台 variant 数据
        mock_engine = MagicMock()
        mock_engine.get_variants = AsyncMock(return_value=[
            {
                "name": "foleycrafter-v1",
                "alias": "FoleyCrafter-v1",
                "base_model": "auffusion-full-no-adapter",
                "precision": "fp16",
                "loader": "foleycrafter",
                "vram_required": "10GB",
                "available": True,
                "model_file_ids": ["fc-temporal-adapter"],
                "missing_file_ids": [],
            },
        ])

        scanner = ModelScanner(variant_dir, engine=mock_engine)
        variants = await scanner.scan_variants_async()

        assert len(variants) == 1
        v = variants[0]
        assert isinstance(v, V2AVariantInfo)
        # 平台返回的通用字段
        assert v.name == "foleycrafter-v1"
        assert v.alias == "FoleyCrafter-v1"
        assert v.base_model == "auffusion-full-no-adapter"
        assert v.precision == "fp16"
        assert v.loader == "foleycrafter"
        assert v.vram_required == "10GB"
        assert v.available is True
        # 引擎特有字段（来自 _KNOWN_VARIANTS）
        assert v.default_steps == 25
        assert v.default_temporal_scale == 0.2
        assert v.default_semantic_scale == 1.0
        assert v.supports_v2a is True
        assert v.supports_t2a is True

    @pytest.mark.asyncio
    async def test_platform_api_returns_unavailable_variant(self, variant_dir: Path):
        """场景 1b：平台 API 返回 available=False 的 variant，应保留状态。"""
        mock_engine = MagicMock()
        mock_engine.get_variants = AsyncMock(return_value=[
            {
                "name": "foleycrafter-v1",
                "alias": "FoleyCrafter-v1",
                "base_model": "auffusion-full-no-adapter",
                "precision": "fp16",
                "loader": "foleycrafter",
                "vram_required": "10GB",
                "available": False,  # 文件未就绪
                "model_file_ids": ["fc-temporal-adapter"],
                "missing_file_ids": ["fc-temporal-adapter"],
            },
        ])

        scanner = ModelScanner(variant_dir, engine=mock_engine)
        variants = await scanner.scan_variants_async()

        assert len(variants) == 1
        assert variants[0].available is False
        # 引擎特有字段仍应被合并
        assert variants[0].default_steps == 25

    @pytest.mark.asyncio
    async def test_platform_api_unreachable_falls_back_to_local(
        self, variant_dir: Path,
    ):
        """场景 2：平台 API 不可达（返回 None），应 fallback 到本地扫描。"""
        mock_engine = MagicMock()
        mock_engine.get_variants = AsyncMock(return_value=None)

        scanner = ModelScanner(variant_dir, engine=mock_engine)
        variants = await scanner.scan_variants_async()

        # fallback 到本地扫描，应识别出完整 variant
        assert len(variants) == 1
        assert variants[0].name == "foleycrafter-v1"
        assert variants[0].available is True  # 本地扫描发现文件齐全

    @pytest.mark.asyncio
    async def test_no_engine_uses_local_scan(self, variant_dir: Path):
        """场景 3：engine 未注入（None），应直接走 fallback 本地扫描。"""
        scanner = ModelScanner(variant_dir, engine=None)
        variants = await scanner.scan_variants_async()

        assert len(variants) == 1
        assert variants[0].name == "foleycrafter-v1"
        assert variants[0].available is True

    @pytest.mark.asyncio
    async def test_platform_api_with_partial_local_files(self, partial_variant_dir: Path):
        """场景 4：平台 API 可用但本地文件不全，应以平台 available 为准。"""
        mock_engine = MagicMock()
        # 平台 API 报告 available=False（因为缺 ip-adapter）
        mock_engine.get_variants = AsyncMock(return_value=[
            {
                "name": "foleycrafter-v1",
                "alias": "FoleyCrafter-v1",
                "base_model": "auffusion-full-no-adapter",
                "precision": "fp16",
                "loader": "foleycrafter",
                "vram_required": "10GB",
                "available": False,
                "model_file_ids": ["fc-temporal-adapter"],
                "missing_file_ids": ["fc-ip-adapter"],
            },
        ])

        scanner = ModelScanner(partial_variant_dir, engine=mock_engine)
        variants = await scanner.scan_variants_async()

        # 平台返回 1 个 variant（available=False），不应被本地扫描覆盖
        assert len(variants) == 1
        assert variants[0].available is False


class TestMergeWithManifest:
    """_merge_with_manifest 测试。"""

    def test_merges_platform_data_with_known_variants(self, variant_dir: Path):
        """应正确合并平台通用字段 + _KNOWN_VARIANTS 引擎特有字段。"""
        scanner = ModelScanner(variant_dir)
        platform_variants = [
            {
                "name": "foleycrafter-v1",
                "alias": "FoleyCrafter-v1",
                "base_model": "auffusion-full-no-adapter",
                "precision": "fp16",
                "loader": "foleycrafter",
                "vram_required": "10GB",
                "available": True,
            },
        ]

        result = scanner._merge_with_manifest(platform_variants)

        assert len(result) == 1
        v = result[0]
        assert v.name == "foleycrafter-v1"
        assert v.available is True
        # 引擎特有字段来自 _KNOWN_VARIANTS
        assert v.default_steps == 25
        assert v.default_temporal_scale == 0.2
        assert v.default_semantic_scale == 1.0

    def test_unknown_variant_uses_defaults(self, variant_dir: Path):
        """平台返回未知 variant 时，应使用默认值。"""
        scanner = ModelScanner(variant_dir)
        platform_variants = [
            {
                "name": "foleycrafter-v2-unknown",
                "alias": "V2",
                "available": True,
            },
        ]

        result = scanner._merge_with_manifest(platform_variants)

        assert len(result) == 1
        v = result[0]
        assert v.name == "foleycrafter-v2-unknown"
        # 未在 _KNOWN_VARIANTS 中找到 → 使用默认值
        assert v.default_steps == 25  # 默认值
        assert v.default_temporal_scale == 0.2  # 默认值

    def test_empty_platform_list_returns_empty(self, variant_dir: Path):
        """平台返回空列表时，应返回空列表。"""
        scanner = ModelScanner(variant_dir)
        result = scanner._merge_with_manifest([])
        assert result == []

