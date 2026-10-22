"""FoleyCrafter 模型文件扫描器。

扫描 models/audio/v2a/foleycrafter/ 目录，识别可用的 FoleyCrafter 模型变体。

v0.2.11+ 方案 B 改造：
    优先调用平台 API（`GET /engines/{id}/variants`）获取 variant 可用性，
    平台不可达时 fallback 到本地文件扫描。

    声明式检测（方案 B）：
        1. 调用 BaseEngine.get_variants() 获取平台 variant 列表（含 available 字段）
        2. 从 manifest.models.variants[] 读取 V2A 特有字段（default_steps 等）
        3. 合并：平台判定 available + 引擎补充特有字段

    Fallback 检测（保留兼容性，测试环境无平台时使用）：
        1. 遍历 _KNOWN_VARIANTS 中定义的已知变体
        2. 检查 model_dir/<subdir>/ 目录是否存在
        3. 检查目录内是否有 4 个子模型 + IP-Adapter encoder
        4. 全部存在则加入可用列表
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

from .schemas import V2AVariantInfo

logger = logging.getLogger(__name__)


# ===== 已知模型变体定义（fallback 用）=====

_KNOWN_VARIANTS: list[dict] = [
    {
        "name": "foleycrafter-v1",
        "alias": "FoleyCrafter-v1",
        "precision": "fp16",
        "vram_required": 10240,  # 10GB
        "subdir": "v1",
        # v0.2.14.3+ 调整 required_files 策略：
        # 本地 fallback 只检查标志文件是否存在（用于判断 variant 目录是否创建过），
        # 不检查大权重文件（unet/vae/text_encoder/model.safetensors 等）。
        # 原因：本地 fallback 路径不检查 key_files，若要求所有大权重齐全，
        # 会导致用户下载了部分文件但还没下完时，variant 列表为空，无法触发下载入口。
        # 实际文件完整性检查由 foleycrafter_wrapper.load() 的 _check_files_integrity 兜底。
        # 平台 API 路径（check_variants）会通过 key_files 机制检查完整权重。
        "required_files": [
            "temporal_adapter.ckpt",
            "timestamp_detector.pth.tar",
            "auffusion/model_index.json",
            "ip-adapter/config.json",
        ],
        "default_steps": 25,
        "default_temporal_scale": 0.2,
        "default_semantic_scale": 1.0,
    },
]


class ModelScanner:
    """模型目录扫描器。

    扫描 model_dir 下的子目录，识别已下载的 FoleyCrafter 模型变体。

    v0.2.11+ 方案 B：优先调用平台 API，fallback 到本地扫描。
    """

    def __init__(self, model_dir: Path, engine=None):
        """初始化扫描器。

        Args:
            model_dir: 模型根目录（如 models/audio/v2a/foleycrafter）
            engine: 可选的 BaseEngine 实例（v0.2.11+ 方案 B 注入）。
                传入时优先调用平台 API 检测 variant；
                为 None 时走 fallback 本地扫描逻辑（兼容旧测试）。
        """
        self.model_dir = model_dir
        self.engine = engine

    async def scan_variants_async(self) -> list[V2AVariantInfo]:
        """v0.2.11+ 方案 B：异步扫描，优先调用平台 API。

        调用链：
            1. engine.get_variants() → 平台 API（声明式检测）
            2. 平台返回 None → fallback 到 _scan_locally()
            3. 合并平台 available 字段 + manifest 引擎特有字段
        """
        if self.engine is not None:
            platform_variants = await self.engine.get_variants()
            if platform_variants is not None:
                return self._merge_with_manifest(platform_variants)

        # fallback：本地扫描
        return self._scan_locally()

    def scan_variants(self) -> list[V2AVariantInfo]:
        """同步扫描接口（向后兼容）。

        v0.2.11+ 注：若需使用方案 B 平台 API，请调用 scan_variants_async()。
        此同步方法走 fallback 本地扫描逻辑。
        """
        return self._scan_locally()

    def _merge_with_manifest(self, platform_variants: list[dict]) -> list[V2AVariantInfo]:
        """合并平台 variant 数据 + manifest 引擎特有字段。

        Args:
            platform_variants: 平台 API 返回的 variant 列表（含 available 字段）

        Returns:
            V2AVariantInfo 列表（available 来自平台，特有字段来自 manifest/_KNOWN_VARIANTS）
        """
        # 从 _KNOWN_VARIANTS 构建 name → 特有字段映射
        # （manifest 已声明这些字段，但引擎进程内 _KNOWN_VARIANTS 是同一份数据）
        manifest_lookup: dict[str, dict] = {
            v["name"]: v for v in _KNOWN_VARIANTS
        }

        result: list[V2AVariantInfo] = []
        for pv in platform_variants:
            v_name = pv.get("name", "")
            mv = manifest_lookup.get(v_name, {})
            vram_mb = mv.get("vram_required", 10240)
            result.append(V2AVariantInfo(
                name=v_name,
                alias=pv.get("alias", mv.get("alias", "")),
                base_model=pv.get("base_model", "auffusion-full-no-adapter"),
                precision=pv.get("precision", mv.get("precision", "fp16")),
                loader=pv.get("loader", "foleycrafter"),
                vram_required=pv.get("vram_required", f"{vram_mb // 1024}GB"),
                available=pv.get("available", False),
                default_steps=mv.get("default_steps", 25),
                default_temporal_scale=mv.get("default_temporal_scale", 0.2),
                default_semantic_scale=mv.get("default_semantic_scale", 1.0),
                supports_v2a=True,
                supports_t2a=True,
            ))

        if not any(r.available for r in result):
            logger.warning(
                "未在 %s 中找到完整的 FoleyCrafter 模型，"
                "请下载模型到对应目录（如 models/audio/v2a/foleycrafter/v1/）。"
                "需要 3 个 HF 仓库的文件：ymzhang319/FoleyCrafter + "
                "auffusion/auffusion-full-no-adapter + h94/IP-Adapter",
                self.model_dir,
            )

        return result

    def _scan_locally(self) -> list[V2AVariantInfo]:
        """fallback 本地扫描（原 scan_variants 逻辑）。"""
        capabilities: list[V2AVariantInfo] = []

        if not self.model_dir.exists():
            logger.warning("模型目录不存在: %s", self.model_dir)
            return capabilities

        for variant_def in _KNOWN_VARIANTS:
            variant_dir = self.model_dir / variant_def["subdir"]

            if not variant_dir.exists():
                logger.debug("变体 %s 目录不存在: %s", variant_def["name"], variant_dir)
                continue

            # 检查 4 子模型 + IP-Adapter encoder 是否齐全
            has_model = self._check_model_files(variant_dir, variant_def["required_files"])
            if not has_model:
                logger.debug(
                    "变体 %s 子模型不齐全: %s（需要 %s）",
                    variant_def["name"], variant_dir, variant_def["required_files"],
                )
                continue

            cap = V2AVariantInfo(
                name=variant_def["name"],
                alias=variant_def["alias"],
                base_model="auffusion-full-no-adapter",
                precision=variant_def["precision"],
                loader="foleycrafter",
                vram_required=f"{variant_def['vram_required'] // 1024}GB",
                available=True,
                default_steps=variant_def["default_steps"],
                default_temporal_scale=variant_def["default_temporal_scale"],
                default_semantic_scale=variant_def["default_semantic_scale"],
                supports_v2a=True,
                supports_t2a=True,
            )
            capabilities.append(cap)
            logger.info("发现模型变体: %s (%s)", cap.name, cap.alias)

        if not capabilities:
            logger.warning(
                "未在 %s 中找到完整的 FoleyCrafter 模型，"
                "请下载模型到对应目录（如 models/audio/v2a/foleycrafter/v1/）。"
                "需要 3 个 HF 仓库的文件：ymzhang319/FoleyCrafter + "
                "auffusion/auffusion-full-no-adapter + h94/IP-Adapter",
                self.model_dir,
            )

        return capabilities

    def _check_model_files(
        self,
        variant_dir: Path,
        required_files: list[str],
    ) -> bool:
        """检查目录内是否有全部必需的子模型文件。

        FoleyCrafter 模型包含 4 个子模型 + IP-Adapter encoder，分散在 3 个 HF 仓库：
            1. temporal_adapter.ckpt（ymzhang319/FoleyCrafter）
            2. semantic/semantic_adapter.bin（ymzhang319/FoleyCrafter）
            3. timestamp_detector.pth.tar（ymzhang319/FoleyCrafter）
            4. auffusion/model_index.json（auffusion/auffusion-full-no-adapter，标志文件）
            5. ip-adapter/config.json（h94/IP-Adapter，标志文件）
        """
        for rel_path in required_files:
            if not (variant_dir / rel_path).exists():
                logger.debug("缺失文件: %s/%s", variant_dir, rel_path)
                return False
        return True

    def resolve_variant_path(self, variant_name: str) -> Optional[Path]:
        """根据变体名解析模型路径。

        Args:
            variant_name: 变体名（如 foleycrafter-v1）

        Returns:
            模型目录路径，不存在返回 None
        """
        for variant_def in _KNOWN_VARIANTS:
            if variant_def["name"] == variant_name:
                path = self.model_dir / variant_def["subdir"]
                return path if path.exists() else None
        return None

    def check_variant_files(self, variant_name: str) -> dict:
        """检查变体的所有子模型文件状态（用于下载指引）。

        Args:
            variant_name: 变体名

        Returns:
            dict: {
                "variant": str,
                "variant_dir": str,
                "files": [{"path": str, "exists": bool}, ...],
                "all_ready": bool,
            }
        """
        for variant_def in _KNOWN_VARIANTS:
            if variant_def["name"] != variant_name:
                continue

            variant_dir = self.model_dir / variant_def["subdir"]
            files_status = []
            all_ready = True

            for rel_path in variant_def["required_files"]:
                full_path = variant_dir / rel_path
                exists = full_path.exists()
                if not exists:
                    all_ready = False
                files_status.append({
                    "path": rel_path,
                    "full_path": str(full_path),
                    "exists": exists,
                })

            return {
                "variant": variant_name,
                "variant_dir": str(variant_dir),
                "files": files_status,
                "all_ready": all_ready,
            }

        return {
            "variant": variant_name,
            "variant_dir": "",
            "files": [],
            "all_ready": False,
        }
