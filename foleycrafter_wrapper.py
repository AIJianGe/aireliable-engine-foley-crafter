"""FoleyCrafter 模型包装器。

封装 FoleyCrafter 原生推理逻辑（基于 inference.py），提供：
    - load(): 加载 4 个子模型 + IP-Adapter encoder
    - generate(): V2A / T2A 推理（同步阻塞，由 engine._infer_impl 用 asyncio.to_thread 包装）
    - unload(): 释放显存

模型组件：
    1. Auffusion 基座（diffusers pipeline）
    2. temporal_adapter（ControlNet 时间对齐适配器）
    3. semantic_adapter（IP-Adapter 语义适配器）
    4. timestamp_detector（视频时间戳检测 CNN）
    5. IP-Adapter image_encoder（CLIP 视觉编码器，semantic_adapter 隐式依赖）

推理流程（V2A 模式）：
    1. moviepy 读视频帧（限制 max_frame_nums = int(fps * max_duration)）
    2. timestamp_detector 推理（提取时间戳条件）
    3. CLIP image_encoder 编码视频帧（生成 image_embeddings）
    4. pipe() 扩散推理（25 步 DDIM，生成频谱图）
    5. vocoder.inference() 解码频谱图 → 音频波形
    6. 截取音频到视频时长

foleycrafter Python 包安装（v0.2.12+）：
    FoleyCrafter 的 Python 代码位于 GitHub 仓库 open-mmlab/FoleyCrafter 的 foleycrafter/ 目录，
    但该仓库没有标准的 setup.py / pyproject.toml [project] 块，无法用 pip install 安装。
    load() 方法开头会调用 _ensure_foleycrafter_package() 自动检测并安装：
        1. git clone --depth 1 到临时目录
        2. 复制 foleycrafter/ 目录到 .venv/Lib/site-packages/
        3. 幂等：已安装时直接返回
    国内用户可设置环境变量 FOLEYCRAFTER_GIT_MIRROR 使用镜像（如 https://ghproxy.com）。
"""

from __future__ import annotations

import logging
import os
import os.path as osp
from pathlib import Path
from typing import Optional

import numpy as np

from .config import EngineConfig

logger = logging.getLogger(__name__)

# v0.3.5+ 设备策略合法取值（平台注入 AIRELIABLE_DEVICE_POLICY 环境变量）
#   auto    ：引擎自行决定（现状行为：不做显存预检，OOM 时由 PyTorch 抛原始错误）
#   gpu     ：强制全 GPU，显存不足直接报错（友好中文提示）
#   gpu+cpu ：GPU 优先，显存不足时警告 + 降级 CPU 加载
#   cpu     ：强制 CPU（跳过显存预检）
DEVICE_POLICIES = ("auto", "gpu", "gpu+cpu", "cpu")

#: 全组件预估所需显存（GB）：auffusion pipeline ~7GB + temporal/semantic
#: adapter + vocoder + CLIP image_encoder ~2GB
_REQUIRED_VRAM_GB = 9.0


def _read_device_policy() -> str:
    """读取设备策略（v0.3.5+，平台注入 AIRELIABLE_DEVICE_POLICY）。"""
    policy = (os.environ.get("AIRELIABLE_DEVICE_POLICY", "auto").strip() or "auto")
    if policy not in DEVICE_POLICIES:
        logger.warning("非法设备策略 %r（合法值：%s），回退 auto", policy, DEVICE_POLICIES)
        return "auto"
    return policy


def _setup_hf_mirror() -> None:
    """设置 HuggingFace 镜像端点，避免直连 huggingface.co 超时。

    v0.2.14.4+ 修复关键 bug：
        之前只设置 os.environ["HF_ENDPOINT"]，但 huggingface_hub.constants.ENDPOINT
        在 import 时（被 transformers/diffusers 提前 import）就已缓存为
        "https://huggingface.co"，后续修改环境变量不会改变 constants.ENDPOINT。
        snapshot_download 内部用 constants.ENDPOINT 而非 os.environ，
        导致镜像设置无效，仍直连 huggingface.co 超时 → 静默返回残缺文件 → 500。

    修复：除设置环境变量外，显式更新 huggingface_hub.constants.ENDPOINT。

    优先级：
        1. 用户已设置的 HF_ENDPOINT 环境变量（不覆盖）
        2. 平台注入的 AIRELIABLE_HF_MIRROR 环境变量（由"重新下载"对话框选择镜像源时设置）
        3. 默认使用 https://hf-mirror.com
    """
    # 1. 设置环境变量（如果未设置）
    if not os.environ.get("HF_ENDPOINT"):
        mirror = os.environ.get("AIRELIABLE_HF_MIRROR", "https://hf-mirror.com").strip()
        if mirror:
            os.environ["HF_ENDPOINT"] = mirror
            logger.info("已设置 HF_ENDPOINT 环境变量: %s", mirror)

    # 2. 关键修复：显式更新 huggingface_hub 的 endpoint 缓存
    #    huggingface_hub 在 import 时缓存多个 endpoint 相关常量，后续修改 os.environ 不会同步：
    #    a) constants.ENDPOINT：endpoint 字符串
    #    b) constants.HUGGINGFACE_CO_URL_TEMPLATE：URL 模板（= ENDPOINT + "/{repo_id}/..."）
    #    c) file_download.HUGGINGFACE_CO_URL_TEMPLATE：file_download 模块的本地引用
    #       （from .constants import HUGGINGFACE_CO_CO_URL_TEMPLATE 创建的本地引用）
    #    只更新 (a) 不够，hf_hub_url() 实际用的是 (b) 和 (c)，必须全部更新，
    #    否则 snapshot_download 构造的 URL 仍是 https://huggingface.co/... → 超时 → 500。
    try:
        import huggingface_hub.constants as hf_constants
        import huggingface_hub.file_download as hf_file_download
        desired = os.environ.get("HF_ENDPOINT", "https://huggingface.co")

        if hf_constants.ENDPOINT != desired:
            logger.warning(
                "huggingface_hub.constants.ENDPOINT 过期，更新: %s -> %s",
                hf_constants.ENDPOINT, desired,
            )
            hf_constants.ENDPOINT = desired
        else:
            logger.info("huggingface_hub 当前 endpoint: %s", hf_constants.ENDPOINT)

        new_template = desired + "/{repo_id}/resolve/{revision}/{filename}"
        # (b) 更新 constants 模块的 HUGGINGFACE_CO_URL_TEMPLATE
        if hf_constants.HUGGINGFACE_CO_URL_TEMPLATE != new_template:
            logger.warning(
                "constants.HUGGINGFACE_CO_URL_TEMPLATE 过期，更新: %s -> %s",
                hf_constants.HUGGINGFACE_CO_URL_TEMPLATE, new_template,
            )
            hf_constants.HUGGINGFACE_CO_URL_TEMPLATE = new_template
        # (c) 更新 file_download 模块的本地引用 HUGGINGFACE_CO_URL_TEMPLATE
        #     （from .constants import 创建的本地引用，不会随 constants 模块属性更新而同步）
        if hf_file_download.HUGGINGFACE_CO_URL_TEMPLATE != new_template:
            logger.warning(
                "file_download.HUGGINGFACE_CO_URL_TEMPLATE 过期，更新: %s -> %s",
                hf_file_download.HUGGINGFACE_CO_URL_TEMPLATE, new_template,
            )
            hf_file_download.HUGGINGFACE_CO_URL_TEMPLATE = new_template
    except ImportError as e:
        logger.warning("无法更新 huggingface_hub.constants（import 失败）: %s", e)


def _ensure_foleycrafter_package() -> None:
    """确保 foleycrafter Python 包已安装到引擎 .venv（v0.2.12+）。

    FoleyCrafter 的 Python 代码位于 GitHub 仓库 open-mmlab/FoleyCrafter 的 foleycrafter/ 目录，
    但该仓库没有标准的 setup.py / pyproject.toml [project] 块，无法用 pip install 安装。
    本函数通过 git clone 克隆仓库到临时目录，然后复制 foleycrafter/ 到 site-packages。

    幂等性：已安装时直接返回，不重复下载。

    环境变量：
        FOLEYCRAFTER_GIT_MIRROR: GitHub 镜像地址（如 https://ghproxy.com），
                                 用于国内加速。设置后克隆 URL 变为
                                 {mirror}/open-mmlab/FoleyCrafter.git

    Raises:
        RuntimeError: git clone 失败 / 仓库结构异常 / 复制后仍无法导入
    """
    import importlib.util
    import shutil
    import subprocess
    import sys
    import tempfile

    # 1. 检查 foleycrafter 包是否已安装
    if importlib.util.find_spec("foleycrafter") is not None:
        return

    logger.info("foleycrafter Python 包未安装，开始从 GitHub 克隆并安装到 .venv...")

    # 2. 定位 site-packages 目录
    # Windows: .venv/Lib/site-packages
    # Linux:   .venv/lib/python{version}/site-packages
    if sys.platform == "win32":
        site_packages = Path(sys.prefix) / "Lib" / "site-packages"
    else:
        lib_dir = Path(sys.prefix) / "lib"
        if not lib_dir.exists():
            raise RuntimeError(f"无法定位 lib 目录: {lib_dir}")
        python_dirs = [d for d in lib_dir.iterdir() if d.name.startswith("python")]
        if not python_dirs:
            raise RuntimeError(f"未找到 python* 目录: {lib_dir}")
        site_packages = python_dirs[0] / "site-packages"

    if not site_packages.exists():
        raise RuntimeError(f"site-packages 目录不存在: {site_packages}")

    # 3. 构造 git clone URL（支持镜像）
    repo_url = "https://github.com/open-mmlab/FoleyCrafter.git"
    mirror = os.environ.get("FOLEYCRAFTER_GIT_MIRROR", "").strip()
    if mirror:
        repo_url = f"{mirror}/open-mmlab/FoleyCrafter.git"
        logger.info("使用 GitHub 镜像: %s", mirror)

    # 4. git clone --depth 1 到临时目录
    with tempfile.TemporaryDirectory(prefix="foleycrafter_clone_") as tmp_dir:
        tmp_repo = Path(tmp_dir) / "FoleyCrafter"
        cmd = ["git", "clone", "--depth", "1", repo_url, str(tmp_repo)]
        logger.info("执行: %s", " ".join(cmd))

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=180,
            )
        except FileNotFoundError:
            raise RuntimeError(
                "未找到 git 命令，请确保 git 已安装并添加到 PATH。"
                "Windows 可从 https://git-scm.com/download/win 下载。"
            )
        except subprocess.TimeoutExpired:
            raise RuntimeError(
                f"git clone 超时（180s）。"
                f"可设置环境变量 FOLEYCRAFTER_GIT_MIRROR 使用镜像（如 https://ghproxy.com）"
            )

        if result.returncode != 0:
            raise RuntimeError(
                f"git clone 失败 (exit {result.returncode}): {result.stderr}\n"
                f"可设置环境变量 FOLEYCRAFTER_GIT_MIRROR 使用镜像（如 https://ghproxy.com）"
            )

        # 5. 复制 foleycrafter/ 目录到 site-packages
        src_dir = tmp_repo / "foleycrafter"
        if not src_dir.exists():
            raise RuntimeError(
                f"克隆的仓库中未找到 foleycrafter/ 目录: {src_dir}。"
                f"可能是仓库结构变更，请检查 open-mmlab/FoleyCrafter 仓库。"
            )

        target_dir = site_packages / "foleycrafter"
        if target_dir.exists():
            logger.warning("site-packages 中已存在 foleycrafter/ 目录，将覆盖")
            shutil.rmtree(target_dir)

        shutil.copytree(src_dir, target_dir)
        logger.info("foleycrafter 包已复制到: %s", target_dir)

    # 6. 验证安装
    if importlib.util.find_spec("foleycrafter") is None:
        raise RuntimeError(
            f"foleycrafter 包复制到 site-packages 后仍无法导入。"
            f"请检查 Python 路径: sys.prefix={sys.prefix}, site-packages={site_packages}"
        )

    logger.info("foleycrafter Python 包安装完成")

    # v0.2.14.8+ 修复 foleycrafter/pipelines/pipeline_controlnet.py 的 T2A 模式 bug
    # 必须在 _ensure_foleycrafter_package() 末尾调用，确保每次启动都校验
    _patch_foleycrafter_pipeline_t2a_bug()
    # v0.2.14.8+ 修复 foleycrafter/models/auffusion_unet.py 的 T2A 模式 bug
    _patch_foleycrafter_unet_t2a_bug()


def _patch_foleycrafter_pipeline_t2a_bug() -> None:
    """修复 foleycrafter 包的 T2A 模式 UnboundLocalError（v0.2.14.8+）。

    Bug 位置：foleycrafter/pipelines/pipeline_controlnet.py 第 1213 行
    Bug 代码：
        added_cond_kwargs = {"image_embeds": image_embeds} if image_embeds is not None else None
    问题：T2A 模式下 ip_adapter_image 和 ip_adapter_image_embeds 都为 None，
          第 1138 行 if 块未执行，image_embeds 变量从未赋值，
          此处引用未定义变量 → "cannot access local variable 'image_embeds'"

    修复：改回 diffusers 原版的判断条件（用入参而非中间变量）：
        added_cond_kwargs = (
            {"image_embeds": image_embeds}
            if ip_adapter_image is not None or ip_adapter_image_embeds is not None
            else None
        )

    幂等性：检测文件内容，若已是修复版则跳过。
    持久化：直接修改 site-packages 中的文件，重启进程后仍生效。
    """
    import importlib.util

    # 1. 定位 foleycrafter 包路径
    if importlib.util.find_spec("foleycrafter") is None:
        logger.warning("无法修复 foleycrafter pipeline bug：包未安装")
        return

    import foleycrafter
    pkg_dir = Path(foleycrafter.__file__).parent
    pipeline_file = pkg_dir / "pipelines" / "pipeline_controlnet.py"

    if not pipeline_file.exists():
        logger.warning("foleycrafter pipeline_controlnet.py 不存在: %s", pipeline_file)
        return

    # 2. 读取文件内容，检测是否需要修复
    text = pipeline_file.read_text(encoding="utf-8")
    bug_pattern = 'added_cond_kwargs = {"image_embeds": image_embeds} if image_embeds is not None else None'
    fix_marker = "v0.2.14.8+ 修复 T2A 模式 UnboundLocalError"

    if fix_marker in text:
        logger.info("foleycrafter pipeline T2A bug 已修复（跳过）")
        return

    if bug_pattern not in text:
        logger.warning(
            "foleycrafter pipeline_controlnet.py 未找到 bug 模式，可能版本已变更，跳过修复: %s",
            pipeline_file,
        )
        return

    # 3. 应用修复
    fixed_text = text.replace(
        bug_pattern,
        (
            "# v0.2.14.8+ 修复 T2A 模式 UnboundLocalError:\n"
            "        #   foleycrafter 原版用 `image_embeds is not None` 判断，但 T2A 模式下\n"
            "        #   ip_adapter_image 和 ip_adapter_image_embeds 都为 None 时，\n"
            "        #   image_embeds 变量从未被赋值（第 1138 行 if 块未执行），\n"
            "        #   导致此处引用未定义变量 → \"cannot access local variable 'image_embeds'\"\n"
            "        #   修复方式：改回 diffusers 原版的判断条件（用入参而非中间变量）\n"
            "        added_cond_kwargs = (\n"
            "            {\"image_embeds\": image_embeds}\n"
            "            if ip_adapter_image is not None or ip_adapter_image_embeds is not None\n"
            "            else None\n"
            "        )"
        ),
    )

    pipeline_file.write_text(fixed_text, encoding="utf-8")
    logger.info("已修复 foleycrafter pipeline T2A bug: %s", pipeline_file)

    # 4. 清理 .pyc 缓存，确保下次 import 加载修复后的源码
    pyc_dir = pipeline_file.parent / "__pycache__"
    if pyc_dir.exists():
        import shutil
        shutil.rmtree(pyc_dir)
        logger.info("已清理 __pycache__ 缓存: %s", pyc_dir)


def _patch_foleycrafter_unet_t2a_bug() -> None:
    """修复 foleycrafter 包的 T2A 模式 TypeError（v0.2.14.8+）。

    Bug 位置：foleycrafter/models/auffusion_unet.py 第 1106 行
    Bug 代码：
        elif self.encoder_hid_proj is not None and self.config.encoder_hid_dim_type == "ip_image_proj":
            if "image_embeds" not in added_cond_kwargs:  # added_cond_kwargs 可能是 None
                raise ValueError(...)
    问题：T2A 模式下 pipeline_controlnet.py 返回 added_cond_kwargs=None，
          unet forward 对 None 做 `in` 操作 →
          TypeError: argument of type 'NoneType' is not iterable

    修复：在 ip_image_proj 分支开头检查 added_cond_kwargs is None，
          若为 None 则跳过 IP-Adapter 注入（等价于 scale=0），不 raise。

    幂等性：检测文件内容，若已是修复版则跳过。
    持久化：直接修改 site-packages 中的文件，重启进程后仍生效。
    """
    import importlib.util

    # 1. 定位 foleycrafter 包路径
    if importlib.util.find_spec("foleycrafter") is None:
        logger.warning("无法修复 foleycrafter unet bug：包未安装")
        return

    import foleycrafter
    pkg_dir = Path(foleycrafter.__file__).parent
    unet_file = pkg_dir / "models" / "auffusion_unet.py"

    if not unet_file.exists():
        logger.warning("foleycrafter auffusion_unet.py 不存在: %s", unet_file)
        return

    # 2. 读取文件内容，检测是否需要修复
    text = unet_file.read_text(encoding="utf-8")
    fix_marker = "v0.2.14.8+ 修复 T2A 模式下 added_cond_kwargs=None 导致的 TypeError"

    if fix_marker in text:
        logger.info("foleycrafter unet T2A bug 已修复（跳过）")
        return

    # 3. 定位 bug 代码块（ip_image_proj 分支）
    #    原始代码（单行 if + raise + 后续处理）：
    #        elif self.encoder_hid_proj is not None and self.config.encoder_hid_dim_type == "ip_image_proj":
    #            if "image_embeds" not in added_cond_kwargs:
    #                raise ValueError(
    #                    f"..."
    #                )
    #            image_embeds = added_cond_kwargs.get("image_embeds")
    #            ...
    #            encoder_hidden_states = (encoder_hidden_states, image_embeds)
    #            # encoder_hidden_states = torch.cat([encoder_hidden_states, image_embeds], dim=1)
    bug_pattern = (
        '        elif self.encoder_hid_proj is not None and self.config.encoder_hid_dim_type == "ip_image_proj":\n'
        '            if "image_embeds" not in added_cond_kwargs:\n'
        '                raise ValueError(\n'
        '                    f"{self.__class__} has the config param `encoder_hid_dim_type` set to \'ip_image_proj\' which requires the keyword argument `image_embeds` to be passed in  `added_conditions`"\n'
        '                )\n'
        '            image_embeds = added_cond_kwargs.get("image_embeds")\n'
        '            image_embeds = self.encoder_hid_proj(image_embeds)\n'
        '            if isinstance(image_embeds, list):\n'
        '                image_embeds = [image_embed.to(encoder_hidden_states.dtype) for image_embed in image_embeds]\n'
        '            else:\n'
        '                image_embeds = image_embeds.to(encoder_hidden_states.dtype)\n'
        '            encoder_hidden_states = (encoder_hidden_states, image_embeds)\n'
        '            # encoder_hidden_states = torch.cat([encoder_hidden_states, image_embeds], dim=1)'
    )

    if bug_pattern not in text:
        logger.warning(
            "foleycrafter auffusion_unet.py 未找到 ip_image_proj bug 模式，可能版本已变更，跳过修复: %s",
            unet_file,
        )
        return

    # 4. 应用修复：在 ip_image_proj 分支开头加 added_cond_kwargs is None 检查
    fixed_block = (
        '        elif self.encoder_hid_proj is not None and self.config.encoder_hid_dim_type == "ip_image_proj":\n'
        '            if added_cond_kwargs is None:\n'
        '                # v0.2.14.8+ 修复 T2A 模式下 added_cond_kwargs=None 导致的 TypeError:\n'
        '                #   argument of type \'NoneType\' is not iterable\n'
        '                # T2A 模式没有 image_embeds（pipeline_controlnet.py 第 1213 行返回 None），\n'
        '                # 跳过 IP-Adapter 注入，等价于 IP-Adapter scale=0\n'
        '                pass\n'
        '            else:\n'
        '                if "image_embeds" not in added_cond_kwargs:\n'
        '                    raise ValueError(\n'
        '                        f"{self.__class__} has the config param `encoder_hid_dim_type` set to \'ip_image_proj\' which requires the keyword argument `image_embeds` to be passed in  `added_conditions`"\n'
        '                    )\n'
        '                image_embeds = added_cond_kwargs.get("image_embeds")\n'
        '                image_embeds = self.encoder_hid_proj(image_embeds)\n'
        '                if isinstance(image_embeds, list):\n'
        '                    image_embeds = [image_embed.to(encoder_hidden_states.dtype) for image_embed in image_embeds]\n'
        '                else:\n'
        '                    image_embeds = image_embeds.to(encoder_hidden_states.dtype)\n'
        '                encoder_hidden_states = (encoder_hidden_states, image_embeds)\n'
        '                # encoder_hidden_states = torch.cat([encoder_hidden_states, image_embeds], dim=1)'
    )

    fixed_text = text.replace(bug_pattern, fixed_block)
    unet_file.write_text(fixed_text, encoding="utf-8")
    logger.info("已修复 foleycrafter unet T2A bug: %s", unet_file)

    # 5. 清理 .pyc 缓存
    pyc_dir = unet_file.parent / "__pycache__"
    if pyc_dir.exists():
        import shutil
        shutil.rmtree(pyc_dir)
        logger.info("已清理 __pycache__ 缓存: %s", pyc_dir)


# 视频预处理变换（与 inference.py 对齐）
def _build_video_transform():
    """构建视频帧预处理变换（与 FoleyCrafter inference.py 一致）。"""
    import torchvision

    vision_transform_list = [
        torchvision.transforms.Resize((128, 128)),
        torchvision.transforms.CenterCrop((112, 112)),
        torchvision.transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ]
    return torchvision.transforms.Compose(vision_transform_list)


class FoleyCrafterWrapper:
    """FoleyCrafter 模型包装器。

    封装 4 个子模型 + IP-Adapter encoder 的加载与推理流程。
    所有方法都是同步阻塞调用，由 engine._infer_impl 用 asyncio.to_thread 包装。
    """

    def __init__(self, config: EngineConfig):
        self.config = config
        # v0.3.5+ 设备策略（平台注入 AIRELIABLE_DEVICE_POLICY）
        self.device_policy: str = _read_device_policy()
        self.device: str = self._resolve_device(config.device)
        # v0.3.5+ cpu 策略强制 CPU（优先于 auto 检测与 config.device）
        if self.device_policy == "cpu":
            self.device = "cpu"

        # 模型组件（load() 后初始化）
        self.pipe = None                     # Auffusion pipeline + temporal/semantic adapter
        self.vocoder = None                  # 音频频谱图 → 波形
        self.time_detector = None            # 视频时间戳检测
        self.image_processor = None          # CLIP 图像预处理
        self.image_encoder = None            # CLIP 视觉编码器

        # 状态
        self.is_loaded: bool = False
        self.loaded_variant: str = ""
        self.sample_rate: int = 16000        # FoleyCrafter 固定 16kHz
        self.device_name: str = ""

        # 视频预处理
        self._video_transform = None

    def _resolve_device(self, device: str) -> str:
        """解析设备字符串。"""
        if device == "auto":
            # 延迟导入 torch（避免模块加载时强制依赖 torch）
            import torch
            if torch.cuda.is_available():
                return "cuda"
            return "cpu"
        return device

    def get_vram_used(self) -> int:
        """获取当前显存占用（MB）。"""
        if not self.is_loaded or "cuda" not in self.device:
            return 0
        try:
            import torch
            return int(torch.cuda.memory_allocated() / 1024 / 1024)
        except Exception:
            return 0

    # ===== 加载 / 卸载 =====

    def load(
        self,
        variant: str,
        variant_path: str,
        device: str = "cuda",
    ) -> None:
        """加载 FoleyCrafter 4 个子模型 + IP-Adapter encoder。

        参考 inference.py 的 build_models() 函数。

        Args:
            variant: 变体名（如 foleycrafter-v1）
            variant_path: 变体目录路径（含 4 个子模型 + auffusion/ + ip-adapter/）
            device: 推理设备（"auto" / "cpu" / "cuda" / "cuda:0"）
        """
        # v0.2.14.2+ 修复 device="auto" 未解析导致 .to("auto") 报错：
        #   RuntimeError: Expected one of cpu, cuda, ipu, ... device type at start of device string: auto
        # 原因：engine.py 调用时传 self.config.device（原始 "auto"），而非 self.device（已解析）
        # 这里在 load 入口统一解析，确保后续 .to(device) 收到的是具体设备字符串
        device = self._resolve_device(device)
        # v0.3.5+ cpu 策略强制 CPU（优先于 auto 检测与显式 device 参数）
        if self.device_policy == "cpu":
            device = "cpu"

        # v0.3.5+ 修复：同步 self.device（推理路径 generate() 用 self.device，
        # 此前 load() 的局部 device 未回写，两者可能不一致）
        self.device = device

        logger.info(
            "开始加载 FoleyCrafter 模型: variant=%s, path=%s, device=%s, device_policy=%s",
            variant, variant_path, device, self.device_policy,
        )

        # v0.3.5+ 显存预检（auffusion ~7GB + 组件 ~2GB）：
        #   gpu+cpu 策略显存不足 → 降级 CPU 加载（警告）
        #   gpu 策略显存不足 → 抛友好错误
        #   auto → 不预检（保持现状，OOM 时由 PyTorch 抛原始错误）
        if device.startswith("cuda") and self.device_policy in ("gpu", "gpu+cpu"):
            import torch as _torch
            if _torch.cuda.is_available():
                free_gb = _torch.cuda.mem_get_info()[0] / (1024 ** 3)
                if free_gb < _REQUIRED_VRAM_GB:
                    if self.device_policy == "gpu+cpu":
                        logger.warning(
                            "显存预检不足: free=%.2fGB < required=%.2fGB（gpu+cpu 策略，降级 CPU）",
                            free_gb, _REQUIRED_VRAM_GB,
                        )
                        device = "cpu"
                        self.device = "cpu"
                    else:
                        raise RuntimeError(
                            f"GPU 显存不足：空闲 {free_gb:.2f}GB < 需要 {_REQUIRED_VRAM_GB}GB"
                            f"（auffusion pipeline ~7GB + 组件 ~2GB）。"
                            f"建议：1) 关闭其他占用显存的程序 2) 卸载其他已加载的引擎 "
                            f"3) 设备策略切换为 gpu+cpu（自动降级 CPU）"
                        )
                else:
                    logger.info(
                        "显存预检通过: free=%.2fGB >= required=%.2fGB",
                        free_gb, _REQUIRED_VRAM_GB,
                    )

        # v0.2.12+ 确保 foleycrafter Python 包已安装到 .venv
        # FoleyCrafter 的 Python 代码位于 GitHub 仓库，不在 HuggingFace 权重中
        # 若未安装则自动 git clone 并复制到 site-packages（幂等）
        _ensure_foleycrafter_package()

        # 延迟导入（避免 SDK 安装时就要求 torch + diffusers）
        import torch
        import numpy as np

        # v0.2.14.3+ 修复 PyTorch 2.6+ weights_only=True 导致权重加载失败：
        #   _pickle.UnpicklingError: Weights only load failed.
        #   Unsupported global: GLOBAL numpy.core.multiarray.scalar
        # 必须在导入 foleycrafter 之前设置，否则 torch_utils.py 内部的
        # torch.load() 调用会使用默认的 weights_only=True。
        #
        # 双重保险：
        # 1. add_safe_globals：将 numpy 标量加入安全列表（PyTorch 官方推荐方案）
        # 2. monkey-patch torch.load：强制 weights_only=False（覆盖所有 torch.load 调用）
        try:
            torch.serialization.add_safe_globals([np.core.multiarray.scalar])
        except Exception as e:
            logger.warning("add_safe_globals 失败（可能 PyTorch 版本不支持）: %s", e)

        _orig_torch_load = torch.load

        def _patched_torch_load(*args, **kwargs):
            if "map_location" not in kwargs:
                kwargs["map_location"] = "cpu"
            if "weights_only" not in kwargs:
                kwargs["weights_only"] = False
            return _orig_torch_load(*args, **kwargs)

        torch.load = _patched_torch_load
        logger.info("已应用 torch.load monkey-patch（weights_only=False + map_location=cpu）")

        # v0.2.14.7+ 修复 foleycrafter 包与 diffusers 0.34.0 的版本不兼容问题：
        # foleycrafter/models/auffusion/loaders/ip_adapter.py 从旧版 diffusers（~0.18.x）复制，
        # 调用 _get_model_file(resume_download=...) 但 diffusers 0.34.0 已移除 resume_download 参数
        # → TypeError: _get_model_file() got an unexpected keyword argument 'resume_download'
        # 修复：monkey-patch _get_model_file，接受并忽略 resume_download 参数
        from diffusers.utils import _get_model_file as _orig_get_model_file

        def _patched_get_model_file(*args, **kwargs):
            # 旧版 diffusers 的 resume_download 参数已废弃，新版用 HF_HUB_DOWNLOAD_TIMEOUT
            kwargs.pop("resume_download", None)
            return _orig_get_model_file(*args, **kwargs)

        # patch diffusers.utils 模块中的 _get_model_file
        import diffusers.utils as _diffusers_utils
        _diffusers_utils._get_model_file = _patched_get_model_file
        # 同时 patch foleycrafter 已经导入的引用（from diffusers.utils import _get_model_file）
        import foleycrafter.models.auffusion.loaders.ip_adapter as _fc_ip_adapter
        _fc_ip_adapter._get_model_file = _patched_get_model_file
        logger.info("已应用 _get_model_file monkey-patch（兼容 diffusers 0.34.0，忽略 resume_download 参数）")

        # v0.2.14.6+ 移除 snapshot_download 导入：load() 不再负责下载，文件缺失直接抛错
        from transformers import CLIPImageProcessor, CLIPVisionModelWithProjection
        from foleycrafter.models.onset import torch_utils
        from foleycrafter.models.time_detector.model import VideoOnsetNet
        from foleycrafter.pipelines.auffusion_pipeline import Generator, denormalize_spectrogram
        from foleycrafter.utils.util import build_foleycrafter, read_frames_with_moviepy

        # 保存导入供 generate() 使用
        self._denormalize_spectrogram = denormalize_spectrogram
        self._read_frames_with_moviepy = read_frames_with_moviepy

        variant_dir = Path(variant_path)
        ckpt_dir = variant_dir
        auffusion_dir = variant_dir / "auffusion"
        ip_adapter_dir = variant_dir / "ip-adapter"

        # ===== 1. 加载 Auffusion 基座（如缺失或大小异常则 snapshot_download） =====
        # 注意：不能只检查 model_index.json 标志文件，因为平台预检下载器只会下载
        # 声明的单个标志文件（几 KB），不会触发整仓 snapshot_download。必须检查
        # 关键权重子目录是否齐全，才能判定 Auffusion 基座真正就绪。
        #
        # v0.2.14.2+ 新增文件大小校验：
        # 断点续传或下载中断可能导致文件存在但大小异常（比如只剩几百字节的残文件）。
        # 仅检查 exists() 会误判为"已就绪"，导致 torch.load 报错。
        # 这里对每个关键文件检查最小合理大小，不达标则触发重新下载。
        _auffusion_key_paths = (
            # ===== pipeline 索引 =====
            "model_index.json",                   # pipeline 索引
            # ===== UNet =====
            "unet/diffusion_pytorch_model.bin",   # UNet 主权重（~3.44GB）
            "unet/config.json",                   # UNet 配置
            # ===== VAE =====
            "vae/diffusion_pytorch_model.bin",    # VAE 权重
            "vae/config.json",                    # VAE 配置
            # ===== CLIP 文本编码器 =====
            "text_encoder/pytorch_model.bin",     # CLIP 文本编码器（~492MB）
            "text_encoder/config.json",           # CLIP 文本编码器配置
            # v0.2.14.6+ 一次性补全所有 diffusers pipeline 必需文件：
            # model_index.json 声明了 6 个组件（unet/vae/text_encoder/tokenizer/scheduler/feature_extractor）
            # StableDiffusionPipeline.from_pretrained() 会加载所有组件，缺任何一个都会抛 OSError → 500
            # 之前只校验 7 个文件，漏了 6 个，导致校验通过但加载失败
            # ===== CLIP tokenizer =====
            "tokenizer/vocab.json",               # tokenizer 词表
            "tokenizer/merges.txt",               # tokenizer BPE 合并表
            "tokenizer/tokenizer_config.json",    # tokenizer 配置
            "tokenizer/special_tokens_map.json",  # tokenizer 特殊 token 映射
            # ===== PNDM 调度器 =====
            "scheduler/scheduler_config.json",    # PNDM 调度器配置
            # ===== CLIP 图像预处理器 =====
            "feature_extractor/preprocessor_config.json",  # CLIP 图像预处理器配置
        )
        # 每个文件对应的最小合理字节数（低于此值视为损坏，触发重下）
        # 权重文件至少 1MB，config 文件至少 100 字节
        _auffusion_min_sizes = {
            # 权重文件
            "unet/diffusion_pytorch_model.bin": 100 * 1024 * 1024,   # 100MB（实际 3.44GB）
            "vae/diffusion_pytorch_model.bin":  10 * 1024 * 1024,    # 10MB（实际 ~100MB）
            "text_encoder/pytorch_model.bin":   50 * 1024 * 1024,    # 50MB（实际 492MB）
            # 配置文件
            "model_index.json":                              100,
            "unet/config.json":                              100,
            "vae/config.json":                               100,
            "text_encoder/config.json":                      100,
            # tokenizer 文件
            "tokenizer/vocab.json":                          100_000,    # ~1MB（CLIP 词表）
            "tokenizer/merges.txt":                          10_000,     # ~500KB
            "tokenizer/tokenizer_config.json":               100,
            "tokenizer/special_tokens_map.json":             100,
            # scheduler / feature_extractor
            "scheduler/scheduler_config.json":               100,
            "feature_extractor/preprocessor_config.json":    100,
        }

        def _check_files_integrity(base_dir: Path, key_paths: tuple, min_sizes: dict) -> bool:
            """校验关键文件是否齐全且大小达标。

            v0.2.14.2+ 新增：防止断点续传导致的不完整文件。
            若文件缺失或大小低于阈值，返回 False 触发重新下载。
            同时清理不完整文件（删除过小的权重文件，避免 snapshot_download 跳过）。
            """
            if not base_dir.exists():
                return False
            all_ok = True
            for rel_path in key_paths:
                full_path = base_dir / rel_path
                if not full_path.exists():
                    logger.warning("文件缺失: %s", full_path)
                    all_ok = False
                    continue
                actual_size = full_path.stat().st_size
                min_size = min_sizes.get(rel_path, 1024)  # 默认至少 1KB
                if actual_size < min_size:
                    logger.warning(
                        "文件大小异常（可能下载不完整）: %s, 实际=%d 字节, 期望>=%d 字节, 将删除并重新下载",
                        full_path, actual_size, min_size,
                    )
                    # 删除损坏文件，避免 snapshot_download 认为文件已存在而跳过
                    try:
                        full_path.unlink()
                    except OSError as e:
                        logger.warning("删除损坏文件失败: %s, error=%s", full_path, e)
                    all_ok = False
            return all_ok

        _auffusion_ready = _check_files_integrity(
            auffusion_dir, _auffusion_key_paths, _auffusion_min_sizes
        )
        if not _auffusion_ready:
            # v0.2.14.6+ 移除兜底下载逻辑：
            # 之前在 load() 中调用 snapshot_download 兜底下载，存在多个问题：
            #   1. huggingface_hub.constants.ENDPOINT 在 import 时缓存，后续修改 os.environ 不生效
            #      （即使设置了 HF_ENDPOINT=https://hf-mirror.com，snapshot_download 仍连 huggingface.co）
            #   2. snapshot_download 网络超时时静默返回本地残缺文件（不抛异常），导致二次校验失败
            #   3. 测试驱动发现文件：每次启动报错才知道哪个文件缺失，用户体验极差
            # 正确架构：load() 只负责加载，不负责下载。文件缺失直接抛错，让用户去模型管理界面下载。
            # 平台模型管理界面支持镜像选择（official / hf-mirror / modelscope）和"全部下载"按钮。
            missing = [
                str(p) for p in _auffusion_key_paths
                if not (auffusion_dir / p).exists()
            ]
            raise RuntimeError(
                f"Auffusion 基座模型文件不完整，缺失 {len(missing)} 个文件：\n"
                f"  {missing}\n\n"
                f"请到引擎详情页 → 模型管理 → 点击「下载缺失文件」或「重新下载」按钮，"
                f"选择镜像源（推荐 hf-mirror.com）后下载。\n"
                f"本地路径：{auffusion_dir}\n"
                f"HuggingFace 仓库：auffusion/auffusion-full-no-adapter"
            )

        # ===== 2. 校验 FoleyCrafter 仓库文件（temporal/semantic/timestamp/vocoder） =====
        # vocoder 文件从 ymzhang319/FoleyCrafter 下载，放在 v1/vocoder/ 下
        # （auffusion 仓库也有 vocoder 文件，但引擎不读取 auffusion/vocoder/，只读 v1/vocoder/）
        fc_ckpt = variant_dir
        _fc_key_paths = (
            "temporal_adapter.ckpt",        # ControlNet 权重
            "timestamp_detector.pth.tar",   # 时间戳检测器
            "semantic/semantic_adapter.bin",  # IP-Adapter 权重
            "vocoder/config.json",          # vocoder 配置
            "vocoder/vocoder.pt",           # vocoder 权重（Generator.from_pretrained 必需）
        )
        _fc_min_sizes = {
            "temporal_adapter.ckpt":         10 * 1024 * 1024,   # 10MB
            "timestamp_detector.pth.tar":    10 * 1024 * 1024,   # 10MB
            "semantic/semantic_adapter.bin":  1 * 1024 * 1024,   # 1MB
            "vocoder/config.json":            100,                # 100 字节
            "vocoder/vocoder.pt":             1 * 1024 * 1024,   # 1MB
        }
        _fc_ready = _check_files_integrity(variant_dir, _fc_key_paths, _fc_min_sizes)
        if not _fc_ready:
            # v0.2.14.6+ 移除兜底下载逻辑（同上）
            missing = [
                str(p) for p in _fc_key_paths
                if not (variant_dir / p).exists()
            ]
            raise RuntimeError(
                f"FoleyCrafter checkpoints 文件不完整，缺失 {len(missing)} 个文件：\n"
                f"  {missing}\n\n"
                f"请到引擎详情页 → 模型管理 → 点击「下载缺失文件」或「重新下载」按钮，"
                f"选择镜像源（推荐 hf-mirror.com）后下载。\n"
                f"本地路径：{variant_dir}\n"
                f"HuggingFace 仓库：ymzhang319/FoleyCrafter"
            )

        # ===== 2.5 校验 IP-Adapter image_encoder 关键文件 =====
        _ip_adapter_key_paths = (
            "config.json",          # 配置文件
            "model.safetensors",    # IP-Adapter image_encoder 权重（~1.5GB）
        )
        _ip_adapter_min_sizes = {
            "config.json":       100,                       # 100 字节
            "model.safetensors": 100 * 1024 * 1024,         # 100MB（实际 1.5GB）
        }
        _ip_adapter_ready = _check_files_integrity(
            ip_adapter_dir, _ip_adapter_key_paths, _ip_adapter_min_sizes
        )
        if not _ip_adapter_ready:
            # v0.2.14.6+ 移除兜底下载逻辑（同上）
            missing = [
                str(p) for p in _ip_adapter_key_paths
                if not (ip_adapter_dir / p).exists()
            ]
            raise RuntimeError(
                f"IP-Adapter image_encoder 文件不完整，缺失 {len(missing)} 个文件：\n"
                f"  {missing}\n\n"
                f"请到引擎详情页 → 模型管理 → 点击「下载缺失文件」或「重新下载」按钮，"
                f"选择镜像源（推荐 hf-mirror.com）后下载。\n"
                f"本地路径：{ip_adapter_dir}\n"
                f"HuggingFace 仓库：h94/IP-Adapter（models/image_encoder/ 子目录）"
            )

        # monkey-patch 已在导入 foleycrafter 之前应用（见上方 line 292-301）
        # 这里直接开始加载模型，try/finally 确保 load 完成后恢复原始 torch.load
        try:
            # ===== 4. 加载 vocoder =====
            logger.info("加载 vocoder...")
            self.vocoder = Generator.from_pretrained(str(fc_ckpt), subfolder="vocoder").to(device)

            # ===== 5. 加载 timestamp_detector =====
            logger.info("加载 timestamp_detector...")
            time_detector_ckpt = str(variant_dir / "timestamp_detector.pth.tar")
            self.time_detector = VideoOnsetNet(False)
            # torch_utils.load_model 内部调用 torch.load，已被 monkey-patch 覆盖
            self.time_detector, _ = torch_utils.load_model(
                time_detector_ckpt,
                self.time_detector,
                device=device,
                strict=True,
            )

            # ===== 6. 加载 Auffusion pipeline + temporal_adapter + semantic_adapter =====
            logger.info("加载 Auffusion pipeline + adapters...")
            # 显式传入本地 auffusion_dir，避免 build_foleycrafter() 走默认 HF repo ID
            # （否则会忽略本地已下载的 auffusion_dir，改从 HF 缓存读取，造成重复下载）
            self.pipe = build_foleycrafter(
                pretrained_model_name_or_path=str(auffusion_dir),
            ).to(device)

            # 加载 temporal_adapter（ControlNet）
            temporal_ckpt_path = str(variant_dir / "temporal_adapter.ckpt")
            ckpt = torch.load(temporal_ckpt_path, map_location=device)
            if "state_dict" in ckpt.keys():
                ckpt = ckpt["state_dict"]
            # 剥离 module. 前缀（DDP 训练产物）
            load_gligen_ckpt = {}
            for key, value in ckpt.items():
                if key.startswith("module."):
                    load_gligen_ckpt[key[len("module."):]] = value
                else:
                    load_gligen_ckpt[key] = value
            m, u = self.pipe.controlnet.load_state_dict(load_gligen_ckpt, strict=False)
            logger.info("ControlNet 加载完成: missing=%d, unexpected=%d", len(m), len(u))

            # 加载 semantic_adapter（IP-Adapter）
            self.pipe.load_ip_adapter(
                osp.join(str(variant_dir), "semantic"),
                subfolder="",
                weight_name="semantic_adapter.bin",
                image_encoder_folder=None,
            )

            # ===== 7. 加载 IP-Adapter image_encoder（CLIP 视觉编码器） =====
            logger.info("加载 CLIP image_encoder...")
            # v0.2.14.6+ 移除兜底下载逻辑：文件完整性已在上方校验（_ip_adapter_ready），
            # 如果执行到这里说明 config.json + model.safetensors 都已存在。
            # 之前这里有第四处 snapshot_download 兜底下载，是重复检查的死代码。
            self.image_processor = CLIPImageProcessor()
            self.image_encoder = CLIPVisionModelWithProjection.from_pretrained(
                str(ip_adapter_dir),
            ).to(device)

            # 视频预处理
            self._video_transform = _build_video_transform()

            # 状态更新
            self.is_loaded = True
            self.loaded_variant = variant
            self.device_name = torch.cuda.get_device_name(0) if "cuda" in device else "CPU"

            logger.info(
                "FoleyCrafter 加载完成: variant=%s, device=%s, sample_rate=%d",
                variant, device, self.sample_rate,
            )
        finally:
            # 无论加载成功或失败，都恢复原始 torch.load 和 _get_model_file
            torch.load = _orig_torch_load
            _diffusers_utils._get_model_file = _orig_get_model_file
            _fc_ip_adapter._get_model_file = _orig_get_model_file

    def unload(self) -> None:
        """卸载所有模型，释放显存。"""
        logger.info("开始卸载 FoleyCrafter 模型...")

        # 释放模型引用
        self.pipe = None
        self.vocoder = None
        self.time_detector = None
        self.image_processor = None
        self.image_encoder = None
        self._video_transform = None

        # 状态重置
        self.is_loaded = False
        self.loaded_variant = ""

        # 触发 GC + 清空 CUDA 缓存
        import gc
        import torch
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

        logger.info("FoleyCrafter 卸载完成")

    # ===== 推理 =====

    def generate(
        self,
        prompt: str = "",
        negative_prompt: str = "",
        seed: int = 42,
        temporal_align: bool = True,
        temporal_scale: float = 0.2,
        semantic_scale: float = 1.0,
        num_inference_steps: int = 25,
        video_path: str = "",
        max_video_duration: float = 10.0,
        progress_callback=None,
    ) -> tuple[np.ndarray, int]:
        """V2A / T2A 推理。

        Args:
            prompt: 正向文本提示（可空）
            negative_prompt: 负向文本提示
            seed: 随机种子
            temporal_align: 是否启用时间对齐（V2A 模式）
            temporal_scale: 时间对齐强度
            semantic_scale: 语义强度（IP-Adapter scale）
            num_inference_steps: 扩散推理步数
            video_path: 视频路径（V2A 模式，空则为 T2A）
            max_video_duration: 视频长度限制（秒）
            progress_callback: 进度回调 fn(step, total_steps)

        Returns:
            (audio_samples, sample_rate)
            audio_samples: np.ndarray, shape (samples,)
            sample_rate: int（固定 16000）
        """
        if not self.is_loaded:
            raise RuntimeError("模型未加载，请先调用 load()")

        # 设置 IP-Adapter scale
        self.pipe.set_ip_adapter_scale(semantic_scale)

        # 设置随机种子
        import torch
        generator = torch.Generator(device=self.device)
        generator.manual_seed(seed)

        # ===== V2A 模式：处理视频输入 =====
        time_condition = None
        image_embeddings = None
        duration = 0.0

        if video_path:
            logger.info("V2A 模式：处理视频 %s", video_path)
            duration, time_condition, image_embeddings = self._process_video(
                video_path,
                max_video_duration,
                temporal_align,
                temporal_scale,
            )
        else:
            logger.info("T2A 模式：仅文本生成")
            # T2A 模式默认 10 秒音频
            duration = 10.0
            # v0.2.14.7+ 修复 T2A 模式下 ControlNet pipeline 报错：
            # StableDiffusionControlNetPipeline 强制要求 image 参数非 None，
            # T2A 模式没有视频，需要构造一个全 -1 的占位条件（表示"无时间戳事件"）
            # shape 与 V2A 模式的 time_condition 一致：[1, 1, 256, 1024]
            time_condition = (
                torch.FloatTensor([-1] * 1024)
                .unsqueeze(0)
                .unsqueeze(0)
                .unsqueeze(0)
                .repeat(1, 1, 256, 1)
                .to(self.device)
            )
            # v0.2.14.8+ 修复 T2A 模式下 IP-Adapter 相关的连锁错误：
            #   1. pipeline_controlnet.py 第 1213 行：image_embeds UnboundLocalError
            #   2. auffusion_unet.py 第 1106 行：added_cond_kwargs None TypeError
            #   3. attention_processor：用错误的 ip_hidden_states（text_embeds 最后 4 token）
            # 根本原因：T2A 模式没有视频帧，无法用 CLIP image_encoder 编码，image_embeddings=None。
            #   但 unet 配置了 encoder_hid_dim_type="ip_image_proj"，attention_processor
            #   期望 encoder_hidden_states 是 (text_embeds, image_embeds) tuple。
            #   image_embeddings=None 导致 pipeline 返回 added_cond_kwargs=None，
            #   unet forward 跳过 ip_image_proj 分支，encoder_hidden_states 保持原始值（纯文本），
            #   attention_processor 误把 text_embeds 最后 4 个 token 当作 ip_hidden_states。
            #
            # 修复方案：构造全零 image_embeddings（与 V2A shape 一致：[1, 2, 1, 1024]），
            #   让 pipeline 正常走 IP-Adapter 流程，但 image_embeds 全零 → scale * 0 = 0，
            #   IP-Adapter 不影响结果（等价于 semantic_scale=0）。
            #   shape 对齐 V2A 的 _process_video 返回值：
            #     [1, 2, 1, 1024] = [batch, neg+pos, num_tokens, embed_dim]
            image_embeddings = torch.zeros(1, 2, 1, 1024, device=self.device)

        # ===== 扩散推理 =====
        logger.info(
            "开始扩散推理: prompt=%r, steps=%d, duration=%.2fs",
            prompt, num_inference_steps, duration,
        )

        # 构造调用参数
        pipe_kwargs = {
            "prompt": prompt,
            "negative_prompt": negative_prompt,
            "num_inference_steps": num_inference_steps,
            "height": 256,
            "width": 1024,
            "output_type": "pt",
            "generator": generator,
        }

        # ControlNet 条件（V2A 和 T2A 都需要传入 image，否则 pipeline 报错）
        if time_condition is not None:
            pipe_kwargs["image"] = time_condition
            # V2A 模式用 temporal_scale，T2A 模式设为 0（让 ControlNet 不影响结果）
            if video_path and temporal_align:
                pipe_kwargs["controlnet_conditioning_scale"] = temporal_scale
            else:
                pipe_kwargs["controlnet_conditioning_scale"] = 0.0

        if image_embeddings is not None:
            pipe_kwargs["ip_adapter_image_embeds"] = image_embeddings

        # 执行推理（带进度回调）
        if progress_callback is not None:
            # diffusers 的 callback 签名：callback(step, timestep, latents)
            def _diffusers_callback(step, timestep, latents):
                progress_callback(step, num_inference_steps)
            pipe_kwargs["callback"] = _diffusers_callback
            pipe_kwargs["callback_steps"] = 1

        sample = self.pipe(**pipe_kwargs)

        # 解码频谱图 → 音频
        audio_img = sample.images[0]
        audio = self._denormalize_spectrogram(audio_img)
        audio = self.vocoder.inference(audio, lengths=160000)[0]

        # 截取到视频时长
        if video_path and duration > 0:
            audio = audio[:int(duration * 16000)]

        logger.info(
            "推理完成: audio_shape=%s, sample_rate=%d, duration=%.2fs",
            audio.shape, self.sample_rate, len(audio) / self.sample_rate,
        )

        return audio, self.sample_rate

    def _process_video(
        self,
        video_path: str,
        max_duration: float,
        temporal_align: bool,
        temporal_scale: float,
    ) -> tuple[float, Optional[torch.Tensor], Optional[torch.Tensor]]:
        """处理视频输入（V2A 模式）。

        Returns:
            (duration, time_condition, image_embeddings)
        """
        import torch
        import torchvision

        # 1. 读视频帧
        frames, duration = self._read_frames_with_moviepy(
            video_path,
            max_frame_nums=int(30 * max_duration),  # 默认 30fps
        )

        # 限制视频时长
        if duration > max_duration:
            duration = max_duration
            logger.info("视频截断到 %.2fs", duration)

        # 2. 时间戳检测
        time_frames = torch.FloatTensor(frames).permute(0, 3, 1, 2)
        if self._video_transform is not None:
            time_frames = self._video_transform(time_frames)
        time_frames = {"frames": time_frames.unsqueeze(0).permute(0, 2, 1, 3, 4)}

        preds = self.time_detector(time_frames)
        preds = torch.sigmoid(preds)

        # 构造时间条件
        time_condition = [
            -1 if preds[0][int(i / (1024 / 10 * duration) * len(frames))] < 0.5 else 1
            for i in range(int(1024 / 10 * duration))
        ]
        time_condition = time_condition + [-1] * (1024 - len(time_condition))
        time_condition = (
            torch.FloatTensor(time_condition)
            .unsqueeze(0)
            .unsqueeze(0)
            .unsqueeze(0)
            .repeat(1, 1, 256, 1)
            .to(self.device)
        )

        # 3. CLIP 视觉编码
        images = self.image_processor(images=frames, return_tensors="pt").to(self.device)
        image_embeddings = self.image_encoder(**images).image_embeds
        image_embeddings = torch.mean(image_embeddings, dim=0, keepdim=True).unsqueeze(0).unsqueeze(0)
        neg_image_embeddings = torch.zeros_like(image_embeddings)
        image_embeddings = torch.cat([neg_image_embeddings, image_embeddings], dim=1)

        return duration, time_condition, image_embeddings
