"""FoleyCrafter 模型下载脚本。

使用 huggingface_hub.snapshot_download() 把 FoleyCrafter 所需的 3 个 HF 仓库文件
聚合到 models/audio/v2a/foleycrafter/<variant>/ 目录。

同时通过 git clone 把 GitHub 仓库 open-mmlab/FoleyCrafter 的 foleycrafter/ Python 包
安装到引擎 .venv 的 site-packages 目录（v0.2.12+）。

3 个 HuggingFace 仓库：
    1. ymzhang319/FoleyCrafter          → temporal_adapter.ckpt + semantic/ + timestamp_detector.pth.tar
    2. auffusion/auffusion-full-no-adapter → auffusion/（diffusers pipeline 目录）
    3. h94/IP-Adapter                    → ip-adapter/（image_encoder 子目录）

1 个 GitHub 仓库（Python 包代码）：
    4. open-mmlab/FoleyCrafter           → foleycrafter/（Python 包，含 models/pipelines/utils）

用法：
    # 通过 uv 环境运行（确保 huggingface_hub 已安装）
    uv run --no-sync --project envs/audio-foleycrafter \\
        python -m engines.foleycrafter_a3b2c1.scripts.download_foleycrafter --variant foleycrafter-v1

    # 使用 HuggingFace 镜像（国内推荐）
    $env:HF_ENDPOINT="https://hf-mirror.com"
    uv run --no-sync --project envs/audio-foleycrafter \\
        python -m engines.foleycrafter_a3b2c1.scripts.download_foleycrafter --variant foleycrafter-v1

    # 使用 GitHub 镜像加速 foleycrafter Python 包克隆（国内推荐）
    $env:FOLEYCRAFTER_GIT_MIRROR="https://ghproxy.com"
    uv run --no-sync --project envs/audio-foleycrafter \\
        python -m engines.foleycrafter_a3b2c1.scripts.download_foleycrafter --variant foleycrafter-v1

支持的变体：
    - foleycrafter-v1  FoleyCrafter 主力模型（Auffusion 基座 + 4 子模型，~10GB）

下载完成后目录结构：
    models/audio/v2a/foleycrafter/v1/
        ├── temporal_adapter.ckpt
        ├── semantic/
        │   └── semantic_adapter.bin
        ├── timestamp_detector.pth.tar
        ├── auffusion/
        │   ├── model_index.json
        │   └── ...（diffusers 目录）
        └── ip-adapter/
            ├── config.json
            └── model.safetensors

    envs/audio-foleycrafter/.venv/Lib/site-packages/foleycrafter/
        ├── data/
        ├── models/
        ├── pipelines/
        └── utils/
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

# 项目根目录（engines/foleycrafter_a3b2c1/scripts/ 往上三级）
PROJECT_ROOT = Path(__file__).resolve().parents[3]

# 已知变体定义（与 model_scanner.py _KNOWN_VARIANTS 保持一致）
VARIANTS: dict[str, dict] = {
    "foleycrafter-v1": {
        "subdir": "v1",
        "description": "FoleyCrafter v1 主力模型（Auffusion 基座 + 4 子模型，~10GB）",
        # 3 个 HF 仓库的下载配置
        "repos": [
            {
                "repo_id": "ymzhang319/FoleyCrafter",
                "target_subdir": ".",                # 直接下载到 variant 根目录
                "description": "FoleyCrafter 3 子模型（temporal/semantic/timestamp，~1.1GB）",
            },
            {
                "repo_id": "auffusion/auffusion-full-no-adapter",
                "target_subdir": "auffusion",        # 下载到 variant/auffusion/
                "description": "Auffusion 基座模型（diffusers pipeline 目录，~8GB）",
            },
            {
                "repo_id": "h94/IP-Adapter",
                "target_subdir": "ip-adapter",       # 下载到 variant/ip-adapter/
                "allow_patterns": ["models/image_encoder/*"],  # 仅下载 image_encoder 子目录
                "description": "IP-Adapter CLIP image_encoder（~1.5GB）",
                # h94/IP-Adapter 仓库结构是 models/image_encoder/，需平移到 ip-adapter/
                "flatten_subdir": "models/image_encoder",
            },
        ],
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="下载 FoleyCrafter 模型到规范目录 models/audio/v2a/foleycrafter/<variant>/"
    )
    parser.add_argument(
        "--variant",
        required=True,
        choices=list(VARIANTS.keys()),
        help="要下载的模型变体名",
    )
    parser.add_argument(
        "--models-root",
        default=str(PROJECT_ROOT / "models"),
        help="模型根目录（默认: <project>/models）",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="即使目标目录已存在也重新下载（覆盖）",
    )
    parser.add_argument(
        "--skip-auffusion",
        action="store_true",
        help="跳过 Auffusion 基座下载（~8GB，假设已存在）",
    )
    parser.add_argument(
        "--skip-package",
        action="store_true",
        help="跳过 foleycrafter Python 包安装（假设 .venv 中已安装）",
    )
    return parser.parse_args()


def _download_repo(
    repo_id: str,
    target_dir: Path,
    allow_patterns: list[str] | None = None,
    force: bool = False,
) -> None:
    """下载单个 HF 仓库到目标目录。

    Args:
        repo_id: HuggingFace 仓库 ID（如 ymzhang319/FoleyCrafter）
        target_dir: 本地目标目录
        allow_patterns: 允许下载的文件模式（None = 全部）
        force: 即使目录已存在也重新下载
    """
    from huggingface_hub import snapshot_download

    target_dir.mkdir(parents=True, exist_ok=True)

    # 检查是否已下载（粗略判断：目录非空且非 force）
    if not force and any(target_dir.iterdir()):
        print(f"  [跳过] {target_dir} 已存在文件，使用 --force 强制重新下载")
        return

    print(f"  [下载] {repo_id} → {target_dir}")
    if allow_patterns:
        print(f"         allow_patterns={allow_patterns}")

    start = time.time()
    snapshot_download(
        repo_id=repo_id,
        local_dir=str(target_dir),
        allow_patterns=allow_patterns,
        max_workers=4,
    )
    elapsed = time.time() - start
    print(f"  [完成] 耗时 {elapsed:.1f}s")


def _flatten_subdir(target_dir: Path, subdir_path: str) -> None:
    """把 target_dir/subdir_path/ 下的文件平移到 target_dir/。

    用于 h94/IP-Adapter 仓库结构：snapshot_download 后路径为
        target_dir/models/image_encoder/config.json
    需平移为
        target_dir/config.json

    Args:
        target_dir: 下载根目录
        subdir_path: 需平移的子目录相对路径（如 "models/image_encoder"）
    """
    src_dir = target_dir / subdir_path
    if not src_dir.exists():
        print(f"  [警告] 子目录不存在: {src_dir}")
        return

    print(f"  [平移] {src_dir} → {target_dir}")
    for f in src_dir.iterdir():
        target = target_dir / f.name
        if target.exists():
            print(f"    [跳过] {f.name} 已存在")
            continue
        f.rename(target)

    # 删除空的 models/ 目录（递归向上）
    try:
        src_dir.rmdir()
        (target_dir / "models").rmdir()
    except OSError:
        pass    # 目录非空，保留


def _install_foleycrafter_package(force: bool = False) -> None:
    """安装 foleycrafter Python 包到引擎 .venv 的 site-packages（v0.2.12+）。

    FoleyCrafter 的 Python 代码位于 GitHub 仓库 open-mmlab/FoleyCrafter 的 foleycrafter/ 目录，
    但该仓库没有标准的 setup.py / pyproject.toml [project] 块，无法用 pip install 安装。
    本函数通过 git clone 克隆仓库到临时目录，然后复制 foleycrafter/ 到 site-packages。

    Args:
        force: 强制重新安装（即使已安装也重新克隆）
    """
    import importlib.util
    import shutil
    import subprocess
    import sys
    import tempfile

    # 1. 检查是否已安装
    if not force and importlib.util.find_spec("foleycrafter") is not None:
        print("  [跳过] foleycrafter Python 包已安装，使用 --force 强制重新安装")
        return

    print("  [安装] foleycrafter Python 包到 .venv site-packages...")

    # 2. 定位 site-packages 目录
    if sys.platform == "win32":
        site_packages = Path(sys.prefix) / "Lib" / "site-packages"
    else:
        lib_dir = Path(sys.prefix) / "lib"
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
        print(f"         使用 GitHub 镜像: {mirror}")

    # 4. git clone --depth 1 到临时目录
    with tempfile.TemporaryDirectory(prefix="foleycrafter_clone_") as tmp_dir:
        tmp_repo = Path(tmp_dir) / "FoleyCrafter"
        cmd = ["git", "clone", "--depth", "1", repo_url, str(tmp_repo)]
        print(f"         执行: {' '.join(cmd)}")

        try:
            result = subprocess.run(
                cmd, capture_output=True, text=True, timeout=180,
            )
        except FileNotFoundError:
            raise RuntimeError(
                "未找到 git 命令，请确保 git 已安装并添加到 PATH。"
                "Windows 可从 https://git-scm.com/download/win 下载。"
            )
        except subprocess.TimeoutExpired:
            raise RuntimeError(
                "git clone 超时（180s）。"
                "可设置环境变量 FOLEYCRAFTER_GIT_MIRROR 使用镜像（如 https://ghproxy.com）"
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
                f"克隆的仓库中未找到 foleycrafter/ 目录: {src_dir}"
            )

        target_dir = site_packages / "foleycrafter"
        if target_dir.exists():
            print(f"         [覆盖] site-packages 中已存在 foleycrafter/ 目录")
            shutil.rmtree(target_dir)

        shutil.copytree(src_dir, target_dir)
        print(f"         [完成] foleycrafter 包已复制到: {target_dir}")

    # 6. 验证安装
    if importlib.util.find_spec("foleycrafter") is None:
        raise RuntimeError(
            f"foleycrafter 包复制到 site-packages 后仍无法导入。"
            f"请检查 Python 路径: sys.prefix={sys.prefix}"
        )

    print("  [成功] foleycrafter Python 包安装完成")


def main() -> int:
    args = parse_args()

    variant_info = VARIANTS[args.variant]
    subdir = variant_info["subdir"]

    # 目标目录：models/audio/v2a/foleycrafter/<variant>/
    target_root = Path(args.models_root) / "audio" / "v2a" / "foleycrafter" / subdir
    target_root.mkdir(parents=True, exist_ok=True)

    print(f"===== FoleyCrafter 模型下载 =====")
    print(f"变体: {args.variant} ({variant_info['description']})")
    print(f"目标目录: {target_root}")
    print(f"Force: {args.force}")
    print()

    if "HF_ENDPOINT" in os.environ:
        print(f"使用 HuggingFace 镜像: {os.environ['HF_ENDPOINT']}")
    else:
        print("提示：国内用户可设置 HF_ENDPOINT=https://hf-mirror.com 加速下载")

    if "FOLEYCRAFTER_GIT_MIRROR" in os.environ:
        print(f"使用 GitHub 镜像: {os.environ['FOLEYCRAFTER_GIT_MIRROR']}")
    else:
        print("提示：国内用户可设置 FOLEYCRAFTER_GIT_MIRROR=https://ghproxy.com 加速 git clone")
    print()

    # ===== 步骤 0：安装 foleycrafter Python 包到 .venv（v0.2.12+）=====
    if args.skip_package:
        print(f"[0/4] 跳过 foleycrafter Python 包安装（--skip-package）")
    else:
        print(f"[0/4] 安装 foleycrafter Python 包到 .venv site-packages")
        try:
            _install_foleycrafter_package(force=args.force)
        except Exception as e:
            print(f"[失败] foleycrafter Python 包安装失败: {e}")
            print(f"       引擎启动时也会自动尝试安装，可忽略此错误继续下载模型")
    print()

    # ===== 步骤 1-3：下载 HuggingFace 模型权重 =====
    for i, repo in enumerate(variant_info["repos"], 1):
        repo_id = repo["repo_id"]
        target_subdir = repo["target_subdir"]
        target_dir = target_root / target_subdir if target_subdir != "." else target_root

        # 跳过 Auffusion（用户指定 --skip-auffusion）
        if args.skip_auffusion and "auffusion" in repo_id:
            print(f"[{i}/{len(variant_info['repos'])}] 跳过 {repo_id}（--skip-auffusion）")
            continue

        print(f"[{i}/{len(variant_info['repos'])}] {repo['description']}")
        print(f"  repo_id: {repo_id}")

        _download_repo(
            repo_id=repo_id,
            target_dir=target_dir,
            allow_patterns=repo.get("allow_patterns"),
            force=args.force,
        )

        # 平移子目录（如 h94/IP-Adapter 的 models/image_encoder/）
        if "flatten_subdir" in repo:
            _flatten_subdir(target_dir, repo["flatten_subdir"])

        print()

    # 验证下载结果
    print(f"===== 验证 =====")
    required_files = [
        "temporal_adapter.ckpt",
        "semantic/semantic_adapter.bin",
        "timestamp_detector.pth.tar",
        "auffusion/model_index.json",
        "ip-adapter/config.json",
    ]
    all_ok = True
    for rel in required_files:
        full = target_root / rel
        exists = full.exists()
        status = "✓" if exists else "✗"
        print(f"  {status} {rel}")
        if not exists:
            all_ok = False

    # 验证 foleycrafter Python 包
    if not args.skip_package:
        try:
            import importlib.util
            pkg_ok = importlib.util.find_spec("foleycrafter") is not None
        except Exception:
            pkg_ok = False
        status = "✓" if pkg_ok else "✗"
        print(f"  {status} foleycrafter Python 包（site-packages）")
        if not pkg_ok:
            all_ok = False

    print()
    if all_ok:
        print(f"[成功] 所有必需文件已就绪: {target_root}")
        print(f"")
        print(f"现在可以启动引擎：")
        print(f"  uv run --no-sync --project envs/audio-foleycrafter \\")
        print(f"      python -m engines.foleycrafter_a3b2c1.server --port 7004")
        return 0
    else:
        print(f"[失败] 部分文件缺失，请检查网络或重新运行（可加 --force）")
        return 1


if __name__ == "__main__":
    sys.exit(main())
