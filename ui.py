"""FoleyCrafter V2A 引擎 Gradio Web UI。

页面元素：
    - 状态栏（引擎状态 / 已加载模型 / 显存）
    - 模型选择 + 加载/卸载/刷新按钮
    - 双模式 Tab（V2A 视频转音效 / T2A 文本转音效）
    - [V2A] 视频上传 + prompt
    - [T2A] prompt（描述要生成的音效）
    - 高级参数（temporal_scale / semantic_scale / steps / seed / max_duration）
    - 生成按钮 + 进度状态
    - 音频播放器 + 视频播放器 + 下载
    - 日志显示

设计原则：
    - 调用自身 API（不直接调 engine.py），保证 UI 和 API 行为一致
    - SSE 流式消费进度帧（v2a.start → v2a.completed）
    - 自动加载逻辑（未加载时自动 /load）
    - 支持 iframe 嵌入（root_path 由 server.py 配置）
"""

from __future__ import annotations

import json
import logging
from typing import Optional

import gradio as gr
import httpx

logger = logging.getLogger(__name__)


def create_ui(engine_url: str, theme: Optional[gr.Theme] = None) -> gr.Blocks:
    """创建 FoleyCrafter V2A 引擎 Web UI。

    Args:
        engine_url: 引擎 HTTP 地址（如 http://127.0.0.1:7004）
        theme: Gradio 主题（Gradio 6.0+ 不再传给 Blocks，由 mount_gradio_app 应用）

    Returns:
        gr.Blocks: UI 实例
    """
    with gr.Blocks(title="FoleyCrafter V2A 音效生成") as demo:
        gr.Markdown("# FoleyCrafter 视频转音效引擎")
        gr.Markdown("Video-to-Audio (V2A) / Text-to-Audio (T2A)")

        # ===== 状态栏 =====
        with gr.Row():
            status_display = gr.Textbox(
                label="引擎状态",
                interactive=False,
                value="未连接",
                scale=2,
            )
            variant_display = gr.Textbox(
                label="已加载模型",
                interactive=False,
                value="-",
                scale=1,
            )
            vram_display = gr.Textbox(
                label="显存占用",
                interactive=False,
                value="-",
                scale=1,
            )

        # ===== 模型选择 + 操作按钮 =====
        with gr.Row():
            variant_select = gr.Dropdown(
                label="模型变体（Variant）",
                choices=[],
                value=None,
                interactive=True,
                info="选择要加载的 FoleyCrafter 模型",
                scale=2,
            )
            refresh_btn = gr.Button("刷新状态", variant="secondary", scale=1)
            load_btn = gr.Button("加载模型", variant="primary", scale=1)
            unload_btn = gr.Button("卸载模型", variant="stop", scale=1)

        # ===== 双模式 Tab =====
        with gr.Tabs() as tabs:
            # ----- Tab 1: V2A 视频转音效 -----
            with gr.Tab("V2A 视频转音效") as tab_v2a:
                gr.Markdown(
                    "**视频转音效**：上传视频，根据视频内容生成匹配的音效，"
                    "输出 wav 音频 + mp4 带音轨视频"
                )
                v2a_video_input = gr.Video(
                    label="输入视频（无声或已有音轨将被替换）",
                    sources=["upload"],
                )
                v2a_prompt = gr.Textbox(
                    label="文本提示（可选，空时仅靠视频语义）",
                    placeholder="如：枪声、雨声、汽车引擎...",
                    lines=2,
                )
                v2a_negative_prompt = gr.Textbox(
                    label="负向提示（可选，控制不希望出现的音效）",
                    placeholder="如：噪声、回声...",
                    lines=1,
                )

            # ----- Tab 2: T2A 文本转音效 -----
            with gr.Tab("T2A 文本转音效") as tab_t2a:
                gr.Markdown(
                    "**文本转音效**：仅根据文本描述生成音效，输出 wav 音频"
                )
                t2a_prompt = gr.Textbox(
                    label="文本提示",
                    placeholder="如：一声雷鸣、紧凑的鼓点、流水声...",
                    lines=3,
                )
                t2a_negative_prompt = gr.Textbox(
                    label="负向提示（可选）",
                    placeholder="如：噪声、人声...",
                    lines=1,
                )

        # ===== 高级参数 =====
        with gr.Accordion("高级参数", open=False):
            with gr.Row():
                temporal_scale_slider = gr.Slider(
                    label="时间对齐强度（temporal_scale）",
                    minimum=0.0,
                    maximum=1.0,
                    value=0.2,
                    step=0.05,
                    info="V2A 模式专用：视频时间戳对齐强度，越高越贴合视频节奏",
                )
                semantic_scale_slider = gr.Slider(
                    label="语义强度（semantic_scale）",
                    minimum=0.0,
                    maximum=2.0,
                    value=1.0,
                    step=0.05,
                    info="IP-Adapter 视觉语义强度，越高越贴合视频内容",
                )
                steps_slider = gr.Slider(
                    label="扩散推理步数",
                    minimum=1,
                    maximum=100,
                    value=25,
                    step=1,
                    info="越多质量越好但越慢",
                )
            with gr.Row():
                seed_input = gr.Number(
                    label="随机种子",
                    value=42,
                    info="固定种子可复现结果",
                )
                max_duration_input = gr.Number(
                    label="视频长度限制（秒）",
                    value=10.0,
                    minimum=1.0,
                    maximum=60.0,
                    step=1.0,
                    info="V2A 模式：超过此长度则截断；T2A 模式：生成音频时长",
                )
                format_select = gr.Dropdown(
                    label="音频输出格式",
                    choices=["wav", "mp3", "flac"],
                    value="wav",
                )

        # ===== 生成按钮 + 进度 =====
        with gr.Row():
            generate_btn = gr.Button("生成音效", variant="primary", scale=2)
            progress_display = gr.Textbox(
                label="进度",
                interactive=False,
                value="就绪",
                scale=3,
            )

        # ===== 结果区 =====
        with gr.Row():
            audio_output = gr.Audio(
                label="生成的音频",
                interactive=False,
                scale=1,
            )
            video_output = gr.Video(
                label="生成的带音轨视频（仅 V2A 模式）",
                interactive=False,
                scale=1,
            )
        with gr.Row():
            audio_download = gr.Textbox(
                label="音频下载链接",
                interactive=False,
                value="-",
                scale=1,
            )
            video_download = gr.Textbox(
                label="视频下载链接",
                interactive=False,
                value="-",
                scale=1,
            )

        # ===== 日志 =====
        log_display = gr.Textbox(
            label="日志",
            interactive=False,
            lines=4,
            max_lines=8,
        )

        # ===== 事件绑定 =====

        # 页面加载 + 刷新按钮
        def _on_refresh():
            return refresh_all(engine_url)

        demo.load(
            fn=_on_refresh,
            outputs=[status_display, variant_display, vram_display, variant_select],
        )
        refresh_btn.click(
            fn=_on_refresh,
            outputs=[status_display, variant_display, vram_display, variant_select],
        )

        # 加载按钮
        def _on_load(variant):
            return load_variant(engine_url, variant)

        load_btn.click(
            fn=_on_load,
            inputs=[variant_select],
            outputs=[status_display, variant_display, vram_display, log_display],
        )

        # 卸载按钮
        def _on_unload():
            return unload_variant(engine_url)

        unload_btn.click(
            fn=_on_unload,
            outputs=[status_display, variant_display, vram_display, log_display],
        )

        # 生成按钮（根据当前 Tab 决定参数）
        current_tab = gr.State("v2a")

        # Tab 切换时更新 current_tab 状态
        tab_v2a.select(fn=lambda: "v2a", outputs=current_tab)
        tab_t2a.select(fn=lambda: "t2a", outputs=current_tab)

        # 生成函数（生成器，流式更新进度）
        def _on_generate(
            tab, variant,
            v2a_video, v2a_prompt, v2a_negative,
            t2a_prompt, t2a_negative,
            temporal_scale, semantic_scale, steps, seed, max_duration, fmt,
        ):
            yield from generate_audio(
                engine_url, tab, variant,
                v2a_video, v2a_prompt, v2a_negative,
                t2a_prompt, t2a_negative,
                temporal_scale, semantic_scale, steps, seed, max_duration, fmt,
            )

        generate_btn.click(
            fn=_on_generate,
            inputs=[
                current_tab, variant_select,
                v2a_video_input, v2a_prompt, v2a_negative_prompt,
                t2a_prompt, t2a_negative_prompt,
                temporal_scale_slider, semantic_scale_slider, steps_slider,
                seed_input, max_duration_input, format_select,
            ],
            outputs=[
                progress_display, audio_output, video_output,
                audio_download, video_download, log_display,
            ],
        )

    return demo


# ===== API 调用函数 =====

def refresh_all(engine_url: str) -> tuple:
    """刷新状态、variant 列表。"""
    try:
        status = _http_get(engine_url, "/status")
        caps = _http_get(engine_url, "/capabilities")

        status_text = f"state={status.get('state', '?')}"
        extra = status.get("extra", {})
        device_display = extra.get("device_name") or extra.get("device")
        if device_display:
            status_text += f" | device={device_display}"
        if extra.get("is_loaded"):
            status_text += " | 已加载"

        variant_display = status.get("loaded_variant") or "-"
        vram_display = status.get("vram_used", "-")

        # variant 列表（显示 name + V2A/T2A 支持）
        variants = []
        for v in caps.get("variants", []):
            name = v.get("name", "")
            supports_v2a = v.get("supports_v2a", False)
            supports_t2a = v.get("supports_t2a", False)
            tags = []
            if supports_v2a:
                tags.append("V2A")
            if supports_t2a:
                tags.append("T2A")
            tag_str = "/".join(tags) if tags else "?"
            label = f"{name} ({tag_str})"
            variants.append(label)

        # 默认选第一个
        default = variants[0].split(" ")[0] if variants else None

        return status_text, variant_display, vram_display, gr.update(choices=variants, value=default)
    except Exception as e:
        logger.error("刷新失败: %s", e)
        return f"连接失败: {e}", "-", "-", gr.update(choices=[])


def load_variant(engine_url: str, variant_label: str) -> tuple:
    """加载 variant。"""
    if not variant_label:
        return "请先选择 variant", "-", "-", "未选择 variant"

    variant = variant_label.split(" ")[0]

    try:
        response = _http_post(engine_url, "/load", {"variant": variant}, timeout=600.0)
        status_text = f"state={response.get('state', '?')}"
        variant_display = response.get("variant", variant)
        vram = response.get("vram_used", "-")
        load_time = response.get("load_time", 0)
        log = f"加载 {variant} 完成，耗时 {load_time:.2f}s，显存 {vram}"
        return status_text, variant_display, vram, log
    except Exception as e:
        logger.error("加载失败: %s", e)
        return f"加载失败: {e}", "-", "-", f"加载失败: {e}"


def unload_variant(engine_url: str) -> tuple:
    """卸载模型。"""
    try:
        response = _http_post(engine_url, "/unload", {}, timeout=60.0)
        return "state=idle", "-", response.get("vram_freed", "0MB"), "卸载完成"
    except Exception as e:
        return f"卸载失败: {e}", "-", "-", f"卸载失败: {e}"


def generate_audio(
    engine_url: str,
    tab: str,
    variant_label: str,
    v2a_video: Optional[str],
    v2a_prompt: str,
    v2a_negative: str,
    t2a_prompt: str,
    t2a_negative: str,
    temporal_scale: float,
    semantic_scale: float,
    steps: int,
    seed: int,
    max_duration: float,
    fmt: str,
):
    """流式 V2A / T2A 生成。

    自动加载逻辑：
        1. 发送前检查 /status
        2. 如果未加载 → 自动 /load
        3. 加载完成后执行 /infer（SSE 流式）
    """
    log_lines: list[str] = []

    # ----- 1. 根据 Tab 解析参数 -----
    variant = (variant_label or "").split(" ")[0]
    if not variant:
        yield "请先选择模型", None, None, "-", "-", "未选择 variant"
        return

    is_v2a = (tab == "v2a")

    if is_v2a:
        # V2A 模式：必须有视频
        if not v2a_video:
            yield "请上传视频", None, None, "-", "-", "视频为空"
            return
        prompt = v2a_prompt or ""
        negative_prompt = v2a_negative or ""
        video_path = v2a_video
        # audios 列表传视频（role=video，与 engine._infer_impl 约定一致）
        audios = [{"path": video_path, "role": "video"}]
        log_lines.append(f"V2A 模式：video={video_path}, prompt={prompt!r}")
    else:
        # T2A 模式：必须有 prompt
        prompt = t2a_prompt or ""
        if not prompt.strip():
            yield "请输入文本提示", None, None, "-", "-", "文本提示为空"
            return
        negative_prompt = t2a_negative or ""
        audios = []
        log_lines.append(f"T2A 模式：prompt={prompt!r}")

    # ----- 2. 自动加载 -----
    try:
        status = _http_get(engine_url, "/status")
    except Exception as e:
        yield "获取状态失败", None, None, "-", "-", f"获取状态失败: {e}"
        return

    current_state = status.get("state", "unknown")
    raw_loaded = status.get("loaded_variant") or ""
    loaded_variant = raw_loaded.replace("\\", "/").rstrip("/").split("/")[-1].split(" ")[0]

    if current_state != "ready" or not loaded_variant:
        log_lines.append(f"模型未加载（state={current_state}），自动加载 {variant}")
        yield "正在加载模型...", None, None, "-", "-", "\n".join(log_lines)
        try:
            _http_post(engine_url, "/load", {"variant": variant}, timeout=600.0)
            log_lines.append("加载完成")
            yield "模型加载完成，开始推理...", None, None, "-", "-", "\n".join(log_lines)
        except Exception as e:
            yield "加载失败", None, None, "-", "-", f"自动加载失败: {e}"
            return
    elif loaded_variant != variant:
        log_lines.append(f"切换模型: {loaded_variant} → {variant}")
        yield "正在切换模型...", None, None, "-", "-", "\n".join(log_lines)
        try:
            _http_post(engine_url, "/unload", {}, timeout=60.0)
            _http_post(engine_url, "/load", {"variant": variant}, timeout=600.0)
            log_lines.append("切换完成")
            yield "模型切换完成，开始推理...", None, None, "-", "-", "\n".join(log_lines)
        except Exception as e:
            yield "切换失败", None, None, "-", "-", f"切换失败: {e}"
            return
    else:
        log_lines.append(f"模型已加载（variant={loaded_variant}），直接推理")
        yield "模型已加载，开始推理...", None, None, "-", "-", "\n".join(log_lines)

    # ----- 3. 构造推理请求 -----
    request_body = {
        "variant": variant,
        "inputs": {
            "prompt": prompt,
            "negative_prompt": negative_prompt,
            "seed": int(seed) if seed is not None else 42,
            "temporal_align": True,
            "temporal_scale": float(temporal_scale),
            "semantic_scale": float(semantic_scale),
            "num_inference_steps": int(steps),
            "max_video_duration": float(max_duration) if max_duration else 10.0,
            "output_format": fmt,
        },
        "audios": audios,
    }

    # ----- 4. SSE 流式推理 -----
    try:
        with httpx.Client(timeout=1800.0) as client:  # V2A 推理可能较慢
            with client.stream(
                "POST",
                f"{engine_url}/infer",
                json=request_body,
                headers={"Accept": "text/event-stream"},
            ) as response:
                response.raise_for_status()

                for line in response.iter_lines():
                    if not line or not line.startswith("data: "):
                        continue
                    data_str = line[6:]
                    try:
                        frame = json.loads(data_str)
                    except Exception:
                        continue

                    stage = frame.get("stage", "")
                    progress = frame.get("progress", 0.0)

                    # V2A 阶段进度
                    if stage == "v2a.start":
                        msg = f"开始推理（{progress:.0%}）"
                        yield msg, None, None, "-", "-", "\n".join(log_lines)

                    elif stage == "v2a.video_decoding":
                        frames_n = frame.get("frames", 0)
                        duration = frame.get("duration", 0)
                        msg = f"解码视频帧（{progress:.0%}）frames={frames_n}, duration={duration:.1f}s"
                        yield msg, None, None, "-", "-", "\n".join(log_lines)

                    elif stage == "v2a.timestamp_detect":
                        msg = f"检测视频时间戳（{progress:.0%}）"
                        yield msg, None, None, "-", "-", "\n".join(log_lines)

                    elif stage == "v2a.image_encoding":
                        msg = f"编码视频视觉特征（{progress:.0%}）"
                        yield msg, None, None, "-", "-", "\n".join(log_lines)

                    elif stage == "v2a.audio_generating":
                        step_n = frame.get("step", 0)
                        total_steps = frame.get("total_steps", 25)
                        msg = f"生成音频中... {step_n}/{total_steps}（{progress:.0%}）"
                        yield msg, None, None, "-", "-", "\n".join(log_lines)

                    elif stage == "v2a.saving":
                        msg = f"保存文件中（{progress:.0%}）"
                        yield msg, None, None, "-", "-", "\n".join(log_lines)

                    elif stage == "completed":
                        result = frame.get("result", {})
                        timing = frame.get("timing", {})
                        audio_url = result.get("audio_url", "")
                        audio_path = result.get("audio_path", "")
                        video_url = result.get("video_url", "")
                        video_path = result.get("video_path", "")
                        duration = result.get("duration", 0)
                        mode = result.get("mode", "")

                        # 拼完整 URL（用于下载链接文本框，可点击下载）
                        full_audio_url = f"{engine_url}{audio_url}" if audio_url else ""
                        full_video_url = f"{engine_url}{video_url}" if video_url else ""

                        # gr.Audio / gr.Video 使用本地绝对路径播放，
                        # 绕过 Gradio safehttpx SSRF 防护对 127.0.0.1 的拦截
                        gradio_audio_value = audio_path if audio_path else full_audio_url
                        gradio_video_value = video_path if video_path else full_video_url

                        # v0.2.14.8+ 修复 T2A 模式下 gr.Video 收到空字符串导致 PermissionError:
                        #   Gradio postprocess 把空字符串当文件路径，解析成 cwd（项目根目录），
                        #   open() 目录 → PermissionError: [Errno 13] Permission denied
                        #   修复：空字符串转为 None，让 Gradio 跳过文件处理
                        if not gradio_audio_value:
                            gradio_audio_value = None
                        if not gradio_video_value:
                            gradio_video_value = None
                        # 下载链接文本框：空字符串显示为 "-"（保持与 error 分支一致）
                        audio_download_display = full_audio_url if full_audio_url else "-"
                        video_download_display = full_video_url if full_video_url else "-"

                        log_lines.append(
                            f"完成: 模式={mode}, 时长 {duration:.2f}s, "
                            f"总耗时 {timing.get('total_seconds', 0):.2f}s"
                        )
                        yield "完成", gradio_audio_value, gradio_video_value, audio_download_display, video_download_display, "\n".join(log_lines)
                        return

                    elif stage == "error":
                        err = frame.get("error", {})
                        msg = f"错误: {err.get('message', '?')}"
                        log_lines.append(msg)
                        yield msg, None, None, "-", "-", "\n".join(log_lines)
                        return

    except Exception as e:
        logger.exception("FoleyCrafter 推理失败")
        yield f"推理异常: {e}", None, None, "-", "-", f"推理异常: {e}"


# ===== HTTP 工具 =====

def _http_get(base_url: str, path: str) -> dict:
    """GET 请求。"""
    with httpx.Client(timeout=10.0) as client:
        response = client.get(f"{base_url}{path}")
        response.raise_for_status()
        return response.json()


def _http_post(base_url: str, path: str, json_body: dict, timeout: float = 600.0) -> dict:
    """POST 请求。"""
    with httpx.Client(timeout=timeout) as client:
        response = client.post(f"{base_url}{path}", json=json_body)
        response.raise_for_status()
        return response.json()
