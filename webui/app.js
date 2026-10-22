/* ============================================================
   FoleyCrafter V2A WebUI 逻辑（原生 JS，直连引擎 API）
   三区联动：设置区（模式/变体/加载） ↔ 参数区（视频/提示词/高级） ↔ 结果区（SSE + 播放器）
   ============================================================ */
"use strict";

const API = "";  // 同源直连（引擎自身挂载 /ui）

// ---------- 状态 ----------
const state = {
  mode: "v2a",
  video: null,       // { name, path, localUrl }（path 为引擎侧绝对路径）
  variants: [],      // capabilities.variants
  generating: false,
  loading: false,
};

// ---------- DOM ----------
const $ = (id) => document.getElementById(id);

const MODE_HINTS = {
  v2a: "视频配音（V2A）：上传视频 → 分析画面语义与时间戳 → 生成同步音效",
  t2a: "文本配音（T2A）：仅凭文字描述生成音效（无需视频，如环境声、拟音）",
};

// SSE 阶段 → 中文（stage 定义见 progress_reporter.py ProgressStage）
const STAGE_TEXT = {
  "v2a.start": "开始推理",
  "v2a.video_decoding": "视频解码",
  "v2a.timestamp_detect": "时间戳检测",
  "v2a.image_encoding": "视觉编码",
  "v2a.audio_generating": "扩散生成音频",
  "v2a.saving": "保存产物",
  "completed": "完成",
};

// ============================================================
// 初始化
// ============================================================
document.addEventListener("DOMContentLoaded", () => {
  bindModeSeg();
  bindSliders();
  bindUpload();
  bindButtons();
  refreshStatus();
  refreshCapabilities();
  setInterval(refreshStatus, 15000);  // 状态轮询（空闲时）
});

// ============================================================
// 设置区：模式切换
// ============================================================
function bindModeSeg() {
  document.querySelectorAll(".seg-btn").forEach((btn) => {
    btn.addEventListener("click", () => {
      document.querySelectorAll(".seg-btn").forEach((b) => b.classList.remove("active"));
      btn.classList.add("active");
      state.mode = btn.dataset.mode;
      $("mode-hint").textContent = MODE_HINTS[state.mode];
      $("video-field").style.display = state.mode === "v2a" ? "" : "none";
      $("footer-mode").textContent = `模式：${btn.textContent}`;
    });
  });
  $("footer-mode").textContent = "模式：视频配音";
}

// ============================================================
// 参数区
// ============================================================
function bindSliders() {
  $("steps").addEventListener("input", (e) => { $("steps-val").textContent = e.target.value; });
  $("temporal-scale").addEventListener("input", (e) => { $("temporal-scale-val").textContent = e.target.value; });
  $("semantic-scale").addEventListener("input", (e) => { $("semantic-scale-val").textContent = e.target.value; });
}

// 视频上传（点击 + 拖拽；选择后立即上传引擎拿 path）
function bindUpload() {
  const dz = $("dropzone");
  const fi = $("file-input");
  dz.addEventListener("click", () => fi.click());
  dz.addEventListener("dragover", (e) => { e.preventDefault(); dz.classList.add("dragover"); });
  dz.addEventListener("dragleave", () => dz.classList.remove("dragover"));
  dz.addEventListener("drop", (e) => {
    e.preventDefault();
    dz.classList.remove("dragover");
    uploadVideo(e.dataTransfer.files[0]);
  });
  fi.addEventListener("change", () => uploadVideo(fi.files[0]));
}

async function uploadVideo(file) {
  if (!file) return;
  if (state.video) {
    showError("已有视频，请先移除后再上传");
    return;
  }
  // 先本地预览
  const localUrl = URL.createObjectURL(file);
  state.video = { name: file.name, path: "", localUrl };
  renderFileChip("上传中…");
  try {
    const fd = new FormData();
    fd.append("file", file);
    const r = await fetch(`${API}/upload`, { method: "POST", body: fd });
    if (!r.ok) {
      const e = await r.json().catch(() => ({}));
      throw new Error(e.detail || `HTTP ${r.status}`);
    }
    const data = await r.json();
    state.video.path = data.path;
    renderFileChip(fmtSize(data.size ?? file.size));
  } catch (err) {
    removeVideo();
    showError(`视频上传失败：${err.message}`);
  }
}

function renderFileChip(sizeText) {
  const box = $("file-chip-box");
  box.innerHTML = `
    <span class="file-chip">
      <span class="fc-name" title="${state.video.name}">🎬 ${state.video.name}</span>
      <span style="color:#1a7f37">${sizeText}</span>
      <button class="fc-rm" title="移除视频">×</button>
    </span>`;
  box.querySelector(".fc-rm").addEventListener("click", removeVideo);
  $("upload-preview").innerHTML =
    `<video src="${state.video.localUrl}" controls muted></video>`;
}

function removeVideo() {
  if (state.video?.localUrl) URL.revokeObjectURL(state.video.localUrl);
  state.video = null;
  $("file-chip-box").innerHTML = "";
  $("upload-preview").innerHTML = "";
  $("file-input").value = "";
}

function fmtSize(bytes) {
  if (bytes >= 1024 * 1024) return (bytes / 1024 / 1024).toFixed(1) + " MB";
  return Math.round(bytes / 1024) + " KB";
}

// ============================================================
// 设置区：变体 / 加载 / 卸载
// ============================================================
async function refreshCapabilities() {
  try {
    const r = await fetch(`${API}/capabilities`);
    const cap = await r.json();
    state.variants = cap.variants || [];
    renderVariantSelect();
  } catch {
    $("mode-hint").textContent = "capabilities 获取失败（引擎未启动？）";
  }
}

function renderVariantSelect() {
  const sel = $("variant-select");
  const cur = sel.value;
  sel.innerHTML = `<option value="">自动（默认变体）</option>`;
  for (const v of state.variants) {
    const opt = document.createElement("option");
    opt.value = v.name;
    const tag = v.available ? "" : "（未下载）";
    opt.textContent = `${v.name} · ${v.precision} · ${v.vram_required}${tag}`;
    if (!v.available) opt.disabled = true;
    sel.appendChild(opt);
  }
  if (cur && [...sel.options].some((o) => o.value === cur)) sel.value = cur;
}

function bindButtons() {
  $("btn-refresh-status").addEventListener("click", refreshStatus);
  $("btn-load").addEventListener("click", loadModel);
  $("btn-unload").addEventListener("click", unloadModel);
  $("btn-generate").addEventListener("click", generate);
}

async function loadModel() {
  if (state.loading) return;
  state.loading = true;
  $("btn-load").disabled = true;
  $("btn-load").textContent = "加载中…";
  try {
    const variant = $("variant-select").value || null;
    const r = await fetch(`${API}/load`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ variant }),
    });
    if (!r.ok) {
      const e = await r.json().catch(() => ({}));
      throw new Error(e.detail || `HTTP ${r.status}`);
    }
  } catch (err) {
    showError(`模型加载失败：${err.message}`);
  } finally {
    state.loading = false;
    $("btn-load").disabled = false;
    $("btn-load").textContent = "加载模型";
    refreshStatus();
  }
}

async function unloadModel() {
  try {
    const r = await fetch(`${API}/unload`, { method: "POST" });
    if (!r.ok) {
      const e = await r.json().catch(() => ({}));
      throw new Error(e.detail || `HTTP ${r.status}`);
    }
  } catch (err) {
    showError(`模型卸载失败：${err.message}`);
  } finally {
    refreshStatus();
  }
}

// ============================================================
// 状态轮询
// ============================================================
async function refreshStatus() {
  if (state.generating) return;
  try {
    const r = await fetch(`${API}/status`);
    const st = await r.json();
    const badge = $("state-badge");
    badge.className = "badge " + (st.state || "");
    badge.textContent = {
      idle: "空闲", loading: "加载中", ready: "就绪",
      inferring: "推理中", error: "异常",
    }[st.state] || st.state;
    const dev = st.extra?.device_name || st.extra?.device || "—";
    $("state-detail").textContent =
      `变体 ${st.loaded_variant || "未加载"} · 显存 ${st.vram_used || "—"} · ${dev}`;
  } catch {
    $("state-badge").className = "badge";
    $("state-badge").textContent = "未连接";
    $("state-detail").textContent = "引擎不可达（127.0.0.1:7004）";
  }
}

// ============================================================
// 结果区：生成（SSE 流）
// ============================================================
async function generate() {
  if (state.generating) return;

  // 校验
  const prompt = $("prompt").value.trim();
  if (state.mode === "v2a" && !state.video?.path) { showError("视频配音模式需要先上传视频"); return; }
  if (state.mode === "t2a" && !prompt) { showError("文本配音模式需要输入提示词"); return; }

  const inputs = {
    prompt,
    negative_prompt: $("negative-prompt").value.trim(),
    seed: parseInt($("seed").value, 10) || 42,
    temporal_align: $("temporal-align").checked,
    temporal_scale: parseFloat($("temporal-scale").value),
    semantic_scale: parseFloat($("semantic-scale").value),
    num_inference_steps: parseInt($("steps").value, 10),
    max_video_duration: parseFloat($("max-duration").value) || 10.0,
    output_format: $("output-format").value,
  };

  // 不携带 task_id：直连链路契约（engine.py 以 request.task_id 是否非空区分
  // gateway 转发/直连）。引擎会自动生成 fc_ 前缀 task_id 上报任务中心。
  const body = { inputs };
  const variant = $("variant-select").value;
  if (variant) body.variant = variant;
  if (state.mode === "v2a") {
    body.audios = [{ path: state.video.path, role: "video" }];
  }

  // UI 状态
  state.generating = true;
  $("btn-generate").disabled = true;
  $("btn-generate").textContent = "生成中…";
  $("progress-wrap").style.display = "flex";
  $("progress-fill").style.width = "0%";
  $("progress-text").textContent = "连接引擎…";
  $("error-box").style.display = "none";
  $("state-badge").className = "badge inferring";
  $("state-badge").textContent = "推理中";

  try {
    const resp = await fetch(`${API}/infer`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });
    if (!resp.ok) {
      const e = await resp.json().catch(() => ({}));
      throw new Error(e.detail || `HTTP ${resp.status}`);
    }

    // 解析 SSE 流（fetch ReadableStream 手动读 data: 行）
    const reader = resp.body.getReader();
    const decoder = new TextDecoder();
    let buf = "";
    let finalResult = null;

    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      buf += decoder.decode(value, { stream: true });
      const lines = buf.split("\n");
      buf = lines.pop();  // 末行可能不完整，留在缓冲
      for (const line of lines) {
        if (!line.startsWith("data:")) continue;
        let frame;
        try { frame = JSON.parse(line.slice(5)); } catch { continue; }
        handleFrame(frame);
        if (frame.stage === "completed" && frame.result) finalResult = frame.result;
      }
    }

    if (finalResult) {
      showResult(finalResult);
    } else {
      throw new Error("SSE 流结束但未收到结果帧");
    }
  } catch (err) {
    showError(`生成失败：${err.message}`);
  } finally {
    state.generating = false;
    $("btn-generate").disabled = false;
    $("btn-generate").textContent = "开始生成";
    $("progress-wrap").style.display = "none";
    refreshStatus();
  }
}

function handleFrame(f) {
  if (f.stage === "error") {
    throw new Error(f.error?.message || "引擎错误");
  }
  const pct = Math.round((f.progress || 0) * 100);
  $("progress-fill").style.width = `${pct}%`;
  const stageText = STAGE_TEXT[f.stage] || f.stage;
  $("progress-text").textContent = `${stageText} ${pct}%${f.detail ? " · " + f.detail : ""}`;
}

function showResult(result) {
  const view = $("result-view");
  const t = Date.now();  // 防缓存
  const audioUrl = result.audio_url + "?t=" + t;
  // result: {audio_url, duration, sample_rate, mode, video_url?, video_path?}
  let html = `<div class="media-stack">`;
  if (result.video_url) {
    html += `
      <div class="media-block">
        <span class="media-label">合成视频（画面 + 生成的音轨）</span>
        <video src="${result.video_url}?t=${t}" controls></video>
      </div>`;
  }
  html += `
      <div class="media-block">
        <span class="media-label">生成的音频</span>
        <audio src="${audioUrl}" controls></audio>
      </div>
    </div>`;
  view.innerHTML = html;

  const modeText = result.mode === "v2a" ? "视频配音" : "文本配音";
  const meta = $("result-meta");
  meta.innerHTML = `
    <span class="meta-chip">模式 ${modeText}</span>
    <span class="meta-chip">音频 ${result.duration?.toFixed(1) ?? "—"}s</span>
    <span class="meta-chip">采样率 ${result.sample_rate ?? "—"}Hz</span>
    <span class="meta-chip">seed ${$("seed").value}</span>
    <span class="meta-chip"><a href="${audioUrl}" download>下载音频</a></span>
    ${result.video_url ? `<span class="meta-chip"><a href="${result.video_url}?t=${t}" download>下载视频</a></span>` : ""}`;
}

// ---------- 错误提示（不自动消失，手动关闭） ----------
function showError(msg) {
  $("error-text").textContent = msg;
  $("error-box").style.display = "block";
}
