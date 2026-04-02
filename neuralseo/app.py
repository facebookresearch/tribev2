"""
NeuralSEO — The only SEO tool that tells you how the human brain
actually responds to your content.

Powered by Meta AI's TRIBE v2 (facebook/tribev2).
"""
import os
import logging
import urllib.request
import tempfile
import uuid
import time
from pathlib import Path
import gradio as gr
import plotly.graph_objects as go
from PIL import Image, ImageDraw
from dotenv import load_dotenv

load_dotenv()
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)

_BRAIN_ASSET_DOWNLOADS = {
    "head.glb": "https://aidemos.atmeta.com/tribev2/assets/head-9ddb57ac.glb",
    "brain-left-hemisphere.glb": "https://aidemos.atmeta.com/tribev2/assets/brain-left-hemisphere-1b9f386f.glb",
    "brain-right-hemisphere.glb": "https://aidemos.atmeta.com/tribev2/assets/brain-right-hemisphere-f0dea562.glb",
    "dynamic/naturalistic/vanessen-2023-timeline-0-start-750/vanessen-2023-timeline-0-start-750-left-hemisphere-face-colors-binary-prediction.json":
        "https://aidemos.atmeta.com/tribev2/data/dynamic/naturalistic/vanessen-2023-timeline-0-start-750/vanessen-2023-timeline-0-start-750-left-hemisphere-face-colors-binary-prediction.json",
    "dynamic/naturalistic/vanessen-2023-timeline-0-start-750/vanessen-2023-timeline-0-start-750-right-hemisphere-face-colors-binary-prediction.json":
        "https://aidemos.atmeta.com/tribev2/data/dynamic/naturalistic/vanessen-2023-timeline-0-start-750/vanessen-2023-timeline-0-start-750-right-hemisphere-face-colors-binary-prediction.json",
    "dynamic/naturalistic/vanessen-2023-timeline-0-start-750/vanessen-2023-timeline-0-start-750-left-hemisphere-face-colors-prediction.uint8rgb.bin":
        "https://aidemos.atmeta.com/tribev2/data/dynamic/naturalistic/vanessen-2023-timeline-0-start-750/vanessen-2023-timeline-0-start-750-left-hemisphere-face-colors-prediction.uint8rgb.bin",
    "dynamic/naturalistic/vanessen-2023-timeline-0-start-750/vanessen-2023-timeline-0-start-750-right-hemisphere-face-colors-prediction.uint8rgb.bin":
        "https://aidemos.atmeta.com/tribev2/data/dynamic/naturalistic/vanessen-2023-timeline-0-start-750/vanessen-2023-timeline-0-start-750-right-hemisphere-face-colors-prediction.uint8rgb.bin",
}


def _ensure_brain_demo_assets() -> None:
    """Download brain mesh + activation assets at runtime if missing."""
    base = Path("assets/brain-demo")
    base.mkdir(parents=True, exist_ok=True)

    for rel_path, url in _BRAIN_ASSET_DOWNLOADS.items():
        target = base / rel_path
        if target.exists():
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        logger.info("Downloading brain demo asset: %s", rel_path)
        urllib.request.urlretrieve(url, target)


def _write_brain_viewer_html() -> None:
    """Write a self-contained static brain viewer page (robust first-load behavior)."""
    viewer_html = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>TRIBE v2 Brain Viewer</title>
  <style>
    html, body {
      margin: 0;
      padding: 0;
      width: 100%;
      height: 100%;
      background: #090513;
      color: #e2e8f0;
      font-family: system-ui, -apple-system, Segoe UI, sans-serif;
      overflow: hidden;
    }
    #app {
      position: relative;
      width: 100%;
      height: 100%;
    }
    #view {
      position: absolute;
      inset: 0;
      background: radial-gradient(circle at 55% 35%, rgba(39, 18, 89, 0.45), rgba(7, 4, 14, 0.95));
    }
    #hud {
      position: absolute;
      left: 10px;
      right: 10px;
      top: 10px;
      display: none;
      justify-content: space-between;
      align-items: center;
      gap: 8px;
      pointer-events: none;
    }
    #status, #error {
      pointer-events: auto;
      background: rgba(18, 10, 41, 0.8);
      border: 1px solid #2d1f6e;
      border-radius: 8px;
      padding: 6px 8px;
      font-size: 12px;
    }
    #status {
      color: #a78bfa;
    }
    #error {
      color: #fda4af;
      display: none;
      max-width: 60%;
      white-space: nowrap;
      overflow: hidden;
      text-overflow: ellipsis;
    }
  </style>
</head>
<body>
  <div id="app">
    <div id="view"></div>
    <div id="hud">
      <div id="status">Booting viewer</div>
      <div id="error"></div>
    </div>
  </div>

  <script type="module">
    const ASSETS = {
      head: "./head.glb",
      leftMesh: "./brain-left-hemisphere.glb",
      rightMesh: "./brain-right-hemisphere.glb",
      dynamicBase: "./dynamic/naturalistic/vanessen-2023-timeline-0-start-750",
      leftMeta: "./dynamic/naturalistic/vanessen-2023-timeline-0-start-750/vanessen-2023-timeline-0-start-750-left-hemisphere-face-colors-binary-prediction.json",
      rightMeta: "./dynamic/naturalistic/vanessen-2023-timeline-0-start-750/vanessen-2023-timeline-0-start-750-right-hemisphere-face-colors-binary-prediction.json",
      threeModules: [
        "https://esm.sh/three@0.160.0",
        "https://esm.sh/three@0.160.0?bundle"
      ],
      gltfLoaderModules: [
        "https://esm.sh/three@0.160.0/examples/jsm/loaders/GLTFLoader.js",
        "https://esm.sh/three@0.160.0/examples/jsm/loaders/GLTFLoader.js?bundle"
      ],
      orbitControlsModules: [
        "https://esm.sh/three@0.160.0/examples/jsm/controls/OrbitControls.js",
        "https://esm.sh/three@0.160.0/examples/jsm/controls/OrbitControls.js?bundle"
      ]
    };

    const statusEl = document.getElementById("status");
    const errorEl = document.getElementById("error");
    const viewEl = document.getElementById("view");

    let booting = false;
    let loaded = false;
    let playing = true;
    let frame = 0;
    let numFrames = 24;
    let lastStepAt = 0;
    let rafId = null;
    let resizeObserver = null;

    const setStatus = (text) => { statusEl.textContent = text; };
    const setError = (text) => {
      if (!text) {
        errorEl.style.display = "none";
        errorEl.textContent = "";
        return;
      }
      errorEl.style.display = "block";
      errorEl.textContent = text;
    };

    const delay = (ms) => new Promise((resolve) => setTimeout(resolve, ms));

    async function importWithFallback(urls) {
      let lastError = null;
      for (const url of urls) {
        try {
          return await import(url);
        } catch (err) {
          lastError = err;
        }
      }
      throw lastError || new Error("Module import failed.");
    }

    async function withRetry(task, retries = 3, waitMs = 700) {
      let lastErr = null;
      for (let i = 0; i < retries; i++) {
        try {
          return await task();
        } catch (err) {
          lastErr = err;
          if (i < retries - 1) await delay(waitMs * (i + 1));
        }
      }
      throw lastErr || new Error("Operation failed.");
    }

    async function fetchJSON(url) {
      const res = await fetch(url);
      if (!res.ok) throw new Error(`JSON fetch failed (${res.status}) for ${url}`);
      return await res.json();
    }

    async function fetchBIN(url) {
      const res = await fetch(url);
      if (!res.ok) throw new Error(`BIN fetch failed (${res.status}) for ${url}`);
      return new Uint8Array(await res.arrayBuffer());
    }

    function cleanupView() {
      if (rafId) {
        cancelAnimationFrame(rafId);
        rafId = null;
      }
      if (resizeObserver) {
        resizeObserver.disconnect();
        resizeObserver = null;
      }
      viewEl.innerHTML = "";
    }

    async function boot(force = false) {
      if (booting) return false;
      if (loaded && !force) return true;
      booting = true;
      setError("");
      cleanupView();

      try {
        setStatus("Loading Three.js modules");
        const THREE = await withRetry(
          () => importWithFallback(ASSETS.threeModules),
          4,
          900
        );
        const { GLTFLoader } = await withRetry(
          () => importWithFallback(ASSETS.gltfLoaderModules),
          4,
          900
        );
        const { OrbitControls } = await withRetry(
          () => importWithFallback(ASSETS.orbitControlsModules),
          4,
          900
        );

        setStatus("Initializing renderer");
        const scene = new THREE.Scene();
        scene.background = new THREE.Color(0x090513);
        const camera = new THREE.PerspectiveCamera(40, 1, 0.01, 100);
        camera.position.set(0, 0.1, 2.95);

        const renderer = new THREE.WebGLRenderer({ antialias: true });
        renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
        viewEl.appendChild(renderer.domElement);

        const controls = new OrbitControls(camera, renderer.domElement);
        controls.enablePan = false;
        controls.enableDamping = true;
        controls.minDistance = 2.1;
        controls.maxDistance = 5.6;
        controls.target.set(0, 0.02, 0);

        scene.add(new THREE.AmbientLight(0xffffff, 0.7));
        const key = new THREE.DirectionalLight(0xffffff, 0.65);
        key.position.set(2.2, 2.4, 2.8);
        scene.add(key);
        const fill = new THREE.DirectionalLight(0x8b5cf6, 0.45);
        fill.position.set(-2.0, 1.0, -1.5);
        scene.add(fill);

        const gltfLoader = new GLTFLoader();
        const loadGLB = (url) =>
          withRetry(
            () => new Promise((resolve, reject) => gltfLoader.load(url, resolve, undefined, reject)),
            4,
            900
          );

        setStatus("Loading meshes + heatmap data");
        const [leftGLB, rightGLB, leftMeta, rightMeta] = await Promise.all([
          loadGLB(ASSETS.leftMesh),
          loadGLB(ASSETS.rightMesh),
          withRetry(() => fetchJSON(ASSETS.leftMeta), 4, 700),
          withRetry(() => fetchJSON(ASSETS.rightMeta), 4, 700),
        ]);

        const [leftBin, rightBin] = await Promise.all([
          withRetry(() => fetchBIN(`${ASSETS.dynamicBase}/${leftMeta.colorsBin}`), 4, 700),
          withRetry(() => fetchBIN(`${ASSETS.dynamicBase}/${rightMeta.colorsBin}`), 4, 700),
        ]);

        const firstMesh = (gltf) => {
          let mesh = null;
          gltf.scene.traverse((obj) => {
            if (!mesh && obj.isMesh) mesh = obj;
          });
          if (!mesh) throw new Error("No mesh found in GLB.");
          return mesh;
        };

        const leftMesh = firstMesh(leftGLB);
        const rightMesh = firstMesh(rightGLB);
        leftMesh.geometry = leftMesh.geometry.toNonIndexed();
        rightMesh.geometry = rightMesh.geometry.toNonIndexed();

        const setupHeat = (mesh, meta, bin) => {
          const faceCount = Math.floor(mesh.geometry.attributes.position.count / 3);
          const usableFaces = Math.min(faceCount, meta.numFaces);
          const colorAttr = new THREE.BufferAttribute(new Float32Array(faceCount * 9), 3);
          colorAttr.array.fill(0.08);
          mesh.geometry.setAttribute("color", colorAttr);
          mesh.material = new THREE.MeshStandardMaterial({
            vertexColors: true,
            roughness: 0.36,
            metalness: 0.05,
            emissive: 0x11091f,
            emissiveIntensity: 0.36
          });
          return { meta, bin, colorAttr, usableFaces };
        };

        const hemis = [
          setupHeat(leftMesh, leftMeta, leftBin),
          setupHeat(rightMesh, rightMeta, rightBin),
        ];

        const brainRoot = new THREE.Group();
        brainRoot.scale.setScalar(0.88);
        brainRoot.position.set(0, -0.02, 0);
        brainRoot.add(leftGLB.scene);
        brainRoot.add(rightGLB.scene);
        scene.add(brainRoot);

        numFrames = Math.max(1, Math.min(leftMeta.numFrames, rightMeta.numFrames));
        frame = 0;
        playing = true;
        lastStepAt = 0;

        const lerp = (a, b, t) => a + (b - a) * t;
        const heatColor = (t) => {
          const x = Math.max(0, Math.min(1, t));
          if (x < 0.33) {
            const q = x / 0.33;
            return [
              lerp(0.08, 0.0, q),
              lerp(0.16, 0.78, q),
              lerp(0.50, 1.0, q),
            ];
          }
          if (x < 0.66) {
            const q = (x - 0.33) / 0.33;
            return [
              lerp(0.0, 1.0, q),
              lerp(0.78, 0.86, q),
              lerp(1.0, 0.18, q),
            ];
          }
          const q = (x - 0.66) / 0.34;
          return [
            1.0,
            lerp(0.86, 0.12, q),
            lerp(0.18, 0.08, q),
          ];
        };

        const updateFrame = (idx) => {
          frame = ((idx % numFrames) + numFrames) % numFrames;
          for (const hemi of hemis) {
            const arr = hemi.colorAttr.array;
            const frameOffset = frame * hemi.meta.frameByteSize;

            let minV = 1.0;
            let maxV = 0.0;
            for (let f = 0; f < hemi.usableFaces; f++) {
              const c = frameOffset + f * 3;
              const raw = (hemi.bin[c] + hemi.bin[c + 1] + hemi.bin[c + 2]) / (3 * 255);
              if (raw < minV) minV = raw;
              if (raw > maxV) maxV = raw;
            }
            const span = Math.max(1e-5, maxV - minV);

            for (let f = 0; f < hemi.usableFaces; f++) {
              const c = frameOffset + f * 3;
              const raw = (hemi.bin[c] + hemi.bin[c + 1] + hemi.bin[c + 2]) / (3 * 255);
              const norm = (raw - minV) / span;
              const boosted = Math.pow(Math.min(1, norm * 1.35), 0.85);
              const [r, g, b] = heatColor(boosted);
              const v = f * 9;
              arr[v] = r; arr[v + 1] = g; arr[v + 2] = b;
              arr[v + 3] = r; arr[v + 4] = g; arr[v + 5] = b;
              arr[v + 6] = r; arr[v + 7] = g; arr[v + 8] = b;
            }
            hemi.colorAttr.needsUpdate = true;
          }
        };

        const resize = () => {
          const w = Math.max(viewEl.clientWidth, 1);
          const h = Math.max(viewEl.clientHeight, 1);
          camera.aspect = w / h;
          camera.updateProjectionMatrix();
          renderer.setSize(w, h, false);
        };
        resizeObserver = new ResizeObserver(resize);
        resizeObserver.observe(viewEl);
        resize();

        // Fit brain geometry to viewport so it never renders oversized/off-screen.
        const bbox = new THREE.Box3().setFromObject(brainRoot);
        const size = bbox.getSize(new THREE.Vector3());
        const center = bbox.getCenter(new THREE.Vector3());
        brainRoot.position.sub(center);

        const maxDim = Math.max(size.x, size.y, size.z, 1e-3);
        const fov = (camera.fov * Math.PI) / 180;
        const fitHeightDistance = (maxDim * 0.5) / Math.tan(fov / 2);
        const fitWidthDistance = fitHeightDistance / Math.max(camera.aspect, 1e-3);
        const distance = Math.max(fitHeightDistance, fitWidthDistance) * 1.6;

        camera.position.set(0, maxDim * 0.06, distance);
        camera.near = Math.max(0.01, distance / 100);
        camera.far = Math.max(50, distance * 8);
        camera.updateProjectionMatrix();
        controls.target.set(0, 0, 0);
        controls.minDistance = distance * 0.55;
        controls.maxDistance = distance * 3.0;
        controls.update();

        const renderLoop = (ts) => {
          rafId = requestAnimationFrame(renderLoop);
          if (playing && ts - lastStepAt > 1000 / 8) {
            updateFrame(frame + 1);
            lastStepAt = ts;
          }
          controls.update();
          renderer.render(scene, camera);
        };

        updateFrame(0);
        setStatus("Ready");
        loaded = true;
        rafId = requestAnimationFrame(renderLoop);
        return true;
      } catch (err) {
        loaded = false;
        setStatus("Viewer failed - auto retrying");
        setError(String(err && err.message ? err.message : err));
        return false;
      } finally {
        booting = false;
      }
    }

    window.addEventListener("load", () => boot(false));
    setTimeout(() => boot(false), 0);
    setTimeout(() => { if (!loaded) boot(true); }, 1200);
    setInterval(() => { if (!loaded && !booting) boot(true); }, 5000);
  </script>
</body>
</html>
"""
    out = Path("assets/brain-demo/viewer.html")
    out.write_text(viewer_html, encoding="utf-8")


_ensure_brain_demo_assets()
_write_brain_viewer_html()

# Serve local static viewer assets (downloaded at startup)
if hasattr(gr, "set_static_paths"):
    gr.set_static_paths(paths=["assets"])

import src.tribe_engine as tribe_engine
tribe_engine.warm_up()

from src.ctr_predictor import predict_ctr
from src.neural_scorer import activation_to_profile, compute_ctr_score
from src.gemini_client import generate_intro_recommendations
from src.screenshot_analyzer import analyze_screenshots


def _score_color(score: int) -> str:
    if score >= 75:   return "#22c55e"
    elif score >= 50: return "#f59e0b"
    elif score >= 30: return "#f97316"
    return "#ef4444"


def _make_radar_chart(dimension_scores: dict) -> go.Figure:
    categories = list(dimension_scores.keys())
    values = [dimension_scores[c] for c in categories]
    categories_closed = categories + [categories[0]]
    values_closed = values + [values[0]]
    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=values_closed, theta=categories_closed, fill="toself",
        fillcolor="rgba(139, 92, 246, 0.3)",
        line=dict(color="#8b5cf6", width=2), name="Your Content",
    ))
    fig.update_layout(
        polar=dict(
            radialaxis=dict(visible=True, range=[0, 100], tickfont=dict(size=10)),
            angularaxis=dict(tickfont=dict(size=11)),
            bgcolor="rgba(0,0,0,0)",
        ),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        showlegend=False,
        title=dict(text="Neural Signal Breakdown", font=dict(size=14, color="#e2e8f0")),
        margin=dict(l=60, r=60, t=60, b=60), font=dict(color="#e2e8f0"),
    )
    return fig


def _make_ctr_bar_chart(ranked_titles: list[dict]) -> go.Figure:
    top = ranked_titles[:15]
    labels = [f"{i+1}. {t['title'][:55]}{'…' if len(t['title']) > 55 else ''}" for i, t in enumerate(top)]
    scores = [round(t["ctr_score"] * 100, 1) for t in top]
    fig = go.Figure(go.Bar(
        x=scores, y=labels, orientation="h",
        marker_color=[_score_color(int(s)) for s in scores],
        text=[f"{s}%" for s in scores], textposition="outside",
    ))
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(range=[0, 115], showgrid=False, title="Neural CTR Score",
                   tickfont=dict(color="#e2e8f0"), title_font=dict(color="#e2e8f0")),
        yaxis=dict(showgrid=False, tickfont=dict(color="#e2e8f0", size=12)),
        margin=dict(l=20, r=80, t=30, b=30),
        height=max(400, len(top) * 42), font=dict(color="#e2e8f0"),
    )
    return fig


def _make_screenshot_bar_chart(ranked_elements: list[dict]) -> go.Figure:
    top = ranked_elements[:20]
    labels = [
        (
            f"{i+1}. "
            f"{item.get('screenshot_name', '')[:22]} · "
            f"{item.get('label', '')[:28]}"
        )
        for i, item in enumerate(top)
    ]
    scores = [round(item["attention_score"] * 100, 1) for item in top]
    fig = go.Figure(
        go.Bar(
            x=scores,
            y=labels,
            orientation="h",
            marker_color=[_score_color(int(s)) for s in scores],
            text=[f"{s}%" for s in scores],
            textposition="outside",
        )
    )
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(
            range=[0, 115],
            showgrid=False,
            title="Neural Attention Score (Element-Level)",
            tickfont=dict(color="#e2e8f0"),
            title_font=dict(color="#e2e8f0"),
        ),
        yaxis=dict(showgrid=False, tickfont=dict(color="#e2e8f0", size=12)),
        margin=dict(l=20, r=80, t=30, b=30),
        height=max(400, len(top) * 42),
        font=dict(color="#e2e8f0"),
    )
    return fig


def _model_status_badge() -> str:
    if tribe_engine.is_ready():       return "🟢 TRIBE v2 Ready"
    err = tribe_engine.get_error()
    if err:                           return f"🔴 Model Error: {err[:60]}"
    return "🟡 TRIBE v2 Loading..."


BRAIN_SYNC_IDLE = (
    "### TRIBEv2 Brain Activity Alignment\n"
    "- Latest completed result: waiting...\n"
    "- Run any analyzer above to sync context.\n"
    "- Brain activity viewer is at the end of this page."
)


def _brain_sync_note(source: str, summary: str) -> str:
    return (
        "### TRIBEv2 Brain Activity Alignment\n"
        f"- Latest completed result: **{source}**\n"
        f"- {summary}"
    )


BRAIN_HEAD_URL = "./gradio_api/file=assets/brain-demo/head.glb"
BRAIN_LEFT_HEMI_URL = "./gradio_api/file=assets/brain-demo/brain-left-hemisphere.glb"
BRAIN_RIGHT_HEMI_URL = "./gradio_api/file=assets/brain-demo/brain-right-hemisphere.glb"
BRAIN_DYNAMIC_BASE = (
    "./gradio_api/file=assets/brain-demo/dynamic/naturalistic/vanessen-2023-timeline-0-start-750"
)
BRAIN_LEFT_META_URL = (
    f"{BRAIN_DYNAMIC_BASE}/vanessen-2023-timeline-0-start-750-left-hemisphere-face-colors-binary-prediction.json"
)
BRAIN_RIGHT_META_URL = (
    f"{BRAIN_DYNAMIC_BASE}/vanessen-2023-timeline-0-start-750-right-hemisphere-face-colors-binary-prediction.json"
)

BRAIN_SIM_HTML = """
<div id="brain-sim-root" class="brain-sim-card">
  <div class="brain-sim-header">
    <div>
      <div class="brain-sim-title">TRIBE v2 Brain Activity Simulation</div>
      <div class="brain-sim-sub">Three.js brain mesh + face-level activation heatmap</div>
    </div>
    <div id="brain-sim-status" class="brain-sim-status">Loading</div>
  </div>
  <div id="brain-sim-view" class="brain-sim-view"></div>
  <div class="brain-sim-controls">
    <button id="brain-prev" class="brain-btn" type="button">Prev</button>
    <button id="brain-play" class="brain-btn" type="button">Pause</button>
    <button id="brain-next" class="brain-btn" type="button">Next</button>
    <input id="brain-frame" class="brain-slider" type="range" min="0" max="23" value="0" step="1" />
    <span id="brain-frame-label" class="brain-frame-label">1 / 24</span>
  </div>
</div>

<style>
.brain-sim-card {
  border: 1px solid #2d1f6e;
  border-radius: 12px;
  background: #120a29;
  padding: 12px;
  margin: 8px 0 24px 0;
}
.brain-sim-header {
  display: flex;
  align-items: center;
  justify-content: space-between;
  margin-bottom: 10px;
}
.brain-sim-title {
  font-weight: 700;
  color: #f1f5f9;
  font-size: 1rem;
}
.brain-sim-sub {
  color: #94a3b8;
  font-size: 0.82rem;
}
.brain-sim-status {
  font-size: 0.75rem;
  color: #a78bfa;
  font-family: 'JetBrains Mono', monospace;
}
.brain-sim-view {
  width: 100%;
  height: 380px;
  border-radius: 10px;
  overflow: hidden;
  background: radial-gradient(circle at 55% 35%, rgba(39, 18, 89, 0.45), rgba(7, 4, 14, 0.95));
}
.brain-sim-controls {
  margin-top: 10px;
  display: flex;
  align-items: center;
  gap: 8px;
}
.brain-btn {
  background: #21124b;
  border: 1px solid #3b2a7b;
  color: #e2e8f0;
  border-radius: 8px;
  padding: 6px 10px;
  font-size: 0.8rem;
  cursor: pointer;
}
.brain-btn:hover {
  border-color: #8b5cf6;
}
.brain-slider {
  flex: 1;
  accent-color: #8b5cf6;
}
.brain-frame-label {
  width: 62px;
  text-align: right;
  font-size: 0.8rem;
  color: #cbd5e1;
  font-family: 'JetBrains Mono', monospace;
}
</style>

<script type="module">
const delay = (ms) => new Promise((resolve) => setTimeout(resolve, ms));

async function withRetry(fn, label, retries = 4, waitMs = 800) {
  let lastErr = null;
  for (let i = 0; i < retries; i++) {
    try {
      return await fn();
    } catch (err) {
      lastErr = err;
      if (i < retries - 1) await delay(waitMs * (i + 1));
    }
  }
  throw new Error(`${label} failed: ${lastErr}`);
}

async function initBrainSim(root) {
  if (!root || root.dataset.loaded === "1" || root.dataset.loading === "1") return false;
  root.dataset.loading = "1";

  const statusEl = root.querySelector("#brain-sim-status");
  const viewEl = root.querySelector("#brain-sim-view");
  const playBtn = root.querySelector("#brain-play");
  const prevBtn = root.querySelector("#brain-prev");
  const nextBtn = root.querySelector("#brain-next");
  const slider = root.querySelector("#brain-frame");
  const frameLabel = root.querySelector("#brain-frame-label");
  if (!statusEl || !viewEl || !playBtn || !prevBtn || !nextBtn || !slider || !frameLabel) {
    root.dataset.loading = "0";
    return false;
  }

  const setStatus = (msg) => { statusEl.textContent = msg; };

  const ASSETS = {
    head: "__BRAIN_HEAD_URL__",
    leftMesh: "__BRAIN_LEFT_HEMI_URL__",
    rightMesh: "__BRAIN_RIGHT_HEMI_URL__",
    leftMeta: "__BRAIN_LEFT_META_URL__",
    rightMeta: "__BRAIN_RIGHT_META_URL__",
    dynamicBase: "__BRAIN_DYNAMIC_BASE__",
  };

  try {
    setStatus("Initializing renderer");
    const THREE = await withRetry(
      () => import("https://esm.sh/three@0.160.0"),
      "three import"
    );
    const { GLTFLoader } = await withRetry(
      () => import("https://esm.sh/three@0.160.0/examples/jsm/loaders/GLTFLoader.js"),
      "gltf loader import"
    );
    const { OrbitControls } = await withRetry(
      () => import("https://esm.sh/three@0.160.0/examples/jsm/controls/OrbitControls.js"),
      "orbit controls import"
    );

    viewEl.innerHTML = "";
    const scene = new THREE.Scene();
    scene.background = new THREE.Color(0x090513);

    const camera = new THREE.PerspectiveCamera(
      32,
      Math.max(viewEl.clientWidth, 1) / Math.max(viewEl.clientHeight, 1),
      0.01,
      100
    );
    camera.position.set(0, 0.2, 2.35);

    const renderer = new THREE.WebGLRenderer({ antialias: true });
    renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
    renderer.setSize(Math.max(viewEl.clientWidth, 1), Math.max(viewEl.clientHeight, 1));
    viewEl.appendChild(renderer.domElement);

    const controls = new OrbitControls(camera, renderer.domElement);
    controls.enablePan = false;
    controls.enableDamping = true;
    controls.minDistance = 1.45;
    controls.maxDistance = 4.0;
    controls.target.set(0, 0.1, 0);

    scene.add(new THREE.AmbientLight(0xffffff, 0.6));
    const key = new THREE.DirectionalLight(0xffffff, 0.65);
    key.position.set(2.2, 2.4, 2.8);
    scene.add(key);
    const fill = new THREE.DirectionalLight(0x8b5cf6, 0.35);
    fill.position.set(-2.0, 1.0, -1.5);
    scene.add(fill);

    const gltfLoader = new GLTFLoader();
    const loadGLB = (url) =>
      withRetry(
        () =>
          new Promise((resolve, reject) => gltfLoader.load(url, resolve, undefined, reject)),
        `load ${url}`
      );

    const getMesh = (gltf) => {
      let mesh = null;
      gltf.scene.traverse((obj) => {
        if (!mesh && obj.isMesh) mesh = obj;
      });
      if (!mesh) throw new Error("No mesh found in GLB.");
      return mesh;
    };

    const fetchJSON = (url) =>
      withRetry(async () => {
        const res = await fetch(url);
        if (!res.ok) throw new Error(`Failed metadata fetch: ${url}`);
        return res.json();
      }, `fetch json ${url}`);

    const fetchBin = (url) =>
      withRetry(async () => {
        const res = await fetch(url);
        if (!res.ok) throw new Error(`Failed binary fetch: ${url}`);
        return new Uint8Array(await res.arrayBuffer());
      }, `fetch bin ${url}`);

    setStatus("Loading mesh and heatmap assets");
    const [headGLB, leftGLB, rightGLB, leftMeta, rightMeta] = await Promise.all([
      loadGLB(ASSETS.head),
      loadGLB(ASSETS.leftMesh),
      loadGLB(ASSETS.rightMesh),
      fetchJSON(ASSETS.leftMeta),
      fetchJSON(ASSETS.rightMeta),
    ]);

    const [leftBin, rightBin] = await Promise.all([
      fetchBin(`${ASSETS.dynamicBase}/${leftMeta.colorsBin}`),
      fetchBin(`${ASSETS.dynamicBase}/${rightMeta.colorsBin}`),
    ]);

    headGLB.scene.traverse((obj) => {
      if (obj.isMesh) {
        obj.material = new THREE.MeshStandardMaterial({
          color: 0xb2bdd6,
          roughness: 0.8,
          metalness: 0.0,
          transparent: true,
          opacity: 0.18,
          side: THREE.DoubleSide,
        });
      }
    });

    const leftMesh = getMesh(leftGLB);
    const rightMesh = getMesh(rightGLB);
    leftMesh.geometry = leftMesh.geometry.toNonIndexed();
    rightMesh.geometry = rightMesh.geometry.toNonIndexed();

    const setupHeatmapMesh = (mesh, meta, bin) => {
      const faceCount = Math.floor(mesh.geometry.attributes.position.count / 3);
      const usableFaces = Math.min(faceCount, meta.numFaces);
      const colorAttr = new THREE.BufferAttribute(new Float32Array(faceCount * 9), 3);
      colorAttr.array.fill(0.08);
      mesh.geometry.setAttribute("color", colorAttr);
      mesh.material = new THREE.MeshStandardMaterial({
        vertexColors: true,
        roughness: 0.42,
        metalness: 0.05,
        emissive: 0x11091f,
        emissiveIntensity: 0.28,
      });
      return { meta, bin, colorAttr, usableFaces };
    };

    const leftState = setupHeatmapMesh(leftMesh, leftMeta, leftBin);
    const rightState = setupHeatmapMesh(rightMesh, rightMeta, rightBin);
    const hemis = [leftState, rightState];

    scene.add(headGLB.scene);
    scene.add(leftGLB.scene);
    scene.add(rightGLB.scene);

    const numFrames = Math.max(1, Math.min(leftMeta.numFrames, rightMeta.numFrames));
    slider.max = String(numFrames - 1);

    let frame = 0;
    let playing = true;
    let lastStepAt = 0;
    const fps = 8;

    const updateFrame = (idx) => {
      frame = ((idx % numFrames) + numFrames) % numFrames;
      for (const hemi of hemis) {
        const arr = hemi.colorAttr.array;
        const frameOffset = frame * hemi.meta.frameByteSize;
        for (let f = 0; f < hemi.usableFaces; f++) {
          const c = frameOffset + f * 3;
          const r = hemi.bin[c] / 255;
          const g = hemi.bin[c + 1] / 255;
          const b = hemi.bin[c + 2] / 255;
          const v = f * 9;
          arr[v] = r; arr[v + 1] = g; arr[v + 2] = b;
          arr[v + 3] = r; arr[v + 4] = g; arr[v + 5] = b;
          arr[v + 6] = r; arr[v + 7] = g; arr[v + 8] = b;
        }
        hemi.colorAttr.needsUpdate = true;
      }
      slider.value = String(frame);
      frameLabel.textContent = `${frame + 1} / ${numFrames}`;
    };

    const syncPlayButton = () => {
      playBtn.textContent = playing ? "Pause" : "Play";
    };
    playBtn.onclick = () => { playing = !playing; syncPlayButton(); };
    prevBtn.onclick = () => { playing = false; syncPlayButton(); updateFrame(frame - 1); };
    nextBtn.onclick = () => { playing = false; syncPlayButton(); updateFrame(frame + 1); };
    slider.oninput = () => { playing = false; syncPlayButton(); updateFrame(Number(slider.value)); };

    const resize = () => {
      const w = Math.max(viewEl.clientWidth, 1);
      const h = Math.max(viewEl.clientHeight, 1);
      camera.aspect = w / h;
      camera.updateProjectionMatrix();
      renderer.setSize(w, h, false);
    };
    new ResizeObserver(resize).observe(viewEl);
    resize();

    const renderLoop = (ts) => {
      requestAnimationFrame(renderLoop);
      if (playing && ts - lastStepAt > 1000 / fps) {
        updateFrame(frame + 1);
        lastStepAt = ts;
      }
      controls.update();
      renderer.render(scene, camera);
    };

    updateFrame(0);
    syncPlayButton();
    setStatus("Ready");
    requestAnimationFrame(renderLoop);
    root.dataset.loaded = "1";
    root.dataset.loading = "0";
    return true;
  } catch (err) {
    console.error(err);
    setStatus("Viewer retrying");
    root.dataset.loading = "0";
    return false;
  }
}

async function bootBrainSim() {
  const root = document.getElementById("brain-sim-root");
  if (!root) return false;
  return initBrainSim(root);
}

(async () => {
  for (let i = 0; i < 12; i++) {
    const ok = await bootBrainSim();
    if (ok) break;
    await delay(700);
  }

  const observer = new MutationObserver(() => {
    bootBrainSim();
  });
  if (document.body) {
    observer.observe(document.body, { childList: true, subtree: true });
  }
})();
</script>
"""

BRAIN_SIM_HTML = (
    BRAIN_SIM_HTML.replace("__BRAIN_HEAD_URL__", BRAIN_HEAD_URL)
    .replace("__BRAIN_LEFT_HEMI_URL__", BRAIN_LEFT_HEMI_URL)
    .replace("__BRAIN_RIGHT_HEMI_URL__", BRAIN_RIGHT_HEMI_URL)
    .replace("__BRAIN_LEFT_META_URL__", BRAIN_LEFT_META_URL)
    .replace("__BRAIN_RIGHT_META_URL__", BRAIN_RIGHT_META_URL)
    .replace("__BRAIN_DYNAMIC_BASE__", BRAIN_DYNAMIC_BASE)
)

# Use a dedicated static iframe viewer to avoid Gradio script timing issues.
BRAIN_VIEWER_URL = f"/gradio_api/file=assets/brain-demo/viewer.html?v={int(time.time())}"
BRAIN_SIM_HTML = """
<div class="brain-sim-card">
  <div class="brain-sim-header">
    <div>
      <div class="brain-sim-title">TRIBE v2 Brain Activity Simulation</div>
      <div class="brain-sim-sub">Three.js brain-only activated pattern heatmap</div>
    </div>
  </div>
  <iframe
    id="brain-viewer-frame"
    src="__BRAIN_VIEWER_URL__"
    style="width:100%;height:430px;border:1px solid #2d1f6e;border-radius:10px;background:#090513"
    loading="eager"
    referrerpolicy="no-referrer"
    allowfullscreen
  ></iframe>
</div>
""".replace("__BRAIN_VIEWER_URL__", BRAIN_VIEWER_URL)


# ── Handlers ──────────────────────────────────────────────────────────────────

def _intro_verdict(score: int) -> str:
    if score >= 75:
        return "Strong Opening - High SEO/attention potential"
    if score >= 55:
        return "Solid Opening - Can be sharpened"
    if score >= 35:
        return "Weak Opening - Needs stronger hook"
    return "Low-impact Opening - Rewrite recommended"


def _intro_dimension_scores(profile) -> dict:
    return {
        "Hook Salience": int(round(profile.peak_salience * 100)),
        "Immediate Engagement": int(round(profile.global_engagement * 100)),
        "Retention Potential": int(round(profile.sustained_attention * 100)),
        "Cognitive Richness": int(round(profile.activation_entropy * 100)),
        "Complexity Balance": int(round(profile.neural_complexity * 100)),
        "Focus Signal": int(round(profile.dmn_suppression * 100)),
    }


def _intro_score(profile) -> int:
    score = (
        0.34 * profile.peak_salience
        + 0.28 * profile.global_engagement
        + 0.20 * profile.sustained_attention
        + 0.18 * profile.activation_entropy
    )
    return int(round(max(0.0, min(1.0, score)) * 100))


def run_intro_paragraph(paragraph: str, get_recs: bool):
    text = (paragraph or "").strip()
    if not text:
        yield "⚠️ Please paste your introduction paragraph.", None, "", "", BRAIN_SYNC_IDLE
        return

    was_trimmed = len(text) > 600
    if was_trimmed:
        text = text[:600]

    progress = "🧠 Encoding introduction with TRIBE v2 ..."
    if was_trimmed:
        progress = "✂️ Input auto-trimmed to 600 chars. Encoding with TRIBE v2 ..."
    yield progress, None, "", "", BRAIN_SYNC_IDLE
    try:
        activations = tribe_engine.predict_texts_quick([text], max_chars=600)
        activation = activations[0] if activations else None
        if activation is None:
            yield "❌ TRIBE returned no activation for this paragraph.", None, "", "", BRAIN_SYNC_IDLE
            return
        profile = activation_to_profile(activation)
    except Exception as exc:
        yield f"❌ TRIBE inference failed: {exc}", None, "", "", BRAIN_SYNC_IDLE
        return

    score = _intro_score(profile)
    verdict = _intro_verdict(score)
    dim_scores = _intro_dimension_scores(profile)
    score_color = _score_color(score)

    status = (
        "### Intro SEO Neural Score: "
        f"<span style='color:{score_color};font-size:2em;font-weight:800'>{score}/100</span>\n\n"
        f"**Verdict:** {verdict}  \n"
        f"**Length:** {len(text)} / 600 chars"
    )
    radar = _make_radar_chart(dim_scores)

    recs = ""
    if get_recs:
        yield "💡 Generating Gemini recommendations...", radar, status, "", BRAIN_SYNC_IDLE
        recs = generate_intro_recommendations(
            intro_text=text,
            intro_score=score,
            verdict=verdict,
            dimension_scores=dim_scores,
        )

    sync = _brain_sync_note("Intro Paragraph Analyzer", f"Intro score {score}/100 · {verdict}")
    yield _model_status_badge(), radar, status, recs, sync


def run_ctr(keyword, n_titles):
    if not keyword or not keyword.strip():
        yield "⚠️ Please enter a keyword.", None, "", "", BRAIN_SYNC_IDLE
        return

    n_titles = int(n_titles)
    yield f"✍️ Generating {n_titles} title variants with Gemini...", None, "", "", BRAIN_SYNC_IDLE

    result = predict_ctr(keyword.strip(), n_titles=n_titles)
    if result.get("error"):
        yield f"❌ {result['error']}", None, "", "", BRAIN_SYNC_IDLE
        return

    ranked = result["ranked_titles"]
    analysis = result.get("analysis", "")

    table_rows = ["| Rank | Title | Neural CTR Score |", "|------|-------|-----------------|"]
    for i, item in enumerate(ranked):
        table_rows.append(f"| {i+1} | {item['title']} | {item['ctr_score'] * 100:.1f}% |")

    chart = _make_ctr_bar_chart(ranked)
    status = f"✅ Scored {len(ranked)} titles for **{keyword}** · ranked by neural attention activation"
    top = ranked[0] if ranked else None
    if top:
        sync = _brain_sync_note(
            "Neural CTR Predictor",
            f"Top title score {top['ctr_score'] * 100:.1f}% for keyword '{keyword}'",
        )
    else:
        sync = BRAIN_SYNC_IDLE
    yield _model_status_badge(), chart, "\n".join(table_rows), analysis, sync


def _normalize_uploaded_files(uploaded_files) -> list[str]:
    if not uploaded_files:
        return []
    if isinstance(uploaded_files, str):
        return [uploaded_files] if os.path.exists(uploaded_files) else []

    normalized = []
    for item in uploaded_files:
        if isinstance(item, str):
            path = item
        elif isinstance(item, dict) and "path" in item:
            path = item["path"]
        elif hasattr(item, "name"):
            path = item.name
        else:
            path = None

        if path and os.path.exists(path):
            normalized.append(path)
    return normalized


def run_screenshot_analysis(uploaded_files):
    image_paths = _normalize_uploaded_files(uploaded_files)
    if not image_paths:
        yield "⚠️ Please upload at least one screenshot image.", None, "", "", [], BRAIN_SYNC_IDLE
        return

    yield f"🧠 Running TRIBE v2 visual inference on {len(image_paths)} screenshot(s)...", None, "", "", [], BRAIN_SYNC_IDLE

    result = analyze_screenshots(image_paths)
    if result.get("error"):
        yield f"❌ {result['error']}", None, "", "", [], BRAIN_SYNC_IDLE
        return

    ranked_elements = result.get("ranked_elements", [])
    if not ranked_elements:
        yield "❌ No screenshots were scored.", None, "", "", [], BRAIN_SYNC_IDLE
        return

    chart = _make_screenshot_bar_chart(ranked_elements)
    table_rows = [
        "| Rank | Screenshot | Element | Type | Neural Score | BBox (norm) | Notes |",
        "|------|------------|---------|------|-------------|-------------|-------|",
    ]
    for i, item in enumerate(ranked_elements[:40]):
        notes = (
            item.get("error")
            or f"engagement={item['profile'].get('global_engagement', 0):.3f}, salience={item['profile'].get('peak_salience', 0):.3f}"
        )
        b = item.get("bbox_norm", {})
        bbox_txt = f"x={b.get('x', 0)},y={b.get('y', 0)},w={b.get('w', 0)},h={b.get('h', 0)}"
        table_rows.append(
            f"| {i+1} | {item.get('screenshot_name', 'N/A')} | {item.get('label', 'UI element')} | {item.get('kind', 'other')} | {item['attention_score'] * 100:.1f}% | {bbox_txt} | {notes} |"
        )

    summary = result.get("summary", "")
    overlay_gallery = result.get("overlay_gallery", [])
    if overlay_gallery:
        summary = summary + "\n\n✅ See **Live TRIBE Region Overlays** below for exact on-image regions."
    top_el = ranked_elements[0] if ranked_elements else None
    if top_el:
        sync = _brain_sync_note(
            "Neural Screenshot Analyzer",
            f"Top region '{top_el.get('label', 'region')}' scored {top_el['attention_score'] * 100:.1f}%",
        )
    else:
        sync = BRAIN_SYNC_IDLE
    yield _model_status_badge(), chart, "\n".join(table_rows), summary, overlay_gallery, sync


def run_mini_text_probe(title: str, meta: str):
    title = (title or "").strip()
    meta = (meta or "").strip()

    candidates: list[tuple[str, str]] = []
    if title:
        candidates.append(("Title", title))
    if meta:
        candidates.append(("Meta Description", meta))
    if title and meta:
        candidates.append(("Title + Meta", f"{title}\n\n{meta}"))

    if not candidates:
        return "⚠️ Enter title and/or meta first.", ""

    try:
        activations = tribe_engine.predict_texts_quick(
            [text for _, text in candidates],
            max_chars=700,
        )
    except Exception as exc:
        return f"❌ TRIBE mini text probe failed: {exc}", ""

    rows = [
        "| Item | Neural CTR | Engagement | Salience | Entropy |",
        "|------|-----------:|-----------:|---------:|--------:|",
    ]
    valid_scores = []
    for (label, _), activation in zip(candidates, activations):
        if activation is None:
            rows.append(f"| {label} | n/a | n/a | n/a | n/a |")
            continue
        profile = activation_to_profile(activation)
        ctr = compute_ctr_score(profile)
        valid_scores.append((label, ctr))
        rows.append(
            f"| {label} | {ctr * 100:.1f}% | {profile.global_engagement * 100:.1f}% | "
            f"{profile.peak_salience * 100:.1f}% | {profile.activation_entropy * 100:.1f}% |"
        )

    if not valid_scores:
        return "❌ TRIBE returned no usable activations for this input.", "\n".join(rows)

    best_label, best_score = sorted(valid_scores, key=lambda x: x[1], reverse=True)[0]
    summary = f"**Top neural performer:** `{best_label}` at **{best_score * 100:.1f}%** CTR proxy."
    return _model_status_badge(), summary + "\n\n" + "\n".join(rows)


def run_mini_title_duel(title_a: str, title_b: str):
    title_a = (title_a or "").strip()
    title_b = (title_b or "").strip()
    if not title_a or not title_b:
        return "⚠️ Provide both title candidates.", ""

    try:
        activations = tribe_engine.predict_texts_quick([title_a, title_b], max_chars=220)
    except Exception as exc:
        return f"❌ TRIBE mini duel failed: {exc}", ""

    if len(activations) != 2 or activations[0] is None or activations[1] is None:
        return "❌ Could not score one of the titles with TRIBE.", ""

    p_a = activation_to_profile(activations[0])
    p_b = activation_to_profile(activations[1])
    s_a = compute_ctr_score(p_a)
    s_b = compute_ctr_score(p_b)

    winner = "A" if s_a >= s_b else "B"
    delta = abs(s_a - s_b) * 100
    md = (
        "### Neural Title Duel\n\n"
        f"- **Winner:** Title {winner}\n"
        f"- **Delta:** {delta:.1f} points\n\n"
        "| Variant | Neural CTR | Engagement | Salience | Sustained Attention |\n"
        "|---------|-----------:|-----------:|---------:|--------------------:|\n"
        f"| A | {s_a * 100:.1f}% | {p_a.global_engagement * 100:.1f}% | {p_a.peak_salience * 100:.1f}% | {p_a.sustained_attention * 100:.1f}% |\n"
        f"| B | {s_b * 100:.1f}% | {p_b.global_engagement * 100:.1f}% | {p_b.peak_salience * 100:.1f}% | {p_b.sustained_attention * 100:.1f}% |"
    )
    return _model_status_badge(), md


def _quick_regions() -> list[dict]:
    return [
        {"label": "query-header", "bbox_norm": {"x": 50, "y": 30, "w": 900, "h": 90}},
        {"label": "top-answer", "bbox_norm": {"x": 50, "y": 140, "w": 900, "h": 190}},
        {"label": "left-result-1", "bbox_norm": {"x": 50, "y": 350, "w": 620, "h": 170}},
        {"label": "left-result-2", "bbox_norm": {"x": 50, "y": 540, "w": 620, "h": 170}},
        {"label": "right-panel", "bbox_norm": {"x": 700, "y": 320, "w": 250, "h": 380}},
    ]


def _crop_quick_regions(image_path: str) -> tuple[list[dict], list[str]]:
    regions = _quick_regions()
    entries: list[dict] = []
    tmp_paths: list[str] = []

    with Image.open(image_path) as image:
        image = image.convert("RGB")
        w, h = image.size
        for region in regions:
            b = region["bbox_norm"]
            x = int((b["x"] / 1000) * w)
            y = int((b["y"] / 1000) * h)
            rw = int((b["w"] / 1000) * w)
            rh = int((b["h"] / 1000) * h)
            x = max(0, min(w - 1, x))
            y = max(0, min(h - 1, y))
            rw = max(20, min(w - x, rw))
            rh = max(20, min(h - y, rh))

            crop = image.crop((x, y, x + rw, y + rh))
            out_path = Path(tempfile.gettempdir()) / f"neuralseo-mini-crop-{uuid.uuid4().hex}.png"
            crop.save(out_path)
            tmp_paths.append(str(out_path))
            entries.append(
                {
                    "label": region["label"],
                    "bbox_norm": b,
                    "crop_path": str(out_path),
                }
            )
    return entries, tmp_paths


def _annotate_scored_regions(image_path: str, scored: list[dict]) -> str:
    with Image.open(image_path) as image:
        image = image.convert("RGB")
        w, h = image.size
        draw = ImageDraw.Draw(image)

        for idx, item in enumerate(scored):
            b = item["bbox_norm"]
            x = int((b["x"] / 1000) * w)
            y = int((b["y"] / 1000) * h)
            rw = int((b["w"] / 1000) * w)
            rh = int((b["h"] / 1000) * h)
            x = max(0, min(w - 1, x))
            y = max(0, min(h - 1, y))
            rw = max(20, min(w - x, rw))
            rh = max(20, min(h - y, rh))

            score = item.get("attention_score", 0.0)
            if score >= 0.7:
                color = (34, 197, 94)
            elif score >= 0.5:
                color = (251, 146, 60)
            else:
                color = (244, 63, 94)

            draw.rectangle((x, y, x + rw, y + rh), outline=color, width=4)
            label = f"{idx+1}. {item['label']} ({score * 100:.1f}%)"
            label_y = max(0, y - 20)
            draw.rectangle((x, label_y, min(w - 1, x + 280), y), fill=(15, 10, 30))
            draw.text((x + 5, label_y + 2), label, fill=color)

        out_path = Path(tempfile.gettempdir()) / f"neuralseo-mini-preview-{uuid.uuid4().hex}.png"
        image.save(out_path)
        return str(out_path)


def run_mini_visual_probe(uploaded_file):
    image_paths = _normalize_uploaded_files(uploaded_file)
    if not image_paths:
        return "⚠️ Upload one screenshot first.", None, ""

    image_path = image_paths[0]
    entries, temp_paths = _crop_quick_regions(image_path)
    try:
        activations = tribe_engine.predict_images_quick(
            [entry["crop_path"] for entry in entries],
            image_duration_sec=1.5,
            fps=6,
        )
    except Exception as exc:
        return f"❌ TRIBE mini visual probe failed: {exc}", None, ""
    finally:
        for path in temp_paths:
            try:
                Path(path).unlink(missing_ok=True)
            except Exception:
                pass

    scored: list[dict] = []
    for entry, activation in zip(entries, activations):
        if activation is None:
            continue
        profile = activation_to_profile(activation)
        score = compute_ctr_score(profile)
        scored.append(
            {
                "label": entry["label"],
                "bbox_norm": entry["bbox_norm"],
                "attention_score": score,
                "profile": profile,
            }
        )

    if not scored:
        return "❌ TRIBE returned no usable activation maps for screenshot regions.", None, ""

    scored.sort(key=lambda x: x["attention_score"], reverse=True)
    preview_path = _annotate_scored_regions(image_path, scored)

    table = [
        "| Rank | Region | Neural Score | Engagement | Salience |",
        "|------|--------|-------------:|-----------:|---------:|",
    ]
    for i, row in enumerate(scored):
        p = row["profile"]
        table.append(
            f"| {i+1} | {row['label']} | {row['attention_score'] * 100:.1f}% | "
            f"{p.global_engagement * 100:.1f}% | {p.peak_salience * 100:.1f}% |"
        )

    return _model_status_badge(), preview_path, "\n".join(table)


# ── CSS ───────────────────────────────────────────────────────────────────────
CSS = """
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;600&display=swap');
body, .gradio-container {
    font-family: 'Space Grotesk', sans-serif !important;
    background: #0f0a1e !important;
    color: #e2e8f0 !important;
}
.hero-title {
    font-size: 2.8rem; font-weight: 700;
    background: linear-gradient(135deg, #a78bfa 0%, #818cf8 50%, #38bdf8 100%);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    background-clip: text; line-height: 1.2; margin-bottom: 0.4rem;
}
.hero-sub { font-size: 1.05rem; color: #94a3b8; font-weight: 400; }
.tribe-badge {
    display: inline-block;
    background: rgba(139,92,246,0.15); border: 1px solid rgba(139,92,246,0.4);
    border-radius: 6px; padding: 4px 14px; font-size: 0.8rem; color: #a78bfa;
    font-family: 'JetBrains Mono', monospace; margin-top: 0.6rem;
}
.gradio-container .tabs > .tab-nav {
    background: #1a1035 !important; border-bottom: 1px solid #2d1f6e !important;
}
.gradio-container .tabs > .tab-nav button {
    color: #94a3b8 !important; font-family: 'Space Grotesk', sans-serif !important;
    font-weight: 500 !important; font-size: 0.95rem !important;
}
.gradio-container .tabs > .tab-nav button.selected {
    color: #a78bfa !important; border-bottom: 2px solid #8b5cf6 !important;
    background: transparent !important;
}
.gr-box, .gr-panel {
    background: #160d30 !important; border: 1px solid #2d1f6e !important;
    border-radius: 12px !important;
}
button.primary {
    background: linear-gradient(135deg, #7c3aed, #4f46e5) !important;
    border: none !important; font-family: 'Space Grotesk', sans-serif !important;
    font-weight: 600 !important; font-size: 0.95rem !important;
    border-radius: 8px !important; padding: 10px 24px !important;
}
button.primary:hover { opacity: 0.85 !important; }
input, textarea {
    background: #0f0a1e !important; border: 1px solid #2d1f6e !important;
    color: #e2e8f0 !important; font-family: 'Space Grotesk', sans-serif !important;
    border-radius: 8px !important;
}
.gradio-container .choice-visible input[type="radio"],
.gradio-container .choice-visible input[type="checkbox"] {
    accent-color: #8b5cf6 !important;
    cursor: pointer !important;
}
.gradio-container .choice-visible label {
    border: 1px solid #2d1f6e !important;
    border-radius: 8px !important;
    padding: 6px 10px !important;
    background: #120a29 !important;
    color: #cbd5e1 !important;
}
.gradio-container .choice-visible label:has(input[type="radio"]:checked),
.gradio-container .choice-visible label:has(input[type="checkbox"]:checked) {
    border-color: #8b5cf6 !important;
    background: rgba(124, 58, 237, 0.28) !important;
    color: #f5f3ff !important;
}
.status-bar {
    font-family: 'JetBrains Mono', monospace; font-size: 0.78rem;
    color: #64748b; padding: 6px 12px; background: #0a0618;
    border-radius: 6px; border: 1px solid #1e1040;
}
"""

# ── Build UI ──────────────────────────────────────────────────────────────────

with gr.Blocks(
    theme=gr.themes.Base(primary_hue="violet", secondary_hue="indigo", neutral_hue="slate"),
    css=CSS,
    title="NeuralSEO — Brain-Powered SEO",
) as demo:

    gr.HTML("""
    <div style='text-align:center;padding:40px 20px 20px'>
        <div class='hero-title'>🧠 NeuralSEO</div>
        <div class='hero-sub'>The only SEO tool that tells you how the human brain actually responds to your content.</div>
        <div class='tribe-badge'>Powered by Meta AI · TRIBE v2 · fMRI-trained on 700+ humans · 1,115 hrs of brain data</div>
    </div>
    """)

    with gr.Tabs():

        with gr.TabItem("🖼️ Neural Screenshot Analyzer"):
            gr.Markdown("""
**Upload SERP and AI-chat screenshots for direct visual neural scoring.**
This pipeline splits each screenshot into deterministic SERP/chat layout regions, crops each region, then uses TRIBE v2 visual inference (image → short video → TRIBE) to assign element-level neural attention scores and draws live overlays directly on the image.
            """)
            with gr.Row():
                with gr.Column(scale=1):
                    screenshot_files = gr.File(
                        file_count="multiple",
                        file_types=["image"],
                        type="filepath",
                        label="Upload screenshots (Google SERP, ChatGPT, Perplexity, AI Mode)",
                    )
                    screenshot_btn = gr.Button("🖼️ Analyze Screenshot Attention", variant="primary")

                with gr.Column(scale=1):
                    screenshot_progress = gr.Textbox(
                        label="Status",
                        interactive=False,
                        value="Ready — upload screenshots above.",
                        elem_classes=["status-bar"],
                    )
                    screenshot_summary = gr.Markdown(label="Summary")

            screenshot_chart = gr.Plot(label="Screenshots Ranked by Neural Attention Score")
            screenshot_overlay_gallery = gr.Gallery(
                label="Live TRIBE Region Overlays",
                show_label=True,
                columns=2,
                height=420,
                object_fit="contain",
                preview=True,
            )
            screenshot_table = gr.Markdown(label="Ranked Screenshots")

        with gr.TabItem("📝 Intro Paragraph Analyzer"):
            gr.Markdown("""
**Analyze your introduction paragraph with TRIBE v2 (auto-trim to 600 chars).**
No URLs. Text only. TRIBE produces neural scores, and Gemini provides rewrite recommendations.
            """)
            with gr.Row():
                with gr.Column(scale=1):
                    intro_input = gr.Textbox(
                        label="Introduction Paragraph (auto-trim to 600 chars)",
                        placeholder="Paste only the first paragraph of your article/page...",
                        lines=6,
                        interactive=True,
                    )
                    intro_recs = gr.Checkbox(
                        value=True,
                        label="Generate Gemini recommendations",
                        interactive=True,
                        elem_classes=["choice-visible"],
                    )
                    intro_btn = gr.Button("🧠 Analyze Introduction", variant="primary")

                with gr.Column(scale=1):
                    intro_progress = gr.Textbox(
                        label="Status",
                        interactive=False,
                        value="Ready — paste intro paragraph above.",
                        elem_classes=["status-bar"],
                    )
                    intro_status = gr.Markdown(label="Result")
                    intro_radar = gr.Plot(label="Neural Signal Breakdown")

            intro_recommendations = gr.Markdown(label="Gemini Recommendations")

        with gr.TabItem("📈 Neural CTR Predictor"):
            gr.Markdown("""
**Know your organic CTR before you publish. No A/B test. No guessing.**
Gemini generates dynamic title variants → TRIBE v2 scores each by frontal attention network activation
+ salience response → ranked by neural CTR score.
            """)
            with gr.Row():
                with gr.Column(scale=1):
                    ctr_keyword = gr.Textbox(
                        label="Keyword / Topic",
                        placeholder="e.g. best running shoes for beginners 2026",
                        lines=2, interactive=True,
                    )
                    ctr_n = gr.Slider(
                        minimum=10, maximum=20, value=15, step=5,
                        label="Title Variants", interactive=True,
                    )
                    ctr_btn = gr.Button("📈 Predict Neural CTR", variant="primary")
                    gr.Markdown("> Each title runs through TRIBE v2 individually. 15 titles takes 2-4 min.")

                with gr.Column(scale=1):
                    ctr_progress = gr.Textbox(
                        label="Status", interactive=False,
                        value="Ready — enter a keyword above.",
                        elem_classes=["status-bar"],
                    )
                    ctr_analysis = gr.Markdown(label="Neural Analysis")

            ctr_chart = gr.Plot(label="Titles Ranked by Neural CTR Score")
            ctr_table = gr.Markdown(label="Ranked Titles")

        with gr.TabItem("⚡ Mini Tools (Fast)"):
            gr.Markdown("""
**Pure TRIBE v2 quick probes.**
All mini tools below run direct TRIBE inference (no external AI APIs).
            """)

            with gr.Group():
                gr.Markdown("### 1) Mini Neural Snippet Probe")
                with gr.Row():
                    with gr.Column(scale=1):
                        mini_title = gr.Textbox(
                            label="Title",
                            lines=2,
                            placeholder="Paste your title tag",
                        )
                        mini_meta = gr.Textbox(
                            label="Meta Description",
                            lines=3,
                            placeholder="Paste your meta description",
                        )
                        mini_probe_btn = gr.Button("🧠 Run TRIBE Probe", variant="primary")
                    with gr.Column(scale=1):
                        mini_probe_status = gr.Textbox(
                            label="Status",
                            interactive=False,
                            value="Ready — TRIBE mini text probe.",
                            elem_classes=["status-bar"],
                        )
                        mini_probe_result = gr.Markdown(value="")

                mini_probe_btn.click(
                    fn=run_mini_text_probe,
                    inputs=[mini_title, mini_meta],
                    outputs=[mini_probe_status, mini_probe_result],
                )

            with gr.Group():
                gr.Markdown("### 2) Mini Neural Title Duel")
                with gr.Row():
                    with gr.Column(scale=1):
                        mini_title_a = gr.Textbox(
                            label="Title A",
                            lines=2,
                            placeholder="First title candidate",
                        )
                        mini_title_b = gr.Textbox(
                            label="Title B",
                            lines=2,
                            placeholder="Second title candidate",
                        )
                        mini_duel_btn = gr.Button("⚔️ Run TRIBE Duel", variant="primary")
                    with gr.Column(scale=1):
                        mini_duel_status = gr.Textbox(
                            label="Status",
                            interactive=False,
                            value="Ready — compare two titles with TRIBE.",
                            elem_classes=["status-bar"],
                        )
                        mini_duel_result = gr.Markdown(value="")

                mini_duel_btn.click(
                    fn=run_mini_title_duel,
                    inputs=[mini_title_a, mini_title_b],
                    outputs=[mini_duel_status, mini_duel_result],
                )

            with gr.Group():
                gr.Markdown("### 3) Mini Visual Region Probe")
                with gr.Row():
                    with gr.Column(scale=1):
                        mini_visual_file = gr.File(
                            file_count="single",
                            file_types=["image"],
                            type="filepath",
                            label="Upload SERP / Chat screenshot (single)",
                        )
                        mini_visual_btn = gr.Button("🖼️ Run TRIBE Visual Probe", variant="primary")
                    with gr.Column(scale=1):
                        mini_visual_status = gr.Textbox(
                            label="Status",
                            interactive=False,
                            value="Ready — score fixed regions with TRIBE.",
                            elem_classes=["status-bar"],
                        )
                        mini_visual_preview = gr.Image(
                            type="filepath",
                            label="TRIBE-Scored Regions",
                        )

                mini_visual_table = gr.Markdown(label="Region Ranking")

                mini_visual_btn.click(
                    fn=run_mini_visual_probe,
                    inputs=[mini_visual_file],
                    outputs=[mini_visual_status, mini_visual_preview, mini_visual_table],
                )

    brain_sync_status = gr.Markdown(value=BRAIN_SYNC_IDLE)
    gr.HTML(BRAIN_SIM_HTML)
    gr.Markdown("""
### About the Builder
- Newsletter: [metehanai.substack.com](https://metehanai.substack.com)
- Website: [metehan.ai](https://metehan.ai)
- X: [x.com/metehan777](https://x.com/metehan777)
- LinkedIn: [linkedin.com/in/metehanyesilyurt](https://linkedin.com/in/metehanyesilyurt)

**This tool is experimental.** Neural scores are directional signals, not ground truth ranking guarantees.
""")
    gr.Markdown("""
### FAQ
**What are T4 limitations for this app?**
- T4 has 16GB VRAM and lower throughput than L4/H200, so large batches and many high-res screenshots are more likely to slow down or timeout.
- On T4, use smaller batches (2-3), fewer screenshots per run, and shorter input text.

**Is L4 better than T4 here?**
- Yes. L4 (24GB VRAM) is noticeably more stable for TRIBEv2 workloads and visual scoring.

**Why can first request be slow?**
- First run warms model weights and caches; later runs are usually faster.

**Are these scores final SEO truth?**
- No. They are TRIBEv2-based neural proxies to support decisions, not direct Google ranking signals.
""")
    gr.HTML("""
    <div style='text-align:center;padding:20px 20px;border-top:1px solid #1e1040;margin-top:18px'>
        <p style='color:#475569;font-size:0.82rem'>
            Built on <a href='https://huggingface.co/facebook/tribev2' target='_blank'
            style='color:#7c3aed'>Meta AI TRIBE v2</a> (CC BY-NC 4.0) ·
            For non-commercial use only ·
            <a href='https://ai.meta.com/research/publications/a-foundation-model-of-vision-audition-and-language-for-in-silico-neuroscience/'
            target='_blank' style='color:#7c3aed'>Research Paper</a>
        </p>
    </div>
    """)

    screenshot_btn.click(
        fn=run_screenshot_analysis,
        inputs=[screenshot_files],
        outputs=[
            screenshot_progress,
            screenshot_chart,
            screenshot_table,
            screenshot_summary,
            screenshot_overlay_gallery,
            brain_sync_status,
        ],
    )

    intro_btn.click(
        fn=run_intro_paragraph,
        inputs=[intro_input, intro_recs],
        outputs=[intro_progress, intro_radar, intro_status, intro_recommendations, brain_sync_status],
    )

    ctr_btn.click(
        fn=run_ctr,
        inputs=[ctr_keyword, ctr_n],
        outputs=[ctr_progress, ctr_chart, ctr_table, ctr_analysis, brain_sync_status],
    )


if __name__ == "__main__":
    demo.queue(max_size=10).launch(
        server_name="0.0.0.0",
        server_port=7860,
        show_error=True,
        ssr_mode=False,
    )
