import os
import sys
import json
import pathlib
import base64
import datetime
import functools

# ============================================================
# WARDOGZ NEURO-UGC v2 — SYSTEM SETUP
# ============================================================

print("Setup Wardogz Brain Scanner v2...")

os.system("curl -LsSf https://astral.sh/uv/install.sh | sh")

uv_path = os.path.expanduser("~/.cargo/bin")
if uv_path not in os.environ.get("PATH", ""):
    os.environ["PATH"] = f"{uv_path}:{os.environ['PATH']}"

os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0"

hf_token = os.environ.get("HF_TOKEN")
if hf_token:
    os.environ["HF_TOKEN"] = hf_token
    os.environ["HUGGING_FACE_HUB_TOKEN"] = hf_token
    os.environ["HUGGINGFACE_TOKEN"] = hf_token
    token_paths = [
        pathlib.Path("/root/.cache/huggingface/token"),
        pathlib.Path("/root/.huggingface/token"),
        pathlib.Path.home() / ".cache" / "huggingface" / "token",
        pathlib.Path.home() / ".huggingface" / "token",
    ]
    for tp in token_paths:
        try:
            tp.parent.mkdir(parents=True, exist_ok=True)
            tp.write_text(hf_token)
        except Exception:
            pass
    from huggingface_hub import login
    login(token=hf_token, add_to_git_credential=False)
    print("Meta Auth OK")
else:
    print("WARNING: HF_TOKEN missing")

# ============================================================
# IMPORTS
# ============================================================

import gradio as gr
import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import io
from PIL import Image

if torch.cuda.is_available():
    torch.set_float32_matmul_precision('high')
    print(f"GPU: {torch.cuda.get_device_name(0)}")

# ============================================================
# MODEL LOADING
# ============================================================

CACHE_DIR      = "./cache/tribev2"
HISTORY_FILE   = "./scan_history.json"
THUMBNAILS_DIR = "./thumbnails"
pathlib.Path(THUMBNAILS_DIR).mkdir(parents=True, exist_ok=True)

try:
    from tribev2 import TribeModel
    model = TribeModel.from_pretrained("facebook/tribev2", cache_folder=CACHE_DIR)
    status_msg = "Brain Scanner v2 Operational"
    print(status_msg)
except Exception as e:
    model = None
    status_msg = f"Loading error: {str(e)}"
    print(status_msg)


# ============================================================
# NEUROLOGICAL MATRIX — ROI DEFINITIONS
#
# Tribe v2 predicts on ~20,484 cortical vertices (fsaverage5).
# We map these vertices to functional areas via the Destrieux
# atlas (nilearn). 6 key metrics for ad auditing:
#
#  Memorability     -> parahippocampal + posterior cingulate
#  Pleasure/Dopamine -> orbitofrontal + vmPFC
#  Stress/Anxiety   -> insula + anterior cingulate
#  Visual Attention -> occipital cortex + cuneus
#  Social Resonance -> TPJ + superior temporal sulcus
#  Audio Impact     -> Heschl's gyrus + superior temporal
# ============================================================

# Destrieux labels -> each ROI is a list of patterns to match
ROI_DESTRIEUX = {
    "memorability": [
        "parahippocampal", "cingul-Mid-Post", "cingul-Post"
    ],
    "pleasure_dopamine": [
        "Orbital", "orbital", "rectus", "frontomargin"
    ],
    "stress_anxiety": [
        "insular", "circular_insula", "cingul-Ant"
    ],
    "visual_attention": [
        "occipital", "cuneus", "calcarine", "Pole_occipital"
    ],
    "social_resonance": [
        "pariet_inf-Angular", "temporal_sup", "temporo_pariet"
    ],
    "audio_impact": [
        "G_T_transv", "temp_sup-Lateral", "temp_sup-Plan"
    ],
}

# UI labels for display
ROI_LABELS = {
    "memorability":       "Memorability",
    "pleasure_dopamine":  "Pleasure / Dopamine",
    "stress_anxiety":     "Stress / Anxiety",
    "visual_attention":   "Visual Attention",
    "social_resonance":   "Social Resonance",
    "audio_impact":       "Audio Impact",
}

# Interpretations for each metric by level
ROI_INTERPRETATIONS = {
    "memorability": {
        "high":   "Memorable ad — message anchored in episodic memory",
        "medium": "Partial memorability — reinforce the logo/slogan",
        "low":    "Low retention — rework the brand signature",
    },
    "pleasure_dopamine": {
        "high":   "Strong anticipatory pleasure — desirable content",
        "medium": "Moderate pleasure — add a visual reward element",
        "low":    "Little dopamine generated — try a narrative reward",
    },
    "stress_anxiety": {
        "high":   "Anxiety-inducing content — watch for repellent effect",
        "medium": "Controlled narrative tension — may be intentional",
        "low":    "Serene atmosphere — suited for trust-based brands",
    },
    "visual_attention": {
        "high":   "Visually captivating — effective art direction",
        "medium": "Standard visual attention — enrich the composition",
        "low":    "Weak visual stimulation — add more movement or color",
    },
    "social_resonance": {
        "high":   "Strong sense of identification — audience captivated",
        "medium": "Partial empathy — humanize the content further",
        "low":    "Low identification — add faces/characters",
    },
    "audio_impact": {
        "high":   "Highly effective soundtrack — music/voice well crafted",
        "medium": "Decent audio — optimize the music or voice tone",
        "low":    "Weak audio impact — rework the sound design",
    },
}


@functools.lru_cache(maxsize=1)
def get_roi_masks(n_vertices: int):
    """
    Load vertex masks for each ROI via nilearn (Destrieux atlas).
    Cached after the first call.
    Returns a dict {roi_name: np.array(bool, shape=(n_vertices,))} or None if nilearn is missing.
    """
    try:
        from nilearn import datasets
        print("Loading Destrieux atlas (nilearn)...")
        destrieux    = datasets.fetch_atlas_surf_destrieux(verbose=0)
        labels_lh    = np.array(destrieux.map_left)    # (10242,)
        labels_rh    = np.array(destrieux.map_right)   # (10242,)
        label_names  = [str(n) for n in destrieux.labels]

        masks = {}
        for roi_name, patterns in ROI_DESTRIEUX.items():
            # Find label indices that match the patterns
            matched_idx = set()
            for i, name in enumerate(label_names):
                if any(p.lower() in name.lower() for p in patterns):
                    matched_idx.add(i)

            if not matched_idx:
                print(f"WARNING: ROI '{roi_name}' — no Destrieux label found")
                continue

            mask_lh   = np.isin(labels_lh, list(matched_idx))
            mask_rh   = np.isin(labels_rh, list(matched_idx))
            full_mask = np.concatenate([mask_lh, mask_rh])  # (20484,)

            # Adapt to actual preds size if different
            if len(full_mask) > n_vertices:
                full_mask = full_mask[:n_vertices]
            elif len(full_mask) < n_vertices:
                full_mask = np.pad(full_mask, (0, n_vertices - len(full_mask)))

            masks[roi_name] = full_mask
            print(f"  {ROI_LABELS[roi_name]}: {full_mask.sum()} vertices")

        return masks if masks else None

    except ImportError:
        print("WARNING: nilearn not available — falling back to uniform split")
        return None
    except Exception as e:
        print(f"WARNING: ROI masks error: {e} — falling back to uniform split")
        return None


def get_roi_masks_fallback(n_vertices: int):
    """
    Fallback if nilearn is absent: uniform cortex split into 6 zones.
    Rough anatomical approximation but functional.
    Zones correspond to known regions on fsaverage5:
    - Occipital (posterior) -> visual
    - Temporal -> audio / social
    - Parietal -> social / memory
    - Inferior frontal -> dopamine / orbitofrontal
    - Insula (approx) -> stress
    - Cingulate (approx) -> memory
    """
    half = n_vertices // 2
    q    = half // 6  # size of one sextant per hemisphere

    def bilateral(start_frac, end_frac):
        s_lh = int(half * start_frac)
        e_lh = int(half * end_frac)
        s_rh = half + int(half * start_frac)
        e_rh = half + int(half * end_frac)
        m = np.zeros(n_vertices, dtype=bool)
        m[s_lh:e_lh] = True
        m[s_rh:e_rh] = True
        return m

    return {
        "memorability":       bilateral(0.70, 0.85),
        "pleasure_dopamine":  bilateral(0.00, 0.20),
        "stress_anxiety":     bilateral(0.40, 0.55),
        "visual_attention":   bilateral(0.80, 1.00),
        "social_resonance":   bilateral(0.55, 0.70),
        "audio_impact":       bilateral(0.25, 0.40),
    }


def compute_brain_matrix(preds: np.ndarray) -> dict:
    """
    Compute the 6 neurological metrics from raw fMRI predictions.
    preds shape: (n_timesteps, n_vertices)
    Returns dict {roi_name: score_0_100}

    Normalization: cross-ROI z-score.
    Each ROI's activation is compared to other ROIs (not the global range).
    score=50 -> average brain activation
    score>70 -> ROI significantly more active than others
    score<30 -> ROI less active than average
    """
    n_vertices = preds.shape[1]

    # Try nilearn, otherwise fallback
    masks = get_roi_masks(n_vertices)
    if masks is None:
        masks = get_roi_masks_fallback(n_vertices)

    # Step 1: raw activation per ROI (spatial + temporal mean)
    roi_activities = {}
    for roi_name in ROI_DESTRIEUX:
        mask = masks.get(roi_name)
        if mask is None or mask.sum() == 0:
            roi_activities[roi_name] = None
            continue
        roi_activities[roi_name] = float(preds[:, mask].mean())

    valid_vals = [v for v in roi_activities.values() if v is not None]
    if not valid_vals:
        return {k: 50 for k in ROI_DESTRIEUX}

    # Step 2: cross-ROI z-score -> each ROI compared to others
    roi_mean = float(np.mean(valid_vals))
    roi_std  = float(np.std(valid_vals))
    if roi_std < 1e-9:
        roi_std = 1.0

    matrix = {}
    for roi_name in ROI_DESTRIEUX:
        val = roi_activities.get(roi_name)
        if val is None:
            matrix[roi_name] = 50
            continue
        z     = (val - roi_mean) / roi_std
        # z=-2 -> 0/100, z=0 -> 50/100, z=+2 -> 100/100
        score = int(np.clip(50 + z * 25, 0, 100))
        matrix[roi_name] = score

    return matrix


def interpret_matrix(matrix: dict) -> str:
    """Generate the interpretation text for the matrix."""
    lines = []
    for roi_name, score in matrix.items():
        label = ROI_LABELS[roi_name]
        interp = ROI_INTERPRETATIONS[roi_name]
        if score >= 65:
            level_key = "high"
            level_str = "High"
        elif score >= 35:
            level_key = "medium"
            level_str = "Medium"
        else:
            level_key = "low"
            level_str = "Low"
        lines.append(f"{label}: {score}/100 [{level_str}]\n  -> {interp[level_key]}")
    return "\n\n".join(lines)


def make_radar_chart(matrix: dict) -> Image.Image:
    """Generate a radar chart (spider web) of the 6 metrics."""
    RADAR_LABELS = {
        "memorability":       "Memorability",
        "pleasure_dopamine":  "Pleasure /\nDopamine",
        "stress_anxiety":     "Stress /\nAnxiety",
        "visual_attention":   "Visual\nAttention",
        "social_resonance":   "Social\nResonance",
        "audio_impact":       "Audio Impact",
    }
    labels  = [RADAR_LABELS[k] for k in ROI_DESTRIEUX]
    values  = [matrix.get(k, 50) for k in ROI_DESTRIEUX]
    N       = len(labels)
    angles  = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]
    values_plot = values + values[:1]

    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))

    # Reference zones
    ax.fill(angles, [65] * (N + 1), alpha=0.04, color='green')
    ax.fill(angles, [35] * (N + 1), alpha=0.04, color='orange')

    # Main curve
    ax.plot(angles, values_plot, 'o-', linewidth=2.5, color='#FF4500')
    ax.fill(angles, values_plot, alpha=0.18, color='#FF4500')

    # Points
    ax.scatter(angles[:-1], values, s=60, color='#FF4500', zorder=5)

    # Labels
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, size=9, wrap=True)
    ax.set_ylim(0, 100)
    ax.set_yticks([25, 50, 75, 100])
    ax.set_yticklabels(['25', '50', '75', '100'], size=7, color='#888')
    ax.grid(color='#ddd', linewidth=0.8)

    ax.set_title("Wardogz Neurological Matrix", size=12, fontweight='bold', pad=20)

    # Annotation for memorable score
    mem_score = matrix.get("memorability", 50)
    dop_score = matrix.get("pleasure_dopamine", 50)
    ax.text(0.5, -0.12,
            f"Memorability {mem_score}/100  ·  Dopamine {dop_score}/100",
            transform=ax.transAxes, ha='center', fontsize=8, color='#555')

    plt.tight_layout()
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight', dpi=150)
    buf.seek(0)
    img = Image.open(buf).copy()
    plt.close(fig)
    return img


def make_brain_scan(preds: np.ndarray) -> Image.Image:
    """
    Render a 3D brain surface map from fMRI predictions using nilearn.
    Shows 4 views: left lateral, right lateral, left medial, right medial.
    preds shape: (n_timesteps, n_vertices) — we average across time.
    """
    try:
        from nilearn import datasets, plotting, surface

        # Average across timesteps to get one activation map
        mean_activation = np.mean(preds, axis=0)  # (n_vertices,)

        # Load fsaverage5 mesh (matches tribev2's 20484 vertices)
        fsaverage = datasets.fetch_surf_fsaverage("fsaverage5")

        n_vertices = len(mean_activation)
        half = n_vertices // 2
        data_lh = mean_activation[:half]
        data_rh = mean_activation[half:]

        fig, axes = plt.subplots(2, 2, figsize=(10, 8),
                                 subplot_kw={"projection": "3d"},
                                 gridspec_kw={"wspace": 0, "hspace": -0.1})

        views = [
            (axes[0, 0], "left",  fsaverage["pial_left"],  data_lh, "Left Lateral"),
            (axes[0, 1], "right", fsaverage["pial_right"], data_rh, "Right Lateral"),
            (axes[1, 0], "right", fsaverage["pial_left"],  data_lh, "Left Medial"),
            (axes[1, 1], "left",  fsaverage["pial_right"], data_rh, "Right Medial"),
        ]

        vmin = np.percentile(mean_activation, 2)
        vmax = np.percentile(mean_activation, 98)

        for ax, hemi, mesh, data, title in views:
            bg_map = fsaverage[f"sulc_{hemi}" if hemi in ("left", "right") else "sulc_left"]
            # Use the correct sulcal map for the hemisphere being plotted
            if "Left" in title:
                bg_map = fsaverage["sulc_left"]
            else:
                bg_map = fsaverage["sulc_right"]

            plotting.plot_surf_stat_map(
                mesh, data,
                hemi=hemi,
                view=(0, 180) if "Lateral" in title and "Left" in title else
                      (0, 0) if "Lateral" in title and "Right" in title else
                      (0, 0) if "Medial" in title and "Left" in title else
                      (0, 180),
                bg_map=bg_map,
                cmap="hot",
                vmin=vmin, vmax=vmax,
                threshold=np.percentile(np.abs(mean_activation), 20),
                axes=ax,
                colorbar=False,
                symmetric_cbar=False,
            )
            ax.set_title(title, fontsize=10, pad=-5)

        fig.suptitle("Brain Activation Map — Wardogz", fontsize=13, fontweight="bold", y=0.95)

        buf = io.BytesIO()
        plt.savefig(buf, format="png", bbox_inches="tight", dpi=150, facecolor="white")
        buf.seek(0)
        img = Image.open(buf).copy()
        plt.close(fig)
        return img

    except Exception as e:
        print(f"WARNING: Brain scan rendering failed: {e}")
        import traceback
        traceback.print_exc()
        return None


def make_brain_timesteps(preds: np.ndarray, segments, n_timesteps: int = 15) -> Image.Image:
    """
    Render a multi-panel brain surface visualization showing activation
    at each timestep with stimulus frames, using PlotBrain (PyVista).
    Same as tribe_demo.ipynb notebook visualization.
    """
    try:
        from tribev2.plotting import PlotBrain
        plotter = PlotBrain(mesh="fsaverage5")

        n = min(n_timesteps, len(preds))
        seg = segments[:n] if segments is not None else None
        fig = plotter.plot_timesteps(
            preds[:n],
            segments=seg,
            cmap="fire",
            norm_percentile=99,
            vmin=0.5,
            alpha_cmap=(0, 0.2),
            show_stimuli=True,
        )

        buf = io.BytesIO()
        fig.savefig(buf, format="png", bbox_inches="tight", dpi=120)
        buf.seek(0)
        img = Image.open(buf).copy()
        plt.close(fig)
        print(f"Brain timesteps rendered: {n} frames (PyVista)")
        return img

    except Exception as e:
        print(f"WARNING: Brain timestep rendering failed: {e}")
        import traceback
        traceback.print_exc()
        return None


# ============================================================
# AUDIT PROFILES
# ============================================================

AUDIT_PROFILES = {
    "Short UGC (< 60s)": {
        "target_height": 720,
        "thresholds": {"excellent": 70, "medium": 45},
        "labels": {
            "excellent": "Highly engaging UGC — strong conversion potential",
            "medium":    "Moderate engagement — try a stronger opening hook",
            "low":       "Low engagement — rework the pacing and first moments",
        },
        "description": "Optimized for Reels, TikTok, Stories — videos under 60 seconds"
    },
    "Motion Design (ad)": {
        "target_height": 1080,
        "thresholds": {"excellent": 65, "medium": 40},
        "labels": {
            "excellent": "Highly effective motion — optimal visual storytelling",
            "medium":    "Moderate effectiveness — check on-screen text readability",
            "low":       "Lacking impact — simplify the message and speed up cuts",
        },
        "description": "For animations, explainers, animated advertisements"
    },
    "Long Format (> 1min)": {
        "target_height": 720,
        "thresholds": {"excellent": 60, "medium": 35},
        "labels": {
            "excellent": "Sustained attention throughout — excellent narrative work",
            "medium":    "Drop-offs detected — strengthen transitions and hooks",
            "low":       "Fragile attention — restructure around 3 key moments max",
        },
        "description": "Testimonials, tutorials, videos over 60 seconds"
    },
    "Custom Audit": {
        "target_height": 720,
        "thresholds": {"excellent": 70, "medium": 45},
        "labels": {
            "excellent": "High score",
            "medium":    "Medium score",
            "low":       "Low score",
        },
        "description": "Manual parameters — configure the thresholds yourself"
    },
}

# ============================================================
# HISTORY
# ============================================================

def load_history():
    if not pathlib.Path(HISTORY_FILE).exists():
        return []
    try:
        with open(HISTORY_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return []


def save_to_history(entry: dict):
    history = load_history()
    history.insert(0, entry)
    history = history[:50]
    with open(HISTORY_FILE, "w", encoding="utf-8") as f:
        json.dump(history, f, ensure_ascii=False, indent=2)


def pil_to_base64(img: Image.Image, fmt="PNG", quality=85) -> str:
    """Convert a PIL image to a base64 data URI."""
    if img is None:
        return ""
    # Convert RGBA to RGB for JPEG compatibility
    if fmt == "JPEG" and img.mode == "RGBA":
        bg = Image.new("RGB", img.size, (255, 255, 255))
        bg.paste(img, mask=img.split()[3])
        img = bg
    buf = io.BytesIO()
    img.save(buf, format=fmt, quality=quality)
    mime = "image/png" if fmt == "PNG" else "image/jpeg"
    return f"data:{mime};base64,{base64.b64encode(buf.getvalue()).decode()}"


def extract_thumbnail(video_path, timestamp=1.0):
    try:
        clip  = _open_video(video_path)
        t     = min(timestamp, clip.duration - 0.1)
        frame = clip.get_frame(t)
        clip.close()
        img = Image.fromarray(frame)
        img.thumbnail((320, 180))
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=80)
        return f"data:image/jpeg;base64,{base64.b64encode(buf.getvalue()).decode()}"
    except Exception as e:
        print(f"WARNING: Thumbnail: {e}")
        return None


def format_history_html(history):
    if not history:
        return "<div style='text-align:center;color:#888;padding:40px'>No analyses recorded.</div>"
    cards = []
    for idx, entry in enumerate(history):
        score  = entry.get("score", 0)
        color  = "#22c55e" if score >= 70 else "#f59e0b" if score >= 45 else "#ef4444"
        thumb  = entry.get("thumbnail", "")
        matrix = entry.get("matrix", {})
        verdict = entry.get("verdict", "")

        thumb_html = (
            f'<img src="{thumb}" style="width:100%;border-radius:6px;object-fit:cover;height:140px">'
            if thumb else
            '<div style="height:140px;background:#eee;border-radius:6px;display:flex;align-items:center;justify-content:center;color:#aaa">Video</div>'
        )

        # Mini bars for the 6 metrics
        mini_bars = ""
        for roi_key, roi_label_full in ROI_LABELS.items():
            val       = matrix.get(roi_key, 0)
            bar_color = "#22c55e" if val >= 65 else "#f59e0b" if val >= 35 else "#ef4444"
            short     = roi_label_full[:14]
            mini_bars += f"""
            <div style="display:flex;align-items:center;gap:6px;margin-bottom:3px">
                <span style="font-size:11px;width:110px;color:#555;white-space:nowrap;overflow:hidden">{short}</span>
                <div style="flex:1;background:#eee;border-radius:3px;height:8px">
                    <div style="width:{val}%;background:{bar_color};height:8px;border-radius:3px"></div>
                </div>
                <span style="font-size:11px;color:#777;width:28px;text-align:right;font-weight:bold">{val}</span>
            </div>"""

        # Plot images
        img_attention  = entry.get("img_attention", "")
        img_radar      = entry.get("img_radar", "")
        img_brain      = entry.get("img_brain", "")
        img_timesteps  = entry.get("img_timesteps", "")

        plots_html = ""
        for img_src, label in [(img_attention, "Attention Curve"), (img_radar, "Neurological Matrix"), (img_brain, "Brain Activation"), (img_timesteps, "Brain Activity Timeline")]:
            if img_src:
                plots_html += f"""
                <div style="flex:1;min-width:250px">
                    <div style="font-size:11px;color:#888;margin-bottom:4px;font-weight:600">{label}</div>
                    <img src="{img_src}" style="width:100%;border-radius:6px;border:1px solid #eee">
                </div>"""

        # Verdict text
        verdict_html = ""
        if verdict:
            verdict_escaped = verdict.replace("\n", "<br>").replace("  ", "&nbsp;&nbsp;")
            verdict_html = f"""
            <div style="background:#f8f9fa;border-radius:6px;padding:12px;margin-top:10px;font-size:12px;font-family:monospace;line-height:1.6;color:#333;white-space:pre-wrap;max-height:300px;overflow-y:auto">
                {verdict_escaped}
            </div>"""

        card = f"""
        <div style="background:white;border-radius:10px;box-shadow:0 2px 12px rgba(0,0,0,0.08);overflow:hidden;margin-bottom:20px;border:1px solid #eee">
            <div style="padding:16px">
                <!-- Header -->
                <div style="display:flex;gap:16px;align-items:flex-start;margin-bottom:14px">
                    <div style="width:200px;flex-shrink:0">
                        {thumb_html}
                    </div>
                    <div style="flex:1">
                        <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:6px">
                            <div style="font-weight:bold;font-size:16px" title="{entry.get('filename','')}">{entry.get('filename','Video')}</div>
                            <div style="font-size:28px;font-weight:bold;color:{color}">{score}<span style="font-size:14px;color:#888">/100</span></div>
                        </div>
                        <div style="font-size:12px;color:#888;margin-bottom:10px">{entry.get('date','')} · {entry.get('profile','')} · {entry.get('duration_label','')}</div>
                        <div style="font-size:12px;color:#666;margin-bottom:4px">{entry.get('trough_label','')} · Trend: {entry.get('trend','')}</div>
                        <div style="margin-top:10px">{mini_bars}</div>
                    </div>
                </div>

                <!-- Plots -->
                <div style="display:flex;gap:12px;flex-wrap:wrap;margin-top:12px">
                    {plots_html}
                </div>

                <!-- Verdict -->
                {verdict_html}
            </div>
        </div>"""
        cards.append(card)
    return f'<div style="padding:10px 0">{"".join(cards)}</div>'

# ============================================================
# VIDEO UTILITIES
# ============================================================

def _open_video(video_path):
    """
    Open a VideoFileClip handling moviepy v1.x (moviepy.editor)
    and moviepy v2.x (moviepy directly).
    """
    try:
        from moviepy import VideoFileClip
        return VideoFileClip(video_path)
    except ImportError:
        from moviepy.editor import VideoFileClip
        return VideoFileClip(video_path)


def get_video_info(video_path):
    try:
        clip = _open_video(video_path)
        duration = clip.duration
        w, h     = clip.size
        clip.close()
        return duration, w, h
    except Exception as e:
        print(f"WARNING: moviepy get_video_info: {e}")
        return 32.0, 1280, 720


def downscale_video(video_path, target_height):
    try:
        clip = _open_video(video_path)
        h, w = clip.size[1], clip.size[0]
        if h <= target_height:
            print(f"Resolution OK ({w}x{h})")
            clip.close()
            return video_path
        ratio    = target_height / h
        new_w    = int(w * ratio)
        out_path = video_path + f"_{target_height}p.mp4"
        print(f"Downscaling: {w}x{h} -> {new_w}x{target_height}")
        # moviepy v2.x renamed resize() to resized()
        try:
            resized = clip.resized((new_w, target_height))   # v2.x
        except AttributeError:
            resized = clip.resize((new_w, target_height))    # v1.x
        resized.write_videofile(out_path, codec="libx264", audio_codec="aac", logger=None)
        clip.close()
        return out_path
    except Exception as e:
        print(f"WARNING: Downscale failed ({e})")
        return video_path


def normalize_curve(curve):
    mn, mx = curve.min(), curve.max()
    if mx > mn:
        return (curve - mn) / (mx - mn) * 100
    return np.full_like(curve, 50.0)


def analyse_curve(curve_100, seconds, real_duration):
    n      = len(curve_100)
    score  = int(np.mean(curve_100))
    window = max(1, int(5 * n / real_duration))
    min_avg, min_start = 999, 0
    max_avg, max_start = -1, 0
    for i in range(n - window + 1):
        avg = np.mean(curve_100[i:i + window])
        if avg < min_avg: min_avg, min_start = avg, i
        if avg > max_avg: max_avg, max_start = avg, i
    trough_s = int(seconds[min_start])
    trough_e = min(int(real_duration), trough_s + 5)
    peak_s   = int(seconds[max_start])
    peak_e   = min(int(real_duration), peak_s + 5)
    mid      = n // 2
    d        = np.mean(curve_100[mid:]) - np.mean(curve_100[:mid])
    trend = ("rising" if d > 10 else "declining" if d < -10 else "stable")
    return score, trough_s, trough_e, int(min_avg), peak_s, peak_e, int(max_avg), trend

# ============================================================
# VISUALIZATION — TEMPORAL CURVE
# ============================================================

def make_attention_plot(curve_100, seconds, real_duration, score, trough_s, trough_e,
                        peak_s, peak_e, level, video_label):
    fig, ax = plt.subplots(figsize=(13, 5))
    ax.plot(seconds, curve_100, color='#FF4500', linewidth=2.5, label='Brain Engagement', zorder=3)
    ax.fill_between(seconds, curve_100, alpha=0.10, color='#FF4500')
    mask_trough = (seconds >= trough_s) & (seconds <= trough_e)
    ax.fill_between(seconds, curve_100, where=mask_trough, alpha=0.30,
                    color='#ef4444', label=f'Trough ({trough_s}s-{trough_e}s)')
    mask_peak = (seconds >= peak_s) & (seconds <= peak_e)
    ax.fill_between(seconds, curve_100, where=mask_peak, alpha=0.22,
                    color='#22c55e', label=f'Peak ({peak_s}s-{peak_e}s)')
    color_line = "#22c55e" if level == "excellent" else "#f59e0b" if level == "medium" else "#ef4444"
    ax.axhline(y=score, color=color_line, linestyle='--', alpha=0.8, linewidth=1.5,
               label=f'Average Score: {score}/100')
    ax.set_title(f"Attention Curve — {video_label or 'Video'} · Wardogz",
                 fontsize=13, fontweight='bold', pad=14)
    ax.set_xlabel("Time (seconds)", fontsize=11)
    ax.set_ylabel("Brain Engagement / 100", fontsize=11)
    ax.legend(fontsize=9, loc='upper right')
    ax.grid(True, alpha=0.18)
    ax.set_xlim(0, real_duration)
    ax.set_ylim(0, 108)
    plt.tight_layout()
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight', dpi=150)
    buf.seek(0)
    img = Image.open(buf).copy()
    plt.close(fig)
    return img

# ============================================================
# MAIN ANALYSIS FUNCTION
# ============================================================

def scan_video(video, profile_name, threshold_excellent, threshold_medium, video_label):

    if model is None:
        return "Model not loaded.", None, None, None, None, format_history_html(load_history())
    if video is None:
        return "No video uploaded.", None, None, None, None, format_history_html(load_history())

    try:
        uv_path = os.path.expanduser("~/.cargo/bin")
        if uv_path not in os.environ.get("PATH", ""):
            os.environ["PATH"] = f"{uv_path}:{os.environ['PATH']}"

        profile = AUDIT_PROFILES.get(profile_name, AUDIT_PROFILES["Short UGC (< 60s)"])
        if profile_name == "Custom Audit":
            profile["thresholds"]["excellent"] = threshold_excellent
            profile["thresholds"]["medium"]    = threshold_medium
        target_height = profile["target_height"]
        thresholds    = profile["thresholds"]
        labels        = profile["labels"]

        real_duration, orig_w, orig_h = get_video_info(video)
        duration_label = f"{int(real_duration)}s · {orig_w}x{orig_h}"
        print(f"Video: {duration_label}")

        thumbnail        = extract_thumbnail(video, timestamp=min(1.0, real_duration * 0.1))
        video_to_analyse = downscale_video(video, target_height)

        df_events = model.get_events_dataframe(video_path=video_to_analyse)

        # Duration fix: inject real duration WITH a 0.1s safety margin.
        # Tribe v2 does a strict assert "end <= clip.duration", so if we round
        # to 32.0 but the video is 31.95s -> AssertionError. Subtract 0.1s.
        safe_duration = real_duration - 0.1
        for col in ['Duration', 'duration']:
            if col in df_events.columns:
                current_val = df_events[col].iloc[0] if len(df_events) > 0 else 0
                if safe_duration > current_val:
                    df_events[col] = safe_duration
                    print(f"Duration injected: {safe_duration:.2f}s (actual {real_duration:.2f}s, margin -0.1s)")

        print("Running Tribe v2 prediction...")
        with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
            preds, segments = model.predict(events=df_events)
        if isinstance(preds, torch.Tensor):
            preds = preds.detach().cpu().numpy()
        n_kept = len(preds)
        print(f"Predictions shape: {preds.shape}")

        # Real timestamps (Segment.start confirmed at 1 Hz)
        seg_times = None
        try:
            if hasattr(segments, '__len__') and len(segments) > 0:
                s0 = segments[0]
                for attr in ['start', 'onset', 'time', 'start_time']:
                    if hasattr(s0, attr):
                        seg_times = np.array([getattr(s, attr) for s in segments], dtype=float)
                        break
                if seg_times is None and hasattr(segments, 'columns'):
                    for col in ['start', 'onset', 'time']:
                        if col in segments.columns:
                            seg_times = segments[col].values.astype(float)
                            break
        except Exception as e:
            print(f"WARNING: seg_times: {e}")

        # Temporal coverage
        # 1 prediction = 1 second of fMRI signal at 1 Hz.
        # n_kept / duration = fraction of video covered (approx 1.0 for spoken videos)
        frac         = n_kept / max(real_duration, 1)
        coverage_pct = min(int(frac * 100), 100)
        confidence   = "High" if frac >= 0.8 else "Medium" if frac >= 0.4 else "Low"
        print(f"Coverage: {n_kept} pts / {real_duration:.1f}s · {coverage_pct}% · {confidence}")

        # Global attention curve
        raw_curve = np.mean(preds, axis=1)
        curve_100 = normalize_curve(raw_curve)
        n_points  = len(curve_100)

        # Time axis: real timestamps if available (more precise), otherwise linspace
        if seg_times is not None and len(seg_times) == n_points:
            seconds = seg_times
        else:
            seconds = np.linspace(0, real_duration, n_points)

        score, trough_s, trough_e, trough_val, peak_s, peak_e, peak_val, trend = analyse_curve(
            curve_100, seconds, real_duration
        )

        level = (
            "excellent" if score >= thresholds["excellent"] else
            "medium"    if score >= thresholds["medium"]    else
            "low"
        )

        # Neurological matrix (6 metrics)
        print("Computing neurological matrix...")
        matrix = compute_brain_matrix(preds)
        matrix_text = interpret_matrix(matrix)

        # Global verdict
        confidence_note = {
            "High":   "Reliable signal",
            "Medium": "Partial signal (video has little speech?)",
            "Low":    "Weak signal — results should be taken with caution",
        }[confidence]
        verdict = "\n".join([
            f"Overall Score: {score}/100",
            labels[level],
            "",
            f"Trough: {trough_s}s-{trough_e}s ({trough_val}/100) -> area to rework",
            f"Peak:   {peak_s}s-{peak_e}s ({peak_val}/100) -> strongest moment",
            f"Trend:  {trend}",
            "",
            f"Signal: {n_kept} segments / {int(real_duration)}s · {confidence_note}",
            "",
            "---- Neurological Matrix ----",
            "  (relative scores: 50=average, >70=salient, <30=under-active)",
            matrix_text,
            "",
            f"Video: {n_points} pts · {duration_label} · Profile: {profile_name}",
        ])

        # Visualizations
        plot_attention = make_attention_plot(
            curve_100, seconds, real_duration, score,
            trough_s, trough_e, peak_s, peak_e, level, video_label
        )
        plot_radar = make_radar_chart(matrix)
        plot_brain = make_brain_scan(preds)
        plot_timesteps = make_brain_timesteps(preds, segments)

        # History
        entry = {
            "date":          datetime.datetime.now().strftime("%d/%m/%Y %H:%M"),
            "filename":      video_label or pathlib.Path(video).name,
            "profile":       profile_name,
            "score":         score,
            "duration_label":duration_label,
            "trough_label":  f"Trough {trough_s}s-{trough_e}s ({trough_val}/100)",
            "trend":         trend,
            "thumbnail":     thumbnail,
            "level":         level,
            "matrix":        matrix,
            "verdict":       verdict,
            "img_attention": pil_to_base64(plot_attention, fmt="JPEG", quality=80),
            "img_radar":     pil_to_base64(plot_radar, fmt="JPEG", quality=80),
            "img_brain":     pil_to_base64(plot_brain, fmt="JPEG", quality=80),
            "img_timesteps": pil_to_base64(plot_timesteps, fmt="JPEG", quality=80),
        }
        save_to_history(entry)

        print(f"Done — Score {score}/100 | Trough {trough_s}s-{trough_e}s")
        return verdict, plot_attention, plot_radar, plot_brain, plot_timesteps, format_history_html(load_history())

    except Exception as e:
        import traceback
        print(f"Error:\n{traceback.format_exc()}")
        return f"Error:\n{str(e)}", None, None, None, None, format_history_html(load_history())

# ============================================================
# GRADIO INTERFACE
# ============================================================

with gr.Blocks(theme=gr.themes.Soft(), title="Wardogz Brain Scanner v2") as demo:

    gr.Markdown("""
    # BlackMotion x Wardogz — Neuro-UGC Scanner v2
    *Attention curve + Neurological matrix · Powered by Meta Tribe v2*
    """)
    gr.Markdown(f"**Status:** {status_msg}")

    with gr.Tabs():

        # Tab 1: Analysis
        with gr.Tab("Analysis"):
            with gr.Row():

                with gr.Column(scale=1):
                    input_video = gr.Video(label="Upload Video (HD, SD, any duration)")
                    video_label = gr.Textbox(
                        label="Project / client name",
                        placeholder="e.g. Fidanimo UGC March 2026"
                    )
                    gr.Markdown("### Audit Profile")
                    profile_selector = gr.Radio(
                        choices=list(AUDIT_PROFILES.keys()),
                        value="Short UGC (< 60s)",
                        label="Content type",
                    )
                    profile_desc = gr.Markdown(
                        f"*{AUDIT_PROFILES['Short UGC (< 60s)']['description']}*"
                    )
                    with gr.Accordion("Thresholds (Custom Audit only)", open=False):
                        threshold_excellent = gr.Slider(50, 90, value=70, step=5, label="Excellent Threshold")
                        threshold_medium    = gr.Slider(20, 70, value=45, step=5, label="Medium Threshold")
                    btn = gr.Button("Launch Brain Scan", variant="primary", size="lg")

                with gr.Column(scale=2):
                    output_verdict = gr.Textbox(label="Verdict + Neurological Matrix", lines=18)
                    with gr.Row():
                        output_attention = gr.Image(label="fMRI Attention Curve")
                        output_radar     = gr.Image(label="Neurological Matrix")
                    output_brain = gr.Image(label="3D Brain Activation Map")
                    output_timesteps = gr.Image(label="Brain Activity Timeline (per second)")

            profile_selector.change(
                fn=lambda p: f"*{AUDIT_PROFILES[p]['description']}*",
                inputs=profile_selector,
                outputs=profile_desc
            )

        # Tab 2: History
        with gr.Tab("Scan History"):
            gr.Markdown("### All your analyses — neurological metrics included")
            history_display = gr.HTML(value=format_history_html(load_history()))
            gr.Button("Refresh", size="sm").click(
                fn=lambda: format_history_html(load_history()),
                outputs=history_display
            )

    btn.click(
        fn=scan_video,
        inputs=[input_video, profile_selector, threshold_excellent, threshold_medium, video_label],
        outputs=[output_verdict, output_attention, output_radar, output_brain, output_timesteps, history_display],
    )

    gr.Markdown("---\n*Meta Tribe v2 (V-JEPA2 + LLaMA 3.2 + Wav2Vec-BERT) · Destrieux Atlas · Wardogz Agency*")

demo.launch()
