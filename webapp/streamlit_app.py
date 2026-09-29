# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""TRIBE v2 — Interactive Brain Response Explorer (Streamlit web app).

Run from the repository root:

    pip install -e ".[plotting]"
    pip install streamlit
    streamlit run webapp/streamlit_app.py

The app wraps the same public inference API used in ``tribe_demo.ipynb``:

    model = TribeModel.from_pretrained("facebook/tribev2")
    events = model.get_events_dataframe(video_path=...)
    preds, segments = model.predict(events)

and renders the predicted cortical maps with the headless Nilearn backend
(``PlotBrainNilearn``), so it works on servers without a display.
"""

import io
import tempfile
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless backend — must be set before pyplot import
import matplotlib.pyplot as plt
import numpy as np
import streamlit as st

REPO_URL = "https://github.com/facebookresearch/tribev2"
PAPER_URL = "https://arxiv.org/abs/2605.04326"
HEMODYNAMIC_LAG_S = 5  # predictions are offset 5 s in the past (see README)

VIEWS = [
    "left",
    "right",
    "medial_left",
    "medial_right",
    "dorsal",
    "ventral",
    "anterior",
    "posterior",
]

st.set_page_config(
    page_title="TRIBE v2 — Brain Response Explorer",
    page_icon="🧠",
    layout="wide",
)

# ----------------------------------------------------------------------
# Cached resources
# ----------------------------------------------------------------------


@st.cache_resource(show_spinner=False)
def load_model(repo_id: str, cache_folder: str):
    """Load TRIBE v2 weights once per session (downloads from HuggingFace)."""
    from tribev2 import TribeModel

    return TribeModel.from_pretrained(repo_id, cache_folder=cache_folder)


@st.cache_resource(show_spinner=False)
def load_plotter():
    """Headless (Nilearn/matplotlib) brain-surface plotter on fsaverage5."""
    from tribev2.plotting import PlotBrainNilearn

    return PlotBrainNilearn(mesh="fsaverage5")


# ----------------------------------------------------------------------
# Sidebar — model + plotting options
# ----------------------------------------------------------------------

with st.sidebar:
    st.header("⚙️ Model")
    repo_id = st.text_input("HuggingFace repo / checkpoint dir", "facebook/tribev2")
    cache_folder = st.text_input("Feature cache folder", "./cache")

    if st.button("Load model", type="primary"):
        with st.spinner(
            "Loading TRIBE v2… the first run downloads the weights, "
            "this can take a while."
        ):
            try:
                st.session_state.model = load_model(repo_id, cache_folder)
            except Exception as exc:  # noqa: BLE001 — surface any load error
                st.error(f"Could not load the model: {exc}")
            else:
                st.success("Model loaded ✅")

    try:
        import torch

        device_msg = (
            "GPU (CUDA) 🚀"
            if torch.cuda.is_available()
            else "CPU — expect slow inference 🐢"
        )
    except ModuleNotFoundError:
        device_msg = "unknown — could not import torch"
    st.caption(f"Compute device: {device_msg}")

    st.divider()
    st.header("🎨 Plotting")
    views = st.multiselect("Views", VIEWS, default=["left", "right"])
    cmap = st.selectbox(
        "Colormap", ["hot", "inferno", "viridis", "magma", "coolwarm"], index=0
    )
    norm_percentile = st.slider(
        "Robust normalization percentile", 0, 100, 100,
        help="100 = no normalization. Lower values boost contrast.",
    )
    show_colorbar = st.checkbox("Colorbar", value=True)

    st.divider()
    st.caption(
        "Built on [TRIBE v2](%s) (Meta) · CC BY-NC 4.0 · "
        "[Paper](%s)" % (REPO_URL, PAPER_URL)
    )

# ----------------------------------------------------------------------
# Header
# ----------------------------------------------------------------------

st.title("🧠 TRIBE v2 — Brain Response Explorer")
st.markdown(
    "Predict **fMRI brain responses** to naturalistic stimuli (text, audio or "
    "video) with Meta's multimodal foundation model, and explore the predicted "
    "cortical activity interactively — no notebook required."
)

model = st.session_state.get("model")
if model is None:
    st.info("👈 Load the model from the sidebar to get started.")
    st.stop()

# ----------------------------------------------------------------------
# Input tabs
# ----------------------------------------------------------------------

tab_text, tab_audio, tab_video = st.tabs(["📝 Text", "🔊 Audio", "🎬 Video"])

input_path: Path | None = None
input_kind: str | None = None

with tab_text:
    text = st.text_area(
        "Stimulus text",
        placeholder="Type or paste text — it is synthesised to speech, then "
        "transcribed to obtain word-level timings…",
        height=150,
    )
    if st.button("Run on text", disabled=not text.strip()):
        tmp = Path(tempfile.mkdtemp()) / "stimulus.txt"
        tmp.write_text(text, encoding="utf-8")
        input_path, input_kind = tmp, "text_path"

with tab_audio:
    audio_file = st.file_uploader(
        "Audio stimulus", type=["wav", "mp3", "flac", "ogg"]
    )
    if audio_file is not None:
        st.audio(audio_file)
    if st.button("Run on audio", disabled=audio_file is None):
        tmp = Path(tempfile.mkdtemp()) / f"stimulus.{audio_file.name.split('.')[-1]}"
        tmp.write_bytes(audio_file.getbuffer())
        input_path, input_kind = tmp, "audio_path"

with tab_video:
    video_file = st.file_uploader(
        "Video stimulus", type=["mp4", "avi", "mkv", "mov", "webm"]
    )
    if video_file is not None:
        st.video(video_file)
    if st.button("Run on video", disabled=video_file is None):
        tmp = Path(tempfile.mkdtemp()) / f"stimulus.{video_file.name.split('.')[-1]}"
        tmp.write_bytes(video_file.getbuffer())
        input_path, input_kind = tmp, "video_path"

if input_path is not None:
    with st.status("Running TRIBE v2 inference…", expanded=True) as status:
        try:
            st.write("Building events dataframe…")
            events = model.get_events_dataframe(**{input_kind: str(input_path)})
            st.write(f"Extracted **{len(events)}** events — running the model…")
            preds, segments = model.predict(events=events, verbose=False)
        except Exception as exc:  # noqa: BLE001
            status.update(label="Inference failed", state="error")
            st.error(str(exc))
            st.stop()
        status.update(label="Inference complete ✅", state="complete")
    st.session_state.preds = preds
    st.session_state.segments = segments
    st.session_state.events = events

# ----------------------------------------------------------------------
# Results
# ----------------------------------------------------------------------

preds = st.session_state.get("preds")
if preds is None:
    st.stop()

events = st.session_state.events
plotter = load_plotter()
norm = norm_percentile if norm_percentile < 100 else None

st.divider()
st.subheader("Results")

c1, c2, c3 = st.columns(3)
c1.metric("Timesteps (TRs)", preds.shape[0])
c2.metric("Cortical vertices", f"{preds.shape[1]:,}")
c3.metric("TR", f"{model.data.TR:g} s")
st.caption(
    f"Predictions are for the *average* subject on the fsaverage5 mesh and are "
    f"offset by {HEMODYNAMIC_LAG_S} s to compensate for the hemodynamic lag."
)

with st.expander("Events dataframe"):
    cols = [c for c in ("type", "start", "duration", "text") if c in events.columns]
    st.dataframe(events[cols].head(200), width="stretch")

# Global activity over time
st.markdown("**Global cortical activity over time**")
fig, ax = plt.subplots(figsize=(10, 2.5))
tr = float(model.data.TR)
time_s = np.arange(preds.shape[0]) * tr - HEMODYNAMIC_LAG_S
ax.plot(time_s, preds.mean(axis=1), color="crimson")
ax.axvline(0, color="grey", ls="--", lw=1)
ax.set_xlabel("Time (s)")
ax.set_ylabel("Mean predicted activity")
ax.spines[["top", "right"]].set_visible(False)
st.pyplot(fig)
plt.close(fig)

# Per-timestep cortical maps
st.markdown("**Predicted cortical map per timestep**")
if preds.shape[0] > 1:
    t = st.slider("Timestep", 0, preds.shape[0] - 1, 0, format="TR %d")
else:
    t = 0
plot_kwargs = dict(
    views=views or ["left", "right"],
    cmap=cmap,
    norm_percentile=norm,
    colorbar=show_colorbar,
)
plotter.plot_surf(preds[t], **plot_kwargs)
fig = plt.gcf()
fig.suptitle(
    f"TR {t} · stimulus time ≈ {time_s[t]:.1f} s", fontsize=12, fontweight="bold"
)
st.pyplot(fig)
plt.close(fig)

# Mean-over-time map
st.markdown("**Mean activity over the whole stimulus**")
plotter.plot_surf(preds.mean(axis=0), **plot_kwargs)
fig = plt.gcf()
fig.suptitle("Mean predicted activity", fontsize=12, fontweight="bold")
st.pyplot(fig)
plt.close(fig)

# Downloads
st.divider()
st.subheader("Download")
buf = io.BytesIO()
np.save(buf, preds)
st.download_button(
    "⬇️ Download predictions (.npy)",
    data=buf.getvalue(),
    file_name="tribev2_predictions.npy",
    mime="application/octet-stream",
    help=f"Array of shape {preds.shape} (n_timesteps, n_vertices, fsaverage5).",
)

with st.expander("🎞️ Export brain-animation video (requires ffmpeg)"):
    st.caption(
        "Renders one brain frame per timestep with "
        "`PlotBrainNilearn.plot_timesteps_mp4` and encodes with ffmpeg."
    )
    fps = st.slider("Interpolated FPS", 1, 30, 10)
    if st.button("Render animation"):
        out_path = Path(tempfile.mkdtemp()) / "tribev2_brain.mp4"
        with st.spinner("Rendering frames and encoding…"):
            try:
                plotter.plot_timesteps_mp4(
                    preds,
                    out_path,
                    segments=st.session_state.segments,
                    interpolated_fps=fps,
                    norm_percentile=norm or 100,
                    views=(views or ["left"])[0],
                )
            except Exception as exc:  # noqa: BLE001
                st.error(f"Video export failed (is ffmpeg installed?): {exc}")
            else:
                st.video(str(out_path))
                st.download_button(
                    "⬇️ Download animation (.mp4)",
                    data=out_path.read_bytes(),
                    file_name="tribev2_brain.mp4",
                    mime="video/mp4",
                )
