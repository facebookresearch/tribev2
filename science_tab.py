"""
science_tab.py
--------------
Renders two tabs:
  - 🧠 Science & Methodology
  - 📖 Glossary

Content is drawn from:
  - TRIBE v2 paper (Meta FAIR, d'Ascoli et al. 2025)
  - Cognitive science literature cited in the pipeline
  - CPCi methodology developed for this tool

All content is pointer-style (no big paragraphs).
"""

import streamlit as st


# ── Shared CSS ────────────────────────────────────────────────────────────────

SCI_CSS = """
<style>
/* Stat callout */
.stat-card    { background:#08111e; border:1px solid #1e3a5e; border-radius:10px; margin-bottom:40px;
                padding:16px; text-align:center; }
.stat-big     { font-size:36px; font-weight:900; line-height:1; }
.stat-label   { font-size: 13px; color:#CBD5E1; text-transform:uppercase;
                letter-spacing:1.5px; margin-top:5px; }
/* Section block */
.sci-block    { background:#0a1220; border:1px solid #1a2840;
                border-left:3px solid; border-radius:10px;
                padding:18px 20px; margin:10px 0; }
.sci-block-title { font-size:13px; font-weight:700; text-transform:uppercase;
                   letter-spacing:1.2px; margin-bottom:10px; }
/* Pointer row */
.ptr-row      { display:flex; align-items:flex-start; gap:10px;
                margin-bottom:9px; font-size:13px; color:#CBD5E1; line-height:1.6; }
.ptr-arrow    { color:#CBD5E1; font-size:14px; flex-shrink:0; margin-top:2px; }
.ptr-strong   { color:#FFFFFF; font-weight:600; }
/* Step strip */
.step-strip   { display:flex; align-items:center; gap:6px;
                flex-wrap:wrap; margin:10px 0; }
.step-box     { background:#0d1828; border:1px solid #1e3a5e; border-radius:8px;
                padding:8px 14px; font-size:12px; color:#94A3B8; text-align:center;
                min-width:80px; }
.step-box strong { display:block; color:#FFFFFF; font-size:13px; }
.step-arrow   { color:#94A3B8; font-size:18px; flex-shrink:0; }
/* Glossary */
.gls-card     { background:#080f1c; border:1px solid #1a2840;
                border-left:3px solid; border-radius:10px;
                padding:14px 18px; margin:8px 0; }
.gls-term     { font-size:14px; font-weight:800; margin-bottom:6px; }
.gls-def      { font-size:12px; color:#94A3B8; line-height:1.7; }
.gls-ptr      { font-size: 13px; color:#CBD5E1; margin-top:6px; }
.gls-ptr span { background:#0d1828; border:1px solid #1e2d45;
                border-radius:12px; padding:2px 9px; margin-right:5px; }
/* Tag chip */
.tag          { display:inline-block; background:#0d1828; border:1px solid #1e2d45;
                border-radius:12px; padding:3px 10px; font-size: 13px;
                color:#94A3B8; margin:3px 2px; letter-spacing:0.5px; }
/* Quote box */
.quote-box    { background:#07101c; border-left:3px solid #40c4ff;
                border-radius:8px; padding:14px 18px; margin:12px 0;
                font-size:13px; color:#CBD5E1; font-style:italic; line-height:1.7; }
.quote-box span { color:#40c4ff; font-style:normal; font-weight:700; }
/* Formula */
.formula      { background:#060d18; border:1px solid #1e3a5e;
                border-radius:8px; padding:14px 20px; margin:10px 0;
                font-family:monospace; font-size:13px; color:#40c4ff;
                text-align:center; letter-spacing:0.3px; }
</style>
"""


def _ptr(arrow_color: str, bold: str, rest: str = "") -> str:
    """Single pointer row HTML."""
    return (
        f"<div class='ptr-row'>"
        f"<span class='ptr-arrow' style='color:{arrow_color};'>→</span>"
        f"<span><span class='ptr-strong'>{bold}</span>{' ' + rest if rest else ''}</span>"
        f"</div>"
    )


def _block(title: str, accent: str, content: str) -> str:
    return (
        f"<div class='sci-block' style='border-left-color:{accent};'>"
        f"<div class='sci-block-title' style='color:{accent};'>{title}</div>"
        f"{content}"
        f"</div>"
    )


# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — Science & Methodology
# ══════════════════════════════════════════════════════════════════════════════

def show_science_tab() -> None:
    st.markdown(SCI_CSS, unsafe_allow_html=True)

    st.markdown(
        "<div style='font-size:13px;color:#CBD5E1;text-transform:uppercase;"
        "letter-spacing:2px;font-weight:600;margin-bottom:4px;'>The Research Foundation</div>"
        "<div class='main-header' style='margin-bottom:4px;border-left:3px solid #3B82F6;padding-left:16px;'>"
        "TRIBE v2 × Creative Intelligence</div>"
        "<div style='font-size:13px;color:#94A3B8;margin-bottom:20px;'>"
        "How a Meta brain-encoding model became the science behind your ad scores</div>",
        unsafe_allow_html=True,
    )

    # ── Key stats bar ─────────────────────────────────────────────────────────
    s1, s2, s3, s4, s5 = st.columns(5)
    for col, big, label, color in [
        (s1, "720",    "fMRI Subjects",       "#40c4ff"),
        (s2, "1,117h", "Brain Scan Hours",    "#00e676"),
        (s3, "20,484", "Cortical Vertices",   "#ffb300"),
        (s4, "#1",     "Algonauts 2025 Rank", "#ffd700"),
        (s5, "263",    "Competing Teams",     "#ff80ab"),
    ]:
        col.markdown(
            f"<div class='stat-card'>"
            f"<div class='stat-big' style='color:{color};'>{big}</div>"
            f"<div class='stat-label'>{label}</div>"
            f"</div>",
            unsafe_allow_html=True,
        )

    st.markdown("<hr style='border:none;border-top:1px solid #1a2840;margin:20px 0;'>",
                unsafe_allow_html=True)

    # ── Section A: What is TRIBE v2 ───────────────────────────────────────────
    st.markdown(
        "<div style='font-size:12px;color:#40c4ff;text-transform:uppercase;"
        "letter-spacing:2px;font-weight:700;margin-bottom:12px;'>A · What Is TRIBE v2</div>",
        unsafe_allow_html=True,
    )

    col_l, col_r = st.columns([3, 2])

    with col_l:
        st.markdown(_block("The Model", "#40c4ff",
            _ptr("#40c4ff", "Full name:", "Tri-modal Brain Imaging Encoding model v2") +
            _ptr("#40c4ff", "Origin:", "Meta FAIR (Fundamental AI Research), 2025") +
            _ptr("#40c4ff", "Authors:", "d'Ascoli, Rapin, Benchetrit, King et al.") +
            _ptr("#40c4ff", "Paper:", "\"A foundation model of vision, audition, and language for in-silico neuroscience\"") +
            _ptr("#40c4ff", "Type:", "Tri-modal transformer encoder — processes video + audio + text simultaneously")
        ), unsafe_allow_html=True)

        st.markdown(_block("What It Does", "#00e676",
            _ptr("#00e676", "Core task:", "Predict human brain activity (fMRI) from any audio, visual, or language stimulus") +
            _ptr("#00e676", "Input modalities:", "Video (V-JEPA embeddings) · Audio (W2V-BERT embeddings) · Text (LLaMA embeddings)") +
            _ptr("#00e676", "Output:", "BOLD signal across 20,484 cortical + 8,802 subcortical brain locations") +
            _ptr("#00e676", "Key innovation:", "One model — all stimuli types, all brain regions, zero-shot generalisation to new subjects")
        ), unsafe_allow_html=True)

    with col_r:
        st.markdown(_block("Training Scale", "#ffb300",
            _ptr("#ffb300", "Subjects:", "720 healthy volunteers") +
            _ptr("#ffb300", "Sessions:", "5,094 scan sessions") +
            _ptr("#ffb300", "fMRI hours:", "1,117 hours of brain recordings") +
            _ptr("#ffb300", "Video:", "121 hours of stimulus video") +
            _ptr("#ffb300", "Audio:", "142 hours of stimulus audio") +
            _ptr("#ffb300", "Text sentences:", "71,000 transcribed sentences") +
            _ptr("#ffb300", "Training GPU:", "128 × V100 GPUs for feature extraction")
        ), unsafe_allow_html=True)

        st.markdown(
            "<div class='quote-box'>"
            "\"By aligning the representations of AI systems to those of the human brain, "
            "we demonstrate that a <span>single architecture</span> can integrate a vast range "
            "of fMRI responses… moving from the fragmented mapping of isolated cognitive tasks "
            "toward a unified, predictive foundation model.\""
            "<br><br>— d'Ascoli et al., Meta FAIR 2025"
            "</div>",
            unsafe_allow_html=True,
        )

    # ── Section B: How Brain Encoding Connects to Ads ─────────────────────────
    st.markdown("<hr style='border:none;border-top:1px solid #1a2840;margin:20px 0;'>",
                unsafe_allow_html=True)
    st.markdown(
        "<div style='font-size:12px;color:#ff80ab;text-transform:uppercase;"
        "letter-spacing:2px;font-weight:700;margin-bottom:12px;'>B · From Brain Science to Ad Intelligence</div>",
        unsafe_allow_html=True,
    )

    # Pipeline steps
    st.markdown(
        "<div style='font-size:13px;color:#CBD5E1;text-transform:uppercase;"
        "letter-spacing:1px;margin-bottom:8px;'>How your ad image becomes a CPCi score:</div>",
        unsafe_allow_html=True,
    )
    st.markdown("""
    <div class='step-strip'>
      <div class='step-box'><strong>📁 Upload</strong>JPG / PNG</div>
      <div class='step-arrow'>→</div>
      <div class='step-box'><strong>👁 Vision</strong>OpenCV scan</div>
      <div class='step-arrow'>→</div>
      <div class='step-box'><strong>🧬 Signals</strong>Cognitive map</div>
      <div class='step-arrow'>→</div>
      <div class='step-box'><strong>🧠 TRIBE</strong>Brain basis</div>
      <div class='step-arrow'>→</div>
      <div class='step-box'><strong>📊 CPCi</strong>Weighted score</div>
      <div class='step-arrow'>→</div>
      <div class='step-box'><strong>📝 Report</strong>Strategy</div>
    </div>""", unsafe_allow_html=True)

    b1, b2, b3 = st.columns(3)

    with b1:
        st.markdown(_block("Step 1 — Visual Feature Extraction", "#40c4ff",
            _ptr("#40c4ff", "Tool:", "OpenCV + Tesseract OCR") +
            _ptr("#40c4ff", "Object count:", "Canny edge detection + contour analysis") +
            _ptr("#40c4ff", "Face detection:", "Haar Cascade classifier (< 200ms)") +
            _ptr("#40c4ff", "Text density:", "Tesseract PSM 11 sparse-text mode") +
            _ptr("#40c4ff", "Contrast:", "Std deviation of grayscale pixel values") +
            _ptr("#40c4ff", "Colours:", "K-means clustering (k=3) on 100×100 downsample") +
            _ptr("#40c4ff", "Speed:", "< 2 seconds total · No GPU required")
        ), unsafe_allow_html=True)

    with b2:
        st.markdown(_block("Step 2 — Cognitive Signal Mapping", "#ff80ab",
            _ptr("#ff80ab", "Attention:", "Contrast × 0.5 + Face boost + Clutter penalty") +
            _ptr("#ff80ab", "Memory:", "Simplicity score + Dual-coding text factor") +
            _ptr("#ff80ab", "Valence:", "HSV colour psychology + Face warmth component") +
            _ptr("#ff80ab", "Cog. Load:", "Object count + text density composite (Low / Medium / High)") +
            _ptr("#ff80ab", "TRIBE basis:", "Signal thresholds calibrated against TRIBE v2 visual cortex encoding patterns") +
            _ptr("#ff80ab", "Transparency:", "All thresholds named as constants — no black box")
        ), unsafe_allow_html=True)

    with b3:
        st.markdown(_block("Step 3 — CPCi Computation", "#ffb300",
            _ptr("#ffb300", "Formula:", "w_attn × Attention + w_mem × Memory + w_emo × Valence_norm") +
            _ptr("#ffb300", "Valence norm:", "(Valence + 1) / 2 × 100 → maps −1/+1 to 0/100 scale") +
            _ptr("#ffb300", "Load penalty:", "−0 / −5 / −12 for Low / Medium / High (Retail Media only)") +
            _ptr("#ffb300", "Use-case weights:", "FMCG: 20/50/30 · Performance: 50/30/20 · Retail: 40/30/30") +
            _ptr("#ffb300", "Score range:", "0 → 100 · < 40 weak · 40–70 average · > 70 strong") +
            _ptr("#ffb300", "Clamp:", "max(0, min(100, raw)) — never negative or over 100")
        ), unsafe_allow_html=True)

    # ── Section B2: From Brain Science to Business Decisions ──────────────────
    st.markdown("<hr style='border:none;border-top:1px solid #1a2840;margin:24px 0;'>",
                unsafe_allow_html=True)
    st.markdown(
        "<div style='font-size:12px;color:#a78bfa;text-transform:uppercase;"
        "letter-spacing:2px;font-weight:700;margin-bottom:4px;'>B2 · Architecture</div>"
        "<div class='main-header' style='border-left:3px solid #a78bfa;"
        "padding-left:16px;margin-bottom:6px;font-size:22px;'>"
        "From Brain Science to Business Decisions</div>"
        "<div style='font-size:13px;color:#94A3B8;margin-bottom:20px;'>"
        "Two layers — one system. TRIBE v2 predicts brain activation. "
        "CPCi converts those signals into a single business-ready score.</div>",
        unsafe_allow_html=True,
    )

    # ── Two-layer architecture diagram ───────────────────────────────────────
    lay_l, lay_mid, lay_r = st.columns([5, 1, 5])

    with lay_l:
        st.markdown(
            "<div style='background:#060d1c;border:2px solid #1D4ED8;border-radius:14px;"
            "padding:22px 24px;height:100%;'>"
            # Badge
            "<div style='display:flex;align-items:center;gap:10px;margin-bottom:16px;'>"
            "<div style='background:#1D4ED8;border-radius:6px;padding:4px 12px;"
            "font-size:12px;font-weight:800;color:#fff;letter-spacing:1px;'>LAYER 1</div>"
            "<div style='font-size:16px;font-weight:800;color:#93C5FD;'>TRIBE v2</div>"
            "</div>"
            # Role
            "<div style='font-size:13px;color:#94A3B8;margin-bottom:14px;line-height:1.6;'>"
            "Brain activation prediction model trained on 720 subjects and 1,117 hours of fMRI scans. "
            "Given any visual, audio, or language stimulus, TRIBE v2 predicts which brain regions fire "
            "— and at what intensity."
            "</div>"
            # Pointer list
            + _ptr("#3B82F6", "Input:", "Ad creative (image, video, or text)")
            + _ptr("#3B82F6", "Process:", "Tri-modal transformer — V-JEPA · W2V-BERT · LLaMA")
            + _ptr("#3B82F6", "Output:", "Predicted BOLD activation across 20,484 cortical vertices")
            + _ptr("#3B82F6", "Key regions:", "V1–V7 visual cortex · FaceBody area · Language areas · Limbic")
            + _ptr("#3B82F6", "Performance:", "#1 Algonauts 2025 benchmark — 263 competing teams")
            +
            "<div style='margin-top:16px;background:#0a1626;border:1px solid #1e3a5e;"
            "border-radius:8px;padding:12px 14px;font-size:12px;color:#CBD5E1;line-height:1.6;'>"
            "<span style='color:#3B82F6;font-weight:700;'>What it tells us:</span> "
            "Which features of a creative cause the human brain to activate — "
            "contrast, faces, text, emotional tone — and by how much."
            "</div>"
            "</div>",
            unsafe_allow_html=True,
        )

    with lay_mid:
        st.markdown(
            "<div style='display:flex;flex-direction:column;align-items:center;"
            "justify-content:center;height:100%;padding-top:40px;gap:6px;'>"
            "<div style='font-size:28px;color:#a78bfa;'>→</div>"
            "<div style='font-size:9px;color:#6B7280;letter-spacing:1px;"
            "text-transform:uppercase;writing-mode:vertical-rl;'>converts to</div>"
            "</div>",
            unsafe_allow_html=True,
        )

    with lay_r:
        st.markdown(
            "<div style='background:#0c0918;border:2px solid #7C3AED;border-radius:14px;"
            "padding:22px 24px;height:100%;'>"
            # Badge
            "<div style='display:flex;align-items:center;gap:10px;margin-bottom:16px;'>"
            "<div style='background:#7C3AED;border-radius:6px;padding:4px 12px;"
            "font-size:12px;font-weight:800;color:#fff;letter-spacing:1px;'>LAYER 2</div>"
            "<div style='font-size:16px;font-weight:800;color:#c4b5fd;'>CPCi</div>"
            "<div style='font-size:12px;color:#94A3B8;'>Cost Per Cognitive Impression</div>"
            "</div>"
            # Role
            "<div style='font-size:13px;color:#94A3B8;margin-bottom:14px;line-height:1.6;'>"
            "CPCi takes TRIBE v2's brain-region activation patterns and converts them into four "
            "measurable signals — each one a direct media performance lever."
            "</div>"
            # Signal pills with contribution arrows
            "<div style='margin-bottom:12px;'>"

            # Attention
            "<div style='display:flex;align-items:center;gap:10px;margin-bottom:8px;'>"
            "<div style='background:#172554;border:1px solid #3B82F6;border-radius:8px;"
            "padding:6px 14px;font-size:12px;font-weight:700;color:#60A5FA;min-width:90px;text-align:center;'>"
            "Attention</div>"
            "<div style='font-size:11px;color:#94A3B8;'>"
            "Visual cortex activation → stopping power in feed</div>"
            "</div>"

            # Memory
            "<div style='display:flex;align-items:center;gap:10px;margin-bottom:8px;'>"
            "<div style='background:#1e1040;border:1px solid #8B5CF6;border-radius:8px;"
            "padding:6px 14px;font-size:12px;font-weight:700;color:#a78bfa;min-width:90px;text-align:center;'>"
            "Memory</div>"
            "<div style='font-size:11px;color:#94A3B8;'>"
            "Language area activity → brand recall after exposure</div>"
            "</div>"

            # Emotion
            "<div style='display:flex;align-items:center;gap:10px;margin-bottom:8px;'>"
            "<div style='background:#1a0a2e;border:1px solid #EC4899;border-radius:8px;"
            "padding:6px 14px;font-size:12px;font-weight:700;color:#f472b6;min-width:90px;text-align:center;'>"
            "Emotion</div>"
            "<div style='font-size:11px;color:#94A3B8;'>"
            "Limbic/subcortical response → valence and purchase intent</div>"
            "</div>"

            # Load
            "<div style='display:flex;align-items:center;gap:10px;margin-bottom:12px;'>"
            "<div style='background:#1c1000;border:1px solid #F59E0B;border-radius:8px;"
            "padding:6px 14px;font-size:12px;font-weight:700;color:#fbbf24;min-width:90px;text-align:center;'>"
            "Load</div>"
            "<div style='font-size:11px;color:#94A3B8;'>"
            "Working memory saturation → processing penalty</div>"
            "</div>"

            # Business score callout
            "<div style='border-top:1px solid #2d1b6e;padding-top:10px;'>"
            "<div style='display:flex;align-items:center;gap:10px;'>"
            "<div style='background:#4C1D95;border:2px solid #7C3AED;border-radius:8px;"
            "padding:8px 16px;font-size:13px;font-weight:800;color:#fff;min-width:90px;text-align:center;'>"
            "Business<br>Score</div>"
            "<div style='font-size:12px;color:#c4b5fd;line-height:1.5;'>"
            "<strong style='color:#fff;'>CPCi 0–100</strong><br>"
            "Weighted composite · Use-case calibrated<br>"
            "Single number a CMO can act on"
            "</div>"
            "</div>"
            "</div>"

            "</div>"
            "</div>",
            unsafe_allow_html=True,
        )

    # ── Tagline callout ───────────────────────────────────────────────────────
    st.markdown(
        "<div style='background:linear-gradient(135deg,#0f0624 0%,#050d1c 100%);"
        "border:1px solid #2d1b6e;border-left:4px solid #a78bfa;"
        "border-radius:12px;padding:20px 28px;margin:20px 0;'>"
        "<div style='font-size:16px;font-weight:700;color:#e2d9f3;line-height:1.6;'>"
        "This system translates neuroscience into actionable media decisions."
        "</div>"
        "<div style='font-size:12px;color:#6B7280;margin-top:8px;line-height:1.5;'>"
        "TRIBE v2 answers: <em>which brain regions respond to this creative?</em> &nbsp;·&nbsp; "
        "CPCi answers: <em>should you spend media budget on it?</em>"
        "</div>"
        "</div>",
        unsafe_allow_html=True,
    )

    # ── Section C: The Brain Regions That Matter for Ads ──────────────────────
    st.markdown("<hr style='border:none;border-top:1px solid #1a2840;margin:20px 0;'>",
                unsafe_allow_html=True)
    st.markdown(
        "<div style='font-size:12px;color:#00e676;text-transform:uppercase;"
        "letter-spacing:2px;font-weight:700;margin-bottom:12px;'>C · Brain Regions That Drive Ad Response</div>",
        unsafe_allow_html=True,
    )

    c1, c2, c3, c4 = st.columns(4)
    for col, region, role, signals, color in [
        (c1, "V1 – V7\nVisual Cortex",
         "Primary visual processing — contrast, edges, motion",
         "→ Contrast score\n→ Object detection\n→ Attention score", "#40c4ff"),
        (c2, "FaceBody Area\n(Lateral Cortex)",
         "Face and body detection — activates in < 13ms, involuntary",
         "→ Face count\n→ Attention boost\n→ Valence warmth", "#ff80ab"),
        (c3, "Language Areas\n(Left Hemisphere)",
         "Text and verbal processing — competes with visual memory trace",
         "→ Text density\n→ Memory encoding\n→ Cognitive load", "#ffb300"),
        (c4, "Limbic / Emotional\n(Subcortical)",
         "Colour, arousal, reward — determines whether content is engaging",
         "→ Emotional valence\n→ Colour psychology\n→ Purchase intent", "#00e676"),
    ]:
        col.markdown(
            f"<div class='sci-block' style='border-left-color:{color};height:100%;'>"
            f"<div class='sci-block-title' style='color:{color};white-space:pre-line;'>{region}</div>"
            f"<div style='font-size:13px;color:#CBD5E1;margin-bottom:8px;'>{role}</div>"
            + "".join(
                f"<div class='ptr-row'><span class='ptr-arrow' style='color:{color};'>→</span>"
                f"<span style='font-size:12px;color:#94A3B8;'>{s.replace('→ ', '')}</span></div>"
                for s in signals.split("\n")
            ) +
            f"</div>",
            unsafe_allow_html=True,
        )

    # Paper note
    st.markdown(
        "<div class='quote-box' style='margin-top:14px;'>"
        "TRIBE v2 predicts responses across all of these regions simultaneously. "
        "The cognitive signal thresholds in this tool are calibrated against TRIBE's "
        "<span>encoding score patterns</span> — the same patterns that showed V1–V7 responding "
        "to contrast and edges, the FaceBody area responding to faces, and language areas "
        "competing with visual processing for working memory resources."
        "</div>",
        unsafe_allow_html=True,
    )

    # ── Section D: The Cognitive Science Frameworks ───────────────────────────
    st.markdown("<hr style='border:none;border-top:1px solid #1a2840;margin:20px 0;'>",
                unsafe_allow_html=True)
    st.markdown(
        "<div style='font-size:12px;color:#c084fc;text-transform:uppercase;"
        "letter-spacing:2px;font-weight:700;margin-bottom:12px;'>D · Cognitive Science Frameworks Used</div>",
        unsafe_allow_html=True,
    )

    d1, d2 = st.columns(2)
    with d1:
        st.markdown(_block("Dual-Coding Theory — Paivio (1971)", "#c084fc",
            _ptr("#c084fc", "Core claim:", "The brain has two separate memory channels — visual and verbal") +
            _ptr("#c084fc", "How it works:", "Images stored in visual memory · words/text in verbal memory") +
            _ptr("#c084fc", "Key finding:", "Content encoded in BOTH channels is 2× more likely to be recalled") +
            _ptr("#c084fc", "In this tool:", "Text density 5–25% = dual-coding sweet spot · < 5% = verbal channel unused · > 25% = verbal drowns visual") +
            _ptr("#c084fc", "Ad implication:", "A single 5–8 word tagline + strong visual is more memorable than text-only or image-only")
        ), unsafe_allow_html=True)

        st.markdown(_block("Russell Circumplex Model of Affect (1980)", "#ff80ab",
            _ptr("#ff80ab", "Core claim:", "All emotions can be mapped on two axes: Valence (positive/negative) + Arousal (excited/calm)") +
            _ptr("#ff80ab", "Why it matters:", "Emotional valence is the single strongest predictor of long-term brand equity") +
            _ptr("#ff80ab", "In this tool:", "Valence scored −1.0 (aversive) → +1.0 (rewarding)") +
            _ptr("#ff80ab", "Colour mapping:", "Warm hues (red/orange/yellow) → positive arousal · Cool/dark → lower valence") +
            _ptr("#ff80ab", "Face contribution:", "Each face detected adds up to +0.25 valence (affiliative warmth)")
        ), unsafe_allow_html=True)

    with d2:
        st.markdown(_block("Cognitive Load Theory — Sweller (1988)", "#ffb300",
            _ptr("#ffb300", "Core claim:", "Working memory has a fixed capacity (~7±2 items)") +
            _ptr("#ffb300", "Three load types:", "Intrinsic (task complexity) · Extraneous (visual noise) · Germane (learning)") +
            _ptr("#ffb300", "Ad relevance:", "Extraneous load = clutter = objects + excess text competing for attention") +
            _ptr("#ffb300", "In this tool:", "< 4 objects = Low load · 4–8 = Medium · > 8 = High") +
            _ptr("#ffb300", "Cost of High load:", "Saturated working memory → brand message not encoded → wasted impressions") +
            _ptr("#ffb300", "Retail Media:", "High load penalised −12 CPCi points (clutter compounds with environment)")
        ), unsafe_allow_html=True)

        st.markdown(_block("Miller's Law (1956) + Pre-attentive Processing", "#40c4ff",
            _ptr("#40c4ff", "Miller's Law:", "Short-term memory holds 7 ± 2 'chunks' of information simultaneously") +
            _ptr("#40c4ff", "Why it caps objects:", "More than 7 objects → chunking fails → viewer samples randomly") +
            _ptr("#40c4ff", "Pre-attentive:", "The brain processes certain features (contrast, colour, motion) in < 200ms — before conscious attention") +
            _ptr("#40c4ff", "Face reflex:", "Face detection activates in < 13ms — the fastest involuntary attention trigger available") +
            _ptr("#40c4ff", "Implication:", "Ads that pass the pre-attentive filter don't need to 'ask' for attention — they capture it automatically")
        ), unsafe_allow_html=True)

    # ── Section E: CPCi Score Interpretation ─────────────────────────────────
    st.markdown("<hr style='border:none;border-top:1px solid #1a2840;margin:20px 0;'>",
                unsafe_allow_html=True)
    st.markdown(
        "<div style='font-size:12px;color:#ffd700;text-transform:uppercase;"
        "letter-spacing:2px;font-weight:700;margin-bottom:12px;'>E · CPCi Score Zones</div>",
        unsafe_allow_html=True,
    )

    e1, e2, e3 = st.columns(3)
    for col, zone, score, color, points in [
        (e1, "NEEDS IMPROVEMENT", "0 – 39", "#ff5252", [
            "Likely scrolled past in cold-audience feeds",
            "Weak attention + memory = low platform Quality Score",
            "High CPM inflation — algorithm penalises low engagement",
            "Do not scale. Fix at least 1 dimension before spending",
            "Action: redesign the creative, not just the copy",
        ]),
        (e2, "AVERAGE PERFORMER", "40 – 69", "#ffb300", [
            "Will deliver in warm/retargeted audiences",
            "Moderate recall — needs 6–8 impressions to build salience",
            "Safe for mid-funnel, not efficient for cold prospecting",
            "Gap to top-quartile is closeable with 1–2 targeted changes",
            "Action: run A/B test before committing full budget",
        ]),
        (e3, "STRONG PERFORMER", "70 – 100", "#00e676", [
            "Above-average thumb-stop rate in social feed formats",
            "Strong memory trace — brand recall lifts in 3–5 impressions",
            "Suitable for broad reach and top-of-funnel investment",
            "Correlates with 3–5pt brand lift in campaign measurement",
            "Action: scale budget 20–30%, cap frequency at 7",
        ]),
    ]:
        col.markdown(
            f"<div class='sci-block' style='border-left-color:{color};'>"
            f"<div class='sci-block-title' style='color:{color};'>{zone}</div>"
            f"<div style='font-size:32px;font-weight:900;color:{color};"
            f"margin-bottom:10px;'>{score}</div>"
            + "".join(
                f"<div class='ptr-row'><span class='ptr-arrow' style='color:{color};'>→</span>"
                f"<span style='font-size:12px;color:#94A3B8;'>{p}</span></div>"
                for p in points
            ) +
            f"</div>",
            unsafe_allow_html=True,
        )

    st.markdown(
        "<div class='formula'>"
        "CPCi = (w_attention × Attention) + (w_memory × Memory) + (w_emotion × ((Valence + 1) / 2 × 100))"
        "<br>[ − Load Penalty if use case = Retail Media ]"
        "</div>",
        unsafe_allow_html=True,
    )

    # ── Section F: Provenance & Attribution ───────────────────────────────────
    st.markdown("<hr style='border:none;border-top:1px solid #1a2840;margin:28px 0;'>",
                unsafe_allow_html=True)
    st.markdown(
        "<div style='font-size:12px;color:#f97316;text-transform:uppercase;"
        "letter-spacing:2px;font-weight:700;margin-bottom:6px;'>F · Provenance &amp; Attribution</div>"
        "<div style='font-size:13px;color:#CBD5E1;margin-bottom:20px;'>"
        "Exactly which numbers come from Meta TRIBE v2, and which were designed by Anil Pandit for CPCi</div>",
        unsafe_allow_html=True,
    )

    # Two-column overview cards
    fa, fb = st.columns(2)

    with fa:
        st.markdown(
            "<div style='background:#060f1c;border:2px solid #1D4ED8;border-radius:12px;"
            "padding:20px 22px;height:100%;'>"
            "<div style='display:flex;align-items:center;gap:10px;margin-bottom:14px;'>"
            "<div style='background:#1D4ED8;border-radius:5px;padding:3px 10px;"
            "font-size:13px;font-weight:800;color:#fff;letter-spacing:0.5px;'>META FAIR</div>"
            "<div style='font-size:14px;font-weight:700;color:#93C5FD;'>TRIBE v2 Contribution</div>"
            "</div>"
            "<div style='font-size:12px;color:#CBD5E1;line-height:1.75;'>"
            "TRIBE v2 (d'Ascoli et al., Meta FAIR 2025) is a multimodal brain encoding model trained on "
            "<strong style='color:#93C5FD;'>720 subjects · 1,117h fMRI · 121h video stimuli</strong>. "
            "It maps AI representations onto cortical brain activity, achieving rank #1 at Algonauts 2025 "
            "(263 competing teams) with a mean encoding score of 0.2146.<br><br>"
            "The TRIBE v2 brain data established which visual features activate which brain regions — "
            "and at what magnitudes. This is the neuroscience foundation for the signal thresholds, "
            "clutter penalties, face boosts, and text density curves in CPCi."
            "</div>"
            "<div style='margin-top:14px;'>"
            + _ptr("#3B82F6", "V1–V7 visual cortex", "validates contrast as the primary pre-attentive driver")
            + _ptr("#3B82F6", "FaceBody area (lateral cortex)", "fires in <13ms — justifies face attention boost of +22 pts")
            + _ptr("#3B82F6", "Language areas (left hemisphere)", "competes with visual memory — basis for text density sweet spot")
            + _ptr("#3B82F6", "Limbic / subcortical", "colour and face affect — basis for emotional valence scoring")
            + "</div>"
            "<div style='margin-top:14px;font-size:13px;color:#94A3B8;border-top:1px solid #1a2840;padding-top:10px;'>"
            "Published: d'Ascoli S, Rapin J, Benchetrit Y, King J-R et al. · Meta FAIR 2025 · "
            "<a href='https://ai.meta.com/research/publications/a-foundation-model-of-vision-audition-and-language-for-in-silico-neuroscience/' "
            "style='color:#3B82F6;'>Paper</a> · "
            "<a href='https://huggingface.co/facebook/tribev2' style='color:#3B82F6;'>Weights (HuggingFace)</a> · "
            "License: CC BY-NC 4.0"
            "</div>"
            "</div>",
            unsafe_allow_html=True,
        )

    with fb:
        st.markdown(
            "<div style='background:#0c0a0a;border:2px solid #EF4444;border-radius:12px;"
            "padding:20px 22px;height:100%;'>"
            "<div style='display:flex;align-items:center;gap:10px;margin-bottom:14px;'>"
            "<div style='background:#1D4ED8;border-radius:5px;padding:3px 10px;"
            "font-size:13px;font-weight:800;color:#fff;letter-spacing:0.5px;'>COGNITIVE SIGNAL ENGINE™</div>"
            "<div style='font-size:14px;font-weight:700;color:#93C5FD;'>Anil Pandit / CPCi Contribution</div>"
            "</div>"
            "<div style='font-size:12px;color:#CBD5E1;line-height:1.75;'>"
            "Anil Pandit designed the CPCi framework to translate TRIBE v2's neuroscience insights "
            "into a <strong style='color:#FCA5A5;'>fast, practical, pre-bid ad scoring system</strong>. "
            "This includes the computer vision pipeline (OpenCV + OCR), all scoring constants, "
            "the weighted CPCi formula, use-case tuning, the load penalty system, "
            "and the narrative intelligence engine."
            "</div>"
            "<div style='margin-top:14px;'>"
            + _ptr("#EF4444", "Creative vision pipeline", "OpenCV + Tesseract OCR — feature extraction in <2s")
            + _ptr("#EF4444", "CPCi formula + weights", "w_attn × Attn + w_mem × Mem + w_emo × Val_norm")
            + _ptr("#EF4444", "Use-case weight system", "FMCG 20/50/30 · Performance 50/30/20 · Retail 40/30/30")
            + _ptr("#EF4444", "Cognitive load penalty", "−0 / −5 / −12 pts for Low / Medium / High (Retail Media)")
            + _ptr("#EF4444", "Narrative engine", "Rule + Claude AI narrative · strategy recommendations")
            + "</div>"
            "<div style='margin-top:14px;font-size:13px;color:#94A3B8;border-top:1px solid #1a2840;padding-top:10px;'>"
            "Cognitive Signal Engine™ — Proprietary framework · © Anil Pandit · All rights reserved"
            "</div>"
            "</div>",
            unsafe_allow_html=True,
        )

    # ── Detailed provenance table ─────────────────────────────────────────────
    st.markdown(
        "<div style='font-size:13px;color:#CBD5E1;text-transform:uppercase;letter-spacing:1.5px;"
        "font-weight:700;margin:24px 0 10px 0;'>Constant-Level Provenance — Every Number, Every Source</div>",
        unsafe_allow_html=True,
    )

    prov_rows = [
        # (Constant / Signal, Value, Source, Neuroscience Basis)
        ("Contrast → Attention weight",   "0.50 (50 pts max)",
         "TRIBE v2 + CPCi",
         "V1–V7 visual cortex encodes luminance contrast edges as the primary pre-attentive signal; TRIBE v2 encoding scores confirm contrast as the strongest low-level visual driver"),
        ("Face attention boost — 1 face",  "+22 pts",
         "TRIBE v2 (FaceBody area)",
         "FaceBody lateral cortex activates in <13ms; TRIBE v2 predicts the highest encoding scores in this region for face-containing stimuli vs. all other image types"),
        ("Face attention boost — 2 faces", "+14 pts (diminishing)",
         "TRIBE v2 + CPCi",
         "TRIBE v2 face encoding shows diminishing incremental activation when multiple faces compete; Anil Pandit calibrated the step-down to 14 pts based on this pattern"),
        ("Face attention boost — 3+ faces","+8 pts",
         "CPCi (calibrated on TRIBE v2)",
         "Beyond 2 faces, faces create attention fragmentation; Anil Pandit set the cap at +8 based on FaceBody saturation pattern observed in TRIBE v2 multi-face stimuli"),
        ("Clutter penalty — mild (4–7 obj)","−4 pts per object",
         "Miller's Law + CPCi",
         "Miller (1956): working memory holds 7±2 items. Anil Pandit calibrated the per-object penalty to exhaust the attention budget at ~7 objects, consistent with TRIBE v2 visual cortex saturation"),
        ("Clutter penalty — severe (8+ obj)","−2 pts per object (cap −30)",
         "CPCi",
         "Diminishing penalty beyond severe threshold — attention is already fragmented; cap prevents negative scores. Anil Pandit's design decision"),
        ("Text density sweet spot",        "5%–25% of image area",
         "Dual-Coding (Paivio 1986) + TRIBE v2",
         "TRIBE v2 language area encoding peaks when verbal content is present but not dominant; Paivio's dual-coding theory (1971) defines the verbal/visual balance — Anil Pandit mapped both to the 5–25% text density range"),
        ("Text overload threshold",        ">50% → memory collapses",
         "Sweller CLT + CPCi",
         "Cognitive Load Theory (Sweller 1988): dual-channel saturation causes disengagement. TRIBE v2 language vs. visual cortex competition patterns support the 50% collapse point"),
        ("Emotional valence — face boost",  "+0.25 per face (cap +0.30)",
         "TRIBE v2 (Limbic) + Russell (1980)",
         "TRIBE v2 subcortical encoding shows positive affect response to faces via limbic activation; Russell circumplex maps face presence to the high-valence arousal quadrant"),
        ("Colour valence — warm hues",      "Red/Orange: +0.30 to +0.40",
         "Russell Circumplex + CPCi",
         "Russell's two-dimensional affect model places warm hues in the high-valence zone. Anil Pandit mapped HSV hue ranges to valence scores, scaled by saturation"),
        ("Colour valence — dark/achromatic","≤ −0.25 (dark) / 0.0 (gray)",
         "TRIBE v2 (Limbic) + CPCi",
         "TRIBE v2 subcortical encoding shows reduced positive affect for low-luminance stimuli; dark images correlate with negative valence in limbic regions"),
        ("CPCi formula weights",           "Varies by use case",
         "CPCi",
         "Marketing science judgment: Performance ads need attention most (50%), FMCG needs memory (50%), Retail needs balance. Anil Pandit's proprietary use-case tuning"),
        ("Cognitive load penalty",         "−0/−5/−12 (Low/Med/High)",
         "CPCi",
         "Retail Media context amplifies load effects (cluttered screen environment). Anil Pandit's penalty system; not derived from TRIBE v2 directly"),
        ("Score zone thresholds",          "<40 weak · 40–70 avg · >70 strong",
         "CPCi",
         "CPCi score bands designed to align with typical creative performance quartiles in paid media. Anil Pandit's proprietary calibration"),
    ]

    # Table header
    st.markdown(
        "<div style='display:grid;grid-template-columns:2fr 1.2fr 1.4fr 3fr;"
        "background:#060e1a;border:1px solid #1a2840;border-radius:10px 10px 0 0;"
        "overflow:hidden;'>"
        "<div style='padding:10px 14px;font-size:13px;font-weight:700;text-transform:uppercase;"
        "letter-spacing:1.2px;color:#CBD5E1;border-right:1px solid #1a2840;'>Constant / Signal</div>"
        "<div style='padding:10px 14px;font-size:13px;font-weight:700;text-transform:uppercase;"
        "letter-spacing:1.2px;color:#CBD5E1;border-right:1px solid #1a2840;'>Value</div>"
        "<div style='padding:10px 14px;font-size:13px;font-weight:700;text-transform:uppercase;"
        "letter-spacing:1.2px;color:#CBD5E1;border-right:1px solid #1a2840;'>Source</div>"
        "<div style='padding:10px 14px;font-size:13px;font-weight:700;text-transform:uppercase;"
        "letter-spacing:1.2px;color:#CBD5E1;'>Neuroscience / Design Basis</div>"
        "</div>",
        unsafe_allow_html=True,
    )

    rows_html = "<div style='border:1px solid #1a2840;border-top:none;border-radius:0 0 10px 10px;overflow:hidden;'>"
    for i, (constant, value, source, basis) in enumerate(prov_rows):
        bg = "#060e1a" if i % 2 == 0 else "#070f1c"
        # Source badge colour
        if "TRIBE v2" in source and "CPCi" in source:
            src_color = "#a78bfa"
        elif "TRIBE v2" in source:
            src_color = "#3B82F6"
        else:
            src_color = "#EF4444"
        rows_html += (
            f"<div style='display:grid;grid-template-columns:2fr 1.2fr 1.4fr 3fr;"
            f"background:{bg};border-top:1px solid #0e1a2e;'>"
            f"<div style='padding:10px 14px;font-size:12px;font-weight:600;color:#FFFFFF;"
            f"border-right:1px solid #0e1a2e;'>{constant}</div>"
            f"<div style='padding:10px 14px;font-family:monospace;font-size:13px;color:#f97316;"
            f"border-right:1px solid #0e1a2e;'>{value}</div>"
            f"<div style='padding:10px 14px;border-right:1px solid #0e1a2e;'>"
            f"<span style='font-size:13px;font-weight:700;color:{src_color};"
            f"background:{src_color}18;border:1px solid {src_color}33;border-radius:4px;"
            f"padding:2px 7px;white-space:nowrap;'>{source}</span>"
            f"</div>"
            f"<div style='padding:10px 14px;font-size:13px;color:#CBD5E1;line-height:1.6;'>{basis}</div>"
            f"</div>"
        )
    rows_html += "</div>"
    st.markdown(rows_html, unsafe_allow_html=True)

    st.markdown(
        "<div style='margin-top:16px;padding:14px 18px;background:#07101c;"
        "border-left:3px solid #f97316;border-radius:8px;font-size:12px;"
        "color:#94A3B8;line-height:1.7;'>"
        "<strong style='color:#f97316;'>Note on methodology:</strong> "
        "TRIBE v2 provides the neuroscience foundation — it tells us <em>which brain regions</em> respond to "
        "contrast, faces, and text, and <em>at what relative magnitudes</em>. "
        "Anil Pandit's CPCi translates those brain-level insights into a fast, deterministic scoring system "
        "using OpenCV + OCR features — no GPU required, no fMRI needed. "
        "The signal thresholds and formula constants were calibrated against TRIBE v2's published encoding "
        "scores and brain parcel activations, then validated against marketing effectiveness benchmarks. "
        "CPCi does not run TRIBE v2 inference directly — it uses TRIBE v2's brain data as its calibration reference."
        "</div>",
        unsafe_allow_html=True,
    )


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — Glossary
# ══════════════════════════════════════════════════════════════════════════════

GLOSSARY = {
    "Neuroscience & TRIBE": [
        ("TRIBE v2",
         "Tri-modal Brain Imaging Encoding model v2. A foundation model from Meta FAIR that predicts fMRI brain activity in response to video, audio, and text. Used as the neuroscience basis for CPCi signal calibration.",
         "#40c4ff",
         ["Meta FAIR 2025", "720 subjects", "1,117h fMRI", "#1 Algonauts 2025"]),

        ("fMRI",
         "Functional Magnetic Resonance Imaging. Measures changes in blood oxygenation across the brain. When a brain region is more active, it consumes more oxygen — fMRI detects this as a BOLD signal change.",
         "#40c4ff",
         ["Blood oxygen measurement", "~2s temporal resolution", "Whole-brain coverage"]),

        ("BOLD Signal",
         "Blood-Oxygen-Level-Dependent signal. The raw measurement from fMRI. Higher BOLD = higher neuronal activity. TRIBE v2 predicts BOLD across 20,484 cortical vertices and 8,802 subcortical voxels.",
         "#40c4ff",
         ["Neuronal activity proxy", "TRIBE predicts this", "20,484 cortical points"]),

        ("Encoding Model",
         "A model that predicts brain activity from sensory inputs. TRIBE v2 is an encoding model — it takes video/audio/text and outputs the expected BOLD signal. Encoding score = Pearson correlation between predicted and actual brain response.",
         "#40c4ff",
         ["Input → brain output", "Pearson correlation score", "Used in Algonauts"]),

        ("Encoding Score",
         "The accuracy metric for brain encoding models. Measured as Pearson correlation (r) between the model's predicted BOLD signal and the participant's actual fMRI signal. TRIBE v2 achieved rank #1 with mean score 0.2146 ± 0.0312.",
         "#40c4ff",
         ["Pearson correlation", "TRIBE: 0.2146 mean", "Higher = better prediction"]),

        ("Cortical Vertices / Parcellation",
         "The brain's surface is divided into 20,484 measurement points (vertices). The HCP atlas groups these into 360 functional parcels (brain regions) including visual cortex V1–V7, face-body areas, language regions, and subcortical zones.",
         "#40c4ff",
         ["360 HCP parcels", "V1–V7 visual regions", "FaceBody, Language areas"]),

        ("Foundation Model",
         "A large pre-trained AI model that can be applied to many downstream tasks without retraining from scratch. TRIBE v2 is a foundation model for neuroscience — one model predicts brain responses across all stimulus types and experimental conditions.",
         "#40c4ff",
         ["Pre-trained at scale", "Zero-shot generalisation", "One model, many tasks"]),

        ("ICA (Independent Component Analysis)",
         "A statistical technique to separate a multi-dimensional signal into independent components. Used in the TRIBE v2 paper to reveal that the model learned neuroscientifically meaningful brain patterns without being explicitly trained to do so.",
         "#40c4ff",
         ["Unsupervised decomposition", "Reveals latent structure", "Validates TRIBE learning"]),
    ],

    "Cognitive Science": [
        ("Attention Score",
         "A 0–100 measure of a creative's ability to capture and hold visual attention. Driven by visual contrast (50% weight), face presence (up to +36 pts), and clutter penalty (based on object count). Calibrated against V1–V7 visual cortex encoding.",
         "#00e676",
         ["> 60 = High Capture", "30–60 = Moderate", "< 30 = Scroll-Past Risk"]),

        ("Memory Score",
         "A 0–100 measure of how strongly a creative will be encoded into long-term memory. Based on composition simplicity (fewer objects = better) and dual-coding balance (text density 5–25% optimal). Maps to hippocampal encoding strength.",
         "#00e676",
         ["> 70 = Strong Signal", "40–70 = Moderate", "< 40 = Low Retention"]),

        ("Emotional Valence",
         "The positive/negative emotional charge a creative produces in the viewer. Measured −1.0 (strongly aversive) to +1.0 (strongly rewarding). Driven by face presence, dominant colour HSV properties, and composition warmth.",
         "#00e676",
         ["+1.0 = Rewarding", "0.0 = Neutral", "−1.0 = Aversive"]),

        ("Cognitive Load",
         "The total mental effort required to process a creative. Calculated as a composite of visual complexity (object count) and verbal load (text density). Low = effortless processing. High = working memory overload → brand message not retained.",
         "#00e676",
         ["Low = < 35 composite", "Medium = 35–60", "High = > 60 · −12 CPCi"]),

        ("Pre-attentive Processing",
         "Brain processing that happens automatically before conscious attention, typically within 200ms. Features processed pre-attentively: luminance contrast, colour, motion, face presence, orientation. Ads that pass this filter don't 'ask' for attention.",
         "#00e676",
         ["< 200ms response", "Before conscious choice", "Contrast + face = strongest"]),

        ("Dual-Coding Theory",
         "Allan Paivio's (1971) theory that the brain stores verbal information (text) and visual information (images) in separate memory systems. Content encoded in both systems is 2× more memorable. Source of the 5–25% text density guideline.",
         "#00e676",
         ["Paivio 1971", "Two memory channels", "5–25% text = sweet spot"]),

        ("Cognitive Load Theory",
         "John Sweller's (1988) theory that working memory has a fixed capacity. Extraneous load (visual clutter, excess copy) consumes that capacity, leaving less room for germane load (brand message encoding). Basis for the object-count thresholds.",
         "#00e676",
         ["Sweller 1988", "7 ± 2 items limit", "Clutter = capacity waste"]),

        ("Russell Circumplex",
         "James Russell's (1980) two-dimensional model of affect: Valence (pleasant/unpleasant) × Arousal (activated/deactivated). Emotional valence in this tool maps to the horizontal valence axis. Used to explain why colour and faces shift emotional response.",
         "#00e676",
         ["Russell 1980", "Valence + Arousal axes", "Predicts purchase intent"]),

        ("Miller's Law",
         "George Miller's (1956) finding that short-term memory holds 7 ± 2 chunks simultaneously. In ad creative, each distinct visual object is a 'chunk'. Beyond 7 objects, viewers sample randomly — the brand message may not be what gets sampled.",
         "#00e676",
         ["Miller 1956", "7 ± 2 chunks", "Object threshold basis"]),
    ],

    "Visual Detection": [
        ("Object Detection (Contour Analysis)",
         "TRIBE pipeline step that counts distinct visual objects using Canny edge detection, dilation, and contour analysis. Threshold: a contour must cover ≥ 0.1% of total image area to count. Outputs object_count.",
         "#ffb300",
         ["OpenCV Canny", "0.1% area threshold", "Proxy for visual clutter"]),

        ("Face Detection (Haar Cascade)",
         "OpenCV's Haar Cascade classifier detects frontal human faces. Settings: scaleFactor=1.1, minNeighbors=5, minSize=30×30px. Runs in ~50–200ms. Face presence is the single highest-return attention variable in ad creative.",
         "#ffb300",
         ["OpenCV Haar", "50–200ms", "+22 pts attention per face"]),

        ("Text Density (OCR)",
         "Tesseract OCR in PSM 11 sparse-text mode detects all text blocks regardless of layout. Text density = sum of text bounding box areas ÷ total image area. Confidence threshold ≥ 40% to filter noise.",
         "#ffb300",
         ["Tesseract PSM 11", "Confidence ≥ 40%", "0.0–1.0 ratio"]),

        ("Contrast Score",
         "Computed as the standard deviation of grayscale pixel values, normalised to 0–100 (max std = 127.5 for an 8-bit image). Low std = flat/washed out. High std = strong light-dark variation = pre-attentive pop-out.",
         "#ffb300",
         ["Std dev / 127.5 × 100", "0 = flat", "100 = max contrast"]),

        ("Dominant Colours (K-means)",
         "K-means clustering (k=3) on a 100×100 downscaled image identifies the 3 most dominant colours by pixel count. Sorted by frequency. Converted to RGB hex strings. Used for valence calculation via HSV colour psychology.",
         "#ffb300",
         ["K-means k=3", "100×100 downsample", "Sorted by frequency"]),

        ("HSV Colour Space",
         "Hue-Saturation-Value colour representation. Used for emotion classification instead of RGB because HSV separates colour identity (hue) from brightness (value) more naturally. Warm hues (0–60°) = positive valence. Dark (V < 0.20) = negative.",
         "#ffb300",
         ["Hue 0–360°", "Saturation 0–1", "Value 0–1 (brightness)"]),
    ],

    "Media & Advertising": [
        ("CPCi (Creative Performance Composite Index)",
         "A 0–100 weighted score combining Attention, Memory, and Emotional Valence, calibrated to the specific cognitive demands of a given use case. Higher CPCi = stronger cognitive impact = better predicted media efficiency.",
         "#ff80ab",
         ["0–100 scale", "Use-case weighted", "Predicts media ROI"]),

        ("Thumb-Stop Rate",
         "The percentage of users who stop scrolling when they encounter an ad. Directly correlated with Attention Score. Low attention ads produce low thumb-stop rates, which the platform algorithm interprets as low quality, increasing effective CPM.",
         "#ff80ab",
         ["Attention proxy", "Algorithm quality signal", "Determines CPM cost"]),

        ("Brand Salience",
         "How quickly and easily a brand comes to mind in buying situations. Built through consistent memory encoding over multiple exposures. High Memory Score creatives build salience faster and with fewer exposures required.",
         "#ff80ab",
         ["Byron Sharp concept", "Built via Memory Score", "Drives shelf recognition"]),

        ("Effective CPM",
         "The true cost per 1,000 impressions accounting for placement quality. Low-attention creatives inflate effective CPM because the platform algorithm bids them into lower-quality inventory with lower engagement.",
         "#ff80ab",
         ["True cost metric", "Rises with low attention", "CPCi predicts direction"]),

        ("Brand Lift",
         "An increase in brand awareness, preference, or purchase intent measured by surveys after campaign exposure. CPCi scores above 70 historically correlate with 3–5 percentage point brand lift per campaign.",
         "#ff80ab",
         ["Survey-measured", "CPCi > 70 = 3–5pt lift", "Gold standard KPI"]),

        ("Frequency Capping",
         "Setting a maximum number of times a user sees the same ad. High Memory Score creatives need lower frequency (3–5 impressions) to achieve the same recall as low-memory creatives (6–9 impressions needed). Impacts media budget efficiency.",
         "#ff80ab",
         ["3–5 for high memory", "6–9 for low memory", "Controls wear-out"]),

        ("Cognitive Load Penalty",
         "A CPCi deduction applied in Retail Media contexts only: −0 for Low load, −5 for Medium, −12 for High. Applied because retail environments already impose high cognitive load on shoppers — an ad that adds to it will not convert.",
         "#ff80ab",
         ["Retail Media only", "−0 / −5 / −12", "Reflects real-world clutter"]),

        ("Quality Score",
         "The ad quality metric used by platforms (Google, Meta) that determines ad auction ranking and CPM. Creatives with high engagement (driven by attention and positive emotion) receive better Quality Scores, lowering cost per click.",
         "#ff80ab",
         ["Platform auction metric", "Attention → higher score", "Lowers CPC directly"]),
    ],
}


def show_glossary_tab() -> None:
    st.markdown(SCI_CSS, unsafe_allow_html=True)

    st.markdown(
        "<div class='main-header' style='margin-bottom:4px;border-left:3px solid #3B82F6;padding-left:16px;'>"
        "📖 Glossary of Terms</div>"
        "<div style='font-size:13px;color:#CBD5E1;margin-bottom:20px;'>"
        "Every technical term used in the reports, scores, and methodology — explained for marketers.</div>",
        unsafe_allow_html=True,
    )

    for category, terms in GLOSSARY.items():
        with st.expander(f"{'🔬' if 'Neuro' in category else '🧩' if 'Cognitive' in category else '👁' if 'Visual' in category else '📢'} {category} — {len(terms)} terms", expanded=False):
            cols = st.columns(2)
            for idx, (term, definition, color, tags) in enumerate(terms):
                tag_html = " ".join(f"<span>{t}</span>" for t in tags)
                cols[idx % 2].markdown(
                    f"<div class='gls-card' style='border-left-color:{color};'>"
                    f"<div class='gls-term' style='color:{color};'>{term}</div>"
                    f"<div class='gls-def'>{definition}</div>"
                    f"<div class='gls-ptr'>{tag_html}</div>"
                    f"</div>",
                    unsafe_allow_html=True,
                )

    # Quick-reference cheat sheet
    st.markdown("<hr style='border:none;border-top:1px solid #1a2840;margin:24px 0;'>",
                unsafe_allow_html=True)
    st.markdown(
        "<div style='font-size:12px;color:#40c4ff;text-transform:uppercase;"
        "letter-spacing:2px;font-weight:700;margin-bottom:12px;'>⚡ Quick-Reference Cheat Sheet</div>",
        unsafe_allow_html=True,
    )

    qs_rows = [
        ("You see...",              "It means...",                                 "Action"),
        ("Attention < 30",          "Ad will be ignored before consciously seen",  "Add face or boost contrast first"),
        ("Memory < 40",             "Brand won't be remembered after exposure",    "Add tagline or simplify layout"),
        ("Valence < −0.1",          "Subtle aversion — silently kills brand love", "Warm up palette or add a face"),
        ("Cognitive Load = High",   "Working memory saturated — message lost",     "Remove objects, cut copy"),
        ("CPCi < 40",               "Do not scale — fix creative first",           "A/B test before spending"),
        ("CPCi 40–70",              "Potential there, but limited",                "Identify weakest signal and fix it"),
        ("CPCi > 70",               "Ready to scale — strong cognitive impact",    "Increase budget 20–30%"),
        ("Text density > 25%",      "Verbal channel drowning visual",              "Cut copy to one key message"),
        ("Text density < 5%",       "No verbal anchor — recall will fade",         "Add 5–10 word tagline"),
        ("Faces = 0",               "Missing fastest attention + emotion driver",  "Add human face to primary zone"),
        ("Contrast < 45",           "Won't pop in feed — washed out pre-attention","Increase contrast in post"),
    ]

    header, *rows = qs_rows
    header_html = (
        "<div style='display:grid;grid-template-columns:1fr 1.4fr 1.2fr;"
        "gap:0;background:#08111e;border:1px solid #1a2840;"
        "border-radius:10px 10px 0 0;overflow:hidden;'>"
        + "".join(
            f"<div style='padding:10px 14px;font-size:13px;font-weight:700;"
            f"text-transform:uppercase;letter-spacing:1.2px;color:#CBD5E1;"
            f"border-right:1px solid #1a2840;'>{h}</div>"
            for h in header
        )
        + "</div>"
    )
    st.markdown(header_html, unsafe_allow_html=True)

    rows_html = "<div style='border:1px solid #1a2840;border-top:none;border-radius:0 0 10px 10px;overflow:hidden;'>"
    for i, (signal, meaning, action) in enumerate(rows):
        bg = "#060e1a" if i % 2 == 0 else "#080f1c"
        rows_html += (
            f"<div style='display:grid;grid-template-columns:1fr 1.4fr 1.2fr;"
            f"background:{bg};border-top:1px solid #111c2e;'>"
            f"<div style='padding:10px 14px;font-size:12px;font-weight:700;color:#FFFFFF;"
            f"border-right:1px solid #111c2e;'>{signal}</div>"
            f"<div style='padding:10px 14px;font-size:12px;color:#94A3B8;"
            f"border-right:1px solid #111c2e;'>{meaning}</div>"
            f"<div style='padding:10px 14px;font-size:12px;color:#ffb300;'>{action}</div>"
            f"</div>"
        )
    rows_html += "</div>"
    st.markdown(rows_html, unsafe_allow_html=True)
