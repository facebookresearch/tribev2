"""
app.py — Creative Intelligence Analyzer
----------------------------------------
Supports single-creative analysis AND multi-creative comparison (2–5 images).

Pipeline (per creative):
  upload → save_upload() → analyze_creative() → map_to_cognitive_signals()
         → compute_cpci() → show_results() / show_comparison()
"""

import os
import json
import time
import hashlib
import colorsys

import streamlit as st

from creative_vision   import analyze_creative
from cognitive_signals import map_to_cognitive_signals
from narrative_engine  import generate_narrative
from science_tab       import show_science_tab, show_glossary_tab


# ── Page config ───────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Creative Intelligence Analyzer",
    page_icon="🧠",
    layout="wide",
)

# ── Global CSS ────────────────────────────────────────────────────────────────

st.markdown("""
<style>
/* ═══════════════════════════════════════════
   GLOBAL TYPOGRAPHY — base defaults only
   Colors applied selectively via named classes.
   NO color: !important on generic elements —
   that blocks all downstream header coloring.
   ═══════════════════════════════════════════ */

/* Base body color — inherits down unless overridden */
body {
    color: #E5E7EB;
    font-size: 17px !important;
}

/* Font size on common elements — no color override */
html, body, [class*="css"] {
    font-size: 17px !important;
}
p, span, div {
    font-size: 17px !important;
    /* color intentionally NOT set here — use named classes */
}
small, label {
    font-size: 15px !important;
    color: #CBD5E1 !important;
}
h1, h2, h3 {
    font-weight: 600 !important;
    /* color set per-instance via .section-header, .ab-subhead, etc. */
}

/* ── Semantic color utility classes ──────────────────────────── */
.main-header   { color: #60A5FA; font-size: 24px; font-weight: 600; letter-spacing: 0.02em; }   /* section headers              */
.sub-header    { color: #93C5FD; font-size: 18px; font-weight: 500; }   /* sub-section / card titles    */
.body-text     { color: #FFFFFF; }   /* primary body content         */
.secondary-text{ color: #CBD5E1; }   /* supporting / secondary text  */

/* Override Streamlit default muted colours — color only, no !important on body text */
.stMarkdown { color: #E5E7EB; }
.stText     { color: #E5E7EB; }

@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&display=swap');

/* ═══════════════════════════════════════════
   DESIGN TOKENS
   bg        : #0B0F14
   surface-1 : #141B24  (primary cards)
   surface-2 : #111827  (nested / secondary)
   border    : #1F2937
   blue      : #3B82F6
   header    : #60A5FA  (section headers)
   subheader : #93C5FD  (sub-section headers)
   text-1    : #FFFFFF   (primary — pure white)
   text-2    : #CBD5E1  (secondary — readable)
   text-3    : #94A3B8  (labels / meta)
   green     : #22C55E  (good)
   amber     : #F59E0B  (warn)
   red       : #EF4444  (bad)
   shadow    : 0 1px 3px rgba(0,0,0,0.2)

   TYPE SCALE
   body     : 16px / 400 / lh 1.75
   label    : 11px / 500 / uppercase
   heading  : 20–28px / 600
   hero     : 120px+ / 600
   eyebrow  : 10px / 500 / uppercase tracked
═══════════════════════════════════════════ */

/* BASE */
html, body {
  background: #0B0F14 !important;
}
.stApp, [data-testid="stAppViewContainer"] {
  font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important;
  background: linear-gradient(180deg, #0B0F14 0%, #0E141B 100%) !important;
  background-attachment: fixed !important;
  font-size: 17px !important;
  min-height: 100vh !important;
}
[data-testid="stSidebar"] { background: #0B0F14 !important; }

/* Constrain content width for large screens */
[data-testid="stMainBlockContainer"] {
  max-width: 1280px !important;
  padding: 2rem 3rem !important;
}
@media (min-width: 1600px) {
  [data-testid="stMainBlockContainer"] {
    padding: 2rem 4rem !important;
  }
}

.stMarkdown, .stMarkdown p, .stMarkdown li,
[data-testid="stMarkdown"], [data-testid="stMarkdown"] p,
.stSelectbox label, .stFileUploader label,
.stExpander summary p, .stTabs [data-testid="stMarkdown"] {
  font-family: 'Inter', -apple-system, sans-serif !important;
}
.stMarkdown p, [data-testid="stMarkdown"] p {
  color: #FFFFFF !important;
  font-size: 17px !important;
  font-weight: 400 !important;
  line-height: 1.7 !important;
}

/* Hide Streamlit anchor icons */
h1 a, h2 a, h3 a,
[data-testid="stHeadingWithActionElements"] a { display: none !important; }

/* Streamlit UI elements */
.stExpander summary {
  color: #FFFFFF !important;
  font-weight: 500 !important;
  font-size: 17px !important;
}
.stSelectbox label, .stFileUploader label {
  color: #FFFFFF !important;
  font-size: 17px !important;
  font-weight: 500 !important;
}
[data-testid="stTabs"] button {
  font-size: 17px !important;
  font-weight: 500 !important;
  color: #94A3B8 !important;
  padding: 10px 20px !important;
  letter-spacing: 0.02em !important;
}
[data-testid="stTabs"] button[aria-selected="true"] {
  color: #FFFFFF !important;
}
/* Remove stray Streamlit top padding */
[data-testid="stVerticalBlock"] > [data-testid="stVerticalBlockBorderWrapper"] {
  border: none !important;
}
/* File uploader clean look */
[data-testid="stFileUploaderDropzone"] {
  background: #141B24 !important;
  border: 1px dashed #1F2937 !important;
  border-radius: 8px !important;
}
/* Selectbox clean look */
[data-testid="stSelectbox"] > div > div {
  background: #141B24 !important;
  border: 1px solid #1F2937 !important;
  border-radius: 6px !important;
  color: #FFFFFF !important;
}

/* ── UTILITIES ── */
.divider { border: none; border-top: 1px solid #1F2937; margin: 40px 0; }

/* ── BADGES ── */
.badge {
  display: inline-block; padding: 4px 12px; border-radius: 4px;
  font-size: 13px; font-weight: 500; text-transform: uppercase; letter-spacing: 0.6px;
}
.badge-good { background: rgba(34,197,94,0.1);  color: #22C55E; }
.badge-warn { background: rgba(245,158,11,0.1); color: #F59E0B; }
.badge-bad  { background: rgba(239,68,68,0.1);  color: #EF4444; }
.badge-avg  { background: rgba(59,130,246,0.1); color: #3B82F6; }

/* ── TIMER / SOURCE ── */
.timer-box {
  display: inline-block; padding: 6px 16px; border-radius: 4px;
  background: #141B24; border: 1px solid #1F2937;
  color: #3B82F6 !important; font-size: 14px; font-weight: 500;
}
.source-badge {
  display: inline-flex; align-items: center; gap: 6px;
  background: #141B24; border: 1px solid #1F2937; border-radius: 4px;
  padding: 6px 14px; font-size: 13px; color: #CBD5E1; font-weight: 500;
}

/* ═══════════════════════════════════════════
   CARDS — surface-1 bg, subtle shadow
═══════════════════════════════════════════ */
.card {
  background: #141B24;
  border: 1px solid #1F2937;
  border-left: 3px solid #3B82F6;
  border-radius: 16px;
  padding: 40px 36px;
  margin-bottom: 40px;
  box-shadow: 0 1px 3px rgba(0,0,0,0.2);
}

/* ── CPCi Hero ── */
.cpci-box {
  background: #141B24;
  border: 1px solid #1F2937;
  border-radius: 16px;
  padding: 36px 28px 32px 28px;
  text-align: center;
  margin: 0 0 24px 0;
  box-shadow: 0 1px 3px rgba(0,0,0,0.2);
}

/* ── Quick Read (primary tier — no card border, just rows) ── */
.qr-wrap    { padding: 0; margin: 0; }
.qr-heading { font-size: 13px; color: #93C5FD; text-transform: uppercase;
              letter-spacing: 1.5px; font-weight: 600; margin-bottom: 28px; }
.qr-row     { display: flex; align-items: baseline; gap: 24px;
              padding: 32px 0; border-bottom: 1px solid #1F2937; }
.qr-row:last-child { border-bottom: none; padding-bottom: 0; }
.qr-num     { font-size: 13px; font-weight: 500; letter-spacing: 0.5px;
              min-width: 20px; flex-shrink: 0; color: #94A3B8; }
.qr-tag     { font-size: 13px; font-weight: 600; text-transform: uppercase;
              letter-spacing: 1px; min-width: 100px; flex-shrink: 0; }
.qr-line    { font-size: 17px !important; color: #FFFFFF !important;
              font-weight: 400 !important; line-height: 1.85; }

/* ── Narrative section cards (secondary — reduced weight) ── */
.ns-card    { background: #141B24; border: 1px solid #1F2937;
              border-left: 3px solid #3B82F6;
              border-radius: 16px;
              padding: 0; margin: 0 0 40px 0;
              box-shadow: 0 1px 3px rgba(0,0,0,0.2); }
.ns-header  { display: flex; align-items: center; justify-content: space-between;
              padding: 24px 36px 20px 36px; border-bottom: 1px solid #1F2937; }
.ns-icon-title { display: flex; align-items: center; gap: 10px; }
.ns-icon    { font-size: 17px; line-height: 1; }
.ns-title   { font-size: 13px; font-weight: 500; color: #60A5FA;
              letter-spacing: 1px; text-transform: uppercase; }
.ns-score-pill { display: flex; flex-direction: column; align-items: center;
                 border-radius: 4px; padding: 4px 12px; text-align: center; }
.ns-score-val  { font-size: 18px; font-weight: 600; line-height: 1.1; }
.ns-score-lbl  { font-size: 13px; text-transform: uppercase; letter-spacing: 0.8px;
                 color: #94A3B8; margin-top: 2px; font-weight: 500; }
.ns-body    { padding: 32px 36px 28px 36px; font-size: 17px; color: #FFFFFF;
              line-height: 1.85; font-weight: 400; }
.ns-body p  { margin: 0 0 28px 0; color: #FFFFFF; }
.ns-body p:last-child { margin-bottom: 0; }
.ns-body strong { color: #FFFFFF; font-weight: 600; }
.ns-pointers { display: flex; flex-wrap: wrap; gap: 8px;
               padding: 20px 36px 28px 36px; border-top: 1px solid #1F2937; }
.ns-ptr     { display: inline-flex; align-items: center; gap: 6px;
              background: #111827; border: 1px solid #1F2937; border-radius: 4px;
              padding: 4px 10px; font-size: 12px; color: #94A3B8; font-weight: 500; }
.ns-ptr-val { font-weight: 600; color: #CBD5E1; }

/* ── Cognitive Diagnosis (most technical — smallest weight) ── */
.cd-wrap    { background: #141B24; border: 1px solid #1F2937;
              border-left: 2px solid #1D4ED8;
              border-radius: 16px;
              padding: 0; margin: 0 0 40px 0;
              box-shadow: 0 1px 3px rgba(0,0,0,0.2); }
.cd-header  { display: flex; align-items: center; justify-content: space-between;
              padding: 24px 36px 20px 36px; border-bottom: 1px solid #1F2937; }
.cd-title   { font-size: 13px; font-weight: 500; color: #60A5FA;
              letter-spacing: 1px; text-transform: uppercase; }
.cd-grid    { display: grid; grid-template-columns: 1fr 1fr 1fr 1fr; gap: 0; }
.cd-block   { padding: 32px 28px; position: relative; }
.cd-block:not(:last-child) { border-right: 1px solid #1F2937; }
@media (max-width: 900px) {
  .cd-grid  { grid-template-columns: 1fr 1fr; }
  .cd-block:nth-child(2) { border-right: none; }
  .cd-block:nth-child(odd):not(:nth-child(3)) { border-right: 1px solid #1F2937; }
  .cd-block:nth-child(1), .cd-block:nth-child(2) { border-bottom: 1px solid #1F2937; }
}
.cd-blk-header { display: flex; align-items: center; justify-content: space-between;
                 margin-bottom: 14px; }
.cd-blk-icon   { font-size: 14px; }
.cd-blk-score  { font-size: 26px; font-weight: 600; line-height: 1; }
.cd-blk-label  { font-size: 13px; text-transform: uppercase; letter-spacing: 0.8px;
                 color: #94A3B8; margin-top: 2px; font-weight: 500; }
.cd-bullet     { display: flex; gap: 8px; align-items: flex-start; margin-bottom: 10px;
                 font-size: 13px; color: #94A3B8; line-height: 1.65; }
.cd-bullet-dot { flex-shrink: 0; margin-top: 7px; width: 3px; height: 3px;
                 border-radius: 50%; background: #1F2937; }
.cd-implication { margin-top: 14px; padding: 12px 14px; border-radius: 4px;
                  font-size: 12px; font-weight: 500; line-height: 1.7;
                  background: #111827; border: 1px solid #1F2937;
                  border-left: 2px solid #1D4ED8; color: #94A3B8; }

/* ── Recommendations ── */
.rp-wrap    { background: #141B24; border: 1px solid #1F2937;
              border-left: 3px solid #3B82F6;
              border-radius: 16px;
              padding: 0; margin: 0 0 40px 0;
              box-shadow: 0 1px 3px rgba(0,0,0,0.2); }
.rp-header  { display: flex; align-items: center; gap: 10px;
              padding: 24px 36px 20px 36px; border-bottom: 1px solid #1F2937; }
.rp-icon    { font-size: 17px; line-height: 1; }
.rp-title   { font-size: 13px; font-weight: 500; color: #60A5FA;
              letter-spacing: 1px; text-transform: uppercase; }
.rp-count   { margin-left: auto; background: rgba(59,130,246,0.1);
              border-radius: 4px; padding: 2px 8px;
              font-size: 13px; color: #3B82F6; font-weight: 500; }
.rp-list    { padding: 0; }
.rp-item    { display: flex; gap: 0; }
.rp-item:not(:last-child) { border-bottom: 1px solid #1F2937; }
.rp-badge-col { width: 56px; flex-shrink: 0; display: flex; align-items: flex-start;
                justify-content: center; padding: 24px 0; }
.rp-badge   { width: 26px; height: 26px; border-radius: 5px; display: flex;
              align-items: center; justify-content: center; font-size: 12px; }
.rp-content { flex: 1; padding: 28px 36px 28px 0; }
.rp-lbl     { font-size: 13px; font-weight: 500; text-transform: uppercase;
              letter-spacing: 0.8px; margin-bottom: 8px;
              display: flex; align-items: center; gap: 6px; }
.rp-lbl-dot { width: 3px; height: 3px; border-radius: 50%; flex-shrink: 0; }
.rp-body    { font-size: 17px; color: #FFFFFF; line-height: 1.85; font-weight: 400; }
.rp-footer  { display: flex; flex-wrap: wrap; gap: 8px;
              padding: 18px 28px 24px 28px; border-top: 1px solid #1F2937; }

/* ── Signal strip ── */
.sig-strip  { display: flex; background: #141B24; border: 1px solid #1F2937;
              border-radius: 16px; margin: 32px 0; overflow: hidden;
              box-shadow: 0 1px 3px rgba(0,0,0,0.2); }
.sig-cell   { flex: 1; padding: 24px 20px; text-align: center; }
.sig-cell:not(:last-child) { border-right: 1px solid #1F2937; }
.sig-label  { font-size: 13px; color: #94A3B8; text-transform: uppercase;
              letter-spacing: 1px; font-weight: 500; margin-bottom: 6px; }
.sig-value  { font-size: 30px; font-weight: 600; line-height: 1.1; }
.sig-sublabel { font-size: 13px; color: #94A3B8; margin-top: 4px; }

/* ── Vis chips ── */
.vis-strip  { display: flex; gap: 8px; flex-wrap: wrap; margin: 12px 0; }
.vis-chip   { background: #111827; border: 1px solid #1F2937; border-radius: 4px;
              padding: 5px 12px; font-size: 13px; color: #94A3B8; }
.vis-chip strong { color: #CBD5E1; font-weight: 500; }

/* ── Color swatch ── */
.color-swatch { display: inline-block; width: 18px; height: 18px;
                border-radius: 3px; margin: 0 2px; vertical-align: middle; }

/* ── Section header ── */
.section-header { font-size: 24px; font-weight: 600; color: #60A5FA;
                  margin: 56px 0 28px 0; line-height: 1.3;
                  letter-spacing: 0.02em;
                  border-left: 3px solid #3B82F6; padding-left: 16px; }

/* ── Metric card (score explanations) ── */
.metric-card  { background: #141B24; border: 1px solid #1F2937; border-radius: 16px;
                padding: 28px 24px; text-align: center; margin: 0 0 40px 0;
                box-shadow: 0 1px 3px rgba(0,0,0,0.2); }
.metric-label { color: #94A3B8; font-size: 13px; font-weight: 500; letter-spacing: 1px;
                text-transform: uppercase; margin-bottom: 10px; }
.metric-value { font-size: 44px; font-weight: 600; margin: 6px 0; line-height: 1; }
.metric-desc  { color: #FFFFFF; font-size: 17px; margin-top: 8px;
                line-height: 1.7; font-weight: 400; }

/* ═══════════════════════════════════════════
   A/B COMPARISON
═══════════════════════════════════════════ */
.ab-header  { font-size: 13px; font-weight: 500; color: #94A3B8; letter-spacing: 1.5px;
              text-transform: uppercase; margin: 40px 0 6px 0; }
.ab-subhead { font-size: 28px; font-weight: 600; color: #60A5FA; margin: 0 0 32px 0;
              line-height: 1.3;
              border-left: 3px solid #3B82F6; padding-left: 16px; }

/* Cards with shadow — winner gets blue border */
.ab-card {
  background: #141B24; border: 1px solid #1F2937;
  border-radius: 16px; padding: 32px 28px;
  box-shadow: 0 1px 3px rgba(0,0,0,0.2);
}
.ab-card-winner {
  background: #141B24; border: 1px solid #3B82F6;
  border-radius: 16px; padding: 32px 28px;
  box-shadow: 0 1px 3px rgba(0,0,0,0.2);
}
.ab-win-badge {
  display: inline-block; background: rgba(59,130,246,0.1); color: #3B82F6;
  font-size: 13px; font-weight: 500; letter-spacing: 1px; text-transform: uppercase;
  border-radius: 4px; padding: 4px 12px; margin-bottom: 14px;
}
.ab-rank-badge {
  display: inline-block; background: #111827; color: #94A3B8;
  font-size: 13px; font-weight: 500; letter-spacing: 1px; text-transform: uppercase;
  border: 1px solid #1F2937; border-radius: 4px; padding: 4px 12px; margin-bottom: 14px;
}
.ab-name      { font-size: 14px; font-weight: 500; color: #FFFFFF; margin-bottom: 16px;
                white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
.ab-score     { font-size: 64px; font-weight: 600; line-height: 1; letter-spacing: -1px; }
.ab-score-sub { font-size: 12px; color: #94A3B8; margin-top: 6px; margin-bottom: 18px; }
.ab-sig-grid  { display: grid; grid-template-columns: 1fr 1fr; gap: 1px;
                background: #1F2937; border-radius: 4px; overflow: hidden; margin: 16px 0; }
.ab-sig-cell  { background: #111827; padding: 16px 12px; text-align: center; }
.ab-sig-label { font-size: 13px; color: #94A3B8; font-weight: 500; letter-spacing: 0.8px;
                text-transform: uppercase; margin-bottom: 4px; }
.ab-sig-val   { font-size: 20px; font-weight: 600; line-height: 1; }
.ab-verdict   { font-size: 13px; font-weight: 500; line-height: 1.65;
                padding: 12px 14px; border-radius: 4px;
                background: #111827; border-left: 2px solid; margin-top: 14px; }

/* Why this wins */
.ab-why       { background: #141B24; border: 1px solid #1F2937;
                border-left: 2px solid #1D4ED8;
                border-radius: 16px;
                padding: 44px 40px; margin: 40px 0 40px 0;
                box-shadow: 0 1px 3px rgba(0,0,0,0.2); }
.ab-why-title { font-size: 13px; font-weight: 500; color: #93C5FD; letter-spacing: 1.5px;
                text-transform: uppercase; margin-bottom: 8px; }
.ab-why-name  { font-size: 22px; font-weight: 600; color: #93C5FD; margin-bottom: 16px;
                line-height: 1.3; }
.ab-why-body  { font-size: 17px; font-weight: 400; color: #FFFFFF; line-height: 1.85; }
.ab-why-stat  { font-weight: 600; color: #3B82F6; }
.ab-why-gap   { color: #EF4444; font-weight: 600; }
.ab-bar-row   { display: flex; align-items: center; gap: 14px; margin: 10px 0; }
.ab-bar-label { font-size: 13px; color: #CBD5E1; font-weight: 500; min-width: 96px; }
.ab-bar-track { flex: 1; height: 3px; background: #1F2937; border-radius: 2px; }
.ab-bar-fill  { height: 3px; border-radius: 2px; }
.ab-bar-val   { font-size: 13px; font-weight: 600; min-width: 32px; text-align: right; }

/* ═══════════════════════════════════════════
   CLIENT MODE
═══════════════════════════════════════════ */
.cm-card    { background: #141B24; border: 1px solid #1F2937;
              border-left: 3px solid #3B82F6;
              border-radius: 16px;
              padding: 48px 44px; margin: 0 0 40px 0;
              box-shadow: 0 1px 3px rgba(0,0,0,0.2); }
.cm-label   { font-size: 13px; font-weight: 500; color: #94A3B8; letter-spacing: 1.5px;
              text-transform: uppercase; margin-bottom: 12px; }
.cm-score   { font-size: 130px; font-weight: 600; line-height: 1; letter-spacing: -2px;
              margin-bottom: 8px; color: #FFFFFF;
              text-shadow: 0 0 20px rgba(255,255,255,0.08); }
.cm-verdict { font-size: 38px; font-weight: 700; line-height: 1.25; letter-spacing: -0.3px;
              margin: 28px 0 0 0; color: #FFFFFF; }
.cm-insight { font-size: 17px; font-weight: 400; color: #FFFFFF; line-height: 1.85; margin: 0; }
.cm-rec     { font-size: 17px; font-weight: 400; color: #FFFFFF; line-height: 1.85; margin: 0; }
.cm-divider { border: none; border-top: 1px solid #1F2937; margin: 36px 0; }

/* ═══════════════════════════════════════════
   CPCI HERO ANIMATIONS
═══════════════════════════════════════════ */

/* Blur-to-focus reveal — number lands decisively */
@keyframes cpciReveal {
  0%   { opacity: 0; transform: scale(0.78) translateY(10px); filter: blur(12px); }
  55%  { filter: blur(0px); }
  100% { opacity: 1; transform: scale(1) translateY(0); filter: blur(0px); }
}

/* Clean fade-up for surrounding text elements */
@keyframes cpciFadeUp {
  from { opacity: 0; transform: translateY(7px); }
  to   { opacity: 1; transform: translateY(0); }
}

.cpci-label-el {
  animation: cpciFadeUp 0.5s cubic-bezier(0.16, 1, 0.3, 1) 0.05s both;
}
.cpci-score-el {
  animation: cpciReveal 0.95s cubic-bezier(0.16, 1, 0.3, 1) 0.15s both;
  display: block;
}
.cpci-subtext-el {
  animation: cpciFadeUp 0.5s cubic-bezier(0.16, 1, 0.3, 1) 0.4s both;
}
.cpci-context-el {
  animation: cpciFadeUp 0.5s cubic-bezier(0.16, 1, 0.3, 1) 0.55s both;
}

/* ═══════════════════════════════════════════
   APP HEADER
═══════════════════════════════════════════ */
.app-wordmark {
  font-size: 17px; font-weight: 600; color: #FFFFFF;
  letter-spacing: 0.03em; line-height: 1;
}
.app-wordmark sup {
  font-size: 9px; font-weight: 500; letter-spacing: 0; vertical-align: super;
}
.app-tagline {
  font-size: 13px; font-weight: 500; color: #94A3B8;
  letter-spacing: 0.14em; text-transform: uppercase;
  margin-top: 12px; padding-bottom: 24px;
}
/* Style Streamlit toggles in header to look minimal */
[data-testid="stToggle"] p {
  font-size: 13px !important;
  font-weight: 500 !important;
  color: #CBD5E1 !important;
}

/* ═══════════════════════════════════════════
   TOOLTIPS
═══════════════════════════════════════════ */
.tt-wrap  { position: relative; display: inline-block; cursor: help; }
.tt-icon  { color: #3B82F6; font-size: 10px; vertical-align: super; margin-left: 2px; }
.tt-body  { display: none; position: absolute; bottom: calc(100% + 8px); left: 50%;
            transform: translateX(-50%); background: #141B24; border: 1px solid #1F2937;
            border-radius: 6px; padding: 14px 18px; font-size: 14px; color: #CBD5E1;
            line-height: 1.7; min-width: 240px; max-width: 280px; z-index: 9999;
            white-space: normal; text-align: left; pointer-events: none; }
.tt-body b    { color: #FFFFFF; font-weight: 600; display: block;
                margin-bottom: 8px; font-size: 14px; }
.tt-body span { display: block; padding: 2px 0; }
.tt-wrap:hover .tt-body { display: block; }
</style>
""", unsafe_allow_html=True)


# ── Tooltip helper ────────────────────────────────────────────────────────────

def _tooltip(label: str, title: str, bullets: list) -> str:
    """Return inline HTML: label + ⓘ that reveals a tooltip card on hover."""
    pts = "".join(f"<span>{b}</span>" for b in bullets)
    return (
        f"<span class='tt-wrap'>{label}"
        f"<span class='tt-icon'>ⓘ</span>"
        f"<span class='tt-body'><b>{title}</b>{pts}</span>"
        f"</span>"
    )

# Pre-built tooltip HTML for the 4 tracked metrics
_TT_ATTN = _tooltip("Attention", "Attention Score", [
    "→ Measures visual stopping power",
    "→ Based on contrast, faces &amp; clutter",
    "→ Predicts scroll-stop probability",
])
_TT_MEM = _tooltip("Memory", "Memory Score", [
    "→ Measures brand recall potential",
    "→ Based on text density &amp; visual simplicity",
    "→ Drives recognition at point of purchase",
])
_TT_VAL = _tooltip("Emotion", "Emotional Valence", [
    "→ Measures positive vs. negative tone",
    "→ Derived from face expression &amp; color warmth",
    "→ Shapes brand affinity &amp; purchase intent",
])
_TT_CPCI = _tooltip("CPCi", "Cost Per Cognitive Impression", [
    "→ Composite cognitive impact score (0–100)",
    "→ Weighted blend of Attention + Memory + Emotion",
    "→ Predicts ad effectiveness before media spend",
])

# ── Core pipeline ─────────────────────────────────────────────────────────────

UPLOAD_DIR = os.path.expanduser("~/tribev2/uploads")

def save_upload(uploaded_file) -> str:
    """Save uploaded file to a permanent MD5-hashed path. Returns absolute path."""
    os.makedirs(UPLOAD_DIR, exist_ok=True)
    file_bytes   = uploaded_file.read()
    content_hash = hashlib.md5(file_bytes).hexdigest()[:10]
    stem         = os.path.splitext(uploaded_file.name)[0].replace(" ", "_")
    ext          = os.path.splitext(uploaded_file.name)[-1].lower()
    path         = os.path.join(UPLOAD_DIR, f"{stem}_{content_hash}{ext}")
    if not os.path.exists(path):
        with open(path, "wb") as f:
            f.write(file_bytes)
    return path


# ── Use-case config ───────────────────────────────────────────────────────────

USE_CASES = {
    "FMCG Branding": {
        "icon":        "🛒",
        "description": "Long-term brand salience · emotional recall · shelf recognition",
        "accent":      "#CBD5E1",
        "weights": {
            "attention": 0.20,
            "memory":    0.50,
            "emotion":   0.30,
        },
        # No cognitive-load penalty — FMCG audiences are patient shoppers
        "load_penalty": False,
        "rationale": (
            "FMCG success depends on memory encoding (recognition at point of purchase) "
            "and emotional warmth (brand affinity). Attention matters less — shoppers scan "
            "shelves slowly. Emotion drives trial and repeat purchase."
        ),
    },
    "Performance Marketing": {
        "icon":        "🎯",
        "description": "Direct response · click-through · immediate conversion",
        "accent":      "#3B82F6",
        "weights": {
            "attention": 0.50,
            "memory":    0.30,
            "emotion":   0.20,
        },
        "load_penalty": False,
        "rationale": (
            "Performance creatives must stop the scroll first — attention is the primary "
            "lever for click-through rate. Memory matters for retargeting sequences. "
            "Emotion plays a smaller role; the CTA does the conversion work."
        ),
    },
    "Retail Media": {
        "icon":        "🏪",
        "description": "In-store / on-platform · price-sensitive · high-clutter context",
        "accent":      "#22C55E",
        "weights": {
            "attention": 0.40,
            "memory":    0.30,
            "emotion":   0.30,
        },
        # Retail environments are visually noisy — high cognitive load is extra damaging
        "load_penalty": True,
        "rationale": (
            "Retail media sits in a high-clutter environment (product carousels, price tags, "
            "competing banners). A cognitive load penalty is applied because confused shoppers "
            "don't buy. Attention and emotion are weighted equally — both drive the impulse "
            "purchase decision."
        ),
    },
}

LOAD_PENALTY_MAP = {"Low": 0, "Medium": -5, "High": -12}


def compute_cpci(signals: dict, weights: dict = None, apply_load_penalty: bool = False) -> float:
    """
    CPCi = w_attn × Attention + w_mem × Memory + w_emotion × Valence_norm
           [− load_penalty  if apply_load_penalty]

    Args:
        signals:            Output of map_to_cognitive_signals().
        weights:            Dict with keys "attention", "memory", "emotion".
                            Defaults to balanced Performance Marketing weights.
        apply_load_penalty: If True, subtracts 0/5/12 pts for Low/Medium/High load.

    Valence is normalised from [-1, +1] → [0, 100] before weighting so all
    three inputs live on the same 0–100 scale.
    """
    if weights is None:
        weights = {"attention": 0.40, "memory": 0.40, "emotion": 0.20}

    # ── Guard: reject None signals immediately ────────────────────────────────
    attn   = signals.get("attention_score")
    mem    = signals.get("memory_score")
    val    = signals.get("emotional_valence")
    cl     = signals.get("cognitive_load", "Medium")

    if attn is None or mem is None or val is None:
        raise ValueError(
            f"compute_cpci: signal is None — "
            f"attention={attn}, memory={mem}, valence={val}. "
            "Check map_to_cognitive_signals() output."
        )

    # ── Debug print (remove once stable) ─────────────────────────────────────
    print(f"[CPCi] attention={attn}, memory={mem}, valence={val}, weights={weights}")

    valence_norm = (val + 1) / 2 * 100   # [-1,+1] → [0,100]

    raw = (
        weights["attention"] * attn
        + weights["memory"]   * mem
        + weights["emotion"]  * valence_norm
    )

    if apply_load_penalty:
        raw += LOAD_PENALTY_MAP.get(cl, 0)

    result = round(max(0, min(100, raw)), 1)
    print(f"[CPCi] raw={raw:.2f}  →  cpci={result}")
    return result


def run_pipeline(
    uploaded_file,
    weights:            dict = None,
    apply_load_penalty: bool = False,
    use_case:           str  = "Performance Marketing",
) -> dict:
    """
    Full analysis pipeline for one creative.
    Generates cognitive signals, CPCi, and the narrative intelligence report.
    """
    file_path   = save_upload(uploaded_file)
    features    = analyze_creative(file_path)
    signals_raw = map_to_cognitive_signals(features)
    reasoning   = signals_raw.pop("_reasoning")
    cpci        = compute_cpci(signals_raw, weights, apply_load_penalty)
    narrative   = generate_narrative(
        features  = features,
        signals   = signals_raw,
        cpci      = cpci,
        use_case  = use_case,
        reasoning = reasoning,
    )
    return {
        "name":            uploaded_file.name,
        "file_path":       file_path,
        "visual_features": features,
        "signals":         signals_raw,
        "reasoning":       reasoning,
        "cpci":            cpci,
        "narrative":       narrative,
    }


# ── UI helpers ────────────────────────────────────────────────────────────────

def badge(label: str, style: str) -> str:
    return f"<span class='badge badge-{style}'>{label}</span>"

def score_badge(v: float, low: int = 40, high: int = 70) -> str:
    if v >= high: return "good"
    if v >= low:  return "warn"
    return "bad"

def cpci_color(v: float) -> str:
    if v >= 70: return "#22C55E"
    if v >= 40: return "#F59E0B"
    return "#EF4444"

def color_swatches(hex_colors: list) -> str:
    return "".join(
        f"<span class='color-swatch' style='background:{c};' title='{c}'></span>"
        for c in hex_colors
    )

def short_name(name: str, max_len: int = 20) -> str:
    """Truncate long filenames for table display."""
    return name if len(name) <= max_len else name[:max_len - 3] + "..."


# ── Section card renderer ────────────────────────────────────────────────────

def _section_card(
    icon:        str,
    title:       str,
    accent:      str,
    body:        str,
    score:       object        = None,
    score_label: str           = "",
    pointers:    list          = None,   # list of (label, value, value_color) tuples
) -> None:
    """
    Full narrative section card with:
      - Coloured left-border header (icon + title + optional score pill)
      - Paragraph body text
      - Pointer strip: data chips at the bottom for quick-scan reference
    """
    # Score pill (top-right of header)
    pill_html = ""
    if score is not None:
        pill_html = (
            f"<div class='ns-score-pill' style='background:{accent}14;border:1px solid {accent}44;'>"
            f"<div class='ns-score-val' style='color:{accent};'>{score}</div>"
            f"<div class='ns-score-lbl'>{score_label}</div>"
            f"</div>"
        )

    # Pointer chips
    ptr_html = ""
    if pointers:
        chips = ""
        for lbl, val, vcol in pointers:
            chips += (
                f"<span class='ns-ptr'>"
                f"<span style='color:#94A3B8;'>→</span>&nbsp;"
                f"{lbl}&nbsp;<span class='ns-ptr-val' style='color:{vcol};'>{val}</span>"
                f"</span>"
            )
        ptr_html = f"<div class='ns-pointers'>{chips}</div>"

    # Render \n\n-separated paragraphs as <p> blocks for readability
    paras = [p.strip() for p in body.split("\n\n") if p.strip()]
    body_html = "".join(f"<p>{p}</p>" for p in paras) if paras else f"<p>{body}</p>"

    st.markdown(f"""
    <div class='ns-card' style='border-left:3px solid {accent};'>
      <div class='ns-header' style='background:{accent}08;'>
        <div class='ns-icon-title'>
          <span class='ns-icon'>{icon}</span>
          <span class='ns-title'>{title}</span>
        </div>
        {pill_html}
      </div>
      <div class='ns-body'>{body_html}</div>
      {ptr_html}
    </div>""", unsafe_allow_html=True)


# ── Cognitive Diagnosis renderer ─────────────────────────────────────────────

def _cognitive_diagnosis(
    attn, mem, val, cl, cl_score, vf,
    a_color, a_label, m_color, m_label,
    v_color, v_label, cl_color,
):
    """
    Renders the merged 🧬 Cognitive Diagnosis section.
    Four compact mini-blocks side by side: Attention · Memory · Emotion · Load.
    Each has: score, 2–3 bullets, 1 implication line.
    """

    def _bullets(*items):
        return "".join(
            f"<div class='cd-bullet'>"
            f"<div class='cd-bullet-dot' style='background:{color};margin-top:5px;'></div>"
            f"<span>{text}</span></div>"
            for text, color in items
        )

    def _implication(text, color):
        return (
            f"<div class='cd-implication' style='background:{color}12;"
            f"border-left:3px solid {color};color:{color};'>"
            f"{text}</div>"
        )

    # ── Attention bullets ─────────────────────────────────────────────────────
    face_txt = (
        f"{vf['face_count']} face(s) — will trigger an orienting response and accelerate processing"
        if vf["face_count"] > 0
        else "No face — will not trigger the fastest biological attention mechanism available"
    )
    contrast_txt = (
        f"Contrast {vf['contrast_score']:.0f}/100 — " +
        ("will pass the visual salience gate in a busy feed" if vf["contrast_score"] >= 60
         else "will not pass the visual salience test — disappears in a competitive feed")
    )
    obj_txt = (
        f"{vf['object_count']} object(s) — " +
        ("clean composition — focus will land on the primary subject" if vf["object_count"] <= 4
         else "visual attention will fragment across elements — no single thing will dominate")
    )
    if attn > 60:
        attn_impl = _implication("→ Will interrupt scrolling and trigger processing in cold audiences", a_color)
    elif attn >= 30:
        attn_impl = _implication("→ Will not reliably stop cold audiences — loses the first cognitive gate before the message is seen", a_color)
    else:
        attn_impl = _implication("→ Will be skipped at near-zero processing depth — the brain never engages with the content", a_color)

    # ── Memory bullets ────────────────────────────────────────────────────────
    text_pct = vf["text_density"] * 100
    if text_pct < 5:
        text_mem_txt = f"Text coverage {text_pct:.0f}% — no verbal anchor; the visual alone will not survive working memory"
    elif text_pct <= 25:
        text_mem_txt = f"Text coverage {text_pct:.0f}% — dual-coding range; both visual and verbal channels will encode simultaneously"
    else:
        text_mem_txt = f"Text coverage {text_pct:.0f}% — verbal overload; neither channel will encode cleanly, recall will suffer"

    mem_obj_txt = (
        f"{vf['object_count']} element(s) — " +
        ("sparse enough to form a dominant memory trace" if vf["object_count"] <= 4
         else "too many competing elements — nothing will be remembered as the primary object")
    )
    if mem > 70:
        mem_impl = _implication("→ Will be recognised at point of purchase — brand memory survives the gap between exposure and decision", m_color)
    elif mem >= 40:
        mem_impl = _implication("→ Will not encode on a single exposure — requires 6–8 impressions to build reliable recall, increasing effective CPM", m_color)
    else:
        mem_impl = _implication("→ Will leave no trace — viewers will not recall the brand or message within minutes of scrolling past", m_color)

    # ── Emotion bullets ───────────────────────────────────────────────────────
    val_norm = (val + 1) / 2 * 100
    face_emo = (
        f"{vf['face_count']} face(s) — will drive affiliative warmth and accelerate positive affect"
        if vf["face_count"] > 0
        else "No face — will forfeit the strongest emotion driver available in still imagery"
    )
    palette_hex = vf["dominant_colors"][0] if vf["dominant_colors"] else "#888888"
    r_hex = int(palette_hex[1:3], 16) if len(palette_hex) == 7 else 128
    b_hex = int(palette_hex[5:7], 16) if len(palette_hex) == 7 else 128
    v_hex = (int(palette_hex[1:3], 16) + int(palette_hex[3:5], 16) + b_hex) // 3
    if r_hex > b_hex + 30:
        pal_txt = f"Warm palette ({palette_hex}) — will activate approach affect and lift emotional valence"
    elif b_hex > r_hex + 20:
        pal_txt = f"Cool palette ({palette_hex}) — will signal credibility but will not drive arousal or urgency"
    elif v_hex < 80:
        pal_txt = f"Dark palette ({palette_hex}) — will project premium cues but will suppress warmth and affinity"
    else:
        pal_txt = f"Neutral palette ({palette_hex}) — will generate no emotional signal — a missed valence opportunity"

    if val > 0.1:
        val_impl = _implication("→ Will build positive brand associations with repeated exposure — emotion compounds into long-term affinity", v_color)
    elif val > -0.1:
        val_impl = _implication("→ Will generate no emotional memory — neutral affect means the brand will not benefit from the exposure beyond the impression", v_color)
    else:
        val_impl = _implication("→ Will silently erode brand equity — negative affect embeds subconsciously and accumulates across each impression served", v_color)

    # ── Cognitive Load bullets ────────────────────────────────────────────────
    vis_complexity = (
        f"{vf['object_count']} visual elements — " +
        ("within comfortable processing capacity — brain will not fragment focus" if vf["object_count"] <= 5
         else "exceeds working memory capacity — brain will abandon full processing")
    )
    text_load = (
        f"Text at {text_pct:.0f}% coverage — " +
        ("low verbal demand — will not compete with visual processing" if text_pct <= 15
         else "high verbal demand — audience will skim or skip rather than read")
    )
    if cl == "Low":
        cl_impl = _implication("→ Will process in under 2 seconds — fits feed, display, and OOH without cognitive friction", cl_color)
    elif cl == "Medium":
        cl_impl = _implication("→ Will underperform in fast-scroll formats — requires dwell time the audience will not give", cl_color)
    else:
        cl_impl = _implication("→ Will saturate working memory before the message lands — high load kills attention, memory, and emotion simultaneously", cl_color)

    # ── Build HTML ────────────────────────────────────────────────────────────
    def _block(icon, label, score_val, score_lbl, color, bullets_html, impl_html, sig_color=None):
        lbl_color = sig_color if sig_color else color
        return (
            f"<div class='cd-block'>"
            f"<div class='cd-blk-header'>"
            f"<span class='cd-blk-icon'>{icon}</span>"
            f"<div style='text-align:right;'>"
            f"<div class='cd-blk-score' style='color:{color};'>{score_val}</div>"
            f"<div class='cd-blk-label'>{score_lbl}</div>"
            f"</div></div>"
            f"<div style='font-size:13px;font-weight:700;color:{lbl_color};"
            f"text-transform:uppercase;letter-spacing:0.8px;margin-bottom:8px;'>{label}</div>"
            f"{bullets_html}"
            f"{impl_html}"
            f"</div>"
        )

    attn_block = _block(
        "🎯", _tooltip("Attention", "Attention Score", [
            "→ Measures visual stopping power",
            "→ Based on contrast, faces &amp; clutter",
            "→ Predicts scroll-stop probability",
        ]), attn, a_label, a_color,
        _bullets(
            (face_txt,     a_color),
            (contrast_txt, "#334455"),
            (obj_txt,      "#334455"),
        ),
        attn_impl,
        sig_color="#3B82F6",
    )
    mem_block = _block(
        "🧩", _tooltip("Memory", "Memory Score", [
            "→ Measures brand recall potential",
            "→ Based on text density &amp; visual simplicity",
            "→ Drives recognition at point of purchase",
        ]), mem, m_label, m_color,
        _bullets(
            (text_mem_txt, m_color),
            (mem_obj_txt,  "#334455"),
        ),
        mem_impl,
        sig_color="#8B5CF6",
    )
    val_block = _block(
        "💭", _tooltip("Emotion", "Emotional Valence", [
            "→ Measures positive vs. negative tone",
            "→ Derived from face expression &amp; color warmth",
            "→ Shapes brand affinity &amp; purchase intent",
        ]), f"{val:+.2f}", v_label, v_color,
        _bullets(
            (face_emo,  v_color),
            (pal_txt,   "#334455"),
            (f"Valence {val_norm:.0f}/100 on the positive–negative scale", "#334455"),
        ),
        val_impl,
        sig_color="#EC4899",
    )
    cl_block = _block(
        "⚙️", "Cog. Load", cl, f"score {cl_score:.0f}/100", cl_color,
        _bullets(
            (vis_complexity, cl_color),
            (text_load,      "#334455"),
        ),
        cl_impl,
        sig_color="#F59E0B",
    )

    st.markdown(f"""
    <div class='cd-wrap'>
      <div class='cd-header'>
        <div class='cd-title'>🧬 Cognitive Diagnosis</div>
        <div style='font-size:13px;color:#94A3B8;'>What each signal predicts about real-world performance</div>
      </div>
      <div class='cd-grid'>
        {attn_block}
        {mem_block}
        {val_block}
        {cl_block}
      </div>
    </div>""", unsafe_allow_html=True)


# ── Recommendations renderer ─────────────────────────────────────────────────

import re as _re

_REC_META = {
    # keyword → (icon, color)
    "priority":  ("🎯", "#3B82F6"),
    "attention": ("🎯", "#3B82F6"),
    "memory":    ("🧩", "#8B5CF6"),
    "emotion":   ("💭", "#EC4899"),
    "load":      ("⚙️", "#F59E0B"),
    "scaling":   ("🧪", "#ce93d8"),
    "a/b":       ("🧪", "#ce93d8"),
    "before":    ("🧪", "#ce93d8"),
}

def _rec_meta(label: str):
    """Return (icon, color) for a recommendation label."""
    lo = label.lower()
    for kw, meta in _REC_META.items():
        if kw in lo:
            return meta
    return ("→", "#667788")

def _render_recommendations(body: str, pointers: list) -> None:
    """
    Premium Recommendations panel.
    Parses **Label:** body paragraphs into numbered, color-coded action cards.
    Falls back to plain paragraph rendering if no bold labels found.
    """
    paragraphs = [p.strip() for p in body.split("\n\n") if p.strip()]

    # Parse each paragraph into (label, body_text)
    parsed = []
    for para in paragraphs:
        m = _re.match(r'^\*\*(.+?):\*\*\s*(.*)', para, _re.DOTALL)
        if m:
            parsed.append((m.group(1).strip(), m.group(2).strip()))
        else:
            parsed.append(("", para))

    # Build item HTML
    items_html = ""
    for idx, (label, text) in enumerate(parsed, 1):
        icon, color = _rec_meta(label) if label else ("→", "#667788")
        badge_html = (
            f"<div class='rp-badge' style='background:{color}18;"
            f"border:1.5px solid {color}55;color:{color};'>{icon}</div>"
        )
        lbl_html = ""
        if label:
            lbl_html = (
                f"<div class='rp-lbl' style='color:{color};'>"
                f"<span class='rp-lbl-dot' style='background:{color};'></span>"
                f"{label}"
                f"</div>"
            )
        items_html += (
            f"<div class='rp-item'>"
            f"  <div class='rp-badge-col'>{badge_html}</div>"
            f"  <div class='rp-content'>{lbl_html}"
            f"    <div class='rp-body'>{text}</div>"
            f"  </div>"
            f"</div>"
        )

    # Build pointer chips
    chips_html = ""
    for lbl, val, vcol in (pointers or []):
        chips_html += (
            f"<span class='ns-ptr'>"
            f"<span style='color:#94A3B8;'>→</span>&nbsp;"
            f"{lbl}&nbsp;<span class='ns-ptr-val' style='color:{vcol};'>{val}</span>"
            f"</span>"
        )

    count = len(parsed)
    st.markdown(f"""
    <div class='rp-wrap'>
      <div class='rp-header'>
        <span class='rp-icon'>🛠</span>
        <span class='rp-title'>Optimization Recommendations</span>
        <span class='rp-count'>{count} ACTION{'S' if count != 1 else ''}</span>
      </div>
      <div class='rp-list'>{items_html}</div>
      <div class='rp-footer'>{chips_html}</div>
    </div>""", unsafe_allow_html=True)


# ── Quick Read generator ──────────────────────────────────────────────────────

def _verdict_color(cpci: float, val: float, cl: str) -> str:
    if cpci >= 70 and cl != "High":        return "#22C55E"
    if cpci >= 55 and val >= -0.05:        return "#F59E0B"
    if cpci >= 40:                          return "#EF4444"
    return "#EF4444"


def _final_verdict(
    cpci: float, attn: int, mem: int, val: float, cl: str, use_case: str,
) -> None:
    """Renders the Final Verdict strip in single-creative view."""
    verdict = _final_verdict_text(cpci, attn, mem, val, cl, use_case)
    vcolor  = _verdict_color(cpci, val, cl)
    st.markdown(
        f"<div style='margin:16px 0 4px 0;padding:14px 16px;background:#0B0F14;"
        f"border-left:3px solid {vcolor};border-radius:0 8px 8px 0;'>"
        f"<div style='font-size:9px;letter-spacing:1.8px;text-transform:uppercase;"
        f"font-weight:700;color:#94A3B8;margin-bottom:6px;'>Final Verdict</div>"
        f"<div style='font-size:13px;font-weight:800;color:{vcolor};line-height:1.4;'>"
        f"{verdict}</div>"
        f"</div>",
        unsafe_allow_html=True,
    )


def _quick_read(
    cpci:     float,
    attn:     int,
    mem:      int,
    val:      float,
    cl:       str,
    vf:       dict,
    use_case: str,
) -> None:
    """
    Renders the 🧠 What This Means (Quick Read) banner.
    Three plain-English lines: performance prediction, core issue, immediate fix.
    No technical language. Max one sentence each.
    """

    # ── Use-case context plain name ────────────────────────────────────────────
    ctx = {
        "FMCG Branding":         "brand awareness campaigns",
        "Performance Marketing": "paid social and search ads",
        "Retail Media":          "retail and in-store placements",
    }.get(use_case, "this campaign type")

    # ── Line 1 — Performance Prediction ───────────────────────────────────────
    if cpci >= 70:
        prediction = f"This creative will earn its media spend in {ctx} — the cognitive gates are clear."
    elif cpci >= 55:
        prediction = f"This creative will generate impressions in {ctx} but will not convert efficiently — one weak signal is costing reach."
    elif cpci >= 40:
        prediction = f"This creative will struggle in cold audiences for {ctx} — it may survive retargeting but will not scale profitably."
    else:
        prediction = f"This creative will underperform regardless of budget — the cognitive barriers are too significant to overcome with spend."

    # ── Line 2 — Core Issue ───────────────────────────────────────────────────
    attn_gap  = max(0, 60 - attn)
    mem_gap   = max(0, 70 - mem)
    val_gap   = max(0, 0.1 - val) * 100
    load_gap  = 25 if cl == "High" else (10 if cl == "Medium" else 0)

    worst     = max(attn_gap, mem_gap, val_gap, load_gap)

    if worst == 0:
        issue = "All signals are above threshold — no single dimension is holding this back."
    elif worst == load_gap and cl == "High":
        issue = "Visual overload will cause viewers to disengage before the message registers — the brain cannot parse this quickly enough in a feed."
    elif worst == attn_gap:
        if vf.get("face_count", 0) == 0:
            issue = "This creative will fail to trigger an orienting response — without a face or salient focal point, the brain will not interrupt scrolling to process it."
        elif vf.get("contrast_score", 50) < 45:
            issue = "This creative will disappear in a busy feed — insufficient contrast means it fails the first visual gate before content is even assessed."
        else:
            issue = "This creative will be processed at low depth — no dominant focal point means the brain distributes attention thinly and nothing is prioritised."
    elif worst == mem_gap:
        if vf.get("text_density", 0) > 0.25:
            issue = "This creative will be remembered only if repeatedly exposed — text overload prevents dual-coding, so neither the visual nor verbal message encodes cleanly."
        elif vf.get("text_density", 0) < 0.05:
            issue = "This creative will be seen but not recalled — without a verbal anchor, the visual alone will not survive beyond a few seconds in working memory."
        else:
            issue = "This creative will not build brand recall efficiently — too many competing elements mean nothing dominates the memory trace."
    elif worst == val_gap:
        if vf.get("face_count", 0) == 0:
            issue = "This creative will generate neutral-to-negative affect on each exposure — without a human face, the palette alone cannot drive positive brand association."
        else:
            issue = "This creative will subtly undermine brand affinity over time — the colour palette is triggering mild avoidance without the viewer being aware of it."
    elif worst == load_gap:
        issue = "This creative demands more cognitive effort than a scrolling audience will commit — the message will not land in fast-feed placements."
    else:
        issue = "Attention, memory, and emotion are pulling in different directions — the creative is sending mixed signals that dilute overall cognitive impact."

    # ── Line 3 — Immediate Fix ────────────────────────────────────────────────
    if vf.get("face_count", 0) == 0 and attn_gap > 15:
        fix = "Introduce a human face as the dominant visual element — it is the single fastest mechanism for triggering an orienting response in feed."
    elif vf.get("contrast_score", 50) < 45 and attn_gap > 10:
        fix = "Increase foreground-to-background contrast significantly — the creative needs to pass the visual salience test before any other signal matters."
    elif vf.get("object_count", 0) > 7 and (worst == load_gap or worst == mem_gap):
        fix = "Eliminate all secondary visual elements and commit to a single hero — cognitive load is suppressing every other signal."
    elif vf.get("text_density", 0) > 0.28:
        fix = "Reduce copy to a single declarative line under six words — audiences process far less text than advertisers write."
    elif vf.get("text_density", 0) < 0.04 and mem_gap > 15:
        fix = "Add a 4–6 word brand or message line — the verbal channel is completely unused, which is costing you recall without any benefit."
    elif val < -0.05 and vf.get("face_count", 0) == 0:
        fix = "Add a person with a natural, warm expression — emotional valence cannot be fixed with colour alone at this deficit."
    elif val < -0.05:
        fix = "Replace the dominant cool or dark tones with warmer equivalents — the palette is the primary driver of the negative valence reading."
    elif attn_gap > 10:
        fix = "Scale up the primary subject and increase its contrast against the background — give the eye an unmissable landing point."
    elif mem_gap > 15:
        fix = "Reduce the composition to one dominant image and one short message — simplicity is the only reliable route to single-exposure recall."
    else:
        fix = "Test a version with a human face replacing the current hero element — it will lift both attention and emotional valence simultaneously."

    # ── Render ─────────────────────────────────────────────────────────────────
    pcolor = "#22C55E" if cpci >= 70 else ("#F59E0B" if cpci >= 40 else "#EF4444")

    st.markdown(f"""
    <div class='qr-wrap'>
      <div class='qr-heading'>🧠&nbsp; What This Means</div>
      <div class='qr-row'>
        <span class='qr-num' style='color:{pcolor};'>01</span>
        <span class='qr-tag' style='color:{pcolor};'>Performance</span>
        <span class='qr-line'>{prediction}</span>
      </div>
      <div class='qr-row'>
        <span class='qr-num' style='color:#F59E0B;'>02</span>
        <span class='qr-tag' style='color:#F59E0B;'>Core Issue</span>
        <span class='qr-line'>{issue}</span>
      </div>
      <div class='qr-row'>
        <span class='qr-num' style='color:#3B82F6;'>03</span>
        <span class='qr-tag' style='color:#3B82F6;'>Fix First</span>
        <span class='qr-line'>{fix}</span>
      </div>
    </div>""", unsafe_allow_html=True)


# ── Why This Matters hero block ───────────────────────────────────────────────

def _why_this_matters(
    cpci: float,
    attn: int,
    mem:  int,
    val:  float,
    cl:   str,
    use_case: str,
) -> None:
    """
    Full-width emotional impact statement placed directly after the CPCi number.
    Goal: make the user FEEL the score, not just read it.
    """

    # ── Tier-adaptive copy ────────────────────────────────────────────────────
    if cpci < 30:
        accent     = "#EF4444"
        bg         = "rgba(239,68,68,0.07)"
        border_col = "#7F1D1D"
        icon       = "🚨"
        bold_line  = (
            "This creative is likely wasting 50–70% of your media budget "
            "due to critically low cognitive engagement."
        )
        support    = (
            f"At CPCi {cpci}, the brain won't reliably process this message. "
            "Impressions served are impressions lost — "
            "no targeting strategy can compensate for a creative the brain ignores."
        )

    elif cpci < 40:
        accent     = "#EF4444"
        bg         = "rgba(239,68,68,0.06)"
        border_col = "#7F1D1D"
        icon       = "🚨"
        bold_line  = (
            "This creative is likely wasting 40–55% of your media budget "
            "due to low cognitive engagement."
        )
        support    = (
            f"At CPCi {cpci}, most impressions will not cognitively register. "
            "The brand message won't be encoded — and won't be recalled at point of purchase."
        )

    elif cpci < 55:
        accent     = "#F59E0B"
        bg         = "rgba(245,158,11,0.06)"
        border_col = "#78350F"
        icon       = "⚠️"
        bold_line  = "This creative is leaving 25–40% of its potential media efficiency on the table."
        support    = (
            f"At CPCi {cpci}, you have signal — but the brain is only partially engaged. "
            "You'll need higher frequency to achieve the same recall as a top-quartile creative "
            "at half the impressions."
        )

    elif cpci < 70:
        accent     = "#F59E0B"
        bg         = "rgba(245,158,11,0.05)"
        border_col = "#78350F"
        icon       = "⚠️"
        bold_line  = (
            "This creative is performing adequately — but there is a gap before it earns "
            "top-quartile media efficiency."
        )
        support    = (
            f"At CPCi {cpci}, you're in the average band. A single targeted fix — "
            "attention, memory, or load — could close the gap and meaningfully reduce "
            "your effective cost per recalled impression."
        )

    else:
        accent     = "#22C55E"
        bg         = "rgba(34,197,94,0.06)"
        border_col = "#14532D"
        icon       = "✅"
        bold_line  = (
            "This creative earns every impression — the brain is processing, "
            "encoding, and responding at above-average efficiency."
        )
        support    = (
            f"At CPCi {cpci}, this is top-quartile cognitive performance. "
            "Your media spend is working at maximum brain-level impact. "
            "Scale with confidence."
        )

    # ── Media cost multiplier (uses mem, the function parameter) ─────────────
    effective_memory = max(mem, 10)
    multiplier       = round(70 / effective_memory, 1)

    if mem < 70:
        mult_color = (
            "#EF4444" if multiplier > 1.8 else
            "#F59E0B" if multiplier >= 1.2 else
            "#22C55E"
        )
        media_cost_html = (
            f"<div style='font-size:18px;font-weight:800;color:{mult_color};"
            f"line-height:1.5;margin-bottom:6px;'>"
            f"🔥 This creative will cost you {multiplier}× more media "
            f"to achieve the same recall."
            f"</div>"
        )
        waste_html = (
            "<div style='font-size:14px;font-weight:600;color:#F87171;"
            "margin-bottom:18px;'>"
            "Equivalent to wasting ~35–50% of your media budget on ineffective impressions."
            "</div>"
        ) if multiplier > 1.5 else "<div style='margin-bottom:18px;'></div>"
    else:
        media_cost_html = (
            "<div style='font-size:18px;font-weight:700;color:#22C55E;"
            "line-height:1.5;margin-bottom:18px;'>"
            "This creative is operating at efficient memory levels — no excess media cost."
            "</div>"
        )
        waste_html = ""

    # ── Render ────────────────────────────────────────────────────────────────
    # Line 1 — eyebrow
    st.markdown(
        f"<div style='background:{bg};border:1px solid {border_col};"
        f"border-left:4px solid {accent};border-radius:16px;"
        f"padding:36px 40px 32px 40px;margin:0 0 40px 0;'>"

        # Eyebrow: "🚨 WHAT THIS MEANS FOR YOUR MEDIA SPEND"
        f"<div style='font-size:11px;font-weight:700;color:{accent};"
        f"letter-spacing:2px;text-transform:uppercase;margin-bottom:18px;'>"
        f"{icon}&nbsp;&nbsp;WHAT THIS MEANS FOR YOUR MEDIA SPEND"
        f"</div>"

        # Line 2 — bold emotional sentence
        f"<div style='font-size:26px;font-weight:800;color:#FFFFFF;"
        f"line-height:1.3;letter-spacing:-0.3px;margin-bottom:14px;'>"
        f"{bold_line}"
        f"</div>"

        # Line 3 — media cost multiplier
        + media_cost_html

        # Line 4 — waste callout (conditional on multiplier > 1.5)
        + waste_html

        # Line 5 — supporting explanation
        + f"<div style='font-size:15px;color:#CBD5E1;line-height:1.7;"
        f"font-weight:400;max-width:760px;'>"
        f"{support}"
        f"</div>"

        f"</div>",
        unsafe_allow_html=True,
    )


# ── Business Impact ───────────────────────────────────────────────────────────

def _business_impact(
    cpci:     float,
    attn:     int,
    mem:      int,
    val:      float,
    cl:       str,
    use_case: str,
) -> None:
    """
    CMO-facing Business Impact panel.
    Translates CPCi into commercial language: waste risk, media efficiency,
    and a single deployment recommendation.
    No technical signal language — decisions only.
    """

    # ── Tier classification ───────────────────────────────────────────────────
    if cpci >= 70:
        risk_label  = "High Efficiency"
        risk_color  = "#22C55E"
        risk_bg     = "#22C55E14"
        risk_border = "#22C55E44"
        headline    = "This creative is ready to scale — cognitive signals clear."
        sub         = (
            "Attention, memory, and emotion are working in alignment. "
            "Media spend behind this creative is likely to generate strong cognitive impact at scale."
        )
        deploy_rec  = "Deploy with confidence. Scale budget progressively."
        deploy_color = "#22C55E"
    elif cpci >= 40:
        risk_label  = "Moderate Performance"
        risk_color  = "#F59E0B"
        risk_bg     = "#F59E0B14"
        risk_border = "#F59E0B44"
        headline    = "This creative will generate results — but not at full media efficiency."
        sub         = (
            "Some impressions will land, many will not. "
            "One weak cognitive signal is suppressing returns. "
            "A targeted fix could unlock significantly better performance before spend increases."
        )
        deploy_rec  = "Deploy selectively. Fix the weakest signal before scaling."
        deploy_color = "#F59E0B"
    else:
        risk_label  = "High Waste Risk"
        risk_color  = "#EF4444"
        risk_bg     = "#EF444414"
        risk_border = "#EF444444"
        headline    = "This creative is likely to underperform in paid media."
        sub         = (
            "High risk of wasted impressions. "
            "The cognitive barriers are significant enough that no targeting, bidding, "
            "or placement strategy will compensate for them at scale."
        )
        deploy_rec  = "Do not deploy. Rebuild creative before any media spend."
        deploy_color = "#EF4444"

    # ── Media efficiency impact bullets ──────────────────────────────────────
    if cpci >= 70:
        efficiency_bullets = [
            ("Impression quality",     "High — creatives at this level typically drive 2–3× better recall than average",  "#22C55E"),
            ("Conversion signal",      "Strong — attention and memory are both above threshold for cold-audience response", "#22C55E"),
            ("Brand equity",           "Positive accumulation — each impression builds durable brand memory",               "#22C55E"),
            ("Recommended budget",     f"Open to full {use_case} spend — cognitive signal supports scale",                 "#22C55E"),
        ]
    elif cpci >= 55:
        efficiency_bullets = [
            ("Impression quality",     "Moderate — mixed signals mean a significant share of impressions will not engage",  "#F59E0B"),
            ("Conversion signal",      "Weak on cold audiences — restrict to warm audiences and lookalikes initially",      "#F59E0B"),
            ("Brand equity",           "Partial — some recall will build but not at efficient frequency",                   "#F59E0B"),
            ("Recommended budget",     "Limit cold-audience spend until the weakest signal is fixed",                       "#F59E0B"),
        ]
    elif cpci >= 40:
        efficiency_bullets = [
            ("Impression quality",     "Low — majority of impressions will generate no measurable cognitive response",      "#EF4444"),
            ("Conversion signal",      "Retargeting only — not viable for cold-audience acquisition",                       "#F59E0B"),
            ("Brand equity",           "Minimal to neutral — recall is unlikely to build at current signal levels",         "#F59E0B"),
            ("Recommended budget",     "Cap spend — retargeting with strict frequency cap (2–3× per user) only",            "#F59E0B"),
        ]
    else:
        efficiency_bullets = [
            ("Impression quality",     "Very low — impressions are likely to generate no cognitive engagement",             "#EF4444"),
            ("Conversion signal",      "None — cognitive barriers prevent any reliable conversion mechanism",                "#EF4444"),
            ("Brand equity",           "Negative risk — repeated exposure to an ineffective creative can erode brand recall","#EF4444"),
            ("Recommended budget",     "Zero — do not commit media spend to this creative in its current state",            "#EF4444"),
        ]

    # ── Visible summary (always shown — 2 lines max) ─────────────────────────
    st.markdown(
        f"<div style='background:#141B24;border:1px solid #1F2937;"
        f"border-left:3px solid {risk_color};"
        f"border-radius:14px;padding:20px 24px;margin:0 0 4px 0;'>"

        f"<div style='font-size:11px;font-weight:700;color:#4B5563;"
        f"letter-spacing:1.8px;text-transform:uppercase;margin-bottom:12px;'>"
        f"💼&nbsp; Business Impact</div>"

        # Badge + headline on one row
        f"<div style='display:flex;align-items:center;gap:14px;flex-wrap:wrap;margin-bottom:12px;'>"
        f"<div style='background:{risk_bg};border:1px solid {risk_border};"
        f"border-radius:6px;padding:4px 12px;flex-shrink:0;'>"
        f"<span style='font-size:11px;font-weight:700;color:{risk_color};"
        f"letter-spacing:1px;text-transform:uppercase;'>{risk_label}</span>"
        f"</div>"
        f"<div style='font-size:17px;font-weight:600;color:#FFFFFF;line-height:1.3;flex:1;'>"
        f"{headline}</div>"
        f"</div>"

        # Deployment decision — single line, always visible
        f"<div style='font-size:13px;color:{deploy_color};font-weight:600;'>"
        f"→&nbsp; {deploy_rec}"
        f"</div>"

        f"</div>",
        unsafe_allow_html=True,
    )

    # ── Detail expander ───────────────────────────────────────────────────────
    with st.expander("View media efficiency breakdown"):
        rows_html = ""
        for label, value, color in efficiency_bullets:
            rows_html += (
                f"<div style='display:flex;align-items:flex-start;gap:12px;"
                f"padding:10px 0;border-bottom:1px solid #1F2937;'>"
                f"<div style='width:160px;flex-shrink:0;font-size:12px;font-weight:600;"
                f"color:#94A3B8;letter-spacing:0.3px;'>{label}</div>"
                f"<div style='flex:1;font-size:12px;color:{color};line-height:1.5;'>{value}</div>"
                f"</div>"
            )
        st.markdown(
            f"<div style='font-size:13px;color:#CBD5E1;line-height:1.7;margin-bottom:16px;'>"
            f"{sub}</div>"
            f"<div style='font-size:11px;font-weight:700;color:#4B5563;"
            f"letter-spacing:1.2px;text-transform:uppercase;margin-bottom:4px;'>"
            f"Expected Media Efficiency Impact</div>"
            f"{rows_html}",
            unsafe_allow_html=True,
        )

    st.markdown("<div style='margin-bottom:32px;'></div>", unsafe_allow_html=True)


# ── Creative Optimization Scenario ───────────────────────────────────────────

def _compute_scenarios(
    cpci: float, attn: int, mem: int, val: float, cl: str, vf: dict, use_case: str
) -> list:
    """
    Compute up to 4 independent CPCi lift scenarios.
    Each uses the real CPCi formula with the actual use-case weights.
    Returns a list of dicts sorted by lift descending.
    """
    w        = USE_CASES[use_case]["weights"]
    val_norm = (val + 1) / 2 * 100          # [-1,+1] → [0,100]
    load_pen = LOAD_PENALTY_MAP.get(cl, 0)

    def _new_cpci(new_attn=attn, new_mem=mem, new_val_norm=val_norm, new_load_pen=load_pen):
        raw = (w["attention"]*new_attn + w["memory"]*new_mem
               + w["emotion"]*new_val_norm + new_load_pen)
        return round(max(0.0, min(100.0, raw)), 1)

    face_count  = vf.get("face_count", 0)
    contrast    = vf.get("contrast_score", 50)
    obj_count   = vf.get("object_count", 4)
    text_pct    = vf.get("text_density", 0.1) * 100

    scenarios = []

    # ── Scenario A: attention improvement ────────────────────────────────────
    attn_gap = max(0, 65 - attn)
    if attn_gap >= 8:
        lift_pts  = min(attn_gap, 22)
        new_a     = min(attn + lift_pts, 100)
        projected = _new_cpci(new_attn=new_a)
        delta     = round(projected - cpci, 1)
        if delta >= 2:
            if face_count == 0:
                action  = "Add a human face as the primary visual subject"
                rationale = "Face presence triggers the brain's fastest attention mechanism — an orienting response in under 13ms."
            elif contrast < 45:
                action  = "Increase contrast ratio to at least 4.5:1 against the background"
                rationale = "Pre-attentive salience is driven by contrast. Below threshold, the creative disappears in a competitive feed."
            else:
                action  = "Simplify to one dominant focal point — remove competing visual elements"
                rationale = "Fragmented compositions split visual attention. A single dominant subject maximises stopping power."
            scenarios.append({
                "signal":     "Attention",
                "sig_color":  "#3B82F6",
                "action":     action,
                "rationale":  rationale,
                "method":     "improving attention signal",
                "from_signal": attn, "to_signal": new_a,
                "from_cpci":  cpci,  "to_cpci":  projected,
                "lift":       delta,
            })

    # ── Scenario B: memory improvement ────────────────────────────────────────
    mem_gap = max(0, 70 - mem)
    if mem_gap >= 8:
        lift_pts  = min(mem_gap, 22)
        new_m     = min(mem + lift_pts, 100)
        projected = _new_cpci(new_mem=new_m)
        delta     = round(projected - cpci, 1)
        if delta >= 2:
            if text_pct < 5:
                action  = "Add a short brand tagline or product callout overlay"
                rationale = "Dual-coding (visual + verbal) simultaneously encodes two memory traces — significantly lifting recall."
            elif obj_count > 5:
                action  = "Reduce to one dominant product or hero image"
                rationale = "Memory encodes the most salient object. Competing elements prevent a single strong trace from forming."
            else:
                action  = "Strengthen logo placement and brand mnemonic visibility"
                rationale = "Recognition at point of purchase requires a clear brand anchor encoded during ad exposure."
            scenarios.append({
                "signal":     "Memory",
                "sig_color":  "#8B5CF6",
                "action":     action,
                "rationale":  rationale,
                "method":     "improving memory encoding",
                "from_signal": mem, "to_signal": new_m,
                "from_cpci":  cpci, "to_cpci": projected,
                "lift":       delta,
            })

    # ── Scenario C: load reduction (High → Medium) ────────────────────────────
    if cl == "High":
        new_pen   = LOAD_PENALTY_MAP.get("Medium", -5)
        projected = _new_cpci(new_load_pen=new_pen)
        delta     = round(projected - cpci, 1)
        if delta >= 2:
            scenarios.append({
                "signal":    "Cognitive Load",
                "sig_color": "#F59E0B",
                "action":    "Reduce on-screen text and visual elements by 40%",
                "rationale": "High cognitive load causes viewers to abandon processing before the message registers — no targeting fix compensates for this.",
                "method":    "reducing cognitive load from High to Medium",
                "from_signal": "High", "to_signal": "Medium",
                "from_cpci": cpci, "to_cpci": projected,
                "lift":      delta,
            })

    # ── Scenario D: valence improvement ───────────────────────────────────────
    val_gap = max(0.0, 0.15 - val)
    if val_gap >= 0.12:
        new_val  = min(val + 0.28, 1.0)
        new_vn   = (new_val + 1) / 2 * 100
        projected = _new_cpci(new_val_norm=new_vn)
        delta    = round(projected - cpci, 1)
        if delta >= 2:
            if face_count == 0:
                action  = "Introduce a human face with a positive expression"
                rationale = "Facial expressions are the most direct driver of emotional valence — warmth and affinity are triggered involuntarily."
            else:
                action  = "Shift colour palette toward warmer tones (amber, coral, warm white)"
                rationale = "Cool and neutral palettes suppress approach affect. Warm palettes activate positive emotional responses without conscious effort."
            scenarios.append({
                "signal":    "Emotion",
                "sig_color": "#EC4899",
                "action":    action,
                "rationale": rationale,
                "method":    "improving emotional valence",
                "from_signal": f"{val:+.2f}", "to_signal": f"{new_val:+.2f}",
                "from_cpci": cpci, "to_cpci": projected,
                "lift":      delta,
            })

    scenarios.sort(key=lambda s: s["lift"], reverse=True)
    return scenarios


def _optimization_scenario(
    cpci: float, attn: int, mem: int, val: float, cl: str, vf: dict, use_case: str
) -> None:
    """
    CMO-facing optimization scenario card.
    Shows before/after CPCi, potential lift, and specific actionable fixes.
    """
    scenarios = _compute_scenarios(cpci, attn, mem, val, cl, vf, use_case)

    if not scenarios:
        # All signals already at or above target — no meaningful lift available
        st.markdown(
            "<div style='background:#141B24;border:1px solid #1F2937;border-left:3px solid #22C55E;"
            "border-radius:16px;padding:28px 36px;margin:0 0 40px 0;'>"
            "<div style='font-size:13px;font-weight:600;color:#94A3B8;letter-spacing:1.5px;"
            "text-transform:uppercase;margin-bottom:12px;'>🎯&nbsp; Creative Optimization Scenario</div>"
            "<div style='font-size:17px;color:#22C55E;font-weight:600;'>"
            "All cognitive signals are above threshold — no single optimization is likely to "
            "produce meaningful additional lift. This creative is ready to scale.</div>"
            "</div>",
            unsafe_allow_html=True,
        )
        return

    best      = scenarios[0]
    from_c    = int(round(best["from_cpci"]))
    to_c      = int(round(best["to_cpci"]))
    lift      = best["lift"]
    sig_color = best["sig_color"]

    # Tier label helper
    def _tier(c):
        if c >= 70:   return ("High Efficiency",    "#22C55E")
        if c >= 40:   return ("Moderate Performance","#F59E0B")
        return              ("High Waste Risk",      "#EF4444")

    from_tier_label, from_tier_color = _tier(from_c)
    to_tier_label,   to_tier_color   = _tier(to_c)
    tier_change = from_tier_label != to_tier_label

    # ── Before/after progress bar ─────────────────────────────────────────────
    bar_from = f"{from_c}%"
    bar_to   = f"{to_c}%"

    # Secondary scenario bullets
    secondary_html = ""
    for s in scenarios[1:3]:            # show up to 2 more
        secondary_html += (
            f"<div style='display:flex;align-items:flex-start;gap:10px;"
            f"padding:10px 0;border-bottom:1px solid #1F2937;'>"
            f"<div style='width:6px;height:6px;border-radius:50%;background:{s['sig_color']};"
            f"margin-top:5px;flex-shrink:0;'></div>"
            f"<div style='flex:1;'>"
            f"<span style='font-size:12px;font-weight:600;color:{s['sig_color']};"
            f"text-transform:uppercase;letter-spacing:0.8px;'>{s['signal']}</span>"
            f"<span style='font-size:12px;color:#94A3B8;'>&nbsp;·&nbsp;"
            f"+{s['lift']:.0f} pts lift&nbsp;·&nbsp;</span>"
            f"<span style='font-size:13px;color:#CBD5E1;'>{s['action']}</span>"
            f"</div>"
            f"</div>"
        )

    tier_transition_html = ""
    if tier_change:
        tier_transition_html = (
            f"<div style='display:inline-flex;align-items:center;gap:8px;"
            f"background:#0B0F14;border:1px solid #1F2937;border-radius:8px;"
            f"padding:6px 12px;margin-bottom:24px;'>"
            f"<span style='font-size:12px;color:{from_tier_color};font-weight:600;"
            f"text-transform:uppercase;letter-spacing:0.8px;'>{from_tier_label}</span>"
            f"<span style='color:#94A3B8;font-size:13px;'>→</span>"
            f"<span style='font-size:12px;color:{to_tier_color};font-weight:600;"
            f"text-transform:uppercase;letter-spacing:0.8px;'>{to_tier_label}</span>"
            f"</div>"
        )

    # ── Visible summary — before/after + primary action (always shown) ───────
    st.markdown(
        f"<div style='background:#141B24;border:1px solid #1F2937;"
        f"border-left:3px solid {sig_color};"
        f"border-radius:14px;padding:20px 24px;margin:0 0 4px 0;'>"

        f"<div style='font-size:11px;font-weight:700;color:#4B5563;"
        f"letter-spacing:1.8px;text-transform:uppercase;margin-bottom:14px;'>"
        f"🎯&nbsp; Optimization Scenario</div>"

        f"{tier_transition_html}"

        # Before / After numbers — compact
        f"<div style='display:flex;align-items:center;gap:16px;margin-bottom:16px;'>"
        f"<div style='text-align:center;'>"
        f"<div style='font-size:10px;color:#4B5563;letter-spacing:1.2px;"
        f"text-transform:uppercase;margin-bottom:4px;'>Now</div>"
        f"<div style='font-size:44px;font-weight:700;color:{from_tier_color};"
        f"line-height:1;letter-spacing:-1px;'>{from_c}</div>"
        f"</div>"
        f"<div style='flex:1;text-align:center;'>"
        f"<div style='font-size:22px;color:#4B5563;'>→</div>"
        f"<div style='display:inline-block;background:{sig_color}18;"
        f"border:1px solid {sig_color}44;border-radius:16px;padding:3px 12px;margin-top:4px;'>"
        f"<span style='font-size:13px;font-weight:700;color:{sig_color};'>+{lift:.0f} pts</span>"
        f"</div>"
        f"</div>"
        f"<div style='text-align:center;'>"
        f"<div style='font-size:10px;color:#4B5563;letter-spacing:1.2px;"
        f"text-transform:uppercase;margin-bottom:4px;'>Projected</div>"
        f"<div style='font-size:44px;font-weight:700;color:{to_tier_color};"
        f"line-height:1;letter-spacing:-1px;'>{to_c}</div>"
        f"</div>"
        f"</div>"

        # Primary action — one line
        f"<div style='background:{sig_color}0D;border-left:3px solid {sig_color};"
        f"border-radius:0 8px 8px 0;padding:10px 14px;'>"
        f"<span style='font-size:11px;font-weight:700;color:{sig_color};"
        f"text-transform:uppercase;letter-spacing:0.8px;'>{best['signal']}&nbsp;·&nbsp;</span>"
        f"<span style='font-size:13px;color:#FFFFFF;font-weight:500;'>{best['action']}</span>"
        f"</div>"

        f"</div>",
        unsafe_allow_html=True,
    )

    # ── Detail expander — rationale + secondary scenarios ─────────────────────
    with st.expander("Why this works + additional opportunities"):
        # Progress bar
        st.markdown(
            f"<div style='background:#0B0F14;border-radius:6px;height:6px;margin-bottom:20px;"
            f"overflow:hidden;position:relative;'>"
            f"<div style='position:absolute;left:0;top:0;height:100%;width:{bar_from};"
            f"background:{from_tier_color};border-radius:6px;opacity:0.35;'></div>"
            f"<div style='position:absolute;left:0;top:0;height:100%;width:{bar_to};"
            f"background:{to_tier_color};border-radius:6px;'></div>"
            f"</div>"
            f"<div style='font-size:14px;color:#FFFFFF;font-weight:500;line-height:1.6;"
            f"margin-bottom:8px;'>"
            f"Improving {best['method']} could increase CPCi from "
            f"<span style='color:{from_tier_color};font-weight:700;'>{from_c}</span> → "
            f"<span style='color:{to_tier_color};font-weight:700;'>{to_c}</span>.</div>"
            f"<div style='font-size:13px;color:#94A3B8;line-height:1.65;margin-bottom:16px;'>"
            f"{best['rationale']}</div>"
            + (
                f"<div style='font-size:11px;font-weight:700;color:#4B5563;"
                f"letter-spacing:1.2px;text-transform:uppercase;margin-bottom:8px;'>"
                f"Additional Opportunities</div>"
                f"{secondary_html}"
                if secondary_html else ""
            ),
            unsafe_allow_html=True,
        )

    st.markdown("<div style='margin-bottom:32px;'></div>", unsafe_allow_html=True)


# ── Creative Brief ────────────────────────────────────────────────────────────

def _creative_brief(
    cpci:     float,
    attn:     int,
    mem:      int,
    val:      float,
    cl:       str,
    vf:       dict,
    use_case: str,
) -> None:
    """
    Compact, executive-style brief. One panel. Every line is a decision.
    Format mirrors a media/creative brief — diagnosis, issue, fix, use / don't use.
    """
    face      = vf.get("face_count", 0) > 0
    contrast  = vf.get("contrast_score", 50)
    objects   = vf.get("object_count", 4)
    text_pct  = vf.get("text_density", 0.1) * 100
    load_high = cl == "High"
    load_low  = cl == "Low"
    attn_str  = attn >= 65
    attn_weak = attn < 45
    mem_str   = mem  >= 65
    mem_weak  = mem  < 45
    val_pos   = val  >  0.10
    val_neg   = val  < -0.05

    cc = "#22C55E" if cpci >= 70 else ("#F59E0B" if cpci >= 40 else "#EF4444")

    # ── Scale label ───────────────────────────────────────────────────────────
    if cpci >= 70:   scale_label = "Ready to scale"
    elif cpci >= 55: scale_label = "Optimise before scaling"
    elif cpci >= 40: scale_label = "Not ready for scale"
    else:            scale_label = "Do not deploy"

    # ── Diagnosis (2 lines max) ───────────────────────────────────────────────
    if cpci >= 70:
        if attn_str and mem_str:
            diag = [
                "This creative clears every cognitive gate — attention, memory, and emotion are all above threshold.",
                "It will work in feed environments against cold audiences and compound brand recall over time.",
            ]
        elif attn_str:
            diag = [
                "This creative has the attention signal to stop cold audiences, but memory encoding limits its long-term brand value.",
                "It will drive clicks and initial engagement but will not build durable recall without frequency.",
            ]
        else:
            diag = [
                "This creative builds strong brand memory and emotional affinity, but its attention signal is below the cold-scroll threshold.",
                "It will perform reliably with warm audiences and frequency-based brand campaigns.",
            ]
    elif cpci >= 55:
        if attn_weak:
            diag = [
                "This creative will not reliably interrupt cold-audience feeds — it lacks the attention signal to win the first cognitive gate.",
                "It can contribute to recall if repeatedly exposed to warm audiences, but acquisition spend is premature.",
            ]
        elif load_high:
            diag = [
                "This creative carries too much visual complexity for fast-scroll environments — the brain abandons processing before the message lands.",
                "Stripping cognitive load is the single highest-leverage fix before any spend increase.",
            ]
        else:
            diag = [
                "This creative has real potential but one signal is suppressing the composite score.",
                "A targeted fix — not a rebuild — is likely all that stands between this and scale-readiness.",
            ]
    elif cpci >= 40:
        if not face and attn_weak:
            diag = [
                "This creative will not perform in feed environments — it lacks an attention trigger and fails to produce an orienting response.",
                "It may work in retargeting environments where familiarity reduces the processing effort required.",
            ]
        elif load_high:
            diag = [
                "This creative asks too much of a scrolling audience — visual overload causes disengagement before the message registers.",
                "No amount of targeting precision will compensate for cognitive friction at this level.",
            ]
        elif val_neg:
            diag = [
                "This creative generates mildly negative affect on each impression — a hidden brand tax at scale.",
                "The emotional tone must be corrected before deployment; otherwise, spend compounds the damage.",
            ]
        else:
            diag = [
                "This creative has insufficient cognitive signal strength to justify cold-audience spend.",
                "Narrow the audience to warm segments while the creative is improved.",
            ]
    else:
        diag = [
            "This creative will not perform regardless of budget, targeting, or placement.",
            "The cognitive barriers are fundamental — a partial fix will not be sufficient.",
        ]

    # ── Primary issue ─────────────────────────────────────────────────────────
    attn_gap = max(0, 60 - attn)
    mem_gap  = max(0, 70 - mem)
    val_gap  = max(0, 0.1 - val) * 100
    load_gap = 25 if cl == "High" else (10 if cl == "Medium" else 0)
    worst    = max(attn_gap, mem_gap, val_gap, load_gap)

    if worst == 0:
        primary_issue = "No dominant weakness — all signals are above threshold."
    elif worst == load_gap and load_high:
        primary_issue = "Visual overload — too many elements competing for attention simultaneously."
    elif worst == attn_gap and not face:
        primary_issue = "No dominant focal point — nothing triggers an orienting response."
    elif worst == attn_gap and contrast < 45:
        primary_issue = "Insufficient contrast — creative is invisible in a competitive feed."
    elif worst == attn_gap:
        primary_issue = "Diffuse composition — attention fragments with no single dominant subject."
    elif worst == mem_gap and text_pct > 25:
        primary_issue = "Text overload — verbal channel is saturated, blocking dual-coding."
    elif worst == mem_gap and text_pct < 5:
        primary_issue = "No verbal anchor — visual alone will not survive working memory."
    elif worst == mem_gap:
        primary_issue = "Too many competing elements — no single memory trace will dominate."
    elif worst == val_gap and not face:
        primary_issue = "No human presence — palette alone cannot generate positive affect."
    else:
        primary_issue = "Negative emotional signal — colour palette is triggering subconscious avoidance."

    # ── Fix ───────────────────────────────────────────────────────────────────
    if not face and attn_gap > 15:
        fix = "Introduce a face or strong visual anchor — it is the highest-leverage single change available."
    elif contrast < 45 and attn_gap > 10:
        fix = "Increase foreground contrast significantly — pass the visual salience threshold first."
    elif objects > 7 and (worst == load_gap or worst == mem_gap):
        fix = "Commit to one hero element — remove everything that is not the primary message."
    elif text_pct > 28:
        fix = "Cut copy to a single line under six words — audiences read far less than advertisers write."
    elif text_pct < 4 and mem_gap > 15:
        fix = "Add a 4–6 word brand line — the verbal channel is unused at zero cost to attention."
    elif val_neg and not face:
        fix = "Add a person with a warm, natural expression — valence cannot be fixed with colour at this deficit."
    elif val_neg:
        fix = "Replace dominant cool or dark tones with warmer equivalents — the palette is the valence driver."
    elif attn_gap > 10:
        fix = "Scale up the primary subject and push contrast — give the eye an unmissable landing point."
    else:
        fix = "Simplify to one dominant image and one short message — single-exposure recall demands it."

    # ── Recommended use ───────────────────────────────────────────────────────
    recommended, avoid = [], []

    if cpci >= 70:
        recommended = ["Cold acquisition", "Full-funnel spend", "High-reach brand campaigns"]
        avoid       = []
    elif cpci >= 55:
        if attn_weak:
            recommended = ["Warm retargeting", "Lookalike audiences (L1–L3)", "Frequency-capped brand campaigns"]
            avoid       = ["Cold acquisition", "Prospecting campaigns"]
        else:
            recommended = ["Warm retargeting", "Mid-funnel conversion", "Frequency builds"]
            avoid       = ["Top-of-funnel cold spend at scale"]
    elif cpci >= 40:
        recommended = ["Retargeting (strict frequency cap: 2–3x)", "High-intent warm audiences only"]
        avoid       = ["Cold acquisition", "Reach campaigns", "OOH or display"]
    else:
        recommended = []
        avoid       = ["Any paid deployment", "Cold audiences", "Retargeting", "Brand campaigns"]

    # ── Build HTML ────────────────────────────────────────────────────────────
    def arrow_list(items, color):
        return "".join(
            f"<div style='display:flex;align-items:baseline;gap:8px;margin-bottom:4px;'>"
            f"<span style='color:{color};font-weight:700;flex-shrink:0;'>→</span>"
            f"<span style='font-size:13px;color:#CBD5E1;'>{item}</span>"
            f"</div>"
            for item in items
        )

    diag_html = "".join(
        f"<div style='font-size:13px;color:#CBD5E1;line-height:1.7;margin-bottom:4px;'>{d}</div>"
        for d in diag
    )

    rec_html  = arrow_list(recommended, "#22C55E") if recommended else (
        f"<div style='font-size:13px;color:#EF4444;'>No deployment recommended — rebuild first.</div>"
    )
    avoid_html = arrow_list(avoid, "#EF4444") if avoid else ""

    st.markdown(
        f"<div style='background:#141B24;border:1px solid #1F2937;border-radius:16px;"
        f"padding:32px 36px;margin:0 0 40px 0;'>"

        # Header row
        f"<div style='display:flex;align-items:center;justify-content:space-between;"
        f"margin-bottom:16px;padding-bottom:14px;border-bottom:1px solid #1F2937;'>"
        f"<div style='font-size:13px;font-weight:700;color:#94A3B8;"
        f"letter-spacing:2px;text-transform:uppercase;'>📄&nbsp; Creative Brief</div>"
        f"<div style='font-size:13px;font-weight:700;color:{cc};"
        f"letter-spacing:0.5px;'>CPCi {cpci} — {scale_label}</div>"
        f"</div>"

        # Diagnosis
        f"{diag_html}"

        # Issue + Fix row
        f"<div style='display:grid;grid-template-columns:1fr 1fr;gap:16px;margin:18px 0;'>"

        f"<div>"
        f"<div style='font-size:13px;font-weight:700;color:#F59E0B;"
        f"letter-spacing:1.5px;text-transform:uppercase;margin-bottom:6px;'>"
        f"Primary Issue</div>"
        f"<div style='font-size:13px;color:#FFFFFF;line-height:1.6;'>{primary_issue}</div>"
        f"</div>"

        f"<div>"
        f"<div style='font-size:13px;font-weight:700;color:#3B82F6;"
        f"letter-spacing:1.5px;text-transform:uppercase;margin-bottom:6px;'>"
        f"Fix</div>"
        f"<div style='font-size:13px;color:#FFFFFF;line-height:1.6;'>{fix}</div>"
        f"</div>"

        f"</div>"

        # Use / Don't use
        f"<div style='border-top:1px solid #1F2937;padding-top:16px;"
        f"display:grid;grid-template-columns:1fr 1fr;gap:16px;'>"

        f"<div>"
        f"<div style='font-size:13px;font-weight:700;color:#22C55E;"
        f"letter-spacing:1.5px;text-transform:uppercase;margin-bottom:8px;'>"
        f"Recommended Use</div>"
        f"{rec_html}"
        f"</div>"

        + (
            f"<div>"
            f"<div style='font-size:13px;font-weight:700;color:#EF4444;"
            f"letter-spacing:1.5px;text-transform:uppercase;margin-bottom:8px;'>"
            f"Do Not Use For</div>"
            f"{avoid_html}"
            f"</div>"
            if avoid_html else ""
        ) +

        f"</div>"
        f"</div>",
        unsafe_allow_html=True,
    )


# ── Creative Classification ───────────────────────────────────────────────────

# Four archetypes — each with a label, icon, colour, one-line reason, and
# a brief description of what this creative type is designed to do.
_CREATIVE_TYPES = {
    "Acquisition": {
        "icon":  "⚡",
        "color": "#3B82F6",
        "tag":   "Acquisition Creative",
        "role":  "Designed to interrupt cold audiences and drive immediate action.",
        "needs": "High attention · Low cognitive load · Neutral-to-positive valence",
    },
    "Recall / Branding": {
        "icon":  "🧠",
        "color": "#a78bfa",
        "tag":   "Recall / Branding Creative",
        "role":  "Designed to build long-term brand memory and emotional affinity.",
        "needs": "High memory encoding · Positive valence · Manageable load",
    },
    "Retargeting": {
        "icon":  "🔁",
        "color": "#F59E0B",
        "tag":   "Retargeting Creative",
        "role":  "Designed to close warm audiences who already have brand awareness.",
        "needs": "Moderate memory · Positive valence · Works with lower attention",
    },
    "Retail Conversion": {
        "icon":  "🛒",
        "color": "#22C55E",
        "tag":   "Retail Conversion Creative",
        "role":  "Designed to drive recognition and action at point of purchase.",
        "needs": "Strong memory encoding · Low cognitive load · Clear product focus",
    },
}


def _classify_creative(
    cpci: float,
    attn: int,
    mem:  int,
    val:  float,
    cl:   str,
    vf:   dict,
    use_case: str,
) -> tuple:
    """
    Returns (archetype_key, reason_string) based on signal balance.

    Decision logic:
      Acquisition     → attn ≥ 60 AND cl ≠ High AND val ≥ 0
                        (stops cold scroll; processable; not brand-negative)
      Retail Conversion → mem ≥ 55 AND cl = Low AND attn ≥ 40
                        (recognition at point of purchase; fast-process format)
      Recall / Branding → mem ≥ 60 AND val > 0.05
                        (memory + emotion = brand building over time)
      Retargeting      → default when cold-audience signals are insufficient
    """
    load_high = cl == "High"
    load_low  = cl == "Low"

    # ── Priority 1: Acquisition ────────────────────────────────────────────────
    if attn >= 60 and not load_high and val >= 0:
        if attn >= 70:
            reason = (
                f"Exceptional attention ({attn}/100) with manageable cognitive load — "
                "will interrupt cold-audience feeds and drive immediate engagement. "
                "The brain's orienting response is reliably triggered at this signal level."
            )
        else:
            reason = (
                f"Strong attention ({attn}/100) clears the cold-scroll threshold, "
                f"cognitive load is {cl.lower()}, and emotional tone is not brand-damaging. "
                "This creative is built to acquire — not to retain."
            )
        return "Acquisition", reason

    # ── Priority 2: Retail Conversion ─────────────────────────────────────────
    if mem >= 55 and load_low and attn >= 40:
        reason = (
            f"Memory encoding is strong ({mem}/100) and cognitive load is low — "
            "ideal for the point-of-purchase context where the viewer must recognise "
            "the brand and make a decision in under two seconds. "
            "Attention is sufficient for warm and in-aisle placements."
        )
        return "Retail Conversion", reason

    # ── Priority 3: Recall / Branding ─────────────────────────────────────────
    if mem >= 60 and val > 0.05:
        reason = (
            f"Memory encoding ({mem}/100) and emotional valence ({val:+.2f}) are both "
            "strong — the two signals required for long-term brand recall. "
            f"Attention is {attn}/100, which is {'sufficient' if attn >= 45 else 'below threshold'} "
            "for cold audiences, but this creative will build equity with frequency. "
            "Best deployed in brand awareness campaigns, not performance."
        )
        return "Recall / Branding", reason

    # ── Priority 4: Retargeting (default) ─────────────────────────────────────
    if attn < 55:
        attn_note = f"attention is below the cold-scroll threshold ({attn}/100)"
    elif load_high:
        attn_note = f"cognitive load is High — unsuitable for fast-scroll cold audiences"
    else:
        attn_note = f"signal profile is mixed across dimensions"

    reason = (
        f"The {attn_note}, which means this creative will not perform efficiently against "
        "cold audiences. However, with warm audiences where brand awareness is pre-established, "
        "the attention bar is lower — this creative can still contribute to conversion "
        "in a retargeting context. Do not scale to prospecting."
    )
    return "Retargeting", reason


def _render_classification(cpci, attn, mem, val, cl, vf, use_case) -> None:
    """Renders the creative classification strip between the signal bar and Quick Read."""
    archetype, reason = _classify_creative(cpci, attn, mem, val, cl, vf, use_case)
    meta = _CREATIVE_TYPES[archetype]
    c    = meta["color"]

    # Check if secondary archetype also applies (for multi-purpose creatives)
    secondary = None
    if archetype == "Acquisition" and mem >= 60 and val > 0.05:
        secondary = "Recall / Branding"
    elif archetype == "Retail Conversion" and attn >= 60:
        secondary = "Acquisition"
    elif archetype == "Recall / Branding" and attn >= 60:
        secondary = "Acquisition"

    sec_html = ""
    if secondary:
        sm      = _CREATIVE_TYPES[secondary]
        sm_c    = sm["color"]
        sm_icon = sm["icon"]
        sec_html = (
            f"<span style='display:inline-flex;align-items:center;gap:5px;"
            f"font-size:13px;color:#94A3B8;margin-left:10px;'>"
            f"also works as&nbsp;"
            f"<span style='color:{sm_c};font-weight:600;'>"
            f"{sm_icon} {secondary}</span></span>"
        )

    st.markdown(
        f"<div style='background:#141B24;border:1px solid #1F2937;"
        f"border-left:3px solid {c};border-radius:0 8px 8px 0;"
        f"padding:20px 24px;margin:0 0 40px 0;'>"

        f"<div style='font-size:13px;font-weight:700;color:#94A3B8;"
        f"letter-spacing:2px;text-transform:uppercase;margin-bottom:8px;'>"
        f"Best Suited For</div>"

        f"<div style='display:flex;align-items:center;gap:8px;margin-bottom:10px;flex-wrap:wrap;'>"
        f"<span style='font-size:18px;font-weight:800;color:{c};'>"
        f"{meta['icon']} {meta['tag']}</span>"
        f"{sec_html}"
        f"</div>"

        f"<div style='font-size:13px;color:#CBD5E1;line-height:1.7;margin-bottom:10px;'>"
        f"{reason}</div>"

        f"<div style='font-size:13px;color:#94A3B8;'>"
        f"<span style='font-weight:600;color:#94A3B8;'>Signal basis:</span>&nbsp;"
        f"{meta['needs']}</div>"

        f"</div>",
        unsafe_allow_html=True,
    )


# ── Media Implications ────────────────────────────────────────────────────────

def _media_implications(
    cpci: float,
    attn: int,
    mem:  int,
    val:  float,
    cl:   str,
    vf:   dict,
    use_case: str,
) -> None:
    """
    Translates cognitive signals into placement-level media planning decisions:
      - Where this creative will work
      - Where it will fail
      - Recommended media strategy
    """

    face      = vf.get("face_count", 0) > 0
    contrast  = vf.get("contrast_score", 50)
    objects   = vf.get("object_count", 4)
    text_pct  = vf.get("text_density", 0.1) * 100
    load_high = cl == "High"
    load_low  = cl == "Low"
    attn_str  = attn >= 65
    attn_weak = attn < 45
    mem_str   = mem  >= 65
    mem_weak  = mem  < 45
    val_pos   = val  >  0.10
    val_neg   = val  < -0.05

    # ── Placement definitions ──────────────────────────────────────────────────
    ALL_PLACEMENTS = {
        "Social Feed":    {"requires": "attn_str or contrast >= 60", "format": "fast-scroll"},
        "YouTube Pre-roll": {"requires": "attn_str and val_pos", "format": "interruptive"},
        "Display / Programmatic": {"requires": "not load_high", "format": "passive"},
        "Retail Media":   {"requires": "mem_str and not load_high", "format": "purchase-adjacent"},
        "OOH / DOOH":     {"requires": "load_low and not load_high and contrast >= 60", "format": "sub-2s"},
        "Retargeting":    {"requires": "not attn_str", "format": "warm-audience"},
        "CTV / Connected TV": {"requires": "val_pos and mem_str", "format": "lean-back"},
    }

    # Evaluate each placement
    works, fails = [], []

    # Social Feed
    if attn_str or contrast >= 60:
        works.append(("Social Feed", "Attention signal is strong enough to interrupt the scroll"))
    else:
        fails.append(("Social Feed", "Will not survive the scroll — attention signal too weak to compete in a competitive feed"))

    # YouTube Pre-roll
    if attn_str and val_pos:
        works.append(("YouTube Pre-roll", "Strong attention + positive valence — will hold viewers through the skip window"))
    elif attn_str and not val_pos:
        fails.append(("YouTube Pre-roll", "Will get noticed but emotional tone will not build the affinity needed to drive post-view action"))
    else:
        fails.append(("YouTube Pre-roll", "Insufficient attention to survive the skip — viewers will disengage before the message lands"))

    # Display / Programmatic
    if not load_high:
        works.append(("Display / Programmatic", "Low cognitive load means the message processes passively — effective for frequency builds"))
    else:
        fails.append(("Display / Programmatic", "High cognitive load in a passive format — the viewer will not invest the effort required to decode this"))

    # Retail Media
    if mem_str and not load_high:
        works.append(("Retail Media", "Strong memory encoding at point of purchase proximity — will drive recognition when intent is highest"))
    else:
        fails.append(("Retail Media", "Weak memory signal means the brand will not be recognised at the shelf or checkout — the moment the placement is designed for"))

    # OOH / DOOH
    if load_low and contrast >= 60:
        works.append(("OOH / DOOH", "Low load + high contrast — will process within the 1.5s average dwell time for outdoor formats"))
    else:
        fails.append(("OOH / DOOH", "Will not process in under 2 seconds — too complex or too low-contrast for outdoor dwell times"))

    # CTV
    if val_pos and mem_str:
        works.append(("CTV / Connected TV", "Positive valence + strong memory encoding — lean-back audiences will absorb and retain the brand message"))
    else:
        fails.append(("CTV / Connected TV", "Neutral or negative valence in a lean-back format — the emotional opportunity of CTV will be wasted"))

    # ── Strategy ───────────────────────────────────────────────────────────────
    if cpci >= 70:
        if attn_str and mem_str:
            strategy = "top_funnel"
            strat_label  = "Full-funnel — prioritise reach"
            strat_detail = (
                "Deploy at scale across cold audiences. This creative clears the cognitive bar "
                "for both acquisition and recall — a rare combination. Allocate the majority of "
                "budget to prospecting. Retargeting will compound the memory already built."
            )
        elif attn_str:
            strategy = "top_funnel"
            strat_label  = "Top-of-funnel acquisition"
            strat_detail = (
                "Strong in cold audiences — deploy for reach and awareness. "
                "Pair with a simpler retargeting creative to close the memory gap, "
                "or add a short text overlay to anchor the brand name."
            )
        else:
            strategy = "frequency"
            strat_label  = "Frequency play — build recall"
            strat_detail = (
                "CPCi is strong but memory needs frequency to compound. Cap at 4–5 "
                "impressions per user per week. Recall will build reliably — avoid "
                "over-exposing or the returns diminish."
            )
    elif cpci >= 55:
        strategy = "warm_audience"
        strat_label  = "Warm audiences — not cold"
        strat_detail = (
            "This creative is not ready for cold-audience acquisition — the cognitive "
            "signal is too mixed to convert efficiently at scale. Restrict spend to "
            "retargeting and lookalike audiences where intent is pre-established. "
            "Fix the weakest signal before opening to prospecting."
        )
    elif cpci >= 40:
        strategy = "retargeting"
        strat_label  = "Retargeting only — strict frequency cap"
        strat_detail = (
            "Limit to retargeting with a hard frequency cap of 2–3 exposures. "
            "The cognitive signal is insufficient for cold acquisition — spend here "
            "will generate impressions but not conversions. Treat as a stopgap "
            "while the creative is rebuilt."
        )
    else:
        strategy = "hold"
        strat_label  = "Hold spend — rebuild before deployment"
        strat_detail = (
            "Do not deploy at any scale. Budget spent on this creative will generate "
            "impressions at near-zero cognitive impact — or worse, accumulate negative "
            "brand associations. The creative requires a rebuild before media spend is justified."
        )

    strat_colors = {
        "top_funnel":  "#22C55E",
        "full_funnel": "#22C55E",
        "frequency":   "#3B82F6",
        "warm_audience": "#F59E0B",
        "retargeting": "#F59E0B",
        "hold":        "#EF4444",
    }
    sc = strat_colors.get(strategy, "#CBD5E1")

    # ── Render ─────────────────────────────────────────────────────────────────
    def _place_row(icon, name, reason, good: bool):
        dot   = "#22C55E" if good else "#EF4444"
        label = "Works" if good else "Fails"
        return (
            f"<div style='display:flex;align-items:flex-start;gap:10px;"
            f"padding:10px 0;border-bottom:1px solid #1F2937;'>"
            f"<div style='flex-shrink:0;margin-top:3px;width:8px;height:8px;"
            f"border-radius:50%;background:{dot};'></div>"
            f"<div style='flex:1;min-width:0;'>"
            f"<div style='display:flex;align-items:center;gap:8px;margin-bottom:2px;'>"
            f"<span style='font-size:13px;font-weight:600;color:#FFFFFF;'>{name}</span>"
            f"<span style='font-size:9px;font-weight:700;color:{dot};"
            f"letter-spacing:1px;text-transform:uppercase;'>{label}</span>"
            f"</div>"
            f"<div style='font-size:12px;color:#94A3B8;line-height:1.5;'>{reason}</div>"
            f"</div>"
            f"</div>"
        )

    works_html = "".join(_place_row("✓", n, r, True)  for n, r in works)
    fails_html = "".join(_place_row("✗", n, r, False) for n, r in fails)

    st.markdown(
        "<div style='margin-bottom:28px;'>"
        "<span style='font-size:13px;font-weight:600;color:#94A3B8;"
        "letter-spacing:1.5px;text-transform:uppercase;'>"
        "📡 Media Implications</span></div>",
        unsafe_allow_html=True,
    )

    pl_col, strat_col = st.columns([3, 2], gap="large")

    with pl_col:
        st.markdown(
            "<div style='font-size:13px;font-weight:700;color:#94A3B8;"
            "letter-spacing:2px;text-transform:uppercase;margin-bottom:4px;'>"
            "Placement Fit</div>",
            unsafe_allow_html=True,
        )
        st.markdown(works_html + fails_html, unsafe_allow_html=True)

    with strat_col:
        st.markdown(
            f"<div style='background:#141B24;border:1px solid #1F2937;"
            f"border-top:2px solid {sc};border-radius:8px;padding:20px 22px;height:100%;'>"
            f"<div style='font-size:13px;font-weight:700;color:#94A3B8;"
            f"letter-spacing:2px;text-transform:uppercase;margin-bottom:10px;'>"
            f"Recommended Strategy</div>"
            f"<div style='font-size:17px;font-weight:700;color:{sc};"
            f"margin-bottom:12px;line-height:1.3;'>{strat_label}</div>"
            f"<div style='font-size:13px;color:#CBD5E1;line-height:1.7;'>"
            f"{strat_detail}"
            f"</div>"
            f"</div>",
            unsafe_allow_html=True,
        )


# ── Export helpers ────────────────────────────────────────────────────────────

def _quick_read_data(cpci, attn, mem, val, cl, vf, use_case):
    """Return (prediction, issue, fix) as plain strings — no HTML."""
    ctx = {
        "FMCG Branding":         "brand awareness campaigns",
        "Performance Marketing": "paid social and search ads",
        "Retail Media":          "retail and in-store placements",
    }.get(use_case, "this campaign type")

    if cpci >= 70:
        prediction = f"This creative will earn its media spend in {ctx} — the cognitive gates are clear."
    elif cpci >= 55:
        prediction = f"This creative will generate impressions in {ctx} but will not convert efficiently — one weak signal is costing reach."
    elif cpci >= 40:
        prediction = f"This creative will struggle in cold audiences for {ctx} — it may survive retargeting but will not scale profitably."
    else:
        prediction = f"This creative will underperform regardless of budget — the cognitive barriers are too significant to overcome with spend."

    attn_gap = max(0, 60 - attn)
    mem_gap  = max(0, 70 - mem)
    val_gap  = max(0, 0.1 - val) * 100
    load_gap = 25 if cl == "High" else (10 if cl == "Medium" else 0)
    worst    = max(attn_gap, mem_gap, val_gap, load_gap)

    if worst == 0:
        issue = "All signals are above threshold — no single dimension is holding this back."
    elif worst == load_gap and cl == "High":
        issue = "Visual overload will cause viewers to disengage before the message registers — the brain cannot parse this quickly enough in a feed."
    elif worst == attn_gap:
        issue = ("This creative will fail to trigger an orienting response — without a face or salient focal point, the brain will not interrupt scrolling to process it."
                 if vf.get("face_count", 0) == 0 else
                 "This creative will disappear in a busy feed — insufficient contrast means it fails the first visual gate before content is even assessed."
                 if vf.get("contrast_score", 50) < 45 else
                 "This creative will be processed at low depth — no dominant focal point means the brain distributes attention thinly and nothing is prioritised.")
    elif worst == mem_gap:
        issue = ("This creative will be remembered only if repeatedly exposed — text overload prevents dual-coding, so neither the visual nor verbal message encodes cleanly."
                 if vf.get("text_density", 0) > 0.25 else
                 "This creative will be seen but not recalled — without a verbal anchor, the visual alone will not survive beyond a few seconds in working memory."
                 if vf.get("text_density", 0) < 0.05 else
                 "This creative will not build brand recall efficiently — too many competing elements mean nothing dominates the memory trace.")
    elif worst == val_gap:
        issue = ("This creative will generate neutral-to-negative affect on each exposure — without a human face, the palette alone cannot drive positive brand association."
                 if vf.get("face_count", 0) == 0 else
                 "This creative will subtly undermine brand affinity over time — the colour palette is triggering mild avoidance without the viewer being aware of it.")
    else:
        issue = "This creative demands more cognitive effort than a scrolling audience will commit — the message will not land in fast-feed placements."

    if vf.get("face_count", 0) == 0 and attn_gap > 15:
        fix = "Introduce a human face as the dominant visual element — it is the single fastest mechanism for triggering an orienting response in feed."
    elif vf.get("contrast_score", 50) < 45 and attn_gap > 10:
        fix = "Increase foreground-to-background contrast significantly — the creative needs to pass the visual salience test before any other signal matters."
    elif vf.get("object_count", 0) > 7 and (worst == load_gap or worst == mem_gap):
        fix = "Eliminate all secondary visual elements and commit to a single hero — cognitive load is suppressing every other signal."
    elif vf.get("text_density", 0) > 0.28:
        fix = "Reduce copy to a single declarative line under six words — audiences process far less text than advertisers write."
    elif vf.get("text_density", 0) < 0.04 and mem_gap > 15:
        fix = "Add a 4–6 word brand or message line — the verbal channel is completely unused, which is costing you recall without any benefit."
    elif val < -0.05 and vf.get("face_count", 0) == 0:
        fix = "Add a person with a natural, warm expression — emotional valence cannot be fixed with colour alone at this deficit."
    elif val < -0.05:
        fix = "Replace the dominant cool or dark tones with warmer equivalents — the palette is the primary driver of the negative valence reading."
    elif attn_gap > 10:
        fix = "Scale up the primary subject and increase its contrast against the background — give the eye an unmissable landing point."
    elif mem_gap > 15:
        fix = "Reduce the composition to one dominant image and one short message — simplicity is the only reliable route to single-exposure recall."
    else:
        fix = "Test a version with a human face replacing the current hero element — it will lift both attention and emotional valence simultaneously."

    return prediction, issue, fix


def _build_report_data(r: dict, use_case: str, client_name: str = "") -> dict:
    """Assemble all client-report content into a plain dict."""
    s       = r["signals"]
    vf      = r["visual_features"]
    narr    = r.get("narrative", {})
    cpci    = r["cpci"]
    attn    = s["attention_score"]
    mem     = s["memory_score"]
    val     = s["emotional_valence"]
    cl      = s["cognitive_load"]
    _rsn    = r.get("reasoning") or {}
    cl_score = (
        _rsn["load"]["composite"]
        if isinstance(_rsn, dict) and "load" in _rsn
        else s.get("cognitive_load_score", 50)
    )

    if cpci >= 70:   perf_label = "Strong Performer"
    elif cpci >= 40: perf_label = "Needs Optimisation"
    else:            perf_label = "Not Ready to Scale"

    if attn > 60:    a_label = "High Attention"
    elif attn >= 30: a_label = "Moderate"
    else:            a_label = "Scroll-Past Risk"

    if mem > 70:     m_label = "Strong Recall"
    elif mem >= 40:  m_label = "Moderate"
    else:            m_label = "Low Retention"

    if val > 0.1:    v_label = "Positive"
    elif val > -0.1: v_label = "Neutral"
    else:            v_label = "Negative"

    verdict        = _final_verdict_text(cpci, attn, mem, val, cl, use_case)
    prediction, issue, fix = _quick_read_data(cpci, attn, mem, val, cl, vf, use_case)
    recommendation = _client_recommendation(narr, cpci, attn, mem, val, cl, vf)

    # Creative Brief data
    if cpci >= 70:   scale_label = "Ready to scale"
    elif cpci >= 55: scale_label = "Optimise before scaling"
    elif cpci >= 40: scale_label = "Not ready for scale"
    else:            scale_label = "Do not deploy"

    uc = USE_CASES.get(use_case, {})
    w  = uc.get("weights", {"attention": 0.4, "memory": 0.3, "emotion": 0.3})

    # Compute top-3 optimization scenarios for PDF recommendations
    scenarios = _compute_scenarios(cpci, attn, mem, val, cl, vf, use_case)

    return {
        "name":           r["name"],
        "use_case":       use_case,
        "client_name":    client_name,
        "cpci":           cpci,
        "perf_label":     perf_label,
        "scale_label":    scale_label,
        "verdict":        verdict,
        "performance":    prediction,
        "core_issue":     issue,
        "fix_first":      fix,
        "recommendation": recommendation,
        # signals
        "attn":       attn,   "attn_label":  a_label,
        "mem":        mem,    "mem_label":   m_label,
        "val":        val,    "val_label":   v_label,
        "cl":         cl,     "cl_score":    cl_score,
        # visual features
        "face_count":     vf.get("face_count", 0),
        "contrast_score": vf.get("contrast_score", 0),
        "object_count":   vf.get("object_count", 0),
        "text_density":   vf.get("text_density", 0),
        "dominant_colors": vf.get("dominant_colors", []),
        # narrative
        "strategic_implication": narr.get("strategic_implication", ""),
        # weights
        "w_attn":  int(w.get("attention", 0.4) * 100),
        "w_mem":   int(w.get("memory",    0.3) * 100),
        "w_emo":   int(w.get("emotion",   0.3) * 100),
        # optimization scenarios (top 3, sorted by lift desc)
        "scenarios":  scenarios[:3],
    }


def _generate_pdf_bytes(data: dict) -> bytes:
    """Build a full multi-section client-ready PDF. Returns raw bytes."""
    import io
    from reportlab.lib.pagesizes import A4
    from reportlab.lib import colors
    from reportlab.lib.units import mm
    from reportlab.platypus import (
        SimpleDocTemplate, Paragraph, Spacer, HRFlowable, Table, TableStyle, PageBreak,
        KeepTogether,
    )
    from reportlab.lib.styles import ParagraphStyle
    from reportlab.lib.enums import TA_LEFT, TA_CENTER, TA_RIGHT
    from datetime import date

    buf = io.BytesIO()
    W, H = A4
    margin = 22 * mm
    inner  = W - 2 * margin

    doc = SimpleDocTemplate(
        buf,
        pagesize=A4,
        leftMargin=margin, rightMargin=margin,
        topMargin=20 * mm, bottomMargin=20 * mm,
    )

    # ── Color palette ─────────────────────────────────────────────────────────
    BLUE   = colors.HexColor("#3B82F6")
    WHITE  = colors.HexColor("#FFFFFF")
    MUTED  = colors.HexColor("#CBD5E1")
    LABEL  = colors.HexColor("#94A3B8")
    BORDER = colors.HexColor("#1F2937")
    CARD   = colors.HexColor("#141B24")
    GREEN  = colors.HexColor("#22C55E")
    AMBER  = colors.HexColor("#F59E0B")
    RED    = colors.HexColor("#EF4444")
    BG     = colors.HexColor("#0B0F14")

    cpci = data["cpci"]
    sc   = GREEN if cpci >= 70 else (AMBER if cpci >= 40 else RED)
    # Plain hex strings for use inside <font color='...'> tags in Paragraph markup.
    # HexColor.hexval() returns "0x..." — the leading "0" makes the slice wrong,
    # so we define the strings directly from the thresholds instead.
    sc_hex  = "#22C55E" if cpci >= 70 else ("#F59E0B" if cpci >= 40 else "#EF4444")

    # signal colors
    def sig_c(v, hi=60, lo=30):
        return GREEN if v >= hi else (AMBER if v >= lo else RED)
    def sig_hex(v, hi=60, lo=30):
        return "#22C55E" if v >= hi else ("#F59E0B" if v >= lo else "#EF4444")
    a_c   = sig_c(data["attn"])
    m_c   = sig_c(data["mem"], 70, 40)
    v_c   = GREEN if data["val"] > 0.1 else (AMBER if data["val"] > -0.1 else RED)
    cl_c  = GREEN if data["cl"] == "Low" else (AMBER if data["cl"] == "Medium" else RED)
    a_hex = sig_hex(data["attn"])
    m_hex = sig_hex(data["mem"], 70, 40)
    v_hex = "#22C55E" if data["val"] > 0.1 else ("#F59E0B" if data["val"] > -0.1 else "#EF4444")
    cl_hex = "#22C55E" if data["cl"] == "Low" else ("#F59E0B" if data["cl"] == "Medium" else "#EF4444")

    # ── Style factory ─────────────────────────────────────────────────────────
    def S(name, **kw):
        return ParagraphStyle(name, **kw)

    sEye   = S("eye",    fontSize=7,  leading=10, textColor=LABEL,
                fontName="Helvetica",  spaceAfter=1, letterSpacing=0.8)
    sTitle = S("title",  fontSize=24, leading=30, textColor=WHITE,
                fontName="Helvetica-Bold", spaceAfter=4)
    sSub   = S("sub",    fontSize=10, leading=14, textColor=MUTED,
                fontName="Helvetica", spaceAfter=0)
    sLbl   = S("lbl",    fontSize=7,  leading=10, textColor=LABEL,
                fontName="Helvetica-Bold", spaceBefore=14, spaceAfter=5,
                letterSpacing=1.2)
    sScore = S("score",  fontSize=72, leading=76, textColor=sc,
                fontName="Helvetica-Bold", spaceAfter=0)
    sPerf  = S("perf",   fontSize=12, leading=16, textColor=sc,
                fontName="Helvetica-Bold")
    sVerd  = S("verd",   fontSize=17, leading=24, textColor=sc,
                fontName="Helvetica-Bold", spaceAfter=0)
    sH2    = S("h2",     fontSize=11, leading=15, textColor=WHITE,
                fontName="Helvetica-Bold", spaceBefore=12, spaceAfter=4)
    sBody  = S("body",   fontSize=11, leading=17, textColor=WHITE,
                fontName="Helvetica", spaceAfter=0)
    sMuted = S("muted",  fontSize=10, leading=15, textColor=MUTED,
                fontName="Helvetica", spaceAfter=0)
    sTag   = S("tag",    fontSize=8,  leading=12, textColor=LABEL,
                fontName="Helvetica-Bold", letterSpacing=0.5)
    sFoot  = S("foot",   fontSize=7,  leading=10, textColor=LABEL,
                fontName="Helvetica", alignment=TA_CENTER)
    sNote  = S("note",   fontSize=9,  leading=13, textColor=LABEL,
                fontName="Helvetica-Oblique", spaceAfter=0)
    sSigV  = S("sigv",   fontSize=22, leading=26, textColor=WHITE,
                fontName="Helvetica-Bold", spaceAfter=0)
    sSigL  = S("sigl",   fontSize=8,  leading=11, textColor=LABEL,
                fontName="Helvetica", spaceAfter=0)
    sSigN  = S("sign",   fontSize=8,  leading=11, textColor=LABEL,
                fontName="Helvetica-Bold", letterSpacing=0.8, spaceAfter=0)

    def rule(c=BORDER, t=0.4):
        return HRFlowable(width="100%", thickness=t, color=c, spaceAfter=6, spaceBefore=6)

    def thick_rule(c=BLUE):
        return HRFlowable(width="100%", thickness=1.5, color=c, spaceAfter=6, spaceBefore=0)

    def section_label(text):
        return Paragraph(text.upper(), sLbl)

    def card_table(rows_data, col_widths, style_extras=None):
        t = Table(rows_data, colWidths=col_widths)
        base = [
            ("BACKGROUND",   (0, 0), (-1, -1), CARD),
            ("BOX",          (0, 0), (-1, -1), 0.5, BORDER),
            ("ROUNDEDCORNERS", [4]),
            ("LEFTPADDING",  (0, 0), (-1, -1), 10),
            ("RIGHTPADDING", (0, 0), (-1, -1), 10),
            ("TOPPADDING",   (0, 0), (-1, -1), 10),
            ("BOTTOMPADDING",(0, 0), (-1, -1), 10),
            ("VALIGN",       (0, 0), (-1, -1), "TOP"),
        ]
        if style_extras:
            base += style_extras
        t.setStyle(TableStyle(base))
        return t

    today = date.today().strftime("%B %d, %Y")
    story = []

    # Shared footer text
    FOOTER_TEXT = "ADVantage Insights  ·  Cognitive Signal Engine™  ·  © Anil Pandit  ·  Confidential"

    # ══════════════════════════════════════════════════════════════════════════
    # COVER PAGE
    # ══════════════════════════════════════════════════════════════════════════

    # Large product name
    sCoverProduct = S("coverprod", fontSize=11, leading=15, textColor=BLUE,
                      fontName="Helvetica-Bold", letterSpacing=2.0, spaceAfter=4)
    sCoverTagline = S("covertagline", fontSize=13, leading=18, textColor=MUTED,
                      fontName="Helvetica", spaceAfter=0)
    sCoverHero    = S("coverhero",   fontSize=36, leading=44, textColor=WHITE,
                      fontName="Helvetica-Bold", spaceAfter=6)
    sCoverMeta    = S("covermeta",   fontSize=10, leading=15, textColor=LABEL,
                      fontName="Helvetica", spaceAfter=0)
    sCoverClient  = S("coverclient", fontSize=22, leading=28, textColor=WHITE,
                      fontName="Helvetica-Bold", spaceAfter=0)
    sCoverSub     = S("coversub",    fontSize=10, leading=15, textColor=MUTED,
                      fontName="Helvetica", spaceAfter=0)

    # ── Top brand strip ───────────────────────────────────────────────────────
    story.append(Spacer(1, 28))
    story.append(Paragraph("COGNITIVE SIGNAL ENGINE™", sCoverProduct))
    story.append(Paragraph("Creative Intelligence Analyzer", sCoverSub))
    story.append(Spacer(1, 32))
    story.append(thick_rule(BLUE))
    story.append(Spacer(1, 40))

    # ── Hero score & product name ─────────────────────────────────────────────
    story.append(Paragraph(
        f"<font color='{sc_hex}'>{cpci}</font>",
        S("covercpci", fontSize=96, leading=100, textColor=sc,
          fontName="Helvetica-Bold", spaceAfter=4),
    ))
    story.append(Paragraph("CPCi — Cost Per Cognitive Impression", sCoverSub))

    # ── 1-line CPCi interpretation ────────────────────────────────────────────
    if cpci >= 70:
        _interp_hex  = "#22C55E"
        _interp_text = "High efficiency — ready for scale"
    elif cpci >= 40:
        _interp_hex  = "#F59E0B"
        _interp_text = "Moderate efficiency — optimise before scaling"
    else:
        _interp_hex  = "#EF4444"
        _interp_text = "High inefficiency — not ready for deployment"

    story.append(Spacer(1, 8))
    story.append(Paragraph(
        f"<font color='{_interp_hex}'>{_interp_text}</font>",
        S("coverinterp", fontSize=11, leading=15, fontName="Helvetica",
          textColor=colors.HexColor(_interp_hex), alignment=TA_CENTER, spaceAfter=0),
    ))
    story.append(Spacer(1, 24))

    # ── Creative title ────────────────────────────────────────────────────────
    story.append(Paragraph(data["name"], sCoverHero))
    story.append(Spacer(1, 12))
    story.append(Paragraph(f"Use Case: {data['use_case']}", sCoverMeta))
    story.append(Spacer(1, 4))

    if data.get("client_name"):
        story.append(Spacer(1, 14))
        story.append(Paragraph(
            "CLIENT",
            S("coverclientlabel", fontSize=9, leading=12, textColor=MUTED,
              fontName="Helvetica-Bold", letterSpacing=1.8, spaceAfter=3),
        ))
        story.append(Paragraph(data["client_name"], sCoverClient))
        story.append(Spacer(1, 14))

    story.append(Paragraph(f"Date: {today}", sCoverMeta))
    story.append(Spacer(1, 40))
    story.append(thick_rule(BLUE))
    story.append(Spacer(1, 16))

    # ── Prepared by line ──────────────────────────────────────────────────────
    story.append(Paragraph(
        "Prepared by  Cognitive Signal Engine™  ·  ADVantage Insights  ·  Anil Pandit",
        S("coverprepby", fontSize=9, leading=13, textColor=LABEL,
          fontName="Helvetica", alignment=TA_CENTER),
    ))

    story.append(Spacer(1, 8))
    story.append(rule())
    story.append(Paragraph(FOOTER_TEXT, sFoot))

    # ══════════════════════════════════════════════════════════════════════════
    # EXECUTIVE SUMMARY PAGE
    # ══════════════════════════════════════════════════════════════════════════
    story.append(PageBreak())

    story.append(Paragraph("COGNITIVE SIGNAL ANALYSIS REPORT", sEye))
    story.append(Paragraph("EXECUTIVE SUMMARY", S("execeyebrow", fontSize=8, leading=11,
                 textColor=BLUE, fontName="Helvetica-Bold", spaceAfter=2, letterSpacing=1.2)))
    story.append(thick_rule(BLUE))
    story.append(Spacer(1, 10))

    # ── CPCi Score callout ────────────────────────────────────────────────────
    story.append(section_label("CPCi — Cost Per Cognitive Impression"))
    exec_score_row = Table(
        [[Paragraph(str(cpci), sScore),
          Spacer(6, 1),
          Paragraph(
              f"<b>{data['perf_label']}</b><br/><br/>"
              f"Scale decision: <b>{data['scale_label']}</b><br/><br/>"
              f"Use-case weights:<br/>"
              f"Attention {data['w_attn']}%  ·  Memory {data['w_mem']}%  ·  Emotion {data['w_emo']}%",
              sMuted,
          )]],
        colWidths=[55*mm, 6*mm, None],
    )
    exec_score_row.setStyle(TableStyle([
        ("VALIGN",       (0,0),(-1,-1),"MIDDLE"),
        ("LEFTPADDING",  (0,0),(-1,-1),0),
        ("RIGHTPADDING", (0,0),(-1,-1),0),
        ("TOPPADDING",   (0,0),(-1,-1),0),
        ("BOTTOMPADDING",(0,0),(-1,-1),0),
    ]))
    story.append(exec_score_row)
    story.append(Spacer(1, 12))
    story.append(rule())

    # ── Verdict ───────────────────────────────────────────────────────────────
    story.append(section_label("Verdict"))
    story.append(Paragraph(data["verdict"], sVerd))
    story.append(Spacer(1, 12))
    story.append(rule())

    # ══════════════════════════════════════════════════════════════════════════
    # EXECUTIVE DECISION BLOCK
    # ══════════════════════════════════════════════════════════════════════════
    _ed_mem        = data["mem"]
    _ed_eff_mem    = max(_ed_mem, 10)
    _ed_multiplier = round(70 / _ed_eff_mem, 1)

    if _ed_mem < 55:
        _ed_risk = "HIGH"
    elif _ed_mem < 70:
        _ed_risk = "MODERATE"
    else:
        _ed_risk = "LOW"

    if _ed_multiplier > 1.8 or _ed_risk == "HIGH":
        _ed_decision   = "DO NOT SCALE IN COLD MEDIA"
        _ed_tone_hex   = "#EF4444"
        _ed_bg_hex     = "#1F0A0A"
        _ed_border_hex = "#7F1D1D"
        _ed_rec        = "Deploy in retargeting / warm audiences only"
    elif _ed_multiplier > 1.3 or _ed_risk == "MODERATE":
        _ed_decision   = "OPTIMISE BEFORE SCALING"
        _ed_tone_hex   = "#F59E0B"
        _ed_bg_hex     = "#1A1200"
        _ed_border_hex = "#78350F"
        _ed_rec        = "Test in controlled budget before scaling"
    else:
        _ed_decision   = "READY TO SCALE"
        _ed_tone_hex   = "#22C55E"
        _ed_bg_hex     = "#071A0F"
        _ed_border_hex = "#14532D"
        _ed_rec        = "Safe for broader deployment"

    _ed_tone_c   = colors.HexColor(_ed_tone_hex)
    _ed_bg_c     = colors.HexColor(_ed_bg_hex)
    _ed_border_c = colors.HexColor(_ed_border_hex)

    # Outer box table (1×1 with background + border)
    _ed_content = [
        # TITLE row
        Paragraph(
            "EXECUTIVE DECISION",
            S("edtitle", fontSize=8, fontName="Helvetica-Bold", leading=11,
              textColor=colors.HexColor(_ed_tone_hex), letterSpacing=2.0,
              spaceAfter=6),
        ),
        # DECISION row — large, bold, colour-coded
        Paragraph(
            f"<font color='{_ed_tone_hex}'><b>{_ed_decision}</b></font>",
            S("eddecision", fontSize=22, fontName="Helvetica-Bold", leading=28,
              textColor=_ed_tone_c, spaceAfter=8),
        ),
        # Supporting line
        Paragraph(
            f"This creative will cost approximately <b>{_ed_multiplier}×</b> more media "
            f"and has <b>{_ed_risk}</b> cold audience deployment risk.",
            S("edsupport", fontSize=10, fontName="Helvetica", leading=15,
              textColor=colors.HexColor("#CBD5E1"), spaceAfter=6),
        ),
        # Recommendation line
        Paragraph(
            f"<font color='{_ed_tone_hex}'>▶  {_ed_rec}</font>",
            S("edrec", fontSize=10, fontName="Helvetica-Bold", leading=14,
              textColor=_ed_tone_c, spaceAfter=0),
        ),
    ]

    _ed_inner = Table(
        [[_ed_content[0]], [_ed_content[1]], [_ed_content[2]], [_ed_content[3]]],
        colWidths=[None],
    )
    _ed_inner.setStyle(TableStyle([
        ("BACKGROUND",   (0,0),(-1,-1), _ed_bg_c),
        ("BOX",          (0,0),(-1,-1), 1.5, _ed_border_c),
        ("LEFTPADDING",  (0,0),(-1,-1), 16),
        ("RIGHTPADDING", (0,0),(-1,-1), 16),
        ("TOPPADDING",   (0,0),(0,0),   14),   # more top padding on title row
        ("TOPPADDING",   (1,0),(-1,-1), 2),
        ("BOTTOMPADDING",(0,0),(-2,-1), 4),
        ("BOTTOMPADDING",(-1,0),(-1,-1), 14),  # more bottom padding on last row
        ("VALIGN",       (0,0),(-1,-1), "TOP"),
        ("LINEBELOW",    (0,0),(0,0),   0.5, colors.HexColor(_ed_border_hex)),  # line under title
    ]))

    story.append(Spacer(1, 4))
    story.append(_ed_inner)
    story.append(Spacer(1, 14))
    story.append(rule())

    # ── Business Impact summary ───────────────────────────────────────────────
    story.append(section_label("Business Impact"))

    if cpci >= 70:
        risk_label = "High Efficiency"
        risk_hex   = "#22C55E"
        impact_bullets = [
            "Impression quality is strong — CPCi above 70 indicates a top-quartile creative.",
            "Conversion signal is optimised — high attention and memory scores drive click-through intent.",
            "Brand equity: positive valence builds cumulative warmth across exposures.",
            "Deployment decision: deploy at full planned budget without optimisation.",
        ]
    elif cpci >= 40:
        risk_label = "Moderate Performance"
        risk_hex   = "#F59E0B"
        impact_bullets = [
            "Impression quality is moderate — some cognitive signal gap remains before peak efficiency.",
            "Conversion signal is partial — memory or attention weakness will require higher frequency.",
            "Brand equity: neutral-to-positive valence; brand sentiment stable but not building.",
            "Deployment decision: optimise the primary signal weakness before full budget deployment.",
        ]
    else:
        risk_label = "High Waste Risk"
        risk_hex   = "#EF4444"
        impact_bullets = [
            "Impression quality is poor — CPCi below 40 means most impressions will not cognitively register.",
            "Conversion signal is absent — media spend is unlikely to generate measurable purchase intent.",
            "Brand equity: low or negative valence risks eroding brand affinity over repeated exposure.",
            "Deployment decision: do not deploy. Revise the creative before any media investment.",
        ]

    risk_c = colors.HexColor(risk_hex)
    story.append(Paragraph(
        f"<font color='{risk_hex}'><b>{risk_label}</b></font>",
        S("risklabel", fontSize=14, leading=18, textColor=risk_c,
          fontName="Helvetica-Bold", spaceAfter=8),
    ))
    story.append(Spacer(1, 4))

    bi_rows = []
    for i, bullet in enumerate(impact_bullets, 1):
        bi_rows.append([
            Paragraph(f"0{i}", S("binum", fontSize=9, fontName="Helvetica-Bold",
                      leading=13, textColor=LABEL)),
            Paragraph(bullet, S("bibody", fontSize=10, fontName="Helvetica",
                      leading=15, textColor=WHITE)),
        ])

    bi_t = Table(bi_rows, colWidths=[14*mm, None])
    bi_t.setStyle(TableStyle([
        ("BACKGROUND",   (0,0),(-1,-1), CARD),
        ("BOX",          (0,0),(-1,-1), 0.5, BORDER),
        ("INNERGRID",    (0,0),(-1,-1), 0.3, BORDER),
        ("LEFTPADDING",  (0,0),(-1,-1), 10),
        ("RIGHTPADDING", (0,0),(-1,-1), 10),
        ("TOPPADDING",   (0,0),(-1,-1), 9),
        ("BOTTOMPADDING",(0,0),(-1,-1), 9),
        ("VALIGN",       (0,0),(-1,-1), "TOP"),
        ("LINEAFTER",    (0,0),(0,-1),  0.5, BORDER),
    ]))
    story.append(bi_t)
    story.append(Spacer(1, 12))
    story.append(rule())

    # ── Media Efficiency Impact ───────────────────────────────────────────────
    story.append(section_label("Media Efficiency Impact"))

    _pdf_mem            = data["mem"]
    _pdf_eff_mem        = max(_pdf_mem, 10)
    _pdf_multiplier     = round(70 / _pdf_eff_mem, 1)
    _pdf_mult_hex       = (
        "#EF4444" if _pdf_multiplier > 1.8 else
        "#F59E0B" if _pdf_multiplier >= 1.2 else
        "#22C55E"
    )
    _pdf_mult_color     = colors.HexColor(_pdf_mult_hex)

    sMedEff = S("medeff", fontSize=13, leading=18, fontName="Helvetica-Bold",
                textColor=WHITE, spaceAfter=6)
    sMedSub = S("medsub", fontSize=11, leading=16, fontName="Helvetica",
                textColor=colors.HexColor("#F87171"), spaceAfter=0)

    if _pdf_mem < 70:
        story.append(Paragraph(
            f"<font color='{_pdf_mult_hex}'>"
            f"This creative will cost approximately <b>{_pdf_multiplier}×</b> more media "
            f"to achieve the same recall as a top-quartile creative."
            f"</font>",
            sMedEff,
        ))
        if _pdf_multiplier > 1.5:
            story.append(Spacer(1, 4))
            story.append(Paragraph(
                "Equivalent to wasting ~35–50% of your media budget on ineffective impressions.",
                sMedSub,
            ))
    else:
        story.append(Paragraph(
            "<font color='#22C55E'>"
            "This creative is operating at efficient memory levels — no excess media cost."
            "</font>",
            sMedEff,
        ))

    story.append(Spacer(1, 12))
    story.append(rule())

    # ── What This Means (executive summary version) ───────────────────────────
    story.append(section_label("Quick Read"))
    wtm_rows_exec = [
        ("PERFORMANCE",  data["performance"]),
        ("CORE ISSUE",   data["core_issue"]),
        ("FIX FIRST",    data["fix_first"]),
    ]
    for tag, body in wtm_rows_exec:
        row = Table(
            [[Paragraph(tag, sTag), Paragraph(body, sBody)]],
            colWidths=[36*mm, None],
        )
        row.setStyle(TableStyle([
            ("VALIGN",       (0,0),(-1,-1),"TOP"),
            ("LEFTPADDING",  (0,0),(-1,-1),0),
            ("RIGHTPADDING", (0,0),(-1,-1),0),
            ("TOPPADDING",   (0,0),(-1,-1),7),
            ("BOTTOMPADDING",(0,0),(-1,-1),7),
            ("LINEBELOW",    (0,0),(-1,-1),0.3,BORDER),
        ]))
        story.append(row)

    story.append(Spacer(1, 8))
    story.append(rule())
    story.append(Paragraph(FOOTER_TEXT, sFoot))

    # ══════════════════════════════════════════════════════════════════════════
    # PAGE — CPCi · Verdict · What This Means (detailed)
    # ══════════════════════════════════════════════════════════════════════════
    story.append(PageBreak())

    # Header
    story.append(Paragraph("COGNITIVE SIGNAL ANALYSIS REPORT", sEye))
    story.append(Paragraph("Cognitive Signal Engine™  ·  Creative Intelligence Analyzer  ·  CPCi", sEye))
    story.append(Spacer(1, 6))
    story.append(thick_rule(BLUE))
    story.append(Spacer(1, 6))

    story.append(Paragraph(data["name"], sTitle))
    story.append(Paragraph(f"{data['use_case']}  ·  {today}", sSub))
    story.append(Spacer(1, 16))
    story.append(rule())

    # ── CPCi Score — technical breakdown ─────────────────────────────────────
    # (business summary already on Page 2; this page shows the formula detail)
    story.append(section_label("CPCi — Score & Formula Detail"))

    score_row = Table(
        [[Paragraph(str(cpci), sScore), Spacer(6, 1), Paragraph(
            f"<b>{data['perf_label']}</b><br/><br/>"
            f"Scale decision: <b>{data['scale_label']}</b><br/><br/>"
            f"Formula weights for <i>{data['use_case']}</i>:<br/>"
            f"Attention {data['w_attn']}%  ·  Memory {data['w_mem']}%  "
            f"·  Emotion {data['w_emo']}%<br/><br/>"
            f"CPCi = (Attention × {data['w_attn']}%) + (Memory × {data['w_mem']}%) "
            f"+ (Emotion × {data['w_emo']}%) − Load Penalty",
            sMuted)]],
        colWidths=[55*mm, 6*mm, None],
    )
    score_row.setStyle(TableStyle([
        ("VALIGN",       (0,0),(-1,-1),"MIDDLE"),
        ("LEFTPADDING",  (0,0),(-1,-1),0),
        ("RIGHTPADDING", (0,0),(-1,-1),0),
        ("TOPPADDING",   (0,0),(-1,-1),0),
        ("BOTTOMPADDING",(0,0),(-1,-1),0),
    ]))
    story.append(score_row)
    story.append(Spacer(1, 10))
    story.append(rule())

    # ── Verdict ───────────────────────────────────────────────────────────────
    story.append(section_label("Deployment Verdict"))
    story.append(Paragraph(data["verdict"], sVerd))
    story.append(Spacer(1, 14))
    story.append(rule())

    story.append(Spacer(1, 6))

    # Footer page
    story.append(rule())
    story.append(Paragraph(FOOTER_TEXT, sFoot))

    # ══════════════════════════════════════════════════════════════════════════
    # PAGE 2 — Signal Breakdown
    # ══════════════════════════════════════════════════════════════════════════
    story.append(PageBreak())

    story.append(Paragraph("COGNITIVE SIGNAL ANALYSIS REPORT", sEye))
    story.append(thick_rule(BLUE))
    story.append(Spacer(1, 6))

    story.append(section_label("Cognitive Signal Breakdown"))
    story.append(Paragraph(
        "The Cognitive Signal Engine measures four independent brain-level signals. "
        "Each is scored separately, then weighted by use-case to produce the CPCi. "
        "The table below shows the detected value, threshold, and what it means for media performance.",
        sMuted,
    ))
    story.append(Spacer(1, 10))

    # ── Visual signal bars ────────────────────────────────────────────────────
    def _make_bar(score_pct: float, hex_color: str, total_mm: float = 95) -> Table:
        """Horizontal progress bar as a two-cell nested Table."""
        filled_w = max(score_pct / 100, 0.02) * total_mm * mm
        empty_w  = max(total_mm * mm - filled_w, 0)
        bar = Table([[Spacer(1, 1), Spacer(1, 1)]],
                    colWidths=[filled_w, empty_w], rowHeights=[7])
        bar.setStyle(TableStyle([
            ("BACKGROUND",   (0,0),(0,0), colors.HexColor(hex_color)),
            ("BACKGROUND",   (1,0),(1,0), colors.HexColor("#1F2937")),
            ("LEFTPADDING",  (0,0),(-1,-1), 0),
            ("RIGHTPADDING", (0,0),(-1,-1), 0),
            ("TOPPADDING",   (0,0),(-1,-1), 0),
            ("BOTTOMPADDING",(0,0),(-1,-1), 0),
        ]))
        return bar

    _val_pct = round((data["val"] + 1) / 2 * 100)  # -1..+1 → 0..100

    _bar_rows = [
        ("Attention",         data["attn"],   a_hex,  f"{data['attn']}/100"),
        ("Memory",            data["mem"],    m_hex,  f"{data['mem']}/100"),
        ("Emotional Valence", _val_pct,       v_hex,  f"{data['val']:+.2f}"),
        ("Cognitive Load",    data["cl_score"], cl_hex,
         f"{data['cl']}  ({data['cl_score']:.0f}/100)"),
    ]

    _sBarLabel = S("blabel", fontSize=9,  fontName="Helvetica-Bold",
                   leading=13, textColor=WHITE)
    _sBarScore = S("bscore", fontSize=10, fontName="Helvetica-Bold",
                   leading=14, textColor=WHITE)

    for _bl, _bs, _bh, _bd in _bar_rows:
        _br = Table(
            [[Paragraph(f"<b>{_bl}</b>", _sBarLabel),
              _make_bar(_bs, _bh),
              Paragraph(f"<font color='{_bh}'><b>{_bd}</b></font>", _sBarScore)]],
            colWidths=[38*mm, 95*mm, None],
        )
        _br.setStyle(TableStyle([
            ("VALIGN",       (0,0),(-1,-1), "MIDDLE"),
            ("LEFTPADDING",  (0,0),(-1,-1), 0),
            ("RIGHTPADDING", (0,0),(-1,-1), 0),
            ("TOPPADDING",   (0,0),(-1,-1), 6),
            ("BOTTOMPADDING",(0,0),(-1,-1), 6),
            ("LINEBELOW",    (0,0),(-1,-1), 0.3, BORDER),
        ]))
        story.append(_br)

    story.append(Spacer(1, 14))
    story.append(rule())
    story.append(section_label("Signal Detail"))
    story.append(Spacer(1, 6))

    # Signal explanation rows
    _RL_COLOR_MAP = {GREEN: "#22C55E", AMBER: "#F59E0B", RED: "#EF4444",
                     WHITE: "#FFFFFF", MUTED: "#CBD5E1", LABEL: "#94A3B8"}

    def sig_row(label, score_str, color, threshold_str, what_it_measures, implication):
        chex = _RL_COLOR_MAP.get(color, "#FFFFFF")
        return [
            Paragraph(f"<b>{label}</b>", S("sl", fontSize=9, fontName="Helvetica-Bold",
                       leading=13, textColor=WHITE)),
            Paragraph(f"<font color='{chex}'><b>{score_str}</b></font>",
                      S("sv", fontSize=14, fontName="Helvetica-Bold",
                        leading=18, textColor=color)),
            Paragraph(threshold_str, S("st", fontSize=8, fontName="Helvetica",
                       leading=12, textColor=MUTED)),
            Paragraph(what_it_measures, S("sm", fontSize=9, fontName="Helvetica",
                       leading=13, textColor=MUTED)),
            Paragraph(implication, S("si", fontSize=9, fontName="Helvetica",
                       leading=13, textColor=WHITE)),
        ]

    # Header row
    hdr_style = S("hdr", fontSize=7, fontName="Helvetica-Bold", leading=10,
                  textColor=LABEL, letterSpacing=0.8)
    sig_table_data = [
        [Paragraph("SIGNAL", hdr_style), Paragraph("SCORE", hdr_style),
         Paragraph("THRESHOLD", hdr_style), Paragraph("WHAT IT MEASURES", hdr_style),
         Paragraph("IMPLICATION", hdr_style)],
    ]

    # Attention row
    attn = data["attn"]
    sig_table_data.append([
        Paragraph("<b>Attention</b>", S("_", fontSize=9, fontName="Helvetica-Bold", leading=13, textColor=WHITE)),
        Paragraph(f"<b>{attn}/100</b>", S("_", fontSize=14, fontName="Helvetica-Bold", leading=18, textColor=a_c)),
        Paragraph("Good ≥ 60\nWeak < 30", S("_", fontSize=8, fontName="Helvetica", leading=12, textColor=MUTED)),
        Paragraph("Visual stopping power in a competitive feed. Based on face presence, contrast, and object clutter.", S("_", fontSize=9, fontName="Helvetica", leading=13, textColor=MUTED)),
        Paragraph(
            ("Will interrupt scrolling and trigger processing in cold audiences." if attn > 60
             else "Will not reliably stop cold audiences — loses the first cognitive gate." if attn >= 30
             else "Will scroll past — no mechanism to trigger an orienting response."),
            S("_", fontSize=9, fontName="Helvetica", leading=13, textColor=WHITE)
        ),
    ])

    # Memory row
    mem = data["mem"]
    sig_table_data.append([
        Paragraph("<b>Memory</b>", S("_", fontSize=9, fontName="Helvetica-Bold", leading=13, textColor=WHITE)),
        Paragraph(f"<b>{mem}/100</b>", S("_", fontSize=14, fontName="Helvetica-Bold", leading=18, textColor=m_c)),
        Paragraph("Good ≥ 70\nWeak < 40", S("_", fontSize=8, fontName="Helvetica", leading=12, textColor=MUTED)),
        Paragraph("Brand recall potential after a single exposure. Based on text density, visual simplicity, and dual-coding principles.", S("_", fontSize=9, fontName="Helvetica", leading=13, textColor=MUTED)),
        Paragraph(
            ("Strong memory encoding — brand will be recognised at point of purchase." if mem > 70
             else "Moderate recall — will require frequency to build durable memory." if mem >= 40
             else "Low retention — brand will not survive a single-exposure feed environment."),
            S("_", fontSize=9, fontName="Helvetica", leading=13, textColor=WHITE)
        ),
    ])

    # Emotional Valence row
    val = data["val"]
    sig_table_data.append([
        Paragraph("<b>Emotional Valence</b>", S("_", fontSize=9, fontName="Helvetica-Bold", leading=13, textColor=WHITE)),
        Paragraph(f"<b>{val:+.2f}</b>", S("_", fontSize=14, fontName="Helvetica-Bold", leading=18, textColor=v_c)),
        Paragraph("Positive > +0.10\nNegative < -0.10", S("_", fontSize=8, fontName="Helvetica", leading=12, textColor=MUTED)),
        Paragraph("Positive vs. negative emotional tone. Derived from colour psychology, face expression cues, and palette warmth. Range: -1.0 to +1.0.", S("_", fontSize=9, fontName="Helvetica", leading=13, textColor=MUTED)),
        Paragraph(
            ("Positive affect — will build brand warmth and purchase intent over repeated exposures." if val > 0.1
             else "Neutral affect — will not build or erode brand sentiment." if val > -0.1
             else "Negative affect — may subtly undermine brand affinity over time."),
            S("_", fontSize=9, fontName="Helvetica", leading=13, textColor=WHITE)
        ),
    ])

    # Cognitive Load row
    cl      = data["cl"]
    cl_score = data["cl_score"]
    sig_table_data.append([
        Paragraph("<b>Cognitive Load</b>", S("_", fontSize=9, fontName="Helvetica-Bold", leading=13, textColor=WHITE)),
        Paragraph(f"<b>{cl}</b><br/>{cl_score:.0f}/100", S("_", fontSize=14, fontName="Helvetica-Bold", leading=18, textColor=cl_c)),
        Paragraph("Low = best\nHigh = penalty", S("_", fontSize=8, fontName="Helvetica", leading=12, textColor=MUTED)),
        Paragraph("Processing effort required to decode the creative. Based on object count, text density, and visual complexity. High load reduces all other signals.", S("_", fontSize=9, fontName="Helvetica", leading=13, textColor=MUTED)),
        Paragraph(
            ("Low cognitive demand — brain can process the message in under 1.5 seconds." if cl == "Low"
             else "Moderate effort required — will work in lean-back formats but may struggle in fast feeds." if cl == "Medium"
             else "High cognitive demand — viewers will abandon processing before the message registers."),
            S("_", fontSize=9, fontName="Helvetica", leading=13, textColor=WHITE)
        ),
    ])

    col_ws = [28*mm, 22*mm, 26*mm, 48*mm, None]
    sig_t  = Table(sig_table_data, colWidths=col_ws, repeatRows=1)
    sig_t.setStyle(TableStyle([
        ("BACKGROUND",   (0,0),(-1,0),  colors.HexColor("#111827")),
        ("BACKGROUND",   (0,1),(-1,-1), CARD),
        ("BOX",          (0,0),(-1,-1), 0.5, BORDER),
        ("INNERGRID",    (0,0),(-1,-1), 0.3, BORDER),
        ("LEFTPADDING",  (0,0),(-1,-1), 8),
        ("RIGHTPADDING", (0,0),(-1,-1), 8),
        ("TOPPADDING",   (0,0),(-1,-1), 8),
        ("BOTTOMPADDING",(0,0),(-1,-1), 8),
        ("VALIGN",       (0,0),(-1,-1), "TOP"),
    ]))
    story.append(sig_t)
    story.append(Spacer(1, 14))
    story.append(rule())

    # ── Visual Feature Detection ───────────────────────────────────────────────
    story.append(section_label("Visual Feature Detection"))
    story.append(Paragraph(
        "Raw computer-vision measurements from the creative. These values directly feed the signal scores above.",
        sMuted,
    ))
    story.append(Spacer(1, 8))

    def feat_explanation(key, value):
        exps = {
            "face":     ("Faces trigger the fusiform face area — the brain's fastest biological attention mechanism. "
                         f"{'1 face present — orienting response will activate.' if value > 0 else 'No face detected — missing the highest-ROI attention driver.'}"),
            "contrast": (f"Contrast {value:.0f}/100. "
                         + ("Passes the visual salience gate — will stand out in a competitive feed." if value >= 60
                            else "Below the 60/100 salience threshold — will blend into feed backgrounds.")),
            "objects":  (f"{value} objects detected. "
                         + ("Clean composition — full attentional focus on primary element." if value <= 3
                            else "Moderate clutter — small attention penalty." if value <= 6
                            else f"High clutter — attention fragments across {value} elements, suppressing all signals.")),
            "text":     (f"Text covers {value*100:.0f}% of the creative area. "
                         + ("No verbal anchor — memory recall will rely on visual alone." if value < 0.04
                            else "Optimal text balance — verbal and visual channels both active." if value <= 0.20
                            else "Text-heavy — cognitive load increases and visual processing is suppressed.")),
        }
        return exps.get(key, "")

    vf_data = [
        [Paragraph("FEATURE", hdr_style), Paragraph("DETECTED VALUE", hdr_style),
         Paragraph("THRESHOLD", hdr_style), Paragraph("INTERPRETATION", hdr_style)],
        [Paragraph("<b>Human Faces</b>", S("_", fontSize=9, fontName="Helvetica-Bold", leading=13, textColor=WHITE)),
         Paragraph(f"<b>{data['face_count']} face(s)</b>", S("_", fontSize=11, fontName="Helvetica-Bold", leading=15, textColor=GREEN if data['face_count'] > 0 else RED)),
         Paragraph("≥ 1 = strong\n0 = penalty", S("_", fontSize=8, fontName="Helvetica", leading=12, textColor=MUTED)),
         Paragraph(feat_explanation("face", data["face_count"]), S("_", fontSize=9, fontName="Helvetica", leading=13, textColor=MUTED))],
        [Paragraph("<b>Contrast Score</b>", S("_", fontSize=9, fontName="Helvetica-Bold", leading=13, textColor=WHITE)),
         Paragraph(f"<b>{data['contrast_score']:.0f} / 100</b>", S("_", fontSize=11, fontName="Helvetica-Bold", leading=15, textColor=GREEN if data['contrast_score'] >= 60 else RED)),
         Paragraph("Good ≥ 60\nWeak < 45", S("_", fontSize=8, fontName="Helvetica", leading=12, textColor=MUTED)),
         Paragraph(feat_explanation("contrast", data["contrast_score"]), S("_", fontSize=9, fontName="Helvetica", leading=13, textColor=MUTED))],
        [Paragraph("<b>Object Count</b>", S("_", fontSize=9, fontName="Helvetica-Bold", leading=13, textColor=WHITE)),
         Paragraph(f"<b>{data['object_count']} objects</b>", S("_", fontSize=11, fontName="Helvetica-Bold", leading=15, textColor=GREEN if data['object_count'] <= 4 else (AMBER if data['object_count'] <= 7 else RED))),
         Paragraph("≤ 4 = clean\n> 7 = high load", S("_", fontSize=8, fontName="Helvetica", leading=12, textColor=MUTED)),
         Paragraph(feat_explanation("objects", data["object_count"]), S("_", fontSize=9, fontName="Helvetica", leading=13, textColor=MUTED))],
        [Paragraph("<b>Text Density</b>", S("_", fontSize=9, fontName="Helvetica-Bold", leading=13, textColor=WHITE)),
         Paragraph(f"<b>{data['text_density']*100:.0f}% of area</b>", S("_", fontSize=11, fontName="Helvetica-Bold", leading=15, textColor=AMBER if data['text_density'] > 0.25 else (GREEN if data['text_density'] >= 0.04 else RED))),
         Paragraph("4–20% = optimal\n> 25% = overloaded", S("_", fontSize=8, fontName="Helvetica", leading=12, textColor=MUTED)),
         Paragraph(feat_explanation("text", data["text_density"]), S("_", fontSize=9, fontName="Helvetica", leading=13, textColor=MUTED))],
    ]
    if data["dominant_colors"]:
        colors_str = "  ·  ".join(data["dominant_colors"][:5])
        vf_data.append([
            Paragraph("<b>Dominant Colours</b>", S("_", fontSize=9, fontName="Helvetica-Bold", leading=13, textColor=WHITE)),
            Paragraph(colors_str, S("_", fontSize=9, fontName="Helvetica", leading=13, textColor=MUTED)),
            Paragraph("Warm = positive\nCool/dark = risk", S("_", fontSize=8, fontName="Helvetica", leading=12, textColor=MUTED)),
            Paragraph("Palette drives emotional valence. Warm tones (reds, oranges, yellows) build positive affect. Cool or dark palettes risk neutral-to-negative brand association.", S("_", fontSize=9, fontName="Helvetica", leading=13, textColor=MUTED)),
        ])

    vf_col_ws = [32*mm, 28*mm, 26*mm, None]
    vf_t = Table(vf_data, colWidths=vf_col_ws, repeatRows=1)
    vf_t.setStyle(TableStyle([
        ("BACKGROUND",   (0,0),(-1,0),  colors.HexColor("#111827")),
        ("BACKGROUND",   (0,1),(-1,-1), CARD),
        ("BOX",          (0,0),(-1,-1), 0.5, BORDER),
        ("INNERGRID",    (0,0),(-1,-1), 0.3, BORDER),
        ("LEFTPADDING",  (0,0),(-1,-1), 8),
        ("RIGHTPADDING", (0,0),(-1,-1), 8),
        ("TOPPADDING",   (0,0),(-1,-1), 8),
        ("BOTTOMPADDING",(0,0),(-1,-1), 8),
        ("VALIGN",       (0,0),(-1,-1), "TOP"),
    ]))
    story.append(vf_t)

    story.append(Spacer(1, 8))
    story.append(rule())
    story.append(Paragraph(FOOTER_TEXT, sFoot))

    # ══════════════════════════════════════════════════════════════════════════
    # PAGE 3 — Strategic Analysis & Recommendations
    # ══════════════════════════════════════════════════════════════════════════
    story.append(PageBreak())

    story.append(Paragraph("COGNITIVE SIGNAL ANALYSIS REPORT", sEye))
    story.append(thick_rule(BLUE))
    story.append(Spacer(1, 6))

    # ── Creative Brief ─────────────────────────────────────────────────────────
    story.append(section_label("Creative Brief — Diagnosis & Scale Decision"))
    story.append(Paragraph(
        "A media-ready brief for creative and planning teams. Every line references a detected value.",
        sMuted,
    ))
    story.append(Spacer(1, 8))

    brief_rows = [
        ("CPCi Score",      f"{cpci} / 100"),
        ("Performance",     data["perf_label"]),
        ("Scale Decision",  data["scale_label"]),
        ("Use Case",        data["use_case"]),
        ("Attention",       f"{data['attn']} / 100  —  {data['attn_label']}"),
        ("Memory",          f"{data['mem']} / 100  —  {data['mem_label']}"),
        ("Emotional Valence",f"{data['val']:+.2f}  —  {data['val_label']}"),
        ("Cognitive Load",  f"{data['cl']}  ({data['cl_score']:.0f} / 100)"),
        ("Faces Detected",  str(data["face_count"])),
        ("Contrast Score",  f"{data['contrast_score']:.0f} / 100"),
        ("Object Count",    str(data["object_count"])),
        ("Text Density",    f"{data['text_density']*100:.0f}%"),
    ]

    b_rows = []
    for k, v in brief_rows:
        b_rows.append([
            Paragraph(k, S("bk", fontSize=9, fontName="Helvetica-Bold", leading=13, textColor=LABEL)),
            Paragraph(v, S("bv", fontSize=9, fontName="Helvetica", leading=13, textColor=WHITE)),
        ])

    brief_t = Table(b_rows, colWidths=[55*mm, None])
    brief_t.setStyle(TableStyle([
        ("BACKGROUND",   (0,0),(-1,-1), CARD),
        ("BOX",          (0,0),(-1,-1), 0.5, BORDER),
        ("INNERGRID",    (0,0),(-1,-1), 0.3, BORDER),
        ("LEFTPADDING",  (0,0),(-1,-1), 10),
        ("RIGHTPADDING", (0,0),(-1,-1), 10),
        ("TOPPADDING",   (0,0),(-1,-1), 7),
        ("BOTTOMPADDING",(0,0),(-1,-1), 7),
        ("VALIGN",       (0,0),(-1,-1), "MIDDLE"),
    ]))
    story.append(brief_t)
    story.append(Spacer(1, 14))
    story.append(rule())

    # ── Strategic Implication ─────────────────────────────────────────────────
    if data.get("strategic_implication"):
        story.append(section_label("Strategic Implication"))
        story.append(Paragraph(data["strategic_implication"], sBody))

        # Cold audience deployment risk — dynamic, memory-driven
        _si_mem = data["mem"]
        if _si_mem < 55:
            _risk_label = "HIGH"
            _risk_hex   = "#EF4444"
        elif _si_mem < 70:
            _risk_label = "MODERATE"
            _risk_hex   = "#F59E0B"
        else:
            _risk_label = "LOW"
            _risk_hex   = "#22C55E"

        story.append(Spacer(1, 10))
        story.append(Paragraph(
            f"<font color='{_risk_hex}'><b>COLD AUDIENCE DEPLOYMENT RISK: {_risk_label}</b></font>",
            S("deployrisk", fontSize=11, fontName="Helvetica-Bold", leading=15,
              textColor=colors.HexColor(_risk_hex), spaceAfter=0),
        ))
        story.append(Spacer(1, 12))
        story.append(rule())

    # ── Recommendation ────────────────────────────────────────────────────────
    story.append(section_label("Recommendation — Priority Actions"))
    story.append(Paragraph(
        "Ranked by expected CPCi impact. Each action references a detected value "
        "and targets the weakest cognitive signal first.",
        sMuted,
    ))
    story.append(Spacer(1, 8))
    story.append(Paragraph(data["recommendation"], sBody))
    story.append(Spacer(1, 14))
    story.append(rule())

    # ── Key Recommendations (Top 3 optimization scenarios) ────────────────────
    story.append(section_label("Key Recommendations — Top 3 Optimization Actions"))
    story.append(Paragraph(
        "Each recommendation is derived from the CPCi formula. Projected lifts use "
        "the real use-case weights — not estimates.",
        sMuted,
    ))
    story.append(Spacer(1, 10))

    scenarios = data.get("scenarios", [])
    if scenarios:
        rec_rows = []
        for idx, sc_item in enumerate(scenarios[:3], 1):
            lift_hex   = "#22C55E" if sc_item["lift"] >= 15 else ("#F59E0B" if sc_item["lift"] >= 8 else "#60A5FA")
            from_cpci  = sc_item["from_cpci"]
            to_cpci    = sc_item["to_cpci"]
            rec_rows.append([
                Paragraph(
                    f"<b>0{idx}</b>",
                    S("recnum", fontSize=16, fontName="Helvetica-Bold",
                      leading=20, textColor=colors.HexColor(sc_item["sig_color"])),
                ),
                Paragraph(
                    f"<b>{sc_item['signal']}</b>  ·  "
                    f"<font color='{lift_hex}'>+{sc_item['lift']} pts</font><br/>"
                    f"<font color='#94A3B8'>{sc_item['action']}</font><br/>"
                    f"<i><font color='#CBD5E1'>{sc_item['rationale']}</font></i>",
                    S("recbody", fontSize=9, fontName="Helvetica", leading=14, textColor=WHITE),
                ),
                Paragraph(
                    f"<b>{from_cpci}</b><br/><font color='#94A3B8'>→</font><br/>"
                    f"<font color='{lift_hex}'><b>{to_cpci}</b></font>",
                    S("reclift", fontSize=12, fontName="Helvetica-Bold",
                      leading=16, textColor=WHITE, alignment=TA_CENTER),
                ),
            ])

        rec_t = Table(rec_rows, colWidths=[12*mm, None, 20*mm])
        rec_t.setStyle(TableStyle([
            ("BACKGROUND",   (0,0),(-1,-1), CARD),
            ("BOX",          (0,0),(-1,-1), 0.5, BORDER),
            ("INNERGRID",    (0,0),(-1,-1), 0.3, BORDER),
            ("LEFTPADDING",  (0,0),(-1,-1), 10),
            ("RIGHTPADDING", (0,0),(-1,-1), 10),
            ("TOPPADDING",   (0,0),(-1,-1), 10),
            ("BOTTOMPADDING",(0,0),(-1,-1), 10),
            ("VALIGN",       (0,0),(-1,-1), "TOP"),
            ("ALIGN",        (2,0),(-1,-1), "CENTER"),
        ]))
        story.append(rec_t)
    else:
        story.append(Paragraph(
            "All cognitive signals are above threshold — no single optimization is likely "
            "to produce meaningful additional lift. This creative is ready to scale.",
            S("recnone", fontSize=11, fontName="Helvetica", leading=16, textColor=GREEN),
        ))

    story.append(Spacer(1, 14))
    story.append(rule())

    # ── CPCi Formula ─────────────────────────────────────────────────────────
    story.append(section_label("How CPCi Is Calculated"))
    story.append(Paragraph(
        "CPCi is a weighted composite score. Weights are calibrated per use-case to reflect "
        "the cognitive priorities of each media environment.",
        sMuted,
    ))
    story.append(Spacer(1, 6))

    formula_rows = [
        [Paragraph("COMPONENT", hdr_style), Paragraph("WEIGHT", hdr_style),
         Paragraph("THIS CREATIVE", hdr_style), Paragraph("CONTRIBUTION", hdr_style)],
        [Paragraph("Attention Score", S("_", fontSize=9, fontName="Helvetica", leading=13, textColor=WHITE)),
         Paragraph(f"{data['w_attn']}%", S("_", fontSize=9, fontName="Helvetica-Bold", leading=13, textColor=MUTED)),
         Paragraph(str(data["attn"]), S("_", fontSize=9, fontName="Helvetica-Bold", leading=13, textColor=a_c)),
         Paragraph(f"{data['attn'] * data['w_attn'] / 100:.1f} pts", S("_", fontSize=9, fontName="Helvetica", leading=13, textColor=WHITE))],
        [Paragraph("Memory Encoding", S("_", fontSize=9, fontName="Helvetica", leading=13, textColor=WHITE)),
         Paragraph(f"{data['w_mem']}%", S("_", fontSize=9, fontName="Helvetica-Bold", leading=13, textColor=MUTED)),
         Paragraph(str(data["mem"]), S("_", fontSize=9, fontName="Helvetica-Bold", leading=13, textColor=m_c)),
         Paragraph(f"{data['mem'] * data['w_mem'] / 100:.1f} pts", S("_", fontSize=9, fontName="Helvetica", leading=13, textColor=WHITE))],
        [Paragraph("Emotional Valence", S("_", fontSize=9, fontName="Helvetica", leading=13, textColor=WHITE)),
         Paragraph(f"{data['w_emo']}%", S("_", fontSize=9, fontName="Helvetica-Bold", leading=13, textColor=MUTED)),
         Paragraph(f"{data['val']:+.2f}", S("_", fontSize=9, fontName="Helvetica-Bold", leading=13, textColor=v_c)),
         Paragraph(f"{((data['val']+1)/2*100) * data['w_emo'] / 100:.1f} pts", S("_", fontSize=9, fontName="Helvetica", leading=13, textColor=WHITE))],
        [Paragraph("Cognitive Load Penalty", S("_", fontSize=9, fontName="Helvetica", leading=13, textColor=WHITE)),
         Paragraph("Applied if High", S("_", fontSize=9, fontName="Helvetica", leading=13, textColor=MUTED)),
         Paragraph(data["cl"], S("_", fontSize=9, fontName="Helvetica-Bold", leading=13, textColor=cl_c)),
         Paragraph("−10 pts" if data["cl"] == "High" else "0 pts", S("_", fontSize=9, fontName="Helvetica", leading=13, textColor=RED if data["cl"] == "High" else MUTED))],
        [Paragraph("<b>CPCi TOTAL</b>", S("_", fontSize=9, fontName="Helvetica-Bold", leading=13, textColor=WHITE)),
         Paragraph("", S("_", fontSize=9, fontName="Helvetica", leading=13, textColor=WHITE)),
         Paragraph("", S("_", fontSize=9, fontName="Helvetica", leading=13, textColor=WHITE)),
         Paragraph(f"<b>{cpci}</b>", S("_", fontSize=14, fontName="Helvetica-Bold", leading=18, textColor=sc))],
    ]

    form_t = Table(formula_rows, colWidths=[55*mm, 25*mm, 35*mm, None], repeatRows=1)
    form_t.setStyle(TableStyle([
        ("BACKGROUND",   (0,0),(-1,0),  colors.HexColor("#111827")),
        ("BACKGROUND",   (0,1),(-1,-2), CARD),
        ("BACKGROUND",   (0,-1),(-1,-1),colors.HexColor("#111827")),
        ("BOX",          (0,0),(-1,-1), 0.5, BORDER),
        ("INNERGRID",    (0,0),(-1,-1), 0.3, BORDER),
        ("LEFTPADDING",  (0,0),(-1,-1), 8),
        ("RIGHTPADDING", (0,0),(-1,-1), 8),
        ("TOPPADDING",   (0,0),(-1,-1), 8),
        ("BOTTOMPADDING",(0,0),(-1,-1), 8),
        ("VALIGN",       (0,0),(-1,-1), "MIDDLE"),
    ]))
    story.append(form_t)
    story.append(Spacer(1, 16))

    # ── Methodology note ──────────────────────────────────────────────────────
    story.append(rule())
    story.append(Paragraph(
        "Methodology: Cognitive Signal Engine™ integrates computer vision (OpenCV), OCR (Tesseract), "
        "Cognitive Load Theory (Sweller, 1988), Dual-Coding Theory (Paivio, 1971), and colour psychology. "
        "Developed by Anil Pandit. All scores are deterministic — identical inputs produce identical outputs.",
        sNote,
    ))
    story.append(Spacer(1, 8))
    story.append(rule())
    story.append(Paragraph(FOOTER_TEXT, sFoot))

    # ── Render with dark background ───────────────────────────────────────────
    def on_page(canvas, doc):
        canvas.saveState()
        canvas.setFillColor(BG)
        canvas.rect(0, 0, W, H, fill=1, stroke=0)
        canvas.restoreState()

    doc.build(story, onFirstPage=on_page, onLaterPages=on_page)
    return buf.getvalue()


def _generate_linkedin_text(data: dict) -> str:
    """
    LinkedIn-ready shareable summary. 3-line insight + CPCi + recommendation.
    Designed to be punchy, viral-friendly, and attribution-stamped.
    """
    cpci   = data["cpci"]
    perf   = data["perf_label"]
    attn   = data["attn"]
    mem    = data["mem"]
    cl     = data["cl"]
    val    = data["val"]
    scens  = data.get("scenarios", [])

    # ── Tier-specific hook ────────────────────────────────────────────────────
    if cpci >= 70:
        hook = (
            f"🧠 We ran a brain-signal test on an ad creative before spending a penny on media.\n\n"
            f"CPCi score: {cpci}/100 — {perf}.\n\n"
            f"The neuroscience said: this one is ready to scale."
        )
        cta = (
            "Most media waste happens before the first impression is served.\n"
            "CPCi catches it at the creative stage — not on the post-campaign report."
        )
    elif cpci >= 40:
        best_lift = scens[0]["lift"] if scens else None
        lift_line = (
            f"One fix could lift CPCi by +{best_lift} points.\n"
            f"That's the difference between average and top-quartile performance."
        ) if best_lift else (
            "There's a gap. The signals show exactly where."
        )
        hook = (
            f"🧠 We brain-scored an ad creative before committing media budget.\n\n"
            f"CPCi: {cpci}/100 — {perf}. There's signal. But there's a gap.\n\n"
            f"{lift_line}"
        )
        cta = (
            "Pre-bid creative scoring means you optimise before you spend.\n"
            "Not after."
        )
    else:
        hook = (
            f"🧠 We almost spent media budget on a creative the brain wouldn't process.\n\n"
            f"CPCi: {cpci}/100 — {perf}.\n\n"
            f"At that score, most impressions won't cognitively register. "
            f"The message won't be encoded. The brand won't be recalled."
        )
        cta = (
            "We caught it before a single impression was served.\n"
            "That's the whole point of pre-bid creative intelligence."
        )

    # ── 3-line signal insights ────────────────────────────────────────────────
    a_insight = (
        "Attention is strong — this creative will stop the scroll."
        if attn > 60 else
        "Attention is moderate — won't reliably interrupt cold-audience feeds."
        if attn >= 30 else
        "Attention is weak — the brain won't pause for this creative."
    )
    m_insight = (
        "Memory encoding is high — one exposure is enough to build brand recall."
        if mem > 70 else
        "Memory encoding is moderate — needs frequency to build durable recall."
        if mem >= 40 else
        "Memory encoding is low — the brand won't survive a single-exposure feed."
    )
    load_insight = (
        "Cognitive load is low — the message reaches the brain cleanly in <1.5s."
        if cl == "Low" else
        "Cognitive load is medium — will work in lean-back formats, struggles in feed."
        if cl == "Medium" else
        "Cognitive load is high — viewers abandon processing before the message lands."
    )

    # ── Primary recommendation ────────────────────────────────────────────────
    if scens:
        top = scens[0]
        rec_line = (
            f"Priority fix: {top['action']}\n"
            f"Projected impact: CPCi {top['from_cpci']} → {top['to_cpci']} "
            f"(+{top['lift']} pts)"
        )
    else:
        rec_line = data.get("fix_first", "")

    lines = [
        hook,
        "",
        "Here's what the 3 cognitive signals revealed:",
        f"→ {a_insight}",
        f"→ {m_insight}",
        f"→ {load_insight}",
        "",
        rec_line,
        "",
        cta,
        "",
        "─" * 40,
        "#CPCi #CreativeIntelligence #MediaEfficiency #Neuroscience "
        "#AdTech #BrainScience #CognitiveScience #BrandStrategy #MediaPlanning",
        "",
        "Powered by Cognitive Signal Engine™ — ADVantage Insights",
    ]
    return "\n".join(lines)


def _generate_summary_text(data: dict) -> str:
    """Plain-text version of the client report — for clipboard copy."""
    lines = [
        f"COGNITIVE SIGNAL ANALYSIS REPORT",
        f"{'─' * 44}",
        f"Creative:    {data['name']}",
        f"Use Case:    {data['use_case']}",
        f"",
        f"CPCi SCORE   {data['cpci']} / 100  —  {data['perf_label']}",
        f"",
        f"VERDICT",
        f"{data['verdict']}",
        f"",
        f"WHAT THIS MEANS",
        f"01 Performance  {data['performance']}",
        f"02 Core Issue   {data['core_issue']}",
        f"03 Fix First    {data['fix_first']}",
        f"",
        f"RECOMMENDATION",
        f"{data['recommendation']}",
        f"",
        f"{'─' * 44}",
        f"Cognitive Signal Engine™ · Creative Intelligence Analyzer · © Anil Pandit",
    ]
    return "\n".join(lines)


# ── Demo Mode — pre-built synthetic results ───────────────────────────────────
#
# Three archetypes that showcase the full scoring range and "Why this wins"
# logic without requiring any file uploads. Used for presentations and demos.
#
_DEMO_RESULTS: list = [
    {
        "name": "Creative_A_Hero_Product.jpg",
        "file_path": "",
        "cpci": 78,
        "signals": {
            "attention_score": 72,
            "memory_score":    68,
            "emotional_valence": 0.22,
            "cognitive_load":  "Low",
            "cognitive_load_score": 28,
        },
        "visual_features": {
            "face_count":      1,
            "contrast_score":  74.0,
            "object_count":    3,
            "text_density":    0.12,
            "dominant_colors": ["#E8F4FD", "#2563EB", "#F9FAFB"],
            "is_video":        False,
            "duration":        0,
            "fps":             0,
            "frame_count":     0,
        },
        "reasoning": "",
        "narrative": {
            "strategic_implication": (
                "Strong hero-product layout — single face, minimal copy, "
                "high contrast. Will stop the scroll in cold audiences and "
                "encode the brand quickly."
            ),
            "recommendations": (
                "Test a version with an even tighter crop on the face to push "
                "attention scores above 80. Consider adding one short headline "
                "to anchor memory."
            ),
        },
    },
    {
        "name": "Creative_B_Lifestyle_Scene.jpg",
        "file_path": "",
        "cpci": 61,
        "signals": {
            "attention_score": 58,
            "memory_score":    64,
            "emotional_valence": 0.14,
            "cognitive_load":  "Medium",
            "cognitive_load_score": 52,
        },
        "visual_features": {
            "face_count":      2,
            "contrast_score":  55.0,
            "object_count":    6,
            "text_density":    0.18,
            "dominant_colors": ["#FEF3C7", "#D97706", "#FFFFFF"],
            "is_video":        False,
            "duration":        0,
            "fps":             0,
            "frame_count":     0,
        },
        "reasoning": "",
        "narrative": {
            "strategic_implication": (
                "Warm lifestyle scene builds emotional affinity and recall "
                "but the composition is slightly busy — the eye doesn't land "
                "cleanly on a single focal point."
            ),
            "recommendations": (
                "Crop tighter to remove background clutter and increase "
                "subject contrast. Reducing objects from 6 to 3–4 should "
                "lift CPCi by 8–12 points."
            ),
        },
    },
    {
        "name": "Creative_C_Text_Heavy.jpg",
        "file_path": "",
        "cpci": 34,
        "signals": {
            "attention_score": 31,
            "memory_score":    40,
            "emotional_valence": -0.08,
            "cognitive_load":  "High",
            "cognitive_load_score": 81,
        },
        "visual_features": {
            "face_count":      0,
            "contrast_score":  38.0,
            "object_count":    11,
            "text_density":    0.41,
            "dominant_colors": ["#6B7280", "#374151", "#9CA3AF"],
            "is_video":        False,
            "duration":        0,
            "fps":             0,
            "frame_count":     0,
        },
        "reasoning": "",
        "narrative": {
            "strategic_implication": (
                "Text-heavy layout with no face and low contrast. "
                "The visual channel is overloaded — viewers will scroll past "
                "before the message registers."
            ),
            "recommendations": (
                "Rebuild with a single dominant image (preferably a face), "
                "reduce copy to one headline under 6 words, and increase "
                "background contrast to at least 60/100."
            ),
        },
    },
]


# ── Demo creative metadata (persona names + descriptions) ─────────────────────
# Keyed by the "name" field in _DEMO_RESULTS
_DEMO_META = {
    "Creative_A_Hero_Product.jpg": {
        "persona":   "Hero Product Ad",
        "archetype": "High Performer",
        "summary":   "Single face, clean layout, high contrast — clears every cognitive gate.",
        "facts":     ["Face present", "Low cognitive load", "Strong contrast (74/100)"],
        "tier_label":"High Efficiency",
        "tier_color":"#22C55E",
    },
    "Creative_B_Lifestyle_Scene.jpg": {
        "persona":   "Lifestyle Scene",
        "archetype": "Moderate Performer",
        "summary":   "Warm and emotional, but the busy composition dilutes attention signal.",
        "facts":     ["2 faces", "Medium load", "6 visual elements"],
        "tier_label":"Moderate Performance",
        "tier_color":"#F59E0B",
    },
    "Creative_C_Text_Heavy.jpg": {
        "persona":   "Text-Heavy Ad",
        "archetype": "Underperformer",
        "summary":   "No face, high load, low contrast — fails the first cognitive gate.",
        "facts":     ["No face", "High cognitive load", "Low contrast (38/100)"],
        "tier_label":"High Waste Risk",
        "tier_color":"#EF4444",
    },
}

_DEMO_STEPS = [
    ("1", "Select Creative",    "Choose a synthetic ad to analyze"),
    ("2", "Read the Analysis",  "Understand CPCi and what drives it"),
    ("3", "See the Opportunity","Optimization scenario + full comparison"),
]

_GUIDED_BANNERS = {
    "step1": (
        "👋 Welcome to the demo",
        "Select one of three synthetic ad creatives below. Each represents a real-world "
        "scenario: a high performer, a moderate performer, and an underperformer. "
        "The analysis will show you exactly why each scores the way it does.",
    ),
    "before_cpci": (
        "📊 CPCi — Cost Per Cognitive Impression",
        "This is the composite score. A score above 70 means the creative is likely to "
        "perform in cold audiences. Below 40 signals high waste risk. The number you see "
        "is derived from brain-level signals — not opinion.",
    ),
    "before_signals": (
        "🧠 Four cognitive signals",
        "These are the inputs that produce CPCi. Each measures a different brain function: "
        "Attention (stopping power), Memory (recall probability), Emotion (brand affinity), "
        "and Cognitive Load (processing friction). A weakness in any one suppresses the score.",
    ),
    "before_business": (
        "💼 Business Impact",
        "This section translates cognitive scores into commercial language. Designed for "
        "budget holders — it tells you whether to deploy, fix, or kill the creative "
        "before any media spend is committed.",
    ),
    "before_optimization": (
        "🎯 The growth story",
        "This is the key output for your creative team. It simulates what one targeted fix "
        "is worth in CPCi points — and maps that to a specific action, not a vague suggestion.",
    ),
    "step3": (
        "📊 Three creatives, one decision",
        "This is what the tool looks like in a real A/B review. It tells you which creative "
        "to put media behind — and exactly why the winner wins.",
    ),
}


def _guided_banner(key: str) -> None:
    """Renders a guided-mode contextual banner. Only shows when guided_walkthrough is ON."""
    if not st.session_state.get("guided_walkthrough", True):
        return
    title, body = _GUIDED_BANNERS.get(key, ("", ""))
    if not title:
        return
    st.markdown(
        f"<div style='background:#0D1117;border:1px solid #1D4ED8;border-left:3px solid #3B82F6;"
        f"border-radius:12px;padding:16px 20px;margin:0 0 24px 0;'>"
        f"<div style='font-size:13px;font-weight:700;color:#60A5FA;"
        f"letter-spacing:0.5px;margin-bottom:6px;'>{title}</div>"
        f"<div style='font-size:13px;color:#CBD5E1;line-height:1.65;'>{body}</div>"
        f"</div>",
        unsafe_allow_html=True,
    )


def _render_step_indicator(current: int) -> None:
    """Horizontal 3-step progress bar."""
    parts = []
    for num, label, sub in _DEMO_STEPS:
        step_n = int(num)
        if step_n < current:
            dot_bg, dot_color, txt_color = "#22C55E", "#FFFFFF", "#22C55E"
            icon = "✓"
        elif step_n == current:
            dot_bg, dot_color, txt_color = "#3B82F6", "#FFFFFF", "#FFFFFF"
            icon = num
        else:
            dot_bg, dot_color, txt_color = "#1F2937", "#94A3B8", "#94A3B8"
            icon = num

        connector = (
            f"<div style='flex:1;height:2px;background:"
            f"{'#22C55E' if step_n <= current else '#1F2937'};"
            f"margin:0 8px;margin-top:14px;align-self:flex-start;'></div>"
            if step_n < 3 else ""
        )

        parts.append(
            f"<div style='display:flex;flex-direction:column;align-items:center;"
            f"min-width:100px;'>"
            f"<div style='width:28px;height:28px;border-radius:50%;"
            f"background:{dot_bg};color:{dot_color};"
            f"display:flex;align-items:center;justify-content:center;"
            f"font-size:13px;font-weight:700;margin-bottom:6px;'>{icon}</div>"
            f"<div style='font-size:13px;font-weight:600;color:{txt_color};"
            f"text-align:center;'>{label}</div>"
            f"<div style='font-size:11px;color:#94A3B8;text-align:center;"
            f"margin-top:2px;max-width:110px;line-height:1.3;'>{sub}</div>"
            f"</div>"
            + connector
        )

    st.markdown(
        f"<div style='display:flex;align-items:flex-start;justify-content:center;"
        f"gap:0;padding:24px 0 32px 0;'>"
        + "".join(parts) +
        f"</div>",
        unsafe_allow_html=True,
    )


def _render_demo_mode(client_mode: bool) -> None:
    """
    Full demo experience with 3 preloaded creatives and optional guided walkthrough.
    Replaces the uploader when demo_mode is active.
    """
    # ── Init session state ────────────────────────────────────────────────────
    if "demo_step"     not in st.session_state: st.session_state["demo_step"]     = 1
    if "demo_creative" not in st.session_state: st.session_state["demo_creative"] = 0
    if "guided_walkthrough" not in st.session_state:
        st.session_state["guided_walkthrough"] = True

    demo_sorted = sorted(_DEMO_RESULTS, key=lambda x: x["cpci"], reverse=True)
    step        = st.session_state["demo_step"]

    # ── Demo header ───────────────────────────────────────────────────────────
    hdr_left, hdr_right = st.columns([3, 1])
    with hdr_left:
        st.markdown(
            "<div style='display:inline-flex;align-items:center;gap:10px;"
            "background:#141B24;border:1px solid #3B82F6;border-radius:8px;"
            "padding:10px 16px;margin-bottom:4px;'>"
            "<span style='font-size:15px;'>🎬</span>"
            "<span style='font-size:14px;font-weight:700;color:#60A5FA;'>"
            "Demo Mode</span>"
            "<span style='width:1px;height:16px;background:#1F2937;display:inline-block;'></span>"
            "<span style='font-size:13px;color:#CBD5E1;'>"
            "3 synthetic creatives · Performance Marketing</span>"
            "</div>",
            unsafe_allow_html=True,
        )
    with hdr_right:
        st.toggle(
            "Guided Walkthrough",
            value=st.session_state.get("guided_walkthrough", True),
            key="guided_walkthrough",
            help="Turn on for step-by-step explanations of each metric and section.",
        )

    # ── Step indicator ────────────────────────────────────────────────────────
    _render_step_indicator(step)

    # ══════════════════════════════════════════════════════════════════════════
    # STEP 1 — Select a creative
    # ══════════════════════════════════════════════════════════════════════════
    if step == 1:
        _guided_banner("step1")

        cols = st.columns(3, gap="medium")
        for col, r in zip(cols, demo_sorted):
            meta = _DEMO_META.get(r["name"], {})
            cpci_v   = r["cpci"]
            tc       = meta.get("tier_color", "#94A3B8")
            tl       = meta.get("tier_label", "")
            persona  = meta.get("persona",   r["name"])
            archetype= meta.get("archetype", "")
            summary  = meta.get("summary",   "")
            facts    = meta.get("facts",     [])

            facts_html = "".join(
                f"<div style='display:flex;align-items:center;gap:6px;"
                f"margin-bottom:4px;'>"
                f"<span style='width:5px;height:5px;border-radius:50%;"
                f"background:{tc};flex-shrink:0;display:inline-block;'></span>"
                f"<span style='font-size:12px;color:#CBD5E1;'>{f}</span>"
                f"</div>"
                for f in facts
            )

            with col:
                st.markdown(
                    f"<div style='background:#141B24;border:1px solid #1F2937;"
                    f"border-top:3px solid {tc};border-radius:12px;"
                    f"padding:24px 20px 20px 20px;height:100%;'>"

                    f"<div style='font-size:11px;font-weight:700;color:{tc};"
                    f"letter-spacing:1.5px;text-transform:uppercase;"
                    f"margin-bottom:10px;'>{archetype}</div>"

                    f"<div style='font-size:17px;font-weight:600;color:#FFFFFF;"
                    f"margin-bottom:6px;line-height:1.25;'>{persona}</div>"

                    f"<div style='display:flex;align-items:baseline;gap:6px;"
                    f"margin-bottom:16px;'>"
                    f"<span style='font-size:48px;font-weight:600;color:{tc};"
                    f"line-height:1;letter-spacing:-1px;'>{cpci_v}</span>"
                    f"<span style='font-size:14px;color:#94A3B8;'>/100 CPCi</span>"
                    f"</div>"

                    f"<div style='font-size:13px;color:#CBD5E1;line-height:1.6;"
                    f"margin-bottom:16px;'>{summary}</div>"

                    f"<div style='margin-bottom:20px;'>{facts_html}</div>"

                    f"<div style='display:inline-block;background:{tc}18;"
                    f"border:1px solid {tc}44;border-radius:6px;"
                    f"padding:4px 10px;margin-bottom:0;'>"
                    f"<span style='font-size:11px;font-weight:700;color:{tc};"
                    f"text-transform:uppercase;letter-spacing:1px;'>{tl}</span>"
                    f"</div>"

                    f"</div>",
                    unsafe_allow_html=True,
                )
                # Spacer then button below the card
                st.markdown("<div style='height:10px;'></div>", unsafe_allow_html=True)
                idx = demo_sorted.index(r)
                if st.button(
                    f"Analyze this creative →",
                    key=f"demo_pick_{idx}",
                    use_container_width=True,
                ):
                    st.session_state["demo_creative"] = idx
                    st.session_state["demo_step"]     = 2
                    st.rerun()

        st.markdown("<hr class='divider'>", unsafe_allow_html=True)
        if st.button("Skip to full comparison →", use_container_width=False):
            st.session_state["demo_step"] = 3
            st.rerun()

    # ══════════════════════════════════════════════════════════════════════════
    # STEP 2 — Full analysis of selected creative
    # ══════════════════════════════════════════════════════════════════════════
    elif step == 2:
        r    = demo_sorted[st.session_state["demo_creative"]]
        meta = _DEMO_META.get(r["name"], {})
        tc   = meta.get("tier_color", "#94A3B8")

        # Creative identifier strip
        st.markdown(
            f"<div style='display:flex;align-items:center;gap:12px;"
            f"background:#141B24;border:1px solid #1F2937;border-left:3px solid {tc};"
            f"border-radius:8px;padding:12px 18px;margin-bottom:24px;'>"
            f"<span style='font-size:13px;font-weight:700;color:{tc};"
            f"text-transform:uppercase;letter-spacing:1px;'>"
            f"{meta.get('archetype','Creative')}</span>"
            f"<span style='color:#1F2937;'>·</span>"
            f"<span style='font-size:14px;font-weight:600;color:#FFFFFF;'>"
            f"{meta.get('persona', r['name'])}</span>"
            f"<span style='font-size:13px;color:#94A3B8;margin-left:auto;'>"
            f"CPCi {r['cpci']} / 100</span>"
            f"</div>",
            unsafe_allow_html=True,
        )

        # Guided banners — fire before key sections
        _guided_banner("before_cpci")

        # Run full analysis
        show_results(r, elapsed=None, use_case="Performance Marketing",
                     client_mode=client_mode)

        # Navigation row
        st.markdown("<hr class='divider'>", unsafe_allow_html=True)
        nav_back, nav_fwd = st.columns([1, 2])
        with nav_back:
            if st.button("← Back to creative selection", use_container_width=True):
                st.session_state["demo_step"] = 1
                st.rerun()
        with nav_fwd:
            if st.button(
                "Next: Compare all 3 creatives →",
                type="primary",
                use_container_width=True,
            ):
                st.session_state["demo_step"] = 3
                st.rerun()

    # ══════════════════════════════════════════════════════════════════════════
    # STEP 3 — Comparison + exit CTA
    # ══════════════════════════════════════════════════════════════════════════
    elif step == 3:
        _guided_banner("step3")

        show_comparison(demo_sorted, "Performance Marketing", client_mode)

        # Exit CTA
        st.markdown("<hr class='divider'>", unsafe_allow_html=True)
        st.markdown(
            "<div style='background:#0D1117;border:1px solid #3B82F6;"
            "border-radius:14px;padding:28px 32px;text-align:center;margin:0 0 32px 0;'>"
            "<div style='font-size:20px;font-weight:600;color:#FFFFFF;"
            "margin-bottom:8px;'>Ready to analyze your own creative?</div>"
            "<div style='font-size:14px;color:#CBD5E1;margin-bottom:24px;'>"
            "Upload any JPG, PNG, or MP4 — results in under 30 seconds.</div>"
            "</div>",
            unsafe_allow_html=True,
        )
        cta_back, cta_exit = st.columns([1, 2])
        with cta_back:
            if st.button("← Back to analysis", use_container_width=True):
                st.session_state["demo_step"] = 2
                st.rerun()
        with cta_exit:
            if st.button(
                "Upload my own creative →",
                type="primary",
                use_container_width=True,
            ):
                # Can't set "demo_mode" directly here — it's bound to a widget key.
                # Signal the header guard (runs before the toggle) to clear it on next rerun.
                st.session_state["_exit_demo"] = True
                st.rerun()


def _render_trust_indicators(cpci: float, attn: int, mem: int, conf_level: str, conf_color: str) -> None:
    """
    Three trust signals rendered below every result:
      1. How this works (collapsed expander)
      2. Limitations (1-line inline note)
      3. Confidence indicator (dot + label)
    """
    st.markdown(
        "<div style='border-top:1px solid #1F2937;margin:32px 0 20px 0;'></div>",
        unsafe_allow_html=True,
    )

    left_col, right_col = st.columns([3, 1], gap="large")

    # ── Left: How this works (collapsed) + Limitations ────────────────────────
    with left_col:
        with st.expander("ℹ️  How this works", expanded=False):
            st.markdown(
                "<div style='font-size:13px;color:#CBD5E1;line-height:1.7;"
                "padding:4px 0 8px 0;'>"

                "<strong style='color:#93C5FD;'>What CPCi measures</strong><br>"
                "CPCi (Cost Per Cognitive Impression) is a composite score produced by the "
                "Cognitive Signal Engine™ — an original multi-layer analytical system that models how "
                "the human brain processes advertising creative. It draws on principles from "
                "neuroscience, behavioural science, and advertising theory to produce three weighted "
                "signals: <span style='color:#3B82F6;'>Attention</span>, "
                "<span style='color:#22C55E;'>Memory encoding</span>, and "
                "<span style='color:#a78bfa;'>Emotional valence</span>.<br><br>"

                "<strong style='color:#93C5FD;'>Signal sources</strong><br>"
                "Each signal is derived from measurable visual properties — face presence, "
                "contrast ratios, object density, text load, colour palette, and spatial "
                "composition — across independent analytical layers. The system does not rely "
                "on a single AI model. Theoretical grounding comes from Sweller's Cognitive "
                "Load Theory, Paivio's Dual-Coding Theory, and empirical attention research.<br><br>"

                "<strong style='color:#93C5FD;'>Use case weighting</strong><br>"
                "The formula shifts weights based on your campaign objective — "
                "Performance Marketing prioritises attention, Brand campaigns prioritise "
                "memory encoding. This changes the final CPCi score without changing "
                "the underlying signal readings.<br><br>"

                "<strong style='color:#93C5FD;'>Origin</strong><br>"
                "Cognitive Signal Engine™ is a proprietary framework developed by "
                "<span style='color:#FFFFFF;font-weight:500;'>Anil Pandit</span>, "
                "integrating neuroscience, behavioral science, and advertising theory "
                "into a decision system for creative effectiveness. "
                "It was built independently using computer vision, signal processing, "
                "and cognitive science theory. TRIBE v2 was the originating research context — "
                "the Cognitive Signal Engine is the productised analytical layer built on top of it."

                "</div>",
                unsafe_allow_html=True,
            )

        st.markdown(
            "<div style='font-size:12px;color:#94A3B8;margin-top:10px;line-height:1.6;'>"
            "<span style='color:#F59E0B;font-weight:600;'>⚠ Limitations</span>&nbsp; "
            "This is a predictive cognitive model based on visual signals, not real user "
            "behaviour. Scores are directional — use them to prioritise testing, not to "
            "replace it."
            "</div>",
            unsafe_allow_html=True,
        )
        st.markdown(
            "<div style='font-size:13px;color:#CBD5E1;margin-top:12px;line-height:1.65;"
            "padding-top:10px;border-top:1px solid #1F2937;'>"
            "<span style='color:#CBD5E1;font-weight:500;'>Methodology note</span>"
            " &mdash; This system is inspired by advances like Meta's TRIBE v2, but extends "
            "them into a practical decision framework for advertising using cognitive signal modeling."
            "</div>",
            unsafe_allow_html=True,
        )

    # ── Right: Confidence indicator ───────────────────────────────────────────
    with right_col:
        conf_bg = {"High": "#22C55E18", "Medium": "#F59E0B18", "Low": "#EF444418"}.get(conf_level, "#1F2937")
        conf_desc = {
            "High":   "Both attention and memory signals are strong. Score is reliable.",
            "Medium": "Signals are mixed. Score is directional — validate with a test.",
            "Low":    "Signals diverge significantly. Treat this score as indicative only.",
        }.get(conf_level, "")
        st.markdown(
            f"<div style='background:{conf_bg};border:1px solid #1F2937;border-radius:16px;"
            f"padding:14px 16px;'>"
            f"<div style='font-size:13px;font-weight:600;color:#94A3B8;"
            f"letter-spacing:1.4px;text-transform:uppercase;margin-bottom:8px;'>"
            f"Model Confidence</div>"
            f"<div style='display:flex;align-items:center;gap:8px;margin-bottom:8px;'>"
            f"<span style='font-size:18px;'>●</span>"
            f"<span style='font-size:17px;font-weight:700;color:{conf_color};'>{conf_level}</span>"
            f"</div>"
            f"<div style='font-size:13px;color:#94A3B8;line-height:1.5;'>{conf_desc}</div>"
            f"</div>",
            unsafe_allow_html=True,
        )


def _render_cta_block(r: dict, use_case: str) -> None:
    """Bottom-of-page CTA — makes the app feel like a product, not a demo."""
    st.markdown(
        "<div style='background:linear-gradient(135deg,#0F172A 0%,#1E293B 60%,#0F172A 100%);"
        "border:1px solid #334155;border-radius:16px;padding:48px 40px;"
        "margin:48px 0 8px 0;text-align:center;'>"
        "<div style='font-size:11px;font-weight:600;letter-spacing:2px;"
        "text-transform:uppercase;color:#60A5FA;margin-bottom:16px;'>"
        "Cognitive Signal Engine™</div>"
        "<div style='font-size:28px;font-weight:800;color:#FFFFFF;"
        "line-height:1.2;letter-spacing:-0.02em;margin-bottom:12px;'>"
        "Ready to test your creatives before media spend?</div>"
        "<div style='font-size:15px;color:#94A3B8;margin-bottom:8px;line-height:1.6;'>"
        "Stop guessing. Know which creatives earn attention, build memory, "
        "and drive response — before you commit budget.</div>"
        "</div>",
        unsafe_allow_html=True,
    )

    # Button row — centred with spacer columns
    _, btn_l, btn_r, _ = st.columns([1.5, 1, 1, 1.5])

    with btn_l:
        if st.button(
            "🧠  Analyze Your Creative",
            use_container_width=True,
            type="primary",
            key=f"cta_analyze_{r.get('name','default')}",
        ):
            # Scroll to top / re-trigger upload flow
            st.session_state["_cta_new_analysis"] = True
            st.rerun()

    with btn_r:
        # Reuse the PDF bytes — carry client_name from the export bar input
        _cta_client_name = st.session_state.get(
            f"pdf_client_name_{r.get('name','default')}", ""
        )
        data = _build_report_data(r, use_case, client_name=_cta_client_name)
        try:
            pdf_bytes = _generate_pdf_bytes(data)
            st.download_button(
                "📄  Download Report",
                pdf_bytes,
                file_name=f"CSE_Report_{r.get('name','creative')}.pdf",
                mime="application/pdf",
                use_container_width=True,
                key=f"cta_dl_{r.get('name','default')}",
            )
        except Exception:
            st.button(
                "📄  Download Report",
                use_container_width=True,
                disabled=True,
                key=f"cta_dl_disabled_{r.get('name','default')}",
            )

    st.markdown(
        "<p style='text-align:center;color:#374151;font-size:11px;"
        "letter-spacing:1.2px;text-transform:uppercase;margin-top:20px;'>"
        "ADVantage Insights · Cognitive Signal Engine™ · © Anil Pandit</p>",
        unsafe_allow_html=True,
    )


def _render_export_bar(r: dict, use_case: str) -> None:
    """Renders the Export Report button row below any result view."""
    st.markdown(
        "<div style='border-top:1px solid #1F2937;margin:32px 0 20px 0;padding-top:20px;'>"
        "<span style='font-size:13px;font-weight:600;color:#94A3B8;"
        "letter-spacing:1.5px;text-transform:uppercase;'>Export Report</span>"
        "</div>",
        unsafe_allow_html=True,
    )

    client_name = st.text_input(
        "Client name (appears on PDF cover page)",
        placeholder="e.g. Unilever, Nike, P&G",
        key=f"pdf_client_name_{r.get('name','default')}",
        label_visibility="visible",
    )

    data = _build_report_data(r, use_case, client_name=client_name)

    pdf_col, txt_col, _ = st.columns([1, 1, 3])

    with pdf_col:
        pdf_bytes = _generate_pdf_bytes(data)
        fname     = data["name"].rsplit(".", 1)[0] + "_cpci_report.pdf"
        st.download_button(
            "⬇️  Download PDF",
            data       = pdf_bytes,
            file_name  = fname,
            mime       = "application/pdf",
            use_container_width=True,
        )

    with txt_col:
        summary = _generate_summary_text(data)
        st.download_button(
            "📋  Download Summary",
            data       = summary,
            file_name  = data["name"].rsplit(".", 1)[0] + "_summary.txt",
            mime       = "text/plain",
            use_container_width=True,
        )

    # ── LinkedIn Share Card (collapsed) ──────────────────────────────────────
    with st.expander("🔗  Share on LinkedIn"):
        linkedin_text = _generate_linkedin_text(data)
        cpci_li = data["cpci"]
        li_badge = ("🟢 Strong Performer" if cpci_li >= 70
                    else "🟡 Average Performer" if cpci_li >= 40
                    else "🔴 Not Ready to Scale")
        st.markdown(
            f"<div style='font-size:12px;color:#64748B;margin-bottom:8px;'>"
            f"CPCi {cpci_li}/100  ·  {li_badge}  —  copy and paste to LinkedIn:</div>",
            unsafe_allow_html=True,
        )
        st.code(linkedin_text, language=None)


# ── Single-creative renderer ──────────────────────────────────────────────────

def show_results(r: dict, elapsed: float = None, use_case: str = "Performance Marketing", client_mode: bool = False) -> None:
    """Full analysis report for a single creative."""
    if client_mode:
        _show_results_client(r, use_case)
        return
    s        = r["signals"]
    vf       = r["visual_features"]
    cpci     = round(r["cpci"], 1)
    narr     = r.get("narrative", {})
    attn     = s["attention_score"]
    mem      = s["memory_score"]
    val      = s["emotional_valence"]
    cl       = s["cognitive_load"]
    _rsn     = r.get("reasoning") or {}
    cl_score = (
        _rsn["load"]["composite"]
        if isinstance(_rsn, dict) and "load" in _rsn
        else s.get("cognitive_load_score", 50)
    )
    uc       = USE_CASES[use_case]
    w        = uc["weights"]

    # ── Derived color + label ─────────────────────────────────────────────────
    if cpci >= 70:   cc, clabel, cstyle = "#22C55E", "Strong Performer",     "good"
    elif cpci >= 40: cc, clabel, cstyle = "#F59E0B", "Average Performer",    "warn"
    else:            cc, clabel, cstyle = "#EF4444", "Needs Improvement",    "bad"

    if attn > 60:    a_color, a_label = "#22C55E", "High Attention"
    elif attn >= 30: a_color, a_label = "#F59E0B", "Moderate"
    else:            a_color, a_label = "#EF4444", "Scroll-Past Risk"

    if mem > 70:     m_color, m_label = "#22C55E", "Strong Recall"
    elif mem >= 40:  m_color, m_label = "#F59E0B", "Moderate"
    else:            m_color, m_label = "#EF4444", "Low Retention"

    if val > 0.1:    v_color, v_label = "#22C55E", "Positive"
    elif val > -0.1: v_color, v_label = "#F59E0B", "Neutral"
    else:            v_color, v_label = "#EF4444", "Negative"

    if cl == "Low":      cl_color = "#22C55E"
    elif cl == "Medium": cl_color = "#F59E0B"
    else:                cl_color = "#EF4444"

    # ── Confidence level ──────────────────────────────────────────────────────
    if attn > 50 and mem > 50:
        conf_level, conf_color = "High",   "#22C55E"
    elif attn < 30 and mem < 30:
        conf_level, conf_color = "Low",    "#EF4444"
    elif abs(attn - mem) > 35 or (attn < 30 or mem < 30):
        conf_level, conf_color = "Low",    "#EF4444"
    else:
        conf_level, conf_color = "Medium", "#F59E0B"

    source      = narr.get("_source", "rules")
    source_icon = "✨ AI · Claude" if source == "claude" else "⚙️ Rule Engine"

    # ── Timer bar (full width) ────────────────────────────────────────────────
    if elapsed:
        st.markdown(
            f"<div style='display:flex;align-items:center;justify-content:space-between;"
            f"margin-bottom:20px;'>"
            f"<div class='timer-box'>⚡ Analysis completed in {elapsed:.2f}s</div>"
            f"<div class='source-badge'>{source_icon}</div>"
            f"</div>",
            unsafe_allow_html=True,
        )

    # ── Video metadata badge (shown only for video creatives) ────────────────
    if vf.get("is_video"):
        _dur = vf.get("duration", 0)
        _fps = vf.get("fps", 0)
        _fc  = vf.get("frame_count", 0)
        _dur_str = f"{int(_dur // 60)}m {int(_dur % 60)}s" if _dur >= 60 else f"{_dur:.1f}s"
        st.markdown(
            f"<div style='display:inline-flex;align-items:center;gap:16px;"
            f"background:#141B24;border:1px solid #1F2937;border-radius:16px;"
            f"padding:8px 16px;margin-bottom:20px;flex-wrap:wrap;'>"
            f"<span style='font-size:13px;font-weight:700;color:#3B82F6;"
            f"letter-spacing:1.5px;text-transform:uppercase;'>🎬 Video Creative</span>"
            f"<span style='font-size:12px;color:#CBD5E1;'>Duration: <b>{_dur_str}</b></span>"
            f"<span style='font-size:12px;color:#CBD5E1;'>FPS: <b>{_fps:.0f}</b></span>"
            f"<span style='font-size:12px;color:#CBD5E1;'>Frames sampled: <b>6</b></span>"
            f"<span style='font-size:13px;color:#94A3B8;'>CPCi scored on frame samples</span>"
            f"</div>",
            unsafe_allow_html=True,
        )

    # ── Creative image — anchors analysis to the actual creative ─────────────
    _render_creative_hero(r.get("file_path", ""), r.get("name", ""), is_video=r.get("visual_features", {}).get("is_video", False))

    # ══════════════════════════════════════════════════════════════════════════
    # TIER 1 — CPCi + Verdict + What This Means  (full-width hero)
    # ══════════════════════════════════════════════════════════════════════════

    # ── CPCi hero — full-width standalone, centered ──────────────────────────
    # CSS @property syntax:'<integer>' rejects floats → always cast to int
    _cpci_int = int(round(cpci))
    _uid = f"cp{abs(hash(str(_cpci_int) + use_case)) % 99991}"
    st.markdown(
        f"<style>"
        f"@property --{_uid} {{"
        f"  syntax:'<integer>';inherits:false;initial-value:0;"
        f"}}"
        f"@keyframes count{_uid}{{"
        f"  from{{--{_uid}:0}}"
        f"  to{{--{_uid}:{_cpci_int}}}"
        f"}}"
        f".ccount{_uid}{{"
        f"  animation:count{_uid} 1.4s cubic-bezier(0.16,1,0.3,1) 0.15s forwards;"
        f"  counter-reset:c var(--{_uid});"
        f"}}"
        f".ccount{_uid}::before{{"
        f"  content:counter(c);"
        f"  font-size:130px;font-weight:600;color:#FFFFFF;"
        f"  text-shadow:0 0 20px rgba(255,255,255,0.08);"
        f"  line-height:1;letter-spacing:-4px;display:block;"
        f"}}"
        f"</style>"
        f"<div style='text-align:center;padding:48px 0 40px 0;'>"
        f"<div class='cpci-label-el' style='font-size:13px;font-weight:600;"
        f"color:#94A3B8;letter-spacing:1.5px;text-transform:uppercase;margin-bottom:20px;'>"
        f"🧠&nbsp; CPCi — Cost Per Cognitive Impression &nbsp;{_TT_CPCI}</div>"
        f"<div class='cpci-score-el ccount{_uid}' style='display:inline-block;'></div>"
        f"<div class='cpci-subtext-el' style='font-size:13px;color:#CBD5E1;"
        f"font-weight:400;margin-top:16px;line-height:1.6;'>"
        f"Predicts cognitive impact before media spend</div>"
        f"<div class='cpci-context-el' style='font-size:12px;color:#94A3B8;margin-top:8px;'>"
        f"out of 100 &nbsp;·&nbsp; {uc['icon']} {use_case}</div>"
        f"</div>",
        unsafe_allow_html=True,
    )

    # ── Plain-text fallback render (always visible regardless of CSS animation) ──
    st.markdown(
        f"<div class='cpci-score' style='text-align:center;font-size:130px;"
        f"font-weight:600;color:#FFFFFF;line-height:1;letter-spacing:-4px;"
        f"text-shadow:0 0 20px rgba(255,255,255,0.08);display:none;'>"
        f"{int(round(cpci))}</div>",
        unsafe_allow_html=True,
    )

    # ── Why This Matters — emotional impact hero ──────────────────────────────
    _why_this_matters(cpci, attn, mem, val, cl, use_case)

    # ── Fallback: warn if CPCi is genuinely zero (pipeline failure) ───────────
    if cpci == 0 and (attn > 0 or mem > 0):
        st.warning(
            "⚠️ CPCi shows 0 but signals are non-zero — check signal pipeline. "
            f"(Attention={attn}, Memory={mem}, Valence={val:.2f})"
        )

    # ── Signal strip — 4 numbers ──────────────────────────────────────────────
    st.markdown(
        f"<div style='display:flex;gap:1px;background:#1F2937;border-radius:16px;"
        f"overflow:hidden;margin:0 0 40px 0;box-shadow:0 1px 3px rgba(0,0,0,0.2);'>"
        f"<div style='flex:1;background:#141B24;padding:24px 20px;text-align:center;'>"
        f"<div style='font-size:13px;color:#3B82F6;text-transform:uppercase;"
        f"letter-spacing:1px;font-weight:600;margin-bottom:8px;'>{_TT_ATTN}</div>"
        f"<div style='font-size:30px;font-weight:700;color:{a_color};line-height:1;'>{attn}</div>"
        f"<div style='font-size:13px;color:#94A3B8;margin-top:6px;'>{a_label}</div>"
        f"</div>"
        f"<div style='flex:1;background:#141B24;padding:24px 20px;text-align:center;'>"
        f"<div style='font-size:13px;color:#8B5CF6;text-transform:uppercase;"
        f"letter-spacing:1px;font-weight:600;margin-bottom:8px;'>{_TT_MEM}</div>"
        f"<div style='font-size:30px;font-weight:700;color:{m_color};line-height:1;'>{mem}</div>"
        f"<div style='font-size:13px;color:#94A3B8;margin-top:6px;'>{m_label}</div>"
        f"</div>"
        f"<div style='flex:1;background:#141B24;padding:24px 20px;text-align:center;'>"
        f"<div style='font-size:13px;color:#EC4899;text-transform:uppercase;"
        f"letter-spacing:1px;font-weight:600;margin-bottom:8px;'>{_TT_VAL}</div>"
        f"<div style='font-size:30px;font-weight:700;color:{v_color};line-height:1;'>{val:+.2f}</div>"
        f"<div style='font-size:13px;color:#94A3B8;margin-top:6px;'>{v_label}</div>"
        f"</div>"
        f"<div style='flex:1;background:#141B24;padding:24px 20px;text-align:center;'>"
        f"<div style='font-size:13px;color:#F59E0B;text-transform:uppercase;"
        f"letter-spacing:1px;font-weight:600;margin-bottom:8px;'>Cognitive Load</div>"
        f"<div style='font-size:30px;font-weight:700;color:{cl_color};line-height:1;'>{cl}</div>"
        f"<div style='font-size:13px;color:#94A3B8;margin-top:6px;'>{cl_score:.0f} / 100</div>"
        f"</div>"
        f"</div>",
        unsafe_allow_html=True,
    )

    # ── Final Verdict — standalone section ────────────────────────────────────
    verdict_txt = _final_verdict_text(cpci, attn, mem, val, cl, use_case)
    st.markdown(
        f"<div style='padding:48px 0 52px 0;border-top:1px solid #1F2937;'>"
        f"<div style='font-size:13px;font-weight:600;color:#94A3B8;"
        f"letter-spacing:1.5px;text-transform:uppercase;margin-bottom:24px;'>"
        f"⚡&nbsp; Final Verdict</div>"
        f"<div style='font-size:38px;font-weight:700;color:#FFFFFF;"
        f"line-height:1.25;letter-spacing:-0.5px;margin-bottom:28px;'>{verdict_txt}</div>"
        f"<div style='display:flex;align-items:center;gap:16px;flex-wrap:wrap;'>"
        f"<div>{badge(clabel, cstyle)}</div>"
        f"<div style='font-size:13px;color:#94A3B8;'>"
        f"<span style='color:{conf_color};font-size:10px;'>●</span>&nbsp;"
        f"Confidence: <span style='color:{conf_color};font-weight:600;'>{conf_level}</span>"
        f"</div>"
        f"</div>"
        f"</div>",
        unsafe_allow_html=True,
    )

    st.markdown("<hr style='border:none;border-top:1px solid #1F2937;margin:0 0 40px 0;'>",
                unsafe_allow_html=True)

    # ── Business Impact — CMO-facing commercial translation ───────────────────
    _business_impact(cpci, attn, mem, val, cl, use_case)

    # ── Creative Optimization Scenario — what one fix is worth ────────────────
    _optimization_scenario(cpci, attn, mem, val, cl, vf, use_case)

    # ── Collapsed details (Brief + Classification) ───────────────────────────
    with st.expander("📋  Creative Brief & Classification"):
        _creative_brief(cpci, attn, mem, val, cl, vf, use_case)
        _render_classification(cpci, attn, mem, val, cl, vf, use_case)

    st.markdown("<hr style='border:none;border-top:1px solid #1F2937;margin:32px 0;'>",
                unsafe_allow_html=True)

    # ── What This Means — full width, immediately below ───────────────────────
    _quick_read(cpci, attn, mem, val, cl, vf, use_case)

    st.markdown("<hr style='border:none;border-top:1px solid #1F2937;margin:40px 0;'>",
                unsafe_allow_html=True)

    # ── Media Implications — placement fit + strategy ─────────────────────────
    _media_implications(cpci, attn, mem, val, cl, vf, use_case)

    # ══════════════════════════════════════════════════════════════════════════
    # TIER 2 — Secondary detail  (visually quiet, below a clear separator)
    # ══════════════════════════════════════════════════════════════════════════
    st.markdown(
        "<div style='border-top:1px solid #1F2937;margin:40px 0 32px 0;"
        "display:flex;align-items:center;gap:12px;'>"
        "<span style='font-size:13px;font-weight:600;color:#94A3B8;"
        "letter-spacing:1.5px;text-transform:uppercase;white-space:nowrap;"
        "position:relative;top:-1px;padding-right:12px;background:#0B0F14;'>"
        "Detailed Analysis</span></div>",
        unsafe_allow_html=True,
    )

    detail_left, detail_right = st.columns([1, 1], gap="large")

    with detail_left:
        _section_card(
            icon  = "📈",
            title = "Strategic Implication",
            accent= "#3B82F6",
            body  = narr.get("strategic_implication", ""),
            pointers=[
                ("Use case",  use_case,      "#3B82F6"),
                ("CPCi",      f"{cpci}/100", cc),
                ("Attention", f"{attn}/100", a_color),
                ("Memory",    f"{mem}/100",  m_color),
            ],
        )

    with detail_right:
        _render_recommendations(
            body     = narr.get("recommendations", ""),
            pointers = [
                ("Priority fix",
                 "Add face"           if vf["face_count"] == 0
                 else "Boost contrast" if vf["contrast_score"] < 60
                 else "Reduce objects" if vf["object_count"] > 6
                 else "Add tagline"    if vf["text_density"] < 0.05
                 else "Trim copy",
                 "#F59E0B"),
                ("Attention gap", f"{max(0, 60-attn)} pts", a_color),
                ("Memory gap",    f"{max(0, 70-mem)} pts",  m_color),
                ("Load",          cl,                        cl_color),
            ],
        )

    # Cognitive Diagnosis — full width, most technical, last
    _cognitive_diagnosis(
        attn=attn, mem=mem, val=val, cl=cl, cl_score=cl_score,
        vf=vf,
        a_color=a_color, a_label=a_label,
        m_color=m_color, m_label=m_label,
        v_color=v_color, v_label=v_label,
        cl_color=cl_color,
    )

    # Trust indicators — how this works, limitations, confidence
    st.markdown("<hr class='divider'>", unsafe_allow_html=True)
    _render_trust_indicators(cpci, attn, mem, conf_level, conf_color)

    # Export report bar
    st.markdown("<hr class='divider'>", unsafe_allow_html=True)
    _render_export_bar(r, use_case)

    # ── CTA block ────────────────────────────────────────────────────────────
    _render_cta_block(r, use_case)


# ── Multi-creative comparison ─────────────────────────────────────────────────

import base64 as _b64

def _client_insight(narr: dict, cpci: float, attn: int, mem: int, val: float, cl: str, use_case: str) -> str:
    """
    One plain-English sentence a client can instantly understand.
    Pulled from narrative if available, otherwise generated from signals.
    No jargon. No scores. Just what it means for the campaign.
    """
    # Try narrative first
    si = narr.get("strategic_implication", "")
    if si and len(si) > 20:
        # Return first sentence only
        return si.split(".")[0].strip() + "."

    # Fallback: signal-driven
    if cpci >= 70:
        if attn >= 65:
            return "This creative will stop the scroll and encode the brand — both are rare in the same creative, and this is ready to scale."
        return "This creative will drive cognitive engagement efficiently — all signals are above the threshold required for reliable performance."
    elif cpci >= 55:
        if attn < 45:
            return "This creative will build recall among viewers who see it, but will not reliably stop cold audiences — acquisition efficiency is at risk."
        if mem < 50:
            return "This creative will attract attention but will not be remembered after a single exposure — reach without recall is wasted media."
        return "This creative is one signal fix away from scale — the foundation is strong but one dimension is suppressing the composite score."
    elif cpci >= 40:
        if cl == "High":
            return "This creative will lose viewers before the message lands — visual overload is causing the brain to abandon processing before engagement occurs."
        return "This creative will not convert efficiently at scale — the cognitive signal profile is too weak to justify full budget deployment."
    else:
        return "This creative will not perform regardless of budget or targeting — the cognitive barriers require a rebuild, not a boost in spend."


def _client_recommendation(narr: dict, cpci: float, attn: int, mem: int, val: float, cl: str, vf: dict) -> str:
    """One actionable sentence a client can take to their creative team."""
    rec = narr.get("recommendations", "")
    if rec and len(rec) > 20:
        return rec.split(".")[0].strip() + "."

    # Fallback
    if vf.get("face_count", 0) == 0 and attn < 50:
        return "Add a human face as the primary visual — it is the fastest single change to trigger an orienting response and lift emotional valence."
    if cl == "High":
        return "Remove at least half the visual elements before scaling — working memory saturation is actively blocking every other signal."
    if mem < 45:
        return "Rebuild around one dominant image and one short line of copy — single-exposure recall requires radical simplicity."
    if val < -0.05:
        return "Replace the dominant cool tones with warmer equivalents — the palette is generating subconscious avoidance that will compound across impressions."
    if attn < 45:
        return "Increase the primary subject's size and contrast — the creative needs to clear the visual salience threshold before targeting or spend can help."
    return "Test a version with a human face as the hero — it is the highest-probability change for lifting attention, emotion, and recall simultaneously."


def _show_results_client(r: dict, use_case: str) -> None:
    """Client Mode — single creative. Clean, no technical metrics."""
    s     = r["signals"]
    narr  = r.get("narrative", {})
    cpci  = r["cpci"]
    attn  = s["attention_score"]
    mem   = s["memory_score"]
    val   = s["emotional_valence"]
    cl    = s["cognitive_load"]
    vf    = r["visual_features"]

    cc      = "#22C55E" if cpci >= 70 else ("#F59E0B" if cpci >= 40 else "#EF4444")
    verdict = _final_verdict_text(cpci, attn, mem, val, cl, use_case)
    insight = _client_insight(narr, cpci, attn, mem, val, cl, use_case)
    rec     = _client_recommendation(narr, cpci, attn, mem, val, cl, vf)

    # Confidence (same logic as expert mode)
    if attn > 50 and mem > 50:
        conf_level, conf_color = "High",   "#22C55E"
    elif attn < 30 and mem < 30:
        conf_level, conf_color = "Low",    "#EF4444"
    elif abs(attn - mem) > 35 or (attn < 30 or mem < 30):
        conf_level, conf_color = "Low",    "#EF4444"
    else:
        conf_level, conf_color = "Medium", "#F59E0B"

    # Label
    if cpci >= 70:   perf_label = "Strong Performer"
    elif cpci >= 40: perf_label = "Needs Optimisation"
    else:            perf_label = "Not Ready to Scale"

    # ── Creative image hero ───────────────────────────────────────────────────
    _render_creative_hero(r.get("file_path", ""), r.get("name", ""), is_video=r.get("visual_features", {}).get("is_video", False))

    left, right = st.columns([2, 3], gap="large")

    with left:
        st.markdown(
            f"<div class='cm-card'>"
            f"<div class='cm-label'>Creative Score</div>"
            f"<div class='cm-score' style='color:#FFFFFF;'>{cpci}</div>"
            f"<div style='font-size:12px;color:#CBD5E1;margin-bottom:20px;'>"
            f"out of 100 &nbsp;·&nbsp; {use_case}</div>"
            f"<hr class='cm-divider'>"
            f"<div class='cm-label'>Performance Outlook</div>"
            f"<div style='font-size:17px;font-weight:600;color:#FFFFFF;margin-bottom:4px;'>"
            f"{perf_label}</div>"
            f"</div>",
            unsafe_allow_html=True,
        )

    with right:
        st.markdown(
            f"<div class='cm-card'>"
            f"<div class='cm-label'>Verdict</div>"
            f"<div class='cm-verdict' style='color:#FFFFFF;margin-top:28px;margin-bottom:32px;'>{verdict}</div>"
            f"<hr class='cm-divider'>"
            f"<div class='cm-label'>Key Insight</div>"
            f"<div class='cm-insight' style='margin-bottom:24px;'>{insight}</div>"
            f"<hr class='cm-divider'>"
            f"<div class='cm-label'>Recommendation</div>"
            f"<div class='cm-rec'>{rec}</div>"
            f"</div>",
            unsafe_allow_html=True,
        )

    st.markdown("<hr style='border:none;border-top:1px solid #1F2937;margin:40px 0;'>",
                unsafe_allow_html=True)

    # Why This Matters — emotional impact hero before business detail
    _why_this_matters(cpci, attn, mem, val, cl, use_case)

    # Business Impact — CMO-facing translation
    _business_impact(cpci, attn, mem, val, cl, use_case)

    # Optimization Scenario — turns the diagnosis into a growth story
    _optimization_scenario(cpci, attn, mem, val, cl, vf, use_case)

    # ── Collapsed details (Brief + Classification) ───────────────────────────
    with st.expander("📋  Creative Brief & Classification"):
        _creative_brief(cpci, attn, mem, val, cl, vf, use_case)
        _render_classification(cpci, attn, mem, val, cl, vf, use_case)

    st.markdown("<hr style='border:none;border-top:1px solid #1F2937;margin:32px 0;'>",
                unsafe_allow_html=True)

    # Media Implications
    _media_implications(cpci, attn, mem, val, cl, vf, use_case)

    st.markdown("<hr style='border:none;border-top:1px solid #1F2937;margin:40px 0;'>",
                unsafe_allow_html=True)

    # Trust indicators — how this works, limitations, confidence
    st.markdown("<hr class='divider'>", unsafe_allow_html=True)
    _render_trust_indicators(cpci, attn, mem, conf_level, conf_color)

    # Export report bar
    st.markdown("<hr class='divider'>", unsafe_allow_html=True)
    _render_export_bar(r, use_case)

    # ── CTA block ────────────────────────────────────────────────────────────
    _render_cta_block(r, use_case)


def _show_comparison_client(sorted_results: list, use_case: str) -> None:
    """Client Mode — comparison. One card per creative, verdict + insight + recommendation only."""
    n = len(sorted_results)

    st.markdown(
        "<div class='ab-header'>Cognitive Signal Engine™</div>"
        "<div style='font-size:11px;color:#64748B;font-weight:500;letter-spacing:1.4px;"
        "text-transform:uppercase;margin:-4px 0 6px 0;'>Creative Intelligence Analyzer</div>"
        f"<div class='ab-subhead'>Client Report · {use_case} · Comparing {n} Creatives</div>",
        unsafe_allow_html=True,
    )

    cols = st.columns(n, gap="medium")
    for i, (col, r) in enumerate(zip(cols, sorted_results)):
        s      = r["signals"]
        narr   = r.get("narrative", {})
        cpci   = r["cpci"]
        attn   = s["attention_score"]
        mem    = s["memory_score"]
        val    = s["emotional_valence"]
        cl     = s["cognitive_load"]
        vf     = r["visual_features"]
        is_win = (i == 0)
        cc     = "#22C55E" if cpci >= 70 else ("#F59E0B" if cpci >= 40 else "#EF4444")
        card_cls = "ab-card-winner" if is_win else "ab-card"

        verdict = _final_verdict_text(cpci, attn, mem, val, cl, use_case)
        insight = _client_insight(narr, cpci, attn, mem, val, cl, use_case)
        rec     = _client_recommendation(narr, cpci, attn, mem, val, cl, vf)

        img_src  = _img_b64(r.get("file_path", ""))
        img_html = (
            f"<img src='{img_src}' style='width:100%;height:140px;object-fit:cover;"
            f"border-radius:8px;margin-bottom:16px;display:block;' />"
            if img_src else ""
        )

        rank_html = (
            "<div class='ab-win-badge'>🏆 Recommended</div>"
            if is_win else
            f"<div class='ab-rank-badge'>Option {i+1}</div>"
        )

        col.markdown(
            f"<div class='{card_cls}'>"
            f"{rank_html}"
            f"{img_html}"
            f"<div class='ab-name'>{short_name(r['name'], 24)}</div>"
            f"<div class='cm-score' style='color:#FFFFFF;font-size:52px;'>{cpci}</div>"
            f"<div style='font-size:13px;color:#94A3B8;margin-bottom:16px;'>/ 100</div>"
            f"<div class='cm-label' style='margin-top:20px;'>Verdict</div>"
            f"<div style='font-size:18px;font-weight:700;color:#FFFFFF;line-height:1.3;"
            f"letter-spacing:-0.2px;margin-top:10px;margin-bottom:20px;'>{verdict}</div>"
            f"<div class='cm-label'>Key Insight</div>"
            f"<div style='font-size:12px;font-weight:500;color:#CBD5E1;line-height:1.6;"
            f"margin-bottom:14px;'>{insight}</div>"
            f"<div class='cm-label'>Recommendation</div>"
            f"<div style='font-size:12px;font-weight:500;color:#CBD5E1;line-height:1.6;'>"
            f"{rec}</div>"
            f"</div>",
            unsafe_allow_html=True,
        )

    # Why the winner wins (client-friendly language)
    if n > 1:
        winner    = sorted_results[0]
        runner_up = sorted_results[1]
        why       = _why_wins(winner, runner_up, use_case)
        st.markdown(
            f"<div class='ab-why' style='margin-top:24px;'>"
            f"<div class='ab-why-title'>Our Recommendation</div>"
            f"<div class='ab-why-name'>🏆 {winner['name']}</div>"
            f"<div class='ab-why-body'>{why}</div>"
            f"</div>",
            unsafe_allow_html=True,
        )

    st.markdown(
        "<p style='text-align:center;color:#4B5563;font-size:11px;letter-spacing:1.2px;"
        "text-transform:uppercase;margin-top:24px;'>Cognitive Signal Engine™</p>",
        unsafe_allow_html=True,
    )


_IMG_EXTS = {"jpg", "jpeg", "png", "gif", "webp"}
_VIDEO_EXTS_B64 = {"mp4", "mov", "avi", "webm", "m4v", "mkv"}

def _img_b64(file_path: str) -> str:
    """Return a base64 data URI for an image file (for inline HTML display).
    For video files, uses the saved thumbnail (generated during analysis).
    Returns '' if unavailable so callers show a 'No preview' fallback.
    """
    if not file_path:
        return ""
    try:
        ext = file_path.rsplit(".", 1)[-1].lower()
        # For video files, use the thumbnail PNG saved alongside it
        if ext in _VIDEO_EXTS_B64:
            thumb = file_path + "_thumb.png"
            if os.path.exists(thumb):
                with open(thumb, "rb") as f:
                    data = _b64.b64encode(f.read()).decode()
                return f"data:image/png;base64,{data}"
            return ""  # no thumbnail → show "No preview" placeholder
        # Image files — only encode known image types to avoid binary garbage
        if ext not in _IMG_EXTS:
            return ""
        with open(file_path, "rb") as f:
            data = _b64.b64encode(f.read()).decode()
        mime = {"jpg": "jpeg", "jpeg": "jpeg", "png": "png",
                "gif": "gif", "webp": "webp"}.get(ext, "jpeg")
        return f"data:image/{mime};base64,{data}"
    except Exception:
        return ""


_HERO_VIDEO_EXTS = {".mp4", ".mov", ".avi", ".webm", ".m4v"}

def _render_creative_hero(file_path: str, name: str = "", is_video: bool = False) -> None:
    """Render the uploaded creative prominently above analysis output."""
    label_html = (
        f"<div style='font-size:13px;font-weight:500;color:#94A3B8;"
        f"letter-spacing:1.5px;text-transform:uppercase;margin-bottom:12px;'>"
        f"{'🎬 Analyzed Video Creative' if is_video else 'Analyzed Creative'}</div>"
    )
    name_html = (
        f"<div style='font-size:12px;color:#94A3B8;margin-top:10px;"
        f"text-align:center;'>{name}</div>"
        if name else ""
    )

    # Check by extension if not explicitly flagged
    if not is_video:
        is_video = os.path.splitext(file_path)[1].lower() in _HERO_VIDEO_EXTS

    if is_video and os.path.exists(file_path):
        st.markdown(
            f"<div style='display:flex;flex-direction:column;align-items:center;"
            f"margin-bottom:36px;'>{label_html}</div>",
            unsafe_allow_html=True,
        )
        # Centre the video player with constrained width
        _vid_l, _vid_c, _vid_r = st.columns([1, 4, 1])
        with _vid_c:
            with open(file_path, "rb") as vf:
                st.video(vf.read())
            if name:
                st.caption(name)
        return

    img_src = _img_b64(file_path)
    if not img_src:
        return
    st.markdown(
        f"<div style='display:flex;flex-direction:column;align-items:center;"
        f"margin-bottom:36px;'>"
        f"{label_html}"
        f"<div style='min-width:60%;max-width:82%;width:fit-content;'>"
        f"<img src='{img_src}' style='"
        f"width:100%;display:block;"
        f"border-radius:14px;"
        f"border:1px solid #1F2937;"
        f"box-shadow:0 2px 16px rgba(0,0,0,0.28);"
        f"' />"
        f"{name_html}"
        f"</div>"
        f"</div>",
        unsafe_allow_html=True,
    )


def _why_wins(winner: dict, runner_up: dict, use_case: str) -> str:
    """
    Return a tight 2–3 sentence decisive summary of why the winner beats the runner-up.
    No hedging. No 'it seems'. Just facts + numbers.
    """
    ws   = winner["signals"];     rs   = runner_up["signals"]
    wvf  = winner["visual_features"]; rvf = runner_up["visual_features"]
    w_cpci = winner["cpci"];      r_cpci = runner_up["cpci"]
    gap    = round(w_cpci - r_cpci, 1)

    parts = []

    # --- CPCi gap opener
    if gap >= 20:
        parts.append(
            f"<b>{winner['name']}</b> dominates by <span class='ab-why-gap'>{gap} CPCi points</span> — "
            f"a margin that translates directly to higher conversion probability in market."
        )
    elif gap >= 10:
        parts.append(
            f"<b>{winner['name']}</b> outperforms by <span class='ab-why-gap'>{gap} CPCi points</span> — "
            f"a clear, meaningful edge across the cognitive signal stack."
        )
    else:
        parts.append(
            f"<b>{winner['name']}</b> edges ahead by <span class='ab-why-gap'>{gap} CPCi points</span> — "
            f"a narrow but consistent advantage across multiple signals."
        )

    # --- Strongest signal advantage
    attn_gap = ws["attention_score"] - rs["attention_score"]
    mem_gap  = ws["memory_score"]    - rs["memory_score"]
    val_gap  = round(ws["emotional_valence"] - rs["emotional_valence"], 2)
    load_adv = {"Low": 0, "Medium": 1, "High": 2}
    load_gap = load_adv.get(rs["cognitive_load"], 1) - load_adv.get(ws["cognitive_load"], 1)

    advantages = sorted(
        [("attn", attn_gap), ("mem", mem_gap), ("val", val_gap * 40), ("load", load_gap * 15)],
        key=lambda x: x[1], reverse=True,
    )
    top_sig, top_gap = advantages[0]

    if top_sig == "attn" and attn_gap > 3:
        reason = (
            f"The primary edge is <span class='ab-why-stat'>attention (+{attn_gap} pts)</span> — "
        )
        if wvf.get("face_count", 0) > rvf.get("face_count", 0):
            reason += "the winner's human face triggers involuntary fixation before the viewer consciously decides to look."
        elif wvf.get("contrast_score", 50) > rvf.get("contrast_score", 50) + 8:
            reason += "higher contrast pops the winner out of feed clutter where the loser blends in."
        else:
            reason += "the winner's cleaner composition gives the eye a clear landing point — the loser fragments attention across too many elements."
    elif top_sig == "mem" and mem_gap > 3:
        reason = (
            f"The decisive factor is <span class='ab-why-stat'>memory encoding (+{mem_gap} pts)</span> — "
        )
        if wvf.get("object_count", 5) < rvf.get("object_count", 5):
            reason += f"the winner's simpler layout ({wvf.get('object_count',0)} objects vs {rvf.get('object_count',0)}) encodes cleanly into long-term memory; the loser's clutter gets discarded."
        elif 0.05 <= wvf.get("text_density", 0) <= 0.25 and rvf.get("text_density", 0) > 0.25:
            reason += "optimal text density gives the brain a verbal anchor; the loser overloads the verbal channel and nothing sticks."
        else:
            reason += "a simpler visual hierarchy lets the brain commit the message — the loser asks viewers to do too much cognitive work."
    elif top_sig == "load" and load_gap > 0:
        reason = (
            f"<span class='ab-why-stat'>Lower cognitive load</span> is the key differentiator — "
            f"the winner processes in milliseconds ({ws['cognitive_load']}), "
            f"the loser demands effort ({rs['cognitive_load']}) that scrolling audiences won't give."
        )
    elif top_sig == "val" and val_gap > 0.05:
        reason = (
            f"<span class='ab-why-stat'>Emotional valence (+{val_gap:+.2f})</span> tips the balance — "
        )
        if wvf.get("face_count", 0) > rvf.get("face_count", 0):
            reason += "the winner's human face generates affiliative warmth the loser's visuals simply cannot replicate."
        else:
            reason += "the winner's color palette encodes positive affect subconsciously; the loser's palette creates mild aversion most viewers never consciously notice."
    else:
        reason = (
            "The winner holds a consistent but slim edge across all signals — "
            "no single metric dominates, but small advantages in contrast, simplicity, "
            "and color warmth compound into a measurable CPCi lead."
        )

    parts.append(reason)

    # --- Use-case verdict
    is_perf = use_case == "Performance Marketing"
    is_brand = use_case == "FMCG Branding"
    if is_perf:
        parts.append(
            f"For <b>Performance Marketing</b>, attention is the first conversion gate — "
            f"<b>{winner['name']}</b> clears it; <b>{runner_up['name']}</b> risks the scroll-past before a click can happen."
        )
    elif is_brand:
        parts.append(
            f"For <b>Brand Building</b>, memory encoding determines whether this spend compounds over time — "
            f"<b>{winner['name']}</b> encodes; <b>{runner_up['name']}</b> fades."
        )
    else:
        parts.append(
            f"Scale <b>{winner['name']}</b> first. Test <b>{runner_up['name']}</b> only after fixing its weakest signal."
        )

    return " ".join(parts)


def show_comparison(sorted_results: list, use_case: str = "Performance Marketing", client_mode: bool = False) -> None:
    """
    A/B testing decision screen:
      1. Side-by-side creative cards — winner glows green
      2. "Why this wins" decisive summary
      3. Signal bar breakdown
      4. Per-creative score explanations + download
    """
    if client_mode:
        _show_comparison_client(sorted_results, use_case)
        return
    winner = sorted_results[0]
    n      = len(sorted_results)

    # ── Header ────────────────────────────────────────────────────────────────
    st.markdown(
        "<div class='ab-header'>Cognitive Signal Engine™</div>"
        "<div style='font-size:11px;color:#64748B;font-weight:500;letter-spacing:1.4px;"
        "text-transform:uppercase;margin:-4px 0 10px 0;'>Creative Intelligence Analyzer</div>"
        f"<div class='ab-subhead'>Comparing {n} Creatives — {use_case}</div>",
        unsafe_allow_html=True,
    )

    # ── Creative cards ────────────────────────────────────────────────────────
    cols = st.columns(n, gap="small")
    for i, (col, r) in enumerate(zip(cols, sorted_results)):
        s       = r["signals"]
        cpci    = r["cpci"]
        is_win  = (i == 0)
        cc      = "#22C55E" if cpci >= 70 else ("#F59E0B" if cpci >= 40 else "#EF4444")
        card_cls = "ab-card-winner" if is_win else "ab-card"

        # Rank label / winner badge
        rank_html = (
            "<div class='ab-win-badge'>🏆 Winner</div>"
            if is_win else
            f"<div class='ab-rank-badge'>#{i+1}</div>"
        )

        # Signal colors
        a_c = "#22C55E" if s["attention_score"] >= 65 else ("#F59E0B" if s["attention_score"] >= 35 else "#EF4444")
        m_c = "#22C55E" if s["memory_score"]    >= 65 else ("#F59E0B" if s["memory_score"]    >= 40 else "#EF4444")
        v   = s["emotional_valence"]
        v_c = "#22C55E" if v > 0.1 else ("#F59E0B" if v > -0.1 else "#EF4444")
        cl  = s["cognitive_load"]
        l_c = "#22C55E" if cl == "Low" else ("#F59E0B" if cl == "Medium" else "#EF4444")

        # Verdict
        vdict_color = cc
        vdict_txt   = _final_verdict_text(cpci, s["attention_score"], s["memory_score"], v, cl, use_case)

        # Image thumbnail
        img_src  = _img_b64(r.get("file_path", ""))
        img_html = (
            f"<img src='{img_src}' style='width:100%;height:130px;object-fit:cover;"
            f"border-radius:6px;margin-bottom:14px;display:block;' />"
            if img_src else
            f"<div style='width:100%;height:80px;background:#0B0F14;border-radius:6px;"
            f"margin-bottom:14px;display:flex;align-items:center;justify-content:center;"
            f"font-size:13px;color:#94A3B8;'>No preview</div>"
        )

        col.markdown(
            f"<div class='{card_cls}'>"
            f"{rank_html}"
            f"{img_html}"
            f"<div class='ab-name'>{short_name(r['name'], 22)}</div>"
            f"<div class='ab-score' style='color:#FFFFFF;'>{cpci}</div>"
            f"<div class='ab-score-sub'>CPCi Score / 100</div>"
            f"<div class='ab-sig-grid'>"
            f"<div class='ab-sig-cell'><div class='ab-sig-label'>Attention</div>"
            f"<div class='ab-sig-val' style='color:{a_c};'>{s['attention_score']}</div></div>"
            f"<div class='ab-sig-cell'><div class='ab-sig-label'>Memory</div>"
            f"<div class='ab-sig-val' style='color:{m_c};'>{s['memory_score']}</div></div>"
            f"<div class='ab-sig-cell'><div class='ab-sig-label'>Valence</div>"
            f"<div class='ab-sig-val' style='color:{v_c};'>{v:+.2f}</div></div>"
            f"<div class='ab-sig-cell'><div class='ab-sig-label'>Load</div>"
            f"<div class='ab-sig-val' style='color:{l_c};font-size:13px;'>{cl}</div></div>"
            f"</div>"
            f"<div class='ab-verdict' style='border-color:{vdict_color};color:{vdict_color};'>"
            f"{vdict_txt}</div>"
            f"</div>",
            unsafe_allow_html=True,
        )

    # ── CPCi score bars ────────────────────────────────────────────────────────
    max_cpci = sorted_results[0]["cpci"] if sorted_results[0]["cpci"] > 0 else 1
    bars_html = ""
    for r in sorted_results:
        pct   = int(r["cpci"] / max_cpci * 100)
        bc    = "#22C55E" if r["cpci"] >= 70 else ("#F59E0B" if r["cpci"] >= 40 else "#EF4444")
        nm    = short_name(r["name"], 18)
        bars_html += (
            f"<div class='ab-bar-row'>"
            f"<div class='ab-bar-label'>{nm}</div>"
            f"<div class='ab-bar-track'>"
            f"<div class='ab-bar-fill' style='width:{pct}%;background:{bc};'></div></div>"
            f"<div class='ab-bar-val' style='color:{bc};'>{r['cpci']}</div>"
            f"</div>"
        )

    st.markdown(
        f"<div style='background:#0B0F14;border-radius:8px;padding:16px 20px;margin:20px 0 0 0;'>"
        f"<div style='font-size:9px;font-weight:800;color:#94A3B8;letter-spacing:2px;"
        f"text-transform:uppercase;margin-bottom:12px;'>CPCi Score Comparison</div>"
        f"{bars_html}"
        f"</div>",
        unsafe_allow_html=True,
    )

    # ── Why this wins ─────────────────────────────────────────────────────────
    runner_up = sorted_results[1] if n > 1 else sorted_results[0]
    why_body  = _why_wins(winner, runner_up, use_case)

    st.markdown(
        f"<div class='ab-why'>"
        f"<div class='ab-why-title'>Why this wins</div>"
        f"<div class='ab-why-name'>🏆 {winner['name']}</div>"
        f"<div class='ab-why-body'>{why_body}</div>"
        f"</div>",
        unsafe_allow_html=True,
    )

    # ── Per-creative detail ────────────────────────────────────────────────────
    st.markdown("<hr class='divider'>", unsafe_allow_html=True)
    show_score_explanations(sorted_results)

    # ── Trust indicators (use winner's signals for confidence) ────────────────
    w_attn = winner["signals"]["attention_score"]
    w_mem  = winner["signals"]["memory_score"]
    w_cpci = winner["cpci"]
    if w_attn > 50 and w_mem > 50:
        w_conf_level, w_conf_color = "High",   "#22C55E"
    elif w_attn < 30 and w_mem < 30:
        w_conf_level, w_conf_color = "Low",    "#EF4444"
    elif abs(w_attn - w_mem) > 35 or (w_attn < 30 or w_mem < 30):
        w_conf_level, w_conf_color = "Low",    "#EF4444"
    else:
        w_conf_level, w_conf_color = "Medium", "#F59E0B"
    _render_trust_indicators(w_cpci, w_attn, w_mem, w_conf_level, w_conf_color)

    # ── Download ──────────────────────────────────────────────────────────────
    st.markdown("<hr class='divider'>", unsafe_allow_html=True)
    export_all = [{k: v for k, v in r.items() if k not in ("reasoning", "file_path")} for r in sorted_results]
    st.download_button(
        "⬇️  Export Comparison Report",
        json.dumps(export_all, indent=2),
        "creative_comparison_report.json",
        "application/json",
        use_container_width=True,
    )
    st.markdown(
        "<p style='text-align:center;color:#94A3B8;font-size:13px;margin-top:14px;'>"
        "Powered by OpenCV · Tesseract OCR · Cognitive Signal Engine™ · "
        "Color Psychology · CLT (Sweller) · Dual-Coding (Paivio)</p>",
        unsafe_allow_html=True,
    )


def _final_verdict_text(cpci, attn, mem, val, cl, use_case) -> str:
    """Return the verdict string only (used by both show_results and comparison cards)."""
    attn_weak = attn < 45;  mem_weak  = mem  < 50
    load_high = cl == "High"; val_neg = val < -0.05
    attn_str  = attn >= 65;  mem_str  = mem  >= 65
    is_perf   = use_case == "Performance Marketing"
    is_brand  = use_case == "FMCG Branding"
    if cpci >= 70:
        if attn_str and mem_str:  return "Ready to scale — strong on every signal."
        if attn_str and mem_weak: return "Strong for attention, weak for recall — add retargeting."
        if mem_str and attn_weak: return "Powerful recall, weak hook — fix opening frame."
        if load_high:             return "High scores, high clutter — simplify before scaling."
        return "Ready to scale — above threshold on all signals."
    elif cpci >= 55:
        if attn_weak and not mem_weak: return "Strong recall, weak acquisition — not cold-audience ready."
        if mem_weak and not attn_weak: return "Stops the scroll but won't be remembered."
        if load_high:                  return "High potential — strip cognitive load first."
        if val_neg and is_brand:       return "Brand risk — emotional tone must improve before spend."
        return "High potential — optimize attention before scaling."
    elif cpci >= 40:
        if attn_weak and mem_weak: return "NOT ready for scale."
        if load_high and attn_weak: return "Too cluttered to convert — rebuild."
        if val_neg: return "Negative signal — will hurt brand at scale."
        if is_perf and mem_weak: return "Will not convert — memory encoding too weak."
        return "Borderline — fix one signal before scaling."
    else:
        if val_neg and load_high: return "Do not run — will damage brand perception."
        if attn < 25: return "NOT ready for scale — will be ignored."
        if mem < 30:  return "Forgettable at any spend level — rebuild."
        return "NOT ready for scale."


MEDALS = ["🥇", "🥈", "🥉", "④", "⑤"]

def show_comparison_table(results: list) -> None:
    """
    Section 1: Comparison table — one row per creative, all key metrics side by side.
    Sorted in upload order (not ranked — ranking is Section 2).
    """
    st.markdown("<hr class='divider'>", unsafe_allow_html=True)
    st.markdown("<div class='section-header main-header'>📊 Side-by-Side Comparison</div>", unsafe_allow_html=True)

    # Header row
    cols = st.columns([2.5, 1, 1, 1, 1.2, 1.2])
    headers = ["Creative", "Attention", "Memory", "Valence", "Cog. Load", "CPCi (Cognitive Impact)"]
    for col, h in zip(cols, headers):
        col.markdown(f"<div style='color:#CBD5E1;font-size:12px;font-weight:700;"
                     f"text-transform:uppercase;letter-spacing:1px;padding:8px 4px;"
                     f"border-bottom:2px solid #1F2937;'>{h}</div>", unsafe_allow_html=True)

    # Data rows
    for i, r in enumerate(results):
        s    = r["signals"]
        bg   = "#141B24" if i % 2 == 0 else "#141B24"
        cols = st.columns([2.5, 1, 1, 1, 1.2, 1.2])

        # Creative name + color palette
        swatch_html = color_swatches(r["visual_features"]["dominant_colors"])
        cols[0].markdown(
            f"<div style='background:{bg};padding:10px 4px;border-top:1px solid #1F2937;'>"
            f"<strong style='color:#FFFFFF;font-size:13px;'>{short_name(r['name'])}</strong><br>"
            f"<span style='font-size:13px;'>{swatch_html}</span></div>",
            unsafe_allow_html=True,
        )

        # Attention
        ac = "#22C55E" if s["attention_score"] >= 70 else ("#F59E0B" if s["attention_score"] >= 40 else "#EF4444")
        cols[1].markdown(
            f"<div style='background:{bg};padding:10px 4px;border-top:1px solid #1F2937;"
            f"text-align:center;font-size:18px;font-weight:700;color:{ac};'>"
            f"{s['attention_score']}</div>", unsafe_allow_html=True,
        )

        # Memory
        mc = "#22C55E" if s["memory_score"] >= 70 else ("#F59E0B" if s["memory_score"] >= 40 else "#EF4444")
        cols[2].markdown(
            f"<div style='background:{bg};padding:10px 4px;border-top:1px solid #1F2937;"
            f"text-align:center;font-size:18px;font-weight:700;color:{mc};'>"
            f"{s['memory_score']}</div>", unsafe_allow_html=True,
        )

        # Valence
        val = s["emotional_valence"]
        vc  = "#22C55E" if val > 0.1 else ("#F59E0B" if val > -0.1 else "#EF4444")
        cols[3].markdown(
            f"<div style='background:{bg};padding:10px 4px;border-top:1px solid #1F2937;"
            f"text-align:center;font-size:18px;font-weight:700;color:{vc};'>"
            f"{val:+.2f}</div>", unsafe_allow_html=True,
        )

        # Cognitive load
        cl    = s["cognitive_load"]
        cl_c  = "#22C55E" if cl == "Low" else ("#F59E0B" if cl == "Medium" else "#EF4444")
        cols[4].markdown(
            f"<div style='background:{bg};padding:10px 4px;border-top:1px solid #1F2937;"
            f"text-align:center;font-size:14px;font-weight:700;color:{cl_c};'>"
            f"{cl}</div>", unsafe_allow_html=True,
        )

        # CPCi
        cc = cpci_color(r["cpci"])
        cols[5].markdown(
            f"<div style='background:{bg};padding:10px 4px;border-top:1px solid #1F2937;"
            f"text-align:center;font-size:22px;font-weight:900;color:#93C5FD;'>"
            f"{r['cpci']}</div>", unsafe_allow_html=True,
        )


def show_ranking(sorted_results: list) -> None:
    """
    Section 2: Ranked list sorted by CPCi descending.
    Winner gets gold highlight + medal. Progress bars show relative CPCi.
    """
    st.markdown("<hr class='divider'>", unsafe_allow_html=True)
    st.markdown("<div class='section-header main-header'>🏆 Performance Ranking</div>", unsafe_allow_html=True)

    max_cpci = sorted_results[0]["cpci"] if sorted_results else 100

    for i, r in enumerate(sorted_results):
        medal      = MEDALS[i] if i < len(MEDALS) else f"#{i+1}"
        card_class = "rank-card rank-winner" if i == 0 else ("rank-card rank-second" if i == 1 else "rank-card")
        cc         = cpci_color(r["cpci"])
        s          = r["signals"]

        st.markdown(f"""
        <div class='{card_class}'>
          <div style='display:flex;align-items:center;justify-content:space-between;'>
            <div>
              <span style='font-size:22px;margin-right:10px;'>{medal}</span>
              <strong style='font-size:17px;color:#FFFFFF;'>{r['name']}</strong>
              {"<span style='margin-left:10px;font-size:13px;background:#141B24;color:#F59E0B;"
               "border:1px solid #F59E0B;border-radius:10px;padding:2px 10px;font-weight:700;"
               "text-transform:uppercase;letter-spacing:1px;'>Winner</span>" if i == 0 else ""}
            </div>
            <div style='font-size:32px;font-weight:900;color:#FFFFFF;'>{r['cpci']}<span style='font-size:14px;color:#94A3B8;'>/100</span></div>
          </div>
          <div style='margin-top:10px;display:flex;gap:20px;font-size:12px;color:#CBD5E1;'>
            <span>🎯 Attention: <strong style='color:#FFFFFF;'>{s['attention_score']}</strong></span>
            <span>🧠 Memory: <strong style='color:#FFFFFF;'>{s['memory_score']}</strong></span>
            <span>❤️ Valence: <strong style='color:#FFFFFF;'>{s['emotional_valence']:+.2f}</strong></span>
            <span>⚙️ Load: <strong style='color:#FFFFFF;'>{s['cognitive_load']}</strong></span>
          </div>
        </div>""", unsafe_allow_html=True)

        # Progress bar showing CPCi relative to the winner
        bar_pct = r["cpci"] / max_cpci if max_cpci > 0 else 0
        st.progress(bar_pct)


def show_winner_explanation(sorted_results: list) -> None:
    """
    Section 3: Explain why the top creative beats each competitor, metric by metric.
    Identifies specific weaknesses in each losing creative.
    """
    st.markdown("<hr class='divider'>", unsafe_allow_html=True)
    st.markdown("<div class='section-header main-header'>🧬 Why the Winner Wins</div>", unsafe_allow_html=True)

    winner  = sorted_results[0]
    losers  = sorted_results[1:]
    ws      = winner["signals"]
    wvf     = winner["visual_features"]
    w_attn  = ws["attention_score"]
    w_mem   = ws["memory_score"]
    w_val   = ws["emotional_valence"]
    w_cl    = ws["cognitive_load"]
    w_cpci  = winner["cpci"]

    load_rank = {"Low": 0, "Medium": 1, "High": 2}

    # Winner summary box
    st.markdown(f"""
    <div class='winner-box'>
      <div style='font-size:13px;color:#94A3B8;text-transform:uppercase;letter-spacing:1px;font-weight:700;margin-bottom:6px;'>
        🥇 Winner
      </div>
      <div style='font-size:22px;font-weight:900;color:#F59E0B;'>{winner['name']}</div>
      <div style='font-size:14px;color:#CBD5E1;margin-top:8px;'>
        CPCi <strong style='color:#F59E0B;font-size:20px;'>{w_cpci}</strong> —
        Attention <strong>{w_attn}</strong> ·
        Memory <strong>{w_mem}</strong> ·
        Valence <strong>{w_val:+.2f}</strong> ·
        Load <strong>{w_cl}</strong>
      </div>
      <div style='font-size:13px;color:#94A3B8;margin-top:12px;padding-top:10px;
                  border-top:1px solid #141B24;line-height:1.6;'>
        This creative wins because it delivers the highest cognitive impact (CPCi) —
        meaning it is most likely to convert attention into memory and action.
      </div>
    </div>""", unsafe_allow_html=True)

    # Per-competitor breakdown
    for loser in losers:
        ls     = loser["signals"]
        lvf    = loser["visual_features"]
        l_attn = ls["attention_score"]
        l_mem  = ls["memory_score"]
        l_val  = ls["emotional_valence"]
        l_cl   = ls["cognitive_load"]
        l_cpci = loser["cpci"]
        cpci_gap = round(w_cpci - l_cpci, 1)

        # Build advantage list
        advantages = []
        weaknesses = []

        if w_attn > l_attn + 5:
            diff = w_attn - l_attn
            advantages.append(
                f"<strong>+{diff} pts Attention</strong> ({w_attn} vs {l_attn}) — "
                + ("Winner has faces creating involuntary fixation. " if wvf["face_count"] > lvf["face_count"] else "")
                + (f"Winner contrast {wvf['contrast_score']:.0f} vs {lvf['contrast_score']:.0f} — "
                   f"higher contrast stops the scroll. " if wvf["contrast_score"] > lvf["contrast_score"] + 10 else "")
                + (f"Loser has {lvf['object_count']} objects vs {wvf['object_count']} — clutter fragments attention." if lvf["object_count"] > wvf["object_count"] + 2 else "")
            )
        elif l_attn > w_attn + 5:
            weaknesses.append(f"Loser has +{l_attn - w_attn} pts higher attention ({l_attn} vs {w_attn}) but weaker on other metrics.")

        if w_mem > l_mem + 5:
            diff = w_mem - l_mem
            advantages.append(
                f"<strong>+{diff} pts Memory Encoding</strong> ({w_mem} vs {l_mem}) — "
                + (f"Winner has {wvf['object_count']} objects vs {lvf['object_count']} — simpler composition encodes more cleanly. " if wvf["object_count"] < lvf["object_count"] else "")
                + ("Winner text density is in optimal 5–25% range. " if 0.05 <= wvf["text_density"] <= 0.25 and not (0.05 <= lvf["text_density"] <= 0.25) else "")
                + (f"Loser text density {lvf['text_density']*100:.0f}% overloads the verbal channel." if lvf["text_density"] > 0.25 else "")
            )

        if w_val > l_val + 0.1:
            diff = round(w_val - l_val, 2)
            advantages.append(
                f"<strong>+{diff} Emotional Valence</strong> ({w_val:+.2f} vs {l_val:+.2f}) — "
                + ("Winner has warmer color palette driving positive affect. " if w_val > 0 else "")
                + ("Loser colors are dark/desaturated, encoding mild aversion. " if l_val < -0.05 else "")
                + (f"Winner has {wvf['face_count']} face(s) adding affiliative warmth. " if wvf["face_count"] > lvf["face_count"] else "")
            )

        if load_rank[w_cl] < load_rank[l_cl]:
            advantages.append(
                f"<strong>Lower Cognitive Load</strong> ({w_cl} vs {l_cl}) — "
                f"winner processes faster in scroll environments. "
                f"Loser has {lvf['object_count']} objects + {lvf['text_density']*100:.0f}% text "
                f"saturating working memory capacity."
            )

        # If winner leads on everything, add a clean sweep note
        if not advantages:
            advantages.append(
                f"<strong>Narrow but consistent margin</strong> — winner outperforms by {cpci_gap} CPCi points "
                f"across a combination of small improvements in contrast, simplicity, and color warmth."
            )

        adv_html = "".join(f"<li style='margin:6px 0;'>{a}</li>" for a in advantages)
        weak_html = (
            "<br><em style='color:#CBD5E1;font-size:12px;'>Note: " + " ".join(weaknesses) + "</em>"
            if weaknesses else ""
        )

        st.markdown(f"""
        <div class='insight-box'>
          <strong style='color:#3B82F6;'>Winner vs {loser['name']}</strong>
          <span style='float:right;color:#EF4444;font-size:13px;font-weight:700;'>
            Loser CPCi: {l_cpci} &nbsp;|&nbsp; Gap: -{cpci_gap}
          </span><br>
          <ul style='margin:10px 0 4px 0;padding-left:20px;'>
            {adv_html}
          </ul>
          {weak_html}
        </div>""", unsafe_allow_html=True)

    # Actionable summary
    st.markdown("<hr class='divider'>", unsafe_allow_html=True)
    export_all = [
        {k: v for k, v in r.items() if k != "reasoning"}
        for r in sorted_results
    ]
    st.download_button(
        "⬇️  Export Comparison Report",
        json.dumps(export_all, indent=2),
        "creative_comparison_report.json",
        "application/json",
        use_container_width=True,
    )
    st.markdown(
        "<p style='text-align:center;color:#94A3B8;font-size:12px;margin-top:16px;'>"
        "Powered by OpenCV · Tesseract OCR · Cognitive Signal Engine™ · "
        "Color Psychology · Cognitive Load Theory (Sweller) · Dual-Coding Theory (Paivio)"
        "</p>",
        unsafe_allow_html=True,
    )


# ── Per-creative score explanation ───────────────────────────────────────────

def _hex_to_hsv(hex_color: str):
    """Parse a hex color string to HSV tuple. Returns (h_degrees, s, v)."""
    hex_color = hex_color.lstrip("#")
    r, g, b   = (int(hex_color[i:i+2], 16) / 255.0 for i in (0, 2, 4))
    h, s, v   = colorsys.rgb_to_hsv(r, g, b)
    return h * 360, s, v


def generate_score_explanation(r: dict) -> dict:
    """
    Generate rule-based bullet point explanations for each metric score.

    Every bullet references a REAL detected value — no generic copy.
    Returns a dict with keys: "attention", "memory", "valence", "load",
    each mapping to a list of (icon, text) tuples.

    Icons: ✅ positive driver  ⚠️ negative driver  ➡️ neutral/moderate
    """
    s          = r["signals"]
    vf         = r["visual_features"]
    obj_count  = vf["object_count"]
    face_count = vf["face_count"]
    contrast   = vf["contrast_score"]
    text_dens  = vf["text_density"]
    colors     = vf["dominant_colors"]
    cl         = s["cognitive_load"]
    reasoning  = r.get("reasoning") or {}
    cl_score   = (
        reasoning["load"]["composite"]
        if isinstance(reasoning, dict) and "load" in reasoning
        else s.get("cognitive_load_score", 50)
    )

    bullets = {"attention": [], "memory": [], "valence": [], "load": []}

    # ── ATTENTION ─────────────────────────────────────────────────────────────

    # Object clutter
    if obj_count > 7:
        penalty = min(30, (obj_count - 3) * 2)
        bullets["attention"].append(
            ("⚠️", f"{obj_count} objects detected — high visual clutter fragments attention "
                   f"(clutter penalty: -{penalty} pts; threshold is >7 objects)")
        )
    elif obj_count <= 3:
        bullets["attention"].append(
            ("✅", f"{obj_count} object(s) — clean composition, full attentional focus on primary element")
        )
    else:
        bullets["attention"].append(
            ("➡️", f"{obj_count} objects — moderate clutter, small attention penalty applied")
        )

    # Face presence
    if face_count > 0:
        boost = 22 if face_count == 1 else (36 if face_count == 2 else 44)
        bullets["attention"].append(
            ("✅", f"{face_count} human face(s) detected — faces trigger involuntary fixation "
                   f"via the fusiform face area (+{boost} pts attention boost)")
        )
    else:
        bullets["attention"].append(
            ("⚠️", "No faces detected — missing the strongest natural attention trigger in ad creative. "
                   "Adding one face can add +22 pts to attention score")
        )

    # Contrast
    if contrast < 40:
        bullets["attention"].append(
            ("⚠️", f"Contrast score {contrast:.0f}/100 — low contrast means the creative blends "
                   f"into the feed; pre-attentive pop is weak")
        )
    elif contrast >= 70:
        bullets["attention"].append(
            ("✅", f"Contrast score {contrast:.0f}/100 — high contrast creates a pre-attentive "
                   f"visual pop that stops scrolling before conscious processing")
        )
    else:
        bullets["attention"].append(
            ("➡️", f"Contrast score {contrast:.0f}/100 — adequate but not scroll-stopping; "
                   f"increasing to 70+ can meaningfully improve attention")
        )

    # ── MEMORY ────────────────────────────────────────────────────────────────

    # Simplicity
    if obj_count <= 4:
        bullets["memory"].append(
            ("✅", f"{obj_count} object(s) — simple composition gives the brain a clear primary "
                   f"element to bind into long-term memory (picture superiority effect)")
        )
    elif obj_count > 7:
        bullets["memory"].append(
            ("⚠️", f"{obj_count} objects — too many competing elements fragment the hippocampal "
                   f"binding event; each extra object above 2 costs ~6 memory points")
        )
    else:
        bullets["memory"].append(
            ("➡️", f"{obj_count} objects — acceptable complexity, but simplifying to ≤4 "
                   f"would improve memory encoding")
        )

    # Text density
    td_pct = text_dens * 100
    if 0.05 <= text_dens <= 0.25:
        bullets["memory"].append(
            ("✅", f"Text coverage {td_pct:.1f}% — optimal zone (5–25%). "
                   f"Enough text to provide a verbal anchor for brand recall "
                   f"without overloading the visual channel (dual-coding theory)")
        )
    elif text_dens < 0.05:
        bullets["memory"].append(
            ("⚠️", f"Text coverage {td_pct:.1f}% — too little text. "
                   f"Without a verbal anchor (tagline, brand name), "
                   f"memory relies entirely on the visual trace which fades faster")
        )
    elif text_dens > 0.30:
        bullets["memory"].append(
            ("⚠️", f"Text coverage {td_pct:.1f}% — excess copy overwhelms the visual channel. "
                   f"Above 25% density, memory encoding declines as the brain "
                   f"shifts from image processing to slow serial text reading")
        )
    else:
        bullets["memory"].append(
            ("➡️", f"Text coverage {td_pct:.1f}% — approaching overload zone (>25%). "
                   f"Trim copy to stay in the optimal range")
        )

    # ── EMOTIONAL VALENCE ─────────────────────────────────────────────────────

    # Face warmth
    if face_count > 0:
        face_boost = min(0.30, face_count * 0.25)
        bullets["valence"].append(
            ("✅", f"{face_count} face(s) — trigger affiliative warmth and emotional mirroring. "
                   f"Valence boost from faces: +{face_boost:.2f} "
                   f"(capped at +0.30 to prevent compounding)")
        )
    else:
        bullets["valence"].append(
            ("⚠️", "No faces detected — faces are the strongest single valence driver. "
                   "Adding a person with visible emotion can shift valence by +0.20 to +0.30")
        )

    # Color temperature analysis
    warm_found = []
    dark_found = []
    for hex_c in colors:
        try:
            h_deg, sat, val_hsv = _hex_to_hsv(hex_c)
            if val_hsv < 0.20:
                dark_found.append(hex_c)
            elif sat > 0.30 and (h_deg < 70 or h_deg >= 345):
                warm_found.append(hex_c)
        except Exception:
            pass

    if warm_found:
        bullets["valence"].append(
            ("✅", f"Warm tones in palette: {', '.join(warm_found[:2])} — "
                   f"red/orange/yellow hues sit in the high-arousal, positive-valence "
                   f"quadrant of Russell's circumplex model")
        )
    else:
        bullets["valence"].append(
            ("➡️", "No dominant warm tones — palette is cool or neutral. "
                   "Cool colors (blue, grey) are emotionally safe but do not actively "
                   "drive positive valence or purchase intent")
        )

    if dark_found:
        bullets["valence"].append(
            ("⚠️", f"Dark colors present: {', '.join(dark_found[:2])} — "
                   f"HSV value < 20% encodes mild negative affect regardless of hue. "
                   f"Dark palettes reduce perceived warmth and energy")
        )

    # ── COGNITIVE LOAD ────────────────────────────────────────────────────────

    high_obj  = obj_count > 7
    high_text = text_dens > 0.30
    low_obj   = obj_count <= 4
    low_text  = text_dens < 0.10

    if high_obj and high_text:
        bullets["load"].append(
            ("⚠️", f"{obj_count} objects AND {td_pct:.0f}% text — both the visual and "
                   f"linguistic processing channels are simultaneously overloaded. "
                   f"Working memory (Miller's 7±2 limit) will saturate; viewer disengages")
        )
    elif high_obj:
        bullets["load"].append(
            ("⚠️", f"{obj_count} objects — visual channel overloaded (>7 exceeds Miller's 7±2 rule). "
                   f"Reduce to ≤7 objects to bring visual complexity into acceptable range")
        )
    elif high_text:
        bullets["load"].append(
            ("⚠️", f"{td_pct:.0f}% text — linguistic channel overloaded (>30% density). "
                   f"Compress copy to one primary message; every extra word costs processing capacity")
        )
    elif low_obj and low_text:
        bullets["load"].append(
            ("✅", f"{obj_count} objects + {td_pct:.0f}% text — both channels are well within "
                   f"processing capacity. Effortless comprehension; ideal for mobile scroll environments")
        )
    else:
        bullets["load"].append(
            ("➡️", f"{obj_count} objects + {td_pct:.0f}% text — moderate load, "
                   f"acceptable for most placements but may reduce performance in "
                   f"passive scroll contexts (Reels, TikTok)")
        )

    icon = "✅" if cl == "Low" else ("➡️" if cl == "Medium" else "⚠️")
    bullets["load"].append(
        (icon, f"Composite load score: {cl_score:.0f}/100 → classified as {cl} load")
    )

    return bullets


def show_score_explanations(results: list) -> None:
    """
    Render 'Why this score?' bullet points for each creative as collapsible expanders.
    Each expander shows 4 metric sections with real-value bullet points.
    """
    st.markdown("<hr class='divider'>", unsafe_allow_html=True)
    st.markdown(
        "<div class='section-header main-header'>📝 Why Each Creative Scored This Way</div>",
        unsafe_allow_html=True,
    )
    st.markdown(
        "<p style='color:#CBD5E1;font-size:13px;margin-bottom:16px;'>"
        "Click any creative to expand its full score explanation — every bullet "
        "references an actual detected value.</p>",
        unsafe_allow_html=True,
    )

    METRIC_CONFIG = [
        ("attention", "🎯 Attention Score",    "#3B82F6"),
        ("memory",    "🧠 Memory Encoding",    "#22C55E"),
        ("valence",   "❤️ Emotional Valence",  "#CBD5E1"),
        ("load",      "⚙️ Cognitive Load",     "#F59E0B"),
    ]

    ICON_COLOR = {"✅": "#22C55E", "⚠️": "#EF4444", "➡️": "#F59E0B"}

    for r in results:
        s        = r["signals"]
        cpci_val = r["cpci"]
        label    = (
            "Strong Performer" if cpci_val >= 70
            else ("Average Performer" if cpci_val >= 45 else "Needs Improvement")
        )
        cc = cpci_color(cpci_val)

        expander_title = (
            f"{r['name']}  —  CPCi {cpci_val}  |  "
            f"Attn {s['attention_score']}  |  Mem {s['memory_score']}  |  "
            f"Val {s['emotional_valence']:+.2f}  |  Load {s['cognitive_load']}"
        )

        with st.expander(expander_title, expanded=False):
            explanation = generate_score_explanation(r)

            # Two-column grid: left = Attention + Memory, right = Valence + Load
            left_col, right_col = st.columns(2)

            for col, keys in [(left_col, ["attention", "memory"]),
                              (right_col, ["valence", "load"])]:
                with col:
                    for key, title, accent in METRIC_CONFIG:
                        if key not in keys:
                            continue
                        score_val = (
                            s["attention_score"] if key == "attention"
                            else s["memory_score"] if key == "memory"
                            else s["emotional_valence"] if key == "valence"
                            else s["cognitive_load"]
                        )
                        score_str = (
                            f"{score_val}/100"    if key in ("attention", "memory")
                            else f"{score_val:+.3f}" if key == "valence"
                            else str(score_val)
                        )

                        st.markdown(
                            f"<div style='font-size:13px;font-weight:700;color:{accent};"
                            f"text-transform:uppercase;letter-spacing:1px;"
                            f"border-bottom:1px solid #1F2937;padding-bottom:4px;"
                            f"margin-bottom:8px;margin-top:12px;'>"
                            f"{title} &nbsp;<span style='color:#FFFFFF;font-size:17px;"
                            f"font-weight:900;'>{score_str}</span></div>",
                            unsafe_allow_html=True,
                        )

                        for icon, text in explanation[key]:
                            icon_color = ICON_COLOR.get(icon, "#FFFFFF")
                            st.markdown(
                                f"<div style='display:flex;gap:8px;align-items:flex-start;"
                                f"margin-bottom:8px;font-size:13px;line-height:1.5;'>"
                                f"<span style='color:{icon_color};font-size:14px;"
                                f"flex-shrink:0;margin-top:1px;'>{icon}</span>"
                                f"<span style='color:#CBD5E1;'>{text}</span></div>",
                                unsafe_allow_html=True,
                            )


# ── Main UI ───────────────────────────────────────────────────────────────────

# ── Landing screen ────────────────────────────────────────────────────────────
if "app_entered" not in st.session_state:
    st.session_state["app_entered"] = False

if not st.session_state["app_entered"]:
    st.markdown("""
    <style>
    /* Landing — hide Streamlit chrome */
    [data-testid="stHeader"] { display: none !important; }

    .landing-wrap {
      display: flex;
      flex-direction: column;
      align-items: center;
      justify-content: center;
      min-height: 88vh;
      text-align: center;
      padding: 0 24px;
      max-width: 720px;
      margin: 0 auto;
    }
    /* H1 — primary platform name */
    .landing-h1 {
      font-size: 52px;
      font-weight: 800;
      color: #FFFFFF;
      line-height: 1.1;
      letter-spacing: -0.02em;
      margin-bottom: 12px;
    }
    .landing-h1 span { color: #60A5FA; }

    /* H2 — module subtitle */
    .landing-h2 {
      font-size: 16px;
      font-weight: 500;
      color: #94A3B8;
      letter-spacing: 0.12em;
      text-transform: uppercase;
      margin-bottom: 28px;
    }

    /* Legacy aliases kept for compat */
    .landing-eyebrow { display: none; }
    .landing-title   { display: none; }

    .landing-tagline {
      font-size: 18px;
      font-weight: 400;
      color: #CBD5E1;
      line-height: 1.75;
      max-width: 520px;
      margin: 0 auto 24px auto;
    }
    .landing-benefits {
      display: flex;
      gap: 32px;
      justify-content: center;
      flex-wrap: wrap;
      margin-bottom: 48px;
    }
    .landing-benefit {
      display: flex;
      align-items: flex-start;
      gap: 12px;
      text-align: left;
      max-width: 180px;
    }
    .landing-benefit-dot {
      width: 6px;
      height: 6px;
      border-radius: 50%;
      background: #3B82F6;
      flex-shrink: 0;
      margin-top: 8px;
    }
    .landing-benefit-text {
      font-size: 14px;
      font-weight: 500;
      color: #CBD5E1;
      line-height: 1.55;
    }
    .landing-divider {
      width: 60px;
      height: 2px;
      background: #3B82F6;
      margin: 24px auto;
      border-radius: 2px;
    }
    .landing-footer {
      font-size: 13px;
      color: #CBD5E1;
      margin-top: 48px;
      letter-spacing: 0.5px;
    }
    </style>

    <div class="landing-wrap">

      <div class="landing-h1">
        Cognitive Signal<br><span>Engine™</span>
      </div>

      <div class="landing-h2">Creative Intelligence Analyzer</div>

      <div class="landing-tagline">
        Measure how creatives perform in the brain<br>— before media spend.
      </div>

      <div class="landing-divider"></div>

      <div class="landing-benefits">
        <div class="landing-benefit">
          <div class="landing-benefit-dot"></div>
          <div class="landing-benefit-text">Pre-bid creative scoring</div>
        </div>
        <div class="landing-benefit">
          <div class="landing-benefit-dot"></div>
          <div class="landing-benefit-text">Reduce wasted media spend</div>
        </div>
        <div class="landing-benefit">
          <div class="landing-benefit-dot"></div>
          <div class="landing-benefit-text">Faster creative decisions</div>
        </div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Scientific Foundation panel (separate call — avoids Streamlit style-block issue) ──
    st.markdown(
        "<div style='max-width:780px;margin:0 auto 40px auto;padding:0 24px;'>"
        "<div style='border:1px solid #1F2937;border-radius:14px;overflow:hidden;'>"

        # Header bar
        "<div style='background:#0D1117;padding:14px 24px;border-bottom:1px solid #1F2937;"
        "display:flex;align-items:center;gap:10px;'>"
        "<div style='width:8px;height:8px;border-radius:50%;background:#EF4444;'></div>"
        "<div style='font-size:13px;font-weight:700;letter-spacing:2px;text-transform:uppercase;"
        "color:#CBD5E1;'>Scientific Foundation — Built on Meta TRIBE v2 · Extended by CPCi</div>"
        "</div>"

        # Two columns via flex
        "<div style='display:flex;'>"

        # LEFT — Meta TRIBE v2
        "<div style='flex:1;padding:22px 24px;'>"
        "<div style='display:flex;align-items:center;gap:10px;margin-bottom:12px;'>"
        "<div style='background:#1D4ED8;border-radius:5px;padding:3px 10px;"
        "font-size:13px;font-weight:800;color:#fff;letter-spacing:0.5px;'>META FAIR</div>"
        "<div style='font-size:13px;font-weight:700;color:#93C5FD;'>TRIBE v2 · 2025</div>"
        "</div>"
        "<div style='font-size:12px;color:#CBD5E1;line-height:1.7;margin-bottom:12px;'>"
        "Tri-modal Brain Imaging Encoding model. Predicts fMRI brain activity from video, audio &amp; text"
        " — trained on <strong style='color:#93C5FD;'>720 subjects</strong>, "
        "<strong style='color:#93C5FD;'>1,117 hours</strong> of brain scans. "
        "Ranked <strong style='color:#93C5FD;'>#1</strong> at Algonauts 2025 (263 teams)."
        "</div>"
        "<div style='font-size:13px;color:#CBD5E1;line-height:1.8;'>"
        "<span style='color:#3B82F6;'>→</span>&nbsp;<strong style='color:#94A3B8;'>Neuroscience weights:</strong> which brain regions respond to contrast, faces, text<br>"
        "<span style='color:#3B82F6;'>→</span>&nbsp;<strong style='color:#94A3B8;'>Signal thresholds:</strong> V1–V7 visual cortex, FaceBody area, Language regions, Limbic<br>"
        "<span style='color:#3B82F6;'>→</span>&nbsp;<strong style='color:#94A3B8;'>Brain data:</strong> face attention fires in &lt;13ms · contrast drives pre-attentive salience"
        "</div>"
        "<div style='margin-top:12px;font-size:13px;color:#94A3B8;'>"
        "d'Ascoli, Rapin, Benchetrit, King et al. · Meta FAIR 2025 · "
        "<span style='color:#1D4ED8;'>CC BY-NC 4.0</span>"
        "</div>"
        "</div>"

        # Divider
        "<div style='width:1px;background:#1F2937;'></div>"

        # RIGHT — CPCi
        "<div style='flex:1;padding:22px 24px;'>"
        "<div style='display:flex;align-items:center;gap:10px;margin-bottom:12px;'>"
        "<div style='background:#1D4ED8;border-radius:5px;padding:3px 10px;"
        "font-size:13px;font-weight:800;color:#fff;letter-spacing:0.5px;'>COGNITIVE SIGNAL ENGINE™</div>"
        "<div style='font-size:13px;font-weight:700;color:#93C5FD;'>CPCi · Anil Pandit</div>"
        "</div>"
        "<div style='font-size:12px;color:#CBD5E1;line-height:1.7;margin-bottom:12px;'>"
        "Cost Per Cognitive Impression. Translates TRIBE v2 brain-encoding patterns into a "
        "<strong style='color:#FCA5A5;'>single 0–100 ad effectiveness score</strong> — "
        "with use-case weights, load penalties, and a full narrative strategy engine."
        "</div>"
        "<div style='font-size:13px;color:#CBD5E1;line-height:1.8;'>"
        "<span style='color:#EF4444;'>→</span>&nbsp;<strong style='color:#94A3B8;'>CPCi formula:</strong> weighted combination of Attention + Memory + Valence signals<br>"
        "<span style='color:#EF4444;'>→</span>&nbsp;<strong style='color:#94A3B8;'>Use-case weights:</strong> FMCG / Performance / Retail tuned for media context<br>"
        "<span style='color:#EF4444;'>→</span>&nbsp;<strong style='color:#94A3B8;'>Narrative engine:</strong> converts scores into actionable creative strategy"
        "</div>"
        "<div style='margin-top:12px;font-size:13px;color:#94A3B8;'>"
        "Anil Pandit · Cognitive Signal Engine™ · AI and Data Leader · "
        "<span style='color:#EF4444;'>Proprietary framework</span>"
        "</div>"
        "</div>"

        "</div>"  # end flex row
        "</div>"  # end border card
        "</div>", # end max-width wrapper
        unsafe_allow_html=True,
    )

    _lnd_gap, _lnd_cta, _lnd_demo, _lnd_gap2 = st.columns([3, 2, 2, 3], gap="small")

    with _lnd_cta:
        if st.button("Start Analysis", type="primary", use_container_width=True):
            st.session_state["app_entered"] = True
            st.rerun()

    with _lnd_demo:
        if st.button("View Demo", use_container_width=True):
            st.session_state["app_entered"] = True
            st.session_state["demo_mode"]   = True
            st.rerun()

    st.markdown(
        "<div style='margin-top:64px;padding:20px 0 14px 0;border-top:1px solid #1F2937;"
        "text-align:center;'>"
        "<div style='font-size:14px;font-weight:800;color:#FFFFFF;letter-spacing:-0.2px;'>"
        "Cognitive Signal Engine™</div>"
        "<div style='font-size:11px;color:#4B5563;margin-top:4px;letter-spacing:1.4px;"
        "text-transform:uppercase;'>Creative Intelligence Analyzer</div>"
        "<div style='font-size:11px;color:#374151;margin-top:10px;'>"
        "© Anil Pandit &nbsp;·&nbsp; ADVantage Insights"
        "</div>"
        "</div>",
        unsafe_allow_html=True,
    )
    st.stop()

_BRAIN_SVG = (
    "<svg width='22' height='22' viewBox='0 0 24 24' fill='none' xmlns='http://www.w3.org/2000/svg'>"
    "<path d='M12 2C9.5 2 7.5 3.5 7 5.5C5.5 5.5 4 7 4 8.5C4 9.5 4.5 10.3 5.2 10.8C4.5 11.5 4 12.5 4 13.5"
    "C4 15.5 5.5 17 7.2 17.3C7.6 18.8 9 20 10.5 20H13.5C15 20 16.4 18.8 16.8 17.3C18.5 17 20 15.5 20 13.5"
    "C20 12.5 19.5 11.5 18.8 10.8C19.5 10.3 20 9.5 20 8.5C20 7 18.5 5.5 17 5.5C16.5 3.5 14.5 2 12 2Z'"
    " fill='#3B82F6' opacity='0.9'/>"
    "<circle cx='9' cy='10' r='1.2' fill='#0B0F14'/>"
    "<circle cx='15' cy='10' r='1.2' fill='#0B0F14'/>"
    "<path d='M9 14 Q12 16 15 14' stroke='#0B0F14' stroke-width='1.2' fill='none' stroke-linecap='round'/>"
    "</svg>"
)
# ── App header ───────────────────────────────────────────────────────────────
_hdr_left, _hdr_right = st.columns([3, 2], gap="large")

with _hdr_left:
    st.markdown(
        "<div style='padding:28px 0 0 0;display:flex;align-items:center;gap:14px;'>"
        "<div style='width:38px;height:38px;background:#e8201a;border-radius:8px;"
        "display:flex;align-items:center;justify-content:center;flex-shrink:0;'>"
        "<svg width='26' height='26' viewBox='0 0 90 90' xmlns='http://www.w3.org/2000/svg'>"
        "<circle cx='45' cy='28' r='24' fill='#5b8ef5' opacity='0.93'/>"
        "<circle cx='28' cy='60' r='24' fill='#3a72e8' opacity='0.88'/>"
        "<circle cx='62' cy='60' r='24' fill='#5ab0f7' opacity='0.88'/>"
        "</svg></div>"
        "<div>"
        "<div style='font-size:18px;font-weight:800;color:#FFFFFF;letter-spacing:-0.3px;line-height:1.15;'>"
        "Cognitive Signal Engine™</div>"
        "<div style='font-size:11px;color:#64748B;margin-top:3px;font-weight:500;"
        "letter-spacing:1.4px;text-transform:uppercase;'>Creative Intelligence Analyzer</div>"
        "</div>"
        "</div>",
        unsafe_allow_html=True,
    )

with _hdr_right:
    # Guard: if demo mode exit was requested programmatically (from inside _render_demo_mode),
    # apply it HERE — before any widget with key="demo_mode" is instantiated, so Streamlit
    # doesn't raise StreamlitAPIException about modifying widget-bound state.
    if st.session_state.pop("_exit_demo", False):
        st.session_state["demo_mode"] = False
        st.session_state["demo_step"] = 1
        st.session_state["demo_creative"] = 0

    st.markdown("<div style='padding:24px 0 0 0;'>", unsafe_allow_html=True)
    _tog_a, _tog_b = st.columns(2, gap="small")
    with _tog_a:
        _expert_on = st.toggle(
            "Expert Mode",
            value=st.session_state.get("expert_mode", False),
            key="expert_mode",
        )
    with _tog_b:
        _demo_on = st.toggle(
            "Demo Mode",
            value=st.session_state.get("demo_mode", False),
            key="demo_mode",
        )
    st.markdown("</div>", unsafe_allow_html=True)

st.markdown(
    "<hr style='border:none;border-top:1px solid #1F2937;margin:0 0 16px 0;'>",
    unsafe_allow_html=True,
)

# ── Top-level tabs ────────────────────────────────────────────────────────────
tab_analyzer, tab_science, tab_glossary = st.tabs([
    "🔬  Analyzer",
    "🧠  Science & Methodology",
    "📖  Glossary",
])

with tab_science:
    show_science_tab()

with tab_glossary:
    show_glossary_tab()

with tab_analyzer:
    # ── Framework ─────────────────────────────────────────────────────────────
    with st.expander("📐  Framework", expanded=False):
        fw_left, fw_right = st.columns([3, 2], gap="large")

        with fw_left:
            st.markdown(
                "<div style='padding:4px 0 8px 0;'>"
                "<div style='font-size:13px;font-weight:700;color:#3B82F6;"
                "letter-spacing:2px;text-transform:uppercase;margin-bottom:14px;'>"
                "Cognitive Signal Engine (CSE)</div>"

                "<div style='font-size:14px;color:#CBD5E1;line-height:1.8;'>"
                "This system is built on the <strong style='color:#FFFFFF;'>Cognitive Signal Engine</strong> — "
                "a proprietary framework that models how advertising stimuli are processed "
                "across four dimensions:"
                "</div>"

                "<div style='margin:20px 0;border-left:2px solid #1F2937;padding-left:20px;"
                "display:flex;flex-direction:column;gap:14px;'>"

                "<div style='display:flex;align-items:baseline;gap:12px;'>"
                "<span style='font-size:13px;font-weight:700;color:#3B82F6;"
                "letter-spacing:1.5px;text-transform:uppercase;flex-shrink:0;min-width:20px;'>01</span>"
                "<div><span style='font-size:14px;font-weight:700;color:#FFFFFF;'>Attention</span>"
                "<span style='font-size:13px;color:#94A3B8;'>&nbsp;·&nbsp;Does it get noticed?</span></div>"
                "</div>"

                "<div style='display:flex;align-items:baseline;gap:12px;'>"
                "<span style='font-size:13px;font-weight:700;color:#8B5CF6;"
                "letter-spacing:1.5px;text-transform:uppercase;flex-shrink:0;min-width:20px;'>02</span>"
                "<div><span style='font-size:14px;font-weight:700;color:#FFFFFF;'>Memory</span>"
                "<span style='font-size:13px;color:#94A3B8;'>&nbsp;·&nbsp;Is it encoded?</span></div>"
                "</div>"

                "<div style='display:flex;align-items:baseline;gap:12px;'>"
                "<span style='font-size:13px;font-weight:700;color:#EC4899;"
                "letter-spacing:1.5px;text-transform:uppercase;flex-shrink:0;min-width:20px;'>03</span>"
                "<div><span style='font-size:14px;font-weight:700;color:#FFFFFF;'>Emotion</span>"
                "<span style='font-size:13px;color:#94A3B8;'>&nbsp;·&nbsp;Does it create affinity?</span></div>"
                "</div>"

                "<div style='display:flex;align-items:baseline;gap:12px;'>"
                "<span style='font-size:13px;font-weight:700;color:#F59E0B;"
                "letter-spacing:1.5px;text-transform:uppercase;flex-shrink:0;min-width:20px;'>04</span>"
                "<div><span style='font-size:14px;font-weight:700;color:#FFFFFF;'>Cognitive Load</span>"
                "<span style='font-size:13px;color:#94A3B8;'>&nbsp;·&nbsp;Is it easy to process?</span></div>"
                "</div>"

                "</div>"

                "<div style='font-size:13px;color:#CBD5E1;line-height:1.7;'>"
                "These signals are combined into "
                "<strong style='color:#FFFFFF;'>CPCi</strong> — a unified measure of "
                "cognitive effectiveness before media spend."
                "</div>"
                "</div>",
                unsafe_allow_html=True,
            )

        with fw_right:
            st.markdown(
                "<div style='background:#141B24;border:1px solid #1F2937;border-radius:16px;"
                "padding:20px 22px;height:100%;'>"

                "<div style='font-size:13px;font-weight:700;color:#94A3B8;"
                "letter-spacing:2px;text-transform:uppercase;margin-bottom:16px;'>"
                "Signal Architecture</div>"

                "<div style='display:flex;flex-direction:column;gap:10px;'>"

                "<div style='display:flex;justify-content:space-between;align-items:center;"
                "border-bottom:1px solid #1F2937;padding-bottom:10px;'>"
                "<span style='font-size:12px;color:#CBD5E1;'>Layer</span>"
                "<span style='font-size:12px;color:#CBD5E1;'>Source</span>"
                "</div>"

                "<div style='display:flex;justify-content:space-between;align-items:center;'>"
                "<span style='font-size:13px;color:#3B82F6;font-weight:600;'>Attention</span>"
                "<span style='font-size:13px;color:#94A3B8;'>Contrast · Face · Clutter</span>"
                "</div>"

                "<div style='display:flex;justify-content:space-between;align-items:center;'>"
                "<span style='font-size:13px;color:#8B5CF6;font-weight:600;'>Memory</span>"
                "<span style='font-size:13px;color:#94A3B8;'>Composition · Text density</span>"
                "</div>"

                "<div style='display:flex;justify-content:space-between;align-items:center;'>"
                "<span style='font-size:13px;color:#EC4899;font-weight:600;'>Emotion</span>"
                "<span style='font-size:13px;color:#94A3B8;'>Colour · Face warmth</span>"
                "</div>"

                "<div style='display:flex;justify-content:space-between;align-items:center;'>"
                "<span style='font-size:13px;color:#F59E0B;font-weight:600;'>Cognitive Load</span>"
                "<span style='font-size:13px;color:#94A3B8;'>Objects · Entropy · Text %</span>"
                "</div>"

                "<div style='border-top:1px solid #1F2937;margin-top:10px;padding-top:12px;'>"
                "<div style='font-size:13px;font-weight:700;color:#3B82F6;"
                "letter-spacing:1.5px;text-transform:uppercase;margin-bottom:4px;'>Output</div>"
                "<div style='font-size:13px;font-weight:700;color:#FFFFFF;'>CPCi Score</div>"
                "<div style='font-size:13px;color:#94A3B8;margin-top:2px;'>"
                "Weighted composite · 0–100 · Use-case calibrated</div>"
                "</div>"

                "</div>"
                "</div>",
                unsafe_allow_html=True,
            )

    st.markdown("<hr class='divider'>", unsafe_allow_html=True)

    # ── Why CPCi? ─────────────────────────────────────────────────────────────
    with st.expander("⚡  Why CPCi?", expanded=False):
        st.markdown(
            "<div style='padding:8px 0 4px 0;'>"

            # — Full form + one-liner ——————————————————————————————————————————
            "<div style='margin-bottom:28px;'>"
            "<span style='font-size:13px;font-weight:700;color:#3B82F6;"
            "letter-spacing:2px;text-transform:uppercase;'>CPCi</span>"
            "<span style='font-size:13px;color:#94A3B8;'>&nbsp;·&nbsp;"
            "Cost Per Cognitive Impression</span>"
            "<div style='font-size:13px;color:#CBD5E1;margin-top:6px;line-height:1.6;'>"
            "A single, pre-spend measure of how effectively an ad creative engages "
            "the human brain — across attention, memory, emotion, and processing ease."
            "</div>"
            "</div>"

            # — The contrast argument ——————————————————————————————————————————
            "<div style='display:grid;grid-template-columns:1fr 1fr;gap:16px;margin-bottom:28px;'>"

            "<div style='background:#141B24;border:1px solid #1F2937;border-radius:16px;"
            "padding:18px 20px;'>"
            "<div style='font-size:13px;font-weight:700;color:#EF4444;"
            "letter-spacing:2px;text-transform:uppercase;margin-bottom:10px;'>"
            "Traditional metrics</div>"
            "<div style='font-size:13px;color:#CBD5E1;line-height:1.7;'>"
            "CTR, CVR, ROAS — measured <em>after</em> exposure.<br>"
            "They tell you what happened. They do not tell you <em>why</em>.<br><br>"
            "By the time you have the data, budget has already been spent on creative "
            "that failed at the first cognitive gate."
            "</div>"
            "</div>"

            "<div style='background:#141B24;border:1px solid #3B82F6;border-radius:16px;"
            "padding:18px 20px;'>"
            "<div style='font-size:13px;font-weight:700;color:#3B82F6;"
            "letter-spacing:2px;text-transform:uppercase;margin-bottom:10px;'>"
            "CPCi</div>"
            "<div style='font-size:13px;color:#CBD5E1;line-height:1.7;'>"
            "Measured <em>before</em> exposure — at the visual signal layer.<br>"
            "Models what happens in the first 1–3 seconds: does the brain notice, "
            "encode, and respond positively?<br><br>"
            "Evaluate creative effectiveness before media spend. "
            "Kill weak creatives before they consume budget."
            "</div>"
            "</div>"

            "</div>"

            # — Benchmark rationale ————————————————————————————————————————————
            "<div style='border-top:1px solid #1F2937;padding-top:20px;margin-bottom:4px;'>"
            "<div style='font-size:13px;font-weight:700;color:#94A3B8;"
            "letter-spacing:2px;text-transform:uppercase;margin-bottom:14px;'>"
            "How the benchmarks were set</div>"

            "<div style='display:grid;grid-template-columns:repeat(3,1fr);gap:12px;'>"

            "<div style='background:#141B24;border:1px solid #1F2937;border-radius:16px;"
            "padding:14px 16px;border-top:2px solid #22C55E;'>"
            "<div style='font-size:22px;font-weight:800;color:#22C55E;"
            "letter-spacing:-1px;margin-bottom:4px;'>70+</div>"
            "<div style='font-size:13px;font-weight:700;color:#22C55E;"
            "letter-spacing:1px;text-transform:uppercase;margin-bottom:8px;'>"
            "Scale-ready</div>"
            "<div style='font-size:12px;color:#94A3B8;line-height:1.6;'>"
            "All three signals (attention, memory, emotion) clear their individual "
            "thresholds. Cognitive Load Theory suggests low-load creative at these "
            "levels is processed fluently — the necessary precondition for action."
            "</div>"
            "</div>"

            "<div style='background:#141B24;border:1px solid #1F2937;border-radius:16px;"
            "padding:14px 16px;border-top:2px solid #F59E0B;'>"
            "<div style='font-size:22px;font-weight:800;color:#F59E0B;"
            "letter-spacing:-1px;margin-bottom:4px;'>40–69</div>"
            "<div style='font-size:13px;font-weight:700;color:#F59E0B;"
            "letter-spacing:1px;text-transform:uppercase;margin-bottom:8px;'>"
            "Optimise first</div>"
            "<div style='font-size:12px;color:#94A3B8;line-height:1.6;'>"
            "Mixed signal profile — at least one dimension is below the encoding "
            "floor. Paivio's Dual-Coding research indicates partial encoding leads "
            "to poor recall and weak brand linkage over time."
            "</div>"
            "</div>"

            "<div style='background:#141B24;border:1px solid #1F2937;border-radius:16px;"
            "padding:14px 16px;border-top:2px solid #EF4444;'>"
            "<div style='font-size:22px;font-weight:800;color:#EF4444;"
            "letter-spacing:-1px;margin-bottom:4px;'>&lt;40</div>"
            "<div style='font-size:13px;font-weight:700;color:#EF4444;"
            "letter-spacing:1px;text-transform:uppercase;margin-bottom:8px;'>"
            "Do not scale</div>"
            "<div style='font-size:12px;color:#94A3B8;line-height:1.6;'>"
            "Insufficient cognitive engagement. Consistent with attention research "
            "showing creatives below contrast and salience minimums are processed "
            "at near-zero depth — spend is wasted before the message lands."
            "</div>"
            "</div>"

            "</div>"
            "</div>"

            "</div>",
            unsafe_allow_html=True,
        )

    st.markdown("<hr class='divider'>", unsafe_allow_html=True)

    # ── What is CPCi? ─────────────────────────────────────────────────────────
    with st.expander("💡 What is CPCi? — Click to learn how scoring works", expanded=False):
        col_a, col_b, col_c, col_d = st.columns(4)
        col_a.markdown("""
        <div style='background:#141B24;border-radius:10px;padding:16px;height:100%;'>
          <div style='color:#3B82F6;font-size:13px;font-weight:700;text-transform:uppercase;
                      letter-spacing:1px;margin-bottom:8px;'>What is CPCi?</div>
          <div style='color:#CBD5E1;font-size:13px;line-height:1.7;'>
            <strong style='color:#FFFFFF;'>CPCi</strong> is produced by the Cognitive Signal Engine —
            a system that models how advertising creative is processed by the human brain,
            before any click or conversion occurs.<br><br>
            Higher CPCi = stronger cognitive processing = better media efficiency.
          </div>
        </div>""", unsafe_allow_html=True)
        col_b.markdown("""
        <div style='background:#141B24;border-radius:10px;padding:16px;height:100%;'>
          <div style='color:#22C55E;font-size:22px;margin-bottom:4px;'>🎯</div>
          <div style='color:#22C55E;font-size:13px;font-weight:700;text-transform:uppercase;
                      letter-spacing:1px;margin-bottom:8px;'>Attention</div>
          <div style='color:#CBD5E1;font-size:13px;line-height:1.7;'>
            Does it stop the scroll?<br><br>
            Driven by visual contrast, face presence, and how clean or cluttered the image is.
          </div>
        </div>""", unsafe_allow_html=True)
        col_c.markdown("""
        <div style='background:#141B24;border-radius:10px;padding:16px;height:100%;'>
          <div style='color:#CBD5E1;font-size:22px;margin-bottom:4px;'>🧠</div>
          <div style='color:#CBD5E1;font-size:13px;font-weight:700;text-transform:uppercase;
                      letter-spacing:1px;margin-bottom:8px;'>Memory</div>
          <div style='color:#CBD5E1;font-size:13px;line-height:1.7;'>
            Will the brand be remembered?<br><br>
            Simple compositions and balanced text leave a stronger memory trace.
          </div>
        </div>""", unsafe_allow_html=True)
        col_d.markdown("""
        <div style='background:#141B24;border-radius:10px;padding:16px;height:100%;'>
          <div style='color:#F59E0B;font-size:22px;margin-bottom:4px;'>❤️</div>
          <div style='color:#F59E0B;font-size:13px;font-weight:700;text-transform:uppercase;
                      letter-spacing:1px;margin-bottom:8px;'>Emotional Valence</div>
          <div style='color:#CBD5E1;font-size:13px;line-height:1.7;'>
            Does it create a positive response?<br><br>
            Warm colours and human faces drive positive valence.
          </div>
        </div>""", unsafe_allow_html=True)
        st.markdown("""
        <div style='background:#141B24;border-radius:8px;padding:12px 16px;margin-top:12px;
                    border-left:3px solid #3B82F6;font-size:13px;color:#CBD5E1;line-height:1.6;'>
          <strong style='color:#FFFFFF;'>How weights work:</strong>
          CPCi formula is adjusted per use case — brand campaign prioritises memory,
          performance campaign prioritises attention. See the 🧠 Science tab for the full formula.
        </div>""", unsafe_allow_html=True)

    # ── Resolve mode flags from header toggles ───────────────────────────────
    client_mode = not st.session_state.get("expert_mode", False)
    demo_mode   = st.session_state.get("demo_mode", False)

    st.markdown("<hr class='divider'>", unsafe_allow_html=True)

    # ── Use-case selector ─────────────────────────────────────────────────────
    uc_col, info_col = st.columns([1, 2])
    with uc_col:
        selected_uc = st.selectbox(
            "🎯 Select Use Case",
            options=list(USE_CASES.keys()),
            index=1,
            help="Changes CPCi weights to match your campaign objective",
        )
    uc_cfg     = USE_CASES[selected_uc]
    uc_weights = uc_cfg["weights"]
    with info_col:
        w = uc_weights
        penalty_note = " · <span style='color:#EF4444;font-weight:700;'>Load Penalty Active</span>" if uc_cfg["load_penalty"] else ""
        st.markdown(
            f"<div style='background:#141B24;border-left:2px solid #3B82F6;"
            f"border-radius:8px;padding:16px 20px;margin-top:4px;'>"
            f"<div style='font-size:14px;font-weight:700;color:#FFFFFF;margin-bottom:6px;'>"
            f"{uc_cfg['icon']} {selected_uc}"
            f"<span style='color:#94A3B8;font-size:12px;font-weight:500;margin-left:10px;'>{uc_cfg['description']}</span></div>"
            f"<div style='font-size:12px;color:#CBD5E1;margin-bottom:8px;'>"
            f"Weights: "
            f"<strong style='color:#3B82F6;'>Attention {int(w['attention']*100)}%</strong> · "
            f"<strong style='color:#22C55E;'>Memory {int(w['memory']*100)}%</strong> · "
            f"<strong style='color:#a78bfa;'>Emotion {int(w['emotion']*100)}%</strong>"
            f"{penalty_note}</div>"
            f"<div style='color:#CBD5E1;font-size:12px;line-height:1.65;'>"
            f"{uc_cfg['rationale']}</div></div>",
            unsafe_allow_html=True,
        )

    st.markdown("<hr class='divider'>", unsafe_allow_html=True)

    # ── Demo Mode — full guided walkthrough experience ────────────────────────
    if demo_mode:
        _render_demo_mode(client_mode)
        st.stop()

    # ── CTA "Analyze Your Creative" — clear cached results and scroll up ────
    if st.session_state.pop("_cta_new_analysis", False):
        for _k in ("cached_results", "analyzed_files", "all_results", "elapsed"):
            st.session_state.pop(_k, None)
        st.toast("Upload a new creative below ↓", icon="🧠")

    # ── File uploader ─────────────────────────────────────────────────────────
    uploaded_files = st.file_uploader(
        "📁 Upload ad creatives — Images (JPG, PNG) or Videos (MP4, MOV, AVI, WEBM) · 1 file for analysis, 2–5 for comparison",
        type=["jpg", "jpeg", "png", "mp4", "mov", "avi", "webm", "m4v"],
        accept_multiple_files=True,
    )

    if uploaded_files:
        n = len(uploaded_files)
        if n > 5:
            st.error("⚠️ Maximum 5 creatives at once. Please remove some files and try again.")
            st.stop()

        st.markdown(
            f"<div style='color:#CBD5E1;font-size:13px;margin-bottom:8px;'>"
            f"{'1 creative — full analysis mode' if n == 1 else f'{n} creatives — comparison mode'}"
            f" &nbsp;·&nbsp; <span style='color:{uc_cfg['accent']};'>{uc_cfg['icon']} {selected_uc}</span>"
            f"</div>",
            unsafe_allow_html=True,
        )

        _VIDEO_EXTS_UI = {".mp4", ".mov", ".avi", ".webm", ".m4v"}
        thumb_cols = st.columns(min(n, 5))
        for col, uf in zip(thumb_cols, uploaded_files):
            ext = os.path.splitext(uf.name)[1].lower()
            if ext in _VIDEO_EXTS_UI:
                col.video(uf)
                col.caption(f"🎬 {short_name(uf.name, 22)}")
            else:
                col.image(uf, caption=short_name(uf.name, 22), use_container_width=True)

        st.markdown("<br>", unsafe_allow_html=True)
        _has_video = any(
            os.path.splitext(uf.name)[1].lower() in _VIDEO_EXTS_UI
            for uf in uploaded_files
        )
        _est_low  = n * 3 if _has_video else n
        _est_high = n * 8 if _has_video else n * 2
        analyze_btn = st.button(
            f"🔬 {'Analyze Creative' if n == 1 else f'Compare {n} Creatives'} [{selected_uc}]",
            type="primary",
            use_container_width=True,
        )
        st.markdown(
            f"<p style='color:#CBD5E1;font-size:12px;text-align:center;margin-top:4px;'>"
            f"~{_est_low}–{_est_high}s · No GPU · Attn {int(uc_weights['attention']*100)}% · "
            f"Mem {int(uc_weights['memory']*100)}% · "
            f"Emo {int(uc_weights['emotion']*100)}%"
            f"{'&nbsp;·&nbsp; 🎬 Video: 6-frame sampling' if _has_video else ''}</p>",
            unsafe_allow_html=True,
        )

        # ── Restore cached results ────────────────────────────────────────────
        cached_names  = st.session_state.get("analyzed_files", [])
        cached_uc     = st.session_state.get("analyzed_uc", "")
        current_names = [uf.name for uf in uploaded_files]
        cache_valid   = (
            "all_results" in st.session_state
            and cached_names == current_names
            and cached_uc == selected_uc
        )
        if cache_valid:
            cached        = st.session_state["all_results"]
            cached_sorted = sorted(cached, key=lambda x: x["cpci"], reverse=True)
            if n == 1:
                show_results(cached[0], st.session_state.get("elapsed"), selected_uc, client_mode)
            else:
                show_comparison(cached_sorted, selected_uc, client_mode)

        # ── Run analysis ──────────────────────────────────────────────────────
        if analyze_btn:
            all_results  = []
            t0           = time.time()
            progress_bar = st.progress(0, text="Starting analysis…")
            for i, uf in enumerate(uploaded_files):
                progress_bar.progress(
                    i / n,
                    text=f"Analyzing {uf.name} ({i+1}/{n})…",
                )
                with st.spinner(f"🔬 Processing {short_name(uf.name)}…"):
                    result = run_pipeline(
                        uf,
                        weights=uc_weights,
                        apply_load_penalty=uc_cfg["load_penalty"],
                        use_case=selected_uc,
                    )
                    all_results.append(result)

            progress_bar.progress(1.0, text="✅ All creatives analyzed.")
            elapsed = time.time() - t0
            st.session_state["all_results"]    = all_results
            st.session_state["analyzed_files"] = [uf.name for uf in uploaded_files]
            st.session_state["analyzed_uc"]    = selected_uc
            st.session_state["elapsed"]        = elapsed
            st.markdown(
                f"<div class='timer-box'>⚡ {n} creative{'s' if n > 1 else ''} analyzed "
                f"in {elapsed:.2f}s &nbsp;·&nbsp; {uc_cfg['icon']} {selected_uc}</div>",
                unsafe_allow_html=True,
            )
            if n == 1:
                show_results(all_results[0], elapsed, selected_uc, client_mode)
            else:
                sorted_results = sorted(all_results, key=lambda x: x["cpci"], reverse=True)
                show_comparison(sorted_results, selected_uc, client_mode)


# ── Global footer (visible on all tabs) ───────────────────────────────────────
_FOOTER_SVG = (
    "<svg width='30' height='30' viewBox='0 0 90 90' xmlns='http://www.w3.org/2000/svg'>"
    "<circle cx='45' cy='28' r='24' fill='#5b8ef5' opacity='0.93'/>"
    "<circle cx='28' cy='60' r='24' fill='#3a72e8' opacity='0.88'/>"
    "<circle cx='62' cy='60' r='24' fill='#5ab0f7' opacity='0.88'/>"
    "</svg>"
)
st.markdown(
    f"<div style='margin-top:72px;padding:20px 0 16px 0;border-top:1px solid #1F2937;'>"
    f"<div style='display:flex;align-items:center;justify-content:space-between;flex-wrap:wrap;gap:12px;'>"

    # Left — logo + name
    f"<div style='display:flex;align-items:center;gap:12px;'>"
    f"<div style='width:36px;height:36px;background:#e8201a;border-radius:8px;"
    f"display:flex;align-items:center;justify-content:center;flex-shrink:0;'>{_FOOTER_SVG}</div>"
    f"<div>"
    f"<div style='font-size:15px;font-weight:800;color:#FFFFFF;letter-spacing:-0.2px;line-height:1.15;'>Cognitive Signal Engine™</div>"
    f"<div style='font-size:10px;color:#4B5563;margin-top:3px;letter-spacing:1.4px;text-transform:uppercase;'>Creative Intelligence Analyzer</div>"
    f"</div>"
    f"</div>"

    # Centre — single clean line
    f"<div style='font-size:12px;color:#64748B;text-align:center;'>"
    f"© Anil Pandit &nbsp;·&nbsp; ADVantage Insights &nbsp;·&nbsp; CPCi — Cost Per Cognitive Impression"
    f"</div>"

    # Right — tagline only
    f"<div style='font-size:12px;color:#64748B;text-align:right;max-width:280px;line-height:1.5;'>"
    f"Neuroscience → Cognitive signals → Media decisions."
    f"</div>"

    f"</div>"
    f"</div>",
    unsafe_allow_html=True,
)
