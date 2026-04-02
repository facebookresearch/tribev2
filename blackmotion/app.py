import os
import sys
import json
import pathlib
import base64
import datetime
import functools

# ============================================================
# WARDOGZ NEURO-UGC v2 — SETUP SYSTÈME
# ============================================================

print("🔧 Setup Wardogz Brain Scanner v2...")

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
    print("✅ Auth Meta OK")
else:
    print("⚠️ HF_TOKEN manquant")

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
    print(f"⚡ GPU : {torch.cuda.get_device_name(0)}")

# ============================================================
# CHARGEMENT MODÈLE
# ============================================================

CACHE_DIR      = "./cache/tribev2"
HISTORY_FILE   = "./scan_history.json"
THUMBNAILS_DIR = "./thumbnails"
pathlib.Path(THUMBNAILS_DIR).mkdir(parents=True, exist_ok=True)

try:
    from tribev2 import TribeModel
    model = TribeModel.from_pretrained("facebook/tribev2", cache_folder=CACHE_DIR)
    status_msg = "✅ Brain Scanner v2 Opérationnel"
    print(status_msg)
except Exception as e:
    model = None
    status_msg = f"❌ Erreur chargement : {str(e)}"
    print(status_msg)

# ============================================================
# MATRICE NEUROLOGIQUE — DÉFINITION DES ROI
#
# Tribe v2 prédit sur ~20 484 vertices corticaux (fsaverage5).
# On mappe ces vertices sur des zones fonctionnelles via l'atlas
# Destrieux (nilearn). 6 métriques clés pour l'audit pub :
#
#  Mémorisation     → parahippocampal + cingulaire postérieur
#  Plaisir/Dopamine → orbitofrontal + vmPFC
#  Stress/Anxiété   → insula + cingulaire antérieur
#  Attention visuelle→ cortex occipital + cuneus
#  Résonance sociale → TPJ + sulcus temporal supérieur
#  Impact audio     → gyrus de Heschl + temporal supérieur
# ============================================================

# Labels Destrieux → chaque ROI est une liste de patterns à matcher
ROI_DESTRIEUX = {
    "memorisation": [
        "parahippocampal", "cingul-Mid-Post", "cingul-Post"
    ],
    "plaisir_dopamine": [
        "Orbital", "orbital", "rectus", "frontomargin"
    ],
    "stress_anxiete": [
        "insular", "circular_insula", "cingul-Ant"
    ],
    "attention_visuelle": [
        "occipital", "cuneus", "calcarine", "Pole_occipital"
    ],
    "resonance_sociale": [
        "pariet_inf-Angular", "temporal_sup", "temporo_pariet"
    ],
    "impact_audio": [
        "G_T_transv", "temp_sup-Lateral", "temp_sup-Plan"
    ],
}

# Labels UI pour l'affichage
ROI_LABELS = {
    "memorisation":      "🧠 Mémorisation",
    "plaisir_dopamine":  "💊 Plaisir / Dopamine",
    "stress_anxiete":    "😰 Stress / Anxiété",
    "attention_visuelle":"👁️ Attention Visuelle",
    "resonance_sociale": "🤝 Résonance Sociale",
    "impact_audio":      "🎵 Impact Audio",
}

# Interprétations de chaque métrique selon le niveau
ROI_INTERPRETATIONS = {
    "memorisation": {
        "haut":   "Pub mémorable — message ancré dans la mémoire épisodique",
        "moyen":  "Mémorisation partielle — renforcer le logo/slogan",
        "bas":    "Faible rétention — retravailler la signature de marque",
    },
    "plaisir_dopamine": {
        "haut":   "Fort plaisir anticipatoire — contenu désirable",
        "moyen":  "Plaisir modéré — ajouter un élément de récompense visuelle",
        "bas":    "Peu de dopamine générée — tester une récompense narrative",
    },
    "stress_anxiete": {
        "haut":   "Contenu anxiogène — attention à l'effet repoussoir",
        "moyen":  "Tension narrative contrôlée — peut être intentionnel",
        "bas":    "Atmosphère sereine — adapté aux marques de confiance",
    },
    "attention_visuelle": {
        "haut":   "Visuellement très captivant — direction artistique efficace",
        "moyen":  "Attention visuelle standard — enrichir la composition",
        "bas":    "Stimulation visuelle faible — plus de mouvement ou couleur",
    },
    "resonance_sociale": {
        "haut":   "Fort sentiment d'identification — audience captivée",
        "moyen":  "Empathie partielle — humaniser davantage le contenu",
        "bas":    "Peu d'identification — ajouter des visages/personnages",
    },
    "impact_audio": {
        "haut":   "Bande-son très efficace — musique/voix bien traitées",
        "moyen":  "Audio correct — optimiser la musique ou le ton de voix",
        "bas":    "Impact audio faible — retravailler le sound design",
    },
}


@functools.lru_cache(maxsize=1)
def get_roi_masks(n_vertices: int):
    """
    Charge les masques de vertices pour chaque ROI via nilearn (Destrieux atlas).
    Mis en cache après le premier appel.
    Retourne un dict {roi_name: np.array(bool, shape=(n_vertices,))} ou None si nilearn absent.
    """
    try:
        from nilearn import datasets
        print("📡 Chargement atlas Destrieux (nilearn)...")
        # fetch_atlas_surf_destrieux retourne fsaverage5 par défaut.
        # Le paramètre 'mesh' n'existe pas dans les versions récentes de nilearn.
        destrieux    = datasets.fetch_atlas_surf_destrieux(verbose=0)
        labels_lh    = np.array(destrieux.map_left)    # (10242,)
        labels_rh    = np.array(destrieux.map_right)   # (10242,)
        label_names  = [str(n) for n in destrieux.labels]

        masks = {}
        for roi_name, patterns in ROI_DESTRIEUX.items():
            # Trouver les indices de labels qui matchent les patterns
            matched_idx = set()
            for i, name in enumerate(label_names):
                if any(p.lower() in name.lower() for p in patterns):
                    matched_idx.add(i)

            if not matched_idx:
                print(f"⚠️ ROI '{roi_name}' — aucun label Destrieux trouvé")
                continue

            mask_lh   = np.isin(labels_lh, list(matched_idx))
            mask_rh   = np.isin(labels_rh, list(matched_idx))
            full_mask = np.concatenate([mask_lh, mask_rh])  # (20484,)

            # Adapter à la taille réelle de preds si différente
            if len(full_mask) > n_vertices:
                full_mask = full_mask[:n_vertices]
            elif len(full_mask) < n_vertices:
                full_mask = np.pad(full_mask, (0, n_vertices - len(full_mask)))

            masks[roi_name] = full_mask
            print(f"  ✅ {ROI_LABELS[roi_name]} : {full_mask.sum()} vertices")

        return masks if masks else None

    except ImportError:
        print("⚠️ nilearn non disponible — fallback sur découpage uniforme")
        return None
    except Exception as e:
        print(f"⚠️ Erreur ROI masks : {e} — fallback sur découpage uniforme")
        return None


def get_roi_masks_fallback(n_vertices: int):
    """
    Fallback si nilearn absent : découpage uniforme du cortex en 6 zones.
    Approximation anatomique grossière mais fonctionnelle.
    Les zones correspondent à des régions connues sur fsaverage5 :
    - Occipital (post) → visuel
    - Temporal → audio / social
    - Pariétal → social / mémoire
    - Frontal inf → dopamine / orbitofrontal
    - Insula (aprox) → stress
    - Cingulaire (aprox) → mémoire
    """
    half = n_vertices // 2
    q    = half // 6  # taille d'un sextant par hémisphère

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
        "memorisation":       bilateral(0.70, 0.85),
        "plaisir_dopamine":   bilateral(0.00, 0.20),
        "stress_anxiete":     bilateral(0.40, 0.55),
        "attention_visuelle": bilateral(0.80, 1.00),
        "resonance_sociale":  bilateral(0.55, 0.70),
        "impact_audio":       bilateral(0.25, 0.40),
    }


def compute_brain_matrix(preds: np.ndarray) -> dict:
    """
    Calcule les 6 métriques neurologiques à partir des prédictions fMRI brutes.
    preds shape : (n_timesteps, n_vertices)
    Retourne dict {roi_name: score_0_100}

    Normalisation : z-score inter-ROI.
    On compare l'activation de chaque ROI aux autres ROIs (pas au range global).
    score=50 → activation dans la moyenne cérébrale
    score>70 → ROI significativement plus active que les autres
    score<30 → ROI moins active que la moyenne
    """
    n_vertices = preds.shape[1]

    # Essayer nilearn, sinon fallback
    masks = get_roi_masks(n_vertices)
    if masks is None:
        masks = get_roi_masks_fallback(n_vertices)

    # Étape 1 : activation brute par ROI (moyenne spatiale + temporelle)
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

    # Étape 2 : z-score cross-ROI → chaque ROI comparée aux autres
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
        # z=-2 → 0/100, z=0 → 50/100, z=+2 → 100/100
        score = int(np.clip(50 + z * 25, 0, 100))
        matrix[roi_name] = score

    return matrix


def interpret_matrix(matrix: dict) -> str:
    """Génère le texte d'interprétation de la matrice."""
    lines = []
    for roi_name, score in matrix.items():
        label = ROI_LABELS[roi_name]
        interp = ROI_INTERPRETATIONS[roi_name]
        if score >= 65:
            niveau_key = "haut"
            niveau_str = "▲ Élevé"
        elif score >= 35:
            niveau_key = "moyen"
            niveau_str = "◆ Moyen"
        else:
            niveau_key = "bas"
            niveau_str = "▼ Faible"
        lines.append(f"{label} : {score}/100 [{niveau_str}]\n  → {interp[niveau_key]}")
    return "\n\n".join(lines)


def make_radar_chart(matrix: dict) -> Image.Image:
    """Génère un radar chart (toile d'araignée) des 6 métriques."""
    # Labels sans emojis pour matplotlib (le serveur n'a pas les polices emoji)
    RADAR_LABELS = {
        "memorisation":       "Memorisation",
        "plaisir_dopamine":   "Plaisir /\nDopamine",
        "stress_anxiete":     "Stress /\nAnxiete",
        "attention_visuelle": "Attention\nVisuelle",
        "resonance_sociale":  "Resonance\nSociale",
        "impact_audio":       "Impact Audio",
    }
    labels  = [RADAR_LABELS[k] for k in ROI_DESTRIEUX]
    values  = [matrix.get(k, 50) for k in ROI_DESTRIEUX]
    N       = len(labels)
    angles  = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]
    values_plot = values + values[:1]

    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))

    # Zones de référence
    ax.fill(angles, [65] * (N + 1), alpha=0.04, color='green')
    ax.fill(angles, [35] * (N + 1), alpha=0.04, color='orange')

    # Courbe principale
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

    ax.set_title("Matrice Neurologique Wardogz", size=12, fontweight='bold', pad=20)

    # Annotation score mémorable
    mem_score = matrix.get("memorisation", 50)
    dop_score = matrix.get("plaisir_dopamine", 50)
    ax.text(0.5, -0.12,
            f"Mémorisation {mem_score}/100  ·  Dopamine {dop_score}/100",
            transform=ax.transAxes, ha='center', fontsize=8, color='#555')

    plt.tight_layout()
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight', dpi=150)
    buf.seek(0)
    img = Image.open(buf).copy()
    plt.close(fig)
    return img

# ============================================================
# PROFILS D'AUDIT
# ============================================================

AUDIT_PROFILES = {
    "UGC Court (< 60s)": {
        "target_height": 720,
        "thresholds": {"excellent": 70, "moyen": 45},
        "labels": {
            "excellent": "🔥 UGC très engageant — fort potentiel de conversion",
            "moyen":     "⚡ Engagement modéré — tester un hook plus fort en ouverture",
            "faible":    "⚠️ Engagement faible — revoir le rythme et les premiers instants",
        },
        "description": "Optimisé pour Reels, TikTok, Stories — vidéos < 60 secondes"
    },
    "Motion Design (pub)": {
        "target_height": 1080,
        "thresholds": {"excellent": 65, "moyen": 40},
        "labels": {
            "excellent": "🔥 Motion très efficace — narration visuelle optimale",
            "moyen":     "⚡ Efficacité modérée — vérifier la lisibilité des textes à l'écran",
            "faible":    "⚠️ Manque d'impact — simplifier le message et accélérer les cuts",
        },
        "description": "Pour animations, explainers, publicités animées"
    },
    "Long Format (> 1min)": {
        "target_height": 720,
        "thresholds": {"excellent": 60, "moyen": 35},
        "labels": {
            "excellent": "🔥 Attention maintenue sur la durée — excellent travail narratif",
            "moyen":     "⚡ Décrochages identifiés — renforcer les transitions et relances",
            "faible":    "⚠️ Attention fragile — restructurer autour de 3 moments forts max",
        },
        "description": "Témoignages, tutoriels, vidéos > 60 secondes"
    },
    "Audit Personnalisé": {
        "target_height": 720,
        "thresholds": {"excellent": 70, "moyen": 45},
        "labels": {
            "excellent": "🔥 Score élevé",
            "moyen":     "⚡ Score moyen",
            "faible":    "⚠️ Score faible",
        },
        "description": "Paramètres manuels — configure les seuils toi-même"
    },
}

# ============================================================
# HISTORIQUE
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
        print(f"⚠️ Thumbnail : {e}")
        return None


def format_history_html(history):
    if not history:
        return "<div style='text-align:center;color:#888;padding:40px'>Aucune analyse enregistrée.</div>"
    cards = []
    for entry in history:
        score  = entry.get("score", 0)
        color  = "#22c55e" if score >= 70 else "#f59e0b" if score >= 45 else "#ef4444"
        thumb  = entry.get("thumbnail", "")
        matrix = entry.get("matrix", {})
        thumb_html = (
            f'<img src="{thumb}" style="width:100%;border-radius:6px 6px 0 0;object-fit:cover;height:110px">'
            if thumb else
            '<div style="height:110px;background:#eee;border-radius:6px 6px 0 0;display:flex;align-items:center;justify-content:center;color:#aaa">📹</div>'
        )
        # Mini-barres pour les 6 métriques
        mini_bars = ""
        for roi_key, roi_label_full in ROI_LABELS.items():
            val       = matrix.get(roi_key, 0)
            bar_color = "#22c55e" if val >= 65 else "#f59e0b" if val >= 35 else "#ef4444"
            short     = roi_label_full.split(" ", 1)[1][:12]
            mini_bars += f"""
            <div style="display:flex;align-items:center;gap:4px;margin-bottom:2px">
                <span style="font-size:9px;width:80px;color:#555;white-space:nowrap;overflow:hidden">{short}</span>
                <div style="flex:1;background:#eee;border-radius:3px;height:5px">
                    <div style="width:{val}%;background:{bar_color};height:5px;border-radius:3px"></div>
                </div>
                <span style="font-size:9px;color:#777;width:22px;text-align:right">{val}</span>
            </div>"""
        card = f"""
        <div style="background:white;border-radius:8px;box-shadow:0 2px 8px rgba(0,0,0,0.1);overflow:hidden;min-width:210px;max-width:250px;flex-shrink:0">
            {thumb_html}
            <div style="padding:10px">
                <div style="font-weight:bold;font-size:12px;white-space:nowrap;overflow:hidden;text-overflow:ellipsis" title="{entry.get('filename','')}">{entry.get('filename','Vidéo')}</div>
                <div style="font-size:10px;color:#888;margin-bottom:5px">{entry.get('date','')} · {entry.get('profile','')}</div>
                <div style="font-size:20px;font-weight:bold;color:{color};margin-bottom:6px">{score}<span style="font-size:11px;color:#888">/100</span></div>
                {mini_bars}
            </div>
        </div>"""
        cards.append(card)
    return f'<div style="display:flex;gap:14px;flex-wrap:wrap;padding:10px 0">{"".join(cards)}</div>'

# ============================================================
# UTILITAIRES VIDÉO
# ============================================================

def _open_video(video_path):
    """
    Ouvre une VideoFileClip en gérant moviepy v1.x (moviepy.editor)
    et moviepy v2.x (moviepy directement).
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
        print(f"⚠️ moviepy get_video_info : {e}")
        return 32.0, 1280, 720


def downscale_video(video_path, target_height):
    try:
        clip = _open_video(video_path)
        h, w = clip.size[1], clip.size[0]
        if h <= target_height:
            print(f"📹 Résolution OK ({w}×{h})")
            clip.close()
            return video_path
        ratio    = target_height / h
        new_w    = int(w * ratio)
        out_path = video_path + f"_{target_height}p.mp4"
        print(f"⚡ Downscale : {w}×{h} → {new_w}×{target_height}")
        # moviepy v2.x a renommé resize() en resized(), on essaie les deux
        try:
            resized = clip.resized((new_w, target_height))   # v2.x
        except AttributeError:
            resized = clip.resize((new_w, target_height))    # v1.x
        resized.write_videofile(out_path, codec="libx264", audio_codec="aac", logger=None)
        clip.close()
        return out_path
    except Exception as e:
        print(f"⚠️ Downscale échoué ({e})")
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
    creux_s = int(seconds[min_start])
    creux_e = min(int(real_duration), creux_s + 5)
    pic_s   = int(seconds[max_start])
    pic_e   = min(int(real_duration), pic_s + 5)
    mid     = n // 2
    d       = np.mean(curve_100[mid:]) - np.mean(curve_100[:mid])
    tendance = ("montante 📈" if d > 10 else "descendante 📉" if d < -10 else "stable ➡️")
    return score, creux_s, creux_e, int(min_avg), pic_s, pic_e, int(max_avg), tendance

# ============================================================
# VISUALISATION — COURBE TEMPORELLE
# ============================================================

def make_attention_plot(curve_100, seconds, real_duration, score, creux_s, creux_e,
                        pic_s, pic_e, niveau, video_label):
    fig, ax = plt.subplots(figsize=(13, 5))
    ax.plot(seconds, curve_100, color='#FF4500', linewidth=2.5, label='Engagement cérébral', zorder=3)
    ax.fill_between(seconds, curve_100, alpha=0.10, color='#FF4500')
    mask_creux = (seconds >= creux_s) & (seconds <= creux_e)
    ax.fill_between(seconds, curve_100, where=mask_creux, alpha=0.30,
                    color='#ef4444', label=f'⚠ Creux ({creux_s}s–{creux_e}s)')
    mask_pic = (seconds >= pic_s) & (seconds <= pic_e)
    ax.fill_between(seconds, curve_100, where=mask_pic, alpha=0.22,
                    color='#22c55e', label=f'★ Pic ({pic_s}s–{pic_e}s)')
    color_line = "#22c55e" if niveau == "excellent" else "#f59e0b" if niveau == "moyen" else "#ef4444"
    ax.axhline(y=score, color=color_line, linestyle='--', alpha=0.8, linewidth=1.5,
               label=f'Score moyen : {score}/100')
    ax.set_title(f"Courbe d'Attention — {video_label or 'Vidéo'} · Wardogz",
                 fontsize=13, fontweight='bold', pad=14)
    ax.set_xlabel("Temps (secondes)", fontsize=11)
    ax.set_ylabel("Engagement Cérébral / 100", fontsize=11)
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
# FONCTION PRINCIPALE D'ANALYSE
# ============================================================

def scan_video(video, profile_name, seuil_excellent, seuil_moyen, video_label):

    if model is None:
        return "❌ Modèle non chargé.", None, None, format_history_html(load_history())
    if video is None:
        return "⚠️ Aucune vidéo uploadée.", None, None, format_history_html(load_history())

    try:
        uv_path = os.path.expanduser("~/.cargo/bin")
        if uv_path not in os.environ.get("PATH", ""):
            os.environ["PATH"] = f"{uv_path}:{os.environ['PATH']}"

        profile = AUDIT_PROFILES.get(profile_name, AUDIT_PROFILES["UGC Court (< 60s)"])
        if profile_name == "Audit Personnalisé":
            profile["thresholds"]["excellent"] = seuil_excellent
            profile["thresholds"]["moyen"]     = seuil_moyen
        target_height = profile["target_height"]
        thresholds    = profile["thresholds"]
        labels        = profile["labels"]

        real_duration, orig_w, orig_h = get_video_info(video)
        duration_label = f"{int(real_duration)}s · {orig_w}×{orig_h}"
        print(f"🎬 {duration_label}")

        thumbnail        = extract_thumbnail(video, timestamp=min(1.0, real_duration * 0.1))
        video_to_analyse = downscale_video(video, target_height)

        df_events = model.get_events_dataframe(video_path=video_to_analyse)

        # ✅ FIX durée : on injecte la durée réelle AVEC une marge de sécurité de 0.1s
        # Tribe v2 fait un assert strict "end <= clip.duration", donc si on arrondit
        # à 32.0 mais que la vidéo fait 31.95s → AssertionError. On soustrait 0.1s.
        safe_duration = real_duration - 0.1
        for col in ['Duration', 'duration']:
            if col in df_events.columns:
                # Ne réduire que si notre valeur est supérieure à ce que Tribe v2 a détecté
                current_val = df_events[col].iloc[0] if len(df_events) > 0 else 0
                if safe_duration > current_val:
                    df_events[col] = safe_duration
                    print(f"✅ Durée injectée : {safe_duration:.2f}s (réelle {real_duration:.2f}s, marge -0.1s)")

        print("🧠 Prédiction Tribe v2...")
        with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
            preds, segments = model.predict(events=df_events)
        if isinstance(preds, torch.Tensor):
            preds = preds.detach().cpu().numpy()
        n_kept = len(preds)
        print(f"✅ preds shape : {preds.shape}")

        # ── Timestamps réels (Segment.start confirmé à 1 Hz) ────────
        # Structure confirmée : list[Segment], chaque Segment a .start (float, en secondes)
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
            print(f"⚠️ seg_times : {e}")

        # ── Couverture temporelle ────────────────────────────────
        # 1 prédiction = 1 seconde de signal fMRI à 1 Hz.
        # n_kept / durée = fraction de la vidéo couverte (≈ 1.0 pour les vidéos parlées)
        frac         = n_kept / max(real_duration, 1)
        coverage_pct = min(int(frac * 100), 100)
        confiance    = "Haute" if frac >= 0.8 else "Moyenne" if frac >= 0.4 else "Faible"
        print(f"📡 Couverture : {n_kept} pts / {real_duration:.1f}s · {coverage_pct}% · {confiance}")

        # ── Courbe d'attention globale ──────────────────────────
        raw_curve = np.mean(preds, axis=1)
        curve_100 = normalize_curve(raw_curve)
        n_points  = len(curve_100)

        # Axe temporel : vrais timestamps si disponibles (plus précis), sinon linspace
        if seg_times is not None and len(seg_times) == n_points:
            seconds = seg_times
        else:
            seconds = np.linspace(0, real_duration, n_points)

        score, creux_s, creux_e, creux_val, pic_s, pic_e, pic_val, tendance = analyse_curve(
            curve_100, seconds, real_duration
        )

        niveau = (
            "excellent" if score >= thresholds["excellent"] else
            "moyen"     if score >= thresholds["moyen"]     else
            "faible"
        )

        # ── Matrice neurologique (6 métriques) ──────────────────
        print("🔬 Calcul matrice neurologique...")
        matrix = compute_brain_matrix(preds)
        matrix_text = interpret_matrix(matrix)

        # ── Verdict global ──────────────────────────────────────
        emoji  = "🔥" if niveau == "excellent" else "⚡" if niveau == "moyen" else "⚠️"
        confiance_note = {
            "Haute":   "✅ Signal fiable",
            "Moyenne": "⚡ Signal partiel (vidéo peu parlée ?)",
            "Faible":  "⚠️ Signal faible — résultats à prendre avec précaution",
        }[confiance]
        verdict = "\n".join([
            f"{emoji} Score Global : {score}/100",
            labels[niveau],
            "",
            f"📉 Creux : {creux_s}s–{creux_e}s ({creux_val}/100) → zone à retravailler",
            f"📈 Pic   : {pic_s}s–{pic_e}s ({pic_val}/100) → moment fort",
            f"📊 Tendance : {tendance}",
            "",
            f"📡 Signal : {n_kept} segments / {int(real_duration)}s · {confiance_note}",
            "",
            "──── Matrice Neurologique ────",
            "  (scores relatifs : 50=dans la moyenne, >70=saillant, <30=sous-actif)",
            matrix_text,
            "",
            f"🎬 {n_points} pts · {duration_label} · Profil : {profile_name}",
        ])

        # ── Visualisations ──────────────────────────────────────
        plot_attention = make_attention_plot(
            curve_100, seconds, real_duration, score,
            creux_s, creux_e, pic_s, pic_e, niveau, video_label
        )
        plot_radar = make_radar_chart(matrix)

        # ── Historique ──────────────────────────────────────────
        entry = {
            "date":          datetime.datetime.now().strftime("%d/%m/%Y %H:%M"),
            "filename":      video_label or pathlib.Path(video).name,
            "profile":       profile_name,
            "score":         score,
            "duration_label":duration_label,
            "creux_label":   f"Creux {creux_s}s–{creux_e}s ({creux_val}/100)",
            "tendance":      tendance,
            "thumbnail":     thumbnail,
            "niveau":        niveau,
            "matrix":        matrix,
        }
        save_to_history(entry)

        print(f"✅ Terminé — Score {score}/100 | Creux {creux_s}s–{creux_e}s")
        return verdict, plot_attention, plot_radar, format_history_html(load_history())

    except Exception as e:
        import traceback
        print(f"❌\n{traceback.format_exc()}")
        return f"❌ Erreur :\n{str(e)}", None, None, format_history_html(load_history())

# ============================================================
# INTERFACE GRADIO
# ============================================================

with gr.Blocks(theme=gr.themes.Soft(), title="Wardogz Brain Scanner v2") as demo:

    gr.Markdown("""
    # 🧠 BlackMotion × Wardogz — Neuro-UGC Scanner v2
    *Courbe d'attention + Matrice neurologique · Powered by Meta Tribe v2*
    """)
    gr.Markdown(f"**Statut :** {status_msg}")

    with gr.Tabs():

        # ── TAB 1 : ANALYSE ─────────────────────────────────────
        with gr.Tab("🔬 Analyse"):
            with gr.Row():

                with gr.Column(scale=1):
                    input_video = gr.Video(label="📹 Upload Vidéo (HD, SD, toute durée)")
                    video_label = gr.Textbox(
                        label="Nom du projet / client",
                        placeholder="Ex: Fidanimo UGC Mars 2026"
                    )
                    gr.Markdown("### ⚙️ Profil d'audit")
                    profile_selector = gr.Radio(
                        choices=list(AUDIT_PROFILES.keys()),
                        value="UGC Court (< 60s)",
                        label="Type de contenu",
                    )
                    profile_desc = gr.Markdown(
                        f"*{AUDIT_PROFILES['UGC Court (< 60s)']['description']}*"
                    )
                    with gr.Accordion("🎛️ Seuils (Audit Personnalisé uniquement)", open=False):
                        seuil_excellent = gr.Slider(50, 90, value=70, step=5, label="Seuil Excellent")
                        seuil_moyen     = gr.Slider(20, 70, value=45, step=5, label="Seuil Moyen")
                    btn = gr.Button("🚀 Lancer le Scan Cérébral", variant="primary", size="lg")

                with gr.Column(scale=2):
                    output_verdict = gr.Textbox(label="🎯 Verdict + Matrice Neurologique", lines=18)
                    with gr.Row():
                        output_attention = gr.Image(label="📊 Courbe d'Attention fMRI")
                        output_radar     = gr.Image(label="🕸️ Matrice Neurologique")

            profile_selector.change(
                fn=lambda p: f"*{AUDIT_PROFILES[p]['description']}*",
                inputs=profile_selector,
                outputs=profile_desc
            )

        # ── TAB 2 : HISTORIQUE ──────────────────────────────────
        with gr.Tab("📁 Historique des Scans"):
            gr.Markdown("### Toutes vos analyses — métriques neurologiques incluses")
            history_display = gr.HTML(value=format_history_html(load_history()))
            gr.Button("🔄 Rafraîchir", size="sm").click(
                fn=lambda: format_history_html(load_history()),
                outputs=history_display
            )

    btn.click(
        fn=scan_video,
        inputs=[input_video, profile_selector, seuil_excellent, seuil_moyen, video_label],
        outputs=[output_verdict, output_attention, output_radar, history_display],
    )

    gr.Markdown("---\n*Meta Tribe v2 (V-JEPA2 + LLaMA 3.2 + Wav2Vec-BERT) · Atlas Destrieux · © Wardogz Agency*")

demo.launch()