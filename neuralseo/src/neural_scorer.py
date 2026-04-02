"""
Maps TRIBE v2 vertex activations to SEO-meaningful neural scores.

Approach:
  - fsaverage5 has ~20,484 vertices (10,242 per hemisphere)
  - We use statistical properties of the activation distribution as proxies
    for neurologically-grounded SEO signals.
  - For proper ROI mapping in future: use FreeSurfer parcellation via nibabel.

Key signals derived:
  1. Global Engagement       — mean activation across all vertices
  2. Neural Complexity       — std dev (expert content → richer, more varied response)
  3. Activation Entropy      — spatial entropy (E-E-A-T proxy: expert = distributed)
  4. Peak Salience           — 95th percentile activation (attention-grabbing moments)
  5. Sustained Attention     — ratio of high-activation vertices (>0.6 normalized)
  6. DMN Suppression         — inverse of low-variance, low-mean (mind-wandering proxy)
"""
import numpy as np
from scipy.stats import entropy as scipy_entropy
from dataclasses import dataclass


@dataclass
class NeuralProfile:
    global_engagement: float      # 0–1
    neural_complexity: float      # 0–1
    activation_entropy: float     # 0–1  ← primary E-E-A-T proxy
    peak_salience: float          # 0–1
    sustained_attention: float    # 0–1
    dmn_suppression: float        # 0–1  (higher = less mind-wandering)
    raw_mean: float
    raw_std: float


def activation_to_profile(activation: np.ndarray) -> NeuralProfile:
    """
    Convert a raw TRIBE v2 spatial activation map (n_vertices,) to a NeuralProfile.
    All returned values are normalized to [0, 1].
    """
    if activation is None or len(activation) == 0:
        raise ValueError("Empty activation map.")

    a = activation.astype(np.float32)

    # Normalize to [0, 1]
    a_min, a_max = a.min(), a.max()
    if a_max > a_min:
        a_norm = (a - a_min) / (a_max - a_min)
    else:
        a_norm = np.zeros_like(a)

    raw_mean = float(a.mean())
    raw_std = float(a.std())

    # 1. Global engagement: mean of normalized map
    global_engagement = float(a_norm.mean())

    # 2. Neural complexity: coefficient of variation (std/mean of raw) clamped
    cv = float(a.std() / (abs(a.mean()) + 1e-8))
    neural_complexity = float(np.clip(cv / 3.0, 0, 1))  # CV ~3 = max complexity

    # 3. Activation entropy: Shannon entropy of histogram
    #    Expert content → high entropy (many regions active at varied levels)
    #    Thin content   → low entropy (sparse, uniform response)
    hist, _ = np.histogram(a_norm, bins=50, range=(0, 1), density=True)
    hist = hist + 1e-10  # avoid log(0)
    ent = float(scipy_entropy(hist))
    # Max entropy for 50 bins ≈ log(50) ≈ 3.91
    activation_entropy = float(np.clip(ent / 3.91, 0, 1))

    # 4. Peak salience: 95th percentile of normalized activations
    peak_salience = float(np.percentile(a_norm, 95))

    # 5. Sustained attention: fraction of vertices with activation > 0.5
    sustained_attention = float((a_norm > 0.5).mean())

    # 6. DMN suppression: inverse of how "flat" the response is
    #    Flat (low std) → high DMN (mind-wandering) → low suppression score
    flatness = 1.0 - neural_complexity
    dmn_suppression = float(np.clip(1.0 - flatness, 0, 1))

    return NeuralProfile(
        global_engagement=global_engagement,
        neural_complexity=neural_complexity,
        activation_entropy=activation_entropy,
        peak_salience=peak_salience,
        sustained_attention=sustained_attention,
        dmn_suppression=dmn_suppression,
        raw_mean=raw_mean,
        raw_std=raw_std,
    )


def compute_eeat_score(
    content_profile: NeuralProfile,
    expert_profiles: list[NeuralProfile],
    shallow_profiles: list[NeuralProfile],
) -> dict:
    """
    Score E-E-A-T as neural distance from expert vs shallow content profiles.

    Returns a dict with:
      - eeat_score: 0–100
      - dimension_scores: per-signal breakdown
      - verdict: string label
    """
    # Expert centroid
    expert_centroid = _mean_profile(expert_profiles)
    shallow_centroid = _mean_profile(shallow_profiles)

    # Feature vector for each profile
    content_vec = _profile_to_vec(content_profile)
    expert_vec = _profile_to_vec(expert_centroid)
    shallow_vec = _profile_to_vec(shallow_centroid)

    # Cosine similarity to expert and shallow
    sim_expert = _cosine_sim(content_vec, expert_vec)
    sim_shallow = _cosine_sim(content_vec, shallow_vec)

    # Score: how much closer to expert than shallow (0–1 → 0–100)
    total = sim_expert + sim_shallow + 1e-8
    eeat_raw = sim_expert / total
    eeat_score = int(round(eeat_raw * 100))

    # Per-dimension breakdown
    dim_labels = [
        "Global Engagement",
        "Neural Complexity",
        "Activation Entropy (E-E-A-T Core)",
        "Peak Salience",
        "Sustained Attention",
        "DMN Suppression",
    ]
    expert_arr = np.array(list(_profile_to_vec(expert_centroid)))
    shallow_arr = np.array(list(_profile_to_vec(shallow_centroid)))
    content_arr = np.array(list(_profile_to_vec(content_profile)))

    dimension_scores = {}
    for i, label in enumerate(dim_labels):
        # Normalize content dim relative to expert/shallow range
        lo, hi = min(expert_arr[i], shallow_arr[i]), max(expert_arr[i], shallow_arr[i])
        if hi > lo:
            val = float(np.clip((content_arr[i] - lo) / (hi - lo), 0, 1))
            # If expert is higher, val is already correct; if shallow is higher, flip
            if shallow_arr[i] > expert_arr[i]:
                val = 1.0 - val
        else:
            val = 0.5
        dimension_scores[label] = round(val * 100)

    if eeat_score >= 75:
        verdict = "Expert-Level Content"
    elif eeat_score >= 55:
        verdict = "Solid — Room for Depth"
    elif eeat_score >= 35:
        verdict = "Shallow — Needs Expertise Signals"
    else:
        verdict = "Thin Content — Low E-E-A-T"

    return {
        "eeat_score": eeat_score,
        "dimension_scores": dimension_scores,
        "verdict": verdict,
        "sim_expert": round(float(sim_expert), 4),
        "sim_shallow": round(float(sim_shallow), 4),
    }


def compute_ctr_score(profile: NeuralProfile) -> float:
    """
    Single attention-weighted CTR score for a title tag.
    Returns value in [0, 1].

    Weights: peak_salience is the dominant factor (it's the "stop the scroll" signal),
    supported by global_engagement and dmn_suppression.
    """
    score = (
        0.45 * profile.peak_salience
        + 0.30 * profile.global_engagement
        + 0.15 * profile.dmn_suppression
        + 0.10 * profile.sustained_attention
    )
    return float(np.clip(score, 0, 1))


# ── Internal helpers ────────────────────────────────────────────────────────

def _profile_to_vec(p: NeuralProfile) -> tuple:
    return (
        p.global_engagement,
        p.neural_complexity,
        p.activation_entropy,
        p.peak_salience,
        p.sustained_attention,
        p.dmn_suppression,
    )


def _cosine_sim(a: tuple, b: tuple) -> float:
    va, vb = np.array(a, dtype=float), np.array(b, dtype=float)
    denom = (np.linalg.norm(va) * np.linalg.norm(vb)) + 1e-10
    return float(np.dot(va, vb) / denom)


def _mean_profile(profiles: list[NeuralProfile]) -> NeuralProfile:
    vecs = [_profile_to_vec(p) for p in profiles]
    mean = np.array(vecs).mean(axis=0)
    return NeuralProfile(
        global_engagement=mean[0],
        neural_complexity=mean[1],
        activation_entropy=mean[2],
        peak_salience=mean[3],
        sustained_attention=mean[4],
        dmn_suppression=mean[5],
        raw_mean=0.0,
        raw_std=0.0,
    )
