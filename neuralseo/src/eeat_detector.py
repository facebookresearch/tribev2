"""
Neural E-E-A-T Detector.

Expert content fires the brain differently than shallow filler:
  - Expert: distributed, high-entropy, spatially complex neural activation
  - Thin:   weak, flat, low-entropy activation

Reference profiles are built lazily on the first E-E-A-T analysis request
(inside a @spaces.GPU call), not at startup.
"""
import logging
from src.tribe_engine import predict_text, predict_texts_batch
from src.neural_scorer import activation_to_profile, compute_eeat_score, NeuralProfile

logger = logging.getLogger(__name__)

_EXPERT_SAMPLES = [
    """
    Transformer architectures have fundamentally reshaped natural language processing
    by replacing recurrence with self-attention mechanisms, enabling parallel computation
    across sequence positions. The scaled dot-product attention computes compatibility
    between queries and keys as Q·K^T / sqrt(d_k), followed by a softmax to yield
    attention weights over values. Multi-head attention projects inputs into h subspaces,
    applies attention independently, then concatenates and projects the outputs.
    This allows the model to jointly attend to information from different representational
    subspaces at different positions — a capability that single-head attention suppresses.
    Positional encodings inject sequence order information since self-attention is
    permutation-invariant. Layer normalization and residual connections stabilize training.
    """,
    """
    Chronic inflammation is implicated in the pathogenesis of atherosclerosis through
    multiple mechanisms. Endothelial dysfunction leads to increased expression of
    adhesion molecules such as VCAM-1 and ICAM-1, facilitating monocyte recruitment
    and transmigration into the subintimal space. Oxidized LDL triggers macrophage
    foam cell formation via scavenger receptors, initiating the formation of fatty
    streaks. Plaque vulnerability is determined by the balance between fibrous cap
    thickness — maintained by smooth muscle cell proliferation and collagen synthesis —
    and the necrotic core size. Matrix metalloproteinases secreted by activated
    macrophages degrade the fibrous cap, predisposing plaques to rupture.
    """,
    """
    Core Web Vitals are Google's set of specific factors used to measure user experience:
    Largest Contentful Paint (LCP) measures loading performance, targeting under 2.5
    seconds; Interaction to Next Paint (INP) replaced First Input Delay in 2024,
    measuring responsiveness with a threshold of 200ms; Cumulative Layout Shift (CLS)
    quantifies visual stability with a threshold of 0.1. These metrics are incorporated
    into Google's page experience signals and influence ranking. Technical optimization
    strategies include preloading LCP image resources, eliminating render-blocking
    scripts, implementing efficient cache policies, and reserving space for dynamically
    injected content to prevent layout shifts.
    """,
]

_SHALLOW_SAMPLES = [
    """
    SEO is very important for websites. If you want to rank higher on Google, you need
    to do SEO. SEO stands for Search Engine Optimization. There are many SEO tips and
    tricks you can use. Keywords are important in SEO. You should use keywords in your
    content. Good content is also important. Make sure your website is fast. Also make
    sure it works on mobile. These are some basic SEO tips to help your website rank
    better on search engines like Google and Bing.
    """,
    """
    Artificial intelligence is changing everything. AI is used in many industries today.
    Companies are using AI to improve their business. AI can help you save time and money.
    There are many types of AI. Machine learning is a type of AI. Deep learning is also
    a type of AI. AI is the future. Many experts believe AI will continue to grow. AI
    is an exciting technology. You should learn more about AI if you want to stay
    competitive in today's fast-changing world.
    """,
    """
    Health and wellness are very important. Eating healthy food is good for you. You
    should exercise regularly to stay fit. Drinking water is also important for your
    health. Getting enough sleep helps your body recover. Stress can be bad for your
    health so try to reduce stress. Meditation is a good way to reduce stress. Taking
    care of your mental health is just as important as physical health. These are some
    tips to help you live a healthier and happier life.
    """,
]

# Lazy-built reference profiles (populated on first analyze_eeat call)
_expert_profiles: list[NeuralProfile] = []
_shallow_profiles: list[NeuralProfile] = []
_references_built = False


def _ensure_references():
    """
    Build expert/shallow reference profiles if not yet done.
    MUST be called from within a context where predict_text can run
    (i.e., inside a @spaces.GPU decorated call chain).
    """
    global _references_built, _expert_profiles, _shallow_profiles
    if _references_built:
        return

    logger.info("Building E-E-A-T reference neural profiles (first run)...")
    all_samples = [t.strip() for t in (_EXPERT_SAMPLES + _SHALLOW_SAMPLES)]
    activations = predict_texts_batch(all_samples, max_chars=2000)

    split = len(_EXPERT_SAMPLES)
    expert_acts = activations[:split]
    shallow_acts = activations[split:]

    _expert_profiles = []
    _shallow_profiles = []

    for idx, act in enumerate(expert_acts):
        if act is None:
            raise RuntimeError(f"Expert reference sample {idx + 1} failed inference.")
        _expert_profiles.append(activation_to_profile(act))

    for idx, act in enumerate(shallow_acts):
        if act is None:
            raise RuntimeError(f"Shallow reference sample {idx + 1} failed inference.")
        _shallow_profiles.append(activation_to_profile(act))

    _references_built = True
    logger.info("E-E-A-T reference profiles ready (%d expert, %d shallow).",
                len(_expert_profiles), len(_shallow_profiles))


def warm_up_references():
    """No-op on ZeroGPU. References build lazily on first analysis."""
    logger.info("ZeroGPU mode: E-E-A-T references will build on first analysis request.")


def analyze_eeat(text: str) -> dict:
    """
    Full E-E-A-T neural analysis.

    Note: This function calls predict_text which is @spaces.GPU decorated.
    On ZeroGPU, each predict_text call gets its own GPU allocation.
    References are built on the first call.
    """
    try:
        # Build references if first run (each predict_text call uses GPU)
        _ensure_references()
    except Exception as exc:
        logger.error("Reference profile build failed: %s", exc)

    refs_ready = _references_built and _expert_profiles and _shallow_profiles

    try:
        activation = predict_text(text, max_chars=3000)
        profile = activation_to_profile(activation)
    except Exception as exc:
        return {"error": str(exc), "eeat_score": None, "references_ready": refs_ready}

    if refs_ready:
        result = compute_eeat_score(profile, _expert_profiles, _shallow_profiles)
    else:
        entropy_score = int(round(profile.activation_entropy * 100))
        result = {
            "eeat_score": entropy_score,
            "verdict": _entropy_verdict(profile.activation_entropy),
            "dimension_scores": {
                "Activation Entropy (E-E-A-T Core)": entropy_score,
                "Global Engagement": int(round(profile.global_engagement * 100)),
                "Neural Complexity": int(round(profile.neural_complexity * 100)),
                "Peak Salience": int(round(profile.peak_salience * 100)),
                "Sustained Attention": int(round(profile.sustained_attention * 100)),
                "DMN Suppression": int(round(profile.dmn_suppression * 100)),
            },
            "sim_expert": None,
            "sim_shallow": None,
        }

    result["neural_profile"] = profile
    result["references_ready"] = refs_ready
    result["error"] = None
    return result


def _entropy_verdict(entropy: float) -> str:
    if entropy >= 0.75:
        return "Expert-Level Content"
    elif entropy >= 0.55:
        return "Solid — Room for Depth"
    elif entropy >= 0.35:
        return "Shallow — Needs Expertise Signals"
    return "Thin Content — Low E-E-A-T"
