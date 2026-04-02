"""
Neural CTR Predictor.

Pipeline:
  1. Gemini generates N dynamic title tag variants for a keyword
  2. TRIBE v2 predicts fMRI response to each title
  3. Titles are ranked by frontal attention activation + salience score
  4. TRIBE-only summary of top winners
"""
import logging
from src.tribe_engine import predict_texts_batch
from src.neural_scorer import activation_to_profile, compute_ctr_score
from src.gemini_client import generate_title_variants

logger = logging.getLogger(__name__)


def _generate_ctr_analysis_tribe(keyword: str, ranked_titles: list[dict]) -> str:
    if not ranked_titles:
        return ""

    top = ranked_titles[:3]
    lines = [
        "### Neural CTR Readout (TRIBE-only)",
        "",
        f"- Keyword: **{keyword}**",
        "- The top titles show stronger salience + engagement activation balance.",
        "",
        "| Rank | Title | Score | Salience | Engagement |",
        "|------|-------|------:|---------:|-----------:|",
    ]
    for i, item in enumerate(top):
        profile = item.get("profile", {})
        lines.append(
            f"| {i+1} | {item['title']} | {item['ctr_score'] * 100:.1f}% | "
            f"{profile.get('peak_salience', 0) * 100:.1f}% | {profile.get('global_engagement', 0) * 100:.1f}% |"
        )
    return "\n".join(lines)


def predict_ctr(keyword: str, n_titles: int = 20) -> dict:
    """
    Run the full Neural CTR Predictor pipeline.

    Args:
        keyword: Target keyword or topic (e.g. "best running shoes 2026")
        n_titles: Number of title variants to generate and test

    Returns:
        {
          "keyword": str,
          "ranked_titles": list of dicts sorted by ctr_score desc,
          "analysis": str (TRIBE-only summary),
          "error": str | None,
        }
    """
    # Step 1: Generate dynamic title variants via Gemini
    try:
        titles = generate_title_variants(keyword, n=n_titles)
        if not titles:
            return {"error": "No title variants generated.", "keyword": keyword}
    except Exception as exc:
        return {"error": f"Gemini title generation failed: {exc}", "keyword": keyword}

    logger.info("Generated %d title variants. Running TRIBE v2 batch scoring ...", len(titles))

    # Step 2: Score ALL titles in a single ZeroGPU allocation (one GPU grab for the batch)
    activations = predict_texts_batch(titles)

    scored = []
    for title, activation in zip(titles, activations):
        if activation is None:
            scored.append({"title": title, "ctr_score": 0.0, "profile": {}, "error": "inference failed"})
            continue
        try:
            profile = activation_to_profile(activation)
            score = compute_ctr_score(profile)
            scored.append({
                "title": title,
                "ctr_score": score,
                "profile": {
                    "global_engagement": round(profile.global_engagement, 3),
                    "peak_salience": round(profile.peak_salience, 3),
                    "dmn_suppression": round(profile.dmn_suppression, 3),
                    "sustained_attention": round(profile.sustained_attention, 3),
                },
            })
        except Exception as exc:
            logger.warning("Score failed for '%s': %s", title[:40], exc)
            scored.append({"title": title, "ctr_score": 0.0, "profile": {}, "error": str(exc)})

    # Step 3: Sort by neural CTR score
    ranked = sorted(scored, key=lambda x: x["ctr_score"], reverse=True)

    # Step 4: TRIBE-only winner summary
    analysis = _generate_ctr_analysis_tribe(keyword, ranked)

    return {
        "keyword": keyword,
        "ranked_titles": ranked,
        "analysis": analysis,
        "error": None,
    }
