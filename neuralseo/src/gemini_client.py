"""
Gemini helper for introduction paragraph recommendations.
"""
import os
import logging
import json
import re
from google import genai

logger = logging.getLogger(__name__)

_client: genai.Client | None = None


def _get_client() -> genai.Client:
    global _client
    if _client is None:
        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key:
            raise RuntimeError("GEMINI_API_KEY is not set.")
        _client = genai.Client(api_key=api_key)
    return _client


def _generate(prompt: str) -> str:
    client = _get_client()
    response = client.models.generate_content(
        model="gemini-3-flash-preview",
        contents=prompt,
    )
    return (response.text or "").strip()


def _extract_json_array(text: str) -> list:
    text = (text or "").strip()
    if not text:
        return []
    try:
        data = json.loads(text)
        if isinstance(data, list):
            return data
    except Exception:
        pass

    match = re.search(r"\[[\s\S]*\]", text)
    if not match:
        return []
    try:
        data = json.loads(match.group(0))
        return data if isinstance(data, list) else []
    except Exception:
        return []


def generate_intro_recommendations(
    intro_text: str,
    intro_score: int,
    verdict: str,
    dimension_scores: dict,
) -> str:
    prompt = f"""You are an elite SEO copy editor.

This introduction paragraph was scored with Meta TRIBE v2 neural signals.

Score: {intro_score}/100
Verdict: {verdict}
Dimension scores: {dimension_scores}

Original introduction (max 600 chars):
\"\"\"{intro_text}\"\"\"

Return concise markdown with exactly:
1) "Top Issues" as 3 bullets
2) "Rewrite Suggestions" as 3 bullets
3) "Improved Intro (<=600 chars)" as one final rewritten paragraph

Focus on hook strength, clarity, salience, and retention.
"""
    try:
        client = _get_client()
        response = client.models.generate_content(
            model="gemini-3-flash-preview",
            contents=prompt,
        )
        text = (response.text or "").strip()
        if text:
            return text
        return "Gemini returned an empty recommendation response."
    except Exception as exc:
        logger.error("Gemini intro recommendations failed: %s", exc)
        return f"Gemini recommendations unavailable: {exc}"


def generate_title_variants(keyword: str, n: int = 20) -> list[str]:
    prompt = f"""You are an elite SEO copywriter and neuromarketing editor.

Generate exactly {n} unique SEO title tags for keyword/topic:
"{keyword}"

Constraints:
- Natural, human phrasing. No template-like repetition.
- Length target: 48-65 characters.
- Include keyword once in a natural way (not always at title start).
- Diverse angles: how-to, list, authority, contrast, curiosity, outcomes.
- No numbering prefix. No quotes.
- Output ONLY a JSON array of strings.
"""
    raw = _generate(prompt)
    items = _extract_json_array(raw)
    cleaned = []
    seen = set()
    for item in items:
        if not isinstance(item, str):
            continue
        title = re.sub(r"\s+", " ", item.strip())
        title = re.sub(r"^[\-\d\.\)\s]+", "", title)
        if len(title) < 24:
            continue
        key = title.lower()
        if key in seen:
            continue
        seen.add(key)
        cleaned.append(title)
        if len(cleaned) >= n:
            break

    if len(cleaned) < max(5, n // 2):
        raise RuntimeError("Gemini returned too few valid title variants.")
    return cleaned[:n]


def generate_ctr_analysis(keyword: str, ranked_titles: list[dict]) -> str:
    top3 = ranked_titles[:3]
    if len(top3) < 3:
        return ""
    prompt = f"""You are an expert in SEO psychology and CTR optimization.

Keyword: "{keyword}"

Top 3 titles ranked by TRIBE v2 neural CTR score:
1) "{top3[0]['title']}" — {top3[0]['ctr_score']:.0%}
2) "{top3[1]['title']}" — {top3[1]['ctr_score']:.0%}
3) "{top3[2]['title']}" — {top3[2]['ctr_score']:.0%}

Return concise markdown:
- "Why These Win" with 3 bullets
- "How to Improve Lower Titles" with 3 bullets
Keep it short and specific.
"""
    try:
        return _generate(prompt)
    except Exception as exc:
        logger.error("Gemini CTR analysis failed: %s", exc)
        return ""

