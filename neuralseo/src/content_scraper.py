"""
Content scraper powered by Parallel.ai Extract API.

Parallel returns clean, LLM-optimized markdown from any URL —
handles JavaScript-heavy pages, paywalls, and PDFs automatically.
No BeautifulSoup, no parsing, no noise.

Docs: https://docs.parallel.ai/api-reference/extract-beta/extract
"""
import os
import logging
import re
from parallel import Parallel

logger = logging.getLogger(__name__)

_client: Parallel | None = None


def _get_client() -> Parallel:
    global _client
    if _client is None:
        api_key = os.environ.get("PARALLEL_API_KEY")
        if not api_key:
            raise RuntimeError("PARALLEL_API_KEY environment variable not set.")
        _client = Parallel(api_key=api_key)
    return _client


def scrape_url(url: str, max_chars: int = 4000) -> dict:
    """
    Fetch and extract clean content from a URL via Parallel.ai.

    Returns:
        {
          "url": str,
          "title": str,
          "meta_description": str,   # first excerpt if available
          "publish_date": str | None,
          "text": str,               # clean markdown body, truncated to max_chars
          "word_count": int,
          "error": str | None,
        }
    """
    result = {
        "url": url,
        "title": "",
        "meta_description": "",
        "publish_date": None,
        "text": "",
        "word_count": 0,
        "error": None,
    }

    try:
        client = _get_client()
        response = client.beta.extract(
            urls=[url],
            objective="Extract the full main article body for SEO and content quality analysis.",
            excerpts=True,
            full_content=True,
        )
    except Exception as exc:
        result["error"] = f"Parallel API error: {exc}"
        return result

    # Handle API-level errors
    if response.errors:
        err = response.errors[0]
        result["error"] = f"{err.error_type} (HTTP {err.http_status_code})"
        return result

    if not response.results:
        result["error"] = "No content returned by Parallel."
        return result

    page = response.results[0]

    result["title"] = page.title or ""
    result["publish_date"] = page.publish_date or None

    # Use full_content as primary source; fall back to joined excerpts
    body = page.full_content or ""
    if not body and page.excerpts:
        body = "\n\n".join(page.excerpts)

    # First excerpt as meta description proxy
    if page.excerpts:
        result["meta_description"] = page.excerpts[0][:300]

    cleaned = _clean_markdown(body)
    result["text"] = cleaned[:max_chars]
    result["word_count"] = len(cleaned.split())

    return result


def _clean_markdown(text: str) -> str:
    """Light cleanup of Parallel's markdown output for TRIBE v2 text input."""
    # Strip markdown image syntax (not useful for text encoder)
    text = re.sub(r"!\[.*?\]\(.*?\)", "", text)
    # Strip bare URLs
    text = re.sub(r"https?://\S+", "", text)
    # Collapse excessive blank lines
    text = re.sub(r"\n{3,}", "\n\n", text)
    # Strip leading/trailing whitespace per line
    lines = [line.strip() for line in text.splitlines()]
    # Drop lines that are too short to be content (nav fragments, etc.)
    lines = [l for l in lines if len(l) > 15 or l == ""]
    return "\n".join(lines).strip()
