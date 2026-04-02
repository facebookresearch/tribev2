"""
Neural Screenshot Analyzer.

Use TRIBE v2 on SERP/AI-chat screenshots (Google, ChatGPT, Perplexity, AI Mode):
- Convert each image to a short silent video
- Run TRIBE v2 via official video input path
- Use deterministic layout regions and score each element
"""
from pathlib import Path
import logging
import tempfile
import uuid

from PIL import Image, ImageDraw
from src.tribe_engine import predict_images_quick
from src.neural_scorer import activation_to_profile, compute_ctr_score

logger = logging.getLogger(__name__)


def _chunked(seq: list[str], size: int) -> list[list[str]]:
    return [seq[i : i + size] for i in range(0, len(seq), size)]


def _fallback_layout_elements() -> list[dict]:
    # Normalized (0..1000) layout fallback if vision detection fails.
    return [
        {"label": "Top query/navigation", "kind": "query", "bbox": {"x": 20, "y": 20, "w": 960, "h": 120}, "confidence": 0.3},
        {"label": "Primary result block", "kind": "result_block", "bbox": {"x": 40, "y": 160, "w": 920, "h": 170}, "confidence": 0.3},
        {"label": "Secondary result block", "kind": "result_block", "bbox": {"x": 40, "y": 350, "w": 920, "h": 170}, "confidence": 0.3},
        {"label": "Tertiary result block", "kind": "result_block", "bbox": {"x": 40, "y": 540, "w": 920, "h": 170}, "confidence": 0.3},
        {"label": "Lower result block", "kind": "result_block", "bbox": {"x": 40, "y": 730, "w": 920, "h": 170}, "confidence": 0.3},
    ]


def _crop_elements(
    image_path: str,
    elements: list[dict],
    max_elements: int = 8,
) -> tuple[list[dict], list[str]]:
    element_entries: list[dict] = []
    temp_paths: list[str] = []

    with Image.open(image_path) as img:
        img = img.convert("RGB")
        width, height = img.size
        for element in elements[:max_elements]:
            bbox = element.get("bbox", {})
            x = int((bbox.get("x", 0) / 1000) * width)
            y = int((bbox.get("y", 0) / 1000) * height)
            w = int((bbox.get("w", 0) / 1000) * width)
            h = int((bbox.get("h", 0) / 1000) * height)

            x = max(0, min(width - 1, x))
            y = max(0, min(height - 1, y))
            w = max(16, min(width - x, w))
            h = max(16, min(height - y, h))
            x2 = min(width, x + w)
            y2 = min(height, y + h)
            if x2 - x < 16 or y2 - y < 16:
                continue

            crop = img.crop((x, y, x2, y2))
            tmp_path = Path(tempfile.gettempdir()) / f"tribe_elem_{uuid.uuid4().hex}.png"
            crop.save(tmp_path, format="PNG")
            temp_paths.append(str(tmp_path))

            element_entries.append(
                {
                    "screenshot_name": Path(image_path).name,
                    "source_path": image_path,
                    "label": str(element.get("label", "UI element")),
                    "kind": str(element.get("kind", "other")),
                    "confidence": float(element.get("confidence", 0.5)),
                    "bbox_norm": {
                        "x": bbox.get("x", 0),
                        "y": bbox.get("y", 0),
                        "w": bbox.get("w", 0),
                        "h": bbox.get("h", 0),
                    },
                    "crop_path": str(tmp_path),
                }
            )

    return element_entries, temp_paths


def _score_color_rgb(score: float) -> tuple[int, int, int]:
    if score >= 0.75:
        return (34, 197, 94)   # green
    if score >= 0.50:
        return (251, 146, 60)  # orange
    return (244, 63, 94)       # red


def _build_scored_overlay(image_path: str, items: list[dict]) -> str:
    with Image.open(image_path) as img:
        img = img.convert("RGB")
        width, height = img.size
        draw = ImageDraw.Draw(img)

        for idx, item in enumerate(items):
            b = item.get("bbox_norm", {})
            x = int((b.get("x", 0) / 1000) * width)
            y = int((b.get("y", 0) / 1000) * height)
            w = int((b.get("w", 0) / 1000) * width)
            h = int((b.get("h", 0) / 1000) * height)

            x = max(0, min(width - 1, x))
            y = max(0, min(height - 1, y))
            w = max(16, min(width - x, w))
            h = max(16, min(height - y, h))
            x2 = x + w
            y2 = y + h

            score = float(item.get("attention_score", 0.0))
            color = _score_color_rgb(score)
            draw.rectangle((x, y, x2, y2), outline=color, width=4)

            label = f"{idx + 1}. {item.get('label', 'region')} {score * 100:.1f}%"
            try:
                text_box = draw.textbbox((0, 0), label)
                tw = text_box[2] - text_box[0]
                th = text_box[3] - text_box[1]
            except Exception:
                tw, th = 180, 14
            label_top = max(0, y - (th + 8))
            label_right = min(width - 1, x + tw + 12)
            draw.rectangle((x, label_top, label_right, y), fill=(15, 10, 30))
            draw.text((x + 6, label_top + 4), label, fill=color)

        out = Path(tempfile.gettempdir()) / f"tribe_overlay_{uuid.uuid4().hex}.png"
        img.save(out, format="PNG")
        return str(out)


def analyze_screenshots(image_paths: list[str]) -> dict:
    if not image_paths:
        return {
            "error": "No images provided.",
            "ranked_images": [],
            "ranked_elements": [],
            "summary": "",
            "overlay_gallery": [],
        }

    cleaned_paths = [str(p) for p in image_paths if p]
    if not cleaned_paths:
        return {
            "error": "No valid image paths provided.",
            "ranked_images": [],
            "ranked_elements": [],
            "summary": "",
            "overlay_gallery": [],
        }

    max_screenshots = 4
    max_elements_per_screen = 6
    cleaned_paths = cleaned_paths[:max_screenshots]

    all_elements: list[dict] = []
    temp_crop_paths: list[str] = []
    for path in cleaned_paths:
        detected = _fallback_layout_elements()

        entries, tmp_paths = _crop_elements(
            path,
            detected,
            max_elements=max_elements_per_screen,
        )
        all_elements.extend(entries)
        temp_crop_paths.extend(tmp_paths)

    if not all_elements:
        return {
            "error": "No screenshot elements could be extracted.",
            "ranked_images": [],
            "ranked_elements": [],
            "summary": "",
            "overlay_gallery": [],
        }

    # Keep each ZeroGPU request short to avoid quota-duration hard blocks.
    activations = []
    crop_paths = [entry["crop_path"] for entry in all_elements]
    for batch in _chunked(crop_paths, size=4):
        activations.extend(predict_images_quick(batch, image_duration_sec=1.5, fps=6))

    scored_elements = []
    for entry, activation in zip(all_elements, activations):
        name = entry["screenshot_name"]
        if activation is None:
            scored_elements.append(
                {
                    **entry,
                    "attention_score": 0.0,
                    "profile": {},
                    "error": "inference failed",
                }
            )
            continue

        try:
            profile = activation_to_profile(activation)
            score = compute_ctr_score(profile)
            scored_elements.append(
                {
                    **entry,
                    "attention_score": score,
                    "profile": {
                        "global_engagement": round(profile.global_engagement, 3),
                        "peak_salience": round(profile.peak_salience, 3),
                        "sustained_attention": round(profile.sustained_attention, 3),
                        "dmn_suppression": round(profile.dmn_suppression, 3),
                    },
                }
            )
        except Exception as exc:
            logger.warning("Screenshot score failed for '%s': %s", name, exc)
            scored_elements.append(
                {
                    **entry,
                    "attention_score": 0.0,
                    "profile": {},
                    "error": str(exc),
                }
            )

    # Cleanup temporary crops
    for tmp in temp_crop_paths:
        try:
            Path(tmp).unlink(missing_ok=True)
        except Exception:
            pass

    ranked_elements = sorted(scored_elements, key=lambda x: x["attention_score"], reverse=True)

    # Aggregate screenshot-level scores from element scores
    by_screen: dict[str, list[float]] = {}
    for item in ranked_elements:
        if "error" in item:
            continue
        by_screen.setdefault(item["screenshot_name"], []).append(item["attention_score"])

    ranked_images = []
    for screen_name, vals in by_screen.items():
        ranked_images.append(
            {
                "name": screen_name,
                "attention_score": sum(vals) / max(len(vals), 1),
                "element_count": len(vals),
            }
        )
    ranked_images.sort(key=lambda x: x["attention_score"], reverse=True)

    ok_elements = [x for x in ranked_elements if "error" not in x]
    overlay_gallery: list[tuple[str, str]] = []
    if not ok_elements:
        summary = "No screenshot elements could be scored by TRIBE v2."
    else:
        top_el = ok_elements[0]
        avg_score = sum(x["attention_score"] for x in ok_elements) / len(ok_elements)
        top_screen = ranked_images[0]["name"] if ranked_images else top_el["screenshot_name"]
        summary = f"Top element: **{top_el['label']}** in **{top_el['screenshot_name']}**. Top screenshot: **{top_screen}**. Average element score: **{avg_score * 100:.1f}%**."

        source_by_name: dict[str, str] = {}
        for item in ranked_elements:
            name = item.get("screenshot_name")
            src = item.get("source_path")
            if name and src and name not in source_by_name:
                source_by_name[name] = src

        for ranked in ranked_images:
            screen_name = ranked["name"]
            src = source_by_name.get(screen_name)
            if not src:
                continue
            screen_items = [
                x for x in ranked_elements
                if x.get("screenshot_name") == screen_name and "error" not in x
            ]
            if not screen_items:
                continue
            screen_items = sorted(screen_items, key=lambda x: x["attention_score"], reverse=True)
            try:
                overlay_path = _build_scored_overlay(src, screen_items)
            except Exception as exc:
                logger.warning("Overlay rendering failed for '%s': %s", screen_name, exc)
                continue
            caption = (
                f"{screen_name} | score {ranked['attention_score'] * 100:.1f}% | "
                f"{ranked['element_count']} regions"
            )
            overlay_gallery.append((overlay_path, caption))

    return {
        "error": None,
        "ranked_images": ranked_images,
        "ranked_elements": ranked_elements,
        "summary": summary,
        "overlay_gallery": overlay_gallery,
    }
