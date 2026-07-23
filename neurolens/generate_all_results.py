"""Automated results generator for all NeuroLens modules.

Generates brain surface plots, match results, eval leaderboards, and radar
charts for all meaningful parameter combinations.  Reads from the pre-computed
neurolens_cache/ and writes structured output to neurolens_results/.

Usage:
    python -m neurolens.generate_all_results [--cache-dir PATH] [--output-dir PATH]
"""

from __future__ import annotations

import argparse
import json
import itertools
from datetime import datetime, timezone
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

from neurolens.cache import CacheManager
from neurolens.stimulus import StimulusLibrary
from neurolens.predict import get_prediction_at_time, get_num_timesteps, get_top_rois
from neurolens.match import find_similar_stimuli, find_contrast_stimuli
from neurolens.eval import (
    compute_all_model_alignments,
    compute_model_brain_alignment,
)
from neurolens.roi import get_roi_group_names
from neurolens.viz import plot_brain_surface, make_radar_chart

VIEWS = ["left", "right", "medial_left", "medial_right", "dorsal"]


def _save_fig(fig: matplotlib.figure.Figure, path: Path, dpi: int = 150) -> None:
    """Save figure and close it to free memory."""
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def _key_frames(n_timesteps: int) -> list[int]:
    """Return first, middle, last timestep indices."""
    if n_timesteps <= 1:
        return [0]
    if n_timesteps == 2:
        return [0, 1]
    return [0, n_timesteps // 2, n_timesteps - 1]


# ── Module 1: Predict ──────────────────────────────────────────────────────


def generate_predict(
    cache: CacheManager,
    library: StimulusLibrary,
    output_dir: Path,
) -> dict:
    """Generate brain surface plots and ROI summaries for all stimuli."""
    predict_dir = output_dir / "predict"
    results = {"stimuli": {}}

    stimuli = library.all()
    total_plots = 0
    for stim in stimuli:
        n_ts = get_num_timesteps(cache, stim.id)
        total_plots += len(_key_frames(n_ts)) * len(VIEWS)

    pbar = tqdm(total=total_plots, desc="Predict: brain plots")

    for stim in stimuli:
        stim_dir = predict_dir / stim.id
        stim_dir.mkdir(parents=True, exist_ok=True)

        n_ts = get_num_timesteps(cache, stim.id)
        key_frames = _key_frames(n_ts)

        stim_result = {
            "name": stim.name,
            "category": stim.category,
            "n_timesteps": n_ts,
            "key_frames": key_frames,
            "brain_plots": [],
            "roi_summary": {},
        }

        # Brain surface plots: each view saved individually
        for t in key_frames:
            pred = get_prediction_at_time(cache, stim.id, t)
            for view in VIEWS:
                fig = plot_brain_surface(
                    pred,
                    views=[view],
                    title=f"{stim.name} (t={t}) — {view}",
                )
                fname = f"brain_t{t:02d}_{view}.png"
                _save_fig(fig, stim_dir / fname)
                stim_result["brain_plots"].append(fname)
                pbar.update(1)

        # ROI summary (time-averaged)
        top_rois = get_top_rois(cache, stim.id, k=9)
        roi_data = {name: score for name, score in top_rois}
        stim_result["roi_summary"] = roi_data
        (stim_dir / "roi_summary.json").write_text(json.dumps(roi_data, indent=2))

        results["stimuli"][stim.id] = stim_result

    pbar.close()
    return results


# ── Module 2: Match ─────────────────────────────────────────────────────────


def generate_match_more_like_this(
    cache: CacheManager,
    library: StimulusLibrary,
    output_dir: Path,
) -> dict:
    """Generate 'More Like This' results for each stimulus."""
    mlt_dir = output_dir / "match" / "more_like_this"
    mlt_dir.mkdir(parents=True, exist_ok=True)
    results = {}

    stimulus_ids = library.ids()

    for stim in tqdm(library.all(), desc="Match: more like this"):
        # Use time-averaged prediction as target
        preds = cache.load_brain_preds(stim.id)
        target = preds.mean(axis=0)

        matches = find_similar_stimuli(
            cache, target, stimulus_ids, top_k=5, time_aggregation="mean"
        )

        # Build match data with names
        match_data = []
        for sid, score in matches:
            s = library.get(sid)
            match_data.append({
                "stimulus_id": sid,
                "name": s.name if s else sid,
                "category": s.category if s else "unknown",
                "similarity": round(score, 4),
            })

        (mlt_dir / f"{stim.id}_matches.json").write_text(
            json.dumps(match_data, indent=2)
        )

        # Radar chart for top 3
        radar_data = {}
        for sid, _ in matches[:3]:
            s = library.get(sid)
            label = s.name if s else sid
            roi_summary = cache.load_roi_summary(sid)
            if roi_summary:
                radar_data[label] = roi_summary

        if len(radar_data) >= 2:
            fig = make_radar_chart(radar_data, title=f"Similar to: {stim.name}")
            _save_fig(fig, mlt_dir / f"{stim.id}_radar.png")

        results[stim.id] = {
            "source": stim.name,
            "matches": match_data,
        }

    return results


def generate_match_contrast(
    cache: CacheManager,
    library: StimulusLibrary,
    output_dir: Path,
) -> dict:
    """Generate contrast results for all directed ROI pairs."""
    contrast_dir = output_dir / "match" / "contrast"
    contrast_dir.mkdir(parents=True, exist_ok=True)
    results = {}

    roi_names = get_roi_group_names()
    stimulus_ids = library.ids()

    pairs = [(a, b) for a, b in itertools.permutations(roi_names, 2)]

    for max_roi, min_roi in tqdm(pairs, desc="Match: contrast pairs"):
        matches = find_contrast_stimuli(
            cache, stimulus_ids, max_roi, min_roi, top_k=5
        )

        match_data = []
        for sid, score in matches:
            s = library.get(sid)
            match_data.append({
                "stimulus_id": sid,
                "name": s.name if s else sid,
                "category": s.category if s else "unknown",
                "contrast_score": round(score, 4),
            })

        safe_max = max_roi.replace(" ", "_")
        safe_min = min_roi.replace(" ", "_")
        prefix = f"max_{safe_max}_min_{safe_min}"

        (contrast_dir / f"{prefix}.json").write_text(
            json.dumps(match_data, indent=2)
        )

        # Radar chart for top 3
        radar_data = {}
        for sid, _ in matches[:3]:
            s = library.get(sid)
            label = s.name if s else sid
            roi_summary = cache.load_roi_summary(sid)
            if roi_summary:
                radar_data[label] = roi_summary

        if len(radar_data) >= 2:
            fig = make_radar_chart(
                radar_data,
                title=f"Contrast: {max_roi} > {min_roi}",
            )
            _save_fig(fig, contrast_dir / f"{prefix}_radar.png")

        results[prefix] = {
            "maximize": max_roi,
            "minimize": min_roi,
            "matches": match_data,
        }

    return results


# ── Module 3: Eval ──────────────────────────────────────────────────────────


def generate_eval(
    cache: CacheManager,
    library: StimulusLibrary,
    output_dir: Path,
) -> dict:
    """Generate leaderboard and pairwise model comparisons."""
    eval_dir = output_dir / "eval"
    eval_dir.mkdir(parents=True, exist_ok=True)

    stimulus_ids = library.ids()
    models = cache.available_models()

    # Leaderboard
    print("Eval: computing leaderboard...")
    alignments = compute_all_model_alignments(cache, stimulus_ids)

    leaderboard = []
    for rank, (model, score) in enumerate(alignments.items(), 1):
        leaderboard.append({
            "rank": rank,
            "model": model,
            "rsa_score": round(score, 4),
            "brain_alignment_pct": round(max(0, score) * 100, 1),
        })

    (eval_dir / "leaderboard.json").write_text(json.dumps(leaderboard, indent=2))

    # Leaderboard bar chart
    fig, ax = plt.subplots(figsize=(8, 5))
    model_names = [e["model"] for e in leaderboard]
    scores = [e["rsa_score"] for e in leaderboard]
    colors = plt.cm.Set2(np.linspace(0, 1, len(model_names)))
    bars = ax.bar(model_names, scores, color=colors, edgecolor="black", linewidth=0.5)
    ax.set_ylabel("RSA Score (Brain Alignment)")
    ax.set_title("Model Leaderboard: Brain Alignment via RSA")
    ax.axhline(y=0, color="gray", linestyle="--", linewidth=0.5)
    for bar, score in zip(bars, scores):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.01,
            f"{score:.4f}",
            ha="center",
            va="bottom",
            fontsize=10,
        )
    _save_fig(fig, eval_dir / "leaderboard.png")

    # Pairwise comparisons
    comparisons = {}
    for m_a, m_b in itertools.combinations(models, 2):
        score_a = compute_model_brain_alignment(cache, m_a, stimulus_ids)
        score_b = compute_model_brain_alignment(cache, m_b, stimulus_ids)
        pair_data = {
            "model_a": m_a,
            "model_b": m_b,
            "rsa_a": round(score_a, 4),
            "rsa_b": round(score_b, 4),
            "alignment_pct_a": round(max(0, score_a) * 100, 1),
            "alignment_pct_b": round(max(0, score_b) * 100, 1),
            "winner": m_a if score_a > score_b else m_b,
        }
        key = f"{m_a}_vs_{m_b}"
        (eval_dir / f"compare_{key}.json").write_text(json.dumps(pair_data, indent=2))
        comparisons[key] = pair_data
        print(f"  {m_a} ({score_a:.4f}) vs {m_b} ({score_b:.4f})")

    return {"leaderboard": leaderboard, "comparisons": comparisons}


# ── Main ────────────────────────────────────────────────────────────────────


def generate_all(cache_dir: str | Path, output_dir: str | Path) -> Path:
    """Run all modules and write results to output_dir."""
    cache_dir = Path(cache_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    cache = CacheManager(cache_dir)
    library = StimulusLibrary(cache_dir)

    print(f"NeuroLens Results Generator")
    print(f"  Cache: {cache_dir}")
    print(f"  Output: {output_dir}")
    print(f"  Stimuli: {len(library)}")
    print(f"  Models: {cache.available_models()}")
    print()

    summary = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "cache_dir": str(cache_dir),
        "stimuli_count": len(library),
        "models": cache.available_models(),
        "modules": {},
    }

    # Module 1: Predict
    print("=" * 60)
    print("MODULE 1: PREDICT")
    print("=" * 60)
    predict_results = generate_predict(cache, library, output_dir)
    summary["modules"]["predict"] = {
        "stimuli": len(predict_results["stimuli"]),
        "total_brain_plots": sum(
            len(s["brain_plots"])
            for s in predict_results["stimuli"].values()
        ),
    }
    print()

    # Module 2: Match
    print("=" * 60)
    print("MODULE 2: MATCH")
    print("=" * 60)
    mlt_results = generate_match_more_like_this(cache, library, output_dir)
    contrast_results = generate_match_contrast(cache, library, output_dir)
    summary["modules"]["match"] = {
        "more_like_this": len(mlt_results),
        "contrast_pairs": len(contrast_results),
    }
    print()

    # Module 3: Eval
    print("=" * 60)
    print("MODULE 3: EVAL")
    print("=" * 60)
    eval_results = generate_eval(cache, library, output_dir)
    summary["modules"]["eval"] = {
        "leaderboard_models": len(eval_results["leaderboard"]),
        "comparisons": len(eval_results["comparisons"]),
    }
    print()

    # Write master summary
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2))

    # Count output files
    all_files = list(output_dir.rglob("*"))
    n_png = len([f for f in all_files if f.suffix == ".png"])
    n_json = len([f for f in all_files if f.suffix == ".json"])
    print("=" * 60)
    print(f"DONE! Generated {n_png} PNGs + {n_json} JSONs")
    print(f"Output: {output_dir}")
    print("=" * 60)

    return output_dir


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate all NeuroLens results")
    parser.add_argument(
        "--cache-dir",
        default="neurolens_cache",
        help="Path to neurolens_cache/ directory",
    )
    parser.add_argument(
        "--output-dir",
        default="neurolens_results",
        help="Path to output directory",
    )
    args = parser.parse_args()
    generate_all(args.cache_dir, args.output_dir)


if __name__ == "__main__":
    main()
