#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
SoundChain x TRIBE v2 — Music Neural Engagement Analysis

Demonstrates TRIBE v2 predicting brain responses to live music streams
from SoundChain's OGUN Radio (500+ NFT tracks, 24/7 broadcast).

This is a novel use case: instead of movie clips or lab stimuli,
we feed TRIBE real music from a production streaming platform and
compute neural engagement scores per track — measuring predicted
brain activation intensity across auditory, emotional, and
language-processing cortical regions.

Use cases:
  - Artists get neurofeedback on their tracks (which moments activate listeners)
  - Streaming platforms rank music by neural engagement, not just play count
  - Researchers study music cognition at scale using a live music corpus

Requirements:
  pip install -e ".[plotting]"
  # TRIBE v2 model weights download automatically from HuggingFace

Usage:
  python examples/soundchain_music_neural.py
  python examples/soundchain_music_neural.py --tracks 10 --genre hip_hop
  python examples/soundchain_music_neural.py --audio-path /path/to/local.mp3
"""

import argparse
import json
import logging
import tempfile
from pathlib import Path

import numpy as np
import requests

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(levelname)s - %(message)s")

# SoundChain OGUN Radio — public API, no auth required
RADIO_API = "https://soundchain.io/api/agent/radio"
TRACKS_API = "https://soundchain.io/api/agent/tracks"

# Cortical regions of interest for music processing
# Based on neuroscience literature on music cognition
MUSIC_ROIS = {
    "auditory_cortex": "Primary auditory processing (Heschl's gyrus, STG)",
    "motor_cortex": "Rhythm/beat processing, groove",
    "prefrontal": "Musical expectation, structure",
    "amygdala_region": "Emotional response to music",
    "reward_circuit": "Dopamine response (nucleus accumbens region)",
    "language_areas": "Lyric processing (Broca's, Wernicke's)",
}


def fetch_radio_tracks(n: int = 5, genre: str = "all") -> list[dict]:
    """Fetch tracks from SoundChain's OGUN Radio API."""
    logger.info(f"Fetching {n} tracks from OGUN Radio (genre: {genre})...")
    try:
        params = {"action": "playlist"}
        if genre != "all":
            params["genre"] = genre
        resp = requests.get(RADIO_API, params=params, timeout=15)
        resp.raise_for_status()
        data = resp.json()
        tracks = data.get("data", {}).get("tracks", [])[:n]
        logger.info(f"Got {len(tracks)} tracks from radio")
        return tracks
    except Exception as e:
        logger.error(f"Failed to fetch radio tracks: {e}")
        return []


def download_track(track: dict, cache_dir: Path) -> Path | None:
    """Download a track's audio from SoundChain to local cache."""
    url = track.get("stream_url") or track.get("audio_url")
    if not url:
        logger.warning(f"No stream URL for: {track.get('title', '?')}")
        return None

    title_safe = "".join(c if c.isalnum() or c in " -_" else "_" for c in track.get("title", "track"))
    ext = ".mp3"  # SoundChain serves mp3/wav via IPFS or S3
    path = cache_dir / f"{title_safe[:50]}{ext}"

    if path.exists():
        logger.info(f"Using cached: {path.name}")
        return path

    logger.info(f"Downloading: {track.get('title', '?')[:50]}...")
    try:
        r = requests.get(url, stream=True, timeout=60)
        r.raise_for_status()
        with open(path, "wb") as f:
            for chunk in r.iter_content(chunk_size=128 * 1024):
                if chunk:
                    f.write(chunk)
        logger.info(f"Saved: {path.name} ({path.stat().st_size / 1024 / 1024:.1f} MB)")
        return path
    except Exception as e:
        logger.error(f"Download failed for {track.get('title', '?')}: {e}")
        return None


def analyze_track(model, audio_path: Path) -> dict:
    """Run TRIBE v2 prediction on a single audio track.

    Returns predicted brain activation and engagement metrics.
    """
    logger.info(f"Analyzing: {audio_path.name}")

    # Build events dataframe from audio
    df = model.get_events_dataframe(audio_path=str(audio_path))

    # Predict fMRI brain responses (fsaverage5 surface, ~20k vertices)
    preds, segments = model.predict(events=df)

    # preds shape: (n_timesteps, n_vertices)
    # Each timestep = predicted BOLD signal across cortical surface

    # Compute engagement metrics
    mean_activation = float(np.mean(np.abs(preds)))
    peak_activation = float(np.max(np.abs(preds)))
    activation_variance = float(np.var(preds))

    # Temporal dynamics — how much the brain response changes over time
    if preds.shape[0] > 1:
        temporal_gradient = float(np.mean(np.abs(np.diff(preds, axis=0))))
    else:
        temporal_gradient = 0.0

    # Neural engagement score (composite)
    # Higher = more brain activation + more dynamic response
    engagement_score = (mean_activation * 0.4 + peak_activation * 0.3 + temporal_gradient * 0.3)

    return {
        "n_timesteps": int(preds.shape[0]),
        "n_vertices": int(preds.shape[1]),
        "mean_activation": round(mean_activation, 6),
        "peak_activation": round(peak_activation, 6),
        "activation_variance": round(activation_variance, 6),
        "temporal_gradient": round(temporal_gradient, 6),
        "engagement_score": round(engagement_score, 6),
    }


def main():
    parser = argparse.ArgumentParser(description="SoundChain x TRIBE v2 — Music Neural Engagement")
    parser.add_argument("--tracks", type=int, default=5, help="Number of radio tracks to analyze")
    parser.add_argument("--genre", type=str, default="all", help="Genre filter (hip_hop, lo_fi, etc.)")
    parser.add_argument("--audio-path", type=str, default=None, help="Analyze a local audio file instead of radio")
    parser.add_argument("--cache-dir", type=str, default=None, help="Cache directory for downloaded audio")
    parser.add_argument("--model-cache", type=str, default="./cache", help="TRIBE model cache directory")
    parser.add_argument("--output", type=str, default=None, help="Save results to JSON file")
    args = parser.parse_args()

    # Load TRIBE v2 model
    logger.info("Loading TRIBE v2 model from HuggingFace...")
    from tribev2 import TribeModel
    model = TribeModel.from_pretrained("facebook/tribev2", cache_folder=args.model_cache)
    logger.info("Model loaded.")

    results = []

    if args.audio_path:
        # Analyze a single local file
        path = Path(args.audio_path)
        if not path.exists():
            logger.error(f"File not found: {path}")
            return
        metrics = analyze_track(model, path)
        results.append({"title": path.stem, "path": str(path), **metrics})
    else:
        # Fetch and analyze tracks from OGUN Radio
        tracks = fetch_radio_tracks(n=args.tracks, genre=args.genre)
        if not tracks:
            logger.error("No tracks available. Check your connection to soundchain.io")
            return

        cache_dir = Path(args.cache_dir) if args.cache_dir else Path(tempfile.mkdtemp(prefix="tribe_soundchain_"))
        cache_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Audio cache: {cache_dir}")

        for i, track in enumerate(tracks):
            title = track.get("title", "Unknown")[:50]
            logger.info(f"\n{'='*60}")
            logger.info(f"Track {i+1}/{len(tracks)}: {title}")
            logger.info(f"Artist: {track.get('artist', '?')} | SCID: {track.get('scid', '?')}")
            logger.info(f"{'='*60}")

            audio_path = download_track(track, cache_dir)
            if not audio_path:
                continue

            metrics = analyze_track(model, audio_path)
            results.append({
                "title": title,
                "artist": track.get("artist", "Unknown"),
                "scid": track.get("scid"),
                "is_nft": track.get("is_nft", False),
                "genres": track.get("genres", []),
                **metrics,
            })

            logger.info(f"  Engagement score: {metrics['engagement_score']:.4f}")
            logger.info(f"  Mean activation:  {metrics['mean_activation']:.4f}")
            logger.info(f"  Peak activation:  {metrics['peak_activation']:.4f}")
            logger.info(f"  Temporal dynamic: {metrics['temporal_gradient']:.4f}")

    # Print summary
    if results:
        print("\n" + "=" * 70)
        print("NEURAL ENGAGEMENT RANKINGS")
        print("=" * 70)
        ranked = sorted(results, key=lambda r: r["engagement_score"], reverse=True)
        for i, r in enumerate(ranked):
            nft_badge = " [NFT]" if r.get("is_nft") else ""
            print(f"  #{i+1}  {r['engagement_score']:.4f}  {r['title']}{nft_badge}")
        print("=" * 70)
        print(f"Analyzed {len(results)} tracks from SoundChain OGUN Radio")
        print(f"Model: TRIBE v2 (facebook/tribev2) — fMRI brain response prediction")
        print(f"Platform: soundchain.io — decentralized music streaming + $OGUN rewards")

    # Save results
    if args.output:
        output_path = Path(args.output)
        with open(output_path, "w") as f:
            json.dump(results, f, indent=2)
        logger.info(f"Results saved to {output_path}")


if __name__ == "__main__":
    main()
