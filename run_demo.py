"""Minimal TRIBE v2 inference demo.

Predicts whole-brain fMRI responses (fsaverage5 cortical mesh, ~20k vertices)
for a video clip. Optionally accepts a separate audio or text file.

Usage:
    python run_demo.py --video path/to/clip.mp4
    python run_demo.py --text  "some caption or transcript"

First run downloads the TRIBE checkpoint (~709 MB) plus the underlying
video/audio/text encoder backbones into ./cache.
"""

import argparse
import os

import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser(description="TRIBE v2 brain-response demo")
    parser.add_argument("--video", type=str, default=None, help="path to a video file")
    parser.add_argument("--audio", type=str, default=None, help="path to an audio file")
    parser.add_argument("--text", type=str, default=None, help="raw caption/transcript text")
    parser.add_argument("--cache", type=str, default="./cache", help="model cache folder")
    parser.add_argument("--out", type=str, default="preds.npy", help="where to save predictions")
    args = parser.parse_args()

    if not any([args.video, args.audio, args.text]):
        parser.error("provide at least one of --video / --audio / --text")

    # Fail fast with a clear message instead of crashing deep in a native
    # video/audio reader when a path is wrong (a missing path can segfault).
    for label, path in (("--video", args.video), ("--audio", args.audio)):
        if path and not os.path.isfile(path):
            parser.error(f"{label} file not found: {path}")

    from tribev2 import TribeModel

    print("Loading TRIBE v2 (downloads on first run)...")
    model = TribeModel.from_pretrained("facebook/tribev2", cache_folder=args.cache)

    events_kwargs = {}
    if args.video:
        events_kwargs["video_path"] = args.video
    if args.audio:
        events_kwargs["audio_path"] = args.audio
    if args.text:
        # write text to a temp file if the API expects a path; else pass through
        text_path = os.path.join(args.cache, "_input_text.txt")
        os.makedirs(args.cache, exist_ok=True)
        with open(text_path, "w") as f:
            f.write(args.text)
        events_kwargs["text_path"] = text_path

    print(f"Building events dataframe from: {list(events_kwargs)}")
    df = model.get_events_dataframe(**events_kwargs)

    print("Predicting brain responses...")
    preds, segments = model.predict(events=df)

    preds_np = preds.detach().cpu().numpy() if hasattr(preds, "detach") else np.asarray(preds)
    np.save(args.out, preds_np)

    print("-" * 50)
    print(f"Prediction shape: {preds_np.shape}  (n_timesteps, n_vertices)")
    print(f"Value range: [{preds_np.min():.3f}, {preds_np.max():.3f}]")
    print(f"Saved predictions to: {args.out}")
    print("Vertices live on the fsaverage5 cortical mesh (~20k vertices).")


if __name__ == "__main__":
    main()
