#!/usr/bin/env python3
"""Make synchronized display videos shorter and smoother."""

import argparse
import os
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path


def compress(path, output_fps, speed):
    temporary = path.with_suffix(path.suffix + ".tmp.mp4")
    vf = (
        f"minterpolate=fps={output_fps}:mi_mode=blend," 
        f"setpts={1.0 / speed:.8f}*PTS"
    )
    subprocess.run([
        "ffmpeg", "-y", "-loglevel", "error", "-i", str(path),
        "-vf", vf, "-an", "-c:v", "libx264", "-preset", "medium",
        "-crf", "18", "-pix_fmt", "yuv420p", "-r", str(output_fps),
        "-movflags", "+faststart", str(temporary),
    ], check=True)
    os.replace(temporary, path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, action="append", required=True)
    parser.add_argument("--fps", type=int, default=60)
    parser.add_argument("--speed", type=float, default=2.0)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument(
        "--name",
        default=None,
        help="Only process files with this basename (for example rgb_stream.mp4).",
    )
    args = parser.parse_args()
    videos = []
    for root in args.root:
        candidates = sorted(root.rglob("*.mp4"))
        videos.extend(
            path for path in candidates
            if args.name is None or path.name == args.name
        )
    if not videos:
        raise RuntimeError("No MP4 files found")
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(compress, path, args.fps, args.speed): path for path in videos}
        for future in as_completed(futures):
            path = futures[future]
            future.result()
            print(path, flush=True)
    print(f"compressed {len(videos)} videos to {args.fps} fps at {args.speed}x speed")


if __name__ == "__main__":
    main()
