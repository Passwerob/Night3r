#!/usr/bin/env python3
import argparse
import json
import subprocess
from pathlib import Path

import cv2
import imageio.v2 as imageio
import numpy as np
from PIL import Image


def parse_args():
    parser = argparse.ArgumentParser("Stabilize the right reconstruction panel in rendered demo frames")
    parser.add_argument("--input-frames", required=True)
    parser.add_argument("--outdir", required=True)
    parser.add_argument("--fps", type=int, default=10)
    parser.add_argument("--left-width", type=int, default=384)
    parser.add_argument("--ema", type=float, default=0.88)
    parser.add_argument("--max-shift", type=int, default=42)
    parser.add_argument("--max-step", type=int, default=5)
    parser.add_argument("--min-pixels", type=int, default=2000)
    parser.add_argument("--video-name", default="stabilized.mp4")
    return parser.parse_args()


def panel_anchor(panel):
    gray = panel.max(axis=2)
    mask = gray > 10
    if int(mask.sum()) < 2000:
        return None
    ys, xs = np.nonzero(mask)
    xlo, xhi = np.percentile(xs, [5, 95])
    ylo, yhi = np.percentile(ys, [5, 95])
    keep = (xs >= xlo) & (xs <= xhi) & (ys >= ylo) & (ys <= yhi)
    if int(keep.sum()) < 2000:
        return None
    return float(np.median(ys[keep]))


def shift_panel(panel, shift):
    h, w = panel.shape[:2]
    shifted = np.zeros_like(panel)
    mat = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, shift]], dtype=np.float32)
    return cv2.warpAffine(panel, mat, (w, h), dst=shifted, flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)


def encode_video(frames_dir, output_path, fps):
    frame_paths = sorted(Path(frames_dir).glob("frame_*.png"))
    try:
        with imageio.get_writer(str(output_path), fps=fps, codec="libx264", quality=8, macro_block_size=16) as writer:
            for frame_path in frame_paths:
                writer.append_data(imageio.imread(frame_path))
    except Exception:
        subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-framerate",
                str(fps),
                "-i",
                str(Path(frames_dir) / "frame_%04d.png"),
                "-vf",
                "pad=ceil(iw/2)*2:ceil(ih/2)*2",
                "-c:v",
                "libx264",
                "-pix_fmt",
                "yuv420p",
                "-movflags",
                "+faststart",
                str(output_path),
            ],
            check=True,
        )


def main():
    args = parse_args()
    input_frames = sorted(Path(args.input_frames).glob("frame_*.png"))
    if not input_frames:
        raise RuntimeError(f"No frames found in {args.input_frames}")

    outdir = Path(args.outdir)
    frames_dir = outdir / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)
    for stale in frames_dir.glob("frame_*.png"):
        stale.unlink()

    anchors = []
    for frame_path in input_frames:
        frame = np.asarray(Image.open(frame_path).convert("RGB"))
        anchors.append(panel_anchor(frame[:, args.left_width :]))

    valid = np.array([a is not None for a in anchors], dtype=bool)
    values = np.array([0.0 if a is None else a for a in anchors], dtype=np.float32)
    if valid.any():
        idx = np.arange(len(values))
        values[~valid] = np.interp(idx[~valid], idx[valid], values[valid])
        target = float(values[valid][0])
    else:
        target = 0.0

    smooth_anchor = target
    prev_shift = 0.0
    records = []
    for idx, frame_path in enumerate(input_frames):
        frame = np.asarray(Image.open(frame_path).convert("RGB"))
        left = frame[:, : args.left_width]
        panel = frame[:, args.left_width :]
        raw_anchor = float(values[idx])
        smooth_anchor = args.ema * smooth_anchor + (1.0 - args.ema) * raw_anchor
        wanted_shift = target - smooth_anchor
        wanted_shift = float(np.clip(wanted_shift, -args.max_shift, args.max_shift))
        wanted_shift = float(np.clip(wanted_shift, prev_shift - args.max_step, prev_shift + args.max_step))
        prev_shift = wanted_shift
        out = np.concatenate([left, shift_panel(panel, wanted_shift)], axis=1)
        Image.fromarray(out).save(frames_dir / f"frame_{idx:04d}.png")
        records.append(
            {
                "frame": idx,
                "raw_anchor_y": raw_anchor,
                "smooth_anchor_y": float(smooth_anchor),
                "shift_y": wanted_shift,
            }
        )

    video_path = outdir / args.video_name
    encode_video(frames_dir, video_path, args.fps)
    manifest = {
        "input_frames": str(args.input_frames),
        "num_frames": len(input_frames),
        "fps": args.fps,
        "left_width": args.left_width,
        "ema": args.ema,
        "max_shift": args.max_shift,
        "max_step": args.max_step,
        "video": args.video_name,
        "frames": records,
    }
    (outdir / "stabilization_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps({"outdir": str(outdir), "video": str(video_path), "frames": len(input_frames)}, indent=2))


if __name__ == "__main__":
    main()
