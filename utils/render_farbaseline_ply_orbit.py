#!/usr/bin/env python3
"""Render a raw far-baseline Night3R RGB point cloud as a short orbit video."""

import argparse
import math
import subprocess
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from utils.export_dsec_darken_pointcloud_contrast import render_cloud


BACKGROUND = (11, 16, 23)
TEXT = (240, 244, 248)
MUTED = (166, 179, 192)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ply", type=Path, required=True)
    parser.add_argument("--merge-ply", type=Path, default=None)
    parser.add_argument("--temporal-ply", type=Path, default=None)
    parser.add_argument("--temporal-view-count", type=int, default=None)
    parser.add_argument("--max-source-view", type=int, default=None)
    parser.add_argument(
        "--merge-transform",
        type=float,
        nargs=16,
        default=None,
        help="Row-major rigid transform from the merge cloud's first camera to the primary cloud's first camera.",
    )
    parser.add_argument("--scene", required=True)
    parser.add_argument("--views", type=int, required=True)
    parser.add_argument("--pairs", type=int, required=True)
    parser.add_argument("--path-length", type=float, required=True)
    parser.add_argument("--outdir", type=Path, required=True)
    parser.add_argument("--frames", type=int, default=120)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--no-overlay", action="store_true")
    return parser.parse_args()


def load_ply(path):
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip() == "end_header":
                break
        rows = np.loadtxt(handle, dtype=np.float32)
    if rows.ndim != 2 or rows.shape[1] < 6:
        raise RuntimeError(f"Expected xyzrgb ASCII PLY: {path}")
    valid = np.isfinite(rows[:, :3]).all(axis=1)
    return rows[valid, :3], np.clip(rows[valid, 3:6], 0, 255).astype(np.uint8)


def temporal_frame_ids(path, view_count):
    _, colors = load_ply(path)
    palette = (
        plt.cm.turbo(np.arange(view_count, dtype=np.float32) / max(view_count - 1, 1))[:, :3]
        * 255.0
        + 0.5
    ).astype(np.uint8)
    distances = (colors[:, None, :].astype(np.int32) - palette[None, :, :].astype(np.int32)) ** 2
    return np.argmin(distances.sum(axis=2), axis=1)


def font(size, bold=False):
    name = "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf"
    path = Path("/usr/share/fonts/truetype/dejavu") / name
    return ImageFont.truetype(str(path), size) if path.is_file() else ImageFont.load_default()


def render_frame(points, colors, angle, args):
    low, high = np.percentile(points, [1.0, 99.0], axis=0)
    center = 0.5 * (low + high)
    rotation = np.asarray(
        [[math.cos(angle), 0.0, math.sin(angle)], [0.0, 1.0, 0.0], [-math.sin(angle), 0.0, math.cos(angle)]],
        dtype=np.float32,
    )
    rotated = (points - center) @ rotation.T + center
    image = Image.fromarray(
        render_cloud(rotated, rgb_colors=colors, color_mode="rgb", width=1280, height=720, point_radius=1)
    )
    if not args.no_overlay:
        draw = ImageDraw.Draw(image)
        draw.text((30, 24), "Night3R far-baseline reconstruction", fill=TEXT, font=font(28, True))
        draw.text((30, 64), args.scene, fill=MUTED, font=font(19))
        draw.text(
            (30, 680),
            f"{args.views} views | {args.pairs} far-baseline pairs | {args.path_length:.2f} m camera path | raw geometry | GT-pose fusion",
            fill=MUTED,
            font=font(17),
        )
    return image


def main():
    args = parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)
    frames = args.outdir / "frames"
    frames.mkdir(exist_ok=True)
    points, colors = load_ply(args.ply)
    if args.temporal_ply is not None:
        if args.temporal_view_count is None or args.max_source_view is None:
            raise ValueError("Temporal filtering requires --temporal-view-count and --max-source-view")
        frame_ids = temporal_frame_ids(args.temporal_ply, args.temporal_view_count)
        if len(frame_ids) != len(points):
            raise RuntimeError("Temporal and RGB PLY vertex counts differ")
        keep = frame_ids < args.max_source_view
        points, colors = points[keep], colors[keep]
    if args.merge_ply is not None:
        if args.merge_transform is None:
            raise ValueError("--merge-ply requires --merge-transform")
        merge_points, merge_colors = load_ply(args.merge_ply)
        transform = np.asarray(args.merge_transform, dtype=np.float32).reshape(4, 4)
        # Long-sequence PLYs use a display-only Y flip. Undo it before applying
        # the dataset camera-to-camera rigid transform, then restore the flip.
        display_flip = np.asarray([1.0, -1.0, 1.0], dtype=np.float32)
        merge_points = merge_points * display_flip
        merge_points = merge_points @ transform[:3, :3].T + transform[:3, 3]
        merge_points = merge_points * display_flip
        points = np.concatenate((points, merge_points), axis=0)
        colors = np.concatenate((colors, merge_colors), axis=0)
    for index in range(args.frames):
        phase = index / max(args.frames - 1, 1)
        # A restrained 24-degree orbit keeps both road layout and facades visible.
        angle = math.radians(24.0) * math.sin(2.0 * math.pi * phase)
        render_frame(points, colors, angle, args).save(frames / f"frame_{index:04d}.png", compress_level=2)
    output = args.outdir / "night3r_farbaseline_raw_rgb.mp4"
    subprocess.run(
        [
            "ffmpeg", "-y", "-loglevel", "error", "-framerate", str(args.fps),
            "-i", str(frames / "frame_%04d.png"), "-c:v", "libx264", "-preset", "slow",
            "-crf", "18", "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(output),
        ],
        check=True,
    )
    print(output)


if __name__ == "__main__":
    main()
