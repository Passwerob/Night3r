#!/usr/bin/env python3
"""Render a camera-following view of an already fused c84 global point cloud."""

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

BACKGROUND = (8, 13, 20)
PANEL = (12, 19, 29)
TEXT = (240, 244, 248)
MUTED = (165, 177, 191)


def load_font(size, bold=False):
    name = "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf"
    path = Path("/usr/share/fonts/truetype/dejavu") / name
    return ImageFont.truetype(str(path), size) if path.is_file() else ImageFont.load_default()


def event_thumbnail(event):
    event = event.detach().float().cpu()
    half = event.shape[0] // 2
    pos = event[:half].abs().sum(dim=0)
    neg = event[half:].abs().sum(dim=0)

    def normalize(value):
        nonzero = value[value > 0]
        if not nonzero.numel():
            return torch.zeros_like(value)
        scale = torch.quantile(torch.log1p(nonzero), 0.995).clamp_min(1.0e-6)
        return (torch.log1p(value) / scale).clamp(0, 1)

    packed = torch.stack((normalize(pos), torch.zeros_like(pos), normalize(neg)), dim=-1)
    return Image.fromarray((packed.numpy() * 255.0 + 0.5).astype(np.uint8))


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cloud-npz", type=Path, required=True)
    parser.add_argument("--trajectory-npz", type=Path, required=True)
    parser.add_argument("--sequence-root", type=Path, required=True)
    parser.add_argument("--num-frames", type=int, required=True)
    parser.add_argument("--history", type=int, default=12)
    parser.add_argument("--fps", type=int, default=12)
    parser.add_argument("--outdir", type=Path, required=True)
    return parser.parse_args()


def elevated_camera(points, frame_index, total):
    """Build a stable elevated virtual camera in the current real-camera frame."""
    lower = np.percentile(points, 1.0, axis=0)
    upper = np.percentile(points, 99.0, axis=0)
    target = (lower + upper) * 0.5
    extent = max(float(np.linalg.norm(upper - lower)), 1.0e-6)

    # x is right, y is image-down, z is forward.  Thus negative y moves the
    # virtual camera upward.  Keep lateral offset small so the learned front-view
    # point map remains legible, while the elevated/backward offset reveals depth.
    phase = 2.0 * math.pi * frame_index / max(total - 1, 1)
    camera = target + np.asarray(
        (0.11 + 0.025 * math.sin(phase), -0.24, -1.20), dtype=np.float32
    ) * extent
    forward = target - camera
    forward /= max(float(np.linalg.norm(forward)), 1.0e-6)
    up_hint = np.asarray((0.0, -1.0, 0.0), dtype=np.float32)
    right = np.cross(forward, up_hint)
    right /= max(float(np.linalg.norm(right)), 1.0e-6)
    up = np.cross(right, forward)
    return camera, right, up, forward


def render_elevated_cloud(points, colors, frame_index, total, width=1260, height=850):
    """Render a mild third-person, elevated view of registered RGB point maps."""
    canvas = np.broadcast_to(np.asarray(BACKGROUND, dtype=np.uint8), (height, width, 3)).copy()
    if not len(points):
        return canvas

    camera, right, up, forward = elevated_camera(points, frame_index, total)
    relative = points - camera[None, :]
    x, y, z = relative @ right, relative @ up, relative @ forward
    valid = np.isfinite(points).all(axis=1) & (z > 1.0e-4)
    if not np.any(valid):
        return canvas

    focal = 0.5 * width / math.tan(math.radians(46.0) * 0.5)
    u = focal * x / np.maximum(z, 1.0e-6) + 0.5 * width
    v = 0.5 * height - focal * y / np.maximum(z, 1.0e-6)
    valid &= np.isfinite(u) & np.isfinite(v)
    valid &= (u >= 0) & (u < width) & (v >= 0) & (v < height)
    if not np.any(valid):
        return canvas

    u = np.rint(u[valid]).astype(np.int32)
    v = np.rint(v[valid]).astype(np.int32)
    depth = z[valid]
    rgb = colors[valid].astype(np.float32) / 255.0
    # The input is intentionally low-light; lift only display gamma so RGB
    # surface colors remain discernible without replacing them with depth colors.
    rgb = np.clip(rgb, 0.0, 1.0) ** 0.66
    rgb = (rgb * 255.0 + 0.5).astype(np.uint8)

    # Paint far to near for correct visible surfaces. A 3x3 splat keeps the
    # cloud readable while making its discrete point structure explicit.
    order = np.argsort(depth)[::-1]
    u, v, rgb = u[order], v[order], rgb[order]
    for delta_y in (-1, 0, 1):
        for delta_x in (-1, 0, 1):
            px, py = u + delta_x, v + delta_y
            inside = (px >= 0) & (px < width) & (py >= 0) & (py < height)
            canvas[py[inside], px[inside]] = rgb[inside]
    return canvas


def render_frame(output, image_path, event_path, local_points, local_colors, index, total):
    canvas = Image.new("RGB", (1920, 1080), BACKGROUND)
    draw = ImageDraw.Draw(canvas)
    draw.text((42, 32), "Night3R", fill=TEXT, font=load_font(29, bold=True))
    draw.text(
        (42, 72),
        "MVSEC outdoor_night2: trajectory-registered 3D point cloud",
        fill=MUTED,
        font=load_font(19),
    )
    draw.text((1878, 46), f"Frame {index + 1:03d} / {total:03d}", fill=MUTED, font=load_font(19), anchor="ra")

    source = Image.open(image_path).convert("RGB")
    source.thumbnail((520, 390), Image.Resampling.LANCZOS)
    sx, sy = 42 + (520 - source.width) // 2, 150 + (390 - source.height) // 2
    draw.rounded_rectangle((42, 124, 562, 540), radius=6, outline=PANEL, width=2)
    draw.text((60, 140), "Low-light RGB input", fill=TEXT, font=load_font(19))
    canvas.paste(source, (sx, sy))

    event = event_thumbnail(torch.load(event_path, map_location="cpu", weights_only=True))
    event.thumbnail((520, 390), Image.Resampling.LANCZOS)
    draw.rounded_rectangle((42, 570, 562, 986), radius=6, outline=PANEL, width=2)
    draw.text((60, 586), "Events (+ red / - blue)", fill=TEXT, font=load_font(19))
    canvas.paste(event, (42 + (520 - event.width) // 2, 612 + (360 - event.height) // 2))

    cloud = Image.fromarray(
        render_elevated_cloud(local_points, local_colors, index, total, width=1260, height=850)
    )
    canvas.paste(cloud, (600, 150))
    draw.rounded_rectangle((600, 124, 1860, 1026), radius=6, outline=PANEL, width=2)
    draw.text((624, 140), "Night3R fused RGB point cloud (elevated following view)", fill=TEXT, font=load_font(29, bold=True))
    draw.text(
        (624, 1044),
        "Virtual elevated camera follows the evaluated pose; RGB colors, no depth pseudo-color.",
        fill=MUTED,
        font=load_font(19),
    )
    canvas.save(output)


def main():
    args = parse_args()
    if args.history < 1 or args.fps < 1:
        raise ValueError("--history and --fps must be positive")
    cloud = np.load(args.cloud_npz)
    points = np.asarray(cloud["points"], dtype=np.float32)
    colors = np.asarray(cloud["colors"], dtype=np.uint8)
    if len(points) % args.num_frames:
        raise ValueError("Cloud point count is not divisible by --num-frames")
    points_per_frame = len(points) // args.num_frames
    trajectory = np.load(args.trajectory_npz)
    poses = np.asarray(trajectory["pred_0_sim3_aligned"], dtype=np.float64)
    source_indices = np.arange(0, args.num_frames * 2, 2, dtype=np.int64)
    if source_indices[-1] >= len(poses):
        raise ValueError("Trajectory is shorter than the requested fused sequence")
    images = [args.sequence_root / "images" / f"{index:06d}.png" for index in source_indices]
    events = [args.sequence_root / "events" / f"{index:06d}.pt" for index in source_indices]
    if not all(path.is_file() for path in images + events):
        raise RuntimeError("Missing matched image or event files")

    args.outdir.mkdir(parents=True, exist_ok=True)
    frame_dir = args.outdir / "frames"
    frame_dir.mkdir(exist_ok=True)
    for index in range(args.num_frames):
        begin = max(0, index - args.history + 1)
        selected = slice(begin * points_per_frame, (index + 1) * points_per_frame)
        world_points = points[selected]
        world_to_camera = np.linalg.inv(poses[source_indices[index]])
        local = world_points @ world_to_camera[:3, :3].T + world_to_camera[:3, 3]
        render_frame(
            frame_dir / f"frame_{index:03d}.png",
            images[index],
            events[index],
            local.astype(np.float32),
            colors[selected],
            index,
            args.num_frames,
        )
    manifest = {
        "cloud_npz": str(args.cloud_npz),
        "trajectory_npz": str(args.trajectory_npz),
        "num_frames": args.num_frames,
        "points_per_frame": points_per_frame,
        "history": args.history,
        "visualization_camera": "A virtual camera, elevated and slightly behind the current evaluated camera, observes the recent registered local cloud. It does not alter the model-predicted or evaluation pose.",
        "claim_scope": "The displayed local cloud contains the recent global Night3R point maps after their trajectory-registration transforms, re-expressed in the current evaluated camera frame.",
    }
    (args.outdir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
