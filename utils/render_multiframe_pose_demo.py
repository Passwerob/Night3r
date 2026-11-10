#!/usr/bin/env python3
"""Render a low-light RGB/event/trajectory video from an evaluated sequence."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont


BACKGROUND = (9, 14, 22)
PANEL = (15, 23, 34)
TEXT = (238, 243, 248)
MUTED = (163, 177, 193)
GT = (188, 198, 210)
NIGHT = (38, 202, 177)
MAST = (94, 132, 255)


def parse_args():
    parser = argparse.ArgumentParser(
        "Render an evaluated low-light sequence with its animated trajectory."
    )
    parser.add_argument("--sequence-root", type=Path, required=True)
    parser.add_argument("--trajectory-npz", type=Path, required=True)
    parser.add_argument("--metrics-csv", type=Path, required=True)
    parser.add_argument("--outdir", type=Path, required=True)
    parser.add_argument("--fps", type=int, default=15)
    parser.add_argument("--frame-step", type=int, default=1)
    parser.add_argument("--width", type=int, default=1920)
    parser.add_argument("--height", type=int, default=1080)
    parser.add_argument("--title", default="MVSEC outdoor_night2 low-light trajectory")
    return parser.parse_args()


def load_font(size, bold=False):
    name = "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf"
    path = Path("/usr/share/fonts/truetype/dejavu") / name
    return ImageFont.truetype(str(path), size) if path.is_file() else ImageFont.load_default()


def fit_image(image, size):
    image = image.convert("RGB")
    scale = min(size[0] / image.width, size[1] / image.height)
    target = (max(1, round(image.width * scale)), max(1, round(image.height * scale)))
    return image.resize(target, Image.Resampling.LANCZOS)


def event_rgb(path):
    try:
        event = torch.load(path, map_location="cpu", weights_only=True).float()
    except TypeError:
        event = torch.load(path, map_location="cpu").float()
    if event.ndim != 3 or event.shape[0] % 2:
        raise ValueError(f"Expected a split-polarity [2B,H,W] event voxel: {path}")
    half = event.shape[0] // 2
    positive = event[:half].abs().sum(dim=0)
    negative = event[half:].abs().sum(dim=0)

    def normalize(value):
        nonzero = value[value > 0]
        if not len(nonzero):
            return torch.zeros_like(value)
        scale = torch.quantile(torch.log1p(nonzero), 0.995).clamp_min(1.0e-6)
        return (torch.log1p(value) / scale).clamp(0, 1)

    pos = normalize(positive)
    neg = normalize(negative)
    rgb = torch.stack((pos, torch.zeros_like(pos), neg), dim=-1)
    return Image.fromarray((rgb.numpy() * 255.0 + 0.5).astype(np.uint8))


def principal_plane(points):
    centered = points - points.mean(axis=0, keepdims=True)
    _, _, vectors = np.linalg.svd(centered, full_matrices=False)
    return vectors[:2].T, points.mean(axis=0)


def project(points, plane, origin):
    return (points - origin[None]) @ plane


def read_metrics(path):
    rows = path.read_text(encoding="utf-8").strip().splitlines()
    header = rows[0].split(",")
    result = {}
    for row in rows[1:]:
        values = row.split(",")
        result[values[0]] = {key: float(value) for key, value in zip(header[1:], values[1:])}
    return result


def draw_trajectory(canvas, rect, paths, index, metrics):
    draw = ImageDraw.Draw(canvas)
    x, y, width, height = rect
    draw.rounded_rectangle((x, y, x + width, y + height), radius=6, fill=PANEL)
    title_font = load_font(26, bold=True)
    small_font = load_font(19)
    tiny_font = load_font(16)
    draw.text((x + 26, y + 22), "GT-aligned camera trajectory", fill=TEXT, font=title_font)
    draw.text((x + 26, y + 57), "147 real low-light frames", fill=MUTED, font=small_font)
    legend = (("GT", GT), ("Night3R", NIGHT), ("MASt3R", MAST))
    legend_x = x + 26
    for label, color in legend:
        draw.line((legend_x, y + 95, legend_x + 34, y + 95), fill=color, width=5)
        draw.text((legend_x + 44, y + 82), label, fill=TEXT, font=tiny_font)
        legend_x += 120

    plot = (x + 26, y + 130, width - 52, height - 300)
    px, py, pw, ph = plot
    draw.rounded_rectangle((px, py, px + pw, py + ph), radius=4, fill=(12, 19, 29))
    all_points = np.concatenate([path for _, path in paths], axis=0)
    low = all_points.min(axis=0)
    high = all_points.max(axis=0)
    extent = max(float(np.max(high - low)), 1.0e-6)
    center = (low + high) * 0.5
    low = center - extent * 0.62
    high = center + extent * 0.62

    def to_pixel(points):
        normalized = (points - low[None]) / np.maximum(high - low, 1.0e-6)
        return np.stack((px + normalized[:, 0] * pw, py + (1.0 - normalized[:, 1]) * ph), axis=1)

    for color, path in paths:
        points = to_pixel(path[: index + 1])
        if len(points) > 1:
            draw.line([tuple(point) for point in points], fill=color, width=4)
        current = points[-1]
        draw.ellipse(
            (current[0] - 8, current[1] - 8, current[0] + 8, current[1] + 8),
            fill=color,
            outline=BACKGROUND,
            width=2,
        )

    metrics_y = y + height - 145
    draw.text(
        (x + 26, metrics_y),
        "Trajectory evaluation over the full sequence",
        fill=MUTED,
        font=small_font,
    )
    night = metrics["Night3R"]
    mast = metrics["MASt3R"]
    draw.text(
        (x + 26, metrics_y + 38),
        f"RPE-t  {night['rpe_t_m']:.3f} m   vs   {mast['rpe_t_m']:.3f} m",
        fill=NIGHT,
        font=small_font,
    )
    draw.text(
        (x + 26, metrics_y + 70),
        f"RPE-R  {night['rpe_r_deg']:.3f} deg   vs   {mast['rpe_r_deg']:.3f} deg",
        fill=NIGHT,
        font=small_font,
    )


def render_frame(output, rgb_path, event_path, paths, index, metrics, args):
    canvas = Image.new("RGB", (args.width, args.height), BACKGROUND)
    draw = ImageDraw.Draw(canvas)
    header_font = load_font(30, bold=True)
    detail_font = load_font(19)
    draw.text((42, 32), "Night3R", fill=TEXT, font=header_font)
    draw.text((42, 72), args.title, fill=MUTED, font=detail_font)
    draw.text(
        (args.width - 44, 46),
        f"Frame {index + 1:03d} / {len(paths[0][1]):03d}",
        fill=MUTED,
        font=detail_font,
        anchor="ra",
    )

    rgb_box = (42, 124, 910, 775)
    draw.rounded_rectangle(
        (rgb_box[0] - 1, rgb_box[1] - 1, rgb_box[0] + rgb_box[2] + 1, rgb_box[1] + rgb_box[3] + 1),
        radius=6,
        fill=PANEL,
    )
    rgb = fit_image(Image.open(rgb_path), (rgb_box[2], rgb_box[3]))
    rgb_x = rgb_box[0] + (rgb_box[2] - rgb.width) // 2
    rgb_y = rgb_box[1] + (rgb_box[3] - rgb.height) // 2
    canvas.paste(rgb, (rgb_x, rgb_y))
    draw.text((rgb_box[0] + 18, rgb_box[1] + 16), "Low-light RGB input", fill=TEXT, font=detail_font)

    event_size = (350, 262)
    event_x = rgb_box[0] + rgb_box[2] - event_size[0] - 18
    event_y = rgb_box[1] + rgb_box[3] - event_size[1] - 18
    event = fit_image(event_rgb(event_path), event_size)
    draw.rounded_rectangle(
        (event_x - 6, event_y - 34, event_x + event_size[0] + 6, event_y + event_size[1] + 6),
        radius=4,
        fill=BACKGROUND,
    )
    draw.text((event_x, event_y - 28), "Events (+ red / - blue)", fill=TEXT, font=load_font(16))
    canvas.paste(event, (event_x, event_y))

    draw_trajectory(canvas, (996, 124, 882, 900), paths, index, metrics)
    canvas.save(output)


def main():
    args = parse_args()
    if args.fps < 1 or args.frame_step < 1:
        raise ValueError("--fps and --frame-step must be positive")
    images = sorted((args.sequence_root / "images").glob("*.png"))
    events = sorted((args.sequence_root / "events").glob("*.pt"))
    trajectory = np.load(args.trajectory_npz)
    gt = np.asarray(trajectory["gt_camera_to_world"][:, :3, 3], dtype=np.float64)
    night = np.asarray(trajectory["pred_0_sim3_aligned"][:, :3, 3], dtype=np.float64)
    mast = np.asarray(trajectory["pred_1_sim3_aligned"][:, :3, 3], dtype=np.float64)
    count = min(len(images), len(events), len(gt), len(night), len(mast))
    if count < 3:
        raise RuntimeError("Need at least three aligned RGB, event, and trajectory frames")
    images, events, gt, night, mast = images[:count], events[:count], gt[:count], night[:count], mast[:count]
    plane, origin = principal_plane(gt)
    paths = ((GT, project(gt, plane, origin)), (NIGHT, project(night, plane, origin)), (MAST, project(mast, plane, origin)))
    metrics = read_metrics(args.metrics_csv)
    if "Night3R" not in metrics or "MASt3R" not in metrics:
        raise KeyError("Metrics must contain Night3R and MASt3R rows")

    frames = list(range(0, count, args.frame_step))
    args.outdir.mkdir(parents=True, exist_ok=True)
    frame_dir = args.outdir / "frames"
    frame_dir.mkdir(exist_ok=True)
    for output_index, frame_index in enumerate(frames):
        render_frame(
            frame_dir / f"frame_{output_index:03d}.png",
            images[frame_index],
            events[frame_index],
            paths,
            frame_index,
            metrics,
            args,
        )
    manifest = {
        "sequence_root": str(args.sequence_root),
        "trajectory_npz": str(args.trajectory_npz),
        "metrics_csv": str(args.metrics_csv),
        "source_frame_count": count,
        "rendered_frame_count": len(frames),
        "frame_step": args.frame_step,
        "fps": args.fps,
        "metrics": metrics,
        "claim_scope": "The animated paths are the existing full-sequence GT-aligned pose evaluation trajectories; RGB and event frames are the matching stored low-light inputs.",
    }
    (args.outdir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
