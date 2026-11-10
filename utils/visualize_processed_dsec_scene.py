#!/usr/bin/env python3
import argparse
import csv
import json
import math
import os
import shutil
import subprocess
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image, ImageDraw


def parse_args():
    parser = argparse.ArgumentParser("Visualize a processed DSEC screen-* scene.")
    parser.add_argument(
        "--scene-dir",
        default="/data/fcr/data/train_data/DSEC/screen-000001-dsec",
        help="Processed screen-* directory containing images, events, depth_gt, and meta.",
    )
    parser.add_argument("--outdir", default=None)
    parser.add_argument("--sample-ids", type=int, nargs="*", default=None)
    parser.add_argument("--max-video-frames", type=int, default=160)
    parser.add_argument("--video-fps", type=int, default=12)
    return parser.parse_args()


def safe_torch_load(path):
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        return torch.load(path, map_location="cpu")


def read_frame_meta(scene_dir):
    path = scene_dir / "meta" / "frames_meta.csv"
    if not path.is_file():
        return []
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def percentile_limits(values, lo=2.0, hi=98.0, fallback=(0.0, 1.0)):
    values = np.asarray(values, dtype=np.float32)
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return fallback
    vmin = float(np.percentile(finite, lo))
    vmax = float(np.percentile(finite, hi))
    if not np.isfinite(vmin) or not np.isfinite(vmax) or vmax <= vmin:
        return fallback
    return vmin, vmax


def normalize01(values, vmin=None, vmax=None):
    values = np.asarray(values, dtype=np.float32)
    if vmin is None or vmax is None:
        vmin, vmax = percentile_limits(values)
    out = (values - vmin) / max(vmax - vmin, 1e-6)
    return np.clip(out, 0.0, 1.0)


def load_rgb(path):
    return np.asarray(Image.open(path).convert("RGB"), dtype=np.uint8)


def load_event(path):
    event = safe_torch_load(path)
    if not torch.is_tensor(event):
        raise TypeError(f"Expected tensor in {path}, got {type(event)}")
    if event.ndim != 3:
        raise ValueError(f"Expected [C,H,W] event tensor in {path}, got {tuple(event.shape)}")
    return event.float().numpy()


def load_depth(path):
    depth = safe_torch_load(path)
    if not torch.is_tensor(depth):
        raise TypeError(f"Expected tensor in {path}, got {type(depth)}")
    if depth.ndim == 3 and depth.shape[0] == 1:
        depth = depth[0]
    if depth.ndim != 2:
        raise ValueError(f"Expected [H,W] depth tensor in {path}, got {tuple(depth.shape)}")
    return depth.float().numpy()


def event_rgb(event):
    bins = event.shape[0] // 2
    pos = np.maximum(event[:bins], 0.0).sum(axis=0)
    neg = np.maximum(event[bins : 2 * bins], 0.0).sum(axis=0)
    scale = percentile_limits(np.concatenate([pos.reshape(-1), neg.reshape(-1)]), 1, 99.5, (0.0, 1.0))[1]
    pos_n = np.clip(pos / max(scale, 1e-6), 0.0, 1.0)
    neg_n = np.clip(neg / max(scale, 1e-6), 0.0, 1.0)
    out = np.zeros((*pos.shape, 3), dtype=np.float32)
    out[..., 0] = pos_n
    out[..., 2] = neg_n
    out[..., 1] = 0.25 * (pos_n + neg_n)
    return out


def event_activity(event):
    return np.abs(event).sum(axis=0)


def depth_vis(depth, vmin=None, vmax=None):
    valid = np.isfinite(depth) & (depth > 0)
    if vmin is None or vmax is None:
        vmin, vmax = percentile_limits(depth[valid], 2, 98, (0.0, 50.0))
    colors = plt.get_cmap("turbo")(normalize01(depth, vmin, vmax))[..., :3]
    colors[~valid] = 0.0
    return colors, vmin, vmax


def image_to_uint8(image):
    image = np.asarray(image)
    if image.dtype == np.uint8:
        return image
    return np.clip(image * 255.0, 0, 255).astype(np.uint8)


def resize_for_panel(image, width=320):
    pil = Image.fromarray(image_to_uint8(image))
    scale = width / pil.width
    return pil.resize((width, max(1, int(round(pil.height * scale)))), Image.BILINEAR)


def draw_label(pil, text):
    draw = ImageDraw.Draw(pil)
    draw.rectangle([0, 0, pil.width, 24], fill=(0, 0, 0))
    draw.text((6, 5), text, fill=(255, 255, 255))
    return pil


def make_contact_sheet(items, out_path, panel_width=280):
    rows = []
    for item in items:
        panels = [resize_for_panel(panel, panel_width) for _, panel in item["panels"]]
        labels = [label for label, _ in item["panels"]]
        for panel, label in zip(panels, labels):
            draw_label(panel, label)
        row_h = max(panel.height for panel in panels)
        row_w = sum(panel.width for panel in panels)
        row = Image.new("RGB", (row_w, row_h), (255, 255, 255))
        x = 0
        for panel in panels:
            row.paste(panel, (x, 0))
            x += panel.width
        rows.append(row)

    gap = 8
    width = max(row.width for row in rows)
    height = sum(row.height for row in rows) + gap * (len(rows) - 1)
    sheet = Image.new("RGB", (width, height), (245, 245, 245))
    y = 0
    for row in rows:
        sheet.paste(row, (0, y))
        y += row.height + gap
    sheet.save(out_path)


def collect_stats(scene_dir, image_paths, event_paths, depth_paths, frame_meta):
    stats = []
    meta_by_idx = {int(row["out_idx"]): row for row in frame_meta if row.get("out_idx", "").isdigit()}
    for idx, (image_path, event_path, depth_path) in enumerate(zip(image_paths, event_paths, depth_paths)):
        image = Image.open(image_path)
        event = load_event(event_path)
        depth = load_depth(depth_path)
        valid_depth = np.isfinite(depth) & (depth > 0)
        activity = event_activity(event)
        row = meta_by_idx.get(idx, {})
        event_start = int(row["event_start_us"]) if row.get("event_start_us") else None
        event_end = int(row["event_end_us"]) if row.get("event_end_us") else None
        num_events = int(row["num_events"]) if row.get("num_events") else None
        depth_valid_values = depth[valid_depth]
        stats.append(
            {
                "idx": idx,
                "image_name": image_path.name,
                "image_width": image.width,
                "image_height": image.height,
                "event_shape": list(event.shape),
                "depth_shape": list(depth.shape),
                "num_events": num_events,
                "event_nonzero_ratio": float(np.count_nonzero(event) / event.size),
                "event_abs_sum": float(np.abs(event).sum()),
                "event_activity_p99": float(np.percentile(activity, 99)) if activity.size else 0.0,
                "depth_valid_ratio": float(valid_depth.mean()),
                "depth_median": float(np.median(depth_valid_values)) if depth_valid_values.size else None,
                "depth_p02": float(np.percentile(depth_valid_values, 2)) if depth_valid_values.size else None,
                "depth_p98": float(np.percentile(depth_valid_values, 98)) if depth_valid_values.size else None,
                "event_window_us": None if event_start is None or event_end is None else event_end - event_start,
            }
        )
    return stats


def plot_stats(stats, out_path):
    idx = np.array([row["idx"] for row in stats])
    num_events = np.array([np.nan if row["num_events"] is None else row["num_events"] for row in stats], dtype=np.float64)
    event_nonzero = np.array([row["event_nonzero_ratio"] for row in stats], dtype=np.float64)
    depth_valid = np.array([row["depth_valid_ratio"] for row in stats], dtype=np.float64)
    depth_median = np.array([np.nan if row["depth_median"] is None else row["depth_median"] for row in stats], dtype=np.float64)
    event_window = np.array([np.nan if row["event_window_us"] is None else row["event_window_us"] for row in stats], dtype=np.float64)

    fig, axes = plt.subplots(4, 1, figsize=(12, 10), sharex=True)
    axes[0].plot(idx, num_events, color="#a61c3c", linewidth=1.4)
    axes[0].set_ylabel("raw events")
    axes[0].grid(alpha=0.25)
    axes[1].plot(idx, event_nonzero * 100.0, color="#005f73", linewidth=1.4)
    axes[1].set_ylabel("voxel nonzero %")
    axes[1].grid(alpha=0.25)
    axes[2].plot(idx, depth_valid * 100.0, color="#2d6a4f", linewidth=1.4)
    axes[2].set_ylabel("valid depth %")
    axes[2].grid(alpha=0.25)
    axes[3].plot(idx, depth_median, color="#7b2cbf", linewidth=1.4, label="median depth")
    axes[3].set_ylabel("depth")
    axes[3].set_xlabel("frame index")
    axes[3].grid(alpha=0.25)

    if np.isfinite(event_window).any():
        ax2 = axes[3].twinx()
        ax2.plot(idx, event_window / 1000.0, color="#555555", linewidth=1.0, alpha=0.5, label="event window ms")
        ax2.set_ylabel("event window ms")

    fig.suptitle("Processed DSEC scene consistency statistics")
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def plot_event_bins(event, out_path, frame_idx):
    channels = event.shape[0]
    cols = min(4, channels)
    rows = int(math.ceil(channels / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3.0, rows * 2.6))
    axes = np.asarray(axes).reshape(rows, cols)
    vmin, vmax = percentile_limits(np.abs(event), 1, 99.5, (0.0, 1.0))
    for c in range(rows * cols):
        ax = axes.flat[c]
        ax.axis("off")
        if c >= channels:
            continue
        name = f"{'pos' if c < channels // 2 else 'neg'} bin {c % (channels // 2)}"
        im = ax.imshow(event[c], cmap="magma", vmin=vmin, vmax=vmax)
        ax.set_title(name, fontsize=10)
    fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.75)
    fig.suptitle(f"Event voxel channels, frame {frame_idx:06d}")
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def make_overview(scene_dir, image_paths, event_paths, depth_paths, sample_ids, out_path):
    depth_values = []
    for idx in sample_ids:
        depth = load_depth(depth_paths[idx])
        valid = np.isfinite(depth) & (depth > 0)
        if valid.any():
            depth_values.append(depth[valid].reshape(-1))
    dmin, dmax = percentile_limits(np.concatenate(depth_values) if depth_values else np.array([]), 2, 98, (0.0, 50.0))

    items = []
    for idx in sample_ids:
        rgb = load_rgb(image_paths[idx])
        event = load_event(event_paths[idx])
        depth = load_depth(depth_paths[idx])
        depth_rgb, _, _ = depth_vis(depth, dmin, dmax)
        activity = normalize01(event_activity(event), None, None)
        activity_rgb = plt.get_cmap("viridis")(activity)[..., :3]
        items.append(
            {
                "idx": idx,
                "panels": [
                    (f"RGB {idx:06d}", rgb),
                    ("event +/-", event_rgb(event)),
                    ("event activity", activity_rgb),
                    ("depth gt", depth_rgb),
                ],
            }
        )
    make_contact_sheet(items, out_path)


def make_video_frames(image_paths, event_paths, depth_paths, outdir, max_frames):
    outdir.mkdir(parents=True, exist_ok=True)
    count = len(image_paths)
    stride = max(1, int(math.ceil(count / max(1, max_frames))))
    selected = list(range(0, count, stride))

    depth_samples = []
    for idx in selected[:: max(1, len(selected) // 24)]:
        depth = load_depth(depth_paths[idx])
        valid = np.isfinite(depth) & (depth > 0)
        if valid.any():
            depth_samples.append(depth[valid].reshape(-1))
    dmin, dmax = percentile_limits(np.concatenate(depth_samples) if depth_samples else np.array([]), 2, 98, (0.0, 50.0))

    frame_paths = []
    for out_idx, idx in enumerate(selected):
        rgb = load_rgb(image_paths[idx])
        event = load_event(event_paths[idx])
        depth = load_depth(depth_paths[idx])
        depth_rgb, _, _ = depth_vis(depth, dmin, dmax)
        panels = [
            resize_for_panel(rgb, 320),
            resize_for_panel(event_rgb(event), 320),
            resize_for_panel(depth_rgb, 320),
        ]
        labels = [f"RGB {idx:06d}", "event +/-", "depth gt"]
        for panel, label in zip(panels, labels):
            draw_label(panel, label)
        width = sum(panel.width for panel in panels)
        height = max(panel.height for panel in panels)
        frame = Image.new("RGB", (width, height), (250, 250, 250))
        x = 0
        for panel in panels:
            frame.paste(panel, (x, 0))
            x += panel.width
        frame_path = outdir / f"{out_idx:06d}.png"
        frame.save(frame_path)
        frame_paths.append(frame_path)
    return frame_paths


def encode_video(frame_dir, out_path, fps):
    if shutil.which("ffmpeg") is None:
        return False
    cmd = [
        "ffmpeg",
        "-y",
        "-hide_banner",
        "-loglevel",
        "error",
        "-framerate",
        str(fps),
        "-i",
        str(frame_dir / "%06d.png"),
        "-pix_fmt",
        "yuv420p",
        str(out_path),
    ]
    subprocess.run(cmd, check=True)
    return True


def summarize(stats):
    def finite_values(key):
        return np.array([row[key] for row in stats if row[key] is not None], dtype=np.float64)

    event_counts = finite_values("num_events")
    depth_valid = finite_values("depth_valid_ratio")
    event_nonzero = finite_values("event_nonzero_ratio")
    med_depth = finite_values("depth_median")
    windows = finite_values("event_window_us")
    first = stats[0]
    return {
        "frames": len(stats),
        "image_size": [first["image_height"], first["image_width"]],
        "event_shape": first["event_shape"],
        "depth_shape": first["depth_shape"],
        "event_count_min_median_max": [
            int(np.min(event_counts)),
            int(np.median(event_counts)),
            int(np.max(event_counts)),
        ]
        if event_counts.size
        else None,
        "event_nonzero_ratio_min_median_max": [
            float(np.min(event_nonzero)),
            float(np.median(event_nonzero)),
            float(np.max(event_nonzero)),
        ],
        "depth_valid_ratio_min_median_max": [
            float(np.min(depth_valid)),
            float(np.median(depth_valid)),
            float(np.max(depth_valid)),
        ],
        "depth_median_min_median_max": [
            float(np.min(med_depth)),
            float(np.median(med_depth)),
            float(np.max(med_depth)),
        ]
        if med_depth.size
        else None,
        "event_window_ms_min_median_max": [
            float(np.min(windows) / 1000.0),
            float(np.median(windows) / 1000.0),
            float(np.max(windows) / 1000.0),
        ]
        if windows.size
        else None,
    }


def main():
    args = parse_args()
    scene_dir = Path(args.scene_dir)
    if not scene_dir.is_dir():
        raise FileNotFoundError(scene_dir)
    outdir = Path(args.outdir) if args.outdir else scene_dir / "visual_check"
    outdir.mkdir(parents=True, exist_ok=True)

    image_paths = sorted((scene_dir / "images").glob("*.png"))
    event_paths = sorted((scene_dir / "events").glob("*.pt"))
    depth_paths = sorted((scene_dir / "depth_gt").glob("*.pt"))
    n = min(len(image_paths), len(event_paths), len(depth_paths))
    if n == 0:
        raise RuntimeError("No complete RGB/event/depth frames found")
    image_paths, event_paths, depth_paths = image_paths[:n], event_paths[:n], depth_paths[:n]

    frame_meta = read_frame_meta(scene_dir)
    stats = collect_stats(scene_dir, image_paths, event_paths, depth_paths, frame_meta)
    summary = summarize(stats)

    if args.sample_ids:
        sample_ids = sorted(set(idx for idx in args.sample_ids if 0 <= idx < n))
    else:
        sample_ids = sorted(set([0, 1, 2, n // 8, n // 4, n // 2, (3 * n) // 4, n - 3, n - 2, n - 1]))
    make_overview(scene_dir, image_paths, event_paths, depth_paths, sample_ids, outdir / "overview_rgb_event_depth.png")
    plot_stats(stats, outdir / "dataset_stats.png")
    mid_idx = n // 2
    plot_event_bins(load_event(event_paths[mid_idx]), outdir / f"event_voxel_bins_{mid_idx:06d}.png", mid_idx)

    video_frame_dir = outdir / "video_frames"
    frame_paths = make_video_frames(image_paths, event_paths, depth_paths, video_frame_dir, args.max_video_frames)
    video_path = outdir / "rgb_event_depth_preview.mp4"
    summary["preview_video_encoded"] = encode_video(video_frame_dir, video_path, args.video_fps)
    summary["preview_video_frames"] = len(frame_paths)

    with open(outdir / "stats_per_frame.json", "w") as f:
        json.dump(stats, f, indent=2)
    with open(outdir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(json.dumps({"outdir": str(outdir), **summary}, indent=2))


if __name__ == "__main__":
    main()
