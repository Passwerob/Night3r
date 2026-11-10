#!/usr/bin/env python3
"""Render a fixed-camera, shared-Viridis sparse-view ablation panel."""

import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw, ImageFont


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input-root",
        type=Path,
        default=Path("outputs/sparse_view_ablation_mvsec208_2_4_6_20260728"),
    )
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args()


def load_font(size):
    path = Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf")
    return ImageFont.truetype(path, size) if path.is_file() else ImageFont.load_default()


def load_ply(path):
    with path.open("r", encoding="utf-8") as handle:
        for line_index, line in enumerate(handle):
            if line.strip() == "end_header":
                break
        else:
            raise RuntimeError(f"Missing PLY header terminator: {path}")
    values = np.loadtxt(path, skiprows=line_index + 1, dtype=np.float32)
    if values.ndim == 1:
        values = values[None]
    return values[:, :3]


def robust_bounds(points):
    return np.quantile(points, (0.01, 0.99), axis=0)


def render(points, camera, target, vmin, vmax, width, height):
    canvas = np.full((height, width, 3), (11, 16, 23), dtype=np.uint8)
    forward = target - camera
    forward /= max(float(np.linalg.norm(forward)), 1.0e-6)
    up_hint = np.asarray([0.0, 1.0, 0.0], dtype=np.float32)
    right = np.cross(forward, up_hint)
    right /= max(float(np.linalg.norm(right)), 1.0e-6)
    up = np.cross(right, forward)
    relative = points - camera[None]
    horizontal, vertical, camera_depth = relative @ right, relative @ up, relative @ forward
    focal = 0.5 * width / np.tan(np.deg2rad(48.0) * 0.5)
    pixel_x = focal * horizontal / np.maximum(camera_depth, 1.0e-6) + 0.5 * width
    pixel_y = 0.5 * height - focal * vertical / np.maximum(camera_depth, 1.0e-6)
    keep = (
        (camera_depth > 1.0e-5)
        & np.isfinite(pixel_x)
        & np.isfinite(pixel_y)
        & (pixel_x >= 0)
        & (pixel_x < width)
        & (pixel_y >= 0)
        & (pixel_y < height)
    )
    pixel_x = np.rint(pixel_x[keep]).astype(np.int32)
    pixel_y = np.rint(pixel_y[keep]).astype(np.int32)
    camera_depth = camera_depth[keep]
    z_values = points[keep, 2]
    normalized = np.clip((z_values - vmin) / max(vmax - vmin, 1.0e-6), 0.0, 1.0)
    colors = (plt.get_cmap("viridis")(normalized)[..., :3] * 255.0 + 0.5).astype(np.uint8)
    order = np.argsort(camera_depth)[::-1]
    canvas[pixel_y[order], pixel_x[order]] = colors[order]
    return Image.fromarray(canvas)


def main():
    args = parse_args()
    input_root = args.input_root.resolve()
    output = (
        args.output.resolve()
        if args.output is not None
        else input_root / "sparse_view_ablation_shared_camera_viridis.png"
    )
    records = []
    for name in ("views2", "views4", "views6"):
        directory = input_root / name
        manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
        points = load_ply(directory / "night3r_long_rgb.ply")
        records.append({"name": name, "points": points, "manifest": manifest})

    all_points = np.concatenate([record["points"] for record in records], axis=0)
    lower, upper = robust_bounds(all_points)
    target = 0.5 * (lower + upper)
    extent = max(float(np.linalg.norm(upper - lower)), 1.0e-6)
    camera = target + np.asarray([0.40 * extent, 0.24 * extent, -1.18 * extent], dtype=np.float32)
    vmin, vmax = (float(value) for value in np.percentile(all_points[:, 2], [2.0, 98.0]))

    panel_width, panel_height, header, footer = 720, 520, 62, 54
    canvas = Image.new("RGB", (3 * panel_width, header + panel_height + footer), (11, 16, 23))
    draw = ImageDraw.Draw(canvas)
    title_font, info_font = load_font(26), load_font(18)
    for index, record in enumerate(records):
        panel = render(record["points"], camera, target, vmin, vmax, panel_width, panel_height)
        left = index * panel_width
        canvas.paste(panel, (left, header))
        sample = record["manifest"]["sample"]
        count = record["manifest"]["fusion"]["points_after_voxelization"]
        title = f"{sample['num_frames']} views"
        subtitle = f"frames {','.join(str(value) for value in sample['frame_indices'])} | {count:,} voxels"
        title_box = draw.textbbox((0, 0), title, font=title_font)
        sub_box = draw.textbbox((0, 0), subtitle, font=info_font)
        draw.text((left + (panel_width - (title_box[2] - title_box[0])) / 2, 12), title, font=title_font, fill=(245, 247, 250))
        draw.text((left + (panel_width - (sub_box[2] - sub_box[0])) / 2, 43), subtitle, font=info_font, fill=(164, 177, 193))

    bar_left, bar_right, bar_top, bar_bottom = 42, canvas.width - 42, header + panel_height + 14, header + panel_height + 31
    gradient = np.linspace(0.0, 1.0, bar_right - bar_left, dtype=np.float32)
    colors = (plt.get_cmap("viridis")(gradient)[..., :3] * 255.0 + 0.5).astype(np.uint8)
    canvas.paste(Image.fromarray(np.repeat(colors[None], bar_bottom - bar_top, axis=0)), (bar_left, bar_top))
    draw.text((bar_left, bar_bottom + 4), f"{vmin:.2f}", font=info_font, fill=(164, 177, 193))
    right_text = f"{vmax:.2f}"
    right_box = draw.textbbox((0, 0), right_text, font=info_font)
    draw.text((bar_right - (right_box[2] - right_box[0]), bar_bottom + 4), right_text, font=info_font, fill=(164, 177, 193))
    output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output, optimize=True)

    protocol = {
        "scene": records[0]["manifest"]["sample"]["scene"],
        "dataset": records[0]["manifest"]["sample"]["dataset"],
        "alignment": "pose_head",
        "shared_camera": camera.tolist(),
        "camera_target": target.tolist(),
        "color": {"colormap": "viridis", "z_range": [vmin, vmax], "range_source": "joint 2nd--98th percentile over the three fused clouds"},
        "variants": [
            {
                "name": record["name"],
                "frame_indices": record["manifest"]["sample"]["frame_indices"],
                "points_after_voxelization": record["manifest"]["fusion"]["points_after_voxelization"],
            }
            for record in records
        ],
        "claim_scope": "Qualitative sparse-view fusion only; each variant uses the same model, temporal endpoints, low-light protocol, camera, and color range.",
    }
    output.with_suffix(".json").write_text(json.dumps(protocol, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(output), **protocol}, indent=2))


if __name__ == "__main__":
    main()
