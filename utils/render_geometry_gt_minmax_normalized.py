#!/usr/bin/env python3
"""Render geometry-only comparisons with display-normalized cloud size.

Geometry is normalized only for the camera render. Colors always use each
method's unnormalized Z-depth with the corresponding GT depth min--max range.
"""

import argparse
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw, ImageFont


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE = (
    REPO_ROOT
    / "outputs/shared_scale_comparison_selected_20260727"
    / "3d_scene_demo_depthcolored_latest_three_20260727"
)
DEFAULT_OUTPUT = (
    REPO_ROOT
    / "outputs/shared_scale_comparison_selected_20260727"
    / "geometry_gt_minmax_display_normalized_20260727"
)
METHODS = (
    ("mast3r", "MASt3R"),
    ("dark3r", "Dark3R"),
    ("vggt", "VGGT"),
    ("night3r", "Night3R"),
    ("gt", "Sensor depth GT"),
)


def parse_args():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--source-root", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--panel-width", type=int, default=620)
    parser.add_argument("--panel-height", type=int, default=500)
    parser.add_argument("--point-radius", type=int, default=1)
    parser.add_argument("--max-points", type=int, default=65000)
    parser.add_argument("--seed", type=int, default=20260727)
    return parser.parse_args()


def load_font(size):
    for path in (
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf",
    ):
        if Path(path).is_file():
            return ImageFont.truetype(path, size)
    return ImageFont.load_default()


def load_cloud(path, max_points, seed):
    payload = np.load(path)
    points = np.asarray(payload["points"], dtype=np.float32)
    depth = np.asarray(payload["depth"], dtype=np.float32)
    valid = np.isfinite(points).all(axis=1) & np.isfinite(depth) & (depth > 0.0)
    points, depth = points[valid], depth[valid]
    if len(points) > max_points:
        rng = np.random.default_rng(seed)
        selected = rng.choice(len(points), size=max_points, replace=False)
        points, depth = points[selected], depth[selected]
    return points, depth


def robust_center_and_extent(points):
    lower, upper = np.quantile(points, (0.01, 0.99), axis=0)
    center = 0.5 * (lower + upper)
    extent = max(float(np.linalg.norm(upper - lower)), 1.0e-6)
    return center.astype(np.float32), extent


def normalize_for_display(points, reference_center, reference_extent):
    source_center, source_extent = robust_center_and_extent(points)
    scale = reference_extent / source_extent
    normalized = (points - source_center[None, :]) * scale + reference_center[None, :]
    return normalized.astype(np.float32), {
        "source_center": source_center.tolist(),
        "source_extent": float(source_extent),
        "display_scale": float(scale),
        "display_translation": reference_center.tolist(),
    }


def depth_colors(depth, minimum, maximum):
    normalized = np.clip((depth - minimum) / max(maximum - minimum, 1.0e-6), 0.0, 1.0)
    return (plt.get_cmap("viridis")(normalized)[..., :3] * 255.0 + 0.5).astype(np.uint8)


def fitted_camera(points, center, direction, width, height, margin=0.08):
    direction = np.asarray(direction, dtype=np.float32)
    direction /= max(float(np.linalg.norm(direction)), 1.0e-6)
    forward = -direction
    up_hint = np.asarray([0.0, -1.0, 0.0], dtype=np.float32)
    right = np.cross(forward, up_hint)
    right /= max(float(np.linalg.norm(right)), 1.0e-6)
    up = np.cross(right, forward)
    centered = points - center[None, :]
    local_x = centered @ right
    local_y = centered @ up
    local_z = centered @ forward
    extent = max(float(np.linalg.norm(np.quantile(points, 0.99, axis=0) - np.quantile(points, 0.01, axis=0))), 1.0e-3)
    focal = 0.5 * width / np.tan(np.deg2rad(52.0) * 0.5)

    def fits(distance):
        z = distance + local_z
        if np.quantile(z, 0.005) <= 1.0e-5:
            return False
        u = focal * local_x / z + width * 0.5
        v = height * 0.5 - focal * local_y / z
        u_low, u_high = np.quantile(u, (0.005, 0.995))
        v_low, v_high = np.quantile(v, (0.005, 0.995))
        return (
            u_low >= width * margin
            and u_high <= width * (1.0 - margin)
            and v_low >= height * margin
            and v_high <= height * (1.0 - margin)
        )

    lower, upper = 0.05 * extent, 6.0 * extent
    for _ in range(36):
        middle = 0.5 * (lower + upper)
        if fits(middle):
            upper = middle
        else:
            lower = middle
    return center + direction * upper


def render_points(points, colors, camera, target, width, height, radius):
    canvas = np.full((height, width, 3), 255, dtype=np.uint8)
    forward = target - camera
    forward /= max(float(np.linalg.norm(forward)), 1.0e-6)
    up_hint = np.asarray([0.0, -1.0, 0.0], dtype=np.float32)
    right = np.cross(forward, up_hint)
    right /= max(float(np.linalg.norm(right)), 1.0e-6)
    up = np.cross(right, forward)

    relative = points - camera[None, :]
    x, y, z = relative @ right, relative @ up, relative @ forward
    focal = 0.5 * width / np.tan(np.deg2rad(52.0) * 0.5)
    u = focal * x / np.maximum(z, 1.0e-6) + width * 0.5
    v = height * 0.5 - focal * y / np.maximum(z, 1.0e-6)
    keep = (z > 1.0e-5) & np.isfinite(u) & np.isfinite(v)
    keep &= (u >= -radius) & (u < width + radius)
    keep &= (v >= -radius) & (v < height + radius)
    u = np.rint(u[keep]).astype(np.int32)
    v = np.rint(v[keep]).astype(np.int32)
    z = z[keep]
    colors = colors[keep]
    order = np.argsort(z)[::-1]
    u, v, colors = u[order], v[order], colors[order]
    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            px, py = u + dx, v + dy
            inside = (px >= 0) & (px < width) & (py >= 0) & (py < height)
            canvas[py[inside], px[inside]] = colors[inside]
    return Image.fromarray(canvas)


def draw_centered(draw, box, text, font, fill):
    left, top, right, bottom = box
    text_box = draw.textbbox((0, 0), text, font=font)
    width, height = text_box[2] - text_box[0], text_box[3] - text_box[1]
    draw.text(((left + right - width) * 0.5, (top + bottom - height) * 0.5), text, font=font, fill=fill)


def compose_row(panels, labels, minimum, maximum, output_path):
    width, height = panels[0].size
    header, footer, gap = 58, 70, 12
    canvas = Image.new("RGB", (len(panels) * width + (len(panels) - 1) * gap, header + height + footer), "white")
    draw = ImageDraw.Draw(canvas)
    title_font = load_font(25)
    meta_font = load_font(18)
    for index, (panel, label) in enumerate(zip(panels, labels)):
        left = index * (width + gap)
        draw_centered(draw, (left, 0, left + width, header), label, title_font, "#1f2937")
        canvas.paste(panel, (left, header))
    bar_left, bar_right = 34, canvas.width - 34
    bar_top, bar_bottom = header + height + 18, header + height + 37
    gradient = np.linspace(0.0, 1.0, bar_right - bar_left, dtype=np.float32)
    rgb = (plt.get_cmap("viridis")(gradient)[..., :3] * 255.0 + 0.5).astype(np.uint8)
    gradient_image = Image.fromarray(np.repeat(rgb[None], bar_bottom - bar_top, axis=0))
    canvas.paste(gradient_image, (bar_left, bar_top))
    draw.text((bar_left, bar_bottom + 6), f"{minimum:.2f} m", font=meta_font, fill="#374151")
    max_text = f"{maximum:.2f} m"
    max_box = draw.textbbox((0, 0), max_text, font=meta_font)
    draw.text((bar_right - (max_box[2] - max_box[0]), bar_bottom + 6), max_text, font=meta_font, fill="#374151")
    depth_text = "Color: raw depth, normalized by sensor-depth GT min--max"
    draw_centered(draw, (bar_left, bar_bottom + 4, bar_right, footer + header + height), depth_text, meta_font, "#374151")
    canvas.save(output_path, optimize=True)


def render_scene(args, source_dir, output_dir, scene_index):
    output_dir.mkdir(parents=True, exist_ok=True)
    raw = {}
    for method_index, (key, _label) in enumerate(METHODS):
        path = source_dir / f"{key}_pointcloud_viridis_shared_scale.npz"
        raw[key] = load_cloud(path, args.max_points, args.seed + 97 * scene_index + method_index)
    gt_points, gt_depth = raw["gt"]
    gt_min, gt_max = float(gt_depth.min()), float(gt_depth.max())
    reference_center, reference_extent = robust_center_and_extent(gt_points)
    display = {}
    transforms = {}
    for key, _label in METHODS:
        display[key], transforms[key] = normalize_for_display(raw[key][0], reference_center, reference_extent)
    union = np.concatenate(list(display.values()), axis=0)
    target = reference_center
    cameras = {
        "front": fitted_camera(union, target, [0.0, 0.0, -1.0], args.panel_width, args.panel_height),
        "elevated": fitted_camera(union, target, [0.38, -0.30, -1.10], args.panel_width, args.panel_height),
    }
    labels = [label for _key, label in METHODS]
    outputs = {}
    for view, camera in cameras.items():
        panels = []
        for key, label in METHODS:
            colors = depth_colors(raw[key][1], gt_min, gt_max)
            panel = render_points(display[key], colors, camera, target, args.panel_width, args.panel_height, args.point_radius)
            panel.save(output_dir / f"{key}_{view}_geometry_only.png", optimize=True)
            panels.append(panel)
        row_path = output_dir / f"{view}_all_methods_geometry_only.png"
        compose_row(panels, labels, gt_min, gt_max, row_path)
        outputs[view] = row_path.name
    front = Image.open(output_dir / outputs["front"]).convert("RGB")
    elevated = Image.open(output_dir / outputs["elevated"]).convert("RGB")
    combined = Image.new("RGB", (front.width, front.height + elevated.height), "white")
    combined.paste(front, (0, 0))
    combined.paste(elevated, (0, front.height))
    combined.save(output_dir / "front_elevated_all_methods_geometry_only.png", optimize=True)

    below = {}
    above = {}
    for key, _label in METHODS:
        depth = raw[key][1]
        below[key] = int((depth < gt_min).sum())
        above[key] = int((depth > gt_max).sum())
        np.savez_compressed(
            output_dir / f"{key}_display_normalized_cloud.npz",
            points=display[key],
            raw_depth=depth,
        )
    manifest = {
        "source": str(source_dir.resolve()),
        "coordinate_frame": "view-1 camera frame",
        "scene_display_policy": (
            "Each method cloud is independently translated to the GT robust center and "
            "uniformly scaled to the GT robust 3D extent for display only."
        ),
        "color_policy": "Raw, unnormalized method Z-depth mapped with exact valid GT min--max.",
        "gt_depth_min_max_m": [gt_min, gt_max],
        "gt_reference_center": reference_center.tolist(),
        "gt_reference_robust_extent": reference_extent,
        "display_transforms": transforms,
        "out_of_gt_range_points": {key: {"below": below[key], "above": above[key]} for key, _label in METHODS},
        "cameras": {key: value.tolist() for key, value in cameras.items()},
        "outputs": outputs,
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


def main():
    args = parse_args()
    source_root = args.source_root.resolve()
    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    records = []
    for scene_index, source_dir in enumerate(sorted(path for path in source_root.iterdir() if path.is_dir() and path.name[:2].isdigit()), start=1):
        output_dir = output_root / source_dir.name
        records.append({"name": source_dir.name, **render_scene(args, source_dir, output_dir, scene_index)})
    (output_root / "manifest.json").write_text(json.dumps(records, indent=2) + "\n", encoding="utf-8")
    readme = """# Geometry-Only, GT-Colorized Comparison

Each row compares MASt3R, Dark3R, VGGT, Night3R, and sensor GT from an identical
camera. Point-cloud geometry is independently centered and uniformly rescaled only
for display so that the scene occupies a comparable visual extent in every panel.
The Viridis color of every point is computed from its original raw Z-depth using the
exact valid GT depth min--max of that scene. No predicted depth is color-normalized.

Thus geometry shape is visually comparable while global scale drift remains visible
as systematic color saturation relative to the GT color range. The display-only
geometry transform must not be used for metrics.
"""
    (output_root / "README.md").write_text(readme, encoding="utf-8")
    print(json.dumps({"output_root": str(output_root), "scenes": [item["name"] for item in records]}, indent=2))


if __name__ == "__main__":
    main()
