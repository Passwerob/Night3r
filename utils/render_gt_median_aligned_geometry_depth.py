#!/usr/bin/env python3
"""Render GT-median-aligned depth and geometry figures for the final three pairs.

Every method is assigned one scale per pair on the valid sensor-depth pixels.
The scale is applied to both its depth map and 3D point cloud. Geometry panels
then share one camera and the GT depth color range, so they compare shape after
the standard scale-only depth alignment without any per-panel display resizing.
"""

import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw, ImageFont


REPO_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = (
    REPO_ROOT
    / "outputs/shared_scale_comparison_selected_20260727"
    / "3d_scene_demo_depthcolored_latest_three_20260727"
)
OUTPUT_ROOT = (
    REPO_ROOT
    / "outputs/shared_scale_comparison_selected_20260727"
    / "gt_median_aligned_geometry_depth_20260728"
)
METHODS = (
    ("mast3r", "MASt3R"),
    ("dark3r", "Dark3R"),
    ("vggt", "VGGT"),
    ("night3r", "Night3R"),
    ("gt", "Sensor depth GT"),
)
PAIR_ROOTS = {
    "01_dsec_000552": (
        REPO_ROOT
        / "outputs/shared_scale_comparison_selected_20260727"
        / "dsec_final_gt_pair_000552/pairs/DSEC/screen-000004-dsec_000552-000553"
    ),
    "02_mvsec_000157": (
        REPO_ROOT
        / "outputs/shared_scale_comparison_selected_20260727"
        / "mvsec_final_gt_pairs_v3/pairs/MVSEC/screen-000157-mvsec_000036-000037"
    ),
    "03_mvsec_000205": (
        REPO_ROOT
        / "outputs/shared_scale_comparison_selected_20260727"
        / "mvsec_final_gt_pairs_v3/pairs/MVSEC/screen-000205-mvsec_000047-000048"
    ),
}


def load_font(size):
    for path in (
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf",
    ):
        if Path(path).is_file():
            return ImageFont.truetype(path, size)
    return ImageFont.load_default()


def valid_mask(values):
    return np.isfinite(values) & (values > 0.0)


def depth_scale(prediction, gt, gt_valid):
    mask = gt_valid & valid_mask(prediction)
    if int(mask.sum()) == 0:
        raise RuntimeError("No overlapping valid depth pixels for median alignment")
    return float(np.median(gt[mask]) / np.median(prediction[mask])), int(mask.sum())


def depth_colors(depth, minimum, maximum):
    normalized = np.clip((depth - minimum) / max(maximum - minimum, 1.0e-6), 0.0, 1.0)
    colors = (plt.get_cmap("viridis")(normalized)[..., :3] * 255.0 + 0.5).astype(np.uint8)
    colors[~valid_mask(depth)] = 255
    return colors


def load_cloud(path, max_points, seed):
    payload = np.load(path)
    points = np.asarray(payload["points"], dtype=np.float32)
    depth = np.asarray(payload["depth"], dtype=np.float32)
    mask = np.isfinite(points).all(axis=1) & valid_mask(depth)
    points, depth = points[mask], depth[mask]
    if len(points) > max_points:
        rng = np.random.default_rng(seed)
        indices = rng.choice(len(points), size=max_points, replace=False)
        points, depth = points[indices], depth[indices]
    return points, depth


def robust_center_and_extent(points):
    lower, upper = np.quantile(points, (0.01, 0.99), axis=0)
    center = 0.5 * (lower + upper)
    extent = max(float(np.linalg.norm(upper - lower)), 1.0e-6)
    return center.astype(np.float32), extent


def fitted_camera(points, center, direction, width, height, margin=0.07):
    direction = np.asarray(direction, dtype=np.float32)
    direction /= max(float(np.linalg.norm(direction)), 1.0e-6)
    forward = -direction
    up_hint = np.asarray([0.0, -1.0, 0.0], dtype=np.float32)
    right = np.cross(forward, up_hint)
    right /= max(float(np.linalg.norm(right)), 1.0e-6)
    up = np.cross(right, forward)
    relative = points - center[None, :]
    local_x, local_y, local_z = relative @ right, relative @ up, relative @ forward
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


def render_points(points, colors, camera, target, width, height, radius=1):
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


def compose_geometry(panels, labels, minimum, maximum, output_path):
    width, height = panels[0].size
    header, footer, gap = 58, 70, 12
    canvas = Image.new("RGB", (len(panels) * width + (len(panels) - 1) * gap, header + height + footer), "white")
    draw = ImageDraw.Draw(canvas)
    title_font, meta_font = load_font(25), load_font(18)
    for index, (panel, label) in enumerate(zip(panels, labels)):
        left = index * (width + gap)
        draw_centered(draw, (left, 0, left + width, header), label, title_font, "#1f2937")
        canvas.paste(panel, (left, header))
    bar_left, bar_right = 34, canvas.width - 34
    bar_top, bar_bottom = header + height + 18, header + height + 37
    gradient = np.linspace(0.0, 1.0, bar_right - bar_left, dtype=np.float32)
    rgb = (plt.get_cmap("viridis")(gradient)[..., :3] * 255.0 + 0.5).astype(np.uint8)
    canvas.paste(Image.fromarray(np.repeat(rgb[None], bar_bottom - bar_top, axis=0)), (bar_left, bar_top))
    draw.text((bar_left, bar_bottom + 6), f"{minimum:.2f} m", font=meta_font, fill="#374151")
    max_text = f"{maximum:.2f} m"
    max_box = draw.textbbox((0, 0), max_text, font=meta_font)
    draw.text((bar_right - (max_box[2] - max_box[0]), bar_bottom + 6), max_text, font=meta_font, fill="#374151")
    draw_centered(draw, (bar_left, bar_bottom + 4, bar_right, footer + header + height), "Color: GT-median-aligned Z-depth; range from sensor GT", meta_font, "#374151")
    canvas.save(output_path, optimize=True)


def compose_depth(depths, labels, output_path):
    panels = [Image.fromarray(depth) for depth in depths]
    width, height = panels[0].size
    header, gap = 44, 8
    canvas = Image.new("RGB", (len(panels) * width + (len(panels) - 1) * gap, header + height), "white")
    draw = ImageDraw.Draw(canvas)
    title_font = load_font(22)
    for index, (panel, label) in enumerate(zip(panels, labels)):
        left = index * (width + gap)
        draw_centered(draw, (left, 0, left + width, header), label, title_font, "#1f2937")
        canvas.paste(panel, (left, header))
    canvas.save(output_path, optimize=True)


def render_scene(scene_name, source_dir, pair_dir, output_dir, seed):
    output_dir.mkdir(parents=True, exist_ok=True)
    arrays = np.load(pair_dir / "source_assets/geometry_arrays.npz")
    gt = np.asarray(arrays["gt"], dtype=np.float32)
    gt_valid = np.asarray(arrays["valid"], dtype=bool) & valid_mask(gt)
    if not gt_valid.any():
        raise RuntimeError(f"{scene_name}: sensor depth has no valid pixels")
    gt_values = gt[gt_valid]
    gt_min, gt_max = (float(value) for value in np.percentile(gt_values, [2.0, 98.0]))

    scales, overlap_counts, aligned_depths = {"gt": 1.0}, {"gt": int(gt_valid.sum())}, {"gt": gt}
    for method, _label in METHODS[:-1]:
        prediction = np.load(pair_dir / method / "depth_raw.npy").astype(np.float32)
        scale, count = depth_scale(prediction, gt, gt_valid)
        scales[method], overlap_counts[method] = scale, count
        aligned_depths[method] = prediction * scale

    depth_panels, labels = [], []
    for method, label in METHODS:
        panel = depth_colors(aligned_depths[method], gt_min, gt_max)
        Image.fromarray(panel).save(output_dir / f"{method}_depth_gt_median_aligned.png", optimize=True)
        depth_panels.append(panel)
        labels.append(label)
    compose_depth(depth_panels, labels, output_dir / "depth_all_methods_gt_median_aligned.png")

    clouds = {}
    for method_index, (method, _label) in enumerate(METHODS):
        points, depth = load_cloud(
            source_dir / f"{method}_pointcloud_viridis_shared_scale.npz",
            max_points=65000,
            seed=seed + method_index,
        )
        if method != "gt":
            source_scale = float(json.loads((source_dir.parent / "manifest.json").read_text())[int(scene_name[:2]) - 1]["scale"])
            relative_scale = scales[method] / source_scale
            points, depth = points * relative_scale, depth * relative_scale
        clouds[method] = (points, depth)

    union = np.concatenate([points for points, _depth in clouds.values()], axis=0)
    gt_center, _gt_extent = robust_center_and_extent(clouds["gt"][0])
    cameras = {
        "front": fitted_camera(union, gt_center, [0.0, 0.0, -1.0], 620, 500),
        "elevated": fitted_camera(union, gt_center, [0.38, -0.30, -1.10], 620, 500),
    }
    geometry_outputs = {}
    for view, camera in cameras.items():
        panels = []
        for method, _label in METHODS:
            points, depth = clouds[method]
            colors = depth_colors(depth, gt_min, gt_max)
            panel = render_points(points, colors, camera, gt_center, 620, 500)
            panel.save(output_dir / f"{method}_{view}_geometry_gt_median_aligned.png", optimize=True)
            panels.append(panel)
        output = output_dir / f"{view}_all_methods_geometry_gt_median_aligned.png"
        compose_geometry(panels, labels, gt_min, gt_max, output)
        geometry_outputs[view] = output.name

    front = Image.open(output_dir / geometry_outputs["front"]).convert("RGB")
    elevated = Image.open(output_dir / geometry_outputs["elevated"]).convert("RGB")
    combined = Image.new("RGB", (front.width, front.height + elevated.height), "white")
    combined.paste(front, (0, 0))
    combined.paste(elevated, (0, front.height))
    combined.save(output_dir / "front_elevated_all_methods_geometry_gt_median_aligned.png", optimize=True)

    manifest = {
        "scene": scene_name,
        "pair_source": str(pair_dir.resolve()),
        "coordinate_frame": "view-1 camera frame",
        "alignment": {
            "policy": "Per-method scalar median alignment on valid sensor-GT pixels.",
            "formula": "s_m = median(D_gt[valid & D_m>0]) / median(D_m[valid & D_m>0])",
            "applied_to": "Each method's depth map and all XYZ coordinates in its point cloud.",
            "scales": scales,
            "overlap_pixel_counts": overlap_counts,
        },
        "geometry_display": {
            "policy": "No per-method centering or display rescaling after GT median alignment; every panel uses one shared camera.",
            "target": gt_center.tolist(),
            "cameras": {key: value.tolist() for key, value in cameras.items()},
        },
        "color_policy": {
            "colormap": "viridis",
            "depth_range_m": [gt_min, gt_max],
            "range_source": "2nd--98th percentile of valid sensor depth GT",
        },
        "outputs": {
            "depth": "depth_all_methods_gt_median_aligned.png",
            **geometry_outputs,
            "combined_geometry": "front_elevated_all_methods_geometry_gt_median_aligned.png",
        },
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


def main():
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    records = []
    for index, scene_name in enumerate(PAIR_ROOTS):
        records.append(
            render_scene(
                scene_name,
                SOURCE_ROOT / scene_name,
                PAIR_ROOTS[scene_name],
                OUTPUT_ROOT / scene_name,
                seed=20260728 + 97 * index,
            )
        )
    (OUTPUT_ROOT / "manifest.json").write_text(json.dumps(records, indent=2) + "\n", encoding="utf-8")
    (OUTPUT_ROOT / "README.md").write_text(
        """# GT-Median-Aligned Depth and Geometry Figures

For every method and pair, one scalar is estimated on the overlapping valid
sensor-depth pixels:

`s_m = median(D_gt) / median(D_m)`.

The scalar is applied to the complete depth map and every XYZ coordinate of
that method's point cloud. The depth images and geometry point colors then use
the same Viridis range, defined by the 2nd--98th percentile of valid sensor
depth GT. All geometry panels for one pair share an identical camera. No
additional per-method centering or display resizing is performed after GT
alignment. These figures isolate local geometric shape; they intentionally do
not visualize absolute-scale error.
""",
        encoding="utf-8",
    )
    print(json.dumps({"output_root": str(OUTPUT_ROOT), "scenes": list(PAIR_ROOTS)}, indent=2))


if __name__ == "__main__":
    main()
