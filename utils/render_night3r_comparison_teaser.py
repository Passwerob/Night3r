#!/usr/bin/env python3
"""Render a paper teaser from an existing DSEC comparison export.

The script does not run either reconstruction network. It reloads the exact
low-light/event input pair and reuses the exported point clouds and metrics.
"""

import argparse
import json
import os
import random
import sys
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from dataset.dsec_event_dataset import DSECEventDataset  # noqa: E402
from utils.export_dsec_darken_pointcloud_contrast import (  # noqa: E402
    depth_to_points,
    render_cloud,
    write_ply,
)


BACKGROUND = (8, 13, 20)
PANEL_BACKGROUND = (11, 16, 23)
TEXT = (244, 247, 250)
MUTED = (165, 176, 188)


def parse_args():
    parser = argparse.ArgumentParser("Render the Night3R DSEC comparison teaser")
    parser.add_argument(
        "--comparison-dir",
        default="outputs/paper_pointcloud_dsec_screen3_pair235_236_20260720",
    )
    parser.add_argument(
        "--candidate-dir",
        default=None,
        help="Stored demo candidate with geometry_arrays.npz and metrics.json.",
    )
    parser.add_argument("--data-root", default="/data/fcr/data/train_data/DSEC")
    parser.add_argument("--scene", default="screen-000003-dsec")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--image-size", type=int, default=512)
    parser.add_argument("--panel-size", type=int, default=480)
    parser.add_argument("--exposure", type=float, default=0.03)
    parser.add_argument(
        "--outdir",
        default="outputs/paper_teaser_night3r_dsec_20260720",
    )
    return parser.parse_args()


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def load_font(size, bold=False):
    name = "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf"
    candidates = (
        Path("/usr/share/fonts/truetype/dejavu") / name,
        Path("/usr/share/fonts/dejavu") / name,
    )
    for path in candidates:
        if path.exists():
            return ImageFont.truetype(str(path), size=size)
    return ImageFont.load_default()


def load_pair(args, comparison_dir):
    pair_file = comparison_dir / "selected_pair.json"
    if not pair_file.is_file():
        raise FileNotFoundError(f"Missing selected pair file: {pair_file}")
    dataset = DSECEventDataset(
        root_path=args.data_root,
        image_size=(args.image_size, args.image_size),
        max_scenes=None,
        max_pairs_per_scene=1,
        pair_sampling="precomputed",
        precomputed_pairs=str(pair_file.resolve()),
        lowlight_enabled=True,
        lowlight_kwargs={
            "min_exposure": args.exposure,
            "max_exposure": args.exposure,
            "read_noise_std": 0.01,
            "shot_noise": True,
            "gamma_space_input": True,
        },
    )
    sample_index = next(
        (
            index
            for index, sample in enumerate(dataset.samples)
            if sample["scene"] == args.scene
        ),
        None,
    )
    if sample_index is None:
        raise RuntimeError(f"Scene {args.scene!r} is not available in {args.data_root}")
    set_seed(args.seed)
    return dataset[sample_index]


def load_real_candidate(args, candidate_dir, outdir):
    candidate_metrics = json.loads(
        (candidate_dir / "metrics.json").read_text(encoding="utf-8")
    )
    scene = candidate_metrics["scene"]
    pair_text = candidate_metrics["sample_id"].rsplit(":", 1)[-1]
    idx1, idx2 = (int(value) for value in pair_text.split("-"))
    pair_file = outdir / "selected_pair.json"
    pair_file.write_text(
        json.dumps([{"idx1": idx1, "idx2": idx2}], indent=2), encoding="utf-8"
    )
    dataset = DSECEventDataset(
        root_path=args.data_root,
        image_size=(args.image_size, args.image_size),
        max_scenes=None,
        max_pairs_per_scene=1,
        pair_sampling="precomputed",
        precomputed_pairs=str(pair_file.resolve()),
        lowlight_enabled=False,
    )
    sample_index = next(
        (
            index
            for index, sample in enumerate(dataset.samples)
            if sample["scene"] == scene
        ),
        None,
    )
    if sample_index is None:
        raise RuntimeError(f"Scene {scene!r} is not available in {args.data_root}")
    sample = dataset[sample_index]
    arrays = np.load(candidate_dir / "geometry_arrays.npz")
    event_arrays = np.load(candidate_dir / "event_voxel.npz")
    metrics = {
        "lowlight_rgb": {
            "absrel": float(candidate_metrics["master_metrics"]["raw_absrel"]),
            "median_aligned_absrel": float(
                candidate_metrics["master_metrics"]["median_aligned_absrel"]
            ),
            "delta_1.25": float(
                candidate_metrics["master_metrics"]["median_aligned_delta_1.25"]
            ),
            "valid_px": int(candidate_metrics["master_metrics"]["valid_px"]),
        },
        "event3r": {
            "absrel": float(candidate_metrics["night_metrics"]["raw_absrel"]),
            "median_aligned_absrel": float(
                candidate_metrics["night_metrics"]["median_aligned_absrel"]
            ),
            "delta_1.25": float(
                candidate_metrics["night_metrics"]["median_aligned_delta_1.25"]
            ),
            "valid_px": int(candidate_metrics["night_metrics"]["valid_px"]),
        },
    }
    sample["event_voxel"] = torch.from_numpy(event_arrays["event_voxel"]).float()
    return sample, arrays, metrics, candidate_metrics


def tensor_rgb_to_image(tensor, size):
    array = tensor.detach().cpu().permute(1, 2, 0).numpy()
    array = (np.clip(array, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)
    return Image.fromarray(array).resize((size, size), Image.Resampling.LANCZOS)


def event_activity_to_image(voxel, size):
    activity = voxel.detach().cpu().float().abs().sum(dim=0).numpy()
    nonzero = activity[activity > 0]
    if nonzero.size:
        scale = max(float(np.percentile(nonzero, 99.5)), 1.0e-6)
        normalized = np.log1p(activity) / np.log1p(scale)
    else:
        normalized = np.zeros_like(activity)
    normalized = np.clip(normalized, 0.0, 1.0)
    rgb = (plt.cm.magma(normalized)[..., :3] * 255.0 + 0.5).astype(np.uint8)
    rgb[activity <= 0] = np.asarray(PANEL_BACKGROUND, dtype=np.uint8)
    return Image.fromarray(rgb).resize((size, size), Image.Resampling.LANCZOS), {
        "mean_activity": float(activity.mean()),
        "occupancy": float((activity > 0).mean()),
        "normalization_percentile": 99.5,
    }


def read_ascii_ply(path):
    header_lines = 0
    vertex_count = None
    with path.open("r", encoding="ascii") as handle:
        for line in handle:
            header_lines += 1
            stripped = line.strip()
            if stripped.startswith("element vertex "):
                vertex_count = int(stripped.rsplit(" ", 1)[-1])
            if stripped == "end_header":
                break
        else:
            raise ValueError(f"Invalid PLY header: {path}")
    values = np.loadtxt(path, skiprows=header_lines, max_rows=vertex_count)
    values = np.atleast_2d(values)
    if values.shape[1] < 3:
        raise ValueError(f"PLY has no XYZ vertices: {path}")
    points = values[:, :3].astype(np.float32)
    colors = values[:, 3:6].astype(np.uint8) if values.shape[1] >= 6 else None
    return points, colors


def cloud_panel(path, size):
    points, colors = read_ascii_ply(path)
    rendered = render_cloud(
        points,
        rgb_colors=colors,
        color_mode="depth",
        width=size,
        height=size,
        point_radius=1,
    )
    foreground = np.any(rendered != np.asarray(PANEL_BACKGROUND, dtype=np.uint8), axis=2)
    yy, xx = np.nonzero(foreground)
    if xx.size:
        crop = Image.fromarray(rendered).crop(
            (int(xx.min()), int(yy.min()), int(xx.max()) + 1, int(yy.max()) + 1)
        )
        target = int(round(size * 0.88))
        scale = target / max(crop.width, crop.height)
        crop = crop.resize(
            (
                max(1, int(round(crop.width * scale))),
                max(1, int(round(crop.height * scale))),
            ),
            Image.Resampling.LANCZOS,
        )
        framed = Image.new("RGB", (size, size), PANEL_BACKGROUND)
        framed.paste(crop, ((size - crop.width) // 2, (size - crop.height) // 2))
        rendered_image = framed
    else:
        rendered_image = Image.fromarray(rendered)
    return rendered_image, int(points.shape[0])


def export_candidate_clouds(sample, arrays, outdir, seed, panel_size):
    intrinsics = sample["intrinsics"]
    image = sample["images_photo"][0]
    valid = torch.from_numpy(arrays["depth_valid"]).bool()
    specs = {
        "lowlight_rgb": arrays["master_raw"],
        "event3r": arrays["night_raw"],
        "gt": arrays["gt_depth"],
    }
    rendered = {}
    point_counts = {}
    ply = {}
    for offset, (name, depth_array) in enumerate(specs.items()):
        points, colors = depth_to_points(
            torch.from_numpy(depth_array).float(),
            image,
            intrinsics,
            valid,
            max_points=45000,
            seed=seed + offset,
        )
        ply_path = outdir / f"{name}_cloud.ply"
        write_ply(ply_path, points, colors)
        rendered[name], point_counts[name] = cloud_panel(ply_path, panel_size)
        ply[name] = ply_path.name
    return rendered, point_counts, ply


def draw_centered(draw, xy, text, font, fill):
    draw.text(xy, text, font=font, fill=fill, anchor="mm")


def compose_teaser(panels, metrics, output_path, panel_size):
    margin_x = 34
    gap = 16
    header = 108
    footer = 38
    border = 3
    width = 2 * margin_x + len(panels) * panel_size + (len(panels) - 1) * gap
    height = header + panel_size + footer
    canvas = Image.new("RGB", (width, height), BACKGROUND)
    draw = ImageDraw.Draw(canvas)
    title_font = load_font(28, bold=True)
    subtitle_font = load_font(19, bold=False)
    footer_font = load_font(17, bold=False)

    for index, panel in enumerate(panels):
        x = margin_x + index * (panel_size + gap)
        y = header
        accent = panel["accent"]
        draw.rectangle(
            (x - border, y - border, x + panel_size + border - 1, y + panel_size + border - 1),
            outline=accent,
            width=border,
        )
        canvas.paste(panel["image"], (x, y))
        draw_centered(draw, (x + panel_size / 2, 35), panel["title"], title_font, TEXT)
        draw_centered(
            draw,
            (x + panel_size / 2, 72),
            panel["subtitle"],
            subtitle_font,
            panel.get("subtitle_color", MUTED),
        )

    separator_x = margin_x + 2 * panel_size + gap + gap // 2
    draw.line((separator_x, 18, separator_x, height - 18), fill=(50, 60, 72), width=2)
    geometry_start = margin_x + 2 * (panel_size + gap)
    geometry_end = margin_x + 5 * panel_size + 4 * gap
    draw_centered(
        draw,
        ((geometry_start + geometry_end) / 2, height - 18),
        "Depth-colored geometry; identical rendering protocol",
        footer_font,
        MUTED,
    )
    canvas.save(output_path, quality=96)

    baseline = metrics["lowlight_rgb"]["absrel"]
    ours = metrics["event3r"]["absrel"]
    return 100.0 * (baseline - ours) / max(baseline, 1.0e-12)


def main():
    args = parse_args()
    comparison_dir = (REPO_ROOT / args.comparison_dir).resolve()
    outdir = (REPO_ROOT / args.outdir).resolve()
    outdir.mkdir(parents=True, exist_ok=True)
    candidate_metrics = None
    if args.candidate_dir:
        candidate_dir = (REPO_ROOT / args.candidate_dir).resolve()
        sample, arrays, metrics, candidate_metrics = load_real_candidate(
            args, candidate_dir, outdir
        )
        cloud_images, point_counts, ply_files = export_candidate_clouds(
            sample, arrays, outdir, args.seed, args.panel_size
        )
        input_tensor = sample["images_photo"][0]
        input_subtitle = "real DSEC low light"
        source_comparison = str(candidate_dir)
        exposure = None
    else:
        manifest_path = comparison_dir / "manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        metrics = manifest["metrics"]
        sample = load_pair(args, comparison_dir)
        cloud_images = {}
        point_counts = {}
        for name in ("lowlight_rgb", "event3r", "gt"):
            cloud_images[name], point_counts[name] = cloud_panel(
                comparison_dir / manifest["ply"][name], args.panel_size
            )
        ply_files = dict(manifest["ply"])
        input_tensor = sample["images_photo_dark"][0]
        input_subtitle = f"{args.exposure * 100:.0f}% exposure"
        source_comparison = str(comparison_dir)
        exposure = args.exposure

    input_image = tensor_rgb_to_image(input_tensor, args.panel_size)
    event_image, event_stats = event_activity_to_image(sample["event_voxel"][0], args.panel_size)
    baseline_image = cloud_images["lowlight_rgb"]
    night3r_image = cloud_images["event3r"]
    gt_image = cloud_images["gt"]

    panel_assets = {
        "01_lowlight_rgb.png": input_image,
        "02_event_activity.png": event_image,
        "03_mast3r_pointcloud.png": baseline_image,
        "04_night3r_pointcloud.png": night3r_image,
        "05_gt_pointcloud.png": gt_image,
    }
    for name, image in panel_assets.items():
        image.save(outdir / name)

    baseline_absrel = metrics["lowlight_rgb"]["absrel"]
    night3r_absrel = metrics["event3r"]["absrel"]
    improvement = 100.0 * (baseline_absrel - night3r_absrel) / max(baseline_absrel, 1.0e-12)
    panels = [
        {
            "title": "Low-light RGB",
            "subtitle": input_subtitle,
            "image": input_image,
            "accent": (116, 128, 142),
        },
        {
            "title": "Aligned events",
            "subtitle": "voxel activity",
            "image": event_image,
            "accent": (211, 102, 179),
        },
        {
            "title": "MASt3R",
            "subtitle": f"AbsRel {baseline_absrel:.3f}",
            "image": baseline_image,
            "accent": (206, 100, 96),
        },
        {
            "title": "Night3R",
            "subtitle": f"AbsRel {night3r_absrel:.3f}  |  {improvement:.1f}% lower",
            "image": night3r_image,
            "accent": (69, 206, 164),
            "subtitle_color": (98, 225, 184),
        },
        {
            "title": "Ground truth",
            "subtitle": "reference geometry",
            "image": gt_image,
            "accent": (204, 212, 220),
        },
    ]
    teaser_path = outdir / "night3r_comparison_teaser.png"
    compose_teaser(panels, metrics, teaser_path, args.panel_size)

    output_manifest = {
        "figure": teaser_path.name,
        "sample_id": sample["sample_id"],
        "source_comparison": source_comparison,
        "exposure": exposure,
        "metrics": metrics,
        "absrel_reduction_percent": improvement,
        "event_statistics": event_stats,
        "point_counts": {
            "mast3r": point_counts["lowlight_rgb"],
            "night3r": point_counts["event3r"],
            "ground_truth": point_counts["gt"],
        },
        "ply": ply_files,
        "rendering": {
            "geometry_color": "per-cloud robust depth colorization",
            "camera": "same relative camera angle and projection for all point clouds",
            "framing": "independent robust 1st--99th percentile extent with tight panel crop",
            "note": "The teaser reuses stored predictions; no network inference is run.",
        },
        "panels": list(panel_assets),
    }
    if candidate_metrics is not None:
        output_manifest["source_candidate_metrics"] = candidate_metrics
    (outdir / "manifest.json").write_text(
        json.dumps(output_manifest, indent=2), encoding="utf-8"
    )
    print(json.dumps({"figure": str(teaser_path), **output_manifest}, indent=2))


if __name__ == "__main__":
    main()
