#!/usr/bin/env python3
"""Render shared-camera geometry comparisons for the MVSEC-003 FT10 check."""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from dataset.dsec_event_dataset import DSECEventDataset  # noqa: E402
from utils.build_pair_review_assets import (  # noqa: E402
    backproject,
    crop_points,
    fitted_camera,
    render_method,
    robust_volume,
)


DEFAULT_RUN_ROOT = (
    REPO_ROOT / "outputs/mvsec003_local_ft10_20260726"
)
SCENE = "screen-000157-mvsec"
MODEL_SIZE = (512, 512)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, default=DEFAULT_RUN_ROOT)
    parser.add_argument("--width", type=int, default=720)
    parser.add_argument("--height", type=int, default=540)
    parser.add_argument("--stride", type=int, default=2)
    parser.add_argument("--point-radius", type=int, default=1)
    return parser.parse_args()


def find_candidate(root):
    matches = sorted(root.glob("candidates/*/rank01_*/geometry_arrays.npz"))
    if len(matches) != 1:
        raise RuntimeError(
            f"Expected one geometry archive below {root}, found {len(matches)}"
        )
    return matches[0]


def load_intrinsics(data_root):
    dataset = DSECEventDataset(
        root_path=str(data_root),
        image_size=MODEL_SIZE,
        pair_stride=1,
        pair_sampling="fixed",
        expected_event_channels=8,
        lowlight_enabled=False,
    )
    matches = [
        intrinsics.numpy().astype(np.float32)
        for scene_dir, intrinsics in dataset.scene_camera.items()
        if scene_dir.name == SCENE and intrinsics is not None
    ]
    if len(matches) != 1:
        raise RuntimeError(f"Could not resolve one calibrated camera for {SCENE}")
    return matches[0]


def load_font(size):
    candidates = (
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/liberation2/LiberationSans-Bold.ttf",
    )
    for path in candidates:
        if Path(path).is_file():
            return ImageFont.truetype(path, size)
    return ImageFont.load_default()


def comparison_sheet(images, labels, output_path):
    image_width, image_height = images[0].size
    label_height = 54
    canvas = Image.new(
        "RGB",
        (image_width * len(images), image_height + label_height),
        "white",
    )
    draw = ImageDraw.Draw(canvas)
    font = load_font(24)
    for index, (image, label) in enumerate(zip(images, labels)):
        x = index * image_width
        canvas.paste(image, (x, label_height))
        box = draw.textbbox((0, 0), label, font=font)
        text_width = box[2] - box[0]
        draw.text(
            (x + (image_width - text_width) // 2, 13),
            label,
            fill=(18, 18, 18),
            font=font,
        )
    canvas.save(output_path, optimize=True)


def main():
    args = parse_args()
    before_path = find_candidate(args.run_root / "visual_before")
    after_path = find_candidate(args.run_root / "visual_after")
    output_root = args.run_root / "paper_effect_check"
    geometry_root = output_root / "geometry_white"
    geometry_root.mkdir(parents=True, exist_ok=True)

    with np.load(before_path) as before, np.load(after_path) as after:
        for key in ("master_aligned", "night_aligned", "zero_aligned", "gt"):
            if key not in before.files or key not in after.files:
                raise KeyError(f"Missing aligned depth array: {key}")
        master_consistent = np.allclose(
            before["master_aligned"],
            after["master_aligned"],
            equal_nan=True,
        )
        gt_consistent = np.allclose(before["gt"], after["gt"], equal_nan=True)
        if not master_consistent or not gt_consistent:
            raise RuntimeError("MASt3R or GT changed between before/after exports")

        depth_arrays = {
            "mast3r": before["master_aligned"].copy(),
            "night3r_before": before["night_aligned"].copy(),
            "night3r_ft10": after["night_aligned"].copy(),
            "night3r_zero_ft10": after["zero_aligned"].copy(),
            "gt": after["gt"].copy(),
        }

    intrinsics = load_intrinsics(args.run_root / "train_data")
    points = {
        name: backproject(depth, intrinsics, args.stride)
        for name, depth in depth_arrays.items()
    }
    reference = points["gt"]
    if len(reference) < 256:
        raise RuntimeError("Insufficient valid GT points")

    lower, upper = robust_volume(reference)
    points = {
        name: crop_points(method_points, lower, upper)
        for name, method_points in points.items()
    }
    reference = points["gt"]
    depth_lo, depth_hi = np.quantile(reference[:, 2], (0.02, 0.98))
    center = 0.5 * (
        np.quantile(reference, 0.01, axis=0)
        + np.quantile(reference, 0.99, axis=0)
    )
    cameras = {
        "front": fitted_camera(
            reference,
            center,
            [0.0, 0.0, -1.0],
            args.width,
            args.height,
        ),
        "elevated": fitted_camera(
            reference,
            center,
            [0.36, -0.34, -1.10],
            args.width,
            args.height,
        ),
    }

    rendered_paths = {}
    for method, method_points in points.items():
        rendered_paths[method] = {}
        for view, camera in cameras.items():
            image = render_method(
                method_points,
                camera,
                center,
                depth_lo,
                depth_hi,
                args.width,
                args.height,
                args.point_radius,
            )
            output_path = geometry_root / f"{method}_{view}.png"
            Image.fromarray(image).save(output_path, optimize=True)
            rendered_paths[method][view] = str(output_path.relative_to(output_root))

    diagnostic_method_order = (
        "mast3r",
        "night3r_before",
        "night3r_ft10",
        "night3r_zero_ft10",
        "gt",
    )
    diagnostic_method_labels = (
        "MASt3R",
        "Night3R Before",
        "Night3R FT10",
        "Night3R Zero",
        "Ground Truth",
    )
    paper_method_order = ("mast3r", "night3r_ft10", "gt")
    paper_method_labels = ("MASt3R", "Night3R FT10", "Ground Truth")
    sheet_paths = {}
    for view, output_name in (
        ("front", "11_geometry_front_comparison.png"),
        ("elevated", "12_geometry_elevated_comparison.png"),
    ):
        images = [
            Image.open(geometry_root / f"{method}_{view}.png").convert("RGB")
            for method in paper_method_order
        ]
        output_path = output_root / output_name
        comparison_sheet(images, paper_method_labels, output_path)
        sheet_paths[view] = output_path.name
    for view, output_name in (
        ("front_diagnostic", "13_geometry_front_full_diagnostic.png"),
        ("elevated_diagnostic", "14_geometry_elevated_full_diagnostic.png"),
    ):
        source_view = view[: -len("_diagnostic")]
        images = [
            Image.open(geometry_root / f"{method}_{source_view}.png").convert("RGB")
            for method in diagnostic_method_order
        ]
        output_path = output_root / output_name
        comparison_sheet(images, diagnostic_method_labels, output_path)
        sheet_paths[view] = output_path.name

    protocol = {
        "scene": SCENE,
        "source_pair": ["000036", "000037"],
        "depth_representation": "metric Z-depth, median-aligned to GT",
        "coordinate_frame": "view-1 calibrated camera frame",
        "model_depth_size": list(MODEL_SIZE),
        "intrinsics_at_model_size": intrinsics.tolist(),
        "backprojection_stride": args.stride,
        "point_radius": args.point_radius,
        "shared_crop_policy": "GT 1%-99% point-cloud volume with 10% margin",
        "shared_crop_lower": lower.tolist(),
        "shared_crop_upper": upper.tolist(),
        "shared_depth_color_range": [float(depth_lo), float(depth_hi)],
        "shared_center": center.tolist(),
        "shared_cameras": {
            name: camera.tolist() for name, camera in cameras.items()
        },
        "camera_directions": {
            "front": [0.0, 0.0, -1.0],
            "elevated": [0.36, -0.34, -1.10],
        },
        "point_counts_after_shared_crop": {
            name: int(len(method_points))
            for name, method_points in points.items()
        },
        "before_geometry_archive": str(before_path),
        "after_geometry_archive": str(after_path),
        "master_consistent_between_exports": master_consistent,
        "gt_consistent_between_exports": gt_consistent,
        "individual_renders": rendered_paths,
        "comparison_sheets": sheet_paths,
    }
    (geometry_root / "geometry_protocol.json").write_text(
        json.dumps(protocol, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(protocol, indent=2))


if __name__ == "__main__":
    main()
