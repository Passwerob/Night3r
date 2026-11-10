#!/usr/bin/env python3
"""Organize the MVSEC-003 local FT10 check into explicit dual-view assets."""

import json
import re
import shutil
import sys
from pathlib import Path

import cv2
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


RUN_ROOT = REPO_ROOT / "outputs/mvsec003_local_ft10_20260726"
PAPER_ROOT = RUN_ROOT / "paper_effect_check"
NATIVE_ROOT = (
    REPO_ROOT
    / "outputs/mvsec_selected_003_005_native_20260726"
    / "003_screen-000157-mvsec_000036_000037/native_346x260"
)
SCENE = "screen-000157-mvsec"
MODEL_SIZE = (512, 512)

VIEW_SOURCES = {
    "view1": {
        "frame": "000036",
        "native_prefix": "view1",
        "before_root": RUN_ROOT / "visual_before",
        "after_root": RUN_ROOT / "visual_after",
        "inference_order": ["000036", "000037"],
    },
    "view2": {
        "frame": "000037",
        "native_prefix": "view2",
        "before_root": RUN_ROOT / "view2_visual_before",
        "after_root": RUN_ROOT / "view2_visual_after",
        "inference_order": ["000037", "000036"],
    },
}


def candidate_dir(root):
    matches = sorted(root.glob("candidates/*/rank01_*"))
    if len(matches) != 1:
        raise RuntimeError(f"Expected one rank01 candidate below {root}")
    return matches[0]


def load_font(size, bold=False):
    candidates = (
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
        if bold
        else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation2/LiberationSans-Bold.ttf"
        if bold
        else "/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf",
    )
    for candidate in candidates:
        if Path(candidate).is_file():
            return ImageFont.truetype(candidate, size)
    return ImageFont.load_default()


def fit_inside(image, size):
    image = image.convert("RGB").copy()
    image.thumbnail(size, Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", size, "white")
    canvas.paste(
        image,
        ((size[0] - image.width) // 2, (size[1] - image.height) // 2),
    )
    return canvas


def labeled_sheet(rows, column_labels, row_labels, output_path, tile_size):
    header_height = 52
    row_header_height = 44
    width = tile_size[0] * len(column_labels)
    height = header_height + len(rows) * (row_header_height + tile_size[1])
    canvas = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(canvas)
    column_font = load_font(22, bold=True)
    row_font = load_font(18, bold=True)
    for column, label in enumerate(column_labels):
        box = draw.textbbox((0, 0), label, font=column_font)
        text_width = box[2] - box[0]
        x = column * tile_size[0] + (tile_size[0] - text_width) // 2
        draw.text((x, 12), label, fill=(18, 18, 18), font=column_font)
    for row_index, (images, row_label) in enumerate(zip(rows, row_labels)):
        top = header_height + row_index * (row_header_height + tile_size[1])
        draw.text((14, top + 10), row_label, fill=(18, 18, 18), font=row_font)
        for column, image in enumerate(images):
            tile = fit_inside(image, tile_size)
            canvas.paste(
                tile,
                (column * tile_size[0], top + row_header_height),
            )
    canvas.save(output_path, optimize=True)


def depth_image(depth, lo, hi, valid=None):
    depth = np.asarray(depth, dtype=np.float32)
    mask = np.isfinite(depth) & (depth > 0)
    if valid is not None:
        mask &= np.asarray(valid).astype(bool)
    normalized = np.clip((depth - lo) / max(float(hi - lo), 1.0e-6), 0, 1)
    normalized = np.nan_to_num(normalized, nan=0.0, posinf=1.0, neginf=0.0)
    indices = np.round(normalized * 255).astype(np.uint8)
    colored = cv2.applyColorMap(indices, cv2.COLORMAP_VIRIDIS)
    colored = cv2.cvtColor(colored, cv2.COLOR_BGR2RGB)
    colored[~mask] = 0
    return Image.fromarray(colored)


def intrinsics():
    dataset = DSECEventDataset(
        root_path=str(RUN_ROOT / "train_data"),
        image_size=MODEL_SIZE,
        pair_stride=1,
        pair_sampling="fixed",
        expected_event_channels=8,
        lowlight_enabled=False,
    )
    values = [
        value.numpy().astype(np.float32)
        for scene_dir, value in dataset.scene_camera.items()
        if scene_dir.name == SCENE and value is not None
    ]
    if len(values) != 1:
        raise RuntimeError(f"Could not resolve intrinsics for {SCENE}")
    return values[0]


def copy_native_inputs(view_name, view_info, view_root):
    prefix = view_info["native_prefix"]
    inputs = view_root / "inputs_native"
    inputs.mkdir(parents=True, exist_ok=True)
    mapping = {
        f"{prefix}_RGB_original.png": "01_RGB_original_346x260.png",
        f"{prefix}_DarkenRGB_native.png": "02_DarkenRGB_native_346x260.png",
        f"{prefix}_Event_main_white.png": "03_Event_main_white_346x260.png",
        f"{prefix}_Event_main_black.png": "04_Event_main_black_346x260.png",
        f"{prefix}_Event_original.pt": "05_Event_original_8x260x346.pt",
    }
    for source_name, output_name in mapping.items():
        shutil.copy2(NATIVE_ROOT / source_name, inputs / output_name)
    for path in inputs.glob("*.png"):
        with Image.open(path) as image:
            if image.size != (346, 260):
                raise RuntimeError(f"Non-native MVSEC input: {path} {image.size}")
    return inputs


def archive_arrays(view_root, before_dir, after_dir):
    arrays_root = view_root / "arrays_and_metrics"
    arrays_root.mkdir(parents=True, exist_ok=True)
    mapping = {
        before_dir / "geometry_arrays.npz": "geometry_arrays_before.npz",
        after_dir / "geometry_arrays.npz": "geometry_arrays_ft10.npz",
        before_dir / "geometry_points.npz": "geometry_points_before.npz",
        after_dir / "geometry_points.npz": "geometry_points_ft10.npz",
        before_dir / "metrics.json": "metrics_before.json",
        after_dir / "metrics.json": "metrics_ft10.json",
    }
    for source, output_name in mapping.items():
        shutil.copy2(source, arrays_root / output_name)
    return arrays_root


def render_geometry(view_root, arrays, k_matrix, depth_lo, depth_hi, view_info):
    geometry_root = view_root / "geometry_white"
    geometry_root.mkdir(parents=True, exist_ok=True)
    depths = {
        "mast3r": arrays["mast3r"],
        "night3r_before": arrays["night3r_before"],
        "night3r_ft10": arrays["night3r_ft10"],
        "night3r_zero_ft10": arrays["night3r_zero_ft10"],
        "gt": arrays["gt"],
    }
    points = {
        name: backproject(depth, k_matrix, stride=2)
        for name, depth in depths.items()
    }
    reference = points["gt"]
    lower, upper = robust_volume(reference)
    points = {
        name: crop_points(method_points, lower, upper)
        for name, method_points in points.items()
    }
    reference = points["gt"]
    center = 0.5 * (
        np.quantile(reference, 0.01, axis=0)
        + np.quantile(reference, 0.99, axis=0)
    )
    cameras = {
        "front": fitted_camera(
            reference, center, [0.0, 0.0, -1.0], 720, 540
        ),
        "elevated": fitted_camera(
            reference, center, [0.36, -0.34, -1.10], 720, 540
        ),
    }
    for method, method_points in points.items():
        for view, camera in cameras.items():
            rendered = render_method(
                method_points,
                camera,
                center,
                depth_lo,
                depth_hi,
                720,
                540,
                1,
            )
            Image.fromarray(rendered).save(
                geometry_root / f"{method}_{view}.png",
                optimize=True,
            )

    protocol = {
        "source_frame": view_info["frame"],
        "inference_order": view_info["inference_order"],
        "coordinate_frame": "first camera of the listed inference order",
        "depth_representation": "metric Z-depth, median-aligned to view GT",
        "intrinsics_at_512x512": k_matrix.tolist(),
        "shared_dual_view_depth_color_range": [depth_lo, depth_hi],
        "crop_policy": "per-view GT 1%-99% volume with 10% margin",
        "crop_lower": lower.tolist(),
        "crop_upper": upper.tolist(),
        "center": center.tolist(),
        "cameras": {
            name: camera.tolist() for name, camera in cameras.items()
        },
        "point_counts": {
            name: int(len(method_points))
            for name, method_points in points.items()
        },
    }
    (geometry_root / "geometry_protocol.json").write_text(
        json.dumps(protocol, indent=2) + "\n",
        encoding="utf-8",
    )
    return geometry_root


def build_view(view_name, view_info, depth_lo, depth_hi, k_matrix):
    view_root = PAPER_ROOT / view_name
    view_root.mkdir(parents=True, exist_ok=True)
    inputs_root = copy_native_inputs(view_name, view_info, view_root)
    before_dir = candidate_dir(view_info["before_root"])
    after_dir = candidate_dir(view_info["after_root"])
    arrays_root = archive_arrays(view_root, before_dir, after_dir)

    with np.load(before_dir / "geometry_arrays.npz") as before, np.load(
        after_dir / "geometry_arrays.npz"
    ) as after:
        arrays = {
            "mast3r": before["master_aligned"].copy(),
            "night3r_before": before["night_aligned"].copy(),
            "night3r_ft10": after["night_aligned"].copy(),
            "night3r_zero_ft10": after["zero_aligned"].copy(),
            "gt": after["gt"].copy(),
            "valid": after["valid"].copy(),
        }

    depth_root = view_root / "depth_512x512_shared_viridis"
    depth_root.mkdir(parents=True, exist_ok=True)
    depth_files = {
        "MASt3R": ("01_MASt3R_depth.png", arrays["mast3r"], None),
        "Night3R Before": (
            "02_Night3R_before_depth.png",
            arrays["night3r_before"],
            None,
        ),
        "Night3R FT10": (
            "03_Night3R_ft10_depth.png",
            arrays["night3r_ft10"],
            None,
        ),
        "Night3R Zero": (
            "04_Night3R_zero_ft10_depth.png",
            arrays["night3r_zero_ft10"],
            None,
        ),
        "Ground Truth": ("05_GT_depth.png", arrays["gt"], arrays["valid"]),
    }
    depth_images = []
    for _, (filename, depth, valid) in depth_files.items():
        image = depth_image(depth, depth_lo, depth_hi, valid)
        image.save(depth_root / filename, optimize=True)
        depth_images.append(image)

    labeled_sheet(
        [depth_images],
        list(depth_files),
        [f"{view_name} | source frame {view_info['frame']}"],
        view_root / "01_depth_comparison.png",
        (512, 512),
    )

    input_names = (
        "01_RGB_original_346x260.png",
        "02_DarkenRGB_native_346x260.png",
        "03_Event_main_white_346x260.png",
        "04_Event_main_black_346x260.png",
    )
    input_labels = ("RGB", "Low-light RGB", "Event (white)", "Event (black)")
    input_images = [Image.open(inputs_root / name).convert("RGB") for name in input_names]
    labeled_sheet(
        [input_images],
        input_labels,
        [f"{view_name} | source frame {view_info['frame']}"],
        view_root / "00_input_comparison.png",
        (346, 260),
    )

    geometry_root = render_geometry(
        view_root,
        arrays,
        k_matrix,
        depth_lo,
        depth_hi,
        view_info,
    )
    for view in ("front", "elevated"):
        geometry_images = [
            Image.open(geometry_root / f"{method}_{view}.png").convert("RGB")
            for method in ("mast3r", "night3r_ft10", "gt")
        ]
        labeled_sheet(
            [geometry_images],
            ["MASt3R", "Night3R FT10", "Ground Truth"],
            [f"{view_name} | source frame {view_info['frame']}"],
            view_root / f"0{2 if view == 'front' else 3}_geometry_{view}.png",
            (720, 540),
        )

    before_metrics = json.loads(
        (arrays_root / "metrics_before.json").read_text(encoding="utf-8")
    )
    after_metrics = json.loads(
        (arrays_root / "metrics_ft10.json").read_text(encoding="utf-8")
    )
    summary = {
        "view": view_name,
        "source_frame": view_info["frame"],
        "inference_order": view_info["inference_order"],
        "metrics_before": before_metrics["metrics"],
        "metrics_ft10": after_metrics["metrics"],
        "shared_dual_view_depth_color_range": [depth_lo, depth_hi],
    }
    (view_root / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n",
        encoding="utf-8",
    )
    return {
        "view_root": view_root,
        "inputs_root": inputs_root,
        "depth_root": depth_root,
        "geometry_root": geometry_root,
        "summary": summary,
    }


def pair_comparisons(outputs):
    pair_root = PAPER_ROOT / "pair_comparisons"
    pair_root.mkdir(parents=True, exist_ok=True)
    view_names = ("view1", "view2")
    row_labels = [
        f"{name} | source frame {VIEW_SOURCES[name]['frame']}"
        for name in view_names
    ]

    input_names = (
        "01_RGB_original_346x260.png",
        "02_DarkenRGB_native_346x260.png",
        "03_Event_main_white_346x260.png",
        "04_Event_main_black_346x260.png",
    )
    input_rows = [
        [
            Image.open(outputs[name]["inputs_root"] / filename).convert("RGB")
            for filename in input_names
        ]
        for name in view_names
    ]
    labeled_sheet(
        input_rows,
        ["RGB", "Low-light RGB", "Event (white)", "Event (black)"],
        row_labels,
        pair_root / "01_inputs_dual_view.png",
        (346, 260),
    )

    depth_names = (
        "01_MASt3R_depth.png",
        "02_Night3R_before_depth.png",
        "03_Night3R_ft10_depth.png",
        "04_Night3R_zero_ft10_depth.png",
        "05_GT_depth.png",
    )
    depth_rows = [
        [
            Image.open(outputs[name]["depth_root"] / filename).convert("RGB")
            for filename in depth_names
        ]
        for name in view_names
    ]
    labeled_sheet(
        depth_rows,
        ["MASt3R", "Night3R Before", "Night3R FT10", "Zero", "GT"],
        row_labels,
        pair_root / "02_depths_dual_view.png",
        (512, 512),
    )

    for view in ("front", "elevated"):
        geometry_rows = [
            [
                Image.open(
                    outputs[name]["geometry_root"] / f"{method}_{view}.png"
                ).convert("RGB")
                for method in ("mast3r", "night3r_ft10", "gt")
            ]
            for name in view_names
        ]
        labeled_sheet(
            geometry_rows,
            ["MASt3R", "Night3R FT10", "Ground Truth"],
            row_labels,
            pair_root / f"0{3 if view == 'front' else 4}_geometry_{view}_dual_view.png",
            (720, 540),
        )
    return pair_root


def archive_legacy_root_files():
    legacy_root = PAPER_ROOT / "legacy_single_view_export"
    legacy_root.mkdir(parents=True, exist_ok=True)
    root_file_pattern = re.compile(r"^(0[1-9]|1[0-4])_.*\.png$")
    for path in list(PAPER_ROOT.iterdir()):
        if path.is_file() and (
            root_file_pattern.match(path.name)
            or path.name
            in {
                "geometry_before.npz",
                "geometry_ft10.npz",
                "geometry_points_before.npz",
                "geometry_points_ft10.npz",
                "metrics_before.json",
                "metrics_ft10.json",
            }
        ):
            destination = legacy_root / path.name
            if destination.exists():
                destination.unlink()
            shutil.move(str(path), str(destination))
    old_geometry = PAPER_ROOT / "geometry_white"
    if old_geometry.is_dir():
        destination = legacy_root / "geometry_white"
        if destination.exists():
            shutil.rmtree(destination)
        shutil.move(str(old_geometry), str(destination))
    return legacy_root


def main():
    if not NATIVE_ROOT.is_dir():
        raise FileNotFoundError(NATIVE_ROOT)
    view_candidates = {}
    gt_values = []
    for view_name, view_info in VIEW_SOURCES.items():
        before_dir = candidate_dir(view_info["before_root"])
        after_dir = candidate_dir(view_info["after_root"])
        view_candidates[view_name] = (before_dir, after_dir)
        with np.load(after_dir / "geometry_arrays.npz") as arrays:
            gt = arrays["gt"]
            valid = arrays["valid"].astype(bool)
            values = gt[valid & np.isfinite(gt) & (gt > 0)]
            gt_values.append(values)
    shared_gt = np.concatenate(gt_values)
    depth_lo, depth_hi = np.quantile(shared_gt, (0.02, 0.98))
    k_matrix = intrinsics()

    outputs = {
        view_name: build_view(
            view_name,
            view_info,
            float(depth_lo),
            float(depth_hi),
            k_matrix,
        )
        for view_name, view_info in VIEW_SOURCES.items()
    }
    pair_root = pair_comparisons(outputs)
    legacy_root = archive_legacy_root_files()

    summary = {
        "scene": SCENE,
        "source_pair": ["000036", "000037"],
        "view1_inference_order": VIEW_SOURCES["view1"]["inference_order"],
        "view2_inference_order": VIEW_SOURCES["view2"]["inference_order"],
        "view2_policy": (
            "The source pair is reversed so frame 000037 is the first camera; "
            "depth and geometry are therefore expressed in the view2 camera frame."
        ),
        "shared_dual_view_depth_color_range": [
            float(depth_lo),
            float(depth_hi),
        ],
        "intrinsics_at_512x512": k_matrix.tolist(),
        "directories": {
            "view1": "view1",
            "view2": "view2",
            "pair_comparisons": str(pair_root.relative_to(PAPER_ROOT)),
            "legacy_single_view_export": str(
                legacy_root.relative_to(PAPER_ROOT)
            ),
        },
    }
    (PAPER_ROOT / "dual_view_manifest.json").write_text(
        json.dumps(summary, indent=2) + "\n",
        encoding="utf-8",
    )

    metrics_rows = []
    for view_name in ("view1", "view2"):
        before = outputs[view_name]["summary"]["metrics_before"]["night"]
        after = outputs[view_name]["summary"]["metrics_ft10"]["night"]
        metrics_rows.append(
            (
                view_name,
                before["raw_absrel"],
                after["raw_absrel"],
                before["raw_delta1"],
                after["raw_delta1"],
            )
        )
    readme = [
        "# MVSEC 003 dual-view local FT10 effect check",
        "",
        "The pair is now organized explicitly by camera view.",
        "",
        "- `view1/`: source frame `000036`, inference order `000036 -> 000037`.",
        "- `view2/`: source frame `000037`, inference order `000037 -> 000036`.",
        "- `pair_comparisons/`: dual-view input, depth, front-geometry, and "
        "elevated-geometry sheets.",
        "- `legacy_single_view_export/`: the previous view1-only root export.",
        "",
        "View2 is not a copied or interpolated view1 prediction. It is produced "
        "by reversing the pair and rerunning MASt3R, Night3R-before, Night3R-FT10, "
        "and the zero-event control, so its depth and geometry use the view2 "
        "camera frame.",
        "",
        "| View | Before raw AbsRel | FT10 raw AbsRel | Before delta1 | FT10 delta1 |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in metrics_rows:
        readme.append(
            f"| {row[0]} | {row[1]:.4f} | {row[2]:.4f} | "
            f"{row[3]:.4f} | {row[4]:.4f} |"
        )
    readme.extend(
        (
            "",
            f"All depth images use one dual-view Viridis range: "
            f"`{depth_lo:.4f}-{depth_hi:.4f} m`. Native RGB/event displays remain "
            "`346x260`; model depth and geometry arrays remain `512x512`.",
            "",
            "This remains a deliberate one-pair local fit and is not held-out "
            "benchmark evidence.",
            "",
        )
    )
    (PAPER_ROOT / "README.md").write_text(
        "\n".join(readme),
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
