#!/usr/bin/env python3
"""Recolor selected depth and depth-colored geometry assets with Viridis."""

import argparse
import json
import shutil
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from utils.build_pair_review_assets import (  # noqa: E402
    backproject,
    crop_points,
    render_method,
    robust_volume,
)


DEFAULT_ROOT = (
    REPO_ROOT
    / "outputs/night3r_selected_pairs_step100_native_rgb_viridis_20260725"
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    return parser.parse_args()


def viridis_scalar(array, lo, hi, valid=None):
    normalized = np.clip(
        (array - float(lo)) / max(float(hi - lo), 1.0e-6), 0.0, 1.0
    )
    normalized = np.nan_to_num(
        normalized, nan=0.0, posinf=1.0, neginf=0.0
    )
    colored = cv2.applyColorMap(
        np.round(normalized * 255).astype(np.uint8), cv2.COLORMAP_VIRIDIS
    )
    colored = cv2.cvtColor(colored, cv2.COLOR_BGR2RGB)
    mask = np.isfinite(array) & (array > 0)
    if valid is not None:
        mask &= valid
    colored[~mask] = 0
    return colored


def preserve_legacy(path):
    legacy = path.with_name(f"{path.stem}_turbo_legacy{path.suffix}")
    if path.exists() and not legacy.exists():
        shutil.copy2(path, legacy)


def recolor_depth_assets(source_dir, protocol):
    arrays = np.load(source_dir / "geometry_arrays.npz")
    lo, hi = protocol["depth_color_range"]
    has_gt = "gt" in arrays.files
    if has_gt:
        valid = arrays["valid"].astype(bool)
        scalar_maps = {
            "master_depth_shared.png": (
                arrays["master_aligned"],
                None,
            ),
            "night_depth_shared.png": (
                arrays["night_aligned"],
                None,
            ),
            "zero_depth_shared.png": (
                arrays["zero_aligned"],
                None,
            ),
            "gt_depth.png": (arrays["gt"], valid),
        }
    else:
        master = arrays["master"]
        night = arrays["night_aligned_to_master"]
        zero = arrays["zero_aligned_to_night"]
        difference = np.abs(
            np.log(np.clip(night, 1.0e-6, None))
            - np.log(np.clip(master, 1.0e-6, None))
        )
        finite = np.isfinite(difference)
        difference_hi = max(
            float(np.percentile(difference[finite], 98)), 1.0e-6
        )
        scalar_maps = {
            "master_depth_shared.png": (master, None),
            "night_depth_shared.png": (night, None),
            "zero_depth_shared.png": (zero, None),
        }
        for filename, (array, valid) in scalar_maps.items():
            path = source_dir / filename
            preserve_legacy(path)
            Image.fromarray(viridis_scalar(array, lo, hi, valid)).save(path)
        difference_path = source_dir / "gt_depth.png"
        preserve_legacy(difference_path)
        Image.fromarray(
            viridis_scalar(difference, 0.0, difference_hi, finite)
        ).save(difference_path)
        return

    for filename, (array, valid) in scalar_maps.items():
        path = source_dir / filename
        preserve_legacy(path)
        Image.fromarray(viridis_scalar(array, lo, hi, valid)).save(path)


def recolor_geometry(source_dir, geometry_dir, protocol):
    arrays = np.load(source_dir / "geometry_arrays.npz")
    if "gt" in arrays.files:
        depths = {
            "mast3r": arrays["master_aligned"],
            "night3r": arrays["night_aligned"],
            "gt": arrays["gt"],
        }
        reference_name = "gt"
    else:
        depths = {
            "mast3r": arrays["master"],
            "night3r": arrays["night_aligned_to_master"],
        }
        reference_name = "mast3r"

    intrinsics = np.asarray(protocol["intrinsics"], dtype=np.float32)
    stride = int(protocol["point_stride"])
    points = {
        name: backproject(depth, intrinsics, stride)
        for name, depth in depths.items()
    }
    lower, upper = robust_volume(points[reference_name])
    points = {
        name: crop_points(method_points, lower, upper)
        for name, method_points in points.items()
    }

    center = np.asarray(protocol["camera_center"], dtype=np.float32)
    cameras = {
        "front": np.asarray(protocol["front_camera"], dtype=np.float32),
        "elevated": np.asarray(
            protocol["elevated_camera"], dtype=np.float32
        ),
    }
    lo, hi = protocol["depth_color_range"]
    for method, method_points in points.items():
        for view, camera in cameras.items():
            path = geometry_dir / f"{method}_{view}.png"
            preserve_legacy(path)
            rendered = render_method(
                method_points,
                camera,
                center,
                lo,
                hi,
                width=720,
                height=540,
                radius=1,
            )
            Image.fromarray(rendered).save(path, optimize=True)


def patch_selection_panel(source_dir):
    panel_path = source_dir / "selection_panel.png"
    preserve_legacy(panel_path)
    panel = Image.open(panel_path).convert("RGB")
    depth_paths = (
        source_dir / "master_depth_shared.png",
        source_dir / "night_depth_shared.png",
        source_dir / "zero_depth_shared.png",
        source_dir / "gt_depth.png",
    )
    if panel.size != (1304, 836):
        return
    for column, depth_path in enumerate(depth_paths):
        depth = Image.open(depth_path).convert("RGB").resize(
            (320, 320), Image.Resampling.BILINEAR
        )
        panel.paste(depth, (column * 328, 516))
    panel.save(panel_path, optimize=True)


def main():
    args = parse_args()
    root = args.root.resolve()
    pair_dirs = sorted(
        path.parent.parent
        for path in root.glob(
            "pairs/*/*/source_assets/geometry_arrays.npz"
        )
    )
    records = []
    for pair_dir in pair_dirs:
        source_dir = pair_dir / "source_assets"
        geometry_dir = pair_dir / "geometry_assets"
        protocol_path = geometry_dir / "geometry_protocol.json"
        with protocol_path.open() as handle:
            protocol = json.load(handle)
        recolor_depth_assets(source_dir, protocol)
        recolor_geometry(source_dir, geometry_dir, protocol)
        patch_selection_panel(source_dir)
        protocol["colormap"] = "viridis"
        protocol["colormap_note"] = (
            "Perceptually uniform Viridis: dark purple to blue to green to "
            "yellow. Invalid depth remains black."
        )
        with protocol_path.open("w") as handle:
            json.dump(protocol, handle, indent=2)
        records.append(
            {
                "pair": str(pair_dir.relative_to(root)),
                "depth_color_range": protocol["depth_color_range"],
                "has_gt": bool(protocol["has_gt"]),
                "colormap": "viridis",
            }
        )

    manifest = {
        "version": 1,
        "colormap": "viridis",
        "scope": [
            "MASt3R/Night3R/Zero/GT depth PNGs",
            "DL3DV model-difference scalar PNG",
            "front/elevated depth-colored geometry renderings",
            "depth tiles in selection panels",
        ],
        "invalid_depth_color": "black",
        "records": records,
    }
    with (root / "metadata/viridis_colormap_manifest.json").open(
        "w"
    ) as handle:
        json.dump(manifest, handle, indent=2)
    print(
        json.dumps(
            {
                "root": str(root),
                "pairs": len(records),
                "colormap": "viridis",
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
