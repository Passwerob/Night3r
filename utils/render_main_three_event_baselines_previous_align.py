#!/usr/bin/env python3
"""Render event-only baselines with the exact previous main-figure Align protocol."""

import json
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.package_main_three_event_baselines import (  # noqa: E402
    backproject,
    resize_depth,
    tensor_2d,
    valid_depth,
)
from utils.render_gt_median_aligned_geometry_depth import (  # noqa: E402
    compose_depth,
    compose_geometry,
    depth_colors,
    render_points,
)


BASE = ROOT / "outputs/main_three_event_baselines_20260729"
OUTPUT = BASE / "previous_align_exact"
PREVIOUS = (
    ROOT
    / "outputs/shared_scale_comparison_selected_20260727"
    / "gt_median_aligned_geometry_depth_20260728/manifest.json"
)
METHODS = (
    ("eventvggt", "EventVGGT"),
    ("depthanyevent", "DepthAnyEvent"),
    ("depthanyevent_r", "DepthAnyEvent-R"),
)
SCENES = (
    ("01_dsec_000552", "01_dsec_screen-000004_000552-000553", "screen-000004-dsec:000552-000553"),
    ("02_mvsec_000157", "02_mvsec_screen-000157_000036-000037", "screen-000157-mvsec:000036-000037"),
    ("03_mvsec_000205", "03_mvsec_screen-000205_000047-000048", "screen-000205-mvsec:000047-000048"),
)


def safe_name(value):
    return "".join(char if char.isalnum() or char in "-_" else "_" for char in value)


def alignment_scale(prediction, gt, gt_valid):
    mask = gt_valid & valid_depth(gt) & valid_depth(prediction)
    if not mask.any():
        raise RuntimeError("No overlapping valid pixels for GT median alignment")
    scale = float(np.median(gt[mask]) / np.median(prediction[mask]))
    return scale, mask


def render_scene(previous, output_name, sample_id):
    output = OUTPUT / output_name
    output.mkdir(parents=True, exist_ok=True)
    minimum, maximum = [float(value) for value in previous["color_policy"]["depth_range_m"]]
    target = np.asarray(previous["geometry_display"]["target"], dtype=np.float32)
    cameras = {
        name: np.asarray(value, dtype=np.float32)
        for name, value in previous["geometry_display"]["cameras"].items()
    }
    records = {}
    depth_panels = []
    front_panels = []
    elevated_panels = []
    labels = []

    for method, label in METHODS:
        prediction_path = BASE / "predictions" / method / f"{safe_name(sample_id)}.pt"
        payload = torch.load(prediction_path, map_location="cpu", weights_only=True)
        gt = tensor_2d(payload["gt_depth"]).numpy()
        gt_valid = tensor_2d(payload["gt_valid"]).numpy().astype(bool)
        prediction = resize_depth(payload["pred_depth"], gt.shape)
        scale, overlap = alignment_scale(prediction, gt, gt_valid)
        aligned = (prediction * scale).astype(np.float32)
        intrinsics = np.asarray(payload["intrinsics"], dtype=np.float32)
        points = backproject(aligned, intrinsics, stride=2)

        method_dir = output / method
        method_dir.mkdir(parents=True, exist_ok=True)
        np.save(method_dir / "depth_GT_median_aligned.npy", aligned)
        np.savez_compressed(method_dir / "geometry_GT_median_aligned.npz", points=points)
        depth_panel = depth_colors(aligned, minimum, maximum)
        depth_path = method_dir / "depth_GT_median_aligned.png"
        Image.fromarray(depth_panel).save(depth_path, optimize=True)

        geometry = {}
        rendered_panels = {}
        point_colors = depth_colors(points[:, 2], minimum, maximum)
        for view in ("front", "elevated"):
            rendered = render_points(
                points,
                point_colors,
                cameras[view],
                target,
                width=620,
                height=500,
                radius=1,
            )
            path = method_dir / f"geometry_{view}_GT_median_aligned.png"
            rendered.save(path, optimize=True)
            geometry[view] = path.name
            rendered_panels[view] = rendered

        records[method] = {
            "display_name": label,
            "prediction": str(prediction_path),
            "alignment_formula": "s_m = median(D_gt[valid & D_m>0]) / median(D_m[valid & D_m>0])",
            "GT_median_scale": scale,
            "overlap_valid_px": int(overlap.sum()),
            "applied_to": "complete depth map and all backprojected XYZ coordinates",
            "depth": depth_path.name,
            "geometry": geometry,
            "point_count": int(len(points)),
        }
        labels.append(label)
        depth_panels.append(depth_panel)
        front_panels.append(rendered_panels["front"])
        elevated_panels.append(rendered_panels["elevated"])

    compose_depth(
        depth_panels,
        labels,
        output / "event_methods_depth_GT_median_aligned.png",
    )
    compose_geometry(
        front_panels,
        labels,
        minimum,
        maximum,
        output / "event_methods_geometry_front_GT_median_aligned.png",
    )
    compose_geometry(
        elevated_panels,
        labels,
        minimum,
        maximum,
        output / "event_methods_geometry_elevated_GT_median_aligned.png",
    )
    record = {
        "sample_id": sample_id,
        "alignment_protocol_source": str(PREVIOUS),
        "alignment_policy": previous["alignment"],
        "geometry_display": {
            **previous["geometry_display"],
            "panel_size_wh": [620, 500],
        },
        "color_policy": previous["color_policy"],
        "methods": records,
    }
    (output / "alignment_manifest.json").write_text(json.dumps(record, indent=2) + "\n")
    return record


def main():
    previous_records = {
        record["scene"]: record for record in json.loads(PREVIOUS.read_text())
    }
    records = [
        render_scene(previous_records[previous_name], output_name, sample_id)
        for previous_name, output_name, sample_id in SCENES
    ]
    (OUTPUT / "manifest.json").write_text(json.dumps(records, indent=2) + "\n")
    (OUTPUT / "README.md").write_text(
        "# Event-only baselines using the previous main-figure Align protocol\n\n"
        "Every depth and XYZ point cloud uses one per-method GT median scale. "
        "The GT Viridis range, target, front/elevated cameras, 620x500 geometry "
        "canvas, and no-auto-fit policy are copied from "
        "`gt_median_aligned_geometry_depth_20260728`. Raw predictions are not "
        "included in this plot-only directory.\n"
    )
    print(json.dumps({"output": str(OUTPUT), "scenes": len(records)}, indent=2))


if __name__ == "__main__":
    main()
