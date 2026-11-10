#!/usr/bin/env python3
"""Evaluate GT-pose-controlled sparse-view fusion against sensor-depth geometry."""

import argparse
import csv
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial import cKDTree

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from dataset.dsec_event_dataset import DSECEventDataset


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prediction-root", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--scene", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--image-size", type=int, default=512)
    parser.add_argument("--depth-stride", type=int, default=4)
    parser.add_argument("--voxel-size", type=float, default=0.08)
    parser.add_argument("--distance-threshold", type=float, default=0.50)
    return parser.parse_args()


def load_ply_points(path):
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


def voxel_downsample(points, voxel_size):
    if voxel_size <= 0 or len(points) == 0:
        return points
    origin = points.min(axis=0)
    keys = np.floor((points - origin) / voxel_size).astype(np.int64)
    _, first = np.unique(keys, axis=0, return_index=True)
    return points[np.sort(first)]


def build_gt_cloud(dataset, scene_dir, frame_indices, depth_stride, voxel_size):
    intrinsics = dataset.scene_camera[scene_dir].detach().cpu().numpy().astype(np.float32)
    poses = dataset.scene_poses[scene_dir].detach().cpu().numpy().astype(np.float32)
    frames = dataset.scene_frames[scene_dir]
    reference_inverse = np.linalg.inv(poses[frame_indices[0]])
    clouds = []
    for frame_index in frame_indices:
        depth = dataset._load_depth_gt(frames["depths"][frame_index]).numpy().astype(np.float32)
        height, width = depth.shape
        pixel_y, pixel_x = np.mgrid[0:height:depth_stride, 0:width:depth_stride]
        sampled_depth = depth[::depth_stride, ::depth_stride]
        valid = np.isfinite(sampled_depth) & (sampled_depth > 0.25) & (sampled_depth < 80.0)
        if not valid.any():
            continue
        z = sampled_depth[valid]
        x = (pixel_x[valid].astype(np.float32) - intrinsics[0, 2]) / intrinsics[0, 0] * z
        y = (pixel_y[valid].astype(np.float32) - intrinsics[1, 2]) / intrinsics[1, 1] * z
        points = np.stack([x, y, z], axis=1)
        transform = reference_inverse @ poses[frame_index]
        points = points @ transform[:3, :3].T + transform[:3, 3]
        clouds.append(points.astype(np.float32))
    if not clouds:
        raise RuntimeError("No valid sensor-depth points were found")
    # Match the display convention used by export_night3r_long_sequence.py.
    cloud = np.concatenate(clouds, axis=0)
    cloud[:, 1] *= -1.0
    return voxel_downsample(cloud, voxel_size)


def median_radial_scale(prediction, ground_truth):
    pred_radius = np.linalg.norm(prediction, axis=1)
    gt_radius = np.linalg.norm(ground_truth, axis=1)
    return float(np.median(gt_radius) / max(np.median(pred_radius), 1.0e-6))


def geometry_metrics(prediction, ground_truth, threshold):
    gt_tree = cKDTree(ground_truth)
    pred_tree = cKDTree(prediction)
    pred_to_gt = gt_tree.query(prediction, workers=-1)[0]
    gt_to_pred = pred_tree.query(ground_truth, workers=-1)[0]
    accuracy = float(pred_to_gt.mean())
    completeness = float(gt_to_pred.mean())
    precision = float((pred_to_gt <= threshold).mean())
    recall = float((gt_to_pred <= threshold).mean())
    fscore = float(2.0 * precision * recall / max(precision + recall, 1.0e-12))
    return {
        "accuracy_m": accuracy,
        "completeness_m": completeness,
        "chamfer_l1_m": 0.5 * (accuracy + completeness),
        "precision": precision,
        "recall": recall,
        "fscore": fscore,
    }


def render_chart(records, output_path):
    view_counts = [record["num_views"] for record in records]
    chart_specs = (
        ("accuracy_m", "Accuracy (m)", "#d95f02"),
        ("completeness_m", "Completeness (m)", "#7570b3"),
        ("chamfer_l1_m", "Chamfer-L1 (m)", "#1b9e77"),
        ("fscore", "F-score @ 0.50 m", "#377eb8"),
    )
    figure, axes = plt.subplots(1, len(chart_specs), figsize=(13.2, 3.2), dpi=220)
    for axis, (key, label, color) in zip(axes, chart_specs):
        values = [record[key] for record in records]
        bars = axis.bar([str(value) for value in view_counts], values, color=color, width=0.62)
        axis.set_title(label, fontsize=10)
        axis.set_xlabel("Input views", fontsize=9)
        axis.grid(axis="y", alpha=0.22, linewidth=0.7)
        axis.set_axisbelow(True)
        upper = max(values) * 1.17 if max(values) > 0 else 1.0
        axis.set_ylim(0.0, upper)
        for bar, value in zip(bars, values):
            axis.text(
                bar.get_x() + bar.get_width() * 0.5,
                value + upper * 0.025,
                f"{value:.3f}",
                ha="center",
                va="bottom",
                fontsize=8,
            )
    figure.suptitle("GT-pose-controlled sparse-view geometry", fontsize=12, y=1.02)
    figure.tight_layout()
    figure.savefig(output_path, bbox_inches="tight", facecolor="white")
    plt.close(figure)


def main():
    args = parse_args()
    prediction_root = args.prediction_root.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    dataset = DSECEventDataset(
        root_path=args.data_root,
        image_size=(args.image_size, args.image_size),
        pair_sampling="fixed",
        pose_output_convention="opencv",
        lowlight_enabled=False,
    )
    scene_dir = next((path for path in dataset.scene_frames if path.name == args.scene), None)
    if scene_dir is None:
        raise RuntimeError(f"Scene {args.scene!r} was not found under {args.data_root}")

    records = []
    for name in ("views2", "views4", "views6"):
        directory = prediction_root / name
        manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
        if manifest["sample"]["scene"] != args.scene:
            raise RuntimeError(f"{name} belongs to a different scene")
        frame_indices = [int(value) for value in manifest["sample"]["frame_indices"]]
        prediction = load_ply_points(directory / "night3r_long_rgb.ply")
        ground_truth = build_gt_cloud(
            dataset,
            scene_dir,
            frame_indices,
            args.depth_stride,
            args.voxel_size,
        )
        scale = median_radial_scale(prediction, ground_truth)
        prediction = prediction * scale
        metrics = geometry_metrics(prediction, ground_truth, args.distance_threshold)
        records.append(
            {
                "variant": name,
                "num_views": int(manifest["sample"]["num_frames"]),
                "frame_indices": frame_indices,
                "prediction_point_count": int(len(prediction)),
                "gt_point_count": int(len(ground_truth)),
                "global_scale": scale,
                **metrics,
            }
        )

    records.sort(key=lambda record: record["num_views"])
    fields = list(records[0])
    with (output_dir / "metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(records)
    protocol = {
        "dataset": "MVSEC",
        "scene": args.scene,
        "fusion_pose": "sensor GT relative poses",
        "geometry_scale_alignment": "one global median radial scale per variant",
        "point_cloud_metric": {
            "accuracy": "mean prediction-to-GT nearest-neighbor distance",
            "completeness": "mean GT-to-prediction nearest-neighbor distance",
            "chamfer_l1": "mean of accuracy and completeness",
            "fscore": f"harmonic mean of precision/recall at {args.distance_threshold:.2f} m",
        },
        "gt_cloud": {
            "sensor_depth": True,
            "depth_stride": args.depth_stride,
            "voxel_size_m": args.voxel_size,
        },
        "records": records,
        "claim_scope": "Controlled multi-view geometry ablation. GT poses are used only to isolate geometry fusion from camera-registration error.",
    }
    (output_dir / "metrics.json").write_text(json.dumps(protocol, indent=2) + "\n", encoding="utf-8")
    render_chart(records, output_dir / "sparse_view_geometry_metrics.png")
    print(json.dumps(protocol, indent=2))


if __name__ == "__main__":
    main()
