#!/usr/bin/env python3
"""Render GT+Night3R and GT+MASt3R as matched 3D trajectory panels."""

import argparse
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


GT_COLOR = "#1f2937"
NIGHT_COLOR = "#0f766e"
MAST_COLOR = "#2563eb"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--aligned-npz", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--display-poses", type=int, default=10)
    return parser.parse_args()


def display_indices(num_poses, count):
    if count < 3:
        raise ValueError("--display-poses must be at least 3")
    return np.unique(np.linspace(0, num_poses - 1, min(count, num_poses), dtype=np.int64))


def draw_panel(axis, gt, prediction, label, color, indices, limits):
    gt_nodes = gt[indices, :3, 3]
    pred_nodes = prediction[indices, :3, 3]
    axis.plot(*gt_nodes.T, color=GT_COLOR, linewidth=3.1, label="Ground truth")
    axis.plot(*pred_nodes.T, color=color, linewidth=2.7, label=label)
    for points, point_color in ((gt_nodes, GT_COLOR), (pred_nodes, color)):
        axis.scatter(*points[0], color=point_color, marker="o", s=44, depthshade=False)
        axis.scatter(*points[-1], color=point_color, marker="s", s=44, depthshade=False)
        if len(points) > 2:
            axis.scatter(*points[1:-1].T, color=point_color, s=14, depthshade=False)
    axis.set_xlim(*limits[0])
    axis.set_ylim(*limits[1])
    axis.set_zlim(*limits[2])
    axis.set_box_aspect((1.0, 1.0, 1.0))
    axis.set_xlabel("World X (m)")
    axis.set_ylabel("World Y (m)")
    axis.set_zlabel("World Z (m)")
    axis.view_init(elev=24, azim=-62)
    axis.set_title(f"Ground truth + {label}", fontsize=13, pad=12)
    axis.legend(frameon=False, loc="upper left", fontsize=10)


def main():
    args = parse_args()
    payload = np.load(args.aligned_npz)
    gt = payload["gt_camera_to_world"]
    night = payload["pred_0_sim3_aligned"]
    mast = payload["pred_1_sim3_aligned"]
    if gt.shape != night.shape or gt.shape != mast.shape:
        raise ValueError("GT and both aligned trajectories must have equal shapes")
    indices = display_indices(len(gt), args.display_poses)

    positions = np.concatenate((gt[:, :3, 3], night[:, :3, 3], mast[:, :3, 3]), axis=0)
    center = positions.mean(axis=0)
    half = max(float(np.max(np.ptp(positions, axis=0))) * 0.56, 0.6)
    limits = tuple((center[dim] - half, center[dim] + half) for dim in range(3))

    figure = plt.figure(figsize=(14.5, 7.0), dpi=200, facecolor="white")
    left = figure.add_subplot(121, projection="3d")
    right = figure.add_subplot(122, projection="3d")
    draw_panel(left, gt, night, "Night3R", NIGHT_COLOR, indices, limits)
    draw_panel(right, gt, mast, "MASt3R", MAST_COLOR, indices, limits)
    figure.tight_layout()

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, bbox_inches="tight", facecolor="white")
    figure.savefig(output.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")


if __name__ == "__main__":
    main()
