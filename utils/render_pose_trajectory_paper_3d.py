#!/usr/bin/env python3
"""Render uncluttered, matched 3D pose-trajectory comparison panels."""

import argparse
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D


GT_COLOR = "#1f2937"
METHOD_COLORS = ("#0f766e", "#2563eb", "#c2410c", "#be123c")


def parse_args():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--aligned-npz", type=Path, required=True)
    parser.add_argument("--labels", nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--display-poses", type=int, default=10)
    parser.add_argument("--elev", type=float, default=24.0)
    parser.add_argument("--azim", type=float, default=-62.0)
    parser.add_argument("--style", choices=("floor", "frame"), default="floor")
    return parser.parse_args()


def display_indices(num_poses, count):
    if count < 3:
        raise ValueError("--display-poses must be at least 3")
    return np.unique(np.linspace(0, num_poses - 1, min(count, num_poses), dtype=np.int64))


def shared_limits(trajectories):
    positions = np.concatenate([trajectory[:, :3, 3] for trajectory in trajectories], axis=0)
    minimum = positions.min(axis=0)
    maximum = positions.max(axis=0)
    span = np.maximum(maximum - minimum, 1.0e-6)
    max_span = float(span.max())
    padding = max(0.055 * max_span, 0.12)
    center = 0.5 * (minimum + maximum)
    limits = tuple((center[index] - 0.5 * span[index] - padding, center[index] + 0.5 * span[index] + padding) for index in range(3))
    aspect = tuple((span / max_span).clip(0.42, 1.0))
    return limits, aspect


def configure_axis(axis, limits, aspect, elev, azim, style):
    axis.set_xlim(*limits[0])
    axis.set_ylim(*limits[1])
    axis.set_zlim(*limits[2])
    axis.set_box_aspect(aspect, zoom=1.13 if style == "frame" else 1.28)
    axis.view_init(elev=elev, azim=azim)
    axis.set_proj_type("persp", focal_length=0.9)
    axis.set_facecolor("white")
    if style == "floor":
        axis.set_axis_off()
        return
    axis.grid(False)
    axis.set_xticks(())
    axis.set_yticks(())
    axis.set_zticks(())
    axis.set_xlabel("")
    axis.set_ylabel("")
    axis.set_zlabel("")
    for pane in (axis.xaxis.pane, axis.yaxis.pane, axis.zaxis.pane):
        pane.fill = False
        pane.set_edgecolor("#cbd5e1")
        pane.set_linewidth(0.65)
    for axis_info in (axis.xaxis, axis.yaxis, axis.zaxis):
        axis_info.line.set_color("#cbd5e1")
        axis_info.line.set_linewidth(0.65)


def add_reference_floor(axis, limits):
    x_min, x_max = limits[0]
    y_min, y_max = limits[1]
    z_floor = limits[2][0]
    x_grid, y_grid = np.meshgrid(np.linspace(x_min, x_max, 2), np.linspace(y_min, y_max, 2))
    axis.plot_surface(
        x_grid,
        y_grid,
        np.full_like(x_grid, z_floor),
        color="#eef2f7",
        alpha=0.55,
        linewidth=0,
        shade=False,
        zorder=0,
    )
    for value in np.linspace(x_min, x_max, 6):
        axis.plot((value, value), (y_min, y_max), (z_floor, z_floor), color="#d4dce7", linewidth=0.65, alpha=0.72, zorder=1)
    for value in np.linspace(y_min, y_max, 6):
        axis.plot((x_min, x_max), (value, value), (z_floor, z_floor), color="#d4dce7", linewidth=0.65, alpha=0.72, zorder=1)


def plot_trajectory(axis, points, color, linewidth, node_size, endpoint_size, floor_z):
    if floor_z is not None:
        shadow = points.copy()
        shadow[:, 2] = floor_z
        axis.plot(*shadow.T, color=color, linewidth=1.05, alpha=0.17, solid_capstyle="round", zorder=1)
    axis.plot(*points.T, color=color, linewidth=linewidth, solid_capstyle="round", zorder=3)
    axis.scatter(*points[1:-1].T, color=color, s=node_size, depthshade=False, zorder=4)
    axis.scatter(
        *points[0], color="white", edgecolors=color, linewidths=1.8, marker="o", s=endpoint_size, depthshade=False, zorder=5
    )
    axis.scatter(
        *points[-1], color=color, edgecolors="white", linewidths=0.9, marker="s", s=endpoint_size, depthshade=False, zorder=5
    )


def main():
    args = parse_args()
    payload = np.load(args.aligned_npz)
    gt = payload["gt_camera_to_world"]
    prediction_keys = sorted(
        (key for key in payload.files if key.startswith("pred_") and key.endswith("_sim3_aligned")),
        key=lambda key: int(key.split("_")[1]),
    )
    if len(prediction_keys) != len(args.labels):
        raise ValueError("--labels must match the number of aligned prediction trajectories")
    predictions = [payload[key] for key in prediction_keys]
    if any(prediction.shape != gt.shape for prediction in predictions):
        raise ValueError("All trajectories must have the same [N,4,4] shape")

    indices = display_indices(len(gt), args.display_poses)
    gt_nodes = gt[indices, :3, 3]
    prediction_nodes = [prediction[indices, :3, 3] for prediction in predictions]
    limits, aspect = shared_limits([gt] + predictions)

    figure = plt.figure(figsize=(13.0, 10.0), dpi=260, facecolor="white")
    for index, (label, nodes) in enumerate(zip(args.labels, prediction_nodes)):
        axis = figure.add_subplot(2, 2, index + 1, projection="3d")
        configure_axis(axis, limits, aspect, args.elev, args.azim, args.style)
        floor_z = limits[2][0]
        if args.style == "floor":
            add_reference_floor(axis, limits)
            gt_floor_z = floor_z
        else:
            gt_floor_z = None
        plot_trajectory(axis, gt_nodes, GT_COLOR, linewidth=2.8, node_size=15, endpoint_size=48, floor_z=gt_floor_z)
        color = METHOD_COLORS[index % len(METHOD_COLORS)]
        plot_trajectory(axis, nodes, color, linewidth=3.2, node_size=18, endpoint_size=54, floor_z=gt_floor_z)
        axis.set_title(label, color=color, fontsize=18, fontweight="semibold", pad=-3)
        axis.text2D(0.02, 0.04, f"({chr(ord('a') + index)})", transform=axis.transAxes, fontsize=13, color="#374151")

    legend = (
        Line2D([0], [0], color=GT_COLOR, linewidth=2.8, label="Ground truth"),
        Line2D([0], [0], color="#6b7280", linewidth=3.2, label="Prediction"),
        Line2D([0], [0], marker="o", color="white", markeredgecolor="#374151", markersize=7, linewidth=0, label="Start"),
        Line2D([0], [0], marker="s", color="#374151", markeredgecolor="white", markersize=7, linewidth=0, label="End"),
    )
    figure.legend(handles=legend, loc="upper center", ncol=4, bbox_to_anchor=(0.5, 0.985), frameon=False, fontsize=12, handlelength=2.2, columnspacing=2.0)
    figure.subplots_adjust(left=0.01, right=0.99, top=0.93, bottom=0.02, wspace=0.005, hspace=0.055)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output, bbox_inches="tight", pad_inches=0.035, facecolor="white")
    figure.savefig(args.output.with_suffix(".pdf"), bbox_inches="tight", pad_inches=0.035, facecolor="white")
    plt.close(figure)


if __name__ == "__main__":
    main()
