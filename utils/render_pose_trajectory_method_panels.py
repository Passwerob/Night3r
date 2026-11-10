#!/usr/bin/env python3
"""Render one matched GT trajectory panel per evaluated method."""

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
METHOD_COLORS = ("#0f766e", "#2563eb", "#b45309", "#be123c")


def parse_args():
    parser = argparse.ArgumentParser("Render matched per-method 3D trajectory panels")
    parser.add_argument("--aligned-npz", required=True)
    parser.add_argument("--labels", nargs="+", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--display-poses", type=int, default=10)
    parser.add_argument("--columns", type=int, default=2)
    parser.add_argument("--elevation", type=float, default=24.0)
    parser.add_argument("--azimuth", type=float, default=-62.0)
    parser.add_argument("--no-panel-titles", action="store_true")
    parser.add_argument(
        "--endpoints-only",
        action="store_true",
        help="Suppress intermediate trajectory nodes and retain only start/end markers.",
    )
    parser.add_argument(
        "--shared-legend",
        action="store_true",
        help="Use one figure-level legend instead of repeating one in every panel.",
    )
    parser.add_argument(
        "--figure-width",
        type=float,
        default=None,
        help="Figure width in inches; defaults to the existing multi-column layout.",
    )
    parser.add_argument(
        "--row-height",
        type=float,
        default=6.9,
        help="Height in inches per panel row.",
    )
    parser.add_argument(
        "--legend-columns",
        type=int,
        default=2,
        help="Number of columns in the shared legend.",
    )
    parser.add_argument(
        "--ground-plane-xz",
        action="store_true",
        help="Display X/Z as the two ground-plane axes and Y as the vertical axis.",
    )
    return parser.parse_args()


def display_indices(num_poses, count):
    if count < 3:
        raise ValueError("--display-poses must be at least 3")
    return np.unique(np.linspace(0, num_poses - 1, min(count, num_poses), dtype=np.int64))


def display_coordinates(poses, indices, ground_plane_xz):
    points = poses[indices, :3, 3]
    if ground_plane_xz:
        return points[:, (0, 2, 1)]
    return points


def draw_panel(
    axis,
    gt,
    prediction,
    label,
    color,
    indices,
    limits,
    ground_plane_xz,
    elevation,
    azimuth,
    show_title,
    show_legend,
    compact,
    show_intermediate_nodes,
):
    gt_nodes = display_coordinates(gt, indices, ground_plane_xz)
    pred_nodes = display_coordinates(prediction, indices, ground_plane_xz)
    line_scale = 0.25 if compact else 1.0
    endpoint_size = 20 if compact else 38
    endpoint_edgewidth = 0.7 if compact else 1.0
    axis.plot(
        *gt_nodes.T,
        color=GT_COLOR,
        linewidth=2.8 * line_scale,
        label="Ground truth",
    )
    axis.plot(
        *pred_nodes.T,
        color=color,
        linewidth=2.5 * line_scale,
        label=label,
    )
    for points, point_color in ((gt_nodes, GT_COLOR), (pred_nodes, color)):
        axis.scatter(
            *points[0],
            color=point_color,
            marker="o",
            s=endpoint_size,
            linewidths=endpoint_edgewidth,
            depthshade=False,
        )
        axis.scatter(
            *points[-1],
            color=point_color,
            marker="s",
            s=endpoint_size,
            linewidths=endpoint_edgewidth,
            depthshade=False,
        )
        if show_intermediate_nodes and len(points) > 2:
            axis.scatter(*points[1:-1].T, color=point_color, s=12, depthshade=False)
    axis.set_xlim(*limits[0])
    axis.set_ylim(*limits[1])
    axis.set_zlim(*limits[2])
    axis.set_box_aspect((1.0, 1.0, 1.0))
    if compact:
        for axis_info in (axis.xaxis, axis.yaxis, axis.zaxis):
            axis_info._axinfo["grid"]["color"] = "#d1d5db"
            axis_info._axinfo["grid"]["linewidth"] = 0.35
            axis_info.line.set_color("#9ca3af")
            axis_info.line.set_linewidth(0.4)
            axis_info.pane.set_edgecolor("#e5e7eb")
    if ground_plane_xz:
        axis_labels = ("World X (m)", "World Z (m)", "World Y (m)")
    else:
        axis_labels = ("World X (m)", "World Y (m)", "World Z (m)")
    if compact:
        axis.set_xlabel("X", fontsize=7, labelpad=-5)
        axis.set_ylabel("Z" if ground_plane_xz else "Y", fontsize=7, labelpad=-5)
        axis.set_zlabel("Y" if ground_plane_xz else "Z", fontsize=7, labelpad=-5)
        axis.tick_params(axis="both", which="major", labelsize=5, pad=-3)
    else:
        axis.set_xlabel(axis_labels[0])
        axis.set_ylabel(axis_labels[1])
        axis.set_zlabel(axis_labels[2])
    axis.view_init(elev=elevation, azim=azimuth)
    if show_title:
        title = label if compact else f"Ground truth + {label}"
        axis.set_title(title, fontsize=8 if compact else 13, fontweight="semibold" if compact else "normal", pad=0 if compact else 10)
    if show_legend:
        axis.legend(frameon=False, loc="upper left", fontsize=9)


def main():
    args = parse_args()
    if args.columns < 1:
        raise ValueError("--columns must be positive")
    payload = np.load(args.aligned_npz)
    gt = payload["gt_camera_to_world"]
    prediction_keys = sorted(
        (key for key in payload.files if key.startswith("pred_") and key.endswith("_sim3_aligned")),
        key=lambda key: int(key.split("_")[1]),
    )
    if len(prediction_keys) != len(args.labels):
        raise ValueError("--labels must match the number of aligned predictions")
    predictions = [payload[key] for key in prediction_keys]
    if any(prediction.shape != gt.shape for prediction in predictions):
        raise ValueError("All predictions must have the same shape as the GT trajectory")

    indices = display_indices(len(gt), args.display_poses)
    positions = np.concatenate(
        [display_coordinates(gt, np.arange(len(gt)), args.ground_plane_xz)]
        + [display_coordinates(pred, np.arange(len(pred)), args.ground_plane_xz) for pred in predictions],
        axis=0,
    )
    center = positions.mean(axis=0)
    half = max(float(np.max(np.ptp(positions, axis=0))) * 0.56, 0.6)
    limits = tuple((center[dim] - half, center[dim] + half) for dim in range(3))

    rows = int(np.ceil(len(predictions) / args.columns))
    figure_width = args.figure_width if args.figure_width is not None else 7.25 * args.columns
    figure = plt.figure(
        figsize=(figure_width, args.row_height * rows), dpi=220, facecolor="white"
    )
    for index, (label, prediction) in enumerate(zip(args.labels, predictions), start=1):
        axis = figure.add_subplot(rows, args.columns, index, projection="3d")
        draw_panel(
            axis,
            gt,
            prediction,
            label,
            METHOD_COLORS[(index - 1) % len(METHOD_COLORS)],
            indices,
            limits,
            args.ground_plane_xz,
            args.elevation,
            args.azimuth,
            not args.no_panel_titles,
            not args.shared_legend,
            args.shared_legend,
            not args.endpoints_only,
        )
    if args.shared_legend:
        legend_handles = [
            Line2D([0], [0], color=GT_COLOR, linewidth=0.9, label="Ground truth")
        ] + [
            Line2D(
                [0],
                [0],
                color=METHOD_COLORS[index % len(METHOD_COLORS)],
                linewidth=0.9,
                label=label,
            )
            for index, label in enumerate(args.labels)
        ]
        figure.legend(
            handles=legend_handles,
            loc="lower center",
            bbox_to_anchor=(0.5, 0.005),
            ncol=args.legend_columns,
            frameon=False,
            prop={"size": 6.8, "weight": "semibold"},
            handlelength=2.1,
            columnspacing=0.8,
            labelspacing=0.35,
        )
        figure.subplots_adjust(
            left=0.01, right=0.99, bottom=0.13, top=0.99, wspace=0.01, hspace=0.08
        )
    else:
        figure.tight_layout()

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, bbox_inches="tight", facecolor="white")
    figure.savefig(output.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")


if __name__ == "__main__":
    main()
