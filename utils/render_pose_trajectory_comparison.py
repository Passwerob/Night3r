#!/usr/bin/env python3
"""Render GT-aligned trajectory comparisons from Night3R pose-evaluation JSON."""

import argparse
import copy
import csv
import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from evo.core.trajectory import PosePath3D


METHOD_COLORS = ("#0f766e", "#2563eb", "#b45309", "#be123c")
GT_COLOR = "#1f2937"


def parse_args():
    parser = argparse.ArgumentParser("Render GT-aligned pose trajectory comparisons")
    parser.add_argument("--inputs", nargs="+", required=True, help="Evaluation JSON files")
    parser.add_argument("--labels", nargs="+", required=True, help="One label per input")
    parser.add_argument("--scene", required=True)
    parser.add_argument("--outdir", required=True)
    parser.add_argument("--title", default=None)
    parser.add_argument(
        "--display-poses",
        type=int,
        default=None,
        help=(
            "Render this many uniformly spaced original pose samples. Metrics and "
            "alignment still use the complete inferred trajectory."
        ),
    )
    parser.add_argument(
        "--show-pose-nodes",
        action="store_true",
        help="Mark every displayed original pose node in the 3D trajectory plot.",
    )
    parser.add_argument(
        "--clean-3d",
        action="store_true",
        help="Render the 3D plot without a title or orientation arrows.",
    )
    return parser.parse_args()


def select_scene(payload, scene_name):
    matches = [item for item in payload.get("scenes", []) if item.get("scene") == scene_name]
    if len(matches) != 1:
        raise RuntimeError(
            f"Expected one exported trajectory for {scene_name!r}, found {len(matches)}"
        )
    result = matches[0]
    trajectory = result.get("trajectory")
    if not isinstance(trajectory, dict):
        raise RuntimeError(
            f"{scene_name!r} has no exported trajectory. Re-run evaluation with "
            "--export-trajectories."
        )
    frame_indices = np.asarray(trajectory.get("frame_indices"), dtype=np.int64)
    predicted = np.asarray(trajectory.get("pred_camera_to_world"), dtype=np.float64)
    ground_truth = np.asarray(trajectory.get("gt_camera_to_world"), dtype=np.float64)
    if predicted.ndim != 3 or predicted.shape[1:] != (4, 4):
        raise RuntimeError("Predicted poses must have shape [N,4,4]")
    if ground_truth.shape != predicted.shape or len(frame_indices) != len(predicted):
        raise RuntimeError("Frame indices, predicted poses, and GT poses must have the same length")
    if len(predicted) < 3:
        raise RuntimeError("At least three contiguous frames are required for trajectory rendering")
    return result, frame_indices, predicted, ground_truth


def align_to_gt(predicted, ground_truth):
    prediction_path = PosePath3D(poses_se3=[pose.copy() for pose in predicted])
    gt_path = PosePath3D(poses_se3=[pose.copy() for pose in ground_truth])
    aligned = copy.deepcopy(prediction_path)
    aligned.align(gt_path, correct_scale=True)
    return np.asarray(aligned.poses_se3), np.asarray(gt_path.poses_se3)


def metric_value(scene, key):
    metrics = scene.get("metrics") or {}
    section = metrics.get(key) or {}
    value = section.get("rmse")
    return None if value is None else float(value)


def fmt(value, digits=3):
    return "--" if value is None else f"{value:.{digits}f}"


def principal_plane(points):
    centered = points - points.mean(axis=0, keepdims=True)
    _, _, vectors = np.linalg.svd(centered, full_matrices=False)
    first = vectors[0]
    second = vectors[1]
    return np.stack((first, second), axis=1)


def project(points, plane, origin):
    return (points - origin) @ plane


def select_display_indices(num_poses, display_poses):
    if display_poses is None or display_poses >= num_poses:
        return np.arange(num_poses, dtype=np.int64)
    if display_poses < 3:
        raise ValueError("--display-poses must be at least three")
    return np.unique(np.linspace(0, num_poses - 1, display_poses, dtype=np.int64))


def draw_path(axis, points, color, label, linewidth=2.4, alpha=1.0):
    axis.plot(points[:, 0], points[:, 1], color=color, linewidth=linewidth, alpha=alpha, label=label)
    axis.scatter(points[0, 0], points[0, 1], color=color, marker="o", s=42, zorder=4)
    axis.scatter(points[-1, 0], points[-1, 1], color=color, marker="s", s=42, zorder=4)
    stride = max(1, len(points) // 8)
    deltas = np.diff(points, axis=0)
    for index in range(0, len(deltas), stride):
        delta = deltas[index]
        length = float(np.linalg.norm(delta))
        if length > 1.0e-8:
            axis.arrow(
                points[index, 0],
                points[index, 1],
                delta[0] * 0.45,
                delta[1] * 0.45,
                color=color,
                width=0.0,
                head_width=0.022,
                head_length=0.035,
                length_includes_head=True,
                alpha=alpha,
                zorder=3,
            )


def render_3d_trajectory(
    outdir,
    title,
    ground_truth,
    methods,
    display_indices,
    show_pose_nodes=False,
    clean=False,
):
    figure = plt.figure(figsize=(10.5, 8.4), dpi=200)
    axis = figure.add_subplot(111, projection="3d")
    gt_positions = ground_truth[display_indices, :3, 3]
    axis.plot(
        gt_positions[:, 0],
        gt_positions[:, 1],
        gt_positions[:, 2],
        color=GT_COLOR,
        linewidth=3.0,
        label="Ground truth",
    )
    axis.scatter(*gt_positions[0], color=GT_COLOR, marker="o", s=55, depthshade=False)
    axis.scatter(*gt_positions[-1], color=GT_COLOR, marker="s", s=55, depthshade=False)
    if show_pose_nodes and len(gt_positions) > 2:
        axis.scatter(
            gt_positions[1:-1, 0],
            gt_positions[1:-1, 1],
            gt_positions[1:-1, 2],
            color=GT_COLOR,
            s=18,
            depthshade=False,
        )
    positions = [gt_positions]
    for index, method in enumerate(methods):
        poses = method["aligned"][display_indices]
        centers = poses[:, :3, 3]
        color = METHOD_COLORS[index % len(METHOD_COLORS)]
        axis.plot(
            centers[:, 0],
            centers[:, 1],
            centers[:, 2],
            color=color,
            linewidth=2.5,
            label=method["label"],
        )
        axis.scatter(*centers[0], color=color, marker="o", s=48, depthshade=False)
        axis.scatter(*centers[-1], color=color, marker="s", s=48, depthshade=False)
        if show_pose_nodes and len(centers) > 2:
            axis.scatter(
                centers[1:-1, 0],
                centers[1:-1, 1],
                centers[1:-1, 2],
                color=color,
                s=16,
                depthshade=False,
            )
        positions.append(centers)
        if not clean:
            arrow_stride = max(1, len(poses) // 5)
            all_positions = np.concatenate(positions, axis=0)
            arrow_length = max(float(np.linalg.norm(np.ptp(all_positions, axis=0))) * 0.035, 0.2)
            for frame_index in range(0, len(poses), arrow_stride):
                direction = poses[frame_index, :3, :3] @ np.array([0.0, 0.0, 1.0])
                axis.quiver(
                    centers[frame_index, 0],
                    centers[frame_index, 1],
                    centers[frame_index, 2],
                    direction[0],
                    direction[1],
                    direction[2],
                    color=color,
                    length=arrow_length,
                    normalize=True,
                    linewidth=0.8,
                    alpha=0.7,
                )
    all_positions = np.concatenate(positions, axis=0)
    span = np.maximum(np.ptp(all_positions, axis=0), 1.0)
    center = all_positions.mean(axis=0)
    half_extent = max(float(np.max(span)) * 0.55, 1.0)
    axis.set_xlim(center[0] - half_extent, center[0] + half_extent)
    axis.set_ylim(center[1] - half_extent, center[1] + half_extent)
    axis.set_zlim(center[2] - half_extent, center[2] + half_extent)
    axis.set_box_aspect((1.0, 1.0, 1.0))
    axis.set_xlabel("World X (m)")
    axis.set_ylabel("World Y (m)")
    axis.set_zlabel("World Z (m)")
    if not clean:
        axis.set_title("3D camera trajectory")
    axis.view_init(elev=24, azim=-62)
    axis.legend(frameon=False, loc="upper left")
    if not clean:
        figure.suptitle(title, fontsize=14, y=0.97)
    figure.tight_layout()
    figure.savefig(outdir / "trajectory_3d_vs_gt.png", bbox_inches="tight")
    figure.savefig(outdir / "trajectory_3d_vs_gt.pdf", bbox_inches="tight")
    plt.close(figure)


def main():
    args = parse_args()
    if len(args.inputs) != len(args.labels):
        raise SystemExit("--inputs and --labels must have the same length")
    if args.display_poses is not None and args.display_poses < 3:
        raise SystemExit("--display-poses must be at least three")
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    methods = []
    reference_frames = None
    ground_truth = None
    for path_text, label in zip(args.inputs, args.labels):
        path = Path(path_text)
        payload = json.loads(path.read_text())
        scene, frame_indices, predicted, gt = select_scene(payload, args.scene)
        if reference_frames is None:
            reference_frames = frame_indices
            ground_truth = gt
        elif not np.array_equal(reference_frames, frame_indices):
            raise RuntimeError(f"Frame protocol mismatch for {path}")
        elif not np.allclose(ground_truth, gt, atol=1.0e-6, rtol=1.0e-6):
            raise RuntimeError(f"GT trajectory mismatch for {path}")
        aligned, aligned_gt = align_to_gt(predicted, gt)
        position_error = np.linalg.norm(aligned[:, :3, 3] - aligned_gt[:, :3, 3], axis=1)
        methods.append(
            {
                "label": label,
                "input": str(path.resolve()),
                "scene": scene,
                "predicted": predicted,
                "aligned": aligned,
                "position_error": position_error,
                "ate": metric_value(scene, "ate_translation_m"),
                "rpe_t": metric_value(scene, "rpe_translation_m"),
                "rpe_r": metric_value(scene, "rpe_rotation_deg"),
            }
        )

    gt_positions = ground_truth[:, :3, 3]
    display_indices = select_display_indices(len(gt_positions), args.display_poses)
    plane_origin = gt_positions.mean(axis=0)
    plane = principal_plane(gt_positions)
    gt_plane = project(gt_positions[display_indices], plane, plane_origin)
    figure, (trajectory_axis, error_axis) = plt.subplots(1, 2, figsize=(14.0, 5.8), dpi=180)
    draw_path(trajectory_axis, gt_plane, GT_COLOR, "Ground truth", linewidth=3.0)
    for index, method in enumerate(methods):
        method_plane = project(
            method["aligned"][display_indices, :3, 3], plane, plane_origin
        )
        draw_path(
            trajectory_axis,
            method_plane,
            METHOD_COLORS[index % len(METHOD_COLORS)],
            method["label"],
        )
    trajectory_axis.set_aspect("equal", adjustable="box")
    trajectory_axis.set_xlabel("Principal trajectory axis 1 (m)")
    trajectory_axis.set_ylabel("Principal trajectory axis 2 (m)")
    trajectory_axis.set_title("GT-aligned camera trajectory")
    trajectory_axis.grid(True, color="#d1d5db", linewidth=0.7, alpha=0.8)
    trajectory_axis.legend(frameon=False, loc="best")

    for index, method in enumerate(methods):
        error_axis.plot(
            reference_frames,
            method["position_error"],
            color=METHOD_COLORS[index % len(METHOD_COLORS)],
            linewidth=2.2,
            label=(
                f"{method['label']}  ATE {fmt(method['ate'])} m | "
                f"RPE-t {fmt(method['rpe_t'])} m | RPE-R {fmt(method['rpe_r'])} deg"
            ),
        )
    error_axis.set_xlabel("Frame index")
    error_axis.set_ylabel("Global Sim(3)-aligned position error (m)")
    error_axis.set_title("Per-frame trajectory error")
    error_axis.grid(True, color="#d1d5db", linewidth=0.7, alpha=0.8)
    error_axis.legend(frameon=False, fontsize=8, loc="best")
    title = args.title or f"{args.scene}: continuous low-light trajectory"
    figure.suptitle(title, fontsize=14, y=0.99)
    figure.tight_layout(rect=(0.0, 0.0, 1.0, 0.95))
    figure.savefig(outdir / "trajectory_vs_gt.png", bbox_inches="tight")
    figure.savefig(outdir / "trajectory_vs_gt.pdf", bbox_inches="tight")
    plt.close(figure)
    render_3d_trajectory(
        outdir,
        title,
        ground_truth,
        methods,
        display_indices,
        show_pose_nodes=args.show_pose_nodes,
        clean=args.clean_3d,
    )

    arrays = {"frame_indices": reference_frames, "gt_camera_to_world": ground_truth}
    for index, method in enumerate(methods):
        arrays[f"pred_{index}_camera_to_world"] = method["predicted"]
        arrays[f"pred_{index}_sim3_aligned"] = method["aligned"]
        arrays[f"pred_{index}_position_error_m"] = method["position_error"]
    np.savez_compressed(outdir / "trajectories_aligned.npz", **arrays)

    with (outdir / "metrics.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=("method", "ate_m", "rpe_t_m", "rpe_r_deg", "max_position_error_m"),
        )
        writer.writeheader()
        for method in methods:
            writer.writerow(
                {
                    "method": method["label"],
                    "ate_m": method["ate"],
                    "rpe_t_m": method["rpe_t"],
                    "rpe_r_deg": method["rpe_r"],
                    "max_position_error_m": float(np.max(method["position_error"])),
                }
            )

    protocol = {
        "scene": args.scene,
        "frame_indices": reference_frames.tolist(),
        "display_frame_indices": reference_frames[display_indices].tolist(),
        "display_pose_count": int(len(display_indices)),
        "show_pose_nodes": bool(args.show_pose_nodes),
        "clean_3d": bool(args.clean_3d),
        "alignment": "one global Sim(3) per method over the full displayed trajectory",
        "trajectory_plane": "first two PCA directions of GT camera centers",
        "methods": [
            {
                "label": method["label"],
                "input": method["input"],
                "ate_m": method["ate"],
                "rpe_t_m": method["rpe_t"],
                "rpe_r_deg": method["rpe_r"],
            }
            for method in methods
        ],
    }
    (outdir / "protocol.json").write_text(json.dumps(protocol, indent=2) + "\n")
    print(json.dumps({"outdir": str(outdir.resolve()), **protocol}, indent=2))


if __name__ == "__main__":
    main()
