#!/usr/bin/env python3
"""Fuse a short Night3R sequence with the separately trained pose head."""

import argparse
import json
import os
import random
import sys
from pathlib import Path
from types import SimpleNamespace

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont, ImageOps

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from dataset.dsec_event_dataset import DSECEventDataset  # noqa: E402
from train_event_dark3r import (  # noqa: E402
    load_config,
    move_batch_to_device,
    resolve_amp_config,
)
from utils.eval_event_dark3r_variant import build_model, resolve_device  # noqa: E402
from utils.eval_night3r_trajectory import (  # noqa: E402
    apply_deterministic_lowlight,
    load_pose_head_checkpoint,
    project_to_so3,
    reciprocal_descriptor_matches,
)
from utils.export_dsec_darken_pointcloud_contrast import (  # noqa: E402
    render_cloud,
    write_ply,
)


BACKGROUND = (8, 13, 20)
PANEL_BACKGROUND = (11, 16, 23)


def parse_args():
    parser = argparse.ArgumentParser("Export a fused Night3R sequence")
    parser.add_argument(
        "--checkpoint",
        default="outputs/night3r_mvsec_only_adapt_1k_from_best/last_event_dark3r.pth",
    )
    parser.add_argument(
        "--pose-head-checkpoint",
        default=(
            "outputs/night3r_pose_separate_branches_w5_dense2k_20260720/"
            "pose_heads/pose_head_step00001200.pth"
        ),
    )
    parser.add_argument(
        "--geometry-head-checkpoint",
        default=None,
        help="Optional compact recovered MASt3R downstream-head checkpoint.",
    )
    parser.add_argument(
        "--config", default="config/train_night3r_pose_separate_branches_w5_dense2k.yaml"
    )
    parser.add_argument("--data-root", default="/data/fcr/data/train_data/DSEC")
    parser.add_argument("--scene", default="screen-000003-dsec")
    parser.add_argument("--start-frame", type=int, default=43)
    parser.add_argument("--num-frames", type=int, default=16)
    parser.add_argument(
        "--frame-indices",
        default=None,
        help=(
            "Optional comma-separated, strictly increasing frame indices. "
            "When supplied, these explicit views replace --start-frame/--num-frames."
        ),
    )
    parser.add_argument("--exposure", type=float, default=0.10)
    parser.add_argument("--read-noise-std", type=float, default=0.01)
    parser.add_argument("--shot-noise-std", type=float, default=0.01)
    parser.add_argument("--lowlight-seed", type=int, default=20260716)
    parser.add_argument(
        "--alignment",
        choices=("anchor_frame", "shared_frame_sim3", "pose_head", "gt_pose"),
        default="anchor_frame",
    )
    parser.add_argument(
        "--gt-pose-scale-mode",
        choices=("shared_frame", "none"),
        default="shared_frame",
        help=(
            "When --alignment=gt_pose, either propagate shared-frame depth scales "
            "between pair predictions or retain native point-map scale."
        ),
    )
    parser.add_argument("--conf-quantile", type=float, default=0.55)
    parser.add_argument("--min-depth", type=float, default=0.25)
    parser.add_argument("--max-depth", type=float, default=80.0)
    parser.add_argument("--max-points-per-frame", type=int, default=18000)
    parser.add_argument("--voxel-size", type=float, default=0.08)
    parser.add_argument(
        "--model-name-or-path",
        default="checkpoints/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric.pth",
    )
    parser.add_argument("--image-size", type=int, default=512)
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--amp-dtype", choices=("fp16", "bf16"), default="bf16")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--allow-unsafe-checkpoint-load", action="store_true")
    parser.add_argument(
        "--restore-frozen-rgb",
        action="store_true",
        help="Use the pristine MASt3R RGB path and load only Night3R adapter weights.",
    )
    parser.add_argument(
        "--restore-frozen-rgb-scope",
        choices=("none", "head2", "view2", "heads", "geometry", "all"),
        default="none",
        help="Restore a selected subset of pristine MASt3R RGB parameters.",
    )
    parser.add_argument(
        "--restore-frozen-rgb-alpha",
        type=float,
        default=1.0,
        help="Blend weight for restored pristine RGB parameters.",
    )
    parser.add_argument(
        "--outdir", default="outputs/night3r_long_dsec_pair43_exp010_16f_20260720"
    )
    args = parser.parse_args()
    if args.frame_indices:
        try:
            indices = [int(value.strip()) for value in args.frame_indices.split(",")]
        except ValueError as error:
            parser.error(f"--frame-indices must contain integers: {error}")
        if len(indices) < 2 or any(next_value <= value for value, next_value in zip(indices, indices[1:])):
            parser.error("--frame-indices must contain at least two strictly increasing values")
        args.frame_indices = indices
        args.start_frame = indices[0]
        args.num_frames = len(indices)
    elif args.num_frames < 2:
        parser.error("--num-frames must be at least 2")
    if not 0.0 < args.exposure <= 1.0:
        parser.error("--exposure must be in (0, 1]")
    return args


def selected_frame_indices(args):
    if args.frame_indices is not None:
        return list(args.frame_indices)
    return list(range(args.start_frame, args.start_frame + args.num_frames))


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_model_args(args, outdir):
    return SimpleNamespace(
        variant="event_dark3r",
        checkpoint=args.checkpoint,
        pose_head_checkpoint=args.pose_head_checkpoint,
        geometry_head_checkpoint=args.geometry_head_checkpoint,
        dark3r_lora_checkpoint=None,
        config=args.config,
        data_root=args.data_root,
        model_name_or_path=args.model_name_or_path,
        allow_unsafe_checkpoint_load=args.allow_unsafe_checkpoint_load,
        restore_frozen_rgb=args.restore_frozen_rgb,
        restore_frozen_rgb_scope=args.restore_frozen_rgb_scope,
        restore_frozen_rgb_alpha=args.restore_frozen_rgb_alpha,
        image_size=(args.image_size, args.image_size),
        max_scenes=None,
        max_frames_per_scene=args.num_frames,
        num_workers=0,
        amp=args.amp,
        amp_dtype=args.amp_dtype,
        device=args.device,
        seed=args.seed,
        lowlight_exposure=args.exposure,
        lowlight_read_noise_std=args.read_noise_std,
        lowlight_shot_noise_std=args.shot_noise_std,
        lowlight_gamma=2.2,
        lowlight_seed=args.lowlight_seed,
        output=str(outdir / "trajectory.json"),
    )


def make_train_args(args, outdir, pair_file):
    return SimpleNamespace(
        config=args.config,
        data_root=args.data_root,
        save_dir=str(outdir),
        model_name_or_path=args.model_name_or_path,
        dark3r_lora_checkpoint=None,
        allow_unsafe_checkpoint_load=args.allow_unsafe_checkpoint_load,
        use_tiny_model=False,
        image_size=(args.image_size, args.image_size),
        batch_size=1,
        max_batches=None,
        num_epochs=None,
        max_steps=None,
        max_scenes=None,
        max_pairs_per_scene=args.num_frames - 1,
        pair_stride=1,
        pair_sampling="precomputed",
        min_pair_distance=1,
        max_pair_distance=1,
        precomputed_pairs=str(pair_file),
        allow_reverse_pairs=False,
        symmetric_pairs=False,
        num_workers=0,
        stage=None,
        partial_finetune_blocks=None,
        amp=args.amp,
        amp_dtype=args.amp_dtype,
        gradient_checkpointing=False,
        device=args.device,
        log_every=1,
        no_save=True,
        seed=args.seed,
        depth_min=None,
        depth_max=None,
    )


def load_font(size, bold=False):
    filename = "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf"
    path = Path("/usr/share/fonts/truetype/dejavu") / filename
    return ImageFont.truetype(str(path), size=size) if path.exists() else ImageFont.load_default()


def select_batch_intrinsics(batch, view_index):
    intrinsics = batch["intrinsics"]
    if intrinsics.ndim == 4:
        return intrinsics[0, view_index]
    if intrinsics.ndim == 3:
        return intrinsics[0]
    if intrinsics.ndim == 2:
        return intrinsics
    raise ValueError(f"Unsupported intrinsics shape: {tuple(intrinsics.shape)}")


def pointmap_to_world(
    points,
    confidence,
    image,
    camera_to_world,
    frame_id,
    args,
    intrinsics=None,
):
    points = points.detach().float().cpu().numpy()
    confidence = confidence.detach().float().cpu().numpy()
    image = image.detach().float().cpu().permute(1, 2, 0).numpy()
    if confidence.ndim == 3 and confidence.shape[-1] == 1:
        confidence = confidence[..., 0]
    valid = np.isfinite(points).all(axis=-1) & np.isfinite(confidence)
    valid &= points[..., 2] > args.min_depth
    valid &= points[..., 2] < args.max_depth
    # Reject point-map outliers outside a generous camera frustum before the
    # points are transformed into the sequence coordinate frame.
    valid &= np.abs(points[..., 0]) <= 1.20 * points[..., 2]
    valid &= np.abs(points[..., 1]) <= 1.00 * points[..., 2]
    if np.any(valid):
        threshold = float(np.quantile(confidence[valid], args.conf_quantile))
        valid &= confidence >= threshold
    flat_indices = np.flatnonzero(valid.reshape(-1))
    if flat_indices.size > args.max_points_per_frame:
        generator = np.random.default_rng(args.seed + frame_id)
        flat_indices = generator.choice(
            flat_indices, size=args.max_points_per_frame, replace=False
        )
    selected = points.reshape(-1, 3)[flat_indices]
    if intrinsics is not None:
        if torch.is_tensor(intrinsics):
            intrinsics = intrinsics.detach().float().cpu().numpy()
        pixel_y, pixel_x = np.unravel_index(flat_indices, points.shape[:2])
        depth = selected[:, 2]
        selected = np.stack(
            [
                (pixel_x - intrinsics[0, 2]) / intrinsics[0, 0] * depth,
                (pixel_y - intrinsics[1, 2]) / intrinsics[1, 1] * depth,
                depth,
            ],
            axis=-1,
        )
    colors = np.clip(image.reshape(-1, 3)[flat_indices] * 255.0 + 0.5, 0, 255).astype(
        np.uint8
    )
    world = selected @ camera_to_world[:3, :3].T + camera_to_world[:3, 3]
    return world.astype(np.float32), colors, np.full(len(world), frame_id, dtype=np.int32)


def voxel_downsample(points, colors, frame_ids, voxel_size):
    if voxel_size <= 0 or len(points) == 0:
        return points, colors, frame_ids
    origin = np.min(points, axis=0)
    keys = np.floor((points - origin) / voxel_size).astype(np.int64)
    _, first = np.unique(keys, axis=0, return_index=True)
    first.sort()
    return points[first], colors[first], frame_ids[first]


def backproject_numpy_depth(points, intrinsics):
    if torch.is_tensor(intrinsics):
        intrinsics = intrinsics.detach().float().cpu().numpy()
    height, width = points.shape[:2]
    pixel_y, pixel_x = np.mgrid[:height, :width]
    depth = points[..., 2]
    return np.stack(
        [
            (pixel_x - intrinsics[0, 2]) / intrinsics[0, 0] * depth,
            (pixel_y - intrinsics[1, 2]) / intrinsics[1, 1] * depth,
            depth,
        ],
        axis=-1,
    )


def estimate_shared_frame_depth_scale(
    current_first_points,
    current_first_confidence,
    previous_second_points,
    previous_second_confidence,
    previous_relative_pose,
    previous_scale,
    intrinsics,
):
    current = current_first_points.detach().float().cpu().numpy()
    current_conf = current_first_confidence.detach().float().cpu().numpy().squeeze()
    previous = previous_second_points.detach().float().cpu().numpy()
    previous_conf = previous_second_confidence.detach().float().cpu().numpy().squeeze()
    previous_in_current = (
        (previous * previous_scale) @ previous_relative_pose[:3, :3].T
        + previous_relative_pose[:3, 3]
    )
    valid = np.isfinite(current).all(axis=-1) & np.isfinite(previous_in_current).all(
        axis=-1
    )
    valid &= np.isfinite(current_conf) & np.isfinite(previous_conf)
    valid &= current[..., 2] > 0.25
    valid &= previous_in_current[..., 2] > 0.25
    if np.any(valid):
        joint_confidence = np.minimum(current_conf, previous_conf)
        valid &= joint_confidence >= np.quantile(joint_confidence[valid], 0.50)
    current_values = current[valid]
    target_values = previous_in_current[valid]
    denominators = np.sum(current_values * current_values, axis=-1)
    ratios = np.sum(current_values * target_values, axis=-1) / np.maximum(
        denominators, 1.0e-12
    )
    ratios = ratios[
        np.isfinite(ratios) & (ratios > 1.0e-4) & (ratios < 1.0e3)
    ]
    if len(ratios) < 512:
        return float(previous_scale), int(len(ratios))
    scale = float(np.median(ratios))
    return scale, int(len(ratios))


def estimate_initial_shared_frame_scales(
    current_first_points,
    current_first_confidence,
    previous_second_points,
    previous_second_confidence,
    previous_relative_pose,
    intrinsics,
):
    current = current_first_points.detach().float().cpu().numpy()
    current_conf = current_first_confidence.detach().float().cpu().numpy().squeeze()
    previous = previous_second_points.detach().float().cpu().numpy()
    previous_conf = previous_second_confidence.detach().float().cpu().numpy().squeeze()
    rotated_previous = previous @ previous_relative_pose[:3, :3].T
    valid = np.isfinite(current).all(axis=-1) & np.isfinite(rotated_previous).all(axis=-1)
    valid &= np.isfinite(current_conf) & np.isfinite(previous_conf)
    valid &= (current[..., 2] > 0.25) & (previous[..., 2] > 0.25)
    if np.any(valid):
        joint_confidence = np.minimum(current_conf, previous_conf)
        valid &= joint_confidence >= np.quantile(joint_confidence[valid], 0.50)
    previous_values = rotated_previous[valid]
    current_values = current[valid]
    if len(previous_values) < 512:
        return 1.0, 1.0, int(len(previous_values))
    matrix = np.stack([previous_values, -current_values], axis=-1).reshape(-1, 2)
    target = np.broadcast_to(
        -previous_relative_pose[:3, 3], previous_values.shape
    ).reshape(-1)
    scales, _, _, _ = np.linalg.lstsq(matrix, target, rcond=None)
    residual = (
        scales[0] * previous_values
        - scales[1] * current_values
        + previous_relative_pose[:3, 3]
    )
    keep = np.linalg.norm(residual, axis=-1) <= np.quantile(
        np.linalg.norm(residual, axis=-1), 0.80
    )
    matrix = np.stack(
        [previous_values[keep], -current_values[keep]], axis=-1
    ).reshape(-1, 2)
    target = np.broadcast_to(
        -previous_relative_pose[:3, 3], previous_values[keep].shape
    ).reshape(-1)
    scales, _, _, _ = np.linalg.lstsq(matrix, target, rcond=None)
    print(
        json.dumps(
            {
                "initial_shared_frame_raw_scales": scales.tolist(),
                "support_pixels": int(keep.sum()),
            }
        ),
        flush=True,
    )
    if (
        not np.isfinite(scales).all()
        or np.any(scales <= 1.0e-4)
        or np.any(scales >= 1.0e3)
    ):
        return 1.0, 1.0, int(keep.sum())
    return float(scales[0]), float(scales[1]), int(keep.sum())


def umeyama_sim3(source, target):
    source_mean = source.mean(axis=0)
    target_mean = target.mean(axis=0)
    source_centered = source - source_mean
    target_centered = target - target_mean
    covariance = target_centered.T @ source_centered / max(len(source), 1)
    u, singular_values, vt = np.linalg.svd(covariance)
    correction = np.eye(3)
    if np.linalg.det(u @ vt) < 0:
        correction[-1, -1] = -1
    rotation = u @ correction @ vt
    variance = float(np.mean(np.sum(source_centered**2, axis=1)))
    if variance <= 1.0e-12:
        raise ValueError("Degenerate shared-frame point map")
    scale = float(np.sum(singular_values * np.diag(correction)) / variance)
    translation = target_mean - scale * (rotation @ source_mean)
    return scale, rotation, translation


def register_shared_frame(
    source,
    target,
    source_conf,
    target_conf,
    source_desc,
    target_desc,
    source_desc_conf,
    target_desc_conf,
    args,
    pair_index,
):
    source_xy, target_xy, match_scores = reciprocal_descriptor_matches(
        source_desc,
        target_desc,
        source_desc_conf,
        target_desc_conf,
        stride=4,
        max_matches=8192,
    )
    source = source.detach().float().cpu().numpy()
    target = target.detach().float().cpu().numpy()
    source_conf = source_conf.detach().float().cpu().numpy().squeeze()
    target_conf = target_conf.detach().float().cpu().numpy().squeeze()
    source_xy = np.rint(source_xy).astype(np.int64)
    target_xy = np.rint(target_xy).astype(np.int64)
    h_source, w_source = source.shape[:2]
    h_target, w_target = target.shape[:2]
    valid_matches = (
        (source_xy[:, 0] >= 0)
        & (source_xy[:, 0] < w_source)
        & (source_xy[:, 1] >= 0)
        & (source_xy[:, 1] < h_source)
        & (target_xy[:, 0] >= 0)
        & (target_xy[:, 0] < w_target)
        & (target_xy[:, 1] >= 0)
        & (target_xy[:, 1] < h_target)
        & np.isfinite(match_scores)
    )
    source_xy = source_xy[valid_matches]
    target_xy = target_xy[valid_matches]
    match_scores = match_scores[valid_matches]
    source_points = source[source_xy[:, 1], source_xy[:, 0]].astype(np.float64)
    target_points = target[target_xy[:, 1], target_xy[:, 0]].astype(np.float64)
    confidence = np.sqrt(
        np.clip(
            source_conf[source_xy[:, 1], source_xy[:, 0]]
            * target_conf[target_xy[:, 1], target_xy[:, 0]],
            0.0,
            None,
        )
    )
    finite = np.isfinite(source_points).all(axis=1) & np.isfinite(target_points).all(axis=1)
    finite &= np.isfinite(confidence)
    finite &= (source_points[:, 2] > args.min_depth) & (source_points[:, 2] < args.max_depth)
    finite &= (target_points[:, 2] > args.min_depth) & (target_points[:, 2] < args.max_depth)
    source_points = source_points[finite]
    target_points = target_points[finite]
    confidence = confidence[finite]
    match_scores = match_scores[finite]
    source_xy = source_xy[finite]
    target_xy = target_xy[finite]
    if len(source_points) < 256:
        raise RuntimeError(f"Insufficient descriptor matches at pair {pair_index}")
    threshold = float(np.quantile(confidence, min(args.conf_quantile, 0.40)))
    keep_conf = confidence >= threshold
    source_points = source_points[keep_conf]
    target_points = target_points[keep_conf]
    match_scores = match_scores[keep_conf]
    source_xy = source_xy[keep_conf]
    target_xy = target_xy[keep_conf]
    scale, rotation, translation = umeyama_sim3(source_points, target_points)
    prediction = (scale * (rotation @ source_points.T)).T + translation
    residual = np.linalg.norm(prediction - target_points, axis=1)
    keep = residual <= np.quantile(residual, 0.80)
    if int(keep.sum()) < 128:
        raise RuntimeError(f"Shared-frame registration failed at pair {pair_index}")
    scale, rotation, translation = umeyama_sim3(
        source_points[keep], target_points[keep]
    )
    prediction = (scale * (rotation @ source_points[keep].T)).T + translation
    residual = np.linalg.norm(prediction - target_points[keep], axis=1)
    target_extent = float(
        np.median(
            np.linalg.norm(
                target_points[keep] - np.median(target_points[keep], axis=0), axis=1
            )
        )
    )
    normalized_rmse = float(np.sqrt(np.mean(residual**2)) / max(target_extent, 1.0e-9))
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = scale * rotation
    transform[:3, 3] = translation
    return transform, {
        "pair_index": pair_index,
        "scale": scale,
        "normalized_rmse": normalized_rmse,
        "selected_points": int(len(source_points)),
        "inliers": int(keep.sum()),
        "median_match_displacement_px": float(
            np.median(np.linalg.norm(source_xy - target_xy, axis=1))
        ),
        "mean_match_score": float(np.mean(match_scores)),
    }


def write_trajectory(path, poses):
    payload = [
        {"frame_offset": index, "camera_to_world": pose.tolist()}
        for index, pose in enumerate(poses)
    ]
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def render_topdown(points, camera_centers, width=560, height=350):
    canvas = Image.new("RGB", (width, height), PANEL_BACKGROUND)
    draw = ImageDraw.Draw(canvas)
    sample = points[:: max(1, len(points) // 25000), :][:, [0, 2]]
    trajectory = camera_centers[:, [0, 2]]
    combined = np.concatenate([sample, trajectory], axis=0)
    lo, hi = np.percentile(combined, [1, 99], axis=0)
    extent = np.maximum(hi - lo, 1.0e-6)
    scale = min((width - 50) / extent[0], (height - 50) / extent[1])

    def project(values):
        xy = (values - (lo + hi) * 0.5) * scale
        xy[:, 0] += width * 0.5
        xy[:, 1] = height * 0.5 - xy[:, 1]
        return xy

    point_xy = project(sample.copy())
    for x, y in point_xy:
        if 0 <= x < width and 0 <= y < height:
            draw.point((int(x), int(y)), fill=(70, 91, 112))
    trajectory_xy = project(trajectory.copy())
    if len(trajectory_xy) > 1:
        draw.line([tuple(value) for value in trajectory_xy], fill=(74, 220, 175), width=4)
    for index, (x, y) in enumerate(trajectory_xy):
        radius = 5 if index in (0, len(trajectory_xy) - 1) else 3
        draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill=(244, 247, 250))
    return canvas


def render_front_cloud(
    points, point_colors=None, width=1800, height=820, point_radius=1
):
    background = np.asarray(PANEL_BACKGROUND, dtype=np.uint8)
    canvas = np.broadcast_to(background, (height, width, 3)).copy()
    z = points[:, 2]
    valid = np.isfinite(points).all(axis=1) & (z > 1.0e-4)
    points = points[valid]
    if point_colors is not None:
        point_colors = point_colors[valid]
    z = points[:, 2]
    rx = points[:, 0] / z
    ry = points[:, 1] / z
    x_lo, x_hi = np.percentile(rx, [0.5, 99.5])
    y_lo, y_hi = np.percentile(ry, [0.5, 99.5])
    x_center = 0.5 * (x_lo + x_hi)
    y_center = 0.5 * (y_lo + y_hi)
    focal = min(
        0.92 * width / max(x_hi - x_lo, 1.0e-6),
        0.92 * height / max(y_hi - y_lo, 1.0e-6),
    )
    u = np.rint((rx - x_center) * focal + width * 0.5).astype(np.int32)
    v = np.rint(height * 0.5 - (ry - y_center) * focal).astype(np.int32)
    inside = (u >= 0) & (u < width) & (v >= 0) & (v < height)
    u, v, z = u[inside], v[inside], z[inside]
    if point_colors is None:
        depth_lo, depth_hi = np.percentile(z, [2, 98])
        depth_norm = np.clip(
            (z - depth_lo) / max(depth_hi - depth_lo, 1.0e-6), 0, 1
        )
        render_colors = (
            plt.cm.turbo(depth_norm)[:, :3] * 255.0 + 0.5
        ).astype(np.uint8)
    else:
        render_colors = point_colors[inside]
    order = np.argsort(z)[::-1]
    u, v, render_colors = u[order], v[order], render_colors[order]
    for dy in range(-point_radius, point_radius + 1):
        for dx in range(-point_radius, point_radius + 1):
            px, py = u + dx, v + dy
            keep = (px >= 0) & (px < width) & (py >= 0) & (py < height)
            canvas[py[keep], px[keep]] = render_colors[keep]
    return canvas


def render_teaser(path, cloud_image, topdown, preview_frames, args, point_count):
    width, height = 2200, 1160
    canvas = Image.new("RGB", (width, height), BACKGROUND)
    draw = ImageDraw.Draw(canvas)
    title_font = load_font(38, bold=True)
    subtitle_font = load_font(24)
    small_font = load_font(21)
    draw.text((50, 38), "Night3R long-range reconstruction", font=title_font, fill=(245, 248, 250))
    alignment_label = {
        "anchor_frame": "one-reference fusion",
        "shared_frame_sim3": "shared-frame Sim(3)",
        "pose_head": "pose-head chaining",
        "gt_pose": "GT-pose geometry fusion",
    }[args.alignment]
    graph_label = "anchor pairs" if args.alignment == "anchor_frame" else "adjacent pairs"
    draw.text(
        (50, 92),
        (
            f"{args.num_frames} frames | {args.num_frames - 1} {graph_label} | "
            f"{args.exposure * 100:.0f}% exposure | {alignment_label}"
        ),
        font=subtitle_font,
        fill=(103, 221, 184),
    )

    thumbnail_y = 140
    thumb_width, thumb_height, gap = 390, 220, 22
    for index, image in enumerate(preview_frames):
        fitted = ImageOps.fit(image, (thumb_width, thumb_height), method=Image.Resampling.LANCZOS)
        x = 50 + index * (thumb_width + gap)
        canvas.paste(fitted, (x, thumbnail_y))
        draw.rectangle((x, thumbnail_y, x + thumb_width, thumbnail_y + thumb_height), outline=(92, 106, 120), width=2)
    draw.text((50, 372), "Input sequence", font=small_font, fill=(168, 179, 190))

    cloud = Image.fromarray(cloud_image)
    cloud = ImageOps.fit(cloud, (1540, 700), method=Image.Resampling.LANCZOS)
    canvas.paste(cloud, (30, 420))
    draw.rectangle((30, 420, 1570, 1120), outline=(69, 206, 164), width=3)
    draw.text((70, 445), f"Fused geometry | {point_count:,} voxels", font=subtitle_font, fill=(245, 248, 250))

    canvas.paste(topdown, (1600, 560))
    draw.rectangle((1600, 560, 2160, 910), outline=(92, 106, 120), width=2)
    trajectory_label = (
        "Ground-truth camera trajectory"
        if args.alignment == "gt_pose"
        else "Estimated camera trajectory"
    )
    draw.text((1600, 525), trajectory_label, font=small_font, fill=(245, 248, 250))
    draw.text((1600, 940), "Top view", font=small_font, fill=(168, 179, 190))
    canvas.save(path)


def main():
    args = parse_args()
    frame_indices = selected_frame_indices(args)
    outdir = (REPO_ROOT / args.outdir).resolve()
    outdir.mkdir(parents=True, exist_ok=True)
    pair_file = outdir / "sequence_pairs.json"
    if args.alignment == "anchor_frame":
        pairs = [
            {"idx1": frame_indices[0], "idx2": frame}
            for frame in frame_indices[1:]
        ]
    else:
        pairs = [
            {"idx1": first, "idx2": second}
            for first, second in zip(frame_indices, frame_indices[1:])
        ]
    pair_file.write_text(json.dumps(pairs, indent=2), encoding="utf-8")

    set_seed(args.seed)
    cfg = dict(load_config(args.config))
    cfg["lowlight_aug"] = {**dict(cfg.get("lowlight_aug", {})), "enabled": False}
    model_args = make_model_args(args, outdir)
    train_args = make_train_args(args, outdir, pair_file.resolve())
    dataset = DSECEventDataset(
        root_path=args.data_root,
        image_size=(args.image_size, args.image_size),
        pair_sampling="precomputed",
        precomputed_pairs=str(pair_file.resolve()),
        pose_output_convention="opencv",
        lowlight_enabled=False,
    )
    if args.alignment == "anchor_frame":
        scene_dir = next(
            (path for path in dataset.scene_frames if path.name == args.scene), None
        )
        if scene_dir is None:
            raise RuntimeError(f"Scene {args.scene!r} was not found")
        dataset.samples = [
            dataset._make_sample(scene_dir, frame_indices[0], frame)
            for frame in frame_indices[1:]
        ]
    else:
        expected_pairs = {(pair["idx1"], pair["idx2"]) for pair in pairs}
        dataset.samples = sorted(
            [
                sample
                for sample in dataset.samples
                if sample["scene"] == args.scene
                and (sample["idx1"], sample["idx2"]) in expected_pairs
            ],
            key=lambda sample: (sample["idx1"], sample["idx2"]),
        )
    if len(dataset.samples) != args.num_frames - 1:
        raise RuntimeError(
            f"Expected {args.num_frames - 1} sequence pairs, found {len(dataset.samples)}"
        )
    loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)

    device = resolve_device(args.device)
    model, load_info = build_model(model_args, cfg, train_args, device)
    non_pose_missing = [
        key for key in load_info.get("missing_keys", []) if not key.startswith("pose_head.")
    ]
    if non_pose_missing or load_info.get("unexpected_keys"):
        raise RuntimeError(
            f"Incomplete geometry checkpoint: missing={non_pose_missing}, "
            f"unexpected={load_info.get('unexpected_keys')}"
        )
    pose_load_info = None
    if args.alignment != "gt_pose":
        pose_load_info = load_pose_head_checkpoint(
            model,
            args.pose_head_checkpoint,
            allow_unsafe=args.allow_unsafe_checkpoint_load,
        )
    amp_enabled, amp_dtype_name, amp_dtype = resolve_amp_config(cfg, train_args, device)

    pair_to_world = np.eye(4, dtype=np.float64)
    poses = []
    all_points, all_colors, all_frame_ids = [], [], []
    preview_frames = []
    relative_poses = []
    alignment_diagnostics = []
    previous_shared_points = None
    previous_shared_confidence = None
    previous_shared_descriptor = None
    previous_shared_descriptor_confidence = None
    previous_pair_second_points = None
    previous_pair_second_confidence = None
    previous_pair_relative_pose = None
    previous_pair_scale = 1.0
    last_pair_scale = 1.0
    last_prediction = None
    last_batch_cpu = None
    last_pair_to_world = None
    with torch.no_grad():
        for pair_index, batch_cpu in enumerate(loader):
            batch_cpu, _ = apply_deterministic_lowlight(batch_cpu, model_args)
            if pair_index in np.linspace(0, len(dataset) - 1, 5, dtype=int):
                preview_view = 1 if args.alignment == "anchor_frame" else 0
                array = (
                    batch_cpu["images_photo_dark"][0, preview_view]
                    .permute(1, 2, 0)
                    .clamp(0, 1)
                    .mul(255)
                    .byte()
                    .numpy()
                )
                preview_frames.append(Image.fromarray(array))
            batch = move_batch_to_device(batch_cpu, device)
            with torch.amp.autocast(
                device_type=device.type,
                dtype=amp_dtype,
                enabled=bool(amp_enabled and device.type == "cuda"),
            ):
                pred1, pred2 = model(batch)
            if args.alignment == "anchor_frame":
                relative = (
                    pred1["relative_pose"][0]
                    .detach()
                    .float()
                    .cpu()
                    .numpy()
                    .astype(np.float64)
                )
                relative[:3, :3] = project_to_so3(relative[:3, :3])
                if not np.isfinite(relative).all():
                    raise RuntimeError(f"Non-finite pose at pair {pair_index}")
                relative_poses.append(relative)
                if pair_index == 0:
                    poses.append(np.eye(4, dtype=np.float64))
                    points, colors, frame_ids = pointmap_to_world(
                        pred1["pts3d"][0],
                        pred1["conf"][0],
                        batch_cpu["images_photo"][0, 0],
                        np.eye(4, dtype=np.float64),
                        0,
                        args,
                    )
                    all_points.append(points)
                    all_colors.append(colors)
                    all_frame_ids.append(frame_ids)
                points, colors, frame_ids = pointmap_to_world(
                    pred2["pts3d_in_other_view"][0],
                    pred2["conf"][0],
                    batch_cpu["images_photo"][0, 1],
                    np.eye(4, dtype=np.float64),
                    pair_index + 1,
                    args,
                )
                all_points.append(points)
                all_colors.append(colors)
                all_frame_ids.append(frame_ids)
                poses.append(np.linalg.inv(relative))
                continue
            if args.alignment == "shared_frame_sim3" and pair_index > 0:
                pair_transform, diagnostics = register_shared_frame(
                    pred1["pts3d"][0],
                    previous_shared_points,
                    pred1["conf"][0],
                    previous_shared_confidence,
                    pred1["desc"],
                    previous_shared_descriptor,
                    pred1.get("desc_conf"),
                    previous_shared_descriptor_confidence,
                    args,
                    pair_index,
                )
                pair_to_world = pair_to_world @ pair_transform
                alignment_diagnostics.append(diagnostics)
            pair_scale = 1.0
            if (
                args.alignment == "gt_pose"
                and args.gt_pose_scale_mode == "shared_frame"
                and pair_index > 0
            ):
                if pair_index == 1:
                    initial_scale, pair_scale, scale_support = (
                        estimate_initial_shared_frame_scales(
                            pred1["pts3d"][0],
                            pred1["conf"][0],
                            previous_pair_second_points,
                            previous_pair_second_confidence,
                            previous_pair_relative_pose,
                            select_batch_intrinsics(batch_cpu, 0),
                        )
                    )
                    all_points[-1] *= initial_scale
                    previous_pair_scale = initial_scale
                else:
                    pair_scale, scale_support = estimate_shared_frame_depth_scale(
                        pred1["pts3d"][0],
                        pred1["conf"][0],
                        previous_pair_second_points,
                        previous_pair_second_confidence,
                        previous_pair_relative_pose,
                        previous_pair_scale,
                        select_batch_intrinsics(batch_cpu, 0),
                    )
                alignment_diagnostics.append(
                    {
                        "pair_index": pair_index,
                        "previous_pair_depth_scale": previous_pair_scale,
                        "shared_frame_depth_scale": pair_scale,
                        "scale_support_pixels": scale_support,
                    }
                )
            poses.append(pair_to_world.copy())
            first_points = pred1["pts3d"][0]
            if args.alignment == "gt_pose" and args.gt_pose_scale_mode == "shared_frame":
                first_points = first_points * pair_scale
            points, colors, frame_ids = pointmap_to_world(
                first_points,
                pred1["conf"][0],
                batch_cpu["images_photo"][0, 0],
                pair_to_world,
                pair_index,
                args,
                intrinsics=(
                    select_batch_intrinsics(batch_cpu, 0)
                    if args.alignment == "gt_pose"
                    else None
                ),
            )
            all_points.append(points)
            all_colors.append(colors)
            all_frame_ids.append(frame_ids)

            if args.alignment == "gt_pose":
                if "relative_pose_gt" not in batch_cpu:
                    raise RuntimeError(
                        f"Ground-truth relative pose is unavailable at pair {pair_index}"
                    )
                relative = batch_cpu["relative_pose_gt"][0].numpy().astype(np.float64)
            else:
                relative = pred1["relative_pose"][0].detach().float().cpu().numpy().astype(np.float64)
            relative[:3, :3] = project_to_so3(relative[:3, :3])
            if not np.isfinite(relative).all():
                raise RuntimeError(f"Non-finite pose at pair {pair_index}")
            relative_poses.append(relative)
            last_pair_to_world = pair_to_world.copy()
            if args.alignment in {"pose_head", "gt_pose"}:
                pair_to_world = pair_to_world @ np.linalg.inv(relative)
            if args.alignment == "gt_pose" and args.gt_pose_scale_mode == "shared_frame":
                previous_pair_second_points = pred2["pts3d_in_other_view"][0].detach()
                previous_pair_second_confidence = pred2["conf"][0].detach()
                previous_pair_relative_pose = relative.copy()
                previous_pair_scale = pair_scale
                last_pair_scale = pair_scale
            previous_shared_points = pred2["pts3d_in_other_view"][0].detach().cpu()
            previous_shared_confidence = pred2["conf"][0].detach().cpu()
            previous_shared_descriptor = pred2["desc"].detach()
            previous_shared_descriptor_confidence = pred2.get("desc_conf")
            if previous_shared_descriptor_confidence is not None:
                previous_shared_descriptor_confidence = previous_shared_descriptor_confidence.detach()
            last_prediction = pred2
            last_batch_cpu = batch_cpu

    if args.alignment != "anchor_frame":
        final_camera_to_world = last_pair_to_world @ np.linalg.inv(relative_poses[-1])
        poses.append(final_camera_to_world)

    if args.alignment != "anchor_frame" and last_prediction is not None:
        final_points = last_prediction["pts3d_in_other_view"][0]
        final_point_to_world = last_pair_to_world
        final_intrinsics = None
        if args.alignment == "gt_pose" and args.gt_pose_scale_mode == "shared_frame":
            relative = torch.as_tensor(
                relative_poses[-1],
                dtype=final_points.dtype,
                device=final_points.device,
            )
            final_points = (
                (final_points * last_pair_scale)
                @ relative[:3, :3].transpose(0, 1)
                + relative[:3, 3]
            )
            final_point_to_world = final_camera_to_world
            final_intrinsics = select_batch_intrinsics(last_batch_cpu, 1)
        points, colors, frame_ids = pointmap_to_world(
            final_points,
            last_prediction["conf"][0],
            last_batch_cpu["images_photo"][0, 1],
            final_point_to_world,
            args.num_frames - 1,
            args,
            intrinsics=final_intrinsics,
        )
        all_points.append(points)
        all_colors.append(colors)
        all_frame_ids.append(frame_ids)

    points = np.concatenate(all_points, axis=0)
    colors = np.concatenate(all_colors, axis=0)
    frame_ids = np.concatenate(all_frame_ids, axis=0)
    points, colors, frame_ids = voxel_downsample(
        points, colors, frame_ids, args.voxel_size
    )

    display_flip = np.diag([1.0, -1.0, 1.0]).astype(np.float32)
    display_points = points @ display_flip.T
    camera_centers = np.asarray([pose[:3, 3] for pose in poses], dtype=np.float32)
    display_centers = camera_centers @ display_flip.T
    frame_norm = frame_ids.astype(np.float32) / max(args.num_frames - 1, 1)
    frame_colors = (plt.cm.turbo(frame_norm)[:, :3] * 255.0 + 0.5).astype(np.uint8)

    write_ply(outdir / "night3r_long_rgb.ply", display_points, colors)
    write_ply(outdir / "night3r_long_temporal.ply", display_points, frame_colors)
    write_trajectory(outdir / "trajectory.json", poses)
    first_camera_image = render_front_cloud(
        display_points, colors, width=1800, height=820, point_radius=1
    )
    Image.fromarray(first_camera_image).save(
        outdir / "night3r_long_first_camera_rgb.png"
    )
    cloud_image = render_cloud(
        display_points,
        rgb_colors=colors,
        color_mode="rgb",
        width=1800,
        height=820,
        point_radius=1,
    )
    Image.fromarray(cloud_image).save(outdir / "night3r_long_rgb_render.png")
    depth_image = render_cloud(
        display_points,
        rgb_colors=colors,
        color_mode="depth",
        width=1800,
        height=820,
        point_radius=1,
    )
    Image.fromarray(depth_image).save(outdir / "night3r_long_depth_render.png")
    topdown = render_topdown(display_points, display_centers)
    topdown.save(outdir / "trajectory_topdown.png")
    teaser_path = outdir / "night3r_long_reconstruction_teaser.png"
    render_teaser(
        teaser_path,
        cloud_image,
        topdown,
        preview_frames,
        args,
        len(display_points),
    )

    data_root_lower = str(args.data_root).lower()
    if "dl3dv" in data_root_lower:
        dataset_name = "DL3DV"
    elif "mvsec" in data_root_lower:
        dataset_name = "MVSEC"
    else:
        dataset_name = "DSEC"
    manifest = {
        "sample": {
            "dataset": dataset_name,
            "scene": args.scene,
            "start_frame": args.start_frame,
            "frame_indices": frame_indices,
            "num_frames": args.num_frames,
            "num_pairs": args.num_frames - 1,
            "exposure": args.exposure,
        },
        "model": {
            "geometry_checkpoint": str((REPO_ROOT / args.checkpoint).resolve()),
            "pose_head_checkpoint": (
                None
                if args.alignment == "gt_pose"
                else str((REPO_ROOT / args.pose_head_checkpoint).resolve())
            ),
            "config": str((REPO_ROOT / args.config).resolve()),
            "geometry_load": load_info,
            "pose_head_load": pose_load_info,
        },
        "fusion": {
            "confidence_quantile": args.conf_quantile,
            "depth_range": [args.min_depth, args.max_depth],
            "voxel_size": args.voxel_size,
            "points_before_voxelization": int(sum(len(value) for value in all_points)),
            "points_after_voxelization": int(len(display_points)),
            "alignment": args.alignment,
            "gt_pose_scale_mode": (
                args.gt_pose_scale_mode if args.alignment == "gt_pose" else None
            ),
            "alignment_diagnostics": alignment_diagnostics,
            "pose_composition": {
                "anchor_frame": (
                    "all pred2 point maps are already expressed in the common anchor frame"
                ),
                "shared_frame_sim3": (
                    "shared frame j registers pair (j,j+1) into pair (j-1,j) with Sim(3)"
                ),
                "pose_head": "T_world_cam_j = T_world_cam_i @ inv(T_cam_j_from_cam_i)",
                "gt_pose": (
                    "Night3R point maps are fused with dataset ground-truth relative poses: "
                    "T_world_cam_j = T_world_cam_i @ inv(T_cam_j_from_cam_i)"
                ),
            }[args.alignment],
        },
        "trajectory": {
            "num_poses": len(poses),
            "path_length": float(
                np.linalg.norm(np.diff(camera_centers, axis=0), axis=1).sum()
            ),
            "relative_poses": [pose.tolist() for pose in relative_poses],
            "ground_truth_available": args.alignment == "gt_pose",
        },
        "runtime": {
            "device": str(device),
            "amp_enabled": bool(amp_enabled),
            "amp_dtype": amp_dtype_name,
        },
        "files": {
            "teaser": teaser_path.name,
            "rgb_render": "night3r_long_rgb_render.png",
            "first_camera_rgb": "night3r_long_first_camera_rgb.png",
            "depth_render": "night3r_long_depth_render.png",
            "trajectory_topdown": "trajectory_topdown.png",
            "rgb_pointcloud": "night3r_long_rgb.ply",
            "temporal_pointcloud": "night3r_long_temporal.ply",
            "trajectory": "trajectory.json",
        },
        "claim_scope": (
            "Geometry-only long-sequence fusion using dataset ground-truth camera poses; "
            "this result isolates Night3R point-map quality and does not evaluate "
            "end-to-end pose estimation."
            if args.alignment == "gt_pose"
            else "Qualitative end-to-end long-sequence reconstruction only. This export "
            "does not compute ATE/RPE, even when the source dataset provides pose GT."
        ),
    }
    (outdir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps({"outdir": str(outdir), **manifest["sample"], **manifest["trajectory"]}, indent=2))


if __name__ == "__main__":
    main()
