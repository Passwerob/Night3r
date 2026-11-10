#!/usr/bin/env python3
"""Render a raw far-baseline PLY from its original camera trajectory."""

import argparse
import json
import subprocess
from pathlib import Path

import numpy as np
import yaml
from PIL import Image
import matplotlib.pyplot as plt


BACKGROUND = np.asarray([11, 16, 23], dtype=np.uint8)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ply", type=Path, required=True)
    parser.add_argument("--trajectory", type=Path, required=True)
    parser.add_argument("--temporal-ply", type=Path, default=None)
    parser.add_argument(
        "--temporal-view-count", type=int, default=None,
        help="Number of source views encoded by --temporal-ply; defaults to trajectory length.",
    )
    parser.add_argument("--local-window", type=int, default=None)
    parser.add_argument(
        "--temporal-blend", action="store_true",
        help="Cross-fade consecutive local point-cloud renders instead of hard switching.",
    )
    parser.add_argument(
        "--max-trajectory-views", type=int, default=None,
        help="Optional prefix length for excluding a failed terminal view.",
    )
    parser.add_argument(
        "--fixed-view-index", type=int, default=None,
        help="Anchor the render at one source camera instead of following the trajectory.",
    )
    parser.add_argument(
        "--view-yaw-degrees", type=float, default=0.0,
        help="Small yaw sweep centered on --fixed-view-index in camera coordinates.",
    )
    parser.add_argument(
        "--view-pitch-degrees", type=float, default=0.0,
        help="Constant downward pitch applied at --fixed-view-index in camera coordinates.",
    )
    parser.add_argument(
        "--view-forward-offset", type=float, default=0.0,
        help="Camera translation along its forward axis for a fixed source view.",
    )
    parser.add_argument("--cameras-yaml", type=Path, required=True)
    parser.add_argument(
        "--intrinsics-json",
        type=Path,
        default=None,
        help="Optional per-view predicted intrinsics in the point-map tensor resolution.",
    )
    parser.add_argument("--scene", required=True)
    parser.add_argument("--views", type=int, required=True)
    parser.add_argument("--pairs", type=int, required=True)
    parser.add_argument("--path-length", type=float, required=True)
    parser.add_argument("--outdir", type=Path, required=True)
    parser.add_argument("--chase", action="store_true")
    parser.add_argument(
        "--adaptive-follow",
        action="store_true",
        help="Follow each predicted pose with an automatically scaled local camera.",
    )
    parser.add_argument(
        "--local-radius-factor",
        type=float,
        default=0.0,
        help="Keep only points within this multiple of the adaptive follow distance.",
    )
    parser.add_argument(
        "--near-depth-percentile",
        type=float,
        default=0.0,
        help="Keep the nearest fraction of points in the current camera frame.",
    )
    parser.add_argument("--frames", type=int, default=120)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--point-radius", type=int, default=1)
    parser.add_argument(
        "--focal-scale",
        type=float,
        default=1.0,
        help="Display-only focal-length multiplier for a tighter virtual view.",
    )
    parser.add_argument(
        "--color-exposure", type=float, default=1.0,
        help="Linear-domain exposure applied to source RGB colors before display.",
    )
    parser.add_argument("--render-scale", type=int, default=1)
    parser.add_argument(
        "--max-output-width", type=int, default=None,
        help="Optional proportional cap for large native camera resolutions.",
    )
    return parser.parse_args()


def load_ply(path):
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip() == "end_header":
                break
        rows = np.loadtxt(handle, dtype=np.float32)
    valid = np.isfinite(rows[:, :3]).all(axis=1)
    # The source PLY contains a display-only Y flip.
    return rows[valid, :3] * np.asarray([1.0, -1.0, 1.0], dtype=np.float32), np.clip(rows[valid, 3:6], 0, 255).astype(np.uint8)


def frame_ids_from_temporal_ply(path, view_count):
    _, colors = load_ply(path)
    palette = (plt.cm.turbo(np.arange(view_count, dtype=np.float32) / max(view_count - 1, 1))[:, :3] * 255.0 + 0.5).astype(np.uint8)
    squared = (colors[:, None, :].astype(np.int32) - palette[None, :, :].astype(np.int32)) ** 2
    return np.argmin(squared.sum(axis=2), axis=1)


def load_camera(path):
    if path.name == "transforms.json":
        with path.open("r", encoding="utf-8") as handle:
            transforms = json.load(handle)
        required = ("fl_x", "fl_y", "cx", "cy", "w", "h")
        missing = [key for key in required if key not in transforms]
        if missing:
            raise KeyError(f"Missing DL3DV camera fields: {missing}")
        fx, fy, cx, cy = [float(transforms[key]) for key in ("fl_x", "fl_y", "cx", "cy")]
        width, height = int(transforms["w"]), int(transforms["h"])
    else:
        with path.open("r", encoding="utf-8") as handle:
            cameras = yaml.safe_load(handle)
        camera = cameras.get("rgb_cam_crop", cameras.get("rgb_cam_full"))
        if camera is None:
            raise KeyError("Expected rgb_cam_crop or rgb_cam_full in cameras.yaml")
        fx, fy, cx, cy = [float(value) for value in camera["intrinsics"]]
        width, height = [int(value) for value in camera["resolution"]]
    return np.asarray([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=np.float32), width, height


def load_predicted_intrinsics(path, output_width, output_height):
    payload = json.loads(path.read_text(encoding="utf-8"))
    reference_height, reference_width = [int(value) for value in payload["image_size_hw"]]
    sequence = np.asarray(payload["intrinsics"], dtype=np.float32)
    if sequence.ndim != 3 or sequence.shape[1:] != (3, 3):
        raise ValueError("intrinsics JSON must contain [views, 3, 3] matrices")
    scale_x = output_width / reference_width
    scale_y = output_height / reference_height
    sequence = sequence.copy()
    sequence[:, 0, 0] *= scale_x
    sequence[:, 0, 2] *= scale_x
    sequence[:, 1, 1] *= scale_y
    sequence[:, 1, 2] *= scale_y
    return sequence


def interpolate_intrinsics(intrinsics, phase):
    if len(intrinsics) == 1:
        return intrinsics[0]
    position = phase * (len(intrinsics) - 1)
    first = min(int(position), len(intrinsics) - 2)
    blend = position - first
    return ((1.0 - blend) * intrinsics[first] + blend * intrinsics[first + 1]).astype(np.float32)


def interpolate_pose(poses, phase):
    position = phase * (len(poses) - 1)
    first = min(int(position), len(poses) - 2)
    blend = position - first
    left, right = poses[first], poses[first + 1]
    rotation = (1.0 - blend) * left[:3, :3] + blend * right[:3, :3]
    u, _, vt = np.linalg.svd(rotation)
    rotation = u @ vt
    if np.linalg.det(rotation) < 0:
        u[:, -1] *= -1.0
        rotation = u @ vt
    translation = (1.0 - blend) * left[:3, 3] + blend * right[:3, 3]
    return rotation.astype(np.float32), translation.astype(np.float32)


def apply_camera_yaw(rotation, degrees):
    angle = np.deg2rad(degrees)
    yaw = np.asarray(
        [
            [np.cos(angle), 0.0, -np.sin(angle)],
            [0.0, 1.0, 0.0],
            [np.sin(angle), 0.0, np.cos(angle)],
        ],
        dtype=np.float32,
    )
    return rotation @ yaw


def apply_camera_pitch(rotation, degrees):
    """Apply a positive downward pitch in the source camera frame."""
    angle = np.deg2rad(-degrees)
    pitch = np.asarray(
        [
            [1.0, 0.0, 0.0],
            [0.0, np.cos(angle), -np.sin(angle)],
            [0.0, np.sin(angle), np.cos(angle)],
        ],
        dtype=np.float32,
    )
    return rotation @ pitch


def rasterize(
    points, colors, rotation, translation, intrinsics, width, height,
    chase=False, point_radius=1, color_exposure=1.0,
    follow_distance=None, follow_center=None, follow_up_fraction=0.12,
):
    if chase or follow_distance is not None:
        forward_hint = rotation[:, 2]
        up_hint = -rotation[:, 1]
        if follow_center is not None:
            candidate = np.asarray(follow_center, dtype=np.float32) - translation
            candidate_norm = float(np.linalg.norm(candidate))
            if candidate_norm > 1.0e-5:
                forward_hint = candidate / candidate_norm
                up_hint = up_hint - forward_hint * float(np.dot(up_hint, forward_hint))
                up_norm = float(np.linalg.norm(up_hint))
                if up_norm > 1.0e-5:
                    up_hint /= up_norm
        distance = 3.5 if follow_distance is None else float(follow_distance)
        camera = translation - distance * forward_hint + follow_up_fraction * distance * up_hint
        target = (
            np.asarray(follow_center, dtype=np.float32)
            if follow_center is not None else translation + (0.65 * distance) * forward_hint
        )
        forward = target - camera
        forward /= max(float(np.linalg.norm(forward)), 1.0e-6)
        right = np.cross(forward, up_hint)
        right /= max(float(np.linalg.norm(right)), 1.0e-6)
        up = np.cross(right, forward)
        local = np.stack(
            ((points - camera) @ right, (points - camera) @ up, (points - camera) @ forward),
            axis=1,
        )
    else:
        # For row vectors, world-to-camera is (p - t) @ R_c2w.
        local = (points - translation) @ rotation
    z = local[:, 2]
    u = intrinsics[0, 0] * local[:, 0] / np.maximum(z, 1.0e-6) + intrinsics[0, 2]
    v = intrinsics[1, 1] * local[:, 1] / np.maximum(z, 1.0e-6) + intrinsics[1, 2]
    valid = (z > 0.25) & np.isfinite(u) & np.isfinite(v)
    valid &= (u >= -1) & (u < width + 1) & (v >= -1) & (v < height + 1)
    canvas = np.broadcast_to(BACKGROUND, (height, width, 3)).copy()
    if not np.any(valid):
        return canvas
    u = np.rint(u[valid]).astype(np.int32)
    v = np.rint(v[valid]).astype(np.int32)
    depth = z[valid]
    source = colors[valid].astype(np.float32) / 255.0
    linear = np.clip(source, 0.0, 1.0) ** 2.2
    display = np.clip(linear * color_exposure, 0.0, 1.0) ** (1.0 / 2.2)
    # Preserve the actual source RGB values.  Geometry rendering must not apply
    # an extra display gamma that would make colors method-dependent.
    rgb = np.clip(display * 255.0 + 0.5, 0, 255).astype(np.uint8)
    order = np.argsort(depth)[::-1]
    u, v, rgb = u[order], v[order], rgb[order]
    for dy in range(-point_radius, point_radius + 1):
        for dx in range(-point_radius, point_radius + 1):
            px, py = u + dx, v + dy
            inside = (px >= 0) & (px < width) & (py >= 0) & (py < height)
            canvas[py[inside], px[inside]] = rgb[inside]
    return canvas


def blend_local_renders(first, second, amount):
    """Time-continuous blend of adjacent local reconstructions."""
    if amount <= 0.0:
        return first
    if amount >= 1.0:
        return second
    first_weight = 1.0 - amount
    return np.clip(
        first.astype(np.float32) * first_weight + second.astype(np.float32) * amount + 0.5,
        0,
        255,
    ).astype(np.uint8)


def main():
    args = parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)
    frame_dir = args.outdir / "frames"
    frame_dir.mkdir(exist_ok=True)
    points, colors = load_ply(args.ply)
    intrinsics, width, height = load_camera(args.cameras_yaml)
    if args.render_scale < 1:
        raise ValueError("--render-scale must be at least one")
    intrinsics[:2] *= args.render_scale
    width *= args.render_scale
    height *= args.render_scale
    if args.max_output_width is not None and width > args.max_output_width:
        scale = args.max_output_width / width
        intrinsics[:2] *= scale
        width = int(round(width * scale))
        height = int(round(height * scale))
    if args.focal_scale <= 0.0:
        raise ValueError("--focal-scale must be positive")
    intrinsics[0, 0] *= args.focal_scale
    intrinsics[1, 1] *= args.focal_scale
    predicted_intrinsics = None
    if args.intrinsics_json is not None:
        predicted_intrinsics = load_predicted_intrinsics(
            args.intrinsics_json, width, height
        )
        predicted_intrinsics[:, 0, 0] *= args.focal_scale
        predicted_intrinsics[:, 1, 1] *= args.focal_scale
    trajectory = json.loads(args.trajectory.read_text(encoding="utf-8"))
    poses = np.asarray([record["camera_to_world"] for record in trajectory], dtype=np.float32)
    if args.max_trajectory_views is not None:
        poses = poses[:args.max_trajectory_views]
    if len(poses) < 2:
        raise ValueError("At least two trajectory views are required")
    adaptive_distance = None
    if args.adaptive_follow:
        # Monocular methods use different arbitrary units.  Scale the chase
        # distance from the robust diameter of this method's own cloud.
        extent = np.ptp(np.percentile(points, [2.0, 98.0], axis=0), axis=0)
        extent_norm = float(np.linalg.norm(extent))
        # Very long Night3R chains can span tens of scene units.  A fixed
        # fraction of that global diameter becomes an overview camera, so use
        # a smaller base and refine it from the local cloud around each pose.
        adaptive_distance = max(extent_norm * (0.08 if extent_norm > 10.0 else 0.42), 1.0e-3)
    if args.fixed_view_index is not None and not 0 <= args.fixed_view_index < len(poses):
        raise ValueError("--fixed-view-index is outside the retained trajectory")
    frame_ids = None
    if args.temporal_ply is not None:
        frame_ids = frame_ids_from_temporal_ply(
            args.temporal_ply,
            args.temporal_view_count or len(trajectory),
        )
        if len(frame_ids) != len(points):
            raise RuntimeError("Temporal and RGB PLY vertex counts differ")
    for index in range(args.frames):
        phase = index / max(args.frames - 1, 1)
        if args.fixed_view_index is None:
            rotation, translation = interpolate_pose(poses, phase)
            position = phase * (len(poses) - 1)
            current_view = int(round(position))
        else:
            rotation = poses[args.fixed_view_index, :3, :3].copy()
            translation = poses[args.fixed_view_index, :3, 3].copy()
            rotation = apply_camera_yaw(
                rotation, args.view_yaw_degrees * (phase - 0.5)
            )
            rotation = apply_camera_pitch(rotation, args.view_pitch_degrees)
            translation = translation + args.view_forward_offset * rotation[:, 2]
            position = float(args.fixed_view_index)
            current_view = args.fixed_view_index
        follow_distance = adaptive_distance
        follow_center = None
        if args.adaptive_follow and adaptive_distance is not None:
            radial = np.linalg.norm(points - translation, axis=1)
            local_scale = float(np.percentile(radial, 10.0))
            if extent_norm > 10.0:
                follow_distance = max(adaptive_distance, 1.2 * local_scale)
            neighborhood = points[radial <= np.percentile(radial, 30.0)]
            if len(neighborhood) >= 256:
                follow_center = np.median(neighborhood, axis=0).astype(np.float32)
        visible_points, visible_colors = points, colors
        blend_amount = 0.0
        next_points = next_colors = None
        if args.local_radius_factor > 0.0 and adaptive_distance is not None:
            local_center = translation + 0.25 * adaptive_distance * rotation[:, 2]
            local_radius = args.local_radius_factor * follow_distance
            keep_local = np.linalg.norm(points - local_center, axis=1) <= local_radius
            # Never replace a valid frame by an empty render when a trajectory
            # estimate drifts outside the merged cloud.
            if int(keep_local.sum()) >= 256:
                visible_points, visible_colors = points[keep_local], colors[keep_local]
        if frame_ids is not None:
            position = phase * (len(poses) - 1)
            if args.temporal_blend:
                current = min(int(np.floor(position)), len(poses) - 1)
                next_view = min(current + 1, len(poses) - 1)
                blend_amount = position - current
                keep = frame_ids == current
                next_keep = frame_ids == next_view
                visible_points, visible_colors = points[keep], colors[keep]
                next_points, next_colors = points[next_keep], colors[next_keep]
            else:
                current = current_view
                keep = np.abs(frame_ids - current) <= args.local_window
                visible_points, visible_colors = points[keep], colors[keep]
        if args.near_depth_percentile > 0.0:
            local_depth = (visible_points - translation) @ rotation
            valid_depth = np.isfinite(local_depth).all(axis=1) & (local_depth[:, 2] > 0.15)
            if int(valid_depth.sum()) >= 256:
                cutoff = np.percentile(
                    local_depth[valid_depth, 2],
                    min(max(float(args.near_depth_percentile), 1.0), 99.0),
                )
                keep_near = valid_depth & (local_depth[:, 2] <= cutoff)
                if int(keep_near.sum()) >= 256:
                    visible_points, visible_colors = visible_points[keep_near], visible_colors[keep_near]
        rendered = rasterize(
            visible_points, visible_colors, rotation, translation,
            interpolate_intrinsics(predicted_intrinsics, phase)
            if predicted_intrinsics is not None else intrinsics,
            width, height, chase=args.chase or args.adaptive_follow,
            point_radius=args.point_radius, color_exposure=args.color_exposure,
            follow_distance=follow_distance if args.adaptive_follow else None,
            follow_center=follow_center,
        )
        if next_points is not None:
            rendered = blend_local_renders(
                rendered,
                rasterize(
                    next_points, next_colors, rotation, translation,
                    intrinsics, width, height, chase=args.chase or args.adaptive_follow,
                    point_radius=args.point_radius, color_exposure=args.color_exposure,
                    follow_distance=follow_distance if args.adaptive_follow else None,
                    follow_center=follow_center,
                ),
                blend_amount,
            )
        image = Image.fromarray(rendered)
        image.save(frame_dir / f"frame_{index:04d}.png", compress_level=2)
    output = args.outdir / ("night3r_farbaseline_camera_chase.mp4" if args.chase else "night3r_farbaseline_camera_follow.mp4")
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-framerate", str(args.fps), "-i", str(frame_dir / "frame_%04d.png"), "-c:v", "libx264", "-preset", "slow", "-crf", "18", "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(output)], check=True)
    print(output)


if __name__ == "__main__":
    main()
