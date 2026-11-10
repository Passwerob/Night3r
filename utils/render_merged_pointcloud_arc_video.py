#!/usr/bin/env python3
"""Render a fused RGB point cloud from a smooth third-person arc camera."""

import argparse
import math
import subprocess
from pathlib import Path

import numpy as np
from PIL import Image


BACKGROUND = np.asarray([11, 16, 23], dtype=np.uint8)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ply", type=Path, required=True)
    parser.add_argument("--outdir", type=Path, required=True)
    parser.add_argument("--frames", type=int, default=480)
    parser.add_argument("--fps", type=int, default=60)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--arc-degrees", type=float, default=92.0)
    parser.add_argument("--camera-distance-scale", type=float, default=1.25)
    parser.add_argument("--camera-distance", type=float, default=None)
    parser.add_argument("--target", type=float, nargs=3, default=None)
    parser.add_argument(
        "--focus-radius", type=float, default=None,
        help="Keep only fused points within this 3D radius of --target.",
    )
    parser.add_argument("--camera-height-offset", type=float, default=None)
    parser.add_argument(
        "--overview-arc", action="store_true",
        help="Use the stable overview camera and make only a small yaw arc around it.",
    )
    parser.add_argument("--focal-scale", type=float, default=0.95)
    parser.add_argument("--point-radius", type=int, default=1)
    parser.add_argument("--max-points", type=int, default=180000)
    parser.add_argument("--seed", type=int, default=17)
    return parser.parse_args()


def load_ply(path):
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip() == "end_header":
                break
        rows = np.loadtxt(handle, dtype=np.float32)
    valid = np.isfinite(rows[:, :3]).all(axis=1)
    return rows[valid, :3], np.clip(rows[valid, 3:6], 0, 255).astype(np.uint8)


def robust_cloud(points, colors, max_points, seed):
    low, high = np.percentile(points, [1.0, 99.0], axis=0)
    keep = np.all((points >= low) & (points <= high), axis=1)
    points, colors = points[keep], colors[keep]
    if len(points) > max_points:
        rng = np.random.default_rng(seed)
        keep = rng.choice(len(points), size=max_points, replace=False)
        points, colors = points[keep], colors[keep]
    return points, colors


def camera_path(points, phase, args):
    low, high = np.percentile(points, [2.0, 98.0], axis=0)
    if args.overview_arc:
        center = 0.5 * (low + high)
        extent = max(float(np.linalg.norm(high - low)), 1.0e-6)
        angle = math.radians(args.arc_degrees) * (phase - 0.5)
        base_offset = np.asarray([0.40 * extent, 0.24 * extent, -1.18 * extent], dtype=np.float32)
        yaw = np.asarray(
            [[math.cos(angle), 0.0, math.sin(angle)], [0.0, 1.0, 0.0], [-math.sin(angle), 0.0, math.cos(angle)]],
            dtype=np.float32,
        )
        camera = center + yaw @ base_offset
        target = center
        forward = target - camera
        forward /= max(float(np.linalg.norm(forward)), 1.0e-6)
        up_hint = np.asarray([0.0, 1.0, 0.0], dtype=np.float32)
        right = np.cross(forward, up_hint)
        right /= max(float(np.linalg.norm(right)), 1.0e-6)
        up = np.cross(right, forward)
        return camera, right, up, forward
    center = np.asarray(args.target, dtype=np.float32) if args.target is not None else 0.5 * (low + high)
    horizontal_extent = max(float(high[0] - low[0]), float(high[2] - low[2]))
    vertical_extent = float(high[1] - low[1])
    distance = args.camera_distance or max(
        args.camera_distance_scale * horizontal_extent, 0.95 * vertical_extent, 0.35
    )
    camera_height = args.camera_height_offset
    if camera_height is None:
        camera_height = 0.45 * vertical_extent
    angle = math.radians(args.arc_degrees) * (phase - 0.5)
    camera = center + np.asarray(
        [math.sin(angle) * distance, camera_height, math.cos(angle) * distance],
        dtype=np.float32,
    )
    target = center + np.asarray([0.0, 0.10 * vertical_extent, 0.0], dtype=np.float32)
    forward = target - camera
    forward /= max(float(np.linalg.norm(forward)), 1.0e-6)
    up_hint = np.asarray([0.0, 1.0, 0.0], dtype=np.float32)
    right = np.cross(forward, up_hint)
    right /= max(float(np.linalg.norm(right)), 1.0e-6)
    up = np.cross(right, forward)
    return camera, right, up, forward


def render(points, colors, phase, args):
    camera, right, up, forward = camera_path(points, phase, args)
    centered = points - camera
    local = np.stack((centered @ right, centered @ up, centered @ forward), axis=1)
    z = local[:, 2]
    focal = args.focal_scale * min(args.width, args.height)
    u = focal * local[:, 0] / np.maximum(z, 1.0e-6) + 0.5 * args.width
    v = -focal * local[:, 1] / np.maximum(z, 1.0e-6) + 0.5 * args.height
    valid = (z > 0.05) & np.isfinite(u) & np.isfinite(v)
    valid &= (u >= -args.point_radius) & (u < args.width + args.point_radius)
    valid &= (v >= -args.point_radius) & (v < args.height + args.point_radius)
    image = np.broadcast_to(BACKGROUND, (args.height, args.width, 3)).copy()
    if not np.any(valid):
        return image
    u = np.rint(u[valid]).astype(np.int32)
    v = np.rint(v[valid]).astype(np.int32)
    z = z[valid]
    rgb = np.clip((colors[valid].astype(np.float32) / 255.0) ** 0.55 * 255.0 + 0.5, 0, 255).astype(np.uint8)
    order = np.argsort(z)[::-1]
    u, v, rgb = u[order], v[order], rgb[order]
    for dy in range(-args.point_radius, args.point_radius + 1):
        for dx in range(-args.point_radius, args.point_radius + 1):
            px, py = u + dx, v + dy
            inside = (px >= 0) & (px < args.width) & (py >= 0) & (py < args.height)
            image[py[inside], px[inside]] = rgb[inside]
    return image


def main():
    args = parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)
    frames = args.outdir / "frames"
    frames.mkdir(exist_ok=True)
    points, colors = load_ply(args.ply)
    if args.focus_radius is not None:
        if args.target is None:
            raise ValueError("--focus-radius requires --target")
        center = np.asarray(args.target, dtype=np.float32)
        keep = np.linalg.norm(points - center, axis=1) <= args.focus_radius
        points, colors = points[keep], colors[keep]
        if len(points) == 0:
            raise RuntimeError("Focus radius removed all points")
    points, colors = robust_cloud(points, colors, args.max_points, args.seed)
    for index in range(args.frames):
        phase = index / max(args.frames - 1, 1)
        Image.fromarray(render(points, colors, phase, args)).save(
            frames / f"frame_{index:04d}.png", compress_level=2
        )
    output = args.outdir / "night3r_merged_pointcloud_arc.mp4"
    subprocess.run(
        [
            "ffmpeg", "-y", "-loglevel", "error", "-framerate", str(args.fps),
            "-i", str(frames / "frame_%04d.png"), "-c:v", "libx264", "-preset", "slow",
            "-crf", "18", "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(output),
        ],
        check=True,
    )
    print(output)


if __name__ == "__main__":
    main()
