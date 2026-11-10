#!/usr/bin/env python3
"""Render local view-1-camera point-cloud videos with RGB colors and raw scale."""

import argparse
import json
import math
import subprocess
from pathlib import Path

import numpy as np
import yaml
from PIL import Image, ImageDraw, ImageFont


BACKGROUND = np.asarray([14, 18, 24], dtype=np.uint8)
TEXT = (238, 243, 247)
SUBTEXT = (170, 184, 197)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source-root",
        type=Path,
        default=Path(
            "/data/fcr/data/Night3R/outputs/"
            "all_test_visual_sweep_step100_final_mildnoise_20260723/shards"
        ),
    )
    parser.add_argument("--data-root", type=Path, default=Path("/data/fcr/data/Night3R/test"))
    parser.add_argument("--outdir", type=Path, required=True)
    parser.add_argument("--mvsec-scenes", type=int, default=10)
    parser.add_argument("--frames", type=int, default=90)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--max-points", type=int, default=80000)
    parser.add_argument("--point-radius", type=int, default=1)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--preview-only", action="store_true")
    return parser.parse_args()


def load_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def candidates(root, pattern):
    result = []
    for path in root.glob(pattern):
        payload = load_json(path)
        if int(payload.get("scene_rank", -1)) == 1:
            result.append((path.parent, payload))
    return result


def select_candidates(root, mvsec_scenes):
    dsec = candidates(root, "gpu4_dsec_strict/candidates/*/rank01_*/metrics.json")
    mvsec = candidates(root, "gpu4_mvsec_first9/candidates/*/rank01_*/metrics.json")
    mvsec += candidates(root, "gpu5_mvsec_last9/candidates/*/rank01_*/metrics.json")
    dsec.sort(key=lambda x: x[1]["sample_id"])
    mvsec.sort(key=lambda x: (x[1]["metrics"]["night"]["median_absrel"], x[1]["sample_id"]))
    selected = dsec + mvsec[:mvsec_scenes]
    if len({x[1]["scene"] for x in selected}) != len(selected):
        raise RuntimeError("Selection contains duplicate scene IDs")
    return selected


def scene_intrinsics(data_root, dataset, scene):
    camera_path = data_root / dataset / scene / "cameras.yaml"
    payload = yaml.safe_load(camera_path.read_text(encoding="utf-8"))
    camera = payload.get("rgb_cam_crop", payload.get("rgb_cam_full"))
    fx, fy, cx, cy = [float(x) for x in camera["intrinsics"]]
    source_w, source_h = [float(x) for x in camera["resolution"]]
    scale_x, scale_y = 512.0 / source_w, 512.0 / source_h
    return np.asarray(
        [[fx * scale_x, 0.0, cx * scale_x], [0.0, fy * scale_y, cy * scale_y], [0, 0, 1]],
        dtype=np.float32,
    ), [int(source_w), int(source_h)]


def rgb_color_points(candidate_dir, payload, data_root, max_points, seed):
    arrays = np.load(candidate_dir / "geometry_points.npz")
    # The cached pair points are already expressed in the first-view camera
    # frame by the sweep exporter. They provide the local second-view coverage
    # while the camera itself remains anchored at view 1.
    points = np.asarray(arrays["night_points"], dtype=np.float32)
    points = points[np.isfinite(points).all(axis=1)]
    points = points[points[:, 2] > 1.0e-5]
    if len(points) < 1000:
        raise RuntimeError(f"No usable raw camera-frame points: {candidate_dir}")
    image = np.asarray(Image.open(candidate_dir / "clean_rgb_view1.png").convert("RGB"))
    K, source_size = scene_intrinsics(data_root, payload["dataset"], payload["scene"])
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    u = fx * points[:, 0] / points[:, 2] + cx
    v = fy * points[:, 1] / points[:, 2] + cy
    visible = np.isfinite(u) & np.isfinite(v)
    visible &= (u >= 1.0) & (u < image.shape[1] - 1.0)
    visible &= (v >= 1.0) & (v < image.shape[0] - 1.0)
    points, u, v = points[visible], u[visible], v[visible]
    if len(points) < 1000:
        raise RuntimeError(f"Too few RGB-projectable points: {candidate_dir}")
    pixel_x = np.rint(u).astype(np.int32)
    pixel_y = np.rint(v).astype(np.int32)
    colors = image[pixel_y, pixel_x].copy()
    # Display-only lift for very dark sensor RGB; hue and relative channel
    # ordering remain the original RGB projection.
    colors = np.clip((colors.astype(np.float32) / 255.0) ** 0.70 * 255.0 + 0.5, 0, 255).astype(np.uint8)
    # Restrict the local camera view to the robust in-frustum depth support.
    z_low, z_high = np.percentile(points[:, 2], (0.5, 99.5))
    keep = (points[:, 2] >= z_low) & (points[:, 2] <= z_high)
    points, colors = points[keep], colors[keep]
    if len(points) > max_points:
        rng = np.random.default_rng(seed)
        indices = rng.choice(len(points), size=max_points, replace=False)
        points, colors = points[indices], colors[indices]

    # The camera starts at the view-1 origin. Focus on a local near-field slab;
    # distant background surfaces are outside this camera-local reconstruction.
    central = points[
        (np.abs(points[:, 0]) < np.percentile(np.abs(points[:, 0]), 70))
        & (np.abs(points[:, 1]) < np.percentile(np.abs(points[:, 1]), 70))
    ]
    focus_depth = float(np.median(central[:, 2] if len(central) else points[:, 2]))
    focus_depth = max(focus_depth, float(np.percentile(points[:, 2], 15)))
    local_keep = (points[:, 2] >= max(0.25 * focus_depth, z_low)) & (
        points[:, 2] <= min(3.0 * focus_depth, z_high)
    )
    if int(local_keep.sum()) >= 1000:
        points, colors = points[local_keep], colors[local_keep]
    return points, colors, K, source_size, focus_depth, {
        "raw_scale": True,
        "gt_median_scale_applied": False,
        "points_before_rgb_projection": int(len(arrays["night_points"])),
        "points_after_rgb_projection": int(len(points)),
        "local_depth_crop": [max(0.25 * focus_depth, z_low), min(3.0 * focus_depth, z_high)],
        "raw_depth_range": [float(z_low), float(z_high)],
        "focus_depth": focus_depth,
    }


def look_at(position, target):
    forward = target - position
    forward /= max(float(np.linalg.norm(forward)), 1.0e-6)
    up_hint = np.asarray([0.0, -1.0, 0.0], dtype=np.float32)
    right = np.cross(forward, up_hint)
    right /= max(float(np.linalg.norm(right)), 1.0e-6)
    up = np.cross(right, forward)
    return forward, right, up


def camera_pose(focus_depth, phase):
    # Small closed local inspection path. phase=0 and phase=1 are exactly the
    # original view-1 camera pose, so the clip can loop without a jump.
    oscillation = math.sin(2.0 * math.pi * phase)
    yaw = math.radians(8.0 * oscillation)
    pitch = math.radians(2.2 * oscillation)
    lateral = 0.045 * focus_depth * oscillation
    vertical = -0.025 * focus_depth * oscillation
    forward_offset = -0.035 * focus_depth * math.sin(math.pi * phase) ** 2
    position = np.asarray([lateral, vertical, forward_offset], dtype=np.float32)
    target = np.asarray(
        [0.0, -0.012 * focus_depth * math.sin(pitch), focus_depth], dtype=np.float32
    )
    return position, target, math.degrees(yaw), math.degrees(pitch)


def rasterize(points, colors, K, focus_depth, phase, width, height, point_radius):
    position, target, yaw, pitch = camera_pose(focus_depth, phase)
    forward, right, up = look_at(position, target)
    relative = points - position[None]
    x = relative @ right
    y = relative @ up
    z = relative @ forward
    viewport = min(height - 140, width - 80)
    scale = viewport / 512.0
    origin_x = (width - viewport) * 0.5
    origin_y = 96.0
    # Preserve the calibrated principal point. The only display transform is
    # a uniform scale from the 512-square model canvas to the square viewport.
    u = scale * (K[0, 0] * x / np.maximum(z, 1.0e-6) + K[0, 2]) + origin_x
    v = scale * (-K[1, 1] * y / np.maximum(z, 1.0e-6) + K[1, 2]) + origin_y
    valid = (z > 1.0e-5) & np.isfinite(u) & np.isfinite(v)
    valid &= (u >= -point_radius) & (u < width + point_radius)
    valid &= (v >= -point_radius) & (v < height + point_radius)
    u = np.rint(u[valid]).astype(np.int32)
    v = np.rint(v[valid]).astype(np.int32)
    depths = z[valid]
    visible_colors = colors[valid]
    order = np.argsort(depths)[::-1]
    u, v, visible_colors = u[order], v[order], visible_colors[order]
    canvas = np.broadcast_to(BACKGROUND, (height, width, 3)).copy()
    occupancy = np.zeros((height, width), dtype=bool)
    for dy in range(-point_radius, point_radius + 1):
        for dx in range(-point_radius, point_radius + 1):
            px, py = u + dx, v + dy
            inside = (px >= 0) & (px < width) & (py >= 0) & (py < height)
            canvas[py[inside], px[inside]] = visible_colors[inside]
            occupancy[py[inside], px[inside]] = True
    return canvas, float(occupancy.mean()), {
        "position": position.tolist(),
        "target": target.tolist(),
        "yaw_degrees": yaw,
        "pitch_degrees": pitch,
    }


def font(size, bold=False):
    name = "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf"
    path = Path("/usr/share/fonts/truetype/dejavu") / name
    return ImageFont.truetype(str(path), size) if path.is_file() else ImageFont.load_default()


def label(image, payload):
    result = Image.fromarray(image)
    draw = ImageDraw.Draw(result)
    draw.text((32, 26), "Night3R", fill=TEXT, font=font(30, True))
    draw.text((32, 70), payload["sample_id"], fill=SUBTEXT, font=font(20))
    draw.text((32, image.shape[0] - 44), "View-1 camera-local point cloud  |  RGB textured  |  raw scale", fill=SUBTEXT, font=font(17))
    return result


def encode(frames, output, fps):
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-framerate", str(fps), "-i", str(frames / "frame_%04d.png"), "-c:v", "libx264", "-preset", "slow", "-crf", "18", "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(output)], check=True)


def render_scene(args, index, candidate_dir, payload):
    name = f"{index:02d}_{payload['dataset'].lower()}_{payload['sample_id'].replace(':', '_')}"
    scene_dir = args.outdir / name
    frames = scene_dir / "frames"
    frames.mkdir(parents=True, exist_ok=True)
    points, colors, K, source_size, focus_depth, cloud_meta = rgb_color_points(
        candidate_dir, payload, args.data_root, args.max_points, args.seed + index
    )
    np.savez_compressed(scene_dir / "night3r_pointcloud_local_raw_rgb.npz", points=points, colors=colors)
    indices = [args.frames // 2] if args.preview_only else list(range(args.frames))
    occupancy, camera_path = [], []
    for frame_index in indices:
        phase = frame_index / max(args.frames - 1, 1)
        array, fraction, camera = rasterize(points, colors, K, focus_depth, phase, args.width, args.height, args.point_radius)
        target = scene_dir / "preview_middle.png" if args.preview_only else frames / f"frame_{frame_index:04d}.png"
        label(array, payload).save(target, compress_level=2)
        occupancy.append(fraction)
        camera_path.append(camera)
    video = scene_dir / "night3r_local_rgb_pointcloud.mp4"
    if not args.preview_only:
        encode(frames, video, args.fps)
        Image.open(frames / f"frame_{args.frames // 2:04d}.png").save(scene_dir / "preview_middle.png", compress_level=2)
    metadata = {
        "dataset": payload["dataset"], "scene": payload["scene"], "sample_id": payload["sample_id"],
        "source_candidate": str(candidate_dir.resolve()), "coordinate_frame": "view-1 camera frame",
        "depth_alignment": "none; raw model point-map scale", "coloring": "clean RGB view-1 projection",
        "camera_path": "local view-1 pose: +/-8 degree yaw, +/-2.2 degree pitch, small lateral/depth offsets",
        "focus_depth_raw": focus_depth, "intrinsics_512": K.tolist(), "source_resolution": source_size,
        "frames": args.frames, "fps": args.fps, "resolution": [args.width, args.height],
        "occupancy_min": min(occupancy), "occupancy_max": max(occupancy), "cloud": cloud_meta,
        "camera_samples": camera_path, "video": video.name if not args.preview_only else None,
    }
    (scene_dir / "render_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    return scene_dir, metadata


def contact_sheet(scene_dirs, output):
    tiles = [Image.open(d / "preview_middle.png").convert("RGB").resize((400, 225), Image.Resampling.LANCZOS) for d in scene_dirs]
    cols, rows = 4, math.ceil(len(tiles) / 4)
    sheet = Image.new("RGB", (cols * 400, rows * 225), tuple(BACKGROUND.tolist()))
    for i, tile in enumerate(tiles): sheet.paste(tile, ((i % cols) * 400, (i // cols) * 225))
    sheet.save(output, compress_level=2)


def reel(scene_dirs, output):
    concat = output.with_suffix(".concat.txt")
    concat.write_text("\n".join(f"file '{(d / 'night3r_local_rgb_pointcloud.mp4').resolve()}'" for d in scene_dirs) + "\n", encoding="utf-8")
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-f", "concat", "-safe", "0", "-i", str(concat), "-c", "copy", "-movflags", "+faststart", str(output)], check=True)


def main():
    args = parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)
    selected = select_candidates(args.source_root, args.mvsec_scenes)
    scene_dirs, records = [], []
    for i, (candidate_dir, payload) in enumerate(selected, 1):
        scene_dir, metadata = render_scene(args, i, candidate_dir, payload)
        scene_dirs.append(scene_dir); records.append(metadata)
        print(f"[{i}/{len(selected)}] {payload['sample_id']}", flush=True)
    contact_sheet(scene_dirs, args.outdir / "overview_local_rgb.png")
    if not args.preview_only: reel(scene_dirs, args.outdir / "night3r_local_rgb_12scene_reel.mp4")
    manifest = {"scene_count": len(records), "distinct_scene_count": len({r["scene"] for r in records}), "depth_alignment": "none", "coloring": "RGB projection", "records": records}
    (args.outdir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__": main()
