#!/usr/bin/env python3
"""Render stable Night3R point-cloud videos for distinct single-pair scenes."""

import argparse
import json
import math
import subprocess
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont


BACKGROUND = np.asarray([246, 248, 251], dtype=np.uint8)
TEXT = (20, 29, 39)
SUBTEXT = (83, 98, 115)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Render one-pair Night3R point-cloud videos across distinct scenes."
    )
    parser.add_argument(
        "--source-root",
        type=Path,
        default=Path(
            "/data/fcr/data/Night3R/outputs/"
            "all_test_visual_sweep_step100_final_mildnoise_20260723/shards"
        ),
    )
    parser.add_argument("--outdir", type=Path, required=True)
    parser.add_argument("--mvsec-scenes", type=int, default=10)
    parser.add_argument("--frames", type=int, default=90)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--max-points", type=int, default=60000)
    parser.add_argument("--point-radius", type=int, default=1)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--preview-only", action="store_true")
    return parser.parse_args()


def load_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def rank_one_candidates(root, pattern):
    candidates = []
    for metrics_path in root.glob(pattern):
        payload = load_json(metrics_path)
        if int(payload.get("scene_rank", -1)) != 1:
            continue
        candidates.append((metrics_path.parent, payload))
    return candidates


def select_candidates(root, mvsec_scenes):
    dsec = rank_one_candidates(root, "gpu4_dsec_strict/candidates/*/rank01_*/metrics.json")
    mvsec = rank_one_candidates(root, "gpu4_mvsec_first9/candidates/*/rank01_*/metrics.json")
    mvsec += rank_one_candidates(root, "gpu5_mvsec_last9/candidates/*/rank01_*/metrics.json")
    dsec.sort(key=lambda item: item[1]["sample_id"])
    mvsec.sort(
        key=lambda item: (
            float(item[1]["metrics"]["night"]["median_absrel"]),
            item[1]["sample_id"],
        )
    )
    selected = dsec + mvsec[:mvsec_scenes]
    scene_ids = [payload["scene"] for _, payload in selected]
    if len(scene_ids) != len(set(scene_ids)):
        raise RuntimeError("Selection contains duplicate scene IDs")
    if len(selected) < 10:
        raise RuntimeError(f"Need at least 10 distinct scenes, found {len(selected)}")
    return selected


def viridis(values):
    anchors = np.asarray(
        [
            [68, 1, 84],
            [59, 82, 139],
            [33, 145, 140],
            [94, 201, 98],
            [253, 231, 37],
        ],
        dtype=np.float32,
    )
    scaled = np.clip(values, 0.0, 1.0) * (len(anchors) - 1)
    low = np.floor(scaled).astype(np.int64)
    high = np.minimum(low + 1, len(anchors) - 1)
    alpha = (scaled - low)[:, None]
    colors = anchors[low] * (1.0 - alpha) + anchors[high] * alpha
    return np.clip(colors + 0.5, 0, 255).astype(np.uint8)


def prepare_cloud(candidate_dir, payload, max_points, seed):
    arrays = np.load(candidate_dir / "geometry_points.npz")
    points = np.asarray(arrays["night_points"], dtype=np.float32)
    scale = float(payload["metrics"]["night"]["median_scale"])
    points = points * scale
    valid = np.isfinite(points).all(axis=1) & (points[:, 2] > 1.0e-6)
    points = points[valid]
    before_robust_filter = len(points)
    if before_robust_filter < 1000:
        raise RuntimeError(f"Too few valid Night3R points in {candidate_dir}")

    lower, upper = np.percentile(points, (1.0, 99.0), axis=0)
    center = ((lower + upper) * 0.5).astype(np.float32)
    distance = np.linalg.norm(points - center[None], axis=1)
    radius_limit = float(np.quantile(distance, 0.995))
    points = points[distance <= radius_limit]
    after_robust_filter = len(points)

    if len(points) > max_points:
        rng = np.random.default_rng(seed)
        indices = rng.choice(len(points), size=max_points, replace=False)
        points = points[indices]

    lower, upper = np.percentile(points, (1.0, 99.0), axis=0)
    center = ((lower + upper) * 0.5).astype(np.float32)
    distance = np.linalg.norm(points - center[None], axis=1)
    radius = max(float(np.quantile(distance, 0.995)), 1.0e-3)
    depth_low, depth_high = np.percentile(points[:, 2], (2.0, 98.0))
    normalized_depth = (points[:, 2] - depth_low) / max(float(depth_high - depth_low), 1.0e-6)
    colors = viridis(normalized_depth)
    return points, colors, center, radius, {
        "gt_median_scale": scale,
        "points_before_robust_filter": int(before_robust_filter),
        "points_after_robust_filter": int(after_robust_filter),
        "points_rendered": int(len(points)),
        "depth_color_range": [float(depth_low), float(depth_high)],
    }


def load_font(size, bold=False):
    name = "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf"
    path = Path("/usr/share/fonts/truetype/dejavu") / name
    return ImageFont.truetype(str(path), size) if path.is_file() else ImageFont.load_default()


def orbit_angles(phase):
    eased = phase * phase * (3.0 - 2.0 * phase)
    azimuth = math.radians(-62.0 + 80.0 * eased)
    elevation = math.radians(12.0 + 5.0 * math.sin(math.pi * phase))
    return azimuth, elevation


def camera_basis(azimuth, elevation):
    offset_direction = np.asarray(
        [
            math.sin(azimuth) * math.cos(elevation),
            -math.sin(elevation),
            -math.cos(azimuth) * math.cos(elevation),
        ],
        dtype=np.float32,
    )
    forward = -offset_direction
    up_hint = np.asarray([0.0, -1.0, 0.0], dtype=np.float32)
    right = np.cross(forward, up_hint)
    right /= max(float(np.linalg.norm(right)), 1.0e-6)
    up = np.cross(right, forward)
    return offset_direction, forward, right, up


def fitted_camera_distance(points, center, width, height):
    relative = points - center[None]
    vertical_tangent = math.tan(math.radians(48.0) * 0.5)
    horizontal_tangent = vertical_tangent * width / height
    required = []
    for phase in np.linspace(0.0, 1.0, 13):
        azimuth, elevation = orbit_angles(float(phase))
        _, forward, right, up = camera_basis(azimuth, elevation)
        x = relative @ right
        y = relative @ up
        z_offset = relative @ forward
        required_x = np.abs(x) / (horizontal_tangent * 0.88) - z_offset
        required_y = np.abs(y) / (vertical_tangent * 0.88) - z_offset
        required.append(float(np.max(np.maximum(required_x, required_y))))
    return max(max(required) * 1.04, 1.0e-3)


def camera_at_phase(center, distance, phase):
    azimuth, elevation = orbit_angles(phase)
    offset_direction, _, _, _ = camera_basis(azimuth, elevation)
    offset = np.asarray(
        offset_direction * distance,
        dtype=np.float32,
    )
    return center + offset, math.degrees(azimuth), math.degrees(elevation), distance


def rasterize(points, colors, center, distance, phase, width, height, point_radius):
    camera, azimuth, elevation, distance = camera_at_phase(center, distance, phase)
    _, forward, right, up = camera_basis(
        math.radians(azimuth), math.radians(elevation)
    )

    relative = points - camera[None]
    camera_x = relative @ right
    camera_y = relative @ up
    camera_z = relative @ forward
    focal = 0.5 * height / math.tan(math.radians(48.0) * 0.5)
    u = focal * camera_x / np.maximum(camera_z, 1.0e-6) + width * 0.5
    v = height * 0.5 - focal * camera_y / np.maximum(camera_z, 1.0e-6)
    valid = camera_z > 1.0e-5
    valid &= np.isfinite(u) & np.isfinite(v)
    valid &= (u >= -point_radius) & (u < width + point_radius)
    valid &= (v >= -point_radius) & (v < height + point_radius)

    u = np.rint(u[valid]).astype(np.int32)
    v = np.rint(v[valid]).astype(np.int32)
    depths = camera_z[valid]
    visible_colors = colors[valid]
    order = np.argsort(depths)[::-1]
    u, v, visible_colors = u[order], v[order], visible_colors[order]
    canvas = np.broadcast_to(BACKGROUND, (height, width, 3)).copy()
    foreground = np.zeros((height, width), dtype=bool)
    for dy in range(-point_radius, point_radius + 1):
        for dx in range(-point_radius, point_radius + 1):
            if dx * dx + dy * dy > point_radius * point_radius + 1:
                continue
            px, py = u + dx, v + dy
            inside = (px >= 0) & (px < width) & (py >= 0) & (py < height)
            canvas[py[inside], px[inside]] = visible_colors[inside]
            foreground[py[inside], px[inside]] = True
    return canvas, float(foreground.mean()), {
        "azimuth_degrees": azimuth,
        "elevation_degrees": elevation,
        "distance": distance,
        "position": camera.tolist(),
    }


def label_frame(array, payload):
    image = Image.fromarray(array)
    draw = ImageDraw.Draw(image)
    draw.text((34, 28), "Night3R", fill=TEXT, font=load_font(30, bold=True))
    draw.text(
        (34, 70),
        payload["sample_id"],
        fill=SUBTEXT,
        font=load_font(20),
    )
    draw.text(
        (34, image.height - 44),
        "Single RGB-event pair  |  GT-median aligned point cloud",
        fill=SUBTEXT,
        font=load_font(17),
    )
    return image


def output_name(index, payload):
    dataset = payload["dataset"].lower()
    sample = payload["sample_id"].replace(":", "_").replace("-", "-")
    return f"{index:02d}_{dataset}_{sample}"


def encode_video(frames_dir, output, fps):
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-loglevel",
            "error",
            "-framerate",
            str(fps),
            "-i",
            str(frames_dir / "frame_%04d.png"),
            "-c:v",
            "libx264",
            "-preset",
            "slow",
            "-crf",
            "18",
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            str(output),
        ],
        check=True,
    )


def render_scene(args, index, candidate_dir, payload):
    scene_dir = args.outdir / output_name(index, payload)
    frames_dir = scene_dir / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)
    points, colors, center, radius, cloud_meta = prepare_cloud(
        candidate_dir, payload, args.max_points, args.seed + index
    )
    camera_distance = fitted_camera_distance(
        points, center, args.width, args.height
    )
    np.savez_compressed(
        scene_dir / "night3r_pointcloud_gt_median_aligned.npz",
        points=points,
        colors=colors,
    )

    frame_indices = [args.frames // 2] if args.preview_only else list(range(args.frames))
    foreground_fractions = []
    camera_path = []
    for frame_index in frame_indices:
        phase = frame_index / max(args.frames - 1, 1)
        array, foreground_fraction, camera = rasterize(
            points,
            colors,
            center,
            camera_distance,
            phase,
            args.width,
            args.height,
            args.point_radius,
        )
        image = label_frame(array, payload)
        target = (
            scene_dir / "preview_middle.png"
            if args.preview_only
            else frames_dir / f"frame_{frame_index:04d}.png"
        )
        image.save(target, compress_level=2)
        foreground_fractions.append(foreground_fraction)
        camera_path.append(camera)

    video_path = scene_dir / "night3r_single_pair_pointcloud.mp4"
    if not args.preview_only:
        encode_video(frames_dir, video_path, args.fps)
        middle = Image.open(frames_dir / f"frame_{args.frames // 2:04d}.png")
        middle.save(scene_dir / "preview_middle.png", compress_level=2)

    metadata = {
        "dataset": payload["dataset"],
        "split_label": payload["split_label"],
        "scene": payload["scene"],
        "sample_id": payload["sample_id"],
        "source_candidate": str(candidate_dir.resolve()),
        "coordinate_frame": "Night3R pair point maps in view-1 camera frame",
        "alignment": "one GT-median scale applied to all XYZ coordinates",
        "geometry_smoothing": "none; static filtered point cloud",
        "camera_smoothing": "cubic smoothstep azimuth with sinusoidal elevation",
        "fixed_camera_target": center.tolist(),
        "fixed_cloud_radius": radius,
        "fixed_camera_distance": camera_distance,
        "frames": args.frames,
        "fps": args.fps,
        "resolution": [args.width, args.height],
        "foreground_fraction_min": float(min(foreground_fractions)),
        "foreground_fraction_max": float(max(foreground_fractions)),
        "night3r_median_absrel": float(payload["metrics"]["night"]["median_absrel"]),
        "night3r_median_delta1": float(payload["metrics"]["night"]["median_delta1"]),
        "cloud": cloud_meta,
        "camera_samples": camera_path,
        "video": str(video_path.name) if not args.preview_only else None,
    }
    (scene_dir / "render_metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
    return scene_dir, metadata


def make_contact_sheet(scene_dirs, output):
    tiles = []
    for scene_dir in scene_dirs:
        image = Image.open(scene_dir / "preview_middle.png").convert("RGB")
        tiles.append(image.resize((400, 225), Image.Resampling.LANCZOS))
    columns = 4
    rows = math.ceil(len(tiles) / columns)
    sheet = Image.new("RGB", (columns * 400, rows * 225), tuple(BACKGROUND.tolist()))
    for index, tile in enumerate(tiles):
        sheet.paste(tile, ((index % columns) * 400, (index // columns) * 225))
    sheet.save(output, compress_level=2)


def make_reel(scene_dirs, output):
    concat_file = output.with_suffix(".concat.txt")
    lines = [f"file '{(scene_dir / 'night3r_single_pair_pointcloud.mp4').resolve()}'" for scene_dir in scene_dirs]
    concat_file.write_text("\n".join(lines) + "\n", encoding="utf-8")
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-loglevel",
            "error",
            "-f",
            "concat",
            "-safe",
            "0",
            "-i",
            str(concat_file),
            "-c",
            "copy",
            "-movflags",
            "+faststart",
            str(output),
        ],
        check=True,
    )


def main():
    args = parse_args()
    if args.frames < 2 or args.fps < 1:
        raise ValueError("frames must be >=2 and fps must be positive")
    if args.width % 2 or args.height % 2:
        raise ValueError("H.264 output dimensions must be even")
    args.outdir.mkdir(parents=True, exist_ok=True)
    selection = select_candidates(args.source_root, args.mvsec_scenes)
    scene_dirs = []
    records = []
    for index, (candidate_dir, payload) in enumerate(selection, start=1):
        scene_dir, metadata = render_scene(args, index, candidate_dir, payload)
        scene_dirs.append(scene_dir)
        records.append(metadata)
        print(f"[{index:02d}/{len(selection):02d}] {payload['sample_id']}", flush=True)

    make_contact_sheet(scene_dirs, args.outdir / "overview_12_scenes.png")
    if not args.preview_only:
        make_reel(scene_dirs, args.outdir / "night3r_12scene_pointcloud_reel.mp4")
    manifest = {
        "scene_count": len(records),
        "distinct_scene_count": len({record["scene"] for record in records}),
        "selection": "all two DSEC rank-1 scenes plus ten MVSEC rank-1 scenes with lowest Night3R median AbsRel",
        "preview_only": args.preview_only,
        "records": records,
    }
    (args.outdir / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({key: manifest[key] for key in manifest if key != "records"}, indent=2))


if __name__ == "__main__":
    main()
