#!/usr/bin/env python3
"""Fuse cached Night3R pair clouds into complete RGB scene videos."""

import argparse
import json
import math
import subprocess
from pathlib import Path

import numpy as np
import yaml
from PIL import Image, ImageDraw, ImageFont


BACKGROUND = np.asarray([12, 16, 22], dtype=np.uint8)
TEXT = (238, 243, 247)
SUBTEXT = (169, 183, 196)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source-root",
        type=Path,
        action="append",
        default=None,
        help=(
            "Candidate-cache shard root. Repeat this option to merge compatible "
            "Night3R runs from the same checkpoint."
        ),
    )
    parser.add_argument("--data-root", type=Path, default=Path("test/MVSEC"))
    parser.add_argument("--outdir", type=Path, required=True)
    parser.add_argument("--scenes", type=int, default=10)
    parser.add_argument("--pairs-per-scene", type=int, default=4)
    parser.add_argument("--frames", type=int, default=90)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--voxel-size", type=float, default=0.035)
    parser.add_argument("--max-points", type=int, default=180000)
    parser.add_argument("--point-radius", type=int, default=1)
    parser.add_argument("--seed", type=int, default=29)
    parser.add_argument("--preview-only", action="store_true")
    args = parser.parse_args()
    if args.source_root is None:
        args.source_root = [
            Path("outputs/all_test_visual_sweep_step100_final_mildnoise_20260723/shards")
        ]
    return args


def load_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def raw_depth_median(candidate_dir):
    arrays = np.load(candidate_dir / "geometry_points.npz")
    points = np.asarray(arrays["night_points"], dtype=np.float32)
    z = points[:, 2]
    z = z[np.isfinite(z) & (z > 1.0e-5)]
    if len(z) < 1000:
        return None
    return float(np.median(z))


def candidate_scenes(source_roots):
    scenes = {}
    patterns = (
        "gpu4_mvsec_first9/candidates/*/rank*_*/metrics.json",
        "gpu5_mvsec_last9/candidates/*/rank*_*/metrics.json",
    )
    for source_index, source_root in enumerate(source_roots):
        for pattern in patterns:
            for path in source_root.glob(pattern):
                payload = load_json(path)
                if payload.get("dataset") != "MVSEC":
                    continue
                median = raw_depth_median(path.parent)
                if median is None:
                    continue
                record = {
                    "candidate_dir": path.parent,
                    "payload": payload,
                    "source_root": source_root,
                    "source_index": source_index,
                    "raw_depth_median": median,
                }
                scenes.setdefault(payload["scene"], []).append(record)

    # Different visual sweeps can contain the same pair. Retain the run with
    # the lower aligned error, while preserving source-root order on ties.
    for scene, scene_records in list(scenes.items()):
        deduplicated = {}
        for record in scene_records:
            payload = record["payload"]
            key = int(payload["idx1"]), int(payload["idx2"])
            quality = float(payload["metrics"]["night"]["median_absrel"])
            previous = deduplicated.get(key)
            if previous is None:
                deduplicated[key] = record
                continue
            previous_quality = float(
                previous["payload"]["metrics"]["night"]["median_absrel"]
            )
            if (quality, record["source_index"]) < (
                previous_quality,
                previous["source_index"],
            ):
                deduplicated[key] = record
        scenes[scene] = sorted(
            deduplicated.values(),
            key=lambda record: (
                int(record["payload"]["idx1"]),
                int(record["payload"]["idx2"]),
            ),
        )
    return scenes


def select_diverse_pairs(records, pairs_per_scene):
    if len(records) <= pairs_per_scene:
        return records
    medians = np.asarray([record["raw_depth_median"] for record in records])
    cluster_median = float(np.median(medians))
    stable = [
        record
        for record in records
        if 0.60 <= record["raw_depth_median"] / cluster_median <= 1.67
    ]
    if len(stable) < pairs_per_scene:
        stable = records
    positions = np.asarray([int(record["payload"]["idx1"]) for record in stable])
    span = max(float(positions.max() - positions.min()), 1.0)
    quality = np.asarray(
        [float(record["payload"]["metrics"]["night"]["median_absrel"]) for record in stable]
    )
    quality = (quality - quality.min()) / max(float(np.ptp(quality)), 1.0e-6)
    selected = []
    for target in np.linspace(float(positions.min()), float(positions.max()), pairs_per_scene):
        scores = np.abs(positions - target) / span + 0.20 * quality
        for index, record in enumerate(stable):
            if record in selected:
                scores[index] = np.inf
        selected.append(stable[int(np.argmin(scores))])
    return sorted(
        selected,
        key=lambda record: (
            int(record["payload"]["idx1"]),
            int(record["payload"]["idx2"]),
        ),
    )


def select_scenes(source_roots, count, pairs_per_scene):
    scenes = candidate_scenes(source_roots)
    ranked = []
    for scene, records in scenes.items():
        if len(records) < pairs_per_scene:
            continue
        selected_records = select_diverse_pairs(records, pairs_per_scene)
        score = float(
            np.mean(
                [
                    record["payload"]["metrics"]["night"]["median_absrel"]
                    for record in selected_records
                ]
            )
        )
        ranked.append((score, scene, selected_records))
    ranked.sort(key=lambda item: (item[0], item[1]))
    selected = ranked[:count]
    if len(selected) < count:
        raise RuntimeError(f"Found only {len(selected)} scenes with {pairs_per_scene} pairs")
    return selected


def intrinsics_512(scene_dir):
    cameras = yaml.safe_load((scene_dir / "cameras.yaml").read_text(encoding="utf-8"))
    camera = cameras.get("rgb_cam_crop", cameras.get("rgb_cam_full"))
    fx, fy, cx, cy = [float(value) for value in camera["intrinsics"]]
    width, height = [float(value) for value in camera["resolution"]]
    sx, sy = 512.0 / width, 512.0 / height
    return np.asarray(
        [[fx * sx, 0.0, cx * sx], [0.0, fy * sy, cy * sy], [0.0, 0.0, 1.0]],
        dtype=np.float32,
    )


def load_poses(scene_dir):
    payload = np.load(scene_dir / "meta" / "poses_selected.npz")
    for key in ("T_world_cam", "T_world_camera", "T_world_body"):
        if key in payload:
            poses = np.asarray(payload[key], dtype=np.float64)
            if poses.ndim == 3 and poses.shape[1:] == (4, 4):
                return key, poses
    raise RuntimeError(f"No camera pose array in {scene_dir}")


def transform_points(points, transform):
    return points @ transform[:3, :3].T + transform[:3, 3]


def project_colors(points, image, intrinsic):
    z = points[:, 2]
    u = intrinsic[0, 0] * points[:, 0] / z + intrinsic[0, 2]
    v = intrinsic[1, 1] * points[:, 1] / z + intrinsic[1, 2]
    valid = np.isfinite(u) & np.isfinite(v) & np.isfinite(z) & (z > 1.0e-5)
    valid &= (u >= 1.0) & (u < image.shape[1] - 1.0)
    valid &= (v >= 1.0) & (v < image.shape[0] - 1.0)
    pixel_x = np.rint(u[valid]).astype(np.int32)
    pixel_y = np.rint(v[valid]).astype(np.int32)
    colors = image[pixel_y, pixel_x].copy()
    # Display-only lift keeps the source RGB hue/order while making the real
    # MVSEC night imagery visible in a dark-background point rendering.
    colors = np.clip(
        (colors.astype(np.float32) / 255.0) ** 0.65 * 255.0 + 0.5, 0, 255
    ).astype(np.uint8)
    return points[valid], colors


def voxel_downsample(points, colors, voxel_size):
    if voxel_size <= 0 or len(points) == 0:
        return points, colors
    keys = np.floor(points / voxel_size).astype(np.int64)
    _, keep = np.unique(keys, axis=0, return_index=True)
    keep.sort()
    return points[keep], colors[keep]


def fuse_scene(scene_dir, records, args):
    intrinsic = intrinsics_512(scene_dir)
    pose_key, poses = load_poses(scene_dir)
    reference_index = min(int(record["payload"]["idx1"]) for record in records)
    world_to_reference = np.linalg.inv(poses[reference_index])
    point_sets, color_sets, pair_records, camera_centers = [], [], [], []

    for record in records:
        candidate_dir = record["candidate_dir"]
        payload = record["payload"]
        idx1, idx2 = int(payload["idx1"]), int(payload["idx2"])
        arrays = np.load(candidate_dir / "geometry_points.npz")
        points = np.asarray(arrays["night_points"], dtype=np.float32)
        points = points[np.isfinite(points).all(axis=1) & (points[:, 2] > 1.0e-5)]
        if len(points) < 1000:
            continue
        z_low, z_high = np.percentile(points[:, 2], (0.2, 99.8))
        points = points[(points[:, 2] >= z_low) & (points[:, 2] <= z_high)]
        image = np.asarray(Image.open(candidate_dir / "clean_rgb_view1.png").convert("RGB"))
        points, colors = project_colors(points, image, intrinsic)
        if len(points) < 1000:
            continue
        reference_from_camera = world_to_reference @ poses[idx1]
        points = transform_points(points.astype(np.float64), reference_from_camera).astype(
            np.float32
        )
        point_sets.append(points)
        color_sets.append(colors)
        for frame_index in (idx1, idx2):
            center = (world_to_reference @ poses[frame_index])[:3, 3]
            camera_centers.append(center.astype(np.float32))
        pair_records.append(
            {
                "sample_id": payload["sample_id"],
                "idx1": idx1,
                "idx2": idx2,
                "source": str(candidate_dir.resolve()),
                "source_root": str(record["source_root"]),
                "points_after_rgb_projection": int(len(points)),
                "raw_depth_percentiles": [float(z_low), float(z_high)],
                "raw_depth_median_before_filter": record["raw_depth_median"],
            }
        )

    if len(point_sets) < 2:
        raise RuntimeError(f"Insufficient usable pair clouds for {scene_dir.name}")
    points = np.concatenate(point_sets, axis=0)
    colors = np.concatenate(color_sets, axis=0)
    points, colors = voxel_downsample(points, colors, args.voxel_size)

    center = np.median(points, axis=0)
    radius = np.linalg.norm(points - center[None], axis=1)
    keep = radius <= np.percentile(radius, 99.7)
    points, colors = points[keep], colors[keep]
    if len(points) > args.max_points:
        rng = np.random.default_rng(args.seed + reference_index)
        keep = rng.choice(len(points), size=args.max_points, replace=False)
        points, colors = points[keep], colors[keep]
    return points, colors, np.asarray(camera_centers), intrinsic, pose_key, pair_records


def robust_bounds(points):
    low = np.percentile(points, 0.5, axis=0)
    high = np.percentile(points, 99.5, axis=0)
    center = (low + high) * 0.5
    radius = max(float(np.linalg.norm(high - low) * 0.5), 0.5)
    return center.astype(np.float32), radius, low.astype(np.float32), high.astype(np.float32)


def look_at(position, target):
    forward = target - position
    forward /= max(float(np.linalg.norm(forward)), 1.0e-6)
    up_hint = np.asarray([0.0, -1.0, 0.0], dtype=np.float32)
    right = np.cross(forward, up_hint)
    if np.linalg.norm(right) < 1.0e-5:
        up_hint = np.asarray([0.0, 0.0, 1.0], dtype=np.float32)
        right = np.cross(forward, up_hint)
    right /= max(float(np.linalg.norm(right)), 1.0e-6)
    up = np.cross(right, forward)
    up /= max(float(np.linalg.norm(up)), 1.0e-6)
    return forward, right, up


def rotate_axis(vector, axis, angle):
    axis = axis / max(float(np.linalg.norm(axis)), 1.0e-6)
    return (
        vector * math.cos(angle)
        + np.cross(axis, vector) * math.sin(angle)
        + axis * float(np.dot(axis, vector)) * (1.0 - math.cos(angle))
    )


def fitted_distance(center, direction, fit_points, focal, width, height, radius):
    target_width = 0.82 * width
    target_height = 0.76 * height
    forward = direction / max(float(np.linalg.norm(direction)), 1.0e-6)
    up_hint = np.asarray([0.0, -1.0, 0.0], dtype=np.float32)
    right = np.cross(forward, up_hint)
    right /= max(float(np.linalg.norm(right)), 1.0e-6)
    camera_up = np.cross(right, forward)
    centered = fit_points - center[None]
    x = centered @ right
    y = centered @ camera_up
    z_offset = centered @ forward

    def fits(distance):
        z = distance + z_offset
        if np.min(z) <= max(0.01 * radius, 1.0e-4):
            return False
        u = focal * x / z
        v = focal * y / z
        return (float(np.max(u) - np.min(u)) <= target_width) and (
            float(np.max(v) - np.min(v)) <= target_height
        )

    low_distance = max(0.05 * radius, 1.0e-3)
    high_distance = max(2.0 * radius, 1.0)
    while not fits(high_distance):
        high_distance *= 1.5
        if high_distance > 1000.0 * max(radius, 1.0):
            raise RuntimeError("Could not fit fused scene into the camera viewport")
    for _ in range(18):
        middle = 0.5 * (low_distance + high_distance)
        if fits(middle):
            high_distance = middle
        else:
            low_distance = middle
    return high_distance * 1.04


def camera_at_phase(center, radius, fit_points, reference_center, phase, width, height):
    base_forward = center - reference_center
    base_forward /= max(float(np.linalg.norm(base_forward)), 1.0e-6)
    up = np.asarray([0.0, -1.0, 0.0], dtype=np.float32)
    right = np.cross(base_forward, up)
    right /= max(float(np.linalg.norm(right)), 1.0e-6)
    oscillation = math.sin(2.0 * math.pi * phase)
    yaw = math.radians(18.0 * oscillation)
    pitch = math.radians(5.0 * math.sin(4.0 * math.pi * phase))
    direction = rotate_axis(base_forward, up, yaw)
    direction = rotate_axis(direction, right, pitch)
    focal = 0.90 * min(width, height)
    fit_distance = fitted_distance(
        center, direction, fit_points, focal, width, height, radius
    )
    position = center - direction * fit_distance
    return (
        position.astype(np.float32),
        center,
        focal,
        math.degrees(yaw),
        math.degrees(pitch),
        fit_distance,
    )


def rasterize(points, colors, center, radius, fit_points, reference_center, phase, args):
    position, target, focal, yaw, pitch, fit_distance = camera_at_phase(
        center, radius, fit_points, reference_center, phase, args.width, args.height
    )
    forward, right, up = look_at(position, target)
    relative = points - position[None]
    x, y, z = relative @ right, relative @ up, relative @ forward
    u = focal * x / np.maximum(z, 1.0e-6) + args.width * 0.5
    v = args.height * 0.5 - focal * y / np.maximum(z, 1.0e-6)
    valid = (z > 1.0e-5) & np.isfinite(u) & np.isfinite(v)
    valid &= (u >= -args.point_radius) & (u < args.width + args.point_radius)
    valid &= (v >= -args.point_radius) & (v < args.height + args.point_radius)
    u = np.rint(u[valid]).astype(np.int32)
    v = np.rint(v[valid]).astype(np.int32)
    z = z[valid]
    visible_colors = colors[valid]
    order = np.argsort(z)[::-1]
    u, v, visible_colors = u[order], v[order], visible_colors[order]
    canvas = np.broadcast_to(BACKGROUND, (args.height, args.width, 3)).copy()
    occupancy = np.zeros((args.height, args.width), dtype=bool)
    for dy in range(-args.point_radius, args.point_radius + 1):
        for dx in range(-args.point_radius, args.point_radius + 1):
            px, py = u + dx, v + dy
            inside = (px >= 0) & (px < args.width) & (py >= 0) & (py < args.height)
            canvas[py[inside], px[inside]] = visible_colors[inside]
            occupancy[py[inside], px[inside]] = True
    return canvas, float(occupancy.mean()), {
        "position": position.tolist(),
        "target": target.tolist(),
        "yaw_degrees": yaw,
        "pitch_degrees": pitch,
        "fit_distance": fit_distance,
    }


def font(size, bold=False):
    name = "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf"
    path = Path("/usr/share/fonts/truetype/dejavu") / name
    return ImageFont.truetype(str(path), size) if path.is_file() else ImageFont.load_default()


def label(image, scene, pair_count, point_count):
    result = Image.fromarray(image)
    draw = ImageDraw.Draw(result)
    draw.text((30, 24), "Night3R multi-pair reconstruction", fill=TEXT, font=font(28, True))
    draw.text((30, 64), scene, fill=SUBTEXT, font=font(19))
    draw.text(
        (30, image.shape[0] - 42),
        f"{pair_count} pairs  |  {point_count:,} RGB points  |  raw geometry  |  GT-pose fusion",
        fill=SUBTEXT,
        font=font(17),
    )
    return result


def encode(frame_dir, output, fps):
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-loglevel",
            "error",
            "-framerate",
            str(fps),
            "-i",
            str(frame_dir / "frame_%04d.png"),
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


def render_scene(args, index, scene, records):
    scene_root = args.data_root / scene
    scene_dir = args.outdir / f"{index:02d}_{scene}"
    frame_dir = scene_dir / "frames"
    frame_dir.mkdir(parents=True, exist_ok=True)
    points, colors, cameras, intrinsic, pose_key, pair_records = fuse_scene(
        scene_root, records, args
    )
    center, radius, low, high = robust_bounds(points)
    reference_center = cameras[0] if len(cameras) else np.zeros(3, dtype=np.float32)
    fit_stride = max(1, math.ceil(len(points) / 20000))
    fit_points = points[::fit_stride]
    np.savez_compressed(
        scene_dir / "night3r_multipair_fullscene_raw_rgb.npz",
        points=points,
        colors=colors,
        camera_centers=cameras,
    )
    indices = [args.frames // 2] if args.preview_only else range(args.frames)
    occupancies, camera_path = [], []
    for frame_index in indices:
        phase = frame_index / max(args.frames - 1, 1)
        image, occupancy, camera = rasterize(
            points, colors, center, radius, fit_points, reference_center, phase, args
        )
        output = (
            scene_dir / "preview_middle.png"
            if args.preview_only
            else frame_dir / f"frame_{frame_index:04d}.png"
        )
        label(image, scene, len(pair_records), len(points)).save(output, compress_level=2)
        occupancies.append(occupancy)
        camera_path.append(camera)
    video = scene_dir / "night3r_multipair_fullscene_rgb.mp4"
    if not args.preview_only:
        encode(frame_dir, video, args.fps)
        Image.open(frame_dir / f"frame_{args.frames // 2:04d}.png").save(
            scene_dir / "preview_middle.png", compress_level=2
        )
    metadata = {
        "dataset": "MVSEC",
        "scene": scene,
        "pair_count": len(pair_records),
        "view_count_nominal": 2 * len(pair_records),
        "pairs": pair_records,
        "pose_source": f"meta/poses_selected.npz:{pose_key}",
        "pose_policy": "dataset camera-to-world poses; first selected camera is the reference frame",
        "depth_alignment": "none; raw Night3R point-map coordinates",
        "depth_crop": "no camera-local depth crop; only 0.2/99.8 per-pair depth outlier rejection and 99.7% fused-radius rejection",
        "coloring": "clean RGB view-1 projection with display-only gamma 0.65; no depth colormap",
        "fusion": "rigid pose transform, direct merge, voxel downsample",
        "voxel_size": args.voxel_size,
        "point_count": int(len(points)),
        "bounds_0p5_99p5": {"min": low.tolist(), "max": high.tolist()},
        "scene_center": center.tolist(),
        "scene_radius": radius,
        "camera_centers": cameras.tolist(),
        "intrinsics_512": intrinsic.tolist(),
        "render_frames": args.frames,
        "fps": args.fps,
        "resolution": [args.width, args.height],
        "occupancy_min": min(occupancies),
        "occupancy_max": max(occupancies),
        "camera_path": camera_path,
        "video": None if args.preview_only else video.name,
    }
    (scene_dir / "render_metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
    return scene_dir, metadata


def contact_sheet(scene_dirs, output):
    tiles = [
        Image.open(path / "preview_middle.png")
        .convert("RGB")
        .resize((400, 225), Image.Resampling.LANCZOS)
        for path in scene_dirs
    ]
    cols, rows = 5, math.ceil(len(tiles) / 5)
    sheet = Image.new("RGB", (cols * 400, rows * 225), tuple(BACKGROUND.tolist()))
    for index, tile in enumerate(tiles):
        sheet.paste(tile, ((index % cols) * 400, (index // cols) * 225))
    sheet.save(output, compress_level=2)


def reel(scene_dirs, output):
    concat = output.with_suffix(".concat.txt")
    concat.write_text(
        "\n".join(
            f"file '{(path / 'night3r_multipair_fullscene_rgb.mp4').resolve()}'"
            for path in scene_dirs
        )
        + "\n",
        encoding="utf-8",
    )
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
            str(concat),
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
    args.outdir.mkdir(parents=True, exist_ok=True)
    selected = select_scenes(args.source_root, args.scenes, args.pairs_per_scene)
    scene_dirs, metadata = [], []
    for index, (_, scene, records) in enumerate(selected, 1):
        scene_dir, scene_metadata = render_scene(args, index, scene, records)
        scene_dirs.append(scene_dir)
        metadata.append(scene_metadata)
        print(f"[{index}/{len(selected)}] {scene}: {scene_metadata['point_count']} points", flush=True)
    contact_sheet(scene_dirs, args.outdir / "overview_multipair_fullscene_rgb.png")
    reel_path = args.outdir / "night3r_multipair_fullscene_10scene_reel.mp4"
    if not args.preview_only:
        reel(scene_dirs, reel_path)
    manifest = {
        "scene_count": len(metadata),
        "pair_count_per_scene": args.pairs_per_scene,
        "candidate_source_roots": [str(path) for path in args.source_root],
        "depth_alignment": "none",
        "pose_policy": "GT camera-to-world rigid fusion",
        "coloring": "direct RGB projection",
        "preview_only": args.preview_only,
        "reel": None if args.preview_only else reel_path.name,
        "records": metadata,
    }
    (args.outdir / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
