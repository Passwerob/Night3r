#!/usr/bin/env python3
"""Build same-view 3D previews and a browser manifest for pair review."""

import argparse
import json
import math
import os
import re
import shutil
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]
REVIEW_UI_ROOT = REPO_ROOT / "web/pair_review"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from dataset.dsec_event_dataset import DSECEventDataset  # noqa: E402
from utils.export_night3r_pair_pointcloud import render_points  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser("Build Night3R pair-review assets")
    parser.add_argument(
        "--sweep-root",
        type=Path,
        default=REPO_ROOT / "outputs/all_test_visual_sweep_final1k_20260721",
    )
    parser.add_argument("--workers", type=int, default=min(12, os.cpu_count() or 4))
    parser.add_argument("--render-width", type=int, default=720)
    parser.add_argument("--render-height", type=int, default=540)
    parser.add_argument("--point-radius", type=int, default=1)
    parser.add_argument("--stride", type=int, default=2)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def safe_name(value):
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value)).strip("_")


def finite_number(value):
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def dataset_intrinsics():
    roots = {
        "DSEC": REPO_ROOT / "test/DSEC",
        "MVSEC": REPO_ROOT / "test/MVSEC",
        "DL3DV": REPO_ROOT / "test/DL3DV",
    }
    result = {}
    for dataset_name, root in roots.items():
        dataset = DSECEventDataset(
            root_path=str(root),
            image_size=(512, 512),
            pair_stride=1,
            pair_sampling="fixed",
            expected_event_channels=8,
            lowlight_enabled=False,
        )
        by_scene = {}
        for scene_dir, intrinsics in dataset.scene_camera.items():
            if intrinsics is None or not np.isfinite(intrinsics.numpy()).all():
                continue
            by_scene[scene_dir.name] = intrinsics.numpy().astype(np.float32).tolist()
        result[dataset_name] = by_scene
    return result


def backproject(depth, intrinsics, stride):
    depth = np.asarray(depth, dtype=np.float32)
    h, w = depth.shape
    yy, xx = np.mgrid[0:h:stride, 0:w:stride]
    z = depth[::stride, ::stride]
    valid = np.isfinite(z) & (z > 1.0e-6)
    x = xx[valid].astype(np.float32)
    y = yy[valid].astype(np.float32)
    z = z[valid]
    fx, fy = float(intrinsics[0, 0]), float(intrinsics[1, 1])
    cx, cy = float(intrinsics[0, 2]), float(intrinsics[1, 2])
    return np.stack(((x - cx) * z / fx, (y - cy) * z / fy, z), axis=1)


def robust_volume(points):
    lower, upper = np.quantile(points, (0.01, 0.99), axis=0)
    margin = np.maximum((upper - lower) * 0.10, 1.0e-4)
    return lower - margin, upper + margin


def crop_points(points, lower, upper):
    keep = np.isfinite(points).all(axis=1)
    keep &= np.all(points >= lower[None], axis=1)
    keep &= np.all(points <= upper[None], axis=1)
    return points[keep]


def depth_colors(points, lo, hi):
    value = np.clip((points[:, 2] - lo) / max(float(hi - lo), 1.0e-6), 0.0, 1.0)
    lut = cv2.applyColorMap(
        np.arange(256, dtype=np.uint8)[:, None], cv2.COLORMAP_VIRIDIS
    )
    lut = cv2.cvtColor(lut, cv2.COLOR_BGR2RGB)[:, 0]
    return lut[np.round(value * 255).astype(np.uint8)]


def white_background(array):
    array = np.asarray(array).copy()
    dark = np.all(array == np.asarray([8, 12, 18], dtype=np.uint8), axis=2)
    array[dark] = 255
    return array


def render_method(points, camera, center, lo, hi, width, height, radius):
    colors = depth_colors(points, lo, hi)
    rendered = render_points(points, colors, camera, center, width, height, radius)
    return white_background(rendered)


def fitted_camera(points, center, direction, width, height, margin=0.08):
    direction = np.asarray(direction, dtype=np.float32)
    direction /= max(float(np.linalg.norm(direction)), 1.0e-6)
    forward = -direction
    up_hint = np.asarray([0.0, -1.0, 0.0], dtype=np.float32)
    right = np.cross(forward, up_hint)
    right /= max(float(np.linalg.norm(right)), 1.0e-6)
    up = np.cross(right, forward)
    centered = points - center[None]
    local_x = centered @ right
    local_y = centered @ up
    local_z = centered @ forward
    extent = max(float(np.linalg.norm(np.quantile(points, 0.99, axis=0) - np.quantile(points, 0.01, axis=0))), 1.0e-3)
    focal = 0.5 * width / np.tan(np.deg2rad(52.0) * 0.5)

    def fits(distance):
        z = distance + local_z
        if np.quantile(z, 0.005) <= 1.0e-5:
            return False
        u = focal * local_x / z + width * 0.5
        v = height * 0.5 - focal * local_y / z
        u_lo, u_hi = np.quantile(u, (0.005, 0.995))
        v_lo, v_hi = np.quantile(v, (0.005, 0.995))
        return (
            u_lo >= width * margin
            and u_hi <= width * (1.0 - margin)
            and v_lo >= height * margin
            and v_hi <= height * (1.0 - margin)
        )

    low, high = 0.05 * extent, 5.0 * extent
    for _ in range(36):
        middle = 0.5 * (low + high)
        if fits(middle):
            high = middle
        else:
            low = middle
    return center + direction * high


def relative_url(path, root):
    return "/" + str(Path(path).resolve().relative_to(root.resolve()))


def method_metrics(record, method):
    metrics = record.get("metrics", {}).get(method, {})
    return {
        key: finite_number(metrics.get(key))
        for key in ("raw_absrel", "raw_delta1", "median_absrel", "median_delta1", "valid_px")
        if key in metrics
    }


def build_one(task):
    record = task["record"]
    root = Path(task["root"])
    geometry_dir = Path(task["geometry_dir"])
    geometry_dir.mkdir(parents=True, exist_ok=True)
    done_path = geometry_dir / "geometry_protocol.json"
    expected = [
        geometry_dir / "mast3r_front.png",
        geometry_dir / "mast3r_elevated.png",
        geometry_dir / "night3r_front.png",
        geometry_dir / "night3r_elevated.png",
    ]
    if record.get("metrics"):
        expected.extend((geometry_dir / "gt_front.png", geometry_dir / "gt_elevated.png"))
    if not task["force"] and done_path.is_file() and all(path.is_file() for path in expected):
        return json.loads(done_path.read_text(encoding="utf-8"))

    asset_dir = Path(record["asset_dir"])
    arrays = np.load(asset_dir / "geometry_arrays.npz")
    point_path = asset_dir / "geometry_points.npz"
    native_pointmaps_archived = point_path.is_file()
    has_gt = "gt" in arrays.files
    if has_gt:
        depth_arrays = {
            "mast3r": arrays["master_aligned"],
            "night3r": arrays["night_aligned"],
            "gt": arrays["gt"],
        }
        reference_name = "gt"
        scale_policy = (
            "View-1 Z-depth backprojection; each prediction is median-aligned "
            "to GT before applying the calibrated intrinsics"
        )
    else:
        depth_arrays = {
            "mast3r": arrays["master"],
            "night3r": arrays["night_aligned_to_master"],
        }
        reference_name = "mast3r"
        scale_policy = (
            "View-1 Z-depth backprojection; Night3R is median-aligned to "
            "MASt3R because no GT is available"
        )
    intrinsics = np.asarray(task["intrinsics"], dtype=np.float32)
    points = {
        name: backproject(depth, intrinsics, task["stride"])
        for name, depth in depth_arrays.items()
    }
    reference = points[reference_name]
    if len(reference) < 256:
        raise RuntimeError(f"Insufficient points for {record['sample_id']}")
    lower, upper = robust_volume(reference)
    points = {name: crop_points(value, lower, upper) for name, value in points.items()}
    reference = points[reference_name]
    depth_lo, depth_hi = np.quantile(reference[:, 2], (0.02, 0.98))
    center = 0.5 * (
        np.quantile(reference, 0.01, axis=0) + np.quantile(reference, 0.99, axis=0)
    )
    front_camera = fitted_camera(
        reference, center, [0.0, 0.0, -1.0], task["width"], task["height"]
    )
    elevated_camera = fitted_camera(
        reference,
        center,
        [0.36, -0.34, -1.10],
        task["width"],
        task["height"],
    )
    cameras = {"front": front_camera, "elevated": elevated_camera}
    output_urls = {}
    for method, method_points in points.items():
        output_urls[method] = {}
        for view, camera in cameras.items():
            rendered = render_method(
                method_points,
                camera,
                center,
                depth_lo,
                depth_hi,
                task["width"],
                task["height"],
                task["radius"],
            )
            path = geometry_dir / f"{method}_{view}.png"
            Image.fromarray(rendered).save(path, optimize=True)
            output_urls[method][view] = relative_url(path, root)

    protocol = {
        "id": task["id"],
        "sample_id": record["sample_id"],
        "dataset": record["dataset"],
        "scene": record["scene"],
        "has_gt": has_gt,
        "coordinate_frame": "view-1 camera frame from calibrated depth backprojection",
        "geometry_source": "view-1 predicted Z-depth plus dataset intrinsics",
        "native_pointmaps_archived": native_pointmaps_archived,
        "intrinsics": task["intrinsics"],
        "scale_policy": scale_policy,
        "crop_policy": "shared 1%-99% reference point-cloud volume with 10% margin",
        "camera_policy": "identical front and elevated cameras for all methods in the pair",
        "depth_color_range": [float(depth_lo), float(depth_hi)],
        "point_stride": int(task["stride"]),
        "confidence_quantile": None,
        "point_counts": {name: int(len(value)) for name, value in points.items()},
        "camera_center": center.tolist(),
        "front_camera": front_camera.tolist(),
        "elevated_camera": elevated_camera.tolist(),
        "assets": output_urls,
    }
    done_path.write_text(json.dumps(protocol, indent=2), encoding="utf-8")
    return protocol


def record_to_review(record, protocol, root):
    asset_dir = Path(record["asset_dir"])
    item_id = protocol["id"]
    has_gt = protocol["has_gt"]
    depth_fourth_label = "GT depth" if has_gt else "Model log-depth difference"
    return {
        "id": item_id,
        "dataset": record["dataset"],
        "scene": record["scene"],
        "sample_id": record["sample_id"],
        "idx1": record["idx1"],
        "idx2": record["idx2"],
        "scene_rank": record["scene_rank"],
        "score": finite_number(record.get("score")),
        "score_kind": record.get("score_kind"),
        "split_label": record.get("split_label"),
        "has_gt": has_gt,
        "warning": record.get("warning"),
        "event_gate_mean": finite_number(record.get("event_gate_mean")),
        "event_occupancy": finite_number(record.get("event_occupancy")),
        "event_activity_mean": finite_number(record.get("event_activity_mean")),
        "lowlight_mean": finite_number(record.get("lowlight_mean")),
        "lowlight_noise": finite_number(record.get("lowlight_noise")),
        "metrics": {
            "mast3r": method_metrics(record, "master"),
            "night3r": method_metrics(record, "night"),
            "zero": method_metrics(record, "zero"),
            "mast3r_vs_night3r": record.get("mast3r_vs_night3r", {}),
            "night3r_real_vs_zero": record.get("night3r_real_vs_zero", {}),
        },
        "protocol": {
            key: protocol[key]
            for key in (
                "coordinate_frame", "geometry_source", "scale_policy", "crop_policy",
                "camera_policy", "depth_color_range", "point_stride", "point_counts",
            )
        },
        "source_asset_dir": str(asset_dir.resolve()),
        "geometry_asset_dir": str((root / "review_app/assets/geometry" / item_id).resolve()),
        "assets": {
            "pair": {
                "rgb1": relative_url(asset_dir / "lowlight_rgb_view1.png", root),
                "rgb2": relative_url(asset_dir / "lowlight_rgb_view2.png", root),
                "clean1": relative_url(asset_dir / "clean_rgb_view1.png", root),
                "clean2": relative_url(asset_dir / "clean_rgb_view2.png", root),
                "event1": relative_url(asset_dir / "event_view1_red_blue.png", root),
                "event2": relative_url(asset_dir / "event_view2_red_blue.png", root),
            },
            "depth": {
                "mast3r": relative_url(asset_dir / "master_depth_shared.png", root),
                "night3r": relative_url(asset_dir / "night_depth_shared.png", root),
                "zero": relative_url(asset_dir / "zero_depth_shared.png", root),
                "reference": relative_url(asset_dir / "gt_depth.png", root),
                "reference_label": depth_fourth_label,
            },
            "geometry": protocol["assets"],
            "overview": relative_url(asset_dir / "selection_panel.png", root),
        },
    }


def main():
    args = parse_args()
    root = args.sweep_root.resolve()
    manifest_path = root / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(manifest_path)
    sweep = json.loads(manifest_path.read_text(encoding="utf-8"))
    app_root = root / "review_app"
    app_root.mkdir(parents=True, exist_ok=True)
    for filename in ("index.html", "styles.css", "app.js", "README.md"):
        source = REVIEW_UI_ROOT / filename
        if not source.is_file():
            raise FileNotFoundError(f"Missing review UI source: {source}")
        shutil.copy2(source, app_root / filename)
    records = sweep.get("records", [])
    intrinsics = dataset_intrinsics()
    geometry_root = root / "review_app/assets/geometry"
    geometry_root.mkdir(parents=True, exist_ok=True)

    tasks = []
    for record in records:
        item_id = safe_name(f"{record['dataset']}_{record['sample_id']}")
        matrix = intrinsics.get(record["dataset"], {}).get(record["scene"])
        if matrix is None:
            raise RuntimeError(f"Missing intrinsics for {record['dataset']} / {record['scene']}")
        tasks.append(
            {
                "id": item_id,
                "record": record,
                "intrinsics": matrix,
                "root": str(root),
                "geometry_dir": str(geometry_root / item_id),
                "width": args.render_width,
                "height": args.render_height,
                "radius": args.point_radius,
                "stride": args.stride,
                "force": args.force,
            }
        )

    protocols = {}
    completed = 0
    with ProcessPoolExecutor(max_workers=max(1, args.workers)) as executor:
        futures = {executor.submit(build_one, task): task for task in tasks}
        for future in as_completed(futures):
            task = futures[future]
            protocol = future.result()
            protocols[protocol["id"]] = protocol
            completed += 1
            if completed % 10 == 0 or completed == len(tasks):
                print(json.dumps({"completed": completed, "total": len(tasks)}), flush=True)

    review_items = [
        record_to_review(record, protocols[safe_name(f"{record['dataset']}_{record['sample_id']}")], root)
        for record in records
    ]
    review_items.sort(key=lambda item: (item["dataset"], item["scene"], item["scene_rank"]))
    review_manifest = {
        "version": 1,
        "title": "Night3R Pair Review",
        "source_sweep": str(manifest_path),
        "total": len(review_items),
        "datasets": {
            name: sum(item["dataset"] == name for item in review_items)
            for name in ("DSEC", "MVSEC", "DL3DV")
        },
        "items": review_items,
    }
    output = app_root / "review_manifest.json"
    output.write_text(json.dumps(review_manifest, indent=2), encoding="utf-8")
    print(json.dumps({"output": str(output), "items": len(review_items)}, indent=2))


if __name__ == "__main__":
    main()
