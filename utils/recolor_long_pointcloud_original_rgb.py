#!/usr/bin/env python3
"""Recolor an exported long-scene cloud from the registered original RGB frames.

Point positions and trajectories are left untouched.  Each point receives the
original sensor RGB value from the valid trajectory view where it projects
closest to the optical center, matching the short-scene display convention.
"""

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from utils.export_multimethod_scene_media import write_ply
from utils.render_farbaseline_follow_video import load_camera, load_ply


def recolor(points, fallback, poses, images, intrinsics):
    h, w = images[0].shape[:2]
    best = np.full(len(points), np.inf, dtype=np.float32)
    colors = fallback.copy()
    fx, fy, cx, cy = intrinsics[0, 0], intrinsics[1, 1], intrinsics[0, 2], intrinsics[1, 2]
    for pose, image in zip(poses, images):
        local = (points - pose[:3, 3]) @ pose[:3, :3]
        z = local[:, 2]
        safe_z = np.maximum(z, 1.0e-6)
        u = fx * local[:, 0] / safe_z + cx
        v = fy * local[:, 1] / safe_z + cy
        valid = (z > 0.15) & (u >= 0) & (u < w) & (v >= 0) & (v < h)
        score = np.abs((u - cx) / max(fx, 1.0)) + np.abs((v - cy) / max(fy, 1.0))
        update = valid & (score < best)
        if np.any(update):
            uu = np.clip(np.rint(u[update]).astype(np.int32), 0, w - 1)
            vv = np.clip(np.rint(v[update]).astype(np.int32), 0, h - 1)
            colors[update] = image[vv, uu]
            best[update] = score[update]
    return colors


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene-dir", type=Path, required=True)
    parser.add_argument("--scene", required=True)
    parser.add_argument("--image-dir", type=Path, required=True)
    parser.add_argument("--methods", nargs="+", default=("mast3r", "dark3r", "vggt", "night3r"))
    args = parser.parse_args()

    cameras = args.scene_dir / "shared_inputs" / "cameras.yaml"
    intrinsics, _, _ = load_camera(cameras)
    image_dir = args.image_dir
    if not image_dir.is_dir():
        raise FileNotFoundError(f"Missing input image directory: {image_dir}")
    image_paths = sorted(image_dir.glob("*.png"))
    images = [cv2.cvtColor(cv2.imread(str(path), cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB) for path in image_paths]
    if not images or any(image is None for image in images):
        raise RuntimeError("Could not load long-scene RGB frames")

    for method in args.methods:
        method_dir = args.scene_dir / method
        points, fallback = load_ply(method_dir / "pointcloud.ply")
        records = json.loads((method_dir / "trajectory.json").read_text(encoding="utf-8"))
        poses = np.asarray([record["camera_to_world"] for record in records], dtype=np.float32)
        count = min(len(poses), len(images))
        colors = recolor(points, fallback, poses[:count], images[:count], intrinsics)
        display_points = points * np.asarray([1.0, -1.0, 1.0], dtype=np.float32)
        write_ply(method_dir / "pointcloud.ply", display_points, colors)
        metadata_path = method_dir / "method.json"
        try:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        except (FileNotFoundError, json.JSONDecodeError):
            metadata = {"method": method}
        metadata["pointcloud_color"] = "original sensor RGB images projected through the predicted pose"
        metadata["color_display"] = "direct sRGB; no display-only brightness exponent"
        metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        print(method, len(points), flush=True)


if __name__ == "__main__":
    main()
