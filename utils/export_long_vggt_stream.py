#!/usr/bin/env python3
"""Streaming VGGT baseline for the 250-pair long-scene comparison."""

import argparse
import json
import cv2
import shutil
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from model.event_dark3r_student import make_view_pair_from_batch  # noqa: E402
from train_event_dark3r import load_config, move_batch_to_device, resolve_amp_config  # noqa: E402
from utils.export_long_multimethod_stream import resize_rgb  # noqa: E402
from utils.export_multimethod_scene_media import (  # noqa: E402
    align_depths_to_gt,
    build_method,
    clean_rgb,
    depth_frames,
    device_of,
    lowlight_batch,
    make_dataset,
    points_for_view,
    register_dense_shared_frame,
    save_video,
    seed_all,
    to_numpy,
    trajectory_video,
    voxel_downsample,
    vggt_camera_poses,
    write_ply,
    write_trajectory,
)
from utils.render_farbaseline_follow_video import load_camera  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene", required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--frame-indices", required=True)
    parser.add_argument("--outdir", type=Path, required=True)
    parser.add_argument("--config", default="config/eval_night3r_da3_ft1k_sweep_20260722.yaml")
    parser.add_argument("--model-name-or-path", default="checkpoints/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric.pth")
    parser.add_argument("--vggt-root", default="/data/fcr/vggt")
    parser.add_argument("--vggt-checkpoint", default="/data/fcr/ckpt/model.pt")
    parser.add_argument("--allow-unsafe-checkpoint-load", action="store_true")
    parser.add_argument("--frames", type=int, default=150)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--max-points-per-view", type=int, default=3000)
    parser.add_argument("--point-conf-quantile", type=float, default=0.65)
    parser.add_argument("--read-noise-std", type=float, default=0.01)
    parser.add_argument("--shot-noise-std", type=float, default=0.01)
    parser.add_argument("--lowlight-seed", type=int, default=20260716)
    parser.add_argument("--exposure", type=float, default=0.30)
    return parser.parse_args()


def main():
    args = parse_args()
    args.image_size = 512
    seed_all(17)
    indices = [int(value) for value in args.frame_indices.split(",")]
    args.outdir.mkdir(parents=True, exist_ok=True)
    pair_file = args.outdir / "sequence_pairs.json"
    pair_file.write_text(json.dumps([{"idx1": a, "idx2": b} for a, b in zip(indices, indices[1:])], indent=2))
    dataset, _ = make_dataset(args, pair_file)
    batches = [lowlight_batch(batch, args) for batch in torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)]
    scene_dir = next(item for item in dataset.scene_frames if item.name == args.scene)
    cameras_yaml = scene_dir / "cameras.yaml"
    _, width, height = load_camera(cameras_yaml)
    shape = (height, width)
    device = device_of("cuda")
    shared = args.outdir / "shared_inputs"
    shared.mkdir(exist_ok=True)
    source_shared = Path("outputs/night3r_long_250pairs_outdoor_night2_20260731/shared_inputs")
    for name in ("event_stream.mp4", "rgb_stream.mp4"):
        shutil.copy2(source_shared / name, shared / name)
    shutil.copy2(cameras_yaml, shared / "cameras.yaml")

    build_args = SimpleNamespace(
        vggt_root=args.vggt_root,
        vggt_checkpoint=args.vggt_checkpoint,
    )
    from utils.export_multimethod_scene_media import load_vggt, preprocess_vggt
    model, load_info = load_vggt(build_args, device)
    points_all, colors_all, ids_all = [], [], []
    depths = [None] * len(indices)
    poses = []
    pair_to_world = np.eye(4, dtype=np.float64)
    previous_second = None
    rgb_sequence = [clean_rgb(batches[0]["images_photo_dark"][0, 0])]
    rgb_sequence += [clean_rgb(batch["images_photo_dark"][0, 1]) for batch in batches]
    with torch.no_grad():
        for pair_index, batch in enumerate(batches):
            gpu = move_batch_to_device(batch, device)
            images = torch.stack([gpu["images_photo_dark"][0, 0], gpu["images_photo_dark"][0, 1]], 0).unsqueeze(0)
            pred = model(images=preprocess_vggt(images))
            vd = pred["depth"][0, ..., 0].float().cpu().numpy()
            vc = pred["depth_conf"][0].float().cpu().numpy()
            from vggt.utils.geometry import unproject_depth_map_to_point_map
            from vggt.utils.pose_enc import pose_encoding_to_extri_intri
            extrinsics, intrinsics = pose_encoding_to_extri_intri(pred["pose_enc"].float(), image_size_hw=(518, 518), build_intrinsics=True)
            pair_points = to_numpy(unproject_depth_map_to_point_map(vd[..., None], extrinsics[0], intrinsics[0]))
            pair_poses = vggt_camera_poses(pred["pose_enc"])
            if previous_second is not None:
                transform, _ = register_dense_shared_frame(pair_points[0], previous_second[0], vc[0], previous_second[1], pair_index)
                pair_to_world = pair_to_world @ transform
            depths[pair_index] = vd[0]
            world_first = pair_points[0] @ pair_to_world[:3, :3].T + pair_to_world[:3, 3]
            pts, col, ids = points_for_view(world_first, vc[0], resize_rgb(rgb_sequence[pair_index], world_first.shape[:2]), None, pair_index, args, points_are_world=True, depth=vd[0])
            points_all.append(pts); colors_all.append(col); ids_all.append(ids)
            poses.append(pair_to_world @ pair_poses[0])
            previous_second = (pair_points[1], vc[1])
            if pair_index == len(batches) - 1:
                depths[pair_index + 1] = vd[1]
                world_second = pair_points[1] @ pair_to_world[:3, :3].T + pair_to_world[:3, 3]
                pts, col, ids = points_for_view(world_second, vc[1], resize_rgb(rgb_sequence[pair_index + 1], world_second.shape[:2]), None, pair_index + 1, args, points_are_world=True, depth=vd[1])
                points_all.append(pts); colors_all.append(col); ids_all.append(ids)
                poses.append(pair_to_world @ pair_poses[1])
    del model
    if device.type == "cuda": torch.cuda.empty_cache()
    method_dir = args.outdir / "vggt"
    method_dir.mkdir(parents=True, exist_ok=True)
    points = np.concatenate(points_all); colors = np.concatenate(colors_all); ids = np.concatenate(ids_all)
    span = np.ptp(np.percentile(points, [1, 99], axis=0), axis=0)
    points, colors, ids = voxel_downsample(points, colors, ids, max(float(np.linalg.norm(span)) * 0.0015, 1e-5))
    display_points = points * np.asarray([1, -1, 1], np.float32)
    write_ply(method_dir / "pointcloud.ply", display_points, colors)
    palette = cv2.applyColorMap(np.linspace(0, 255, len(indices), dtype=np.uint8).reshape(-1, 1), cv2.COLORMAP_TURBO)[:, 0, ::-1]
    write_ply(method_dir / "pointcloud_temporal.ply", display_points, palette[np.clip(ids, 0, len(indices) - 1)])
    write_trajectory(method_dir / "trajectory.json", poses)
    save_video(depth_frames(depths, shape, args.frames)[0], method_dir / "depth.mp4", args.fps)
    trajectory_video(poses, method_dir / "camera_trajectory.mp4", args.frames, args.fps)
    path_length = float(np.linalg.norm(np.diff(np.asarray([p[:3, 3] for p in poses]), axis=0), axis=1).sum())
    render_cmd = [sys.executable, str(REPO_ROOT / "utils/render_farbaseline_follow_video.py"), "--ply", str(method_dir / "pointcloud.ply"), "--trajectory", str(method_dir / "trajectory.json"), "--cameras-yaml", str(cameras_yaml), "--scene", args.scene, "--views", str(len(indices)), "--pairs", str(len(indices) - 1), "--path-length", str(path_length), "--outdir", str(method_dir), "--frames", str(args.frames), "--fps", str(args.fps), "--point-radius", "1", "--color-exposure", "1.0", "--temporal-ply", str(method_dir / "pointcloud_temporal.ply"), "--temporal-view-count", str(len(indices)), "--local-window", "8", "--temporal-blend"]
    subprocess.run(render_cmd, check=True)
    (method_dir / "night3r_farbaseline_camera_follow.mp4").replace(method_dir / "pointcloud.mp4")
    shutil.copy2(shared / "event_stream.mp4", method_dir / "event_stream.mp4")
    shutil.copy2(shared / "rgb_stream.mp4", method_dir / "rgb_stream.mp4")
    (method_dir / "method.json").write_text(json.dumps({"method": "vggt", "inference": "250-pair sequential official depth+camera stream", "pose_source": "official VGGT camera head", "pair_count": len(indices) - 1, "load_info": load_info}, indent=2))


if __name__ == "__main__":
    main()
