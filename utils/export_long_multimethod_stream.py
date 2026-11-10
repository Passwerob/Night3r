#!/usr/bin/env python3
"""Export 250-pair streaming comparisons for the long MVSEC display scene."""

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from utils.export_multimethod_scene_media import (  # noqa: E402
    build_method,
    clean_rgb,
    depth_frames,
    device_of,
    lowlight_batch,
    make_dataset,
    move_batch_to_device,
    native_depth,
    points_for_view,
    pred_conf,
    pred_points,
    register_dense_shared_frame,
    register_shared_frame,
    resolve_amp_config,
    save_video,
    seed_all,
    trajectory_video,
    voxel_downsample,
    write_ply,
    write_trajectory,
    align_depths_to_gt,
    vggt_camera_poses,
    preprocess_vggt,
    to_numpy,
)
from train_event_dark3r import load_config  # noqa: E402
from model.event_dark3r_student import make_view_pair_from_batch  # noqa: E402
from utils.render_farbaseline_follow_video import load_camera  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene", required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--frame-indices", required=True)
    parser.add_argument("--outdir", type=Path, required=True)
    parser.add_argument(
        "--input-stream-dir",
        type=Path,
        default=Path("outputs/night3r_long_250pairs_outdoor_night2_20260731/shared_inputs"),
    )
    parser.add_argument("--config", default="config/eval_night3r_da3_ft1k_sweep_20260722.yaml")
    parser.add_argument("--model-name-or-path", default="checkpoints/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric.pth")
    parser.add_argument("--dark3r-lora-checkpoint", default="checkpoints/dark3r_pretrained/weights/0049_lora.pth")
    parser.add_argument("--vggt-root", default="/data/fcr/vggt")
    parser.add_argument("--vggt-checkpoint", default="/data/fcr/ckpt/model.pt")
    parser.add_argument("--allow-unsafe-checkpoint-load", action="store_true")
    parser.add_argument("--frames", type=int, default=150)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--max-points-per-view", type=int, default=3000)
    parser.add_argument("--point-conf-quantile", type=float, default=0.65)
    parser.add_argument("--exposure", type=float, default=0.30)
    parser.add_argument("--read-noise-std", type=float, default=0.01)
    parser.add_argument("--shot-noise-std", type=float, default=0.01)
    parser.add_argument("--lowlight-seed", type=int, default=20260716)
    parser.add_argument(
        "--methods",
        nargs="+",
        choices=("mast3r", "dark3r", "night3r"),
        default=("mast3r", "dark3r"),
    )
    return parser.parse_args()


def transform_world(points, transform):
    return points @ transform[:3, :3].T + transform[:3, 3]


def args_for_model(args, method):
    return SimpleNamespace(
        variant={
            "mast3r": "mast3r",
            "dark3r": "dark3r_lora",
            "night3r": "event_dark3r",
        }[method],
        checkpoint="outputs/night3r_final_step100_20260723/event_dark3r_night3r_final.pth",
        geometry_head_checkpoint=None,
        config=args.config,
        data_root=str(args.data_root),
        model_name_or_path=args.model_name_or_path,
        dark3r_lora_checkpoint=args.dark3r_lora_checkpoint if method == "dark3r" else None,
        allow_unsafe_checkpoint_load=args.allow_unsafe_checkpoint_load,
        restore_frozen_rgb=False,
        restore_frozen_rgb_scope="none",
        restore_frozen_rgb_alpha=1.0,
    )


def export_point_stream(args, method, batches, indices, cameras_yaml, shape, device):
    build_args = SimpleNamespace(
        model_name_or_path=args.model_name_or_path,
        night3r_checkpoint="outputs/night3r_final_step100_20260723/event_dark3r_night3r_final.pth",
        night3r_pose_head_checkpoint="outputs/night3r_final_step100_pose_adapt100_20260726/pose_heads/pose_head_step00000100.pth",
        night3r_pose_config="config/train_night3r_final_step100_pose_adapt100.yaml",
        config=args.config,
        data_root=str(args.data_root),
        dark3r_lora_checkpoint=args.dark3r_lora_checkpoint,
        allow_unsafe_checkpoint_load=args.allow_unsafe_checkpoint_load,
        image_size=512,
        device="cuda",
        amp_dtype="bf16",
    )
    model, load_info = build_method(build_args, method, device)
    cfg = dict(load_config(args.config))
    amp_enabled, _, amp_dtype = resolve_amp_config(
        cfg, SimpleNamespace(amp=True, amp_dtype="bf16"), device
    )
    method_dir = args.outdir / method
    method_dir.mkdir(parents=True, exist_ok=True)
    all_points, all_colors, all_ids = [], [], []
    depths = [None] * len(indices)
    poses = []
    # Color the long-stream point clouds with the same darkened observations
    # seen by the model, so method geometry differences remain visible under
    # the controlled low-light input.
    rgb_sequence = [clean_rgb(batches[0]["images_photo_dark"][0, 0])]
    rgb_sequence += [clean_rgb(batch["images_photo_dark"][0, 1]) for batch in batches]
    previous_second = None
    pair_to_world = np.eye(4, dtype=np.float64)
    pose_diags = []
    with torch.no_grad():
        for pair_index, batch in enumerate(batches):
            gpu = move_batch_to_device(batch, device)
            if method in ("mast3r", "dark3r"):
                with torch.amp.autocast(device_type="cuda", dtype=amp_dtype, enabled=bool(amp_enabled)):
                    view1, view2 = make_view_pair_from_batch(gpu, prefer_dark=True)
                    pred1, pred2 = model(view1, view2)
            elif method == "night3r":
                with torch.amp.autocast(device_type="cuda", dtype=amp_dtype, enabled=bool(amp_enabled)):
                    pred1, pred2 = model(gpu)
            else:
                raise ValueError(method)
            first = pred_points(pred1)
            first_conf = pred_conf(pred1)
            second = pred_points(pred2, other=True)
            second_conf = pred_conf(pred2, other=True)
            if previous_second is not None:
                try:
                    transform, diag = register_shared_frame(
                        pred1["pts3d"][0], previous_second[0],
                        pred1["conf"][0], previous_second[1],
                        pred1["desc"], previous_second[2],
                        pred1.get("desc_conf"), previous_second[3],
                        SimpleNamespace(min_depth=0.15, max_depth=100.0, conf_quantile=0.35, seed=17),
                        pair_index,
                    )
                except RuntimeError as error:
                    transform, diag = register_dense_shared_frame(
                        pred1["pts3d"][0], previous_second[0],
                        pred1["conf"][0], previous_second[1], pair_index,
                    )
                    diag["descriptor_failure"] = str(error)
                if method == "night3r":
                    # Per-pair Night3R point maps have an arbitrary scale.
                    # Accumulating those Sim(3) scales over a fast 128-pair
                    # sequence collapses the entire display trajectory.  Keep
                    # the predicted rotation/translation registration but make
                    # this local-fusion chain rigid.
                    scale = float(diag.get("scale", diag.get("sim3_scale", 1.0)))
                    if np.isfinite(scale) and scale > 1.0e-6:
                        transform[:3, :3] /= scale
                        diag["display_sim3_scale_removed"] = scale
                pair_to_world = pair_to_world @ transform
                pose_diags.append(diag)
            else:
                pose_diags.append({"pair_index": 0, "pose_source": "first-camera gauge"})
            depths[pair_index] = native_depth(pred1)
            pts, col, ids = points_for_view(
                first, first_conf, resize_rgb(rgb_sequence[pair_index], first.shape[:2]),
                pair_to_world, pair_index, args, depth=first[..., 2],
            )
            all_points.append(pts); all_colors.append(col); all_ids.append(ids)
            if pair_index == len(batches) - 1:
                depths[pair_index + 1] = native_depth(pred2, other=True)
                pts, col, ids = points_for_view(
                    second, second_conf, resize_rgb(rgb_sequence[pair_index + 1], second.shape[:2]),
                    pair_to_world, pair_index + 1, args, depth=second[..., 2],
                )
                all_points.append(pts); all_colors.append(col); all_ids.append(ids)
            previous_second = (
                pred2["pts3d_in_other_view"][0].detach(),
                pred2["conf"][0].detach(),
                pred2["desc"].detach(),
                None if pred2.get("desc_conf") is None else pred2["desc_conf"].detach(),
            )
            poses.append(pair_to_world.copy())
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    poses.append(pair_to_world.copy())
    return all_points, all_colors, all_ids, depths, poses, load_info, pose_diags


def resize_rgb(image, hw):
    return cv2.resize(image, (hw[1], hw[0]), interpolation=cv2.INTER_AREA)


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
    if len(batches) != len(indices) - 1:
        raise RuntimeError(f"Expected {len(indices)-1} batches, got {len(batches)}")
    scene_dir = next(item for item in dataset.scene_frames if item.name == args.scene)
    cameras_yaml = scene_dir / "cameras.yaml"
    _, width, height = load_camera(cameras_yaml)
    shape = (height, width)
    device = device_of("cuda")
    shared = args.outdir / "shared_inputs"
    shared.mkdir(exist_ok=True)
    shutil.copy2(cameras_yaml, shared / "cameras.yaml")
    for stream_name in ("event_stream.mp4", "rgb_stream.mp4"):
        shutil.copy2(args.input_stream_dir / stream_name, shared / stream_name)
    for method in args.methods:
        print(f"[long-stream] {method}", flush=True)
        points, colors, ids, depths, poses, info, diags = export_point_stream(
            args, method, batches, indices, cameras_yaml, shape, device
        )
        points = np.concatenate(points); colors = np.concatenate(colors); ids = np.concatenate(ids)
        span = np.ptp(np.percentile(points, [1, 99], axis=0), axis=0)
        points, colors, ids = voxel_downsample(points, colors, ids, max(float(np.linalg.norm(span)) * 0.0015, 1e-5))
        display_points = points * np.asarray([1, -1, 1], np.float32)
        write_ply(args.outdir / method / "pointcloud.ply", display_points, colors)
        palette = cv2.applyColorMap(
            np.linspace(0, 255, len(indices), dtype=np.uint8).reshape(-1, 1),
            cv2.COLORMAP_TURBO,
        )[:, 0, ::-1]
        write_ply(args.outdir / method / "pointcloud_temporal.ply", display_points, palette[np.clip(ids, 0, len(indices) - 1)])
        write_trajectory(args.outdir / method / "trajectory.json", poses)
        aligned, _ = align_depths_to_gt(depths, [None] * len(depths))
        save_video(depth_frames(aligned, shape, args.frames)[0], args.outdir / method / "depth.mp4", args.fps)
        trajectory_video(poses, args.outdir / method / "camera_trajectory.mp4", args.frames, args.fps)
        path_length = float(np.linalg.norm(np.diff(np.asarray([p[:3, 3] for p in poses]), axis=0), axis=1).sum())
        render_cmd = [
            sys.executable, str(REPO_ROOT / "utils/render_farbaseline_follow_video.py"),
            "--ply", str(args.outdir / method / "pointcloud.ply"),
            "--trajectory", str(args.outdir / method / "trajectory.json"),
            "--cameras-yaml", str(cameras_yaml), "--scene", args.scene,
            "--views", str(len(indices)), "--pairs", str(len(indices) - 1),
            "--path-length", str(path_length), "--outdir", str(args.outdir / method),
            "--frames", str(args.frames), "--fps", str(args.fps),
            "--point-radius", "1", "--color-exposure", "1.0",
            "--temporal-ply", str(args.outdir / method / "pointcloud_temporal.ply"),
            "--temporal-view-count", str(len(indices)), "--local-window", "8", "--temporal-blend",
        ]
        subprocess.run(render_cmd, check=True)
        os_path = args.outdir / method / "night3r_farbaseline_camera_follow.mp4"
        os_path.replace(args.outdir / method / "pointcloud.mp4")
        shutil.copy2(shared / "event_stream.mp4", args.outdir / method / "event_stream.mp4")
        shutil.copy2(shared / "rgb_stream.mp4", args.outdir / method / "rgb_stream.mp4")
        (args.outdir / method / "method.json").write_text(json.dumps({"method": method, "inference": "250-pair sequential shared-view Sim(3) stream", "pose_source": "method-predicted shared-view chain", "pair_count": len(indices)-1, "load_info": info, "pose_diagnostics": diags}, indent=2))
    (args.outdir / "manifest.json").write_text(json.dumps({"scene": args.scene, "pair_count": len(indices)-1, "fps": args.fps, "frames": args.frames, "methods": list(args.methods), "rgb_stream": "shared original RGB", "event_stream": "shared 4-bin temporal expansion"}, indent=2))


if __name__ == "__main__":
    main()
