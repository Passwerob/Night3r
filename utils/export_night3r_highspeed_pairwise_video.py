#!/usr/bin/env python3
"""Render stable high-speed Night3R geometry from native adjacent-pair maps.

Each frame fuses the two point maps predicted for one adjacent RGB/event pair.
Both maps are expressed in view-1 coordinates, so no long-horizon pose chain
is used for display.
"""

import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from model.event_dark3r_student import make_view_pair_from_batch
from train_event_dark3r import load_config
from utils.export_multimethod_scene_media import (
    build_method,
    clean_rgb,
    device_of,
    lowlight_batch,
    make_dataset,
    move_batch_to_device,
    pred_conf,
    pred_points,
    resolve_amp_config,
    save_video,
)
from utils.render_farbaseline_follow_video import load_camera


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene", required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--frame-indices", required=True)
    parser.add_argument("--outdir", type=Path, required=True)
    parser.add_argument("--config", default="config/eval_night3r_da3_ft1k_sweep_20260722.yaml")
    parser.add_argument("--model-name-or-path", default="checkpoints/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric.pth")
    parser.add_argument("--night3r-checkpoint", default="outputs/night3r_final_step100_20260723/event_dark3r_night3r_final.pth")
    parser.add_argument("--night3r-pose-head-checkpoint", default="outputs/night3r_final_step100_pose_adapt100_20260726/pose_heads/pose_head_step00000100.pth")
    parser.add_argument("--night3r-pose-config", default="config/train_night3r_final_step100_pose_adapt100.yaml")
    parser.add_argument("--frames", type=int, default=150)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--exposure", type=float, default=0.20)
    parser.add_argument("--read-noise-std", type=float, default=0.0)
    parser.add_argument("--shot-noise-std", type=float, default=0.0)
    parser.add_argument("--lowlight-seed", type=int, default=20260716)
    parser.add_argument("--max-points", type=int, default=12000)
    parser.add_argument("--conf-quantile", type=float, default=0.75)
    parser.add_argument("--allow-unsafe-checkpoint-load", action="store_true")
    return parser.parse_args()


def select_points(points, confidence, image, args, seed):
    valid = np.isfinite(points).all(axis=-1) & np.isfinite(confidence)
    valid &= (points[..., 2] > 0.15) & (points[..., 2] < 100.0)
    if np.any(valid):
        valid &= confidence >= np.quantile(confidence[valid], args.conf_quantile)
    selected = np.flatnonzero(valid.reshape(-1))
    if len(selected) > args.max_points:
        selected = np.random.default_rng(seed).choice(selected, args.max_points, replace=False)
    colors = cv2.resize(image, (points.shape[1], points.shape[0]), interpolation=cv2.INTER_AREA)
    return points.reshape(-1, 3)[selected], np.clip(colors.reshape(-1, 3)[selected] * 255.0 + 0.5, 0, 255).astype(np.uint8)


def render_camera_local(points, colors, width, height):
    valid = np.isfinite(points).all(axis=1) & (points[:, 2] > 0.15)
    points, colors = points[valid], colors[valid]
    canvas = np.full((height, width, 3), (11, 16, 23), dtype=np.uint8)
    if len(points) < 32:
        return canvas
    z = points[:, 2]
    rx, ry = points[:, 0] / z, points[:, 1] / z
    x_lo, x_hi = np.percentile(rx, [1.0, 99.0])
    y_lo, y_hi = np.percentile(ry, [1.0, 99.0])
    focal = min(0.92 * width / max(x_hi - x_lo, 1.0e-6), 0.92 * height / max(y_hi - y_lo, 1.0e-6))
    u = np.rint((rx - 0.5 * (x_lo + x_hi)) * focal + 0.5 * width).astype(np.int32)
    v = np.rint(0.5 * height - (ry - 0.5 * (y_lo + y_hi)) * focal).astype(np.int32)
    inside = (u >= 0) & (u < width) & (v >= 0) & (v < height)
    u, v, z, colors = u[inside], v[inside], z[inside], colors[inside]
    order = np.argsort(z)[::-1]
    for dy in range(-1, 2):
        for dx in range(-1, 2):
            px, py = u[order] + dx, v[order] + dy
            keep = (px >= 0) & (px < width) & (py >= 0) & (py < height)
            canvas[py[keep], px[keep]] = colors[order][keep]
    return canvas


def main():
    args = parse_args()
    args.image_size = 512
    args.device = "cuda"
    args.amp_dtype = "bf16"
    indices = [int(value) for value in args.frame_indices.split(",")]
    args.outdir.mkdir(parents=True, exist_ok=True)
    pair_file = args.outdir / "sequence_pairs.json"
    pair_file.write_text(json.dumps([{"idx1": a, "idx2": b} for a, b in zip(indices, indices[1:])]))
    dataset, _ = make_dataset(args, pair_file)
    batches = [lowlight_batch(batch, args) for batch in torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)]
    scene_dir = next(item for item in dataset.scene_frames if item.name == args.scene)
    _, width, height = load_camera(scene_dir / "cameras.yaml")
    device = device_of("cuda")
    build_args = SimpleNamespace(
        model_name_or_path=args.model_name_or_path,
        night3r_checkpoint=args.night3r_checkpoint,
        night3r_pose_head_checkpoint=args.night3r_pose_head_checkpoint,
        night3r_pose_config=args.night3r_pose_config,
        config=args.config,
        data_root=str(args.data_root),
        dark3r_lora_checkpoint=None,
        allow_unsafe_checkpoint_load=args.allow_unsafe_checkpoint_load,
        image_size=512,
        device="cuda",
        amp_dtype="bf16",
    )
    model, _ = build_method(build_args, "night3r", device)
    amp_enabled, _, amp_dtype = resolve_amp_config(dict(load_config(args.config)), SimpleNamespace(amp=True, amp_dtype="bf16"), device)
    pair_frames = []
    with torch.no_grad():
        for pair_index, batch in enumerate(batches):
            gpu = move_batch_to_device(batch, device)
            with torch.amp.autocast(device_type="cuda", dtype=amp_dtype, enabled=bool(amp_enabled)):
                pred1, pred2 = model(gpu)
            first = pred_points(pred1)
            second = pred_points(pred2, other=True)
            image1 = clean_rgb(batch["images_photo"][0, 0])
            image2 = clean_rgb(batch["images_photo"][0, 1])
            p1, c1 = select_points(first, pred_conf(pred1), image1, args, 17 + pair_index)
            p2, c2 = select_points(second, pred_conf(pred2, other=True), image2, args, 10017 + pair_index)
            pair_frames.append(render_camera_local(
                np.concatenate((p1, p2)), np.concatenate((c1, c2)), width, height
            ))
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    frames = []
    for output_index in range(args.frames):
        phase = output_index * (len(pair_frames) - 1) / max(args.frames - 1, 1)
        first = int(np.floor(phase))
        second = min(first + 1, len(pair_frames) - 1)
        amount = phase - first
        frames.append(cv2.addWeighted(pair_frames[first], 1.0 - amount, pair_frames[second], amount, 0.0))
    method_dir = args.outdir / "night3r"
    method_dir.mkdir(exist_ok=True)
    save_video(frames, method_dir / "pointcloud_pairwise_local.mp4", args.fps)
    (method_dir / "pairwise_local_manifest.json").write_text(json.dumps({
        "pair_count": len(batches), "fusion": "pred1 pts3d + pred2 pts3d_in_other_view in native view-1 coordinates",
        "pose_policy": "no long-horizon pose accumulation", "point_conf_quantile": args.conf_quantile,
    }, indent=2))


if __name__ == "__main__":
    main()
