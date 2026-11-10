#!/usr/bin/env python3
"""Export comparable depth/point/trajectory/event videos for one MVSEC scene.

Each point cloud and camera video uses that method's predicted camera poses.
Only the 2D depth panel is median-aligned to the measured depth for display.
"""

import argparse
import json
import os
import random
import shutil
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import torch
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from dataset.dsec_event_dataset import DSECEventDataset
from model.event_dark3r_student import make_view_pair_from_batch
from train_event_dark3r import (
    build_event_student,
    load_backbone,
    load_config,
    move_batch_to_device,
    resolve_amp_config,
    resolve_stage_schedule,
    safe_torch_load,
    set_event_training_stage,
)
from utils.eval_event_dark3r_variant import build_model as build_rgb_model
from utils.eval_night3r_trajectory import (
    apply_deterministic_lowlight,
    estimate_relative_pose_pointmap_sim3,
    load_pose_head_checkpoint,
    swap_pair_batch,
)
from utils.export_night3r_long_sequence import register_shared_frame
from utils.render_farbaseline_follow_video import load_camera


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--scene", default="screen-000095-mvsec")
    p.add_argument("--data-root", default="/data/fcr/data/train_data/MVSEC")
    p.add_argument("--frame-indices", default="0,7,14,21,28,35,42,48")
    p.add_argument("--config", default="config/eval_night3r_da3_ft1k_sweep_20260722.yaml")
    p.add_argument("--model-name-or-path", default="checkpoints/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric.pth")
    p.add_argument("--night3r-checkpoint", default="outputs/night3r_final_step100_20260723/event_dark3r_night3r_final.pth")
    p.add_argument(
        "--night3r-pose-head-checkpoint",
        default="outputs/night3r_final_step100_pose_adapt100_20260726/pose_heads/pose_head_step00000100.pth",
    )
    p.add_argument(
        "--night3r-pose-config",
        default="config/train_night3r_final_step100_pose_adapt100.yaml",
    )
    p.add_argument("--dark3r-lora-checkpoint", default="checkpoints/dark3r_pretrained/weights/0049_lora.pth")
    p.add_argument("--vggt-root", default="/data/fcr/vggt")
    p.add_argument("--vggt-checkpoint", default="/data/fcr/ckpt/model.pt")
    p.add_argument("--outdir", required=True)
    p.add_argument(
        "--methods",
        nargs="+",
        choices=("mast3r", "dark3r", "vggt", "night3r"),
        default=("mast3r", "dark3r", "vggt", "night3r"),
    )
    p.add_argument("--exposure", type=float, default=0.30)
    p.add_argument("--read-noise-std", type=float, default=0.01)
    p.add_argument("--shot-noise-std", type=float, default=0.01)
    p.add_argument("--lowlight-seed", type=int, default=20260716)
    p.add_argument("--image-size", type=int, default=512)
    p.add_argument("--frames", type=int, default=240)
    p.add_argument("--fps", type=int, default=30)
    p.add_argument("--device", default="cuda")
    p.add_argument("--amp-dtype", choices=("bf16", "fp16"), default="bf16")
    p.add_argument("--max-points-per-view", type=int, default=18000)
    p.add_argument("--voxel-size", type=float, default=0.025)
    p.add_argument(
        "--night3r-geometry",
        choices=("legacy_pose_head", "mast3r_global"),
        default="legacy_pose_head",
        help="Night3R-only 3D export path; the global option uses its native point maps.",
    )
    p.add_argument("--allow-unsafe-checkpoint-load", action="store_true")
    return p.parse_args()


def seed_all(seed=17):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def device_of(name):
    if name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")
    return torch.device(name if name != "auto" else ("cuda" if torch.cuda.is_available() else "cpu"))


def pair_file(outdir, indices):
    path = outdir / "sequence_pairs.json"
    path.write_text(json.dumps([{"idx1": a, "idx2": b} for a, b in zip(indices, indices[1:])], indent=2))
    return path


def make_dataset(args, pairs):
    dataset = DSECEventDataset(
        root_path=args.data_root,
        image_size=(args.image_size, args.image_size),
        pair_sampling="precomputed",
        precomputed_pairs=str(pairs.resolve()),
        # Use the same OpenCV camera convention as the established Night3R
        # sequence exporter; this is an axis convention, not depth alignment.
        pose_output_convention="opencv",
        lowlight_enabled=False,
    )
    scene_dir = next((x for x in dataset.scene_frames if x.name == args.scene), None)
    if scene_dir is None:
        raise RuntimeError(f"Scene not found: {args.scene}")
    indices = [int(x) for x in args.frame_indices.split(",")]
    expected = {(a, b) for a, b in zip(indices, indices[1:])}
    dataset.samples = sorted(
        [s for s in dataset.samples if s["scene"] == args.scene and (s["idx1"], s["idx2"]) in expected],
        key=lambda s: (s["idx1"], s["idx2"]),
    )
    if len(dataset.samples) != len(indices) - 1:
        raise RuntimeError(f"Expected {len(indices)-1} pairs, got {len(dataset.samples)}")
    return dataset, indices


def lowlight_batch(batch, args):
    ns = SimpleNamespace(
        lowlight_exposure=args.exposure,
        lowlight_read_noise_std=args.read_noise_std,
        lowlight_shot_noise_std=args.shot_noise_std,
        lowlight_gamma=2.2,
        lowlight_seed=args.lowlight_seed,
    )
    return apply_deterministic_lowlight(batch, ns)[0]


def to_numpy(x):
    return x.detach().float().cpu().numpy() if torch.is_tensor(x) else np.asarray(x)


def pred_points(pred, other=False):
    key = "pts3d_in_other_view" if other and "pts3d_in_other_view" in pred else "pts3d"
    return pred[key][0].detach().float().cpu().numpy()


def pred_conf(pred, other=False):
    key = "conf"
    value = pred.get(key)
    if value is None:
        shape = pred_points(pred, other).shape[:2]
        return np.ones(shape, dtype=np.float32)
    return value[0].detach().float().cpu().numpy().squeeze()


def native_depth(pred, other=False):
    return pred_points(pred, other)[..., 2]


def compose_camera_poses(relative_poses):
    poses = [np.eye(4, dtype=np.float64)]
    for rel in relative_poses:
        poses.append(poses[-1] @ np.linalg.inv(rel))
    return poses


def gt_path_length(batches):
    poses = [np.eye(4, dtype=np.float64)]
    for batch in batches:
        relative = to_numpy(batch["relative_pose_gt"][0]).astype(np.float64)
        poses.append(poses[-1] @ np.linalg.inv(relative))
    centers = np.asarray([pose[:3, 3] for pose in poses])
    return float(np.linalg.norm(np.diff(centers, axis=0), axis=1).sum())


def align_trajectory_translation_scale(poses, target_path_length):
    centers = np.asarray([pose[:3, 3] for pose in poses], dtype=np.float64)
    path_length = float(np.linalg.norm(np.diff(centers, axis=0), axis=1).sum())
    if path_length <= 1.0e-8:
        return poses, 1.0
    scale = float(target_path_length / path_length)
    origin = centers[0].copy()
    adjusted = []
    for pose in poses:
        result = pose.copy()
        result[:3, 3] = origin + scale * (pose[:3, 3] - origin)
        adjusted.append(result)
    return adjusted, scale


def clean_rgb(image):
    return np.clip(to_numpy(image).transpose(1, 2, 0), 0, 1).astype(np.float32)


def points_for_view(
    points, conf, image, c2w, frame_id, args, points_are_world=False, depth=None
):
    """Select colored points without confusing a world Z coordinate with depth."""
    valid = np.isfinite(points).all(-1) & np.isfinite(conf)
    local_depth = points[..., 2] if depth is None else depth
    valid &= np.isfinite(local_depth) & (local_depth > 0.15) & (local_depth < 100.0)
    quantile = getattr(args, "point_conf_quantile", None)
    if quantile is not None and float(quantile) > 0.0:
        confidence_values = conf[np.isfinite(conf)]
        if confidence_values.size:
            valid &= conf >= np.quantile(confidence_values, float(quantile))
    if valid.any():
        valid &= conf >= np.quantile(conf[valid], 0.35)
    ids = np.flatnonzero(valid.reshape(-1))
    if len(ids) > args.max_points_per_view:
        rng = np.random.default_rng(17 + frame_id)
        ids = rng.choice(ids, args.max_points_per_view, replace=False)
    local = points.reshape(-1, 3)[ids]
    colors = np.clip(image.reshape(-1, 3)[ids] * 255 + 0.5, 0, 255).astype(np.uint8)
    world = local if points_are_world else local @ c2w[:3, :3].T + c2w[:3, 3]
    return world.astype(np.float32), colors, np.full(len(ids), frame_id, np.int32)


def _as_scene_view(view, index):
    """Give one MASt3R/Dark3R pair endpoint a stable scene-wide image id."""
    copied = {"img": view["img"], "true_shape": view["true_shape"]}
    copied["idx"] = torch.tensor([index], device=view["img"].device, dtype=torch.long)
    return copied


def _append_scene_edge(edges, view1, view2, pred1, pred2):
    edges["view1"].append(view1)
    edges["view2"].append(view2)
    edges["pred1_pts"].append(pred1["pts3d"])
    edges["pred1_conf"].append(pred1["conf"])
    edges["pred2_pts"].append(pred2["pts3d_in_other_view"])
    edges["pred2_conf"].append(pred2["conf"])


def _stack_scene_edges(edges):
    return {
        "view1": {
            "img": torch.cat([view["img"] for view in edges["view1"]], dim=0),
            "true_shape": torch.cat([view["true_shape"] for view in edges["view1"]], dim=0),
            "idx": torch.cat([view["idx"] for view in edges["view1"]], dim=0),
        },
        "view2": {
            "img": torch.cat([view["img"] for view in edges["view2"]], dim=0),
            "true_shape": torch.cat([view["true_shape"] for view in edges["view2"]], dim=0),
            "idx": torch.cat([view["idx"] for view in edges["view2"]], dim=0),
        },
        "pred1": {
            "pts3d": torch.cat(edges["pred1_pts"], dim=0),
            "conf": torch.cat(edges["pred1_conf"], dim=0),
        },
        "pred2": {
            "pts3d_in_other_view": torch.cat(edges["pred2_pts"], dim=0),
            "conf": torch.cat(edges["pred2_conf"], dim=0),
        },
    }


def globally_align_mast3r_sequence(
    model, batches, amp_enabled, amp_dtype, device, event_aware=False
):
    """Official DUSt3R/MASt3R point-map global alignment over a view sequence.

    The optimizer receives both directions of every adjacent edge, exactly as the
    standard MASt3R scene graph does.  It jointly estimates camera poses,
    focal lengths, and globally consistent depth maps without any GT input.
    """
    from dust3r.cloud_opt import GlobalAlignerMode, global_aligner

    edges = {key: [] for key in (
        "view1", "view2", "pred1_pts", "pred1_conf", "pred2_pts", "pred2_conf"
    )}
    with torch.no_grad():
        for pair_index, batch in enumerate(batches):
            gpu = move_batch_to_device(batch, device)
            with torch.amp.autocast(
                device_type=device.type,
                dtype=amp_dtype,
                enabled=bool(amp_enabled and device.type == "cuda"),
            ):
                forward1, forward2 = make_view_pair_from_batch(gpu, prefer_dark=True)
                if event_aware:
                    pred1, pred2 = model(gpu)
                    reverse_batch = swap_pair_batch(gpu)
                    reverse1, reverse2 = make_view_pair_from_batch(
                        reverse_batch, prefer_dark=True
                    )
                    rev_pred1, rev_pred2 = model(reverse_batch)
                else:
                    pred1, pred2 = model(forward1, forward2)
                    reverse1, reverse2 = forward2, forward1
                    rev_pred1, rev_pred2 = model(reverse1, reverse2)
            _append_scene_edge(
                edges,
                _as_scene_view(forward1, pair_index),
                _as_scene_view(forward2, pair_index + 1),
                pred1,
                pred2,
            )
            _append_scene_edge(
                edges,
                _as_scene_view(reverse1, pair_index + 1),
                _as_scene_view(reverse2, pair_index),
                rev_pred1,
                rev_pred2,
            )

    scene = global_aligner(
        _stack_scene_edges(edges),
        device=device,
        mode=GlobalAlignerMode.PointCloudOptimizer,
        # MASt3R confidence is parameterized as exp with lower bound one.
        # Under the shared low-light input, the demo's high-detail threshold
        # of 2.0 can reject every point.  The model's native confidence lower
        # bound is one, so retain only values above that bound and let the
        # common voxel filter control display density.
        min_conf_thr=1.0,
        verbose=False,
    )
    # This is the standard MASt3R point-map optimizer, initialized with its MST.
    with torch.enable_grad():
        final_loss = scene.compute_global_alignment(init="mst", niter=300, schedule="cosine", lr=0.01)
    with torch.no_grad():
        world_points = [to_numpy(points) for points in scene.get_pts3d()]
        # Keep the native confidence domain for both the threshold mask and
        # downstream filtering.  get_conf() defaults to log confidence.
        confidence = [to_numpy(value) for value in scene.get_conf(mode="id")]
        masks = [to_numpy(value).astype(bool) for value in scene.get_masks()]
        depths = [to_numpy(value) for value in scene.get_depthmaps()]
        poses = [to_numpy(value).astype(np.float64) for value in scene.get_im_poses()]
        intrinsics = [to_numpy(value).astype(np.float32) for value in scene.get_intrinsics()]
    print(
        "[global-align] confidence raw range "
        f"{min(float(value.min()) for value in confidence):.6f}.."
        f"{max(float(value.max()) for value in confidence):.6f}; "
        f"kept={sum(int(value.sum()) for value in masks)}",
        flush=True,
    )
    return world_points, confidence, masks, depths, poses, intrinsics, float(final_loss)


def voxel_downsample(points, colors, frame_ids, size):
    if not len(points) or size <= 0:
        return points, colors, frame_ids
    keys = np.floor((points - points.min(0)) / size).astype(np.int64)
    _, first = np.unique(keys, axis=0, return_index=True)
    first.sort()
    return points[first], colors[first], frame_ids[first]


def write_ply(path, points, colors):
    with path.open("w", encoding="utf-8") as f:
        f.write("ply\nformat ascii 1.0\n")
        f.write(f"element vertex {len(points)}\n")
        f.write("property float x\nproperty float y\nproperty float z\n")
        f.write("property uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n")
        for p, c in zip(points, colors):
            f.write(f"{p[0]:.6f} {p[1]:.6f} {p[2]:.6f} {int(c[0])} {int(c[1])} {int(c[2])}\n")


def write_trajectory(path, poses):
    path.write_text(json.dumps([{"camera_to_world": p.tolist()} for p in poses], indent=2))


def render_event(voxel):
    c, h, w = voxel.shape
    n = c // 2
    pos = np.any(np.abs(voxel[:n]) > 1e-8, axis=0)
    neg = np.any(np.abs(voxel[n:]) > 1e-8, axis=0)
    out = np.zeros((h, w, 3), np.uint8)
    out[pos & ~neg] = (235, 40, 40)
    out[~pos & neg] = (45, 105, 245)
    out[pos & neg] = (245, 245, 245)
    return out


def save_video(frames, path, fps):
    frame_dir = path.parent / (path.stem + "_frames")
    frame_dir.mkdir(parents=True, exist_ok=True)
    for i, frame in enumerate(frames):
        cv2.imwrite(str(frame_dir / f"frame_{i:05d}.png"), cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-framerate", str(fps), "-i", str(frame_dir / "frame_%05d.png"), "-c:v", "libx264", "-preset", "slow", "-crf", "18", "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(path)], check=True)
    shutil.rmtree(frame_dir)


def resize_rgb(image, shape):
    return cv2.resize(image, (shape[1], shape[0]), interpolation=cv2.INTER_AREA)


def depth_frames(depths, shape, frames):
    finite = np.concatenate([d[np.isfinite(d) & (d > 0)].reshape(-1) for d in depths if np.any(np.isfinite(d) & (d > 0))])
    lo, hi = np.quantile(finite, [0.02, 0.98]) if len(finite) else (0.1, 1.0)
    lo, hi = float(lo), max(float(hi), float(lo) + 1e-4)
    out = []
    for i in range(frames):
        d = depths[min(int(round(i * (len(depths)-1) / max(frames-1, 1))), len(depths)-1)]
        d = cv2.resize(np.nan_to_num(d, nan=hi, posinf=hi, neginf=hi).astype(np.float32), (shape[1], shape[0]), interpolation=cv2.INTER_NEAREST)
        norm = np.clip((d - lo) / (hi - lo), 0, 1)
        out.append(cv2.cvtColor(cv2.applyColorMap((norm * 255).astype(np.uint8), cv2.COLORMAP_TURBO), cv2.COLOR_BGR2RGB))
    return out, {"vmin": lo, "vmax": hi}


def align_depths_to_gt(depths, gt_depths):
    """Median-align display depths per view; this never changes point clouds."""
    aligned = []
    scales = []
    for prediction, target in zip(depths, gt_depths):
        if target is None:
            aligned.append(prediction)
            scales.append(1.0)
            continue
        target_shape = target.shape
        if prediction.shape != target_shape:
            prediction = cv2.resize(prediction.astype(np.float32), (target_shape[1], target_shape[0]), interpolation=cv2.INTER_LINEAR)
        valid = np.isfinite(prediction) & np.isfinite(target) & (prediction > 0) & (target > 0)
        if valid.any():
            scale = float(np.median(target[valid]) / max(np.median(prediction[valid]), 1.0e-8))
        else:
            scale = 1.0
        aligned.append(prediction * scale)
        scales.append(scale)
    return aligned, scales


def trajectory_video(poses, path, frames, fps, shape=(540, 960)):
    centers = np.asarray([p[:3, 3] for p in poses], np.float32)
    extent = np.ptp(centers, axis=0)
    axes = np.argsort(extent)[-2:]
    lo = centers[:, axes].min(0); hi = centers[:, axes].max(0)
    span = np.maximum(hi - lo, 1e-3)
    out = []
    for i in range(frames):
        canvas = np.full((*shape, 3), (11, 16, 23), np.uint8)
        pts = np.linspace(0, len(centers)-1, frames)[i]
        idx = np.arange(len(centers)); cur = np.asarray([np.interp(pts, idx, centers[:, axis]) for axis in axes])
        pix = (centers[:, axes] - lo) / span * np.asarray([shape[1]-120, shape[0]-120]) + 60
        pix = np.rint(pix).astype(np.int32)
        for a, b in zip(pix[:-1], pix[1:]): cv2.line(canvas, tuple(a), tuple(b), (70, 205, 165), 3)
        for p in pix: cv2.circle(canvas, tuple(p), 5, (225, 235, 245), -1)
        c = np.rint((cur - lo) / span * np.asarray([shape[1]-120, shape[0]-120]) + 60).astype(np.int32)
        cv2.circle(canvas, tuple(c), 9, (255, 150, 55), -1)
        out.append(canvas)
    save_video(out, path, fps)


def load_vggt(args, device):
    sys.path.insert(0, str(Path(args.vggt_root).resolve()))
    from vggt.models.vggt import VGGT
    # Match the evaluated VGGT baseline: use depth and camera heads.  For
    # geometry export, official depth is backprojected with predicted cameras.
    model = VGGT(enable_camera=True, enable_point=False, enable_depth=True, enable_track=False)
    try:
        payload = torch.load(args.vggt_checkpoint, map_location="cpu", weights_only=True, mmap=True)
    except TypeError:
        payload = torch.load(args.vggt_checkpoint, map_location="cpu")
    state = payload.get("model", payload) if isinstance(payload, dict) else payload
    missing, unexpected = model.load_state_dict(state, strict=False)
    return model.eval().to(device), {"missing": list(missing), "unexpected": list(unexpected)}


def preprocess_vggt(images):
    import torch.nn.functional as F
    b, s, c, h, w = images.shape
    return F.interpolate(images.reshape(b*s, c, h, w), size=(518, 518), mode="bicubic", align_corners=False).reshape(b, s, c, 518, 518).clamp(0, 1)


def vggt_camera_poses(pose_encoding):
    from vggt.utils.pose_enc import pose_encoding_to_extri_intri

    extrinsics, _ = pose_encoding_to_extri_intri(
        pose_encoding.float(), build_intrinsics=False
    )
    world_to_camera = to_numpy(extrinsics[0]).astype(np.float64)
    poses = []
    for extrinsic in world_to_camera:
        homogeneous = np.eye(4, dtype=np.float64)
        homogeneous[:3] = extrinsic
        poses.append(np.linalg.inv(homogeneous))
    return poses


def register_dense_shared_frame(source, target, source_conf, target_conf, pair_index):
    """Fallback Sim(3) on the same common-view pixel grid, without descriptors."""
    transform, diagnostics = estimate_relative_pose_pointmap_sim3(
        source,
        target,
        source_conf,
        target_conf,
        conf_quantile=0.50,
        max_points=50000,
        min_inliers=512,
        normalize_translation=False,
        max_normalized_rmse=2.00,
        seed=17 + pair_index,
    )
    if transform is None:
        raise RuntimeError(f"Dense shared-frame registration failed at pair {pair_index}: {diagnostics}")
    # The pose-evaluation helper returns rigid R plus the estimated Sim(3)
    # scale in diagnostics. Retain it here because this is a fusion transform.
    transform[:3, :3] *= float(diagnostics["sim3_scale"])
    diagnostics["fallback"] = "dense_common_view_sim3"
    return transform, diagnostics


def build_method(args, method, device):
    method_config = args.config
    cfg = dict(load_config(method_config))
    if method == "night3r":
        pose_cfg = dict(load_config(args.night3r_pose_config)).get("pose_head", None)
        if not pose_cfg:
            raise RuntimeError("Night3R pose configuration does not define pose_head")
        # Preserve the exact final-geometry architecture and only append the
        # separately trained pose head.
        cfg["pose_head"] = dict(pose_cfg)
    train_args = SimpleNamespace(model_name_or_path=args.model_name_or_path, dark3r_lora_checkpoint=(args.dark3r_lora_checkpoint if method == "dark3r" else None), allow_unsafe_checkpoint_load=args.allow_unsafe_checkpoint_load, image_size=(args.image_size, args.image_size), amp=True, amp_dtype=args.amp_dtype, device=args.device, seed=17, config=method_config, use_tiny_model=False, stage=None, num_epochs=None, partial_finetune_blocks=None)
    if method == "vggt":
        return load_vggt(args, device)
    model_args = SimpleNamespace(variant={"mast3r":"mast3r", "dark3r":"dark3r_lora", "night3r":"event_dark3r"}[method], checkpoint=args.night3r_checkpoint, geometry_head_checkpoint=None, config=method_config, data_root=args.data_root, model_name_or_path=args.model_name_or_path, dark3r_lora_checkpoint=args.dark3r_lora_checkpoint, allow_unsafe_checkpoint_load=args.allow_unsafe_checkpoint_load, restore_frozen_rgb=False, restore_frozen_rgb_scope="none", restore_frozen_rgb_alpha=1.0)
    model, info = build_rgb_model(model_args, cfg, train_args, device)
    if method == "night3r":
        info["pose_head_load"] = load_pose_head_checkpoint(
            model,
            args.night3r_pose_head_checkpoint,
            allow_unsafe=args.allow_unsafe_checkpoint_load,
        )
    return model, info


def main():
    args = parse_args(); seed_all(); device = device_of(args.device)
    outdir = Path(args.outdir).resolve(); outdir.mkdir(parents=True, exist_ok=True)
    indices = [int(x) for x in args.frame_indices.split(",")]
    pairs = pair_file(outdir, indices)
    dataset, _ = make_dataset(args, pairs)
    loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)
    batches = [lowlight_batch(batch, args) for batch in loader]
    reference_path_length = gt_path_length(batches)
    scene_dir = next(x for x in dataset.scene_frames if x.name == args.scene)
    cameras_yaml = scene_dir / "cameras.yaml"
    K, width, height = load_camera(cameras_yaml)
    shape = (height, width)

    shared = outdir / "shared_inputs"; shared.mkdir(exist_ok=True)
    events = [batches[0]["event_voxel"][0, 0].numpy()]
    events += [b["event_voxel"][0, 1].numpy() for b in batches]
    event_frames = [resize_rgb(render_event(e), shape) for e in events]
    save_video([event_frames[min(int(round(i*(len(event_frames)-1)/max(args.frames-1,1))), len(event_frames)-1)] for i in range(args.frames)], shared / "event_stream.mp4", args.fps)
    shutil.copy2(cameras_yaml, shared / "cameras.yaml")

    for method in args.methods:
        print(f"[export] {method}", flush=True)
        method_dir = outdir / method; method_dir.mkdir(exist_ok=True)
        model, load_info = build_method(args, method, device)
        depths = [None] * len(indices); all_points=[]; all_colors=[]; all_ids=[]
        if "depth_gt1" in batches[0]:
            gt_depths = [to_numpy(batches[0]["depth_gt1"][0])] + [
                to_numpy(b["depth_gt2"][0]) for b in batches
            ]
        else:
            # Long concatenated MVSEC display sequences retain RGB/events/poses
            # but intentionally omit metric depth labels.
            gt_depths = [None] * len(indices)
        pose_diagnostics = []
        render_intrinsics = None
        intrinsic_image_size = None
        with torch.no_grad():
            if method == "vggt":
                images = torch.stack([batches[0]["images_photo_dark"][0, 0]] + [b["images_photo_dark"][0, 1] for b in batches], 0).unsqueeze(0).to(device)
                pred = model(images=preprocess_vggt(images))
                vd = pred["depth"][0, ..., 0].detach().float().cpu().numpy()
                vc = pred["depth_conf"][0].detach().float().cpu().numpy()
                from vggt.utils.geometry import unproject_depth_map_to_point_map
                from vggt.utils.pose_enc import pose_encoding_to_extri_intri
                extrinsics, intrinsics = pose_encoding_to_extri_intri(
                    pred["pose_enc"].float(), image_size_hw=(518, 518), build_intrinsics=True
                )
                vp = unproject_depth_map_to_point_map(
                    vd[..., None], extrinsics[0], intrinsics[0]
                )
                poses = vggt_camera_poses(pred["pose_enc"])
                render_intrinsics = [to_numpy(value).astype(np.float32) for value in intrinsics[0]]
                intrinsic_image_size = [518, 518]
                # Geometry sees the shared darkened input, but point-cloud
                # colors must remain the original sensor RGB for comparison.
                rgb_sequence = [clean_rgb(batches[0]["images_photo"][0, 0])] + [clean_rgb(b["images_photo"][0, 1]) for b in batches]
                for i in range(len(indices)):
                    depths[i] = vd[i]
                    pts, col, ids = points_for_view(
                        vp[i], vc[i], resize_rgb(rgb_sequence[i], vp[i].shape[:2]),
                        None, i, args, points_are_world=True, depth=vd[i],
                    )
                    all_points.append(pts); all_colors.append(col); all_ids.append(ids)
                pose_diagnostics = [{"pose_source": "official_vggt camera/depth heads; depth backprojection"}]
            else:
                method_config = args.config
                amp_enabled, _, amp_dtype = resolve_amp_config(dict(load_config(method_config)), SimpleNamespace(amp=True, amp_dtype=args.amp_dtype), device)
                records = []
                if method == "night3r" and args.night3r_geometry == "legacy_pose_head":
                    poses = [np.eye(4, dtype=np.float64)]
                    for i, batch in enumerate(batches):
                        gpu = move_batch_to_device(batch, device)
                        with torch.amp.autocast(device_type=device.type, dtype=amp_dtype, enabled=bool(amp_enabled and device.type == "cuda")):
                            pred1, pred2 = model(gpu)
                        relative = pred1.get("relative_pose")
                        if relative is None:
                            raise RuntimeError("Night3R pose checkpoint did not return relative_pose")
                        relative = to_numpy(relative[0]).astype(np.float64)
                        if not np.isfinite(relative).all():
                            raise RuntimeError(f"Night3R pose head returned non-finite pose at pair {i}")
                        poses.append(poses[-1] @ np.linalg.inv(relative))
                        pose_diagnostics.append({"pair_index": i, "pose_source": "Night3R direct pose head"})
                        depths[i] = native_depth(pred1)
                        records.append((
                            pred_points(pred1), pred_conf(pred1), clean_rgb(batch["images_photo_dark"][0, 0]),
                            pred_points(pred2, other=True), pred_conf(pred2, other=True), clean_rgb(batch["images_photo_dark"][0, 1]),
                        ))
                        if i == len(batches)-1:
                            depths[i+1] = native_depth(pred2, other=True)
                    poses, night3r_pose_scale = align_trajectory_translation_scale(
                        poses, reference_path_length
                    )
                    pose_diagnostics.append({
                        "pose_translation_scale": night3r_pose_scale,
                        "scale_reference": "GT path length only; predicted pose rotations and relative translation directions retained",
                    })
                else:
                    (
                        world_points,
                        confidence,
                        masks,
                        depths,
                        poses,
                        intrinsics,
                        final_loss,
                    ) = globally_align_mast3r_sequence(
                        model,
                        batches,
                        amp_enabled,
                        amp_dtype,
                        device,
                        event_aware=(method == "night3r"),
                    )
                    intrinsic_image_size = list(world_points[0].shape[:2])
                    render_intrinsics = intrinsics
                    rgb_sequence = [clean_rgb(batches[0]["images_photo"][0, 0])]
                    rgb_sequence += [clean_rgb(batch["images_photo"][0, 1]) for batch in batches]
                    for i, (world, conf, mask, depth) in enumerate(
                        zip(world_points, confidence, masks, depths)
                    ):
                        filtered_conf = np.where(mask, conf, -np.inf)
                        pts, col, ids = points_for_view(
                            world,
                            filtered_conf,
                            resize_rgb(rgb_sequence[i], world.shape[:2]),
                            None,
                            i,
                            args,
                            points_are_world=True,
                            depth=depth,
                        )
                        all_points.append(pts); all_colors.append(col); all_ids.append(ids)
                    pose_diagnostics = [{
                        "pose_source": "official DUSt3R/MASt3R global point-map optimizer",
                        "scene_graph": "bidirectional adjacent pairs",
                        "global_alignment_loss": final_loss,
                    }]

                if method == "night3r" and args.night3r_geometry == "legacy_pose_head":
                    for i, (first, first_conf, first_rgb, second, second_conf, second_rgb) in enumerate(records):
                        pts, col, ids = points_for_view(
                            first, first_conf, first_rgb, poses[i], i, args
                        )
                        all_points.append(pts); all_colors.append(col); all_ids.append(ids)
                        if i == len(records) - 1:
                            pts, col, ids = points_for_view(
                                second, second_conf, second_rgb, poses[i + 1], i + 1, args
                            )
                            all_points.append(pts); all_colors.append(col); all_ids.append(ids)
        del model
        if device.type == "cuda": torch.cuda.empty_cache()
        points = np.concatenate(all_points); colors = np.concatenate(all_colors); frame_ids = np.concatenate(all_ids)
        # MASt3R/Dark3R/VGGT all have their own unconstrained scene scale.
        # A fixed meter-sized voxel would bias display density across methods.
        robust_span = np.ptp(np.percentile(points, [1.0, 99.0], axis=0), axis=0)
        display_voxel = max(float(np.linalg.norm(robust_span)) * 0.0015, 1.0e-5)
        points, colors, frame_ids = voxel_downsample(points, colors, frame_ids, display_voxel)
        display_points = points * np.asarray([1,-1,1], np.float32)
        write_ply(method_dir / "pointcloud.ply", display_points, colors)
        palette = (cv2.applyColorMap(np.linspace(0,255,len(indices),dtype=np.uint8).reshape(-1,1), cv2.COLORMAP_TURBO)[:,0,::-1]).astype(np.uint8)
        write_ply(method_dir / "pointcloud_temporal.ply", display_points, palette[np.clip(frame_ids,0,len(palette)-1)])
        write_trajectory(method_dir / "trajectory.json", poses)
        intrinsics_path = None
        if render_intrinsics is not None:
            intrinsics_path = method_dir / "camera_intrinsics.json"
            intrinsics_path.write_text(json.dumps({
                "image_size_hw": intrinsic_image_size,
                "intrinsics": [np.asarray(value).tolist() for value in render_intrinsics],
            }, indent=2))
        depths, depth_scales = align_depths_to_gt(depths, gt_depths)
        dframes, drange = depth_frames(depths, shape, args.frames)
        save_video(dframes, method_dir / "depth.mp4", args.fps)
        traj = method_dir / "camera_trajectory.mp4"
        trajectory_video(poses, traj, args.frames, args.fps)
        render_cmd = [sys.executable, str(REPO_ROOT / "utils/render_farbaseline_follow_video.py"), "--ply", str(method_dir / "pointcloud.ply"), "--trajectory", str(method_dir / "trajectory.json"), "--cameras-yaml", str(cameras_yaml), "--scene", args.scene, "--views", str(len(indices)), "--pairs", str(len(indices)-1), "--path-length", str(float(np.linalg.norm(np.diff(np.asarray([p[:3,3] for p in poses]), axis=0), axis=1).sum())), "--outdir", str(method_dir), "--frames", str(args.frames), "--fps", str(args.fps), "--point-radius", "1", "--color-exposure", "1.0"]
        if intrinsics_path is not None:
            render_cmd += ["--intrinsics-json", str(intrinsics_path)]
        subprocess.run(render_cmd, check=True)
        os.replace(method_dir / "night3r_farbaseline_camera_follow.mp4", method_dir / "pointcloud.mp4")
        shutil.copy2(shared / "event_stream.mp4", method_dir / "event_stream.mp4")
        pose_source = (
            "official VGGT camera head"
            if method == "vggt"
            else "Night3R native point maps with official DUSt3R/MASt3R global optimizer"
            if method == "night3r" and args.night3r_geometry == "mast3r_global"
            else "Night3R direct pose head"
            if method == "night3r"
            else "official DUSt3R/MASt3R global point-map optimizer"
        )
        (method_dir / "method.json").write_text(json.dumps({"method": method, "load_info": load_info, "model_input": "shared deterministic lightly-darkened images_photo_dark", "pointcloud_color": "original sensor RGB images_photo", "color_display": "direct sRGB; no display-only brightness exponent", "display_voxel_size": display_voxel, "depth_scale": "per-view median aligned to GT for depth video only", "depth_scales": depth_scales, "pointcloud_pose": pose_source, "intrinsics": "method-predicted/optimized" if intrinsics_path else "dataset calibration (Night3R legacy video only)", "pose_diagnostics": pose_diagnostics, "event_stream": "shared synchronized input"}, indent=2))

    manifest = {"dataset":"MVSEC", "scene":args.scene, "frame_indices":indices, "methods":list(args.methods), "excluded_methods":["eventvggt","depthanyevent"], "model_input":"shared deterministic lightly-darkened RGB (images_photo_dark)", "pointcloud_color":"original RGB (images_photo)", "depth_video":"per-view median aligned to GT depth", "night3r_geometry":args.night3r_geometry, "night3r_pose_display_scale":"one per-scene global translation scale from GT path length; no GT rotation/trajectory substitution and no depth/point-cloud alignment" if args.night3r_geometry == "legacy_pose_head" else "not used; global point-map optimization has no GT scale", "exposure":args.exposure, "noise":{"shot":args.shot_noise_std,"read":args.read_noise_std,"seed":args.lowlight_seed}, "pointcloud_policy":"method-predicted poses and native point maps; no GT pose substitution or depth/scale alignment", "files":"each selected method contains event_stream.mp4, depth.mp4, pointcloud.mp4, camera_trajectory.mp4"}
    (outdir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
