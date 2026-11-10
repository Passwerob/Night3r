#!/usr/bin/env python3
"""Run MASt3R sparse global alignment with Night3R RGB-event pair inference."""

import argparse
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import torch
from PIL import Image

import mast3r.utils.path_to_dust3r  # noqa: F401
from dust3r.utils.image import load_images
from mast3r.cloud_opt import sparse_ga as sparse_ga_module
from mast3r.cloud_opt.sparse_ga import sparse_global_alignment
from mast3r.image_pairs import make_pairs
from train_event_dark3r import load_config, move_batch_to_device, resolve_amp_config
from train_event_dark3r import is_descriptor_head_parameter, safe_torch_load
from utils.eval_event_dark3r_variant import build_model, resolve_device
from utils.export_dsec_darken_pointcloud_contrast import render_cloud, write_ply
from utils.export_night3r_long_sequence import (
    DSECEventDataset,
    apply_deterministic_lowlight,
    make_model_args,
    make_train_args,
    set_seed,
)

def parse_args():
    parser = argparse.ArgumentParser("Night3R sparse global sequence export")
    parser.add_argument(
        "--checkpoint",
        default="outputs/night3r_mvsec_only_adapt_1k_from_best/last_event_dark3r.pth",
    )
    parser.add_argument("--pose-head-checkpoint", default=None)
    parser.add_argument(
        "--descriptor-head-checkpoint",
        default=None,
        help="Compact descriptor-head checkpoint to overlay on --checkpoint.",
    )
    parser.add_argument(
        "--config", default="config/train_night3r_pose_separate_branches_w5_dense2k.yaml"
    )
    parser.add_argument(
        "--model-name-or-path",
        default="checkpoints/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric.pth",
    )
    parser.add_argument("--data-root", default="/data/fcr/data/eval_data/DL3DV")
    parser.add_argument("--scene", default="screen-000090")
    parser.add_argument("--start-frame", type=int, default=0)
    parser.add_argument("--num-frames", type=int, default=8)
    parser.add_argument("--image-size", type=int, default=512)
    parser.add_argument("--exposure", type=float, default=0.10)
    parser.add_argument("--read-noise-std", type=float, default=0.01)
    parser.add_argument("--shot-noise-std", type=float, default=0.01)
    parser.add_argument("--lowlight-seed", type=int, default=20260716)
    parser.add_argument("--scene-graph", default="swin-2-noncyclic")
    parser.add_argument(
        "--kinematic-mode",
        choices=("mst", "hclust-ward"),
        default="hclust-ward",
    )
    parser.add_argument("--subsample", type=int, default=8)
    parser.add_argument("--niter1", type=int, default=150)
    parser.add_argument("--niter2", type=int, default=100)
    parser.add_argument("--lr1", type=float, default=0.07)
    parser.add_argument("--lr2", type=float, default=0.01)
    parser.add_argument("--init-focal", type=float, default=300.0)
    parser.add_argument("--focal-reg-w", type=float, default=100.0)
    parser.add_argument("--matching-conf-thr", type=float, default=5.0)
    parser.add_argument("--min-corres-conf-thr", type=float, default=0.0)
    parser.add_argument("--conf-quantile", type=float, default=0.45)
    parser.add_argument("--voxel-size", type=float, default=0.04)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--amp-dtype", choices=("fp16", "bf16"), default="bf16")
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--allow-unsafe-checkpoint-load", action="store_true")
    parser.add_argument(
        "--outdir", default="outputs/night3r_sparse_ga_dl3dv_000090_8f_20260720"
    )
    return parser.parse_args()


def save_tensor_rgb(tensor, path):
    array = (
        tensor.detach().float().permute(1, 2, 0).clamp(0, 1).mul(255).byte().numpy()
    )
    Image.fromarray(array).save(path)


def load_descriptor_head_checkpoint(model, path, allow_unsafe=False):
    payload = safe_torch_load(
        path,
        map_location="cpu",
        allow_unsafe=allow_unsafe,
        purpose="Night3R descriptor-head checkpoint",
    )
    if payload.get("checkpoint_kind") != "descriptor_head_only":
        raise RuntimeError(f"Not a descriptor-head-only checkpoint: {path}")
    state = payload.get("descriptor_head", None)
    if not isinstance(state, dict) or not state:
        raise RuntimeError(f"Descriptor checkpoint has no descriptor_head state: {path}")
    invalid = [key for key in state if not is_descriptor_head_parameter(key)]
    if invalid:
        raise RuntimeError(f"Invalid descriptor checkpoint keys: {invalid[:10]}")
    missing, unexpected = model.load_state_dict(state, strict=False)
    loaded = set(state)
    missing_descriptor = [
        key
        for key in model.state_dict()
        if is_descriptor_head_parameter(key) and key not in loaded
    ]
    if missing_descriptor or unexpected:
        raise RuntimeError(
            "Incomplete descriptor overlay: "
            f"missing={missing_descriptor}, unexpected={list(unexpected)}"
        )
    return {
        "checkpoint": str(Path(path).resolve()),
        "step": payload.get("step"),
        "base_checkpoint": payload.get("base_checkpoint"),
        "non_descriptor_missing_count": len(missing),
    }


def prepare_sequence(args, outdir, model_args):
    pairs = [
        {"idx1": index, "idx2": index + 1}
        for index in range(args.start_frame, args.start_frame + args.num_frames - 1)
    ]
    pair_file = outdir / "sequence_pairs.json"
    pair_file.write_text(json.dumps(pairs, indent=2), encoding="utf-8")
    dataset = DSECEventDataset(
        root_path=args.data_root,
        image_size=(args.image_size, args.image_size),
        pair_sampling="precomputed",
        precomputed_pairs=str(pair_file),
        pose_output_convention="opencv",
        lowlight_enabled=False,
    )
    dataset.samples = sorted(
        [sample for sample in dataset.samples if sample["scene"] == args.scene],
        key=lambda sample: (sample["idx1"], sample["idx2"]),
    )
    if len(dataset.samples) != args.num_frames - 1:
        raise RuntimeError(
            f"Expected {args.num_frames - 1} pairs, found {len(dataset.samples)}"
        )

    input_dir = outdir / "dark_inputs"
    input_dir.mkdir(parents=True, exist_ok=True)
    frame_payload = {}
    loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)
    for batch in loader:
        batch, _ = apply_deterministic_lowlight(batch, model_args)
        for view_index, frame_index in enumerate((int(batch["idx1"][0]), int(batch["idx2"][0]))):
            if frame_index in frame_payload:
                continue
            path = (input_dir / f"frame_{frame_index:06d}.png").resolve()
            save_tensor_rgb(batch["images_photo_dark"][0, view_index], path)
            frame_payload[frame_index] = {
                "path": str(path),
                "event": batch["event_voxel"][0, view_index].detach().cpu(),
                "clean": batch["images_photo"][0, view_index].detach().cpu(),
            }
    ordered = [frame_payload[index] for index in range(args.start_frame, args.start_frame + args.num_frames)]
    return ordered, pair_file


def make_event_symmetric_inference(event_lookup, amp_enabled, amp_dtype):
    @torch.no_grad()
    def event_symmetric_inference(model, image1, image2, device):
        def run(first, second):
            images = torch.stack([first["img"], second["img"]], dim=1).to(
                device, non_blocking=True
            )
            events = torch.stack(
                [event_lookup[first["instance"]], event_lookup[second["instance"]]], dim=0
            ).unsqueeze(0).to(device, non_blocking=True)
            batch = {
                "images_dark": images,
                "event_voxel": events,
                "dataset_source": "dl3dv",
            }
            with torch.amp.autocast(
                device_type=torch.device(device).type,
                dtype=amp_dtype,
                enabled=bool(amp_enabled and torch.device(device).type == "cuda"),
            ):
                pred_first, pred_second = model(batch)
            pred_second = dict(pred_second)
            pred_second["pts3d"] = pred_second.pop("pts3d_in_other_view")
            return pred_first, pred_second

        result11, result21 = run(image1, image2)
        result22, result12 = run(image2, image1)
        return result11, result21, result22, result12

    return event_symmetric_inference


def aggregate_scene(scene, clean_images, conf_quantile, voxel_size):
    points, _, confidences = scene.get_dense_pts3d(clean_depth=True, subsample=8)
    all_points = []
    all_colors = []
    for pointmap, confidence, clean in zip(points, confidences, clean_images):
        pointmap = pointmap.detach().float().cpu().numpy().reshape(-1, 3)
        confidence = confidence.detach().float().cpu().numpy().reshape(-1)
        colors = clean.permute(1, 2, 0).numpy().reshape(-1, 3)
        valid = np.isfinite(pointmap).all(axis=1) & np.isfinite(confidence)
        if np.any(valid):
            valid &= confidence >= np.quantile(confidence[valid], conf_quantile)
        all_points.append(pointmap[valid])
        all_colors.append(np.clip(colors[valid] * 255.0 + 0.5, 0, 255).astype(np.uint8))
    points = np.concatenate(all_points, axis=0)
    colors = np.concatenate(all_colors, axis=0)
    if voxel_size > 0:
        origin = points.min(axis=0)
        keys = np.floor((points - origin) / voxel_size).astype(np.int64)
        _, first = np.unique(keys, axis=0, return_index=True)
        points, colors = points[first], colors[first]
    return points.astype(np.float32), colors


def main():
    args = parse_args()
    outdir = (REPO_ROOT / args.outdir).resolve()
    outdir.mkdir(parents=True, exist_ok=True)
    set_seed(args.seed)
    cfg = dict(load_config(args.config))
    cfg["lowlight_aug"] = {**dict(cfg.get("lowlight_aug", {})), "enabled": False}
    model_args = make_model_args(args, outdir)
    sequence, pair_file = prepare_sequence(args, outdir, model_args)
    train_args = make_train_args(args, outdir, pair_file)
    device = resolve_device(args.device)
    model, load_info = build_model(model_args, cfg, train_args, device)
    non_pose_missing = [
        key for key in load_info.get("missing_keys", []) if not key.startswith("pose_head.")
    ]
    if non_pose_missing or load_info.get("unexpected_keys"):
        raise RuntimeError(
            f"Incomplete geometry checkpoint: missing={non_pose_missing}, "
            f"unexpected={load_info.get('unexpected_keys')}"
        )
    descriptor_load_info = None
    if args.descriptor_head_checkpoint:
        descriptor_load_info = load_descriptor_head_checkpoint(
            model,
            args.descriptor_head_checkpoint,
            allow_unsafe=args.allow_unsafe_checkpoint_load,
        )
    model.eval()
    amp_enabled, amp_dtype_name, amp_dtype = resolve_amp_config(cfg, train_args, device)

    filelist = [item["path"] for item in sequence]
    images = load_images(filelist, size=args.image_size, square_ok=True, verbose=True)
    pairs = make_pairs(images, scene_graph=args.scene_graph, prefilter=None, symmetrize=True)
    event_lookup = {item["path"]: item["event"] for item in sequence}
    original_symmetric_inference = sparse_ga_module.symmetric_inference
    sparse_ga_module.symmetric_inference = make_event_symmetric_inference(
        event_lookup, amp_enabled, amp_dtype
    )
    try:
        scene = sparse_global_alignment(
            filelist,
            pairs,
            str(outdir / "cache"),
            model,
            subsample=args.subsample,
            device=str(device),
            kinematic_mode=args.kinematic_mode,
            lr1=args.lr1,
            niter1=args.niter1,
            lr2=args.lr2,
            niter2=args.niter2,
            opt_depth=True,
            shared_intrinsics=True,
            matching_conf_thr=args.matching_conf_thr,
            min_corres_conf_thr=args.min_corres_conf_thr,
            init_focal=args.init_focal,
            focal_reg_w=args.focal_reg_w,
        )
    finally:
        sparse_ga_module.symmetric_inference = original_symmetric_inference

    points, colors = aggregate_scene(
        scene, [item["clean"] for item in sequence], args.conf_quantile, args.voxel_size
    )
    write_ply(outdir / "night3r_sparse_global_rgb.ply", points, colors)
    rgb_render = render_cloud(
        points, rgb_colors=colors, color_mode="rgb", width=1800, height=1000, point_radius=1
    )
    depth_render = render_cloud(
        points, rgb_colors=colors, color_mode="depth", width=1800, height=1000, point_radius=1
    )
    Image.fromarray(rgb_render).save(outdir / "night3r_sparse_global_rgb.png")
    Image.fromarray(depth_render).save(outdir / "night3r_sparse_global_depth.png")
    poses = scene.get_im_poses().detach().float().cpu().numpy()
    camera_centers = poses[:, :3, 3]
    trajectory_steps = np.linalg.norm(np.diff(camera_centers, axis=0), axis=1)
    final_focal = float(scene.get_focals()[0].detach().float().cpu())
    manifest = {
        "dataset": "DL3DV",
        "scene": args.scene,
        "start_frame": args.start_frame,
        "num_frames": args.num_frames,
        "exposure": args.exposure,
        "scene_graph": args.scene_graph,
        "status": "requires_visual_and_trajectory_validation",
        "optimization": {
            "subsample": args.subsample,
            "niter1": args.niter1,
            "niter2": args.niter2,
            "lr1": args.lr1,
            "lr2": args.lr2,
            "init_focal": args.init_focal,
            "focal_reg_w": args.focal_reg_w,
            "matching_conf_thr": args.matching_conf_thr,
            "min_corres_conf_thr": args.min_corres_conf_thr,
            "kinematic_mode": args.kinematic_mode,
            "conf_quantile": args.conf_quantile,
            "voxel_size": args.voxel_size,
        },
        "point_count": int(len(points)),
        "final_focal_px": final_focal,
        "trajectory_diagnostics": {
            "path_length_internal_units": float(trajectory_steps.sum()),
            "step_coefficient_of_variation": float(
                trajectory_steps.std() / max(trajectory_steps.mean(), 1.0e-12)
            ),
        },
        "camera_to_world": poses.tolist(),
        "amp": {"enabled": bool(amp_enabled), "dtype": amp_dtype_name},
        "checkpoint": str((REPO_ROOT / args.checkpoint).resolve()),
        "descriptor_head_checkpoint": descriptor_load_info,
        "claim_scope": (
            "Night3R pair predictions globally aligned by the official MASt3R sparse "
            "global optimizer; no ground-truth pose or depth is used in alignment. "
            "The export is diagnostic until visual coherence and trajectory stability "
            "have both been verified."
        ),
    }
    (outdir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps({"outdir": str(outdir), "point_count": len(points)}, indent=2))


if __name__ == "__main__":
    main()
