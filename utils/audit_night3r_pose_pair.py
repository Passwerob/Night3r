#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from model.event_dark3r_student import make_view_pair_from_batch
from train_event_dark3r import build_dataset, load_config, move_batch_to_device
from utils.eval_event_dark3r_variant import apply_event_mode, build_model, cfg_without_lora
from utils.eval_night3r_trajectory import (
    _as_numpy,
    _rotation_error_deg,
    convert_dl3dv_relative_pose_to_opencv,
    estimate_relative_pose_pointmap_sim3,
    make_train_args,
    set_seed,
    swap_pair_batch,
)


def parse_args():
    parser = argparse.ArgumentParser("Audit one Night3R pose pair")
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--model-name-or-path", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--max-scenes", type=int, default=1)
    parser.add_argument("--max-frames-per-scene", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--image-size", type=int, nargs=2, default=None)
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--amp-dtype", choices=("fp16", "bf16"), default="bf16")
    parser.add_argument("--allow-unsafe-checkpoint-load", action="store_true")
    parser.add_argument("--pose-recovery", default="pointmap_sim3")
    parser.add_argument("--event-mode", default="real")
    parser.add_argument("--conf-quantile", type=float, default=0.70)
    parser.add_argument("--max-pnp-points", type=int, default=50000)
    parser.add_argument("--min-pnp-inliers", type=int, default=512)
    return parser.parse_args()


def point_stats(value):
    points = _as_numpy(value)
    if points.ndim == 4:
        points = points[0]
    finite = np.isfinite(points).all(axis=-1)
    points = points[finite]
    return {
        "finite_fraction": float(finite.mean()),
        "norm_median": float(np.median(np.linalg.norm(points, axis=-1))),
        "z_median": float(np.median(points[:, 2])),
        "z_positive_fraction": float(np.mean(points[:, 2] > 0.0)),
    }


def compare_tensors(left, right):
    left = left.detach().float()
    right = right.detach().float()
    delta = (left - right).abs()
    denominator = right.abs().mean().clamp_min(1.0e-8)
    return {
        "mean_abs": float(delta.mean().cpu()),
        "max_abs": float(delta.max().cpu()),
        "relative_mean_abs": float((delta.mean() / denominator).cpu()),
    }


def audit_prediction(label, forward, reverse, gt_rel, args):
    pred1, pred2 = forward
    reverse1, reverse2 = reverse
    candidates = {
        "view2": (
            pred2["pts3d_in_other_view"],
            reverse1["pts3d"],
            pred2["conf"],
            reverse1["conf"],
        ),
        "view1": (
            pred1["pts3d"],
            reverse2["pts3d_in_other_view"],
            pred1["conf"],
            reverse2["conf"],
        ),
    }
    registrations = {}
    for direction, values in candidates.items():
        transform, diagnostics = estimate_relative_pose_pointmap_sim3(
            *values,
            conf_quantile=args.conf_quantile,
            max_points=args.max_pnp_points,
            min_inliers=args.min_pnp_inliers,
            normalize_translation=True,
            seed=args.seed,
        )
        row = dict(diagnostics)
        if transform is not None:
            row.update(
                {
                    "rotation_error_gt_deg": _rotation_error_deg(
                        transform[:3, :3], gt_rel[:3, :3]
                    ),
                    "rotation_error_inverse_gt_deg": _rotation_error_deg(
                        transform[:3, :3], gt_rel[:3, :3].T
                    ),
                    "rotation_matrix": transform[:3, :3].tolist(),
                }
            )
        registrations[direction] = row
    return {
        "label": label,
        "forward_view1": point_stats(pred1["pts3d"]),
        "forward_view2_in_view1": point_stats(pred2["pts3d_in_other_view"]),
        "reverse_view2": point_stats(reverse1["pts3d"]),
        "reverse_view1_in_view2": point_stats(reverse2["pts3d_in_other_view"]),
        "registrations": registrations,
    }


def main():
    args = parse_args()
    set_seed(args.seed)
    cfg = dict(load_config(args.config))
    cfg["lowlight_aug"] = {**dict(cfg.get("lowlight_aug", {})), "enabled": False}
    train_args = make_train_args(args)
    dataset = build_dataset(train_args, cfg)
    batch_cpu = next(iter(torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False)))
    device = torch.device(args.device)
    batch = move_batch_to_device(batch_cpu, device)
    zero_batch = apply_event_mode(batch, "zero")
    reverse_batch = swap_pair_batch(batch)
    reverse_zero_batch = swap_pair_batch(zero_batch)

    event_args = argparse.Namespace(**vars(args))
    event_args.variant = "event_dark3r"
    event_args.dark3r_lora_checkpoint = None
    event_model, event_load = build_model(event_args, cfg, train_args, device)

    mast3r_args = argparse.Namespace(**vars(args))
    mast3r_args.variant = "mast3r"
    mast3r_args.checkpoint = None
    mast3r_args.dark3r_lora_checkpoint = None
    mast3r_model, mast3r_load = build_model(
        mast3r_args, cfg_without_lora(cfg), train_args, device
    )

    view1, view2 = make_view_pair_from_batch(batch, prefer_dark=True)
    reverse1, reverse2 = make_view_pair_from_batch(reverse_batch, prefer_dark=True)
    dtype = torch.bfloat16 if args.amp_dtype == "bf16" else torch.float16
    with torch.no_grad(), torch.amp.autocast(
        device_type=device.type, dtype=dtype, enabled=args.amp and device.type == "cuda"
    ):
        base_forward = mast3r_model(view1, view2)
        base_reverse = mast3r_model(reverse1, reverse2)
        internal_forward = event_model.dark3r(view1, view2)
        internal_reverse = event_model.dark3r(reverse1, reverse2)
        zero_forward = event_model(zero_batch)
        zero_reverse = event_model(reverse_zero_batch)
        real_forward = event_model(batch)
        real_reverse = event_model(reverse_batch)

    gt_rel = _as_numpy(batch["relative_pose_gt"])[0]
    dataset_source = str(batch_cpu["dataset_source"][0]).lower()
    if dataset_source == "dl3dv":
        gt_rel = convert_dl3dv_relative_pose_to_opencv(gt_rel)

    result = {
        "sample_id": str(batch_cpu["sample_id"][0]),
        "dataset_source": dataset_source,
        "gt_relative": gt_rel.tolist(),
        "load": {"event": event_load, "mast3r": mast3r_load},
        "predictions": [
            audit_prediction("mast3r_base", base_forward, base_reverse, gt_rel, args),
            audit_prediction(
                "checkpoint_internal_rgb", internal_forward, internal_reverse, gt_rel, args
            ),
            audit_prediction("night3r_zero", zero_forward, zero_reverse, gt_rel, args),
            audit_prediction("night3r_real", real_forward, real_reverse, gt_rel, args),
        ],
        "differences": {
            "zero_vs_internal_view1": compare_tensors(
                zero_forward[0]["pts3d"], internal_forward[0]["pts3d"]
            ),
            "zero_vs_internal_view2": compare_tensors(
                zero_forward[1]["pts3d_in_other_view"],
                internal_forward[1]["pts3d_in_other_view"],
            ),
            "internal_vs_base_view1": compare_tensors(
                internal_forward[0]["pts3d"], base_forward[0]["pts3d"]
            ),
            "internal_vs_base_view2": compare_tensors(
                internal_forward[1]["pts3d_in_other_view"],
                base_forward[1]["pts3d_in_other_view"],
            ),
        },
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True))
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
