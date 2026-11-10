#!/usr/bin/env python3
"""Evaluate and rank Night3R checkpoints for depth/XYZ trade-offs."""

import argparse
import copy
import csv
import json
import math
import random
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from train_event_dark3r import (  # noqa: E402
    build_dataset,
    build_event_student,
    compute_eval_loss,
    load_backbone,
    load_config,
    make_frozen_teacher,
    make_loader,
    move_batch_to_device,
    resolve_amp_config,
    resolve_stage_schedule,
    safe_torch_load,
    set_event_training_stage,
)
from utils.compare_event_dark3r_checkpoint import (  # noqa: E402
    DenseMetricAccumulator,
    as_bchw,
    depth_from_pred,
)


DATASETS = {
    "DSEC": {
        "root": "test/DSEC",
        "exclude_scenes": ["screen-000002-dsec"],
        "max_scenes": None,
        "max_pairs_per_scene": 8,
        "note": "Strict held-out DSEC scenes; screen-000002-dsec is excluded.",
    },
    "MVSEC": {
        "root": "test/MVSEC",
        "exclude_scenes": [],
        "max_scenes": 8,
        "max_pairs_per_scene": 4,
        "note": "MVSEC adaptation-domain diagnostic subset.",
    },
    "DL3DV": {
        "root": "test/DL3DV",
        "exclude_scenes": ["screen-000087", "screen-000098", "screen-000130"],
        "max_scenes": 8,
        "max_pairs_per_scene": 4,
        "note": "Strict held-out DL3DV subset; no sensor depth GT.",
    },
}


def parse_args():
    parser = argparse.ArgumentParser("Sweep one Night3R XYZ-repair checkpoint")
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--step-label", type=int)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("config/train_night3r_xyz_repair_joint_800_20260723.yaml"),
    )
    parser.add_argument(
        "--model-name-or-path",
        type=Path,
        default=Path(
            "checkpoints/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric.pth"
        ),
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--seed", type=int, default=23)
    parser.add_argument("--max-batches", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--allow-unsafe-checkpoint-load", action="store_true")
    parser.add_argument(
        "--restore-scope",
        choices=("none", "head2", "heads", "geometry", "all"),
        default="none",
        help="Keep selected pristine MASt3R RGB weights instead of checkpoint copies.",
    )
    parser.add_argument(
        "--summarize-dir",
        type=Path,
        help="Summarize existing step*.json files instead of running inference.",
    )
    return parser.parse_args()


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_train_args(args, data_root=None, dataset_spec=None):
    dataset_spec = dataset_spec or {}
    return SimpleNamespace(
        config=str(args.config),
        data_root=data_root,
        save_dir=str(args.output.parent if args.output else Path(".")),
        model_name_or_path=str(args.model_name_or_path),
        allow_unsafe_checkpoint_load=args.allow_unsafe_checkpoint_load,
        use_tiny_model=False,
        image_size=[512, 512],
        batch_size=1,
        max_batches=args.max_batches,
        num_epochs=None,
        max_steps=None,
        max_scenes=dataset_spec.get("max_scenes"),
        max_pairs_per_scene=dataset_spec.get("max_pairs_per_scene"),
        pair_stride=1,
        pair_sampling="fixed",
        min_pair_distance=1,
        max_pair_distance=1,
        precomputed_pairs=None,
        allow_reverse_pairs=False,
        symmetric_pairs=True,
        num_workers=args.num_workers,
        stage=0,
        partial_finetune_blocks=0,
        amp=True,
        amp_dtype="bf16",
        gradient_checkpointing=False,
        device=args.device,
        log_every=1,
        no_save=True,
        seed=args.seed,
        depth_psnr_max=None,
    )


class MeanAccumulator:
    def __init__(self):
        self.values = {}

    def add(self, key, value, weight=1.0):
        value = float(value)
        if not math.isfinite(value) or weight <= 0:
            return
        total, count = self.values.get(key, (0.0, 0.0))
        self.values[key] = (total + value * weight, count + weight)

    def finalize(self):
        return {
            key: total / count
            for key, (total, count) in sorted(self.values.items())
            if count > 0
        }


def merge_scalar_dict(accum, values):
    for key, value in values.items():
        if isinstance(value, (int, float)):
            accum.add(key, value)


def point_tensor(pred):
    if "pts3d" in pred:
        return pred["pts3d"]
    return pred.get("pts3d_in_other_view")


def point_diagnostics(pred, prefix, accum):
    points = point_tensor(pred)
    if points is None:
        return
    points = points.detach().float()
    finite = torch.isfinite(points).all(dim=-1)
    accum.add(f"{prefix}_finite_fraction", finite.float().mean())
    if not finite.any():
        return
    values = points[finite]
    z = values[:, 2]
    median_abs_z = z.abs().median().clamp_min(1.0e-6)
    accum.add(f"{prefix}_positive_z_fraction", (z > 0).float().mean())
    accum.add(f"{prefix}_median_abs_x_over_z", values[:, 0].abs().median() / median_abs_z)
    accum.add(f"{prefix}_median_abs_y_over_z", values[:, 1].abs().median() / median_abs_z)


def resize_points(points, target_hw):
    if points.shape[1:3] == target_hw:
        return points
    return F.interpolate(
        points.permute(0, 3, 1, 2),
        size=target_hw,
        mode="bilinear",
        align_corners=False,
    ).permute(0, 2, 3, 1)


def add_gt_xyz_metrics(pred, batch, accum):
    points = pred.get("pts3d")
    if points is None or "depth_gt" not in batch or "intrinsics" not in batch:
        return
    gt = as_bchw(batch["depth_gt"]).float()
    valid = as_bchw(
        batch.get("depth_valid", torch.isfinite(gt) & (gt > 0))
    ).bool()
    points = resize_points(points.float(), gt.shape[-2:])
    valid = (
        valid
        & torch.isfinite(gt)
        & (gt > 0)
        & torch.isfinite(points).all(dim=-1).unsqueeze(1)
    )
    for batch_index in range(gt.shape[0]):
        mask = valid[batch_index, 0]
        if int(mask.sum()) < 32:
            continue
        depth = gt[batch_index, 0]
        intrinsics = batch["intrinsics"][batch_index].float()
        height, width = depth.shape
        yy, xx = torch.meshgrid(
            torch.arange(height, device=depth.device, dtype=depth.dtype),
            torch.arange(width, device=depth.device, dtype=depth.dtype),
            indexing="ij",
        )
        fx = intrinsics[0, 0].clamp_min(1.0e-6)
        fy = intrinsics[1, 1].clamp_min(1.0e-6)
        cx = intrinsics[0, 2]
        cy = intrinsics[1, 2]
        gt_xyz = torch.stack(
            ((xx - cx) * depth / fx, (yy - cy) * depth / fy, depth),
            dim=-1,
        )
        pred_xyz = points[batch_index]
        gt_valid = gt_xyz[mask]
        pred_valid = pred_xyz[mask]
        raw_residual = pred_valid - gt_valid
        gt_scale = torch.linalg.vector_norm(gt_valid, dim=-1).median().clamp_min(1.0e-6)
        raw_nerr = torch.linalg.vector_norm(raw_residual, dim=-1).mean() / gt_scale

        pred_z_median = pred_valid[:, 2].median()
        gt_z_median = gt_valid[:, 2].median()
        scale = gt_z_median / pred_z_median.clamp_min(1.0e-6)
        aligned_residual = pred_valid * scale - gt_valid
        aligned_nerr = (
            torch.linalg.vector_norm(aligned_residual, dim=-1).mean() / gt_scale
        )
        accum.add("raw_xyz_nerr", raw_nerr)
        accum.add("median_aligned_xyz_nerr", aligned_nerr)
        accum.add("median_alignment_scale", scale)
        for component, name in enumerate(("x", "y", "z")):
            accum.add(
                f"median_aligned_{name}_nmae",
                aligned_residual[:, component].abs().mean() / gt_scale,
            )


def build_models(args, cfg, device):
    train_args = make_train_args(args)
    backbone = load_backbone(train_args, cfg)
    teacher = make_frozen_teacher(backbone).to(device).eval()
    student = build_event_student(backbone, config=cfg).to(device)
    payload = safe_torch_load(
        args.checkpoint,
        map_location="cpu",
        allow_unsafe=args.allow_unsafe_checkpoint_load,
        purpose="Night3R XYZ-repair sweep checkpoint",
    )
    checkpoint_state = payload["model"]
    restore_scope = args.restore_scope

    def restore_key(key):
        if not key.startswith("dark3r.") or restore_scope == "none":
            return False
        rgb_key = key[len("dark3r.") :]
        if restore_scope == "all":
            return True
        if restore_scope == "head2":
            return rgb_key.startswith("downstream_head2.")
        if restore_scope == "heads":
            return rgb_key.startswith(("downstream_head1.", "downstream_head2."))
        if restore_scope == "geometry":
            return rgb_key.startswith(
                (
                    "decoder_embed.",
                    "dec_blocks.",
                    "dec_blocks2.",
                    "dec_norm.",
                    "downstream_head1.",
                    "downstream_head2.",
                )
            )
        return False

    restored_keys = [key for key in checkpoint_state if restore_key(key)]
    if restored_keys:
        checkpoint_state = {
            key: value for key, value in checkpoint_state.items() if not restore_key(key)
        }
    missing, unexpected = student.load_state_dict(checkpoint_state, strict=False)
    restored_key_set = set(restored_keys)
    missing = [key for key in missing if key not in restored_key_set]
    final_stage = resolve_stage_schedule(cfg, train_args)[-1]
    set_event_training_stage(
        student,
        stage=final_stage["stage"],
        partial_finetune_blocks=final_stage["partial_finetune_blocks"],
    )
    student.eval()
    return (
        student,
        teacher,
        payload,
        list(missing),
        list(unexpected),
        restored_keys,
    )


def evaluate_dataset(args, base_cfg, student, teacher, device, name, spec):
    cfg = copy.deepcopy(base_cfg)
    dataset_cfg = dict(cfg.get("dataset", {}))
    dataset_cfg["exclude_scenes"] = list(spec.get("exclude_scenes", []))
    cfg["dataset"] = dataset_cfg
    train_args = make_train_args(args, data_root=spec["root"], dataset_spec=spec)
    dataset = build_dataset(train_args, cfg)
    loader, _ = make_loader(
        dataset,
        batch_size=1,
        num_workers=args.num_workers,
        device=device,
        distributed=False,
        shuffle=False,
    )
    amp_enabled, amp_dtype_name, amp_dtype = resolve_amp_config(
        cfg, train_args, device
    )
    dense_metrics = DenseMetricAccumulator()
    scalar_metrics = MeanAccumulator()
    xyz_metrics = MeanAccumulator()
    point_metrics = MeanAccumulator()
    sample_ids = []
    num_batches = 0

    set_seed(args.seed)
    with torch.inference_mode():
        for batch_index, batch in enumerate(loader):
            if args.max_batches is not None and batch_index >= args.max_batches:
                break
            batch = move_batch_to_device(batch, device)
            loss, loss_dict, student_out = compute_eval_loss(
                student,
                teacher,
                batch,
                cfg.get("loss", {}),
                symmetric=True,
                amp_enabled=amp_enabled,
                amp_dtype=amp_dtype,
                device_type=device.type,
            )
            scalar_metrics.add("loss", loss.detach())
            merge_scalar_dict(scalar_metrics, loss_dict)
            pred_depth = depth_from_pred(student_out[0])
            if pred_depth is not None and "depth_gt" in batch:
                gt_depth = as_bchw(batch["depth_gt"]).to(
                    device=pred_depth.device, dtype=pred_depth.dtype
                )
                gt_valid = as_bchw(
                    batch.get(
                        "depth_valid",
                        torch.isfinite(gt_depth) & (gt_depth > 0),
                    )
                ).to(device=pred_depth.device)
                dense_metrics.add_depth(
                    pred_depth, gt_depth, gt_valid, variant="raw"
                )
                dense_metrics.add_depth(
                    pred_depth, gt_depth, gt_valid, variant="median_aligned"
                )
                add_gt_xyz_metrics(student_out[0], batch, xyz_metrics)
            point_diagnostics(student_out[0], "view1", point_metrics)
            point_diagnostics(student_out[1], "view2_in_view1", point_metrics)
            ids = batch.get("sample_id", [])
            if isinstance(ids, (list, tuple)):
                sample_ids.extend(str(item) for item in ids)
            elif ids:
                sample_ids.append(str(ids))
            num_batches += 1

    scalars = scalar_metrics.finalize()
    structure_weight = float(cfg.get("loss", {}).get("structure_consistency", 0.0))
    structure_values = [
        value
        for key, value in scalars.items()
        if key in {
            "structure_consistency",
            "forward/structure_consistency",
            "reverse/structure_consistency",
        }
    ]
    structure_weighted = (
        float(np.mean(structure_values)) if structure_values else float("nan")
    )
    return {
        "root": str(Path(spec["root"]).resolve()),
        "note": spec["note"],
        "dataset_len": len(dataset),
        "num_batches": num_batches,
        "sample_ids": sample_ids,
        "amp": amp_enabled,
        "amp_dtype": amp_dtype_name if amp_enabled else None,
        "dense_metrics": dense_metrics.finalize(),
        "xyz_gt": xyz_metrics.finalize(),
        "point_diagnostics": point_metrics.finalize(),
        "loss_metrics": scalars,
        "structure_consistency_weight": structure_weight,
        "structure_consistency_weighted": structure_weighted,
        "structure_consistency_unweighted": (
            structure_weighted / structure_weight
            if structure_weight > 0 and math.isfinite(structure_weighted)
            else float("nan")
        ),
    }


def nested_number(payload, *keys):
    value = payload
    for key in keys:
        if not isinstance(value, dict) or key not in value:
            return None
        value = value[key]
    if not isinstance(value, (int, float)) or not math.isfinite(float(value)):
        return None
    return float(value)


def sanitize(value):
    if isinstance(value, dict):
        return {key: sanitize(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [sanitize(item) for item in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        value = float(value)
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def summarize(directory):
    records = []
    for path in sorted(directory.glob("step*.json")):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if "datasets" in payload:
            records.append(payload)
    if not records:
        raise RuntimeError(f"No completed step*.json files found in {directory}")
    records.sort(key=lambda item: int(item["step_label"]))
    baseline = next(
        (item for item in records if int(item["step_label"]) == 0), records[0]
    )
    metric_specs = [
        (
            "dsec_depth",
            0.30,
            ("datasets", "DSEC", "dense_metrics", "depth_median_aligned_absrel"),
        ),
        (
            "mvsec_depth",
            0.25,
            ("datasets", "MVSEC", "dense_metrics", "depth_median_aligned_absrel"),
        ),
        (
            "dsec_xyz",
            0.20,
            ("datasets", "DSEC", "xyz_gt", "median_aligned_xyz_nerr"),
        ),
        (
            "mvsec_xyz",
            0.15,
            ("datasets", "MVSEC", "xyz_gt", "median_aligned_xyz_nerr"),
        ),
        (
            "dl3dv_structure",
            0.10,
            ("datasets", "DL3DV", "structure_consistency_unweighted"),
        ),
    ]
    rows = []
    for payload in records:
        weighted = 0.0
        weight_sum = 0.0
        stability_penalty = 0.0
        row = {
            "step": int(payload["step_label"]),
            "checkpoint": payload["checkpoint"],
        }
        for name, weight, path in metric_specs:
            value = nested_number(payload, *path)
            base_value = nested_number(baseline, *path)
            row[name] = value
            if value is None or base_value is None or base_value <= 0:
                continue
            ratio = value / base_value
            row[f"{name}_vs_init"] = ratio
            weighted += weight * ratio
            weight_sum += weight
            if name in {"dsec_depth", "mvsec_depth"} and ratio > 1.15:
                stability_penalty += 0.25 * (ratio - 1.15) / 0.15
        row["tradeoff_score"] = (
            weighted / weight_sum + stability_penalty
            if weight_sum > 0
            else float("inf")
        )
        row["stability_penalty"] = stability_penalty
        rows.append(row)
    rows.sort(key=lambda item: item["tradeoff_score"])
    best = rows[0]

    output = {
        "selection_note": (
            "Lower is better. The score combines median-aligned depth, GT-derived "
            "view-1 XYZ error, and DL3DV teacher-geometry consistency relative to "
            "the initialization checkpoint. MVSEC is adaptation-domain diagnostic."
        ),
        "best": best,
        "ranking": rows,
    }
    output = sanitize(output)
    (directory / "tradeoff_summary.json").write_text(
        json.dumps(output, indent=2, allow_nan=False), encoding="utf-8"
    )
    columns = [
        "step",
        "tradeoff_score",
        "stability_penalty",
        "dsec_depth",
        "mvsec_depth",
        "dsec_xyz",
        "mvsec_xyz",
        "dl3dv_structure",
        "checkpoint",
    ]
    with (directory / "tradeoff_summary.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    lines = [
        "# Night3R XYZ-repair checkpoint sweep",
        "",
        output["selection_note"],
        "",
        "| Rank | Step | Score | DSEC AbsRel | MVSEC AbsRel | DSEC XYZ | MVSEC XYZ | DL3DV structure |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for rank, row in enumerate(rows, start=1):
        def fmt(key):
            value = row.get(key)
            return "n/a" if value is None else f"{value:.5f}"

        lines.append(
            f"| {rank} | {row['step']} | {row['tradeoff_score']:.5f} | "
            f"{fmt('dsec_depth')} | {fmt('mvsec_depth')} | {fmt('dsec_xyz')} | "
            f"{fmt('mvsec_xyz')} | {fmt('dl3dv_structure')} |"
        )
    lines.extend(
        [
            "",
            f"Selected step: **{best['step']}**",
            "",
            "Checkpoint selection on these diagnostics is for engineering comparison; "
            "the held-out test metrics must be reported separately without retuning.",
        ]
    )
    (directory / "tradeoff_summary.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    print(json.dumps(output, indent=2, allow_nan=False))


def run(args):
    if args.checkpoint is None or args.output is None or args.step_label is None:
        raise ValueError("--checkpoint, --step-label, and --output are required")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    set_seed(args.seed)
    cfg = load_config(str(args.config))
    device = torch.device(args.device)
    (
        student,
        teacher,
        payload,
        missing,
        unexpected,
        restored_keys,
    ) = build_models(args, cfg, device)
    result = {
        "checkpoint": str(args.checkpoint.resolve()),
        "step_label": int(args.step_label),
        "checkpoint_step": payload.get("step"),
        "config": str(args.config.resolve()),
        "device": str(device),
        "visible_device_count": torch.cuda.device_count() if args.device == "cuda" else 0,
        "restore_scope": args.restore_scope,
        "restored_rgb_key_count": len(restored_keys),
        "missing_keys": missing,
        "unexpected_keys": unexpected,
        "datasets": {},
    }
    for name, spec in DATASETS.items():
        result["datasets"][name] = evaluate_dataset(
            args, cfg, student, teacher, device, name, spec
        )
        print(
            json.dumps(
                {
                    "event": "dataset_complete",
                    "step": args.step_label,
                    "dataset": name,
                    "metrics": result["datasets"][name],
                },
                default=str,
            ),
            flush=True,
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result = sanitize(result)
    args.output.write_text(
        json.dumps(result, indent=2, allow_nan=False), encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "event": "checkpoint_complete",
                "step": args.step_label,
                "output": str(args.output.resolve()),
            }
        ),
        flush=True,
    )


def main():
    args = parse_args()
    if args.summarize_dir is not None:
        summarize(args.summarize_dir)
    else:
        run(args)


if __name__ == "__main__":
    main()
