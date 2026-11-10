#!/usr/bin/env python3
"""Evaluate depth quality on edge and event-active regions.

This script complements global AbsRel/RMSE metrics with structure-sensitive
regions that are important for event/RGB reconstruction:

- RGB image edges: where visual boundaries are likely.
- Event-active edges: where events indicate motion/contrast boundaries.
- GT depth edges: where geometry discontinuities exist.
- Near/far depth halves: where global scale drift often hides.

Metrics are reported in raw depth and per-sample median-aligned depth. The
median-aligned numbers help separate shape/boundary quality from global scale
error.
"""

import argparse
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
    load_backbone,
    load_config,
    move_batch_to_device,
    resolve_amp_config,
    resolve_stage_schedule,
    safe_torch_load,
    set_event_training_stage,
)
from utils.compare_event_dark3r_checkpoint import as_bchw, depth_from_pred  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser("Evaluate Event-Dark3R depth on edge regions")
    parser.add_argument("--trained-checkpoint", required=True)
    parser.add_argument("--config", default="config/train_event_dark3r.yaml")
    parser.add_argument("--data-root", default=None)
    parser.add_argument("--model-name-or-path", required=True)
    parser.add_argument("--allow-unsafe-checkpoint-load", action="store_true")
    parser.add_argument("--image-size", type=int, nargs=2, default=None)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--max-batches", type=int, default=128)
    parser.add_argument("--max-scenes", type=int, default=None)
    parser.add_argument("--max-pairs-per-scene", type=int, default=None)
    parser.add_argument("--pair-stride", type=int, default=None)
    parser.add_argument("--pair-sampling", choices=("fixed", "random", "precomputed"), default=None)
    parser.add_argument("--min-pair-distance", type=int, default=None)
    parser.add_argument("--max-pair-distance", type=int, default=None)
    parser.add_argument("--precomputed-pairs", type=str, default=None)
    parser.add_argument("--allow-reverse-pairs", dest="allow_reverse_pairs", action="store_true", default=None)
    parser.add_argument("--no-allow-reverse-pairs", dest="allow_reverse_pairs", action="store_false")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--amp", dest="amp", action="store_true", default=None)
    parser.add_argument("--no-amp", dest="amp", action="store_false")
    parser.add_argument("--amp-dtype", choices=("fp16", "bf16"), default=None)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--edge-quantile", type=float, default=0.90)
    parser.add_argument("--event-quantile", type=float, default=0.90)
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def make_train_args(args):
    return SimpleNamespace(
        config=args.config,
        trained_checkpoint=args.trained_checkpoint,
        data_root=args.data_root,
        save_dir=str(Path(args.output).parent),
        model_name_or_path=args.model_name_or_path,
        allow_unsafe_checkpoint_load=args.allow_unsafe_checkpoint_load,
        use_tiny_model=False,
        image_size=args.image_size,
        batch_size=args.batch_size,
        max_batches=args.max_batches,
        num_epochs=None,
        max_steps=None,
        max_scenes=args.max_scenes,
        max_pairs_per_scene=args.max_pairs_per_scene,
        pair_stride=args.pair_stride,
        pair_sampling=args.pair_sampling,
        min_pair_distance=args.min_pair_distance,
        max_pair_distance=args.max_pair_distance,
        precomputed_pairs=args.precomputed_pairs,
        allow_reverse_pairs=args.allow_reverse_pairs,
        symmetric_pairs=False,
        num_workers=args.num_workers,
        stage=None,
        partial_finetune_blocks=None,
        amp=args.amp,
        amp_dtype=args.amp_dtype,
        gradient_checkpointing=False,
        device=args.device,
        log_every=1,
        no_save=True,
        seed=args.seed,
    )


def resolve_device(device_arg):
    if device_arg == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("--device cuda requested but CUDA is unavailable")
        return torch.device("cuda")
    if device_arg == "cpu":
        return torch.device("cpu")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def sobel_mag(x):
    """Return Sobel gradient magnitude for a BCHW single-channel tensor."""
    if x.ndim != 4 or x.shape[1] != 1:
        raise ValueError("sobel_mag expects [B,1,H,W]")
    kx = torch.tensor(
        [[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]],
        device=x.device,
        dtype=x.dtype,
    ).view(1, 1, 3, 3)
    ky = torch.tensor(
        [[-1.0, -2.0, -1.0], [0.0, 0.0, 0.0], [1.0, 2.0, 1.0]],
        device=x.device,
        dtype=x.dtype,
    ).view(1, 1, 3, 3)
    gx = F.conv2d(x, kx, padding=1)
    gy = F.conv2d(x, ky, padding=1)
    return torch.sqrt(gx * gx + gy * gy + 1e-12)


def resize_like(x, target_hw, mode="bilinear"):
    if x.shape[-2:] == target_hw:
        return x
    kwargs = {"size": target_hw, "mode": mode}
    if mode in {"bilinear", "bicubic"}:
        kwargs["align_corners"] = False
    return F.interpolate(x, **kwargs)


def top_quantile_mask(values, valid, quantile, min_pixels=32):
    """Select the top values inside valid pixels."""
    values = values.float()
    valid = valid.bool() & torch.isfinite(values)
    selected_values = values[valid]
    if selected_values.numel() < min_pixels:
        return torch.zeros_like(valid)
    threshold = torch.quantile(selected_values, float(quantile))
    mask = valid & (values >= threshold)
    if int(mask.sum().item()) < min_pixels:
        return torch.zeros_like(valid)
    return mask


def median_split_masks(gt_depth, valid):
    selected = gt_depth[valid]
    if selected.numel() == 0:
        empty = torch.zeros_like(valid)
        return empty, empty
    median = selected.median()
    near = valid & (gt_depth <= median)
    far = valid & (gt_depth > median)
    return near, far


class RegionAccumulator:
    def __init__(self):
        self.stats = {}

    @staticmethod
    def _empty():
        return {
            "valid_px": 0,
            "absrel_sum": 0.0,
            "abs_sum": 0.0,
            "sq_sum": 0.0,
            "log_sq_sum": 0.0,
            "delta_1_count": 0,
            "delta_2_count": 0,
            "delta_3_count": 0,
            "pred_grad_sum": 0.0,
            "gt_grad_sum": 0.0,
            "grad_abs_sum": 0.0,
            "grad_px": 0,
            "max_gt": 0.0,
        }

    def add(self, region, alignment, pred, gt, mask, pred_grad, gt_grad):
        mask = mask.bool() & torch.isfinite(pred) & torch.isfinite(gt) & (pred > 0) & (gt > 0)
        if not mask.any():
            return
        key = (region, alignment)
        stats = self.stats.setdefault(key, self._empty())
        pred_values = pred[mask].clamp_min(1e-6)
        gt_values = gt[mask].clamp_min(1e-6)
        diff = pred_values - gt_values
        abs_diff = diff.abs()
        ratio = torch.maximum(pred_values / gt_values, gt_values / pred_values)
        log_diff = torch.log(pred_values) - torch.log(gt_values)
        n = int(gt_values.numel())

        stats["valid_px"] += n
        stats["absrel_sum"] += float((abs_diff / gt_values).sum().detach().cpu())
        stats["abs_sum"] += float(abs_diff.sum().detach().cpu())
        stats["sq_sum"] += float((diff * diff).sum().detach().cpu())
        stats["log_sq_sum"] += float((log_diff * log_diff).sum().detach().cpu())
        stats["delta_1_count"] += int((ratio < 1.25).sum().item())
        stats["delta_2_count"] += int((ratio < 1.25**2).sum().item())
        stats["delta_3_count"] += int((ratio < 1.25**3).sum().item())
        grad_mask = mask & torch.isfinite(pred_grad) & torch.isfinite(gt_grad)
        if grad_mask.any():
            grad_diff = (pred_grad[grad_mask] - gt_grad[grad_mask]).abs()
            stats["pred_grad_sum"] += float(pred_grad[grad_mask].sum().detach().cpu())
            stats["gt_grad_sum"] += float(gt_grad[grad_mask].sum().detach().cpu())
            stats["grad_abs_sum"] += float(grad_diff.sum().detach().cpu())
            stats["grad_px"] += int(grad_mask.sum().item())
        stats["max_gt"] = max(stats["max_gt"], float(gt_values.max().detach().cpu()))

    def finalize(self):
        rows = []
        for (region, alignment), stats in sorted(self.stats.items()):
            n = stats["valid_px"]
            if n <= 0:
                continue
            mse = stats["sq_sum"] / n
            mae = stats["abs_sum"] / n
            rmse = math.sqrt(max(mse, 0.0))
            max_gt = stats["max_gt"]
            grad_n = stats["grad_px"]
            row = {
                "region": region,
                "alignment": alignment,
                "valid_px": n,
                "grad_px": grad_n,
                "absrel": stats["absrel_sum"] / n,
                "mae": mae,
                "rmse": rmse,
                "rmse_log": math.sqrt(max(stats["log_sq_sum"] / n, 0.0)),
                "delta_1.25": stats["delta_1_count"] / n,
                "delta_1.25^2": stats["delta_2_count"] / n,
                "delta_1.25^3": stats["delta_3_count"] / n,
                "pred_grad_mean": stats["pred_grad_sum"] / grad_n if grad_n > 0 else float("nan"),
                "gt_grad_mean": stats["gt_grad_sum"] / grad_n if grad_n > 0 else float("nan"),
                "grad_l1": stats["grad_abs_sum"] / grad_n if grad_n > 0 else float("nan"),
            }
            if max_gt > 0 and mse > 0:
                row["psnr_db"] = 20.0 * math.log10(max_gt) - 10.0 * math.log10(mse)
            else:
                row["psnr_db"] = float("nan")
            row["pred_to_gt_grad_ratio"] = (
                row["pred_grad_mean"] / row["gt_grad_mean"]
                if abs(row["gt_grad_mean"]) > 1e-12
                else float("nan")
            )
            rows.append(row)
        return rows


def build_model(args, cfg, device):
    backbone = load_backbone(args, cfg)
    model = build_event_student(backbone, config=cfg)
    final_stage = resolve_stage_schedule(cfg, args)[-1]
    set_event_training_stage(
        model,
        stage=final_stage["stage"],
        partial_finetune_blocks=final_stage["partial_finetune_blocks"],
    )
    payload = safe_torch_load(
        args.trained_checkpoint,
        map_location="cpu",
        allow_unsafe=args.allow_unsafe_checkpoint_load,
        purpose="Event-Dark3R edge-region eval checkpoint",
    )
    state_dict = payload.get("model", payload)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    model = model.to(device)
    model.eval()
    load_info = {
        "missing_keys": list(missing),
        "unexpected_keys": list(unexpected),
        "step": payload.get("step"),
        "stage": payload.get("stage"),
        "epoch": payload.get("epoch"),
    }
    return model, load_info


def region_masks(batch, pred_depth, gt_depth, gt_valid, edge_quantile, event_quantile):
    target_hw = gt_depth.shape[-2:]
    valid = gt_valid.bool() & torch.isfinite(pred_depth) & torch.isfinite(gt_depth) & (pred_depth > 0) & (gt_depth > 0)

    image = batch["images_photo"][:, 0].float()
    gray = image.mean(dim=1, keepdim=True)
    gray = resize_like(gray, target_hw)
    rgb_grad = sobel_mag(gray)

    event = batch["event_voxel"][:, 0].float().abs().sum(dim=1, keepdim=True)
    event = resize_like(event, target_hw)

    gt_grad = sobel_mag(gt_depth.float())
    rgb_edge = top_quantile_mask(rgb_grad, valid, edge_quantile)
    event_edge = top_quantile_mask(event, valid, event_quantile)
    gt_depth_edge = top_quantile_mask(gt_grad, valid, edge_quantile)
    near, far = median_split_masks(gt_depth, valid)

    masks = {
        "all": valid,
        f"rgb_edge_q{int(edge_quantile * 100):02d}": rgb_edge,
        f"event_active_q{int(event_quantile * 100):02d}": event_edge,
        f"gt_depth_edge_q{int(edge_quantile * 100):02d}": gt_depth_edge,
        f"rgb_or_event_q{int(edge_quantile * 100):02d}": (rgb_edge | event_edge) & valid,
        "near_depth_half": near,
        "far_depth_half": far,
        f"non_gt_depth_edge_q{int(edge_quantile * 100):02d}": valid & (~gt_depth_edge),
    }
    return masks


def evaluate(args):
    cfg = load_config(args.config)
    train_args = make_train_args(args)
    device = resolve_device(args.device)
    set_seed(args.seed)
    dataset = build_dataset(train_args, cfg)
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )
    model, load_info = build_model(train_args, cfg, device)
    amp_enabled, amp_dtype_name, amp_dtype = resolve_amp_config(cfg, train_args, device)
    accumulator = RegionAccumulator()
    sample_ids = []

    with torch.no_grad():
        for batch_idx, batch in enumerate(loader):
            if args.max_batches is not None and batch_idx >= args.max_batches:
                break
            batch = move_batch_to_device(batch, device)
            if "depth_gt" not in batch:
                continue
            with torch.amp.autocast(
                device_type=device.type,
                dtype=amp_dtype,
                enabled=bool(amp_enabled and device.type == "cuda"),
            ):
                pred1, _ = model(batch)
            pred_depth = depth_from_pred(pred1)
            if pred_depth is None:
                continue

            gt_depth = as_bchw(batch["depth_gt"]).to(device=device, dtype=torch.float32)
            default_valid = torch.isfinite(gt_depth) & (gt_depth > 0)
            gt_valid = as_bchw(batch.get("depth_valid", default_valid)).to(device=device).bool()
            pred_depth = resize_like(pred_depth.float(), gt_depth.shape[-2:])

            valid = gt_valid & torch.isfinite(pred_depth) & torch.isfinite(gt_depth) & (pred_depth > 0) & (gt_depth > 0)
            if not valid.any():
                continue
            pred_median = pred_depth[valid].median().clamp_min(1e-6)
            scale = gt_depth[valid].median() / pred_median

            masks = region_masks(
                batch,
                pred_depth,
                gt_depth,
                gt_valid,
                args.edge_quantile,
                args.event_quantile,
            )
            gt_grad = sobel_mag(gt_depth.float())
            raw_pred_grad = sobel_mag(pred_depth.float())
            aligned_depth = pred_depth * scale
            aligned_pred_grad = sobel_mag(aligned_depth.float())

            for region, mask in masks.items():
                accumulator.add(region, "raw", pred_depth, gt_depth, mask, raw_pred_grad, gt_grad)
                accumulator.add(
                    region,
                    "median_aligned",
                    aligned_depth,
                    gt_depth,
                    mask,
                    aligned_pred_grad,
                    gt_grad,
                )

            ids = batch.get("sample_id", None)
            if isinstance(ids, (list, tuple)):
                sample_ids.extend(str(item) for item in ids)
            elif ids is not None:
                sample_ids.append(str(ids))

    rows = accumulator.finalize()
    result = {
        "trained_checkpoint": str(Path(args.trained_checkpoint).resolve()),
        "config": str(Path(args.config).resolve()),
        "data_root": args.data_root,
        "load_info": load_info,
        "num_batches": min(len(loader), args.max_batches) if args.max_batches is not None else len(loader),
        "num_samples_recorded": len(sample_ids),
        "sample_ids": sample_ids,
        "amp": amp_enabled,
        "amp_dtype": amp_dtype_name if amp_enabled else None,
        "edge_quantile": args.edge_quantile,
        "event_quantile": args.event_quantile,
        "metrics": rows,
    }
    return result


def write_outputs(result, output_path):
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w") as f:
        json.dump(result, f, indent=2)

    csv_path = output.with_suffix(".csv")
    rows = result["metrics"]
    fieldnames = [
        "region",
        "alignment",
        "valid_px",
        "grad_px",
        "absrel",
        "mae",
        "rmse",
        "rmse_log",
        "psnr_db",
        "delta_1.25",
        "delta_1.25^2",
        "delta_1.25^3",
        "grad_l1",
        "pred_grad_mean",
        "gt_grad_mean",
        "pred_to_gt_grad_ratio",
    ]
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})

    md_path = output.with_suffix(".md")
    lines = [
        "| Region | Align | AbsRel | RMSE | PSNR | Grad L1 | Pred/GT Grad | Valid px |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            "| {region} | {alignment} | {absrel:.4f} | {rmse:.3f} | {psnr_db:.2f} | "
            "{grad_l1:.3f} | {pred_to_gt_grad_ratio:.3f} | {valid_px} |".format(**row)
        )
    with open(md_path, "w") as f:
        f.write("\n".join(lines) + "\n")


def main():
    args = parse_args()
    result = evaluate(args)
    write_outputs(result, args.output)
    print(json.dumps({"output": str(Path(args.output).resolve()), "metrics": len(result["metrics"])}, indent=2))


if __name__ == "__main__":
    main()
