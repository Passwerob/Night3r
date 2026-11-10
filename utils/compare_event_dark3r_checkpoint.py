#!/usr/bin/env python3
import argparse
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
    cfg_get,
    compute_eval_loss,
    load_backbone,
    load_config,
    make_frozen_teacher,
    make_loader,
    merge_loss_dict,
    resolve_amp_config,
    resolve_stage_schedule,
    safe_torch_load,
    set_event_training_stage,
    should_use_symmetric,
    validate_loss_config,
)


def parse_args():
    parser = argparse.ArgumentParser("Compare trained Event-Dark3R against the pretrained-init baseline")
    parser.add_argument("--trained-checkpoint", required=True)
    parser.add_argument("--config", default="config/train_event_dark3r.yaml")
    parser.add_argument("--data-root", default=None)
    parser.add_argument("--model-name-or-path", required=True)
    parser.add_argument("--allow-unsafe-checkpoint-load", action="store_true")
    parser.add_argument("--image-size", type=int, nargs=2, default=None, metavar=("H", "W"))
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--max-batches", type=int, default=32)
    parser.add_argument("--max-scenes", type=int, default=None)
    parser.add_argument("--max-pairs-per-scene", type=int, default=None)
    parser.add_argument("--pair-stride", type=int, default=None)
    parser.add_argument("--pair-sampling", choices=("fixed", "random", "precomputed"), default=None)
    parser.add_argument("--min-pair-distance", type=int, default=None)
    parser.add_argument("--max-pair-distance", type=int, default=None)
    parser.add_argument("--precomputed-pairs", type=str, default=None)
    parser.add_argument("--allow-reverse-pairs", dest="allow_reverse_pairs", action="store_true", default=None)
    parser.add_argument("--no-allow-reverse-pairs", dest="allow_reverse_pairs", action="store_false")
    parser.add_argument("--symmetric-pairs", dest="symmetric_pairs", action="store_true", default=None)
    parser.add_argument("--no-symmetric-pairs", dest="symmetric_pairs", action="store_false")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--stage", type=int, default=None)
    parser.add_argument("--partial-finetune-blocks", type=int, default=None)
    parser.add_argument("--amp", dest="amp", action="store_true", default=None)
    parser.add_argument("--no-amp", dest="amp", action="store_false")
    parser.add_argument("--amp-dtype", choices=("fp16", "bf16"), default=None)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--depth-psnr-max",
        type=float,
        default=None,
        help="Max depth value for depth PSNR. Defaults to max GT depth observed in the evaluated subset.",
    )
    parser.add_argument(
        "--lowlight-min-exposure",
        type=float,
        default=None,
        help="Override lowlight_aug.min_exposure and force low-light augmentation on.",
    )
    parser.add_argument(
        "--lowlight-max-exposure",
        type=float,
        default=None,
        help="Override lowlight_aug.max_exposure and force low-light augmentation on.",
    )
    parser.add_argument("--output", default=None)
    return parser.parse_args()


def apply_lowlight_overrides(cfg, args):
    cfg = dict(cfg)
    lowlight_cfg = dict(cfg.get("lowlight_aug", {}))
    if args.lowlight_min_exposure is not None:
        lowlight_cfg["enabled"] = True
        lowlight_cfg["min_exposure"] = float(args.lowlight_min_exposure)
    if args.lowlight_max_exposure is not None:
        lowlight_cfg["enabled"] = True
        lowlight_cfg["max_exposure"] = float(args.lowlight_max_exposure)
    cfg["lowlight_aug"] = lowlight_cfg
    return cfg


def make_train_arg_namespace(args):
    return SimpleNamespace(
        config=args.config,
        data_root=args.data_root,
        save_dir=str(Path(args.output).parent if args.output else Path(args.trained_checkpoint).parent),
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
        symmetric_pairs=args.symmetric_pairs,
        num_workers=args.num_workers,
        stage=args.stage,
        partial_finetune_blocks=args.partial_finetune_blocks,
        amp=args.amp,
        amp_dtype=args.amp_dtype,
        gradient_checkpointing=False,
        device=args.device,
        log_every=1,
        no_save=True,
        seed=args.seed,
        depth_psnr_max=args.depth_psnr_max,
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


def as_bchw(x):
    if x is None:
        return None
    if x.ndim == 2:
        return x[None, None]
    if x.ndim == 3:
        return x[:, None]
    if x.ndim == 4:
        return x if x.shape[1] == 1 else x[:, :1]
    return None


def depth_from_pred(pred):
    if pred is None:
        return None
    if "depth" in pred:
        return as_bchw(pred["depth"])
    key = "pts3d" if "pts3d" in pred else "pts3d_in_other_view"
    if key not in pred:
        return None
    pts = pred[key]
    if pts.ndim != 4 or pts.shape[-1] < 3:
        return None
    return pts[..., 2].unsqueeze(1)


def warp_second_to_first(image_j, depth_i, intrinsics, relative_pose):
    b, _, h, w = depth_i.shape
    device = depth_i.device
    dtype = depth_i.dtype
    ys, xs = torch.meshgrid(
        torch.arange(h, device=device, dtype=dtype),
        torch.arange(w, device=device, dtype=dtype),
        indexing="ij",
    )
    pix = torch.stack([xs, ys, torch.ones_like(xs)], dim=0).view(1, 3, -1).repeat(b, 1, 1)
    intrinsics = intrinsics.to(device=device, dtype=dtype)
    relative_pose = relative_pose.to(device=device, dtype=dtype)
    xyz_i = torch.inverse(intrinsics) @ pix * depth_i.view(b, 1, -1)
    xyz_i_h = torch.cat([xyz_i, torch.ones(b, 1, xyz_i.shape[-1], device=device, dtype=dtype)], dim=1)
    xyz_j = relative_pose @ xyz_i_h
    z = xyz_j[:, 2:3]
    proj = intrinsics @ xyz_j[:, :3]
    uv = proj[:, :2] / z.clamp_min(1e-6)
    x_norm = 2.0 * uv[:, 0] / max(w - 1, 1) - 1.0
    y_norm = 2.0 * uv[:, 1] / max(h - 1, 1) - 1.0
    grid = torch.stack([x_norm, y_norm], dim=-1).view(b, h, w, 2)
    warped = F.grid_sample(image_j, grid, mode="bilinear", padding_mode="zeros", align_corners=True)
    valid = (
        (z.view(b, 1, h, w) > 1e-6)
        & (grid[..., 0].abs().unsqueeze(1) <= 1)
        & (grid[..., 1].abs().unsqueeze(1) <= 1)
    )
    return warped, valid


class MetricAccumulator:
    def __init__(self):
        self.values = {}

    def add(self, key, value, weight=1.0):
        if value is None:
            return
        if not math.isfinite(float(value)):
            return
        total, count = self.values.get(key, (0.0, 0.0))
        self.values[key] = (total + float(value) * float(weight), count + float(weight))

    def mean(self, key):
        total, count = self.values.get(key, (0.0, 0.0))
        if count <= 0:
            return float("nan")
        return total / count

    def finalize(self):
        return {key: total / count for key, (total, count) in sorted(self.values.items()) if count > 0}


class DenseMetricAccumulator:
    def __init__(self):
        self.depth = {
            "raw": self._empty_depth_stats(),
            "median_aligned": self._empty_depth_stats(),
        }
        self.photo_sq_sum = 0.0
        self.photo_count = 0
        self.depth_gt_max = 0.0

    @staticmethod
    def _empty_depth_stats():
        return {
            "valid_px": 0,
            "gt_valid_px": 0,
            "abs_rel_sum": 0.0,
            "sq_sum": 0.0,
            "abs_sum": 0.0,
            "log_sum": 0.0,
            "log_sq_sum": 0.0,
            "delta_1_count": 0,
            "delta_2_count": 0,
            "delta_3_count": 0,
        }

    def add_depth(self, pred_depth, gt_depth, gt_valid, variant="raw"):
        stats = self.depth[variant]
        pred_depth = pred_depth.float()
        gt_depth = gt_depth.float()
        gt_valid = gt_valid.bool()
        if pred_depth.shape[-2:] != gt_depth.shape[-2:]:
            pred_depth = F.interpolate(pred_depth, size=gt_depth.shape[-2:], mode="bilinear", align_corners=False)
        gt_mask = gt_valid & torch.isfinite(gt_depth) & (gt_depth > 0)
        valid = gt_mask & torch.isfinite(pred_depth) & (pred_depth > 0)
        stats["gt_valid_px"] += int(gt_mask.sum().item())
        if not valid.any():
            return

        pred = pred_depth[valid]
        gt = gt_depth[valid]
        self.depth_gt_max = max(self.depth_gt_max, float(gt.max().detach().cpu()))
        if variant == "median_aligned":
            pred_median = pred.median().clamp_min(1e-6)
            scale = gt.median() / pred_median
            pred = pred * scale

        pred = pred.clamp_min(1e-6)
        gt = gt.clamp_min(1e-6)
        diff = pred - gt
        abs_diff = diff.abs()
        log_diff = torch.log(pred) - torch.log(gt)
        ratio = torch.maximum(pred / gt, gt / pred)
        n = int(gt.numel())
        stats["valid_px"] += n
        stats["abs_rel_sum"] += float((abs_diff / gt).sum().detach().cpu())
        stats["sq_sum"] += float((diff * diff).sum().detach().cpu())
        stats["abs_sum"] += float(abs_diff.sum().detach().cpu())
        stats["log_sum"] += float(log_diff.sum().detach().cpu())
        stats["log_sq_sum"] += float((log_diff * log_diff).sum().detach().cpu())
        stats["delta_1_count"] += int((ratio < 1.25).sum().item())
        stats["delta_2_count"] += int((ratio < 1.25**2).sum().item())
        stats["delta_3_count"] += int((ratio < 1.25**3).sum().item())

    def add_photo_psnr(self, pred_depth, batch):
        if "images_photo" not in batch or "intrinsics" not in batch or "relative_pose" not in batch:
            return
        images = batch["images_photo"].to(device=pred_depth.device, dtype=torch.float32)
        if images.ndim != 5 or images.shape[1] < 2:
            return
        image_i = images[:, 0]
        image_j = images[:, 1]
        depth = pred_depth.float()
        if depth.shape[-2:] != image_i.shape[-2:]:
            depth = F.interpolate(depth, size=image_i.shape[-2:], mode="bilinear", align_corners=False)
        valid_depth = torch.isfinite(depth) & (depth > 0)
        warped_j, valid_warp = warp_second_to_first(
            image_j,
            depth,
            batch["intrinsics"],
            batch["relative_pose"],
        )
        valid = valid_depth & valid_warp & torch.isfinite(warped_j) & torch.isfinite(image_i)
        if not valid.any():
            return
        valid_rgb = valid.expand_as(image_i)
        diff = warped_j[valid_rgb] - image_i[valid_rgb]
        self.photo_sq_sum += float((diff * diff).sum().detach().cpu())
        self.photo_count += int(diff.numel())

    def finalize(self, depth_psnr_max=None):
        out = {}
        max_depth = float(depth_psnr_max or self.depth_gt_max)
        for variant, stats in self.depth.items():
            n = stats["valid_px"]
            prefix = f"depth_{variant}"
            out[f"{prefix}_valid_px"] = n
            out[f"{prefix}_coverage"] = n / max(stats["gt_valid_px"], 1)
            if n <= 0:
                continue
            mse = stats["sq_sum"] / n
            mae = stats["abs_sum"] / n
            mean_log = stats["log_sum"] / n
            mean_log_sq = stats["log_sq_sum"] / n
            silog = 10.0 * math.sqrt(max(mean_log_sq - mean_log * mean_log + 0.15 * mean_log * mean_log, 1e-12))
            out[f"{prefix}_absrel"] = stats["abs_rel_sum"] / n
            out[f"{prefix}_mae"] = mae
            out[f"{prefix}_rmse"] = math.sqrt(max(mse, 0.0))
            out[f"{prefix}_rmse_log"] = math.sqrt(max(mean_log_sq, 0.0))
            out[f"{prefix}_silog"] = silog
            out[f"{prefix}_delta_1.25"] = stats["delta_1_count"] / n
            out[f"{prefix}_delta_1.25^2"] = stats["delta_2_count"] / n
            out[f"{prefix}_delta_1.25^3"] = stats["delta_3_count"] / n
            if max_depth > 0 and mse > 0:
                out[f"{prefix}_psnr_db"] = 20.0 * math.log10(max_depth) - 10.0 * math.log10(mse)
        out["depth_psnr_max"] = max_depth
        if self.photo_count > 0:
            photo_mse = self.photo_sq_sum / self.photo_count
            out["warp_rgb_psnr_db"] = -10.0 * math.log10(max(photo_mse, 1e-12))
            out["warp_rgb_mse"] = photo_mse
            out["warp_rgb_valid_values"] = self.photo_count
        return out


def build_model_bundle(args, cfg, device, trained_checkpoint=None):
    backbone = load_backbone(args, cfg)
    teacher = make_frozen_teacher(backbone).to(device)
    student = build_event_student(backbone, config=cfg).to(device)
    stage_schedule = resolve_stage_schedule(cfg, args)
    final_stage = stage_schedule[-1]
    set_event_training_stage(
        student,
        stage=final_stage["stage"],
        partial_finetune_blocks=final_stage["partial_finetune_blocks"],
    )
    load_info = {"missing_keys": [], "unexpected_keys": [], "step": None, "stage": None, "epoch": None}
    if trained_checkpoint is not None:
        ckpt = safe_torch_load(
            trained_checkpoint,
            map_location="cpu",
            allow_unsafe=args.allow_unsafe_checkpoint_load,
            purpose="event_dark3r checkpoint",
        )
        missing, unexpected = student.load_state_dict(ckpt["model"], strict=False)
        load_info = {
            "missing_keys": list(missing),
            "unexpected_keys": list(unexpected),
            "step": ckpt.get("step"),
            "stage": ckpt.get("stage"),
            "epoch": ckpt.get("epoch"),
        }
    student.eval()
    teacher.eval()
    return student, teacher, load_info


def evaluate_one_model(label, args, cfg, device, trained_checkpoint=None):
    set_seed(args.seed)
    dataset = build_dataset(args, cfg)
    validate_loss_config(cfg, dataset, None)

    set_seed(args.seed)
    loader, _ = make_loader(
        dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        device=device,
        distributed=False,
        shuffle=False,
    )
    student, teacher, load_info = build_model_bundle(args, cfg, device, trained_checkpoint)
    amp_enabled, amp_dtype_name, amp_dtype = resolve_amp_config(cfg, args, device)
    symmetric = should_use_symmetric(cfg, args)
    loss_total = 0.0
    num_batches = 0
    loss_accum = {}
    dense_metrics = DenseMetricAccumulator()
    sample_ids = []

    with torch.no_grad():
        for batch_idx, batch in enumerate(loader):
            if args.max_batches is not None and batch_idx >= args.max_batches:
                break
            batch = {
                key: value.to(device, non_blocking=True) if torch.is_tensor(value) else value
                for key, value in batch.items()
            }
            loss, loss_dict, student_out = compute_eval_loss(
                student,
                teacher,
                batch,
                cfg.get("loss", {}),
                symmetric=symmetric,
                amp_enabled=amp_enabled,
                amp_dtype=amp_dtype,
                device_type=device.type,
            )
            loss_total += float(loss.detach().cpu())
            num_batches += 1
            merge_loss_dict(loss_accum, loss_dict)

            pred_depth = depth_from_pred(student_out[0])
            if pred_depth is not None and "depth_gt" in batch:
                gt_depth = as_bchw(batch["depth_gt"].to(device=pred_depth.device, dtype=pred_depth.dtype))
                gt_valid = as_bchw(batch.get("depth_valid", torch.isfinite(gt_depth) & (gt_depth > 0)).to(device=pred_depth.device))
                dense_metrics.add_depth(pred_depth, gt_depth, gt_valid, variant="raw")
                dense_metrics.add_depth(pred_depth, gt_depth, gt_valid, variant="median_aligned")
                dense_metrics.add_photo_psnr(pred_depth, batch)

            batch_sample_ids = batch.get("sample_id", None)
            if isinstance(batch_sample_ids, (list, tuple)):
                sample_ids.extend(str(item) for item in batch_sample_ids)
            elif batch_sample_ids is not None:
                sample_ids.append(str(batch_sample_ids))

    loss_metrics = {}
    for key, value in sorted(loss_accum.items()):
        if isinstance(value, tuple):
            total, count = value
            loss_metrics[key] = total / max(count, 1)
        else:
            loss_metrics[key] = value
    result = {
        "label": label,
        "dataset_len": len(dataset),
        "num_batches": num_batches,
        "batch_size": args.batch_size,
        "sample_ids": sample_ids,
        "eval_loss": loss_total / max(num_batches, 1),
        "eval_loss_dict": loss_metrics,
        "dense_metrics": dense_metrics.finalize(depth_psnr_max=args.depth_psnr_max),
        "load_info": load_info,
        "symmetric_pairs": symmetric,
        "amp": amp_enabled,
        "amp_dtype": amp_dtype_name if amp_enabled else None,
    }
    del student, teacher, dataset, loader
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return result


def metric_delta(baseline, trained):
    out = {}
    base_sources = {
        "eval_loss": baseline.get("eval_loss"),
        **{f"loss/{k}": v for k, v in baseline.get("eval_loss_dict", {}).items()},
        **{f"dense/{k}": v for k, v in baseline.get("dense_metrics", {}).items()},
    }
    trained_sources = {
        "eval_loss": trained.get("eval_loss"),
        **{f"loss/{k}": v for k, v in trained.get("eval_loss_dict", {}).items()},
        **{f"dense/{k}": v for k, v in trained.get("dense_metrics", {}).items()},
    }
    for key, base_value in base_sources.items():
        trained_value = trained_sources.get(key)
        if isinstance(base_value, (int, float)) and isinstance(trained_value, (int, float)):
            diff = float(trained_value) - float(base_value)
            item = {"trained_minus_baseline": diff}
            if abs(float(base_value)) > 1e-12:
                item["relative_change"] = diff / abs(float(base_value))
            out[key] = item
    return out


def config_data_root_text(cfg, data_root):
    root = data_root if data_root is not None else cfg_get(cfg, "dataset", "root", "")
    if isinstance(root, (list, tuple)):
        return ", ".join(str(item) for item in root)
    return str(root)


def main():
    cli_args = parse_args()
    cfg = load_config(cli_args.config)
    cfg = apply_lowlight_overrides(cfg, cli_args)
    train_args = make_train_arg_namespace(cli_args)
    device = resolve_device(cli_args.device)
    set_seed(cli_args.seed)

    baseline = evaluate_one_model(
        "pretrained_init_baseline",
        train_args,
        cfg,
        device,
        trained_checkpoint=None,
    )
    trained = evaluate_one_model(
        "trained_event_dark3r",
        train_args,
        cfg,
        device,
        trained_checkpoint=cli_args.trained_checkpoint,
    )
    sample_ids_match = baseline["sample_ids"] == trained["sample_ids"]
    result = {
        "baseline_model": str(Path(cli_args.model_name_or_path).resolve()),
        "trained_checkpoint": str(Path(cli_args.trained_checkpoint).resolve()),
        "config": str(Path(cli_args.config).resolve()),
        "data_root": config_data_root_text(cfg, cli_args.data_root),
        "device": str(device),
        "seed": cli_args.seed,
        "max_batches": cli_args.max_batches,
        "sample_ids_match": sample_ids_match,
        "lowlight_aug": cfg.get("lowlight_aug", {}),
        "baseline": baseline,
        "trained": trained,
        "delta": metric_delta(baseline, trained),
    }
    print(json.dumps(result, indent=2, sort_keys=True))
    if cli_args.output:
        output = Path(cli_args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        with open(output, "w") as f:
            json.dump(result, f, indent=2, sort_keys=True)


if __name__ == "__main__":
    main()
