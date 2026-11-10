#!/usr/bin/env python3
import argparse
import copy
import json
import math
import os
import sys
from pathlib import Path
from types import SimpleNamespace

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from dataset.lowlight_augmentation import ContinuousLowLightAug  # noqa: E402
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
from utils.compare_event_dark3r_checkpoint import (  # noqa: E402
    as_bchw,
    depth_from_pred,
    set_seed,
    warp_second_to_first,
)


def parse_args():
    parser = argparse.ArgumentParser("Compare two Event-Dark3R checkpoints with different model inputs")
    parser.add_argument("--old-config", required=True)
    parser.add_argument("--old-checkpoint", required=True)
    parser.add_argument("--old-label", default="NoAug 6k")
    parser.add_argument(
        "--old-input-mode",
        choices=("darken", "original"),
        default="original",
        help="Use darkened input for the old checkpoint, or the original dataset input.",
    )
    parser.add_argument("--new-config", required=True)
    parser.add_argument("--new-checkpoint", required=True)
    parser.add_argument("--new-label", default="DarkAug+DL3DV 10k")
    parser.add_argument(
        "--new-input-mode",
        choices=("darken", "original"),
        default="darken",
        help="Use darkened input for the new checkpoint, or the same original input as the old checkpoint.",
    )
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--model-name-or-path", required=True)
    parser.add_argument("--allow-unsafe-checkpoint-load", action="store_true")
    parser.add_argument("--image-size", type=int, nargs=2, default=None, metavar=("H", "W"))
    parser.add_argument("--max-pairs", type=int, default=8)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--max-scenes", type=int, default=None)
    parser.add_argument("--max-pairs-per-scene", type=int, default=None)
    parser.add_argument("--pair-stride", type=int, default=None)
    parser.add_argument("--pair-sampling", choices=("fixed", "random", "precomputed"), default=None)
    parser.add_argument("--min-pair-distance", type=int, default=None)
    parser.add_argument("--max-pair-distance", type=int, default=None)
    parser.add_argument("--precomputed-pairs", type=str, default=None)
    parser.add_argument("--allow-reverse-pairs", dest="allow_reverse_pairs", action="store_true", default=None)
    parser.add_argument("--no-allow-reverse-pairs", dest="allow_reverse_pairs", action="store_false")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--amp", dest="amp", action="store_true", default=None)
    parser.add_argument("--no-amp", dest="amp", action="store_false")
    parser.add_argument("--amp-dtype", choices=("fp16", "bf16"), default=None)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--darken-seed-offset", type=int, default=10000)
    parser.add_argument("--outdir", required=True)
    return parser.parse_args()


def cfg_with_lowlight(cfg, enabled):
    cfg = dict(cfg)
    lowlight_cfg = dict(cfg.get("lowlight_aug", {}))
    lowlight_cfg["enabled"] = bool(enabled)
    cfg["lowlight_aug"] = lowlight_cfg
    return cfg


def make_train_args(args, config, outdir):
    return SimpleNamespace(
        config=config,
        data_root=args.data_root,
        save_dir=outdir,
        model_name_or_path=args.model_name_or_path,
        allow_unsafe_checkpoint_load=args.allow_unsafe_checkpoint_load,
        use_tiny_model=False,
        image_size=args.image_size,
        batch_size=args.batch_size,
        max_pairs=args.max_pairs,
        start_index=args.start_index,
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


def cache_base_batches(args, cfg):
    set_seed(args.seed)
    dataset = build_dataset(args, cfg)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=False,
    )
    batches = []
    start_index = max(int(getattr(args, "start_index", 0) or 0), 0)
    for batch_idx, batch in enumerate(loader):
        if batch_idx < start_index:
            continue
        batches.append({key: value.detach().cpu() if torch.is_tensor(value) else value for key, value in batch.items()})
        if len(batches) >= args.max_pairs:
            break
    return dataset, batches


def clone_batch(batch):
    out = {}
    for key, value in batch.items():
        if torch.is_tensor(value):
            out[key] = value.clone()
        else:
            out[key] = copy.deepcopy(value)
    return out


def make_darkened_batches(base_batches, cfg, seed_offset):
    lowlight_cfg = dict(cfg.get("lowlight_aug", {}))
    lowlight_cfg.pop("enabled", None)
    lowlight_cfg.pop("apply_sources", None)
    aug = ContinuousLowLightAug(**lowlight_cfg)
    out = []
    for idx, batch in enumerate(base_batches):
        item = clone_batch(batch)
        torch.manual_seed(int(seed_offset) + idx)
        images_photo_dark = aug(item["images_photo"])
        item["images_photo_dark"] = images_photo_dark
        item["images_dark"] = images_photo_dark * 2.0 - 1.0
        item["darken_metadata"] = dict(getattr(aug, "last_metadata", {}))
        out.append(item)
    return out


def build_model(args, cfg, device, checkpoint):
    backbone = load_backbone(args, cfg)
    model = build_event_student(backbone, config=cfg)
    final_stage = resolve_stage_schedule(cfg, args)[-1]
    set_event_training_stage(
        model,
        stage=final_stage["stage"],
        partial_finetune_blocks=final_stage["partial_finetune_blocks"],
    )
    payload = safe_torch_load(
        checkpoint,
        map_location="cpu",
        allow_unsafe=args.allow_unsafe_checkpoint_load,
        purpose="Event-Dark3R comparison checkpoint",
    )
    state_dict = payload.get("model", payload)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    return model.to(device).eval(), {
        "checkpoint": str(Path(checkpoint).resolve()),
        "step": payload.get("step") if isinstance(payload, dict) else None,
        "stage": payload.get("stage") if isinstance(payload, dict) else None,
        "epoch": payload.get("epoch") if isinstance(payload, dict) else None,
        "missing_keys": list(missing),
        "unexpected_keys": list(unexpected),
    }


def resize_depth(depth, shape):
    if depth.shape[-2:] == shape:
        return depth
    return F.interpolate(depth, size=shape, mode="bilinear", align_corners=False)


def predict(model, batches, args, cfg, device):
    amp_enabled, _, amp_dtype = resolve_amp_config(cfg, args, device)
    outputs = []
    with torch.no_grad():
        for batch in batches:
            batch_gpu = move_batch_to_device(batch, device)
            with torch.amp.autocast(
                device_type=device.type,
                dtype=amp_dtype,
                enabled=bool(amp_enabled and device.type == "cuda"),
            ):
                pred1, _ = model(batch_gpu)
            depth = depth_from_pred(pred1)
            if depth is None:
                raise RuntimeError("Model output does not contain depth/pts3d")
            warped = None
            warp_valid = None
            if "images_photo_dark" in batch_gpu and "intrinsics" in batch_gpu and "relative_pose" in batch_gpu:
                image2 = batch_gpu["images_photo_dark"][:, 1].to(dtype=torch.float32)
                pred_depth = resize_depth(depth.to(dtype=torch.float32), image2.shape[-2:])
                warped, warp_valid = warp_second_to_first(
                    image2,
                    pred_depth,
                    batch_gpu["intrinsics"],
                    batch_gpu["relative_pose"],
                )
            outputs.append(
                {
                    "depth": depth.detach().cpu(),
                    "warped": None if warped is None else warped.detach().cpu(),
                    "warp_valid": None if warp_valid is None else warp_valid.detach().cpu(),
                }
            )
    return outputs


def image_hwc(image):
    return image.detach().cpu().float().clamp(0, 1).permute(1, 2, 0).numpy()


def finite_percentile(values, percentile, default=0.0):
    values = np.asarray(values, dtype=np.float32)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return float(default)
    return float(np.percentile(values, percentile))


def depth_metrics(pred_depth, gt_depth, valid):
    pred_depth = resize_depth(pred_depth.float(), gt_depth.shape[-2:])
    valid = valid.bool() & torch.isfinite(pred_depth) & torch.isfinite(gt_depth) & (pred_depth > 0) & (gt_depth > 0)
    if not valid.any():
        return {}
    pred = pred_depth[valid].clamp_min(1e-6)
    gt = gt_depth[valid].clamp_min(1e-6)
    diff = pred - gt
    mse = float((diff * diff).mean().detach().cpu())
    ratio = torch.maximum(pred / gt, gt / pred)
    max_depth = float(gt.max().detach().cpu())
    return {
        "absrel": float((diff.abs() / gt).mean().detach().cpu()),
        "rmse": math.sqrt(max(mse, 0.0)),
        "psnr_db": 20.0 * math.log10(max_depth) - 10.0 * math.log10(max(mse, 1e-12)),
        "delta_1.25": float((ratio < 1.25).float().mean().detach().cpu()),
        "valid_px": int(valid.sum().item()),
    }


def warp_psnr(warped, target, valid):
    if warped is None or valid is None:
        return float("nan")
    valid = valid.bool().expand_as(target) & torch.isfinite(warped) & torch.isfinite(target)
    if not valid.any():
        return float("nan")
    mse = float(((warped[valid].float() - target[valid].float()) ** 2).mean().detach().cpu())
    return -10.0 * math.log10(max(mse, 1e-12))


def sample_id_from_batch(batch, fallback):
    sample_id = batch.get("sample_id", [f"sample_{fallback:03d}"])
    if isinstance(sample_id, (list, tuple)):
        return str(sample_id[0])
    return str(sample_id)


def render_panel(out_path, idx, old_batch, new_batch, old_pred, new_pred, old_label, new_label):
    night1 = old_batch["images_photo"][0, 0]
    night2 = old_batch["images_photo"][0, 1]
    old_input1 = old_batch["images_photo_dark"][0, 0]
    old_input2 = old_batch["images_photo_dark"][0, 1]
    new_input1 = new_batch["images_photo_dark"][0, 0]
    new_input2 = new_batch["images_photo_dark"][0, 1]
    event = old_batch["event_voxel"][0, 0].abs().sum(dim=0).float().numpy()

    gt_depth = as_bchw(old_batch["depth_gt"])[0:1]
    gt_valid = as_bchw(old_batch.get("depth_valid", torch.isfinite(gt_depth) & (gt_depth > 0)))[0:1].bool()
    gt_depth_2d = gt_depth[0, 0]
    valid_2d = gt_valid[0, 0].numpy()
    old_depth = resize_depth(old_pred["depth"][0:1], gt_depth.shape[-2:])[0, 0]
    new_depth = resize_depth(new_pred["depth"][0:1], gt_depth.shape[-2:])[0, 0]
    old_err = (old_depth - gt_depth_2d).abs()
    new_err = (new_depth - gt_depth_2d).abs()
    improvement = (old_err - new_err).numpy()

    if valid_2d.any():
        depth_vmin = finite_percentile(gt_depth_2d.numpy()[valid_2d], 2)
        depth_vmax = finite_percentile(gt_depth_2d.numpy()[valid_2d], 98, default=depth_vmin + 1.0)
        err_values = np.concatenate([old_err.numpy()[valid_2d], new_err.numpy()[valid_2d]])
        err_vmax = finite_percentile(err_values, 95, default=1.0)
        imp_abs = finite_percentile(np.abs(improvement[valid_2d]), 98, default=1.0)
    else:
        depth_vmin, depth_vmax, err_vmax, imp_abs = 0.0, 1.0, 1.0, 1.0
    if depth_vmax <= depth_vmin:
        depth_vmax = depth_vmin + 1e-6
    err_vmax = max(err_vmax, 1e-6)
    imp_abs = max(imp_abs, 1e-6)

    old_metrics = depth_metrics(old_depth[None, None], gt_depth, gt_valid)
    new_metrics = depth_metrics(new_depth[None, None], gt_depth, gt_valid)
    old_warp_psnr = warp_psnr(old_pred["warped"], old_batch["images_photo_dark"][:, 0], old_pred["warp_valid"])
    new_warp_psnr = warp_psnr(new_pred["warped"], new_batch["images_photo_dark"][:, 0], new_pred["warp_valid"])

    sample_id = sample_id_from_batch(old_batch, idx)
    exposure = new_batch.get("darken_metadata", {}).get("exposure_scale", None)
    exposure_text = "" if exposure is None else f"; new exposure {float(exposure):.3f}"

    fig, axes = plt.subplots(4, 4, figsize=(20, 16), constrained_layout=True)
    fig.suptitle(f"{sample_id}{exposure_text}", fontsize=14)
    panels = [
        (axes[0, 0], image_hwc(night1), "Original night view1", None, None, None),
        (axes[0, 1], image_hwc(old_input1), f"{old_label} input view1", None, None, None),
        (axes[0, 2], image_hwc(new_input1), f"{new_label} input view1", None, None, None),
        (axes[0, 3], event, "Event activity view1", "magma", None, None),
        (axes[1, 0], image_hwc(night2), "Original night view2", None, None, None),
        (axes[1, 1], image_hwc(old_input2), f"{old_label} input view2", None, None, None),
        (axes[1, 2], image_hwc(new_input2), f"{new_label} input view2", None, None, None),
        (axes[1, 3], gt_depth_2d.numpy(), "GT depth", "viridis", depth_vmin, depth_vmax),
        (
            axes[2, 0],
            old_depth.numpy(),
            f"{old_label} depth\nAbsRel {old_metrics.get('absrel', float('nan')):.3f}",
            "viridis",
            depth_vmin,
            depth_vmax,
        ),
        (
            axes[2, 1],
            new_depth.numpy(),
            f"{new_label} depth\nAbsRel {new_metrics.get('absrel', float('nan')):.3f}",
            "viridis",
            depth_vmin,
            depth_vmax,
        ),
        (
            axes[2, 2],
            old_err.numpy(),
            f"{old_label} abs error\nRMSE {old_metrics.get('rmse', float('nan')):.2f}",
            "inferno",
            0.0,
            err_vmax,
        ),
        (
            axes[2, 3],
            new_err.numpy(),
            f"{new_label} abs error\nRMSE {new_metrics.get('rmse', float('nan')):.2f}",
            "inferno",
            0.0,
            err_vmax,
        ),
        (
            axes[3, 0],
            improvement,
            "Abs error improvement\npositive = new better",
            "coolwarm",
            -imp_abs,
            imp_abs,
        ),
        (
            axes[3, 1],
            image_hwc(old_pred["warped"][0]) if old_pred["warped"] is not None else np.zeros_like(image_hwc(night1)),
            f"{old_label} input warp\nPSNR {old_warp_psnr:.2f} dB",
            None,
            None,
            None,
        ),
        (
            axes[3, 2],
            image_hwc(new_pred["warped"][0]) if new_pred["warped"] is not None else np.zeros_like(image_hwc(night1)),
            f"{new_label} input warp\nPSNR {new_warp_psnr:.2f} dB",
            None,
            None,
            None,
        ),
        (axes[3, 3], image_hwc(night1), "Original night reference", None, None, None),
    ]

    for ax, data, title, cmap, vmin, vmax in panels:
        if cmap is None:
            ax.imshow(np.clip(data, 0, 1))
        else:
            shown = ax.imshow(data, cmap=cmap, vmin=vmin, vmax=vmax)
            fig.colorbar(shown, ax=ax, fraction=0.046, pad=0.02)
        ax.set_title(title, fontsize=10)
        ax.axis("off")
    fig.savefig(out_path, dpi=140)
    plt.close(fig)
    return {
        "sample_id": sample_id,
        "panel": str(out_path),
        "new_darken_metadata": new_batch.get("darken_metadata", {}),
        "old_depth": old_metrics,
        "new_depth": new_metrics,
        "old_input_warp_psnr_db": old_warp_psnr,
        "new_input_warp_psnr_db": new_warp_psnr,
    }


def depth_stats(depth):
    values = depth.detach().cpu().float().numpy()
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return {"p02": None, "median": None, "p98": None}
    return {
        "p02": float(np.percentile(finite, 2)),
        "median": float(np.median(finite)),
        "p98": float(np.percentile(finite, 98)),
    }


def render_panel_no_gt(out_path, idx, old_batch, new_batch, old_pred, new_pred, old_label, new_label):
    night1 = old_batch["images_photo"][0, 0]
    night2 = old_batch["images_photo"][0, 1]
    old_input1 = old_batch["images_photo_dark"][0, 0]
    old_input2 = old_batch["images_photo_dark"][0, 1]
    new_input1 = new_batch["images_photo_dark"][0, 0]
    new_input2 = new_batch["images_photo_dark"][0, 1]
    event = old_batch["event_voxel"][0, 0].abs().sum(dim=0).float().numpy()

    target_shape = night1.shape[-2:]
    old_depth = resize_depth(old_pred["depth"][0:1], target_shape)[0, 0]
    new_depth = resize_depth(new_pred["depth"][0:1], target_shape)[0, 0]
    old_np = old_depth.numpy()
    new_np = new_depth.numpy()
    all_depth = np.concatenate([old_np.reshape(-1), new_np.reshape(-1)])
    depth_vmin = finite_percentile(all_depth, 2, default=0.0)
    depth_vmax = finite_percentile(all_depth, 98, default=1.0)
    if depth_vmax <= depth_vmin:
        depth_vmax = depth_vmin + 1e-6
    diff = new_np - old_np
    diff_abs = finite_percentile(np.abs(diff), 98, default=1.0)
    diff_abs = max(diff_abs, 1e-6)

    old_warp_psnr = warp_psnr(old_pred["warped"], old_batch["images_photo_dark"][:, 0], old_pred["warp_valid"])
    new_warp_psnr = warp_psnr(new_pred["warped"], new_batch["images_photo_dark"][:, 0], new_pred["warp_valid"])
    sample_id = sample_id_from_batch(old_batch, idx)
    exposure = new_batch.get("darken_metadata", {}).get("exposure_scale", None)
    exposure_text = "" if exposure is None else f"; new exposure {float(exposure):.3f}"
    old_stats = depth_stats(old_depth)
    new_stats = depth_stats(new_depth)

    fig, axes = plt.subplots(4, 4, figsize=(20, 16), constrained_layout=True)
    fig.suptitle(f"{sample_id}{exposure_text} (DL3DV qualitative, no GT depth)", fontsize=14)
    panels = [
        (axes[0, 0], image_hwc(night1), "Original DL3DV view1", None, None, None),
        (axes[0, 1], image_hwc(old_input1), f"{old_label} input view1", None, None, None),
        (axes[0, 2], image_hwc(new_input1), f"{new_label} input view1", None, None, None),
        (axes[0, 3], event, "Event activity view1", "magma", None, None),
        (axes[1, 0], image_hwc(night2), "Original DL3DV view2", None, None, None),
        (axes[1, 1], image_hwc(old_input2), f"{old_label} input view2", None, None, None),
        (axes[1, 2], image_hwc(new_input2), f"{new_label} input view2", None, None, None),
        (axes[1, 3], image_hwc(night1), "Reference view1", None, None, None),
        (
            axes[2, 0],
            old_np,
            f"{old_label} depth\nmedian {old_stats['median']:.2f}",
            "viridis",
            depth_vmin,
            depth_vmax,
        ),
        (
            axes[2, 1],
            new_np,
            f"{new_label} depth\nmedian {new_stats['median']:.2f}",
            "viridis",
            depth_vmin,
            depth_vmax,
        ),
        (
            axes[2, 2],
            diff,
            f"{new_label} - {old_label} depth",
            "coolwarm",
            -diff_abs,
            diff_abs,
        ),
        (axes[2, 3], image_hwc(new_input1), "New darkened input reference", None, None, None),
        (
            axes[3, 0],
            image_hwc(old_pred["warped"][0]) if old_pred["warped"] is not None else np.zeros_like(image_hwc(night1)),
            f"{old_label} input warp\nPSNR {old_warp_psnr:.2f} dB",
            None,
            None,
            None,
        ),
        (
            axes[3, 1],
            image_hwc(new_pred["warped"][0]) if new_pred["warped"] is not None else np.zeros_like(image_hwc(night1)),
            f"{new_label} input warp\nPSNR {new_warp_psnr:.2f} dB",
            None,
            None,
            None,
        ),
        (axes[3, 2], image_hwc(old_input1), f"{old_label} input view1", None, None, None),
        (axes[3, 3], image_hwc(new_input1), f"{new_label} input view1", None, None, None),
    ]

    for ax, data, title, cmap, vmin, vmax in panels:
        if cmap is None:
            ax.imshow(np.clip(data, 0, 1))
        else:
            shown = ax.imshow(data, cmap=cmap, vmin=vmin, vmax=vmax)
            fig.colorbar(shown, ax=ax, fraction=0.046, pad=0.02)
        ax.set_title(title, fontsize=10)
        ax.axis("off")
    fig.savefig(out_path, dpi=140)
    plt.close(fig)
    return {
        "sample_id": sample_id,
        "panel": str(out_path),
        "new_darken_metadata": new_batch.get("darken_metadata", {}),
        "old_depth": {},
        "new_depth": {},
        "old_depth_stats": old_stats,
        "new_depth_stats": new_stats,
        "old_input_warp_psnr_db": old_warp_psnr,
        "new_input_warp_psnr_db": new_warp_psnr,
    }


def write_index(outdir, records, old_label, new_label):
    html = [
        "<!doctype html><meta charset='utf-8'>",
        "<title>Event-Dark3R Dual Input Comparison</title>",
        "<style>body{font-family:Arial,sans-serif;margin:24px;background:#f6f6f6;color:#111}"
        "img{max-width:100%;display:block;margin:12px 0 36px;border:1px solid #ccc;background:white}</style>",
        "<h1>Event-Dark3R Dual Input Comparison</h1>",
        f"<p>{old_label}: original night input. {new_label}: night input darkened again.</p>",
    ]
    for item in records:
        rel = Path(item["panel"]).name
        old_abs = item["old_depth"].get("absrel")
        new_abs = item["new_depth"].get("absrel")
        exposure = item.get("new_darken_metadata", {}).get("exposure_scale", float("nan"))
        html.append(f"<h2>{item['sample_id']}</h2>")
        if old_abs is None or new_abs is None:
            old_med = item.get("old_depth_stats", {}).get("median", float("nan"))
            new_med = item.get("new_depth_stats", {}).get("median", float("nan"))
            html.append(
                "<p>"
                "No GT depth available. "
                f"Pred depth median: {old_label} {old_med:.3f}, {new_label} {new_med:.3f}. "
                f"Input-warp PSNR: {old_label} {item['old_input_warp_psnr_db']:.2f} dB, "
                f"{new_label} {item['new_input_warp_psnr_db']:.2f} dB. "
                f"New exposure: {exposure:.3f}."
                "</p>"
            )
        else:
            html.append(
                "<p>"
                f"AbsRel: {old_label} {old_abs:.3f}, {new_label} {new_abs:.3f}. "
                f"Input-warp PSNR: {old_label} {item['old_input_warp_psnr_db']:.2f} dB, "
                f"{new_label} {item['new_input_warp_psnr_db']:.2f} dB. "
                f"New exposure: {exposure:.3f}."
                "</p>"
            )
        html.append(f"<img src='{rel}' alt='{item['sample_id']}'>")
    with open(Path(outdir) / "index.html", "w") as f:
        f.write("\n".join(html))


def main():
    args = parse_args()
    old_uses_darken = args.old_input_mode == "darken"
    old_cfg = cfg_with_lowlight(load_config(args.old_config), enabled=old_uses_darken)
    new_uses_darken = args.new_input_mode == "darken"
    new_cfg = cfg_with_lowlight(load_config(args.new_config), enabled=new_uses_darken)
    base_cfg = cfg_with_lowlight(load_config(args.old_config), enabled=False)
    old_train_args = make_train_args(args, args.old_config, args.outdir)
    new_train_args = make_train_args(args, args.new_config, args.outdir)
    device = resolve_device(args.device)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    dataset, base_batches = cache_base_batches(old_train_args, base_cfg)
    if old_uses_darken:
        old_batches = make_darkened_batches(base_batches, old_cfg, args.seed + args.darken_seed_offset)
    else:
        old_batches = [clone_batch(batch) for batch in base_batches]
    if new_uses_darken:
        new_batches = make_darkened_batches(base_batches, new_cfg, args.seed + args.darken_seed_offset)
    else:
        new_batches = [clone_batch(batch) for batch in base_batches]

    old_model, old_info = build_model(old_train_args, old_cfg, device, args.old_checkpoint)
    old_outputs = predict(old_model, old_batches, old_train_args, old_cfg, device)
    del old_model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    new_model, new_info = build_model(new_train_args, new_cfg, device, args.new_checkpoint)
    new_outputs = predict(new_model, new_batches, new_train_args, new_cfg, device)
    del new_model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    records = []
    for idx, (old_batch, new_batch, old_pred, new_pred) in enumerate(
        zip(old_batches, new_batches, old_outputs, new_outputs)
    ):
        out_path = outdir / f"sample_{idx:03d}_dual_input_panel.png"
        if "depth_gt" in old_batch:
            record = render_panel(
                out_path,
                idx,
                old_batch,
                new_batch,
                old_pred,
                new_pred,
                args.old_label,
                args.new_label,
            )
        else:
            record = render_panel_no_gt(
                out_path,
                idx,
                old_batch,
                new_batch,
                old_pred,
                new_pred,
                args.old_label,
                args.new_label,
            )
        records.append(record)

    manifest = {
        "data_root": getattr(dataset, "root_paths_text", str(dataset.root_path)),
        "dataset_len": len(dataset),
        "old_label": args.old_label,
        "new_label": args.new_label,
        "old_config": str(Path(args.old_config).resolve()),
        "new_config": str(Path(args.new_config).resolve()),
        "old_load_info": old_info,
        "new_load_info": new_info,
        "old_lowlight_aug_enabled": old_uses_darken,
        "new_lowlight_aug_enabled": new_uses_darken,
        "old_input_mode": args.old_input_mode,
        "new_input_mode": args.new_input_mode,
        "records": records,
    }
    with open(outdir / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)
    write_index(outdir, records, args.old_label, args.new_label)
    print(json.dumps({"outdir": str(outdir.resolve()), "panels": [item["panel"] for item in records]}, indent=2))


if __name__ == "__main__":
    main()
