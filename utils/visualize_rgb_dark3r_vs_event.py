#!/usr/bin/env python3
import argparse
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

from model.event_dark3r_student import make_view_pair_from_batch  # noqa: E402
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
    parser = argparse.ArgumentParser("Render Master3r/Dark3r baselines vs trained Night3r")
    parser.add_argument("--trained-checkpoint", required=True)
    parser.add_argument("--config", default="config/train_event_dark3r.yaml")
    parser.add_argument("--data-root", default=None)
    parser.add_argument("--model-name-or-path", required=True)
    parser.add_argument(
        "--dark3r-lora-checkpoint",
        default="checkpoints/dark3r_pretrained/weights/0049_lora.pth",
        help="LoRA checkpoint used for the Dark3r low-light RGB baseline.",
    )
    parser.add_argument("--allow-unsafe-checkpoint-load", action="store_true")
    parser.add_argument("--image-size", type=int, nargs=2, default=None, metavar=("H", "W"))
    parser.add_argument("--max-pairs", type=int, default=4)
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
    parser.add_argument("--skip-dark3r", action="store_true", help="Only compare original MASt3R against trained Night3R.")
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
    parser.add_argument("--outdir", required=True)
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


def make_train_args(args):
    return SimpleNamespace(
        config=args.config,
        data_root=args.data_root,
        save_dir=args.outdir,
        trained_checkpoint=args.trained_checkpoint,
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


def cache_batches(args, cfg):
    set_seed(args.seed)
    dataset = build_dataset(args, cfg)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=False,
    )
    cached = []
    start_index = max(int(getattr(args, "start_index", 0) or 0), 0)
    for batch_idx, batch in enumerate(loader):
        if batch_idx < start_index:
            continue
        cached.append({key: value.detach().cpu() if torch.is_tensor(value) else value for key, value in batch.items()})
        if len(cached) >= args.max_pairs:
            break
    return dataset, cached


def image_hwc(image):
    return image.detach().cpu().float().clamp(0, 1).permute(1, 2, 0).numpy()


def resolve_repo_path(path):
    path = Path(path)
    if path.is_absolute():
        return path
    return REPO_ROOT / path


def build_event_model(args, cfg, device):
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
        purpose="trained Night3r checkpoint",
    )
    missing, unexpected = model.load_state_dict(payload["model"], strict=False)
    model = model.to(device).eval()
    return model, {"step": payload.get("step"), "missing_keys": list(missing), "unexpected_keys": list(unexpected)}


def resize_depth(depth, shape):
    if depth.shape[-2:] == shape:
        return depth
    return F.interpolate(depth, size=shape, mode="bilinear", align_corners=False)


def compute_warp(pred_depth, batch_gpu):
    if (
        "relative_pose" not in batch_gpu
        or "intrinsics" not in batch_gpu
        or pred_depth is None
    ):
        return None, None
    image2 = batch_gpu["images_photo"][:, 1].to(dtype=torch.float32)
    depth = resize_depth(pred_depth.float(), image2.shape[-2:])
    return warp_second_to_first(
        image2,
        depth,
        batch_gpu["intrinsics"],
        batch_gpu["relative_pose"],
    )


def predict_rgb(model, batches, args, cfg, device, prefer_dark):
    amp_enabled, _, amp_dtype = resolve_amp_config(cfg, args, device)
    outputs = []
    with torch.no_grad():
        for batch in batches:
            batch_gpu = move_batch_to_device(batch, device)
            view1, view2 = make_view_pair_from_batch(batch_gpu, prefer_dark=prefer_dark)
            with torch.amp.autocast(
                device_type=device.type,
                dtype=amp_dtype,
                enabled=bool(amp_enabled and device.type == "cuda"),
            ):
                pred1, _ = model(view1, view2)
            depth = depth_from_pred(pred1)
            warped, valid = compute_warp(depth, batch_gpu)
            item = {"depth": depth.detach().cpu()}
            if warped is not None and valid is not None:
                item.update({"warped": warped.detach().cpu(), "warp_valid": valid.detach().cpu()})
            outputs.append(item)
    return outputs


def predict_event(model, batches, args, cfg, device):
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
            warped, valid = compute_warp(depth, batch_gpu)
            item = {"depth": depth.detach().cpu()}
            if warped is not None and valid is not None:
                item.update({"warped": warped.detach().cpu(), "warp_valid": valid.detach().cpu()})
            outputs.append(item)
    return outputs


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


def warp_psnr(warped, image1, valid):
    if warped is None or valid is None:
        return float("nan")
    valid = valid.bool().expand_as(image1) & torch.isfinite(warped) & torch.isfinite(image1)
    if not valid.any():
        return float("nan")
    mse = float(((warped[valid] - image1[valid]) ** 2).mean().detach().cpu())
    return -10.0 * math.log10(max(mse, 1e-12))


def color_limits(values, lo=2, hi=98):
    values = np.asarray(values)
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return 0.0, 1.0
    vmin = float(np.percentile(finite, lo))
    vmax = float(np.percentile(finite, hi))
    if vmax <= vmin:
        vmax = vmin + 1e-6
    return vmin, vmax


def render_panel(out_path, idx, batch, clean, lowlight, event):
    image1 = batch["images_photo"][0, 0]
    image2 = batch["images_photo"][0, 1]
    dark1 = batch["images_photo_dark"][0, 0]
    event_activity = batch["event_voxel"][0, 0].abs().sum(dim=0).float().numpy()
    gt_depth = as_bchw(batch["depth_gt"])[0:1]
    gt_valid = as_bchw(batch["depth_valid"])[0:1].bool()
    gt_depth_2d = gt_depth[0, 0]
    valid_2d = gt_valid[0, 0].numpy()

    clean_depth = resize_depth(clean["depth"][0:1], gt_depth.shape[-2:])[0, 0]
    low_depth = resize_depth(lowlight["depth"][0:1], gt_depth.shape[-2:])[0, 0]
    event_depth = resize_depth(event["depth"][0:1], gt_depth.shape[-2:])[0, 0]

    clean_err = (clean_depth - gt_depth_2d).abs()
    low_err = (low_depth - gt_depth_2d).abs()
    event_err = (event_depth - gt_depth_2d).abs()
    depth_vmin, depth_vmax = color_limits(gt_depth_2d.numpy()[valid_2d])
    err_stack = np.concatenate([clean_err.numpy()[valid_2d], low_err.numpy()[valid_2d], event_err.numpy()[valid_2d]])
    _, err_vmax = color_limits(err_stack, lo=0, hi=95)
    improvement = (low_err - event_err).numpy()
    imp_abs = max(float(np.percentile(np.abs(improvement[valid_2d]), 98)), 1e-6)

    clean_m = depth_metrics(clean_depth[None, None], gt_depth, gt_valid)
    low_m = depth_metrics(low_depth[None, None], gt_depth, gt_valid)
    event_m = depth_metrics(event_depth[None, None], gt_depth, gt_valid)
    clean_wpsnr = warp_psnr(clean.get("warped"), batch["images_photo"][:, 0], clean.get("warp_valid"))
    low_wpsnr = warp_psnr(lowlight.get("warped"), batch["images_photo"][:, 0], lowlight.get("warp_valid"))
    event_wpsnr = warp_psnr(event.get("warped"), batch["images_photo"][:, 0], event.get("warp_valid"))

    target_view = image_hwc(image1)
    skipped_warp = np.zeros_like(target_view)

    def warp_image(output):
        if output.get("warped") is None:
            return skipped_warp
        return image_hwc(output["warped"][0])

    def warp_title(name, value):
        if math.isfinite(float(value)):
            return f"{name} warp\nPSNR {value:.2f} dB"
        return f"{name} warp\nskipped: missing pose"

    sample_id = batch.get("sample_id", [f"sample_{idx:03d}"])
    if isinstance(sample_id, (list, tuple)):
        sample_id = sample_id[0]

    fig, axes = plt.subplots(4, 4, figsize=(20, 16), constrained_layout=True)
    fig.suptitle(str(sample_id), fontsize=14)
    panels = [
        (axes[0, 0], image_hwc(image1), "Clean RGB reference", None, None, None),
        (axes[0, 1], image_hwc(dark1), "Low-light input to all models", None, None, None),
        (axes[0, 2], image_hwc(image2), "RGB view2", None, None, None),
        (axes[0, 3], event_activity, "Event activity", "magma", None, None),
        (axes[1, 0], gt_depth_2d.numpy(), "GT depth", "viridis", depth_vmin, depth_vmax),
        (axes[1, 1], clean_depth.numpy(), f"Master3r depth\nPSNR {clean_m['psnr_db']:.2f} dB", "viridis", depth_vmin, depth_vmax),
        (axes[1, 2], low_depth.numpy(), f"Dark3r depth\nPSNR {low_m['psnr_db']:.2f} dB", "viridis", depth_vmin, depth_vmax),
        (axes[1, 3], event_depth.numpy(), f"Night3r depth\nPSNR {event_m['psnr_db']:.2f} dB", "viridis", depth_vmin, depth_vmax),
        (axes[2, 0], clean_err.numpy(), f"Master3r abs error\nAbsRel {clean_m['absrel']:.3f}", "inferno", 0.0, err_vmax),
        (axes[2, 1], low_err.numpy(), f"Dark3r abs error\nAbsRel {low_m['absrel']:.3f}", "inferno", 0.0, err_vmax),
        (axes[2, 2], event_err.numpy(), f"Night3r abs error\nAbsRel {event_m['absrel']:.3f}", "inferno", 0.0, err_vmax),
        (axes[2, 3], improvement, "Abs error improvement\npositive = Night3r better than Dark3r", "coolwarm", -imp_abs, imp_abs),
        (axes[3, 0], warp_image(clean), warp_title("Master3r", clean_wpsnr), None, None, None),
        (axes[3, 1], warp_image(lowlight), warp_title("Dark3r", low_wpsnr), None, None, None),
        (axes[3, 2], warp_image(event), warp_title("Night3r", event_wpsnr), None, None, None),
        (axes[3, 3], target_view, "Clean RGB reference", None, None, None),
    ]
    for ax, data, title, cmap, vmin, vmax in panels:
        if cmap is None:
            ax.imshow(np.clip(data, 0, 1))
        else:
            im = ax.imshow(data, cmap=cmap, vmin=vmin, vmax=vmax)
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
        ax.set_title(title, fontsize=10)
        ax.axis("off")
    fig.savefig(out_path, dpi=140)
    plt.close(fig)
    return {
        "sample_id": str(sample_id),
        "panel": str(out_path),
        "master3r_depth": clean_m,
        "dark3r_depth": low_m,
        "night3r_depth": event_m,
        "master3r_warp_psnr_db": clean_wpsnr,
        "dark3r_warp_psnr_db": low_wpsnr,
        "night3r_warp_psnr_db": event_wpsnr,
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


def render_panel_no_gt(out_path, idx, batch, master3r, night3r):
    image1 = batch["images_photo"][0, 0]
    image2 = batch["images_photo"][0, 1]
    dark1 = batch["images_photo_dark"][0, 0]
    dark2 = batch["images_photo_dark"][0, 1]
    event_activity = batch["event_voxel"][0, 0].abs().sum(dim=0).float().numpy()

    target_shape = image1.shape[-2:]
    master_depth = resize_depth(master3r["depth"][0:1], target_shape)[0, 0]
    night_depth = resize_depth(night3r["depth"][0:1], target_shape)[0, 0]
    master_np = master_depth.numpy()
    night_np = night_depth.numpy()
    depth_values = np.concatenate([master_np.reshape(-1), night_np.reshape(-1)])
    depth_vmin, depth_vmax = color_limits(depth_values)
    diff = night_np - master_np
    diff_abs = max(float(np.percentile(np.abs(diff[np.isfinite(diff)]), 98)), 1e-6)

    master_stats = depth_stats(master_depth)
    night_stats = depth_stats(night_depth)
    master_wpsnr = warp_psnr(master3r.get("warped"), batch["images_photo"][:, 0], master3r.get("warp_valid"))
    night_wpsnr = warp_psnr(night3r.get("warped"), batch["images_photo"][:, 0], night3r.get("warp_valid"))

    sample_id = batch.get("sample_id", [f"sample_{idx:03d}"])
    if isinstance(sample_id, (list, tuple)):
        sample_id = sample_id[0]

    def warp_image(output):
        if output.get("warped") is None:
            return np.zeros_like(image_hwc(image1))
        return image_hwc(output["warped"][0])

    fig, axes = plt.subplots(3, 4, figsize=(20, 12), constrained_layout=True)
    fig.suptitle(f"{sample_id} | forced dark input, no GT depth", fontsize=14)
    panels = [
        (axes[0, 0], image_hwc(image1), "Clean RGB view1", None, None, None),
        (axes[0, 1], image_hwc(dark1), "Dark input view1", None, None, None),
        (axes[0, 2], image_hwc(image2), "Clean RGB view2", None, None, None),
        (axes[0, 3], event_activity, "Event activity", "magma", None, None),
        (
            axes[1, 0],
            master_np,
            f"Original MASt3R on dark RGB\nmedian {master_stats['median']:.2f} / p98 {master_stats['p98']:.2f}",
            "viridis",
            depth_vmin,
            depth_vmax,
        ),
        (
            axes[1, 1],
            night_np,
            f"Night3R on dark RGB+event\nmedian {night_stats['median']:.2f} / p98 {night_stats['p98']:.2f}",
            "viridis",
            depth_vmin,
            depth_vmax,
        ),
        (axes[1, 2], diff, "Depth diff\nNight3R - MASt3R", "coolwarm", -diff_abs, diff_abs),
        (axes[1, 3], image_hwc(dark2), "Dark input view2", None, None, None),
        (
            axes[2, 0],
            warp_image(master3r),
            f"MASt3R warp view2->view1\nRGB PSNR {master_wpsnr:.2f} dB",
            None,
            None,
            None,
        ),
        (
            axes[2, 1],
            warp_image(night3r),
            f"Night3R warp view2->view1\nRGB PSNR {night_wpsnr:.2f} dB",
            None,
            None,
            None,
        ),
        (axes[2, 2], image_hwc(image1), "Clean RGB reference", None, None, None),
        (axes[2, 3], image_hwc(dark1), "Dark model input", None, None, None),
    ]
    for ax, data, title, cmap, vmin, vmax in panels:
        if cmap is None:
            ax.imshow(np.clip(data, 0, 1))
        else:
            im = ax.imshow(data, cmap=cmap, vmin=vmin, vmax=vmax)
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
        ax.set_title(title, fontsize=10)
        ax.axis("off")
    fig.savefig(out_path, dpi=140)
    plt.close(fig)
    return {
        "sample_id": str(sample_id),
        "panel": str(out_path),
        "master3r_depth_stats": master_stats,
        "night3r_depth_stats": night_stats,
        "master3r_warp_psnr_db": master_wpsnr,
        "night3r_warp_psnr_db": night_wpsnr,
    }


def write_index(outdir, records):
    html = [
        "<!doctype html><meta charset='utf-8'>",
        "<title>Master3r / Dark3r / Night3r</title>",
        "<style>body{font-family:Arial,sans-serif;margin:24px;background:#f6f6f6;color:#111}"
        "img{max-width:100%;display:block;margin:12px 0 36px;border:1px solid #ccc;background:white}</style>",
        "<h1>Master3r / Dark3r / Night3r</h1>",
    ]
    for item in records:
        rel = Path(item["panel"]).name
        html.append(f"<h2>{item['sample_id']}</h2>")
        if "master3r_depth" in item:
            html.append(
                "<p>"
                f"Depth PSNR: Master3r {item['master3r_depth']['psnr_db']:.2f} dB, "
                f"Dark3r {item['dark3r_depth']['psnr_db']:.2f} dB, "
                f"Night3r {item['night3r_depth']['psnr_db']:.2f} dB. "
                f"Warp PSNR: Master3r {item['master3r_warp_psnr_db']:.2f} dB, "
                f"Dark3r {item['dark3r_warp_psnr_db']:.2f} dB, "
                f"Night3r {item['night3r_warp_psnr_db']:.2f} dB."
                "</p>"
            )
        else:
            html.append(
                "<p>"
                f"MASt3R p98: {item['master3r_depth_stats']['p98']:.2f}; "
                f"Night3R p98: {item['night3r_depth_stats']['p98']:.2f}. "
                f"Warp PSNR: MASt3R {item['master3r_warp_psnr_db']:.2f} dB, "
                f"Night3R {item['night3r_warp_psnr_db']:.2f} dB."
                "</p>"
            )
        html.append(f"<img src='{rel}' alt='{item['sample_id']}'>")
    with open(Path(outdir) / "index.html", "w") as f:
        f.write("\n".join(html))


def main():
    cli_args = parse_args()
    cfg = load_config(cli_args.config)
    cfg = apply_lowlight_overrides(cfg, cli_args)
    args = make_train_args(cli_args)
    device = resolve_device(args.device)
    outdir = Path(cli_args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    dataset, batches = cache_batches(args, cfg)
    master3r_model = load_backbone(args, cfg).to(device).eval()
    master3r_outputs = predict_rgb(master3r_model, batches, args, cfg, device, prefer_dark=True)
    del master3r_model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    if cli_args.skip_dark3r:
        low_outputs = [None for _ in batches]
    else:
        dark3r_args = SimpleNamespace(**vars(args))
        dark3r_args.backbone_lora_checkpoint = str(resolve_repo_path(cli_args.dark3r_lora_checkpoint))
        dark3r_model = load_backbone(dark3r_args, cfg).to(device).eval()
        low_outputs = predict_rgb(dark3r_model, batches, args, cfg, device, prefer_dark=True)
        del dark3r_model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    event_model, load_info = build_event_model(args, cfg, device)
    event_outputs = predict_event(event_model, batches, args, cfg, device)
    del event_model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    records = []
    for idx, (batch, master3r, dark3r, event) in enumerate(zip(batches, master3r_outputs, low_outputs, event_outputs)):
        if "depth_gt" in batch and dark3r is not None:
            records.append(render_panel(outdir / f"sample_{idx:03d}_panel.png", idx, batch, master3r, dark3r, event))
        else:
            records.append(render_panel_no_gt(outdir / f"sample_{idx:03d}_panel.png", idx, batch, master3r, event))

    manifest = {
        "note": "All model predictions use low-light RGB input. Night3r also uses event voxel.",
        "data_root": getattr(dataset, "root_paths_text", str(dataset.root_path)),
        "dataset_len": len(dataset),
        "model_name_or_path": str(Path(cli_args.model_name_or_path).resolve()),
        "dark3r_lora_checkpoint": str(resolve_repo_path(cli_args.dark3r_lora_checkpoint).resolve()),
        "dark3r_enabled": not cli_args.skip_dark3r,
        "lowlight_aug": cfg.get("lowlight_aug", {}),
        "trained_checkpoint": str(Path(cli_args.trained_checkpoint).resolve()),
        "trained_load_info": load_info,
        "records": records,
    }
    with open(outdir / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)
    write_index(outdir, records)
    print(json.dumps({"outdir": str(outdir.resolve()), "panels": [item["panel"] for item in records]}, indent=2))


if __name__ == "__main__":
    main()
