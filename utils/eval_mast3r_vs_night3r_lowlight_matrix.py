#!/usr/bin/env python3
"""Compare original MASt3R and trained Night3R under controlled DL3DV darkening.

The script is intentionally evaluation-only.  It does not require GT depth and
therefore reports prediction scale statistics plus view2->view1 warp PSNR.
Those numbers are not a full geometry benchmark, but they are useful for
finding cases where event injection helps or hurts relative to the RGB-only
MASt3R path.
"""

import argparse
import csv
import json
import math
import os
import sys
from pathlib import Path
from types import SimpleNamespace

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw
from torch.utils.data._utils.collate import default_collate

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

try:
    import cv2
except Exception:  # pragma: no cover - optional panel coloring dependency
    cv2 = None


def parse_args():
    parser = argparse.ArgumentParser("MASt3R vs Night3R low-light matrix")
    parser.add_argument("--trained-checkpoint", required=True)
    parser.add_argument("--config", default="config/train_event_dark3r.yaml")
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--model-name-or-path", required=True)
    parser.add_argument("--allow-unsafe-checkpoint-load", action="store_true")
    parser.add_argument("--image-size", type=int, nargs=2, default=None, metavar=("H", "W"))
    parser.add_argument("--scenes", nargs="*", default=None, help="Scene names such as screen-000090.")
    parser.add_argument("--max-scenes", type=int, default=None)
    parser.add_argument("--frame-index", type=int, default=24)
    parser.add_argument("--pair-stride", type=int, default=1)
    parser.add_argument("--exposures", type=float, nargs="+", default=[0.2, 0.1, 0.05])
    parser.add_argument("--amp", dest="amp", action="store_true", default=None)
    parser.add_argument("--no-amp", dest="amp", action="store_false")
    parser.add_argument("--amp-dtype", choices=("fp16", "bf16"), default=None)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--save-panels", action="store_true")
    parser.add_argument(
        "--panel-protocol",
        choices=("legacy", "v2"),
        default="legacy",
        help=(
            "legacy writes the original compact panel. v2 writes a stricter "
            "comparison panel with shared MASt3R/Night3R color ranges, "
            "median-aligned depth, log-ratio maps, and warp error maps."
        ),
    )
    parser.add_argument("--outdir", required=True)
    return parser.parse_args()


def apply_lowlight_exposure(cfg, exposure):
    cfg = dict(cfg)
    lowlight_cfg = dict(cfg.get("lowlight_aug", {}))
    lowlight_cfg["enabled"] = True
    lowlight_cfg["min_exposure"] = float(exposure)
    lowlight_cfg["max_exposure"] = float(exposure)
    cfg["lowlight_aug"] = lowlight_cfg
    return cfg


def make_train_args(args, outdir):
    return SimpleNamespace(
        config=args.config,
        data_root=args.data_root,
        save_dir=str(outdir),
        trained_checkpoint=args.trained_checkpoint,
        model_name_or_path=args.model_name_or_path,
        allow_unsafe_checkpoint_load=args.allow_unsafe_checkpoint_load,
        use_tiny_model=False,
        image_size=args.image_size,
        batch_size=1,
        max_batches=None,
        max_pairs=None,
        start_index=0,
        num_epochs=None,
        max_steps=None,
        max_scenes=args.max_scenes,
        max_pairs_per_scene=None,
        pair_stride=args.pair_stride,
        pair_sampling="fixed",
        min_pair_distance=args.pair_stride,
        max_pair_distance=args.pair_stride,
        precomputed_pairs=None,
        allow_reverse_pairs=False,
        symmetric_pairs=False,
        num_workers=0,
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
        purpose="trained Night3R checkpoint",
    )
    missing, unexpected = model.load_state_dict(payload["model"], strict=False)
    model = model.to(device).eval()
    return model, {
        "step": payload.get("step"),
        "stage": payload.get("stage"),
        "epoch": payload.get("epoch"),
        "missing_keys": list(missing),
        "unexpected_keys": list(unexpected),
    }


def resize_depth(depth, shape):
    if depth.shape[-2:] == shape:
        return depth
    return F.interpolate(depth, size=shape, mode="bilinear", align_corners=False)


def compute_warp(pred_depth, batch_gpu):
    if "relative_pose" not in batch_gpu or "intrinsics" not in batch_gpu or pred_depth is None:
        return None, None
    image2 = batch_gpu["images_photo"][:, 1].to(dtype=torch.float32)
    depth = resize_depth(pred_depth.float(), image2.shape[-2:])
    return warp_second_to_first(
        image2,
        depth,
        batch_gpu["intrinsics"],
        batch_gpu["relative_pose"],
    )


def warp_psnr(warped, target, valid):
    if warped is None or valid is None:
        return float("nan")
    valid = valid.bool().expand_as(target) & torch.isfinite(warped) & torch.isfinite(target)
    if not valid.any():
        return float("nan")
    mse = float(((warped[valid] - target[valid]) ** 2).mean().detach().cpu())
    return -10.0 * math.log10(max(mse, 1e-12))


def depth_stats(depth):
    values = depth.detach().cpu().float().numpy()
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return {"p02": float("nan"), "median": float("nan"), "p98": float("nan")}
    return {
        "p02": float(np.percentile(finite, 2)),
        "median": float(np.median(finite)),
        "p98": float(np.percentile(finite, 98)),
    }


def event_stats(batch):
    event = batch["event_voxel"][0, 0].detach().cpu().float()
    activity = event.abs().sum(dim=0)
    return {
        "event_activity_mean": float(activity.mean().item()),
        "event_activity_p95": float(torch.quantile(activity.flatten(), 0.95).item()),
        "event_occupancy": float((activity > 0).float().mean().item()),
    }


def select_samples(dataset, scenes, frame_index):
    scene_filter = {scene.replace("_", "-") for scene in scenes} if scenes else None
    selected = []
    seen = set()
    for idx, sample in enumerate(dataset.samples):
        scene = str(sample["scene"])
        if scene_filter is not None and scene not in scene_filter:
            continue
        if int(sample["idx1"]) != int(frame_index):
            continue
        if scene in seen:
            continue
        selected.append(idx)
        seen.add(scene)
    if selected:
        return selected

    # Fallback for short scenes: use one central pair per selected scene.
    by_scene = {}
    for idx, sample in enumerate(dataset.samples):
        scene = str(sample["scene"])
        if scene_filter is not None and scene not in scene_filter:
            continue
        by_scene.setdefault(scene, []).append((idx, sample))
    for scene, items in sorted(by_scene.items()):
        mid = len(items) // 2
        selected.append(items[mid][0])
    return selected


def sample_to_batch(dataset, sample_index):
    sample = dataset[sample_index]
    return default_collate([sample])


def predict_master(model, batch_gpu, args, cfg, device):
    amp_enabled, _, amp_dtype = resolve_amp_config(cfg, args, device)
    view1, view2 = make_view_pair_from_batch(batch_gpu, prefer_dark=True)
    with torch.no_grad():
        with torch.amp.autocast(
            device_type=device.type,
            dtype=amp_dtype,
            enabled=bool(amp_enabled and device.type == "cuda"),
        ):
            pred1, _ = model(view1, view2)
    return depth_from_pred(pred1)


def predict_night(model, batch_gpu, args, cfg, device):
    amp_enabled, _, amp_dtype = resolve_amp_config(cfg, args, device)
    with torch.no_grad():
        with torch.amp.autocast(
            device_type=device.type,
            dtype=amp_dtype,
            enabled=bool(amp_enabled and device.type == "cuda"),
        ):
            pred1, _ = model(batch_gpu)
    return depth_from_pred(pred1)


def image_hwc_uint8(image_chw):
    arr = image_chw.detach().cpu().float().clamp(0, 1).permute(1, 2, 0).numpy()
    return (arr * 255.0 + 0.5).astype(np.uint8)


def colorize_scalar(values, vmin=None, vmax=None, cmap="viridis"):
    arr = values.detach().cpu().float().numpy() if torch.is_tensor(values) else np.asarray(values)
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        norm = np.zeros_like(arr, dtype=np.float32)
    else:
        if vmin is None:
            vmin = float(np.percentile(finite, 2))
        if vmax is None:
            vmax = float(np.percentile(finite, 98))
        if vmax <= vmin:
            vmax = vmin + 1e-6
        norm = np.clip((arr - vmin) / (vmax - vmin), 0, 1).astype(np.float32)
    gray = (norm * 255.0 + 0.5).astype(np.uint8)
    if cv2 is not None:
        if cmap == "viridis":
            code = cv2.COLORMAP_VIRIDIS
        elif cmap == "magma":
            code = cv2.COLORMAP_MAGMA
        else:
            code = cv2.COLORMAP_INFERNO
        return cv2.cvtColor(cv2.applyColorMap(gray, code), cv2.COLOR_BGR2RGB)
    return np.repeat(gray[..., None], 3, axis=2)


def colorize_diff(values, scale=None):
    arr = values.detach().cpu().float().numpy() if torch.is_tensor(values) else np.asarray(values)
    finite = arr[np.isfinite(arr)]
    if scale is None:
        scale = float(np.percentile(np.abs(finite), 98)) if finite.size else 1.0
    scale = max(scale, 1e-6)
    norm = np.clip(arr / scale, -1, 1)
    out = np.zeros((*arr.shape, 3), dtype=np.uint8)
    out[..., 0] = np.where(norm > 0, norm * 255, 0).astype(np.uint8)
    out[..., 2] = np.where(norm < 0, -norm * 255, 0).astype(np.uint8)
    out[..., 1] = (255 * (1 - np.abs(norm)) * 0.85).astype(np.uint8)
    return out


def tile_with_title(image, title, subtitle=None, size=(320, 320)):
    pil = Image.fromarray(image).resize(size, Image.BILINEAR)
    title_h = 48 if subtitle else 36
    canvas = Image.new("RGB", (size[0], size[1] + title_h), "white")
    canvas.paste(pil, (0, title_h))
    draw = ImageDraw.Draw(canvas)
    draw.text((8, 8), title[:70], fill=(0, 0, 0))
    if subtitle:
        draw.text((8, 25), subtitle[:80], fill=(64, 64, 64))
    return canvas


def text_tile(lines, size=(320, 320)):
    canvas = Image.new("RGB", (size[0], size[1] + 48), "white")
    draw = ImageDraw.Draw(canvas)
    draw.text((8, 8), "protocol + metrics", fill=(0, 0, 0))
    y = 58
    for line in lines:
        draw.text((10, y), str(line), fill=(20, 20, 20))
        y += 22
    return canvas


def robust_range(*arrays, qlo=2.0, qhi=98.0):
    values = []
    for array in arrays:
        arr = array.detach().cpu().float().numpy() if torch.is_tensor(array) else np.asarray(array)
        finite = arr[np.isfinite(arr)]
        if finite.size:
            values.append(finite.reshape(-1))
    if not values:
        return 0.0, 1.0
    values = np.concatenate(values)
    vmin = float(np.percentile(values, qlo))
    vmax = float(np.percentile(values, qhi))
    if vmax <= vmin:
        vmax = vmin + 1e-6
    return vmin, vmax


def safe_median(depth):
    finite = depth[torch.isfinite(depth) & (depth > 0)]
    if finite.numel() == 0:
        return depth.new_tensor(1.0)
    return finite.median().clamp_min(1e-6)


def warp_error_image(warped, target, valid):
    if warped is None or valid is None:
        return torch.zeros_like(target[:, :1])
    valid = valid.bool().expand_as(target) & torch.isfinite(warped) & torch.isfinite(target)
    err = (warped.float() - target.float()).abs().mean(dim=1, keepdim=True)
    return torch.where(valid[:, :1], err, torch.zeros_like(err))


def save_panel(path, batch, master_depth, night_depth, master_warped, night_warped, record):
    image1 = batch["images_photo"][0, 0]
    dark1 = batch["images_photo_dark"][0, 0]
    event_activity = batch["event_voxel"][0, 0].abs().sum(dim=0)
    target_shape = image1.shape[-2:]
    master = resize_depth(master_depth.cpu(), target_shape)[0, 0]
    night = resize_depth(night_depth.cpu(), target_shape)[0, 0]
    all_depth = torch.cat([master.flatten(), night.flatten()])
    finite = all_depth[torch.isfinite(all_depth)]
    if finite.numel():
        vmin = float(torch.quantile(finite, 0.02).item())
        vmax = float(torch.quantile(finite, 0.98).item())
    else:
        vmin, vmax = 0.0, 1.0

    def warp_img(warped):
        if warped is None:
            return np.zeros_like(image_hwc_uint8(image1))
        return image_hwc_uint8(warped[0])

    tiles = [
        tile_with_title(image_hwc_uint8(image1), "clean RGB"),
        tile_with_title(image_hwc_uint8(dark1), f"dark RGB exp={record['exposure']:.2f}"),
        tile_with_title(colorize_scalar(event_activity, cmap="magma"), "event activity"),
        tile_with_title(
            colorize_scalar(master, vmin=vmin, vmax=vmax),
            f"MASt3R depth p98 {record['master_p98']:.2f}",
        ),
        tile_with_title(
            colorize_scalar(night, vmin=vmin, vmax=vmax),
            f"Night3R depth p98 {record['night_p98']:.2f}",
        ),
        tile_with_title(colorize_diff(night - master), "depth diff N-M"),
        tile_with_title(warp_img(master_warped), f"MASt3R warp {record['master_warp_psnr_db']:.2f}"),
        tile_with_title(warp_img(night_warped), f"Night3R warp {record['night_warp_psnr_db']:.2f}"),
    ]
    w, h = tiles[0].size
    grid = Image.new("RGB", (4 * w, 2 * h), "white")
    for i, tile in enumerate(tiles):
        grid.paste(tile, ((i % 4) * w, (i // 4) * h))
    path.parent.mkdir(parents=True, exist_ok=True)
    grid.save(path)


def save_panel_v2(path, batch, master_depth, night_depth, master_warped, night_warped, record):
    """Write a strict comparison panel with explicit shared visualization rules."""
    image1 = batch["images_photo"][0, 0]
    dark1 = batch["images_photo_dark"][0, 0]
    event_activity = batch["event_voxel"][0, 0].abs().sum(dim=0)
    target_shape = image1.shape[-2:]
    master = resize_depth(master_depth.cpu(), target_shape)[0, 0].clamp_min(1e-6)
    night = resize_depth(night_depth.cpu(), target_shape)[0, 0].clamp_min(1e-6)

    raw_vmin, raw_vmax = robust_range(master, night, qlo=2, qhi=98)
    master_med = safe_median(master)
    night_med = safe_median(night)
    master_norm = master / master_med
    night_norm = night / night_med
    norm_vmin, norm_vmax = robust_range(master_norm, night_norm, qlo=2, qhi=98)

    raw_log_ratio = torch.log2(night / master).clamp(-1.0, 1.0)
    norm_log_ratio = torch.log2(night_norm.clamp_min(1e-6) / master_norm.clamp_min(1e-6)).clamp(-1.0, 1.0)

    target = batch["images_photo"][:, 0].float()
    master_err = warp_error_image(master_warped.cpu() if master_warped is not None else None, target, None)
    night_err = warp_error_image(night_warped.cpu() if night_warped is not None else None, target, None)
    if master_warped is not None:
        master_err = (master_warped.cpu().float() - target).abs().mean(dim=1, keepdim=True)
    if night_warped is not None:
        night_err = (night_warped.cpu().float() - target).abs().mean(dim=1, keepdim=True)

    def warp_img(warped):
        if warped is None:
            return np.zeros_like(image_hwc_uint8(image1))
        return image_hwc_uint8(warped.cpu()[0])

    metric_lines = [
        "Depth protocol:",
        f"raw depth shared [{raw_vmin:.2f}, {raw_vmax:.2f}]",
        f"median aligned shared [{norm_vmin:.2f}, {norm_vmax:.2f}]",
        "ratio maps fixed: log2(Night3R/MASt3R)",
        "red: Night3R farther, blue: Night3R nearer",
        "",
        f"MASt3R med/p98: {record['master_median']:.2f} / {record['master_p98']:.2f}",
        f"Night3R med/p98: {record['night_median']:.2f} / {record['night_p98']:.2f}",
        f"warp PSNR M/N: {record['master_warp_psnr_db']:.2f} / {record['night_warp_psnr_db']:.2f}",
    ]

    tiles = [
        tile_with_title(image_hwc_uint8(image1), "clean RGB"),
        tile_with_title(image_hwc_uint8(dark1), f"dark RGB exp={record['exposure']:.2f}"),
        tile_with_title(colorize_scalar(event_activity, cmap="magma"), "event activity", "same input for Night3R"),
        text_tile(metric_lines),
        tile_with_title(colorize_scalar(master, raw_vmin, raw_vmax), "MASt3R raw depth", f"shared raw scale [{raw_vmin:.2f}, {raw_vmax:.2f}]"),
        tile_with_title(colorize_scalar(night, raw_vmin, raw_vmax), "Night3R raw depth", f"shared raw scale [{raw_vmin:.2f}, {raw_vmax:.2f}]"),
        tile_with_title(colorize_diff(raw_log_ratio, scale=1.0), "raw log2 ratio N/M", "fixed [-1, 1]"),
        tile_with_title(colorize_scalar((night - master).abs(), cmap="inferno"), "|raw depth diff|", "display only, not scale-corrected"),
        tile_with_title(colorize_scalar(master_norm, norm_vmin, norm_vmax), "MASt3R depth / median", f"median {float(master_med):.2f}"),
        tile_with_title(colorize_scalar(night_norm, norm_vmin, norm_vmax), "Night3R depth / median", f"median {float(night_med):.2f}"),
        tile_with_title(colorize_diff(norm_log_ratio, scale=1.0), "median-aligned log2 ratio", "fixed [-1, 1]"),
        tile_with_title(colorize_scalar((night_norm - master_norm).abs(), cmap="inferno"), "|median-aligned diff|", "structure/shape disagreement"),
        tile_with_title(warp_img(master_warped), f"MASt3R warp {record['master_warp_psnr_db']:.2f} dB"),
        tile_with_title(warp_img(night_warped), f"Night3R warp {record['night_warp_psnr_db']:.2f} dB"),
        tile_with_title(colorize_scalar(master_err[0, 0], 0.0, 0.35, cmap="inferno"), "MASt3R warp error", "same error scale [0, 0.35]"),
        tile_with_title(colorize_scalar(night_err[0, 0], 0.0, 0.35, cmap="inferno"), "Night3R warp error", "same error scale [0, 0.35]"),
    ]
    w, h = tiles[0].size
    grid = Image.new("RGB", (4 * w, 4 * h), "white")
    for i, tile in enumerate(tiles):
        grid.paste(tile, ((i % 4) * w, (i // 4) * h))
    path.parent.mkdir(parents=True, exist_ok=True)
    grid.save(path)


def write_csv(path, records):
    if not records:
        return
    fieldnames = list(records[0].keys())
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)


def finite_mean(values):
    values = [float(v) for v in values if isinstance(v, (int, float)) and math.isfinite(float(v))]
    return float(np.mean(values)) if values else float("nan")


def write_markdown(path, records):
    lines = [
        "| Exposure | Scenes | MASt3R p98 | Night3R p98 | Delta p98 | MASt3R warp | Night3R warp | Delta warp | Night3R better p98 |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for exposure in sorted({record["exposure"] for record in records}, reverse=True):
        subset = [record for record in records if record["exposure"] == exposure]
        better = [
            record
            for record in subset
            if math.isfinite(record["master_p98"]) and math.isfinite(record["night_p98"]) and record["night_p98"] < record["master_p98"]
        ]
        lines.append(
            "| "
            f"{exposure:.2f} | {len(subset)} | "
            f"{finite_mean([r['master_p98'] for r in subset]):.2f} | "
            f"{finite_mean([r['night_p98'] for r in subset]):.2f} | "
            f"{finite_mean([r['delta_p98_night_minus_master'] for r in subset]):+.2f} | "
            f"{finite_mean([r['master_warp_psnr_db'] for r in subset]):.2f} | "
            f"{finite_mean([r['night_warp_psnr_db'] for r in subset]):.2f} | "
            f"{finite_mean([r['delta_warp_night_minus_master'] for r in subset]):+.2f} | "
            f"{len(better)}/{len(subset)} |"
        )
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")


def main():
    cli_args = parse_args()
    outdir = Path(cli_args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    base_cfg = load_config(cli_args.config)
    device = resolve_device(cli_args.device)
    set_seed(cli_args.seed)

    model_args = make_train_args(cli_args, outdir)
    first_cfg = apply_lowlight_exposure(base_cfg, cli_args.exposures[0])
    master = load_backbone(model_args, first_cfg).to(device).eval()
    night, load_info = build_event_model(model_args, first_cfg, device)

    records = []
    for exposure in cli_args.exposures:
        cfg = apply_lowlight_exposure(base_cfg, exposure)
        args = make_train_args(cli_args, outdir)
        dataset = build_dataset(args, cfg)
        selected = select_samples(dataset, cli_args.scenes, cli_args.frame_index)
        if not selected:
            raise RuntimeError("No samples selected. Check --scenes and --frame-index.")
        for sample_index in selected:
            batch_cpu = sample_to_batch(dataset, sample_index)
            batch_gpu = move_batch_to_device(batch_cpu, device)
            master_depth = predict_master(master, batch_gpu, args, cfg, device)
            night_depth = predict_night(night, batch_gpu, args, cfg, device)
            master_warped, master_valid = compute_warp(master_depth, batch_gpu)
            night_warped, night_valid = compute_warp(night_depth, batch_gpu)
            target = batch_gpu["images_photo"][:, 0].to(dtype=torch.float32)
            master_stat = depth_stats(resize_depth(master_depth.cpu(), target.shape[-2:]))
            night_stat = depth_stats(resize_depth(night_depth.cpu(), target.shape[-2:]))
            sample = dataset.samples[sample_index]
            sample_id = batch_cpu.get("sample_id", [""])[0]
            rec = {
                "exposure": float(exposure),
                "scene": str(sample["scene"]),
                "sample_id": str(sample_id),
                "idx1": int(sample["idx1"]),
                "idx2": int(sample["idx2"]),
                "master_median": master_stat["median"],
                "master_p98": master_stat["p98"],
                "night_median": night_stat["median"],
                "night_p98": night_stat["p98"],
                "delta_p98_night_minus_master": night_stat["p98"] - master_stat["p98"],
                "master_warp_psnr_db": warp_psnr(master_warped, target, master_valid),
                "night_warp_psnr_db": warp_psnr(night_warped, target, night_valid),
            }
            rec["delta_warp_night_minus_master"] = rec["night_warp_psnr_db"] - rec["master_warp_psnr_db"]
            rec.update(event_stats(batch_cpu))
            if cli_args.save_panels:
                panel_name = f"exp_{exposure:.2f}_{rec['scene']}_{rec['idx1']:06d}_{rec['idx2']:06d}.png"
                panel_path = outdir / "panels" / panel_name
                if cli_args.panel_protocol == "v2":
                    save_panel_v2(panel_path, batch_cpu, master_depth, night_depth, master_warped, night_warped, rec)
                else:
                    save_panel(panel_path, batch_cpu, master_depth, night_depth, master_warped, night_warped, rec)
                rec["panel"] = str(panel_path)
            records.append(rec)
            print(
                f"exp={exposure:.2f} {rec['sample_id']} "
                f"p98 M/N={rec['master_p98']:.2f}/{rec['night_p98']:.2f} "
                f"warp M/N={rec['master_warp_psnr_db']:.2f}/{rec['night_warp_psnr_db']:.2f}",
                flush=True,
            )

    write_csv(outdir / "mast3r_vs_night3r_lowlight_matrix.csv", records)
    write_markdown(outdir / "summary.md", records)
    manifest = {
        "data_root": str(Path(cli_args.data_root).resolve()),
        "config": str(Path(cli_args.config).resolve()),
        "model_name_or_path": str(Path(cli_args.model_name_or_path).resolve()),
        "trained_checkpoint": str(Path(cli_args.trained_checkpoint).resolve()),
        "trained_load_info": load_info,
        "frame_index": cli_args.frame_index,
        "pair_stride": cli_args.pair_stride,
        "panel_protocol": cli_args.panel_protocol,
        "exposures": cli_args.exposures,
        "scenes": cli_args.scenes,
        "records": records,
    }
    with open(outdir / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)
    print(json.dumps({"outdir": str(outdir.resolve()), "num_records": len(records)}, indent=2))


if __name__ == "__main__":
    main()
