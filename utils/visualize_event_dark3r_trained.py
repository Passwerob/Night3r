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
from utils.eval_event_dark3r_variant import apply_event_mode  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser("Render trained Event-Dark3R prediction panels")
    parser.add_argument("--trained-checkpoint", required=True)
    parser.add_argument("--config", default="config/train_event_dark3r.yaml")
    parser.add_argument("--data-root", default=None)
    parser.add_argument("--model-name-or-path", required=True)
    parser.add_argument("--allow-unsafe-checkpoint-load", action="store_true")
    parser.add_argument("--image-size", type=int, nargs=2, default=None, metavar=("H", "W"))
    parser.add_argument("--max-pairs", type=int, default=6)
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
    parser.add_argument(
        "--no-lowlight-aug",
        action="store_true",
        help="Disable low-light augmentation so night datasets are used directly.",
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
    parser.add_argument(
        "--event-mode",
        choices=("real", "zero", "shuffle"),
        default="real",
        help="Event ablation mode for the trained Event-Dark3R model.",
    )
    parser.add_argument("--outdir", required=True)
    return parser.parse_args()


def apply_lowlight_overrides(cfg, args):
    cfg = dict(cfg)
    lowlight_cfg = dict(cfg.get("lowlight_aug", {}))
    if args.no_lowlight_aug:
        lowlight_cfg["enabled"] = False
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


def build_trained_model(args, cfg, device, checkpoint):
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
        purpose="Event-Dark3R trained visualization checkpoint",
    )
    state_dict = payload.get("model", payload)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    model = model.to(device)
    model.eval()
    return model, {
        "missing_keys": list(missing),
        "unexpected_keys": list(unexpected),
        "step": payload.get("step"),
    }


def predict(model, batches, args, cfg, device):
    amp_enabled, _, amp_dtype = resolve_amp_config(cfg, args, device)
    outputs = []
    with torch.no_grad():
        for batch in batches:
            batch_gpu = move_batch_to_device(batch, device)
            batch_gpu = apply_event_mode(batch_gpu, getattr(args, "event_mode", "real"), batch_idx=len(outputs))
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
            if "images_photo" in batch_gpu and "intrinsics" in batch_gpu and "relative_pose" in batch_gpu:
                image2 = batch_gpu["images_photo"][:, 1].to(dtype=torch.float32)
                pred_depth = depth.to(dtype=torch.float32)
                if pred_depth.shape[-2:] != image2.shape[-2:]:
                    pred_depth = F.interpolate(
                        pred_depth,
                        size=image2.shape[-2:],
                        mode="bilinear",
                        align_corners=False,
                    )
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


def tensor_image_chw_to_hwc(image):
    image = image.detach().cpu().float().clamp(0, 1)
    return image.permute(1, 2, 0).numpy()


def finite_percentile(values, percentile, default=0.0):
    values = np.asarray(values, dtype=np.float32)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return float(default)
    return float(np.percentile(values, percentile))


def depth_metrics(pred_depth, gt_depth, valid):
    pred_depth = pred_depth.float()
    gt_depth = gt_depth.float()
    if pred_depth.shape[-2:] != gt_depth.shape[-2:]:
        pred_depth = F.interpolate(pred_depth, size=gt_depth.shape[-2:], mode="bilinear", align_corners=False)
    valid = valid.bool() & torch.isfinite(pred_depth) & torch.isfinite(gt_depth) & (pred_depth > 0) & (gt_depth > 0)
    if not valid.any():
        return {}
    pred = pred_depth[valid].clamp_min(1e-6)
    gt = gt_depth[valid].clamp_min(1e-6)
    diff = pred - gt
    mse = float((diff * diff).mean().detach().cpu())
    ratio = torch.maximum(pred / gt, gt / pred)
    max_depth = float(gt.max().detach().cpu())
    psnr = 20.0 * math.log10(max_depth) - 10.0 * math.log10(max(mse, 1e-12))
    return {
        "absrel": float(((pred - gt).abs() / gt).mean().detach().cpu()),
        "rmse": math.sqrt(max(mse, 0.0)),
        "psnr_db": psnr,
        "delta_1.25": float((ratio < 1.25).float().mean().detach().cpu()),
        "valid_px": int(valid.sum().item()),
    }


def warp_psnr(warped, image1, valid):
    if warped is None or valid is None:
        return float("nan")
    warped = warped.float()
    image1 = image1.float()
    valid = valid.bool().expand_as(image1) & torch.isfinite(warped) & torch.isfinite(image1)
    if not valid.any():
        return float("nan")
    mse = float(((warped[valid] - image1[valid]) ** 2).mean().detach().cpu())
    return -10.0 * math.log10(max(mse, 1e-12))


def sample_id_from_batch(batch, sample_index):
    sample_id = batch.get("sample_id", [f"sample_{sample_index:03d}"])
    if isinstance(sample_id, (list, tuple)):
        return str(sample_id[0])
    return str(sample_id)


def render_panel(out_path, sample_index, batch, pred):
    image1 = batch["images_photo"][0, 0]
    image2 = batch["images_photo"][0, 1]
    dark1 = batch.get("images_photo_dark", batch["images_photo"])[0, 0]
    event = batch["event_voxel"][0, 0].abs().sum(dim=0).float().numpy()
    gt_depth = as_bchw(batch["depth_gt"])[0, 0]
    default_valid = torch.isfinite(gt_depth) & (gt_depth > 0)
    gt_valid = as_bchw(batch.get("depth_valid", default_valid))[0, 0].bool()
    pred_depth = pred["depth"][0, 0]

    if pred_depth.shape != gt_depth.shape:
        pred_depth = F.interpolate(pred_depth[None, None], size=gt_depth.shape, mode="bilinear", align_corners=False)[0, 0]

    pred_err = (pred_depth - gt_depth).abs()
    valid_np = gt_valid.numpy() & np.isfinite(gt_depth.numpy()) & (gt_depth.numpy() > 0)
    if valid_np.any():
        depth_vmin = finite_percentile(gt_depth.numpy()[valid_np], 2)
        depth_vmax = finite_percentile(gt_depth.numpy()[valid_np], 98, default=depth_vmin + 1.0)
        err_vmax = finite_percentile(pred_err.numpy()[valid_np], 95, default=1.0)
    else:
        depth_vmin, depth_vmax, err_vmax = 0.0, 1.0, 1.0
    if depth_vmax <= depth_vmin:
        depth_vmax = depth_vmin + 1e-6
    err_vmax = max(err_vmax, 1e-6)

    metrics = depth_metrics(pred_depth[None, None], gt_depth[None, None], gt_valid[None, None])
    pred_warp_psnr = warp_psnr(pred["warped"], batch["images_photo"][:, 0], pred["warp_valid"])

    fig, axes = plt.subplots(2, 4, figsize=(19, 8), constrained_layout=True)
    sample_id = sample_id_from_batch(batch, sample_index)
    fig.suptitle(str(sample_id), fontsize=14)

    panels = [
        (axes[0, 0], tensor_image_chw_to_hwc(image1), "RGB view1", None, None, None),
        (axes[0, 1], tensor_image_chw_to_hwc(dark1), "Model input view1", None, None, None),
        (axes[0, 2], tensor_image_chw_to_hwc(image2), "RGB view2", None, None, None),
        (axes[0, 3], event, "Event activity", "magma", None, None),
        (axes[1, 0], gt_depth.numpy(), "GT depth", "viridis", depth_vmin, depth_vmax),
        (
            axes[1, 1],
            pred_depth.numpy(),
            f"Pred depth\nPSNR {metrics.get('psnr_db', float('nan')):.2f} dB",
            "viridis",
            depth_vmin,
            depth_vmax,
        ),
        (
            axes[1, 2],
            pred_err.numpy(),
            f"Abs error\nAbsRel {metrics.get('absrel', float('nan')):.3f}",
            "inferno",
            0.0,
            err_vmax,
        ),
        (
            axes[1, 3],
            tensor_image_chw_to_hwc(pred["warped"][0])
            if pred["warped"] is not None
            else np.zeros_like(tensor_image_chw_to_hwc(image1)),
            f"Warp view2->view1\nRGB PSNR {pred_warp_psnr:.2f} dB",
            None,
            None,
            None,
        ),
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
        "trained_depth": metrics,
        "trained_warp_psnr_db": pred_warp_psnr,
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


def render_panel_no_gt(out_path, sample_index, batch, pred):
    image1 = batch["images_photo"][0, 0]
    image2 = batch["images_photo"][0, 1]
    dark1 = batch.get("images_photo_dark", batch["images_photo"])[0, 0]
    dark2 = batch.get("images_photo_dark", batch["images_photo"])[0, 1]
    event = batch["event_voxel"][0, 0].abs().sum(dim=0).float().numpy()
    pred_depth = pred["depth"][0, 0]
    target_shape = image1.shape[-2:]
    if pred_depth.shape != target_shape:
        pred_depth = F.interpolate(
            pred_depth[None, None],
            size=target_shape,
            mode="bilinear",
            align_corners=False,
        )[0, 0]

    pred_np = pred_depth.numpy()
    depth_vmin = finite_percentile(pred_np.reshape(-1), 2, default=0.0)
    depth_vmax = finite_percentile(pred_np.reshape(-1), 98, default=1.0)
    if depth_vmax <= depth_vmin:
        depth_vmax = depth_vmin + 1e-6
    stats = depth_stats(pred_depth)
    pred_warp_psnr = warp_psnr(pred["warped"], batch["images_photo"][:, 0], pred["warp_valid"])

    fig, axes = plt.subplots(2, 4, figsize=(19, 8), constrained_layout=True)
    sample_id = sample_id_from_batch(batch, sample_index)
    fig.suptitle(f"{sample_id} (no GT depth)", fontsize=14)

    panels = [
        (axes[0, 0], tensor_image_chw_to_hwc(image1), "RGB view1", None, None, None),
        (axes[0, 1], tensor_image_chw_to_hwc(dark1), "Model input view1", None, None, None),
        (axes[0, 2], tensor_image_chw_to_hwc(image2), "RGB view2", None, None, None),
        (axes[0, 3], event, "Event activity", "magma", None, None),
        (
            axes[1, 0],
            pred_np,
            f"Pred depth\nmedian {stats['median']:.2f}",
            "viridis",
            depth_vmin,
            depth_vmax,
        ),
        (
            axes[1, 1],
            pred_np,
            f"Pred depth detail\np02 {stats['p02']:.2f} / p98 {stats['p98']:.2f}",
            "viridis",
            depth_vmin,
            depth_vmax,
        ),
        (
            axes[1, 2],
            tensor_image_chw_to_hwc(pred["warped"][0])
            if pred["warped"] is not None
            else np.zeros_like(tensor_image_chw_to_hwc(image1)),
            f"Warp view2->view1\nRGB PSNR {pred_warp_psnr:.2f} dB",
            None,
            None,
            None,
        ),
        (axes[1, 3], tensor_image_chw_to_hwc(dark2), "Model input view2", None, None, None),
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
        "trained_depth": {},
        "trained_depth_stats": stats,
        "trained_warp_psnr_db": pred_warp_psnr,
    }


def write_index(outdir, records):
    html = [
        "<!doctype html><meta charset='utf-8'>",
        "<title>Event-Dark3R MVSEC Trained Visualization</title>",
        "<style>body{font-family:Arial,sans-serif;margin:24px;background:#f6f6f6;color:#111}"
        "img{max-width:100%;display:block;margin:12px 0 36px;border:1px solid #ccc;background:white}"
        "code{background:#eee;padding:2px 4px}</style>",
        "<h1>Event-Dark3R MVSEC Trained Visualization</h1>",
    ]
    for item in records:
        rel = Path(item["panel"]).name
        html.append(f"<h2>{item['sample_id']}</h2>")
        html.append(
            "<p>"
            f"depth PSNR: {item['trained_depth'].get('psnr_db', float('nan')):.2f} dB; "
            f"AbsRel: {item['trained_depth'].get('absrel', float('nan')):.3f}; "
            f"warp RGB PSNR: {item['trained_warp_psnr_db']:.2f} dB"
            "</p>"
        )
        html.append(f"<img src='{rel}' alt='{item['sample_id']}'>")
    with open(Path(outdir) / "index.html", "w") as f:
        f.write("\n".join(html))


def main():
    args = parse_args()
    cfg = load_config(args.config)
    cfg = apply_lowlight_overrides(cfg, args)
    train_args = make_train_args(args)
    device = resolve_device(args.device)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    dataset, batches = cache_batches(train_args, cfg)
    model, load_info = build_trained_model(train_args, cfg, device, args.trained_checkpoint)
    predictions = predict(model, batches, train_args, cfg, device)
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    records = []
    for idx, (batch, pred) in enumerate(zip(batches, predictions)):
        out_path = outdir / f"sample_{idx:03d}_trained_panel.png"
        if "depth_gt" in batch:
            records.append(render_panel(out_path, idx, batch, pred))
        else:
            records.append(render_panel_no_gt(out_path, idx, batch, pred))

    manifest = {
        "data_root": getattr(dataset, "root_paths_text", str(dataset.root_path)),
        "dataset_len": len(dataset),
        "trained_checkpoint": str(Path(args.trained_checkpoint).resolve()),
        "model_name_or_path": str(Path(args.model_name_or_path).resolve()),
        "lowlight_aug_enabled": bool(cfg.get("lowlight_aug", {}).get("enabled", False)),
        "lowlight_aug": cfg.get("lowlight_aug", {}),
        "event_mode": args.event_mode,
        "trained_load_info": load_info,
        "records": records,
    }
    with open(outdir / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)
    write_index(outdir, records)
    print(json.dumps({"outdir": str(outdir.resolve()), "panels": [item["panel"] for item in records]}, indent=2))


if __name__ == "__main__":
    main()
