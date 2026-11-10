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


def parse_args():
    parser = argparse.ArgumentParser("Render Event-Dark3R baseline/trained comparison panels")
    parser.add_argument("--trained-checkpoint", required=True)
    parser.add_argument("--config", default="config/train_event_dark3r.yaml")
    parser.add_argument("--data-root", default=None)
    parser.add_argument("--model-name-or-path", required=True)
    parser.add_argument("--allow-unsafe-checkpoint-load", action="store_true")
    parser.add_argument("--image-size", type=int, nargs=2, default=None, metavar=("H", "W"))
    parser.add_argument("--max-pairs", type=int, default=4)
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
    parser.add_argument("--outdir", required=True)
    return parser.parse_args()


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


def build_student(args, cfg, device, checkpoint=None):
    backbone = load_backbone(args, cfg)
    model = build_event_student(backbone, config=cfg)
    final_stage = resolve_stage_schedule(cfg, args)[-1]
    set_event_training_stage(
        model,
        stage=final_stage["stage"],
        partial_finetune_blocks=final_stage["partial_finetune_blocks"],
    )
    load_info = {"missing_keys": [], "unexpected_keys": [], "step": None}
    if checkpoint is not None:
        payload = safe_torch_load(
            checkpoint,
            map_location="cpu",
            allow_unsafe=args.allow_unsafe_checkpoint_load,
            purpose="Event-Dark3R visualization checkpoint",
        )
        state_dict = payload.get("model", payload)
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        load_info = {
            "missing_keys": list(missing),
            "unexpected_keys": list(unexpected),
            "step": payload.get("step"),
        }
    model = model.to(device)
    model.eval()
    return model, load_info


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
    for batch in loader:
        cached.append(batch)
        if len(cached) >= args.max_pairs:
            break
    return dataset, cached


def detach_batch(batch):
    return {key: value.detach().cpu() if torch.is_tensor(value) else value for key, value in batch.items()}


def tensor_image_chw_to_hwc(image):
    image = image.detach().cpu().float().clamp(0, 1)
    return image.permute(1, 2, 0).numpy()


def normalize_map(x, vmin=None, vmax=None):
    x = np.asarray(x, dtype=np.float32)
    finite = np.isfinite(x)
    if not finite.any():
        return x, 0.0, 1.0
    values = x[finite]
    if vmin is None:
        vmin = float(np.percentile(values, 2))
    if vmax is None:
        vmax = float(np.percentile(values, 98))
    if vmax <= vmin:
        vmax = vmin + 1e-6
    return x, vmin, vmax


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
                pred1, pred2 = model(batch_gpu)
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
                    "event_gate": pred1.get("event_gate", None),
                    "warped": None if warped is None else warped.detach().cpu(),
                    "warp_valid": None if warp_valid is None else warp_valid.detach().cpu(),
                }
            )
    return outputs


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


def render_panel(out_path, sample_index, batch, baseline, trained):
    image1 = batch["images_photo"][0, 0]
    image2 = batch["images_photo"][0, 1]
    dark1 = batch["images_photo_dark"][0, 0]
    event = batch["event_voxel"][0, 0].abs().sum(dim=0).float().numpy()
    gt_depth = as_bchw(batch["depth_gt"])[0, 0]
    gt_valid = as_bchw(batch.get("depth_valid", torch.isfinite(gt_depth) & (gt_depth > 0)))[0, 0].bool()
    base_depth = baseline["depth"][0, 0]
    trained_depth = trained["depth"][0, 0]

    if base_depth.shape != gt_depth.shape:
        base_depth = F.interpolate(base_depth[None, None], size=gt_depth.shape, mode="bilinear", align_corners=False)[0, 0]
    if trained_depth.shape != gt_depth.shape:
        trained_depth = F.interpolate(trained_depth[None, None], size=gt_depth.shape, mode="bilinear", align_corners=False)[0, 0]

    base_err = (base_depth - gt_depth).abs()
    trained_err = (trained_depth - gt_depth).abs()
    valid_np = gt_valid.numpy()
    gt_np, depth_vmin, depth_vmax = normalize_map(gt_depth.numpy()[valid_np])
    depth_vmin = float(np.percentile(gt_depth.numpy()[valid_np], 2))
    depth_vmax = float(np.percentile(gt_depth.numpy()[valid_np], 98))
    err_values = np.concatenate([base_err.numpy()[valid_np].reshape(-1), trained_err.numpy()[valid_np].reshape(-1)])
    err_vmax = float(np.percentile(err_values[np.isfinite(err_values)], 95))
    err_vmax = max(err_vmax, 1e-6)
    improvement = (base_err - trained_err).numpy()
    imp_abs = float(np.percentile(np.abs(improvement[valid_np]), 98))
    imp_abs = max(imp_abs, 1e-6)

    base_metrics = depth_metrics(base_depth[None, None], gt_depth[None, None], gt_valid[None, None])
    trained_metrics = depth_metrics(trained_depth[None, None], gt_depth[None, None], gt_valid[None, None])
    base_warp_psnr = warp_psnr(baseline["warped"], batch["images_photo"][:, 0], baseline["warp_valid"])
    trained_warp_psnr = warp_psnr(trained["warped"], batch["images_photo"][:, 0], trained["warp_valid"])

    fig, axes = plt.subplots(3, 4, figsize=(19, 12), constrained_layout=True)
    sample_id = batch.get("sample_id", [f"sample_{sample_index:03d}"])
    if isinstance(sample_id, (list, tuple)):
        sample_id = sample_id[0]
    fig.suptitle(str(sample_id), fontsize=14)

    panels = [
        (axes[0, 0], tensor_image_chw_to_hwc(image1), "RGB view1", None, None, None),
        (axes[0, 1], tensor_image_chw_to_hwc(dark1), "Low-light view1", None, None, None),
        (axes[0, 2], tensor_image_chw_to_hwc(image2), "RGB view2", None, None, None),
        (axes[0, 3], event, "Event activity", "magma", None, None),
        (axes[1, 0], gt_depth.numpy(), "GT depth", "viridis", depth_vmin, depth_vmax),
        (
            axes[1, 1],
            base_depth.numpy(),
            f"Baseline depth\nPSNR {base_metrics.get('psnr_db', float('nan')):.2f} dB",
            "viridis",
            depth_vmin,
            depth_vmax,
        ),
        (
            axes[1, 2],
            trained_depth.numpy(),
            f"Trained depth\nPSNR {trained_metrics.get('psnr_db', float('nan')):.2f} dB",
            "viridis",
            depth_vmin,
            depth_vmax,
        ),
        (axes[1, 3], improvement, "Abs error improvement\npositive = trained better", "coolwarm", -imp_abs, imp_abs),
        (
            axes[2, 0],
            base_err.numpy(),
            f"Baseline abs error\nAbsRel {base_metrics.get('absrel', float('nan')):.3f}",
            "inferno",
            0.0,
            err_vmax,
        ),
        (
            axes[2, 1],
            trained_err.numpy(),
            f"Trained abs error\nAbsRel {trained_metrics.get('absrel', float('nan')):.3f}",
            "inferno",
            0.0,
            err_vmax,
        ),
        (
            axes[2, 2],
            tensor_image_chw_to_hwc(baseline["warped"][0]) if baseline["warped"] is not None else np.zeros_like(tensor_image_chw_to_hwc(image1)),
            f"Baseline warp view2->view1\nRGB PSNR {base_warp_psnr:.2f} dB",
            None,
            None,
            None,
        ),
        (
            axes[2, 3],
            tensor_image_chw_to_hwc(trained["warped"][0]) if trained["warped"] is not None else np.zeros_like(tensor_image_chw_to_hwc(image1)),
            f"Trained warp view2->view1\nRGB PSNR {trained_warp_psnr:.2f} dB",
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
        "sample_id": str(sample_id),
        "panel": str(out_path),
        "baseline_depth": base_metrics,
        "trained_depth": trained_metrics,
        "baseline_warp_psnr_db": base_warp_psnr,
        "trained_warp_psnr_db": trained_warp_psnr,
    }


def write_index(outdir, records):
    html = [
        "<!doctype html><meta charset='utf-8'>",
        "<title>Event-Dark3R MVSEC Visual Comparison</title>",
        "<style>body{font-family:Arial,sans-serif;margin:24px;background:#f6f6f6;color:#111}"
        "img{max-width:100%;display:block;margin:12px 0 36px;border:1px solid #ccc;background:white}"
        "code{background:#eee;padding:2px 4px}</style>",
        "<h1>Event-Dark3R MVSEC Visual Comparison</h1>",
    ]
    for item in records:
        rel = Path(item["panel"]).name
        html.append(f"<h2>{item['sample_id']}</h2>")
        html.append(
            "<p>"
            f"depth PSNR: baseline {item['baseline_depth'].get('psnr_db', float('nan')):.2f} dB, "
            f"trained {item['trained_depth'].get('psnr_db', float('nan')):.2f} dB; "
            f"warp RGB PSNR: baseline {item['baseline_warp_psnr_db']:.2f} dB, "
            f"trained {item['trained_warp_psnr_db']:.2f} dB"
            "</p>"
        )
        html.append(f"<img src='{rel}' alt='{item['sample_id']}'>")
    with open(Path(outdir) / "index.html", "w") as f:
        f.write("\n".join(html))


def main():
    args = parse_args()
    cfg = load_config(args.config)
    train_args = make_train_args(args)
    device = resolve_device(args.device)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    dataset, batches = cache_batches(train_args, cfg)
    batches = [detach_batch(batch) for batch in batches]

    baseline_model, baseline_info = build_student(train_args, cfg, device, checkpoint=None)
    baseline_outputs = predict(baseline_model, batches, train_args, cfg, device)
    del baseline_model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    trained_model, trained_info = build_student(train_args, cfg, device, checkpoint=args.trained_checkpoint)
    trained_outputs = predict(trained_model, batches, train_args, cfg, device)
    del trained_model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    records = []
    for idx, (batch, base_out, trained_out) in enumerate(zip(batches, baseline_outputs, trained_outputs)):
        out_path = outdir / f"sample_{idx:03d}_panel.png"
        records.append(render_panel(out_path, idx, batch, base_out, trained_out))

    manifest = {
        "data_root": getattr(dataset, "root_paths_text", str(dataset.root_path)),
        "dataset_len": len(dataset),
        "trained_checkpoint": str(Path(args.trained_checkpoint).resolve()),
        "model_name_or_path": str(Path(args.model_name_or_path).resolve()),
        "baseline_load_info": baseline_info,
        "trained_load_info": trained_info,
        "records": records,
    }
    with open(outdir / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)
    write_index(outdir, records)
    print(json.dumps({"outdir": str(outdir.resolve()), "panels": [item["panel"] for item in records]}, indent=2))


if __name__ == "__main__":
    main()
