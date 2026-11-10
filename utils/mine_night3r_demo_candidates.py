#!/usr/bin/env python3
import argparse
import csv
import html
import json
import math
import os
import re
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
from PIL import Image
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from model.event_dark3r_student import make_view_pair_from_batch  # noqa: E402
from train_event_dark3r import (  # noqa: E402
    build_dataset,
    load_config,
    move_batch_to_device,
    resolve_amp_config,
)
from utils.compare_event_dark3r_checkpoint import as_bchw, depth_from_pred  # noqa: E402
from utils.eval_event_dark3r_variant import build_model  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser("Mine reproducible Night3R demo candidates")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--model-name-or-path", required=True)
    parser.add_argument(
        "--datasets",
        nargs="+",
        required=True,
        metavar="NAME=ROOT",
        help="Dataset labels and roots, for example mvsec=/path dsec=/path.",
    )
    parser.add_argument("--outdir", required=True)
    parser.add_argument("--max-pairs-per-dataset", type=int, default=256)
    parser.add_argument("--top-k-per-dataset", type=int, default=12)
    parser.add_argument(
        "--diversity-bucket-size",
        type=int,
        default=16,
        help="Keep only the best pair per contiguous traversal bucket before ranking.",
    )
    parser.add_argument("--image-size", type=int, nargs=2, default=[512, 512])
    parser.add_argument("--amp-dtype", choices=("fp16", "bf16"), default="bf16")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="cuda")
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--allow-unsafe-checkpoint-load", action="store_true")
    return parser.parse_args()


def parse_datasets(items):
    datasets = []
    for item in items:
        if "=" not in item:
            raise ValueError(f"Invalid --datasets item: {item!r}; expected NAME=ROOT")
        name, root = item.split("=", 1)
        if not name or not root:
            raise ValueError(f"Invalid --datasets item: {item!r}")
        datasets.append((name, root))
    return datasets


def make_train_args(args, data_root):
    return SimpleNamespace(
        config=args.config,
        data_root=data_root,
        save_dir=args.outdir,
        model_name_or_path=args.model_name_or_path,
        dark3r_lora_checkpoint=None,
        allow_unsafe_checkpoint_load=args.allow_unsafe_checkpoint_load,
        use_tiny_model=False,
        image_size=args.image_size,
        batch_size=1,
        max_batches=args.max_pairs_per_dataset,
        num_epochs=None,
        max_steps=None,
        max_scenes=None,
        max_pairs_per_scene=None,
        pair_stride=1,
        pair_sampling="fixed",
        min_pair_distance=1,
        max_pair_distance=1,
        precomputed_pairs=None,
        allow_reverse_pairs=False,
        symmetric_pairs=False,
        num_workers=0,
        stage=None,
        partial_finetune_blocks=None,
        amp=True,
        amp_dtype=args.amp_dtype,
        gradient_checkpointing=False,
        device=args.device,
        log_every=1,
        no_save=True,
        seed=args.seed,
        depth_min=None,
        depth_max=None,
    )


def make_model_args(args, variant):
    return SimpleNamespace(
        variant=variant,
        checkpoint=args.checkpoint if variant == "event_dark3r" else None,
        dark3r_lora_checkpoint=None,
        model_name_or_path=args.model_name_or_path,
        allow_unsafe_checkpoint_load=args.allow_unsafe_checkpoint_load,
    )


def resolve_device(name):
    if name == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is unavailable")
        return torch.device("cuda")
    if name == "cpu":
        return torch.device("cpu")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def cpu_batch(batch):
    return {
        key: value.detach().cpu() if torch.is_tensor(value) else value
        for key, value in batch.items()
    }


def resize_depth(depth, shape):
    depth = as_bchw(depth).float()
    if depth.shape[-2:] != shape:
        depth = F.interpolate(depth, size=shape, mode="bilinear", align_corners=False)
    return depth


def depth_metrics(pred_depth, gt_depth, valid):
    gt_depth = as_bchw(gt_depth).float()
    valid = as_bchw(valid).bool()
    pred_depth = resize_depth(pred_depth, gt_depth.shape[-2:])
    valid = valid & torch.isfinite(gt_depth) & torch.isfinite(pred_depth)
    valid = valid & (gt_depth > 0) & (pred_depth > 0)
    if not valid.any():
        return None, pred_depth

    gt = gt_depth[valid].clamp_min(1e-6)
    pred = pred_depth[valid].clamp_min(1e-6)
    raw_absrel = float(((pred - gt).abs() / gt).mean())
    scale = float(torch.median(gt) / torch.median(pred).clamp_min(1e-6))
    aligned = pred_depth * scale
    aligned_values = aligned[valid].clamp_min(1e-6)
    median_absrel = float(((aligned_values - gt).abs() / gt).mean())
    ratio = torch.maximum(aligned_values / gt, gt / aligned_values)
    return {
        "raw_absrel": raw_absrel,
        "median_aligned_absrel": median_absrel,
        "median_scale": scale,
        "median_aligned_delta_1.25": float((ratio < 1.25).float().mean()),
        "valid_px": int(valid.sum()),
    }, aligned


def sample_id_from_batch(batch, fallback):
    value = batch.get("sample_id", fallback)
    if isinstance(value, (list, tuple)):
        value = value[0]
    return str(value)


def infer_triplet(master, night, batch, train_args, cfg, device):
    batch_gpu = move_batch_to_device(batch, device)
    amp_enabled, _, amp_dtype = resolve_amp_config(cfg, train_args, device)
    with torch.no_grad(), torch.amp.autocast(
        device_type=device.type,
        dtype=amp_dtype,
        enabled=bool(amp_enabled and device.type == "cuda"),
    ):
        view1, view2 = make_view_pair_from_batch(batch_gpu, prefer_dark=True)
        master_pred, _ = master(view1, view2)
        night_pred, _ = night(batch_gpu)
        zero_batch = dict(batch_gpu)
        zero_batch["event_voxel"] = torch.zeros_like(batch_gpu["event_voxel"])
        zero_pred, _ = night(zero_batch)

    gate_mean = night_pred.get("event_gate", float("nan"))
    if torch.is_tensor(gate_mean):
        gate_mean = float(gate_mean.detach().float().cpu())
    else:
        gate_mean = float(gate_mean)

    return {
        "master": depth_from_pred(master_pred).detach().cpu(),
        "night": depth_from_pred(night_pred).detach().cpu(),
        "zero": depth_from_pred(zero_pred).detach().cpu(),
        "gate_map": None
        if night_pred.get("event_gate_map") is None
        else night_pred["event_gate_map"].detach().cpu(),
        "gate_mean": gate_mean,
    }


def evaluate_candidate(dataset_name, index, batch, outputs, bucket_size):
    gt = as_bchw(batch["depth_gt"]).float()
    valid = as_bchw(batch.get("depth_valid", torch.isfinite(gt) & (gt > 0))).bool()
    master_metrics, master_aligned = depth_metrics(outputs["master"], gt, valid)
    night_metrics, night_aligned = depth_metrics(outputs["night"], gt, valid)
    zero_metrics, zero_aligned = depth_metrics(outputs["zero"], gt, valid)
    if master_metrics is None or night_metrics is None or zero_metrics is None:
        return None

    sample_id = sample_id_from_batch(batch, f"{dataset_name}_{index:06d}")
    scene = sample_id.split(":", 1)[0]
    geometry_gain = master_metrics["median_aligned_absrel"] - night_metrics["median_aligned_absrel"]
    event_gain = zero_metrics["median_aligned_absrel"] - night_metrics["median_aligned_absrel"]
    score = geometry_gain + 0.5 * event_gain
    event = batch["event_voxel"].float()
    activity = event.abs().sum(dim=(-3, -2, -1)).mean()

    return {
        "dataset": dataset_name,
        "dataset_index": index,
        "sample_id": sample_id,
        "scene": scene,
        "diversity_key": f"{scene}:bucket-{index // max(bucket_size, 1):04d}",
        "score": float(score),
        "geometry_gain": float(geometry_gain),
        "event_gain": float(event_gain),
        "event_activity_mean": float(activity),
        "event_gate_mean": outputs["gate_mean"],
        "master_metrics": master_metrics,
        "night_metrics": night_metrics,
        "zero_metrics": zero_metrics,
        "batch": cpu_batch(batch),
        "depths": {
            "master_raw": outputs["master"],
            "night_raw": outputs["night"],
            "zero_raw": outputs["zero"],
            "master_aligned": master_aligned,
            "night_aligned": night_aligned,
            "zero_aligned": zero_aligned,
            "gate_map": outputs["gate_map"],
        },
    }


def to_rgb_uint8(image):
    image = image.detach().cpu().float().clamp(0, 1)
    if image.ndim == 4:
        image = image[0]
    return (image.permute(1, 2, 0).numpy() * 255.0 + 0.5).astype(np.uint8)


def save_rgb(path, image):
    Image.fromarray(to_rgb_uint8(image)).save(path)


def safe_name(value):
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_")


def valid_depth_limits(gt, valid):
    values = gt[valid & torch.isfinite(gt) & (gt > 0)].numpy()
    if values.size == 0:
        return 0.0, 1.0
    lo, hi = np.percentile(values, [2, 98])
    return float(lo), float(max(hi, lo + 1e-6))


def save_candidate(outdir, rank, item):
    sample_dir = outdir / f"rank_{rank:02d}_{safe_name(item['sample_id'])}"
    sample_dir.mkdir(parents=True, exist_ok=True)
    batch = item.pop("batch")
    depths = item.pop("depths")

    images = batch.get("images_photo_dark", batch.get("images_photo", batch["images"]))[0]
    save_rgb(sample_dir / "rgb_view1.png", images[0])
    save_rgb(sample_dir / "rgb_view2.png", images[1])

    event = batch["event_voxel"][0].float()
    event_activity = event[0].abs().sum(dim=0).numpy()
    np.savez_compressed(sample_dir / "event_voxel.npz", event_voxel=event.numpy())
    vmax_event = max(float(np.percentile(event_activity, 99)), 1e-6)
    plt.imsave(sample_dir / "event_activity.png", event_activity, cmap="magma", vmin=0, vmax=vmax_event)

    gt = as_bchw(batch["depth_gt"])[0, 0].float()
    valid = as_bchw(batch.get("depth_valid", torch.isfinite(gt) & (gt > 0)))[0, 0].bool()
    arrays = {"gt_depth": gt.numpy(), "depth_valid": valid.numpy()}
    for name, value in depths.items():
        if value is not None:
            arrays[name] = value.detach().cpu().float().squeeze().numpy()
    np.savez_compressed(sample_dir / "geometry_arrays.npz", **arrays)

    gate = arrays.get("gate_map")
    if gate is not None:
        plt.imsave(sample_dir / "event_gate.png", gate, cmap="viridis")

    lo, hi = valid_depth_limits(gt, valid)
    master = torch.from_numpy(arrays["master_aligned"]).float()
    night = torch.from_numpy(arrays["night_aligned"]).float()
    zero = torch.from_numpy(arrays["zero_aligned"]).float()
    master_err = (master - gt).abs().numpy()
    night_err = (night - gt).abs().numpy()
    error_values = np.concatenate([master_err[valid.numpy()], night_err[valid.numpy()]])
    err_hi = max(float(np.percentile(error_values, 95)), 1e-6)

    fig, axes = plt.subplots(2, 5, figsize=(19, 8), constrained_layout=True)
    panels = [
        (axes[0, 0], to_rgb_uint8(images[0]), "RGB view 1", None, None, None),
        (axes[0, 1], to_rgb_uint8(images[1]), "RGB view 2", None, None, None),
        (axes[0, 2], event_activity, "Event activity", "magma", 0, vmax_event),
        (axes[0, 3], gate if gate is not None else np.zeros((8, 8)), "BRG gate", "viridis", None, None),
        (axes[0, 4], gt.numpy(), "GT depth", "viridis", lo, hi),
        (axes[1, 0], master.numpy(), f"MASt3R aligned\nAbsRel {item['master_metrics']['median_aligned_absrel']:.3f}", "viridis", lo, hi),
        (axes[1, 1], night.numpy(), f"Night3R real event\nAbsRel {item['night_metrics']['median_aligned_absrel']:.3f}", "viridis", lo, hi),
        (axes[1, 2], zero.numpy(), f"Night3R zero event\nAbsRel {item['zero_metrics']['median_aligned_absrel']:.3f}", "viridis", lo, hi),
        (axes[1, 3], master_err, "MASt3R abs error", "inferno", 0, err_hi),
        (axes[1, 4], night_err, "Night3R abs error", "inferno", 0, err_hi),
    ]
    for ax, data, title, cmap, vmin, vmax in panels:
        if cmap is None:
            ax.imshow(data)
        else:
            ax.imshow(data, cmap=cmap, vmin=vmin, vmax=vmax)
        ax.set_title(title)
        ax.axis("off")
    fig.suptitle(
        f"{item['sample_id']} | score={item['score']:.4f} | "
        f"geometry gain={item['geometry_gain']:.4f} | event gain={item['event_gain']:.4f}"
    )
    panel_path = sample_dir / "comparison_panel.png"
    fig.savefig(panel_path, dpi=140)
    plt.close(fig)

    item["rank"] = rank
    item["asset_dir"] = str(sample_dir)
    item["panel"] = str(panel_path)
    with open(sample_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(item, f, indent=2)
    return item


def write_index(outdir, records):
    rows = [
        "<!doctype html><meta charset='utf-8'>",
        "<title>Night3R demo candidates</title>",
        "<style>body{font-family:Arial,sans-serif;margin:24px;background:#f5f5f5;color:#111}"
        "img{max-width:100%;background:white;border:1px solid #bbb;margin-bottom:32px}"
        "table{border-collapse:collapse;background:white}td,th{padding:6px 10px;border:1px solid #ccc}</style>",
        "<h1>Night3R demo candidates</h1>",
    ]
    for item in records:
        panel = Path(item["panel"]).relative_to(outdir)
        rows.append(f"<h2>#{item['rank']} {html.escape(item['sample_id'])}</h2>")
        rows.append(
            f"<p>dataset={html.escape(item['dataset'])}, score={item['score']:.4f}, "
            f"geometry gain={item['geometry_gain']:.4f}, event gain={item['event_gain']:.4f}</p>"
        )
        rows.append(f"<img src='{html.escape(str(panel))}'>")
    with open(outdir / "index.html", "w", encoding="utf-8") as f:
        f.write("\n".join(rows))


def main():
    args = parse_args()
    set_seed(args.seed)
    device = resolve_device(args.device)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    cfg = load_config(args.config)
    cfg = dict(cfg)
    lowlight_cfg = dict(cfg.get("lowlight_aug", {}))
    lowlight_cfg["enabled"] = False
    cfg["lowlight_aug"] = lowlight_cfg

    first_root = parse_datasets(args.datasets)[0][1]
    model_train_args = make_train_args(args, first_root)
    master, master_info = build_model(
        make_model_args(args, "mast3r"), cfg, model_train_args, device
    )
    night, night_info = build_model(
        make_model_args(args, "event_dark3r"), cfg, model_train_args, device
    )

    all_selected = []
    traversal = {}
    for dataset_name, data_root in parse_datasets(args.datasets):
        train_args = make_train_args(args, data_root)
        dataset = build_dataset(train_args, cfg)
        loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)
        best_by_bucket = {}
        processed = 0
        for index, batch in enumerate(loader):
            if index >= args.max_pairs_per_dataset:
                break
            outputs = infer_triplet(master, night, batch, train_args, cfg, device)
            candidate = evaluate_candidate(
                dataset_name, index, batch, outputs, args.diversity_bucket_size
            )
            processed += 1
            if candidate is None:
                continue
            key = candidate["diversity_key"]
            if key not in best_by_bucket or candidate["score"] > best_by_bucket[key]["score"]:
                best_by_bucket[key] = candidate
            if processed % 16 == 0:
                print(
                    json.dumps(
                        {
                            "dataset": dataset_name,
                            "processed": processed,
                            "retained_buckets": len(best_by_bucket),
                        }
                    ),
                    flush=True,
                )

        selected = sorted(best_by_bucket.values(), key=lambda item: item["score"], reverse=True)
        selected = selected[: args.top_k_per_dataset]
        traversal[dataset_name] = {
            "root": data_root,
            "dataset_len": len(dataset),
            "processed": processed,
            "retained_buckets": len(best_by_bucket),
            "selected": len(selected),
        }
        all_selected.extend(selected)

    saved = []
    for dataset_name, _ in parse_datasets(args.datasets):
        dataset_items = [item for item in all_selected if item["dataset"] == dataset_name]
        dataset_dir = outdir / dataset_name
        dataset_dir.mkdir(parents=True, exist_ok=True)
        for rank, item in enumerate(dataset_items, start=1):
            saved.append(save_candidate(dataset_dir, rank, item))

    manifest = {
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "config": str(Path(args.config).resolve()),
        "model_name_or_path": str(Path(args.model_name_or_path).resolve()),
        "selection_score": "(MASt3R_med_absrel - Night3R_real_med_absrel) + 0.5 * (Night3R_zero_med_absrel - Night3R_real_med_absrel)",
        "master_load_info": master_info,
        "night_load_info": night_info,
        "traversal": traversal,
        "records": saved,
    }
    with open(outdir / "manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    csv_fields = [
        "dataset",
        "rank",
        "sample_id",
        "score",
        "geometry_gain",
        "event_gain",
        "event_activity_mean",
        "event_gate_mean",
        "asset_dir",
        "panel",
    ]
    with open(outdir / "ranking.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=csv_fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(saved)
    write_index(outdir, saved)
    print(json.dumps({"outdir": str(outdir.resolve()), "saved": len(saved)}, indent=2))


if __name__ == "__main__":
    main()
