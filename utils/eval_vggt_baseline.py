#!/usr/bin/env python3
"""Evaluate official VGGT or EventVGGT on the Night3R fixed-pair protocol."""

import argparse
import json
import random
import statistics
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn.functional as F


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from train_event_dark3r import build_dataset, load_config  # noqa: E402
from utils.compare_event_dark3r_checkpoint import (  # noqa: E402
    DenseMetricAccumulator,
    as_bchw,
)
from utils.depth_cutoff_metrics import DepthCutoffAccumulator  # noqa: E402
from utils.eval_event_dark3r_variant import apply_event_mode  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser("Evaluate VGGT/EventVGGT on Night3R data")
    parser.add_argument("--variant", required=True, choices=("vggt", "eventvggt"))
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--vggt-root", default="/data/fcr/vggt")
    parser.add_argument("--eventvggt-root", default="/data/fcr/EventVGGT")
    parser.add_argument("--max-batches", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--image-size", type=int, nargs=2, default=(512, 512))
    parser.add_argument("--pair-stride", type=int, default=1)
    parser.add_argument("--pair-sampling", choices=("fixed", "random", "precomputed"), default="fixed")
    parser.add_argument("--min-pair-distance", type=int, default=1)
    parser.add_argument("--max-pair-distance", type=int, default=1)
    parser.add_argument("--max-scenes", type=int, default=None)
    parser.add_argument("--max-pairs-per-scene", type=int, default=None)
    parser.add_argument("--precomputed-pairs", default=None)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--amp-dtype", choices=("bf16", "fp16"), default="bf16")
    parser.add_argument("--depth-min", type=float, default=None)
    parser.add_argument("--depth-max", type=float, default=None)
    parser.add_argument("--event-mode", choices=("real", "zero", "shuffle"), default="real")
    parser.add_argument(
        "--event-size",
        type=int,
        nargs=2,
        default=None,
        metavar=("H", "W"),
        help="EventVGGT input size. Defaults to 252x336 for MVSEC and 252x504 otherwise.",
    )
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--prediction-dir", type=Path, default=None)
    parser.add_argument("--prediction-count", type=int, default=0)
    return parser.parse_args()


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def dataset_args(args):
    return SimpleNamespace(
        data_root=args.data_root,
        image_size=tuple(args.image_size),
        pair_stride=args.pair_stride,
        max_scenes=args.max_scenes,
        max_pairs_per_scene=args.max_pairs_per_scene,
        pair_sampling=args.pair_sampling,
        min_pair_distance=args.min_pair_distance,
        max_pair_distance=args.max_pair_distance,
        precomputed_pairs=args.precomputed_pairs,
        allow_reverse_pairs=False,
    )


def build_loader(args, cfg, device):
    dataset = build_dataset(dataset_args(args), cfg)
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )
    return dataset, loader


def load_vggt(args, device):
    sys.path.insert(0, str(Path(args.vggt_root).resolve()))
    from vggt.models.vggt import VGGT

    model = VGGT(
        enable_camera=False,
        enable_point=False,
        enable_depth=True,
        enable_track=False,
    )
    try:
        payload = torch.load(
            args.checkpoint,
            map_location="cpu",
            weights_only=True,
            mmap=True,
        )
    except TypeError:
        payload = torch.load(args.checkpoint, map_location="cpu")
    state = payload.get("model", payload) if isinstance(payload, dict) else payload
    incompatible = model.load_state_dict(state, strict=False)
    model = model.eval().to(device)
    return model, {
        "implementation": str(Path(args.vggt_root).resolve()),
        "missing_key_count": len(incompatible.missing_keys),
        "unexpected_key_count": len(incompatible.unexpected_keys),
        "unexpected_key_groups": sorted(
            {key.split(".", 1)[0] for key in incompatible.unexpected_keys}
        ),
        "note": "Unused camera, point, and track heads are omitted to reduce evaluation memory.",
    }


def load_eventvggt(args, device):
    sys.path.insert(0, str(Path(args.eventvggt_root).resolve()))
    from infer import load_model

    model = load_model(
        args.checkpoint,
        device=str(device),
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
    )
    return model, {
        "implementation": str(Path(args.eventvggt_root).resolve()),
        "lora_r": args.lora_r,
        "lora_alpha": args.lora_alpha,
    }


def preprocess_vggt(images):
    """Use the official 518-square model size on the common Night3R FOV."""
    b, s, c, _, _ = images.shape
    images = F.interpolate(
        images.reshape(b * s, c, *images.shape[-2:]),
        size=(518, 518),
        mode="bicubic",
        align_corners=False,
    )
    return images.reshape(b, s, c, 518, 518).clamp(0, 1)


def event_voxel_to_red_blue(event_voxel, output_size):
    """Render split-polarity voxels like EventVGGT's white/red/blue event PNGs.

    The Night3R voxels are normalized differently across source datasets, so
    occupancy is used instead of magnitude. This preserves polarity and spatial
    support without treating normalized negative values as opposite polarity.
    """
    if event_voxel.ndim != 5:
        raise ValueError(f"Expected event voxel [B,S,C,H,W], got {tuple(event_voxel.shape)}")
    channels = event_voxel.shape[2]
    if channels % 2:
        raise ValueError(f"Expected an even number of split-polarity channels, got {channels}")
    num_bins = channels // 2
    positive = (event_voxel[:, :, :num_bins] != 0).any(dim=2)
    negative = (event_voxel[:, :, num_bins:] != 0).any(dim=2)
    both = positive & negative

    red = (~negative).to(event_voxel.dtype)
    green = (~(positive | negative)).to(event_voxel.dtype)
    blue = (~positive).to(event_voxel.dtype)
    red = torch.where(both, torch.ones_like(red), red)
    blue = torch.where(both, torch.ones_like(blue), blue)
    image = torch.stack((red, green, blue), dim=2)

    b, s, c, h, w = image.shape
    image = F.interpolate(
        image.reshape(b * s, c, h, w),
        size=tuple(output_size),
        mode="bilinear",
        align_corners=False,
    )
    return image.reshape(b, s, c, *output_size).clamp(0, 1)


def prediction_depth(prediction):
    depth = prediction["depth"]
    if depth.ndim != 5 or depth.shape[-1] != 1:
        raise ValueError(f"Expected model depth [B,S,H,W,1], got {tuple(depth.shape)}")
    return depth[:, 0].permute(0, 3, 1, 2).contiguous()


def sample_ids(batch):
    values = batch.get("sample_id", [])
    if isinstance(values, (list, tuple)):
        return [str(value) for value in values]
    return [str(values)] if values else []


def safe_sample_name(sample_id):
    return "".join(char if char.isalnum() or char in "-_" else "_" for char in sample_id)


def save_prediction_batch(args, batch, pred_depth, already_saved):
    if args.prediction_dir is None or already_saved >= args.prediction_count:
        return already_saved
    args.prediction_dir.mkdir(parents=True, exist_ok=True)
    identifiers = sample_ids(batch)
    remaining = args.prediction_count - already_saved
    count = min(pred_depth.shape[0], remaining)
    for item_idx in range(count):
        sample_id = identifiers[item_idx]
        payload = {
            "sample_id": sample_id,
            "variant": args.variant,
            "pred_depth": pred_depth[item_idx].detach().float().cpu(),
            "gt_depth": as_bchw(batch["depth_gt"])[item_idx].detach().float().cpu(),
            "gt_valid": as_bchw(batch["depth_valid"])[item_idx].detach().bool().cpu(),
            "dark_rgb": batch["images_photo_dark"][item_idx, 0].detach().float().cpu(),
            "clean_rgb": batch["images_photo"][item_idx, 0].detach().float().cpu(),
            "event_voxel": batch["event_voxel"][item_idx, 0].detach().float().cpu(),
            "intrinsics": batch["intrinsics"][item_idx].detach().float().cpu(),
        }
        torch.save(payload, args.prediction_dir / f"{safe_sample_name(sample_id)}.pt")
    return already_saved + count


def evaluate(args):
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    dtype = torch.bfloat16 if args.amp_dtype == "bf16" else torch.float16
    cfg = load_config(args.config)
    set_seed(args.seed)
    dataset, loader = build_loader(args, cfg, device)

    checkpoint = Path(args.checkpoint)
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    if args.variant == "vggt":
        model, load_info = load_vggt(args, device)
    else:
        model, load_info = load_eventvggt(args, device)

    event_size = args.event_size
    if event_size is None:
        event_size = (252, 336) if "mvsec" in args.data_root.lower() else (252, 504)
    if event_size[0] % 14 or event_size[1] % 14:
        raise ValueError(f"EventVGGT H/W must be divisible by 14, got {event_size}")

    metrics = DenseMetricAccumulator()
    cutoff_metrics = DepthCutoffAccumulator()
    ids = []
    timings_ms = []
    input_stats = []
    num_batches = 0
    predictions_saved = 0
    set_seed(args.seed)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    with torch.inference_mode():
        for batch_idx, batch in enumerate(loader):
            if args.max_batches is not None and batch_idx >= args.max_batches:
                break
            ids.extend(sample_ids(batch))
            gt_depth = as_bchw(batch["depth_gt"]).to(device, non_blocking=True)
            gt_valid = as_bchw(batch["depth_valid"]).to(device, non_blocking=True).bool()
            if args.depth_min is not None:
                gt_valid &= gt_depth >= float(args.depth_min)
            if args.depth_max is not None:
                gt_valid &= gt_depth <= float(args.depth_max)

            if args.variant == "vggt":
                images = batch["images_photo_dark"].to(device, non_blocking=True)
                model_input = preprocess_vggt(images)
                input_stats.append(float(images.float().mean()))
            else:
                controlled = apply_event_mode(batch, args.event_mode, batch_idx=batch_idx)
                events = controlled["event_voxel"].to(device, non_blocking=True)
                model_input = event_voxel_to_red_blue(events, event_size)
                input_stats.append(float((events != 0).float().mean()))

            if device.type == "cuda":
                torch.cuda.synchronize(device)
            started = time.perf_counter()
            with torch.autocast(device_type=device.type, dtype=dtype, enabled=device.type == "cuda"):
                prediction = model(images=model_input)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            timings_ms.append((time.perf_counter() - started) * 1000.0)

            pred_depth = prediction_depth(prediction)
            metrics.add_depth(pred_depth, gt_depth, gt_valid, variant="raw")
            metrics.add_depth(pred_depth, gt_depth, gt_valid, variant="median_aligned")
            cutoff_metrics.add_depth(pred_depth, gt_depth, gt_valid)
            predictions_saved = save_prediction_batch(
                args,
                batch,
                pred_depth,
                predictions_saved,
            )
            num_batches += 1

    peak_memory = (
        int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else None
    )
    return {
        "variant": args.variant,
        "checkpoint": str(checkpoint.resolve()),
        "config": str(Path(args.config).resolve()),
        "data_root": str(Path(args.data_root).resolve()),
        "dataset_len": len(dataset),
        "num_batches": num_batches,
        "batch_size": args.batch_size,
        "pair_protocol": {
            "sampling": args.pair_sampling,
            "stride": args.pair_stride,
            "min_distance": args.min_pair_distance,
            "max_distance": args.max_pair_distance,
        },
        "seed": args.seed,
        "lowlight_aug_enabled": bool(cfg.get("lowlight_aug", {}).get("enabled", False)),
        "lowlight_aug": dict(cfg.get("lowlight_aug", {})),
        "input": (
            (
                "darkened_rgb"
                if cfg.get("lowlight_aug", {}).get("enabled", False)
                else "clean_rgb"
            )
            if args.variant == "vggt"
            else f"event_only:{args.event_mode}"
        ),
        "input_adapter": (
            {"resize": [518, 518], "range": [0, 1]}
            if args.variant == "vggt"
            else {
                "representation": "binary split-polarity occupancy on white background; positive=red, negative=blue",
                "resize": list(event_size),
                "resize_mode": "bilinear",
                "range": [0, 1],
                "note": "Adapter from Night3R 8-bin voxels to the official EventVGGT 3-channel event-image interface.",
            }
        ),
        "input_stat_mean": float(np.mean(input_stats)) if input_stats else None,
        "amp_dtype": args.amp_dtype,
        "load_info": load_info,
        "dense_metrics": metrics.finalize(),
        "depth_cutoff_metrics": cutoff_metrics.finalize(),
        "prediction_dir": (
            str(args.prediction_dir.resolve()) if args.prediction_dir is not None else None
        ),
        "predictions_saved": predictions_saved,
        "latency_ms_mean": float(np.mean(timings_ms)) if timings_ms else None,
        "latency_ms_median": statistics.median(timings_ms) if timings_ms else None,
        "peak_memory_bytes": peak_memory,
        "sample_ids": ids,
    }


def main():
    args = parse_args()
    result = evaluate(args)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
