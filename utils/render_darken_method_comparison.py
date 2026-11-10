#!/usr/bin/env python3
import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F


METHODS = ("vggt", "eventvggt", "night3r")
DISPLAY_NAMES = {"vggt": "VGGT", "eventvggt": "EventVGGT", "night3r": "Night3R"}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prediction-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def load_records(root):
    records = {}
    for method in METHODS:
        method_records = {}
        for path in sorted((root / method).glob("*.pt")):
            payload = torch.load(path, map_location="cpu", weights_only=True)
            method_records[payload["sample_id"]] = payload
        records[method] = method_records
    common = set.intersection(*(set(records[method]) for method in METHODS))
    if not common:
        raise RuntimeError(f"No common prediction samples under {root}")
    return records, sorted(common)


def as_hw(tensor):
    tensor = tensor.float()
    while tensor.ndim > 2 and tensor.shape[0] == 1:
        tensor = tensor[0]
    if tensor.ndim != 2:
        raise ValueError(f"Expected a depth map, got {tuple(tensor.shape)}")
    return tensor


def resize_hw(tensor, size, mode="bilinear"):
    tensor = tensor.float()
    if tensor.ndim == 2:
        tensor = tensor[None, None]
    elif tensor.ndim == 3:
        tensor = tensor[None]
    kwargs = {"size": size, "mode": mode}
    if mode in {"bilinear", "bicubic"}:
        kwargs["align_corners"] = False
    return F.interpolate(tensor, **kwargs)[0]


def event_rgb(event_voxel, size):
    event = event_voxel.float()
    half = event.shape[0] // 2
    positive = (event[:half] != 0).any(dim=0)
    negative = (event[half:] != 0).any(dim=0)
    both = positive & negative
    red = (~negative).float()
    green = (~(positive | negative)).float()
    blue = (~positive).float()
    red[both] = 1.0
    blue[both] = 1.0
    return resize_hw(torch.stack((red, green, blue)), size).permute(1, 2, 0).numpy()


def aligned_prediction(payload):
    gt = as_hw(payload["gt_depth"])
    valid = as_hw(payload["gt_valid"].float()).bool() & torch.isfinite(gt) & (gt > 0)
    pred = as_hw(resize_hw(payload["pred_depth"], gt.shape[-2:]))
    valid &= torch.isfinite(pred) & (pred > 0)
    scale = gt[valid].median() / pred[valid].median().clamp_min(1e-6)
    pred = pred * scale
    absrel = float(((pred[valid] - gt[valid]).abs() / gt[valid]).mean())
    error = (pred - gt).abs()
    return gt, valid, pred, error, float(scale), absrel


def render_depth_figure(output, records, sample_ids):
    fig, axes = plt.subplots(len(sample_ids), 6, figsize=(20, 4.25 * len(sample_ids)), squeeze=False)
    manifest = []
    for row_idx, sample_id in enumerate(sample_ids):
        reference = records["night3r"][sample_id]
        gt = as_hw(reference["gt_depth"])
        valid = as_hw(reference["gt_valid"].float()).bool() & torch.isfinite(gt) & (gt > 0)
        dark = resize_hw(reference["dark_rgb"], gt.shape[-2:]).permute(1, 2, 0).clamp(0, 1).numpy()
        event = event_rgb(reference["event_voxel"], gt.shape[-2:])
        valid_values = gt[valid].numpy()
        vmin, vmax = np.percentile(valid_values, [2, 98])

        axes[row_idx, 0].imshow(dark)
        axes[row_idx, 1].imshow(event)
        gt_display = gt.numpy().copy()
        gt_display[~valid.numpy()] = np.nan
        axes[row_idx, 2].imshow(gt_display, cmap="turbo", vmin=vmin, vmax=vmax)
        axes[row_idx, 0].set_ylabel(sample_id, fontsize=8)

        row_manifest = {"sample_id": sample_id, "methods": {}}
        for col_idx, method in enumerate(METHODS, start=3):
            _, method_valid, pred, _, scale, absrel = aligned_prediction(records[method][sample_id])
            display = pred.numpy().copy()
            display[~method_valid.numpy()] = np.nan
            axes[row_idx, col_idx].imshow(display, cmap="turbo", vmin=vmin, vmax=vmax)
            axes[row_idx, col_idx].set_title(f"{DISPLAY_NAMES[method]}\nAbsRel {absrel:.3f}")
            row_manifest["methods"][method] = {"median_scale": scale, "absrel": absrel}
        manifest.append(row_manifest)

    for col_idx, title in enumerate(("Dark RGB (e=0.05)", "Event", "GT depth", "VGGT", "EventVGGT", "Night3R")):
        if col_idx < 3:
            axes[0, col_idx].set_title(title)
    for axis in axes.flat:
        axis.set_xticks([])
        axis.set_yticks([])
    fig.suptitle("Deterministic low-light depth comparison", fontsize=15)
    fig.tight_layout(rect=(0, 0, 1, 0.97), h_pad=3.2, w_pad=0.8)
    fig.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return manifest


def render_error_figure(output, records, sample_ids):
    fig, axes = plt.subplots(len(sample_ids), 3, figsize=(12, 4.25 * len(sample_ids)), squeeze=False)
    for row_idx, sample_id in enumerate(sample_ids):
        errors = []
        prepared = []
        for method in METHODS:
            _, valid, _, error, _, absrel = aligned_prediction(records[method][sample_id])
            errors.append(error[valid].numpy())
            prepared.append((method, valid, error, absrel))
        vmax = np.percentile(np.concatenate(errors), 95)
        for col_idx, (method, valid, error, absrel) in enumerate(prepared):
            display = error.numpy().copy()
            display[~valid.numpy()] = np.nan
            axes[row_idx, col_idx].imshow(display, cmap="magma", vmin=0, vmax=vmax)
            axes[row_idx, col_idx].set_title(f"{DISPLAY_NAMES[method]} error\nAbsRel {absrel:.3f}")
            axes[row_idx, col_idx].set_xticks([])
            axes[row_idx, col_idx].set_yticks([])
            if col_idx == 0:
                axes[row_idx, col_idx].set_ylabel(sample_id, fontsize=8)
    fig.suptitle("Median-aligned absolute depth error", fontsize=15)
    fig.tight_layout(rect=(0, 0, 1, 0.97), h_pad=3.2, w_pad=0.8)
    fig.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main():
    args = parse_args()
    records, sample_ids = load_records(args.prediction_root)
    sample_ids = sample_ids[:3]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest = render_depth_figure(
        args.output_dir / "depth_comparison_e005.png",
        records,
        sample_ids,
    )
    render_error_figure(
        args.output_dir / "absolute_error_e005.png",
        records,
        sample_ids,
    )
    (args.output_dir / "manifest.json").write_text(
        json.dumps({"samples": manifest}, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"output_dir": str(args.output_dir), "sample_ids": sample_ids}, indent=2))


if __name__ == "__main__":
    main()
