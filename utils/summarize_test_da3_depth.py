#!/usr/bin/env python3
"""Validate and summarize DA3 depth generated for Night3R test datasets."""

import argparse
import json
import os
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFont, ImageOps


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--allow-incomplete", action="store_true")
    return parser.parse_args()


def read_json(path):
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def atomic_write_text(path, text):
    temporary = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    with temporary.open("w", encoding="utf-8") as handle:
        handle.write(text)
    os.replace(temporary, path)


def load_tensor(path):
    try:
        value = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        value = torch.load(path, map_location="cpu")
    return torch.as_tensor(value).float().squeeze()


def align_gt_shape(gt, shape):
    if tuple(gt.shape) == tuple(shape):
        return gt
    return F.interpolate(
        gt[None, None], size=tuple(shape), mode="nearest"
    )[0, 0]


def depth_metrics(prediction, target):
    target = align_gt_shape(target, prediction.shape)
    valid = (
        torch.isfinite(prediction)
        & torch.isfinite(target)
        & (prediction > 0)
        & (target > 0)
    )
    if int(valid.sum()) < 64:
        return None
    pred = prediction[valid]
    gt = target[valid]
    ratio = torch.median(gt) / torch.median(pred).clamp_min(1e-8)
    aligned = pred * ratio

    def metrics_for(values):
        absrel = torch.mean(torch.abs(values - gt) / gt.clamp_min(1e-8))
        threshold = torch.maximum(
            values / gt.clamp_min(1e-8), gt / values.clamp_min(1e-8)
        )
        delta1 = torch.mean((threshold < 1.25).float())
        return float(absrel), float(delta1)

    raw_absrel, raw_delta1 = metrics_for(pred)
    median_absrel, median_delta1 = metrics_for(aligned)
    return {
        "raw_absrel": raw_absrel,
        "raw_delta1": raw_delta1,
        "median_absrel": median_absrel,
        "median_delta1": median_delta1,
        "median_scale": float(ratio),
        "valid_px": int(valid.sum()),
    }


def make_contact_sheet(root, complete):
    chosen = complete[:: max(1, len(complete) // 12)][:12]
    if not chosen:
        return None
    tile_size = (630, 371)
    label_height = 24
    columns = 2
    rows = (len(chosen) + columns - 1) // columns
    sheet = Image.new(
        "RGB",
        (
            columns * tile_size[0],
            rows * (tile_size[1] + label_height),
        ),
        "white",
    )
    draw = ImageDraw.Draw(sheet)
    font = ImageFont.load_default()
    for index, item in enumerate(chosen):
        row, column = divmod(index, columns)
        x = column * tile_size[0]
        y = row * (tile_size[1] + label_height)
        overview = root / item["dataset"] / item["scene"] / "da3_depth_overview.jpg"
        image = Image.open(overview).convert("RGB")
        image = ImageOps.fit(image, tile_size, method=Image.Resampling.LANCZOS)
        sheet.paste(image, (x, y + label_height))
        draw.text(
            (x + 6, y + 6),
            f"{item['dataset']}/{item['scene']}",
            fill="black",
            font=font,
        )
    output = root / "da3_depth_contact_sheet.jpg"
    sheet.save(output, quality=94)
    return output


def main():
    args = parse_args()
    args.root = args.root.resolve()
    selection = read_json(args.root / "selection_manifest.json")
    complete = []
    incomplete = []
    dataset_metrics = defaultdict(list)

    for record in selection["records"]:
        output_scene = args.root / record["dataset"] / record["scene"]
        source_scene = Path(record["source_scene_dir"])
        image_paths = sorted((source_scene / "images").glob("*.png"))
        manifest_path = output_scene / "da3_depth_manifest.json"
        if not manifest_path.is_file():
            incomplete.append(
                {
                    "dataset": record["dataset"],
                    "scene": record["scene"],
                    "reason": "missing manifest",
                }
            )
            continue
        manifest = read_json(manifest_path)
        depth_paths = sorted((output_scene / "depth_da3").glob("*.pt"))
        confidence_paths = sorted(
            (output_scene / "depth_da3_conf").glob("*.pt")
        )
        if (
            manifest.get("status") != "complete"
            or len(depth_paths) != len(image_paths)
            or len(confidence_paths) != len(image_paths)
        ):
            incomplete.append(
                {
                    "dataset": record["dataset"],
                    "scene": record["scene"],
                    "reason": "incomplete files",
                    "depth_count": len(depth_paths),
                    "confidence_count": len(confidence_paths),
                    "expected": len(image_paths),
                }
            )
            continue

        gt_dir = source_scene / "depth_gt"
        frame_metrics = []
        if gt_dir.is_dir():
            for depth_path in depth_paths:
                gt_path = gt_dir / depth_path.name
                if not gt_path.is_file():
                    continue
                metrics = depth_metrics(
                    load_tensor(depth_path), load_tensor(gt_path)
                )
                if metrics is not None:
                    frame_metrics.append(metrics)
                    dataset_metrics[record["dataset"]].append(metrics)
        manifest_summary = {
            **manifest,
            "evaluated_gt_frames": len(frame_metrics),
        }
        if frame_metrics:
            for key in (
                "raw_absrel",
                "raw_delta1",
                "median_absrel",
                "median_delta1",
                "median_scale",
            ):
                manifest_summary[f"gt_{key}_mean"] = float(
                    np.mean([item[key] for item in frame_metrics])
                )
        complete.append(manifest_summary)

    metrics_summary = {}
    for dataset, records in sorted(dataset_metrics.items()):
        metrics_summary[dataset] = {
            "frame_count": len(records),
            **{
                f"{key}_mean": float(np.mean([item[key] for item in records]))
                for key in (
                    "raw_absrel",
                    "raw_delta1",
                    "median_absrel",
                    "median_delta1",
                    "median_scale",
                )
            },
        }

    overlap_scales = [
        float(scale)
        for item in complete
        for scale in item.get("window_overlap_scales", [])
    ]
    summary = {
        "status": "complete" if not incomplete else "incomplete",
        "label_type": "DA3 prediction; not sensor/dataset ground truth",
        "requested_scene_count": selection["scene_count"],
        "requested_frame_count": selection["frame_count"],
        "complete_scene_count": len(complete),
        "complete_frame_count": sum(item["frame_count"] for item in complete),
        "datasets": {
            dataset: {
                "scene_count": sum(
                    item["dataset"] == dataset for item in complete
                ),
                "frame_count": sum(
                    item["frame_count"]
                    for item in complete
                    if item["dataset"] == dataset
                ),
            }
            for dataset in selection["datasets"]
        },
        "gt_comparison": metrics_summary,
        "window_overlap_scale": {
            "min": min(overlap_scales) if overlap_scales else None,
            "max": max(overlap_scales) if overlap_scales else None,
            "outside_0.5_to_2.0": sum(
                not 0.5 <= scale <= 2.0 for scale in overlap_scales
            ),
            "count": len(overlap_scales),
        },
        "incomplete": incomplete,
        "generation_elapsed_scene_seconds": float(
            sum(item["elapsed_seconds"] for item in complete)
        ),
    }
    atomic_write_text(
        args.root / "generation_summary.json",
        json.dumps(summary, indent=2) + "\n",
    )
    make_contact_sheet(args.root, complete)
    semantics = {
        "DL3DV": (
            "- DL3DV: pose-conditioned and aligned to the supplied scene "
            "trajectory scale; absolute metric scale is not guaranteed."
        ),
        "MVSEC": (
            "- MVSEC: pose-conditioned using `T_world_body` as a camera-pose proxy. "
            "The unavailable camera-body transform is a documented approximation."
        ),
        "DSEC": (
            "- DSEC: no pose conditioning is available; interpret the output as relative depth."
        ),
    }
    semantics_text = "\n".join(
        semantics[dataset] for dataset in selection["datasets"]
    )
    readme = f"""# Night3R test-set DA3 depth

This directory contains DA3-LARGE-1.1 predictions for the Night3R test sets.
They are external-model predictions and are not sensor or dataset ground truth.

- Complete scenes: `{summary['complete_scene_count']}/{summary['requested_scene_count']}`
- Complete frames: `{summary['complete_frame_count']}/{summary['requested_frame_count']}`
- Saved shape: `{selection['target_size_hw'][0]} x {selection['target_size_hw'][1]}`
- Window size / overlap: `{selection['window_size']} / {selection['window_overlap']}`
- Model license: `Apache-2.0`

## Depth semantics

{semantics_text}

Depth is stored under `DATASET/SCENE/depth_da3/`; raw DA3 confidence is stored
under `DATASET/SCENE/depth_da3_conf/`. File stems exactly match the source RGB
frames. Per-scene details are in `da3_depth_manifest.json`.

Aggregate validation and GT comparisons are recorded in
`generation_summary.json`. The contact sheet is
`da3_depth_contact_sheet.jpg`.

DL3DV scene trajectories do not guarantee a common absolute metric scale.
Downstream evaluation or distillation should therefore use per-pair or
per-scene robust scale alignment.
"""
    atomic_write_text(args.root / "README.md", readme)
    print(json.dumps(summary, indent=2))
    if incomplete and not args.allow_incomplete:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
