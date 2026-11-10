#!/usr/bin/env python3
"""Run Night3R on selected figure pairs and archive depth/GT beside the assets."""

import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from train_event_dark3r import load_backbone, load_config, safe_torch_load  # noqa: E402
from training.event_teacher_student import build_event_student  # noqa: E402
from utils.compare_event_dark3r_checkpoint import depth_from_pred  # noqa: E402


PAIR_SPECS = (
    {
        "prefix": "02_test_DL3DV_temple",
        "source": "dl3dv",
        "root": REPO_ROOT / "test/DL3DV/screen-000095",
        "indices": (32, 48),
    },
    {
        "prefix": "03_train_DSEC_alpine_road",
        "source": "dsec",
        "root": Path("/data/fcr/data/train_data/DSEC/screen-000001-dsec"),
        "indices": (252, 268),
    },
    {
        "prefix": "04_test_DSEC_church_road",
        "source": "dsec",
        "root": REPO_ROOT / "test/DSEC/screen-000002-dsec",
        "indices": (481, 497),
    },
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--asset-dir",
        type=Path,
        default=REPO_ROOT
        / "outputs/paper_assets_selected_rgb_darken_event_low_noise_20260721",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=REPO_ROOT / "config/train_night3r_mvsec_only_adapt_1k_from_best.yaml",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=REPO_ROOT
        / "outputs/night3r_mvsec_only_adapt_1k_from_best/last_event_dark3r.pth",
    )
    parser.add_argument(
        "--backbone",
        type=Path,
        default=REPO_ROOT
        / "checkpoints/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric.pth",
    )
    parser.add_argument("--image-size", type=int, nargs=2, default=(512, 512))
    parser.add_argument("--device", default="cuda:4")
    return parser.parse_args()


def load_image(path, size):
    image = Image.open(path).convert("RGB")
    array = np.asarray(image, dtype=np.float32) / 255.0
    tensor = torch.from_numpy(array).permute(2, 0, 1).contiguous()
    tensor = F.interpolate(
        tensor[None], size=size, mode="bilinear", align_corners=False
    )[0]
    return tensor, (image.height, image.width)


def load_event(path, size):
    try:
        event = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        event = torch.load(path, map_location="cpu")
    if not torch.is_tensor(event) or event.ndim != 3 or event.shape[0] != 8:
        raise ValueError(f"Expected an 8-channel event voxel at {path}, got {type(event)}")
    event = event.float()
    if tuple(event.shape[-2:]) != tuple(size):
        event = F.interpolate(
            event[None], size=size, mode="bilinear", align_corners=False
        )[0]
    return event.contiguous()


def build_model(args, cfg, device):
    model_args = SimpleNamespace(
        use_tiny_model=False,
        model_name_or_path=str(args.backbone),
        backbone_lora_checkpoint=None,
        allow_unsafe_checkpoint_load=True,
    )
    backbone = load_backbone(
        model_args, cfg, model_name_or_path=str(args.backbone)
    )
    model = build_event_student(backbone, config=cfg)
    payload = safe_torch_load(
        args.checkpoint,
        map_location="cpu",
        allow_unsafe=True,
        purpose="selected paper-asset Night3R depth export",
    )
    state_dict = payload.get("model", payload)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing or unexpected:
        raise RuntimeError(
            f"Checkpoint/model mismatch: missing={missing[:10]}, unexpected={unexpected[:10]}"
        )
    model = model.to(device).eval()
    return model, {
        "step": payload.get("step"),
        "stage": payload.get("stage"),
        "missing_keys": [],
        "unexpected_keys": [],
    }


def make_batch(images, events, source, device):
    images = torch.stack(images, dim=0)[None].to(device)
    events = torch.stack(events, dim=0)[None].to(device)
    return {
        "images_dark": images * 2.0 - 1.0,
        "event_voxel": events,
        "dataset_source": source,
    }


def predict_first_view(model, batch, target_hw, device):
    with torch.inference_mode(), torch.amp.autocast(
        device_type=device.type,
        dtype=torch.bfloat16,
        enabled=device.type == "cuda",
    ):
        pred1, _ = model(batch)
    depth = depth_from_pred(pred1)
    if depth is None:
        raise RuntimeError("Night3R output does not contain depth or pts3d")
    depth = F.interpolate(
        depth.float(), size=target_hw, mode="bilinear", align_corners=False
    )[0, 0]
    return depth.detach().cpu().contiguous()


def finite_positive(depth):
    return depth[torch.isfinite(depth) & (depth > 0)]


def visualization_range(depths):
    values = [finite_positive(depth).reshape(-1) for depth in depths]
    values = [value for value in values if value.numel()]
    if not values:
        return 0.0, 1.0
    values = torch.cat(values).float().numpy()
    vmin = float(np.percentile(values, 2.0))
    vmax = float(np.percentile(values, 98.0))
    if not np.isfinite(vmin) or not np.isfinite(vmax) or vmax <= vmin:
        return 0.0, 1.0
    return vmin, vmax


def save_depth_png(depth, path, vmin, vmax):
    array = depth.detach().cpu().float().numpy()
    valid = np.isfinite(array) & (array > 0)
    normalized = np.zeros_like(array, dtype=np.float32)
    normalized[valid] = np.clip(
        (array[valid] - float(vmin)) / max(float(vmax - vmin), 1.0e-8),
        0.0,
        1.0,
    )
    gray = np.round(normalized * 255.0).astype(np.uint8)
    colored_bgr = cv2.applyColorMap(gray, cv2.COLORMAP_TURBO)
    colored_rgb = cv2.cvtColor(colored_bgr, cv2.COLOR_BGR2RGB)
    colored_rgb[~valid] = 0
    Image.fromarray(colored_rgb).save(path)


def depth_metrics(pred, gt):
    valid = (
        torch.isfinite(pred)
        & torch.isfinite(gt)
        & (pred > 0)
        & (gt > 0)
    )
    if not valid.any():
        return None
    pred_values = pred[valid]
    gt_values = gt[valid]
    ratio = torch.maximum(
        pred_values.clamp_min(1.0e-8) / gt_values.clamp_min(1.0e-8),
        gt_values.clamp_min(1.0e-8) / pred_values.clamp_min(1.0e-8),
    )
    scale = gt_values.median() / pred_values.median().clamp_min(1.0e-8)
    aligned = pred_values * scale
    aligned_ratio = torch.maximum(
        aligned.clamp_min(1.0e-8) / gt_values.clamp_min(1.0e-8),
        gt_values.clamp_min(1.0e-8) / aligned.clamp_min(1.0e-8),
    )
    return {
        "valid_pixels": int(valid.sum()),
        "raw_absrel": float(((pred_values - gt_values).abs() / gt_values).mean()),
        "raw_delta1": float((ratio < 1.25).float().mean()),
        "median_scale": float(scale),
        "median_absrel": float(((aligned - gt_values).abs() / gt_values).mean()),
        "median_delta1": float((aligned_ratio < 1.25).float().mean()),
    }


def load_gt(path, target_hw):
    if not path.is_file():
        return None
    try:
        depth = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        depth = torch.load(path, map_location="cpu")
    depth = torch.as_tensor(depth).float().squeeze()
    if tuple(depth.shape) != tuple(target_hw):
        depth = F.interpolate(depth[None, None], size=target_hw, mode="nearest")[0, 0]
    return depth.contiguous()


def process_pair(model, spec, args, device):
    images = []
    events = []
    target_shapes = []
    stems = []
    gts = []
    source_events = []
    source_gts = []

    for view, index in enumerate(spec["indices"], start=1):
        stem = f"{spec['prefix']}_view{view:02d}"
        image, target_hw = load_image(
            args.asset_dir / f"{stem}_DarkenRGB.png", args.image_size
        )
        event_path = spec["root"] / "events" / f"{index:06d}.pt"
        gt_path = spec["root"] / "depth_gt" / f"{index:06d}.pt"
        images.append(image)
        events.append(load_event(event_path, args.image_size))
        target_shapes.append(target_hw)
        stems.append(stem)
        gts.append(load_gt(gt_path, target_hw))
        source_events.append(str(event_path.resolve()))
        source_gts.append(str(gt_path.resolve()) if gt_path.is_file() else None)

    if target_shapes[0] != target_shapes[1]:
        raise ValueError(f"Pair views have different output shapes: {target_shapes}")
    batch = make_batch(images, events, spec["source"], device)
    depth1 = predict_first_view(model, batch, target_shapes[0], device)
    reverse_batch = {
        "images_dark": batch["images_dark"].flip(1),
        "event_voxel": batch["event_voxel"].flip(1),
        "dataset_source": batch["dataset_source"],
    }
    depth2 = predict_first_view(model, reverse_batch, target_shapes[1], device)
    depths = [depth1, depth2]

    range_source = [gt for gt in gts if gt is not None]
    vmin, vmax = visualization_range(range_source if range_source else depths)
    records = []
    for stem, index, pred, gt, event_path, gt_path in zip(
        stems, spec["indices"], depths, gts, source_events, source_gts
    ):
        pred_pt = args.asset_dir / f"{stem}_Depth.pt"
        pred_png = args.asset_dir / f"{stem}_Depth.png"
        torch.save(pred, pred_pt)
        save_depth_png(pred, pred_png, vmin, vmax)
        record = {
            "prefix": stem,
            "source": spec["source"],
            "frame_index": index,
            "source_event": event_path,
            "prediction_pt": str(pred_pt.resolve()),
            "prediction_png": str(pred_png.resolve()),
            "prediction_shape": list(pred.shape),
            "prediction_positive_fraction": float(
                (torch.isfinite(pred) & (pred > 0)).float().mean()
            ),
            "prediction_median": (
                float(finite_positive(pred).median())
                if finite_positive(pred).numel()
                else None
            ),
            "visualization_range": [vmin, vmax],
            "gt_available": gt is not None,
        }
        if gt is not None:
            gt_pt = args.asset_dir / f"{stem}_DepthGT.pt"
            gt_png = args.asset_dir / f"{stem}_DepthGT.png"
            torch.save(gt, gt_pt)
            save_depth_png(gt, gt_png, vmin, vmax)
            record.update(
                {
                    "source_gt": gt_path,
                    "gt_pt": str(gt_pt.resolve()),
                    "gt_png": str(gt_png.resolve()),
                    "gt_shape": list(gt.shape),
                    "metrics": depth_metrics(pred, gt),
                }
            )
        records.append(record)
    return records


def main():
    args = parse_args()
    args.asset_dir = args.asset_dir.resolve()
    if not args.asset_dir.is_dir():
        raise FileNotFoundError(args.asset_dir)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"CUDA is unavailable for requested device {device}")

    cfg = load_config(args.config)
    model, checkpoint_info = build_model(args, cfg, device)
    records = []
    for spec in PAIR_SPECS:
        records.extend(process_pair(model, spec, args, device))

    manifest = {
        "method": "Night3R",
        "checkpoint": str(args.checkpoint.resolve()),
        "backbone": str(args.backbone.resolve()),
        "config": str(args.config.resolve()),
        "checkpoint_info": checkpoint_info,
        "input": (
            "The archived DarkenRGB files are the RGB inputs; source event voxel "
            "tensors are paired by scene/frame index."
        ),
        "view_depth_protocol": (
            "Each view is predicted as the first view of an ordered Night3R pair. "
            "The pair is reversed for view02, so pts3d_in_other_view Z is never "
            "misreported as camera-local view02 depth."
        ),
        "prediction_scale": "raw model output; no median alignment or post-processing",
        "visualization": (
            "Turbo colormap. DSEC uses the joint 2nd-98th percentile GT range per "
            "scene pair; no-GT DL3DV uses the joint prediction range. Invalid GT is black."
        ),
        "records": records,
    }
    manifest_path = args.asset_dir / "depth_archive_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(
        json.dumps(
            {
                "asset_dir": str(args.asset_dir),
                "views": len(records),
                "predictions": len(records),
                "ground_truth": sum(record["gt_available"] for record in records),
                "manifest": str(manifest_path),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
