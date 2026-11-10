#!/usr/bin/env python3
"""Evaluate center-frame reconstruction stability under changing temporal context.

For every selected center frame t, the evaluator constructs two ordered pairs:
(t, t-1) and (t, t+1). The center frame is therefore view 1 in both pairs,
so both predicted center-frame point maps use the same camera coordinate frame.
This avoids using pose estimates when measuring temporal depth consistency.

Sensor depth is used for MVSEC/DSEC when available. DL3DV pseudo depth is
reported only as teacher agreement. Stability without a depth reference is a
diagnostic and cannot establish geometric accuracy by itself.
"""

import argparse
import hashlib
import json
import math
import os
import random
import re
import sys
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFont
from torch.utils.data._utils.collate import default_collate

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from dataset.dsec_event_dataset import DSECEventDataset  # noqa: E402
from model.event_dark3r_student import make_view_pair_from_batch  # noqa: E402
from train_event_dark3r import (  # noqa: E402
    load_config,
    move_batch_to_device,
    resolve_amp_config,
)
from utils.compare_event_dark3r_checkpoint import as_bchw, depth_from_pred  # noqa: E402
from utils.eval_event_dark3r_variant import build_model  # noqa: E402
from utils.export_rgb_lowlight_event_assets import event_red_blue  # noqa: E402


METHODS = ("mast3r", "night3r", "night3r_zero")


def parse_args():
    parser = argparse.ArgumentParser(
        "Evaluate Night3R dynamic center-frame reconstruction stability"
    )
    parser.add_argument(
        "--dataset-name",
        required=True,
        choices=("MVSEC", "DSEC", "DL3DV", "DL3DV_CHUNK016"),
    )
    parser.add_argument("--data-root", required=True, type=Path)
    parser.add_argument("--scenes", nargs="+", default=None)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--model-name-or-path", required=True, type=Path)
    parser.add_argument("--outdir", required=True, type=Path)
    parser.add_argument("--image-size", nargs=2, type=int, default=(512, 512))
    parser.add_argument("--max-centers-per-scene", type=int, default=12)
    parser.add_argument("--center-stride", type=int, default=1)
    parser.add_argument(
        "--synthetic-exposure",
        type=float,
        default=None,
        help="Fixed exposure in approximate linear RGB; omitted for native low-light input.",
    )
    parser.add_argument("--read-noise-std", type=float, default=0.01)
    parser.add_argument("--shot-noise-std", type=float, default=0.01)
    parser.add_argument("--gamma", type=float, default=2.2)
    parser.add_argument("--lowlight-seed", type=int, default=20260726)
    parser.add_argument("--visuals-per-scene", type=int, default=2)
    parser.add_argument("--amp-dtype", choices=("fp16", "bf16"), default="bf16")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="cuda")
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--progress-every", type=int, default=10)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--allow-unsafe-checkpoint-load", action="store_true")
    args = parser.parse_args()
    if args.max_centers_per_scene <= 0:
        parser.error("--max-centers-per-scene must be positive")
    if args.center_stride <= 0:
        parser.error("--center-stride must be positive")
    if args.synthetic_exposure is not None and not 0 < args.synthetic_exposure <= 1:
        parser.error("--synthetic-exposure must be in (0, 1]")
    if args.read_noise_std < 0 or args.shot_noise_std < 0:
        parser.error("noise standard deviations must be non-negative")
    if args.gamma <= 0:
        parser.error("--gamma must be positive")
    return args


def configure_cpu_threads():
    threads = int(os.environ.get("NIGHT3R_CPU_THREADS", "4"))
    torch.set_num_threads(max(1, threads))
    torch.set_num_interop_threads(max(1, min(threads, 4)))


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(name):
    if name == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but unavailable")
        return torch.device("cuda")
    if name == "cpu":
        return torch.device("cpu")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def make_train_args(args):
    return SimpleNamespace(
        config=str(args.config),
        data_root=str(args.data_root),
        save_dir=str(args.outdir),
        model_name_or_path=str(args.model_name_or_path),
        dark3r_lora_checkpoint=None,
        allow_unsafe_checkpoint_load=args.allow_unsafe_checkpoint_load,
        use_tiny_model=False,
        image_size=list(args.image_size),
        batch_size=2,
        max_batches=None,
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
        checkpoint=str(args.checkpoint) if variant == "event_dark3r" else None,
        dark3r_lora_checkpoint=None,
        model_name_or_path=str(args.model_name_or_path),
        allow_unsafe_checkpoint_load=args.allow_unsafe_checkpoint_load,
        restore_frozen_rgb_scope="none",
        restore_frozen_rgb=False,
        restore_frozen_rgb_alpha=1.0,
        geometry_head_checkpoint=None,
    )


def build_dataset(args, cfg):
    return DSECEventDataset(
        root_path=str(args.data_root),
        image_size=tuple(args.image_size),
        pair_stride=1,
        pair_sampling="fixed",
        expected_event_channels=int(cfg.get("event", {}).get("voxel_channels", 8)),
        lowlight_enabled=False,
        pose_output_convention="opencv",
    )


def pair_index(dataset):
    return {
        (str(sample["scene"]), int(sample["idx1"]), int(sample["idx2"])): index
        for index, sample in enumerate(dataset.samples)
    }


def swap_sample(sample):
    """Swap an unbatched pair and keep view-specific GT fields consistent."""
    result = dict(sample)
    for key in (
        "images",
        "images_dark",
        "images_photo",
        "images_photo_dark",
        "event_voxel",
    ):
        if key in result and torch.is_tensor(result[key]):
            result[key] = result[key].flip(0)
    for left, right in (
        ("image1_path", "image2_path"),
        ("event1_path", "event2_path"),
        ("depth1_path", "depth2_path"),
        ("depth_pseudo1_path", "depth_pseudo2_path"),
        ("depth_pseudo_conf1_path", "depth_pseudo_conf2_path"),
    ):
        if left in result and right in result:
            result[left], result[right] = result[right], result[left]
    result["idx1"], result["idx2"] = int(sample["idx2"]), int(sample["idx1"])
    result["sample_id"] = (
        f"{sample['scene']}:{result['idx1']:06d}-{result['idx2']:06d}"
    )
    for prefix in ("depth_gt", "depth_pseudo", "depth_pseudo_conf"):
        key1, key2 = f"{prefix}1", f"{prefix}2"
        if key1 in result and key2 in result:
            result[key1], result[key2] = result[key2], result[key1]
            if prefix in result:
                result[prefix] = result[key1]
    if "depth_gt1" in result:
        result["depth_gt"] = result["depth_gt1"]
        result["depth_valid"] = (
            torch.isfinite(result["depth_gt1"]) & (result["depth_gt1"] > 0)
        )
    if "depth_pseudo1" in result:
        result["depth_pseudo"] = result["depth_pseudo1"]
        result["depth_pseudo_valid"] = (
            torch.isfinite(result["depth_pseudo1"])
            & (result["depth_pseudo1"] > 0)
        )
    if "depth_pseudo_conf1" in result:
        result["depth_pseudo_conf"] = result["depth_pseudo_conf1"]
    for pose_key in ("relative_pose", "relative_pose_gt"):
        if pose_key in result:
            result[pose_key] = torch.linalg.inv(result[pose_key])
    return result


def keyed_seed(*values, base_seed):
    text = "|".join(str(value) for value in values)
    digest = hashlib.blake2b(text.encode("utf-8"), digest_size=8).digest()
    return (int.from_bytes(digest, "little") + int(base_seed)) % (2**63 - 1)


def darken_frame(image, exposure, read_noise, shot_noise, gamma, seed):
    if exposure is None:
        return image.clone()
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed))
    image = image.detach().float().cpu().clamp(0, 1)
    linear = image.pow(float(gamma))
    dark = linear * float(exposure)
    if shot_noise > 0:
        std = torch.sqrt(dark.clamp_min(0) + 1e-6) * float(shot_noise)
        dark = dark + torch.randn(
            dark.shape, generator=generator, dtype=dark.dtype
        ) * std
    if read_noise > 0:
        dark = dark + torch.randn(
            dark.shape, generator=generator, dtype=dark.dtype
        ) * float(read_noise)
    return dark.clamp(0, 1).pow(1.0 / float(gamma))


def apply_deterministic_lowlight(sample, args):
    clean = sample["images_photo"]
    dark = []
    for view, frame_index in enumerate((sample["idx1"], sample["idx2"])):
        seed = keyed_seed(
            args.dataset_name,
            sample["scene"],
            int(frame_index),
            base_seed=args.lowlight_seed,
        )
        dark.append(
            darken_frame(
                clean[view],
                args.synthetic_exposure,
                args.read_noise_std,
                args.shot_noise_std,
                args.gamma,
                seed,
            )
        )
    sample = dict(sample)
    sample["images_photo_dark"] = torch.stack(dark, dim=0).to(clean.dtype)
    sample["images_dark"] = sample["images_photo_dark"] * 2.0 - 1.0
    return sample


def selected_centers(num_frames, max_centers, stride):
    values = np.arange(1, num_frames - 1, stride, dtype=np.int64)
    if values.size <= max_centers:
        return [int(value) for value in values]
    positions = np.linspace(0, values.size - 1, max_centers)
    return sorted({int(values[int(round(position))]) for position in positions})


def resize_depth(depth, target_hw):
    value = as_bchw(depth).float()
    if value.shape[-2:] != tuple(target_hw):
        value = F.interpolate(
            value, size=tuple(target_hw), mode="bilinear", align_corners=False
        )
    return value


def valid_values(left, right, extra_valid=None):
    valid = (
        torch.isfinite(left)
        & torch.isfinite(right)
        & (left > 0)
        & (right > 0)
    )
    if extra_valid is not None:
        valid &= extra_valid.bool()
    return valid


def temporal_metrics(left, right, reference_valid=None):
    left = as_bchw(left).float()
    right = resize_depth(right, left.shape[-2:])
    if reference_valid is not None:
        reference_valid = as_bchw(reference_valid).bool()
        if reference_valid.shape[-2:] != left.shape[-2:]:
            reference_valid = F.interpolate(
                reference_valid.float(),
                size=left.shape[-2:],
                mode="nearest",
            ).bool()
    valid = valid_values(left, right, reference_valid)
    if int(valid.sum()) < 32:
        return None
    a = left[valid].clamp_min(1e-6)
    b = right[valid].clamp_min(1e-6)
    log_difference = (torch.log(a) - torch.log(b)).abs()
    ratio = torch.maximum(a / b, b / a)
    scale = torch.median(a) / torch.median(b).clamp_min(1e-6)
    aligned = b * scale
    aligned_log_difference = (torch.log(a) - torch.log(aligned)).abs()
    aligned_ratio = torch.maximum(a / aligned, aligned / a)
    symmetric_absrel = (2.0 * (a - b).abs() / (a + b).clamp_min(1e-6)).mean()
    return {
        "raw_mean_abs_log": float(log_difference.mean()),
        "raw_p90_abs_log": float(torch.quantile(log_difference, 0.90)),
        "raw_delta1_agreement": float((ratio < 1.25).float().mean()),
        "raw_symmetric_absrel": float(symmetric_absrel),
        "median_scale_right_to_left": float(scale),
        "scale_drift_abs_log": float(torch.log(scale.clamp_min(1e-6)).abs()),
        "median_mean_abs_log": float(aligned_log_difference.mean()),
        "median_p90_abs_log": float(torch.quantile(aligned_log_difference, 0.90)),
        "median_delta1_agreement": float(
            (aligned_ratio < 1.25).float().mean()
        ),
        "valid_px": int(valid.sum()),
    }


def accuracy_metrics(prediction, reference, reference_valid):
    reference = as_bchw(reference).float()
    prediction = resize_depth(prediction, reference.shape[-2:])
    reference_valid = as_bchw(reference_valid).bool()
    valid = valid_values(prediction, reference, reference_valid)
    if int(valid.sum()) < 32:
        return None
    pred = prediction[valid].clamp_min(1e-6)
    target = reference[valid].clamp_min(1e-6)
    ratio = torch.maximum(pred / target, target / pred)
    scale = torch.median(target) / torch.median(pred).clamp_min(1e-6)
    aligned = pred * scale
    aligned_ratio = torch.maximum(aligned / target, target / aligned)
    return {
        "raw_absrel": float(((pred - target).abs() / target).mean()),
        "raw_delta1": float((ratio < 1.25).float().mean()),
        "median_scale": float(scale),
        "median_absrel": float(((aligned - target).abs() / target).mean()),
        "median_delta1": float((aligned_ratio < 1.25).float().mean()),
        "valid_px": int(valid.sum()),
    }


def mean_numeric_dict(items):
    items = [item for item in items if item]
    if not items:
        return None
    result = {}
    for key in sorted(set.intersection(*(set(item) for item in items))):
        values = [item[key] for item in items]
        if key == "valid_px":
            result[key] = int(sum(values))
        elif all(isinstance(value, (int, float)) for value in values):
            result[key] = float(np.mean(values))
    return result


def rotation_angle_deg(relative_pose):
    rotation = relative_pose[:3, :3].detach().float().cpu()
    trace = torch.trace(rotation)
    cosine = ((trace - 1.0) * 0.5).clamp(-1.0, 1.0)
    return float(torch.rad2deg(torch.acos(cosine)))


def input_diagnostics(batch):
    event = batch["event_voxel"][:, 0].detach().cpu().float()
    activity = event.abs().sum(dim=1)
    image = batch["images_photo_dark"][:, 0].detach().cpu().float()
    result = {
        "event_activity_mean": float(activity.mean()),
        "event_occupancy": float((activity > 0).float().mean()),
        "dark_rgb_mean": float(image.mean()),
        "dark_rgb_std": float(image.std()),
    }
    if "relative_pose_gt" in batch:
        poses = batch["relative_pose_gt"].detach().cpu().float()
        translations = torch.linalg.vector_norm(poses[:, :3, 3], dim=-1)
        rotations = [rotation_angle_deg(pose) for pose in poses]
        result.update(
            {
                "context_translation_mean": float(translations.mean()),
                "context_rotation_deg_mean": float(np.mean(rotations)),
            }
        )
    return result


def reference_from_batch(batch):
    if "depth_gt1" in batch:
        reference = batch["depth_gt1"][0:1]
        valid = torch.isfinite(reference) & (reference > 0)
        return reference, valid, "sensor_depth_gt"
    if "depth_pseudo1" in batch:
        reference = batch["depth_pseudo1"][0:1]
        valid = torch.isfinite(reference) & (reference > 0)
        if "depth_pseudo_conf1" in batch:
            confidence = batch["depth_pseudo_conf1"][0:1]
            finite = torch.isfinite(confidence)
            if int(finite.sum()) >= 32:
                threshold = torch.quantile(confidence[finite], 0.25)
                valid &= confidence >= threshold
        return reference, valid, "da3_pseudo_depth"
    return None, None, "none"


def infer_center_contexts(master, night, batch_cpu, train_args, cfg, device):
    batch = move_batch_to_device(batch_cpu, device)
    amp_enabled, _, amp_dtype = resolve_amp_config(cfg, train_args, device)
    with torch.no_grad(), torch.amp.autocast(
        device_type=device.type,
        dtype=amp_dtype,
        enabled=bool(amp_enabled and device.type == "cuda"),
    ):
        view1, view2 = make_view_pair_from_batch(batch, prefer_dark=True)
        master_pred, _ = master(view1, view2)
        night_pred, _ = night(batch)
        zero_batch = dict(batch)
        zero_batch["event_voxel"] = torch.zeros_like(batch["event_voxel"])
        zero_pred, _ = night(zero_batch)
    outputs = {
        "mast3r": depth_from_pred(master_pred).detach().float().cpu(),
        "night3r": depth_from_pred(night_pred).detach().float().cpu(),
        "night3r_zero": depth_from_pred(zero_pred).detach().float().cpu(),
    }
    del (
        batch,
        view1,
        view2,
        master_pred,
        night_pred,
        zero_batch,
        zero_pred,
    )
    return outputs


def split_label(dataset_name, scene):
    if dataset_name == "DSEC":
        return "excluded_overlap_diagnostic" if scene == "screen-000002-dsec" else "strict_heldout"
    if dataset_name == "MVSEC":
        return "adaptation_domain_diagnostic"
    if dataset_name == "DL3DV_CHUNK016":
        return "training_source_diagnostic"
    return "strict_heldout"


def record_for_center(args, scene, center, batch, outputs):
    reference, reference_valid, reference_kind = reference_from_batch(batch)
    record = {
        "protocol_id": (
            f"{args.dataset_name}|{args.data_root.resolve()}|{scene}|{center:06d}"
        ),
        "dataset": args.dataset_name,
        "data_root": str(args.data_root.resolve()),
        "scene": scene,
        "center_frame": int(center),
        "left_context_frame": int(center - 1),
        "right_context_frame": int(center + 1),
        "split_label": split_label(args.dataset_name, scene),
        "reference_kind": reference_kind,
        "synthetic_exposure": args.synthetic_exposure,
        **input_diagnostics(batch),
        "methods": {},
    }
    for method, depth in outputs.items():
        temporal = temporal_metrics(depth[0:1], depth[1:2], reference_valid)
        accuracy = None
        if reference is not None:
            left_accuracy = accuracy_metrics(depth[0:1], reference, reference_valid)
            right_accuracy = accuracy_metrics(depth[1:2], reference, reference_valid)
            accuracy = {
                "left_context": left_accuracy,
                "right_context": right_accuracy,
                "mean": mean_numeric_dict([left_accuracy, right_accuracy]),
            }
        record["methods"][method] = {
            "temporal": temporal,
            "accuracy": accuracy,
        }
    master_temporal = record["methods"]["mast3r"]["temporal"]
    night_temporal = record["methods"]["night3r"]["temporal"]
    zero_temporal = record["methods"]["night3r_zero"]["temporal"]
    record["night3r_vs_mast3r_temporal_gain"] = (
        None
        if not master_temporal or not night_temporal
        else float(
            master_temporal["median_mean_abs_log"]
            - night_temporal["median_mean_abs_log"]
        )
    )
    record["real_vs_zero_temporal_gain"] = (
        None
        if not zero_temporal or not night_temporal
        else float(
            zero_temporal["median_mean_abs_log"]
            - night_temporal["median_mean_abs_log"]
        )
    )
    if reference is not None:
        master_accuracy = record["methods"]["mast3r"]["accuracy"]["mean"]
        night_accuracy = record["methods"]["night3r"]["accuracy"]["mean"]
        zero_accuracy = record["methods"]["night3r_zero"]["accuracy"]["mean"]
        record["night3r_vs_mast3r_accuracy_gain"] = float(
            master_accuracy["median_absrel"] - night_accuracy["median_absrel"]
        )
        record["real_vs_zero_accuracy_gain"] = float(
            zero_accuracy["median_absrel"] - night_accuracy["median_absrel"]
        )
    else:
        record["night3r_vs_mast3r_accuracy_gain"] = None
        record["real_vs_zero_accuracy_gain"] = None
    return record


def sanitize(value):
    if isinstance(value, dict):
        return {key: sanitize(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [sanitize(item) for item in value]
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, (np.floating, float)):
        value = float(value)
        return value if math.isfinite(value) else None
    return value


def safe_name(value):
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value)).strip("_")


def tensor_rgb(value):
    value = value.detach().cpu().float().clamp(0, 1)
    return (value.permute(1, 2, 0).numpy() * 255.0 + 0.5).astype(np.uint8)


def depth_limits(arrays, valid=None):
    values = []
    for array in arrays:
        mask = np.isfinite(array) & (array > 0)
        if valid is not None:
            mask &= valid
        if mask.any():
            values.append(array[mask])
    if not values:
        return 0.0, 1.0
    joined = np.concatenate(values)
    low, high = np.percentile(joined, (2, 98))
    return float(low), float(max(high, low + 1e-6))


def color_depth(array, low, high, valid=None):
    normalized = np.clip((array - low) / max(high - low, 1e-6), 0, 1)
    normalized = np.nan_to_num(normalized, nan=0.0, posinf=1.0, neginf=0.0)
    color = cv2.applyColorMap(
        (normalized * 255).astype(np.uint8), cv2.COLORMAP_VIRIDIS
    )
    color = cv2.cvtColor(color, cv2.COLOR_BGR2RGB)
    mask = np.isfinite(array) & (array > 0)
    if valid is not None:
        mask &= valid
    color[~mask] = 0
    return color


def color_difference(array, valid=None):
    values = array[np.isfinite(array)]
    high = float(np.percentile(values, 98)) if values.size else 1.0
    normalized = np.clip(array / max(high, 1e-6), 0, 1)
    normalized = np.nan_to_num(normalized, nan=0.0, posinf=1.0, neginf=0.0)
    color = cv2.applyColorMap(
        (normalized * 255).astype(np.uint8), cv2.COLORMAP_MAGMA
    )
    color = cv2.cvtColor(color, cv2.COLOR_BGR2RGB)
    if valid is not None:
        color[~valid] = 0
    return color


def align_array(source, target, valid):
    mask = (
        valid
        & np.isfinite(source)
        & np.isfinite(target)
        & (source > 0)
        & (target > 0)
    )
    if int(mask.sum()) < 32:
        return source
    scale = np.median(target[mask]) / max(float(np.median(source[mask])), 1e-6)
    return source * scale


def load_font(size, bold=False):
    filename = "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf"
    path = Path("/usr/share/fonts/truetype/dejavu") / filename
    if path.is_file():
        return ImageFont.truetype(str(path), size=size)
    return ImageFont.load_default()


def labeled_tile(image, title, subtitle="", width=300):
    image = Image.fromarray(image).convert("RGB")
    height = int(round(width * image.height / max(image.width, 1)))
    image = image.resize((width, height), Image.Resampling.BILINEAR)
    header = 58
    tile = Image.new("RGB", (width, height + header), "white")
    draw = ImageDraw.Draw(tile)
    draw.text((9, 6), title, fill=(15, 23, 42), font=load_font(16, bold=True))
    if subtitle:
        draw.text((9, 31), subtitle, fill=(71, 85, 105), font=load_font(12))
    tile.paste(image, (0, header))
    return tile


def compose_rows(rows, heading, subtitle):
    gap = 8
    width = sum(tile.width for tile in rows[0]) + gap * (len(rows[0]) - 1)
    body_height = sum(row[0].height for row in rows) + gap * (len(rows) - 1)
    top = 74
    canvas = Image.new("RGB", (width, top + body_height), (241, 245, 249))
    draw = ImageDraw.Draw(canvas)
    draw.text((12, 8), heading, fill=(15, 23, 42), font=load_font(21, bold=True))
    draw.text((12, 42), subtitle, fill=(71, 85, 105), font=load_font(13))
    y = top
    for row in rows:
        x = 0
        for tile in row:
            canvas.paste(tile, (x, y))
            x += tile.width + gap
        y += row[0].height + gap
    return canvas


def visual_payload(batch, outputs, record):
    reference, reference_valid, _ = reference_from_batch(batch)
    reference_bchw = None if reference is None else as_bchw(reference)
    reference_valid_bchw = (
        None if reference_valid is None else as_bchw(reference_valid)
    )
    payload = {
        "record": sanitize(record),
        "rgb": tensor_rgb(batch["images_photo_dark"][0, 0]),
        "event": np.asarray(
            event_red_blue(batch["event_voxel"][0, 0], background="white")[0]
        ),
        "depths": {
            method: depth.detach().cpu().float().numpy()[:, 0]
            for method, depth in outputs.items()
        },
        "reference": (
            None
            if reference_bchw is None
            else reference_bchw.detach().cpu().float().numpy()[0, 0]
        ),
        "reference_valid": (
            None
            if reference_valid_bchw is None
            else reference_valid_bchw.detach().cpu().bool().numpy()[0, 0]
        ),
    }
    return payload


def save_visual(path, payload):
    record = payload["record"]
    reference = payload["reference"]
    reference_valid = payload["reference_valid"]
    depths = payload["depths"]
    if reference is not None:
        valid = reference_valid
        low, high = depth_limits([reference], valid)
        display = {}
        for method, pair_depth in depths.items():
            display[method] = [
                align_array(pair_depth[index], reference, valid)
                for index in (0, 1)
            ]
        reference_image = color_depth(reference, low, high, valid)
        reference_title = (
            "Sensor depth GT"
            if record["reference_kind"] == "sensor_depth_gt"
            else "DA3 pseudo depth"
        )
    else:
        valid = None
        anchor = depths["mast3r"][0]
        display = {}
        for method, pair_depth in depths.items():
            display[method] = [
                align_array(pair_depth[index], anchor, np.isfinite(anchor) & (anchor > 0))
                for index in (0, 1)
            ]
        low, high = depth_limits(
            [value for pair_depth in display.values() for value in pair_depth]
        )
        reference_image = np.full_like(payload["rgb"], 255)
        reference_title = "No depth reference"
    rows = [
        [
            labeled_tile(payload["rgb"], "Center low-light RGB", f"frame {record['center_frame']}"),
            labeled_tile(payload["event"], "Center events", "+ red / - blue"),
            labeled_tile(reference_image, reference_title, f"shared {low:.2f}-{high:.2f}"),
        ]
    ]
    method_titles = {
        "mast3r": "MASt3R",
        "night3r": "Night3R real",
        "night3r_zero": "Night3R zero",
    }
    for method in METHODS:
        left, right = display[method]
        mask = (
            np.isfinite(left)
            & np.isfinite(right)
            & (left > 0)
            & (right > 0)
        )
        right_aligned = align_array(right, left, mask)
        difference = np.abs(
            np.log(np.clip(left, 1e-6, None))
            - np.log(np.clip(right_aligned, 1e-6, None))
        )
        temporal = record["methods"][method]["temporal"]
        rows.append(
            [
                labeled_tile(
                    color_depth(left, low, high),
                    f"{method_titles[method]} | left",
                    "center t with context t-1",
                ),
                labeled_tile(
                    color_depth(right, low, high),
                    f"{method_titles[method]} | right",
                    "center t with context t+1",
                ),
                labeled_tile(
                    color_difference(difference, mask),
                    f"{method_titles[method]} | temporal diff",
                    f"SI |log d| {temporal['median_mean_abs_log']:.3f}",
                ),
            ]
        )
    panel = compose_rows(
        rows,
        f"{record['dataset']} | {record['scene']} | dynamic center {record['center_frame']}",
        (
            f"{record['split_label']} | event activity "
            f"{record['event_activity_mean']:.4f} | exposure {record['synthetic_exposure']}"
        ),
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    panel.save(path)
    metadata_path = path.with_suffix(".json")
    metadata_path.write_text(json.dumps(record, indent=2), encoding="utf-8")


def visual_rank(record):
    temporal_gain = record.get("night3r_vs_mast3r_temporal_gain")
    accuracy_gain = record.get("night3r_vs_mast3r_accuracy_gain")
    return (
        0.0 if temporal_gain is None else float(temporal_gain)
    ) + 0.5 * (0.0 if accuracy_gain is None else float(accuracy_gain))


def update_visual_candidates(candidates, scene, payload, limit):
    if limit <= 0:
        return
    record = payload["record"]
    entries = candidates[scene]
    entries.append(
        {
            "gain_score": visual_rank(record),
            "event_score": float(record["event_activity_mean"]),
            "payload": payload,
        }
    )
    ranked_gain = sorted(entries, key=lambda item: item["gain_score"], reverse=True)
    ranked_event = sorted(entries, key=lambda item: item["event_score"], reverse=True)
    selected = []
    seen = set()
    for item in ranked_gain + ranked_event:
        center = item["payload"]["record"]["center_frame"]
        if center in seen:
            continue
        selected.append(item)
        seen.add(center)
        if len(selected) >= limit:
            break
    candidates[scene] = selected


def flatten_method_values(records, method, section):
    values = defaultdict(list)
    for record in records:
        item = record["methods"][method].get(section)
        if section == "accuracy" and item:
            item = item.get("mean")
        if not item:
            continue
        for key, value in item.items():
            if key != "valid_px" and isinstance(value, (int, float)):
                values[key].append(float(value))
    return {
        key: float(np.mean(items))
        for key, items in sorted(values.items())
        if items
    }


def summarize_records(records):
    summary = {"num_centers": len(records), "methods": {}}
    for method in METHODS:
        summary["methods"][method] = {
            "temporal": flatten_method_values(records, method, "temporal"),
            "accuracy": flatten_method_values(records, method, "accuracy"),
        }
    for key in (
        "night3r_vs_mast3r_temporal_gain",
        "real_vs_zero_temporal_gain",
        "night3r_vs_mast3r_accuracy_gain",
        "real_vs_zero_accuracy_gain",
    ):
        values = [
            float(record[key])
            for record in records
            if record.get(key) is not None
        ]
        summary[key] = None if not values else float(np.mean(values))
    return summary


def load_completed(path):
    records = {}
    if not path.is_file():
        return records
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            records[str(record["protocol_id"])] = record
    return records


def main():
    args = parse_args()
    configure_cpu_threads()
    set_seed(args.seed)
    args.outdir.mkdir(parents=True, exist_ok=True)
    record_path = args.outdir / "records.jsonl"
    completed = load_completed(record_path) if args.resume else {}

    cfg = dict(load_config(args.config))
    cfg["lowlight_aug"] = {
        **dict(cfg.get("lowlight_aug", {})),
        "enabled": False,
    }
    train_args = make_train_args(args)
    dataset = build_dataset(args, cfg)
    requested_scenes = None if args.scenes is None else set(args.scenes)
    available_scenes = {
        scene_dir.name: scene_dir for scene_dir in dataset.scene_frames
    }
    if requested_scenes is not None:
        missing = sorted(requested_scenes - set(available_scenes))
        if missing:
            raise RuntimeError(f"Requested scenes were not found: {missing}")
        scene_names = sorted(requested_scenes)
    else:
        scene_names = sorted(available_scenes)
    indices = pair_index(dataset)

    device = resolve_device(args.device)
    master, master_load = build_model(
        make_model_args(args, "mast3r"), cfg, train_args, device
    )
    night, night_load = build_model(
        make_model_args(args, "event_dark3r"), cfg, train_args, device
    )

    records = dict(completed)
    visual_candidates = defaultdict(list)
    requested_centers = []
    for scene in scene_names:
        scene_dir = available_scenes[scene]
        num_frames = int(dataset.scene_frames[scene_dir]["num_frames"])
        for center in selected_centers(
            num_frames,
            args.max_centers_per_scene,
            args.center_stride,
        ):
            requested_centers.append((scene, center))

    mode = "a" if args.resume else "w"
    processed_now = 0
    with record_path.open(mode, encoding="utf-8") as handle:
        for scene, center in requested_centers:
            protocol_id = (
                f"{args.dataset_name}|{args.data_root.resolve()}|"
                f"{scene}|{center:06d}"
            )
            if protocol_id in records:
                continue
            left_key = (scene, center - 1, center)
            right_key = (scene, center, center + 1)
            if left_key not in indices or right_key not in indices:
                raise RuntimeError(
                    f"Adjacent pair missing for {scene} center={center}"
                )
            left = swap_sample(dataset[indices[left_key]])
            right = dataset[indices[right_key]]
            left = apply_deterministic_lowlight(left, args)
            right = apply_deterministic_lowlight(right, args)
            batch = default_collate([left, right])
            if not torch.equal(batch["idx1"], torch.tensor([center, center])):
                raise RuntimeError("Center frame is not view1 in both contexts")
            center_difference = (
                batch["images_photo_dark"][0, 0]
                - batch["images_photo_dark"][1, 0]
            ).abs().max()
            if float(center_difference) > 0:
                raise RuntimeError(
                    "Deterministic low-light center image differs across contexts"
                )
            outputs = infer_center_contexts(
                master, night, batch, train_args, cfg, device
            )
            record = sanitize(record_for_center(args, scene, center, batch, outputs))
            records[protocol_id] = record
            handle.write(json.dumps(record, sort_keys=True) + "\n")
            handle.flush()
            payload = visual_payload(batch, outputs, record)
            update_visual_candidates(
                visual_candidates,
                scene,
                payload,
                args.visuals_per_scene,
            )
            processed_now += 1
            if processed_now % args.progress_every == 0:
                print(
                    json.dumps(
                        {
                            "processed_now": processed_now,
                            "completed_total": len(records),
                            "requested_total": len(requested_centers),
                            "scene": scene,
                            "center": center,
                        }
                    ),
                    flush=True,
                )
            del batch, outputs, payload

    visuals_root = args.outdir / "visuals"
    for scene, entries in sorted(visual_candidates.items()):
        for rank, item in enumerate(
            sorted(
                entries,
                key=lambda value: (
                    value["gain_score"],
                    value["event_score"],
                ),
                reverse=True,
            ),
            start=1,
        ):
            center = item["payload"]["record"]["center_frame"]
            path = (
                visuals_root
                / safe_name(scene)
                / f"rank{rank:02d}_center{center:06d}.png"
            )
            save_visual(path, item["payload"])

    ordered_records = [
        records[
            (
                f"{args.dataset_name}|{args.data_root.resolve()}|"
                f"{scene}|{center:06d}"
            )
        ]
        for scene, center in requested_centers
        if (
            f"{args.dataset_name}|{args.data_root.resolve()}|"
            f"{scene}|{center:06d}"
        )
        in records
    ]
    summary = {
        "protocol": {
            "name": "bidirectional_center_frame_dynamic_reconstruction",
            "pair_orders": ["(t,t-1)", "(t,t+1)"],
            "center_frame_role": "view1 in both pairs",
            "temporal_alignment": "same pixel grid and same camera-t coordinate frame",
            "dynamic_object_mask": "unavailable",
            "interpretation": (
                "Evaluates temporal-context stability and depth accuracy in "
                "dynamic sequences; it does not isolate non-rigid objects."
            ),
        },
        "dataset": args.dataset_name,
        "data_root": str(args.data_root.resolve()),
        "scenes": scene_names,
        "requested_centers": len(requested_centers),
        "completed_centers": len(ordered_records),
        "synthetic_lowlight": {
            "exposure": args.synthetic_exposure,
            "read_noise_std": args.read_noise_std,
            "shot_noise_std": args.shot_noise_std,
            "gamma": args.gamma,
            "sample_keyed_seed": args.lowlight_seed,
        },
        "checkpoint": str(args.checkpoint.resolve()),
        "model_checkpoint": str(args.model_name_or_path.resolve()),
        "master_load": master_load,
        "night_load": night_load,
        "aggregate": summarize_records(ordered_records),
    }
    (args.outdir / "summary.json").write_text(
        json.dumps(sanitize(summary), indent=2), encoding="utf-8"
    )
    (args.outdir / "EVAL_COMPLETE").touch()
    print(json.dumps(sanitize(summary), indent=2), flush=True)


if __name__ == "__main__":
    main()
