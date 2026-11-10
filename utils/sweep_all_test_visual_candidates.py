#!/usr/bin/env python3
"""Traverse complete test scenes and export temporally diverse visual candidates.

GT-enabled datasets are ranked by median-aligned depth gains.  Datasets without
GT are ranked only by prediction discrepancy and event sensitivity; those scores
are explicitly marked as diagnostics rather than reconstruction quality.
"""

import argparse
import csv
import html
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
from train_event_dark3r import load_config, move_batch_to_device, resolve_amp_config  # noqa: E402
from utils.compare_event_dark3r_checkpoint import as_bchw, depth_from_pred  # noqa: E402
from utils.eval_event_dark3r_variant import build_model  # noqa: E402
from utils.export_rgb_lowlight_event_assets import event_red_blue  # noqa: E402


def configure_cpu_threads():
    value = os.environ.get("NIGHT3R_CPU_THREADS")
    if value is None:
        return
    threads = max(1, int(value))
    torch.set_num_threads(threads)
    torch.set_num_interop_threads(min(threads, 4))


def parse_args():
    parser = argparse.ArgumentParser("Sweep Night3R test scenes for visual candidates")
    parser.add_argument("--dataset-name", required=True, choices=("DSEC", "MVSEC", "DL3DV"))
    parser.add_argument("--data-root", required=True, type=Path)
    parser.add_argument("--scenes", nargs="+", required=True)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--model-name-or-path", required=True, type=Path)
    parser.add_argument("--outdir", required=True, type=Path)
    parser.add_argument("--image-size", nargs=2, type=int, default=[512, 512])
    parser.add_argument(
        "--darkening-mode",
        choices=("linear_exposure", "gamma"),
        default="linear_exposure",
        help="Controlled RGB darkening used by the sweep.",
    )
    parser.add_argument("--exposure", type=float, default=None)
    parser.add_argument(
        "--gamma-exponent",
        type=float,
        default=None,
        help="Exponent for gamma darkening I_dark = I ** gamma.",
    )
    parser.add_argument("--read-noise-std", type=float, default=0.01)
    parser.add_argument("--shot-noise-std", type=float, default=0.01)
    parser.add_argument(
        "--allow-mvsec-synthetic-darkening",
        action="store_true",
        help="Explicitly allow an additional controlled darkening pass on already-nighttime MVSEC frames.",
    )
    parser.add_argument("--selection-noise-penalty", type=float, default=0.0)
    parser.add_argument("--selection-brightness-target", type=float, default=None)
    parser.add_argument("--selection-brightness-penalty", type=float, default=0.0)
    parser.add_argument("--selection-black-floor", type=float, default=0.015)
    parser.add_argument("--selection-black-penalty", type=float, default=0.0)
    parser.add_argument("--temporal-buckets", type=int, default=8)
    parser.add_argument("--top-per-scene", type=int, default=4)
    parser.add_argument(
        "--pair-subsample-stride",
        type=int,
        default=1,
        help="Uniformly subsample adjacent pairs within each scene for qualitative sweeps.",
    )
    parser.add_argument("--amp-dtype", choices=("fp16", "bf16"), default="bf16")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="cuda")
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--progress-every", type=int, default=25)
    parser.add_argument("--allow-unsafe-checkpoint-load", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--refresh-visual-stats",
        action="store_true",
        help="Recompute display statistics for resumed records without repeating full inference.",
    )
    return parser.parse_args()


def resolve_device(name):
    if name == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but unavailable")
        return torch.device("cuda")
    if name == "cpu":
        return torch.device("cpu")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def safe_name(value):
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value)).strip("_")


def split_label(dataset, scene):
    if dataset == "MVSEC":
        return "adaptation-domain/train-scene-overlap"
    if dataset == "DSEC" and scene == "screen-000002-dsec":
        return "excluded-overlap"
    if dataset == "DL3DV":
        return "strict-heldout/no-depth-gt"
    return "strict-heldout"


def make_train_args(args):
    return SimpleNamespace(
        config=str(args.config),
        data_root=str(args.data_root),
        save_dir=str(args.outdir),
        model_name_or_path=str(args.model_name_or_path),
        dark3r_lora_checkpoint=None,
        allow_unsafe_checkpoint_load=args.allow_unsafe_checkpoint_load,
        use_tiny_model=False,
        image_size=args.image_size,
        batch_size=1,
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
    lowlight_enabled = (
        args.darkening_mode == "linear_exposure" and args.exposure is not None
    )
    lowlight_kwargs = None
    if lowlight_enabled:
        lowlight_kwargs = {
            "min_exposure": args.exposure,
            "max_exposure": args.exposure,
            "read_noise_std": args.read_noise_std,
            "shot_noise": True,
            "shot_noise_std": args.shot_noise_std,
            "gamma_space_input": True,
        }
    return DSECEventDataset(
        root_path=str(args.data_root),
        image_size=tuple(args.image_size),
        pair_stride=1,
        pair_sampling="fixed",
        expected_event_channels=int(cfg.get("event", {}).get("voxel_channels", 8)),
        lowlight_enabled=lowlight_enabled,
        lowlight_kwargs=lowlight_kwargs,
    )


def selected_indices(dataset, scenes, pair_subsample_stride=1):
    requested = set(scenes)
    available = {str(item["scene"]) for item in dataset.samples}
    missing = sorted(requested - available)
    if missing:
        raise RuntimeError("Requested scenes are unavailable: " + ", ".join(missing))
    stride = max(int(pair_subsample_stride), 1)
    grouped = defaultdict(list)
    for index, item in enumerate(dataset.samples):
        scene = str(item["scene"])
        if scene in requested:
            grouped[scene].append(index)
    selected = []
    for scene in sorted(grouped):
        scene_indices = grouped[scene]
        sampled = scene_indices[::stride]
        if scene_indices and sampled[-1] != scene_indices[-1]:
            sampled.append(scene_indices[-1])
        selected.extend(sampled)
    return selected


def deterministic_batch(dataset, dataset_index, seed):
    local_seed = int(seed + dataset_index * 1009)
    random.seed(local_seed)
    np.random.seed(local_seed % (2**32 - 1))
    torch.manual_seed(local_seed)
    return default_collate([dataset[dataset_index]])


def apply_requested_darkening(args, batch):
    if args.darkening_mode != "gamma":
        return batch
    dark = batch["images_photo"].float().clamp(0, 1).pow(args.gamma_exponent)
    batch["images_photo_dark"] = dark
    batch["images_dark"] = dark * 2.0 - 1.0
    return batch


def filtered_pair_points(pred1, pred2, stride=2, confidence_quantile=0.55):
    entries = (
        (pred1.get("pts3d"), pred1.get("conf")),
        (
            pred2.get("pts3d_in_other_view", pred2.get("pts3d")),
            pred2.get("conf"),
        ),
    )
    point_sets = []
    for points, confidence in entries:
        if points is None:
            continue
        points = points[0, ::stride, ::stride].detach().cpu().float()
        point_norm = torch.linalg.vector_norm(points, dim=-1)
        valid = torch.isfinite(points).all(dim=-1) & (point_norm > 1.0e-6)
        if confidence is not None:
            confidence = confidence[0, ::stride, ::stride].detach().cpu().float()
            finite_conf = torch.isfinite(confidence) & valid
            if int(finite_conf.sum()) >= 32:
                threshold = torch.quantile(
                    confidence[finite_conf], float(confidence_quantile)
                )
                valid &= confidence >= threshold
        point_sets.append(points[valid].numpy().astype(np.float32))
    if not point_sets:
        raise RuntimeError("Prediction does not contain a valid pair point map")

    reference_norm = np.linalg.norm(point_sets[0], axis=1)
    norm_low, norm_high = np.quantile(reference_norm, (0.005, 0.995))
    filtered = []
    for points in point_sets:
        point_norm = np.linalg.norm(points, axis=1)
        keep = point_norm >= max(float(norm_low) * 0.25, 1.0e-6)
        keep &= point_norm <= float(norm_high) * 1.5
        filtered.append(points[keep])

    points = np.concatenate(filtered, axis=0)
    center = np.median(points, axis=0)
    radius = np.linalg.norm(points - center[None], axis=1)
    pair_points = points[radius <= np.quantile(radius, 0.99)]
    result = {"pair": pair_points, "view1": filtered[0]}
    if len(filtered) > 1:
        result["view2"] = filtered[1]
    return result


def gt_pair_points(batch, stride=2):
    if "depth_gt1" not in batch or "intrinsics" not in batch:
        return None
    intrinsics = batch["intrinsics"][0].detach().cpu().float()
    fx, fy = float(intrinsics[0, 0]), float(intrinsics[1, 1])
    cx, cy = float(intrinsics[0, 2]), float(intrinsics[1, 2])
    point_sets = []
    for view_index, depth_key in enumerate(("depth_gt1", "depth_gt2")):
        if depth_key not in batch:
            continue
        depth = batch[depth_key][0].detach().cpu().float()
        h, w = depth.shape
        yy, xx = torch.meshgrid(
            torch.arange(0, h, stride),
            torch.arange(0, w, stride),
            indexing="ij",
        )
        z = depth[::stride, ::stride]
        valid = torch.isfinite(z) & (z > 0)
        points = torch.stack(
            (
                (xx.float() - cx) * z / fx,
                (yy.float() - cy) * z / fy,
                z,
            ),
            dim=-1,
        )[valid]
        if view_index == 1:
            if "relative_pose" not in batch:
                continue
            camera1_from_camera2 = torch.linalg.inv(
                batch["relative_pose"][0].detach().cpu().float()
            )
            points_h = torch.cat(
                (points, torch.ones((len(points), 1), dtype=points.dtype)), dim=1
            )
            points = (camera1_from_camera2 @ points_h.T).T[:, :3]
        point_sets.append(points.numpy().astype(np.float32))
    if not point_sets:
        return None
    result = {
        "pair": np.concatenate(point_sets, axis=0),
        "view1": point_sets[0],
    }
    if len(point_sets) > 1:
        result["view2"] = point_sets[1]
    return result


def infer_triplet(
    master,
    night,
    batch,
    train_args,
    cfg,
    device,
    include_geometry=False,
):
    batch_gpu = move_batch_to_device(batch, device)
    amp_enabled, _, amp_dtype = resolve_amp_config(cfg, train_args, device)
    with torch.inference_mode(), torch.amp.autocast(
        device_type=device.type,
        dtype=amp_dtype,
        enabled=bool(amp_enabled and device.type == "cuda"),
    ):
        view1, view2 = make_view_pair_from_batch(batch_gpu, prefer_dark=True)
        master_pred, master_pred2 = master(view1, view2)
        night_pred, night_pred2 = night(batch_gpu)
        zero_batch = dict(batch_gpu)
        zero_batch["event_voxel"] = torch.zeros_like(batch_gpu["event_voxel"])
        zero_pred, zero_pred2 = night(zero_batch)
    gate = night_pred.get("event_gate", float("nan"))
    gate = float(gate.detach().float().cpu()) if torch.is_tensor(gate) else float(gate)
    result = {
        "master": depth_from_pred(master_pred).detach().cpu().float(),
        "night": depth_from_pred(night_pred).detach().cpu().float(),
        "zero": depth_from_pred(zero_pred).detach().cpu().float(),
        "gate_mean": gate,
    }
    if include_geometry:
        master_geometry = filtered_pair_points(master_pred, master_pred2)
        night_geometry = filtered_pair_points(night_pred, night_pred2)
        result["geometry"] = {
            "master_points": master_geometry["pair"],
            "master_points_view1": master_geometry["view1"],
            "night_points": night_geometry["pair"],
            "night_points_view1": night_geometry["view1"],
        }
        if "view2" in master_geometry:
            result["geometry"]["master_points_view2"] = master_geometry["view2"]
        if "view2" in night_geometry:
            result["geometry"]["night_points_view2"] = night_geometry["view2"]
        gt_points = gt_pair_points(batch)
        if gt_points is not None:
            result["geometry"]["gt_points"] = gt_points["pair"]
            result["geometry"]["gt_points_view1"] = gt_points["view1"]
            if "view2" in gt_points:
                result["geometry"]["gt_points_view2"] = gt_points["view2"]
    del (
        batch_gpu,
        view1,
        view2,
        master_pred,
        master_pred2,
        night_pred,
        night_pred2,
        zero_pred,
        zero_pred2,
        zero_batch,
    )
    return result


def resize_depth(depth, target_hw):
    value = as_bchw(depth).float()
    if value.shape[-2:] != target_hw:
        value = F.interpolate(value, size=target_hw, mode="bilinear", align_corners=False)
    return value


def depth_metrics(depth, gt, valid):
    gt = as_bchw(gt).float()
    valid = as_bchw(valid).bool()
    depth = resize_depth(depth, gt.shape[-2:])
    valid = valid & torch.isfinite(gt) & torch.isfinite(depth) & (gt > 0) & (depth > 0)
    if int(valid.sum()) < 32:
        return None, depth
    target = gt[valid].clamp_min(1e-6)
    pred = depth[valid].clamp_min(1e-6)
    raw_ratio = torch.maximum(pred / target, target / pred)
    scale = torch.median(target) / torch.median(pred).clamp_min(1e-6)
    aligned = depth * scale
    aligned_pred = aligned[valid].clamp_min(1e-6)
    aligned_ratio = torch.maximum(aligned_pred / target, target / aligned_pred)
    return {
        "raw_absrel": float(((pred - target).abs() / target).mean()),
        "raw_delta1": float((raw_ratio < 1.25).float().mean()),
        "median_scale": float(scale),
        "median_absrel": float(((aligned_pred - target).abs() / target).mean()),
        "median_delta1": float((aligned_ratio < 1.25).float().mean()),
        "valid_px": int(valid.sum()),
    }, aligned


def scale_invariant_discrepancy(reference, estimate):
    reference = as_bchw(reference).float()
    estimate = resize_depth(estimate, reference.shape[-2:])
    valid = torch.isfinite(reference) & torch.isfinite(estimate) & (reference > 0) & (estimate > 0)
    if int(valid.sum()) < 32:
        return {"mean_abs_log": float("nan"), "p90_abs_log": float("nan"), "scale": 1.0}, estimate
    ref = reference[valid].clamp_min(1e-6)
    est = estimate[valid].clamp_min(1e-6)
    scale = torch.median(ref) / torch.median(est).clamp_min(1e-6)
    aligned = estimate * scale
    values = (torch.log(aligned[valid].clamp_min(1e-6)) - torch.log(ref)).abs()
    return {
        "mean_abs_log": float(values.mean()),
        "p90_abs_log": float(torch.quantile(values, 0.90)),
        "scale": float(scale),
        "valid_px": int(valid.sum()),
    }, aligned


def robust_dark_noise(image):
    """Estimate visible high-frequency noise in darker, non-saturated pixels."""
    gray = image.float().mean(dim=0, keepdim=True).unsqueeze(0)
    smooth = F.avg_pool2d(gray, kernel_size=5, stride=1, padding=2)
    residual = (gray - smooth).abs().flatten()
    intensity = gray.flatten()
    nonblack = intensity > (1.0 / 255.0)
    if int(nonblack.sum()) >= 32:
        positive = intensity[nonblack]
        lower = torch.quantile(positive, 0.05)
        upper = torch.quantile(positive, 0.65)
        dark = nonblack & (intensity >= lower) & (intensity <= upper)
    else:
        dark = torch.ones_like(nonblack)
    values = residual[dark]
    if values.numel() < 32:
        values = residual
    return float(values.median() * 1.4826)


def visual_stats(batch):
    image = batch["images_photo_dark"][0, 0].float()
    gray = image.mean(dim=0)
    event = batch["event_voxel"][0, 0].float()
    activity = event.abs().sum(dim=0)
    return {
        "lowlight_mean": float(gray.mean()),
        "lowlight_contrast": float(gray.std()),
        "lowlight_noise": robust_dark_noise(image),
        "event_activity_mean": float(activity.mean()),
        "event_occupancy": float((activity > 0).float().mean()),
    }


def apply_visual_selection_score(args, record):
    selection_score = float(record["score"])
    selection_score -= args.selection_noise_penalty * record["lowlight_noise"]
    if args.selection_brightness_target is not None:
        selection_score -= args.selection_brightness_penalty * max(
            record["lowlight_mean"] - args.selection_brightness_target, 0.0
        )
    selection_score -= args.selection_black_penalty * max(
        args.selection_black_floor - record["lowlight_mean"], 0.0
    )
    record["selection_score"] = float(selection_score)
    return record


def scalar_record(args, dataset_index, batch, outputs, scene_position, scene_count):
    scene = str(batch["scene"][0])
    idx1 = int(batch["idx1"][0])
    idx2 = int(batch["idx2"][0])
    bucket = min(
        args.temporal_buckets - 1,
        int(scene_position * args.temporal_buckets / max(scene_count, 1)),
    )
    record = {
        "dataset": args.dataset_name,
        "scene": scene,
        "sample_id": str(batch["sample_id"][0]),
        "dataset_index": int(dataset_index),
        "idx1": idx1,
        "idx2": idx2,
        "scene_position": int(scene_position),
        "scene_pair_count": int(scene_count),
        "temporal_bucket": int(bucket),
        "split_label": split_label(args.dataset_name, scene),
        "event_gate_mean": outputs["gate_mean"],
        **visual_stats(batch),
    }
    if "depth_gt" in batch:
        gt = batch["depth_gt"]
        valid = batch.get("depth_valid", torch.isfinite(gt) & (gt > 0))
        metric_values = {}
        for name in ("master", "night", "zero"):
            metrics, _ = depth_metrics(outputs[name], gt, valid)
            if metrics is None:
                return None
            metric_values[name] = metrics
        geometry_gain = (
            metric_values["master"]["median_absrel"]
            - metric_values["night"]["median_absrel"]
        )
        event_gain = (
            metric_values["zero"]["median_absrel"]
            - metric_values["night"]["median_absrel"]
        )
        geometry_gain_relative = geometry_gain / max(
            metric_values["master"]["median_absrel"], 0.05
        )
        event_gain_relative = event_gain / max(metric_values["zero"]["median_absrel"], 0.05)
        score = 0.70 * np.clip(geometry_gain_relative, -2.0, 2.0)
        score += 0.30 * np.clip(event_gain_relative, -2.0, 2.0)
        score += 0.03 * min(record["event_occupancy"], 0.5)
        record.update(
            {
                "score_kind": "gt_depth_improvement",
                "score": float(score),
                "geometry_gain": float(geometry_gain),
                "event_gain": float(event_gain),
                "metrics": metric_values,
            }
        )
    else:
        model_diff, _ = scale_invariant_discrepancy(outputs["master"], outputs["night"])
        event_diff, _ = scale_invariant_discrepancy(outputs["night"], outputs["zero"])
        score = min(model_diff["mean_abs_log"], 1.0)
        score += 0.5 * min(event_diff["mean_abs_log"], 1.0)
        score += 0.05 * min(record["event_occupancy"], 0.5)
        record.update(
            {
                "score_kind": "discrepancy_diagnostic_not_quality",
                "score": float(score),
                "mast3r_vs_night3r": model_diff,
                "night3r_real_vs_zero": event_diff,
                "warning": "No GT depth: discrepancy does not establish which prediction is better.",
            }
        )
    return apply_visual_selection_score(args, record)


def sanitize(value):
    if isinstance(value, dict):
        return {key: sanitize(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [sanitize(item) for item in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        value = float(value)
        return value if math.isfinite(value) else None
    return value


def load_existing_records(path):
    records = {}
    if not path.is_file():
        return records
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            records[int(item["dataset_index"])] = item
    return records


def scene_positions(dataset, indices):
    grouped = defaultdict(list)
    for index in indices:
        grouped[str(dataset.samples[index]["scene"])].append(index)
    positions = {}
    for scene, scene_indices in grouped.items():
        for position, index in enumerate(scene_indices):
            positions[index] = (position, len(scene_indices))
    return positions


def select_candidates(records, top_per_scene):
    grouped = defaultdict(list)
    for record in records:
        grouped[record["scene"]].append(record)
    selected = []
    for scene in sorted(grouped):
        best_by_bucket = {}
        for record in grouped[scene]:
            key = int(record["temporal_bucket"])
            rank_score = record.get("selection_score", record["score"])
            previous_score = (
                best_by_bucket[key].get("selection_score", best_by_bucket[key]["score"])
                if key in best_by_bucket
                else None
            )
            if previous_score is None or rank_score > previous_score:
                best_by_bucket[key] = record
        ranked = sorted(
            best_by_bucket.values(),
            key=lambda item: item.get("selection_score", item["score"]),
            reverse=True,
        )
        for rank, record in enumerate(ranked[:top_per_scene], start=1):
            copy = dict(record)
            copy["scene_rank"] = rank
            selected.append(copy)
    return selected


def tensor_rgb(tensor):
    value = tensor.detach().cpu().float().clamp(0, 1)
    if value.ndim == 4:
        value = value[0]
    return (value.permute(1, 2, 0).numpy() * 255.0 + 0.5).astype(np.uint8)


def depth_limits(arrays, valid=None):
    values = []
    for array in arrays:
        mask = np.isfinite(array) & (array > 0)
        if valid is not None:
            mask &= valid
        values.append(array[mask])
    values = [value for value in values if value.size]
    if not values:
        return 0.0, 1.0
    joined = np.concatenate(values)
    lo, hi = np.percentile(joined, [2, 98])
    return float(lo), float(max(hi, lo + 1e-6))


def color_depth(array, lo, hi, valid=None):
    normalized = np.clip((array - lo) / max(hi - lo, 1e-6), 0, 1)
    normalized = np.nan_to_num(normalized, nan=0.0, posinf=1.0, neginf=0.0)
    colored = cv2.applyColorMap(
        (normalized * 255).astype(np.uint8), cv2.COLORMAP_VIRIDIS
    )
    colored = cv2.cvtColor(colored, cv2.COLOR_BGR2RGB)
    mask = np.isfinite(array) & (array > 0)
    if valid is not None:
        mask &= valid
    colored[~mask] = 0
    return colored


def load_font(size, bold=False):
    filename = "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf"
    path = Path("/usr/share/fonts/truetype/dejavu") / filename
    return ImageFont.truetype(str(path), size=size) if path.is_file() else ImageFont.load_default()


def labeled_tile(image, title, subtitle="", width=320):
    image = Image.fromarray(image) if isinstance(image, np.ndarray) else image.convert("RGB")
    height = int(round(width * image.height / max(image.width, 1)))
    image = image.resize((width, height), Image.Resampling.BILINEAR)
    header = 58
    tile = Image.new("RGB", (width, height + header), "white")
    draw = ImageDraw.Draw(tile)
    draw.text((10, 7), title, fill=(15, 23, 42), font=load_font(17, bold=True))
    if subtitle:
        draw.text((10, 32), subtitle, fill=(71, 85, 105), font=load_font(13))
    tile.paste(image, (0, header))
    return tile


def compose_panel(rows, heading, badge):
    gap = 8
    width = sum(tile.width for tile in rows[0]) + gap * (len(rows[0]) - 1)
    body_h = sum(row[0].height for row in rows) + gap * (len(rows) - 1)
    top = 72
    canvas = Image.new("RGB", (width, top + body_h), (241, 245, 249))
    draw = ImageDraw.Draw(canvas)
    draw.text((14, 10), heading, fill=(15, 23, 42), font=load_font(22, bold=True))
    badge_color = (22, 101, 52) if badge == "strict-heldout" else (154, 52, 18)
    draw.text((14, 42), badge, fill=badge_color, font=load_font(15, bold=True))
    y = top
    for row in rows:
        x = 0
        for tile in row:
            canvas.paste(tile, (x, y))
            x += tile.width + gap
        y += row[0].height + gap
    return canvas


def export_candidate(args, dataset, master, night, train_args, cfg, device, record):
    batch = deterministic_batch(dataset, int(record["dataset_index"]), args.seed)
    batch = apply_requested_darkening(args, batch)
    outputs = infer_triplet(
        master,
        night,
        batch,
        train_args,
        cfg,
        device,
        include_geometry=True,
    )
    sample_dir = (
        args.outdir
        / "candidates"
        / safe_name(record["scene"])
        / f"rank{int(record['scene_rank']):02d}_{int(record['idx1']):06d}_{int(record['idx2']):06d}"
    )
    sample_dir.mkdir(parents=True, exist_ok=True)

    clean = batch["images_photo"][0]
    dark = batch["images_photo_dark"][0]
    event = batch["event_voxel"][0]
    images = {}
    for view in range(2):
        images[f"clean{view + 1}"] = tensor_rgb(clean[view])
        images[f"dark{view + 1}"] = tensor_rgb(dark[view])
        images[f"event{view + 1}"] = np.asarray(event_red_blue(event[view], background="white")[0])
        Image.fromarray(images[f"clean{view + 1}"]).save(sample_dir / f"clean_rgb_view{view + 1}.png")
        Image.fromarray(images[f"dark{view + 1}"]).save(sample_dir / f"lowlight_rgb_view{view + 1}.png")
        Image.fromarray(images[f"event{view + 1}"]).save(sample_dir / f"event_view{view + 1}_red_blue.png")

    arrays = {name: outputs[name].squeeze().numpy() for name in ("master", "night", "zero")}
    subtitles = {}
    if "depth_gt" in batch:
        gt = as_bchw(batch["depth_gt"])[0, 0].float()
        valid = as_bchw(batch["depth_valid"])[0, 0].bool()
        arrays["gt"] = gt.numpy()
        arrays["valid"] = valid.numpy()
        for name in ("master", "night", "zero"):
            metrics, aligned = depth_metrics(outputs[name], gt, valid)
            arrays[f"{name}_aligned"] = aligned.squeeze().numpy()
            subtitles[name] = (
                f"raw {metrics['raw_absrel']:.3f} | "
                f"med {metrics['median_absrel']:.3f}"
            )
        lo, hi = depth_limits([arrays["gt"]], arrays["valid"])
        depth_images = {
            name: color_depth(arrays[f"{name}_aligned"], lo, hi)
            for name in ("master", "night", "zero")
        }
        depth_images["gt"] = color_depth(arrays["gt"], lo, hi, arrays["valid"])
        fourth_title = "GT depth"
        fourth_subtitle = f"shared {lo:.2f}-{hi:.2f} m"
    else:
        _, night_aligned = scale_invariant_discrepancy(outputs["master"], outputs["night"])
        _, zero_aligned = scale_invariant_discrepancy(outputs["night"], outputs["zero"])
        arrays["night_aligned_to_master"] = night_aligned.squeeze().numpy()
        arrays["zero_aligned_to_night"] = zero_aligned.squeeze().numpy()
        lo, hi = depth_limits([arrays["master"], arrays["night_aligned_to_master"]])
        depth_images = {
            "master": color_depth(arrays["master"], lo, hi),
            "night": color_depth(arrays["night_aligned_to_master"], lo, hi),
            "zero": color_depth(arrays["zero_aligned_to_night"], lo, hi),
        }
        diff = np.abs(
            np.log(np.clip(arrays["night_aligned_to_master"], 1e-6, None))
            - np.log(np.clip(arrays["master"], 1e-6, None))
        )
        diff_hi = max(float(np.percentile(diff[np.isfinite(diff)], 98)), 1e-6)
        depth_images["gt"] = color_depth(diff, 0.0, diff_hi, np.isfinite(diff))
        subtitles["master"] = f"shared relative range {lo:.2f}-{hi:.2f}"
        subtitles["night"] = f"model diff {record['mast3r_vs_night3r']['mean_abs_log']:.3f}"
        subtitles["zero"] = f"event diff {record['night3r_real_vs_zero']['mean_abs_log']:.3f}"
        fourth_title = "Model log-depth difference"
        fourth_subtitle = "diagnostic only; no GT"

    for name, image in depth_images.items():
        output_name = "gt_depth.png" if name == "gt" else f"{name}_depth_shared.png"
        Image.fromarray(image).save(sample_dir / output_name)
    np.savez_compressed(sample_dir / "geometry_arrays.npz", **arrays)
    np.savez_compressed(sample_dir / "geometry_points.npz", **outputs["geometry"])

    top_row = [
        labeled_tile(images["dark1"], "Low-light RGB view 1"),
        labeled_tile(images["dark2"], "Low-light RGB view 2"),
        labeled_tile(images["event1"], "Events view 1", "+ red / - blue"),
        labeled_tile(images["event2"], "Events view 2", "+ red / - blue"),
    ]
    bottom_row = [
        labeled_tile(
            depth_images["master"],
            "MASt3R (median-aligned)" if "depth_gt" in batch else "MASt3R",
            subtitles.get("master", ""),
        ),
        labeled_tile(
            depth_images["night"],
            "Night3R real (median-aligned)" if "depth_gt" in batch else "Night3R real",
            subtitles.get("night", ""),
        ),
        labeled_tile(
            depth_images["zero"],
            "Night3R zero (median-aligned)" if "depth_gt" in batch else "Night3R zero",
            subtitles.get("zero", ""),
        ),
        labeled_tile(depth_images["gt"], fourth_title, fourth_subtitle),
    ]
    panel = compose_panel(
        [top_row, bottom_row],
        f"{args.dataset_name} | {record['sample_id']} | score {record['score']:.3f}",
        record["split_label"],
    )
    panel_path = sample_dir / "selection_panel.png"
    panel.save(panel_path)
    saved = dict(record)
    saved["asset_dir"] = str(sample_dir.resolve())
    saved["panel"] = str(panel_path.resolve())
    (sample_dir / "metrics.json").write_text(
        json.dumps(sanitize(saved), indent=2), encoding="utf-8"
    )
    return saved


def flatten_record(record):
    row = {key: value for key, value in record.items() if not isinstance(value, dict)}
    metrics = record.get("metrics", {})
    for model, values in metrics.items():
        for key, value in values.items():
            row[f"{model}_{key}"] = value
    for group in ("mast3r_vs_night3r", "night3r_real_vs_zero"):
        for key, value in record.get(group, {}).items():
            row[f"{group}_{key}"] = value
    return row


def write_csv(path, records):
    rows = [flatten_record(record) for record in records]
    fields = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_index(outdir, records):
    lines = [
        "<!doctype html><html><head><meta charset='utf-8'><title>Night3R test sweep</title>",
        "<style>body{font-family:Arial,sans-serif;margin:24px;background:#f1f5f9;color:#0f172a}"
        ".item{background:white;padding:14px;margin:0 0 22px;border:1px solid #cbd5e1}"
        "img{width:100%;height:auto}.warn{color:#9a3412;font-weight:700}"
        "code{background:#e2e8f0;padding:2px 5px}</style></head><body>",
        f"<h1>{html.escape(records[0]['dataset'] if records else 'Night3R')} full-scene candidate sweep</h1>",
        "<p>GT scenes are ranked by depth improvement. No-GT scenes are ranked by discrepancy only; discrepancy is not accuracy.</p>",
    ]
    for item in sorted(records, key=lambda value: (value["scene"], value["scene_rank"])):
        panel = Path(item["panel"]).relative_to(outdir)
        lines.append("<div class='item'>")
        lines.append(f"<h2>{html.escape(item['scene'])} #{item['scene_rank']}: {html.escape(item['sample_id'])}</h2>")
        lines.append(
            f"<p class='warn'>{html.escape(item['split_label'])}</p>"
            f"<p>score=<code>{item['score']:.4f}</code>, kind=<code>{html.escape(item['score_kind'])}</code></p>"
        )
        lines.append(f"<a href='{html.escape(str(panel))}'><img loading='lazy' src='{html.escape(str(panel))}'></a></div>")
    lines.append("</body></html>")
    (outdir / "index.html").write_text("\n".join(lines), encoding="utf-8")


def main():
    configure_cpu_threads()
    args = parse_args()
    args.outdir = args.outdir.resolve()
    if args.darkening_mode == "gamma":
        if args.gamma_exponent is None or args.gamma_exponent <= 1.0:
            raise ValueError("Gamma darkening requires --gamma-exponent greater than 1")
        if args.exposure is not None:
            raise ValueError("--exposure cannot be combined with --darkening-mode gamma")
    elif args.gamma_exponent is not None:
        raise ValueError("--gamma-exponent is only valid with --darkening-mode gamma")
    synthetic_darkening = (
        args.exposure is not None
        if args.darkening_mode == "linear_exposure"
        else args.gamma_exponent is not None
    )
    if (
        args.dataset_name == "MVSEC"
        and synthetic_darkening
        and not args.allow_mvsec_synthetic_darkening
    ):
        raise ValueError(
            "MVSEC is already nighttime; pass --allow-mvsec-synthetic-darkening "
            "to opt in to an additional controlled darkening pass"
        )
    if args.dataset_name in {"DSEC", "DL3DV"} and not synthetic_darkening:
        raise ValueError(
            f"{args.dataset_name} sweep requires an explicit controlled darkening"
        )
    set_seed(args.seed)
    args.outdir.mkdir(parents=True, exist_ok=True)
    device = resolve_device(args.device)
    cfg = load_config(str(args.config))
    train_args = make_train_args(args)
    dataset = build_dataset(args, cfg)
    indices = selected_indices(dataset, args.scenes, args.pair_subsample_stride)
    positions = scene_positions(dataset, indices)
    print(
        json.dumps(
            {
                "event": "sweep_start",
                "dataset": args.dataset_name,
                "scenes": args.scenes,
                "pair_count": len(indices),
                "visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
                "device": str(device),
                "cpu_threads": torch.get_num_threads(),
                "cpu_interop_threads": torch.get_num_interop_threads(),
            }
        ),
        flush=True,
    )

    master, master_info = build_model(make_model_args(args, "mast3r"), cfg, train_args, device)
    night, night_info = build_model(make_model_args(args, "event_dark3r"), cfg, train_args, device)
    jsonl = args.outdir / "all_pairs.jsonl"
    existing = load_existing_records(jsonl) if args.resume else {}
    if args.refresh_visual_stats and existing:
        for dataset_index in indices:
            if dataset_index not in existing:
                continue
            batch = deterministic_batch(dataset, dataset_index, args.seed)
            batch = apply_requested_darkening(args, batch)
            existing[dataset_index].update(visual_stats(batch))
            apply_visual_selection_score(args, existing[dataset_index])
        temporary = jsonl.with_suffix(".jsonl.tmp")
        with temporary.open("w", encoding="utf-8") as handle:
            for dataset_index in indices:
                if dataset_index in existing:
                    handle.write(json.dumps(sanitize(existing[dataset_index]), allow_nan=False) + "\n")
        os.replace(temporary, jsonl)
        print(
            json.dumps(
                {
                    "event": "visual_stats_refreshed",
                    "dataset": args.dataset_name,
                    "records": len(existing),
                }
            ),
            flush=True,
        )
    mode = "a" if args.resume and jsonl.is_file() else "w"
    processed_now = 0
    with jsonl.open(mode, encoding="utf-8") as handle:
        for dataset_index in indices:
            if dataset_index in existing:
                continue
            batch = deterministic_batch(dataset, dataset_index, args.seed)
            batch = apply_requested_darkening(args, batch)
            outputs = infer_triplet(master, night, batch, train_args, cfg, device)
            position, count = positions[dataset_index]
            record = scalar_record(args, dataset_index, batch, outputs, position, count)
            if record is not None:
                existing[dataset_index] = sanitize(record)
                handle.write(json.dumps(existing[dataset_index], allow_nan=False) + "\n")
                handle.flush()
            processed_now += 1
            if processed_now % args.progress_every == 0:
                print(
                    json.dumps(
                        {
                            "event": "progress",
                            "dataset": args.dataset_name,
                            "processed_now": processed_now,
                            "completed_total": len(existing),
                            "target": len(indices),
                        }
                    ),
                    flush=True,
                )

    all_records = [existing[index] for index in indices if index in existing]
    selected = select_candidates(all_records, args.top_per_scene)
    saved = []
    for number, record in enumerate(selected, start=1):
        saved.append(export_candidate(args, dataset, master, night, train_args, cfg, device, record))
        print(
            json.dumps({"event": "candidate_export", "number": number, "total": len(selected), "sample_id": record["sample_id"]}),
            flush=True,
        )

    write_csv(args.outdir / "all_pairs.csv", all_records)
    write_csv(args.outdir / "ranking.csv", saved)
    manifest = {
        "dataset": args.dataset_name,
        "data_root": str(args.data_root.resolve()),
        "scenes": args.scenes,
        "requested_pairs": len(indices),
        "processed_pairs": len(all_records),
        "selected_candidates": len(saved),
        "top_per_scene": args.top_per_scene,
        "temporal_buckets": args.temporal_buckets,
        "lowlight": {
            "mode": args.darkening_mode,
            "formula": (
                "I_dark = I ** gamma"
                if args.darkening_mode == "gamma"
                else "I_dark = gamma_encode(exposure * gamma_decode(I) + sensor_noise)"
            ),
            "gamma_exponent": (
                args.gamma_exponent if args.darkening_mode == "gamma" else None
            ),
            "exposure": (
                args.exposure if args.darkening_mode == "linear_exposure" else None
            ),
            "read_noise_std": (
                args.read_noise_std
                if args.darkening_mode == "linear_exposure" and args.exposure is not None
                else None
            ),
            "shot_noise_std": (
                args.shot_noise_std
                if args.darkening_mode == "linear_exposure" and args.exposure is not None
                else None
            ),
            "gamma_space_input": (
                True
                if args.darkening_mode == "linear_exposure" and args.exposure is not None
                else None
            ),
            "mvsec_synthetic_darkening_opt_in": bool(
                args.dataset_name == "MVSEC"
                and synthetic_darkening
                and args.allow_mvsec_synthetic_darkening
            ),
        },
        "checkpoint": str(args.checkpoint.resolve()),
        "config": str(args.config.resolve()),
        "backbone": str(args.model_name_or_path.resolve()),
        "visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "master_load_info": master_info,
        "night_load_info": night_info,
        "selection_protocol": {
            "gt": "0.70 relative MASt3R-to-Night3R median-AbsRel gain + 0.30 real-to-zero event gain, after one best candidate per temporal bucket",
            "no_gt": "MASt3R/Night3R and real/zero scale-invariant log-depth discrepancy; diagnostic only, not quality",
            "visual_ranking": {
                "noise_estimator": "1.4826 times median 5x5 high-pass magnitude over the 5%-65% nonblack intensity range",
                "noise_penalty": args.selection_noise_penalty,
                "brightness_target": args.selection_brightness_target,
                "brightness_penalty": args.selection_brightness_penalty,
                "black_floor": args.selection_black_floor,
                "black_penalty": args.selection_black_penalty,
            },
        },
        "records": saved,
    }
    (args.outdir / "manifest.json").write_text(
        json.dumps(sanitize(manifest), indent=2), encoding="utf-8"
    )
    write_index(args.outdir, saved)
    print(
        json.dumps(
            {
                "event": "sweep_complete",
                "outdir": str(args.outdir.resolve()),
                "processed": len(all_records),
                "selected": len(saved),
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
