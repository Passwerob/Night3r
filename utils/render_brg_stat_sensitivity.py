#!/usr/bin/env python3
"""Evaluate post-normalization BRG statistic sensitivity and render paper figures."""

import argparse
import copy
import csv
import json
import math
import os
import sys
from collections import defaultdict
from contextlib import contextmanager
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
    DenseMetricAccumulator,
    as_bchw,
    depth_from_pred,
    set_seed,
)
from utils.depth_cutoff_metrics import DepthCutoffAccumulator  # noqa: E402


DEFAULT_CHECKPOINT = REPO_ROOT / "outputs/night3r_final_step100_20260723/event_dark3r_night3r_final.pth"
DEFAULT_BACKBONE = REPO_ROOT / "checkpoints/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric.pth"
DEFAULT_DSEC = Path("/data/fcr/data/train_data/DSEC_test")
DEFAULT_MVSEC = Path("/data/fcr/data/train_data/MVSEC_outdoor_night2_test")

VARIANTS = (
    ("full", "Full statistics", (1.0, 1.0, 1.0)),
    ("no_density", "- Density", (0.0, 1.0, 1.0)),
    ("no_occupancy", "- Occupancy", (1.0, 0.0, 1.0)),
    ("no_polarity", "- Polarity", (1.0, 1.0, 0.0)),
)


def parse_args():
    parser = argparse.ArgumentParser(
        "Render post-normalization leave-one-statistic-out sensitivity for BRG."
    )
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument(
        "--config",
        default="config/train_night3r_global_final_continue_200_20260723.yaml",
        help="Architecture-compatible Night3R config used to instantiate the checkpoint.",
    )
    parser.add_argument("--model-name-or-path", type=Path, default=DEFAULT_BACKBONE)
    parser.add_argument("--allow-unsafe-checkpoint-load", action="store_true")
    parser.add_argument(
        "--dataset",
        action="append",
        default=None,
        metavar="NAME=ROOT",
        help="Repeatable evaluation dataset. Defaults to DSEC and MVSEC held-out roots.",
    )
    parser.add_argument("--max-batches", type=int, default=128)
    parser.add_argument("--visual-samples", type=int, default=3)
    parser.add_argument(
        "--render-only",
        action="store_true",
        help="Render only selected qualitative samples; skip metric evaluation.",
    )
    parser.add_argument(
        "--visual-batch-index",
        action="append",
        type=int,
        default=None,
        metavar="INDEX",
        help="Dataset batch index to render in --render-only mode; repeatable.",
    )
    parser.add_argument(
        "--paper-compact",
        action="store_true",
        help="Render the ablation-focused compact qualitative layout without absolute gate maps.",
    )
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--image-size", type=int, nargs=2, default=None, metavar=("H", "W"))
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="cuda")
    parser.add_argument("--amp", dest="amp", action="store_true", default=True)
    parser.add_argument("--no-amp", dest="amp", action="store_false")
    parser.add_argument("--amp-dtype", choices=("fp16", "bf16"), default="bf16")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "outputs/brg_stat_sensitivity_step100_20260726",
    )
    return parser.parse_args()


def parse_datasets(specs):
    if not specs:
        return (("DSEC", DEFAULT_DSEC), ("MVSEC", DEFAULT_MVSEC))
    parsed = []
    for spec in specs:
        if "=" not in spec:
            raise ValueError(f"Expected NAME=ROOT, got {spec!r}")
        name, root = spec.split("=", 1)
        name = name.strip()
        root = Path(root.strip())
        if not name or not root:
            raise ValueError(f"Expected NAME=ROOT, got {spec!r}")
        parsed.append((name, root))
    return tuple(parsed)


def resolve_device(device_arg):
    if device_arg == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("--device cuda requested but CUDA is unavailable")
        return torch.device("cuda")
    if device_arg == "cpu":
        return torch.device("cpu")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def make_train_args(args, data_root):
    return SimpleNamespace(
        config=args.config,
        data_root=str(data_root),
        save_dir=str(args.output_dir),
        model_name_or_path=str(args.model_name_or_path),
        allow_unsafe_checkpoint_load=args.allow_unsafe_checkpoint_load,
        use_tiny_model=False,
        image_size=args.image_size,
        batch_size=args.batch_size,
        max_pairs=args.max_batches,
        start_index=0,
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


def evaluation_config(config):
    config = copy.deepcopy(config)
    lowlight = dict(config.get("lowlight_aug", {}))
    lowlight["enabled"] = False
    config["lowlight_aug"] = lowlight
    return config


def build_model(args, config, device):
    train_args = make_train_args(args, DEFAULT_DSEC)
    backbone = load_backbone(train_args, config)
    model = build_event_student(backbone, config=config)
    final_stage = resolve_stage_schedule(config, train_args)[-1]
    set_event_training_stage(
        model,
        stage=final_stage["stage"],
        partial_finetune_blocks=final_stage["partial_finetune_blocks"],
    )
    payload = safe_torch_load(
        args.checkpoint,
        map_location="cpu",
        allow_unsafe=args.allow_unsafe_checkpoint_load,
        purpose="BRG statistic sensitivity checkpoint",
    )
    state_dict = payload.get("model", payload)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    model = model.to(device)
    model.eval()
    fusion = getattr(model, "fusion_adapter", None)
    if fusion is None or not hasattr(fusion, "stat_norm") or not hasattr(fusion, "token_gate"):
        raise RuntimeError("This script requires a BRG checkpoint with stat_norm and token_gate.")
    if getattr(model, "event_film_enabled", False):
        raise RuntimeError(
            "Event FiLM is enabled. This test masks only the BRG gate input; use a checkpoint "
            "with event FiLM disabled to make the claimed statistic intervention unambiguous."
        )
    return model, {
        "step": payload.get("step"),
        "stage": payload.get("stage"),
        "epoch": payload.get("epoch"),
        "missing_keys": list(missing),
        "unexpected_keys": list(unexpected),
    }


@contextmanager
def post_norm_stat_mask(model, mask_values):
    """Mask statistic dimensions after normalization without touching occupancy safety."""
    fusion = model.fusion_adapter
    stat_dim = int(getattr(fusion, "stat_dim", 0))
    if len(mask_values) != stat_dim:
        raise ValueError(f"Expected {stat_dim} mask values, got {mask_values}")
    mask = torch.tensor(mask_values, dtype=torch.float32).view(1, 1, stat_dim)

    def mask_hook(_module, _inputs, output):
        return output * mask.to(device=output.device, dtype=output.dtype)

    handle = fusion.stat_norm.register_forward_hook(mask_hook)
    try:
        yield
    finally:
        handle.remove()


def make_loader(args, config, data_root):
    train_args = make_train_args(args, data_root)
    set_seed(args.seed)
    dataset = build_dataset(train_args, config)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=False,
    )
    return train_args, dataset, loader


def batch_sample_id(batch, batch_idx):
    value = batch.get("sample_id")
    if isinstance(value, (list, tuple)) and value:
        return str(value[0])
    if value is not None:
        return str(value)
    return f"batch_{batch_idx:04d}"


def depth_reference(batch):
    depth = as_bchw(batch.get("depth_gt"))
    if depth is None:
        return None, None
    valid_value = batch.get("depth_valid")
    valid = as_bchw(valid_value).bool() if valid_value is not None else torch.isfinite(depth) & (depth > 0)
    valid = valid & torch.isfinite(depth) & (depth > 0)
    return depth, valid


def finite_scalar(value):
    if torch.is_tensor(value):
        value = value.detach().float()
        return float(value.mean().cpu()) if value.numel() else float("nan")
    if isinstance(value, (int, float)):
        return float(value)
    return float("nan")


def evaluate_variant(model, args, config, device, data_root, mask_values):
    train_args, dataset, loader = make_loader(args, config, data_root)
    amp_enabled, _, amp_dtype = resolve_amp_config(config, train_args, device)
    dense_metrics = DenseMetricAccumulator()
    cutoff_metrics = DepthCutoffAccumulator()
    diagnostics = defaultdict(list)
    samples = []

    with post_norm_stat_mask(model, mask_values), torch.no_grad():
        for batch_idx, batch in enumerate(loader):
            if args.max_batches is not None and batch_idx >= args.max_batches:
                break
            sample_id = batch_sample_id(batch, batch_idx)
            batch_gpu = move_batch_to_device(batch, device)
            with torch.amp.autocast(
                device_type=device.type,
                dtype=amp_dtype,
                enabled=bool(amp_enabled and device.type == "cuda"),
            ):
                pred1, _ = model(batch_gpu)
            prediction = depth_from_pred(pred1)
            gt_depth, gt_valid = depth_reference(batch_gpu)
            if prediction is not None and gt_depth is not None:
                dense_metrics.add_depth(prediction, gt_depth, gt_valid, variant="raw")
                dense_metrics.add_depth(prediction, gt_depth, gt_valid, variant="median_aligned")
                cutoff_metrics.add_depth(prediction, gt_depth, gt_valid)
            gate_mean = finite_scalar(pred1.get("event_gate"))
            diagnostics["gate_mean"].append(gate_mean)
            samples.append({"sample_id": sample_id, "gate_mean": gate_mean, "batch_index": batch_idx})

    summary = {
        key: float(np.nanmean(values)) if values else float("nan")
        for key, values in diagnostics.items()
    }
    return {
        "dataset_len": len(dataset),
        "num_batches": len(samples),
        "dense_metrics": dense_metrics.finalize(),
        "depth_cutoff_metrics": cutoff_metrics.finalize(),
        "diagnostics": summary,
        "samples": samples,
    }


def select_visual_samples(results, dataset_name, limit):
    if limit <= 0:
        return []
    per_sample = defaultdict(dict)
    for variant_key, payload in results[dataset_name].items():
        for item in payload["samples"]:
            per_sample[item["sample_id"]][variant_key] = item
    scores = []
    for sample_id, items in per_sample.items():
        if len(items) != len(VARIANTS):
            continue
        values = [item["gate_mean"] for item in items.values()]
        if not all(math.isfinite(value) for value in values):
            continue
        scores.append((max(values) - min(values), sample_id, items["full"]["batch_index"]))
    scores.sort(reverse=True)
    return [(sample_id, batch_index) for _, sample_id, batch_index in scores[:limit]]


def first_view_image(batch):
    images = batch.get("images_photo_dark", batch.get("images_photo"))
    if images is None:
        raise RuntimeError("Expected images_photo_dark or images_photo in batch")
    return images[0, 0].detach().cpu().float().clamp(0.0, 1.0)


def first_view_event_activity(batch):
    event = batch.get("event_voxel")
    if event is None:
        raise RuntimeError("Expected event_voxel in batch")
    return event[0, 0].detach().cpu().float().abs().sum(dim=0)


def display_gate_before_occupancy_mask(model, batch, pred, gate_map):
    """Recompute the BRG gate for visualization before the no-event hard mask."""
    fusion = model.fusion_adapter
    if getattr(fusion, "visibility_gate_enabled", False) or getattr(
        fusion, "structural_gate_enabled", False
    ):
        raise RuntimeError(
            "Pre-occupancy gate rendering currently requires visibility and structural gates disabled."
        )
    event_stat = pred.get("event_stat")
    if event_stat is None:
        raise RuntimeError("Pre-occupancy gate rendering requires event_stat")
    if event_stat.shape[1] != gate_map.shape[-2] * gate_map.shape[-1]:
        raise RuntimeError(
            "event_stat and event_gate_map use incompatible token grids: "
            f"{event_stat.shape} vs {gate_map.shape}"
        )

    with torch.amp.autocast(device_type=event_stat.device.type, enabled=False):
        event_stat = event_stat.float()
        source_id = model._source_id_from_batch(batch, event_stat.device, event_stat.shape[0])
        normalized_stat = fusion.stat_norm(event_stat * fusion.event_stat_mask, source_id=source_id)
        raw_logit = fusion.token_gate(normalized_stat)
        if fusion.source_bias is not None and source_id is not None:
            ids = source_id.long().view(-1)
            valid_source = (ids >= 0) & (ids < fusion.num_sources)
            if valid_source.all():
                raw_logit = raw_logit + fusion.source_bias(ids).view(-1, 1, 1)
        bounded_logit = fusion.logit_bound * torch.tanh(raw_logit / fusion.logit_bound)
        gate = fusion.get_gate_cap(step=model._current_step()) * torch.sigmoid(
            bounded_logit / fusion.gate_temperature
        )
    return gate[..., 0].view_as(gate_map)


def collect_visuals(model, args, config, device, data_root, selections):
    if not selections:
        return {}
    selected_by_index = {batch_index: sample_id for sample_id, batch_index in selections}
    train_args, _, loader = make_loader(args, config, data_root)
    amp_enabled, _, amp_dtype = resolve_amp_config(config, train_args, device)
    records = {sample_id: {"variants": {}} for sample_id, _ in selections}

    for variant_key, _label, mask_values in VARIANTS:
        with post_norm_stat_mask(model, mask_values), torch.no_grad():
            for batch_idx, batch in enumerate(loader):
                if batch_idx > max(selected_by_index):
                    break
                sample_id = selected_by_index.get(batch_idx)
                if sample_id is None:
                    continue
                batch_gpu = move_batch_to_device(batch, device)
                with torch.amp.autocast(
                    device_type=device.type,
                    dtype=amp_dtype,
                    enabled=bool(amp_enabled and device.type == "cuda"),
                ):
                    pred1, _ = model(batch_gpu)
                depth = depth_from_pred(pred1)
                gate_map = pred1.get("event_gate_map")
                if depth is None or gate_map is None:
                    raise RuntimeError("Expected depth and event_gate_map from BRG model output")
                display_gate_map = display_gate_before_occupancy_mask(
                    model, batch_gpu, pred1, gate_map
                )
                gt_depth, gt_valid = depth_reference(batch_gpu)
                if gt_depth is None:
                    raise RuntimeError("Visualization requires GT depth; choose DSEC or MVSEC")
                record = records[sample_id]
                if "input" not in record:
                    event_stat = pred1.get("event_stat")
                    if event_stat is None:
                        raise RuntimeError("Visualization requires token-level event statistics")
                    if event_stat.shape[1] != gate_map.shape[-2] * gate_map.shape[-1]:
                        raise RuntimeError(
                            "event_stat and event_gate_map use incompatible token grids: "
                            f"{event_stat.shape} vs {gate_map.shape}"
                        )
                    record["input"] = {
                        "image": first_view_image(batch),
                        "event": first_view_event_activity(batch),
                        "gt_depth": gt_depth[0, 0].detach().cpu().float(),
                        "valid": gt_valid[0, 0].detach().cpu().bool(),
                        "occupancy": event_stat[0, :, 1]
                        .detach()
                        .cpu()
                        .float()
                        .view(gate_map.shape[-2:]),
                        "sample_id": batch_sample_id(batch, batch_idx),
                    }
                record["variants"][variant_key] = {
                    "depth": depth[0, 0].detach().cpu().float(),
                    "gate": gate_map[0].detach().cpu().float(),
                    "display_gate": display_gate_map[0].detach().cpu().float(),
                    "gate_mean": finite_scalar(pred1.get("event_gate")),
                }
    return records


def finite_percentile(values, percentile, default=1.0):
    values = np.asarray(values, dtype=np.float32)
    values = values[np.isfinite(values)]
    return float(np.percentile(values, percentile)) if values.size else float(default)


def align_depth_to_gt(depth, gt_depth, valid):
    if depth.shape != gt_depth.shape:
        depth = F.interpolate(depth[None, None], size=gt_depth.shape, mode="bilinear", align_corners=False)[0, 0]
    valid = valid & torch.isfinite(depth) & torch.isfinite(gt_depth) & (depth > 0) & (gt_depth > 0)
    if valid.any():
        depth = depth * (gt_depth[valid].median() / depth[valid].median().clamp_min(1e-6))
    return depth, valid


def chw_to_hwc(image):
    return image.permute(1, 2, 0).numpy()


def render_visual_panel(output_dir, dataset_name, record):
    data = record["input"]
    variants = record["variants"]
    if len(variants) != len(VARIANTS):
        return None
    gt_depth = data["gt_depth"]
    valid = data["valid"] & torch.isfinite(gt_depth) & (gt_depth > 0)
    gt_values = gt_depth[valid].numpy() if valid.any() else gt_depth.numpy().reshape(-1)
    depth_vmin = finite_percentile(gt_values, 2, default=0.0)
    depth_vmax = finite_percentile(gt_values, 98, default=depth_vmin + 1.0)
    if depth_vmax <= depth_vmin:
        depth_vmax = depth_vmin + 1e-6

    aligned = {}
    errors = {}
    gates = {}
    for key, _label, _mask in VARIANTS:
        depth, depth_valid = align_depth_to_gt(variants[key]["depth"], gt_depth, valid)
        aligned[key] = depth
        error = (depth - gt_depth).abs()
        errors[key] = error.masked_fill(~depth_valid, float("nan"))
        gate = variants[key]["display_gate"]
        gate = F.interpolate(
            gate[None, None], size=gt_depth.shape, mode="bilinear", align_corners=False
        )[0, 0]
        gates[key] = gate
    gate_values = torch.cat([item.reshape(-1) for item in gates.values()]).numpy()
    gate_vmin = finite_percentile(gate_values, 2, default=0.0)
    gate_vmax = finite_percentile(gate_values, 98, default=gate_vmin + 1.0)
    if gate_vmax <= gate_vmin:
        gate_vmax = gate_vmin + 1e-6
    err_vmax = max(
        finite_percentile(torch.cat([item.reshape(-1) for item in errors.values()]).numpy(), 95, 1.0),
        1e-6,
    )
    delta_vmax = max(
        finite_percentile(
            torch.cat([(gates[key] - gates["full"]).abs().reshape(-1) for key in gates]).numpy(),
            99,
            1e-6,
        ),
        1e-6,
    )

    gate_cmap = plt.get_cmap("viridis")
    delta_cmap = plt.get_cmap("coolwarm")
    fig, axes = plt.subplots(5, 4, figsize=(15.5, 18), constrained_layout=True)
    fig.suptitle(
        f"{dataset_name}: {data['sample_id']} | post-normalization statistic sensitivity",
        fontsize=15,
    )
    top_panels = (
        (data["image"], "Low-light RGB", None, None, None),
        (data["event"], "Event activity", "magma", None, None),
        (gt_depth, "GT depth", "viridis", depth_vmin, depth_vmax),
        (None, "", None, None, None),
    )
    for axis, (image, title, cmap, vmin, vmax) in zip(axes[0], top_panels):
        if image is None:
            axis.axis("off")
            continue
        if cmap is None:
            axis.imshow(np.clip(chw_to_hwc(image), 0.0, 1.0))
        else:
            shown = axis.imshow(image.numpy(), cmap=cmap, vmin=vmin, vmax=vmax)
            fig.colorbar(shown, ax=axis, fraction=0.046, pad=0.02)
        axis.set_title(title, fontsize=11)
        axis.axis("off")
    axes[0, 3].text(
        0.02,
        0.88,
        "Intervention\n\nEach ablation masks one\nnormalized statistic before\nthe BRG token-gate MLP.\n\nGate maps are displayed\nbefore the occupancy safety\nmask. Model inference\nremains unchanged.\n\nGate colors: p2-p98.",
        transform=axes[0, 3].transAxes,
        va="top",
        fontsize=11,
    )

    for row, (key, label, _mask) in enumerate(VARIANTS, start=1):
        gate = gates[key]
        delta = gate - gates["full"]
        panels = (
            (
                gate,
                f"{label}\nPre-occupancy gate map (display only)",
                gate_cmap,
                gate_vmin,
                gate_vmax,
            ),
            (delta, "Pre-occupancy gate\nchange vs Full", delta_cmap, -delta_vmax, delta_vmax),
            (aligned[key], "Median-aligned depth", "viridis", depth_vmin, depth_vmax),
            (errors[key], "Absolute depth error", "inferno", 0.0, err_vmax),
        )
        for axis, (image, title, cmap, vmin, vmax) in zip(axes[row], panels):
            shown = axis.imshow(image.numpy(), cmap=cmap, vmin=vmin, vmax=vmax)
            fig.colorbar(shown, ax=axis, fraction=0.046, pad=0.02)
            axis.set_title(title, fontsize=10)
            axis.axis("off")
        axes[row, 0].set_ylabel(label, fontsize=12, rotation=90, labelpad=24)

    stem = f"{dataset_name.lower()}_{data['sample_id'].replace('/', '_').replace(':', '_')}"
    for suffix in ("png", "pdf"):
        fig.savefig(output_dir / f"{stem}.{suffix}", dpi=220 if suffix == "png" else None, bbox_inches="tight")
    plt.close(fig)
    return stem


def render_compact_ablation_panel(output_dir, dataset_name, record):
    """Render a publication-focused layout showing only ablation-induced changes."""
    data = record["input"]
    variants = record["variants"]
    if len(variants) != len(VARIANTS):
        return None
    gt_depth = data["gt_depth"]
    valid = data["valid"] & torch.isfinite(gt_depth) & (gt_depth > 0)
    gt_values = gt_depth[valid].numpy() if valid.any() else gt_depth.numpy().reshape(-1)
    depth_vmin = finite_percentile(gt_values, 2, default=0.0)
    depth_vmax = finite_percentile(gt_values, 98, default=depth_vmin + 1.0)
    if depth_vmax <= depth_vmin:
        depth_vmax = depth_vmin + 1e-6

    aligned = {}
    errors = {}
    gates = {}
    for key, _label, _mask in VARIANTS:
        depth, depth_valid = align_depth_to_gt(variants[key]["depth"], gt_depth, valid)
        aligned[key] = depth
        errors[key] = (depth - gt_depth).abs().masked_fill(~depth_valid, float("nan"))
        gate = F.interpolate(
            variants[key]["display_gate"][None, None],
            size=gt_depth.shape,
            mode="bilinear",
            align_corners=False,
        )[0, 0]
        gates[key] = gate

    gate_delta_vmax = max(
        finite_percentile(
            torch.cat(
                [(gates[key] - gates["full"]).abs().reshape(-1) for key, *_ in VARIANTS[1:]]
            ).numpy(),
            99,
            1e-6,
        ),
        1e-6,
    )
    error_delta_vmax = max(
        finite_percentile(
            torch.cat(
                [(errors[key] - errors["full"]).abs().reshape(-1) for key, *_ in VARIANTS[1:]]
            ).numpy(),
            99,
            1e-6,
        ),
        1e-6,
    )
    error_vmax = max(
        finite_percentile(
            torch.cat([item.reshape(-1) for item in errors.values()]).numpy(), 95, 1.0
        ),
        1e-6,
    )

    gate_delta_cmap = plt.get_cmap("coolwarm")
    error_delta_cmap = plt.get_cmap("coolwarm").copy()
    error_delta_cmap.set_bad(color="#ffffff")
    fig, axes = plt.subplots(4, 4, figsize=(14.2, 13.6), constrained_layout=True)
    fig.suptitle(
        f"{dataset_name}: {data['sample_id']} | BRG statistic ablation effect",
        fontsize=15,
    )
    top_panels = (
        (data["image"], "Low-light RGB", None, None, None),
        (data["event"], "Event activity", "magma", None, None),
        (gt_depth, "GT depth", "viridis", depth_vmin, depth_vmax),
        (aligned["full"], "Full statistics depth", "viridis", depth_vmin, depth_vmax),
    )
    for axis, (image, title, cmap, vmin, vmax) in zip(axes[0], top_panels):
        if cmap is None:
            axis.imshow(np.clip(chw_to_hwc(image), 0.0, 1.0))
        else:
            shown = axis.imshow(image.numpy(), cmap=cmap, vmin=vmin, vmax=vmax)
            fig.colorbar(shown, ax=axis, fraction=0.046, pad=0.02)
        axis.set_title(title, fontsize=11)
        axis.axis("off")

    for row, (key, label, _mask) in enumerate(VARIANTS[1:], start=1):
        gate_delta = gates[key] - gates["full"]
        error_delta = errors[key] - errors["full"]
        panels = (
            (
                gate_delta,
                "Pre-occupancy gate\nchange vs Full",
                gate_delta_cmap,
                -gate_delta_vmax,
                gate_delta_vmax,
            ),
            (error_delta, "Absolute-error change vs Full", error_delta_cmap, -error_delta_vmax, error_delta_vmax),
            (aligned[key], "Median-aligned depth", "viridis", depth_vmin, depth_vmax),
            (errors[key], "Absolute depth error", "inferno", 0.0, error_vmax),
        )
        for axis, (image, title, cmap, vmin, vmax) in zip(axes[row], panels):
            shown = axis.imshow(image.numpy(), cmap=cmap, vmin=vmin, vmax=vmax)
            fig.colorbar(shown, ax=axis, fraction=0.046, pad=0.02)
            axis.set_title(title, fontsize=10)
            axis.axis("off")
        axes[row, 0].set_ylabel(label, fontsize=12, rotation=90, labelpad=24)

    stem = f"{dataset_name.lower()}_{data['sample_id'].replace('/', '_').replace(':', '_')}_compact"
    for suffix in ("png", "pdf"):
        fig.savefig(output_dir / f"{stem}.{suffix}", dpi=240 if suffix == "png" else None, bbox_inches="tight")
    plt.close(fig)
    return stem


def metric_value(payload, key):
    value = payload["dense_metrics"].get(key)
    return float(value) if value is not None and math.isfinite(float(value)) else float("nan")


def render_metric_heatmap(output_dir, results, dataset_names):
    labels = [item[1] for item in VARIANTS]
    keys = [item[0] for item in VARIANTS]
    matrices = []
    titles = []
    for metric_key, title, transform in (
        (
            "depth_median_aligned_absrel",
            "Median-aligned AbsRel change vs Full (%)\npositive = worse",
            lambda value, baseline: 100.0 * (value / baseline - 1.0),
        ),
        (
            "depth_median_aligned_delta_1.25",
            "Median-aligned delta1 change vs Full (pp)\npositive = better",
            lambda value, baseline: 100.0 * (value - baseline),
        ),
    ):
        matrix = np.full((len(keys), len(dataset_names)), np.nan, dtype=np.float32)
        annotations = [["" for _ in dataset_names] for _ in keys]
        for col, dataset_name in enumerate(dataset_names):
            baseline = metric_value(results[dataset_name]["full"], metric_key)
            for row, key in enumerate(keys):
                value = metric_value(results[dataset_name][key], metric_key)
                if math.isfinite(value) and math.isfinite(baseline) and abs(baseline) > 1e-12:
                    matrix[row, col] = transform(value, baseline)
                    annotations[row][col] = f"{value:.4f}\n{matrix[row, col]:+.1f}"
        matrices.append((matrix, annotations))
        titles.append(title)

    fig, axes = plt.subplots(1, 2, figsize=(12.5, 6.2), constrained_layout=True)
    for index, (axis, (matrix, annotations), title) in enumerate(zip(axes, matrices, titles)):
        extent = max(
            0.1,
            float(np.nanmax(np.abs(matrix))) if np.isfinite(matrix).any() else 0.1,
        )
        shown = axis.imshow(
            matrix,
            cmap="RdYlGn_r" if index == 0 else "RdYlGn",
            vmin=-extent,
            vmax=extent,
        )
        axis.set_xticks(range(len(dataset_names)), dataset_names)
        axis.set_yticks(range(len(labels)), labels)
        axis.set_title(title, fontsize=12)
        for row in range(matrix.shape[0]):
            for col in range(matrix.shape[1]):
                axis.text(col, row, annotations[row][col], ha="center", va="center", fontsize=10)
        fig.colorbar(shown, ax=axis, fraction=0.046, pad=0.04)
    fig.suptitle(
        "Final Night3R: test-time post-normalization leave-one-statistic-out sensitivity",
        fontsize=14,
    )
    for suffix in ("png", "pdf"):
        fig.savefig(
            output_dir / f"brg_stat_sensitivity_heatmap.{suffix}",
            dpi=240 if suffix == "png" else None,
            bbox_inches="tight",
        )
    plt.close(fig)


def write_metrics_csv(output_dir, results, dataset_names):
    path = output_dir / "brg_stat_sensitivity_metrics.csv"
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=(
                "dataset",
                "variant",
                "num_batches",
                "gate_mean",
                "median_absrel",
                "median_delta1",
                "raw_absrel",
                "raw_delta1",
            ),
        )
        writer.writeheader()
        for dataset_name in dataset_names:
            for key, label, _mask in VARIANTS:
                payload = results[dataset_name][key]
                dense = payload["dense_metrics"]
                writer.writerow(
                    {
                        "dataset": dataset_name,
                        "variant": label,
                        "num_batches": payload["num_batches"],
                        "gate_mean": payload["diagnostics"].get("gate_mean"),
                        "median_absrel": dense.get("depth_median_aligned_absrel"),
                        "median_delta1": dense.get("depth_median_aligned_delta_1.25"),
                        "raw_absrel": dense.get("depth_raw_absrel"),
                        "raw_delta1": dense.get("depth_raw_delta_1.25"),
                    }
                )
    return path


def main():
    args = parse_args()
    args.checkpoint = args.checkpoint.resolve()
    args.model_name_or_path = args.model_name_or_path.resolve()
    if not args.checkpoint.is_file():
        raise FileNotFoundError(f"Missing checkpoint: {args.checkpoint}")
    if not args.model_name_or_path.is_file():
        raise FileNotFoundError(f"Missing backbone checkpoint: {args.model_name_or_path}")
    datasets = parse_datasets(args.dataset)
    for name, root in datasets:
        if not root.is_dir():
            raise FileNotFoundError(f"Missing {name} root: {root}")
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    config = evaluation_config(load_config(args.config))
    device = resolve_device(args.device)
    model, load_info = build_model(args, config, device)
    render_panel = render_compact_ablation_panel if args.paper_compact else render_visual_panel

    results = {}
    selections = {}
    if args.render_only:
        if not args.visual_batch_index:
            raise ValueError("--render-only requires at least one --visual-batch-index")
        requested = sorted(set(args.visual_batch_index))
        if requested[0] < 0:
            raise ValueError("--visual-batch-index must be non-negative")
        if requested[-1] >= args.max_batches:
            raise ValueError("--max-batches must exceed every --visual-batch-index")

        visual_outputs = {}
        rendered_selections = {}
        for dataset_name, data_root in datasets:
            _train_args, _dataset, loader = make_loader(args, config, data_root)
            remaining = set(requested)
            selected = []
            for batch_idx, batch in enumerate(loader):
                if batch_idx > requested[-1] or not remaining:
                    break
                if batch_idx in remaining:
                    selected.append((batch_sample_id(batch, batch_idx), batch_idx))
                    remaining.remove(batch_idx)
            if remaining:
                raise RuntimeError(
                    f"Requested batch indices are outside {dataset_name} loader: {sorted(remaining)}"
                )
            records = collect_visuals(model, args, config, device, data_root, selected)
            visual_outputs[dataset_name] = [
                render_panel(output_dir, dataset_name, record)
                for record in records.values()
            ]
            rendered_selections[dataset_name] = selected

        manifest = {
            "render_mode": "qualitative_only",
            "checkpoint": str(args.checkpoint),
            "config": str(Path(args.config).resolve()),
            "device": str(device),
            "visual_selections": rendered_selections,
            "visual_outputs": visual_outputs,
            "gate_map_note": (
                "Gate maps are visualization-only pre-occupancy gates; model inference and "
                "all metrics retain the original occupancy hard mask."
            ),
        }
        with (output_dir / "manifest.json").open("w") as handle:
            json.dump(manifest, handle, indent=2)
        print(json.dumps({"output_dir": str(output_dir), "visual_outputs": visual_outputs}, indent=2))
        return

    for dataset_name, data_root in datasets:
        dataset_results = {}
        for variant_key, _label, mask_values in VARIANTS:
            dataset_results[variant_key] = evaluate_variant(
                model, args, config, device, data_root, mask_values
            )
        results[dataset_name] = dataset_results
        selections[dataset_name] = select_visual_samples(
            results, dataset_name, args.visual_samples
        )

    visual_outputs = {}
    for dataset_name, data_root in datasets:
        records = collect_visuals(
            model,
            args,
            config,
            device,
            data_root,
            selections[dataset_name],
        )
        visual_outputs[dataset_name] = [
            render_panel(output_dir, dataset_name, record)
            for record in records.values()
        ]

    dataset_names = [name for name, _root in datasets]
    render_metric_heatmap(output_dir, results, dataset_names)
    csv_path = write_metrics_csv(output_dir, results, dataset_names)
    manifest = {
        "claim": "Test-time post-normalization leave-one-statistic-out sensitivity; not retrained ablation.",
        "checkpoint": str(args.checkpoint),
        "config": str(Path(args.config).resolve()),
        "backbone": str(args.model_name_or_path),
        "load_info": load_info,
        "device": str(device),
        "max_batches": args.max_batches,
        "seed": args.seed,
        "stat_order": ["density", "occupancy", "polarity_balance"],
        "occupancy_safety_mask_preserved": True,
        "variants": [
            {"key": key, "label": label, "post_norm_mask": list(mask)}
            for key, label, mask in VARIANTS
        ],
        "results": results,
        "visual_selections": selections,
        "visual_outputs": visual_outputs,
        "metrics_csv": str(csv_path),
    }
    with (output_dir / "manifest.json").open("w") as handle:
        json.dump(manifest, handle, indent=2)
    print(
        json.dumps(
            {
                "output_dir": str(output_dir),
                "heatmap": str(output_dir / "brg_stat_sensitivity_heatmap.png"),
                "metrics_csv": str(csv_path),
                "visual_outputs": visual_outputs,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
