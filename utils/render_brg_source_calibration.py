#!/usr/bin/env python3
"""Audit source-specific BRG calibration in a trained Night3R checkpoint."""

import argparse
import copy
import csv
import json
import math
import os
import sys
from contextlib import contextmanager
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from model.event_dark3r_student import SOURCE_NAME_TO_ID  # noqa: E402
from model.event_fusion import compute_event_stats  # noqa: E402
from utils.render_brg_stat_sensitivity import (  # noqa: E402
    DEFAULT_BACKBONE,
    DEFAULT_CHECKPOINT,
    evaluation_config,
    evaluate_variant,
    make_loader,
    resolve_device,
    build_model,
)
from train_event_dark3r import load_config  # noqa: E402


DEFAULT_SOURCE_ROOTS = (
    ("MVSEC", Path("/data/fcr/data/train_data/MVSEC")),
    ("DSEC", REPO_ROOT / "outputs/splits/night3r_dsec_holdout_v1/dsec_train"),
    ("DL3DV", Path("/data/fcr/data/train_data/DL3DV")),
)
DEFAULT_EVAL_ROOTS = (
    ("DSEC", Path("/data/fcr/data/train_data/DSEC_test")),
    ("MVSEC", Path("/data/fcr/data/train_data/MVSEC_outdoor_night2_test")),
)


def parse_args():
    parser = argparse.ArgumentParser("Audit source calibration in a BRG checkpoint.")
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument(
        "--config", default="config/train_night3r_global_final_continue_200_20260723.yaml"
    )
    parser.add_argument("--model-name-or-path", type=Path, default=DEFAULT_BACKBONE)
    parser.add_argument("--allow-unsafe-checkpoint-load", action="store_true")
    parser.add_argument("--source-root", action="append", default=None, metavar="NAME=ROOT")
    parser.add_argument("--eval-root", action="append", default=None, metavar="NAME=ROOT")
    parser.add_argument("--source-batches", type=int, default=128)
    parser.add_argument("--eval-batches", type=int, default=128)
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
        default=REPO_ROOT / "outputs/brg_source_calibration_step100_20260726",
    )
    return parser.parse_args()


def parse_roots(specs, defaults):
    if not specs:
        return defaults
    parsed = []
    for spec in specs:
        if "=" not in spec:
            raise ValueError(f"Expected NAME=ROOT, got {spec!r}")
        name, root = spec.split("=", 1)
        name, root = name.strip(), Path(root.strip())
        if not name:
            raise ValueError(f"Expected NAME=ROOT, got {spec!r}")
        parsed.append((name, root))
    return tuple(parsed)


def source_id(name):
    key = name.lower()
    if key not in SOURCE_NAME_TO_ID:
        raise ValueError(f"Unsupported source name: {name}")
    return SOURCE_NAME_TO_ID[key]


def extract_source_parameters(model):
    fusion = model.fusion_adapter
    norm = fusion.stat_norm
    output = {
        "source_names": ["MVSEC", "DSEC", "DL3DV"],
        "stat_order": ["density", "occupancy", "polarity_balance"],
        "global_mean": norm.mean.detach().float().cpu().reshape(-1).tolist(),
        "global_std": norm.std.detach().float().cpu().reshape(-1).tolist(),
        "source_mean": norm.source_mean.detach().float().cpu().squeeze(1).tolist(),
        "source_std": norm.source_std.detach().float().cpu().squeeze(1).tolist(),
        "learned_affine": norm.affine.detach().float().cpu().tolist(),
        "learned_bias": norm.bias.detach().float().cpu().tolist(),
        "source_bias": (
            fusion.source_bias.weight.detach().float().cpu().reshape(-1).tolist()
            if fusion.source_bias is not None
            else None
        ),
    }
    output["source_stat_norm_is_identity"] = bool(
        torch.equal(norm.source_mean, norm.mean.expand_as(norm.source_mean))
        and torch.equal(norm.source_std, norm.std.expand_as(norm.source_std))
    )
    return output


def event_views(event):
    if event is None:
        return []
    if event.ndim == 5:
        return [event[:, index] for index in range(event.shape[1])]
    if event.ndim == 4:
        return [event]
    raise ValueError(f"Unsupported event voxel shape: {tuple(event.shape)}")


def image_hw(batch, event):
    images = batch.get("images_photo_dark", batch.get("images_photo", batch.get("images")))
    if torch.is_tensor(images):
        return int(images.shape[-2]), int(images.shape[-1])
    return int(event.shape[-2]), int(event.shape[-1])


def source_normalize(stat, params, source_index):
    mean = torch.tensor(params["source_mean"][source_index], dtype=torch.float32)
    std = torch.tensor(params["source_std"][source_index], dtype=torch.float32)
    affine = torch.tensor(params["learned_affine"], dtype=torch.float32)
    bias = torch.tensor(params["learned_bias"], dtype=torch.float32)
    z_score = torch.clamp((stat - mean) / (std + 1.0e-4), -5.0, 5.0)
    return z_score * affine + bias


def summarize(values):
    values = np.asarray(values, dtype=np.float32)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return {"count": 0}
    return {
        "count": int(values.size),
        "mean": float(values.mean()),
        "std": float(values.std()),
        "p05": float(np.percentile(values, 5)),
        "p50": float(np.percentile(values, 50)),
        "p95": float(np.percentile(values, 95)),
    }


def collect_source_statistics(args, config, roots, params):
    output = {}
    raw_values = {}
    normalized_values = {}
    for name, root in roots:
        train_args, dataset, _loader = make_loader(args, config, root)
        del train_args
        sample_count = min(int(args.source_batches) * int(args.batch_size), len(dataset))
        generator = np.random.default_rng(int(args.seed) + source_id(name))
        indices = generator.choice(len(dataset), size=sample_count, replace=False).tolist()
        loader = DataLoader(
            Subset(dataset, indices),
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=False,
        )
        index = source_id(name)
        raw = [[] for _ in range(3)]
        normalized = [[] for _ in range(3)]
        num_pairs = 0
        for batch_index, batch in enumerate(loader):
            event = batch.get("event_voxel")
            h, w = image_hw(batch, event)
            target_hw = (max(h // 16, 1), max(w // 16, 1))
            for view in event_views(event):
                stat = compute_event_stats(view.float(), target_hw=target_hw).cpu()
                normalized_stat = source_normalize(stat, params, index)
                for stat_index in range(3):
                    raw[stat_index].append(stat[..., stat_index].reshape(-1).numpy())
                    normalized[stat_index].append(normalized_stat[..., stat_index].reshape(-1).numpy())
            num_pairs += int(event.shape[0])
        raw_values[name] = [np.concatenate(values) if values else np.empty(0) for values in raw]
        normalized_values[name] = [
            np.concatenate(values) if values else np.empty(0) for values in normalized
        ]
        output[name] = {
            "root": str(root),
            "dataset_len": len(dataset),
            "pairs_sampled": num_pairs,
            "raw": {
                label: summarize(raw_values[name][stat_index])
                for stat_index, label in enumerate(params["stat_order"])
            },
            "source_normalized": {
                label: summarize(normalized_values[name][stat_index])
                for stat_index, label in enumerate(params["stat_order"])
            },
        }
    return output, raw_values, normalized_values


def finite_range(values):
    values = [array[np.isfinite(array)] for array in values]
    values = [array for array in values if array.size]
    if not values:
        return 0.0, 1.0
    merged = np.concatenate(values)
    lo, hi = np.percentile(merged, [1, 99])
    if hi <= lo:
        hi = lo + 1.0e-6
    return float(lo), float(hi)


def render_distributions(output_dir, roots, raw_values, normalized_values, params):
    names = [name for name, _ in roots]
    colors = {"MVSEC": "#2166ac", "DSEC": "#b2182b", "DL3DV": "#4d9221"}
    fig, axes = plt.subplots(3, 2, figsize=(14, 11), constrained_layout=True)
    for stat_index, label in enumerate(params["stat_order"]):
        for col, (title, values_by_source) in enumerate(
            (("Raw token statistic", raw_values), ("Current source-normalized statistic", normalized_values))
        ):
            axis = axes[stat_index, col]
            lo, hi = finite_range([values_by_source[name][stat_index] for name in names])
            bins = np.linspace(lo, hi, 90)
            for name in names:
                values = values_by_source[name][stat_index]
                values = values[np.isfinite(values)]
                if values.size:
                    axis.hist(
                        values,
                        bins=bins,
                        density=True,
                        histtype="step",
                        linewidth=1.7,
                        color=colors.get(name, None),
                        label=name,
                    )
            axis.set_title(f"{label}: {title}", fontsize=11)
            axis.set_xlabel(label)
            axis.set_ylabel("Token density")
            axis.grid(alpha=0.2)
            if stat_index == 0 and col == 1:
                axis.legend(frameon=False)
    source_bias = params["source_bias"]
    bias_text = "none" if source_bias is None else ", ".join(
        f"{name}={value:+.3f}" for name, value in zip(params["source_names"], source_bias)
    )
    fig.suptitle(
        "Final Night3R BRG source-stat audit\n"
        f"source-stat identity={params['source_stat_norm_is_identity']}; learned gate bias: {bias_text}",
        fontsize=14,
    )
    for suffix in ("png", "pdf"):
        fig.savefig(
            output_dir / f"brg_source_stat_distributions.{suffix}",
            dpi=220 if suffix == "png" else None,
            bbox_inches="tight",
        )
    plt.close(fig)


@contextmanager
def zero_source_bias(model):
    source_bias = model.fusion_adapter.source_bias
    if source_bias is None:
        yield
        return
    original = source_bias.weight.detach().clone()
    try:
        with torch.no_grad():
            source_bias.weight.zero_()
        yield
    finally:
        with torch.no_grad():
            source_bias.weight.copy_(original)


def evaluate_source_bias_modes(model, args, config, device, roots):
    results = {}
    full_mask = (1.0, 1.0, 1.0)
    for name, root in roots:
        full = evaluate_variant(model, args, config, device, root, full_mask)
        with zero_source_bias(model):
            no_bias = evaluate_variant(model, args, config, device, root, full_mask)
        results[name] = {"source_aware": full, "source_bias_zero": no_bias}
    return results


def metric(payload, key):
    value = payload["dense_metrics"].get(key)
    return float(value) if value is not None and math.isfinite(float(value)) else float("nan")


def gate_mean(payload):
    value = payload["diagnostics"].get("gate_mean")
    return float(value) if value is not None and math.isfinite(float(value)) else float("nan")


def render_source_bias_effect(output_dir, results, params):
    names = list(results)
    fig, axes = plt.subplots(1, 3, figsize=(15.5, 5.4), constrained_layout=True)
    panels = (
        ("gate", "Mean BRG gate", lambda payload: gate_mean(payload), "higher = more event injection"),
        (
            "absrel",
            "Median-aligned AbsRel",
            lambda payload: metric(payload, "depth_median_aligned_absrel"),
            "DSEC/MVSEC only; lower is better",
        ),
        (
            "delta1",
            "Median-aligned delta1",
            lambda payload: metric(payload, "depth_median_aligned_delta_1.25"),
            "DSEC/MVSEC only; higher is better",
        ),
    )
    positions = np.arange(len(names))
    width = 0.36
    for axis, (_key, title, getter, subtitle) in zip(axes, panels):
        full = [getter(results[name]["source_aware"]) for name in names]
        zero = [getter(results[name]["source_bias_zero"]) for name in names]
        axis.bar(positions - width / 2, full, width, label="Source-aware", color="#2166ac")
        axis.bar(positions + width / 2, zero, width, label="Source bias = 0", color="#bdbdbd")
        axis.set_title(f"{title}\n{subtitle}", fontsize=11)
        axis.set_xticks(positions, names)
        axis.grid(axis="y", alpha=0.2)
        for values, offset in ((full, -width / 2), (zero, width / 2)):
            for index, value in enumerate(values):
                if math.isfinite(value):
                    axis.text(index + offset, value, f"{value:.4f}", ha="center", va="bottom", fontsize=8)
        if axis is axes[0]:
            axis.legend(frameon=False)
    bias_text = "none" if params["source_bias"] is None else ", ".join(
        f"{name}={value:+.3f}" for name, value in zip(params["source_names"], params["source_bias"])
    )
    fig.suptitle(f"Final Night3R source-bias intervention | learned offsets: {bias_text}", fontsize=14)
    for suffix in ("png", "pdf"):
        fig.savefig(
            output_dir / f"brg_source_bias_effect.{suffix}",
            dpi=220 if suffix == "png" else None,
            bbox_inches="tight",
        )
    plt.close(fig)


def write_csv(output_dir, results):
    path = output_dir / "brg_source_bias_metrics.csv"
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=(
                "source",
                "mode",
                "num_batches",
                "gate_mean",
                "median_absrel",
                "median_delta1",
                "raw_absrel",
                "raw_delta1",
            ),
        )
        writer.writeheader()
        for source_name, modes in results.items():
            for mode, payload in modes.items():
                writer.writerow(
                    {
                        "source": source_name,
                        "mode": mode,
                        "num_batches": payload["num_batches"],
                        "gate_mean": gate_mean(payload),
                        "median_absrel": metric(payload, "depth_median_aligned_absrel"),
                        "median_delta1": metric(payload, "depth_median_aligned_delta_1.25"),
                        "raw_absrel": metric(payload, "depth_raw_absrel"),
                        "raw_delta1": metric(payload, "depth_raw_delta_1.25"),
                    }
                )
    return path


def main():
    args = parse_args()
    args.checkpoint = args.checkpoint.resolve()
    args.model_name_or_path = args.model_name_or_path.resolve()
    if not args.checkpoint.is_file() or not args.model_name_or_path.is_file():
        raise FileNotFoundError("Checkpoint or MASt3R backbone is missing")
    source_roots = parse_roots(args.source_root, DEFAULT_SOURCE_ROOTS)
    eval_roots = parse_roots(args.eval_root, DEFAULT_EVAL_ROOTS)
    for name, root in source_roots + eval_roots:
        if not root.is_dir():
            raise FileNotFoundError(f"Missing {name} root: {root}")
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    config = evaluation_config(load_config(args.config))
    device = resolve_device(args.device)
    # The shared model loader expects the evaluation-batch name used by the
    # statistic-sensitivity utility.
    args.max_batches = args.eval_batches
    model, load_info = build_model(args, config, device)
    params = extract_source_parameters(model)
    source_statistics, raw_values, normalized_values = collect_source_statistics(
        args, config, source_roots, params
    )
    render_distributions(output_dir, source_roots, raw_values, normalized_values, params)
    source_bias_results = evaluate_source_bias_modes(model, args, config, device, eval_roots)
    render_source_bias_effect(output_dir, source_bias_results, params)
    csv_path = write_csv(output_dir, source_bias_results)
    manifest = {
        "checkpoint": str(args.checkpoint),
        "config": str(Path(args.config).resolve()),
        "load_info": load_info,
        "device": str(device),
        "source_parameters": params,
        "source_statistics": source_statistics,
        "source_bias_intervention": {
            "description": "Sets the learned source gate-bias embedding to zero while retaining all other final checkpoint parameters.",
            "results": source_bias_results,
        },
        "source_stat_normalization_interpretation": (
            "The source mean/std buffers are identical to global mean/std, so source-stat normalization "
            "is an identity transform in this checkpoint."
            if params["source_stat_norm_is_identity"]
            else "Source mean/std buffers differ from the global statistics."
        ),
        "metrics_csv": str(csv_path),
    }
    with (output_dir / "manifest.json").open("w") as handle:
        json.dump(manifest, handle, indent=2)
    print(
        json.dumps(
            {
                "output_dir": str(output_dir),
                "distribution_figure": str(output_dir / "brg_source_stat_distributions.png"),
                "source_bias_figure": str(output_dir / "brg_source_bias_effect.png"),
                "metrics_csv": str(csv_path),
                "source_stat_norm_is_identity": params["source_stat_norm_is_identity"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
