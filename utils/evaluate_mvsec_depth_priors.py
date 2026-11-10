#!/usr/bin/env python3
"""Evaluate simple depth priors on MVSEC-format event/RGB datasets.

This script is intentionally model-free. It checks whether MVSEC depth metrics
can be explained by dataset/layout priors such as "center is far, lower image is
near" without using RGB or event inputs.
"""

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from train_event_dark3r import build_dataset, load_config  # noqa: E402
from utils.compare_event_dark3r_checkpoint import DenseMetricAccumulator, as_bchw  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser("Evaluate MVSEC depth prior sanity-check baselines")
    parser.add_argument("--config", default="config/train_night3r_oldtest_recovery_from_best_currenttest_train.yaml")
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--image-size", type=int, nargs=2, default=None, metavar=("H", "W"))
    parser.add_argument("--max-scenes", type=int, default=None)
    parser.add_argument("--max-pairs-per-scene", type=int, default=None)
    parser.add_argument("--max-batches", type=int, default=128)
    parser.add_argument("--pair-stride", type=int, default=1)
    parser.add_argument("--pair-sampling", choices=("fixed", "random", "precomputed"), default="fixed")
    parser.add_argument("--min-pair-distance", type=int, default=1)
    parser.add_argument("--max-pair-distance", type=int, default=1)
    parser.add_argument("--precomputed-pairs", type=str, default=None)
    parser.add_argument("--allow-reverse-pairs", action="store_true")
    parser.add_argument("--depth-min", type=float, default=None)
    parser.add_argument("--depth-max", type=float, default=None)
    parser.add_argument(
        "--fit-fraction",
        type=float,
        default=0.5,
        help="Per-scene prefix fraction used for fit-prefix/test-suffix prior.",
    )
    parser.add_argument(
        "--poly-degree",
        type=int,
        default=2,
        choices=(1, 2),
        help="Polynomial degree for xy inverse-depth prior.",
    )
    parser.add_argument(
        "--max-fit-pixels-per-depth",
        type=int,
        default=4096,
        help="Subsample valid pixels per depth map when fitting xy polynomial.",
    )
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def make_train_args(args):
    return SimpleNamespace(
        config=args.config,
        data_root=args.data_root,
        save_dir=args.output_dir,
        model_name_or_path=None,
        image_size=args.image_size,
        batch_size=1,
        max_batches=args.max_batches,
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
        num_workers=0,
        stage=None,
        partial_finetune_blocks=None,
        amp=False,
        amp_dtype=None,
        gradient_checkpointing=False,
        device="cpu",
        log_every=1,
        no_save=True,
        seed=args.seed,
    )


def disable_lowlight(cfg):
    cfg = dict(cfg)
    lowlight_cfg = dict(cfg.get("lowlight_aug", {}))
    lowlight_cfg["enabled"] = False
    cfg["lowlight_aug"] = lowlight_cfg
    return cfg


def load_depths(dataset, max_batches):
    records = []
    samples = dataset.samples
    if max_batches is not None:
        samples = samples[: int(max_batches)]
    for sample_index, item in enumerate(samples):
        frames = dataset.scene_frames[item["scene_dir"]]
        idx1 = int(item["idx1"])
        if not frames.get("depths"):
            continue
        depth_path = frames["depths"][idx1]
        depth = dataset._load_depth_gt(depth_path)
        valid = torch.isfinite(depth) & (depth > 0)
        records.append(
            {
                "sample_index": sample_index,
                "scene": str(item["scene"]),
                "idx1": idx1,
                "idx2": int(item["idx2"]),
                "sample_id": f"{item['scene']}:{idx1:06d}-{int(item['idx2']):06d}",
                "depth": depth.contiguous(),
                "valid": valid.contiguous(),
                "depth_path": str(depth_path),
            }
        )
    if not records:
        raise RuntimeError(f"No depth_gt records found under {dataset.root_paths_text}")
    return records


def valid_mask(record, depth_min=None, depth_max=None):
    depth = record["depth"]
    mask = record["valid"].clone()
    if depth_min is not None:
        mask &= depth >= float(depth_min)
    if depth_max is not None:
        mask &= depth <= float(depth_max)
    return mask


def stack_depth_maps(records, depth_min=None, depth_max=None):
    depths = []
    masks = []
    for record in records:
        depth = record["depth"].float()
        mask = valid_mask(record, depth_min, depth_max)
        depths.append(torch.where(mask, depth, torch.full_like(depth, float("nan"))))
        masks.append(mask)
    return torch.stack(depths, dim=0), torch.stack(masks, dim=0)


def nanmedian_stack(values):
    finite = torch.isfinite(values)
    replaced = torch.where(finite, values, torch.full_like(values, float("nan")))
    try:
        return torch.nanmedian(replaced, dim=0).values
    except AttributeError:
        # Compatibility for older PyTorch: sort finite values by replacing NaNs
        # with +inf, then pick the middle finite value per pixel.
        inf = torch.full_like(replaced, float("inf"))
        sorted_values, _ = torch.sort(torch.where(finite, replaced, inf), dim=0)
        counts = finite.sum(dim=0)
        mid = torch.clamp((counts - 1) // 2, min=0)
        gathered = torch.gather(sorted_values, 0, mid.unsqueeze(0)).squeeze(0)
        return torch.where(counts > 0, gathered, torch.full_like(gathered, float("nan")))


def fill_invalid_template(template, fallback):
    template = template.float()
    if torch.isfinite(template).all() and (template > 0).all():
        return template
    return torch.where(torch.isfinite(template) & (template > 0), template, torch.full_like(template, float(fallback)))


def global_valid_median(records, depth_min=None, depth_max=None):
    values = []
    for record in records:
        mask = valid_mask(record, depth_min, depth_max)
        if mask.any():
            values.append(record["depth"][mask].float())
    if not values:
        return 1.0
    return float(torch.cat(values).median())


def mean_depth_map(records, depth_min=None, depth_max=None, reducer="median"):
    depths, _ = stack_depth_maps(records, depth_min, depth_max)
    fallback = global_valid_median(records, depth_min, depth_max)
    if reducer == "mean":
        finite = torch.isfinite(depths)
        summed = torch.where(finite, depths, torch.zeros_like(depths)).sum(dim=0)
        counts = finite.sum(dim=0).clamp_min(1)
        template = summed / counts
    elif reducer == "median":
        template = nanmedian_stack(depths)
    else:
        raise ValueError(f"Unknown reducer: {reducer}")
    return fill_invalid_template(template, fallback)


def xy_features(height, width, degree=2, device="cpu"):
    ys, xs = torch.meshgrid(
        torch.linspace(-1.0, 1.0, height, device=device),
        torch.linspace(-1.0, 1.0, width, device=device),
        indexing="ij",
    )
    features = [torch.ones_like(xs), xs, ys]
    if degree >= 2:
        features.extend([xs * xs, ys * ys, xs * ys])
    return torch.stack(features, dim=-1)


def fit_xy_inverse_depth(records, degree=2, max_pixels_per_depth=4096, seed=0, depth_min=None, depth_max=None):
    rng = np.random.default_rng(int(seed))
    h, w = records[0]["depth"].shape[-2:]
    grid = xy_features(h, w, degree=degree).view(-1, 3 if degree == 1 else 6)
    x_parts = []
    y_parts = []
    for record in records:
        depth = record["depth"].float()
        mask = valid_mask(record, depth_min, depth_max).view(-1)
        valid_indices = torch.nonzero(mask, as_tuple=False).flatten()
        if valid_indices.numel() <= 0:
            continue
        if valid_indices.numel() > max_pixels_per_depth:
            choice = rng.choice(valid_indices.numel(), size=max_pixels_per_depth, replace=False)
            valid_indices = valid_indices[torch.from_numpy(choice).long()]
        target = 1.0 / depth.view(-1)[valid_indices].clamp_min(1e-6)
        x_parts.append(grid[valid_indices])
        y_parts.append(target)
    if not x_parts:
        fallback = global_valid_median(records, depth_min, depth_max)
        return torch.full((h, w), fallback, dtype=torch.float32), None
    x = torch.cat(x_parts, dim=0).float()
    y = torch.cat(y_parts, dim=0).float().unsqueeze(1)
    reg = 1e-5 * torch.eye(x.shape[1], dtype=torch.float32)
    coef = torch.linalg.solve(x.T @ x + reg, x.T @ y).squeeze(1)
    inv_depth = (grid.float() @ coef).view(h, w).clamp_min(1e-4)
    pred_depth = (1.0 / inv_depth).clamp(1e-3, 200.0)
    return pred_depth, coef


def evaluate_template(records, template, depth_min=None, depth_max=None):
    metrics = DenseMetricAccumulator()
    pred = template.float()[None, None]
    for record in records:
        gt = record["depth"].float()[None, None]
        mask = valid_mask(record, depth_min, depth_max)[None, None]
        metrics.add_depth(pred, gt, mask, variant="raw")
        metrics.add_depth(pred, gt, mask, variant="median_aligned")
    return metrics.finalize()


def evaluate_constant(records, value, depth_min=None, depth_max=None):
    h, w = records[0]["depth"].shape[-2:]
    template = torch.full((h, w), float(value), dtype=torch.float32)
    return evaluate_template(records, template, depth_min, depth_max), template


def split_prefix_suffix(records, fit_fraction):
    fit = []
    test = []
    by_scene = {}
    for record in records:
        by_scene.setdefault(record["scene"], []).append(record)
    for scene_records in by_scene.values():
        scene_records = sorted(scene_records, key=lambda item: (item["idx1"], item["idx2"], item["sample_index"]))
        n_fit = max(1, min(len(scene_records) - 1, int(round(len(scene_records) * float(fit_fraction)))))
        fit.extend(scene_records[:n_fit])
        test.extend(scene_records[n_fit:])
    return fit, test


def evaluate_leave_one_scene_out(records, depth_min=None, depth_max=None, reducer="median"):
    rows = []
    all_metrics = DenseMetricAccumulator()
    scenes = sorted({record["scene"] for record in records})
    for scene in scenes:
        fit_records = [record for record in records if record["scene"] != scene]
        test_records = [record for record in records if record["scene"] == scene]
        if not fit_records or not test_records:
            continue
        template = mean_depth_map(fit_records, depth_min, depth_max, reducer=reducer)
        metrics = DenseMetricAccumulator()
        pred = template[None, None]
        for record in test_records:
            gt = record["depth"].float()[None, None]
            mask = valid_mask(record, depth_min, depth_max)[None, None]
            metrics.add_depth(pred, gt, mask, variant="raw")
            metrics.add_depth(pred, gt, mask, variant="median_aligned")
            all_metrics.add_depth(pred, gt, mask, variant="raw")
            all_metrics.add_depth(pred, gt, mask, variant="median_aligned")
        row = {"scene": scene, "num_test": len(test_records), **metrics.finalize()}
        rows.append(row)
    return all_metrics.finalize(), rows


def evaluate_leave_one_scene_out_xy(records, degree=2, max_pixels_per_depth=4096, seed=0, depth_min=None, depth_max=None):
    rows = []
    all_metrics = DenseMetricAccumulator()
    scenes = sorted({record["scene"] for record in records})
    for scene_index, scene in enumerate(scenes):
        fit_records = [record for record in records if record["scene"] != scene]
        test_records = [record for record in records if record["scene"] == scene]
        if not fit_records or not test_records:
            continue
        template, _ = fit_xy_inverse_depth(
            fit_records,
            degree=degree,
            max_pixels_per_depth=max_pixels_per_depth,
            seed=seed + scene_index,
            depth_min=depth_min,
            depth_max=depth_max,
        )
        metrics = DenseMetricAccumulator()
        pred = template[None, None]
        for record in test_records:
            gt = record["depth"].float()[None, None]
            mask = valid_mask(record, depth_min, depth_max)[None, None]
            metrics.add_depth(pred, gt, mask, variant="raw")
            metrics.add_depth(pred, gt, mask, variant="median_aligned")
            all_metrics.add_depth(pred, gt, mask, variant="raw")
            all_metrics.add_depth(pred, gt, mask, variant="median_aligned")
        rows.append({"scene": scene, "num_test": len(test_records), **metrics.finalize()})
    return all_metrics.finalize(), rows


def metric_subset(metrics):
    keys = [
        "depth_raw_absrel",
        "depth_raw_rmse_log",
        "depth_raw_delta_1.25",
        "depth_median_aligned_absrel",
        "depth_median_aligned_rmse_log",
        "depth_median_aligned_delta_1.25",
        "depth_raw_valid_px",
        "depth_raw_coverage",
    ]
    return {key: metrics.get(key, float("nan")) for key in keys}


def write_csv(path, rows):
    if not rows:
        return
    fieldnames = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def save_template_png(path, template):
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return
    arr = template.detach().cpu().float().numpy()
    finite = np.isfinite(arr) & (arr > 0)
    if not finite.any():
        return
    vmax = float(np.percentile(arr[finite], 95))
    plt.figure(figsize=(6, 4))
    plt.imshow(arr, cmap="magma_r", vmin=0.0, vmax=vmax)
    plt.colorbar(label="depth")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()


def make_markdown(rows):
    lines = [
        "| Prior | Fit/Eval split | Raw AbsRel ↓ | δ1 ↑ | RMSE log ↓ | Median AbsRel ↓ | Num samples |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        metrics = row["metrics"]
        lines.append(
            "| {method} | {split} | {absrel:.4f} | {delta:.4f} | {rmse_log:.4f} | {med_absrel:.4f} | {num_samples} |".format(
                method=row["method"],
                split=row["split"],
                absrel=float(metrics.get("depth_raw_absrel", float("nan"))),
                delta=float(metrics.get("depth_raw_delta_1.25", float("nan"))),
                rmse_log=float(metrics.get("depth_raw_rmse_log", float("nan"))),
                med_absrel=float(metrics.get("depth_median_aligned_absrel", float("nan"))),
                num_samples=row["num_samples"],
            )
        )
    return "\n".join(lines) + "\n"


def main():
    args = parse_args()
    torch.manual_seed(int(args.seed))
    np.random.seed(int(args.seed))
    cfg = disable_lowlight(load_config(args.config))
    train_args = make_train_args(args)
    dataset = build_dataset(train_args, cfg)
    records = load_depths(dataset, args.max_batches)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    h, w = records[0]["depth"].shape[-2:]
    summary_rows = []
    artifacts = {
        "data_root": getattr(dataset, "root_paths_text", str(dataset.root_path)),
        "dataset_len": len(dataset),
        "num_eval_records": len(records),
        "image_size": [h, w],
        "max_batches": args.max_batches,
        "max_pairs_per_scene": args.max_pairs_per_scene,
        "pair_stride": args.pair_stride,
        "depth_min": args.depth_min,
        "depth_max": args.depth_max,
        "methods": {},
    }

    global_median = global_valid_median(records, args.depth_min, args.depth_max)
    metrics, template = evaluate_constant(records, global_median, args.depth_min, args.depth_max)
    summary_rows.append(
        {
            "method": "constant_global_median",
            "split": "fit=eval",
            "num_samples": len(records),
            "metrics": metrics,
        }
    )
    artifacts["methods"]["constant_global_median"] = {"value": global_median, "metrics": metrics}
    save_template_png(output_dir / "constant_global_median.png", template)

    median_template = mean_depth_map(records, args.depth_min, args.depth_max, reducer="median")
    metrics = evaluate_template(records, median_template, args.depth_min, args.depth_max)
    summary_rows.append(
        {
            "method": "median_depth_map",
            "split": "fit=eval",
            "num_samples": len(records),
            "metrics": metrics,
        }
    )
    artifacts["methods"]["median_depth_map_fit_eval"] = {"metrics": metrics}
    save_template_png(output_dir / "median_depth_map_fit_eval.png", median_template)

    mean_template = mean_depth_map(records, args.depth_min, args.depth_max, reducer="mean")
    metrics = evaluate_template(records, mean_template, args.depth_min, args.depth_max)
    summary_rows.append(
        {
            "method": "mean_depth_map",
            "split": "fit=eval",
            "num_samples": len(records),
            "metrics": metrics,
        }
    )
    artifacts["methods"]["mean_depth_map_fit_eval"] = {"metrics": metrics}
    save_template_png(output_dir / "mean_depth_map_fit_eval.png", mean_template)

    xy_template, xy_coef = fit_xy_inverse_depth(
        records,
        degree=args.poly_degree,
        max_pixels_per_depth=args.max_fit_pixels_per_depth,
        seed=args.seed,
        depth_min=args.depth_min,
        depth_max=args.depth_max,
    )
    metrics = evaluate_template(records, xy_template, args.depth_min, args.depth_max)
    summary_rows.append(
        {
            "method": f"xy_poly{args.poly_degree}_inv_depth",
            "split": "fit=eval",
            "num_samples": len(records),
            "metrics": metrics,
        }
    )
    artifacts["methods"][f"xy_poly{args.poly_degree}_fit_eval"] = {
        "coef": None if xy_coef is None else [float(x) for x in xy_coef.detach().cpu()],
        "metrics": metrics,
    }
    save_template_png(output_dir / f"xy_poly{args.poly_degree}_fit_eval.png", xy_template)

    fit_records, test_records = split_prefix_suffix(records, args.fit_fraction)
    if fit_records and test_records:
        prefix_template = mean_depth_map(fit_records, args.depth_min, args.depth_max, reducer="median")
        metrics = evaluate_template(test_records, prefix_template, args.depth_min, args.depth_max)
        summary_rows.append(
            {
                "method": "median_depth_map",
                "split": f"scene_prefix_{args.fit_fraction:.2f}->suffix",
                "num_samples": len(test_records),
                "metrics": metrics,
            }
        )
        artifacts["methods"]["median_depth_map_prefix_suffix"] = {
            "num_fit": len(fit_records),
            "num_test": len(test_records),
            "metrics": metrics,
        }
        save_template_png(output_dir / "median_depth_map_prefix_suffix.png", prefix_template)

    loso_metrics, loso_rows = evaluate_leave_one_scene_out(
        records,
        depth_min=args.depth_min,
        depth_max=args.depth_max,
        reducer="median",
    )
    if loso_metrics:
        summary_rows.append(
            {
                "method": "median_depth_map",
                "split": "leave_one_scene_out",
                "num_samples": len(records),
                "metrics": loso_metrics,
            }
        )
        artifacts["methods"]["median_depth_map_loso"] = {"metrics": loso_metrics}
        write_csv(output_dir / "per_scene_loso_median_depth_map.csv", loso_rows)

    loso_xy_metrics, loso_xy_rows = evaluate_leave_one_scene_out_xy(
        records,
        degree=args.poly_degree,
        max_pixels_per_depth=args.max_fit_pixels_per_depth,
        seed=args.seed,
        depth_min=args.depth_min,
        depth_max=args.depth_max,
    )
    if loso_xy_metrics:
        summary_rows.append(
            {
                "method": f"xy_poly{args.poly_degree}_inv_depth",
                "split": "leave_one_scene_out",
                "num_samples": len(records),
                "metrics": loso_xy_metrics,
            }
        )
        artifacts["methods"][f"xy_poly{args.poly_degree}_loso"] = {"metrics": loso_xy_metrics}
        write_csv(output_dir / f"per_scene_loso_xy_poly{args.poly_degree}.csv", loso_xy_rows)

    flat_rows = []
    for row in summary_rows:
        flat = {
            "method": row["method"],
            "split": row["split"],
            "num_samples": row["num_samples"],
        }
        flat.update(metric_subset(row["metrics"]))
        flat_rows.append(flat)
    write_csv(output_dir / "summary.csv", flat_rows)
    (output_dir / "summary.md").write_text(make_markdown(summary_rows))
    (output_dir / "summary.json").write_text(json.dumps(artifacts, indent=2, sort_keys=True))
    print(make_markdown(summary_rows))
    print(f"Wrote {output_dir}")


if __name__ == "__main__":
    main()
