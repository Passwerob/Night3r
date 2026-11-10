#!/usr/bin/env python3
"""Run low-light synthesis validity checks used by mamba.txt.

The script compares real night images against several darkening variants:

- sRGB multiplication, a deliberately weak baseline.
- Linear-domain exposure only.
- Linear-domain exposure with read noise.
- Linear-domain exposure with shot + read noise, matching the training aug.

It writes machine-readable metrics plus small visual artifacts. It does not
modify dataset files.
"""

import argparse
import csv
import json
import math
import random
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw


IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp"}
LUMA_WEIGHTS = np.array([0.2126, 0.7152, 0.0722], dtype=np.float32)


def parse_args():
    parser = argparse.ArgumentParser("Low-light data validity statistics")
    parser.add_argument(
        "--real-roots",
        nargs="+",
        default=["/data/fcr/data/train_data/MVSEC_outdoor_night2_test"],
        help="Real night roots with screen-*/images.",
    )
    parser.add_argument(
        "--source-roots",
        nargs="+",
        default=["/data/fcr/data/train_data/DL3DV"],
        help="Normal-light source roots used for synthetic darkening.",
    )
    parser.add_argument(
        "--output-dir",
        default="/data/fcr/data/Night3R/outputs/lowlight_validity_mamba_20260710",
    )
    parser.add_argument("--max-real", type=int, default=240)
    parser.add_argument("--max-source", type=int, default=240)
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--patch-size", type=int, default=16)
    parser.add_argument("--hist-bins", type=int, default=64)
    parser.add_argument("--grad-bins", type=int, default=64)
    parser.add_argument("--exposures", type=float, nargs="+", default=[0.2, 0.1, 0.05, 0.03])
    parser.add_argument("--gamma", type=float, default=2.2)
    parser.add_argument("--read-noise-std", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--contact-sheet-count", type=int, default=8)
    return parser.parse_args()


def discover_images(roots, max_images, seed):
    paths = []
    for root in roots:
        root_path = Path(root)
        if not root_path.exists():
            continue
        candidates = []
        for scene_dir in sorted(root_path.glob("screen-*")):
            image_dir = scene_dir / "images"
            if image_dir.is_dir():
                candidates.extend(
                    sorted(
                        path
                        for path in image_dir.iterdir()
                        if path.is_file() and path.suffix.lower() in IMAGE_EXTS
                    )
                )
        if not candidates:
            candidates = sorted(
                path
                for path in root_path.rglob("*")
                if path.is_file() and path.suffix.lower() in IMAGE_EXTS
            )
        paths.extend(candidates)
    rng = random.Random(seed)
    rng.shuffle(paths)
    if max_images is not None and max_images > 0:
        paths = paths[:max_images]
    return paths


def load_rgb(path, image_size):
    image = Image.open(path).convert("RGB")
    image.thumbnail((image_size, image_size), Image.Resampling.BILINEAR)
    canvas = Image.new("RGB", (image_size, image_size))
    left = (image_size - image.width) // 2
    top = (image_size - image.height) // 2
    canvas.paste(image, (left, top))
    return np.asarray(canvas, dtype=np.float32) / 255.0


def luminance(rgb):
    return np.tensordot(rgb, LUMA_WEIGHTS, axes=([-1], [0])).astype(np.float32)


def srgb_multiply(rgb, exposure):
    return np.clip(rgb * exposure, 0.0, 1.0)


def linear_exposure(rgb, exposure, gamma):
    linear = np.clip(rgb, 0.0, 1.0) ** gamma
    dark = linear * exposure
    return np.clip(dark, 0.0, 1.0) ** (1.0 / gamma)


def linear_noise(rgb, exposure, gamma, read_noise_std, shot_noise, rng):
    linear = np.clip(rgb, 0.0, 1.0) ** gamma
    dark = linear * exposure
    if shot_noise:
        shot_std = np.sqrt(np.maximum(dark, 0.0) + 1e-6) * read_noise_std
        dark = dark + rng.normal(0.0, shot_std, size=dark.shape).astype(np.float32)
    if read_noise_std > 0.0:
        dark = dark + rng.normal(0.0, read_noise_std, size=dark.shape).astype(np.float32)
    return np.clip(dark, 0.0, 1.0) ** (1.0 / gamma)


def grad_magnitude(gray):
    gx = np.diff(gray, axis=1, append=gray[:, -1:])
    gy = np.diff(gray, axis=0, append=gray[-1:, :])
    return np.sqrt(gx * gx + gy * gy).astype(np.float32)


def patch_mean_variance(gray, patch_size):
    h, w = gray.shape
    h = (h // patch_size) * patch_size
    w = (w // patch_size) * patch_size
    if h == 0 or w == 0:
        return np.zeros((0,), dtype=np.float32), np.zeros((0,), dtype=np.float32)
    patches = gray[:h, :w].reshape(h // patch_size, patch_size, w // patch_size, patch_size)
    patches = patches.transpose(0, 2, 1, 3).reshape(-1, patch_size * patch_size)
    return patches.mean(axis=1).astype(np.float32), patches.var(axis=1).astype(np.float32)


def normalized_hist(values, bins, value_range):
    hist, _ = np.histogram(values, bins=bins, range=value_range)
    hist = hist.astype(np.float64)
    total = hist.sum()
    if total > 0:
        hist /= total
    return hist


def summarize_group(name, image_paths, image_size, patch_size, hist_bins, grad_bins):
    luma_values = []
    grad_values = []
    patch_means = []
    patch_vars = []
    brightness = []
    for path in image_paths:
        rgb = load_rgb(path, image_size)
        gray = luminance(rgb)
        luma_values.append(gray.reshape(-1))
        grad_values.append(grad_magnitude(gray).reshape(-1))
        mean, var = patch_mean_variance(gray, patch_size)
        patch_means.append(mean)
        patch_vars.append(var)
        brightness.append(float(gray.mean()))
    if not luma_values:
        raise RuntimeError(f"No images available for group {name}")
    luma_all = np.concatenate(luma_values)
    grad_all = np.concatenate(grad_values)
    mean_all = np.concatenate(patch_means) if patch_means else np.zeros((0,), dtype=np.float32)
    var_all = np.concatenate(patch_vars) if patch_vars else np.zeros((0,), dtype=np.float32)
    return {
        "name": name,
        "num_images": len(image_paths),
        "luma_hist": normalized_hist(luma_all, hist_bins, (0.0, 1.0)),
        "grad_hist": normalized_hist(grad_all, grad_bins, (0.0, 0.5)),
        "patch_mean": mean_all,
        "patch_var": var_all,
        "mean_luma": float(luma_all.mean()),
        "median_luma": float(np.median(luma_all)),
        "p05_luma": float(np.quantile(luma_all, 0.05)),
        "p95_luma": float(np.quantile(luma_all, 0.95)),
        "dark005": float((luma_all < 0.05).mean()),
        "dark010": float((luma_all < 0.10).mean()),
        "grad_mean": float(grad_all.mean()),
        "grad_p95": float(np.quantile(grad_all, 0.95)),
        "image_mean_luma_std": float(np.std(brightness)),
    }


def summarize_synthetic(
    name,
    image_paths,
    image_size,
    patch_size,
    hist_bins,
    grad_bins,
    transform,
):
    luma_values = []
    grad_values = []
    patch_means = []
    patch_vars = []
    brightness = []
    for idx, path in enumerate(image_paths):
        rgb = load_rgb(path, image_size)
        synth = transform(rgb, idx)
        gray = luminance(synth)
        luma_values.append(gray.reshape(-1))
        grad_values.append(grad_magnitude(gray).reshape(-1))
        mean, var = patch_mean_variance(gray, patch_size)
        patch_means.append(mean)
        patch_vars.append(var)
        brightness.append(float(gray.mean()))
    if not luma_values:
        raise RuntimeError(f"No images available for group {name}")
    luma_all = np.concatenate(luma_values)
    grad_all = np.concatenate(grad_values)
    mean_all = np.concatenate(patch_means) if patch_means else np.zeros((0,), dtype=np.float32)
    var_all = np.concatenate(patch_vars) if patch_vars else np.zeros((0,), dtype=np.float32)
    return {
        "name": name,
        "num_images": len(image_paths),
        "luma_hist": normalized_hist(luma_all, hist_bins, (0.0, 1.0)),
        "grad_hist": normalized_hist(grad_all, grad_bins, (0.0, 0.5)),
        "patch_mean": mean_all,
        "patch_var": var_all,
        "mean_luma": float(luma_all.mean()),
        "median_luma": float(np.median(luma_all)),
        "p05_luma": float(np.quantile(luma_all, 0.05)),
        "p95_luma": float(np.quantile(luma_all, 0.95)),
        "dark005": float((luma_all < 0.05).mean()),
        "dark010": float((luma_all < 0.10).mean()),
        "grad_mean": float(grad_all.mean()),
        "grad_p95": float(np.quantile(grad_all, 0.95)),
        "image_mean_luma_std": float(np.std(brightness)),
    }


def mean_variance_curve(patch_mean, patch_var, bins=20):
    edges = np.linspace(0.0, 1.0, bins + 1)
    centers = (edges[:-1] + edges[1:]) * 0.5
    values = np.full((bins,), np.nan, dtype=np.float64)
    counts = np.zeros((bins,), dtype=np.int64)
    for idx in range(bins):
        mask = (patch_mean >= edges[idx]) & (patch_mean < edges[idx + 1])
        if idx == bins - 1:
            mask = (patch_mean >= edges[idx]) & (patch_mean <= edges[idx + 1])
        counts[idx] = int(mask.sum())
        if counts[idx] > 0:
            values[idx] = float(np.mean(patch_var[mask]))
    return centers, values, counts


def l1_hist(a, b):
    return float(0.5 * np.abs(a - b).sum())


def curve_l1(curve_a, curve_b):
    mask = np.isfinite(curve_a) & np.isfinite(curve_b)
    if not np.any(mask):
        return float("nan")
    return float(np.mean(np.abs(curve_a[mask] - curve_b[mask])))


def linear_fit_mean_variance(patch_mean, patch_var):
    mask = np.isfinite(patch_mean) & np.isfinite(patch_var)
    mask &= patch_mean >= 0.0
    if int(mask.sum()) < 4:
        return {"slope": float("nan"), "intercept": float("nan"), "r2": float("nan")}
    x = patch_mean[mask].astype(np.float64)
    y = patch_var[mask].astype(np.float64)
    a, b = np.polyfit(x, y, deg=1)
    pred = a * x + b
    ss_res = float(np.square(y - pred).sum())
    ss_tot = float(np.square(y - y.mean()).sum())
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0.0 else float("nan")
    return {"slope": float(a), "intercept": float(b), "r2": float(r2)}


def serializable_group(group, keep_arrays=False):
    output = {
        "name": group["name"],
        "num_images": group["num_images"],
        "mean_luma": group["mean_luma"],
        "median_luma": group["median_luma"],
        "p05_luma": group["p05_luma"],
        "p95_luma": group["p95_luma"],
        "dark005": group["dark005"],
        "dark010": group["dark010"],
        "grad_mean": group["grad_mean"],
        "grad_p95": group["grad_p95"],
        "image_mean_luma_std": group["image_mean_luma_std"],
        "mean_variance_fit": linear_fit_mean_variance(group["patch_mean"], group["patch_var"]),
    }
    if keep_arrays:
        output["luma_hist"] = group["luma_hist"].tolist()
        output["grad_hist"] = group["grad_hist"].tolist()
    return output


def write_csv(path, rows, fieldnames):
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def save_contact_sheet(output_path, real_paths, source_paths, args):
    n = min(args.contact_sheet_count, len(real_paths), len(source_paths))
    if n <= 0:
        return
    labels = [
        "real_night",
        "source",
        "srgb_x0.05",
        "linear_x0.05",
        "linear_read_x0.05",
        "linear_shot_read_x0.05",
    ]
    tile = args.image_size
    label_h = 22
    sheet = Image.new("RGB", (len(labels) * tile, n * (tile + label_h)), (255, 255, 255))
    draw = ImageDraw.Draw(sheet)
    rng = np.random.default_rng(args.seed)
    for row in range(n):
        real = load_rgb(real_paths[row], args.image_size)
        src = load_rgb(source_paths[row], args.image_size)
        variants = [
            real,
            src,
            srgb_multiply(src, 0.05),
            linear_exposure(src, 0.05, args.gamma),
            linear_noise(src, 0.05, args.gamma, args.read_noise_std, False, rng),
            linear_noise(src, 0.05, args.gamma, args.read_noise_std, True, rng),
        ]
        for col, (label, array) in enumerate(zip(labels, variants)):
            img = Image.fromarray((np.clip(array, 0.0, 1.0) * 255.0).astype(np.uint8))
            x = col * tile
            y = row * (tile + label_h)
            sheet.paste(img, (x, y + label_h))
            draw.text((x + 4, y + 4), label, fill=(0, 0, 0))
    sheet.save(output_path)


def try_write_plots(output_dir, groups, rows, hist_bins, mv_bins=20):
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:
        return f"matplotlib unavailable: {exc}"

    hist_x = np.linspace(0.0, 1.0, hist_bins, endpoint=False) + 0.5 / hist_bins
    plt.figure(figsize=(10, 5))
    for group in groups:
        if group["name"] in {"real_night", "source_clean"} or group["name"].endswith("x0.05"):
            plt.plot(hist_x, group["luma_hist"], label=group["name"])
    plt.xlabel("luminance")
    plt.ylabel("normalized frequency")
    plt.yscale("log")
    plt.grid(True, alpha=0.25)
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(output_dir / "luminance_histograms.png", dpi=180)
    plt.close()

    plt.figure(figsize=(10, 5))
    for group in groups:
        if group["name"] in {"real_night", "source_clean"} or group["name"].endswith("x0.05"):
            centers, curve, _ = mean_variance_curve(group["patch_mean"], group["patch_var"], bins=mv_bins)
            plt.plot(centers, curve, marker="o", markersize=3, label=group["name"])
    plt.xlabel("patch mean luminance")
    plt.ylabel("patch variance")
    plt.grid(True, alpha=0.25)
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(output_dir / "mean_variance_curves.png", dpi=180)
    plt.close()

    plt.figure(figsize=(10, 4))
    names = [row["variant"] for row in rows]
    scores = [row["combined_distance"] for row in rows]
    plt.bar(range(len(names)), scores)
    plt.xticks(range(len(names)), names, rotation=75, ha="right", fontsize=7)
    plt.ylabel("combined distance to real night")
    plt.tight_layout()
    plt.savefig(output_dir / "variant_distance_ranking.png", dpi=180)
    plt.close()
    return None


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    real_paths = discover_images(args.real_roots, args.max_real, args.seed)
    source_paths = discover_images(args.source_roots, args.max_source, args.seed + 1)
    if not real_paths:
        raise RuntimeError(f"No real-night images found under {args.real_roots}")
    if not source_paths:
        raise RuntimeError(f"No source images found under {args.source_roots}")

    rng_master = np.random.default_rng(args.seed)
    groups = [summarize_group("real_night", real_paths, args.image_size, args.patch_size, args.hist_bins, args.grad_bins)]
    source_clean = summarize_group(
        "source_clean",
        source_paths,
        args.image_size,
        args.patch_size,
        args.hist_bins,
        args.grad_bins,
    )
    groups.append(source_clean)

    for exposure in args.exposures:
        exp_tag = f"x{exposure:g}"
        groups.append(
            summarize_synthetic(
                f"srgb_multiply_{exp_tag}",
                source_paths,
                args.image_size,
                args.patch_size,
                args.hist_bins,
                args.grad_bins,
                lambda rgb, idx, e=exposure: srgb_multiply(rgb, e),
            )
        )
        groups.append(
            summarize_synthetic(
                f"linear_exposure_{exp_tag}",
                source_paths,
                args.image_size,
                args.patch_size,
                args.hist_bins,
                args.grad_bins,
                lambda rgb, idx, e=exposure: linear_exposure(rgb, e, args.gamma),
            )
        )
        groups.append(
            summarize_synthetic(
                f"linear_read_{exp_tag}",
                source_paths,
                args.image_size,
                args.patch_size,
                args.hist_bins,
                args.grad_bins,
                lambda rgb, idx, e=exposure: linear_noise(
                    rgb,
                    e,
                    args.gamma,
                    args.read_noise_std,
                    False,
                    np.random.default_rng(int(rng_master.integers(0, 2**31 - 1)) + idx),
                ),
            )
        )
        groups.append(
            summarize_synthetic(
                f"linear_shot_read_{exp_tag}",
                source_paths,
                args.image_size,
                args.patch_size,
                args.hist_bins,
                args.grad_bins,
                lambda rgb, idx, e=exposure: linear_noise(
                    rgb,
                    e,
                    args.gamma,
                    args.read_noise_std,
                    True,
                    np.random.default_rng(int(rng_master.integers(0, 2**31 - 1)) + idx),
                ),
            )
        )

    real = groups[0]
    _, real_mv_curve, _ = mean_variance_curve(real["patch_mean"], real["patch_var"])
    rows = []
    for group in groups[1:]:
        _, mv_curve, _ = mean_variance_curve(group["patch_mean"], group["patch_var"])
        luma_dist = l1_hist(group["luma_hist"], real["luma_hist"])
        grad_dist = l1_hist(group["grad_hist"], real["grad_hist"])
        mv_dist = curve_l1(mv_curve, real_mv_curve)
        dark_dist = 0.5 * (
            abs(group["dark005"] - real["dark005"]) + abs(group["dark010"] - real["dark010"])
        )
        mean_dist = abs(group["mean_luma"] - real["mean_luma"])
        combined = luma_dist + grad_dist + 20.0 * mv_dist + dark_dist + mean_dist
        rows.append(
            {
                "variant": group["name"],
                "num_images": group["num_images"],
                "mean_luma": group["mean_luma"],
                "median_luma": group["median_luma"],
                "dark005": group["dark005"],
                "dark010": group["dark010"],
                "grad_mean": group["grad_mean"],
                "luma_hist_l1": luma_dist,
                "grad_hist_l1": grad_dist,
                "mean_variance_l1": mv_dist,
                "dark_ratio_error": dark_dist,
                "mean_luma_error": mean_dist,
                "combined_distance": combined,
            }
        )
    rows.sort(key=lambda item: (math.inf if not np.isfinite(item["combined_distance"]) else item["combined_distance"]))

    write_csv(
        output_dir / "summary.csv",
        rows,
        [
            "variant",
            "num_images",
            "mean_luma",
            "median_luma",
            "dark005",
            "dark010",
            "grad_mean",
            "luma_hist_l1",
            "grad_hist_l1",
            "mean_variance_l1",
            "dark_ratio_error",
            "mean_luma_error",
            "combined_distance",
        ],
    )

    mv_rows = []
    for group in groups:
        centers, curve, counts = mean_variance_curve(group["patch_mean"], group["patch_var"])
        for center, value, count in zip(centers, curve, counts):
            mv_rows.append(
                {
                    "variant": group["name"],
                    "patch_mean_center": float(center),
                    "patch_var_mean": "" if not np.isfinite(value) else float(value),
                    "num_patches": int(count),
                }
            )
    write_csv(output_dir / "mean_variance_curves.csv", mv_rows, ["variant", "patch_mean_center", "patch_var_mean", "num_patches"])

    metrics = {
        "args": vars(args),
        "real_image_examples": [str(path) for path in real_paths[:10]],
        "source_image_examples": [str(path) for path in source_paths[:10]],
        "groups": [serializable_group(group, keep_arrays=True) for group in groups],
        "ranking": rows,
    }
    with open(output_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2, sort_keys=True)

    save_contact_sheet(output_dir / "contact_sheet_x005.png", real_paths, source_paths, args)
    plot_note = try_write_plots(output_dir, groups, rows, args.hist_bins)

    best = rows[0]
    md_lines = [
        "# Low-light validity statistics",
        "",
        f"Real night roots: {', '.join(args.real_roots)}",
        f"Source roots: {', '.join(args.source_roots)}",
        f"Real images: {len(real_paths)}",
        f"Source images: {len(source_paths)}",
        f"Image size: {args.image_size} x {args.image_size}",
        f"Patch size: {args.patch_size}",
        f"Gamma: {args.gamma}",
        f"Read noise std: {args.read_noise_std}",
        "",
        "## Real night reference",
        "",
        f"- mean luminance: {real['mean_luma']:.6f}",
        f"- median luminance: {real['median_luma']:.6f}",
        f"- dark ratio <0.05: {real['dark005']:.6f}",
        f"- dark ratio <0.10: {real['dark010']:.6f}",
        f"- mean gradient magnitude: {real['grad_mean']:.6f}",
        "",
        "## Best synthetic variant",
        "",
        f"- variant: {best['variant']}",
        f"- combined distance: {best['combined_distance']:.6f}",
        f"- luminance hist L1: {best['luma_hist_l1']:.6f}",
        f"- gradient hist L1: {best['grad_hist_l1']:.6f}",
        f"- mean-variance L1: {best['mean_variance_l1']:.8f}",
        f"- dark ratio error: {best['dark_ratio_error']:.6f}",
        "",
        "## Ranking",
        "",
        "| rank | variant | combined | luma_hist | grad_hist | mean_var | dark_err | mean_luma | dark<0.05 | dark<0.10 |",
        "|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for rank, row in enumerate(rows[:16], start=1):
        md_lines.append(
            "| {rank} | {variant} | {combined:.6f} | {luma:.6f} | {grad:.6f} | {mv:.8f} | {dark:.6f} | {mean:.6f} | {d05:.6f} | {d10:.6f} |".format(
                rank=rank,
                variant=row["variant"],
                combined=row["combined_distance"],
                luma=row["luma_hist_l1"],
                grad=row["grad_hist_l1"],
                mv=row["mean_variance_l1"],
                dark=row["dark_ratio_error"],
                mean=row["mean_luma"],
                d05=row["dark005"],
                d10=row["dark010"],
            )
        )
    if plot_note:
        md_lines.extend(["", f"Plot note: {plot_note}"])
    with open(output_dir / "summary.md", "w") as f:
        f.write("\n".join(md_lines) + "\n")

    print(json.dumps({"output_dir": str(output_dir), "best": best, "num_rows": len(rows)}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
