#!/usr/bin/env python3
"""Render paper-ready low-light statistics from the saved validity results."""

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


DISPLAY = {
    "real_night": ("Real MVSEC night", "#111111", "-", 2.4),
    "source_clean": ("Clean DL3DV", "#7A7A7A", "--", 1.6),
    "srgb_multiply_x0.03": ("Direct sRGB x0.03", "#D55E00", "-.", 1.9),
    "linear_exposure_x0.03": ("Linear exposure x0.03", "#0072B2", "-", 2.2),
    "linear_shot_read_x0.03": ("Linear + shot/read x0.03", "#009E73", ":", 2.2),
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("outputs/lowlight_validity_mamba_20260710"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("figures/lowlight_data_validity"),
    )
    return parser.parse_args()


def load_mean_variance(path):
    curves = {name: ([], []) for name in DISPLAY}
    with path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            name = row["variant"]
            if name not in curves or not row["patch_var_mean"]:
                continue
            curves[name][0].append(float(row["patch_mean_center"]))
            curves[name][1].append(float(row["patch_var_mean"]))
    return curves


def main():
    args = parse_args()
    metrics = json.loads((args.input_dir / "metrics.json").read_text())
    groups = {group["name"]: group for group in metrics["groups"]}
    curves = load_mean_variance(args.input_dir / "mean_variance_curves.csv")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update(
        {
            "font.size": 9,
            "axes.labelsize": 10,
            "axes.titlesize": 11,
            "legend.fontsize": 8,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    fig, axes = plt.subplots(1, 2, figsize=(10.8, 3.8))

    ax = axes[0]
    for name, (label, color, linestyle, width) in DISPLAY.items():
        hist = np.asarray(groups[name]["luma_hist"], dtype=np.float64)
        centers = (np.arange(hist.size, dtype=np.float64) + 0.5) / hist.size
        ax.plot(centers, np.maximum(hist, 1e-6), label=label, color=color,
                linestyle=linestyle, linewidth=width)
    ax.axvline(0.05, color="#B0B0B0", linewidth=0.8)
    ax.axvline(0.10, color="#B0B0B0", linewidth=0.8)
    ax.set_yscale("log")
    ax.set_xlim(0.0, 0.45)
    ax.set_ylim(1e-5, 1.0)
    ax.set_xlabel("Luminance")
    ax.set_ylabel("Normalized frequency")
    ax.set_title("(a) Luminance distribution")
    ax.grid(True, which="major", color="#D9D9D9", linewidth=0.6, alpha=0.75)

    ax = axes[1]
    for name, (label, color, linestyle, width) in DISPLAY.items():
        x, y = curves[name]
        ax.plot(x, y, label=label, color=color, linestyle=linestyle,
                linewidth=width, marker="o", markersize=2.8)
    ax.set_xlim(0.0, 0.50)
    ax.set_ylim(bottom=0.0)
    ax.set_xlabel("Patch mean luminance")
    ax.set_ylabel("Patch variance")
    ax.set_title("(b) Patch mean-variance")
    ax.grid(True, color="#D9D9D9", linewidth=0.6, alpha=0.75)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=5, frameon=False,
               bbox_to_anchor=(0.5, -0.01))
    fig.tight_layout(rect=(0.0, 0.12, 1.0, 1.0), w_pad=2.5)
    for suffix in ("png", "pdf"):
        fig.savefig(
            args.output_dir / f"lowlight_statistics.{suffix}",
            dpi=240 if suffix == "png" else None,
            bbox_inches="tight",
        )
    plt.close(fig)


if __name__ == "__main__":
    main()
