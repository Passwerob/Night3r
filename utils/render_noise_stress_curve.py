#!/usr/bin/env python3
"""Render paper-style raw-depth and scale curves from a noise summary."""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


METHODS = (
    ("MASt3R", "#334155", "o"),
    ("Night3R-real", "#0f766e", "s"),
    ("Night3R-zero", "#c2410c", "^"),
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary-json", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args()


def main():
    args = parse_args()
    payload = json.loads(args.summary_json.read_text(encoding="utf-8"))
    rows = payload["rows"]
    levels = ["Stress-A", "Stress-B", "Stress-C"]
    x = list(range(len(levels)))
    xlabels = [
        "A\n$\\sigma_r=8\\times10^{-5}$",
        "B\n$\\sigma_r=3\\times10^{-4}$",
        "C\n$\\sigma_r=10^{-3}$",
    ]

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "axes.titlesize": 11,
            "axes.labelsize": 9,
            "legend.fontsize": 8,
            "figure.dpi": 160,
        }
    )
    fig, axes = plt.subplots(2, 2, figsize=(8.2, 5.6), constrained_layout=True)
    for row_index, dataset in enumerate(("DSEC", "MVSEC")):
        for method, color, marker in METHODS:
            selected = {
                row["level"]: row
                for row in rows
                if row["dataset"] == dataset and row["method"] == method
            }
            axes[row_index, 0].plot(
                x,
                [selected[level]["raw_absrel"] for level in levels],
                color=color,
                marker=marker,
                linewidth=2,
                markersize=5,
                label=method,
            )
            axes[row_index, 1].plot(
                x,
                [selected[level]["scale_median"] for level in levels],
                color=color,
                marker=marker,
                linewidth=2,
                markersize=5,
                label=method,
            )

        axes[row_index, 0].set_title(f"{dataset}: raw depth error")
        axes[row_index, 0].set_ylabel("Raw AbsRel $\\downarrow$")
        axes[row_index, 1].set_title(f"{dataset}: metric-scale stability")
        axes[row_index, 1].set_ylabel("Median scale factor $\\downarrow$")
        for axis in axes[row_index]:
            axis.set_xticks(x, xlabels)
            axis.grid(axis="y", color="#cbd5e1", linewidth=0.7, alpha=0.8)
            axis.spines["top"].set_visible(False)
            axis.spines["right"].set_visible(False)
            axis.set_axisbelow(True)

    axes[0, 0].legend(loc="upper left", frameon=False)
    fig.suptitle(
        "Linear-domain noise stress: identical exposure, pairs, and events",
        fontsize=12,
        fontweight="bold",
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(json.dumps({"output": str(args.output.resolve())}))


if __name__ == "__main__":
    main()
