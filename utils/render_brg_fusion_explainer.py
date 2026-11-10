#!/usr/bin/env python3
"""Render a concise diagram of the implemented Night3R BRG fusion path."""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


COLORS = {
    "rgb": "#DCEBF7",
    "event": "#FCE5CD",
    "gate": "#E2F0D9",
    "output": "#EDE1F5",
    "fixed": "#F2F2F2",
    "ink": "#1F2933",
    "muted": "#52606D",
    "arrow": "#52606D",
}


def box(axis, x, y, width, height, text, color, title=None, fontsize=12):
    patch = FancyBboxPatch(
        (x, y), width, height,
        boxstyle="round,pad=0.012,rounding_size=0.018",
        linewidth=1.4,
        edgecolor=COLORS["ink"],
        facecolor=color,
        zorder=2,
    )
    axis.add_patch(patch)
    if title:
        axis.text(
            x + width / 2, y + height * 0.68, title,
            ha="center", va="center", fontsize=fontsize + 1,
            weight="bold", color=COLORS["ink"], zorder=3,
        )
        axis.text(
            x + width / 2, y + height * 0.33, text,
            ha="center", va="center", fontsize=fontsize - 1,
            color=COLORS["muted"], zorder=3, wrap=True,
        )
    else:
        axis.text(
            x + width / 2, y + height / 2, text,
            ha="center", va="center", fontsize=fontsize,
            color=COLORS["ink"], zorder=3, wrap=True,
        )
    return patch


def arrow(axis, start, end, label=None, bend=0.0):
    connection = "arc3" if bend == 0 else f"arc3,rad={bend}"
    patch = FancyArrowPatch(
        start, end, arrowstyle="-|>", mutation_scale=14,
        linewidth=1.5, color=COLORS["arrow"],
        connectionstyle=connection, zorder=1,
    )
    axis.add_patch(patch)
    if label:
        mid_x = (start[0] + end[0]) / 2
        mid_y = (start[1] + end[1]) / 2
        axis.text(
            mid_x, mid_y + 0.025, label,
            ha="center", va="bottom", fontsize=9.5,
            color=COLORS["muted"], zorder=4,
        )


def step(axis, x, y, number):
    axis.text(
        x, y, str(number), ha="center", va="center", fontsize=11,
        weight="bold", color="white", zorder=5,
        bbox={"boxstyle": "circle,pad=0.30", "facecolor": "#334E68", "edgecolor": "none"},
    )


def render(output):
    figure, axis = plt.subplots(figsize=(18, 10), dpi=180)
    figure.patch.set_facecolor("white")
    axis.set_xlim(0, 18)
    axis.set_ylim(0, 9)
    axis.axis("off")

    axis.text(
        0.6, 8.45, "Night3R BRG Fusion: What Actually Happens",
        fontsize=22, weight="bold", color=COLORS["ink"], va="center",
    )
    axis.text(
        0.6, 8.05,
        "Fusion is a gated residual addition, not RGB-event cross-attention.",
        fontsize=12.5, color=COLORS["muted"], va="center",
    )

    # RGB path
    box(axis, 0.65, 5.95, 2.15, 1.1, "low-light frame", COLORS["rgb"], "Low-light RGB")
    box(axis, 3.2, 5.95, 2.25, 1.1, "original MASt3R", COLORS["rgb"], "RGB\nEncoder")
    box(axis, 5.85, 5.95, 2.05, 1.1, "one token per patch", COLORS["rgb"], "RGB\nTokens Z")
    arrow(axis, (2.8, 6.5), (3.2, 6.5))
    arrow(axis, (5.45, 6.5), (5.85, 6.5))
    step(axis, 0.35, 6.5, 1)

    # Event content path
    box(axis, 0.65, 3.35, 2.15, 1.1, "4 bins x 2 polarities", COLORS["event"], "Event Voxel")
    box(axis, 3.2, 3.35, 2.25, 1.1, "Conv patch embedding\n+ LayerNorm", COLORS["event"], "Patch + LN")
    box(axis, 5.85, 3.35, 2.05, 1.1, "event Transformer", COLORS["event"], "Event\nEncoder")
    box(axis, 8.3, 3.35, 2.15, 1.1, "LayerNorm + MLP", COLORS["event"], "Event\nResidual")
    arrow(axis, (2.8, 3.9), (3.2, 3.9))
    arrow(axis, (5.45, 3.9), (5.85, 3.9))
    arrow(axis, (7.9, 3.9), (8.3, 3.9))
    step(axis, 0.35, 3.9, 2)

    # BRG branch
    box(axis, 2.75, 1.0, 2.35, 1.15, "density, occupancy,\npolarity balance", COLORS["fixed"], "Raw Event\nStats", fontsize=10.5)
    box(axis, 5.5, 1.0, 1.85, 1.15, "learnable\naffine + bias", COLORS["gate"], "StatNorm", fontsize=10.5)
    box(axis, 7.75, 1.0, 2.05, 1.15, "learnable MLP\n+ source bias", COLORS["gate"], "BRG Gate", fontsize=10.5)
    box(axis, 10.2, 1.0, 1.55, 1.15, "one scalar\nper patch", COLORS["gate"], "Gate g", fontsize=10.5)
    arrow(axis, (1.8, 3.35), (3.5, 2.15), bend=-0.18)
    arrow(axis, (5.1, 1.58), (5.5, 1.58))
    arrow(axis, (7.35, 1.58), (7.75, 1.58))
    arrow(axis, (9.8, 1.58), (10.2, 1.58))
    step(axis, 2.3, 1.58, 3)

    # Fusion and decoder
    box(
        axis, 12.15, 3.95, 2.85, 1.95,
        "Add the gated event residual\nto the matching RGB token.\n\nLearnable global scale\nlimits event strength.",
        COLORS["gate"], "BRG Residual\nFusion", fontsize=11.5,
    )
    arrow(axis, (7.9, 6.5), (12.15, 5.25), bend=-0.12)
    arrow(axis, (10.45, 3.9), (12.15, 4.85), bend=0.10)
    arrow(axis, (10.98, 2.15), (13.55, 3.95), bend=-0.12)
    step(axis, 11.7, 5.25, 4)

    box(
        axis, 15.45, 3.95, 1.95, 1.95,
        "Normal cross-view\nattention happens here.\n\nQ, K, V use fused\nRGB tokens.",
        COLORS["output"], "MASt3R\nDecoder", fontsize=10.5,
    )
    arrow(axis, (15.0, 4.95), (15.45, 4.95))
    step(axis, 15.18, 4.95, 5)

    axis.text(
        0.7, 0.28,
        "Important: Event tokens do not directly provide Q/K/V to RGB tokens. "
        "The BRG first adds a gated event residual; the original decoder then performs cross-view attention.",
        fontsize=11.5, color=COLORS["ink"], va="center",
        bbox={"boxstyle": "round,pad=0.45", "facecolor": "#FFF7E6", "edgecolor": "#D9A441", "linewidth": 1.0},
    )

    figure.savefig(output, bbox_inches="tight", facecolor=figure.get_facecolor())
    plt.close(figure)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        default="outputs/brg_fusion_explainer_20260726/brg_fusion_explainer.png",
    )
    args = parser.parse_args()
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    render(output)
    print(output)


if __name__ == "__main__":
    main()
