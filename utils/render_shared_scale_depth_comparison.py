#!/usr/bin/env python3
"""Render raw method depths with one shared pair-level scale and color range."""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw


METHODS = ("mast3r", "dark3r", "vggt", "night3r")
DISPLAY_NAMES = {
    "mast3r": "MASt3R",
    "dark3r": "Dark3R",
    "vggt": "VGGT",
    "night3r": "Night3R",
}


def parse_args():
    parser = argparse.ArgumentParser("Render shared-scale raw depth comparison")
    parser.add_argument("--pair-dir", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--reference", required=True, type=Path)
    parser.add_argument("--reference-key", default=None)
    parser.add_argument("--reference-kind", required=True)
    parser.add_argument("--clean-rgb", required=True, type=Path)
    parser.add_argument("--stress-rgb", required=True, type=Path)
    parser.add_argument("--events", required=True, type=Path)
    parser.add_argument("--low", type=float, default=2.0)
    parser.add_argument("--high", type=float, default=98.0)
    return parser.parse_args()


def valid(values):
    values = np.asarray(values, dtype=np.float32)
    return values[np.isfinite(values) & (values > 0)]


def load_reference(path, key):
    if path.suffix == ".npz":
        payload = np.load(path)
        if key is None:
            raise ValueError("--reference-key is required for .npz references")
        if key not in payload.files:
            raise KeyError(f"Reference key {key!r} is missing from {path}")
        return payload[key].astype(np.float32)
    return np.load(path).astype(np.float32)


def normalize(values, vmin, vmax):
    values = np.asarray(values, dtype=np.float32)
    out = np.zeros((*values.shape, 4), dtype=np.float32)
    mask = np.isfinite(values) & (values > 0)
    norm = np.clip((values - vmin) / max(vmax - vmin, 1.0e-8), 0.0, 1.0)
    out[mask] = plt.get_cmap("viridis")(norm[mask])
    return out


def rgb(path, size):
    image = Image.open(path).convert("RGB")
    return np.asarray(image.resize(size, Image.Resampling.BILINEAR))


def event_image(path, size):
    image = Image.open(path).convert("RGB")
    return np.asarray(image.resize(size, Image.Resampling.BILINEAR))


def main():
    args = parse_args()
    arrays = {method: np.load(args.pair_dir / method / "depth_raw.npy").astype(np.float32) for method in METHODS}
    reference = load_reference(args.reference, args.reference_key)
    reference_values = valid(reference)
    if reference_values.size == 0:
        raise RuntimeError("Reference depth has no valid pixels")

    # One scalar maps the Night3R raw median to the reference median. It is
    # applied unchanged to every method to preserve their relative scale.
    scale = float(np.median(reference_values) / max(np.median(valid(arrays["night3r"])), 1.0e-8))
    scaled = {method: depth * scale for method, depth in arrays.items()}
    all_values = np.concatenate([reference_values] + [valid(depth) for depth in scaled.values()])
    vmin, vmax = np.percentile(all_values, [args.low, args.high])
    if vmax <= vmin:
        vmax = vmin + 1.0e-6

    size = (arrays["night3r"].shape[1], arrays["night3r"].shape[0])
    panels = [
        (rgb(args.clean_rgb, size), "Clean RGB"),
        (rgb(args.stress_rgb, size), "Stress RGB"),
        (event_image(args.events, size), "Aligned events"),
    ]
    for method in METHODS:
        panels.append((normalize(scaled[method], vmin, vmax), DISPLAY_NAMES[method]))
    panels.append((normalize(reference, vmin, vmax), args.reference_kind))

    figure, axes = plt.subplots(1, len(panels), figsize=(3.0 * len(panels), 2.6), dpi=220)
    if len(panels) == 1:
        axes = [axes]
    for axis, (image, title) in zip(axes, panels):
        axis.imshow(image)
        axis.set_title(title, fontsize=9)
        axis.axis("off")
    figure.tight_layout(pad=0.4)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output, bbox_inches="tight", facecolor="white")
    plt.close(figure)

    protocol = {
        "visualization": "raw_depth_shared_pair_scale",
        "methods": list(METHODS),
        "shared_scale": scale,
        "reference_kind": args.reference_kind,
        "reference": str(args.reference.resolve()),
        "reference_key": args.reference_key,
        "display_range": [float(vmin), float(vmax)],
        "percentiles": [args.low, args.high],
        "metric_warning": "qualitative only; formal metrics retain per-method median/Sim(3) alignment",
    }
    args.output.with_suffix(".json").write_text(json.dumps(protocol, indent=2) + "\n")
    print(json.dumps(protocol, indent=2))


if __name__ == "__main__":
    main()
