#!/usr/bin/env python3
"""Export low-noise MVSEC figure candidates with explicit 4-bin events."""

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import torch
from PIL import Image, ImageDraw

from export_rgb_lowlight_event_assets import (
    event_red_blue,
    image_tensor,
    load_event,
    lowlight_image,
    to_image,
)


REPO_ROOT = Path(__file__).resolve().parents[1]

# These scenes have clear image structure and useful event support in both views.
SCENES = (
    ("01_MVSEC_screen000105", "screen-000105-mvsec", (6, 18)),
    ("02_MVSEC_screen000114", "screen-000114-mvsec", (36, 48)),
    ("03_MVSEC_screen000099", "screen-000099-mvsec", (24, 36)),
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--outdir",
        type=Path,
        default=REPO_ROOT
        / "outputs/paper_assets_mvsec_recent4bins_ultralow_noise_20260721",
    )
    parser.add_argument("--recent-bins", type=int, default=4)
    parser.add_argument("--exposure", type=float, default=0.045)
    parser.add_argument("--read-noise-std", type=float, default=0.0001)
    parser.add_argument("--shot-noise-std", type=float, default=0.0005)
    parser.add_argument("--seed", type=int, default=20260721)
    return parser.parse_args()


def select_recent_polarity_bins(event, recent_bins):
    """Return recent positive/negative bins without confusing polarity with time."""
    if event.ndim != 3 or event.shape[0] % 2:
        raise ValueError(f"Expected [2B,H,W] event voxel, got {tuple(event.shape)}")
    source_bins = event.shape[0] // 2
    selected_bins = min(int(recent_bins), source_bins)
    if selected_bins <= 0:
        raise ValueError("recent_bins must be positive")
    positive = event[source_bins - selected_bins : source_bins].abs()
    negative = event[2 * source_bins - selected_bins : 2 * source_bins].abs()
    indices = list(range(source_bins - selected_bins, source_bins))
    return positive, negative, source_bins, indices


def render_recent_event(positive, negative):
    event = torch.cat((positive, negative), dim=0)
    image, _, _ = event_red_blue(event, background="white")
    return image


def render_bin_strip(positive, negative):
    panels = []
    label_height = 26
    for index in range(positive.shape[0]):
        event = torch.stack((positive[index], negative[index]), dim=0)
        panel, _, _ = event_red_blue(event, background="white")
        canvas = Image.new("RGB", (panel.width, panel.height + label_height), "white")
        ImageDraw.Draw(canvas).text((8, 7), f"t{index + 1}", fill="black")
        canvas.paste(panel, (0, label_height))
        panels.append(canvas)
    strip = Image.new("RGB", (sum(panel.width for panel in panels), panels[0].height), "white")
    left = 0
    for panel in panels:
        strip.paste(panel, (left, 0))
        left += panel.width
    return strip


def event_statistics(positive, negative):
    activity = positive.sum(dim=0) + negative.sum(dim=0)
    positive_sum = float(positive.sum())
    negative_sum = float(negative.sum())
    return {
        "occupancy": float((activity > 0).float().mean()),
        "positive_sum": positive_sum,
        "negative_sum": negative_sum,
        "polarity_balance": 1.0
        - abs(positive_sum - negative_sum) / max(positive_sum + negative_sum, 1.0e-6),
    }


def make_contact_sheet(rows, output):
    panel_width = 500
    label_height = 34
    gap = 8
    rendered = []
    for label, paths in rows:
        images = [Image.open(path).convert("RGB") for path in paths]
        panel_height = round(images[0].height * panel_width / images[0].width)
        row = Image.new("RGB", (3 * panel_width, panel_height + label_height), "white")
        draw = ImageDraw.Draw(row)
        draw.text((10, 10), label, fill="black")
        for column, image in enumerate(images):
            image = image.resize((panel_width, panel_height), Image.Resampling.LANCZOS)
            row.paste(image, (column * panel_width, label_height))
        rendered.append(row)

    sheet = Image.new(
        "RGB",
        (3 * panel_width, sum(row.height for row in rendered) + gap * (len(rendered) - 1)),
        "white",
    )
    top = 0
    for row in rendered:
        sheet.paste(row, (0, top))
        top += row.height + gap
    sheet.save(output)


def main():
    args = parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)
    lowlight_args = SimpleNamespace(
        exposure=args.exposure,
        read_noise_std=args.read_noise_std,
        shot_noise_std=args.shot_noise_std,
        seed=args.seed,
    )
    records = []
    contact_rows = []

    for prefix, scene, indices in SCENES:
        root = Path("/data/fcr/data/train_data/MVSEC") / scene
        scene_out = args.outdir / prefix
        scene_out.mkdir(parents=True, exist_ok=True)
        image_paths = sorted((root / "images").glob("*.png"))
        event_paths = sorted((root / "events").glob("*.pt"))

        for view, index in enumerate(indices, start=1):
            stem = f"view{view:02d}"
            clean = image_tensor(image_paths[index])
            event = load_event(event_paths[index])
            positive, negative, source_bins, selected_indices = select_recent_polarity_bins(
                event, args.recent_bins
            )
            dark = lowlight_image(clean, lowlight_args, f"{prefix}/{stem}")
            event_image = render_recent_event(positive, negative)
            bin_strip = render_bin_strip(positive, negative)

            rgb_path = scene_out / f"{stem}_RGB.png"
            dark_path = scene_out / f"{stem}_DarkenRGB.png"
            event_path = scene_out / f"{stem}_Event_recent4bins.png"
            strip_path = scene_out / f"{stem}_Event_recent4bins_strip.png"
            to_image(clean).save(rgb_path)
            to_image(dark).save(dark_path)
            event_image.save(event_path)
            bin_strip.save(strip_path)
            contact_rows.append((f"{prefix}_{stem}", (rgb_path, dark_path, event_path)))
            records.append(
                {
                    "prefix": prefix,
                    "view": stem,
                    "scene": scene,
                    "frame_index": index,
                    "source_rgb": str(image_paths[index]),
                    "source_event": str(event_paths[index]),
                    "source_event_shape": list(event.shape),
                    "source_temporal_bins": source_bins,
                    "selected_temporal_bin_indices_oldest_to_newest": selected_indices,
                    "output_signed_temporal_bins": len(selected_indices),
                    "event": event_statistics(positive, negative),
                }
            )

    make_contact_sheet(contact_rows, args.outdir / "contact_sheet.png")
    manifest = {
        "darkening": {
            "exposure": args.exposure,
            "read_noise_std": args.read_noise_std,
            "shot_noise_std": args.shot_noise_std,
            "gamma_space_input": True,
        },
        "event_representation": {
            "source_layout": "[positive temporal bins, negative temporal bins]",
            "requested_recent_temporal_bins": args.recent_bins,
            "display_layout": "four signed temporal bins; positive red, negative blue",
            "note": (
                "The source has eight channels because four temporal bins are stored "
                "separately for two polarities. It does not contain eight temporal bins."
            ),
        },
        "records": records,
    }
    (args.outdir / "index.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )

    lines = [
        "# MVSEC low-noise recent-4-bin figure candidates",
        "",
        "Each scene folder contains two aligned views. For every view, `RGB`,",
        "`DarkenRGB`, the aggregate red/blue event image, and a four-bin temporal",
        "strip are provided.",
        "",
        "The saved source event voxel has eight channels: four temporal bins for",
        "positive polarity followed by four temporal bins for negative polarity.",
        "The visualization combines matching polarity channels into four signed",
        "temporal bins; it does not treat the eight channels as eight time bins.",
        "",
        f"Darkening: exposure `{args.exposure}`, read noise `{args.read_noise_std}`,",
        f"shot-noise scale `{args.shot_noise_std}`.",
        "",
        "| Scene | View | Frame | Event occupancy | Polarity balance |",
        "|---|---|---:|---:|---:|",
    ]
    for record in records:
        lines.append(
            f"| `{record['scene']}` | {record['view']} | {record['frame_index']} | "
            f"{record['event']['occupancy']:.4f} | "
            f"{record['event']['polarity_balance']:.4f} |"
        )
    (args.outdir / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"outdir": str(args.outdir), "triplets": len(records)}))


if __name__ == "__main__":
    main()
