#!/usr/bin/env python3
"""Export event-rich MVSEC RGB/dark-RGB/event pairs for paper figures."""

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

from PIL import Image, ImageDraw

from export_rgb_lowlight_event_assets import (
    event_activity,
    event_red_blue,
    image_tensor,
    load_event,
    lowlight_image,
    to_image,
)


REPO_ROOT = Path(__file__).resolve().parents[1]


SCENES = (
    ("01_train_MVSEC_screen000105", "train", "screen-000105-mvsec", (6, 18)),
    ("02_train_MVSEC_screen000114", "train", "screen-000114-mvsec", (36, 48)),
    ("03_train_MVSEC_screen000099", "train", "screen-000099-mvsec", (24, 36)),
    ("04_test_MVSEC_screen000129", "test", "screen-000129-mvsec", (30, 48)),
    ("05_test_MVSEC_screen000208", "test", "screen-000208-mvsec", (12, 24)),
    ("06_test_MVSEC_screen000123", "test", "screen-000123-mvsec", (30, 48)),
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--outdir",
        type=Path,
        default=REPO_ROOT
        / "outputs/paper_assets_selected_mvsec_event_rich_low_noise_20260721",
    )
    parser.add_argument("--exposure", type=float, default=0.06)
    parser.add_argument("--read-noise-std", type=float, default=0.0005)
    parser.add_argument("--shot-noise-std", type=float, default=0.002)
    parser.add_argument("--seed", type=int, default=20260721)
    return parser.parse_args()


def event_statistics(event):
    positive, negative = event_activity(event)
    activity = positive + negative
    occupancy = float((activity > 0).float().mean())
    positive_sum = float(positive.sum())
    negative_sum = float(negative.sum())
    balance = 1.0 - abs(positive_sum - negative_sum) / max(
        positive_sum + negative_sum, 1.0e-6
    )
    return {
        "occupancy": occupancy,
        "polarity_balance": balance,
        "positive_sum": positive_sum,
        "negative_sum": negative_sum,
    }


def make_contact_sheet(rows, output):
    panel_width = 420
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
    rows = []

    for prefix, split, scene, indices in SCENES:
        if split == "train":
            root = Path("/data/fcr/data/train_data/MVSEC") / scene
        else:
            root = REPO_ROOT / "test/MVSEC" / scene
        image_paths = sorted((root / "images").glob("*.png"))
        event_paths = sorted((root / "events").glob("*.pt"))

        for view, index in enumerate(indices, start=1):
            stem = f"{prefix}_view{view:02d}"
            clean = image_tensor(image_paths[index])
            event = load_event(event_paths[index])
            dark = lowlight_image(clean, lowlight_args, stem)
            event_image, _, _ = event_red_blue(event, background="white")

            rgb_path = args.outdir / f"{stem}_RGB.png"
            dark_path = args.outdir / f"{stem}_DarkenRGB.png"
            event_path = args.outdir / f"{stem}_Event.png"
            to_image(clean).save(rgb_path)
            to_image(dark).save(dark_path)
            event_image.save(event_path)
            rows.append((stem, (rgb_path, dark_path, event_path)))
            records.append(
                {
                    "prefix": stem,
                    "split": split,
                    "scene": scene,
                    "frame_index": index,
                    "source_rgb": str(image_paths[index]),
                    "source_event": str(event_paths[index]),
                    "event": event_statistics(event),
                }
            )

    make_contact_sheet(rows, args.outdir / "contact_sheet.png")
    index = {
        "selection": (
            "Scenes were ranked by the minimum event occupancy across both views, "
            "mean occupancy, polarity balance, and visible image structure."
        ),
        "darkening": {
            "exposure": args.exposure,
            "read_noise_std": args.read_noise_std,
            "shot_noise_std": args.shot_noise_std,
        },
        "records": records,
    }
    (args.outdir / "index.json").write_text(
        json.dumps(index, indent=2), encoding="utf-8"
    )

    lines = [
        "# Event-rich MVSEC figure assets",
        "",
        "Each scene contains two views. Files sharing a view prefix are aligned RGB,",
        "DarkenRGB, and Event images. Positive events are red and negative events are blue.",
        "",
        "| Prefix | Split | Frame | Event occupancy | Polarity balance |",
        "|---|---|---:|---:|---:|",
    ]
    for record in records:
        lines.append(
            f"| `{record['prefix']}` | {record['split']} | {record['frame_index']} | "
            f"{record['event']['occupancy']:.4f} | "
            f"{record['event']['polarity_balance']:.4f} |"
        )
    (args.outdir / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"outdir": str(args.outdir), "triplets": len(records)}))


if __name__ == "__main__":
    main()
