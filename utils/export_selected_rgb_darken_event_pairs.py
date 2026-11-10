#!/usr/bin/env python3
"""Export two aligned RGB/dark-RGB/event views for selected paper scenes."""

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

from PIL import Image, ImageDraw

from export_rgb_lowlight_event_assets import (
    event_red_blue,
    image_tensor,
    load_event,
    lowlight_image,
    to_image,
)


REPO_ROOT = Path(__file__).resolve().parents[1]


SCENES = (
    {
        "prefix": "01_train_DL3DV_bikes",
        "dataset": "DL3DV",
        "split": "train",
        "scene": "screen-000059",
        "root": Path("/data/fcr/data/train_data/DL3DV/screen-000059"),
        "indices": (32, 48),
    },
    {
        "prefix": "02_test_DL3DV_temple",
        "dataset": "DL3DV",
        "split": "test",
        "scene": "screen-000095",
        "root": REPO_ROOT / "test/DL3DV/screen-000095",
        "indices": (32, 48),
    },
    {
        "prefix": "03_train_DSEC_alpine_road",
        "dataset": "DSEC",
        "split": "train",
        "scene": "screen-000001-dsec",
        "root": Path("/data/fcr/data/train_data/DSEC/screen-000001-dsec"),
        "indices": (252, 268),
    },
    {
        "prefix": "04_test_DSEC_church_road",
        "dataset": "DSEC",
        "split": "test",
        "scene": "screen-000002-dsec",
        "root": REPO_ROOT / "test/DSEC/screen-000002-dsec",
        "indices": (481, 497),
    },
    {
        "prefix": "05_train_MVSEC_urban_drive",
        "dataset": "MVSEC",
        "split": "train",
        "scene": "screen-000099-mvsec",
        "root": Path("/data/fcr/data/train_data/MVSEC/screen-000099-mvsec"),
        "indices": (32, 48),
    },
    {
        "prefix": "06_test_MVSEC_church_drive",
        "dataset": "MVSEC",
        "split": "test",
        "scene": "screen-000110-mvsec",
        "root": REPO_ROOT / "test/MVSEC/screen-000110-mvsec",
        "indices": (32, 48),
    },
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--outdir",
        type=Path,
        default=REPO_ROOT
        / "outputs/paper_assets_selected_rgb_darken_event_low_noise_20260721",
    )
    parser.add_argument("--exposure", type=float, default=0.06)
    parser.add_argument("--read-noise-std", type=float, default=0.0005)
    parser.add_argument("--shot-noise-std", type=float, default=0.002)
    parser.add_argument("--seed", type=int, default=20260721)
    return parser.parse_args()


def make_contact_sheet(rows, output):
    panel_width = 420
    label_height = 34
    row_gap = 8
    rendered_rows = []
    for label, paths in rows:
        images = [Image.open(path).convert("RGB") for path in paths]
        panel_height = round(images[0].height * panel_width / images[0].width)
        canvas = Image.new(
            "RGB", (3 * panel_width, panel_height + label_height), "white"
        )
        draw = ImageDraw.Draw(canvas)
        draw.text((10, 10), label, fill="black")
        for column, image in enumerate(images):
            image = image.resize((panel_width, panel_height), Image.Resampling.LANCZOS)
            canvas.paste(image, (column * panel_width, label_height))
        rendered_rows.append(canvas)

    height = sum(row.height for row in rendered_rows) + row_gap * (
        len(rendered_rows) - 1
    )
    sheet = Image.new("RGB", (3 * panel_width, height), "white")
    top = 0
    for row in rendered_rows:
        sheet.paste(row, (0, top))
        top += row.height + row_gap
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

    for scene in SCENES:
        image_paths = sorted((scene["root"] / "images").glob("*.png"))
        event_paths = sorted((scene["root"] / "events").glob("*.pt"))
        for view_number, index in enumerate(scene["indices"], start=1):
            if index >= min(len(image_paths), len(event_paths)):
                raise IndexError(f"Frame {index} is unavailable for {scene['root']}")

            stem = f"{scene['prefix']}_view{view_number:02d}"
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
            contact_rows.append((stem, (rgb_path, dark_path, event_path)))
            records.append(
                {
                    "prefix": stem,
                    "split": scene["split"],
                    "dataset": scene["dataset"],
                    "scene": scene["scene"],
                    "frame_index": index,
                    "source_rgb": str(image_paths[index]),
                    "source_event": str(event_paths[index]),
                }
            )

    make_contact_sheet(contact_rows, args.outdir / "contact_sheet.png")
    metadata = {
        "darkening": {
            "exposure": args.exposure,
            "read_noise_std": args.read_noise_std,
            "shot_noise_std": args.shot_noise_std,
            "linear_domain": True,
            "gamma_space_input": True,
        },
        "event_visualization": {
            "positive": "red",
            "negative": "blue",
            "background": "white",
            "scaling": "log1p with nonzero 99.5-percentile normalization",
        },
        "records": records,
    }
    (args.outdir / "index.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )

    lines = [
        "# Selected RGB / DarkenRGB / Event Pairs",
        "",
        "Files sharing the same scene prefix and `view01` or `view02` suffix are",
        "spatially and temporally aligned. Each scene contains two reconstruction",
        "views, and each view contains RGB, DarkenRGB, and Event images.",
        "",
        f"Darkening uses linear-domain exposure {args.exposure}, read-noise standard ",
        f"deviation {args.read_noise_std}, and shot-noise scale {args.shot_noise_std}.",
        "Positive events are red and negative events are blue on a white background.",
        "",
        "| Scene prefix | View 1 frame | View 2 frame |",
        "|---|---:|---:|",
    ]
    for scene in SCENES:
        lines.append(
            f"| `{scene['prefix']}` | {scene['indices'][0]} | {scene['indices'][1]} |"
        )
    (args.outdir / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"outdir": str(args.outdir), "triplets": len(records)}))


if __name__ == "__main__":
    main()
