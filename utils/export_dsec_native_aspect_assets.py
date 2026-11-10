#!/usr/bin/env python3
"""Export DSEC figure assets at the native rectified RGB aspect ratio."""

import argparse
import csv
import json
import zipfile
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from PIL import Image, ImageDraw

from export_rgb_lowlight_event_assets import (
    event_red_blue,
    load_event,
    lowlight_image,
    to_image,
)


REPO_ROOT = Path(__file__).resolve().parents[1]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--scene-dir",
        type=Path,
        default=REPO_ROOT / "test/DSEC/screen-000004-dsec",
    )
    parser.add_argument(
        "--image-zip",
        type=Path,
        default=REPO_ROOT
        / "downloads/DSEC/zurich_city_11_b/zurich_city_11_b_images_rectified_left.zip",
    )
    parser.add_argument(
        "--outdir",
        type=Path,
        default=REPO_ROOT / "outputs/paper_assets_dsec_native4by3_20260721",
    )
    parser.add_argument("--frame-indices", type=int, nargs="+", default=(732, 744))
    parser.add_argument("--exposure", type=float, default=0.045)
    parser.add_argument("--read-noise-std", type=float, default=0.0001)
    parser.add_argument("--shot-noise-std", type=float, default=0.0005)
    parser.add_argument("--seed", type=int, default=20260721)
    return parser.parse_args()


def read_frame_metadata(scene_dir):
    path = scene_dir / "meta/frames_meta.csv"
    with path.open(newline="") as handle:
        return {int(row["out_idx"]): row for row in csv.DictReader(handle)}


def read_zip_rgb(archive, name):
    with archive.open(name) as handle:
        image = Image.open(handle).convert("RGB")
        image.load()
    return image


def image_tensor(image):
    array = np.asarray(image, dtype=np.float32) / 255.0
    return torch.from_numpy(array).permute(2, 0, 1).contiguous()


def make_contact_sheet(rows, output):
    panel_width = 480
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
            resized = image.resize((panel_width, panel_height), Image.Resampling.LANCZOS)
            row.paste(resized, (column * panel_width, label_height))
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
    metadata = read_frame_metadata(args.scene_dir)
    lowlight_args = SimpleNamespace(
        exposure=args.exposure,
        read_noise_std=args.read_noise_std,
        shot_noise_std=args.shot_noise_std,
        seed=args.seed,
    )
    records = []
    contact_rows = []

    with zipfile.ZipFile(args.image_zip) as archive:
        for view, frame_index in enumerate(args.frame_indices, start=1):
            row = metadata[frame_index]
            source_name = row["source_file"]
            source_rgb = read_zip_rgb(archive, source_name)
            processed_rgb = Image.open(
                args.scene_dir / "images" / f"{frame_index:06d}.png"
            ).convert("RGB")
            event = load_event(args.scene_dir / "events" / f"{frame_index:06d}.pt")
            event_image, _, _ = event_red_blue(event, background="white")
            event_image = event_image.resize(source_rgb.size, Image.Resampling.BILINEAR)
            clean = image_tensor(source_rgb)
            dark = lowlight_image(clean, lowlight_args, f"dsec-native-view{view:02d}")

            stem = f"view{view:02d}_frame{frame_index:06d}"
            rgb_path = args.outdir / f"{stem}_RGB_native4by3.png"
            dark_path = args.outdir / f"{stem}_DarkenRGB_native4by3.png"
            event_path = args.outdir / f"{stem}_Event_display_native4by3.png"
            processed_path = args.outdir / f"{stem}_RGB_processed512.png"
            source_rgb.save(rgb_path)
            to_image(dark).save(dark_path)
            event_image.save(event_path)
            processed_rgb.save(processed_path)
            contact_rows.append((stem, (rgb_path, dark_path, event_path)))
            records.append(
                {
                    "view": view,
                    "processed_frame_index": frame_index,
                    "source_file": source_name,
                    "source_rgb_size_wh": list(source_rgb.size),
                    "processed_rgb_size_wh": list(processed_rgb.size),
                    "event_source_shape_chw": list(event.shape),
                    "rgb_path": str(rgb_path),
                    "dark_rgb_path": str(dark_path),
                    "event_display_path": str(event_path),
                }
            )

    make_contact_sheet(contact_rows, args.outdir / "contact_sheet_native4by3.png")
    index = {
        "source_sequence": "zurich_city_11_b",
        "native_rectified_rgb_aspect_ratio": "4:3",
        "native_rectified_rgb_size_wh": [1440, 1080],
        "processed_model_input_size_wh": [512, 512],
        "darkening": {
            "exposure": args.exposure,
            "read_noise_std": args.read_noise_std,
            "shot_noise_std": args.shot_noise_std,
        },
        "event_display_note": (
            "The processed 512x512 event visualization is resampled to 1440x1080 "
            "only for the native-aspect paper display. Model evaluation still uses "
            "the stored 512x512 tensors."
        ),
        "records": records,
    }
    (args.outdir / "index.json").write_text(
        json.dumps(index, indent=2), encoding="utf-8"
    )
    (args.outdir / "README.md").write_text(
        "# DSEC native-aspect figure assets\n\n"
        "DSEC rectified RGB is 1440 x 1080 (4:3), not 16:9. The square images in\n"
        "the processed Night3R scene were resized to 512 x 512 during packing.\n"
        "These files recover the native RGB directly from the official image zip.\n"
        "Event images are resized back to the native canvas for display only.\n",
        encoding="utf-8",
    )
    print(json.dumps({"outdir": str(args.outdir), "views": len(records)}))


if __name__ == "__main__":
    main()
