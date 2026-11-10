#!/usr/bin/env python3
"""Compose paper-ready multi-scene Night3R geometry figures."""

import argparse
import csv
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont


REPO_ROOT = Path(__file__).resolve().parents[1]
WHITE = (255, 255, 255)
TEXT = (20, 24, 29)
MUTED = (82, 89, 99)
ACCENT = (0, 122, 82)
SEPARATOR = (222, 226, 231)

DEFAULT_SCENES = (
    (
        "A",
        REPO_ROOT
        / "outputs/paper_main_geometry_candidates_20260721/mvsec_157_034_035",
    ),
    (
        "B",
        REPO_ROOT
        / "outputs/paper_main_geometry_candidates_20260721/mvsec_205_042_043",
    ),
    (
        "C",
        REPO_ROOT
        / "outputs/paper_main_geometry_candidates_20260721/mvsec_200_000_001",
    ),
    (
        "D",
        REPO_ROOT
        / "outputs/paper_showcase_mvsec207_final1k_mast3r_vs_night3r_20260721",
    ),
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--outdir",
        type=Path,
        default=REPO_ROOT / "outputs/paper_main_geometry_figure_20260721",
    )
    parser.add_argument(
        "--scene-root",
        type=Path,
        action="append",
        default=None,
        help="Pair-export directory. Repeat to define the figure rows.",
    )
    return parser.parse_args()


def font(size, bold=False):
    name = "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf"
    path = Path("/usr/share/fonts/truetype/dejavu") / name
    return ImageFont.truetype(str(path), size) if path.is_file() else ImageFont.load_default()


def contain(image, size, background=WHITE):
    image = image.convert("RGB")
    scale = min(size[0] / image.width, size[1] / image.height)
    resized = image.resize(
        (max(1, round(image.width * scale)), max(1, round(image.height * scale))),
        Image.Resampling.LANCZOS,
    )
    canvas = Image.new("RGB", size, background)
    canvas.paste(
        resized,
        ((size[0] - resized.width) // 2, (size[1] - resized.height) // 2),
    )
    return canvas


def pair_panel(image1, image2, size, background=WHITE):
    gap = 6
    half_width = (size[0] - gap) // 2
    panel = Image.new("RGB", size, background)
    first = contain(image1, (half_width, size[1]), background)
    second = contain(image2, (half_width, size[1]), background)
    panel.paste(first, (0, 0))
    panel.paste(second, (half_width + gap, 0))
    return panel


def common_geometry_crop(images, margin=18):
    masks = []
    for image in images:
        array = np.asarray(image.convert("RGB"))
        masks.append(np.any(array < 248, axis=2))
    union = np.logical_or.reduce(masks)
    yy, xx = np.nonzero(union)
    if not len(xx):
        return images
    left = max(int(xx.min()) - margin, 0)
    top = max(int(yy.min()) - margin, 0)
    right = min(int(xx.max()) + margin + 1, images[0].width)
    bottom = min(int(yy.max()) + margin + 1, images[0].height)
    return [image.crop((left, top, right, bottom)) for image in images]


def draw_depth_legend(canvas, y):
    width, height = 250, 14
    gradient = np.arange(256, dtype=np.uint8)[None]
    gradient = cv2.applyColorMap(gradient, cv2.COLORMAP_VIRIDIS)
    gradient = cv2.cvtColor(gradient, cv2.COLOR_BGR2RGB)
    gradient = Image.fromarray(gradient).resize((width, height), Image.Resampling.BILINEAR)
    x = (canvas.width - width) // 2
    canvas.paste(gradient, (x, y))
    draw = ImageDraw.Draw(canvas)
    legend_font = font(14)
    draw.text((x - 10, y + height / 2), "Near", fill=MUTED, font=legend_font, anchor="rm")
    draw.text((x + width + 10, y + height / 2), "Far", fill=MUTED, font=legend_font, anchor="lm")


def load_scene(label, root):
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    geometry_dir = root / "geometry_single_images_white"
    geometry = common_geometry_crop(
        [
            Image.open(geometry_dir / "mast3r_elevated.png").convert("RGB"),
            Image.open(geometry_dir / "night3r_elevated.png").convert("RGB"),
            Image.open(geometry_dir / "ground_truth_elevated.png").convert("RGB"),
        ]
    )
    view1 = root / "single_images/view01"
    view2 = root / "single_images/view02"
    metrics = manifest["depth_evaluation"]["pair_view_mean"]
    metric_type = manifest["pair_geometry"].get("metric_type", "Aligned AbsRel")
    if metric_type == "Raw AbsRel":
        mast3r_absrel = metrics["mast3r_raw_absrel"]
        night3r_absrel = metrics["night3r_raw_absrel"]
    else:
        mast3r_absrel = metrics["mast3r_median_aligned_absrel"]
        night3r_absrel = metrics["night3r_median_aligned_absrel"]
    return {
        "label": label,
        "root": root,
        "sample_id": manifest["sample_id"],
        "rgb": [
            Image.open(view1 / "01_RGB_native.png").convert("RGB"),
            Image.open(view2 / "01_RGB_native.png").convert("RGB"),
        ],
        "event": [
            Image.open(view1 / "02_Event_recent4bins_red_blue.png").convert("RGB"),
            Image.open(view2 / "02_Event_recent4bins_red_blue.png").convert("RGB"),
        ],
        "geometry": geometry,
        "dataset_name": manifest.get("dataset_name", "DSEC"),
        "metric_type": metric_type,
        "mast3r_absrel": mast3r_absrel,
        "night3r_absrel": night3r_absrel,
    }


def compose_geometry_only(scenes, output):
    panel_width, panel_height = 410, 245
    left_margin, right_margin = 142, 28
    top, header_height = 24, 60
    row_label_height, metric_height, row_gap = 0, 34, 12
    row_height = panel_height + metric_height
    gap = 14
    width = left_margin + 3 * panel_width + 2 * gap + right_margin
    content_bottom = top + header_height + len(scenes) * row_height + (len(scenes) - 1) * row_gap
    height = content_bottom + 52
    canvas = Image.new("RGB", (width, height), WHITE)
    draw = ImageDraw.Draw(canvas)
    header_font, label_font = font(24, True), font(22, True)
    small_font, metric_font = font(15), font(17)
    headers = ("MASt3R", "Night3R (Ours)", "Ground Truth")
    for column, header in enumerate(headers):
        x = left_margin + column * (panel_width + gap) + panel_width / 2
        draw.text((x, top + 23), header, fill=ACCENT if column == 1 else TEXT, font=header_font, anchor="mm")

    for row, scene in enumerate(scenes):
        y = top + header_height + row * (row_height + row_gap)
        draw.text((22, y + panel_height / 2 - 12), f"Pair {scene['label']}", fill=TEXT, font=label_font)
        draw.text((22, y + panel_height / 2 + 20), scene["dataset_name"], fill=MUTED, font=small_font)
        for column, image in enumerate(scene["geometry"]):
            x = left_margin + column * (panel_width + gap)
            canvas.paste(contain(image, (panel_width, panel_height)), (x, y))
        values = (
            f"Raw AbsRel {scene['mast3r_absrel']:.3f}",
            f"Raw AbsRel {scene['night3r_absrel']:.3f}",
            "Reference geometry",
        )
        for column, value in enumerate(values):
            x = left_margin + column * (panel_width + gap) + panel_width / 2
            draw.text(
                (x, y + panel_height + 17),
                value,
                fill=ACCENT if column == 1 else MUTED,
                font=metric_font,
                anchor="mm",
            )
        if row < len(scenes) - 1:
            line_y = y + row_height + row_gap / 2
            draw.line((18, line_y, width - 18, line_y), fill=SEPARATOR, width=1)
    draw_depth_legend(canvas, content_bottom + 20)
    canvas.save(output)


def compose_full(scenes, output):
    panel_width, panel_height = 300, 218
    left_margin, right_margin = 118, 24
    top, header_height = 24, 60
    metric_height, row_gap, gap = 32, 12, 12
    row_height = panel_height + metric_height
    width = left_margin + 5 * panel_width + 4 * gap + right_margin
    content_bottom = top + header_height + len(scenes) * row_height + (len(scenes) - 1) * row_gap
    height = content_bottom + 50
    canvas = Image.new("RGB", (width, height), WHITE)
    draw = ImageDraw.Draw(canvas)
    header_font, label_font = font(21, True), font(20, True)
    small_font, metric_font = font(14), font(15)
    headers = ("Low-light RGB Pair", "Event Pair", "MASt3R", "Night3R (Ours)", "Ground Truth")
    for column, header in enumerate(headers):
        x = left_margin + column * (panel_width + gap) + panel_width / 2
        draw.text((x, top + 23), header, fill=ACCENT if column == 3 else TEXT, font=header_font, anchor="mm")

    for row, scene in enumerate(scenes):
        y = top + header_height + row * (row_height + row_gap)
        draw.text((18, y + panel_height / 2 - 10), f"Pair {scene['label']}", fill=TEXT, font=label_font)
        draw.text((18, y + panel_height / 2 + 20), scene["dataset_name"], fill=MUTED, font=small_font)
        panels = [
            pair_panel(*scene["rgb"], (panel_width, panel_height)),
            pair_panel(*scene["event"], (panel_width, panel_height)),
            contain(scene["geometry"][0], (panel_width, panel_height)),
            contain(scene["geometry"][1], (panel_width, panel_height)),
            contain(scene["geometry"][2], (panel_width, panel_height)),
        ]
        for column, panel in enumerate(panels):
            x = left_margin + column * (panel_width + gap)
            canvas.paste(panel, (x, y))
        metrics = ("", "", f"Raw AbsRel {scene['mast3r_absrel']:.3f}", f"Raw AbsRel {scene['night3r_absrel']:.3f}", "Reference")
        for column, value in enumerate(metrics):
            if not value:
                continue
            x = left_margin + column * (panel_width + gap) + panel_width / 2
            draw.text(
                (x, y + panel_height + 16),
                value,
                fill=ACCENT if column == 3 else MUTED,
                font=metric_font,
                anchor="mm",
            )
        if row < len(scenes) - 1:
            line_y = y + row_height + row_gap / 2
            draw.line((14, line_y, width - 14, line_y), fill=SEPARATOR, width=1)
    draw_depth_legend(canvas, content_bottom + 18)
    canvas.save(output)


def main():
    args = parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)
    roots = args.scene_root or [root for _, root in DEFAULT_SCENES]
    scenes = [load_scene(chr(ord("A") + index), root) for index, root in enumerate(roots)]
    compose_geometry_only(scenes, args.outdir / "main_geometry_comparison_white.png")
    compose_full(scenes, args.outdir / "main_rgb_event_geometry_comparison_white.png")
    with (args.outdir / "selection_metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=("scene_label", "sample_id", "mast3r_absrel", "night3r_absrel"),
        )
        writer.writeheader()
        for scene in scenes:
            writer.writerow(
                {
                    "scene_label": scene["label"],
                    "sample_id": scene["sample_id"],
                    "mast3r_absrel": f"{scene['mast3r_absrel']:.6f}",
                    "night3r_absrel": f"{scene['night3r_absrel']:.6f}",
                }
            )
    (args.outdir / "main_figure_caption.tex").write_text(
        "\\caption{Qualitative metric-scale geometry on the source-disjoint DSEC "
        "Zurich City 11-b sequence under controlled low-light input. The four "
        "pairs are sampled at pre-registered temporal positions. Camera-local depth predictions "
        "are fused in the first-camera frame using the same ground-truth relative "
        "pose. Within each row, MASt3R, Night3R, and the reference use the same "
        "elevated virtual camera, spatial bounds, and Viridis depth colors from "
        "near (purple) to far (yellow). No scale alignment is applied; the displayed "
        "Raw AbsRel therefore includes metric-scale error.}\n",
        encoding="utf-8",
    )
    (args.outdir / "README.md").write_text(
        "# Main qualitative geometry figure\n\n"
        "The geometry-only and RGB/event-inclusive white-background figures use "
        "the final MVSEC-1k checkpoint on the held-out DSEC Zurich City 11-b sequence. "
        "Each row has a fixed camera and color "
        "scale shared by MASt3R, Night3R, and GT. Camera-local depth maps are "
        "fused with the same GT relative pose, so the figure isolates depth "
        "geometry rather than pose estimation. No scale alignment is applied.\n\n"
        "The source sequence was downloaded and packed after the final checkpoint "
        "was produced, and its overlap audit reports no overlap with the DSEC training sequences.\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
