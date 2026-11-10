#!/usr/bin/env python3
"""Package severe low-light baseline-collapse scenes for paper review."""

import json
import shutil
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


REPO_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = (
    REPO_ROOT
    / "outputs/night3r_selected_pairs_step100_native_rgb_viridis_20260725/pairs"
)
OUTPUT_ROOT = (
    REPO_ROOT / "outputs/paper_next_scene_shortlist_severe_baseline_20260726"
)

SELECTIONS = (
    {
        "order": 1,
        "dataset": "MVSEC",
        "source_dir": "MVSEC/screen-000156-mvsec_000038-000039",
        "short_name": "mvsec156_000038_000039",
        "recommendation": "strongest geometry-collapse comparison",
    },
    {
        "order": 2,
        "dataset": "MVSEC",
        "source_dir": "MVSEC/screen-000115-mvsec_000026-000027",
        "short_name": "mvsec115_000026_000027",
        "recommendation": "dark road scene with dense event structure",
    },
    {
        "order": 3,
        "dataset": "DSEC",
        "source_dir": "DSEC/screen-000003-dsec_000048-000049",
        "short_name": "dsec003_000048_000049",
        "recommendation": "strict-heldout main quantitative figure candidate",
    },
    {
        "order": 4,
        "dataset": "DSEC",
        "source_dir": "DSEC/screen-000003-dsec_000408-000409",
        "short_name": "dsec003_000408_000409",
        "recommendation": "strict-heldout scale-collapse figure candidate",
    },
)


def load_font(size, bold=False):
    candidates = (
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
        if bold
        else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation2/LiberationSans-Bold.ttf"
        if bold
        else "/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf",
    )
    for candidate in candidates:
        if Path(candidate).is_file():
            return ImageFont.truetype(candidate, size)
    return ImageFont.load_default()


def fit_inside(image, size):
    image = image.copy()
    image.thumbnail(size, Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", size, "white")
    x = (size[0] - image.width) // 2
    y = (size[1] - image.height) // 2
    canvas.paste(image, (x, y))
    return canvas


def selection_panel_overview(records):
    tile_width = 1040
    row_height = 710
    header_height = 68
    canvas = Image.new(
        "RGB",
        (tile_width, header_height + row_height * len(records)),
        "white",
    )
    draw = ImageDraw.Draw(canvas)
    title_font = load_font(26, bold=True)
    row_font = load_font(20, bold=True)
    detail_font = load_font(16)
    draw.text(
        (20, 17),
        "Severe low-light scene shortlist",
        fill=(16, 20, 28),
        font=title_font,
    )
    for row, record in enumerate(records):
        top = header_height + row * row_height
        panel = Image.open(record["selection_panel"]).convert("RGB")
        panel = fit_inside(panel, (tile_width, row_height - 52))
        canvas.paste(panel, (0, top + 52))
        metrics = record["metrics"]
        label = (
            f"{record['order']:02d}  {record['sample_id']}  "
            f"[{record['split_label']}]"
        )
        draw.text((16, top + 6), label, fill=(16, 20, 28), font=row_font)
        detail = (
            f"MASt3R AbsRel {metrics['mast3r_absrel']:.3f}  |  "
            f"Night3R AbsRel {metrics['night3r_absrel']:.3f}"
        )
        draw.text((16, top + 31), detail, fill=(68, 77, 91), font=detail_font)
    output = OUTPUT_ROOT / "00_scene_shortlist_overview.png"
    canvas.save(output, optimize=True)
    return output


def geometry_overview(records, view):
    method_files = (
        ("MASt3R", "mast3r"),
        ("Night3R", "night3r"),
        ("Ground Truth", "gt"),
    )
    tile_size = (720, 540)
    column_header = 54
    row_label = 58
    canvas = Image.new(
        "RGB",
        (
            tile_size[0] * len(method_files),
            column_header + (row_label + tile_size[1]) * len(records),
        ),
        "white",
    )
    draw = ImageDraw.Draw(canvas)
    header_font = load_font(24, bold=True)
    row_font = load_font(19, bold=True)
    for column, (label, _) in enumerate(method_files):
        bbox = draw.textbbox((0, 0), label, font=header_font)
        text_width = bbox[2] - bbox[0]
        x = column * tile_size[0] + (tile_size[0] - text_width) // 2
        draw.text((x, 13), label, fill=(18, 18, 18), font=header_font)

    for row, record in enumerate(records):
        top = column_header + row * (row_label + tile_size[1])
        row_text = (
            f"{record['order']:02d}  {record['sample_id']}  "
            f"[{record['split_label']}]"
        )
        draw.text((18, top + 15), row_text, fill=(18, 18, 18), font=row_font)
        for column, (_, method) in enumerate(method_files):
            source = (
                Path(record["geometry_dir"]) / f"{method}_{view}.png"
            )
            image = Image.open(source).convert("RGB")
            image = fit_inside(image, tile_size)
            canvas.paste(
                image,
                (
                    column * tile_size[0],
                    top + row_label,
                ),
            )
    output = OUTPUT_ROOT / f"0{1 if view == 'front' else 2}_geometry_{view}_overview.png"
    canvas.save(output, optimize=True)
    return output


def compact_metrics(selection):
    metrics = selection["metrics"]
    mast3r = metrics["mast3r"]
    night3r = metrics["night3r"]
    zero = metrics["zero"]
    use_median = selection["dataset"] == "MVSEC"
    prefix = "median" if use_median else "raw"
    return {
        "reported_alignment": prefix,
        "mast3r_absrel": float(mast3r[f"{prefix}_absrel"]),
        "mast3r_delta1": float(mast3r[f"{prefix}_delta1"]),
        "night3r_absrel": float(night3r[f"{prefix}_absrel"]),
        "night3r_delta1": float(night3r[f"{prefix}_delta1"]),
        "zero_absrel": float(zero[f"{prefix}_absrel"]),
        "zero_delta1": float(zero[f"{prefix}_delta1"]),
    }


def main():
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    records = []
    for item in SELECTIONS:
        source = SOURCE_ROOT / item["source_dir"]
        if not source.is_dir():
            raise FileNotFoundError(source)
        destination = OUTPUT_ROOT / f"{item['order']:02d}_{item['short_name']}"
        shutil.copytree(source, destination, dirs_exist_ok=True)

        selection = json.loads(
            (destination / "selection.json").read_text(encoding="utf-8")
        )
        source_assets = destination / "source_assets"
        geometry_dir = destination / "geometry_assets"
        native_sizes = {}
        for name in (
            "clean_rgb_view1.png",
            "clean_rgb_view2.png",
            "lowlight_rgb_view1.png",
            "lowlight_rgb_view2.png",
        ):
            with Image.open(source_assets / name) as image:
                native_sizes[name] = list(image.size)

        record = {
            **item,
            "sample_id": selection["sample_id"],
            "split_label": selection["split_label"],
            "metrics": compact_metrics(selection),
            "recommendation": item["recommendation"],
            "directory": str(destination.relative_to(OUTPUT_ROOT)),
            "selection_panel": str(source_assets / "selection_panel.png"),
            "geometry_dir": str(geometry_dir),
            "native_rgb_sizes_wh": native_sizes,
        }
        records.append(record)

    overview = selection_panel_overview(records)
    front = geometry_overview(records, "front")
    elevated = geometry_overview(records, "elevated")
    manifest = {
        "version": 1,
        "checkpoint": (
            "outputs/night3r_final_step100_20260723/"
            "event_dark3r_night3r_final.pth"
        ),
        "selection_policy": (
            "Severe low-light MASt3R degradation, preserved Night3R structure, "
            "clear event support, low visual noise, and available front/elevated "
            "same-camera geometry."
        ),
        "overview_files": [
            overview.name,
            front.name,
            elevated.name,
        ],
        "pairs": records,
    }
    (OUTPUT_ROOT / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n",
        encoding="utf-8",
    )

    lines = [
        "# Severe low-light scene shortlist",
        "",
        "All four candidates retain original-resolution RGB assets, model-grid "
        "event/depth outputs, and same-camera white-background front/elevated "
        "geometry renders. Viridis is used consistently.",
        "",
        "| Order | Pair | Split | Metric | MASt3R | Night3R | Recommendation |",
        "|---:|---|---|---|---:|---:|---|",
    ]
    for record in records:
        metrics = record["metrics"]
        lines.append(
            f"| {record['order']} | `{record['sample_id']}` | "
            f"{record['split_label']} | {metrics['reported_alignment']} AbsRel | "
            f"{metrics['mast3r_absrel']:.4f} | "
            f"{metrics['night3r_absrel']:.4f} | "
            f"{record['recommendation']} |"
        )
    lines.extend(
        (
            "",
            "The two DSEC pairs are strict held-out candidates and may support "
            "held-out qualitative claims. The two MVSEC pairs overlap the final "
            "adaptation domain and must be described as adaptation-domain "
            "qualitative examples.",
            "",
            "`00_scene_shortlist_overview.png` shows RGB, events and depth. "
            "`01_geometry_front_overview.png` and "
            "`02_geometry_elevated_overview.png` show the shared-view geometry.",
            "",
        )
    )
    (OUTPUT_ROOT / "README.md").write_text(
        "\n".join(lines),
        encoding="utf-8",
    )
    print(json.dumps({"output": str(OUTPUT_ROOT), "pairs": len(records)}))


if __name__ == "__main__":
    main()
