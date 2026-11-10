#!/usr/bin/env python3
"""Extract the BRG display-only gate maps and their matching RGB/event assets."""

import json
import shutil
from pathlib import Path

from PIL import Image


REPO_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ASSETS = REPO_ROOT / "outputs/brg_display_only_assets_20260727/dsec_screen-000003-dsec_000023_000024"
SOURCE_PANEL = (
    REPO_ROOT
    / "outputs/brg_stat_sensitivity_gate_contrast_step100_20260726"
    / "dsec_screen-000003-dsec_000023-000024.png"
)
OUTPUT_ROOT = REPO_ROOT / "outputs/brg_gate_map_display_only_assets_20260727"

# Pixel boxes in SOURCE_PANEL. Each retains the rendered title, gate map, and colorbar.
PANELS = (
    ("full_statistics", "Full statistics", (0, 840, 900, 1610)),
    ("w_o_density", "w/o density", (0, 1620, 900, 2390)),
    ("w_o_occupancy", "w/o occupancy", (0, 2400, 900, 3170)),
    ("w_o_polarity", "w/o polarity", (0, 3180, 900, 3950)),
)
COMMON_ASSETS = (
    "view1_rgb_source.png",
    "view1_rgb_model_input.png",
    "view1_event_source_red_blue_black.png",
    "view1_event_source_red_blue_white.png",
    "view1_event_model_input_red_blue_black.png",
    "view1_event_model_input_red_blue_white.png",
    "view1_event_model_input_activity_magma.png",
)


def copy(source, destination):
    if not source.is_file():
        raise FileNotFoundError(source)
    shutil.copy2(source, destination)


def main():
    if not SOURCE_PANEL.is_file():
        raise FileNotFoundError(SOURCE_PANEL)
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    image = Image.open(SOURCE_PANEL).convert("RGB")
    if image.size != (3427, 3975):
        raise RuntimeError(f"Unexpected BRG panel size: {image.size}")

    records = []
    for directory_name, label, crop_box in PANELS:
        output_dir = OUTPUT_ROOT / directory_name
        output_dir.mkdir(parents=True, exist_ok=True)
        image.crop(crop_box).save(output_dir / "brg_gate_map_display_only.png")
        for filename in COMMON_ASSETS:
            copy(SOURCE_ASSETS / filename, output_dir / filename)
        metadata = {
            "sample_id": "screen-000003-dsec:000023-000024",
            "dataset": "DSEC",
            "variant": label,
            "source_panel": str(SOURCE_PANEL.resolve()),
            "source_crop_xyxy": list(crop_box),
            "gate_map_semantics": (
                "BRG token gate recomputed before the occupancy hard mask for display only. "
                "The actual model inference preserves the occupancy hard mask."
            ),
            "rgb_event_view": "view1, the view displayed in the BRG panel",
        }
        (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
        records.append({"directory": directory_name, **metadata})

    readme = """# BRG Display-Only Gate Maps

These are the first-column gate maps from the established BRG statistic-sensitivity
figure for DSEC `screen-000003-dsec:000023-000024`. Each subdirectory contains one
gate map plus the exact view-1 RGB and event images used by the panel.

`w_o_density`, `w_o_occupancy`, and `w_o_polarity` each mask one normalized BRG
statistic before the gate MLP. The gate maps are display-only pre-occupancy maps;
the model's inference path still applies the occupancy hard mask.

Positive events are red and negative events are blue.
"""
    (OUTPUT_ROOT / "README.md").write_text(readme, encoding="utf-8")
    (OUTPUT_ROOT / "manifest.json").write_text(
        json.dumps({"count": len(records), "records": records}, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"output_dir": str(OUTPUT_ROOT), "count": len(records)}, indent=2))


if __name__ == "__main__":
    main()
