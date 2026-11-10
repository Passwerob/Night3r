#!/usr/bin/env python3
"""Export the four BRG display-only gate maps as feature-map-only PNGs."""

import json
from pathlib import Path

from PIL import Image


REPO_ROOT = Path(__file__).resolve().parents[1]
SOURCE_PANEL = (
    REPO_ROOT
    / "outputs/brg_stat_sensitivity_gate_contrast_step100_20260726"
    / "dsec_screen-000003-dsec_000023-000024.png"
)
OUTPUT_ROOT = REPO_ROOT / "outputs/brg_gate_feature_maps_only_20260728"

# These coordinates retain only the imshow raster in each first-column gate panel.
# They deliberately exclude the title, axes, white margin, and colour bar.
FEATURE_MAPS = (
    ("full_statistics", "Full statistics", (29, 916, 711, 1598)),
    ("w_o_density", "w/o density", (29, 1701, 711, 2383)),
    ("w_o_occupancy", "w/o occupancy", (29, 2486, 711, 3168)),
    ("w_o_polarity", "w/o polarity", (29, 3271, 711, 3953)),
)


def main():
    if not SOURCE_PANEL.is_file():
        raise FileNotFoundError(SOURCE_PANEL)

    image = Image.open(SOURCE_PANEL).convert("RGB")
    if image.size != (3427, 3975):
        raise RuntimeError(f"Unexpected BRG panel size: {image.size}")

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    records = []
    for stem, label, box in FEATURE_MAPS:
        feature_map = image.crop(box)
        if feature_map.size != (682, 682):
            raise RuntimeError(f"Unexpected crop size for {stem}: {feature_map.size}")
        output_path = OUTPUT_ROOT / f"{stem}_feature_map.png"
        feature_map.save(output_path)
        records.append(
            {
                "variant": label,
                "file": output_path.name,
                "source_crop_xyxy": list(box),
                "pixel_size": list(feature_map.size),
            }
        )

    readme = """# BRG Feature Maps Only

One display-only pre-occupancy BRG gate feature map is exported per statistic
variant for DSEC `screen-000003-dsec:000023-000024`. The PNGs contain only the
feature-map raster: no title, axes, border, or colour bar.

The maps preserve the shared p2--p98 viridis colour normalization from the
source BRG sensitivity panel, so colours remain comparable between variants.
The occupancy hard mask remains part of model inference; these are visual-only
pre-occupancy gate maps.
"""
    (OUTPUT_ROOT / "README.md").write_text(readme, encoding="utf-8")
    manifest = {
        "sample_id": "screen-000003-dsec:000023-000024",
        "dataset": "DSEC",
        "source_panel": str(SOURCE_PANEL.resolve()),
        "feature_map_semantics": (
            "BRG token gate recomputed before the occupancy hard mask for display only. "
            "Model inference preserves the occupancy hard mask."
        ),
        "colour_normalization": "Shared p2--p98 viridis normalization from the source panel.",
        "records": records,
    }
    (OUTPUT_ROOT / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({"output_dir": str(OUTPUT_ROOT), "count": len(records)}, indent=2))


if __name__ == "__main__":
    main()
