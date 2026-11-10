#!/usr/bin/env python3
"""Package the source RGB and event assets behind BRG display-only figures."""

import argparse
import json
import re
import shutil
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from utils.export_rgb_lowlight_event_assets import event_red_blue, load_event


DEFAULT_BRG_DIR = REPO_ROOT / "outputs/brg_stat_sensitivity_step100_20260726"
DEFAULT_OUTPUT = REPO_ROOT / "outputs/brg_display_only_assets_20260727"
DATASET_ROOTS = {
    "DSEC": Path("/data/fcr/data/train_data/DSEC_test"),
    "MVSEC": Path("/data/fcr/data/train_data/MVSEC_outdoor_night2_test"),
}
SAMPLE_PATTERN = re.compile(r"^(?P<scene>[^:]+):(?P<first>\d+)-(?P<second>\d+)$")


def parse_args():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--brg-dir", type=Path, default=DEFAULT_BRG_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--paper-display-dir",
        type=Path,
        default=REPO_ROOT / "outputs/brg_stat_sensitivity_paper_preoccupancy_step100_20260726",
        help="Directory containing the compact pre-occupancy display-only panel.",
    )
    parser.add_argument("--model-size", type=int, nargs=2, default=(512, 512), metavar=("H", "W"))
    return parser.parse_args()


def load_json(path):
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def parse_sample_id(sample_id):
    match = SAMPLE_PATTERN.match(sample_id)
    if not match:
        raise ValueError(f"Unexpected sample id: {sample_id}")
    return match.group("scene"), int(match.group("first")), int(match.group("second"))


def figure_stem(dataset, sample_id):
    return f"{dataset.lower()}_{sample_id.replace('/', '_').replace(':', '_')}"


def copy_file(source, destination):
    if not source.is_file():
        raise FileNotFoundError(source)
    shutil.copy2(source, destination)


def model_event(event, size_hw):
    if tuple(event.shape[-2:]) == tuple(size_hw):
        return event
    return F.interpolate(event.unsqueeze(0), size=size_hw, mode="bilinear", align_corners=False)[0]


def save_activity(event, output):
    activity = event.float().abs().sum(dim=0).numpy()
    plt.imsave(output, activity, cmap="magma")


def save_rgb_assets(image_path, event_path, output_dir, view, model_size_hw):
    source = Image.open(image_path).convert("RGB")
    source.save(output_dir / f"view{view}_rgb_source.png")
    source.resize((model_size_hw[1], model_size_hw[0]), Image.Resampling.BILINEAR).save(
        output_dir / f"view{view}_rgb_model_input.png"
    )

    event = load_event(event_path)
    event_source_black, _, _ = event_red_blue(event, background="black")
    event_source_white, _, _ = event_red_blue(event, background="white")
    event_source_black.save(output_dir / f"view{view}_event_source_red_blue_black.png")
    event_source_white.save(output_dir / f"view{view}_event_source_red_blue_white.png")

    event_input = model_event(event, model_size_hw)
    event_input_black, _, _ = event_red_blue(event_input, background="black")
    event_input_white, _, _ = event_red_blue(event_input, background="white")
    event_input_black.save(output_dir / f"view{view}_event_model_input_red_blue_black.png")
    event_input_white.save(output_dir / f"view{view}_event_model_input_red_blue_white.png")
    save_activity(event_input, output_dir / f"view{view}_event_model_input_activity_magma.png")

    return {
        "frame_index": int(image_path.stem),
        "source_rgb": str(image_path.resolve()),
        "source_event": str(event_path.resolve()),
        "source_rgb_size_wh": list(source.size),
        "source_event_shape_chw": list(event.shape),
        "model_input_size_hw": list(model_size_hw),
    }


def write_readme(output_dir, records):
    lines = [
        "# BRG Display-Only Source Assets",
        "",
        "Each sample directory is the source pair behind one BRG statistic-sensitivity panel.",
        "`view1` is the view rendered in the BRG panel; `view2` is retained as its paired observation.",
        "",
        "- `brg_stat_sensitivity_display_only.*`: the original 5x4 BRG panel. Its gate maps are recomputed before the occupancy safety mask for visualization only.",
        "- `view*_rgb_source.png`: stored test RGB at its source resolution.",
        "- `view*_rgb_model_input.png`: the RGB after the evaluation loader's 512x512 resize. The BRG render used this image, with extra low-light augmentation disabled.",
        "- `view*_event_*_red_blue_{black,white}.png`: positive events are red and negative events are blue; magnitudes use log1p normalization at the nonzero 99.5 percentile.",
        "- `view*_event_model_input_activity_magma.png`: the absolute event activity used by the BRG panel's top-row event display.",
        "",
        "The BRG figure label `Low-light RGB` is inherited from the renderer. These exports preserve the actual test-time input: no additional synthetic darkening was applied while rendering this sensitivity figure.",
        "",
        "## Samples",
        "",
    ]
    for record in records:
        lines.append(f"- `{record['asset_dir']}`: {record['dataset']} `{record['sample_id']}`")
    (output_dir / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    args = parse_args()
    brg_dir = args.brg_dir.resolve()
    output_dir = args.output_dir.resolve()
    model_size_hw = tuple(args.model_size)
    manifest = load_json(brg_dir / "manifest.json")
    selections = manifest.get("visual_selections", {})
    if not selections:
        raise RuntimeError(f"No visual_selections in {brg_dir / 'manifest.json'}")
    output_dir.mkdir(parents=True, exist_ok=True)
    records = []

    for dataset, items in selections.items():
        dataset_root = DATASET_ROOTS.get(dataset)
        if dataset_root is None:
            raise KeyError(f"No source root registered for {dataset}")
        for sample_id, _batch_index in items:
            scene, first, second = parse_sample_id(sample_id)
            source_dir = dataset_root / scene
            asset_name = f"{dataset.lower()}_{scene}_{first:06d}_{second:06d}"
            asset_dir = output_dir / asset_name
            asset_dir.mkdir(parents=True, exist_ok=True)

            stem = figure_stem(dataset, sample_id)
            copy_file(brg_dir / f"{stem}.png", asset_dir / "brg_stat_sensitivity_display_only.png")
            copy_file(brg_dir / f"{stem}.pdf", asset_dir / "brg_stat_sensitivity_display_only.pdf")

            compact_stem = f"{stem}_compact"
            compact_png = args.paper_display_dir / f"{compact_stem}.png"
            compact_pdf = args.paper_display_dir / f"{compact_stem}.pdf"
            has_compact = compact_png.is_file() and compact_pdf.is_file()
            if has_compact:
                copy_file(compact_png, asset_dir / "brg_preoccupancy_display_only_compact.png")
                copy_file(compact_pdf, asset_dir / "brg_preoccupancy_display_only_compact.pdf")

            views = []
            for view, frame_index in ((1, first), (2, second)):
                image_path = source_dir / "images" / f"{frame_index:06d}.png"
                event_path = source_dir / "events" / f"{frame_index:06d}.pt"
                views.append(save_rgb_assets(image_path, event_path, asset_dir, view, model_size_hw))

            record = {
                "dataset": dataset,
                "sample_id": sample_id,
                "source_scene": str(source_dir.resolve()),
                "asset_dir": asset_name,
                "brg_panel_source": str((brg_dir / f"{stem}.png").resolve()),
                "has_compact_preoccupancy_display": has_compact,
                "visualization_note": (
                    "The BRG display-only gate is computed before the occupancy hard mask. "
                    "Model inference retains the occupancy hard mask."
                ),
                "views": views,
            }
            (asset_dir / "metadata.json").write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
            records.append(record)

    root_manifest = {
        "source_brg_manifest": str((brg_dir / "manifest.json").resolve()),
        "model_input_size_hw": list(model_size_hw),
        "count": len(records),
        "records": records,
    }
    (output_dir / "manifest.json").write_text(json.dumps(root_manifest, indent=2) + "\n", encoding="utf-8")
    write_readme(output_dir, records)
    print(json.dumps({"output_dir": str(output_dir), "count": len(records)}, indent=2))


if __name__ == "__main__":
    main()
