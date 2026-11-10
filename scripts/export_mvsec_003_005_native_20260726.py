#!/usr/bin/env python3
"""Export selected MVSEC pairs 003/005 at the native 346x260 resolution."""

import json
import random
import shutil
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from dataset.lowlight_augmentation import ContinuousLowLightAug
SOURCE_ROOT = REPO_ROOT / "test/MVSEC"
SELECTION_ROOT = (
    REPO_ROOT / "outputs/mvsec_collapse_event_rich_selected_20260726"
)
OUTPUT_ROOT = (
    REPO_ROOT / "outputs/mvsec_selected_003_005_native_20260726"
)
SELECTIONS = (
    {
        "order": "003",
        "dataset_index": 516,
        "scene": "screen-000157-mvsec",
        "frames": (36, 37),
        "selection_dir": "03_screen-000157-mvsec_000036_000037",
    },
    {
        "order": "005",
        "dataset_index": 284,
        "scene": "screen-000122-mvsec",
        "frames": (44, 45),
        "selection_dir": "05_screen-000122-mvsec_000044_000045",
    },
)
SWEEP_SEED = 17
EXPOSURE = 0.08
READ_NOISE_STD = 3.0e-5
SHOT_NOISE_STD = 1.0e-4
EVENT_DISPLAY_QUANTILE = 0.985
EVENT_DISPLAY_GAMMA = 0.55


def load_rgb(path):
    with Image.open(path) as image:
        array = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
    return torch.from_numpy(array).permute(2, 0, 1).contiguous()


def save_rgb(tensor, path):
    array = (
        tensor.detach()
        .float()
        .clamp(0, 1)
        .permute(1, 2, 0)
        .mul(255)
        .round()
        .byte()
        .numpy()
    )
    Image.fromarray(array).save(path)


def load_event(path):
    try:
        event = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        event = torch.load(path, map_location="cpu")
    if not torch.is_tensor(event) or event.ndim != 3:
        raise ValueError(f"Invalid event tensor: {path}")
    return event.float()


def event_activity(event):
    half = event.shape[0] // 2
    positive = event[:half].abs().sum(dim=0)
    negative = event[half:].abs().sum(dim=0)
    return positive, negative


def enhanced_event_strength(activity):
    transformed = torch.log1p(activity)
    nonzero = transformed[transformed > 0]
    if nonzero.numel() == 0:
        return torch.zeros_like(activity)
    scale = torch.quantile(
        nonzero, EVENT_DISPLAY_QUANTILE
    ).clamp_min(1.0e-6)
    return (transformed / scale).clamp(0, 1).pow(EVENT_DISPLAY_GAMMA)


def enhanced_event_red_blue(event, background):
    positive, negative = event_activity(event)
    pos = enhanced_event_strength(positive)
    neg = enhanced_event_strength(negative)
    strength = torch.maximum(pos, neg)
    total = (pos + neg).clamp_min(1.0e-6)
    color = torch.stack(
        (pos / total, torch.zeros_like(pos), neg / total), dim=0
    )
    if background == "black":
        rgb = strength.unsqueeze(0) * color
    elif background == "white":
        rgb = (1.0 - strength.unsqueeze(0)) + strength.unsqueeze(0) * color
    else:
        raise ValueError(f"Unsupported event background: {background}")
    return rgb


def faithful_event_red_blue(event, background):
    """Render native event support as one colored pixel per active location."""
    positive, negative = event_activity(event)
    active = (positive > 0) | (negative > 0)
    positive_dominant = active & (positive >= negative)
    negative_dominant = active & (negative > positive)
    if background == "black":
        rgb = torch.zeros((3, *active.shape), dtype=torch.float32)
    elif background == "white":
        rgb = torch.ones((3, *active.shape), dtype=torch.float32)
    else:
        raise ValueError(f"Unsupported event background: {background}")
    rgb[:, active] = 0.0
    rgb[0, positive_dominant] = 1.0
    rgb[2, negative_dominant] = 1.0
    return rgb


def native_lowlight(rgb_pair, dataset_index):
    local_seed = SWEEP_SEED + int(dataset_index) * 1009
    random.seed(local_seed)
    np.random.seed(local_seed % (2**32 - 1))
    torch_state = torch.random.get_rng_state()
    torch.manual_seed(local_seed)
    try:
        augment = ContinuousLowLightAug(
            min_exposure=EXPOSURE,
            max_exposure=EXPOSURE,
            read_noise_std=READ_NOISE_STD,
            shot_noise=True,
            shot_noise_std=SHOT_NOISE_STD,
            gamma_space_input=True,
        )
        return augment(rgb_pair), local_seed
    finally:
        torch.random.set_rng_state(torch_state)


def verify_size(path, expected=(346, 260)):
    with Image.open(path) as image:
        if image.size != expected:
            raise RuntimeError(
                f"{path} has size {image.size}; expected native MVSEC {expected}"
            )


def main():
    if OUTPUT_ROOT.exists():
        shutil.rmtree(OUTPUT_ROOT)
    OUTPUT_ROOT.mkdir(parents=True)
    records = []

    for selection in SELECTIONS:
        pair_name = (
            f"{selection['order']}_{selection['scene']}_"
            f"{selection['frames'][0]:06d}_{selection['frames'][1]:06d}"
        )
        pair_root = OUTPUT_ROOT / pair_name
        native_root = pair_root / "native_346x260"
        model_root = pair_root / "model_outputs_512x512"
        native_root.mkdir(parents=True)
        model_root.mkdir()

        source_scene = SOURCE_ROOT / selection["scene"]
        rgb_tensors = []
        source_records = []
        for view, frame in enumerate(selection["frames"], start=1):
            rgb_source = source_scene / "images" / f"{frame:06d}.png"
            event_source = source_scene / "events" / f"{frame:06d}.pt"
            rgb_output = native_root / f"view{view}_RGB_original.png"
            event_tensor_output = native_root / f"view{view}_Event_original.pt"
            event_white_output = (
                native_root / f"view{view}_Event_red_blue_white_enhanced.png"
            )
            event_black_output = (
                native_root / f"view{view}_Event_red_blue_black_enhanced.png"
            )
            event_main_white_output = (
                native_root / f"view{view}_Event_main_white.png"
            )
            event_main_black_output = (
                native_root / f"view{view}_Event_main_black.png"
            )

            shutil.copy2(rgb_source, rgb_output)
            shutil.copy2(event_source, event_tensor_output)
            verify_size(rgb_output)

            rgb_tensors.append(load_rgb(rgb_source))
            event = load_event(event_source)
            save_rgb(
                enhanced_event_red_blue(event, background="white"),
                event_white_output,
            )
            save_rgb(
                enhanced_event_red_blue(event, background="black"),
                event_black_output,
            )
            save_rgb(
                faithful_event_red_blue(event, background="white"),
                event_main_white_output,
            )
            save_rgb(
                faithful_event_red_blue(event, background="black"),
                event_main_black_output,
            )
            verify_size(event_white_output)
            verify_size(event_black_output)
            verify_size(event_main_white_output)
            verify_size(event_main_black_output)
            source_records.append(
                {
                    "view": view,
                    "frame": frame,
                    "rgb_source": str(rgb_source),
                    "event_source": str(event_source),
                    "event_shape": list(event.shape),
                }
            )

        rgb_pair = torch.stack(rgb_tensors)
        dark_pair, local_seed = native_lowlight(
            rgb_pair, selection["dataset_index"]
        )
        for view, dark in enumerate(dark_pair, start=1):
            dark_path = native_root / f"view{view}_DarkenRGB_native.png"
            save_rgb(dark, dark_path)
            verify_size(dark_path)

        selection_assets = (
            SELECTION_ROOT / selection["selection_dir"] / "visualization"
        )
        for name in (
            "master_depth_shared.png",
            "night_depth_shared.png",
            "zero_depth_shared.png",
            "gt_depth.png",
            "geometry_arrays.npz",
            "geometry_points.npz",
            "metrics.json",
        ):
            source = selection_assets / name
            if source.is_file():
                shutil.copy2(source, model_root / name)

        record = {
            **selection,
            "pair_directory": pair_name,
            "native_resolution_wh": [346, 260],
            "source_files": source_records,
            "darkening": {
                "exposure": EXPOSURE,
                "read_noise_std": READ_NOISE_STD,
                "shot_noise_std": SHOT_NOISE_STD,
                "gamma_space_input": True,
                "sweep_seed": SWEEP_SEED,
                "pair_local_seed": local_seed,
            },
            "event_visualization": {
                "positive_color": "red",
                "negative_color": "blue",
                "backgrounds": ["white", "black"],
                "paper_main": {
                    "filenames": "view{1,2}_Event_main_{white,black}.png",
                    "mapping": (
                        "binary native event support; one output pixel per active "
                        "sensor location; dominant accumulated polarity selects "
                        "red or blue"
                    ),
                    "magnitude_enhancement": "none",
                    "spatial_filtering": "none",
                    "resizing": "none",
                },
                "diagnostic_enhanced": {
                    "filenames": (
                        "view{1,2}_Event_red_blue_{white,black}_enhanced.png"
                    ),
                    "magnitude_transform": "log1p",
                    "nonzero_quantile": EVENT_DISPLAY_QUANTILE,
                    "display_gamma": EVENT_DISPLAY_GAMMA,
                },
                "spatial_filtering": "none",
                "note": (
                    "Use Event_main files in the paper figure. Enhanced files "
                    "are display diagnostics only. The saved event tensor is "
                    "unchanged."
                ),
            },
            "resolution_note": (
                "Files under native_346x260 are generated directly from the "
                "native MVSEC RGB/event tensors without spatial resizing. Files "
                "under model_outputs_512x512 remain model-grid predictions."
            ),
        }
        (pair_root / "manifest.json").write_text(
            json.dumps(record, indent=2) + "\n", encoding="utf-8"
        )
        records.append(record)

    (OUTPUT_ROOT / "manifest.json").write_text(
        json.dumps({"version": 1, "pairs": records}, indent=2) + "\n",
        encoding="utf-8",
    )
    (OUTPUT_ROOT / "README.md").write_text(
        "# MVSEC selections 003 and 005 at native resolution\n\n"
        "Every PNG under `native_346x260/` is exactly 346x260. The original "
        "RGB files are byte-for-byte copies from `test/MVSEC`; dark RGB is "
        "generated at native resolution, and both white- and black-background "
        "event renders use the native 8x260x346 voxel tensors. Use "
        "`Event_main_white.png` or `Event_main_black.png` for the paper: these "
        "are binary native-support renders with no magnitude enhancement, "
        "spatial filtering, or resizing. Files ending in `_enhanced.png` are "
        "diagnostic alternatives only. Model predictions remain separately "
        "stored under `model_outputs_512x512/`.\n",
        encoding="utf-8",
    )
    print(json.dumps({"output": str(OUTPUT_ROOT), "pairs": len(records)}))


if __name__ == "__main__":
    main()
