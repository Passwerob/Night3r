#!/usr/bin/env python3
"""Replace selected-pair RGB assets with native-resolution source images."""

import argparse
import csv
import json
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


DEFAULT_SOURCE = (
    REPO_ROOT / "outputs/night3r_selected_pairs_step100_20260724"
)
DEFAULT_OUTPUT = (
    REPO_ROOT / "outputs/night3r_selected_pairs_step100_native_rgb_20260725"
)
TEST_ROOT = REPO_ROOT / "test"
DSEC_RAW_ROOTS = (
    Path("/data/fcr/code/fcr/StreamVGGT/data/test"),
    Path("/data/fcr/code/event_vggt/Data_Part/StreamVGGT/data/test"),
    Path("/data/fcr/code/event_vggt/Data_Part/Original/StreamVGGT/data/test"),
)
LOWLIGHT_BY_DATASET = {
    "DSEC": {
        "exposure": 0.012,
        "read_noise_std": 3.0e-5,
        "shot_noise_std": 1.0e-4,
    },
    "MVSEC": {
        "exposure": 0.080,
        "read_noise_std": 3.0e-5,
        "shot_noise_std": 1.0e-4,
    },
    "DL3DV": {
        "exposure": 0.008,
        "read_noise_std": 3.0e-5,
        "shot_noise_std": 1.0e-4,
    },
}
SWEEP_SEED = 17


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--skip-copy",
        action="store_true",
        help="Modify an already copied output directory.",
    )
    return parser.parse_args()


def image_tensor(image):
    array = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
    return torch.from_numpy(array).permute(2, 0, 1).contiguous()


def tensor_image(tensor):
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
    return Image.fromarray(array)


def dsec_source_path(scene, frame_index):
    scene_dir = TEST_ROOT / "DSEC" / scene
    with (scene_dir / "meta.json").open() as handle:
        scene_meta = json.load(handle)
    sequence = scene_meta["sequence"]

    source_file = None
    with (scene_dir / "meta/frames_meta.csv").open(newline="") as handle:
        for row in csv.DictReader(handle):
            if int(row["out_idx"]) == int(frame_index):
                source_file = row["source_file"]
                break
    if source_file is None:
        raise KeyError(f"No DSEC source mapping for {scene}:{frame_index}")

    candidates = [
        root / sequence / f"{sequence}_images_rectified_left" / source_file
        for root in DSEC_RAW_ROOTS
    ]
    for candidate in candidates:
        if candidate.is_file():
            return candidate, {
                "sequence": sequence,
                "processed_frame_index": int(frame_index),
                "source_frame": source_file,
            }
    raise FileNotFoundError(
        f"Native DSEC frame {sequence}/{source_file} was not found in "
        + ", ".join(str(path) for path in DSEC_RAW_ROOTS)
    )


def native_source_path(dataset, scene, frame_index):
    if dataset == "DSEC":
        return dsec_source_path(scene, frame_index)
    source = TEST_ROOT / dataset / scene / "images" / f"{frame_index:06d}.png"
    if not source.is_file():
        raise FileNotFoundError(source)
    return source, {
        "processed_frame_index": int(frame_index),
        "source_frame": source.name,
    }


def model512_consistency(native_image, model_image_path):
    with Image.open(model_image_path) as model_image:
        model = np.asarray(model_image.convert("RGB"), dtype=np.float32)
    resized = native_image.convert("RGB").resize((512, 512), Image.BILINEAR)
    resized_array = np.asarray(resized, dtype=np.float32)
    error = np.abs(resized_array - model)
    return {
        "resized_to_model512_mae_uint8": float(error.mean()),
        "resized_to_model512_max_uint8": float(error.max()),
    }


def pair_output_dir(output, item):
    pair_name = (
        f"{item['scene']}_{int(item['idx1']):06d}-{int(item['idx2']):06d}"
    )
    return output / "pairs" / item["dataset"] / pair_name / "source_assets"


def save_native_pair(output, item):
    dataset = item["dataset"]
    indices = (int(item["idx1"]), int(item["idx2"]))
    source_paths = []
    source_details = []
    native_images = []
    for frame_index in indices:
        path, details = native_source_path(dataset, item["scene"], frame_index)
        source_paths.append(path)
        source_details.append(details)
        with Image.open(path) as image:
            native_images.append(image.convert("RGB").copy())

    output_dir = pair_output_dir(output, item)
    if not output_dir.is_dir():
        raise FileNotFoundError(output_dir)

    consistency = []
    for view, (source_path, native_image) in enumerate(
        zip(source_paths, native_images), start=1
    ):
        clean_path = output_dir / f"clean_rgb_view{view}.png"
        lowlight_path = output_dir / f"lowlight_rgb_view{view}.png"
        model_clean_path = output_dir / f"clean_rgb_view{view}_model512.png"
        model_lowlight_path = output_dir / f"lowlight_rgb_view{view}_model512.png"
        if not model_clean_path.exists():
            shutil.copy2(clean_path, model_clean_path)
        if not model_lowlight_path.exists():
            shutil.copy2(lowlight_path, model_lowlight_path)
        consistency.append(
            model512_consistency(native_image, model_clean_path)
        )
        shutil.copy2(source_path, clean_path)

    clean_pair = torch.stack([image_tensor(image) for image in native_images])
    params = LOWLIGHT_BY_DATASET[dataset]
    dataset_index = item.get("dataset_index")
    if dataset_index is None:
        with (output_dir / "metrics.json").open() as handle:
            dataset_index = json.load(handle)["dataset_index"]
    local_seed = SWEEP_SEED + int(dataset_index) * 1009
    rng_state = torch.random.get_rng_state()
    torch.manual_seed(local_seed)
    try:
        augment = ContinuousLowLightAug(
            min_exposure=params["exposure"],
            max_exposure=params["exposure"],
            read_noise_std=params["read_noise_std"],
            shot_noise=True,
            shot_noise_std=params["shot_noise_std"],
            gamma_space_input=True,
        )
        dark_pair = augment(clean_pair)
    finally:
        torch.random.set_rng_state(rng_state)

    for view in range(2):
        tensor_image(dark_pair[view]).save(
            output_dir / f"lowlight_rgb_view{view + 1}.png"
        )

    width, height = native_images[0].size
    return {
        "id": item["id"],
        "dataset": dataset,
        "scene": item["scene"],
        "indices": list(indices),
        "dataset_index": int(dataset_index),
        "native_rgb_size_wh": [width, height],
        "native_rgb_sources": [str(path) for path in source_paths],
        "source_mapping": source_details,
        "lowlight": {
            **params,
            "gamma_space_input": True,
            "seed": SWEEP_SEED,
            "pair_local_seed": local_seed,
            "applied_jointly_to_both_native_views": True,
        },
        "model512_consistency": consistency,
        "event_depth_geometry_note": (
            "Event visualizations, depth maps, geometry arrays, and selection "
            "panels remain model-grid outputs and are not native RGB assets."
        ),
    }


def main():
    args = parse_args()
    source = args.source.resolve()
    output = args.output.resolve()
    if not args.skip_copy:
        if output.exists():
            raise FileExistsError(output)
        shutil.copytree(source, output, copy_function=shutil.copy2)
    elif not output.is_dir():
        raise FileNotFoundError(output)

    manifest_path = output / "metadata/kept_manifest.json"
    with manifest_path.open() as handle:
        kept_manifest = json.load(handle)
    records = [save_native_pair(output, item) for item in kept_manifest["items"]]

    native_manifest = {
        "version": 1,
        "source_package": str(source),
        "output_package": str(output),
        "canonical_rgb_policy": (
            "clean_rgb_view*.png and lowlight_rgb_view*.png are native "
            "resolution. The previous 512-by-512 exports are retained with "
            "the _model512 suffix."
        ),
        "records": records,
    }
    with (output / "metadata/native_rgb_manifest.json").open("w") as handle:
        json.dump(native_manifest, handle, indent=2)

    readme = output / "README_NATIVE_RGB.md"
    readme.write_text(
        "# Native-resolution RGB correction\n\n"
        "The canonical `clean_rgb_view*.png` and `lowlight_rgb_view*.png` "
        "files in every selected pair preserve the source resolution and "
        "aspect ratio:\n\n"
        "- DSEC: 1440 x 1080\n"
        "- MVSEC: 346 x 260\n"
        "- DL3DV: 960 x 540\n\n"
        "The prior 512 x 512 RGB exports are retained as "
        "`*_model512.png`; they are model-input diagnostics, not source "
        "RGB figure assets. Event visualizations, predicted depth, and 3D "
        "geometry remain on the model grid. Exact frame provenance and "
        "darkening parameters are recorded in "
        "`metadata/native_rgb_manifest.json`.\n"
    )
    print(
        json.dumps(
            {
                "output": str(output),
                "pairs": len(records),
                "sizes": {
                    dataset: sorted(
                        {
                            tuple(record["native_rgb_size_wh"])
                            for record in records
                            if record["dataset"] == dataset
                        }
                    )
                    for dataset in ("DSEC", "MVSEC", "DL3DV")
                },
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
