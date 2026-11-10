#!/usr/bin/env python3
"""Run a matched low-light stress comparison on selected DSEC/DL3DV pairs."""

import argparse
import csv
import json
import random
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFont
from torch.utils.data._utils.collate import default_collate


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from dataset.dsec_event_dataset import DSECEventDataset  # noqa: E402
from model.event_dark3r_student import make_view_pair_from_batch  # noqa: E402
from train_event_dark3r import load_config, move_batch_to_device  # noqa: E402
from utils.compare_event_dark3r_checkpoint import depth_from_pred  # noqa: E402
from utils.eval_event_dark3r_variant import build_model  # noqa: E402
from utils.eval_vggt_baseline import load_vggt, prediction_depth, preprocess_vggt  # noqa: E402
from utils.export_selected_dark3r_vggt_baselines import (  # noqa: E402
    DEFAULT_CONFIG,
    DEFAULT_DARK3R_LORA,
    DEFAULT_MAST3R,
    DEFAULT_SELECTED_ROOT,
    DEFAULT_VGGT,
    DEFAULT_VGGT_ROOT,
    backproject,
    crop_points,
    depth_metrics,
    discrepancy_metrics,
    render_geometry,
    robust_volume,
    save_depth_png,
    sha256,
    tensor_rgb,
    tensor_sha256,
    valid_depth,
)
from utils.sweep_all_test_visual_candidates import robust_dark_noise  # noqa: E402


DEFAULT_OUTPUT_ROOT = (
    REPO_ROOT / "outputs/selected_lowlight_stress_A_20260725"
)
DEFAULT_NIGHT3R = (
    DEFAULT_SELECTED_ROOT / "weights/event_dark3r_night3r_final_step100.pth"
)
METHODS = ("mast3r", "dark3r", "vggt", "night3r")
DISPLAY_NAMES = {
    "mast3r": "MASt3R",
    "dark3r": "Dark3R",
    "vggt": "VGGT",
    "night3r": "Night3R",
}
STRESS_PROTOCOL = {
    "DSEC": {
        "exposure": 0.004,
        "read_noise_std": 8.0e-5,
        "shot_noise_std": 2.0e-4,
    },
    "DL3DV": {
        "exposure": 0.0025,
        "read_noise_std": 8.0e-5,
        "shot_noise_std": 2.0e-4,
    },
}


def parse_args():
    parser = argparse.ArgumentParser("Selected-pair low-light stress comparison")
    parser.add_argument("--selected-root", type=Path, default=DEFAULT_SELECTED_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--variant", choices=METHODS, default=None)
    parser.add_argument("--finalize", action="store_true")
    parser.add_argument("--datasets", nargs="+", choices=("DSEC", "DL3DV"), default=["DSEC", "DL3DV"])
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--mast3r-checkpoint", type=Path, default=DEFAULT_MAST3R)
    parser.add_argument("--dark3r-lora-checkpoint", type=Path, default=DEFAULT_DARK3R_LORA)
    parser.add_argument("--vggt-checkpoint", type=Path, default=DEFAULT_VGGT)
    parser.add_argument("--vggt-root", type=Path, default=DEFAULT_VGGT_ROOT)
    parser.add_argument("--night3r-checkpoint", type=Path, default=DEFAULT_NIGHT3R)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--amp-dtype", choices=("bf16", "fp16"), default="bf16")
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    if args.finalize == (args.variant is not None):
        parser.error("choose exactly one of --variant or --finalize")
    return args


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_datasets(datasets):
    result = {}
    for dataset_name in datasets:
        protocol = STRESS_PROTOCOL[dataset_name]
        result[dataset_name] = DSECEventDataset(
            root_path=str(REPO_ROOT / "test" / dataset_name),
            image_size=(512, 512),
            pair_stride=1,
            pair_sampling="fixed",
            expected_event_channels=8,
            lowlight_enabled=True,
            lowlight_kwargs={
                "min_exposure": protocol["exposure"],
                "max_exposure": protocol["exposure"],
                "read_noise_std": protocol["read_noise_std"],
                "shot_noise": True,
                "shot_noise_std": protocol["shot_noise_std"],
                "gamma_space_input": True,
            },
        )
    return result


def pair_directories(selected_root, datasets):
    pairs = []
    for dataset in datasets:
        pairs.extend(sorted((selected_root / "pairs" / dataset).glob("*")))
    return [path for path in pairs if path.is_dir()]


def deterministic_batch(dataset, dataset_index, seed):
    local_seed = int(seed + dataset_index * 1009)
    random.seed(local_seed)
    np.random.seed(local_seed % (2**32 - 1))
    torch.manual_seed(local_seed)
    return default_collate([dataset[dataset_index]])


def load_pair_batch(pair_dir, datasets, seed):
    record = json.loads(
        (pair_dir / "source_assets/metrics.json").read_text(encoding="utf-8")
    )
    batch = deterministic_batch(
        datasets[record["dataset"]], int(record["dataset_index"]), seed
    )
    if str(batch["sample_id"][0]) != str(record["sample_id"]):
        raise RuntimeError(
            f"Dataset ordering changed: {batch['sample_id'][0]} != {record['sample_id']}"
        )
    return batch, record


def model_namespaces(args, variant):
    train_args = SimpleNamespace(
        config=str(args.config),
        data_root=None,
        save_dir=str(args.output_root),
        model_name_or_path=str(args.mast3r_checkpoint),
        dark3r_lora_checkpoint=(
            str(args.dark3r_lora_checkpoint) if variant == "dark3r" else None
        ),
        backbone_lora_checkpoint=None,
        allow_unsafe_checkpoint_load=True,
        use_tiny_model=False,
        image_size=[512, 512],
        batch_size=1,
        max_batches=None,
        num_epochs=None,
        max_steps=None,
        max_scenes=None,
        max_pairs_per_scene=None,
        pair_stride=1,
        pair_sampling="fixed",
        min_pair_distance=1,
        max_pair_distance=1,
        precomputed_pairs=None,
        allow_reverse_pairs=False,
        symmetric_pairs=False,
        num_workers=0,
        stage=None,
        partial_finetune_blocks=None,
        amp=True,
        amp_dtype=args.amp_dtype,
        gradient_checkpointing=False,
        device=args.device,
        log_every=1,
        no_save=True,
        seed=args.seed,
        depth_min=None,
        depth_max=None,
    )
    model_args = SimpleNamespace(
        variant=(
            "dark3r_lora"
            if variant == "dark3r"
            else "event_dark3r"
            if variant == "night3r"
            else "mast3r"
        ),
        checkpoint=str(args.night3r_checkpoint) if variant == "night3r" else None,
        geometry_head_checkpoint=None,
        config=str(args.config),
        data_root=None,
        model_name_or_path=str(args.mast3r_checkpoint),
        dark3r_lora_checkpoint=(
            str(args.dark3r_lora_checkpoint) if variant == "dark3r" else None
        ),
        allow_unsafe_checkpoint_load=True,
        restore_frozen_rgb=False,
        restore_frozen_rgb_scope="none",
        restore_frozen_rgb_alpha=1.0,
    )
    return train_args, model_args


def load_variant_model(args, device):
    if args.variant == "vggt":
        vggt_args = SimpleNamespace(
            vggt_root=str(args.vggt_root),
            checkpoint=str(args.vggt_checkpoint),
        )
        return load_vggt(vggt_args, device)
    cfg = load_config(args.config)
    train_args, model_args = model_namespaces(args, args.variant)
    return build_model(model_args, cfg, train_args, device)


def resize_depth(depth):
    if depth.shape[-2:] != (512, 512):
        depth = F.interpolate(
            depth,
            size=(512, 512),
            mode="bilinear",
            align_corners=False,
        )
    return depth[0, 0].detach().float().cpu().numpy().astype(np.float32)


def infer_rgb_pair(model, normalized_images):
    view1, view2 = make_view_pair_from_batch(
        {"images_dark": normalized_images}, prefer_dark=True
    )
    prediction, _ = model(view1, view2)
    depth = depth_from_pred(prediction)
    if depth is None:
        raise RuntimeError("RGB reconstruction model returned no depth")
    return depth


def infer_variant(args, model, batch, device):
    dtype = torch.bfloat16 if args.amp_dtype == "bf16" else torch.float16
    with torch.inference_mode(), torch.autocast(
        device_type=device.type,
        dtype=dtype,
        enabled=device.type == "cuda",
    ):
        if args.variant == "night3r":
            batch_gpu = move_batch_to_device(batch, device)
            prediction, _ = model(batch_gpu)
            depth = depth_from_pred(prediction)
            clean_reference = None
        elif args.variant == "vggt":
            images = batch["images_photo_dark"].to(device)
            prediction = model(images=preprocess_vggt(images))
            depth = prediction_depth(prediction)
            clean_reference = None
        else:
            images_dark = batch["images_dark"].to(device)
            depth = infer_rgb_pair(model, images_dark)
            clean_reference = None
            if args.variant == "mast3r":
                clean_images = batch["images"].to(device)
                clean_reference = infer_rgb_pair(model, clean_images)
    if depth is None:
        raise RuntimeError(f"{args.variant} returned no depth")
    result = {"stress_depth": resize_depth(depth)}
    if clean_reference is not None:
        result["clean_reference_depth"] = resize_depth(clean_reference)
    return result


def process_variant(args, model, load_info, datasets, device):
    records = []
    for pair_dir in pair_directories(args.selected_root, args.datasets):
        batch, source_record = load_pair_batch(pair_dir, datasets, args.seed)
        output_dir = (
            args.output_root
            / "pairs"
            / source_record["dataset"]
            / pair_dir.name
            / args.variant
        )
        metadata_path = output_dir / "inference.json"
        if metadata_path.is_file() and not args.force:
            records.append(json.loads(metadata_path.read_text(encoding="utf-8")))
            continue
        output_dir.mkdir(parents=True, exist_ok=True)
        outputs = infer_variant(args, model, batch, device)
        np.save(output_dir / "depth_raw.npy", outputs["stress_depth"])
        if "clean_reference_depth" in outputs:
            np.save(
                output_dir / "clean_reference_depth.npy",
                outputs["clean_reference_depth"],
            )

        images = batch["images_photo_dark"].float()
        input_record = {
            "dataset": source_record["dataset"],
            "scene": source_record["scene"],
            "sample_id": source_record["sample_id"],
            "pair_name": pair_dir.name,
            "variant": args.variant,
            "display_name": DISPLAY_NAMES[args.variant],
            "dataset_index": int(source_record["dataset_index"]),
            "seed": int(args.seed),
            "stress_protocol": STRESS_PROTOCOL[source_record["dataset"]],
            "input_float_sha256": tensor_sha256(images),
            "input_mean_both_views": float(images.mean()),
            "input_view1_mean": float(images[0, 0].mean()),
            "input_view1_noise": robust_dark_noise(images[0, 0]),
            "model_load": load_info,
            "outputs": {
                "depth_raw": "depth_raw.npy",
                "clean_reference_depth": (
                    "clean_reference_depth.npy"
                    if "clean_reference_depth" in outputs
                    else None
                ),
            },
        }
        metadata_path.write_text(
            json.dumps(input_record, indent=2) + "\n", encoding="utf-8"
        )
        records.append(input_record)
        print(
            f"[{args.variant}] {source_record['dataset']} "
            f"{source_record['sample_id']}",
            flush=True,
        )
    manifest = args.output_root / f"{args.variant}_manifest.json"
    manifest.parent.mkdir(parents=True, exist_ok=True)
    manifest.write_text(
        json.dumps(
            {
                "variant": args.variant,
                "stress_protocol": STRESS_PROTOCOL,
                "records": records,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


def load_font(size):
    path = Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf")
    return ImageFont.truetype(str(path), size=size) if path.is_file() else ImageFont.load_default()


def labeled_tile(path, title, width=320):
    image = Image.open(path).convert("RGB")
    height = int(round(width * image.height / max(image.width, 1)))
    image = image.resize((width, height), Image.Resampling.BILINEAR)
    tile = Image.new("RGB", (width, height + 40), "white")
    ImageDraw.Draw(tile).text((10, 7), title, fill=(20, 24, 28), font=load_font(20))
    tile.paste(image, (0, 40))
    return tile


def compose_panel(output, columns):
    tiles = [labeled_tile(path, title) for path, title in columns]
    panel = Image.new("RGB", (sum(tile.width for tile in tiles), tiles[0].height), "white")
    offset = 0
    for tile in tiles:
        panel.paste(tile, (offset, 0))
        offset += tile.width
    panel.save(output, optimize=True)


def mild_metrics(pair_dir, method):
    if method in {"mast3r", "night3r"}:
        source = json.loads(
            (pair_dir / "source_assets/metrics.json").read_text(encoding="utf-8")
        )
        return source["metrics"]["master" if method == "mast3r" else "night"]
    return json.loads(
        (pair_dir / "baseline_assets" / method / "metrics.json").read_text(
            encoding="utf-8"
        )
    )["metrics"]


def method_mean(records, method, field):
    values = [
        item["stress_metrics"][field]
        for item in records
        if item["dataset"] == "DSEC" and item["method"] == method
    ]
    return float(np.mean(values))


def finalize(args, datasets):
    records = []
    panels = []
    dsec_mild = {method: [] for method in METHODS}
    for pair_dir in pair_directories(args.selected_root, args.datasets):
        batch, source_record = load_pair_batch(pair_dir, datasets, args.seed)
        pair_output = (
            args.output_root
            / "pairs"
            / source_record["dataset"]
            / pair_dir.name
        )
        pair_output.mkdir(parents=True, exist_ok=True)
        stress_images = batch["images_photo_dark"][0]
        for view_index in range(2):
            Image.fromarray(tensor_rgb(stress_images[view_index])).save(
                pair_output / f"lowlight_stress_view{view_index + 1}_model512.png"
            )

        source_arrays = np.load(pair_dir / "source_assets/geometry_arrays.npz")
        protocol = json.loads(
            (pair_dir / "geometry_assets/geometry_protocol.json").read_text(
                encoding="utf-8"
            )
        )
        if source_record["dataset"] == "DSEC":
            reference = source_arrays["gt"].astype(np.float32)
            reference_valid = source_arrays["valid"].astype(bool)
            reference_kind = "sensor_depth_gt"
            lo, hi = protocol["depth_color_range"]
        else:
            reference = np.load(
                pair_output / "mast3r/clean_reference_depth.npy"
            ).astype(np.float32)
            reference_valid = valid_depth(reference)
            reference_kind = "clean_rgb_mast3r_proxy"
            lo, hi = np.quantile(reference[reference_valid], (0.02, 0.98))
            protocol = dict(protocol)
            protocol["depth_color_range"] = [float(lo), float(hi)]

        save_depth_png(pair_output / "reference_depth_viridis.png", reference, lo, hi)
        for method in METHODS:
            method_dir = pair_output / method
            inference = json.loads(
                (method_dir / "inference.json").read_text(encoding="utf-8")
            )
            raw_depth = np.load(method_dir / "depth_raw.npy").astype(np.float32)
            if source_record["dataset"] == "DSEC":
                metrics, aligned = depth_metrics(
                    raw_depth, reference, reference_valid
                )
                metric_kind = "depth_accuracy"
                dsec_mild[method].append(mild_metrics(pair_dir, method))
            else:
                metrics, aligned = discrepancy_metrics(raw_depth, reference)
                metric_kind = "clean_reference_disagreement_diagnostic"
                metrics["metric_note"] = (
                    "Median-aligned disagreement with clean-RGB MASt3R proxy; "
                    "DL3DV has no sensor depth GT, so this is not accuracy."
                )
            np.save(method_dir / "depth_aligned.npy", aligned)
            save_depth_png(
                method_dir / "depth_aligned_viridis.png", aligned, lo, hi
            )
            geometry_paths, point_count = render_geometry(
                method_dir,
                aligned,
                reference,
                reference_valid,
                protocol,
            )
            final_record = {
                "dataset": source_record["dataset"],
                "scene": source_record["scene"],
                "sample_id": source_record["sample_id"],
                "method": method,
                "display_name": DISPLAY_NAMES[method],
                "reference": reference_kind,
                "metric_kind": metric_kind,
                "stress_metrics": metrics,
                "input": {
                    key: inference[key]
                    for key in (
                        "dataset_index",
                        "seed",
                        "stress_protocol",
                        "input_float_sha256",
                        "input_mean_both_views",
                        "input_view1_mean",
                        "input_view1_noise",
                    )
                },
                "geometry": {
                    "front": geometry_paths["front"],
                    "elevated": geometry_paths["elevated"],
                    "point_count": point_count,
                    "colormap": "viridis",
                },
            }
            (method_dir / "metrics.json").write_text(
                json.dumps(final_record, indent=2) + "\n", encoding="utf-8"
            )
            records.append(final_record)

        clean_source = pair_dir / "source_assets/clean_rgb_view1_model512.png"
        event_source = pair_dir / "source_assets/event_view1_red_blue.png"
        columns = [
            (clean_source, "Clean RGB"),
            (pair_output / "lowlight_stress_view1_model512.png", "Stress RGB"),
            (event_source, "Aligned events"),
            (pair_output / "mast3r/depth_aligned_viridis.png", "MASt3R"),
            (pair_output / "dark3r/depth_aligned_viridis.png", "Dark3R"),
            (pair_output / "vggt/depth_aligned_viridis.png", "VGGT"),
            (pair_output / "night3r/depth_aligned_viridis.png", "Night3R"),
            (
                pair_output / "reference_depth_viridis.png",
                "GT depth" if source_record["dataset"] == "DSEC" else "Clean proxy",
            ),
        ]
        panel_path = pair_output / "comparison_8column_viridis.png"
        compose_panel(panel_path, columns)
        panels.append(str(panel_path.resolve()))

    dsec_mild_mean = {}
    dsec_stress_mean = {}
    dsec_degradation = {}
    fields = ("raw_absrel", "raw_delta1", "median_absrel", "median_delta1")
    for method in METHODS:
        dsec_mild_mean[method] = {
            field: float(np.mean([item[field] for item in dsec_mild[method]]))
            for field in fields
        }
        dsec_stress_mean[method] = {
            field: method_mean(records, method, field) for field in fields
        }
        mild_absrel = dsec_mild_mean[method]["median_absrel"]
        stress_absrel = dsec_stress_mean[method]["median_absrel"]
        dsec_degradation[method] = {
            "median_absrel_absolute_increase": stress_absrel - mild_absrel,
            "median_absrel_relative_increase_percent": (
                (stress_absrel / max(mild_absrel, 1.0e-8) - 1.0) * 100.0
            ),
            "median_delta1_drop": (
                dsec_mild_mean[method]["median_delta1"]
                - dsec_stress_mean[method]["median_delta1"]
            ),
        }

    summary = {
        "version": 1,
        "stress_protocol": STRESS_PROTOCOL,
        "fairness": (
            "All RGB-dependent methods receive the same deterministic degraded "
            "float RGB pair. Night3R additionally receives the unchanged aligned "
            "event voxel. No method-specific degradation is used."
        ),
        "scope": (
            "Two selected DSEC qualitative pairs and three selected DL3DV "
            "qualitative pairs; not a replacement for full-test metrics."
        ),
        "dsec_mild_mean": dsec_mild_mean,
        "dsec_stress_mean": dsec_stress_mean,
        "dsec_degradation": dsec_degradation,
        "records": records,
        "panels": panels,
        "dl3dv_note": (
            "DL3DV has no sensor depth GT. Clean-RGB MASt3R is used only as a "
            "display-scale proxy, and disagreement values are not accuracy."
        ),
        "mvsec_note": "MVSEC is unchanged and was not rerun.",
    }
    args.output_root.mkdir(parents=True, exist_ok=True)
    (args.output_root / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )

    csv_path = args.output_root / "summary.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "Method",
                "Mild Median AbsRel",
                "Stress Median AbsRel",
                "Relative Increase (%)",
                "Mild Median delta1",
                "Stress Median delta1",
                "delta1 Drop",
            ]
        )
        for method in METHODS:
            writer.writerow(
                [
                    DISPLAY_NAMES[method],
                    dsec_mild_mean[method]["median_absrel"],
                    dsec_stress_mean[method]["median_absrel"],
                    dsec_degradation[method][
                        "median_absrel_relative_increase_percent"
                    ],
                    dsec_mild_mean[method]["median_delta1"],
                    dsec_stress_mean[method]["median_delta1"],
                    dsec_degradation[method]["median_delta1_drop"],
                ]
            )

    lines = [
        "# Matched low-light stress comparison",
        "",
        "All RGB-dependent methods receive the same deterministic degraded float "
        "RGB. Night3R additionally receives the unchanged aligned event voxel. "
        "No baseline-specific degradation is used.",
        "",
        "## DSEC selected-pair degradation",
        "",
        "| Method | Mild AbsRel | Stress AbsRel | Increase | Mild delta1 | Stress delta1 | Drop |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for method in METHODS:
        mild = dsec_mild_mean[method]
        stress = dsec_stress_mean[method]
        degradation = dsec_degradation[method]
        lines.append(
            f"| {DISPLAY_NAMES[method]} | {mild['median_absrel']:.4f} | "
            f"{stress['median_absrel']:.4f} | "
            f"{degradation['median_absrel_relative_increase_percent']:+.1f}% | "
            f"{mild['median_delta1']:.4f} | {stress['median_delta1']:.4f} | "
            f"{degradation['median_delta1_drop']:+.4f} |"
        )
    lines.extend(
        [
            "",
            "These values cover two selected DSEC qualitative pairs and demonstrate "
            "the controlled stress behavior of the displayed examples. Full-test "
            "claims must use the complete benchmark protocol.",
            "",
            "DL3DV has no sensor depth GT; its clean-RGB proxy comparison is "
            "qualitative only. MVSEC was not rerun.",
            "",
        ]
    )
    (args.output_root / "summary.md").write_text(
        "\n".join(lines), encoding="utf-8"
    )
    print(json.dumps({"output_root": str(args.output_root), "records": len(records)}, indent=2))


def main():
    args = parse_args()
    args.selected_root = args.selected_root.resolve()
    args.output_root = args.output_root.resolve()
    datasets = build_datasets(args.datasets)
    if args.finalize:
        finalize(args, datasets)
        return
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    set_seed(args.seed)
    model, load_info = load_variant_model(args, device)
    process_variant(args, model, load_info, datasets, device)


if __name__ == "__main__":
    main()
