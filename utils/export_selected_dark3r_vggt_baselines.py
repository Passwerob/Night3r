#!/usr/bin/env python3
"""Export Dark3R and VGGT baselines for the final selected paper pairs.

The script consumes the archived 512x512 low-light tensors rendered as PNGs in
the selected-pair package. This keeps the RGB degradation exactly identical to
the one already used for MASt3R and Night3R.
"""

import argparse
import csv
import hashlib
import json
import os
import random
import sys
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFont
from torch.utils.data._utils.collate import default_collate


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from model.event_dark3r_student import make_view_pair_from_batch  # noqa: E402
from dataset.dsec_event_dataset import DSECEventDataset  # noqa: E402
from train_event_dark3r import load_config  # noqa: E402
from utils.compare_event_dark3r_checkpoint import depth_from_pred  # noqa: E402
from utils.eval_event_dark3r_variant import build_model  # noqa: E402
from utils.eval_vggt_baseline import load_vggt, prediction_depth, preprocess_vggt  # noqa: E402
from utils.export_night3r_pair_pointcloud import render_points  # noqa: E402


DEFAULT_SELECTED_ROOT = (
    REPO_ROOT
    / "outputs/night3r_selected_pairs_step100_native_rgb_viridis_20260725"
)
DEFAULT_CONFIG = REPO_ROOT / "config/eval_night3r_da3_ft1k_sweep_20260722.yaml"
DEFAULT_MAST3R = (
    REPO_ROOT
    / "checkpoints/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric.pth"
)
DEFAULT_DARK3R_LORA = (
    REPO_ROOT / "checkpoints/dark3r_pretrained/weights/0049_lora.pth"
)
DEFAULT_VGGT = Path("/data/fcr/ckpt/model.pt")
DEFAULT_VGGT_ROOT = Path("/data/fcr/vggt")
DATASETS = ("DSEC", "MVSEC", "DL3DV")
METHOD_NAMES = {"dark3r": "Dark3R", "vggt": "VGGT"}
COMPARISON_NAMES = {
    "mast3r": "MASt3R",
    "dark3r": "Dark3R",
    "vggt": "VGGT",
    "night3r": "Night3R",
}
LOWLIGHT_PROTOCOL = {
    "MVSEC": {
        "exposure": 0.08,
        "read_noise_std": 3.0e-5,
        "shot_noise_std": 1.0e-4,
    },
    "DSEC": {
        "exposure": 0.012,
        "read_noise_std": 3.0e-5,
        "shot_noise_std": 1.0e-4,
    },
    "DL3DV": {
        "exposure": 0.008,
        "read_noise_std": 3.0e-5,
        "shot_noise_std": 1.0e-4,
    },
}


def parse_args():
    parser = argparse.ArgumentParser(
        "Export matched Dark3R/VGGT results for selected Night3R paper pairs"
    )
    parser.add_argument("--selected-root", type=Path, default=DEFAULT_SELECTED_ROOT)
    parser.add_argument("--variant", choices=("dark3r", "vggt"), default=None)
    parser.add_argument("--finalize", action="store_true")
    parser.add_argument("--datasets", nargs="+", choices=DATASETS, default=list(DATASETS))
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--mast3r-checkpoint", type=Path, default=DEFAULT_MAST3R)
    parser.add_argument("--dark3r-lora-checkpoint", type=Path, default=DEFAULT_DARK3R_LORA)
    parser.add_argument("--vggt-checkpoint", type=Path, default=DEFAULT_VGGT)
    parser.add_argument("--vggt-root", type=Path, default=DEFAULT_VGGT_ROOT)
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


def resolve_device(value):
    device = torch.device(value)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    return device


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def pair_directories(root, datasets):
    result = []
    for dataset in datasets:
        result.extend(sorted((root / "pairs" / dataset).glob("*")))
    return [path for path in result if path.is_dir()]


def tensor_sha256(tensor):
    value = tensor.detach().cpu().contiguous().numpy()
    return hashlib.sha256(value.tobytes()).hexdigest()


def tensor_rgb(tensor):
    value = tensor.detach().cpu().float().clamp(0, 1)
    return (value.permute(1, 2, 0).numpy() * 255.0 + 0.5).astype(np.uint8)


def build_eval_datasets(args):
    datasets = {}
    for dataset_name in args.datasets:
        protocol = LOWLIGHT_PROTOCOL[dataset_name]
        datasets[dataset_name] = DSECEventDataset(
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
    return datasets


def deterministic_batch(dataset, dataset_index, seed):
    local_seed = int(seed + dataset_index * 1009)
    random.seed(local_seed)
    np.random.seed(local_seed % (2**32 - 1))
    torch.manual_seed(local_seed)
    return default_collate([dataset[dataset_index]])


def load_pair_images(pair_dir, datasets, seed):
    source = pair_dir / "source_assets"
    paths = [
        source / "lowlight_rgb_view1_model512.png",
        source / "lowlight_rgb_view2_model512.png",
    ]
    missing = [path for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(", ".join(str(path) for path in missing))
    record = json.loads((source / "metrics.json").read_text(encoding="utf-8"))
    dataset_name = str(record["dataset"])
    dataset_index = int(record["dataset_index"])
    batch = deterministic_batch(datasets[dataset_name], dataset_index, seed)
    if str(batch["sample_id"][0]) != str(record["sample_id"]):
        raise RuntimeError(
            "Dataset ordering changed: "
            f"{batch['sample_id'][0]} != {record['sample_id']}"
        )
    images = batch["images_photo_dark"].float()
    png_exact = []
    for view_index, path in enumerate(paths):
        archived = np.asarray(Image.open(path).convert("RGB"), dtype=np.uint8)
        reconstructed = tensor_rgb(images[0, view_index])
        png_exact.append(bool(np.array_equal(archived, reconstructed)))
    if not all(png_exact):
        raise RuntimeError(
            f"Deterministic low-light reconstruction does not match archived PNG: {pair_dir}"
        )
    return images, paths, record, png_exact


def dark3r_namespaces(args):
    train_args = SimpleNamespace(
        config=str(args.config),
        data_root=None,
        save_dir=str(args.selected_root),
        model_name_or_path=str(args.mast3r_checkpoint),
        dark3r_lora_checkpoint=str(args.dark3r_lora_checkpoint),
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
        variant="dark3r_lora",
        checkpoint=None,
        geometry_head_checkpoint=None,
        config=str(args.config),
        data_root=None,
        model_name_or_path=str(args.mast3r_checkpoint),
        dark3r_lora_checkpoint=str(args.dark3r_lora_checkpoint),
        allow_unsafe_checkpoint_load=True,
        restore_frozen_rgb=False,
        restore_frozen_rgb_scope="none",
        restore_frozen_rgb_alpha=1.0,
    )
    return train_args, model_args


def load_model(args, device):
    if args.variant == "dark3r":
        cfg = load_config(args.config)
        train_args, model_args = dark3r_namespaces(args)
        model, load_info = build_model(model_args, cfg, train_args, device)
        return model, load_info

    vggt_args = SimpleNamespace(
        vggt_root=str(args.vggt_root),
        checkpoint=str(args.vggt_checkpoint),
    )
    return load_vggt(vggt_args, device)


def infer_depth(args, model, images, device):
    dtype = torch.bfloat16 if args.amp_dtype == "bf16" else torch.float16
    images = images.to(device)
    with torch.inference_mode(), torch.autocast(
        device_type=device.type,
        dtype=dtype,
        enabled=device.type == "cuda",
    ):
        if args.variant == "dark3r":
            normalized = images * 2.0 - 1.0
            view1, view2 = make_view_pair_from_batch(
                {"images_dark": normalized}, prefer_dark=True
            )
            prediction, _ = model(view1, view2)
            depth = depth_from_pred(prediction)
        else:
            prediction = model(images=preprocess_vggt(images))
            depth = prediction_depth(prediction)
    if depth is None:
        raise RuntimeError(f"{args.variant} did not return a depth prediction")
    depth = depth[0, 0].detach().float().cpu()
    if depth.shape != (512, 512):
        depth = F.interpolate(
            depth[None, None],
            size=(512, 512),
            mode="bilinear",
            align_corners=False,
        )[0, 0]
    return depth.numpy().astype(np.float32)


def valid_depth(depth):
    return np.isfinite(depth) & (depth > 1.0e-6)


def depth_metrics(pred, reference, valid):
    valid = valid & valid_depth(pred) & valid_depth(reference)
    if int(valid.sum()) < 32:
        raise RuntimeError("Insufficient valid depth pixels")
    target = reference[valid]
    estimate = pred[valid]
    ratio = np.maximum(estimate / target, target / estimate)
    scale = float(np.median(target) / max(float(np.median(estimate)), 1.0e-6))
    aligned = pred * scale
    aligned_estimate = aligned[valid]
    aligned_ratio = np.maximum(aligned_estimate / target, target / aligned_estimate)
    return (
        {
            "valid_px": int(valid.sum()),
            "raw_absrel": float(np.mean(np.abs(estimate - target) / target)),
            "raw_delta1": float(np.mean(ratio < 1.25)),
            "median_scale": scale,
            "median_absrel": float(
                np.mean(np.abs(aligned_estimate - target) / target)
            ),
            "median_delta1": float(np.mean(aligned_ratio < 1.25)),
        },
        aligned.astype(np.float32),
    )


def discrepancy_metrics(pred, reference):
    valid = valid_depth(pred) & valid_depth(reference)
    if int(valid.sum()) < 32:
        raise RuntimeError("Insufficient valid depth pixels")
    scale = float(
        np.median(reference[valid]) / max(float(np.median(pred[valid])), 1.0e-6)
    )
    aligned = pred * scale
    difference = np.abs(
        np.log(np.clip(aligned[valid], 1.0e-6, None))
        - np.log(np.clip(reference[valid], 1.0e-6, None))
    )
    return (
        {
            "valid_px": int(valid.sum()),
            "median_scale_to_mast3r": scale,
            "mean_abs_log_difference": float(np.mean(difference)),
            "p90_abs_log_difference": float(np.quantile(difference, 0.90)),
            "metric_note": (
                "Diagnostic disagreement with MASt3R after median alignment; "
                "DL3DV has no sensor depth GT, so this is not an accuracy metric."
            ),
        },
        aligned.astype(np.float32),
    )


def viridis_lut():
    values = np.arange(256, dtype=np.uint8)[:, None]
    return cv2.cvtColor(
        cv2.applyColorMap(values, cv2.COLORMAP_VIRIDIS),
        cv2.COLOR_BGR2RGB,
    )[:, 0]


def depth_colors(values, lo, hi):
    normalized = np.clip(
        (np.asarray(values, dtype=np.float32) - float(lo))
        / max(float(hi - lo), 1.0e-6),
        0.0,
        1.0,
    )
    return viridis_lut()[np.round(normalized * 255.0).astype(np.uint8)]


def save_depth_png(path, depth, lo, hi):
    valid = valid_depth(depth)
    output = np.zeros((*depth.shape, 3), dtype=np.uint8)
    output[valid] = depth_colors(depth[valid], lo, hi)
    Image.fromarray(output).save(path, optimize=True)


def backproject(depth, intrinsics, stride=2, valid=None):
    depth = np.asarray(depth, dtype=np.float32)
    h, w = depth.shape
    yy, xx = np.mgrid[0:h:stride, 0:w:stride]
    z = depth[::stride, ::stride]
    keep = valid_depth(z)
    if valid is not None:
        keep &= np.asarray(valid, dtype=bool)[::stride, ::stride]
    x = xx[keep].astype(np.float32)
    y = yy[keep].astype(np.float32)
    z = z[keep]
    fx, fy = float(intrinsics[0, 0]), float(intrinsics[1, 1])
    cx, cy = float(intrinsics[0, 2]), float(intrinsics[1, 2])
    return np.stack(((x - cx) * z / fx, (y - cy) * z / fy, z), axis=1)


def robust_volume(points):
    lower, upper = np.quantile(points, (0.01, 0.99), axis=0)
    margin = np.maximum((upper - lower) * 0.10, 1.0e-4)
    return lower - margin, upper + margin


def crop_points(points, lower, upper):
    keep = np.isfinite(points).all(axis=1)
    keep &= np.all(points >= lower[None], axis=1)
    keep &= np.all(points <= upper[None], axis=1)
    return points[keep]


def white_background(image):
    image = np.asarray(image).copy()
    image[np.all(image == np.asarray([8, 12, 18], dtype=np.uint8), axis=2)] = 255
    return image


def render_geometry(method_dir, aligned_depth, reference, reference_valid, protocol):
    intrinsics = np.asarray(protocol["intrinsics"], dtype=np.float32)
    reference_points = backproject(
        reference,
        intrinsics,
        stride=int(protocol.get("point_stride", 2)),
        valid=reference_valid,
    )
    lower, upper = robust_volume(reference_points)
    method_points = backproject(
        aligned_depth,
        intrinsics,
        stride=int(protocol.get("point_stride", 2)),
    )
    method_points = crop_points(method_points, lower, upper)
    lo, hi = protocol["depth_color_range"]
    colors = depth_colors(method_points[:, 2], lo, hi)
    center = np.asarray(protocol["camera_center"], dtype=np.float32)
    cameras = {
        "front": np.asarray(protocol["front_camera"], dtype=np.float32),
        "elevated": np.asarray(protocol["elevated_camera"], dtype=np.float32),
    }
    paths = {}
    for view_name, camera in cameras.items():
        rendered = render_points(
            method_points,
            colors,
            camera,
            center,
            width=720,
            height=540,
            radius=1,
        )
        path = method_dir / f"geometry_{view_name}.png"
        Image.fromarray(white_background(rendered)).save(path, optimize=True)
        paths[view_name] = path.name
    np.savez_compressed(method_dir / "geometry_points.npz", points=method_points)
    return paths, int(len(method_points))


def process_pair(args, model, load_info, pair_dir, device, datasets):
    method_dir = pair_dir / "baseline_assets" / args.variant
    metadata_path = method_dir / "metrics.json"
    if metadata_path.is_file() and not args.force:
        return json.loads(metadata_path.read_text(encoding="utf-8"))
    method_dir.mkdir(parents=True, exist_ok=True)

    images, image_paths, source_record, png_exact = load_pair_images(
        pair_dir, datasets, args.seed
    )
    raw_depth = infer_depth(args, model, images, device)
    if not np.isfinite(raw_depth).all() or not np.any(raw_depth > 0):
        raise RuntimeError(f"Invalid {args.variant} depth for {pair_dir}")

    arrays_path = pair_dir / "source_assets/geometry_arrays.npz"
    protocol_path = pair_dir / "geometry_assets/geometry_protocol.json"
    arrays = np.load(arrays_path)
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    if bool(protocol["has_gt"]):
        reference = arrays["gt"].astype(np.float32)
        reference_valid = arrays["valid"].astype(bool)
        metrics, aligned_depth = depth_metrics(raw_depth, reference, reference_valid)
        reference_name = "sensor_depth_gt"
        metric_kind = "depth_accuracy"
    else:
        reference = arrays["master"].astype(np.float32)
        reference_valid = valid_depth(reference)
        metrics, aligned_depth = discrepancy_metrics(raw_depth, reference)
        reference_name = "mast3r_prediction"
        metric_kind = "scale_invariant_disagreement_diagnostic"

    np.save(method_dir / "depth_raw.npy", raw_depth)
    np.save(method_dir / "depth_aligned.npy", aligned_depth)
    lo, hi = protocol["depth_color_range"]
    save_depth_png(method_dir / "depth_aligned_viridis.png", aligned_depth, lo, hi)
    geometry_paths, point_count = render_geometry(
        method_dir,
        aligned_depth,
        reference,
        reference_valid,
        protocol,
    )
    record = {
        "dataset": protocol["dataset"],
        "scene": protocol["scene"],
        "sample_id": protocol["sample_id"],
        "variant": args.variant,
        "display_name": METHOD_NAMES[args.variant],
        "input": {
            "type": "deterministically_reconstructed_matched_lowlight_rgb",
            "view1": str(image_paths[0].resolve()),
            "view2": str(image_paths[1].resolve()),
            "view1_sha256": sha256(image_paths[0]),
            "view2_sha256": sha256(image_paths[1]),
            "float_tensor_sha256": tensor_sha256(images),
            "resolution": [512, 512],
            "mean": float(images.mean()),
            "dataset_index": int(source_record["dataset_index"]),
            "seed": int(args.seed),
            "lowlight_protocol": LOWLIGHT_PROTOCOL[protocol["dataset"]],
            "archived_png_exact_match": png_exact,
            "note": (
                "The float model input is deterministically reconstructed from the "
                "same dataset index, seed, exposure, and noise parameters used by "
                "the existing MASt3R/Night3R sweep. Its 8-bit rendering matches the "
                "archived PNG pixel-for-pixel. No baseline-specific noise was added."
            ),
        },
        "reference": reference_name,
        "metric_kind": metric_kind,
        "metrics": metrics,
        "depth_color_range": [float(lo), float(hi)],
        "colormap": "viridis",
        "geometry": {
            "coordinate_frame": protocol["coordinate_frame"],
            "scale_policy": protocol["scale_policy"],
            "camera_policy": protocol["camera_policy"],
            "front": geometry_paths["front"],
            "elevated": geometry_paths["elevated"],
            "point_count": point_count,
        },
        "model_load": load_info,
        "outputs": {
            "depth_raw": "depth_raw.npy",
            "depth_aligned": "depth_aligned.npy",
            "depth_visualization": "depth_aligned_viridis.png",
        },
    }
    metadata_path.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    return record


def font(size=24):
    candidates = (
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf",
    )
    for path in candidates:
        if Path(path).is_file():
            return ImageFont.truetype(path, size)
    return ImageFont.load_default()


def labeled_tile(path, label, size=(512, 512)):
    image = Image.open(path).convert("RGB").resize(size, Image.Resampling.BILINEAR)
    tile = Image.new("RGB", (size[0], size[1] + 42), "white")
    tile.paste(image, (0, 42))
    draw = ImageDraw.Draw(tile)
    draw.text((12, 7), label, fill=(20, 24, 28), font=font(22))
    return tile


def build_pair_panel(pair_dir, protocol):
    source = pair_dir / "source_assets"
    columns = [
        (source / "lowlight_rgb_view1_model512.png", "Low-light RGB"),
        (source / "master_depth_shared.png", "MASt3R"),
        (
            pair_dir / "baseline_assets/dark3r/depth_aligned_viridis.png",
            "Dark3R",
        ),
        (pair_dir / "baseline_assets/vggt/depth_aligned_viridis.png", "VGGT"),
        (source / "night_depth_shared.png", "Night3R"),
        (
            source / "gt_depth.png",
            "GT depth" if protocol["has_gt"] else "MASt3R/Night3R difference",
        ),
    ]
    missing = [path for path, _ in columns if not path.is_file()]
    if missing:
        raise FileNotFoundError(", ".join(str(path) for path in missing))
    tiles = [labeled_tile(path, label) for path, label in columns]
    panel = Image.new("RGB", (sum(tile.width for tile in tiles), tiles[0].height), "white")
    offset = 0
    for tile in tiles:
        panel.paste(tile, (offset, 0))
        offset += tile.width
    output = pair_dir / "baseline_assets/comparison_depths_viridis.png"
    output.parent.mkdir(parents=True, exist_ok=True)
    panel.save(output, optimize=True)
    return output


def mean_metrics(records, method, dataset):
    selected = [
        item["metrics"]
        for item in records
        if item["variant"] == method
        and item["dataset"] == dataset
        and item["metric_kind"] == "depth_accuracy"
    ]
    if not selected:
        return {}
    keys = ("raw_absrel", "raw_delta1", "median_absrel", "median_delta1")
    return {key: float(np.mean([item[key] for item in selected])) for key in keys}


def existing_dsec_means(selected_root):
    values = {"mast3r": [], "night3r": []}
    source_keys = {"mast3r": "master", "night3r": "night"}
    for pair_dir in sorted((selected_root / "pairs" / "DSEC").glob("*")):
        path = pair_dir / "source_assets/metrics.json"
        if not path.is_file():
            continue
        record = json.loads(path.read_text(encoding="utf-8"))
        for method, source_key in source_keys.items():
            values[method].append(record["metrics"][source_key])
    keys = ("raw_absrel", "raw_delta1", "median_absrel", "median_delta1")
    return {
        method: {
            key: float(np.mean([item[key] for item in method_values]))
            for key in keys
        }
        for method, method_values in values.items()
        if method_values
    }


def finalize(args):
    records = []
    panels = []
    for pair_dir in pair_directories(args.selected_root, args.datasets):
        protocol = json.loads(
            (pair_dir / "geometry_assets/geometry_protocol.json").read_text(
                encoding="utf-8"
            )
        )
        for method in METHOD_NAMES:
            path = pair_dir / "baseline_assets" / method / "metrics.json"
            if not path.is_file():
                raise FileNotFoundError(path)
            records.append(json.loads(path.read_text(encoding="utf-8")))
        panels.append(str(build_pair_panel(pair_dir, protocol).resolve()))

    dsec_means = existing_dsec_means(args.selected_root)
    dsec_means.update(
        {method: mean_metrics(records, method, "DSEC") for method in METHOD_NAMES}
    )
    summary = {
        "version": 1,
        "selected_root": str(args.selected_root.resolve()),
        "protocol": {
            "DSEC": (
                "Deterministically reconstructed mildly noisy low-light RGB matching "
                "the original float sweep input; metrics use sensor depth GT and "
                "the original valid mask."
            ),
            "DL3DV": (
                "Deterministically reconstructed mildly noisy low-light RGB matching "
                "the original float sweep input; no sensor depth GT. Reported "
                "log-depth disagreement is diagnostic only."
            ),
            "fairness": (
                "Dark3R, VGGT, MASt3R, and Night3R receive the same archived RGB "
                "degradation for each pair. Noise is not applied selectively."
            ),
            "MVSEC": (
                "Not rerun. Existing results remain at "
                "outputs/exp1_follow_20260711_128/main_results/dark3r_lora and "
                "outputs/table_missing_completion_20260716/vggt_clean."
            ),
        },
        "dsec_selected_pair_mean": dsec_means,
        "dsec_selected_pair_mean_note": (
            "Mean over the two hand-selected qualitative DSEC pairs. This supports "
            "asset inspection and is not a replacement for the full benchmark table."
        ),
        "records": records,
        "comparison_panels": panels,
    }
    metadata = args.selected_root / "metadata"
    metadata.mkdir(parents=True, exist_ok=True)
    summary_path = metadata / "dark3r_vggt_baseline_supplement.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    csv_path = metadata / "dark3r_vggt_baseline_supplement.csv"
    fields = [
        "dataset",
        "scene",
        "sample_id",
        "variant",
        "metric_kind",
        "raw_absrel",
        "raw_delta1",
        "median_absrel",
        "median_delta1",
        "mean_abs_log_difference",
        "p90_abs_log_difference",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for record in records:
            row = {key: record.get(key, "") for key in fields}
            row.update(
                {
                    key: record["metrics"].get(key, "")
                    for key in fields
                    if key in record["metrics"]
                }
            )
            writer.writerow(row)

    md_path = metadata / "dark3r_vggt_baseline_supplement.md"
    lines = [
        "# Dark3R and VGGT baseline supplement",
        "",
        "Dark3R and VGGT use deterministically reconstructed float low-light RGB "
        "inputs from the same dataset indices, seed, exposure, and noise parameters "
        "as the selected MASt3R/Night3R comparison. Their 8-bit renderings match "
        "the archived PNGs pixel-for-pixel. No method-specific degradation is applied.",
        "",
        "## DSEC selected-pair mean",
        "",
        "| Method | Raw AbsRel | Raw delta1 | Median AbsRel | Median delta1 |",
        "|---|---:|---:|---:|---:|",
    ]
    for method, display_name in COMPARISON_NAMES.items():
        values = summary["dsec_selected_pair_mean"][method]
        lines.append(
            f"| {display_name} | {values['raw_absrel']:.4f} | "
            f"{values['raw_delta1']:.4f} | {values['median_absrel']:.4f} | "
            f"{values['median_delta1']:.4f} |"
        )
    lines.extend(
        [
            "",
            "These means cover only the two selected qualitative DSEC pairs and "
            "must not replace the full-test benchmark results.",
            "",
            "DL3DV has no sensor depth ground truth. Its per-pair JSON files retain "
            "only median-aligned disagreement with MASt3R as a visualization "
            "diagnostic; these values must not be reported as reconstruction accuracy.",
            "",
            "MVSEC was not rerun. Existing Dark3R and VGGT results are referenced "
            "from their original output directories.",
            "",
        ]
    )
    md_path.write_text("\n".join(lines), encoding="utf-8")
    print(
        json.dumps(
            {
                "summary": str(summary_path),
                "csv": str(csv_path),
                "markdown": str(md_path),
                "records": len(records),
                "panels": len(panels),
            },
            indent=2,
        )
    )


def main():
    args = parse_args()
    args.selected_root = args.selected_root.resolve()
    if args.finalize:
        finalize(args)
        return

    device = resolve_device(args.device)
    set_seed(args.seed)
    datasets = build_eval_datasets(args)
    model, load_info = load_model(args, device)
    records = []
    for pair_dir in pair_directories(args.selected_root, args.datasets):
        records.append(process_pair(args, model, load_info, pair_dir, device, datasets))
        print(
            f"[{args.variant}] {records[-1]['dataset']} "
            f"{records[-1]['sample_id']}",
            flush=True,
        )
    output = args.selected_root / "metadata" / f"{args.variant}_selected_manifest.json"
    output.write_text(
        json.dumps(
            {
                "variant": args.variant,
                "records": records,
                "model_load": load_info,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"manifest": str(output), "count": len(records)}, indent=2))


if __name__ == "__main__":
    main()
