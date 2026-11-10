#!/usr/bin/env python3
"""Generate DA3 depth and confidence maps for the Night3R test datasets.

The outputs are external-model predictions, not sensor ground truth. DL3DV
uses its calibrated camera trajectory, MVSEC uses the exported body trajectory
as a camera-pose proxy, and DSEC runs without pose conditioning.
"""

import argparse
import json
import math
import os
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
import yaml
from PIL import Image, ImageDraw, ImageFont, ImageOps

from depth_anything_3.api import DepthAnything3


OPENGL_TO_OPENCV = np.diag([1.0, -1.0, -1.0, 1.0]).astype(np.float32)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--test-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=("DL3DV", "DSEC", "MVSEC"),
        choices=("DL3DV", "DSEC", "MVSEC"),
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--window-size", type=int, default=8)
    parser.add_argument("--window-overlap", type=int, default=2)
    parser.add_argument("--process-res", type=int, default=504)
    parser.add_argument(
        "--target-size", type=int, nargs=2, default=(512, 512), metavar=("H", "W")
    )
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--max-scenes", type=int, default=None)
    parser.add_argument("--selection-only", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def read_json(path):
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def atomic_write_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
        handle.write("\n")
    os.replace(temporary, path)


def save_tensor(path, tensor):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    torch.save(tensor, temporary)
    os.replace(temporary, path)


def discover_scenes(args):
    records = []
    for dataset in args.datasets:
        dataset_root = args.test_root / dataset
        if not dataset_root.is_dir():
            raise FileNotFoundError(f"Missing test dataset: {dataset_root}")
        for scene_dir in sorted(dataset_root.glob("screen-*")):
            image_paths = sorted((scene_dir / "images").glob("*.png"))
            if not image_paths:
                continue
            if dataset == "DL3DV":
                pose_mode = "dl3dv_opengl_c2w"
            elif dataset == "MVSEC" and (
                scene_dir / "meta" / "poses_selected.npz"
            ).is_file():
                pose_mode = "mvsec_world_body_proxy"
            else:
                pose_mode = "none"
            records.append(
                {
                    "dataset": dataset,
                    "scene": scene_dir.name,
                    "source_scene_dir": str(scene_dir.resolve()),
                    "frame_count": len(image_paths),
                    "pose_mode": pose_mode,
                }
            )
    records.sort(key=lambda item: (item["dataset"], item["scene"]))
    if args.max_scenes is not None:
        records = records[: args.max_scenes]
    return records


def balanced_assignments(records, num_shards):
    assignments = [[] for _ in range(num_shards)]
    loads = [0] * num_shards
    for record in sorted(
        records,
        key=lambda item: (-item["frame_count"], item["dataset"], item["scene"]),
    ):
        shard_index = min(range(num_shards), key=lambda index: (loads[index], index))
        assignments[shard_index].append(record)
        loads[shard_index] += record["frame_count"]
    for shard_records in assignments:
        shard_records.sort(key=lambda item: (item["dataset"], item["scene"]))
    return assignments, loads


def load_yaml(path):
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def image_intrinsic_from_yaml(scene_dir, image_size):
    camera_path = scene_dir / "cameras.yaml"
    if not camera_path.is_file():
        return None
    cameras = load_yaml(camera_path)
    camera = cameras.get("rgb_cam_crop") or cameras.get("rgb_cam_full")
    if not camera or "intrinsics" not in camera:
        return None
    fx, fy, cx, cy = [float(value) for value in camera["intrinsics"]]
    source_w, source_h = [float(value) for value in camera["resolution"]]
    image_w, image_h = image_size
    sx = image_w / source_w
    sy = image_h / source_h
    return np.array(
        [[fx * sx, 0.0, cx * sx], [0.0, fy * sy, cy * sy], [0.0, 0.0, 1.0]],
        dtype=np.float32,
    )


def dl3dv_cameras(scene_dir, image_paths, image_size):
    transforms_path = scene_dir / "transforms.json"
    if not transforms_path.is_file():
        raise FileNotFoundError(f"Missing DL3DV transforms: {transforms_path}")
    transforms = read_json(transforms_path)
    frames = transforms.get("frames", [])
    if len(frames) < len(image_paths):
        raise ValueError(
            f"{scene_dir}: {len(frames)} poses for {len(image_paths)} images"
        )

    image_w, image_h = image_size
    sx = image_w / float(transforms["w"])
    sy = image_h / float(transforms["h"])
    intrinsic = np.array(
        [
            [float(transforms["fl_x"]) * sx, 0.0, float(transforms["cx"]) * sx],
            [0.0, float(transforms["fl_y"]) * sy, float(transforms["cy"]) * sy],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )
    extrinsics = []
    for frame in frames[: len(image_paths)]:
        camera_to_world_opengl = np.asarray(
            frame["transform_matrix"], dtype=np.float32
        )
        if camera_to_world_opengl.shape != (4, 4):
            raise ValueError(f"{scene_dir}: invalid transform shape")
        camera_to_world_opencv = camera_to_world_opengl @ OPENGL_TO_OPENCV
        extrinsics.append(np.linalg.inv(camera_to_world_opencv).astype(np.float32))
    intrinsics = np.repeat(intrinsic[None], len(image_paths), axis=0)
    return np.stack(extrinsics), intrinsics, {
        "pose_input": "DL3DV OpenGL camera-to-world converted to OpenCV world-to-camera",
        "pose_is_proxy": False,
        "scale_semantics": (
            "aligned to the supplied scene trajectory scale; absolute metric "
            "scale is not guaranteed"
        ),
        "intrinsic_source": "transforms.json scaled to stored image size",
    }


def mvsec_cameras(scene_dir, image_paths, image_size):
    intrinsic = image_intrinsic_from_yaml(scene_dir, image_size)
    if intrinsic is None:
        raise ValueError(f"{scene_dir}: missing usable cameras.yaml")
    pose_path = scene_dir / "meta" / "poses_selected.npz"
    poses_npz = np.load(pose_path)
    if "T_world_body" not in poses_npz:
        raise KeyError(f"{pose_path}: missing T_world_body")
    camera_to_world_proxy = np.asarray(poses_npz["T_world_body"], dtype=np.float32)
    if (
        camera_to_world_proxy.ndim != 3
        or camera_to_world_proxy.shape[-2:] != (4, 4)
        or len(camera_to_world_proxy) < len(image_paths)
    ):
        raise ValueError(f"{pose_path}: invalid pose array {camera_to_world_proxy.shape}")
    extrinsics = np.linalg.inv(camera_to_world_proxy[: len(image_paths)]).astype(
        np.float32
    )
    intrinsics = np.repeat(intrinsic[None], len(image_paths), axis=0)
    return extrinsics, intrinsics, {
        "pose_input": "MVSEC T_world_body inverted and used as a camera-pose proxy",
        "pose_is_proxy": True,
        "scale_semantics": "aligned to the metric body-trajectory scale; camera-body offset is unavailable",
        "intrinsic_source": "rectified cameras.yaml scaled to stored image size",
    }


def dsec_cameras(scene_dir, image_paths, image_size):
    intrinsic = image_intrinsic_from_yaml(scene_dir, image_size)
    if intrinsic is None:
        raise ValueError(f"{scene_dir}: missing usable cameras.yaml")
    intrinsics = np.repeat(intrinsic[None], len(image_paths), axis=0)
    return None, intrinsics, {
        "pose_input": "none",
        "pose_is_proxy": False,
        "scale_semantics": "DA3 relative depth; no input trajectory is available",
        "intrinsic_source": "rectified cameras.yaml scaled to stored image size",
    }


def scene_cameras(record, image_paths):
    scene_dir = Path(record["source_scene_dir"])
    with Image.open(image_paths[0]) as image:
        image_size = image.size
    if record["dataset"] == "DL3DV":
        return (*dl3dv_cameras(scene_dir, image_paths, image_size), image_size)
    if record["dataset"] == "MVSEC":
        return (*mvsec_cameras(scene_dir, image_paths, image_size), image_size)
    return (*dsec_cameras(scene_dir, image_paths, image_size), image_size)


def window_starts(num_frames, window_size, overlap):
    if num_frames <= window_size:
        return [0]
    stride = window_size - overlap
    if stride <= 0:
        raise ValueError("window-overlap must be smaller than window-size")
    starts = list(range(0, num_frames - window_size + 1, stride))
    final_start = num_frames - window_size
    if starts[-1] != final_start:
        starts.append(final_start)
    return starts


def resize_maps(array, target_size):
    tensor = torch.from_numpy(np.asarray(array)).float().unsqueeze(1)
    return (
        F.interpolate(
            tensor, size=tuple(target_size), mode="bilinear", align_corners=False
        )
        .squeeze(1)
        .numpy()
    )


def confidence_weights(confidence):
    confidence = np.asarray(confidence, dtype=np.float32)
    result = np.empty_like(confidence)
    for index, frame_confidence in enumerate(confidence):
        finite = np.isfinite(frame_confidence)
        if not finite.any():
            result[index] = 1.0
            continue
        low, high = np.percentile(frame_confidence[finite], [5.0, 95.0])
        scale = max(float(high - low), 1e-6)
        result[index] = np.clip(
            (frame_confidence - low) / scale, 0.05, 1.0
        )
    return result


def new_accumulator(shape):
    return {
        "log_sum": np.zeros(shape, dtype=np.float64),
        "weight_sum": np.zeros(shape, dtype=np.float64),
        "conf_sum": np.zeros(shape, dtype=np.float64),
        "conf_count": np.zeros(shape, dtype=np.float64),
    }


def estimate_pending_scale(pending, start, new_depth):
    residuals = []
    for local_index, frame_depth in enumerate(new_depth):
        accumulator = pending.get(start + local_index)
        if accumulator is None:
            continue
        existing = accumulator["weight_sum"] > 0
        valid = existing & np.isfinite(frame_depth) & (frame_depth > 0)
        if valid.sum() < 1024:
            continue
        old_log = (
            accumulator["log_sum"][valid] / accumulator["weight_sum"][valid]
        )
        delta = old_log - np.log(frame_depth[valid])
        delta = delta[np.isfinite(delta)]
        if delta.size:
            residuals.append(delta[:: max(1, delta.size // 20000)])
    if not residuals:
        return 1.0
    log_scale = float(np.median(np.concatenate(residuals)))
    return float(np.clip(math.exp(log_scale), 0.25, 4.0))


def colorize_depth(depth):
    valid = np.isfinite(depth) & (depth > 0)
    if not valid.any():
        return np.zeros((*depth.shape, 3), dtype=np.uint8)
    low, high = np.percentile(depth[valid], [2.0, 98.0])
    normalized = np.clip((depth - low) / max(float(high - low), 1e-6), 0.0, 1.0)
    colored = cv2.applyColorMap(
        (normalized * 255).astype(np.uint8), cv2.COLORMAP_TURBO
    )
    colored = cv2.cvtColor(colored, cv2.COLOR_BGR2RGB)
    colored[~valid] = 0
    return colored


def colorize_confidence(confidence):
    finite = np.isfinite(confidence)
    if not finite.any():
        return np.zeros((*confidence.shape, 3), dtype=np.uint8)
    low, high = np.percentile(confidence[finite], [2.0, 98.0])
    normalized = np.clip(
        (confidence - low) / max(float(high - low), 1e-6), 0.0, 1.0
    )
    return np.repeat((normalized[..., None] * 255).astype(np.uint8), 3, axis=2)


def save_overview(output_scene, image_paths, examples):
    indices = sorted(examples)
    if not indices:
        return
    tile_size = (420, 236)
    label_height = 28
    canvas = Image.new(
        "RGB",
        (3 * tile_size[0], label_height + len(indices) * tile_size[1]),
        "white",
    )
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()
    for column, label in enumerate(("Test RGB", "DA3 depth", "DA3 confidence")):
        draw.text((column * tile_size[0] + 8, 9), label, fill="black", font=font)
    for row, frame_index in enumerate(indices):
        y = label_height + row * tile_size[1]
        rgb = Image.open(image_paths[frame_index]).convert("RGB")
        rgb = ImageOps.fit(rgb, tile_size, method=Image.Resampling.LANCZOS)
        depth, confidence = examples[frame_index]
        depth_image = Image.fromarray(colorize_depth(depth)).resize(tile_size)
        confidence_image = Image.fromarray(
            colorize_confidence(confidence)
        ).resize(tile_size)
        canvas.paste(rgb, (0, y))
        canvas.paste(depth_image, (tile_size[0], y))
        canvas.paste(confidence_image, (2 * tile_size[0], y))
        draw.text(
            (6, y + 6),
            f"frame {frame_index:06d}",
            fill="white",
            stroke_width=2,
            stroke_fill="black",
            font=font,
        )
    canvas.save(output_scene / "da3_depth_overview.jpg", quality=94)


def complete_scene(output_scene, image_paths):
    manifest_path = output_scene / "da3_depth_manifest.json"
    if not manifest_path.is_file():
        return False
    try:
        manifest = read_json(manifest_path)
    except (OSError, json.JSONDecodeError):
        return False
    if manifest.get("status") != "complete":
        return False
    return all(
        (output_scene / "depth_da3" / f"{path.stem}.pt").is_file()
        and (output_scene / "depth_da3_conf" / f"{path.stem}.pt").is_file()
        for path in image_paths
    )


def process_scene(model, record, args):
    scene_dir = Path(record["source_scene_dir"])
    output_scene = args.output_root / record["dataset"] / record["scene"]
    image_paths = sorted((scene_dir / "images").glob("*.png"))
    if complete_scene(output_scene, image_paths) and not args.overwrite:
        return {
            "dataset": record["dataset"],
            "scene": record["scene"],
            "status": "skipped-complete",
            "frame_count": len(image_paths),
        }

    output_scene.mkdir(parents=True, exist_ok=True)
    extrinsics, intrinsics, camera_protocol, original_image_size = scene_cameras(
        record, image_paths
    )
    num_frames = len(image_paths)
    window_size = min(args.window_size, num_frames)
    starts = window_starts(num_frames, window_size, args.window_overlap)
    target_shape = tuple(args.target_size)
    target_indices = {0, num_frames // 2, num_frames - 1}
    examples = {}
    pending = {}
    scale_factors = []
    frame_medians = []
    valid_fractions = []
    depth_min = float("inf")
    depth_max = float("-inf")
    started = time.time()

    for window_index, start in enumerate(starts):
        stop = min(start + window_size, num_frames)
        window_extrinsics = None if extrinsics is None else extrinsics[start:stop]
        prediction = model.inference(
            [str(path) for path in image_paths[start:stop]],
            extrinsics=window_extrinsics,
            intrinsics=intrinsics[start:stop],
            align_to_input_ext_scale=window_extrinsics is not None,
            process_res=args.process_res,
            process_res_method="upper_bound_resize",
            export_dir=None,
            ref_view_strategy="middle",
        )
        depth = resize_maps(prediction.depth, target_shape)
        if prediction.conf is None:
            confidence = np.ones_like(depth, dtype=np.float32)
        else:
            confidence = resize_maps(prediction.conf, target_shape)
        scale = estimate_pending_scale(pending, start, depth)
        depth *= scale
        scale_factors.append(scale)
        weights = confidence_weights(confidence)
        valid = (
            np.isfinite(depth)
            & (depth > 1e-4)
            & (depth < 1e4)
            & np.isfinite(weights)
        )

        for local_index in range(stop - start):
            frame_index = start + local_index
            accumulator = pending.setdefault(
                frame_index, new_accumulator(target_shape)
            )
            frame_valid = valid[local_index]
            frame_weight = np.where(frame_valid, weights[local_index], 0.0)
            accumulator["log_sum"] += np.where(
                frame_valid,
                np.log(np.clip(depth[local_index], 1e-6, None)),
                0.0,
            ) * frame_weight
            accumulator["weight_sum"] += frame_weight
            accumulator["conf_sum"] += np.where(
                frame_valid, confidence[local_index], 0.0
            )
            accumulator["conf_count"] += frame_valid.astype(np.float32)

        next_start = starts[window_index + 1] if window_index + 1 < len(starts) else num_frames
        finalized = sorted(index for index in pending if index < next_start)
        for frame_index in finalized:
            accumulator = pending.pop(frame_index)
            frame_valid = accumulator["weight_sum"] > 0
            frame_depth = np.full(target_shape, np.nan, dtype=np.float32)
            frame_depth[frame_valid] = np.exp(
                accumulator["log_sum"][frame_valid]
                / accumulator["weight_sum"][frame_valid]
            ).astype(np.float32)
            conf_valid = accumulator["conf_count"] > 0
            frame_confidence = np.zeros(target_shape, dtype=np.float32)
            frame_confidence[conf_valid] = (
                accumulator["conf_sum"][conf_valid]
                / accumulator["conf_count"][conf_valid]
            ).astype(np.float32)

            stem = image_paths[frame_index].stem
            save_tensor(
                output_scene / "depth_da3" / f"{stem}.pt",
                torch.from_numpy(frame_depth),
            )
            save_tensor(
                output_scene / "depth_da3_conf" / f"{stem}.pt",
                torch.from_numpy(frame_confidence).to(torch.float16),
            )
            finite_depth = frame_depth[frame_valid]
            valid_fractions.append(float(frame_valid.mean()))
            if finite_depth.size:
                frame_medians.append(float(np.median(finite_depth)))
                depth_min = min(depth_min, float(finite_depth.min()))
                depth_max = max(depth_max, float(finite_depth.max()))
            if frame_index in target_indices:
                examples[frame_index] = (frame_depth, frame_confidence)

        print(
            f"[{record['dataset']}/{record['scene']}] "
            f"window {window_index + 1}/{len(starts)} frames {start}:{stop} "
            f"overlap_scale={scale:.4f}",
            flush=True,
        )
        del prediction, depth, confidence, weights, valid

    if pending:
        raise RuntimeError(
            f"{record['dataset']}/{record['scene']}: unfinalized frames {sorted(pending)}"
        )
    save_overview(output_scene, image_paths, examples)
    manifest = {
        "status": "complete",
        "dataset": record["dataset"],
        "scene": record["scene"],
        "source_scene_dir": str(scene_dir.resolve()),
        "model_dir": str(args.model_dir.resolve()),
        "model_license": "Apache-2.0",
        "label_type": "DA3 prediction; not sensor/dataset ground truth",
        "frame_count": num_frames,
        "original_image_size_wh": list(original_image_size),
        "saved_depth_size_hw": list(args.target_size),
        "window_size": args.window_size,
        "window_overlap": args.window_overlap,
        "window_count": len(starts),
        "window_overlap_scales": scale_factors,
        "process_res": args.process_res,
        "valid_fraction_mean": float(np.mean(valid_fractions)),
        "depth_frame_median": float(np.median(frame_medians)),
        "depth_min": depth_min if math.isfinite(depth_min) else None,
        "depth_max": depth_max if math.isfinite(depth_max) else None,
        "elapsed_seconds": time.time() - started,
        **camera_protocol,
    }
    atomic_write_json(output_scene / "da3_depth_manifest.json", manifest)
    return manifest


def main():
    args = parse_args()
    if args.num_shards <= 0 or not 0 <= args.shard_index < args.num_shards:
        raise ValueError("shard-index must be in [0, num-shards)")
    if args.window_size <= 0 or args.window_overlap < 0:
        raise ValueError("window sizes must be non-negative and window-size positive")
    if args.window_overlap >= args.window_size:
        raise ValueError("window-overlap must be smaller than window-size")

    args.test_root = args.test_root.resolve()
    args.output_root = args.output_root.resolve()
    args.model_dir = args.model_dir.resolve()
    args.output_root.mkdir(parents=True, exist_ok=True)
    records = discover_scenes(args)
    assignments, shard_loads = balanced_assignments(records, args.num_shards)
    selection = {
        "status": "selected",
        "test_root": str(args.test_root),
        "output_root": str(args.output_root),
        "model_dir": str(args.model_dir),
        "model_license": "Apache-2.0",
        "label_type": "DA3 prediction; not sensor/dataset ground truth",
        "datasets": list(args.datasets),
        "scene_count": len(records),
        "frame_count": sum(item["frame_count"] for item in records),
        "target_size_hw": list(args.target_size),
        "window_size": args.window_size,
        "window_overlap": args.window_overlap,
        "process_res": args.process_res,
        "num_shards": args.num_shards,
        "shard_frame_loads": shard_loads,
        "records": records,
    }
    atomic_write_json(args.output_root / "selection_manifest.json", selection)
    assigned = assignments[args.shard_index]
    assigned_frames = sum(item["frame_count"] for item in assigned)
    print(
        f"Shard {args.shard_index}/{args.num_shards}: "
        f"{len(assigned)} scenes, {assigned_frames} frames",
        flush=True,
    )
    if args.selection_only:
        return

    model = DepthAnything3.from_pretrained(str(args.model_dir)).to(args.device)
    model.eval()
    results = []
    for scene_index, record in enumerate(assigned):
        print(
            f"Scene {scene_index + 1}/{len(assigned)}: "
            f"{record['dataset']}/{record['scene']}",
            flush=True,
        )
        result = process_scene(model, record, args)
        results.append(result)
        atomic_write_json(
            args.output_root / f"shard_{args.shard_index:02d}_progress.json",
            {
                "status": "running",
                "shard_index": args.shard_index,
                "num_shards": args.num_shards,
                "completed_scenes": len(results),
                "assigned_scenes": len(assigned),
                "assigned_frames": assigned_frames,
                "results": results,
            },
        )
    atomic_write_json(
        args.output_root / f"shard_{args.shard_index:02d}_complete.json",
        {
            "status": "complete",
            "shard_index": args.shard_index,
            "num_shards": args.num_shards,
            "assigned_scenes": len(assigned),
            "assigned_frames": assigned_frames,
            "results": results,
        },
    )


if __name__ == "__main__":
    main()
