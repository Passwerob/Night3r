#!/usr/bin/env python3
"""Generate pose-conditioned DA3 pseudo depth for a leakage-safe DL3DV subset.

The output intentionally uses ``depth_pseudo`` rather than ``depth_gt``.  DA3
predictions are teacher labels and must not be reported as sensor ground truth.
"""

import argparse
import json
import math
import os
import shutil
import time
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFont

from depth_anything_3.api import DepthAnything3


OPENGL_TO_OPENCV = np.diag([1.0, -1.0, -1.0, 1.0]).astype(np.float32)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--test-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--clips-per-source", type=int, default=2)
    parser.add_argument("--window-size", type=int, default=8)
    parser.add_argument("--window-overlap", type=int, default=2)
    parser.add_argument("--process-res", type=int, default=504)
    parser.add_argument("--target-size", type=int, nargs=2, default=(512, 512), metavar=("H", "W"))
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--selection-only", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def read_json(path):
    with open(path, "r") as handle:
        return json.load(handle)


def atomic_write_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    with open(tmp, "w") as handle:
        json.dump(payload, handle, indent=2)
        handle.write("\n")
    os.replace(tmp, path)


def source_scene_id(scene_dir):
    meta_path = scene_dir / "meta.json"
    if not meta_path.is_file():
        return None
    return str(read_json(meta_path).get("scene", "")).strip() or None


def test_source_scenes(test_root):
    sources = set()
    missing_meta = []
    for scene_dir in sorted(test_root.glob("screen-*")):
        source = source_scene_id(scene_dir)
        if source is None:
            missing_meta.append(scene_dir.name)
        else:
            sources.add(source)
    return sources, missing_meta


def evenly_spaced(items, count):
    if count >= len(items):
        return list(items)
    positions = np.linspace(0, len(items) - 1, count + 2)[1:-1]
    indices = []
    for position in positions:
        index = int(round(float(position)))
        while index in indices and index + 1 < len(items):
            index += 1
        indices.append(index)
    return [items[index] for index in sorted(set(indices))]


def build_selection(args):
    excluded_sources, missing_test_meta = test_source_scenes(args.test_root)
    grouped = defaultdict(list)
    skipped = []
    for scene_dir in sorted(args.source_root.glob("screen-*")):
        source = source_scene_id(scene_dir)
        if source is None:
            skipped.append({"scene": scene_dir.name, "reason": "missing source scene metadata"})
            continue
        if source in excluded_sources:
            skipped.append({"scene": scene_dir.name, "source_scene": source, "reason": "test source overlap"})
            continue
        required = [scene_dir / "images", scene_dir / "events", scene_dir / "transforms.json"]
        if not all(path.exists() for path in required):
            skipped.append({"scene": scene_dir.name, "source_scene": source, "reason": "incomplete scene"})
            continue
        grouped[source].append(scene_dir)

    selected = []
    for source in sorted(grouped):
        choices = evenly_spaced(sorted(grouped[source]), args.clips_per_source)
        selected.extend(choices)

    records = []
    for scene_dir in sorted(selected):
        meta = read_json(scene_dir / "meta.json")
        records.append(
            {
                "scene": scene_dir.name,
                "source_scene": str(meta.get("scene")),
                "source_clip": str(meta.get("clip")),
                "source_dir": str(scene_dir),
                "image_count": len(list((scene_dir / "images").glob("*.png"))),
                "event_count": len(list((scene_dir / "events").glob("*.pt"))),
            }
        )
    return {
        "protocol": "DA3 pose-conditioned pseudo-depth; selected source scenes do not overlap DL3DV test source scenes",
        "source_root": str(args.source_root),
        "test_root": str(args.test_root),
        "model_dir": str(args.model_dir),
        "clips_per_source": args.clips_per_source,
        "excluded_test_source_scenes": sorted(excluded_sources),
        "test_scenes_missing_meta": missing_test_meta,
        "selected_source_scene_count": len(grouped),
        "selected_clip_count": len(records),
        "selected_frame_count": sum(item["image_count"] for item in records),
        "records": records,
        "skipped": skipped,
    }


def scene_cameras(scene_dir, image_paths):
    transforms = read_json(scene_dir / "transforms.json")
    frames = transforms.get("frames", [])
    if len(frames) < len(image_paths):
        raise ValueError(f"{scene_dir}: only {len(frames)} poses for {len(image_paths)} images")

    with Image.open(image_paths[0]) as image:
        image_w, image_h = image.size
    source_w = float(transforms["w"])
    source_h = float(transforms["h"])
    sx = image_w / source_w
    sy = image_h / source_h
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
        c2w_opengl = np.asarray(frame["transform_matrix"], dtype=np.float32)
        if c2w_opengl.shape != (4, 4):
            raise ValueError(f"{scene_dir}: invalid transform shape {c2w_opengl.shape}")
        c2w_opencv = c2w_opengl @ OPENGL_TO_OPENCV
        extrinsics.append(np.linalg.inv(c2w_opencv).astype(np.float32))
    intrinsics = np.repeat(intrinsic[None], len(image_paths), axis=0)
    return np.stack(extrinsics), intrinsics, (image_h, image_w), transforms


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


def resize_maps(array, target_size, mode):
    tensor = torch.from_numpy(np.asarray(array)).float().unsqueeze(1)
    kwargs = {"size": tuple(target_size), "mode": mode}
    if mode in {"bilinear", "bicubic"}:
        kwargs["align_corners"] = False
    return F.interpolate(tensor, **kwargs).squeeze(1).numpy()


def confidence_weights(confidence):
    confidence = np.asarray(confidence, dtype=np.float32)
    result = np.empty_like(confidence)
    for index, conf in enumerate(confidence):
        finite = np.isfinite(conf)
        if not finite.any():
            result[index] = 1.0
            continue
        low, high = np.percentile(conf[finite], [5.0, 95.0])
        scale = max(float(high - low), 1e-6)
        result[index] = np.clip((conf - low) / scale, 0.05, 1.0)
    return result


def estimate_overlap_scale(log_sum, weight_sum, start, new_depth):
    residuals = []
    for local_index in range(len(new_depth)):
        frame_index = start + local_index
        existing = weight_sum[frame_index] > 0
        valid = existing & np.isfinite(new_depth[local_index]) & (new_depth[local_index] > 0)
        if valid.sum() < 1024:
            continue
        old_log = log_sum[frame_index][valid] / weight_sum[frame_index][valid]
        new_log = np.log(new_depth[local_index][valid])
        delta = old_log - new_log
        delta = delta[np.isfinite(delta)]
        if delta.size:
            residuals.append(delta[:: max(1, delta.size // 20000)])
    if not residuals:
        return 1.0
    log_scale = float(np.median(np.concatenate(residuals)))
    return float(np.clip(math.exp(log_scale), 0.25, 4.0))


def safe_symlink(source, destination):
    source = source.resolve()
    if destination.is_symlink():
        if destination.resolve() != source:
            raise RuntimeError(f"Existing symlink points elsewhere: {destination}")
        return
    if destination.exists():
        raise RuntimeError(f"Refusing to replace existing path: {destination}")
    destination.symlink_to(source, target_is_directory=source.is_dir())


def save_tensor(path, tensor):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    torch.save(tensor, tmp)
    os.replace(tmp, path)


def colorize_depth(depth, valid):
    values = depth[valid]
    if values.size == 0:
        return np.zeros((*depth.shape, 3), dtype=np.uint8)
    lo, hi = np.percentile(values, [2.0, 98.0])
    normalized = np.clip((depth - lo) / max(float(hi - lo), 1e-6), 0.0, 1.0)
    colored = cv2.applyColorMap((normalized * 255).astype(np.uint8), cv2.COLORMAP_TURBO)
    colored = cv2.cvtColor(colored, cv2.COLOR_BGR2RGB)
    colored[~valid] = 0
    return colored


def save_scene_overview(scene_dir, output_scene, image_paths, depths, confidences):
    indices = sorted(set([0, len(image_paths) // 2, len(image_paths) - 1]))
    tile_w, tile_h = 420, 236
    header = 34
    canvas = Image.new("RGB", (3 * tile_w, header + len(indices) * tile_h), "white")
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()
    for column, label in enumerate(("Clean RGB", "DA3 pseudo depth", "DA3 confidence")):
        draw.text((column * tile_w + 8, 10), label, fill="black", font=font)
    for row, index in enumerate(indices):
        y = header + row * tile_h
        rgb = Image.open(image_paths[index]).convert("RGB").resize((tile_w, tile_h), Image.Resampling.LANCZOS)
        valid = np.isfinite(depths[index]) & (depths[index] > 0)
        depth_vis = Image.fromarray(colorize_depth(depths[index], valid)).resize((tile_w, tile_h))
        conf = confidences[index]
        finite = np.isfinite(conf)
        if finite.any():
            lo, hi = np.percentile(conf[finite], [2.0, 98.0])
            conf_vis = np.clip((conf - lo) / max(float(hi - lo), 1e-6), 0.0, 1.0)
        else:
            conf_vis = np.zeros_like(conf)
        conf_vis = Image.fromarray((conf_vis * 255).astype(np.uint8)).convert("RGB").resize((tile_w, tile_h))
        canvas.paste(rgb, (0, y))
        canvas.paste(depth_vis, (tile_w, y))
        canvas.paste(conf_vis, (2 * tile_w, y))
        draw.text((5, y + 5), f"frame {index:06d}", fill="white", stroke_width=2, stroke_fill="black")
    canvas.save(output_scene / "pseudo_depth_overview.jpg", quality=94)


def prepare_output_scene(source_scene, output_scene, scene_manifest):
    output_scene.mkdir(parents=True, exist_ok=True)
    safe_symlink(source_scene / "images", output_scene / "images")
    safe_symlink(source_scene / "events", output_scene / "events")
    safe_symlink(source_scene / "transforms.json", output_scene / "transforms.json")
    camera_yaml = source_scene / "cameras.yaml"
    if camera_yaml.exists():
        safe_symlink(camera_yaml, output_scene / "cameras.yaml")

    meta = read_json(source_scene / "meta.json")
    meta.update(
        {
            "pseudo_depth_source": "DA3-LARGE-1.1 pose-conditioned clean-RGB teacher",
            "pseudo_depth_protocol": "OpenGL c2w converted to OpenCV w2c; DA3 aligned to input extrinsic scale",
            "pseudo_depth_model_dir": scene_manifest["model_dir"],
            "pseudo_depth_is_ground_truth": False,
        }
    )
    atomic_write_json(output_scene / "meta.json", meta)


def complete_scene(output_scene, image_paths):
    manifest_path = output_scene / "pseudo_depth_manifest.json"
    if not manifest_path.is_file():
        return False
    try:
        manifest = read_json(manifest_path)
    except (OSError, json.JSONDecodeError):
        return False
    if manifest.get("status") != "complete":
        return False
    stems = [path.stem for path in image_paths]
    return all(
        (output_scene / "depth_pseudo" / f"{stem}.pt").is_file()
        and (output_scene / "depth_pseudo_conf" / f"{stem}.pt").is_file()
        for stem in stems
    )


def process_scene(model, record, args, selection):
    source_scene = Path(record["source_dir"])
    output_scene = args.output_root / record["scene"]
    image_paths = sorted((source_scene / "images").glob("*.png"))
    if complete_scene(output_scene, image_paths) and not args.overwrite:
        return {"scene": record["scene"], "status": "skipped-complete"}

    prepare_output_scene(source_scene, output_scene, selection)
    extrinsics, intrinsics, original_size, transforms = scene_cameras(source_scene, image_paths)
    num_frames = len(image_paths)
    target_h, target_w = args.target_size
    log_sum = np.zeros((num_frames, target_h, target_w), dtype=np.float64)
    weight_sum = np.zeros((num_frames, target_h, target_w), dtype=np.float64)
    conf_sum = np.zeros((num_frames, target_h, target_w), dtype=np.float64)
    conf_count = np.zeros((num_frames, target_h, target_w), dtype=np.float64)
    scale_factors = []
    starts = window_starts(num_frames, min(args.window_size, num_frames), args.window_overlap)
    started = time.time()

    for window_index, start in enumerate(starts):
        stop = min(start + args.window_size, num_frames)
        prediction = model.inference(
            [str(path) for path in image_paths[start:stop]],
            extrinsics=extrinsics[start:stop],
            intrinsics=intrinsics[start:stop],
            align_to_input_ext_scale=True,
            process_res=args.process_res,
            process_res_method="upper_bound_resize",
            export_dir=None,
            ref_view_strategy="middle",
        )
        depth = resize_maps(prediction.depth, args.target_size, "bilinear")
        if prediction.conf is None:
            confidence = np.ones_like(depth, dtype=np.float32)
        else:
            confidence = resize_maps(prediction.conf, args.target_size, "bilinear")
        scale = estimate_overlap_scale(log_sum, weight_sum, start, depth)
        depth = depth * scale
        scale_factors.append(scale)
        weights = confidence_weights(confidence)
        valid = np.isfinite(depth) & (depth > 1e-4) & (depth < 1e4) & np.isfinite(weights)
        for local_index in range(stop - start):
            frame_index = start + local_index
            frame_valid = valid[local_index]
            frame_weight = np.where(frame_valid, weights[local_index], 0.0)
            log_sum[frame_index] += np.where(frame_valid, np.log(np.clip(depth[local_index], 1e-6, None)), 0.0) * frame_weight
            weight_sum[frame_index] += frame_weight
            conf_sum[frame_index] += np.where(frame_valid, confidence[local_index], 0.0)
            conf_count[frame_index] += frame_valid.astype(np.float32)
        print(
            f"[{record['scene']}] window {window_index + 1}/{len(starts)} "
            f"frames {start}:{stop} overlap_scale={scale:.4f}",
            flush=True,
        )

    valid = weight_sum > 0
    depths = np.full_like(weight_sum, np.nan, dtype=np.float32)
    depths[valid] = np.exp(log_sum[valid] / weight_sum[valid]).astype(np.float32)
    confidences = np.zeros_like(depths, dtype=np.float32)
    conf_valid = conf_count > 0
    confidences[conf_valid] = (conf_sum[conf_valid] / conf_count[conf_valid]).astype(np.float32)

    for index, image_path in enumerate(image_paths):
        stem = image_path.stem
        save_tensor(output_scene / "depth_pseudo" / f"{stem}.pt", torch.from_numpy(depths[index]))
        save_tensor(
            output_scene / "depth_pseudo_conf" / f"{stem}.pt",
            torch.from_numpy(confidences[index]).to(torch.float16),
        )

    save_scene_overview(source_scene, output_scene, image_paths, depths, confidences)
    finite_depth = depths[np.isfinite(depths) & (depths > 0)]
    scene_manifest = {
        "status": "complete",
        "scene": record["scene"],
        "source_scene": record["source_scene"],
        "source_clip": record["source_clip"],
        "source_dir": str(source_scene),
        "model_dir": str(args.model_dir),
        "model_license": "CC BY-NC 4.0",
        "label_type": "pseudo depth; not ground truth",
        "pose_input": "OpenCV world-to-camera converted from DL3DV OpenGL camera-to-world",
        "input_intrinsics": "scaled from transforms.json to the stored RGB image resolution",
        "align_to_input_ext_scale": True,
        "frame_count": num_frames,
        "original_image_size": list(original_size),
        "saved_depth_size": [target_h, target_w],
        "window_size": args.window_size,
        "window_overlap": args.window_overlap,
        "window_count": len(starts),
        "window_overlap_scales": scale_factors,
        "process_res": args.process_res,
        "valid_fraction": float(valid.mean()),
        "depth_min": float(finite_depth.min()) if finite_depth.size else None,
        "depth_median": float(np.median(finite_depth)) if finite_depth.size else None,
        "depth_max": float(finite_depth.max()) if finite_depth.size else None,
        "elapsed_seconds": time.time() - started,
        "distortion": {
            key: transforms.get(key) for key in ("camera_model", "k1", "k2", "p1", "p2")
        },
    }
    atomic_write_json(output_scene / "pseudo_depth_manifest.json", scene_manifest)
    return scene_manifest


def main():
    args = parse_args()
    if args.num_shards <= 0 or not 0 <= args.shard_index < args.num_shards:
        raise ValueError("shard-index must be in [0, num-shards)")
    if args.clips_per_source <= 0:
        raise ValueError("clips-per-source must be positive")
    args.output_root.mkdir(parents=True, exist_ok=True)
    selection = build_selection(args)
    selection.update(
        {
            "model_dir": str(args.model_dir),
            "model_license": "CC BY-NC 4.0",
            "label_type": "DA3 pseudo depth, not sensor/dataset ground truth",
            "target_size": list(args.target_size),
            "window_size": args.window_size,
            "window_overlap": args.window_overlap,
            "process_res": args.process_res,
        }
    )
    atomic_write_json(args.output_root / "selection_manifest.json", selection)
    print(
        f"Selected {selection['selected_clip_count']} clips / "
        f"{selection['selected_frame_count']} frames from "
        f"{selection['selected_source_scene_count']} non-test source scenes",
        flush=True,
    )
    if args.selection_only:
        return

    records = selection["records"][args.shard_index :: args.num_shards]
    print(f"Shard {args.shard_index}/{args.num_shards}: {len(records)} clips", flush=True)
    model = DepthAnything3.from_pretrained(str(args.model_dir)).to(args.device)
    model.eval()
    results = []
    for scene_index, record in enumerate(records):
        print(f"Scene {scene_index + 1}/{len(records)}: {record['scene']}", flush=True)
        result = process_scene(model, record, args, selection)
        results.append(result)
        atomic_write_json(
            args.output_root / f"shard_{args.shard_index:02d}_progress.json",
            {
                "shard_index": args.shard_index,
                "num_shards": args.num_shards,
                "completed": len(results),
                "assigned": len(records),
                "results": results,
            },
        )
    atomic_write_json(
        args.output_root / f"shard_{args.shard_index:02d}_complete.json",
        {
            "status": "complete",
            "shard_index": args.shard_index,
            "num_shards": args.num_shards,
            "assigned": len(records),
            "results": results,
        },
    )


if __name__ == "__main__":
    main()
