#!/usr/bin/env python3
"""Package native-aspect inputs and aligned event-only baseline geometry."""

import io
import csv
import json
import random
import shutil
import sys
import zipfile
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.export_night3r_pair_pointcloud import render_points
from utils.export_rgb_lowlight_event_assets import event_red_blue, load_event
from dataset.lowlight_augmentation import ContinuousLowLightAug


OUTPUT = ROOT / "outputs/main_three_event_baselines_20260729"
PREDICTIONS = OUTPUT / "predictions"
METHODS = ("eventvggt", "depthanyevent", "depthanyevent_r")
DISPLAY_NAMES = {
    "eventvggt": "EventVGGT",
    "depthanyevent": "DepthAnyEvent",
    "depthanyevent_r": "DepthAnyEvent-R",
}
SCENES = (
    {
        "key": "01_dsec_screen-000004_000552-000553",
        "dataset": "DSEC",
        "scene": "screen-000004-dsec",
        "frames": (552, 553),
        "sample_id": "screen-000004-dsec:000552-000553",
        "rgb_zip": ROOT / "downloads/DSEC/zurich_city_11_b/zurich_city_11_b_images_rectified_left.zip",
        "rgb_members": ("001104.png", "001106.png"),
        "event_native_hw": (480, 640),
        "dataset_index": 2159,
        "lowlight": {"exposure": 0.004, "read_noise_std": 8.0e-5, "shot_noise_std": 2.0e-4, "seed": 17},
        "geometry_protocol": ROOT / "outputs/shared_scale_comparison_selected_20260727/dsec_final_gt_pair_000552/pairs/DSEC/screen-000004-dsec_000552-000553/geometry_assets/geometry_protocol.json",
    },
    {
        "key": "02_mvsec_screen-000157_000036-000037",
        "dataset": "MVSEC",
        "scene": "screen-000157-mvsec",
        "frames": (36, 37),
        "sample_id": "screen-000157-mvsec:000036-000037",
        "event_native_hw": (260, 346),
        "dataset_index": 516,
        "lowlight": {"exposure": 0.08, "read_noise_std": 3.0e-5, "shot_noise_std": 1.0e-4, "seed": 17},
        "geometry_protocol": ROOT / "outputs/shared_scale_comparison_selected_20260727/mvsec_final_gt_pairs_v3/pairs/MVSEC/screen-000157-mvsec_000036-000037/geometry_assets/geometry_protocol.json",
    },
    {
        "key": "03_mvsec_screen-000205_000047-000048",
        "dataset": "MVSEC",
        "scene": "screen-000205-mvsec",
        "frames": (47, 48),
        "sample_id": "screen-000205-mvsec:000047-000048",
        "event_native_hw": (260, 346),
        "dataset_index": 671,
        "lowlight": {"exposure": 0.08, "read_noise_std": 3.0e-5, "shot_noise_std": 1.0e-4, "seed": 17},
        "geometry_protocol": ROOT / "outputs/shared_scale_comparison_selected_20260727/mvsec_final_gt_pairs_v3/pairs/MVSEC/screen-000205-mvsec_000047-000048/geometry_assets/geometry_protocol.json",
    },
)


def safe_sample_name(value):
    return "".join(char if char.isalnum() or char in "-_" else "_" for char in value)


def tensor_2d(value):
    value = torch.as_tensor(value).detach().float().cpu().squeeze()
    if value.ndim != 2:
        raise ValueError(f"Expected a 2D tensor, got {tuple(value.shape)}")
    return value


def resize_depth(value, shape):
    value = tensor_2d(value)
    if tuple(value.shape) == tuple(shape):
        return value.numpy()
    resized = F.interpolate(
        value[None, None], size=tuple(shape), mode="bilinear", align_corners=False
    )[0, 0]
    return resized.numpy()


def valid_depth(value):
    return np.isfinite(value) & (value > 1.0e-6)


def align_and_measure(prediction, reference, reference_valid):
    valid = reference_valid & valid_depth(reference) & valid_depth(prediction)
    scale = float(np.median(reference[valid]) / max(np.median(prediction[valid]), 1.0e-8))
    aligned = prediction * scale
    ratio = np.maximum(aligned[valid] / reference[valid], reference[valid] / aligned[valid])
    return aligned.astype(np.float32), {
        "median_scale": scale,
        "valid_px": int(valid.sum()),
        "absrel": float(np.mean(np.abs(aligned[valid] - reference[valid]) / reference[valid])),
        "delta1": float(np.mean(ratio < 1.25)),
        "rmse_log": float(
            np.sqrt(np.mean((np.log(aligned[valid]) - np.log(reference[valid])) ** 2))
        ),
    }


def viridis_lut():
    values = np.arange(256, dtype=np.uint8)[:, None]
    return cv2.cvtColor(
        cv2.applyColorMap(values, cv2.COLORMAP_VIRIDIS), cv2.COLOR_BGR2RGB
    )[:, 0]


def colors(values, lo, hi):
    normalized = np.clip((values - lo) / max(hi - lo, 1.0e-6), 0.0, 1.0)
    return viridis_lut()[np.round(normalized * 255.0).astype(np.uint8)]


def save_depth(path, depth, lo, hi):
    valid = valid_depth(depth)
    image = np.zeros((*depth.shape, 3), dtype=np.uint8)
    image[valid] = colors(depth[valid], lo, hi)
    Image.fromarray(image).save(path, optimize=True)


def backproject(depth, intrinsics, stride=2, valid=None):
    height, width = depth.shape
    yy, xx = np.mgrid[0:height:stride, 0:width:stride]
    z = depth[::stride, ::stride]
    keep = valid_depth(z)
    if valid is not None:
        keep &= valid[::stride, ::stride]
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
    image[np.all(image == np.array([8, 12, 18], dtype=np.uint8), axis=2)] = 255
    return image


def native_rgb(scene, view):
    if scene["dataset"] == "MVSEC":
        frame = scene["frames"][view - 1]
        return Image.open(
            ROOT / "test" / scene["dataset"] / scene["scene"] / "images" / f"{frame:06d}.png"
        ).convert("RGB")
    suffix = scene["rgb_members"][view - 1]
    with zipfile.ZipFile(scene["rgb_zip"]) as archive:
        matches = [name for name in archive.namelist() if name.endswith(suffix)]
        if len(matches) != 1:
            raise RuntimeError(f"Could not uniquely resolve {suffix}: {matches}")
        image = Image.open(io.BytesIO(archive.read(matches[0]))).convert("RGB")
        image.load()
        return image


def native_event(scene, view):
    frame = scene["frames"][view - 1]
    path = ROOT / "test" / scene["dataset"] / scene["scene"] / "events" / f"{frame:06d}.pt"
    event = load_event(path)
    target = tuple(scene["event_native_hw"])
    if tuple(event.shape[-2:]) != target:
        event = F.interpolate(
            event[None], size=target, mode="bilinear", align_corners=False
        )[0]
    return event_red_blue(event, background="white")[0], path, tuple(event.shape)


def image_tensor(image):
    array = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
    return torch.from_numpy(array).permute(2, 0, 1).contiguous()


def tensor_image(tensor):
    array = (
        tensor.detach().float().clamp(0, 1).permute(1, 2, 0).mul(255).round().byte().numpy()
    )
    return Image.fromarray(array)


def native_dark_pair(scene, clean_images):
    protocol = scene["lowlight"]
    local_seed = int(protocol["seed"] + scene["dataset_index"] * 1009)
    python_state = random.getstate()
    numpy_state = np.random.get_state()
    torch_state = torch.random.get_rng_state()
    random.seed(local_seed)
    np.random.seed(local_seed % (2**32 - 1))
    torch.manual_seed(local_seed)
    try:
        augment = ContinuousLowLightAug(
            min_exposure=protocol["exposure"],
            max_exposure=protocol["exposure"],
            read_noise_std=protocol["read_noise_std"],
            shot_noise=True,
            shot_noise_std=protocol["shot_noise_std"],
            gamma_space_input=True,
        )
        dark = augment(torch.stack([image_tensor(image) for image in clean_images]))
    finally:
        random.setstate(python_state)
        np.random.set_state(numpy_state)
        torch.random.set_rng_state(torch_state)
    return [tensor_image(value) for value in dark], local_seed


def source_metadata(scene):
    scene_root = ROOT / "test" / scene["dataset"] / scene["scene"]
    scene_meta = json.loads((scene_root / "meta.json").read_text())
    with (scene_root / "meta/frames_meta.csv").open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    index_key = "out_idx" if scene["dataset"] == "DSEC" else "k"
    selected = {
        int(row[index_key]): row for row in rows if int(row[index_key]) in scene["frames"]
    }
    if set(selected) != set(scene["frames"]):
        raise RuntimeError(f"Missing frame metadata for {scene['sample_id']}")

    if scene["dataset"] == "DSEC":
        return {
            "original_dataset": "DSEC",
            "original_sequence": scene_meta["sequence"],
            "packed_scene": scene["scene"],
            "packed_pair": list(scene["frames"]),
            "original_frames": [
                {
                    "view": view,
                    "packed_frame_index": frame,
                    "source_frame_index": int(selected[frame]["source_idx"]),
                    "source_file": selected[frame]["source_file"],
                    "image_timestamp_us": int(selected[frame]["image_timestamp_us"]),
                    "disparity_timestamp_us": int(selected[frame]["disparity_timestamp_us"]),
                    "event_start_us": int(selected[frame]["event_start_us"]),
                    "event_end_us": int(selected[frame]["event_end_us"]),
                    "event_window_us": int(selected[frame]["event_end_us"])
                    - int(selected[frame]["event_start_us"]),
                    "num_events": int(selected[frame]["num_events"]),
                }
                for view, frame in enumerate(scene["frames"], start=1)
            ],
            "image_source": "official DSEC rectified-left image archive",
            "event_source": "DSEC events_left rectified with rectify_map.h5",
            "rectification": {
                "rgb_rectified": True,
                "rgb_method": "official images_rectified_left product",
                "event_rectified": bool(scene_meta["event_rectified"]),
                "event_method": "events_left/rectify_map.h5",
                "darkening_order": "after RGB rectification",
            },
            "depth_source": scene_meta["depth_source"],
            "packing_note": scene_meta["event_alignment_note"],
        }

    return {
        "original_dataset": "MVSEC",
        "original_sequence": scene_meta["source_scene"],
        "source_chunk_id": int(scene_meta["source_chunk_id"]),
        "global_screen_index": int(scene_meta["global_screen_index"]),
        "packed_scene": scene["scene"],
        "packed_pair": list(scene["frames"]),
        "original_frames": [
            {
                "view": view,
                "packed_frame_index": frame,
                "source_rgb_index": int(selected[frame]["rgb_index"]),
                "source_rgb_timestamp_us": int(selected[frame]["rgb_ts"]),
                "event_center_timestamp_us": int(selected[frame]["center_event_t"]),
                "event_start_us": int(selected[frame]["t_start"]),
                "event_end_us": int(selected[frame]["t_end"]),
                "event_window_us": int(selected[frame]["window_diameter"]),
                "num_events": int(selected[frame]["num_events"]),
                "source_depth_index": int(selected[frame]["depth_idx"]),
                "source_depth_timestamp_us": int(selected[frame]["depth_ts"]),
                "source_pose_index": int(selected[frame]["pose_idx"]),
                "source_pose_timestamp_us": int(selected[frame]["pose_ts"]),
            }
            for view, frame in enumerate(scene["frames"], start=1)
        ],
        "image_source": scene_meta["image_source"],
        "event_source": scene_meta["event_source"],
        "rectification": {
            "rgb_rectified": bool(scene_meta["export"]["rectified"]),
            "rgb_method": "outdoor_night_left_{x,y}_map",
            "event_rectified": bool(scene_meta["export"]["rectified"]),
            "event_method": "fisheye.undistortPoints(K,D,R,P)",
            "darkening_order": "after RGB rectification",
        },
        "depth_source": scene_meta["depth_source"],
        "pose_source": scene_meta["pose_source"],
        "export": scene_meta["export"],
    }


def labeled_grid(images, labels, output, panel_size=512):
    label_height = 34
    gap = 8
    canvas = Image.new(
        "RGB", (len(images) * panel_size + (len(images) - 1) * gap, panel_size + label_height), "white"
    )
    draw = ImageDraw.Draw(canvas)
    for index, (image, label) in enumerate(zip(images, labels)):
        left = index * (panel_size + gap)
        canvas.paste(image.resize((panel_size, panel_size), Image.Resampling.LANCZOS), (left, label_height))
        draw.text((left + 8, 10), label, fill="black")
    canvas.save(output, optimize=True)


def native_aspect_grid(images, labels, output, panel_width=640):
    resized = [
        image.resize(
            (panel_width, round(image.height * panel_width / image.width)),
            Image.Resampling.LANCZOS,
        )
        for image in images
    ]
    panel_height = max(image.height for image in resized)
    label_height = 34
    gap = 8
    canvas = Image.new(
        "RGB",
        (len(images) * panel_width + (len(images) - 1) * gap, panel_height + label_height),
        "white",
    )
    draw = ImageDraw.Draw(canvas)
    for index, (image, label) in enumerate(zip(resized, labels)):
        left = index * (panel_width + gap)
        top = label_height + (panel_height - image.height) // 2
        canvas.paste(image, (left, top))
        draw.text((left + 8, 10), label, fill="black")
    canvas.save(output, optimize=True)


def package_scene(scene):
    scene_dir = OUTPUT / "assets" / scene["key"]
    inputs_dir = scene_dir / "inputs_native_aspect"
    inputs_dir.mkdir(parents=True, exist_ok=True)
    provenance = source_metadata(scene)
    (scene_dir / "source_metadata.json").write_text(
        json.dumps(provenance, indent=2) + "\n"
    )
    clean_images = [native_rgb(scene, view) for view in (1, 2)]
    dark_images, pair_local_seed = native_dark_pair(scene, clean_images)
    exact_model_root = (
        ROOT
        / "outputs/shared_scale_comparison_selected_20260727/latest_three_groups_figure_assets_20260727"
        / scene["key"]
    )
    input_records = []
    input_images = []
    input_labels = []
    for view in (1, 2):
        rgb = clean_images[view - 1]
        dark = dark_images[view - 1]
        event, event_path, event_shape = native_event(scene, view)
        rgb_path = inputs_dir / f"view{view}_rgb_clean_native.png"
        dark_path = inputs_dir / f"view{view}_rgb_darken_native.png"
        exact_dark_path = inputs_dir / f"view{view}_rgb_darken_model512_exact.png"
        event_path_out = inputs_dir / f"view{view}_event_red_blue_native.png"
        rgb.save(rgb_path, optimize=True)
        dark.save(dark_path, optimize=True)
        shutil.copy2(
            exact_model_root / f"0{view + 2}_dark_rgb_view{view}_figure_512.png",
            exact_dark_path,
        )
        event.save(event_path_out, optimize=True)
        input_images.extend((dark, event))
        input_labels.extend((f"Darken RGB view {view}", f"Events view {view}"))
        input_records.append(
            {
                "view": view,
                "clean_rgb_size_wh": list(rgb.size),
                "darken_rgb_native_size_wh": list(dark.size),
                "darken_rgb_model_exact_size_wh": [512, 512],
                "event_size_wh": list(event.size),
                "rgb_aspect": rgb.width / rgb.height,
                "event_aspect": event.width / event.height,
                "clean_rgb": str(rgb_path),
                "darken_rgb_native": str(dark_path),
                "darken_rgb_model512_exact": str(exact_dark_path),
                "event_tensor_source": str(event_path),
                "event_tensor_shape_chw_after_aspect_restore": list(event_shape),
            }
        )
    native_aspect_grid(
        input_images,
        input_labels,
        scene_dir / "comparison_rgb_event_native_aspect.png",
    )

    protocol = json.loads(Path(scene["geometry_protocol"]).read_text())
    method_records = {}
    panels_depth = []
    panels_front = []
    panels_elevated = []
    reference_points = None
    lower = upper = None
    for method in METHODS:
        prediction_path = PREDICTIONS / method / f"{safe_sample_name(scene['sample_id'])}.pt"
        payload = torch.load(prediction_path, map_location="cpu", weights_only=True)
        reference = tensor_2d(payload["gt_depth"]).numpy()
        reference_valid = tensor_2d(payload["gt_valid"]).numpy().astype(bool)
        prediction = resize_depth(payload["pred_depth"], reference.shape)
        aligned, metrics = align_and_measure(prediction, reference, reference_valid)
        intrinsics = np.asarray(payload["intrinsics"], dtype=np.float32)
        if reference_points is None:
            reference_points = backproject(reference, intrinsics, stride=2, valid=reference_valid)
            lower, upper = robust_volume(reference_points)
        points = crop_points(backproject(aligned, intrinsics, stride=2), lower, upper)
        method_dir = scene_dir / method
        method_dir.mkdir(parents=True, exist_ok=True)
        np.save(method_dir / "depth_raw.npy", prediction)
        np.save(method_dir / "depth_median_aligned.npy", aligned)
        np.savez_compressed(method_dir / "geometry_points.npz", points=points)
        lo, hi = [float(value) for value in protocol["depth_color_range"]]
        depth_path = method_dir / "depth_median_aligned_viridis_512.png"
        save_depth(depth_path, aligned, lo, hi)
        geometry_paths = {}
        for view_name, camera_key in (("front", "front_camera"), ("elevated", "elevated_camera")):
            rendered = render_points(
                points,
                colors(points[:, 2], lo, hi),
                np.asarray(protocol[camera_key], dtype=np.float32),
                np.asarray(protocol["camera_center"], dtype=np.float32),
                512,
                512,
                1,
            )
            image = Image.fromarray(white_background(rendered))
            geometry_path = method_dir / f"geometry_{view_name}_512.png"
            image.save(geometry_path, optimize=True)
            geometry_paths[view_name] = geometry_path.name
        method_records[method] = {
            "display_name": DISPLAY_NAMES[method],
            "prediction": str(prediction_path),
            "alignment": metrics,
            "point_count_after_shared_gt_crop": int(len(points)),
            "depth": depth_path.name,
            "geometry": geometry_paths,
        }
        panels_depth.append(Image.open(depth_path).convert("RGB"))
        panels_front.append(Image.open(method_dir / "geometry_front_512.png").convert("RGB"))
        panels_elevated.append(Image.open(method_dir / "geometry_elevated_512.png").convert("RGB"))

    labels = [DISPLAY_NAMES[method] for method in METHODS]
    labeled_grid(panels_depth, labels, scene_dir / "comparison_aligned_depth_512.png")
    labeled_grid(panels_front, labels, scene_dir / "comparison_geometry_front_512.png")
    labeled_grid(panels_elevated, labels, scene_dir / "comparison_geometry_elevated_512.png")
    record = {
        "dataset": scene["dataset"],
        "scene": scene["scene"],
        "sample_id": scene["sample_id"],
        "source_metadata": provenance,
        "lowlight_protocol": {
            **scene["lowlight"],
            "dataset_index": scene["dataset_index"],
            "pair_local_seed": pair_local_seed,
            "gamma_space_input": True,
            "native_display_note": (
                "Protocol-matched darkening applied jointly at native resolution. "
                "Use darken_rgb_model512_exact for the pixel-exact inference input."
            ),
        },
        "inputs": input_records,
        "alignment_policy": "single per-sample median scale to sensor GT; no shift",
        "geometry_policy": {
            "coordinate_frame": "view-1 camera frame",
            "intrinsics": protocol["intrinsics"],
            "shared_crop": "GT 1%-99% point-cloud volume with 10% margin",
            "shared_cameras": {
                "center": protocol["camera_center"],
                "front": protocol["front_camera"],
                "elevated": protocol["elevated_camera"],
            },
            "canvas_wh": [512, 512],
            "point_stride": 2,
        },
        "methods": method_records,
    }
    (scene_dir / "manifest.json").write_text(json.dumps(record, indent=2) + "\n")
    return record


def main():
    records = [package_scene(scene) for scene in SCENES]
    rows = []
    for record in records:
        for method in METHODS:
            metrics = record["methods"][method]["alignment"]
            rows.append(
                (record["sample_id"], DISPLAY_NAMES[method], metrics["absrel"], metrics["delta1"], metrics["rmse_log"])
            )
    lines = [
        "# Main-three event-only baseline assets",
        "",
        "Darkened RGB and event input assets preserve the native sensor aspect ratio; clean RGB is retained for provenance.",
        "DSEC RGB is recovered from the official 1440x1080 rectified archive; its packed event voxel is inverse-resampled to the native 640x480 event aspect for display.",
        "MVSEC RGB and events remain at their native 346x260 resolution.",
        "Depth uses one median scale to sensor GT with no shift. All geometry panels share GT crop, camera, depth color range, and a 512x512 canvas within each scene.",
        "",
        "| Sample | Method | Aligned AbsRel | Aligned delta1 | Aligned RMSElog |",
        "|---|---|---:|---:|---:|",
    ]
    for sample, method, absrel, delta1, rmse_log in rows:
        lines.append(f"| {sample} | {method} | {absrel:.4f} | {delta1:.4f} | {rmse_log:.4f} |")
    (OUTPUT / "summary.md").write_text("\n".join(lines) + "\n")
    (OUTPUT / "manifest.json").write_text(json.dumps({"scenes": records}, indent=2) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
