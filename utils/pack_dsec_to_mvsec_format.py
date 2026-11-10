#!/usr/bin/env python3
import argparse
import csv
import json
import shutil
import sys
import tempfile
import zipfile
from bisect import bisect_left, bisect_right
from pathlib import Path

import h5py
import numpy as np
import torch
import yaml
from PIL import Image

try:
    import hdf5plugin  # noqa: F401
except ImportError:
    hdf5plugin = None

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from dataset.event_voxel import build_event_voxel


def parse_args():
    parser = argparse.ArgumentParser("Pack DSEC zips into the screen-* format used by DSECEventDataset/MVSEC")
    parser.add_argument("--root", default="/data/fcr/data/train_data/DSEC")
    parser.add_argument("--sequence", default="interlaken_00_c")
    parser.add_argument("--output-root", default=None)
    parser.add_argument("--scene-name", default=None)
    parser.add_argument("--image-size", type=int, nargs=2, default=None, metavar=("H", "W"))
    parser.add_argument("--num-bins", type=int, default=4)
    parser.add_argument("--event-window", choices=("between_frames", "centered"), default="between_frames")
    parser.add_argument("--polarity-separate", action="store_true", default=True)
    parser.add_argument("--no-normalize-events", dest="normalize_events", action="store_false")
    parser.add_argument("--disparity-scale", type=float, default=256.0)
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--keep-temp", action="store_true")
    return parser.parse_args()


def read_lines_int(path):
    out = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            out.append(int(line.split(",")[0].strip()))
    return out


def read_exposure(path):
    starts, ends = [], []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            a, b = [int(x.strip()) for x in line.split(",")[:2]]
            starts.append(a)
            ends.append(b)
    return starts, ends


def zip_png_names(path):
    with zipfile.ZipFile(path) as zf:
        return sorted(name for name in zf.namelist() if name.lower().endswith(".png"))


def read_zip_png(zf, name):
    with zf.open(name) as f:
        image = Image.open(f)
        image.load()
        return image


def resize_image(image, image_size, mode=Image.BILINEAR):
    if image_size is None:
        return image
    target_h, target_w = image_size
    if image.size == (target_w, target_h):
        return image
    return image.resize((target_w, target_h), mode)


def resize_depth_torch(depth, image_size):
    if image_size is None:
        return depth
    if tuple(depth.shape[-2:]) == tuple(image_size):
        return depth
    return (
        torch.nn.functional.interpolate(
            depth[None, None],
            size=tuple(image_size),
            mode="nearest",
        )[0, 0]
        .contiguous()
    )


def load_calibration(root, sequence):
    cal_zip = root / f"{sequence}_calibration.zip"
    with zipfile.ZipFile(cal_zip) as zf:
        raw = zf.read("cam_to_cam.yaml")
    return yaml.safe_load(raw)


def make_camera_yaml(calib, image_size=None):
    cam = calib["intrinsics"]["camRect1"]
    fx, fy, cx, cy = [float(x) for x in cam["camera_matrix"]]
    width, height = [int(x) for x in cam["resolution"]]
    if image_size is not None:
        target_h, target_w = image_size
        sx = target_w / float(width)
        sy = target_h / float(height)
        out_resolution = [target_w, target_h]
        out_intrinsics = [fx * sx, fy * sy, cx * sx, cy * sy]
    else:
        out_resolution = [width, height]
        out_intrinsics = [fx, fy, cx, cy]
    camera = {
        "distortion_model": "none",
        "distortion_coeffs": [0.0, 0.0, 0.0, 0.0],
        "intrinsics": out_intrinsics,
        "resolution": out_resolution,
    }
    return {
        "event_cam": camera,
        "rgb_cam_full": camera,
        "rgb_cam_crop": {**camera, "crop_offset_xy": [0, 0]},
    }


def disparity_to_depth(disparity_png, calib, disparity_scale):
    arr = np.asarray(disparity_png, dtype=np.float32)
    cam = calib["intrinsics"]["camRect1"]
    fx = float(cam["camera_matrix"][0])
    q32 = float(calib["disparity_to_depth"]["cams_12"][3][2])
    disp = arr / float(disparity_scale)
    depth = np.zeros_like(disp, dtype=np.float32)
    valid = disp > 0
    # DSEC Q[3,2] is approximately 1 / stereo_baseline.
    depth[valid] = fx / (disp[valid] * q32)
    return torch.from_numpy(depth)


def infer_event_keys(events_group):
    if {"x", "y", "t", "p"}.issubset(events_group.keys()):
        return "x", "y", "t", "p"
    if {"xs", "ys", "ts", "ps"}.issubset(events_group.keys()):
        return "xs", "ys", "ts", "ps"
    raise KeyError(f"Unsupported events.h5 keys: {list(events_group.keys())}")


def load_events_h5(events_zip, temp_root):
    temp_dir = Path(tempfile.mkdtemp(prefix="dsec_events_", dir=temp_root))
    with zipfile.ZipFile(events_zip) as zf:
        zf.extract("events.h5", temp_dir)
        rectify_path = None
        if "rectify_map.h5" in zf.namelist():
            zf.extract("rectify_map.h5", temp_dir)
            rectify_path = temp_dir / "rectify_map.h5"
    return temp_dir, temp_dir / "events.h5", rectify_path


def load_rectify_map(path):
    if path is None or not Path(path).is_file():
        return None
    with h5py.File(path, "r") as f:
        if "rectify_map" not in f:
            return None
        return f["rectify_map"][:].astype(np.float32)


def event_time_offset_us(events_file):
    if "t_offset" not in events_file:
        return 0
    value = events_file["t_offset"][()]
    if isinstance(value, np.ndarray):
        value = value.item()
    return int(value)


def absolute_event_timestamps(events_file, events_group, t_key):
    raw = events_group[t_key][:]
    if raw.dtype.kind == "f":
        raw = raw.astype(np.float64)
        if float(np.nanmax(raw)) < 1e5:
            raw = raw * 1_000_000.0
        raw = raw.astype(np.int64)
    else:
        raw = raw.astype(np.int64)
    return raw + event_time_offset_us(events_file)


def event_window_for_frame(frame_idx, image_ts, exposure_start, exposure_end, mode):
    if mode == "centered":
        if exposure_start and exposure_end and frame_idx < len(exposure_start):
            return int(exposure_start[frame_idx]), int(exposure_end[frame_idx])
        center = int(image_ts[frame_idx])
        if frame_idx == 0:
            half = (int(image_ts[1]) - center) // 2
        elif frame_idx + 1 >= len(image_ts):
            half = (center - int(image_ts[frame_idx - 1])) // 2
        else:
            half = (int(image_ts[frame_idx + 1]) - int(image_ts[frame_idx - 1])) // 4
        return center - half, center + half

    start = int(image_ts[frame_idx - 1]) if frame_idx > 0 else int(image_ts[frame_idx])
    end = int(image_ts[frame_idx])
    if end <= start and frame_idx + 1 < len(image_ts):
        end = int(image_ts[frame_idx + 1])
    return start, end


def slice_events(events_group, keys, ts_us, start_us, end_us):
    x_key, y_key, t_key, p_key = keys
    left = bisect_left(ts_us, start_us)
    right = bisect_right(ts_us, end_us)
    if right <= left:
        return np.empty((0, 4), dtype=np.float32)
    x = events_group[x_key][left:right].astype(np.float32)
    y = events_group[y_key][left:right].astype(np.float32)
    t = (ts_us[left:right] - int(start_us)).astype(np.float32)
    p = events_group[p_key][left:right].astype(np.float32)
    # DSEC polarity is commonly {0,1}; build_event_voxel treats p > 0 as positive.
    return np.stack([x, y, t, p], axis=1)


def transform_event_coordinates(events, rectify_map, source_resolution, target_hw):
    if events.size == 0:
        return events
    target_h, target_w = target_hw
    events = events.copy()
    if rectify_map is not None:
        map_h, map_w = rectify_map.shape[:2]
        x_raw = np.rint(events[:, 0]).astype(np.int64)
        y_raw = np.rint(events[:, 1]).astype(np.int64)
        valid = (x_raw >= 0) & (x_raw < map_w) & (y_raw >= 0) & (y_raw < map_h)
        if not np.any(valid):
            return np.empty((0, 4), dtype=np.float32)
        events = events[valid]
        xy_rect = rectify_map[y_raw[valid], x_raw[valid]]
        events[:, 0] = xy_rect[:, 0]
        events[:, 1] = xy_rect[:, 1]
        source_w, source_h = map_w, map_h
    else:
        source_w, source_h = source_resolution

    events[:, 0] *= float(target_w) / float(source_w)
    events[:, 1] *= float(target_h) / float(source_h)
    finite = np.isfinite(events[:, 0]) & np.isfinite(events[:, 1])
    return events[finite].astype(np.float32, copy=False)


def safe_write_text(path, lines):
    with open(path, "w") as f:
        for line in lines:
            f.write(f"{line}\n")


def main():
    args = parse_args()
    root = Path(args.root)
    output_root = Path(args.output_root) if args.output_root else root
    scene_name = args.scene_name or f"screen-000001-dsec-{args.sequence}"
    scene_dir = output_root / scene_name
    if scene_dir.exists():
        if not args.overwrite:
            raise FileExistsError(f"{scene_dir} exists. Pass --overwrite to replace it.")
        shutil.rmtree(scene_dir)

    image_size = tuple(args.image_size) if args.image_size is not None else None
    calib = load_calibration(root, args.sequence)

    image_zip = root / f"{args.sequence}_images_rectified_left.zip"
    event_zip = root / f"{args.sequence}_events_left.zip"
    disp_zip = root / f"{args.sequence}_disparity_image.zip"
    image_ts = read_lines_int(root / f"{args.sequence}_image_timestamps.txt")
    disp_ts = read_lines_int(root / f"{args.sequence}_disparity_timestamps.txt")
    exposure_start, exposure_end = read_exposure(root / f"{args.sequence}_image_exposure_timestamps_left.txt")

    image_names = set(zip_png_names(image_zip))
    disp_names = zip_png_names(disp_zip)
    selected = [name for name in disp_names if name in image_names]
    if args.max_frames is not None:
        selected = selected[: int(args.max_frames)]
    if not selected:
        raise RuntimeError("No image/disparity frames overlap.")

    for subdir in ("images", "events", "depth_gt", "meta"):
        (scene_dir / subdir).mkdir(parents=True, exist_ok=True)
    with open(scene_dir / "cameras.yaml", "w") as f:
        yaml.safe_dump(make_camera_yaml(calib, image_size=image_size), f, sort_keys=False)

    temp_root = Path("/tmp")
    temp_dir, events_h5_path, rectify_map_path = load_events_h5(event_zip, temp_root)
    rectify_map = load_rectify_map(rectify_map_path)
    event_source_resolution = tuple(calib["intrinsics"]["cam0"]["resolution"])
    records = []
    event_windows = []
    try:
        with h5py.File(events_h5_path, "r") as events_file, zipfile.ZipFile(image_zip) as image_zf, zipfile.ZipFile(disp_zip) as disp_zf:
            events_group = events_file["events"] if "events" in events_file else events_file
            keys = infer_event_keys(events_group)
            t_key = keys[2]
            ts_us = absolute_event_timestamps(events_file, events_group, t_key)
            event_t_offset_us = event_time_offset_us(events_file)

            for out_idx, name in enumerate(selected):
                source_idx = int(Path(name).stem)
                out_name = f"{out_idx:06d}"
                image = read_zip_png(image_zf, name).convert("RGB")
                image = resize_image(image, image_size, Image.BILINEAR)
                image.save(scene_dir / "images" / f"{out_name}.png")

                disp_img = read_zip_png(disp_zf, name)
                depth = disparity_to_depth(disp_img, calib, args.disparity_scale)
                depth = resize_depth_torch(depth.float(), image_size)
                torch.save(depth.contiguous(), scene_dir / "depth_gt" / f"{out_name}.pt")

                start_us, end_us = event_window_for_frame(
                    source_idx,
                    image_ts,
                    exposure_start,
                    exposure_end,
                    args.event_window,
                )
                events_np = slice_events(events_group, keys, ts_us, start_us, end_us)
                h, w = image_size if image_size is not None else (image.height, image.width)
                events_np = transform_event_coordinates(
                    events_np,
                    rectify_map,
                    event_source_resolution,
                    (h, w),
                )
                voxel = build_event_voxel(
                    events_np,
                    h,
                    w,
                    num_bins=args.num_bins,
                    polarity_separate=args.polarity_separate,
                    normalize=args.normalize_events,
                )
                if voxel is None:
                    voxel = torch.zeros((2 * args.num_bins, h, w), dtype=torch.float32)
                torch.save(voxel.float().contiguous(), scene_dir / "events" / f"{out_name}.pt")

                event_windows.append(f"{start_us},{end_us}")
                records.append(
                    {
                        "out_idx": out_idx,
                        "source_idx": source_idx,
                        "source_file": name,
                        "image_timestamp_us": int(image_ts[source_idx]),
                        "disparity_timestamp_us": int(disp_ts[source_idx // 2])
                        if source_idx // 2 < len(disp_ts)
                        else None,
                        "event_start_us": int(start_us),
                        "event_end_us": int(end_us),
                        "num_events": int(events_np.shape[0]),
                    }
                )
                if (out_idx + 1) % 25 == 0 or out_idx + 1 == len(selected):
                    print("packed_dsec_frame", {"done": out_idx + 1, "total": len(selected)})
    finally:
        if args.keep_temp:
            print("keeping temp events dir", temp_dir)
        else:
            shutil.rmtree(temp_dir, ignore_errors=True)

    safe_write_text(scene_dir / "meta" / "timestamp_rgb_us.txt", [r["image_timestamp_us"] for r in records])
    safe_write_text(scene_dir / "meta" / "timestamp_event_center_us.txt", [int((r["event_start_us"] + r["event_end_us"]) // 2) for r in records])
    safe_write_text(scene_dir / "meta" / "event_windows_us.txt", event_windows)

    with open(scene_dir / "meta" / "frames_meta.csv", "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "out_idx",
                "source_idx",
                "source_file",
                "image_timestamp_us",
                "disparity_timestamp_us",
                "event_start_us",
                "event_end_us",
                "num_events",
            ],
        )
        writer.writeheader()
        writer.writerows(records)

    with open(scene_dir / "meta.json", "w") as f:
        json.dump(
            {
                "source": "DSEC",
                "sequence": args.sequence,
                "frames": len(records),
                "image_size": list(image_size) if image_size is not None else None,
                "event_window": args.event_window,
                "num_bins": args.num_bins,
                "disparity_scale": args.disparity_scale,
                "depth_formula": "depth = fx / ((disp_raw / disparity_scale) * Q[3,2])",
                "depth_source": "dsec/disparity_image",
                "depth_units": "meters",
                "depth_invalid_value": 0.0,
                "depth_protocol": "sparse_metric_depth_zero_invalid",
                "depth_protocol_note": (
                    "Metric depth in the rectified RGB/event camera plane with zero as invalid. "
                    "This matches the MVSEC depth_gt convention semantically, although DSEC is "
                    "usually much sparser because the source is a disparity image."
                ),
                "event_time_offset_us": event_t_offset_us,
                "event_rectified": rectify_map is not None,
                "event_alignment_note": (
                    "DSEC event x/y are mapped through events_left/rectify_map.h5 to the "
                    "rectified event camera plane, then resized to the training image size. "
                    "This is not a full depth-aware event-camera to frame-camera projection."
                ),
            },
            f,
            indent=2,
        )

    summary = {
        "output_scene": str(scene_dir.resolve()),
        "frames": len(records),
        "has_depth_gt": True,
        "has_events": True,
        "has_images": True,
    }
    print("dsec_pack_done", summary)


if __name__ == "__main__":
    main()
