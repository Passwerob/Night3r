#!/usr/bin/env python3
import argparse
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
os.environ.setdefault("HF_HUB_OFFLINE", "1")

import imageio.v2 as imageio
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image


def parse_args():
    parser = argparse.ArgumentParser("Render DA3 camera-following point-cloud video")
    parser.add_argument("--da3-root", default="/data/fcr/Depth-Anything-3")
    parser.add_argument("--model-dir", default="/data/fcr/models/DA3/DA3-LARGE-1.1")
    parser.add_argument("--scene-dir", default="/data/fcr/data/train_data/DSEC/screen-000001-dsec")
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--num-frames", type=int, default=150)
    parser.add_argument("--fps", type=int, default=10)
    parser.add_argument("--process-res", type=int, default=504)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--points-per-frame", type=int, default=90000)
    parser.add_argument("--conf-percentile", type=float, default=20.0)
    parser.add_argument("--render-size", type=int, default=768)
    parser.add_argument("--point-radius", type=int, default=1)
    parser.add_argument("--display-gamma", type=float, default=0.7)
    parser.add_argument("--display-gain", type=float, default=1.15)
    parser.add_argument("--camera-height-offset", type=float, default=0.35)
    parser.add_argument("--camera-y-offset", type=float, default=0.0)
    parser.add_argument("--camera-lookahead", type=float, default=1.2)
    parser.add_argument("--focal-scale", type=float, default=0.9)
    parser.add_argument("--principal-y-shift", type=float, default=0.08)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--depth-scale-smoothing-window", type=int, default=1)
    parser.add_argument("--no-sky-filter", action="store_true")
    parser.add_argument("--outdir", required=True)
    return parser.parse_args()


def import_da3(root):
    root = Path(root)
    sys.path.insert(0, str(root / "src"))
    from depth_anything_3.api import DepthAnything3  # noqa: WPS433

    return DepthAnything3


def frame_paths(scene_dir, start, count):
    scene_dir = Path(scene_dir)
    images = sorted((scene_dir / "images").glob("*.png"))
    events = sorted((scene_dir / "events").glob("*.pt"))
    end = min(len(images), len(events), start + count)
    if end <= start:
        raise RuntimeError(f"No frames found in {scene_dir} from start {start}")
    return images[start:end], events[start:end]


def save_original_inputs(image_paths, outdir):
    outdir.mkdir(parents=True, exist_ok=True)
    copied_paths = []
    for idx, path in enumerate(image_paths):
        image = Image.open(path).convert("RGB")
        out_path = outdir / f"{idx:06d}.png"
        image.save(out_path)
        copied_paths.append(out_path)
    return copied_paths


def read_rgb_01(path):
    return np.asarray(Image.open(path).convert("RGB"), dtype=np.float32) / 255.0


def event_activity_image(event_path):
    try:
        event = torch.load(event_path, map_location="cpu", weights_only=True)
    except TypeError:
        event = torch.load(event_path, map_location="cpu")
    activity = event.float().abs().sum(dim=0).numpy()
    finite = activity[np.isfinite(activity)]
    vmax = float(np.percentile(finite, 99.5)) if finite.size else 1.0
    activity = np.clip(activity / max(vmax, 1e-6), 0, 1)
    return activity


def resize_u8(image, size):
    image_u8 = (np.clip(image, 0, 1) * 255.0 + 0.5).astype(np.uint8)
    return np.asarray(Image.fromarray(image_u8).resize((size[1], size[0]), Image.BILINEAR))


def event_to_rgb(event_img, size):
    cmap = plt.get_cmap("magma")
    rgb = (cmap(event_img)[..., :3] * 255.0 + 0.5).astype(np.uint8)
    if rgb.shape[:2] != size:
        rgb = np.asarray(Image.fromarray(rgb).resize((size[1], size[0]), Image.BILINEAR))
    return rgb


def visualize_depth(depth_np, percentile=2, cmap_name="Spectral"):
    finite = depth_np[np.isfinite(depth_np)]
    if finite.size == 0:
        return np.zeros((*depth_np.shape, 3), dtype=np.uint8)
    vmin = np.percentile(finite, percentile)
    vmax = np.percentile(finite, 100 - percentile)
    depth_norm = np.clip((depth_np - vmin) / (vmax - vmin + 1e-8), 0, 1)
    cmap = matplotlib.colormaps.get_cmap(cmap_name)
    return (cmap(depth_norm)[..., :3] * 255.0 + 0.5).astype(np.uint8)


def scale_intrinsic(intrinsic, model_hw, render_hw):
    model_h, model_w = model_hw
    render_h, render_w = render_hw
    scaled = intrinsic.astype(np.float32).copy()
    sx = render_w / float(model_w)
    sy = render_h / float(model_h)
    scaled[0, 0] *= sx
    scaled[0, 2] *= sx
    scaled[1, 1] *= sy
    scaled[1, 2] *= sy
    return scaled


def fallback_intrinsic(hw):
    h, w = hw
    intrinsic = np.eye(3, dtype=np.float32)
    focal = max(h, w) * 0.85
    intrinsic[0, 0] = focal
    intrinsic[1, 1] = focal
    intrinsic[0, 2] = (w - 1) * 0.5
    intrinsic[1, 2] = (h - 1) * 0.5
    return intrinsic


def point_cloud_from_depth(depth, conf, sky, image, intrinsic, max_points, conf_percentile, seed, use_sky_filter):
    depth = depth.astype(np.float32)
    h, w = depth.shape
    if image.shape[:2] != (h, w):
        image_u8 = (np.clip(image, 0, 1) * 255.0 + 0.5).astype(np.uint8)
        image = np.asarray(Image.fromarray(image_u8).resize((w, h), Image.BILINEAR), dtype=np.float32) / 255.0

    ys, xs = np.mgrid[0:h, 0:w].astype(np.float32)
    fx, fy = intrinsic[0, 0], intrinsic[1, 1]
    cx, cy = intrinsic[0, 2], intrinsic[1, 2]
    z = depth
    x = (xs - cx) / max(float(fx), 1e-6) * z
    y = (ys - cy) / max(float(fy), 1e-6) * z
    pts = np.stack([x, y, z], axis=-1).reshape(-1, 3).astype(np.float32)
    colors = (image.reshape(-1, 3) * 255.0 + 0.5).astype(np.uint8)

    finite = np.isfinite(pts).all(axis=1) & (pts[:, 2] > 1e-5)
    if conf is not None:
        conf_flat = conf.reshape(-1).astype(np.float32)
        conf_ok = np.isfinite(conf_flat)
        if (finite & conf_ok).any():
            thr = float(np.percentile(conf_flat[finite & conf_ok], conf_percentile))
            finite = finite & conf_ok & (conf_flat >= thr)
    if use_sky_filter and sky is not None:
        sky_flat = sky.reshape(-1).astype(np.float32)
        finite = finite & (sky_flat < 0.5)

    pts = pts[finite]
    colors = colors[finite]
    if len(pts) > max_points:
        rng = np.random.default_rng(seed)
        keep = rng.choice(len(pts), size=max_points, replace=False)
        pts = pts[keep]
        colors = colors[keep]
    return pts, colors


def robust_depth_median(depth, conf=None, sky=None, conf_percentile=0.0, use_sky_filter=True):
    valid = np.isfinite(depth) & (depth > 1e-5)
    if conf is not None:
        conf = conf.astype(np.float32)
        conf_valid = valid & np.isfinite(conf)
        if conf_valid.any():
            thr = float(np.percentile(conf[conf_valid], conf_percentile))
            valid = valid & (conf >= thr)
    if use_sky_filter and sky is not None:
        valid = valid & (sky.astype(np.float32) < 0.5)
    vals = depth[valid]
    if vals.size == 0:
        return float("nan")
    return float(np.median(vals))


def smooth_depth_scales(depth_medians, window):
    depth_medians = np.asarray(depth_medians, dtype=np.float32)
    if window <= 1 or len(depth_medians) == 0:
        return np.ones_like(depth_medians, dtype=np.float32)
    if window % 2 == 0:
        window += 1

    finite = np.isfinite(depth_medians) & (depth_medians > 1e-6)
    if not finite.any():
        return np.ones_like(depth_medians, dtype=np.float32)
    filled = depth_medians.copy()
    idx = np.arange(len(filled))
    filled[~finite] = np.interp(idx[~finite], idx[finite], filled[finite])
    half = window // 2
    smoothed = np.empty_like(filled)
    for i in range(len(filled)):
        lo = max(0, i - half)
        hi = min(len(filled), i + half + 1)
        smoothed[i] = np.median(filled[lo:hi])
    return np.clip(smoothed / np.maximum(filled, 1e-6), 0.5, 2.0).astype(np.float32)


def apply_local_virtual_camera(points_cam, height_offset, y_offset, lookahead):
    if height_offset == 0.0 and y_offset == 0.0:
        return points_cam
    if lookahead <= 0.0:
        points_cam = points_cam.copy()
        points_cam[:, 1] += height_offset
        points_cam[:, 2] -= y_offset
        return points_cam

    # User-local frame: x right, y into screen, z up.
    user_pts = np.stack([points_cam[:, 0], points_cam[:, 2], -points_cam[:, 1]], axis=1)
    cam_pos = np.array([0.0, y_offset, height_offset], dtype=np.float32)
    target = np.array([0.0, lookahead, 0.0], dtype=np.float32)
    up_hint = np.array([0.0, 0.0, 1.0], dtype=np.float32)

    forward = target - cam_pos
    forward = forward / max(np.linalg.norm(forward), 1e-6)
    right = np.cross(forward, up_hint)
    right = right / max(np.linalg.norm(right), 1e-6)
    down = np.cross(forward, right)
    down = down / max(np.linalg.norm(down), 1e-6)

    rel = user_pts - cam_pos[None, :]
    return np.stack([rel @ right, rel @ down, rel @ forward], axis=1).astype(np.float32)


def render_projected_cloud(points_cam, colors, intrinsic, render_hw, args):
    render_h, render_w = render_hw
    canvas = np.zeros((render_h, render_w, 3), dtype=np.uint8)
    if len(points_cam) == 0:
        return canvas

    z = points_cam[:, 2]
    positive = z > 1e-5
    negative = z < -1e-5
    if int(negative.sum()) > int(positive.sum()):
        denom = -z
        valid_z = negative
    else:
        denom = z
        valid_z = positive

    fx, fy = intrinsic[0, 0], intrinsic[1, 1]
    cx, cy = intrinsic[0, 2], intrinsic[1, 2]
    u = fx * (points_cam[:, 0] / np.maximum(denom, 1e-6)) + cx
    v = fy * (points_cam[:, 1] / np.maximum(denom, 1e-6)) + cy
    valid = (
        valid_z
        & np.isfinite(u)
        & np.isfinite(v)
        & (u >= -args.point_radius)
        & (u < render_w + args.point_radius)
        & (v >= -args.point_radius)
        & (v < render_h + args.point_radius)
    )
    if not valid.any():
        return canvas

    base_u = np.rint(u[valid]).astype(np.int32)
    base_v = np.rint(v[valid]).astype(np.int32)
    depth = denom[valid].astype(np.float32)
    color = colors[valid].astype(np.float32) / 255.0
    color = np.clip(np.power(color, args.display_gamma) * args.display_gain, 0.0, 1.0)
    color = (color * 255.0 + 0.5).astype(np.uint8)

    flat = canvas.reshape(-1, 3)
    for dy in range(-args.point_radius, args.point_radius + 1):
        for dx in range(-args.point_radius, args.point_radius + 1):
            px = base_u + dx
            py = base_v + dy
            inside = (px >= 0) & (px < render_w) & (py >= 0) & (py < render_h)
            if not inside.any():
                continue
            lin = py[inside] * render_w + px[inside]
            zbuf = np.full(render_h * render_w, np.inf, dtype=np.float32)
            np.minimum.at(zbuf, lin, depth[inside])
            keep = depth[inside] <= zbuf[lin] + 1e-6
            flat[lin[keep]] = color[inside][keep]
    return canvas


def render_frame(path, image_path, event_path, points_cam, colors, intrinsic, model_hw, args):
    image = read_rgb_01(image_path)
    event_img = event_activity_image(event_path)
    pts = apply_local_virtual_camera(
        points_cam,
        args.camera_height_offset,
        args.camera_y_offset,
        args.camera_lookahead,
    )
    render_hw = (args.render_size, args.render_size)
    intrinsic = scale_intrinsic(intrinsic, model_hw, render_hw)
    intrinsic[0, 0] *= args.focal_scale
    intrinsic[1, 1] *= args.focal_scale
    intrinsic[1, 2] += args.principal_y_shift * args.render_size
    reconstruction = render_projected_cloud(pts, colors, intrinsic, render_hw, args)

    left_size = args.render_size // 2
    image_panel = resize_u8(image, (left_size, left_size))
    event_panel = event_to_rgb(event_img, (left_size, left_size))
    canvas = np.zeros((args.render_size, args.render_size + left_size, 3), dtype=np.uint8)
    canvas[:left_size, :left_size] = image_panel
    canvas[left_size:, :left_size] = event_panel
    canvas[:, left_size:] = reconstruction
    Image.fromarray(canvas).save(path)


def write_ply(path, points, colors):
    with open(path, "w") as f:
        f.write("ply\nformat ascii 1.0\n")
        f.write(f"element vertex {len(points)}\n")
        f.write("property float x\nproperty float y\nproperty float z\n")
        f.write("property uchar red\nproperty uchar green\nproperty uchar blue\n")
        f.write("end_header\n")
        for p, c in zip(points, colors):
            f.write(f"{p[0]:.6f} {p[1]:.6f} {p[2]:.6f} {int(c[0])} {int(c[1])} {int(c[2])}\n")


def encode_video(frames_dir, output_path, fps):
    frame_paths = sorted(Path(frames_dir).glob("frame_*.png"))
    try:
        with imageio.get_writer(str(output_path), fps=fps, codec="libx264", quality=8, macro_block_size=16) as writer:
            for frame_path in frame_paths:
                writer.append_data(imageio.imread(frame_path))
    except Exception:
        import subprocess

        subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-framerate",
                str(fps),
                "-i",
                str(Path(frames_dir) / "frame_%04d.png"),
                "-vf",
                "pad=ceil(iw/2)*2:ceil(ih/2)*2",
                "-c:v",
                "libx264",
                "-pix_fmt",
                "yuv420p",
                "-movflags",
                "+faststart",
                str(output_path),
            ],
            check=True,
        )


def write_index(outdir, manifest):
    html = [
        "<!doctype html><meta charset='utf-8'>",
        "<title>DA3 Follow Video</title>",
        "<style>body{font-family:Arial,sans-serif;margin:24px;background:#111;color:#eee}"
        "video,img{max-width:100%;display:block;margin:12px 0 28px;background:#000;border:1px solid #333}"
        "a{color:#9bd}code{color:#ddd}</style>",
        "<h1>DA3 Pose-following Reconstruction</h1>",
        f"<video controls src='{manifest['outputs']['video']}'></video>",
        "<ul>",
    ]
    for key, value in manifest["outputs"].items():
        html.append(f"<li>{key}: <a href='{value}'>{value}</a></li>")
    html.extend(["</ul><h2>Preview frames</h2>"])
    for idx in [0, len(manifest["frames"]) // 2, len(manifest["frames"]) - 1]:
        frame = manifest["frames"][idx]
        html.append(f"<h3>frame {frame['frame']}</h3><img src='frames/frame_{frame['frame']:04d}.png'>")
    html.append("</body>")
    (Path(outdir) / "index.html").write_text("\n".join(html), encoding="utf-8")


def main():
    args = parse_args()
    outdir = Path(args.outdir)
    input_dir = outdir / "input_images"
    frames_dir = outdir / "frames"
    ply_dir = outdir / "ply"
    depth_dir = outdir / "depth_da3"
    depth_vis_dir = outdir / "depth_da3_vis"
    for path in (input_dir, frames_dir, ply_dir, depth_dir, depth_vis_dir):
        path.mkdir(parents=True, exist_ok=True)
    for pattern, directory in (
        ("frame_*.png", frames_dir),
        ("da3_frame_*.ply", ply_dir),
        ("*.pt", depth_dir),
        ("*.png", depth_vis_dir),
    ):
        for stale in directory.glob(pattern):
            stale.unlink()

    image_paths, event_paths = frame_paths(args.scene_dir, args.start_index, args.num_frames)
    input_paths = save_original_inputs(image_paths, input_dir)

    DepthAnything3 = import_da3(args.da3_root)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = DepthAnything3.from_pretrained(args.model_dir).to(device).eval()

    predictions = []
    for batch_start in range(0, len(input_paths), args.batch_size):
        batch_end = min(len(input_paths), batch_start + args.batch_size)
        batch_paths = input_paths[batch_start:batch_end]
        with torch.no_grad():
            pred = model.inference(
                image=[str(path) for path in batch_paths],
                process_res=args.process_res,
                process_res_method="upper_bound_resize",
                export_dir=None,
            )

        depths = pred.depth
        confs = pred.conf if pred.conf is not None else [None] * len(batch_paths)
        skies = pred.sky if pred.sky is not None else [None] * len(batch_paths)
        images = pred.processed_images
        if images is None:
            images = np.stack([read_rgb_01(path) for path in batch_paths], axis=0)
        intrinsics = pred.intrinsics

        for local_idx, image_path in enumerate(batch_paths):
            frame_idx = batch_start + local_idx
            depth = depths[local_idx].astype(np.float32)
            model_hw = depth.shape
            intrinsic = intrinsics[local_idx].astype(np.float32) if intrinsics is not None else fallback_intrinsic(model_hw)
            image = images[local_idx].astype(np.float32)
            if image.max() > 2.0:
                image = image / 255.0
            conf = confs[local_idx]
            sky = skies[local_idx]
            depth_median = robust_depth_median(
                depth,
                conf,
                sky,
                conf_percentile=args.conf_percentile,
                use_sky_filter=not args.no_sky_filter,
            )
            predictions.append(
                {
                    "frame_idx": frame_idx,
                    "image_path": image_path,
                    "depth": depth,
                    "conf": conf,
                    "sky": sky,
                    "image": image,
                    "intrinsic": intrinsic,
                    "model_hw": model_hw,
                    "batch_start": batch_start,
                    "batch_end": batch_end,
                    "depth_median": depth_median,
                }
            )

        del pred, depths, confs, skies, images, intrinsics
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    depth_scales = smooth_depth_scales(
        [item["depth_median"] for item in predictions],
        args.depth_scale_smoothing_window,
    )

    ply_save_indices = {0, len(input_paths) // 2, len(input_paths) - 1}
    records = []
    for item, depth_scale in zip(predictions, depth_scales):
            frame_idx = item["frame_idx"]
            image_path = item["image_path"]
            depth = item["depth"] * float(depth_scale)
            model_hw = item["model_hw"]
            intrinsic = item["intrinsic"]
            image = item["image"]
            pts_cam, colors = point_cloud_from_depth(
                depth,
                item["conf"],
                item["sky"],
                image,
                intrinsic,
                max_points=args.points_per_frame,
                conf_percentile=args.conf_percentile,
                seed=args.seed + frame_idx,
                use_sky_filter=not args.no_sky_filter,
            )
            torch.save(torch.from_numpy(depth).float(), depth_dir / f"{frame_idx:06d}.pt")
            Image.fromarray(visualize_depth(depth)).save(depth_vis_dir / f"{frame_idx:06d}.png")
            render_frame(
                frames_dir / f"frame_{frame_idx:04d}.png",
                image_path,
                event_paths[frame_idx],
                pts_cam,
                colors,
                intrinsic,
                model_hw,
                args,
            )
            if frame_idx in ply_save_indices:
                write_ply(ply_dir / f"da3_frame_{frame_idx:04d}_camera_color.ply", pts_cam, colors)

            records.append(
                {
                    "frame": int(frame_idx),
                    "source_image": str(image_paths[frame_idx]),
                    "input_image": str(image_path),
                    "event": str(event_paths[frame_idx]),
                    "depth": f"depth_da3/{frame_idx:06d}.pt",
                    "depth_vis": f"depth_da3_vis/{frame_idx:06d}.png",
                    "num_points": int(len(pts_cam)),
                    "model_hw": [int(v) for v in model_hw],
                    "batch_start": int(item["batch_start"]),
                    "batch_end_exclusive": int(item["batch_end"]),
                    "raw_depth_median": float(item["depth_median"]),
                    "depth_scale": float(depth_scale),
                }
            )

    video_path = outdir / "da3_follow.mp4"
    encode_video(frames_dir, video_path, args.fps)
    manifest = {
        "scene_dir": args.scene_dir,
        "model_dir": args.model_dir,
        "start_index": args.start_index,
        "num_frames": len(input_paths),
        "fps": args.fps,
        "duration_s": len(input_paths) / float(args.fps),
        "process_res": args.process_res,
        "batch_size": args.batch_size,
        "points_per_frame": args.points_per_frame,
        "conf_percentile": args.conf_percentile,
        "camera_height_offset": args.camera_height_offset,
        "camera_y_offset": args.camera_y_offset,
        "camera_lookahead": args.camera_lookahead,
        "focal_scale": args.focal_scale,
        "principal_y_shift": args.principal_y_shift,
        "depth_scale_smoothing_window": args.depth_scale_smoothing_window,
        "render_note": "Left panels are original RGB and event activity. Right panel is a raised camera-local projection of DA3 depth points for each current frame. The virtual camera follows the current frame and is offset upward in the user-local z axis, looking slightly down toward the original xy plane.",
        "outputs": {
            "video": "da3_follow.mp4",
            "frames_dir": "frames",
            "input_images": "input_images",
            "depth_dir": "depth_da3",
            "depth_vis_dir": "depth_da3_vis",
            "ply_dir": "ply",
        },
        "frames": records,
    }
    (outdir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    write_index(outdir, manifest)
    print(json.dumps({"outdir": str(outdir), "video": str(video_path), "frames": len(input_paths)}, indent=2))


if __name__ == "__main__":
    main()
