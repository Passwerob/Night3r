#!/usr/bin/env python3
import argparse
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from dataset.lowlight_augmentation import ContinuousLowLightAug  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser("Render StreamVGGT camera-following point-cloud video")
    parser.add_argument("--streamvggt-root", default="/data/fcr/code/fcr/StreamVGGT")
    parser.add_argument("--checkpoint", default="/data/fcr/code/fcr/StreamVGGT/ckpts/model.pt")
    parser.add_argument("--scene-dir", default="/data/fcr/data/train_data/DSEC/screen-000001-dsec")
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--num-frames", type=int, default=100)
    parser.add_argument("--fps", type=int, default=10)
    parser.add_argument("--exposure", type=float, default=0.03)
    parser.add_argument("--use-original-images", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--history", type=int, default=28)
    parser.add_argument("--points-per-frame", type=int, default=6500)
    parser.add_argument("--conf-percentile", type=float, default=50.0)
    parser.add_argument("--render-size", type=int, default=1024)
    parser.add_argument("--point-radius", type=int, default=1)
    parser.add_argument("--display-gamma", type=float, default=0.7)
    parser.add_argument("--display-gain", type=float, default=1.15)
    parser.add_argument("--camera-height-offset", type=float, default=0.0)
    parser.add_argument("--camera-y-offset", type=float, default=0.0)
    parser.add_argument("--camera-lookahead", type=float, default=0.0)
    parser.add_argument("--focal-scale", type=float, default=1.0)
    parser.add_argument("--principal-y-shift", type=float, default=0.0)
    parser.add_argument("--chunk-size", type=int, default=0)
    parser.add_argument("--outdir", required=True)
    return parser.parse_args()


def import_streamvggt(root):
    root = Path(root)
    sys.path.insert(0, str(root))
    sys.path.insert(0, str(root / "src"))
    from inference import StreamVGGTInference  # noqa: WPS433

    return StreamVGGTInference


def frame_paths(scene_dir, start, count):
    scene_dir = Path(scene_dir)
    images = sorted((scene_dir / "images").glob("*.png"))
    events = sorted((scene_dir / "events").glob("*.pt"))
    end = min(len(images), len(events), start + count)
    if end <= start:
        raise RuntimeError(f"No frames found in {scene_dir} from start {start}")
    return images[start:end], events[start:end]


def load_image_01(path, size=(512, 512)):
    image = Image.open(path).convert("RGB")
    if image.size != (size[1], size[0]):
        image = image.resize((size[1], size[0]), Image.BILINEAR)
    return np.asarray(image, dtype=np.float32) / 255.0


def save_darkened_inputs(image_paths, outdir, exposure, seed):
    torch.manual_seed(seed)
    outdir.mkdir(parents=True, exist_ok=True)
    aug = ContinuousLowLightAug(
        min_exposure=exposure,
        max_exposure=exposure,
        read_noise_std=0.01,
        shot_noise=True,
        gamma_space_input=True,
    )
    dark_paths = []
    for idx, path in enumerate(image_paths):
        image = torch.from_numpy(load_image_01(path)).permute(2, 0, 1)
        dark = aug(image).clamp(0, 1).permute(1, 2, 0).numpy()
        out_path = outdir / f"{idx:06d}.png"
        Image.fromarray((dark * 255.0 + 0.5).astype(np.uint8)).save(out_path)
        dark_paths.append(out_path)
    return dark_paths


def save_original_inputs(image_paths, outdir):
    outdir.mkdir(parents=True, exist_ok=True)
    copied_paths = []
    for idx, path in enumerate(image_paths):
        image = Image.open(path).convert("RGB")
        out_path = outdir / f"{idx:06d}.png"
        image.save(out_path)
        copied_paths.append(out_path)
    return copied_paths


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


def read_dark_image(path):
    return np.asarray(Image.open(path).convert("RGB"), dtype=np.float32) / 255.0


def closed_form_inverse_se3(se3):
    out = np.eye(4, dtype=np.float32)
    if se3.shape == (3, 4):
        mat = np.eye(4, dtype=np.float32)
        mat[:3, :4] = se3
    else:
        mat = se3.astype(np.float32)
    r = mat[:3, :3]
    t = mat[:3, 3]
    out[:3, :3] = r.T
    out[:3, 3] = -r.T @ t
    return out


def transform_points(mat, points):
    return points @ mat[:3, :3].T + mat[:3, 3]


def point_cloud_from_prediction(points, conf, image, max_points, conf_percentile, seed):
    if image.shape[:2] != points.shape[:2]:
        image_u8 = (np.clip(image, 0, 1) * 255.0 + 0.5).astype(np.uint8)
        image = np.asarray(
            Image.fromarray(image_u8).resize((points.shape[1], points.shape[0]), Image.BILINEAR),
            dtype=np.float32,
        ) / 255.0
    pts = points.reshape(-1, 3).astype(np.float32)
    colors = (image.reshape(-1, 3) * 255.0 + 0.5).astype(np.uint8)
    conf = conf.reshape(-1)
    finite = np.isfinite(pts).all(axis=1) & np.isfinite(conf) & (np.linalg.norm(pts, axis=1) > 1e-6)
    if finite.any():
        thr = float(np.percentile(conf[finite], conf_percentile))
        finite = finite & (conf >= thr)
    pts = pts[finite]
    colors = colors[finite]
    if len(pts) > max_points:
        rng = np.random.default_rng(seed)
        keep = rng.choice(len(pts), size=max_points, replace=False)
        pts = pts[keep]
        colors = colors[keep]
    return pts, colors


def write_ply(path, points, colors):
    with open(path, "w") as f:
        f.write("ply\nformat ascii 1.0\n")
        f.write(f"element vertex {len(points)}\n")
        f.write("property float x\nproperty float y\nproperty float z\n")
        f.write("property uchar red\nproperty uchar green\nproperty uchar blue\n")
        f.write("end_header\n")
        for p, c in zip(points, colors):
            f.write(f"{p[0]:.6f} {p[1]:.6f} {p[2]:.6f} {int(c[0])} {int(c[1])} {int(c[2])}\n")


def camera_to_display(points):
    return np.stack([points[:, 0], points[:, 2], -points[:, 1]], axis=1)


def set_equal_axes(ax, pts):
    if pts.size == 0:
        return
    mins = np.percentile(pts, 2, axis=0)
    maxs = np.percentile(pts, 98, axis=0)
    center = (mins + maxs) / 2.0
    radius = max(float(np.max(maxs - mins) / 2.0), 1e-3)
    ax.set_xlim(center[0] - radius, center[0] + radius)
    ax.set_ylim(center[1] - radius, center[1] + radius)
    ax.set_zlim(center[2] - radius, center[2] + radius)


def resize_u8(image, size):
    image_u8 = (np.clip(image, 0, 1) * 255.0 + 0.5).astype(np.uint8)
    return np.asarray(Image.fromarray(image_u8).resize((size[1], size[0]), Image.BILINEAR))


def event_to_rgb(event_img, size):
    cmap = plt.get_cmap("magma")
    rgb = (cmap(event_img)[..., :3] * 255.0 + 0.5).astype(np.uint8)
    if rgb.shape[:2] != size:
        rgb = np.asarray(Image.fromarray(rgb).resize((size[1], size[0]), Image.BILINEAR))
    return rgb


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


def render_frame(
    path,
    idx,
    dark_path,
    event_path,
    history_clouds,
    current_world_to_camera,
    current_intrinsic,
    model_hw,
    args,
):
    dark = read_dark_image(dark_path)
    event_img = event_activity_image(event_path)
    pts_world = np.concatenate([item["points_world"] for item in history_clouds], axis=0)
    cols = np.concatenate([item["colors"] for item in history_clouds], axis=0)
    pts = transform_points(current_world_to_camera, pts_world)
    pts = apply_local_virtual_camera(
        pts,
        args.camera_height_offset,
        args.camera_y_offset,
        args.camera_lookahead,
    )
    render_hw = (args.render_size, args.render_size)
    intrinsic = scale_intrinsic(current_intrinsic, model_hw, render_hw)
    intrinsic[0, 0] *= args.focal_scale
    intrinsic[1, 1] *= args.focal_scale
    intrinsic[1, 2] += args.principal_y_shift * args.render_size
    reconstruction = render_projected_cloud(pts, cols, intrinsic, render_hw, args)

    left_size = args.render_size // 2
    dark_panel = resize_u8(dark, (left_size, left_size))
    event_panel = event_to_rgb(event_img, (left_size, left_size))
    canvas = np.zeros((args.render_size, args.render_size + left_size, 3), dtype=np.uint8)
    canvas[:left_size, :left_size] = dark_panel
    canvas[left_size:, :left_size] = event_panel
    canvas[:, left_size:] = reconstruction
    Image.fromarray(canvas).save(path)


def encode_video(frames_dir, output_path, fps):
    import imageio.v2 as imageio

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
        "<title>StreamVGGT Follow Video</title>",
        "<style>body{font-family:Arial,sans-serif;margin:24px;background:#111;color:#eee}"
        "video,img{max-width:100%;display:block;margin:12px 0 28px;background:#000;border:1px solid #333}"
        "a{color:#9bd}code{color:#ddd}</style>",
        "<h1>StreamVGGT Pose-following Reconstruction</h1>",
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
    input_dir = outdir / ("input_images" if args.use_original_images else "darkened_inputs")
    frames_dir = outdir / "frames"
    ply_dir = outdir / "ply"
    pred_dir = outdir / "predictions"
    for path in (input_dir, frames_dir, ply_dir, pred_dir):
        path.mkdir(parents=True, exist_ok=True)
    for pattern, directory in (
        ("frame_*.png", frames_dir),
        ("streamvggt_*.ply", ply_dir),
        ("chunk_*.npz", pred_dir),
    ):
        for stale in directory.glob(pattern):
            stale.unlink()

    image_paths, event_paths = frame_paths(args.scene_dir, args.start_index, args.num_frames)
    if args.use_original_images:
        dark_paths = save_original_inputs(image_paths, input_dir)
    else:
        dark_paths = save_darkened_inputs(image_paths, input_dir, args.exposure, args.seed)

    StreamVGGTInference = import_streamvggt(args.streamvggt_root)
    infer = StreamVGGTInference(checkpoint_path=args.checkpoint, device="cuda" if torch.cuda.is_available() else "cpu")
    chunk_size = args.chunk_size if args.chunk_size and args.chunk_size > 0 else len(dark_paths)
    chunk_size = min(chunk_size, len(dark_paths))
    chunked = chunk_size < len(dark_paths)
    records = []
    merged_pts = [] if not chunked else None
    merged_cols = [] if not chunked else None
    ply_save_indices = {0, len(dark_paths) // 2, len(dark_paths) - 1}

    for chunk_start in range(0, len(dark_paths), chunk_size):
        chunk_end = min(len(dark_paths), chunk_start + chunk_size)
        predictions, _ = infer.inference([str(path) for path in dark_paths[chunk_start:chunk_end]])
        world_points = predictions["world_points"].detach().float().cpu().numpy()
        world_conf = predictions["world_points_conf"].detach().float().cpu().numpy()
        extrinsic = predictions["extrinsic"].detach().float().cpu().numpy()
        intrinsic = predictions["intrinsic"].detach().float().cpu().numpy()
        model_hw = world_points.shape[1:3]
        np.savez(
            pred_dir / f"chunk_{chunk_start:04d}_{chunk_end - 1:04d}.npz",
            extrinsic=extrinsic,
            intrinsic=intrinsic,
        )

        camera_to_world = np.stack([closed_form_inverse_se3(ext) for ext in extrinsic], axis=0)
        clouds = []
        for local_idx in range(chunk_end - chunk_start):
            idx = chunk_start + local_idx
            image = read_dark_image(dark_paths[idx])
            pts_world, colors = point_cloud_from_prediction(
                world_points[local_idx],
                world_conf[local_idx],
                image,
                max_points=args.points_per_frame,
                conf_percentile=args.conf_percentile,
                seed=args.seed + idx,
            )
            current_world_to_camera = extrinsic[local_idx]
            current_world_to_camera_4x4 = np.eye(4, dtype=np.float32)
            current_world_to_camera_4x4[:3, :4] = current_world_to_camera
            clouds.append({"points_world": pts_world, "colors": colors})
            if not chunked:
                merged_pts.append(pts_world)
                merged_cols.append(colors)

            history = clouds[max(0, len(clouds) - args.history) :]
            render_frame(
                frames_dir / f"frame_{idx:04d}.png",
                idx,
                dark_paths[idx],
                event_paths[idx],
                history,
                current_world_to_camera_4x4,
                intrinsic[local_idx],
                model_hw,
                args,
            )
            if idx in ply_save_indices:
                write_ply(ply_dir / f"streamvggt_frame_{idx:04d}_world_color.ply", pts_world, colors)

            cam_pos = camera_to_world[local_idx][:3, 3]
            records.append(
                {
                    "frame": idx,
                    "source_image": str(image_paths[idx]),
                    "input_image": str(dark_paths[idx]),
                    "event": str(event_paths[idx]),
                    "num_points": int(len(pts_world)),
                    "chunk_start": int(chunk_start),
                    "chunk_end_exclusive": int(chunk_end),
                    "camera_position_chunk_world": [float(v) for v in cam_pos],
                }
            )

        del predictions, world_points, world_conf, extrinsic, intrinsic
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    merged_ply = None
    if not chunked and merged_pts:
        merged_pts = np.concatenate(merged_pts, axis=0)
        merged_cols = np.concatenate(merged_cols, axis=0)
        max_merged = args.points_per_frame * min(args.num_frames, args.history)
        if len(merged_pts) > max_merged:
            rng = np.random.default_rng(args.seed)
            keep = rng.choice(len(merged_pts), size=max_merged, replace=False)
            merged_pts = merged_pts[keep]
            merged_cols = merged_cols[keep]
        merged_ply = "ply/streamvggt_merged_world_color_sampled.ply"
        write_ply(ply_dir / "streamvggt_merged_world_color_sampled.ply", merged_pts, merged_cols)

    video_path = outdir / "streamvggt_follow.mp4"
    encode_video(frames_dir, video_path, args.fps)

    manifest = {
        "scene_dir": args.scene_dir,
        "start_index": args.start_index,
        "num_frames": len(dark_paths),
        "fps": args.fps,
        "duration_s": len(dark_paths) / float(args.fps),
        "input_mode": "original_images" if args.use_original_images else "darkened_images",
        "exposure": None if args.use_original_images else args.exposure,
        "history": args.history,
        "points_per_frame": args.points_per_frame,
        "conf_percentile": args.conf_percentile,
        "camera_height_offset": args.camera_height_offset,
        "camera_y_offset": args.camera_y_offset,
        "camera_lookahead": args.camera_lookahead,
        "focal_scale": args.focal_scale,
        "principal_y_shift": args.principal_y_shift,
        "chunk_size": chunk_size,
        "chunked": chunked,
        "render_note": "Right panel is a camera-view projection, not a third-person 3D orbit. For every video frame, all retained world points from the history window are transformed by the current StreamVGGT extrinsic and projected with the current StreamVGGT intrinsic, so the rendered reconstruction uses the same view as the original captured RGB/event frame. Point colors come from the StreamVGGT input RGB frames.",
        "chunk_note": "When chunked is true, StreamVGGT is run independently per chunk to fit GPU memory. The right panel still uses each frame's own predicted camera view, so the displayed camera-view video remains continuous, but chunk-local world coordinates should not be interpreted as one global trajectory.",
        "outputs": {
            "video": "streamvggt_follow.mp4",
            "frames_dir": "frames",
            "input_images": "input_images" if args.use_original_images else "darkened_inputs",
            "ply_dir": "ply",
            "merged_ply": merged_ply,
            "predictions_dir": "predictions",
        },
        "frames": records,
    }
    (outdir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    write_index(outdir, manifest)
    print(json.dumps({"outdir": str(outdir), "video": str(video_path), "frames": len(dark_paths)}, indent=2))


if __name__ == "__main__":
    main()
