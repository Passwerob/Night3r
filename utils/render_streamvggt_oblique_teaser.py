#!/usr/bin/env python3
import argparse
import json
import os
import sys
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
import torch
from PIL import Image


def parse_args():
    parser = argparse.ArgumentParser("Render an elevated oblique StreamVGGT point-cloud teaser")
    parser.add_argument("--streamvggt-root", default="/data/fcr/code/fcr/StreamVGGT")
    parser.add_argument("--checkpoint", default="/data/fcr/code/fcr/StreamVGGT/ckpts/model.pt")
    parser.add_argument("--scene-dir", default="/data/fcr/data/train_data/DSEC/screen-000001-dsec")
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--num-scene-frames", type=int, default=100)
    parser.add_argument("--render-frames", type=int, default=150)
    parser.add_argument("--fps", type=int, default=10)
    parser.add_argument("--points-per-frame", type=int, default=7000)
    parser.add_argument("--conf-percentile", type=float, default=35.0)
    parser.add_argument("--max-render-points", type=int, default=650000)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--fov-deg", type=float, default=55.0)
    parser.add_argument("--point-radius", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
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
    end = min(len(images), start + count)
    if end <= start:
        raise RuntimeError(f"No images found in {scene_dir} from start {start}")
    return images[start:end]


def save_input_images(image_paths, outdir):
    outdir.mkdir(parents=True, exist_ok=True)
    copied = []
    for idx, path in enumerate(image_paths):
        out_path = outdir / f"{idx:06d}.png"
        Image.open(path).convert("RGB").save(out_path)
        copied.append(out_path)
    return copied


def read_image_01(path, size_hw):
    image = Image.open(path).convert("RGB").resize((size_hw[1], size_hw[0]), Image.BILINEAR)
    return np.asarray(image, dtype=np.float32) / 255.0


def closed_form_inverse_se3(se3):
    mat = np.eye(4, dtype=np.float32)
    mat[:3, :4] = se3.astype(np.float32)
    out = np.eye(4, dtype=np.float32)
    r = mat[:3, :3]
    t = mat[:3, 3]
    out[:3, :3] = r.T
    out[:3, 3] = -r.T @ t
    return out


def sample_points(points, conf, image, max_points, conf_percentile, seed):
    pts = points.reshape(-1, 3).astype(np.float32)
    cols = (image.reshape(-1, 3) * 255.0 + 0.5).astype(np.uint8)
    conf = conf.reshape(-1)
    finite = np.isfinite(pts).all(axis=1) & np.isfinite(conf) & (np.linalg.norm(pts, axis=1) > 1e-6)
    if finite.any():
        thr = float(np.percentile(conf[finite], conf_percentile))
        finite = finite & (conf >= thr)
    pts = pts[finite]
    cols = cols[finite]
    if len(pts) > max_points:
        rng = np.random.default_rng(seed)
        keep = rng.choice(len(pts), size=max_points, replace=False)
        pts = pts[keep]
        cols = cols[keep]
    return pts, cols


def write_ply(path, points, colors):
    with open(path, "w") as f:
        f.write("ply\nformat ascii 1.0\n")
        f.write(f"element vertex {len(points)}\n")
        f.write("property float x\nproperty float y\nproperty float z\n")
        f.write("property uchar red\nproperty uchar green\nproperty uchar blue\n")
        f.write("end_header\n")
        for p, c in zip(points, colors):
            f.write(f"{p[0]:.6f} {p[1]:.6f} {p[2]:.6f} {int(c[0])} {int(c[1])} {int(c[2])}\n")


def robust_filter(points, colors):
    lo = np.percentile(points, 0.5, axis=0)
    hi = np.percentile(points, 99.5, axis=0)
    mask = np.all((points >= lo) & (points <= hi), axis=1)
    points = points[mask]
    colors = colors[mask]
    center = np.median(points, axis=0)
    radius = np.linalg.norm(points - center, axis=1)
    keep = radius <= np.percentile(radius, 98.5)
    return points[keep], colors[keep]


def camera_basis_from_extrinsics(extrinsic):
    c2w = np.stack([closed_form_inverse_se3(ext) for ext in extrinsic], axis=0)
    positions = c2w[:, :3, 3]
    right = c2w[:, :3, 0].mean(axis=0)
    up = (-c2w[:, :3, 1]).mean(axis=0)
    forward = c2w[:, :3, 2].mean(axis=0)
    traj = positions[-1] - positions[0]
    if np.linalg.norm(traj) > 1e-5:
        forward = 0.65 * forward + 0.35 * traj
    right = right / max(np.linalg.norm(right), 1e-6)
    up = up - right * np.dot(up, right)
    up = up / max(np.linalg.norm(up), 1e-6)
    forward = forward - right * np.dot(forward, right) - up * np.dot(forward, up)
    forward = forward / max(np.linalg.norm(forward), 1e-6)
    return positions, right, up, forward


def make_camera_path(points, extrinsic, nframes):
    cam_pos, right, up, forward = camera_basis_from_extrinsics(extrinsic)
    lo = np.percentile(points, 2, axis=0)
    hi = np.percentile(points, 98, axis=0)
    center = (lo + hi) * 0.5
    ext = hi - lo
    radius = max(float(np.linalg.norm(ext) * 0.55), 1e-3)
    traj_center = np.median(cam_pos, axis=0)
    look_center = 0.72 * center + 0.28 * traj_center

    base_dist = radius * 2.25
    base_height = radius * 0.62
    base_side = radius * 0.42
    positions = []
    targets = []
    for i in range(nframes):
        t = i / max(nframes - 1, 1)
        s = t * t * (3.0 - 2.0 * t)
        angle = np.deg2rad(-8.0 + 18.0 * s)
        side = base_side * (0.82 + 0.28 * np.sin(np.pi * s))
        dist = base_dist * (1.04 - 0.14 * s)
        height = base_height * (1.0 + 0.05 * np.sin(2.0 * np.pi * s))

        view_forward = np.cos(angle) * forward + np.sin(angle) * right
        view_forward = view_forward / max(np.linalg.norm(view_forward), 1e-6)
        side_dir = np.cos(angle) * right - np.sin(angle) * forward
        side_dir = side_dir / max(np.linalg.norm(side_dir), 1e-6)
        target = look_center + forward * (radius * 0.10 * (s - 0.5)) + up * (radius * 0.02)
        pos = target - view_forward * dist + up * height + side_dir * side
        positions.append(pos.astype(np.float32))
        targets.append(target.astype(np.float32))
    return np.stack(positions), np.stack(targets), up.astype(np.float32), radius, look_center


def render_points(points, colors, cam_pos, target, up_hint, width, height, fov_deg, point_radius):
    canvas = np.zeros((height, width, 3), dtype=np.uint8)
    forward = target - cam_pos
    forward = forward / max(np.linalg.norm(forward), 1e-6)
    right = np.cross(forward, up_hint)
    right = right / max(np.linalg.norm(right), 1e-6)
    up = np.cross(right, forward)
    rel = points - cam_pos[None, :]
    x = rel @ right
    y = rel @ up
    z = rel @ forward
    valid = z > 1e-5
    if not valid.any():
        return canvas

    focal = 0.5 * width / np.tan(np.deg2rad(fov_deg) * 0.5)
    u = focal * (x / z) + width * 0.5
    v = height * 0.5 - focal * (y / z)
    valid = valid & np.isfinite(u) & np.isfinite(v)
    valid = valid & (u >= -point_radius) & (u < width + point_radius)
    valid = valid & (v >= -point_radius) & (v < height + point_radius)
    if not valid.any():
        return canvas

    u = np.rint(u[valid]).astype(np.int32)
    v = np.rint(v[valid]).astype(np.int32)
    depth = z[valid].astype(np.float32)
    cols = colors[valid].astype(np.float32)

    d_lo, d_hi = np.percentile(depth, [3, 97])
    depth_t = np.clip((depth - d_lo) / max(d_hi - d_lo, 1e-6), 0, 1)
    fog = (1.0 - 0.36 * depth_t)[:, None]
    cols = np.clip(cols * fog + 8.0 * (1.0 - fog), 0, 255).astype(np.uint8)

    flat = canvas.reshape(-1, 3)
    zbuf = np.full(height * width, np.inf, dtype=np.float32)
    for dy in range(-point_radius, point_radius + 1):
        for dx in range(-point_radius, point_radius + 1):
            px = u + dx
            py = v + dy
            inside = (px >= 0) & (px < width) & (py >= 0) & (py < height)
            if not inside.any():
                continue
            lin = py[inside] * width + px[inside]
            d = depth[inside]
            np.minimum.at(zbuf, lin, d)
            keep = d <= zbuf[lin] + 1e-6
            flat[lin[keep]] = cols[inside][keep]
    return canvas


def encode_video(frames_dir, output_path, fps):
    import subprocess

    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-framerate",
            str(fps),
            "-i",
            str(Path(frames_dir) / "frame_%04d.png"),
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
        "<title>StreamVGGT Oblique Teaser</title>",
        "<style>body{font-family:Arial,sans-serif;margin:24px;background:#111;color:#eee}"
        "video,img{max-width:100%;display:block;margin:12px 0 28px;background:#000;border:1px solid #333}"
        "a{color:#9bd}code{color:#ddd}</style>",
        "<h1>StreamVGGT Elevated Oblique Teaser</h1>",
        "<video controls src='streamvggt_oblique_teaser.mp4'></video>",
        "<ul>",
    ]
    for key, value in manifest["outputs"].items():
        html.append(f"<li>{key}: <a href='{value}'>{value}</a></li>")
    html.extend(["</ul><h2>Preview frames</h2>"])
    for idx in [0, len(manifest["render_frames"]) // 2, len(manifest["render_frames"]) - 1]:
        frame = manifest["render_frames"][idx]
        html.append(f"<h3>frame {frame}</h3><img src='frames/frame_{frame:04d}.png'>")
    html.append("</body>")
    (Path(outdir) / "index.html").write_text("\n".join(html), encoding="utf-8")


def main():
    args = parse_args()
    outdir = Path(args.outdir)
    input_dir = outdir / "input_images"
    frames_dir = outdir / "frames"
    ply_dir = outdir / "ply"
    for directory in (input_dir, frames_dir, ply_dir):
        directory.mkdir(parents=True, exist_ok=True)
    for stale in frames_dir.glob("frame_*.png"):
        stale.unlink()

    image_paths = frame_paths(args.scene_dir, args.start_index, args.num_scene_frames)
    input_paths = save_input_images(image_paths, input_dir)
    StreamVGGTInference = import_streamvggt(args.streamvggt_root)
    infer = StreamVGGTInference(checkpoint_path=args.checkpoint, device="cuda" if torch.cuda.is_available() else "cpu")
    predictions, _ = infer.inference([str(p) for p in input_paths])

    world_points = predictions["world_points"].detach().float().cpu().numpy()
    world_conf = predictions["world_points_conf"].detach().float().cpu().numpy()
    extrinsic = predictions["extrinsic"].detach().float().cpu().numpy()
    model_hw = world_points.shape[1:3]

    all_points = []
    all_colors = []
    for idx, path in enumerate(input_paths):
        image = read_image_01(path, model_hw)
        pts, cols = sample_points(
            world_points[idx],
            world_conf[idx],
            image,
            args.points_per_frame,
            args.conf_percentile,
            args.seed + idx,
        )
        all_points.append(pts)
        all_colors.append(cols)

    points = np.concatenate(all_points, axis=0)
    colors = np.concatenate(all_colors, axis=0)
    points, colors = robust_filter(points, colors)
    if len(points) > args.max_render_points:
        rng = np.random.default_rng(args.seed)
        keep = rng.choice(len(points), size=args.max_render_points, replace=False)
        points = points[keep]
        colors = colors[keep]

    write_ply(ply_dir / "streamvggt_merged_original_rgb_100f_sampled.ply", points, colors)
    cam_positions, targets, up_hint, scene_radius, look_center = make_camera_path(points, extrinsic, args.render_frames)
    for idx in range(args.render_frames):
        frame = render_points(
            points,
            colors,
            cam_positions[idx],
            targets[idx],
            up_hint,
            args.width,
            args.height,
            args.fov_deg,
            args.point_radius,
        )
        Image.fromarray(frame).save(frames_dir / f"frame_{idx:04d}.png")

    video_path = outdir / "streamvggt_oblique_teaser.mp4"
    encode_video(frames_dir, video_path, args.fps)
    manifest = {
        "scene_dir": str(args.scene_dir),
        "input_mode": "original_images",
        "num_scene_frames": len(input_paths),
        "render_frames": list(range(args.render_frames)),
        "fps": args.fps,
        "duration_s": args.render_frames / float(args.fps),
        "resolution": [args.width, args.height],
        "points_per_frame": args.points_per_frame,
        "conf_percentile": args.conf_percentile,
        "render_points": int(len(points)),
        "scene_radius": float(scene_radius),
        "look_center": [float(v) for v in look_center],
        "camera_style": "slightly elevated oblique, front-above, mild top-down, three-quarter perspective, smooth slow orbit/forward motion",
        "outputs": {
            "video": "streamvggt_oblique_teaser.mp4",
            "frames_dir": "frames",
            "input_images": "input_images",
            "merged_ply": "ply/streamvggt_merged_original_rgb_100f_sampled.ply",
        },
    }
    (outdir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    write_index(outdir, manifest)
    print(json.dumps({"outdir": str(outdir), "video": str(video_path), "render_points": int(len(points))}, indent=2))


if __name__ == "__main__":
    main()
