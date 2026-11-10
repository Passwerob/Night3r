#!/usr/bin/env python3
import argparse
import json
import subprocess
from pathlib import Path

import numpy as np
from PIL import Image


def parse_args():
    parser = argparse.ArgumentParser("Render a stable elevated oblique video from a StreamVGGT PLY")
    parser.add_argument("--ply", required=True)
    parser.add_argument("--outdir", required=True)
    parser.add_argument("--frames", type=int, default=150)
    parser.add_argument("--fps", type=int, default=10)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--fov-deg", type=float, default=50.0)
    parser.add_argument("--point-radius", type=int, default=1)
    parser.add_argument("--max-points", type=int, default=650000)
    parser.add_argument("--seed", type=int, default=1)
    return parser.parse_args()


def read_ply(path):
    with Path(path).open() as f:
        count = None
        for line in f:
            if line.startswith("element vertex"):
                count = int(line.split()[-1])
            if line.strip() == "end_header":
                break
        data = np.loadtxt(f, dtype=np.float32, max_rows=count)
    return data[:, :3], data[:, 3:6].astype(np.uint8)


def streamvggt_to_scene(points):
    return np.stack([points[:, 0], points[:, 2], -points[:, 1]], axis=1).astype(np.float32)


def crop_points(points, colors):
    lo = np.percentile(points, [0.3, 0.4, 0.3], axis=0).diagonal()
    hi = np.percentile(points, [99.5, 99.4, 98.8], axis=0).diagonal()
    mask = np.all((points >= lo) & (points <= hi), axis=1)
    points = points[mask]
    colors = colors[mask]
    center = np.median(points, axis=0)
    radius = np.linalg.norm(points - center, axis=1)
    keep = radius <= np.percentile(radius, 99.0)
    return points[keep], colors[keep]


def smoothstep(t):
    return t * t * (3.0 - 2.0 * t)


def make_path(points, nframes):
    q = np.percentile(points, [2, 8, 25, 45, 55, 75, 92, 98], axis=0)
    x_mid = float(np.median(points[:, 0]))
    y_front = float(q[1, 1])
    y_mid = float(q[4, 1])
    y_back = float(q[6, 1])
    z_low = float(q[2, 2])
    z_mid = float(q[4, 2])
    x_span = max(float(q[-1, 0] - q[0, 0]), 1e-3)
    y_span = max(float(q[-1, 1] - q[0, 1]), 1e-3)
    z_span = max(float(q[-1, 2] - q[0, 2]), 1e-3)

    base_target = np.array([x_mid, 0.63 * y_mid + 0.37 * y_back, z_low + 0.18 * z_span], dtype=np.float32)
    dist = 1.82 * y_span
    height = 0.30 * y_span + 0.18 * z_span
    side_mag = 0.50 * x_span

    positions = []
    targets = []
    for idx in range(nframes):
        s = smoothstep(idx / max(nframes - 1, 1))
        yaw = np.deg2rad(-20.0 + 18.0 * s)
        forward = np.array([np.sin(yaw), np.cos(yaw), 0.0], dtype=np.float32)
        side = np.array([np.cos(yaw), -np.sin(yaw), 0.0], dtype=np.float32)
        target = base_target.copy()
        target[1] += (s - 0.5) * 0.10 * y_span
        target[2] += np.sin(np.pi * s) * 0.015 * y_span
        pos = target - forward * (dist * (1.02 - 0.06 * s))
        pos += side * (side_mag * (0.54 - 0.12 * s))
        pos += np.array([0.0, 0.0, height * (1.0 + 0.015 * np.sin(np.pi * s))], dtype=np.float32)
        positions.append(pos)
        targets.append(target)
    return np.stack(positions), np.stack(targets), {
        "x_span": x_span,
        "y_span": y_span,
        "z_span": z_span,
        "base_target": base_target.tolist(),
        "y_front": y_front,
    }


def render(points, colors, cam_pos, target, width, height, fov_deg, point_radius):
    canvas = np.zeros((height, width, 3), dtype=np.uint8)
    up_hint = np.array([0.0, 0.0, 1.0], dtype=np.float32)
    forward = target - cam_pos
    forward = forward / max(np.linalg.norm(forward), 1e-6)
    right = np.cross(forward, up_hint)
    right = right / max(np.linalg.norm(right), 1e-6)
    up = np.cross(right, forward)

    rel = points - cam_pos[None, :]
    x = rel @ right
    y = rel @ up
    z = rel @ forward
    valid = z > 1e-4
    focal = 0.5 * width / np.tan(np.deg2rad(fov_deg) * 0.5)
    u = focal * (x / np.maximum(z, 1e-6)) + width * 0.5
    v = height * 0.5 - focal * (y / np.maximum(z, 1e-6))
    valid &= np.isfinite(u) & np.isfinite(v)
    valid &= (u >= -point_radius) & (u < width + point_radius)
    valid &= (v >= -point_radius) & (v < height + point_radius)
    if not valid.any():
        return canvas

    u = np.rint(u[valid]).astype(np.int32)
    v = np.rint(v[valid]).astype(np.int32)
    z = z[valid].astype(np.float32)
    cols = colors[valid].astype(np.float32)

    near, far = np.percentile(z, [3, 98])
    depth_t = np.clip((z - near) / max(far - near, 1e-6), 0.0, 1.0)
    shade = (1.05 - 0.38 * depth_t)[:, None]
    cols = np.clip(cols * shade + 4.0 * (1.0 - shade), 0, 255).astype(np.uint8)

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
            d = z[inside]
            np.minimum.at(zbuf, lin, d)
            keep = d <= zbuf[lin] + 1e-6
            flat[lin[keep]] = cols[inside][keep]
    return canvas


def encode(frames_dir, output, fps):
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
            str(output),
        ],
        check=True,
    )


def write_index(outdir, manifest):
    html = [
        "<!doctype html><meta charset='utf-8'>",
        "<title>StreamVGGT Oblique Teaser V2</title>",
        "<style>body{font-family:Arial,sans-serif;margin:24px;background:#111;color:#eee}"
        "video,img{max-width:100%;display:block;margin:12px 0 28px;background:#000;border:1px solid #333}"
        "a{color:#9bd}</style>",
        "<h1>StreamVGGT Elevated Oblique Teaser V2</h1>",
        "<video controls src='streamvggt_oblique_teaser_v2.mp4'></video>",
        "<ul>",
    ]
    for key, value in manifest["outputs"].items():
        html.append(f"<li>{key}: <a href='{value}'>{value}</a></li>")
    html.extend(["</ul><h2>Preview frames</h2>"])
    for idx in [0, manifest["frames"] // 2, manifest["frames"] - 1]:
        html.append(f"<h3>frame {idx}</h3><img src='frames/frame_{idx:04d}.png'>")
    html.append("</body>")
    (Path(outdir) / "index.html").write_text("\n".join(html), encoding="utf-8")


def main():
    args = parse_args()
    outdir = Path(args.outdir)
    frames_dir = outdir / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)
    for stale in frames_dir.glob("frame_*.png"):
        stale.unlink()

    points_raw, colors = read_ply(args.ply)
    points = streamvggt_to_scene(points_raw)
    points, colors = crop_points(points, colors)
    if len(points) > args.max_points:
        rng = np.random.default_rng(args.seed)
        keep = rng.choice(len(points), size=args.max_points, replace=False)
        points = points[keep]
        colors = colors[keep]

    cam_positions, targets, path_info = make_path(points, args.frames)
    for idx in range(args.frames):
        frame = render(
            points,
            colors,
            cam_positions[idx],
            targets[idx],
            args.width,
            args.height,
            args.fov_deg,
            args.point_radius,
        )
        Image.fromarray(frame).save(frames_dir / f"frame_{idx:04d}.png")

    video = outdir / "streamvggt_oblique_teaser_v2.mp4"
    encode(frames_dir, video, args.fps)
    manifest = {
        "source_ply": str(args.ply),
        "frames": args.frames,
        "fps": args.fps,
        "duration_s": args.frames / float(args.fps),
        "resolution": [args.width, args.height],
        "render_points": int(len(points)),
        "coordinate_note": "StreamVGGT points are rendered as scene coordinates [x, z, -y], so z/depth becomes forward and -y becomes height.",
        "camera_style": "front-above, slightly side-biased, mild top-down three-quarter perspective with smooth forward/orbit motion.",
        "path_info": path_info,
        "outputs": {
            "video": "streamvggt_oblique_teaser_v2.mp4",
            "frames_dir": "frames",
        },
    }
    (outdir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    write_index(outdir, manifest)
    print(json.dumps({"outdir": str(outdir), "video": str(video), "points": int(len(points))}, indent=2))


if __name__ == "__main__":
    main()
