#!/usr/bin/env python3
"""Fuse Night3R c84 point maps with its evaluated pose trajectory for a demo video."""

import argparse
import json
import math
import os
import sys
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

from train_event_dark3r import load_config, move_batch_to_device, resolve_amp_config
from utils.render_event3r_third_person_recon import (
    build_event_model,
    cache_batches,
    make_train_args,
    resolve_device,
)


BACKGROUND = (8, 13, 20)
PANEL = (12, 19, 29)
TEXT = (240, 244, 248)
MUTED = (165, 177, 191)


def parse_args():
    parser = argparse.ArgumentParser(
        "Fuse c84 Night3R point maps using the corresponding evaluated poses."
    )
    parser.add_argument("--trained-checkpoint", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--model-name-or-path", required=True)
    parser.add_argument("--trajectory-npz", type=Path, required=True)
    parser.add_argument("--allow-unsafe-checkpoint-load", action="store_true")
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--max-pairs", type=int, default=146)
    parser.add_argument("--pair-stride", type=int, default=1)
    parser.add_argument("--fixed-lowlight-exposure", type=float, default=None)
    parser.add_argument("--frame-step", type=int, default=2)
    parser.add_argument("--image-size", type=int, nargs=2, default=(512, 512))
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--amp", dest="amp", action="store_true", default=True)
    parser.add_argument("--no-amp", dest="amp", action="store_false")
    parser.add_argument("--amp-dtype", choices=("fp16", "bf16"), default="bf16")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--conf-quantile", type=float, default=0.65)
    parser.add_argument("--max-points-per-frame", type=int, default=9000)
    parser.add_argument("--max-render-points", type=int, default=135000)
    parser.add_argument("--fps", type=int, default=12)
    parser.add_argument("--outdir", type=Path, required=True)
    return parser.parse_args()


def load_font(size, bold=False):
    name = "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf"
    path = Path("/usr/share/fonts/truetype/dejavu") / name
    return ImageFont.truetype(str(path), size) if path.is_file() else ImageFont.load_default()


def umeyama_similarity(source, target):
    """Return s, R, t mapping source row vectors to target row vectors."""
    source_mean = source.mean(axis=0)
    target_mean = target.mean(axis=0)
    source_zero = source - source_mean
    target_zero = target - target_mean
    covariance = target_zero.T @ source_zero / len(source)
    left, singular, right_t = np.linalg.svd(covariance)
    sign = np.eye(3)
    if np.linalg.det(left @ right_t) < 0:
        sign[-1, -1] = -1.0
    rotation = left @ sign @ right_t
    variance = float(np.sum(source_zero * source_zero) / len(source))
    scale = float(np.trace(np.diag(singular) @ sign) / max(variance, 1.0e-12))
    translation = target_mean - scale * (rotation @ source_mean)
    return scale, rotation, translation


def project_points(points, camera, right, up, forward, width, height, focal):
    relative = points - camera[None]
    x = relative @ right
    y = relative @ up
    z = relative @ forward
    u = focal * x / np.maximum(z, 1.0e-6) + 0.5 * width
    v = 0.5 * height - focal * y / np.maximum(z, 1.0e-6)
    return u, v, z


def fixed_render_camera(points):
    lower = np.percentile(points, 1.0, axis=0)
    upper = np.percentile(points, 99.0, axis=0)
    center = (lower + upper) * 0.5
    extent = max(float(np.linalg.norm(upper - lower)), 1.0e-6)
    camera = center + np.asarray([0.55, 0.40, -1.28], dtype=np.float32) * extent
    forward = center - camera
    forward /= max(float(np.linalg.norm(forward)), 1.0e-6)
    up_hint = np.asarray([0.0, 1.0, 0.0], dtype=np.float32)
    right = np.cross(forward, up_hint)
    right /= max(float(np.linalg.norm(right)), 1.0e-6)
    up = np.cross(right, forward)
    return camera, right, up, forward


def render_cloud(points, colors, camera_data, width=1050, height=790):
    canvas = np.broadcast_to(np.asarray(BACKGROUND, dtype=np.uint8), (height, width, 3)).copy()
    camera, right, up, forward = camera_data
    focal = 0.5 * width / math.tan(math.radians(49.0) * 0.5)
    u, v, z = project_points(points, camera, right, up, forward, width, height, focal)
    valid = (z > 1.0e-5) & np.isfinite(u) & np.isfinite(v)
    valid &= (u >= -1) & (u < width + 1) & (v >= -1) & (v < height + 1)
    if not np.any(valid):
        return canvas
    u = np.rint(u[valid]).astype(np.int32)
    v = np.rint(v[valid]).astype(np.int32)
    depth = z[valid]
    rgb = colors[valid].astype(np.float32) / 255.0
    luminance = rgb @ np.asarray([0.2126, 0.7152, 0.0722], dtype=np.float32)
    white = max(float(np.percentile(luminance, 98.0)), 0.04)
    rgb = np.clip(rgb / white, 0.0, 1.0) ** 0.72
    rgb = (rgb * 255.0 + 0.5).astype(np.uint8)
    order = np.argsort(depth)[::-1]
    u, v, rgb = u[order], v[order], rgb[order]
    for delta_y in (-1, 0, 1):
        for delta_x in (-1, 0, 1):
            px = u + delta_x
            py = v + delta_y
            inside = (px >= 0) & (px < width) & (py >= 0) & (py < height)
            canvas[py[inside], px[inside]] = rgb[inside]
    return canvas


def event_thumbnail(event):
    event = event.detach().float().cpu()
    half = event.shape[0] // 2
    pos = event[:half].abs().sum(dim=0)
    neg = event[half:].abs().sum(dim=0)

    def normalize(value):
        nz = value[value > 0]
        if not nz.numel():
            return torch.zeros_like(value)
        return (torch.log1p(value) / torch.quantile(torch.log1p(nz), 0.995).clamp_min(1.0e-6)).clamp(0, 1)

    return Image.fromarray(
        (torch.stack((normalize(pos), torch.zeros_like(pos), normalize(neg)), dim=-1).numpy() * 255.0 + 0.5).astype(np.uint8)
    )


def select_pointmap(prediction, dark_image, pose_raw, similarity, frame_index, args):
    pointmap = prediction["pts3d"][0].detach().float().cpu().numpy()
    confidence = prediction.get("conf")
    if confidence is None:
        confidence = np.ones(pointmap.shape[:2], dtype=np.float32)
    else:
        confidence = confidence[0].detach().float().cpu().numpy().squeeze()
    colors = dark_image.detach().float().cpu().permute(1, 2, 0).numpy()
    valid = np.isfinite(pointmap).all(axis=-1) & np.isfinite(confidence)
    valid &= pointmap[..., 2] > 1.0e-4
    norm = np.linalg.norm(pointmap, axis=-1)
    valid &= norm <= np.quantile(norm[valid], 0.995)
    valid &= confidence >= np.quantile(confidence[valid], args.conf_quantile)
    selected = np.flatnonzero(valid.reshape(-1))
    if len(selected) > args.max_points_per_frame:
        rng = np.random.default_rng(args.seed + frame_index)
        selected = rng.choice(selected, size=args.max_points_per_frame, replace=False)
    local = pointmap.reshape(-1, 3)[selected]
    raw_world = local @ pose_raw[:3, :3].T + pose_raw[:3, 3]
    scale, rotation, translation = similarity
    aligned_world = scale * (raw_world @ rotation.T) + translation
    rgb = np.clip(colors.reshape(-1, 3)[selected] * 255.0 + 0.5, 0, 255).astype(np.uint8)
    return aligned_world.astype(np.float32), rgb


def render_video_frame(output, batch, cloud, camera_data, count, args):
    canvas = Image.new("RGB", (1920, 1080), BACKGROUND)
    draw = ImageDraw.Draw(canvas)
    title_font = load_font(29, bold=True)
    text_font = load_font(19)
    draw.text((42, 32), "Night3R", fill=TEXT, font=title_font)
    draw.text((42, 72), "MVSEC outdoor_night2: trajectory-registered global point cloud", fill=MUTED, font=text_font)
    draw.text((1878, 46), f"{count} integrated frames", fill=MUTED, font=text_font, anchor="ra")

    dark = batch["images_photo_dark"][0, 0].detach().float().cpu()
    image = Image.fromarray((dark.permute(1, 2, 0).numpy().clip(0, 1) * 255.0 + 0.5).astype(np.uint8))
    image.thumbnail((760, 570), Image.Resampling.LANCZOS)
    canvas.paste(image, (42 + (760 - image.width) // 2, 148 + (570 - image.height) // 2))
    draw.rounded_rectangle((42, 124, 802, 742), radius=6, outline=PANEL, width=2)
    draw.text((60, 140), "Low-light RGB input", fill=TEXT, font=text_font)
    event = event_thumbnail(batch["event_voxel"][0, 0])
    event.thumbnail((330, 248), Image.Resampling.LANCZOS)
    ex, ey = 448, 470
    draw.rounded_rectangle((ex - 8, ey - 34, ex + 338, ey + 256), radius=5, fill=PANEL)
    draw.text((ex, ey - 28), "Events (+ red / - blue)", fill=TEXT, font=load_font(16))
    canvas.paste(event, (ex + (330 - event.width) // 2, ey + (248 - event.height) // 2))

    cloud_image = Image.fromarray(render_cloud(*cloud, camera_data))
    canvas.paste(cloud_image, (826, 180))
    draw.rounded_rectangle((826, 124, 1876, 994), radius=6, outline=PANEL, width=2)
    draw.text((850, 140), "Night3R global RGB point cloud", fill=TEXT, font=title_font)
    draw.text((850, 1018), "Each frame is transformed by the evaluated c84 camera trajectory before fusion.", fill=MUTED, font=text_font)
    canvas.save(output)


def main():
    cli = parse_args()
    if not 0.0 < cli.conf_quantile < 1.0:
        raise ValueError("--conf-quantile must be in (0, 1)")
    if cli.frame_step < 1 or cli.fps < 1:
        raise ValueError("--frame-step and --fps must be positive")
    cli.outdir.mkdir(parents=True, exist_ok=True)
    frames_dir = cli.outdir / "frames"
    frames_dir.mkdir(exist_ok=True)

    trajectory = np.load(cli.trajectory_npz)
    raw_poses = np.asarray(trajectory["pred_0_camera_to_world"], dtype=np.float64)
    aligned_poses = np.asarray(trajectory["pred_0_sim3_aligned"], dtype=np.float64)
    similarity = umeyama_similarity(raw_poses[:, :3, 3], aligned_poses[:, :3, 3])

    cfg = dict(load_config(cli.config))
    cfg["lowlight_aug"] = {**dict(cfg.get("lowlight_aug", {})), "enabled": False}
    train_args = make_train_args(cli)
    train_args.pair_stride = 1
    train_args.max_pairs = cli.max_pairs
    train_args.fixed_lowlight_exposure = 1.0
    train_args.disable_lowlight_aug = True
    device = resolve_device(cli.device)
    _, batches = cache_batches(train_args, cfg)
    pair_count = min(len(batches), len(raw_poses) - 1, cli.max_pairs)
    if pair_count < 3:
        raise RuntimeError("The c84 input does not provide enough aligned frame pairs")
    selected = list(range(0, pair_count, cli.frame_step))

    model, load_info = build_event_model(train_args, cfg, device)
    model.eval()
    amp_enabled, _, amp_dtype = resolve_amp_config(cfg, train_args, device)
    clouds = []
    with torch.inference_mode():
        for frame_index in selected:
            batch = batches[frame_index]
            batch_gpu = move_batch_to_device(batch, device)
            with torch.amp.autocast(
                device_type=device.type,
                dtype=amp_dtype,
                enabled=bool(amp_enabled and device.type == "cuda"),
            ):
                prediction, _ = model(batch_gpu)
            points, colors = select_pointmap(
                prediction,
                batch["images_photo_dark"][0, 0],
                raw_poses[frame_index],
                similarity,
                frame_index,
                cli,
            )
            clouds.append((frame_index, batch, points, colors))
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    final_points = np.concatenate([points for _, _, points, _ in clouds], axis=0)
    camera_data = fixed_render_camera(final_points)
    cumulative_points, cumulative_colors = [], []
    rendered = []
    for output_index, (frame_index, batch, points, colors) in enumerate(clouds):
        cumulative_points.append(points)
        cumulative_colors.append(colors)
        merged_points = np.concatenate(cumulative_points, axis=0)
        merged_colors = np.concatenate(cumulative_colors, axis=0)
        if len(merged_points) > cli.max_render_points:
            rng = np.random.default_rng(cli.seed + output_index)
            keep = rng.choice(len(merged_points), size=cli.max_render_points, replace=False)
            visible = (merged_points[keep], merged_colors[keep])
        else:
            visible = (merged_points, merged_colors)
        render_video_frame(
            frames_dir / f"frame_{output_index:03d}.png",
            batch,
            visible,
            camera_data,
            output_index + 1,
            cli,
        )
        rendered.append({"source_frame": frame_index, "points_in_frame": int(len(points))})

    np.savez_compressed(
        cli.outdir / "final_global_cloud.npz",
        points=final_points,
        colors=np.concatenate([colors for _, _, _, colors in clouds], axis=0),
    )
    manifest = {
        "trajectory_npz": str(cli.trajectory_npz),
        "source_pair_count": pair_count,
        "integrated_frames": len(clouds),
        "frame_step": cli.frame_step,
        "fps": cli.fps,
        "point_count": int(len(final_points)),
        "similarity_raw_pose_to_gt_aligned": {
            "scale": float(similarity[0]),
            "rotation": similarity[1].tolist(),
            "translation": similarity[2].tolist(),
        },
        "checkpoint_load": load_info,
        "frames": rendered,
        "claim_scope": "Each point map is inferred from the matching stored low-light RGB/event pair, transformed by the saved Night3R c84 camera pose, then by the same whole-trajectory Sim(3) used for GT-aligned pose evaluation.",
    }
    (cli.outdir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"outdir": str(cli.outdir), **manifest}, indent=2))


if __name__ == "__main__":
    main()
