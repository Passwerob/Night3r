#!/usr/bin/env python3
import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from PIL import Image, ImageDraw

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from dataset.dsec_event_dataset import DSECEventDataset  # noqa: E402
from train_event_dark3r import (  # noqa: E402
    build_event_student,
    load_backbone,
    load_config,
    move_batch_to_device,
    safe_torch_load,
)


def parse_args():
    parser = argparse.ArgumentParser(
        "Export a pose-free Night3R pair point cloud in the first-camera frame"
    )
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--model-name-or-path", required=True)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--scene", required=True)
    parser.add_argument("--idx1", type=int, required=True)
    parser.add_argument("--idx2", type=int, required=True)
    parser.add_argument("--outdir", required=True)
    parser.add_argument("--image-size", type=int, nargs=2, default=(512, 512))
    parser.add_argument("--confidence-quantile", type=float, default=0.55)
    parser.add_argument("--max-points", type=int, default=240000)
    parser.add_argument("--width", type=int, default=1200)
    parser.add_argument("--height", type=int, default=800)
    parser.add_argument("--point-radius", type=int, default=1)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="cuda")
    parser.add_argument("--allow-unsafe-checkpoint-load", action="store_true")
    return parser.parse_args()


def resolve_device(name):
    if name == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is unavailable")
        return torch.device("cuda")
    if name == "cpu":
        return torch.device("cpu")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_model(args, cfg, device):
    model_args = SimpleNamespace(
        use_tiny_model=False,
        model_name_or_path=args.model_name_or_path,
        backbone_lora_checkpoint=None,
        allow_unsafe_checkpoint_load=args.allow_unsafe_checkpoint_load,
    )
    backbone = load_backbone(model_args, cfg).to(device)
    model = build_event_student(backbone, config=cfg).to(device)
    payload = safe_torch_load(
        args.checkpoint,
        map_location="cpu",
        allow_unsafe=args.allow_unsafe_checkpoint_load,
        purpose="Night3R pair point-cloud export",
    )
    state_dict = payload.get("model", payload)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    model.eval()
    return model, {
        "step": payload.get("step") if isinstance(payload, dict) else None,
        "missing_keys": list(missing),
        "unexpected_keys": list(unexpected),
    }


def load_pair(args, cfg):
    dataset = DSECEventDataset(
        root_path=args.data_root,
        image_size=args.image_size,
        pair_stride=1,
        expected_event_channels=int(cfg.get("event", {}).get("voxel_channels", 8)),
        pair_sampling="fixed",
        pose_output_convention=cfg.get("dataset", {}).get(
            "pose_output_convention", "native"
        ),
        lowlight_enabled=False,
    )
    for sample_idx, sample in enumerate(dataset.samples):
        if (
            sample["scene"] == args.scene
            and int(sample["idx1"]) == args.idx1
            and int(sample["idx2"]) == args.idx2
        ):
            batch = dataset[sample_idx]
            return {
                key: value.unsqueeze(0) if torch.is_tensor(value) else [value]
                for key, value in batch.items()
            }
    raise RuntimeError(
        f"Pair {args.scene}:{args.idx1:06d}-{args.idx2:06d} was not found"
    )


def image_to_uint8(image):
    array = image.detach().cpu().float().permute(1, 2, 0).numpy()
    return (np.clip(array, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)


def collect_cloud(pred1, pred2, batch, quantile, max_points, seed):
    entries = (
        (pred1["pts3d"], pred1["conf"], batch["images_photo"][0, 0]),
        (
            pred2["pts3d_in_other_view"],
            pred2["conf"],
            batch["images_photo"][0, 1],
        ),
    )
    clouds = []
    diagnostics = []
    for view_idx, (points_tensor, confidence_tensor, image) in enumerate(entries):
        points = points_tensor[0].detach().cpu().float().numpy().reshape(-1, 3)
        confidence = confidence_tensor[0].detach().cpu().float().numpy().reshape(-1)
        colors = image_to_uint8(image).reshape(-1, 3)
        valid = np.isfinite(points).all(axis=1) & np.isfinite(confidence)
        valid &= np.linalg.norm(points, axis=1) > 1.0e-6
        threshold = float(np.quantile(confidence[valid], quantile))
        valid &= confidence >= threshold
        points = points[valid]
        colors = colors[valid]
        confidence = confidence[valid]
        diagnostics.append(
            {
                "view": view_idx + 1,
                "confidence_min": float(confidence.min()),
                "confidence_median": float(np.median(confidence)),
                "confidence_max": float(confidence.max()),
                "points_before_scale_filter": int(len(points)),
            }
        )
        clouds.append((points, colors, confidence, threshold, view_idx))

    reference_norm = np.linalg.norm(clouds[0][0], axis=1)
    norm_low, norm_high = np.quantile(reference_norm, (0.005, 0.995))
    filtered_clouds = []
    for item, view_diagnostics in zip(clouds, diagnostics):
        point_norm = np.linalg.norm(item[0], axis=1)
        keep = (point_norm >= max(float(norm_low) * 0.25, 1.0e-6))
        keep &= point_norm <= float(norm_high) * 1.5
        view_diagnostics["points_after_scale_filter"] = int(keep.sum())
        filtered_clouds.append((item[0][keep], item[1][keep], item[2][keep]))

    points = np.concatenate([item[0] for item in filtered_clouds], axis=0)
    colors = np.concatenate([item[1] for item in filtered_clouds], axis=0)
    confidence = np.concatenate([item[2] for item in filtered_clouds], axis=0)
    center = np.median(points, axis=0)
    radius = np.linalg.norm(points - center, axis=1)
    keep = radius <= np.quantile(radius, 0.99)
    points, colors, confidence = points[keep], colors[keep], confidence[keep]
    if len(points) > max_points:
        rng = np.random.default_rng(seed)
        indices = rng.choice(len(points), size=max_points, replace=False)
        points, colors, confidence = points[indices], colors[indices], confidence[indices]
    return points, colors, confidence, [item[3] for item in clouds], diagnostics


def depth_colors(points):
    depth = points[:, 2]
    lo, hi = np.quantile(depth, (0.02, 0.98))
    value = np.clip((depth - lo) / max(float(hi - lo), 1.0e-6), 0.0, 1.0)
    anchors = np.asarray(
        [
            [36, 64, 142],
            [25, 160, 173],
            [81, 190, 105],
            [246, 205, 67],
            [213, 73, 69],
        ],
        dtype=np.float32,
    )
    scaled = value * (len(anchors) - 1)
    lower = np.floor(scaled).astype(np.int64).clip(0, len(anchors) - 1)
    upper = np.minimum(lower + 1, len(anchors) - 1)
    alpha = (scaled - lower)[:, None]
    return np.clip(anchors[lower] * (1.0 - alpha) + anchors[upper] * alpha, 0, 255).astype(np.uint8)


def write_ply(path, points, colors):
    with open(path, "w", encoding="ascii") as handle:
        handle.write("ply\nformat ascii 1.0\n")
        handle.write(f"element vertex {len(points)}\n")
        handle.write("property float x\nproperty float y\nproperty float z\n")
        handle.write("property uchar red\nproperty uchar green\nproperty uchar blue\n")
        handle.write("end_header\n")
        for point, color in zip(points, colors):
            handle.write(
                f"{point[0]:.6f} {point[1]:.6f} {point[2]:.6f} "
                f"{int(color[0])} {int(color[1])} {int(color[2])}\n"
            )


def render_points(points, colors, camera_position, target, width, height, radius):
    background = np.asarray([8, 12, 18], dtype=np.uint8)
    canvas = np.broadcast_to(background, (height, width, 3)).copy()
    forward = target - camera_position
    forward /= max(float(np.linalg.norm(forward)), 1.0e-6)
    up_hint = np.asarray([0.0, -1.0, 0.0], dtype=np.float32)
    right = np.cross(forward, up_hint)
    right /= max(float(np.linalg.norm(right)), 1.0e-6)
    up = np.cross(right, forward)

    relative = points - camera_position[None]
    x, y, z = relative @ right, relative @ up, relative @ forward
    valid = z > 1.0e-5
    focal = 0.5 * width / np.tan(np.deg2rad(52.0) * 0.5)
    u = focal * x / np.maximum(z, 1.0e-6) + width * 0.5
    v = height * 0.5 - focal * y / np.maximum(z, 1.0e-6)
    valid &= np.isfinite(u) & np.isfinite(v)
    valid &= (u >= -radius) & (u < width + radius)
    valid &= (v >= -radius) & (v < height + radius)

    u = np.rint(u[valid]).astype(np.int32)
    v = np.rint(v[valid]).astype(np.int32)
    z = z[valid].astype(np.float32)
    colors = colors[valid]
    order = np.argsort(z)[::-1]
    u, v, colors = u[order], v[order], colors[order]
    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            px, py = u + dx, v + dy
            inside = (px >= 0) & (px < width) & (py >= 0) & (py < height)
            canvas[py[inside], px[inside]] = colors[inside]
    return canvas


def camera_views(points):
    lo, hi = np.quantile(points, (0.02, 0.98), axis=0)
    center = 0.5 * (lo + hi)
    extent = max(float(np.linalg.norm(hi - lo)), 1.0e-3)
    front = center + np.asarray([0.0, 0.0, -1.25 * extent], dtype=np.float32)
    oblique = center + np.asarray(
        [0.50 * extent, -0.30 * extent, -1.15 * extent], dtype=np.float32
    )
    return center.astype(np.float32), front, oblique


def add_label(image, label):
    result = Image.fromarray(image)
    draw = ImageDraw.Draw(result)
    draw.rounded_rectangle((22, 20, 190, 62), radius=5, fill=(8, 12, 18))
    draw.text((38, 31), label, fill=(245, 247, 250))
    return np.asarray(result)


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    cfg = load_config(args.config)
    batch = load_pair(args, cfg)
    device = resolve_device(args.device)
    model, load_info = load_model(args, cfg, device)
    batch_gpu = move_batch_to_device(batch, device)
    with torch.no_grad(), torch.amp.autocast(
        device_type=device.type,
        dtype=torch.bfloat16,
        enabled=device.type == "cuda",
    ):
        pred1, pred2 = model(batch_gpu)

    points, rgb, confidence, thresholds, cloud_diagnostics = collect_cloud(
        pred1,
        pred2,
        batch,
        args.confidence_quantile,
        args.max_points,
        args.seed,
    )
    colors_depth = depth_colors(points)
    write_ply(outdir / "night3r_pair_rgb.ply", points, rgb)
    write_ply(outdir / "night3r_pair_depth_color.ply", points, colors_depth)

    center, front_camera, oblique_camera = camera_views(points)
    front = render_points(
        points, colors_depth, front_camera, center, args.width, args.height, args.point_radius
    )
    oblique = render_points(
        points, colors_depth, oblique_camera, center, args.width, args.height, args.point_radius
    )
    Image.fromarray(front).save(outdir / "night3r_pointcloud_front.png")
    Image.fromarray(oblique).save(outdir / "night3r_pointcloud_oblique.png")
    panel = np.concatenate((add_label(front, "Front view"), add_label(oblique, "Oblique view")), axis=1)
    Image.fromarray(panel).save(outdir / "night3r_pointcloud_paper_panel.png")

    input1 = Image.fromarray(image_to_uint8(batch["images_photo"][0, 0]))
    input2 = Image.fromarray(image_to_uint8(batch["images_photo"][0, 1]))
    input1.save(outdir / "input_view1.png")
    input2.save(outdir / "input_view2.png")
    manifest = {
        "sample_id": f"{args.scene}:{args.idx1:06d}-{args.idx2:06d}",
        "coordinate_frame": "both point maps expressed in input view 1 camera frame",
        "pose_head_used": False,
        "confidence_quantile": args.confidence_quantile,
        "confidence_thresholds": thresholds,
        "cloud_diagnostics": cloud_diagnostics,
        "points": int(len(points)),
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "model_load": load_info,
        "outputs": {
            "rgb_ply": "night3r_pair_rgb.ply",
            "depth_color_ply": "night3r_pair_depth_color.ply",
            "paper_panel": "night3r_pointcloud_paper_panel.png",
        },
    }
    (outdir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
