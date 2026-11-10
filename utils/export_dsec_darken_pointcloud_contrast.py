#!/usr/bin/env python3
import argparse
import json
import math
import os
import sys
from pathlib import Path
from types import SimpleNamespace

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from model.event_dark3r_student import make_view_pair_from_batch  # noqa: E402
from train_event_dark3r import (  # noqa: E402
    build_dataset,
    build_event_student,
    load_backbone,
    load_config,
    move_batch_to_device,
    resolve_amp_config,
    resolve_stage_schedule,
    safe_torch_load,
    set_event_training_stage,
)
from utils.compare_event_dark3r_checkpoint import as_bchw, depth_from_pred, set_seed  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser("Export DSEC darkened RGB vs Event3R point-cloud comparison")
    parser.add_argument("--trained-checkpoint", required=True)
    parser.add_argument("--config", default="config/train_event_dark3r_mvsec_dsec.yaml")
    parser.add_argument("--data-root", default="/data/fcr/data/train_data/DSEC")
    parser.add_argument("--model-name-or-path", required=True)
    parser.add_argument("--allow-unsafe-checkpoint-load", action="store_true")
    parser.add_argument("--scene-index", type=int, default=0)
    parser.add_argument("--idx1", type=int, default=1)
    parser.add_argument("--idx2", type=int, default=7)
    parser.add_argument("--image-size", type=int, nargs=2, default=(512, 512), metavar=("H", "W"))
    parser.add_argument("--fixed-lowlight-exposure", type=float, default=0.03)
    parser.add_argument("--amp", dest="amp", action="store_true", default=None)
    parser.add_argument("--no-amp", dest="amp", action="store_false")
    parser.add_argument("--amp-dtype", choices=("fp16", "bf16"), default=None)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num-points", type=int, default=45000)
    parser.add_argument("--outdir", required=True)
    return parser.parse_args()


def make_train_args(args, precomputed_pairs):
    return SimpleNamespace(
        config=args.config,
        data_root=args.data_root,
        save_dir=args.outdir,
        trained_checkpoint=args.trained_checkpoint,
        model_name_or_path=args.model_name_or_path,
        allow_unsafe_checkpoint_load=args.allow_unsafe_checkpoint_load,
        use_tiny_model=False,
        image_size=args.image_size,
        batch_size=1,
        max_batches=1,
        max_pairs=1,
        num_epochs=None,
        max_steps=None,
        max_scenes=args.scene_index + 1,
        max_pairs_per_scene=None,
        pair_stride=1,
        pair_sampling="precomputed",
        min_pair_distance=1,
        max_pair_distance=max(1, args.idx2 - args.idx1),
        precomputed_pairs=str(precomputed_pairs),
        allow_reverse_pairs=False,
        symmetric_pairs=False,
        num_workers=0,
        stage=None,
        partial_finetune_blocks=None,
        amp=args.amp,
        amp_dtype=args.amp_dtype,
        gradient_checkpointing=False,
        device=args.device,
        log_every=1,
        no_save=True,
        seed=args.seed,
    )


def resolve_device(device_arg):
    if device_arg == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("--device cuda requested but CUDA is unavailable")
        return torch.device("cuda")
    if device_arg == "cpu":
        return torch.device("cpu")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def build_event_model(args, cfg, device):
    backbone = load_backbone(args, cfg)
    model = build_event_student(backbone, config=cfg)
    final_stage = resolve_stage_schedule(cfg, args)[-1]
    set_event_training_stage(
        model,
        stage=final_stage["stage"],
        partial_finetune_blocks=final_stage["partial_finetune_blocks"],
    )
    payload = safe_torch_load(
        args.trained_checkpoint,
        map_location="cpu",
        allow_unsafe=args.allow_unsafe_checkpoint_load,
        purpose="trained Event3R checkpoint",
    )
    state_dict = payload.get("model", payload)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    return model.to(device).eval(), {
        "step": payload.get("step") if isinstance(payload, dict) else None,
        "missing_keys": list(missing),
        "unexpected_keys": list(unexpected),
    }


def resize_depth(depth, shape):
    if depth.shape[-2:] == shape:
        return depth
    return F.interpolate(depth.float(), size=shape, mode="bilinear", align_corners=False)


def predict_rgb(model, batch, args, cfg, device, prefer_dark):
    amp_enabled, _, amp_dtype = resolve_amp_config(cfg, args, device)
    batch_gpu = move_batch_to_device(batch, device)
    view1, view2 = make_view_pair_from_batch(batch_gpu, prefer_dark=prefer_dark)
    with torch.no_grad(), torch.amp.autocast(
        device_type=device.type,
        dtype=amp_dtype,
        enabled=bool(amp_enabled and device.type == "cuda"),
    ):
        pred1, _ = model(view1, view2)
    depth = depth_from_pred(pred1)
    if depth is None:
        raise RuntimeError("RGB model did not return depth")
    return depth.detach().cpu()


def predict_event(model, batch, args, cfg, device):
    amp_enabled, _, amp_dtype = resolve_amp_config(cfg, args, device)
    batch_gpu = move_batch_to_device(batch, device)
    with torch.no_grad(), torch.amp.autocast(
        device_type=device.type,
        dtype=amp_dtype,
        enabled=bool(amp_enabled and device.type == "cuda"),
    ):
        pred1, _ = model(batch_gpu)
    depth = depth_from_pred(pred1)
    if depth is None:
        raise RuntimeError("Event3R model did not return depth")
    return depth.detach().cpu()


def metric(pred_depth, gt_depth, valid):
    pred_depth = resize_depth(pred_depth.float(), gt_depth.shape[-2:])[0, 0]
    gt = gt_depth[0, 0].float()
    mask = valid[0, 0].bool() & torch.isfinite(pred_depth) & torch.isfinite(gt) & (pred_depth > 0) & (gt > 0)
    pred = pred_depth[mask].clamp_min(1e-6)
    gt = gt[mask].clamp_min(1e-6)
    diff = pred - gt
    mse = float((diff * diff).mean())
    ratio = torch.maximum(pred / gt, gt / pred)
    return {
        "absrel": float((diff.abs() / gt).mean()),
        "rmse": math.sqrt(max(mse, 0.0)),
        "delta_1.25": float((ratio < 1.25).float().mean()),
        "valid_px": int(mask.sum()),
    }


def depth_to_points(depth, image, intrinsics, valid, max_points, seed):
    depth = depth.detach().cpu().float()
    image = image.detach().cpu().float().clamp(0, 1)
    valid = valid.detach().cpu().bool()
    h, w = depth.shape
    yy, xx = torch.meshgrid(torch.arange(h), torch.arange(w), indexing="ij")
    fx, fy = float(intrinsics[0, 0]), float(intrinsics[1, 1])
    cx, cy = float(intrinsics[0, 2]), float(intrinsics[1, 2])
    mask = valid & torch.isfinite(depth) & (depth > 0)
    # Clip only extreme outliers for visual stability; do not median-align predictions.
    zvals = depth[mask]
    if zvals.numel() > 100:
        zmax = torch.quantile(zvals, 0.995)
        mask = mask & (depth <= zmax)
    idx = torch.nonzero(mask.reshape(-1), as_tuple=False).squeeze(1)
    if idx.numel() > max_points:
        g = torch.Generator().manual_seed(seed)
        idx = idx[torch.randperm(idx.numel(), generator=g)[:max_points]]
    y = yy.reshape(-1)[idx].float()
    x = xx.reshape(-1)[idx].float()
    z = depth.reshape(-1)[idx]
    pts = torch.stack(((x - cx) * z / fx, -(y - cy) * z / fy, z), dim=1).numpy()
    colors = (image.permute(1, 2, 0).reshape(-1, 3)[idx].numpy() * 255.0 + 0.5).astype(np.uint8)
    return pts.astype(np.float32), colors


def write_ply(path, points, colors):
    with open(path, "w") as f:
        f.write("ply\nformat ascii 1.0\n")
        f.write(f"element vertex {len(points)}\n")
        f.write("property float x\nproperty float y\nproperty float z\n")
        f.write("property uchar red\nproperty uchar green\nproperty uchar blue\n")
        f.write("end_header\n")
        for p, c in zip(points, colors):
            f.write(f"{p[0]:.6f} {p[1]:.6f} {p[2]:.6f} {int(c[0])} {int(c[1])} {int(c[2])}\n")


def color_depth(depth, valid, vmin=None, vmax=None):
    arr = depth.detach().cpu().float().numpy()
    mask = valid.detach().cpu().bool().numpy() & np.isfinite(arr) & (arr > 0)
    vals = arr[mask]
    if vmin is None:
        vmin = float(np.percentile(vals, 2))
    if vmax is None:
        vmax = float(np.percentile(vals, 98))
    norm = np.clip((arr - vmin) / max(vmax - vmin, 1e-6), 0, 1)
    rgb = plt.cm.viridis(norm)[..., :3]
    rgb[~mask] = 0
    return rgb, vmin, vmax


def save_depth_bundle(outdir, name, depth, valid, vmin=None, vmax=None):
    depth = depth.detach().cpu().float()
    valid = valid.detach().cpu().bool()
    rgb, vmin, vmax = color_depth(depth, valid, vmin=vmin, vmax=vmax)
    np.save(outdir / f"{name}.npy", depth.numpy())
    plt.imsave(outdir / f"{name}.png", rgb)
    plt.imsave(outdir / f"{name}_vis.png", rgb)
    meta = {
        "vmin": float(vmin),
        "vmax": float(vmax),
        "shape": list(depth.shape),
    }
    (outdir / f"{name}_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return rgb, vmin, vmax


def save_overview(path, batch, depths, metrics):
    image = batch["images_photo"][0, 0]
    dark = batch["images_photo_dark"][0, 0]
    gt = as_bchw(batch["depth_gt"])[0, 0]
    valid = as_bchw(batch["depth_valid"])[0, 0].bool()
    low = resize_depth(depths["lowlight_rgb"], gt.shape[-2:])[0, 0]
    event = resize_depth(depths["event3r"], gt.shape[-2:])[0, 0]
    vals = gt[valid]
    vmin, vmax = float(torch.quantile(vals, 0.02)), float(torch.quantile(vals, 0.98))
    gt_rgb, _, _ = color_depth(gt, valid, vmin, vmax)
    low_rgb, _, _ = color_depth(low, valid, vmin, vmax)
    event_rgb, _, _ = color_depth(event, valid, vmin, vmax)
    low_err = (low - gt).abs().numpy()
    event_err = (event - gt).abs().numpy()
    err_vmax = float(np.percentile(np.concatenate([low_err[valid.numpy()], event_err[valid.numpy()]]), 95))
    imp = low_err - event_err
    imp_abs = float(np.percentile(np.abs(imp[valid.numpy()]), 98))

    fig, axes = plt.subplots(2, 4, figsize=(18, 9), constrained_layout=True)
    panels = [
        (image.permute(1, 2, 0).numpy(), "RGB", None, None, None),
        (dark.permute(1, 2, 0).numpy(), "Darkened RGB", None, None, None),
        (gt_rgb, "GT depth", None, None, None),
        (low_rgb, f"Low-light RGB depth\nabsrel {metrics['lowlight_rgb']['absrel']:.3f}", None, None, None),
        (event_rgb, f"Event3R depth\nabsrel {metrics['event3r']['absrel']:.3f}", None, None, None),
        (low_err, "Low-light abs error", "inferno", 0, err_vmax),
        (event_err, "Event3R abs error", "inferno", 0, err_vmax),
        (imp, "Error improvement", "coolwarm", -imp_abs, imp_abs),
    ]
    for ax, (data, title, cmap, vmin_i, vmax_i) in zip(axes.reshape(-1), panels):
        if cmap is None:
            ax.imshow(np.clip(data, 0, 1))
        else:
            im = ax.imshow(data, cmap=cmap, vmin=vmin_i, vmax=vmax_i)
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
        ax.set_title(title)
        ax.axis("off")
    fig.savefig(path, dpi=150)
    plt.close(fig)


def equal_axes(ax, pts):
    mins = np.percentile(pts, 1.0, axis=0)
    maxs = np.percentile(pts, 99.0, axis=0)
    center = (mins + maxs) / 2
    radius = max(float(np.max(maxs - mins) / 2), 1.0e-6)
    ax.set_xlim(center[0] - radius, center[0] + radius)
    ax.set_ylim(center[1] - radius, center[1] + radius)
    ax.set_zlim(center[2] - radius, center[2] + radius)


def render_cloud(points, rgb_colors=None, color_mode="depth", width=900, height=620, point_radius=1):
    background = np.asarray([11, 16, 23], dtype=np.uint8)
    canvas = np.broadcast_to(background, (height, width, 3)).copy()
    lo = np.percentile(points, 1, axis=0)
    hi = np.percentile(points, 99, axis=0)
    center = 0.5 * (lo + hi)
    extent = max(float(np.linalg.norm(hi - lo)), 1.0e-6)
    camera = center + np.asarray(
        [0.40 * extent, 0.24 * extent, -1.18 * extent], dtype=np.float32
    )
    forward = center - camera
    forward /= max(float(np.linalg.norm(forward)), 1.0e-6)
    up_hint = np.asarray([0.0, 1.0, 0.0], dtype=np.float32)
    right = np.cross(forward, up_hint)
    right /= max(float(np.linalg.norm(right)), 1.0e-6)
    up = np.cross(right, forward)

    relative = points - camera[None]
    x, y, z = relative @ right, relative @ up, relative @ forward
    focal = 0.5 * width / np.tan(np.deg2rad(48.0) * 0.5)
    u = focal * x / np.maximum(z, 1.0e-6) + 0.5 * width
    v = 0.5 * height - focal * y / np.maximum(z, 1.0e-6)
    valid = (z > 1.0e-5) & np.isfinite(u) & np.isfinite(v)
    valid &= (u >= -point_radius) & (u < width + point_radius)
    valid &= (v >= -point_radius) & (v < height + point_radius)

    u = np.rint(u[valid]).astype(np.int32)
    v = np.rint(v[valid]).astype(np.int32)
    camera_depth = z[valid]
    if color_mode == "rgb":
        if rgb_colors is None:
            raise ValueError("RGB rendering requires point colors")
        colors_float = rgb_colors[valid].astype(np.float32) / 255.0
        luminance = colors_float @ np.asarray([0.2126, 0.7152, 0.0722], dtype=np.float32)
        white = max(float(np.percentile(luminance, 98)), 0.05)
        colors = np.clip(colors_float / white, 0.0, 1.0) ** 0.72
        colors = (colors * 255.0 + 0.5).astype(np.uint8)
    elif color_mode == "depth":
        scene_depth = points[valid, 2]
        depth_lo, depth_hi = np.percentile(scene_depth, [2, 98])
        depth_norm = np.clip(
            (scene_depth - depth_lo) / max(float(depth_hi - depth_lo), 1.0e-6), 0, 1
        )
        colors = (plt.cm.turbo(depth_norm)[:, :3] * 255.0 + 0.5).astype(np.uint8)
    else:
        raise ValueError(f"Unsupported point-cloud color mode: {color_mode}")
    order = np.argsort(camera_depth)[::-1]
    u, v, colors = u[order], v[order], colors[order]
    for dy in range(-point_radius, point_radius + 1):
        for dx in range(-point_radius, point_radius + 1):
            px, py = u + dx, v + dy
            inside = (px >= 0) & (px < width) & (py >= 0) & (py < height)
            canvas[py[inside], px[inside]] = colors[inside]
    return canvas


def save_cloud_view(path, clouds, color_mode="depth"):
    panel_width, panel_height, header = 900, 620, 70
    output = Image.new(
        "RGB", (panel_width * len(clouds), panel_height + header), (11, 16, 23)
    )
    draw = ImageDraw.Draw(output)
    for index, cloud in enumerate(clouds):
        rendered = render_cloud(
            cloud["points"],
            rgb_colors=cloud["colors"],
            color_mode=color_mode,
            width=panel_width,
            height=panel_height,
        )
        output.paste(Image.fromarray(rendered), (index * panel_width, header))
        draw.text(
            (index * panel_width + panel_width // 2, 28),
            cloud["title"],
            fill=(245, 247, 250),
            anchor="mm",
        )
    output.save(path)


def write_index(path, manifest):
    html = [
        "<!doctype html><meta charset='utf-8'>",
        "<title>DSEC darkened point cloud contrast</title>",
        "<style>body{font-family:Arial,sans-serif;margin:24px;background:#111;color:#eee}img{max-width:100%;display:block;margin:12px 0 28px}a{color:#9bd}</style>",
        "<h1>DSEC darkened point cloud contrast</h1>",
        f"<p>{manifest['sample_id']}</p>",
        "<img src='overview_depth_error.png'>",
        "<img src='pointcloud_three_way.png'>",
        "<ul>",
    ]
    for key, file_name in manifest["ply"].items():
        html.append(f"<li><a href='{file_name}'>{key}: {file_name}</a></li>")
    html.append("</ul></body>")
    path.write_text("\n".join(html), encoding="utf-8")


def main():
    args = parse_args()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    precomputed_pairs = (outdir / "selected_pair.json").resolve()
    precomputed_pairs.write_text(json.dumps([{"idx1": args.idx1, "idx2": args.idx2}], indent=2), encoding="utf-8")
    train_args = make_train_args(args, precomputed_pairs)
    set_seed(args.seed)

    cfg = load_config(args.config)
    cfg = dict(cfg)
    cfg["lowlight_aug"] = {
        **dict(cfg.get("lowlight_aug", {})),
        "enabled": True,
        "min_exposure": float(args.fixed_lowlight_exposure),
        "max_exposure": float(args.fixed_lowlight_exposure),
    }
    dataset = build_dataset(train_args, cfg)
    if args.scene_index > 0:
        scene_name = sorted({s["scene"] for s in dataset.samples})[args.scene_index]
        sample_indices = [i for i, s in enumerate(dataset.samples) if s["scene"] == scene_name]
        dataset.samples = [dataset.samples[i] for i in sample_indices]
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)
    batch = next(iter(loader))
    device = resolve_device(args.device)
    rgb_model = load_backbone(train_args, cfg).to(device).eval()
    event_model, load_info = build_event_model(train_args, cfg, device)

    depths = {
        "lowlight_rgb": predict_rgb(rgb_model, batch, train_args, cfg, device, prefer_dark=True),
        "event3r": predict_event(event_model, batch, train_args, cfg, device),
    }
    gt = as_bchw(batch["depth_gt"])
    valid = as_bchw(batch["depth_valid"]).bool()
    metrics = {name: metric(depth, gt, valid) for name, depth in depths.items()}

    target_shape = gt.shape[-2:]
    image = batch["images_photo"][0, 0]
    dark = batch["images_photo_dark"][0, 0]
    intrinsics = batch["intrinsics"][0]
    valid2d = valid[0, 0]
    gt_valid = valid2d & torch.isfinite(gt[0, 0]) & (gt[0, 0] > 0)
    depth_vmin = float(torch.quantile(gt[0, 0][gt_valid], 0.02))
    depth_vmax = float(torch.quantile(gt[0, 0][gt_valid], 0.98))
    save_depth_bundle(outdir, "gt_depth", gt[0, 0], valid2d, vmin=depth_vmin, vmax=depth_vmax)
    save_depth_bundle(
        outdir,
        "lowlight_rgb_depth",
        resize_depth(depths["lowlight_rgb"], target_shape)[0, 0],
        valid2d,
        vmin=depth_vmin,
        vmax=depth_vmax,
    )
    save_depth_bundle(
        outdir,
        "event3r_depth",
        resize_depth(depths["event3r"], target_shape)[0, 0],
        valid2d,
        vmin=depth_vmin,
        vmax=depth_vmax,
    )
    clouds = []
    ply_files = {}
    cloud_specs = [
        ("gt", gt[0, 0], image, "GT point cloud"),
        ("lowlight_rgb", resize_depth(depths["lowlight_rgb"], target_shape)[0, 0], image, "Low-light RGB point cloud"),
        ("event3r", resize_depth(depths["event3r"], target_shape)[0, 0], image, "Night3R point cloud"),
    ]
    for offset, (name, depth, color_image, title) in enumerate(cloud_specs):
        pts, colors = depth_to_points(depth, color_image, intrinsics, valid2d, args.num_points, args.seed + offset)
        ply_name = f"{name}_cloud.ply"
        write_ply(outdir / ply_name, pts, colors)
        ply_files[name] = ply_name
        clouds.append({"name": name, "title": title, "points": pts, "colors": colors})

    save_overview(outdir / "overview_depth_error.png", batch, depths, metrics)
    save_cloud_view(outdir / "pointcloud_three_way_depth.png", clouds, color_mode="depth")
    save_cloud_view(outdir / "pointcloud_three_way_rgb.png", clouds, color_mode="rgb")
    save_cloud_view(outdir / "pointcloud_three_way.png", clouds, color_mode="rgb")
    sample_id = batch["sample_id"][0] if isinstance(batch.get("sample_id"), (list, tuple)) else str(batch.get("sample_id"))
    manifest = {
        "sample_id": sample_id,
        "data_root": args.data_root,
        "fixed_lowlight_exposure": args.fixed_lowlight_exposure,
        "metrics": metrics,
        "event_load_info": load_info,
        "ply": ply_files,
        "pointcloud_renders": {
            "rgb": "pointcloud_three_way_rgb.png",
            "depth": "pointcloud_three_way_depth.png",
            "rgb_note": "Geometry is predicted from low-light inputs; all clouds use the registered clean reference RGB only as visualization texture.",
        },
    }
    (outdir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    write_index(outdir / "index.html", manifest)
    print(json.dumps({"outdir": str(outdir), "sample_id": sample_id, "metrics": metrics}, indent=2))


if __name__ == "__main__":
    main()
