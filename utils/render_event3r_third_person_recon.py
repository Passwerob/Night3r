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
from utils.compare_event_dark3r_checkpoint import depth_from_pred, set_seed  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser("Render Event3R multi-frame third-person reconstruction")
    parser.add_argument("--trained-checkpoint", required=True)
    parser.add_argument("--config", default="config/train_event_dark3r_mvsec_dsec.yaml")
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--model-name-or-path", required=True)
    parser.add_argument("--allow-unsafe-checkpoint-load", action="store_true")
    parser.add_argument("--image-size", type=int, nargs=2, default=(512, 512), metavar=("H", "W"))
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--max-pairs", type=int, default=32)
    parser.add_argument("--pair-stride", type=int, default=7)
    parser.add_argument("--fixed-lowlight-exposure", type=float, default=0.03)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--amp", dest="amp", action="store_true", default=None)
    parser.add_argument("--no-amp", dest="amp", action="store_false")
    parser.add_argument("--amp-dtype", choices=("fp16", "bf16"), default=None)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--fps", type=int, default=8)
    parser.add_argument("--num-points-per-frame", type=int, default=14000)
    parser.add_argument("--history", type=int, default=10)
    parser.add_argument("--depth-percentile", type=float, default=99.0)
    parser.add_argument("--view-elev", type=float, default=15.0)
    parser.add_argument("--view-azim", type=float, default=-68.0)
    parser.add_argument("--outdir", required=True)
    return parser.parse_args()


def make_train_args(args):
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
        max_batches=None,
        max_pairs=args.max_pairs,
        start_index=args.start_index,
        num_epochs=None,
        max_steps=None,
        max_scenes=1,
        max_pairs_per_scene=args.start_index + args.max_pairs,
        pair_stride=args.pair_stride,
        pair_sampling="fixed",
        min_pair_distance=None,
        max_pair_distance=None,
        precomputed_pairs=None,
        allow_reverse_pairs=False,
        symmetric_pairs=False,
        num_workers=args.num_workers,
        stage=None,
        partial_finetune_blocks=None,
        amp=args.amp,
        amp_dtype=args.amp_dtype,
        gradient_checkpointing=False,
        device=args.device,
        log_every=1,
        no_save=True,
        seed=args.seed,
        fixed_lowlight_exposure=args.fixed_lowlight_exposure,
    )


def resolve_device(device_arg):
    if device_arg == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("--device cuda requested but CUDA is unavailable")
        return torch.device("cuda")
    if device_arg == "cpu":
        return torch.device("cpu")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def resize_depth(depth, shape):
    if depth.shape[-2:] == shape:
        return depth
    return F.interpolate(depth.float(), size=shape, mode="bilinear", align_corners=False)


def tensor_to_cpu(value):
    if torch.is_tensor(value):
        return value.detach().cpu()
    if isinstance(value, dict):
        return {key: tensor_to_cpu(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return type(value)(tensor_to_cpu(item) for item in value)
    return value


def cache_batches(args, cfg):
    set_seed(args.seed)
    cfg = dict(cfg)
    exposure = float(args.fixed_lowlight_exposure)
    cfg["lowlight_aug"] = {
        **dict(cfg.get("lowlight_aug", {})),
        "enabled": True,
        "min_exposure": exposure,
        "max_exposure": exposure,
    }
    dataset = build_dataset(args, cfg)
    if args.start_index > 0:
        indices = list(range(args.start_index, min(len(dataset), args.start_index + args.max_pairs)))
        dataset_for_loader = torch.utils.data.Subset(dataset, indices)
    else:
        dataset_for_loader = torch.utils.data.Subset(dataset, list(range(min(len(dataset), args.max_pairs))))
    loader = DataLoader(dataset_for_loader, batch_size=1, shuffle=False, num_workers=args.num_workers)
    return dataset, [tensor_to_cpu(batch) for batch in loader]


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
    event_gate = pred1.get("event_gate", None)
    if torch.is_tensor(event_gate):
        event_gate = float(event_gate.detach().cpu().reshape(-1)[0])
    return depth.detach().cpu(), event_gate


def predict_lowlight_rgb(model, batch, args, cfg, device):
    amp_enabled, _, amp_dtype = resolve_amp_config(cfg, args, device)
    batch_gpu = move_batch_to_device(batch, device)
    view1, view2 = make_view_pair_from_batch(batch_gpu, prefer_dark=True)
    with torch.no_grad(), torch.amp.autocast(
        device_type=device.type,
        dtype=amp_dtype,
        enabled=bool(amp_enabled and device.type == "cuda"),
    ):
        pred1, _ = model(view1, view2)
    depth = depth_from_pred(pred1)
    if depth is None:
        raise RuntimeError("Low-light RGB model did not return depth")
    return depth.detach().cpu()


def depth_to_cloud(depth, image, intrinsics, max_points, seed, depth_percentile):
    depth = depth.detach().cpu().float()
    image = image.detach().cpu().float().clamp(0, 1)
    h, w = depth.shape
    yy, xx = torch.meshgrid(torch.arange(h), torch.arange(w), indexing="ij")
    fx, fy = float(intrinsics[0, 0]), float(intrinsics[1, 1])
    cx, cy = float(intrinsics[0, 2]), float(intrinsics[1, 2])
    valid = torch.isfinite(depth) & (depth > 0)
    zvals = depth[valid]
    if zvals.numel() > 100:
        zmax = torch.quantile(zvals, float(depth_percentile) / 100.0)
        valid = valid & (depth <= zmax)
    idx = torch.nonzero(valid.reshape(-1), as_tuple=False).squeeze(1)
    if idx.numel() > max_points:
        g = torch.Generator().manual_seed(int(seed))
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


def set_equal_axes(ax, pts):
    if pts.size == 0:
        return
    mins = np.percentile(pts, 2, axis=0)
    maxs = np.percentile(pts, 98, axis=0)
    center = (mins + maxs) / 2.0
    radius = float(np.max(maxs - mins) / 2.0)
    radius = max(radius, 1e-3)
    ax.set_xlim(center[0] - radius, center[0] + radius)
    ax.set_ylim(center[2] - radius, center[2] + radius)
    ax.set_zlim(center[1] - radius, center[1] + radius)


def sample_id_from_batch(batch, fallback):
    sample_id = batch.get("sample_id", [fallback])
    if isinstance(sample_id, (list, tuple)):
        return str(sample_id[0])
    return str(sample_id)


def render_recon_frame(path, frame_idx, batch, frame_clouds, args):
    dark = batch["images_photo_dark"][0, 0]
    event_activity = batch["event_voxel"][0, 0].abs().sum(dim=0).numpy()
    current_pts, current_colors = frame_clouds[-1]["points"], frame_clouds[-1]["colors"]
    history_pts = np.concatenate([cloud["points"] for cloud in frame_clouds], axis=0)
    history_colors = np.concatenate([cloud["colors"] for cloud in frame_clouds], axis=0)
    sample_id = sample_id_from_batch(batch, f"frame_{frame_idx:04d}")

    fig = plt.figure(figsize=(14, 8), constrained_layout=True)
    grid = fig.add_gridspec(2, 3, width_ratios=[1.0, 1.0, 1.7])
    ax_img = fig.add_subplot(grid[0, 0])
    ax_evt = fig.add_subplot(grid[1, 0])
    ax_cur = fig.add_subplot(grid[:, 1], projection="3d")
    ax_hist = fig.add_subplot(grid[:, 2], projection="3d")

    ax_img.imshow(dark.permute(1, 2, 0).numpy().clip(0, 1))
    ax_img.set_title("Darkened RGB")
    ax_img.axis("off")
    ax_evt.imshow(event_activity, cmap="magma")
    ax_evt.set_title("Event activity")
    ax_evt.axis("off")

    for ax, pts, colors, title, point_size in (
        (ax_cur, current_pts, current_colors, "Current Event3R cloud", 0.18),
        (ax_hist, history_pts, history_colors, f"Following reconstruction, last {len(frame_clouds)} frames", 0.10),
    ):
        stride = max(1, len(pts) // (args.num_points_per_frame * 2))
        pts_show = pts[::stride]
        colors_show = colors[::stride].astype(np.float32) / 255.0
        ax.scatter(pts_show[:, 0], pts_show[:, 2], pts_show[:, 1], c=colors_show, s=point_size, linewidths=0)
        ax.view_init(elev=args.view_elev, azim=args.view_azim)
        ax.set_title(title)
        ax.set_axis_off()
        set_equal_axes(ax, history_pts)

    fig.suptitle(sample_id, fontsize=12)
    fig.savefig(path, dpi=150)
    plt.close(fig)


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
        "<title>Event3R Third-person Reconstruction</title>",
        "<style>body{font-family:Arial,sans-serif;margin:24px;background:#111;color:#eee}"
        "video,img{max-width:100%;display:block;margin:12px 0 28px;background:#000;border:1px solid #333}"
        "a{color:#9bd}code{color:#ddd}</style>",
        "<h1>Event3R Third-person Reconstruction</h1>",
        f"<p><code>{manifest['data_root']}</code></p>",
        "<video controls src='third_person_follow.mp4'></video>",
        "<h2>Outputs</h2><ul>",
    ]
    for key, value in manifest["outputs"].items():
        html.append(f"<li>{key}: <a href='{value}'>{value}</a></li>")
    html.extend(["</ul><h2>Preview frames</h2>"])
    for item in manifest["frames"][:6]:
        html.append(f"<h3>{item['sample_id']}</h3><img src='{Path(item['image']).name}'>")
    html.append("</body>")
    (Path(outdir) / "index.html").write_text("\n".join(html), encoding="utf-8")


def main():
    cli = parse_args()
    outdir = Path(cli.outdir)
    frames_dir = outdir / "frames"
    ply_dir = outdir / "ply"
    frames_dir.mkdir(parents=True, exist_ok=True)
    ply_dir.mkdir(parents=True, exist_ok=True)

    cfg = load_config(cli.config)
    args = make_train_args(cli)
    device = resolve_device(args.device)
    dataset, batches = cache_batches(args, cfg)
    if not batches:
        raise RuntimeError("No batches found for third-person reconstruction")

    model, load_info = build_event_model(args, cfg, device)
    rgb_model = load_backbone(args, cfg).to(device).eval()

    clouds = []
    records = []
    all_points = []
    all_colors = []
    for idx, batch in enumerate(batches):
        event_depth, event_gate = predict_event(model, batch, args, cfg, device)
        low_depth = predict_lowlight_rgb(rgb_model, batch, args, cfg, device)
        target_shape = batch["images_photo_dark"].shape[-2:]
        depth = resize_depth(event_depth.float(), target_shape)[0, 0]
        low_depth = resize_depth(low_depth.float(), target_shape)[0, 0]
        dark = batch["images_photo_dark"][0, 0]
        intrinsics = batch["intrinsics"][0]
        pts, colors = depth_to_cloud(
            depth,
            dark,
            intrinsics,
            max_points=cli.num_points_per_frame,
            seed=cli.seed + idx,
            depth_percentile=cli.depth_percentile,
        )
        cloud = {"points": pts, "colors": colors}
        clouds.append(cloud)
        all_points.append(pts)
        all_colors.append(colors)
        history_clouds = clouds[max(0, len(clouds) - cli.history) :]
        frame_path = frames_dir / f"frame_{idx:04d}.png"
        render_recon_frame(frame_path, idx, batch, history_clouds, cli)

        if idx in {0, len(batches) // 2, len(batches) - 1}:
            write_ply(ply_dir / f"event3r_frame_{idx:04d}.ply", pts, colors)

        records.append(
            {
                "frame": idx,
                "sample_id": sample_id_from_batch(batch, f"frame_{idx:04d}"),
                "idx1": int(batch["idx1"][0]) if torch.is_tensor(batch.get("idx1")) else None,
                "idx2": int(batch["idx2"][0]) if torch.is_tensor(batch.get("idx2")) else None,
                "num_points": int(len(pts)),
                "event_gate": event_gate,
                "event_depth_min": float(depth[torch.isfinite(depth) & (depth > 0)].min().item()),
                "event_depth_median": float(depth[torch.isfinite(depth) & (depth > 0)].median().item()),
                "event_depth_max": float(depth[torch.isfinite(depth) & (depth > 0)].max().item()),
                "lowlight_depth_median": float(low_depth[torch.isfinite(low_depth) & (low_depth > 0)].median().item()),
                "image": str(frame_path),
            }
        )

    merged_points = np.concatenate(all_points, axis=0)
    merged_colors = np.concatenate(all_colors, axis=0)
    if len(merged_points) > cli.num_points_per_frame * cli.history:
        rng = np.random.default_rng(cli.seed)
        keep = rng.choice(len(merged_points), size=cli.num_points_per_frame * cli.history, replace=False)
        merged_points = merged_points[keep]
        merged_colors = merged_colors[keep]
    write_ply(ply_dir / "event3r_follow_merged_sampled.ply", merged_points, merged_colors)

    video_path = outdir / "third_person_follow.mp4"
    encode_video(frames_dir, video_path, cli.fps)

    manifest = {
        "data_root": cli.data_root,
        "config": cli.config,
        "trained_checkpoint": cli.trained_checkpoint,
        "model_name_or_path": cli.model_name_or_path,
        "event_load_info": load_info,
        "fixed_lowlight_exposure": cli.fixed_lowlight_exposure,
        "start_index": cli.start_index,
        "max_pairs": cli.max_pairs,
        "pair_stride": cli.pair_stride,
        "num_points_per_frame": cli.num_points_per_frame,
        "history": cli.history,
        "pose_note": "This DSEC packed scene has no poses_selected.npz; video is a third-person following render in each current camera coordinate frame, not a global pose-registered reconstruction.",
        "outputs": {
            "video": "third_person_follow.mp4",
            "frames_dir": "frames",
            "ply_dir": "ply",
            "merged_sampled_ply": "ply/event3r_follow_merged_sampled.ply",
        },
        "frames": records,
    }
    (outdir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    write_index(outdir, manifest)
    print(json.dumps({"outdir": str(outdir), "video": str(video_path), "frames": len(records)}, indent=2))


if __name__ == "__main__":
    main()
