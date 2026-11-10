#!/usr/bin/env python3
import argparse
import csv
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
from utils.compare_event_dark3r_checkpoint import (  # noqa: E402
    as_bchw,
    depth_from_pred,
    set_seed,
    warp_second_to_first,
)


def parse_args():
    parser = argparse.ArgumentParser("Render Event3R sequence video and dense metrics")
    parser.add_argument("--trained-checkpoint", required=True)
    parser.add_argument("--config", default="config/train_event_dark3r_mvsec_dsec.yaml")
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--model-name-or-path", required=True)
    parser.add_argument("--allow-unsafe-checkpoint-load", action="store_true")
    parser.add_argument("--image-size", type=int, nargs=2, default=(512, 512), metavar=("H", "W"))
    parser.add_argument("--max-pairs", type=int, default=24)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--pair-stride", type=int, default=7)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--amp", dest="amp", action="store_true", default=None)
    parser.add_argument("--no-amp", dest="amp", action="store_false")
    parser.add_argument("--amp-dtype", choices=("fp16", "bf16"), default=None)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--fps", type=int, default=6)
    parser.add_argument(
        "--disable-lowlight-aug",
        action="store_true",
        help="Use original RGB as images_dark. Useful when the scene is already low-light.",
    )
    parser.add_argument(
        "--fixed-lowlight-exposure",
        type=float,
        default=None,
        help="Lock low-light augmentation to one exposure scale by setting min/max exposure.",
    )
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
        batch_size=args.batch_size,
        max_pairs=args.max_pairs,
        start_index=args.start_index,
        num_epochs=None,
        max_steps=None,
        max_scenes=None,
        max_pairs_per_scene=None,
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
        disable_lowlight_aug=args.disable_lowlight_aug,
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
    if getattr(args, "disable_lowlight_aug", False):
        cfg = dict(cfg)
        cfg["lowlight_aug"] = {**dict(cfg.get("lowlight_aug", {})), "enabled": False}
    elif getattr(args, "fixed_lowlight_exposure", None) is not None:
        exposure = float(args.fixed_lowlight_exposure)
        if exposure <= 0:
            raise ValueError("--fixed-lowlight-exposure must be positive")
        cfg = dict(cfg)
        lowlight_cfg = {**dict(cfg.get("lowlight_aug", {}))}
        lowlight_cfg.update({"enabled": True, "min_exposure": exposure, "max_exposure": exposure})
        cfg["lowlight_aug"] = lowlight_cfg
    dataset = build_dataset(args, cfg)
    if args.start_index > 0:
        indices = list(range(args.start_index, len(dataset)))
        dataset_for_loader = torch.utils.data.Subset(dataset, indices)
    else:
        dataset_for_loader = dataset
    loader = DataLoader(
        dataset_for_loader,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=False,
    )
    batches = []
    for batch in loader:
        batches.append(tensor_to_cpu(batch))
        if len(batches) >= args.max_pairs:
            break
    return dataset, batches


def resize_like(x, shape, mode="bilinear"):
    if x is None or x.shape[-2:] == shape:
        return x
    kwargs = {"size": shape, "mode": mode}
    if mode in {"bilinear", "bicubic"}:
        kwargs["align_corners"] = False
    return F.interpolate(x.float(), **kwargs)


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
    model = model.to(device).eval()
    return model, {
        "step": payload.get("step") if isinstance(payload, dict) else None,
        "missing_keys": list(missing),
        "unexpected_keys": list(unexpected),
    }


def predict_rgb(model, batches, args, cfg, device, prefer_dark):
    amp_enabled, _, amp_dtype = resolve_amp_config(cfg, args, device)
    outputs = []
    with torch.no_grad():
        for batch in batches:
            batch_gpu = move_batch_to_device(batch, device)
            view1, view2 = make_view_pair_from_batch(batch_gpu, prefer_dark=prefer_dark)
            with torch.amp.autocast(
                device_type=device.type,
                dtype=amp_dtype,
                enabled=bool(amp_enabled and device.type == "cuda"),
            ):
                pred1, _ = model(view1, view2)
            depth = depth_from_pred(pred1)
            warped, warp_valid = compute_warp(depth, batch_gpu)
            outputs.append(
                {
                    "depth": None if depth is None else depth.detach().cpu(),
                    "warped": None if warped is None else warped.detach().cpu(),
                    "warp_valid": None if warp_valid is None else warp_valid.detach().cpu(),
                }
            )
    return outputs


def predict_event(model, batches, args, cfg, device):
    amp_enabled, _, amp_dtype = resolve_amp_config(cfg, args, device)
    outputs = []
    with torch.no_grad():
        for batch in batches:
            batch_gpu = move_batch_to_device(batch, device)
            with torch.amp.autocast(
                device_type=device.type,
                dtype=amp_dtype,
                enabled=bool(amp_enabled and device.type == "cuda"),
            ):
                pred1, _ = model(batch_gpu)
            depth = depth_from_pred(pred1)
            warped, warp_valid = compute_warp(depth, batch_gpu)
            outputs.append(
                {
                    "depth": None if depth is None else depth.detach().cpu(),
                    "warped": None if warped is None else warped.detach().cpu(),
                    "warp_valid": None if warp_valid is None else warp_valid.detach().cpu(),
                    "event_gate": pred1.get("event_gate", None),
                }
            )
    return outputs


def compute_warp(depth, batch_gpu):
    if depth is None:
        return None, None
    if "images_photo" not in batch_gpu or "intrinsics" not in batch_gpu or "relative_pose" not in batch_gpu:
        return None, None
    image2 = batch_gpu["images_photo"][:, 1].to(dtype=torch.float32)
    depth = resize_like(depth.float(), image2.shape[-2:])
    return warp_second_to_first(
        image2,
        depth,
        batch_gpu["intrinsics"],
        batch_gpu["relative_pose"],
    )


class DepthStats:
    def __init__(self):
        self.n = 0
        self.gt_n = 0
        self.absrel = 0.0
        self.mae = 0.0
        self.sq = 0.0
        self.log = 0.0
        self.log_sq = 0.0
        self.d1 = 0
        self.d2 = 0
        self.d3 = 0
        self.max_gt = 0.0

    def add(self, pred_depth, gt_depth, valid):
        gt_mask = valid.bool() & torch.isfinite(gt_depth) & (gt_depth > 0)
        mask = gt_mask & torch.isfinite(pred_depth) & (pred_depth > 0)
        self.gt_n += int(gt_mask.sum().item())
        if not mask.any():
            return
        pred = pred_depth[mask].float().clamp_min(1e-6)
        gt = gt_depth[mask].float().clamp_min(1e-6)
        diff = pred - gt
        abs_diff = diff.abs()
        log_diff = torch.log(pred) - torch.log(gt)
        ratio = torch.maximum(pred / gt, gt / pred)
        n = int(gt.numel())
        self.n += n
        self.absrel += float((abs_diff / gt).sum().item())
        self.mae += float(abs_diff.sum().item())
        self.sq += float((diff * diff).sum().item())
        self.log += float(log_diff.sum().item())
        self.log_sq += float((log_diff * log_diff).sum().item())
        self.d1 += int((ratio < 1.25).sum().item())
        self.d2 += int((ratio < 1.25**2).sum().item())
        self.d3 += int((ratio < 1.25**3).sum().item())
        self.max_gt = max(self.max_gt, float(gt.max().item()))

    def finalize(self):
        out = {"valid_px": self.n, "coverage": self.n / max(self.gt_n, 1)}
        if self.n <= 0:
            return out
        mse = self.sq / self.n
        mean_log = self.log / self.n
        mean_log_sq = self.log_sq / self.n
        out.update(
            {
                "absrel": self.absrel / self.n,
                "mae": self.mae / self.n,
                "rmse": math.sqrt(max(mse, 0.0)),
                "rmse_log": math.sqrt(max(mean_log_sq, 0.0)),
                "silog": 10.0
                * math.sqrt(max(mean_log_sq - mean_log * mean_log + 0.15 * mean_log * mean_log, 1e-12)),
                "delta_1.25": self.d1 / self.n,
                "delta_1.25^2": self.d2 / self.n,
                "delta_1.25^3": self.d3 / self.n,
            }
        )
        if self.max_gt > 0 and mse > 0:
            out["psnr_db"] = 20.0 * math.log10(self.max_gt) - 10.0 * math.log10(mse)
        return out


def depth_metrics(pred_depth, gt_depth, valid):
    pred_depth = resize_like(pred_depth.float(), gt_depth.shape[-2:])
    raw = DepthStats()
    raw.add(pred_depth, gt_depth, valid)

    med = DepthStats()
    gt_mask = valid.bool() & torch.isfinite(gt_depth) & (gt_depth > 0)
    mask = gt_mask & torch.isfinite(pred_depth) & (pred_depth > 0)
    aligned = pred_depth
    scale = float("nan")
    if mask.any():
        pred_med = pred_depth[mask].float().median().clamp_min(1e-6)
        gt_med = gt_depth[mask].float().median()
        scale = float((gt_med / pred_med).item())
        aligned = pred_depth * scale
    med.add(aligned, gt_depth, valid)
    return {"raw": raw.finalize(), "median_aligned": med.finalize(), "median_scale": scale}


def warp_psnr(warped, image1, valid):
    if warped is None or valid is None:
        return float("nan")
    warped = warped.float()
    image1 = image1.float()
    valid = valid.bool().expand_as(image1) & torch.isfinite(warped) & torch.isfinite(image1)
    if not valid.any():
        return float("nan")
    mse = float(((warped[valid] - image1[valid]) ** 2).mean().item())
    return -10.0 * math.log10(max(mse, 1e-12))


def image_hwc(image):
    return image.detach().cpu().float().clamp(0, 1).permute(1, 2, 0).numpy()


def color_limits(values, lo=2, hi=98):
    values = np.asarray(values, dtype=np.float32)
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return 0.0, 1.0
    vmin = float(np.percentile(finite, lo))
    vmax = float(np.percentile(finite, hi))
    if vmax <= vmin:
        vmax = vmin + 1e-6
    return vmin, vmax


def render_frame(out_path, frame_idx, batch, clean, lowlight, event, has_gt):
    image1 = batch["images_photo"][0, 0]
    dark1 = batch["images_photo_dark"][0, 0]
    event_activity = batch["event_voxel"][0, 0].abs().sum(dim=0).float().numpy()
    target_shape = image1.shape[-2:]

    clean_depth = resize_like(clean["depth"][0:1], target_shape)[0, 0]
    low_depth = resize_like(lowlight["depth"][0:1], target_shape)[0, 0]
    event_depth = resize_like(event["depth"][0:1], target_shape)[0, 0]

    sample_id = batch.get("sample_id", [f"frame_{frame_idx:04d}"])
    if isinstance(sample_id, (list, tuple)):
        sample_id = sample_id[0]

    if has_gt:
        gt_depth = as_bchw(batch["depth_gt"])[0:1]
        gt_valid = as_bchw(batch["depth_valid"])[0:1].bool()
        gt_2d = gt_depth[0, 0]
        valid_2d = gt_valid[0, 0].numpy()
        depth_values = gt_2d.numpy()[valid_2d]
        depth_vmin, depth_vmax = color_limits(depth_values)
        event_err = (event_depth - gt_2d).abs().numpy()
        low_err = (low_depth - gt_2d).abs().numpy()
        imp = low_err - event_err
        err_values = np.concatenate([event_err[valid_2d].reshape(-1), low_err[valid_2d].reshape(-1)])
        _, err_vmax = color_limits(err_values, lo=0, hi=95)
        imp_abs = max(float(np.percentile(np.abs(imp[valid_2d]), 98)), 1e-6)
        row2 = [
            (gt_2d.numpy(), "GT depth", "viridis", depth_vmin, depth_vmax),
            (event_err, "Event abs error", "inferno", 0.0, err_vmax),
            (imp, "Error improvement\npositive=event better", "coolwarm", -imp_abs, imp_abs),
        ]
    else:
        all_depth = np.concatenate([clean_depth.numpy().reshape(-1), low_depth.numpy().reshape(-1), event_depth.numpy().reshape(-1)])
        depth_vmin, depth_vmax = color_limits(all_depth)
        diff = (event_depth - low_depth).numpy()
        diff_abs = max(float(np.percentile(np.abs(diff[np.isfinite(diff)]), 98)), 1e-6)
        row2 = [
            (clean_depth.numpy(), "Clean RGB depth", "viridis", depth_vmin, depth_vmax),
            (low_depth.numpy(), "Low-light depth", "viridis", depth_vmin, depth_vmax),
            (diff, "Event - low-light depth", "coolwarm", -diff_abs, diff_abs),
        ]

    fig, axes = plt.subplots(3, 3, figsize=(15, 12), constrained_layout=True)
    fig.suptitle(str(sample_id), fontsize=13)
    panels = [
        (image_hwc(image1), "RGB view1", None, None, None),
        (image_hwc(dark1), "Simulated low-light", None, None, None),
        (event_activity, "Event activity", "magma", None, None),
        (clean_depth.numpy(), "Clean RGB depth", "viridis", depth_vmin, depth_vmax),
        (low_depth.numpy(), "Low-light depth", "viridis", depth_vmin, depth_vmax),
        (event_depth.numpy(), "Event3R depth", "viridis", depth_vmin, depth_vmax),
        *row2,
    ]
    for ax, (data, title, cmap, vmin, vmax) in zip(axes.reshape(-1), panels):
        if cmap is None:
            ax.imshow(np.clip(data, 0, 1))
        else:
            im = ax.imshow(data, cmap=cmap, vmin=vmin, vmax=vmax)
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
        ax.set_title(title, fontsize=10)
        ax.axis("off")
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def frame_record(idx, batch, clean, lowlight, event):
    sample_id = batch.get("sample_id", [f"frame_{idx:04d}"])
    if isinstance(sample_id, (list, tuple)):
        sample_id = sample_id[0]
    rec = {"frame": idx, "sample_id": str(sample_id)}
    has_gt = "depth_gt" in batch and "depth_valid" in batch
    if has_gt:
        gt = as_bchw(batch["depth_gt"])[0:1]
        valid = as_bchw(batch["depth_valid"])[0:1].bool()
        for name, output in (("clean_rgb", clean), ("lowlight_rgb", lowlight), ("event3r", event)):
            rec[name] = depth_metrics(output["depth"][0:1], gt, valid)
    for name, output in (("clean_rgb", clean), ("lowlight_rgb", lowlight), ("event3r", event)):
        rec[f"{name}_warp_psnr_db"] = warp_psnr(output.get("warped"), batch["images_photo"][:, 0], output.get("warp_valid"))
    return rec


def summarize(records):
    summary = {"num_frames": len(records)}
    for model in ("clean_rgb", "lowlight_rgb", "event3r"):
        for mode in ("raw", "median_aligned"):
            values = {}
            for rec in records:
                metrics = rec.get(model, {}).get(mode, {})
                for key, value in metrics.items():
                    if isinstance(value, (int, float)) and math.isfinite(float(value)):
                        values.setdefault(key, []).append(float(value))
            if values:
                summary[f"{model}_{mode}"] = {
                    key: sum(vals) / len(vals) for key, vals in sorted(values.items())
                }
        warp_vals = [
            rec.get(f"{model}_warp_psnr_db")
            for rec in records
            if isinstance(rec.get(f"{model}_warp_psnr_db"), (int, float))
            and math.isfinite(float(rec.get(f"{model}_warp_psnr_db")))
        ]
        if warp_vals:
            summary[f"{model}_warp_psnr_db"] = sum(warp_vals) / len(warp_vals)
    return summary


def write_csv(path, records):
    rows = []
    for rec in records:
        row = {"frame": rec["frame"], "sample_id": rec["sample_id"]}
        for model in ("clean_rgb", "lowlight_rgb", "event3r"):
            for mode in ("raw", "median_aligned"):
                for key, value in rec.get(model, {}).get(mode, {}).items():
                    row[f"{model}_{mode}_{key}"] = value
            row[f"{model}_warp_psnr_db"] = rec.get(f"{model}_warp_psnr_db")
        rows.append(row)
    fieldnames = sorted({key for row in rows for key in row.keys()})
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_html(outdir, manifest):
    summary = manifest.get("summary", {})
    html = [
        "<!doctype html><meta charset='utf-8'>",
        "<title>Event3R Sequence Video</title>",
        "<style>body{font-family:Arial,sans-serif;margin:24px;background:#f6f6f6;color:#111}"
        "video,img{max-width:100%;display:block;margin:12px 0 24px;border:1px solid #ccc;background:white}"
        "table{border-collapse:collapse;background:white}td,th{border:1px solid #ccc;padding:4px 8px}</style>",
        "<h1>Event3R Sequence Video</h1>",
        f"<p><code>{manifest['data_root']}</code></p>",
        "<video controls src='sequence.mp4'></video>",
        "<h2>Summary</h2>",
        "<table><tr><th>metric</th><th>value</th></tr>",
    ]
    flat = {}
    for key, value in summary.items():
        if isinstance(value, dict):
            for sub_key, sub_value in value.items():
                flat[f"{key}/{sub_key}"] = sub_value
        else:
            flat[key] = value
    for key, value in sorted(flat.items()):
        if isinstance(value, float):
            value = f"{value:.6g}"
        html.append(f"<tr><td>{key}</td><td>{value}</td></tr>")
    html.extend(
        [
            "</table>",
            "<p><a href='metrics.csv'>metrics.csv</a> | <a href='metrics.json'>metrics.json</a> | <a href='summary.json'>summary.json</a></p>",
            "<h2>First Frames</h2>",
        ]
    )
    for frame in manifest.get("frames", [])[:6]:
        html.append(f"<h3>{frame['sample_id']}</h3>")
        html.append(f"<img src='{Path(frame['image']).name}'>")
    with open(Path(outdir) / "index.html", "w") as f:
        f.write("\n".join(html))


def encode_video(frames_dir, output_path, fps):
    import imageio.v2 as imageio

    frame_paths = sorted(Path(frames_dir).glob("frame_*.png"))
    try:
        with imageio.get_writer(str(output_path), fps=fps, codec="libx264", quality=8, macro_block_size=16) as writer:
            for path in frame_paths:
                writer.append_data(imageio.imread(path))
    except Exception:
        import subprocess

        pattern = str(Path(frames_dir) / "frame_%04d.png")
        subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-framerate",
                str(fps),
                "-i",
                pattern,
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


def main():
    cli = parse_args()
    cfg = load_config(cli.config)
    args = make_train_args(cli)
    device = resolve_device(args.device)
    outdir = Path(cli.outdir)
    frames_dir = outdir / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)

    dataset, batches = cache_batches(args, cfg)
    rgb_model = load_backbone(args, cfg).to(device).eval()
    clean_outputs = predict_rgb(rgb_model, batches, args, cfg, device, prefer_dark=False)
    low_outputs = predict_rgb(rgb_model, batches, args, cfg, device, prefer_dark=True)
    del rgb_model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    event_model, load_info = build_event_model(args, cfg, device)
    event_outputs = predict_event(event_model, batches, args, cfg, device)
    del event_model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    records = []
    frame_items = []
    for idx, (batch, clean, low, event) in enumerate(zip(batches, clean_outputs, low_outputs, event_outputs)):
        rec = frame_record(idx, batch, clean, low, event)
        out_path = frames_dir / f"frame_{idx:04d}.png"
        render_frame(out_path, idx, batch, clean, low, event, has_gt="depth_gt" in batch and "depth_valid" in batch)
        rec["image"] = str(out_path)
        records.append(rec)
        frame_items.append({"sample_id": rec["sample_id"], "image": str(out_path)})

    summary = summarize(records)
    write_csv(outdir / "metrics.csv", records)
    with open(outdir / "metrics.json", "w") as f:
        json.dump(records, f, indent=2)
    with open(outdir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2, sort_keys=True)
    video_path = outdir / "sequence.mp4"
    encode_video(frames_dir, video_path, cli.fps)

    manifest = {
        "data_root": getattr(dataset, "root_paths_text", str(dataset.root_path)),
        "dataset_len": len(dataset),
        "num_frames": len(records),
        "fps": cli.fps,
        "video": str(video_path),
        "trained_checkpoint": str(Path(cli.trained_checkpoint).resolve()),
        "trained_load_info": load_info,
        "summary": summary,
        "frames": frame_items,
    }
    with open(outdir / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)
    write_html(outdir, manifest)
    print(json.dumps({"outdir": str(outdir.resolve()), "video": str(video_path), "num_frames": len(records)}, indent=2))


if __name__ == "__main__":
    main()
