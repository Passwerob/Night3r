#!/usr/bin/env python3
import argparse
import json
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
    parser = argparse.ArgumentParser("Render qualitative predictions for Master3r/Dark3r/Night3r")
    parser.add_argument("--trained-checkpoint", required=True)
    parser.add_argument("--config", default="config/train_event_dark3r_dsec.yaml")
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--model-name-or-path", required=True)
    parser.add_argument(
        "--dark3r-lora-checkpoint",
        default="checkpoints/dark3r_pretrained/weights/0049_lora.pth",
        help="LoRA checkpoint used for the Dark3r low-light RGB baseline.",
    )
    parser.add_argument("--allow-unsafe-checkpoint-load", action="store_true")
    parser.add_argument("--image-size", type=int, nargs=2, default=None, metavar=("H", "W"))
    parser.add_argument("--max-pairs", type=int, default=6)
    parser.add_argument("--max-scenes", type=int, default=None)
    parser.add_argument("--max-pairs-per-scene", type=int, default=None)
    parser.add_argument("--pair-stride", type=int, default=None)
    parser.add_argument("--pair-sampling", choices=("fixed", "random", "precomputed"), default=None)
    parser.add_argument("--min-pair-distance", type=int, default=None)
    parser.add_argument("--max-pair-distance", type=int, default=None)
    parser.add_argument("--precomputed-pairs", type=str, default=None)
    parser.add_argument("--allow-reverse-pairs", dest="allow_reverse_pairs", action="store_true", default=None)
    parser.add_argument("--no-allow-reverse-pairs", dest="allow_reverse_pairs", action="store_false")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--amp", dest="amp", action="store_true", default=None)
    parser.add_argument("--no-amp", dest="amp", action="store_false")
    parser.add_argument("--amp-dtype", choices=("fp16", "bf16"), default=None)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--seed", type=int, default=0)
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
        num_epochs=None,
        max_steps=None,
        max_scenes=args.max_scenes,
        max_pairs_per_scene=args.max_pairs_per_scene,
        pair_stride=args.pair_stride,
        pair_sampling=args.pair_sampling,
        min_pair_distance=args.min_pair_distance,
        max_pair_distance=args.max_pair_distance,
        precomputed_pairs=args.precomputed_pairs,
        allow_reverse_pairs=args.allow_reverse_pairs,
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
    )


def resolve_device(device_arg):
    if device_arg == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("--device cuda requested but CUDA is unavailable")
        return torch.device("cuda")
    if device_arg == "cpu":
        return torch.device("cpu")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def cache_batches(args, cfg):
    set_seed(args.seed)
    dataset = build_dataset(args, cfg)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=False,
    )
    cached = []
    for batch in loader:
        cached.append({key: value.detach().cpu() if torch.is_tensor(value) else value for key, value in batch.items()})
        if len(cached) >= args.max_pairs:
            break
    return dataset, cached


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
        purpose="trained Night3r checkpoint",
    )
    state_dict = payload.get("model", payload)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    model = model.to(device).eval()
    return model, {"step": payload.get("step"), "missing_keys": list(missing), "unexpected_keys": list(unexpected)}


def resolve_repo_path(path):
    path = Path(path)
    if path.is_absolute():
        return path
    return REPO_ROOT / path


def image_hwc(image):
    return image.detach().cpu().float().clamp(0, 1).permute(1, 2, 0).numpy()


def conf_from_pred(pred):
    for key in ("conf", "desc_conf"):
        value = pred.get(key)
        if torch.is_tensor(value):
            if value.ndim == 4 and value.shape[-1] == 1:
                value = value[..., 0]
            if value.ndim == 4 and value.shape[1] == 1:
                value = value[:, 0]
            if value.ndim == 3:
                return value[:, None]
    depth = depth_from_pred(pred)
    if depth is None:
        return None
    return torch.isfinite(depth).float()


def resize_like(x, shape):
    if x is None or x.shape[-2:] == shape:
        return x
    return F.interpolate(x.float(), size=shape, mode="bilinear", align_corners=False)


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
            outputs.append(
                {
                    "depth": depth_from_pred(pred1).detach().cpu(),
                    "conf": None if conf_from_pred(pred1) is None else conf_from_pred(pred1).detach().cpu(),
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
            outputs.append(
                {
                    "depth": depth_from_pred(pred1).detach().cpu(),
                    "conf": None if conf_from_pred(pred1) is None else conf_from_pred(pred1).detach().cpu(),
                    "event_gate": pred1.get("event_gate", None),
                }
            )
    return outputs


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


def depth_stats(depth):
    d = depth.detach().cpu().float().numpy()
    finite = d[np.isfinite(d)]
    if finite.size == 0:
        return {"min": None, "median": None, "max": None}
    return {
        "min": float(np.percentile(finite, 2)),
        "median": float(np.median(finite)),
        "max": float(np.percentile(finite, 98)),
    }


def dataset_label(batch, fallback_root=None):
    source = batch.get("dataset_source", None)
    if isinstance(source, (list, tuple)) and source:
        source = source[0]
    if source is not None:
        source = str(source).strip().lower()
        if source and source != "unknown":
            return source.upper()
    text = "" if fallback_root is None else str(fallback_root).lower()
    if "dl3dv" in text:
        return "DL3DV"
    if "dsec" in text:
        return "DSEC"
    if "mvsec" in text:
        return "MVSEC"
    return "DATA"


def render_panel(out_path, idx, batch, clean, lowlight, event, label="DATA"):
    image1 = batch["images_photo"][0, 0]
    image2 = batch["images_photo"][0, 1]
    dark1 = batch["images_photo_dark"][0, 0]
    event_activity = batch["event_voxel"][0, 0].abs().sum(dim=0).float().numpy()
    target_shape = image1.shape[-2:]

    clean_depth = resize_like(clean["depth"][0:1], target_shape)[0, 0]
    low_depth = resize_like(lowlight["depth"][0:1], target_shape)[0, 0]
    event_depth = resize_like(event["depth"][0:1], target_shape)[0, 0]
    all_depth = np.concatenate(
        [
            clean_depth.numpy().reshape(-1),
            low_depth.numpy().reshape(-1),
            event_depth.numpy().reshape(-1),
        ]
    )
    depth_vmin, depth_vmax = color_limits(all_depth)
    diff_low_event = (event_depth - low_depth).numpy()
    diff_clean_event = (event_depth - clean_depth).numpy()
    diff_abs = np.concatenate([np.abs(diff_low_event).reshape(-1), np.abs(diff_clean_event).reshape(-1)])
    diff_lim = max(float(np.percentile(diff_abs[np.isfinite(diff_abs)], 98)), 1e-6)

    clean_conf = resize_like(clean["conf"][0:1], target_shape)[0, 0] if clean["conf"] is not None else torch.zeros_like(clean_depth)
    low_conf = resize_like(lowlight["conf"][0:1], target_shape)[0, 0] if lowlight["conf"] is not None else torch.zeros_like(clean_depth)
    event_conf = resize_like(event["conf"][0:1], target_shape)[0, 0] if event["conf"] is not None else torch.zeros_like(clean_depth)
    conf_values = np.concatenate([clean_conf.numpy().reshape(-1), low_conf.numpy().reshape(-1), event_conf.numpy().reshape(-1)])
    conf_vmin, conf_vmax = color_limits(conf_values, lo=2, hi=98)

    sample_id = batch.get("sample_id", [f"sample_{idx:03d}"])
    if isinstance(sample_id, (list, tuple)):
        sample_id = sample_id[0]

    fig, axes = plt.subplots(4, 4, figsize=(20, 16), constrained_layout=True)
    fig.suptitle(str(sample_id), fontsize=14)
    panels = [
        (axes[0, 0], image_hwc(image1), f"{label} clean RGB reference", None, None, None),
        (axes[0, 1], image_hwc(dark1), "Low-light input to all models", None, None, None),
        (axes[0, 2], image_hwc(image2), f"{label} RGB view2", None, None, None),
        (axes[0, 3], event_activity, "Event activity", "magma", None, None),
        (axes[1, 0], clean_depth.numpy(), "Master3r depth", "viridis", depth_vmin, depth_vmax),
        (axes[1, 1], low_depth.numpy(), "Dark3r depth", "viridis", depth_vmin, depth_vmax),
        (axes[1, 2], event_depth.numpy(), "Night3r depth", "viridis", depth_vmin, depth_vmax),
        (axes[1, 3], diff_low_event, "Night3r depth - Dark3r depth", "coolwarm", -diff_lim, diff_lim),
        (axes[2, 0], clean_conf.numpy(), "Master3r confidence", "plasma", conf_vmin, conf_vmax),
        (axes[2, 1], low_conf.numpy(), "Dark3r confidence", "plasma", conf_vmin, conf_vmax),
        (axes[2, 2], event_conf.numpy(), "Night3r confidence", "plasma", conf_vmin, conf_vmax),
        (axes[2, 3], diff_clean_event, "Night3r depth - Master3r depth", "coolwarm", -diff_lim, diff_lim),
        (axes[3, 0], clean_depth.numpy(), f"Master3r depth\nmed {depth_stats(clean_depth)['median']:.2f}", "viridis", depth_vmin, depth_vmax),
        (axes[3, 1], low_depth.numpy(), f"Dark3r depth\nmed {depth_stats(low_depth)['median']:.2f}", "viridis", depth_vmin, depth_vmax),
        (axes[3, 2], event_depth.numpy(), f"Night3r depth\nmed {depth_stats(event_depth)['median']:.2f}", "viridis", depth_vmin, depth_vmax),
        (axes[3, 3], image_hwc(image1), "Clean RGB reference", None, None, None),
    ]
    for ax, data, title, cmap, vmin, vmax in panels:
        if cmap is None:
            ax.imshow(np.clip(data, 0, 1))
        else:
            im = ax.imshow(data, cmap=cmap, vmin=vmin, vmax=vmax)
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
        ax.set_title(title, fontsize=10)
        ax.axis("off")
    fig.savefig(out_path, dpi=140)
    plt.close(fig)
    return {
        "sample_id": str(sample_id),
        "panel": str(out_path),
        "master3r_depth": depth_stats(clean_depth),
        "dark3r_depth": depth_stats(low_depth),
        "night3r_depth": depth_stats(event_depth),
    }


def write_index(outdir, records):
    html = [
        "<!doctype html><meta charset='utf-8'>",
        "<title>Master3r / Dark3r / Night3r</title>",
        "<style>body{font-family:Arial,sans-serif;margin:24px;background:#f6f6f6;color:#111}"
        "img{max-width:100%;display:block;margin:12px 0 36px;border:1px solid #ccc;background:white}</style>",
        "<h1>Qualitative Test: Master3r / Dark3r / Night3r</h1>",
        "<p>No depth_gt/pose is required for these qualitative panels.</p>",
    ]
    for item in records:
        rel = Path(item["panel"]).name
        html.append(f"<h2>{item['sample_id']}</h2>")
        html.append(f"<img src='{rel}' alt='{item['sample_id']}'>")
    with open(Path(outdir) / "index.html", "w") as f:
        f.write("\n".join(html))


def main():
    cli_args = parse_args()
    cfg = load_config(cli_args.config)
    args = make_train_args(cli_args)
    device = resolve_device(args.device)
    outdir = Path(cli_args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    dataset, batches = cache_batches(args, cfg)
    master3r_model = load_backbone(args, cfg).to(device).eval()
    master3r_outputs = predict_rgb(master3r_model, batches, args, cfg, device, prefer_dark=True)
    del master3r_model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    dark3r_args = SimpleNamespace(**vars(args))
    dark3r_args.backbone_lora_checkpoint = str(resolve_repo_path(cli_args.dark3r_lora_checkpoint))
    dark3r_model = load_backbone(dark3r_args, cfg).to(device).eval()
    low_outputs = predict_rgb(dark3r_model, batches, args, cfg, device, prefer_dark=True)
    del dark3r_model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    event_model, load_info = build_event_model(args, cfg, device)
    event_outputs = predict_event(event_model, batches, args, cfg, device)
    del event_model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    records = []
    for idx, (batch, master3r, dark3r, event) in enumerate(zip(batches, master3r_outputs, low_outputs, event_outputs)):
        label = dataset_label(batch, fallback_root=cli_args.data_root)
        prefix = label.lower()
        records.append(
            render_panel(
                outdir / f"{prefix}_sample_{idx:03d}_panel.png",
                idx,
                batch,
                master3r,
                dark3r,
                event,
                label=label,
            )
        )

    manifest = {
        "note": "All model predictions use low-light RGB input. Night3r also uses event voxel. These panels report qualitative depth/confidence stats only.",
        "data_root": getattr(dataset, "root_paths_text", str(dataset.root_path)),
        "dataset_len": len(dataset),
        "model_name_or_path": str(Path(cli_args.model_name_or_path).resolve()),
        "dark3r_lora_checkpoint": str(resolve_repo_path(cli_args.dark3r_lora_checkpoint).resolve()),
        "trained_checkpoint": str(Path(cli_args.trained_checkpoint).resolve()),
        "trained_load_info": load_info,
        "records": records,
    }
    with open(outdir / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)
    write_index(outdir, records)
    print(json.dumps({"outdir": str(outdir.resolve()), "panels": [item["panel"] for item in records]}, indent=2))


if __name__ == "__main__":
    main()
