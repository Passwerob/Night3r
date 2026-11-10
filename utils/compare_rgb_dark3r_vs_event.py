#!/usr/bin/env python3
import argparse
import json
import random
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
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
    DenseMetricAccumulator,
    as_bchw,
    depth_from_pred,
)


def parse_args():
    parser = argparse.ArgumentParser("Compare true RGB-only MASt3R/Dark3R baselines against trained Event-Dark3R")
    parser.add_argument("--trained-checkpoint", required=True)
    parser.add_argument("--config", default="config/train_event_dark3r.yaml")
    parser.add_argument("--data-root", default=None)
    parser.add_argument("--model-name-or-path", required=True)
    parser.add_argument(
        "--dark3r-lora-checkpoint",
        default=None,
        help=(
            "Optional Dark3R LoRA checkpoint, for example "
            "checkpoints/dark3r_pretrained/weights/0049_lora.pth. "
            "Requires loralib. When set, the result includes dark3r_clean "
            "and dark3r_lowlight in addition to the MASt3R rgb_* baselines."
        ),
    )
    parser.add_argument("--allow-unsafe-checkpoint-load", action="store_true")
    parser.add_argument("--image-size", type=int, nargs=2, default=None, metavar=("H", "W"))
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--max-batches", type=int, default=32)
    parser.add_argument("--max-scenes", type=int, default=None)
    parser.add_argument("--max-pairs-per-scene", type=int, default=None)
    parser.add_argument("--pair-stride", type=int, default=None)
    parser.add_argument("--pair-sampling", choices=("fixed", "random", "precomputed"), default=None)
    parser.add_argument("--min-pair-distance", type=int, default=None)
    parser.add_argument("--max-pair-distance", type=int, default=None)
    parser.add_argument("--precomputed-pairs", type=str, default=None)
    parser.add_argument("--allow-reverse-pairs", dest="allow_reverse_pairs", action="store_true", default=None)
    parser.add_argument("--no-allow-reverse-pairs", dest="allow_reverse_pairs", action="store_false")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--amp", dest="amp", action="store_true", default=None)
    parser.add_argument("--no-amp", dest="amp", action="store_false")
    parser.add_argument("--amp-dtype", choices=("fp16", "bf16"), default=None)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", default=None)
    return parser.parse_args()


def make_train_args(args):
    return SimpleNamespace(
        config=args.config,
        data_root=args.data_root,
        save_dir=str(Path(args.output).parent if args.output else "."),
        trained_checkpoint=args.trained_checkpoint,
        model_name_or_path=args.model_name_or_path,
        dark3r_lora_checkpoint=args.dark3r_lora_checkpoint,
        allow_unsafe_checkpoint_load=args.allow_unsafe_checkpoint_load,
        use_tiny_model=False,
        image_size=args.image_size,
        batch_size=args.batch_size,
        max_batches=args.max_batches,
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


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


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
    for batch_idx, batch in enumerate(loader):
        if args.max_batches is not None and batch_idx >= args.max_batches:
            break
        cached.append({key: value.detach().cpu() if torch.is_tensor(value) else value for key, value in batch.items()})
    return dataset, cached


def build_trained_event_model(args, cfg, device):
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
        purpose="trained Event-Dark3R checkpoint",
    )
    missing, unexpected = model.load_state_dict(payload["model"], strict=False)
    model = model.to(device)
    model.eval()
    return model, {
        "step": payload.get("step"),
        "missing_keys": list(missing),
        "unexpected_keys": list(unexpected),
    }


def build_dark3r_lora_model(args, device):
    try:
        from model.lora_model import LoRAMast3r
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "Dark3R LoRA evaluation requires loralib. Install it in this "
            "environment before using --dark3r-lora-checkpoint."
        ) from exc

    source = Path(str(args.model_name_or_path))
    if source.is_file():
        model = load_backbone(args, None)
        # The checkpoint loader constructs AsymmetricMASt3R from serialized args.
        # LoRAMast3r has the same layout and adds only LoRA helper methods.
        model.__class__ = LoRAMast3r
    else:
        model = LoRAMast3r.from_pretrained(
            str(args.model_name_or_path),
            allow_unsafe_checkpoint_load=args.allow_unsafe_checkpoint_load,
        )
    model.load_pretrained_lora(args.dark3r_lora_checkpoint)
    return model.to(device).eval()


def eval_rgb_backbone(model, batches, device, amp_enabled, amp_dtype, prefer_dark):
    metrics = DenseMetricAccumulator()
    sample_ids = []
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
            pred_depth = depth_from_pred(pred1)
            if pred_depth is not None and "depth_gt" in batch_gpu:
                gt_depth = as_bchw(batch_gpu["depth_gt"])
                gt_valid = as_bchw(batch_gpu["depth_valid"])
                metrics.add_depth(pred_depth, gt_depth, gt_valid, variant="raw")
                metrics.add_depth(pred_depth, gt_depth, gt_valid, variant="median_aligned")
                metrics.add_photo_psnr(pred_depth, batch_gpu)
            batch_sample_ids = batch.get("sample_id", None)
            if isinstance(batch_sample_ids, (list, tuple)):
                sample_ids.extend(str(item) for item in batch_sample_ids)
            elif batch_sample_ids is not None:
                sample_ids.append(str(batch_sample_ids))
    return {"dense_metrics": metrics.finalize(), "sample_ids": sample_ids}


def eval_event_model(model, batches, device, amp_enabled, amp_dtype):
    metrics = DenseMetricAccumulator()
    sample_ids = []
    with torch.no_grad():
        for batch in batches:
            batch_gpu = move_batch_to_device(batch, device)
            with torch.amp.autocast(
                device_type=device.type,
                dtype=amp_dtype,
                enabled=bool(amp_enabled and device.type == "cuda"),
            ):
                pred1, _ = model(batch_gpu)
            pred_depth = depth_from_pred(pred1)
            if pred_depth is not None and "depth_gt" in batch_gpu:
                gt_depth = as_bchw(batch_gpu["depth_gt"])
                gt_valid = as_bchw(batch_gpu["depth_valid"])
                metrics.add_depth(pred_depth, gt_depth, gt_valid, variant="raw")
                metrics.add_depth(pred_depth, gt_depth, gt_valid, variant="median_aligned")
                metrics.add_photo_psnr(pred_depth, batch_gpu)
            batch_sample_ids = batch.get("sample_id", None)
            if isinstance(batch_sample_ids, (list, tuple)):
                sample_ids.extend(str(item) for item in batch_sample_ids)
            elif batch_sample_ids is not None:
                sample_ids.append(str(batch_sample_ids))
    return {"dense_metrics": metrics.finalize(), "sample_ids": sample_ids}


def metric_delta(reference, candidate):
    out = {}
    ref = reference.get("dense_metrics", {})
    cand = candidate.get("dense_metrics", {})
    for key, ref_value in ref.items():
        cand_value = cand.get(key)
        if isinstance(ref_value, (int, float)) and isinstance(cand_value, (int, float)):
            diff = float(cand_value) - float(ref_value)
            item = {"candidate_minus_reference": diff}
            if abs(float(ref_value)) > 1e-12:
                item["relative_change"] = diff / abs(float(ref_value))
            out[key] = item
    return out


def dataset_root_text(dataset):
    return getattr(dataset, "root_paths_text", str(getattr(dataset, "root_path", "")))


def main():
    cli_args = parse_args()
    cfg = load_config(cli_args.config)
    train_args = make_train_args(cli_args)
    device = resolve_device(cli_args.device)
    set_seed(cli_args.seed)

    dataset, batches = cache_batches(train_args, cfg)
    amp_enabled, amp_dtype_name, amp_dtype = resolve_amp_config(cfg, train_args, device)

    rgb_model = load_backbone(train_args, cfg).to(device).eval()
    rgb_clean = eval_rgb_backbone(rgb_model, batches, device, amp_enabled, amp_dtype, prefer_dark=False)
    rgb_lowlight = eval_rgb_backbone(rgb_model, batches, device, amp_enabled, amp_dtype, prefer_dark=True)
    del rgb_model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    event_model, load_info = build_trained_event_model(train_args, cfg, device)
    trained_event = eval_event_model(event_model, batches, device, amp_enabled, amp_dtype)
    del event_model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    result = {
        "note": (
            "rgb_clean/rgb_lowlight are RGB-only MASt3R checkpoint baselines. "
            "If dark3r_lora_checkpoint is set, dark3r_clean/dark3r_lowlight "
            "are RGB-only Dark3R LoRA baselines. All RGB-only baselines bypass "
            "the Event-Dark3R wrapper and do not use event encoder/fusion."
        ),
        "model_name_or_path": str(Path(cli_args.model_name_or_path).resolve()),
        "dark3r_lora_checkpoint": (
            None
            if cli_args.dark3r_lora_checkpoint is None
            else str(Path(cli_args.dark3r_lora_checkpoint).resolve())
        ),
        "trained_checkpoint": str(Path(cli_args.trained_checkpoint).resolve()),
        "data_root": dataset_root_text(dataset),
        "dataset_len": len(dataset),
        "num_batches": len(batches),
        "batch_size": cli_args.batch_size,
        "amp": amp_enabled,
        "amp_dtype": amp_dtype_name if amp_enabled else None,
        "rgb_clean": rgb_clean,
        "rgb_lowlight": rgb_lowlight,
        "trained_event": {**trained_event, "load_info": load_info},
        "sample_ids_match": (
            rgb_clean["sample_ids"] == rgb_lowlight["sample_ids"] == trained_event["sample_ids"]
        ),
        "delta_trained_vs_rgb_clean": metric_delta(rgb_clean, trained_event),
        "delta_trained_vs_rgb_lowlight": metric_delta(rgb_lowlight, trained_event),
    }
    if cli_args.dark3r_lora_checkpoint is not None:
        dark3r_model = build_dark3r_lora_model(train_args, device)
        dark3r_clean = eval_rgb_backbone(dark3r_model, batches, device, amp_enabled, amp_dtype, prefer_dark=False)
        dark3r_lowlight = eval_rgb_backbone(dark3r_model, batches, device, amp_enabled, amp_dtype, prefer_dark=True)
        del dark3r_model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        result.update(
            {
                "dark3r_clean": dark3r_clean,
                "dark3r_lowlight": dark3r_lowlight,
                "dark3r_sample_ids_match": (
                    dark3r_clean["sample_ids"]
                    == dark3r_lowlight["sample_ids"]
                    == trained_event["sample_ids"]
                ),
                "delta_trained_vs_dark3r_clean": metric_delta(dark3r_clean, trained_event),
                "delta_trained_vs_dark3r_lowlight": metric_delta(dark3r_lowlight, trained_event),
            }
        )
    print(json.dumps(result, indent=2, sort_keys=True))
    if cli_args.output:
        output = Path(cli_args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        with open(output, "w") as f:
            json.dump(result, f, indent=2, sort_keys=True)


if __name__ == "__main__":
    main()
