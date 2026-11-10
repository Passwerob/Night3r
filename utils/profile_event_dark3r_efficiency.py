#!/usr/bin/env python3
import argparse
import json
import statistics
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from model.event_dark3r_student import make_view_pair_from_batch  # noqa: E402
from train_event_dark3r import (  # noqa: E402
    build_dataset,
    load_config,
    move_batch_to_device,
    resolve_amp_config,
)
from utils.eval_event_dark3r_variant import build_model  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser("Profile MASt3R/Dark3R/Night3R forward efficiency")
    parser.add_argument("--variant", choices=("mast3r", "dark3r_lora", "event_dark3r"), required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--dark3r-lora-checkpoint", default=None)
    parser.add_argument("--model-name-or-path", required=True)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iterations", type=int, default=50)
    parser.add_argument("--output", required=True)
    parser.add_argument("--allow-unsafe-checkpoint-load", action="store_true")
    return parser.parse_args()


def train_args(args):
    return SimpleNamespace(
        config=args.config,
        data_root=args.data_root,
        save_dir=str(Path(args.output).parent),
        model_name_or_path=args.model_name_or_path,
        dark3r_lora_checkpoint=args.dark3r_lora_checkpoint,
        allow_unsafe_checkpoint_load=args.allow_unsafe_checkpoint_load,
        use_tiny_model=False,
        image_size=None,
        batch_size=1,
        max_batches=1,
        num_epochs=None,
        max_steps=None,
        max_scenes=None,
        max_pairs_per_scene=None,
        pair_stride=1,
        pair_sampling="fixed",
        min_pair_distance=1,
        max_pair_distance=1,
        precomputed_pairs=None,
        allow_reverse_pairs=False,
        symmetric_pairs=False,
        num_workers=0,
        stage=None,
        partial_finetune_blocks=None,
        amp=True,
        amp_dtype="bf16",
        gradient_checkpointing=False,
        device="cuda",
        log_every=1,
        no_save=True,
        seed=17,
        depth_min=None,
        depth_max=None,
    )


def percentile(values, fraction):
    ordered = sorted(values)
    index = min(int(round((len(ordered) - 1) * fraction)), len(ordered) - 1)
    return ordered[index]


def main():
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for efficiency profiling")
    device = torch.device("cuda")
    cfg = load_config(args.config)
    cfg = dict(cfg)
    cfg["lowlight_aug"] = dict(cfg.get("lowlight_aug", {}), enabled=False)
    targs = train_args(args)

    dataset = build_dataset(targs, cfg)
    loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)
    batch = move_batch_to_device(next(iter(loader)), device)
    model, load_info = build_model(args, cfg, targs, device)
    model.eval()

    amp_enabled, amp_name, amp_dtype = resolve_amp_config(cfg, targs, device)
    if args.variant == "event_dark3r":
        def forward():
            return model(batch)
    else:
        view1, view2 = make_view_pair_from_batch(batch, prefer_dark=True)

        def forward():
            return model(view1, view2)

    total_params = sum(parameter.numel() for parameter in model.parameters())
    trainable_params = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    model_parameter_bytes = sum(
        parameter.numel() * parameter.element_size() for parameter in model.parameters()
    )

    with torch.inference_mode(), torch.amp.autocast(
        device_type="cuda", dtype=amp_dtype, enabled=bool(amp_enabled)
    ):
        for _ in range(args.warmup):
            forward()
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats(device)
        baseline_allocated = torch.cuda.memory_allocated(device)
        baseline_reserved = torch.cuda.memory_reserved(device)
        timings_ms = []
        for _ in range(args.iterations):
            start = time.perf_counter()
            forward()
            torch.cuda.synchronize()
            timings_ms.append((time.perf_counter() - start) * 1000.0)

    mean_ms = statistics.fmean(timings_ms)
    payload = {
        "variant": args.variant,
        "config": str(Path(args.config).resolve()),
        "checkpoint": args.checkpoint,
        "data_root": str(Path(args.data_root).resolve()),
        "image_size": [512, 512],
        "batch_size": 1,
        "amp": bool(amp_enabled),
        "amp_dtype": amp_name,
        "warmup_iterations": args.warmup,
        "timed_iterations": args.iterations,
        "latency_ms_mean": mean_ms,
        "latency_ms_median": statistics.median(timings_ms),
        "latency_ms_p90": percentile(timings_ms, 0.90),
        "fps": 1000.0 / mean_ms,
        "peak_allocated_mib": torch.cuda.max_memory_allocated(device) / (1024 ** 2),
        "peak_reserved_mib": torch.cuda.max_memory_reserved(device) / (1024 ** 2),
        "baseline_allocated_mib": baseline_allocated / (1024 ** 2),
        "baseline_reserved_mib": baseline_reserved / (1024 ** 2),
        "total_params": total_params,
        "trainable_params_at_profile": trainable_params,
        "parameter_storage_mib": model_parameter_bytes / (1024 ** 2),
        "gpu_name": torch.cuda.get_device_name(device),
        "torch_version": torch.__version__,
        "load_info": load_info,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
