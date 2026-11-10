#!/usr/bin/env python3
import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from train_event_dark3r import (
    build_dataset,
    build_event_student,
    cfg_get,
    evaluate,
    load_backbone,
    load_config,
    make_frozen_teacher,
    make_loader,
    resolve_amp_config,
    resolve_stage_schedule,
    safe_torch_load,
    set_event_training_stage,
    should_use_symmetric,
    validate_loss_config,
)


def parse_args():
    parser = argparse.ArgumentParser("Evaluate an Event-Dark3R checkpoint")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--config", default="config/train_event_dark3r.yaml")
    parser.add_argument("--data-root", default=None)
    parser.add_argument("--model-name-or-path", default=None)
    parser.add_argument("--allow-unsafe-checkpoint-load", action="store_true")
    parser.add_argument("--image-size", type=int, nargs=2, default=None, metavar=("H", "W"))
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--max-batches", type=int, default=None)
    parser.add_argument("--max-scenes", type=int, default=None)
    parser.add_argument("--max-pairs-per-scene", type=int, default=None)
    parser.add_argument("--pair-stride", type=int, default=None)
    parser.add_argument(
        "--pair-sampling",
        choices=("fixed", "random", "precomputed"),
        default=None,
    )
    parser.add_argument("--min-pair-distance", type=int, default=None)
    parser.add_argument("--max-pair-distance", type=int, default=None)
    parser.add_argument("--precomputed-pairs", type=str, default=None)
    parser.add_argument("--allow-reverse-pairs", dest="allow_reverse_pairs", action="store_true", default=None)
    parser.add_argument("--no-allow-reverse-pairs", dest="allow_reverse_pairs", action="store_false")
    parser.add_argument("--symmetric-pairs", dest="symmetric_pairs", action="store_true", default=None)
    parser.add_argument("--no-symmetric-pairs", dest="symmetric_pairs", action="store_false")
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--stage", type=int, default=None)
    parser.add_argument("--partial-finetune-blocks", type=int, default=None)
    parser.add_argument("--amp", dest="amp", action="store_true", default=None)
    parser.add_argument("--no-amp", dest="amp", action="store_false")
    parser.add_argument("--amp-dtype", choices=("fp16", "bf16"), default=None)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", default=None)
    return parser.parse_args()


def load_checkpoint_config(args):
    ckpt = safe_torch_load(
        args.checkpoint,
        map_location="cpu",
        allow_unsafe=args.allow_unsafe_checkpoint_load,
        purpose="event_dark3r checkpoint",
    )
    cfg = ckpt.get("config") or load_config(args.config)
    if args.config:
        file_cfg = load_config(args.config)
        if file_cfg:
            cfg = file_cfg if not ckpt.get("config") else cfg
    return ckpt, cfg


def make_train_arg_namespace(args):
    return SimpleNamespace(
        config=args.config,
        data_root=args.data_root,
        save_dir=str(Path(args.output).parent if args.output else Path(args.checkpoint).parent),
        model_name_or_path=args.model_name_or_path,
        allow_unsafe_checkpoint_load=args.allow_unsafe_checkpoint_load,
        use_tiny_model=False,
        image_size=args.image_size,
        batch_size=args.batch_size,
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
        symmetric_pairs=args.symmetric_pairs,
        num_workers=args.num_workers,
        stage=args.stage,
        partial_finetune_blocks=args.partial_finetune_blocks,
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


def main():
    args = parse_args()
    torch.manual_seed(args.seed)

    ckpt, cfg = load_checkpoint_config(args)
    train_args = make_train_arg_namespace(args)
    device = resolve_device(args.device)

    dataset = build_dataset(train_args, cfg)
    batch_size = args.batch_size or int(cfg_get(cfg, "eval", "batch_size", cfg_get(cfg, "training", "batch_size", 1)))
    num_workers = args.num_workers
    if num_workers is None:
        num_workers = int(cfg_get(cfg, "eval", "num_workers", cfg_get(cfg, "dataset", "num_workers", 0)))
    max_batches = args.max_batches
    if max_batches is None:
        max_batches = cfg.get("eval", {}).get("max_batches", None)
        max_batches = int(max_batches) if max_batches is not None else None

    loader, _ = make_loader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        device=device,
        distributed=False,
        shuffle=False,
    )

    backbone = load_backbone(train_args, cfg)
    teacher = make_frozen_teacher(backbone).to(device)
    student = build_event_student(backbone, config=cfg).to(device)

    missing, unexpected = student.load_state_dict(ckpt["model"], strict=False)
    stage_schedule = resolve_stage_schedule(cfg, train_args)
    final_stage = stage_schedule[-1]
    set_event_training_stage(
        student,
        stage=final_stage["stage"],
        partial_finetune_blocks=final_stage["partial_finetune_blocks"],
    )
    validate_loss_config(cfg, dataset, student)

    amp_enabled, amp_dtype_name, amp_dtype = resolve_amp_config(cfg, train_args, device)
    symmetric = should_use_symmetric(cfg, train_args)
    metrics = evaluate(
        student,
        teacher,
        loader,
        device,
        cfg.get("loss", {}),
        max_batches=max_batches,
        symmetric=symmetric,
        amp_enabled=amp_enabled,
        amp_dtype=amp_dtype,
    )

    result = {
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "step": ckpt.get("step"),
        "stage": ckpt.get("stage"),
        "epoch": ckpt.get("epoch"),
        "device": str(device),
        "data_root": getattr(dataset, "root_paths_text", str(dataset.root_path)),
        "dataset_len": len(dataset),
        "batch_size": batch_size,
        "max_batches": max_batches,
        "symmetric_pairs": symmetric,
        "amp": amp_enabled,
        "amp_dtype": amp_dtype_name if amp_enabled else None,
        "missing_keys": list(missing),
        "unexpected_keys": list(unexpected),
        "metrics": metrics,
    }

    print(json.dumps(result, indent=2, sort_keys=True))
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        with open(output, "w") as f:
            json.dump(result, f, indent=2, sort_keys=True)


if __name__ == "__main__":
    main()
