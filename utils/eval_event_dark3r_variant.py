#!/usr/bin/env python3
import argparse
import json
import random
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

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
from utils.depth_cutoff_metrics import DepthCutoffAccumulator  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser("Evaluate one Master3r/Dark3r/Night3r variant")
    parser.add_argument("--variant", required=True, choices=("mast3r", "dark3r_lora", "event_dark3r"))
    parser.add_argument("--checkpoint", default=None, help="Required for event_dark3r.")
    parser.add_argument(
        "--geometry-head-checkpoint",
        default=None,
        help="Optional compact downstream_head1/2 recovery checkpoint.",
    )
    parser.add_argument("--config", default="config/train_event_dark3r_mvsec_dsec_lora.yaml")
    parser.add_argument("--data-root", default=None)
    parser.add_argument("--model-name-or-path", default=None)
    parser.add_argument("--dark3r-lora-checkpoint", default=None)
    parser.add_argument("--allow-unsafe-checkpoint-load", action="store_true")
    parser.add_argument(
        "--restore-frozen-rgb",
        action="store_true",
        help=(
            "Keep the pristine RGB backbone loaded from --model-name-or-path and "
            "load only the Night3R adapter parameters from --checkpoint."
        ),
    )
    parser.add_argument(
        "--restore-frozen-rgb-scope",
        choices=("none", "head2", "view2", "heads", "geometry", "all"),
        default="none",
        help=(
            "Select which pristine MASt3R RGB parameters replace drifted checkpoint "
            "parameters. --restore-frozen-rgb is an alias for scope=all."
        ),
    )
    parser.add_argument(
        "--restore-frozen-rgb-alpha",
        type=float,
        default=1.0,
        help=(
            "Interpolation weight for restored pristine RGB parameters: "
            "0 keeps the Night3R checkpoint and 1 uses pristine MASt3R weights."
        ),
    )
    parser.add_argument("--image-size", type=int, nargs=2, default=None, metavar=("H", "W"))
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--max-batches", type=int, default=32)
    parser.add_argument("--max-scenes", type=int, default=None)
    parser.add_argument("--max-pairs-per-scene", type=int, default=None)
    parser.add_argument(
        "--start-index",
        type=int,
        default=0,
        help="Dataset/cache index offset before collecting or evaluating batches.",
    )
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
    parser.add_argument(
        "--no-lowlight-aug",
        action="store_true",
        help="Disable low-light augmentation so already-night datasets are used directly.",
    )
    parser.add_argument("--depth-min", type=float, default=None, help="Optional GT depth lower bound for metrics.")
    parser.add_argument("--depth-max", type=float, default=None, help="Optional GT depth upper bound for metrics.")
    parser.add_argument(
        "--depth-reference",
        choices=("gt", "pseudo", "auto"),
        default="gt",
        help=(
            "Depth target used for metrics. 'pseudo' evaluates agreement with "
            "depth_pseudo tensors and never relabels them as sensor ground truth."
        ),
    )
    parser.add_argument(
        "--pseudo-confidence-percentile",
        type=float,
        default=40.0,
        help="Discard pseudo-depth pixels below this per-sample confidence percentile.",
    )
    parser.add_argument(
        "--batch-cache",
        default=None,
        help="Optional torch-saved eval batch cache. Use the same cache for fair multi-process comparison.",
    )
    parser.add_argument(
        "--write-batch-cache",
        action="store_true",
        help="Write --batch-cache from the configured dataset and exit before model construction.",
    )
    parser.add_argument(
        "--input",
        choices=("clean", "lowlight"),
        default="lowlight",
        help="RGB input for mast3r/dark3r_lora. event_dark3r always uses its wrapper default.",
    )
    parser.add_argument(
        "--event-mode",
        choices=(
            "real",
            "zero",
            "shuffle",
            "polarity_flip",
            "temporal_reverse",
            "dropout_25",
            "dropout_50",
            "dropout_75",
            "hot_pixels",
        ),
        default="real",
        help=(
            "Event ablation for event_dark3r. real uses the dataset event voxel, "
            "zero removes event evidence, and shuffle keeps event statistics but "
            "spatially misaligns the voxel as a negative control. polarity_flip "
            "swaps positive/negative polarity channels, and temporal_reverse "
            "reverses the event-bin order inside each polarity stream. dropout_* "
            "deterministically removes a spatial fraction of events, while "
            "hot_pixels injects sparse high-activity sensor artifacts."
        ),
    )
    parser.add_argument("--output", required=True)
    parser.add_argument("--prediction-dir", type=Path, default=None)
    parser.add_argument("--prediction-count", type=int, default=0)
    return parser.parse_args()


def make_train_args(args):
    return SimpleNamespace(
        config=args.config,
        data_root=args.data_root,
        save_dir=str(Path(args.output).parent),
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
        depth_min=args.depth_min,
        depth_max=args.depth_max,
    )


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(device_arg):
    if device_arg == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("--device cuda requested but CUDA is unavailable")
        return torch.device("cuda")
    if device_arg == "cpu":
        return torch.device("cpu")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def cfg_without_lora(cfg):
    cfg = dict(cfg)
    model_cfg = dict(cfg.get("model", {}))
    model_cfg["backbone_lora_checkpoint"] = None
    cfg["model"] = model_cfg
    return cfg


def cfg_with_lora(cfg, lora_checkpoint):
    cfg = dict(cfg)
    model_cfg = dict(cfg.get("model", {}))
    if lora_checkpoint is not None:
        model_cfg["backbone_lora_checkpoint"] = lora_checkpoint
    cfg["model"] = model_cfg
    return cfg


def build_model(args, cfg, train_args, device):
    if args.variant == "mast3r":
        model = load_backbone(train_args, cfg_without_lora(cfg)).to(device).eval()
        return model, {
            "kind": "rgb_backbone",
            "model": "mast3r",
            "lora_checkpoint": None,
            "checkpoint": str(Path(args.model_name_or_path or cfg["model"]["checkpoint"]).resolve()),
        }

    if args.variant == "dark3r_lora":
        lora_ckpt = args.dark3r_lora_checkpoint or cfg.get("model", {}).get("backbone_lora_checkpoint")
        model = load_backbone(train_args, cfg_with_lora(cfg, lora_ckpt)).to(device).eval()
        return model, {
            "kind": "rgb_backbone",
            "model": "dark3r_lora",
            "lora_checkpoint": None if lora_ckpt is None else str(Path(lora_ckpt).resolve()),
            "checkpoint": str(Path(args.model_name_or_path or cfg["model"]["checkpoint"]).resolve()),
        }

    if args.checkpoint is None:
        raise RuntimeError("--checkpoint is required for --variant event_dark3r")
    backbone = load_backbone(train_args, cfg).to(device)
    model = build_event_student(backbone, config=cfg).to(device)
    final_stage = resolve_stage_schedule(cfg, train_args)[-1]
    set_event_training_stage(
        model,
        stage=final_stage["stage"],
        partial_finetune_blocks=final_stage["partial_finetune_blocks"],
    )
    payload = safe_torch_load(
        args.checkpoint,
        map_location="cpu",
        allow_unsafe=args.allow_unsafe_checkpoint_load,
        purpose="trained Night3r checkpoint",
    )
    checkpoint_state = payload["model"]
    restore_scope = str(
        getattr(args, "restore_frozen_rgb_scope", "none") or "none"
    ).lower()
    if bool(getattr(args, "restore_frozen_rgb", False)):
        restore_scope = "all"

    def should_restore_rgb_key(key):
        if not key.startswith("dark3r.") or restore_scope == "none":
            return False
        rgb_key = key[len("dark3r.") :]
        if restore_scope == "all":
            return True
        if restore_scope == "head2":
            return rgb_key.startswith("downstream_head2.")
        if restore_scope == "view2":
            return rgb_key.startswith(("dec_blocks2.", "downstream_head2."))
        if restore_scope == "heads":
            return rgb_key.startswith(("downstream_head1.", "downstream_head2."))
        if restore_scope == "geometry":
            return rgb_key.startswith(
                (
                    "decoder_embed.",
                    "dec_blocks.",
                    "dec_blocks2.",
                    "dec_norm.",
                    "downstream_head1.",
                    "downstream_head2.",
                )
            )
        raise ValueError(f"Unsupported frozen RGB restore scope: {restore_scope}")

    restore_alpha = float(getattr(args, "restore_frozen_rgb_alpha", 1.0))
    if not 0.0 <= restore_alpha <= 1.0:
        raise ValueError("restore_frozen_rgb_alpha must be in [0, 1]")
    restore_frozen_rgb = restore_scope != "none" and restore_alpha > 0.0
    restored_rgb_keys = []
    if restore_frozen_rgb:
        restored_rgb_keys = [key for key in checkpoint_state if should_restore_rgb_key(key)]
        if restore_alpha >= 1.0:
            checkpoint_state = {
                key: value
                for key, value in checkpoint_state.items()
                if not should_restore_rgb_key(key)
            }
        else:
            pristine_state = model.state_dict()
            checkpoint_state = dict(checkpoint_state)
            for key in restored_rgb_keys:
                checkpoint_value = checkpoint_state[key]
                pristine_value = pristine_state[key].detach().to(
                    device="cpu", dtype=checkpoint_value.dtype
                )
                checkpoint_state[key] = torch.lerp(
                    checkpoint_value, pristine_value, restore_alpha
                )
    missing, unexpected = model.load_state_dict(checkpoint_state, strict=False)
    if restore_frozen_rgb and restore_alpha >= 1.0:
        restored_rgb_key_set = set(restored_rgb_keys)
        missing = [key for key in missing if key not in restored_rgb_key_set]
    geometry_head_load = None
    geometry_head_checkpoint = getattr(args, "geometry_head_checkpoint", None)
    if geometry_head_checkpoint:
        head_payload = safe_torch_load(
            geometry_head_checkpoint,
            map_location="cpu",
            allow_unsafe=args.allow_unsafe_checkpoint_load,
            purpose="trained Night3R geometry-head checkpoint",
        )
        head_state = head_payload.get("geometry_head", head_payload)
        invalid_head_keys = [
            key
            for key in head_state
            if not key.startswith(
                ("dark3r.downstream_head1.", "dark3r.downstream_head2.")
            )
        ]
        if invalid_head_keys:
            raise RuntimeError(
                "Geometry-head checkpoint contains non-head keys: "
                + ", ".join(invalid_head_keys[:20])
            )
        _, head_unexpected = model.load_state_dict(head_state, strict=False)
        if head_unexpected:
            raise RuntimeError(
                "Unexpected geometry-head checkpoint keys: "
                + ", ".join(head_unexpected[:20])
            )
        geometry_head_load = {
            "checkpoint": str(Path(geometry_head_checkpoint).resolve()),
            "step": head_payload.get("step") if isinstance(head_payload, dict) else None,
            "key_count": len(head_state),
        }
    model.eval()
    return model, {
        "kind": "event_wrapper",
        "model": "event_dark3r",
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "step": payload.get("step"),
        "stage": payload.get("stage"),
        "epoch": payload.get("epoch"),
        "missing_keys": list(missing),
        "unexpected_keys": list(unexpected),
        "restored_frozen_rgb": restore_frozen_rgb,
        "restored_frozen_rgb_scope": restore_scope,
        "restored_frozen_rgb_alpha": restore_alpha,
        "restored_rgb_key_count": len(restored_rgb_keys),
        "geometry_head_load": geometry_head_load,
    }


def tensor_to_cpu(value):
    if torch.is_tensor(value):
        return value.detach().cpu()
    if isinstance(value, dict):
        return {key: tensor_to_cpu(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return type(value)(tensor_to_cpu(item) for item in value)
    return value


def make_loader_from_dataset(dataset, args, train_args, device):
    if args.start_index > 0:
        indices = list(range(int(args.start_index), len(dataset)))
        dataset = torch.utils.data.Subset(dataset, indices)
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=train_args.num_workers,
        pin_memory=device.type == "cuda",
    )


def write_batch_cache(args, cfg, train_args, device):
    if not args.batch_cache:
        raise RuntimeError("--write-batch-cache requires --batch-cache")
    set_seed(args.seed)
    dataset = build_dataset(train_args, cfg)
    loader = make_loader_from_dataset(dataset, args, train_args, device)
    batches = []
    for batch_idx, batch in enumerate(loader):
        if args.max_batches is not None and batch_idx >= args.max_batches:
            break
        batches.append(tensor_to_cpu(batch))
    payload = {
        "config": str(Path(args.config).resolve()),
        "data_root": getattr(dataset, "root_paths_text", str(dataset.root_path)),
        "dataset_len": len(dataset),
        "start_index": args.start_index,
        "batch_size": args.batch_size,
        "max_batches": args.max_batches,
        "num_batches": len(batches),
        "seed": args.seed,
        "batches": batches,
    }
    output = Path(args.batch_cache)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, output)
    print(json.dumps({key: value for key, value in payload.items() if key != "batches"}, indent=2, sort_keys=True))


def load_batch_cache(path):
    try:
        payload = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        payload = torch.load(path, map_location="cpu")
    if not isinstance(payload, dict) or "batches" not in payload:
        raise ValueError(f"Invalid batch cache: {path}")
    return payload


def batch_sample_ids(batch):
    values = batch.get("sample_id", None)
    if values is None:
        return []
    if isinstance(values, (list, tuple)):
        return [str(item) for item in values]
    return [str(values)]


def safe_sample_name(sample_id):
    return "".join(char if char.isalnum() or char in "-_" else "_" for char in sample_id)


def save_prediction_batch(args, batch, pred_depth, already_saved):
    if args.prediction_dir is None or already_saved >= args.prediction_count:
        return already_saved
    args.prediction_dir.mkdir(parents=True, exist_ok=True)
    identifiers = batch_sample_ids(batch)
    remaining = args.prediction_count - already_saved
    count = min(pred_depth.shape[0], remaining)
    for item_idx in range(count):
        sample_id = identifiers[item_idx]
        payload = {
            "sample_id": sample_id,
            "variant": args.variant,
            "pred_depth": pred_depth[item_idx].detach().float().cpu(),
            "gt_depth": as_bchw(batch["depth_gt"])[item_idx].detach().float().cpu(),
            "gt_valid": as_bchw(batch["depth_valid"])[item_idx].detach().bool().cpu(),
            "dark_rgb": batch["images_photo_dark"][item_idx, 0].detach().float().cpu(),
            "clean_rgb": batch["images_photo"][item_idx, 0].detach().float().cpu(),
            "event_voxel": batch["event_voxel"][item_idx, 0].detach().float().cpu(),
        }
        torch.save(payload, args.prediction_dir / f"{safe_sample_name(sample_id)}.pt")
    return already_saved + count


def apply_event_mode(batch, mode, batch_idx=0):
    """Apply a deterministic event ablation without changing RGB or GT tensors."""
    if mode == "real":
        return batch
    if "event_voxel" not in batch:
        raise KeyError("--event-mode requires batch['event_voxel']")
    event = batch["event_voxel"]
    if not torch.is_tensor(event):
        raise TypeError("batch['event_voxel'] must be a tensor")

    batch = dict(batch)
    if mode == "zero":
        batch["event_voxel"] = torch.zeros_like(event)
        return batch

    if mode == "shuffle":
        # Keep the same value distribution and tensor shape, but break spatial
        # alignment to RGB. This is a stronger control than shuffling across a
        # batch, because most eval runs use batch_size=1.
        h, w = event.shape[-2:]
        shift_y = max(h // 3, 1)
        shift_x = max(w // 5, 1)
        shuffled = torch.roll(event, shifts=(shift_y, shift_x), dims=(-2, -1))
        if event.shape[-3] > 1:
            shuffled = torch.roll(shuffled, shifts=1 + (int(batch_idx) % max(event.shape[-3] - 1, 1)), dims=-3)
        batch["event_voxel"] = shuffled
        return batch

    if mode == "polarity_flip":
        if event.shape[-3] % 2 != 0:
            batch["event_voxel"] = -event
            return batch
        num_bins = event.shape[-3] // 2
        flipped = torch.cat([event[..., num_bins:, :, :], event[..., :num_bins, :, :]], dim=-3)
        batch["event_voxel"] = flipped
        return batch

    if mode == "temporal_reverse":
        if event.shape[-3] % 2 != 0:
            batch["event_voxel"] = torch.flip(event, dims=(-3,))
            return batch
        num_bins = event.shape[-3] // 2
        pos = torch.flip(event[..., :num_bins, :, :], dims=(-3,))
        neg = torch.flip(event[..., num_bins:, :, :], dims=(-3,))
        batch["event_voxel"] = torch.cat([pos, neg], dim=-3)
        return batch

    if mode in {"dropout_25", "dropout_50", "dropout_75"}:
        drop_quarters = {"dropout_25": 1, "dropout_50": 2, "dropout_75": 3}[mode]
        h, w = event.shape[-2:]
        yy = torch.arange(h, device=event.device).view(h, 1)
        xx = torch.arange(w, device=event.device).view(1, w)
        spatial_hash = (3 * yy + 5 * xx + int(batch_idx)) % 4
        keep = (spatial_hash >= drop_quarters).to(dtype=event.dtype)
        batch["event_voxel"] = event * keep.view(*([1] * (event.ndim - 2)), h, w)
        return batch

    if mode == "hot_pixels":
        h, w = event.shape[-2:]
        yy = torch.arange(h, device=event.device).view(h, 1)
        xx = torch.arange(w, device=event.device).view(1, w)
        offset_y = (7 * int(batch_idx)) % 29
        offset_x = (11 * int(batch_idx)) % 31
        hot_mask = ((yy - offset_y) % 29 == 0) & ((xx - offset_x) % 31 == 0)
        amplitude = event.detach().abs().amax().clamp_min(1.0)
        channel = torch.arange(event.shape[-3], device=event.device)
        signs = torch.where(channel % 2 == 0, 1.0, -1.0).to(event.dtype)
        signs = signs.view(*([1] * (event.ndim - 3)), event.shape[-3], 1, 1)
        batch["event_voxel"] = event + amplitude * signs * hot_mask.to(event.dtype)
        return batch

    raise ValueError(f"Unknown event mode: {mode}")


def _collect_numeric(prefix, value, out):
    if torch.is_tensor(value):
        if value.numel() == 1:
            out[prefix] = float(value.detach().float().cpu().reshape(()))
        return
    if isinstance(value, (int, float)):
        out[prefix] = float(value)
        return
    if isinstance(value, dict):
        for key, item in value.items():
            _collect_numeric(f"{prefix}.{key}", item, out)


def flatten_event_diagnostics(pred):
    out = {}
    for key in (
        "event_gate",
        "event_scale",
        "event_token_norm_before_film",
        "event_token_norm_after_film",
        "rgb_token_norm",
        "visibility_mean",
        "visibility_std",
        "visibility_min",
        "visibility_max",
    ):
        if key in pred:
            _collect_numeric(key, pred[key], out)
    for key in ("event_gate_summary", "event_stat_summary"):
        if key in pred:
            _collect_numeric(key, pred[key], out)
    return out


def summarize_event_diagnostics(values):
    if not values:
        return {}
    keys = sorted({key for item in values for key in item})
    summary = {}
    for key in keys:
        vals = [item[key] for item in values if key in item and np.isfinite(item[key])]
        if vals:
            arr = np.asarray(vals, dtype=np.float64)
            summary[key] = {
                "mean": float(arr.mean()),
                "std": float(arr.std()),
                "min": float(arr.min()),
                "max": float(arr.max()),
                "num": int(arr.size),
            }
    return summary


def summarize_rgb_input_diagnostics(values):
    if not values:
        return {}
    return {
        key: float(np.mean([item[key] for item in values]))
        for key in values[0]
    }


def select_depth_reference(batch, args):
    reference = args.depth_reference
    if reference == "auto":
        reference = "gt" if "depth_gt" in batch else "pseudo"

    if reference == "gt":
        if "depth_gt" not in batch:
            return None, None, "gt"
        return as_bchw(batch["depth_gt"]), as_bchw(batch["depth_valid"]).bool(), "gt"

    if "depth_pseudo" not in batch:
        return None, None, "pseudo"
    depth = as_bchw(batch["depth_pseudo"])
    valid_value = batch.get("depth_pseudo_valid")
    valid = as_bchw(valid_value).bool() if valid_value is not None else torch.isfinite(depth) & (depth > 0)
    valid = valid & torch.isfinite(depth) & (depth > 0)

    confidence = as_bchw(batch.get("depth_pseudo_conf"))
    percentile = float(args.pseudo_confidence_percentile)
    if confidence is not None and percentile > 0:
        confidence = confidence.float()
        for item_idx in range(confidence.shape[0]):
            finite = torch.isfinite(confidence[item_idx]) & valid[item_idx]
            if finite.any():
                threshold = torch.quantile(
                    confidence[item_idx][finite], percentile / 100.0
                )
                valid[item_idx] &= confidence[item_idx] >= threshold
    return depth, valid, "pseudo"


def evaluate(args, cfg, train_args, device):
    cache_payload = load_batch_cache(args.batch_cache) if args.batch_cache else None
    if cache_payload is None:
        set_seed(args.seed)
        dataset = build_dataset(train_args, cfg)
        loader = make_loader_from_dataset(dataset, args, train_args, device)
        data_root = getattr(dataset, "root_paths_text", str(dataset.root_path))
        dataset_len = len(dataset)
    else:
        batches = cache_payload["batches"]
        if args.max_batches is not None:
            batches = batches[: args.max_batches]
        loader = batches
        data_root = cache_payload.get("data_root", "")
        dataset_len = int(cache_payload.get("dataset_len", 0))

    model, load_info = build_model(args, cfg, train_args, device)
    amp_enabled, amp_dtype_name, amp_dtype = resolve_amp_config(cfg, train_args, device)
    metrics = DenseMetricAccumulator()
    cutoff_metrics = DepthCutoffAccumulator()
    sample_ids = []
    event_diagnostics = []
    rgb_input_diagnostics = []
    num_batches = 0
    predictions_saved = 0
    evaluated_depth_reference = None

    prefer_dark = args.input == "lowlight"
    # Dataset low-light noise is sampled lazily when num_workers=0. Reset after
    # model construction so different model variants receive the same RGB
    # degradation sequence for a shared seed and fixed dataset order.
    if cache_payload is None and train_args.num_workers == 0:
        set_seed(args.seed)
    with torch.no_grad():
        for batch_idx, batch in enumerate(loader):
            if args.max_batches is not None and batch_idx >= args.max_batches:
                break
            clean_rgb = batch.get("images_photo")
            dark_rgb = batch.get("images_photo_dark")
            if torch.is_tensor(clean_rgb) and torch.is_tensor(dark_rgb):
                clean_mean = clean_rgb.detach().float().mean()
                dark_mean = dark_rgb.detach().float().mean()
                rgb_input_diagnostics.append(
                    {
                        "clean_mean": float(clean_mean),
                        "dark_mean": float(dark_mean),
                        "dark_to_clean_mean_ratio": float(
                            dark_mean / clean_mean.clamp_min(1.0e-8)
                        ),
                        "dark_zero_fraction": float(
                            (dark_rgb.detach() <= 0.0).float().mean()
                        ),
                    }
                )
            batch = move_batch_to_device(batch, device)
            if args.variant == "event_dark3r":
                batch = apply_event_mode(batch, args.event_mode, batch_idx=batch_idx)
            num_batches += 1
            with torch.amp.autocast(
                device_type=device.type,
                dtype=amp_dtype,
                enabled=bool(amp_enabled and device.type == "cuda"),
            ):
                if args.variant == "event_dark3r":
                    pred1, _ = model(batch)
                else:
                    view1, view2 = make_view_pair_from_batch(batch, prefer_dark=prefer_dark)
                    pred1, _ = model(view1, view2)
            if args.variant == "event_dark3r":
                event_diagnostics.append(flatten_event_diagnostics(pred1))
            pred_depth = depth_from_pred(pred1)
            reference_depth, reference_valid, reference_name = select_depth_reference(
                batch, args
            )
            if pred_depth is not None and reference_depth is not None:
                evaluated_depth_reference = reference_name
                gt_depth = reference_depth
                gt_valid = reference_valid
                if args.depth_min is not None:
                    gt_valid = gt_valid & (gt_depth >= float(args.depth_min))
                if args.depth_max is not None:
                    gt_valid = gt_valid & (gt_depth <= float(args.depth_max))
                metrics.add_depth(pred_depth, gt_depth, gt_valid, variant="raw")
                metrics.add_depth(pred_depth, gt_depth, gt_valid, variant="median_aligned")
                cutoff_metrics.add_depth(pred_depth, gt_depth, gt_valid)
                if reference_name == "gt":
                    metrics.add_photo_psnr(pred_depth, batch)
                    predictions_saved = save_prediction_batch(
                        args,
                        batch,
                        pred_depth,
                        predictions_saved,
                    )
            sample_ids.extend(batch_sample_ids(batch))

    return {
        "variant": args.variant,
        "input": args.input if args.variant != "event_dark3r" else f"lowlight+event:{args.event_mode}",
        "event_mode": args.event_mode if args.variant == "event_dark3r" else None,
        "event_mode_note": (
            "shuffle uses deterministic spatial roll plus channel roll; it preserves event value statistics "
            "but breaks RGB/event alignment."
            if args.variant == "event_dark3r" and args.event_mode == "shuffle"
            else None
        ),
        "config": str(Path(args.config).resolve()),
        "data_root": data_root,
        "dataset_len": dataset_len,
        "num_batches": num_batches,
        "batch_size": args.batch_size,
        "batch_cache": None if args.batch_cache is None else str(Path(args.batch_cache).resolve()),
        "device": str(device),
        "seed": args.seed,
        "lowlight_aug_enabled": bool(cfg.get("lowlight_aug", {}).get("enabled", False)),
        "lowlight_aug": dict(cfg.get("lowlight_aug", {})),
        "rgb_input_diagnostics": summarize_rgb_input_diagnostics(
            rgb_input_diagnostics
        ),
        "depth_min": args.depth_min,
        "depth_max": args.depth_max,
        "depth_reference_requested": args.depth_reference,
        "depth_reference_evaluated": evaluated_depth_reference,
        "pseudo_confidence_percentile": args.pseudo_confidence_percentile,
        "amp": amp_enabled,
        "amp_dtype": amp_dtype_name if amp_enabled else None,
        "load_info": load_info,
        "dense_metrics": metrics.finalize(),
        "depth_cutoff_metrics": cutoff_metrics.finalize(),
        "prediction_dir": (
            str(args.prediction_dir.resolve()) if args.prediction_dir is not None else None
        ),
        "predictions_saved": predictions_saved,
        "event_diagnostics": summarize_event_diagnostics(event_diagnostics),
        "sample_ids": sample_ids,
    }


def main():
    args = parse_args()
    cfg = load_config(args.config)
    if args.no_lowlight_aug:
        cfg = dict(cfg)
        lowlight_cfg = dict(cfg.get("lowlight_aug", {}))
        lowlight_cfg["enabled"] = False
        cfg["lowlight_aug"] = lowlight_cfg
    train_args = make_train_args(args)
    device = resolve_device(args.device)
    if args.write_batch_cache:
        write_batch_cache(args, cfg, train_args, device)
        return
    result = evaluate(args, cfg, train_args, device)
    print(json.dumps(result, indent=2, sort_keys=True))
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w") as f:
        json.dump(result, f, indent=2, sort_keys=True)


if __name__ == "__main__":
    main()
