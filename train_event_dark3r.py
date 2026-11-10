#!/usr/bin/env python3
import argparse
import gc
import json
import os
import sys
from pathlib import Path

import torch
import torch.distributed as dist
import yaml
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Sampler
from torch.utils.data.distributed import DistributedSampler

from dataset.dsec_event_dataset import DSECEventDataset
from losses.distillation_losses import feature_distillation_loss, structure_consistency_loss
from losses.event_geometry_losses import (
    depth_gradient_gt_loss,
    event_depth_edge_loss,
    event_photo_consistency_loss,
    pose_gt_loss,
    pseudo_depth_distillation_loss,
    supervised_depth_gt_loss,
)
from training.event_teacher_student import (
    build_event_optimizer,
    build_event_student,
    make_frozen_teacher,
    set_event_training_stage,
)
from model.event_dark3r_student import decode_local_descriptors, make_view_pair_from_batch


DEFAULT_MODEL_NAME_OR_PATH = (
    "/data/fcr/data/Night3R/checkpoints/"
    "MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric.pth"
)


def parse_args():
    parser = argparse.ArgumentParser("Train Night3R on event/RGB data")
    parser.add_argument("--config", default="config/train_event_dark3r.yaml")
    parser.add_argument("--data-root", default=None)
    parser.add_argument("--save-dir", required=True)
    parser.add_argument("--model-name-or-path", default=None)
    parser.add_argument(
        "--backbone-lora-checkpoint",
        default=None,
        help=(
            "Initialize the RGB backbone and teacher with an optional Dark3R "
            "LoRA checkpoint before wrapping it as Night3R."
        ),
    )
    parser.add_argument(
        "--init-event-checkpoint",
        default=None,
        help=(
            "Initialize the Night3R student from a previous Night3R/Event-Dark3R "
            "checkpoint model state, but start a fresh optimizer/schedule."
        ),
    )
    parser.add_argument(
        "--allow-unsafe-checkpoint-load",
        action="store_true",
        help=(
            "Allow legacy pickle checkpoint loading for trusted local MASt3R/Dark3R "
            "checkpoints. Leave disabled for untrusted files."
        ),
    )
    parser.add_argument("--use-tiny-model", action="store_true")
    parser.add_argument("--image-size", type=int, nargs=2, default=None, metavar=("H", "W"))
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--num-epochs", type=int, default=None)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--max-scenes", type=int, default=None)
    parser.add_argument("--max-pairs-per-scene", type=int, default=None)
    parser.add_argument("--pair-stride", type=int, default=None)
    parser.add_argument(
        "--pair-sampling",
        choices=("fixed", "random", "precomputed"),
        default=None,
        help="Override dataset.pair_sampling.",
    )
    parser.add_argument("--min-pair-distance", type=int, default=None)
    parser.add_argument("--max-pair-distance", type=int, default=None)
    parser.add_argument("--precomputed-pairs", type=str, default=None)
    parser.add_argument(
        "--allow-reverse-pairs",
        dest="allow_reverse_pairs",
        action="store_true",
        default=None,
        help="Allow random sampling to choose idx2 < idx1.",
    )
    parser.add_argument(
        "--no-allow-reverse-pairs",
        dest="allow_reverse_pairs",
        action="store_false",
    )
    parser.add_argument(
        "--symmetric-pairs",
        dest="symmetric_pairs",
        action="store_true",
        default=None,
        help="Train/evaluate both A->B and B->A for each sampled pair.",
    )
    parser.add_argument(
        "--no-symmetric-pairs",
        dest="symmetric_pairs",
        action="store_false",
    )
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--stage", type=int, default=None)
    parser.add_argument("--partial-finetune-blocks", type=int, default=None)
    parser.add_argument(
        "--amp",
        dest="amp",
        action="store_true",
        default=None,
        help="Enable CUDA autocast mixed precision for lower activation memory.",
    )
    parser.add_argument(
        "--no-amp",
        dest="amp",
        action="store_false",
        help="Disable mixed precision even if training.amp is enabled in the config.",
    )
    parser.add_argument(
        "--amp-dtype",
        choices=("fp16", "bf16"),
        default=None,
        help="Autocast dtype when AMP is enabled. Defaults to training.amp_dtype or fp16.",
    )
    parser.add_argument(
        "--gradient-checkpointing",
        dest="gradient_checkpointing",
        action="store_true",
        default=None,
        help="Checkpoint MASt3R/event transformer blocks to reduce memory at lower speed.",
    )
    parser.add_argument(
        "--no-gradient-checkpointing",
        dest="gradient_checkpointing",
        action="store_false",
        help="Disable gradient checkpointing even if enabled in the config.",
    )
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--log-every", type=int, default=1)
    parser.add_argument("--no-save", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def _deep_merge_config(base, override):
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge_config(merged[key], value)
        else:
            merged[key] = value
    return merged


def load_config(path, _stack=None):
    path = Path(path).resolve()
    stack = [] if _stack is None else list(_stack)
    if path in stack:
        chain = " -> ".join(str(item) for item in stack + [path])
        raise ValueError(f"Cyclic config inheritance: {chain}")
    with path.open("r") as f:
        cfg = yaml.safe_load(f)
    cfg = {} if cfg is None else cfg
    base_path = cfg.pop("_base_", None)
    if base_path is None:
        return cfg
    base_path = Path(base_path)
    if not base_path.is_absolute():
        base_path = path.parent / base_path
    base_cfg = load_config(base_path, _stack=stack + [path])
    return _deep_merge_config(base_cfg, cfg)


def cfg_get(cfg, section, key, default=None):
    return cfg.get(section, {}).get(key, default)


def resolve_data_root(args, cfg):
    dataset_cfg = cfg.get("dataset", {})
    data_root = args.data_root if args.data_root is not None else dataset_cfg.get("root")
    if not data_root:
        raise RuntimeError("No dataset root configured. Set --data-root or dataset.root.")
    return data_root


def resolve_model_name_or_path(args, cfg, key="checkpoint", fallback=None):
    model_cfg = cfg.get("model", {})
    if key == "checkpoint" and args.model_name_or_path is not None:
        return args.model_name_or_path

    value = model_cfg.get(key, None)
    if value:
        return value

    if args.model_name_or_path is not None:
        return args.model_name_or_path
    if fallback is not None:
        return fallback
    return DEFAULT_MODEL_NAME_OR_PATH


def resolve_backbone_lora_checkpoint(args, cfg):
    cli_value = getattr(args, "backbone_lora_checkpoint", None)
    if cli_value:
        return cli_value
    return cfg.get("model", {}).get("backbone_lora_checkpoint", None)


def safe_torch_load(path, map_location="cpu", allow_unsafe=False, purpose="checkpoint"):
    try:
        return torch.load(path, map_location=map_location, weights_only=True)
    except TypeError as exc:
        if allow_unsafe:
            return torch.load(path, map_location=map_location)
        raise RuntimeError(
            f"Safe torch.load(weights_only=True) is unavailable for {purpose}: {path}. "
            "Upgrade PyTorch or pass --allow-unsafe-checkpoint-load only for trusted files."
        ) from exc
    except Exception as exc:
        if allow_unsafe:
            return torch.load(path, map_location=map_location, weights_only=False)
        raise RuntimeError(
            f"Safe checkpoint load failed for {purpose}: {path}. "
            "If this is a trusted legacy pickle checkpoint, rerun with "
            "--allow-unsafe-checkpoint-load."
        ) from exc


def setup_distributed(device_arg):
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    distributed = world_size > 1
    if distributed:
        local_rank = int(os.environ["LOCAL_RANK"])
        if torch.cuda.is_available() and device_arg != "cpu":
            torch.cuda.set_device(local_rank)
            device = torch.device("cuda", local_rank)
            backend = "nccl"
        else:
            device = torch.device("cpu")
            backend = "gloo"
        dist.init_process_group(backend=backend)
        return device, distributed, local_rank, dist.get_rank(), world_size

    if device_arg == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("--device cuda requested but CUDA is unavailable")
        device = torch.device("cuda")
    elif device_arg == "cpu":
        device = torch.device("cpu")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return device, distributed, 0, 0, 1


def cleanup_distributed(distributed):
    if distributed:
        dist.destroy_process_group()


def distributed_is_initialized():
    return dist.is_available() and dist.is_initialized()


def sync_error_message(message, device):
    if not distributed_is_initialized():
        return message
    gathered = [None for _ in range(dist.get_world_size())]
    dist.all_gather_object(gathered, message)
    for rank_idx, item in enumerate(gathered):
        if item:
            return f"rank {rank_idx}: {item}"
    return None


def sync_finite_tensor(value, device, where, name):
    """Return True if value is finite on all ranks, False otherwise."""
    if not torch.is_tensor(value):
        value = torch.as_tensor(value, device=device)
    local_bad = int(not torch.isfinite(value).all().item())
    if distributed_is_initialized():
        bad = torch.tensor([local_bad], device=device, dtype=torch.int32)
        dist.all_reduce(bad, op=dist.ReduceOp.MAX)
        if int(bad.item()) == 0:
            return True
        rank = dist.get_rank()
        if local_bad:
            print(
                f"[rank {rank}] WARNING: Non-finite {name} at {where}: "
                f"{value.detach().cpu()}, skipping"
            )
        else:
            print(f"[rank {rank}] WARNING: Another rank has non-finite {name} at {where}, skipping")
        return False
    if local_bad:
        print(f"WARNING: Non-finite {name} at {where}: {value.detach().cpu()}, skipping")
        return False
    return True


def sync_finite_loss(loss, device, where):
    """Return True if loss is finite on all ranks, False otherwise."""
    return sync_finite_tensor(loss, device, where, "loss")


def sync_loss_below_threshold(loss, max_abs, device, where):
    if max_abs is None or max_abs <= 0:
        return True
    local_bad = int(
        torch.isfinite(loss).all().item()
        and torch.abs(loss.detach()).gt(float(max_abs)).any().item()
    )
    if distributed_is_initialized():
        bad = torch.tensor([local_bad], device=device, dtype=torch.int32)
        dist.all_reduce(bad, op=dist.ReduceOp.MAX)
        if int(bad.item()) == 0:
            return True
        rank = dist.get_rank()
        if local_bad:
            print(
                f"[rank {rank}] WARNING: Loss above max_total_loss at {where}: "
                f"{loss.detach().cpu()} > {max_abs}, skipping"
            )
        else:
            print(f"[rank {rank}] WARNING: Another rank has loss above max_total_loss at {where}, skipping")
        return False
    if local_bad:
        print(f"WARNING: Loss above max_total_loss at {where}: {loss.detach().cpu()} > {max_abs}, skipping")
        return False
    return True


def clip_grad_norm_if_enabled(parameters, max_norm):
    params = [p for p in parameters if p.requires_grad and p.grad is not None]
    if not params:
        return None
    if max_norm is None or max_norm <= 0:
        norms = [torch.linalg.vector_norm(p.grad.detach(), 2) for p in params]
        return torch.linalg.vector_norm(torch.stack(norms), 2)
    return torch.nn.utils.clip_grad_norm_(params, max_norm=max_norm)


def log_skipped_train_step(rank, global_step, reason, batch, loss, loss_dict, grad_norm=None):
    payload = {
        "rank": rank,
        "step": global_step,
        "reason": reason,
        "sample_id": batch.get("sample_id", None) if isinstance(batch, dict) else None,
        "loss": float(loss.detach().cpu()) if torch.is_tensor(loss) else loss,
        "loss_dict": loss_dict,
    }
    if grad_norm is not None:
        payload["grad_norm"] = float(grad_norm.detach().cpu()) if torch.is_tensor(grad_norm) else grad_norm
    print("event_train_skip", payload)


def barrier_if_distributed():
    if distributed_is_initialized():
        dist.barrier()


def move_batch_to_device(batch, device):
    out = {}
    for key, value in batch.items():
        if torch.is_tensor(value):
            out[key] = value.to(device, non_blocking=True)
        else:
            out[key] = value
    return out


def _sample_source(dataset, idx):
    item = dataset.samples[int(idx)]
    scene_dir = item.get("scene_dir", None)
    if scene_dir is None:
        return "unknown"
    if hasattr(dataset, "_scene_source"):
        return str(dataset._scene_source(scene_dir)).strip().lower()
    return "unknown"


class SourceBalancedSampler(Sampler):
    """Deterministic weighted round-robin sampler for mixed-source datasets."""

    def __init__(
        self,
        dataset,
        shuffle=True,
        seed=0,
        rank=0,
        world_size=1,
        source_weights=None,
    ):
        self.dataset = dataset
        self.shuffle = bool(shuffle)
        self.seed = int(seed)
        self.rank = int(rank)
        self.world_size = int(world_size)
        self.epoch = 0
        by_source = {}
        for idx in range(len(dataset)):
            by_source.setdefault(_sample_source(dataset, idx), []).append(idx)
        self.by_source = {key: value for key, value in by_source.items() if value}
        configured_weights = source_weights or {}
        if not isinstance(configured_weights, dict):
            raise TypeError("training.source_sampling_weights must be a mapping")
        unknown_sources = sorted(
            key
            for key in configured_weights
            if key != "default" and key not in self.by_source
        )
        if unknown_sources:
            raise ValueError(
                "training.source_sampling_weights contains undiscovered sources: "
                f"{unknown_sources}; discovered={sorted(self.by_source)}"
            )
        default_weight = configured_weights.get("default", 1)
        self.source_weights = {}
        for source in sorted(self.by_source):
            raw_weight = configured_weights.get(source, default_weight)
            weight = float(raw_weight)
            if weight < 0 or not weight.is_integer():
                raise ValueError(
                    "training.source_sampling_weights values must be non-negative "
                    f"integers, got {source}={raw_weight}"
                )
            self.source_weights[source] = int(weight)
        self.sources = [
            source for source in sorted(self.by_source) if self.source_weights[source] > 0
        ]
        if not self.sources:
            raise ValueError(
                "training.source_sampling_weights disables every discovered source"
            )
        self.source_cycle = [
            source
            for source in self.sources
            for _ in range(self.source_weights[source])
        ]
        self.num_samples = (len(dataset) + self.world_size - 1) // self.world_size

    def __len__(self):
        return self.num_samples

    def set_epoch(self, epoch):
        self.epoch = int(epoch)

    def __iter__(self):
        generator = torch.Generator()
        generator.manual_seed(self.seed + self.epoch)
        pools = {}
        for source, indices in self.by_source.items():
            values = list(indices)
            if self.shuffle:
                order = torch.randperm(len(values), generator=generator).tolist()
                values = [values[i] for i in order]
            pools[source] = values

        cursors = {source: 0 for source in self.sources}
        active = set(self.sources)
        order = []
        while active:
            made_progress = False
            for source in self.source_cycle:
                if source not in active:
                    continue
                cursor = cursors[source]
                values = pools[source]
                if cursor >= len(values):
                    active.remove(source)
                    continue
                order.append(values[cursor])
                cursors[source] = cursor + 1
                made_progress = True
            if not made_progress:
                break

        total_size = self.num_samples * self.world_size
        if order and len(order) < total_size:
            repeats = (total_size + len(order) - 1) // len(order)
            order = (order * repeats)[:total_size]
        else:
            order = order[:total_size]
        return iter(order[self.rank : total_size : self.world_size])


def make_loader(dataset, batch_size, num_workers, device, distributed, shuffle, drop_last=False):
    sampler = DistributedSampler(dataset, shuffle=shuffle) if distributed else None
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        shuffle=shuffle and sampler is None,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
        drop_last=drop_last,
    )
    return loader, sampler


def make_train_loader(dataset, batch_size, num_workers, device, distributed, shuffle, cfg, seed=0, drop_last=False):
    training_cfg = cfg.get("training", {}) if isinstance(cfg, dict) else {}
    balanced = bool(training_cfg.get("balanced_dataset_sampling", False))
    if not balanced:
        return make_loader(dataset, batch_size, num_workers, device, distributed, shuffle, drop_last)

    rank = dist.get_rank() if distributed_is_initialized() else 0
    world_size = dist.get_world_size() if distributed_is_initialized() else 1
    sampler = SourceBalancedSampler(
        dataset,
        shuffle=shuffle,
        seed=seed,
        rank=rank,
        world_size=world_size,
        source_weights=training_cfg.get("source_sampling_weights", None),
    )
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
        drop_last=drop_last,
    )
    return loader, sampler


def build_dataset(args, cfg):
    dataset_cfg = cfg.get("dataset", {})
    lowlight_cfg = dict(cfg.get("lowlight_aug", {}))
    lowlight_enabled = bool(lowlight_cfg.pop("enabled", False))
    lowlight_sources = lowlight_cfg.pop("apply_sources", None)
    image_size = args.image_size or dataset_cfg.get("image_size", (512, 512))
    return DSECEventDataset(
        root_path=resolve_data_root(args, cfg),
        image_size=image_size,
        pair_stride=args.pair_stride
        if args.pair_stride is not None
        else dataset_cfg.get("pair_stride", 1),
        max_scenes=args.max_scenes
        if args.max_scenes is not None
        else dataset_cfg.get("max_scenes", None),
        max_pairs_per_scene=args.max_pairs_per_scene
        if args.max_pairs_per_scene is not None
        else dataset_cfg.get("max_pairs_per_scene", None),
        exclude_scenes=dataset_cfg.get("exclude_scenes", None),
        expected_event_channels=int(cfg_get(cfg, "event", "voxel_channels", 8)),
        lowlight_enabled=lowlight_enabled,
        lowlight_sources=lowlight_sources,
        lowlight_kwargs=lowlight_cfg,
        pair_sampling=args.pair_sampling
        if args.pair_sampling is not None
        else dataset_cfg.get("pair_sampling", "fixed"),
        min_pair_distance=args.min_pair_distance
        if args.min_pair_distance is not None
        else dataset_cfg.get("min_pair_distance", None),
        max_pair_distance=args.max_pair_distance
        if args.max_pair_distance is not None
        else dataset_cfg.get("max_pair_distance", None),
        precomputed_pairs=args.precomputed_pairs
        if args.precomputed_pairs is not None
        else dataset_cfg.get("precomputed_pairs", None),
        allow_reverse_pairs=args.allow_reverse_pairs
        if args.allow_reverse_pairs is not None
        else dataset_cfg.get("allow_reverse_pairs", False),
        pose_output_convention=dataset_cfg.get("pose_output_convention", "native"),
    )


def load_backbone(args, cfg=None, checkpoint_key="checkpoint", model_name_or_path=None):
    if args.use_tiny_model:
        from utils.smoke_event_dark3r import TinyDark3R

        return TinyDark3R()
    from mast3r.model import AsymmetricMASt3R

    source = model_name_or_path or resolve_model_name_or_path(
        args, {} if cfg is None else cfg, checkpoint_key
    )
    model = AsymmetricMASt3R.from_pretrained(
        source,
        allow_unsafe_checkpoint_load=args.allow_unsafe_checkpoint_load,
    )
    lora_checkpoint = resolve_backbone_lora_checkpoint(args, {} if cfg is None else cfg)
    if lora_checkpoint:
        try:
            from model.lora_model import LoRAMast3r
        except ModuleNotFoundError as exc:
            raise RuntimeError(
                "Dark3R LoRA backbone initialization requires loralib in this "
                "environment. Install loralib or omit --backbone-lora-checkpoint."
            ) from exc
        model.__class__ = LoRAMast3r
        model.load_pretrained_lora(lora_checkpoint)
    return model


def pred_feature(pred):
    return pred.get("feature", pred.get("desc", None))


def add_loss(total, loss):
    return loss if total is None else total + loss


def loss_was_skipped(loss_dict, prefix):
    skipped_key = f"skipped {prefix}"
    return any(key == skipped_key or key.startswith(f"{skipped_key}/") for key in loss_dict)


def add_loss_component(total, active_weight, loss, weight, loss_dict, prefix):
    if loss_was_skipped(loss_dict, prefix):
        return total, active_weight
    return add_loss(total, loss), active_weight + float(weight)


def _batch_values(value):
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    if isinstance(value, (list, tuple)):
        return list(value)
    return [value]


def _source_key(value):
    return str(value).strip().lower()


def resolve_source_weight(base_weight, loss_cfg, batch, loss_dict, prefix):
    source_weights = loss_cfg.get(f"{prefix}_source_weights", None)
    if not source_weights:
        return base_weight

    sources = [_source_key(value) for value in _batch_values(batch.get("dataset_source", None))]
    sources = [source for source in sources if source]
    if not sources:
        loss_dict[f"{prefix}_source_weight_scale"] = 1.0
        loss_dict[f"{prefix}_source_weight_note"] = "missing dataset_source"
        return base_weight

    default_scale = float(source_weights.get("default", 1.0))
    scales = [float(source_weights.get(source, default_scale)) for source in sources]
    scale = sum(scales) / max(len(scales), 1)
    unique_sources = sorted(set(sources))
    loss_dict[f"{prefix}_source"] = ",".join(unique_sources)
    loss_dict[f"{prefix}_source_weight_scale"] = float(scale)
    if len(unique_sources) > 1:
        loss_dict[f"{prefix}_source_weight_note"] = "mixed source batch uses mean scale"
    return float(base_weight) * float(scale)


def finalize_total_loss(total, active_weight, loss_cfg, loss_dict):
    if total is None:
        return total
    loss_dict["loss_before_normalize"] = float(total.detach().cpu())
    loss_dict["loss_active_weight"] = float(active_weight)
    if bool(loss_cfg.get("normalize_by_active_weight", True)) and active_weight > 0:
        target_weight = float(loss_cfg.get("loss_normalize_target_weight", 1.0))
        min_weight = float(loss_cfg.get("loss_normalize_min_weight", 1e-6))
        factor = target_weight / max(float(active_weight), min_weight)
        total = total * factor
        loss_dict["loss_normalize_factor"] = float(factor)
        loss_dict["loss_after_normalize"] = float(total.detach().cpu())
    return total


def initialize_optimizer_lrs(optimizer):
    for group in optimizer.param_groups:
        group["base_lr"] = float(group["lr"])


def optimizer_lrs(optimizer):
    return {
        group.get("name", f"group{idx}"): float(group["lr"])
        for idx, group in enumerate(optimizer.param_groups)
    }


def _batch_sources(batch):
    sources = [_source_key(value) for value in _batch_values(batch.get("dataset_source", None))]
    return [source for source in sources if source]


def event_debug_log_fields(student_out, batch):
    pred = student_out[0] if isinstance(student_out, (list, tuple)) else student_out
    fields = {}
    stat_summary = pred.get("event_stat_summary", {}) if isinstance(pred, dict) else {}
    gate_summary = pred.get("event_gate_summary", {}) if isinstance(pred, dict) else {}
    for key, value in stat_summary.items():
        fields[key] = value
    for key in ("mean", "std", "min", "max"):
        if key in gate_summary:
            fields[f"event_gate_{key}"] = gate_summary[key]
    for key in (
        "event_scale",
        "event_gate_sat_low",
        "event_gate_sat_high",
        "event_channel_dev",
        "event_residual_ratio",
        "event_gate_logit_abs",
        "event_low_event_gate_mean",
        "event_high_event_gate_mean",
        "visibility_mean",
        "visibility_std",
        "visibility_min",
        "visibility_max",
        "event_visibility_mean",
        "event_event_allow_mean",
        "event_structural_reliability_mean",
        "event_structural_allow_mean",
        "event_temporal_enabled",
        "event_temporal_visibility_mean",
        "event_temporal_outer_weight_mean",
        "event_temporal_outer_weight_min",
        "event_temporal_outer_weight_max",
        "event_temporal_center_bins",
        "event_low_visibility_gate_mean",
        "event_high_visibility_gate_mean",
        "event_low_structural_gate_mean",
        "event_high_structural_gate_mean",
        "event_gate_visibility_corr",
        "event_gate_cap",
        "event_gate_mean_cap",
        "event_token_norm_before_film",
        "event_token_norm_after_film",
        "rgb_token_norm",
    ):
        value = pred.get(key, None) if isinstance(pred, dict) else None
        if value is not None:
            fields[key] = value

    sources = sorted(set(_batch_sources(batch)))
    if len(sources) == 1:
        source_prefix = sources[0].upper()
        if "event_density_mean" in fields:
            fields[f"{source_prefix}/event_density_mean"] = fields["event_density_mean"]
        if "event_gate_mean" in fields:
            fields[f"{source_prefix}/event_gate_mean"] = fields["event_gate_mean"]
    return fields


def apply_stage_lr_warmup(optimizer, stage_cfg, stage_step):
    warmup_steps = int(stage_cfg.get("warmup_steps", 0))
    start_factor = float(stage_cfg.get("warmup_start_factor", 1.0))
    start_factor = max(0.0, min(start_factor, 1.0))
    if warmup_steps <= 0:
        factor = 1.0
    else:
        progress = min(max(int(stage_step), 0), warmup_steps) / max(warmup_steps, 1)
        factor = start_factor + (1.0 - start_factor) * progress
    for group in optimizer.param_groups:
        base_lr = float(group.get("base_lr", group["lr"]))
        group["base_lr"] = base_lr
        group["lr"] = base_lr * factor
    return factor


def flip_pair_stack(value):
    if not torch.is_tensor(value):
        return value
    if value.ndim >= 5 and value.shape[1] == 2:
        return value.flip(1)
    return value


def infer_batch_size(batch):
    if not isinstance(batch, dict):
        return None
    for key in ("images", "images_dark", "event_voxel"):
        value = batch.get(key, None)
        if torch.is_tensor(value) and value.ndim > 0:
            return int(value.shape[0])
    for value in batch.values():
        if torch.is_tensor(value) and value.ndim > 0:
            return int(value.shape[0])
    return None


def concat_batch_values(value1, value2, batch_size):
    if torch.is_tensor(value1) and torch.is_tensor(value2):
        if value1.ndim > 0 and value2.ndim > 0:
            if value1.shape[0] == batch_size and value2.shape[0] == batch_size:
                return torch.cat([value1, value2], dim=0)
        return value1
    if isinstance(value1, list) and isinstance(value2, list):
        if len(value1) == batch_size and len(value2) == batch_size:
            return value1 + value2
        return value1
    if isinstance(value1, tuple) and isinstance(value2, tuple):
        if len(value1) == batch_size and len(value2) == batch_size:
            return value1 + value2
        return value1
    return value1


def concat_batches(batch1, batch2):
    batch_size = infer_batch_size(batch1)
    if batch_size is None:
        raise ValueError("Cannot infer batch size for symmetric pair batching")

    out = {}
    keys = list(batch1.keys()) + [key for key in batch2.keys() if key not in batch1]
    for key in keys:
        if key in batch1 and key in batch2:
            out[key] = concat_batch_values(batch1[key], batch2[key], batch_size)
        elif key in batch1:
            out[key] = batch1[key]
        else:
            out[key] = batch2[key]
    return out


def slice_batch(batch, start, end, total_size):
    out = {}
    for key, value in batch.items():
        if torch.is_tensor(value) and value.ndim > 0 and value.shape[0] == total_size:
            out[key] = value[start:end]
        elif isinstance(value, list) and len(value) == total_size:
            out[key] = value[start:end]
        elif isinstance(value, tuple) and len(value) == total_size:
            out[key] = value[start:end]
        else:
            out[key] = value
    return out


def slice_pred(pred, start, end, total_size):
    if pred is None:
        return None
    out = {}
    for key, value in pred.items():
        if torch.is_tensor(value) and value.ndim > 0 and value.shape[0] == total_size:
            out[key] = value[start:end]
        else:
            out[key] = value
    return out


def slice_outputs(outputs, start, end, total_size):
    return tuple(slice_pred(pred, start, end, total_size) for pred in outputs)


def reverse_pair_batch(batch):
    out = {}
    for key, value in batch.items():
        if key in {
            "images",
            "images_dark",
            "images_photo",
            "images_photo_dark",
            "event_voxel",
        }:
            out[key] = flip_pair_stack(value)
        else:
            out[key] = value

    for key1, key2 in (
        ("image1_path", "image2_path"),
        ("event1_path", "event2_path"),
        ("depth1_path", "depth2_path"),
        ("depth_pseudo1_path", "depth_pseudo2_path"),
        ("depth_pseudo_conf1_path", "depth_pseudo_conf2_path"),
        ("idx1", "idx2"),
        ("depth_gt1", "depth_gt2"),
        ("depth_pseudo1", "depth_pseudo2"),
        ("depth_pseudo_conf1", "depth_pseudo_conf2"),
    ):
        if key1 in batch and key2 in batch:
            out[key1] = batch[key2]
            out[key2] = batch[key1]

    if "depth_gt2" in batch:
        out["depth_gt"] = batch["depth_gt2"]
        out["depth_valid"] = torch.isfinite(batch["depth_gt2"]) & (batch["depth_gt2"] > 0)

    if "depth_pseudo2" in batch:
        out["depth_pseudo"] = batch["depth_pseudo2"]
        out["depth_pseudo_valid"] = torch.isfinite(batch["depth_pseudo2"]) & (
            batch["depth_pseudo2"] > 0
        )
    if "depth_pseudo_conf2" in batch:
        out["depth_pseudo_conf"] = batch["depth_pseudo_conf2"]

    if "relative_pose" in batch:
        out["relative_pose"] = torch.linalg.inv(batch["relative_pose"])
    if "relative_pose_gt" in batch:
        out["relative_pose_gt"] = torch.linalg.inv(batch["relative_pose_gt"])

    sample_id = batch.get("sample_id", None)
    if isinstance(sample_id, list):
        out["sample_id"] = [f"{item}:rev" for item in sample_id]
    elif isinstance(sample_id, tuple):
        out["sample_id"] = tuple(f"{item}:rev" for item in sample_id)
    elif isinstance(sample_id, str):
        out["sample_id"] = f"{sample_id}:rev"
    return out


def prefix_loss_dict(loss_dict, prefix):
    out = {}
    for key, value in loss_dict.items():
        if isinstance(key, str) and key.startswith("skipped "):
            out[f"skipped {prefix}/{key[len('skipped '):]}"] = value
        else:
            out[f"{prefix}/{key}"] = value
    return out


def merge_step_loss_dict(*items):
    merged = {}
    for item in items:
        merged.update(item)
    return merged


def resolve_amp_config(cfg, args, device):
    training_cfg = cfg.get("training", {})
    amp_enabled = (
        bool(args.amp)
        if args.amp is not None
        else bool(training_cfg.get("amp", False))
    )
    if device.type != "cuda":
        amp_enabled = False

    dtype_name = str(
        args.amp_dtype
        or training_cfg.get("amp_dtype", cfg.get("precision", "fp16"))
    ).lower()
    if dtype_name not in {"fp16", "bf16"}:
        raise ValueError(f"Unsupported AMP dtype: {dtype_name}")
    amp_dtype = torch.float16 if dtype_name == "fp16" else torch.bfloat16
    return amp_enabled, dtype_name, amp_dtype


def resolve_gradient_checkpointing(cfg, args):
    training_cfg = cfg.get("training", {})
    if args.gradient_checkpointing is not None:
        return bool(args.gradient_checkpointing)
    return bool(training_cfg.get("gradient_checkpointing", False))


def autocast_context(device_type, amp_enabled, amp_dtype):
    return torch.amp.autocast(
        device_type=device_type,
        dtype=amp_dtype,
        enabled=bool(amp_enabled),
    )


def make_grad_scaler(device, amp_enabled, amp_dtype):
    if device.type != "cuda" or not amp_enabled or amp_dtype != torch.float16:
        return None
    return torch.amp.GradScaler(device.type, enabled=True)


def set_student_step(student, step):
    model = student.module if isinstance(student, DDP) else student
    if hasattr(model, "set_train_step"):
        model.set_train_step(step)
    else:
        setattr(model, "current_step", None if step is None else int(step))


def configure_gradient_checkpointing(student, enabled):
    model = unwrap_model(student)
    if hasattr(model, "dark3r"):
        setattr(model.dark3r, "gradient_checkpointing", bool(enabled))
    if hasattr(model, "event_encoder"):
        setattr(model.event_encoder, "gradient_checkpointing", bool(enabled))


def descriptor_only_teacher_forward(teacher, batch):
    view1, view2 = make_view_pair_from_batch(batch, prefer_dark=False)
    image1, image2 = view1["img"], view2["img"]
    shape1, shape2 = view1["true_shape"], view2["true_shape"]
    feat1, feat2, pos1, pos2 = teacher._encode_image_pairs(
        image1, image2, shape1, shape2
    )
    dec1, dec2 = teacher._decoder(feat1, pos1, feat2, pos2)
    return (
        decode_local_descriptors(teacher, 1, dec1, shape1),
        decode_local_descriptors(teacher, 2, dec2, shape2),
    )


def should_use_symmetric(cfg, args=None):
    if args is not None and getattr(args, "symmetric_pairs", None) is not None:
        return bool(args.symmetric_pairs)
    training_cfg = cfg.get("training", {})
    return bool(training_cfg.get("symmetric_pairs", False))


def compute_pair_loss(
    student,
    teacher,
    batch,
    loss_cfg,
    teacher_no_grad=True,
    amp_enabled=False,
    amp_dtype=torch.float16,
    device_type="cuda",
    step=None,
):
    set_student_step(student, step)
    with autocast_context(device_type, amp_enabled, amp_dtype):
        needs_teacher = (
            float(loss_cfg.get("feat_distill", 0.0)) > 0
            or float(loss_cfg.get("structure_consistency", 0.0)) > 0
            or float(loss_cfg.get("descriptor_distill", 0.0)) > 0
        )
        if not needs_teacher:
            teacher_out = (None, None)
        elif teacher_no_grad:
            with torch.no_grad():
                if float(loss_cfg.get("descriptor_distill", 0.0)) > 0 and not (
                    float(loss_cfg.get("feat_distill", 0.0)) > 0
                    or float(loss_cfg.get("structure_consistency", 0.0)) > 0
                ):
                    teacher_out = descriptor_only_teacher_forward(teacher, batch)
                else:
                    view1, view2 = make_view_pair_from_batch(batch, prefer_dark=False)
                    teacher_out = teacher(view1, view2)
        else:
            if float(loss_cfg.get("descriptor_distill", 0.0)) > 0 and not (
                float(loss_cfg.get("feat_distill", 0.0)) > 0
                or float(loss_cfg.get("structure_consistency", 0.0)) > 0
            ):
                teacher_out = descriptor_only_teacher_forward(teacher, batch)
            else:
                view1, view2 = make_view_pair_from_batch(batch, prefer_dark=False)
                teacher_out = teacher(view1, view2)
        student_out = student(batch)
    loss, loss_dict = compute_losses(student_out, teacher_out, batch, loss_cfg)
    return loss, loss_dict, student_out


def compute_symmetric_pair_loss(
    student,
    teacher,
    batch,
    loss_cfg,
    teacher_no_grad=True,
    amp_enabled=False,
    amp_dtype=torch.float16,
    device_type="cuda",
    step=None,
):
    set_student_step(student, step)
    batch_size = infer_batch_size(batch)
    if batch_size is None:
        raise ValueError("Cannot infer batch size for symmetric pair loss")

    reverse_batch = reverse_pair_batch(batch)
    combined_batch = concat_batches(batch, reverse_batch)
    total_size = batch_size * 2

    with autocast_context(device_type, amp_enabled, amp_dtype):
        needs_teacher = (
            float(loss_cfg.get("feat_distill", 0.0)) > 0
            or float(loss_cfg.get("structure_consistency", 0.0)) > 0
            or float(loss_cfg.get("descriptor_distill", 0.0)) > 0
        )
        if not needs_teacher:
            teacher_out = (None, None)
        elif teacher_no_grad:
            with torch.no_grad():
                if float(loss_cfg.get("descriptor_distill", 0.0)) > 0 and not (
                    float(loss_cfg.get("feat_distill", 0.0)) > 0
                    or float(loss_cfg.get("structure_consistency", 0.0)) > 0
                ):
                    teacher_out = descriptor_only_teacher_forward(
                        teacher, combined_batch
                    )
                else:
                    view1, view2 = make_view_pair_from_batch(
                        combined_batch, prefer_dark=False
                    )
                    teacher_out = teacher(view1, view2)
        else:
            if float(loss_cfg.get("descriptor_distill", 0.0)) > 0 and not (
                float(loss_cfg.get("feat_distill", 0.0)) > 0
                or float(loss_cfg.get("structure_consistency", 0.0)) > 0
            ):
                teacher_out = descriptor_only_teacher_forward(teacher, combined_batch)
            else:
                view1, view2 = make_view_pair_from_batch(combined_batch, prefer_dark=False)
                teacher_out = teacher(view1, view2)
        student_out = student(combined_batch)

    forward_batch = slice_batch(combined_batch, 0, batch_size, total_size)
    reverse_batch = slice_batch(combined_batch, batch_size, total_size, total_size)
    forward_student_out = slice_outputs(student_out, 0, batch_size, total_size)
    reverse_student_out = slice_outputs(student_out, batch_size, total_size, total_size)
    forward_teacher_out = slice_outputs(teacher_out, 0, batch_size, total_size)
    reverse_teacher_out = slice_outputs(teacher_out, batch_size, total_size, total_size)

    loss, loss_dict = compute_losses(
        forward_student_out,
        forward_teacher_out,
        forward_batch,
        loss_cfg,
    )
    reverse_loss, reverse_loss_dict = compute_losses(
        reverse_student_out,
        reverse_teacher_out,
        reverse_batch,
        loss_cfg,
    )
    merged_loss_dict = merge_step_loss_dict(
        prefix_loss_dict(loss_dict, "forward"),
        prefix_loss_dict(reverse_loss_dict, "reverse"),
    )
    merged_loss_dict["symmetric_loss_forward"] = float(loss.detach().cpu())
    merged_loss_dict["symmetric_loss_reverse"] = float(reverse_loss.detach().cpu())
    return 0.5 * (loss + reverse_loss), merged_loss_dict, forward_student_out


def compute_training_loss(
    student,
    teacher,
    batch,
    loss_cfg,
    symmetric=False,
    amp_enabled=False,
    amp_dtype=torch.float16,
    device_type="cuda",
    step=None,
):
    if symmetric:
        return compute_symmetric_pair_loss(
            student,
            teacher,
            batch,
            loss_cfg,
            teacher_no_grad=True,
            amp_enabled=amp_enabled,
            amp_dtype=amp_dtype,
            device_type=device_type,
            step=step,
        )
    return compute_pair_loss(
        student,
        teacher,
        batch,
        loss_cfg,
        teacher_no_grad=True,
        amp_enabled=amp_enabled,
        amp_dtype=amp_dtype,
        device_type=device_type,
        step=step,
    )


def compute_eval_loss(
    student,
    teacher,
    batch,
    loss_cfg,
    symmetric=False,
    amp_enabled=False,
    amp_dtype=torch.float16,
    device_type="cuda",
):
    if symmetric:
        return compute_symmetric_pair_loss(
            student,
            teacher,
            batch,
            loss_cfg,
            teacher_no_grad=False,
            amp_enabled=amp_enabled,
            amp_dtype=amp_dtype,
            device_type=device_type,
        )
    return compute_pair_loss(
        student,
        teacher,
        batch,
        loss_cfg,
        teacher_no_grad=False,
        amp_enabled=amp_enabled,
        amp_dtype=amp_dtype,
        device_type=device_type,
    )


def validate_loss_config(cfg, dataset, student):
    loss_cfg = cfg.get("loss", {})
    data_root = getattr(dataset, "root_paths_text", str(getattr(dataset, "root_path", "")))
    unsupported = {
        "dark3r_original": (
            "dark3r_original loss is not implemented in train_event_dark3r.py; "
            "use feat_distill/structure_consistency or implement this term first"
        ),
        "pose_cycle": (
            "pose_cycle requires predicted pose sequences, which EventDark3R/MASt3R "
            "does not currently output"
        ),
    }
    for key, message in unsupported.items():
        if float(loss_cfg.get(key, 0.0)) > 0:
            raise RuntimeError(f"Invalid loss config: {key} > 0. {message}.")
    if float(loss_cfg.get("pose_gt", 0.0)) > 0 and not getattr(
        student, "pose_head_enabled", False
    ):
        raise RuntimeError(
            "Invalid loss config: pose_gt > 0 requires pose_head.enabled=true."
        )

    def scene_source(scene_dir):
        if hasattr(dataset, "_scene_source"):
            try:
                return str(dataset._scene_source(scene_dir)).strip().lower()
            except Exception:
                pass
        return "unknown"

    def source_weight_scale(prefix, source):
        source_weights = loss_cfg.get(f"{prefix}_source_weights", {}) or {}
        default_scale = float(source_weights.get("default", 1.0))
        return float(source_weights.get(source, default_scale))

    def has_positive_scene_weight(prefix):
        scene_frames = getattr(dataset, "scene_frames", {}) or {}
        if not scene_frames:
            return True
        return any(source_weight_scale(prefix, scene_source(scene_dir)) > 0 for scene_dir in scene_frames)

    if (
        float(loss_cfg.get("depth_gt", 0.0)) > 0
        and not getattr(dataset, "has_depth_gt", False)
        and has_positive_scene_weight("depth_gt")
    ):
        raise RuntimeError(
            "Invalid loss config: depth_gt > 0 but dataset has no depth_gt/ tensors. "
            f"data_root={data_root}"
        )
    if float(loss_cfg.get("depth_gt", 0.0)) > 0 and not getattr(dataset, "has_all_depth_gt", True):
        scene_frames = getattr(dataset, "scene_frames", {})
        source_weights = loss_cfg.get("depth_gt_source_weights", {}) or {}
        default_scale = float(source_weights.get("default", 1.0))
        missing_depth_scenes = []
        uncovered_sources = set()
        for scene_dir, frames in scene_frames.items():
            if frames.get("depths"):
                continue
            missing_depth_scenes.append(str(scene_dir))
            scene_source_value = scene_source(scene_dir)
            scale = float(source_weights.get(scene_source_value, default_scale))
            if scale > 0:
                uncovered_sources.add(scene_source_value)
        if uncovered_sources:
            raise RuntimeError(
                "Invalid loss config: depth_gt > 0 but at least one discovered scene has no "
                "depth_gt/ tensors and its source still has nonzero depth weight. "
                f"uncovered_sources={sorted(uncovered_sources)} data_root={data_root}"
            )
    if (
        float(loss_cfg.get("pseudo_depth", 0.0)) > 0
        and not getattr(dataset, "has_pseudo_depth", False)
        and has_positive_scene_weight("pseudo_depth")
    ):
        raise RuntimeError(
            "Invalid loss config: pseudo_depth > 0 but dataset has no depth_pseudo/ tensors. "
            f"data_root={data_root}"
        )
    if float(loss_cfg.get("pseudo_depth", 0.0)) > 0 and not getattr(
        dataset, "has_all_pseudo_depth", True
    ):
        scene_frames = getattr(dataset, "scene_frames", {})
        source_weights = loss_cfg.get("pseudo_depth_source_weights", {}) or {}
        default_scale = float(source_weights.get("default", 1.0))
        uncovered_sources = set()
        for scene_dir, frames in scene_frames.items():
            if frames.get("pseudo_depths"):
                continue
            scene_source_value = scene_source(scene_dir)
            scale = float(source_weights.get(scene_source_value, default_scale))
            if scale > 0:
                uncovered_sources.add(scene_source_value)
        if uncovered_sources:
            raise RuntimeError(
                "Invalid loss config: pseudo_depth > 0 but at least one discovered scene has "
                "no depth_pseudo/ tensors and its source still has nonzero pseudo-depth weight. "
                f"uncovered_sources={sorted(uncovered_sources)} data_root={data_root}"
            )
    if (
        float(loss_cfg.get("event_photo", 0.0)) > 0
        and not getattr(dataset, "has_relative_pose", False)
        and has_positive_scene_weight("event_photo")
    ):
        raise RuntimeError(
            "Invalid loss config: event_photo > 0 but dataset has no relative poses "
            "from scene metadata (poses_selected.npz or transforms.json). "
            f"data_root={data_root}"
        )
    if float(loss_cfg.get("event_photo", 0.0)) > 0 and not getattr(
        dataset, "has_all_relative_pose", True
    ):
        source_weights = loss_cfg.get("event_photo_source_weights", {}) or {}
        default_scale = float(source_weights.get("default", 1.0))
        scene_frames = getattr(dataset, "scene_frames", {})
        scene_poses = getattr(dataset, "scene_poses", {})
        uncovered_sources = set()
        for scene_dir in scene_frames:
            if scene_poses.get(scene_dir, None) is not None:
                continue
            scene_source_value = scene_source(scene_dir)
            scale = float(source_weights.get(scene_source_value, default_scale))
            if scale > 0:
                uncovered_sources.add(scene_source_value)
        if uncovered_sources:
            raise RuntimeError(
                "Invalid loss config: event_photo > 0 but at least one discovered scene has no "
                "relative poses and its source still has nonzero event_photo weight. "
                f"uncovered_sources={sorted(uncovered_sources)} data_root={data_root}"
            )

    if float(loss_cfg.get("pose_gt", 0.0)) > 0:
        if not getattr(dataset, "has_relative_pose", False):
            raise RuntimeError(
                "Invalid loss config: pose_gt > 0 but the dataset has no relative poses. "
                f"data_root={data_root}"
            )
        if not getattr(dataset, "has_all_relative_pose", True):
            raise RuntimeError(
                "Invalid loss config: pose_gt > 0 but at least one discovered scene has "
                f"no relative poses. data_root={data_root}"
            )


def validate_pose_head_only_trainability(student):
    if not getattr(student, "pose_head_only", False):
        return
    trainable = [name for name, param in student.named_parameters() if param.requires_grad]
    invalid = [name for name in trainable if not name.startswith("pose_head.")]
    if invalid:
        raise RuntimeError(
            "pose-head-only training exposed non-pose parameters: " + ", ".join(invalid[:20])
        )
    if not trainable:
        raise RuntimeError("pose-head-only training has no trainable pose_head parameters")


def is_descriptor_head_parameter(name):
    return (
        name.startswith("dark3r.downstream_head1.head_local_features.")
        or name.startswith("dark3r.downstream_head2.head_local_features.")
    )


def validate_descriptor_head_only_trainability(student):
    if not getattr(student, "descriptor_head_only", False):
        return
    trainable = [name for name, param in student.named_parameters() if param.requires_grad]
    invalid = [name for name in trainable if not is_descriptor_head_parameter(name)]
    if invalid:
        raise RuntimeError(
            "descriptor-head-only training exposed non-descriptor parameters: "
            + ", ".join(invalid[:20])
        )
    if not trainable:
        raise RuntimeError(
            "descriptor-head-only training has no trainable head_local_features parameters"
        )


def is_geometry_head_parameter(name):
    return name.startswith(
        ("dark3r.downstream_head1.", "dark3r.downstream_head2.")
    )


def validate_geometry_head_only_trainability(student):
    if not getattr(student, "geometry_head_only", False):
        return
    trainable = [name for name, param in student.named_parameters() if param.requires_grad]
    invalid = [name for name in trainable if not is_geometry_head_parameter(name)]
    if invalid:
        raise RuntimeError(
            "geometry-head-only training exposed non-head parameters: "
            + ", ".join(invalid[:20])
        )
    if not trainable:
        raise RuntimeError("geometry-head-only training has no trainable parameters")


def fail_on_configured_skips(loss_dict, loss_cfg, where):
    message = configured_skip_message(loss_dict, loss_cfg, where)
    if message:
        raise RuntimeError(message)


def configured_skip_message(loss_dict, loss_cfg, where):
    nonfatal_skips = {
        ("event_photo", "no valid warped pixels"),
    }
    required_prefixes = []
    if float(loss_cfg.get("event_depth_edge", 0.0)) > 0:
        required_prefixes.extend(["event_depth_edge_view1", "event_depth_edge_view2"])
    if float(loss_cfg.get("event_photo", 0.0)) > 0:
        required_prefixes.append("event_photo")
    if float(loss_cfg.get("depth_gt", 0.0)) > 0:
        required_prefixes.append("depth_gt")
    if float(loss_cfg.get("pseudo_depth", 0.0)) > 0:
        required_prefixes.append("pseudo_depth")
    if float(loss_cfg.get("depth_gradient_gt", 0.0)) > 0:
        required_prefixes.append("depth_gradient_gt")
    if float(loss_cfg.get("pose_gt", 0.0)) > 0:
        required_prefixes.append("pose_gt")
    if float(loss_cfg.get("descriptor_distill", 0.0)) > 0:
        required_prefixes.extend(
            ["descriptor_distill_view1", "descriptor_distill_view2"]
        )

    for skipped_key, reason in loss_dict.items():
        if not isinstance(skipped_key, str) or not skipped_key.startswith("skipped "):
            continue
        skipped_name = skipped_key[len("skipped ") :]
        base_skipped_name = skipped_name.split("/")[-1]
        reason_text = str(reason)
        if (base_skipped_name, reason_text) in nonfatal_skips or (skipped_name, reason_text) in nonfatal_skips:
            continue
        if any(
            skipped_name.startswith(prefix) or base_skipped_name.startswith(prefix)
            for prefix in required_prefixes
        ):
            return (
                f"Configured loss was skipped at {where}: {skipped_key}={reason}. "
                "Disable this loss or fix the dataset/model fields before training."
            )
    return None


def sync_configured_skips(loss_dict, loss_cfg, where, device):
    message = configured_skip_message(loss_dict, loss_cfg, where)
    message = sync_error_message(message, device)
    if message:
        raise RuntimeError(message)


def resolve_stage_schedule(cfg, args):
    training_cfg = cfg.get("training", {})
    base_stage = {
        "stage": args.stage if args.stage is not None else int(training_cfg.get("stage", 1)),
        "num_epochs": args.num_epochs
        if args.num_epochs is not None
        else int(training_cfg.get("num_epochs", 1)),
        "partial_finetune_blocks": args.partial_finetune_blocks
        if args.partial_finetune_blocks is not None
        else int(cfg_get(cfg, "model", "partial_finetune_blocks", 4)),
        "lr_event_encoder": float(training_cfg.get("lr_event_encoder", 1e-4)),
        "lr_fusion_adapter": float(training_cfg.get("lr_fusion_adapter", 1e-4)),
        "lr_dark3r": float(training_cfg.get("lr_dark3r", 1e-5)),
        "lr_head": float(training_cfg.get("lr_head", 1e-4)),
        "lr_pose_head": float(training_cfg.get("lr_pose_head", 1e-4)),
        "warmup_steps": int(training_cfg.get("warmup_steps", 0)),
        "warmup_start_factor": float(training_cfg.get("warmup_start_factor", 1.0)),
    }
    if args.stage is not None:
        return [base_stage]

    stages = training_cfg.get("stages", None)
    if not stages:
        return [base_stage]

    resolved = []
    for stage_cfg in stages:
        merged = dict(base_stage)
        merged.update({k: v for k, v in stage_cfg.items() if v is not None})
        if args.num_epochs is not None:
            merged["num_epochs"] = args.num_epochs
        if args.partial_finetune_blocks is not None:
            merged["partial_finetune_blocks"] = args.partial_finetune_blocks
        merged["stage"] = int(merged["stage"])
        merged["num_epochs"] = int(merged.get("num_epochs", 1))
        merged["partial_finetune_blocks"] = int(
            merged.get("partial_finetune_blocks", base_stage["partial_finetune_blocks"])
        )
        for key in (
            "lr_event_encoder",
            "lr_fusion_adapter",
            "lr_dark3r",
            "lr_head",
            "lr_pose_head",
        ):
            merged[key] = float(merged[key])
        merged["warmup_steps"] = int(merged.get("warmup_steps", base_stage["warmup_steps"]))
        merged["warmup_start_factor"] = float(
            merged.get("warmup_start_factor", base_stage["warmup_start_factor"])
        )
        resolved.append(merged)
    return resolved


def unwrap_model(model):
    return model.module if isinstance(model, DDP) else model


def build_optimizer_for_stage(student, stage_cfg):
    model = unwrap_model(student)
    return build_event_optimizer(
        model,
        lr_event_encoder=stage_cfg["lr_event_encoder"],
        lr_fusion_adapter=stage_cfg["lr_fusion_adapter"],
        lr_dark3r=stage_cfg["lr_dark3r"],
        lr_head=stage_cfg["lr_head"],
        lr_pose_head=stage_cfg["lr_pose_head"],
    )


def add_event_aux_losses(total, active_weight, student_out, loss_dict):
    if not isinstance(student_out, (list, tuple)) or not student_out:
        return total, active_weight
    pred = student_out[0]
    if not isinstance(pred, dict):
        return total, active_weight
    aux_losses = pred.get("event_aux_losses", None)
    if not isinstance(aux_losses, dict) or "total" not in aux_losses:
        return total, active_weight

    aux_total = aux_losses.get("total", None)
    if not torch.is_tensor(aux_total):
        return total, active_weight
    for key, value in aux_losses.items():
        if torch.is_tensor(value) and value.numel() == 1:
            loss_dict[f"event_aux/{key}"] = float(value.detach().cpu())

    weight_sum = aux_losses.get("weight_sum", None)
    if torch.is_tensor(weight_sum):
        weight_value = float(weight_sum.detach().cpu())
    elif weight_sum is None:
        weight_value = 0.0
    else:
        weight_value = float(weight_sum)
    if weight_value <= 0:
        return total, active_weight
    return add_loss(total, aux_total), active_weight + weight_value


def compute_losses(student_out, teacher_out, batch, loss_cfg):
    loss_dict = {}
    total = None
    active_weight = 0.0

    descriptor_weight = float(loss_cfg.get("descriptor_distill", 0.0))
    if descriptor_weight > 0:
        effective_descriptor_weight = resolve_source_weight(
            descriptor_weight,
            loss_cfg,
            batch,
            loss_dict,
            "descriptor_distill",
        )
        if effective_descriptor_weight > 0:
            for view_idx, (student_pred, teacher_pred) in enumerate(
                zip(student_out, teacher_out), start=1
            ):
                prefix = f"descriptor_distill_view{view_idx}"
                component_weight = effective_descriptor_weight / 2.0
                loss, loss_dict = feature_distillation_loss(
                    student_pred.get("desc", None),
                    teacher_pred.get("desc", None),
                    weight=component_weight,
                    loss_dict=loss_dict,
                    prefix=prefix,
                )
                total, active_weight = add_loss_component(
                    total,
                    active_weight,
                    loss,
                    component_weight,
                    loss_dict,
                    prefix,
                )

    feat_weight = float(loss_cfg.get("feat_distill", 0.0))
    if feat_weight > 0:
        for view_idx, (student_pred, teacher_pred) in enumerate(
            zip(student_out, teacher_out), start=1
        ):
            prefix = f"feat_distill_view{view_idx}"
            component_weight = feat_weight / 2.0
            loss, loss_dict = feature_distillation_loss(
                pred_feature(student_pred),
                pred_feature(teacher_pred),
                weight=component_weight,
                loss_dict=loss_dict,
                prefix=prefix,
            )
            total, active_weight = add_loss_component(
                total, active_weight, loss, component_weight, loss_dict, prefix
            )

    struct_weight = float(loss_cfg.get("structure_consistency", 0.0))
    if struct_weight > 0:
        prefix = "structure_consistency"
        loss, loss_dict = structure_consistency_loss(
            student_out,
            teacher_out,
            weight=struct_weight,
            conf_thr=float(loss_cfg.get("structure_conf_thr", 1.0)),
            min_scale=float(loss_cfg.get("structure_min_scale", 1.0)),
            max_scale=float(loss_cfg.get("structure_max_scale", 1000.0)),
            max_point_norm=float(loss_cfg.get("structure_max_point_norm", 10000.0)),
            max_residual=float(loss_cfg.get("structure_max_residual", 50.0)),
            min_valid_fraction=float(loss_cfg.get("structure_min_valid_fraction", 0.01)),
            loss_dict=loss_dict,
            prefix=prefix,
        )
        total, active_weight = add_loss_component(
            total, active_weight, loss, struct_weight, loss_dict, prefix
        )

    edge_weight = float(loss_cfg.get("event_depth_edge", 0.0))
    if edge_weight > 0 and "event_voxel" in batch:
        effective_edge_weight = resolve_source_weight(
            edge_weight,
            loss_cfg,
            batch,
            loss_dict,
            "event_depth_edge",
        )
        if effective_edge_weight <= 0:
            edge_weight = 0.0
        else:
            edge_weight = effective_edge_weight
    if edge_weight > 0 and "event_voxel" in batch:
        event_voxel = batch["event_voxel"]
        event1 = event_voxel[:, 0] if event_voxel.ndim == 5 else event_voxel
        event2 = event_voxel[:, 1] if event_voxel.ndim == 5 and event_voxel.shape[1] > 1 else None
        prefix = "event_depth_edge_view1"
        component_weight = edge_weight / 2.0 if event2 is not None else edge_weight
        loss, loss_dict = event_depth_edge_loss(
            student_out[0],
            event1,
            weight=component_weight,
            loss_dict=loss_dict,
            prefix=prefix,
        )
        total, active_weight = add_loss_component(
            total, active_weight, loss, component_weight, loss_dict, prefix
        )
        if event2 is not None:
            prefix = "event_depth_edge_view2"
            component_weight = edge_weight / 2.0
            loss, loss_dict = event_depth_edge_loss(
                student_out[1],
                event2,
                weight=component_weight,
                loss_dict=loss_dict,
                prefix=prefix,
            )
            total, active_weight = add_loss_component(
                total, active_weight, loss, component_weight, loss_dict, prefix
            )

    photo_weight = float(loss_cfg.get("event_photo", 0.0))
    if photo_weight > 0:
        prefix = "event_photo"
        effective_photo_weight = resolve_source_weight(
            photo_weight,
            loss_cfg,
            batch,
            loss_dict,
            prefix,
        )
        if effective_photo_weight > 0:
            loss, loss_dict = event_photo_consistency_loss(
                batch,
                student_out[0],
                weight=effective_photo_weight,
                loss_dict=loss_dict,
                prefix=prefix,
            )
            total, active_weight = add_loss_component(
                total, active_weight, loss, effective_photo_weight, loss_dict, prefix
            )

    depth_weight = float(loss_cfg.get("depth_gt", 0.0))
    if depth_weight > 0:
        prefix = "depth_gt"
        effective_depth_weight = resolve_source_weight(
            depth_weight,
            loss_cfg,
            batch,
            loss_dict,
            prefix,
        )
        if effective_depth_weight > 0:
            loss, loss_dict = supervised_depth_gt_loss(
                student_out[0],
                batch,
                weight=effective_depth_weight,
                loss_dict=loss_dict,
                prefix=prefix,
            )
            total, active_weight = add_loss_component(
                total, active_weight, loss, effective_depth_weight, loss_dict, prefix
            )

    pseudo_depth_weight = float(loss_cfg.get("pseudo_depth", 0.0))
    if pseudo_depth_weight > 0:
        prefix = "pseudo_depth"
        effective_pseudo_depth_weight = resolve_source_weight(
            pseudo_depth_weight,
            loss_cfg,
            batch,
            loss_dict,
            prefix,
        )
        if effective_pseudo_depth_weight > 0:
            loss, loss_dict = pseudo_depth_distillation_loss(
                student_out[0],
                batch,
                weight=effective_pseudo_depth_weight,
                confidence_percentile=float(
                    loss_cfg.get("pseudo_depth_confidence_percentile", 40.0)
                ),
                huber_delta=float(loss_cfg.get("pseudo_depth_huber_delta", 0.1)),
                min_valid_fraction=float(
                    loss_cfg.get("pseudo_depth_min_valid_fraction", 0.05)
                ),
                metric_scale_weight=float(
                    loss_cfg.get("pseudo_depth_metric_scale_weight", 0.0)
                ),
                loss_dict=loss_dict,
                prefix=prefix,
            )
            total, active_weight = add_loss_component(
                total,
                active_weight,
                loss,
                effective_pseudo_depth_weight,
                loss_dict,
                prefix,
            )

    depth_grad_weight = float(loss_cfg.get("depth_gradient_gt", 0.0))
    if depth_grad_weight > 0:
        prefix = "depth_gradient_gt"
        effective_depth_grad_weight = resolve_source_weight(
            depth_grad_weight,
            loss_cfg,
            batch,
            loss_dict,
            prefix,
        )
        if effective_depth_grad_weight > 0:
            loss, loss_dict = depth_gradient_gt_loss(
                student_out[0],
                batch,
                weight=effective_depth_grad_weight,
                image_edge_weight=float(loss_cfg.get("depth_gradient_image_edge_weight", 1.0)),
                event_edge_weight=float(loss_cfg.get("depth_gradient_event_edge_weight", 0.5)),
                log_depth=bool(loss_cfg.get("depth_gradient_log_depth", True)),
                loss_dict=loss_dict,
                prefix=prefix,
            )
            total, active_weight = add_loss_component(
                total,
                active_weight,
                loss,
                effective_depth_grad_weight,
                loss_dict,
                prefix,
            )

    pose_weight = float(loss_cfg.get("pose_gt", 0.0))
    if pose_weight > 0:
        prefix = "pose_gt"
        effective_pose_weight = resolve_source_weight(
            pose_weight,
            loss_cfg,
            batch,
            loss_dict,
            prefix,
        )
        if effective_pose_weight > 0:
            loss, loss_dict = pose_gt_loss(
                student_out[0],
                batch,
                weight=effective_pose_weight,
                cycle_weight=float(loss_cfg.get("pose_cycle", 0.0)),
                rotation_weight=float(loss_cfg.get("pose_rotation_weight", 1.0)),
                rotation_chordal_weight=float(
                    loss_cfg.get("pose_rotation_chordal_weight", 0.0)
                ),
                translation_weight=float(loss_cfg.get("pose_translation_weight", 1.0)),
                min_translation_norm=float(
                    loss_cfg.get("pose_min_translation_norm", 1.0e-3)
                ),
                loss_dict=loss_dict,
            )
            total, active_weight = add_loss_component(
                total, active_weight, loss, effective_pose_weight, loss_dict, prefix
            )

    total, active_weight = add_event_aux_losses(total, active_weight, student_out, loss_dict)

    if total is None:
        anchor = student_out[0].get("pts3d", student_out[0].get("desc"))
        total = anchor.new_zeros(())
        loss_dict["skipped all_losses"] = "all configured loss weights are zero"
    else:
        total = finalize_total_loss(total, active_weight, loss_cfg, loss_dict)
    return total, loss_dict


def save_checkpoint(
    save_dir,
    student,
    optimizer,
    cfg,
    args,
    step,
    rank,
    name=None,
    extra=None,
    scaler=None,
):
    if rank == 0 and not args.no_save:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        model = student.module if isinstance(student, DDP) else student
        checkpoint_mode = str(
            cfg.get("training", {}).get("checkpoint_mode", "full")
        ).lower()
        if checkpoint_mode == "pose_head_only":
            if not getattr(model, "pose_head_enabled", False) or model.pose_head is None:
                raise RuntimeError(
                    "checkpoint_mode=pose_head_only requires pose_head.enabled=true"
                )
            payload = {
                "checkpoint_kind": "pose_head_only",
                "pose_head": {
                    key: value.detach().cpu()
                    for key, value in model.pose_head.state_dict().items()
                },
                "pose_head_config": dict(cfg.get("pose_head", {})),
                "base_checkpoint": (
                    None
                    if not getattr(args, "init_event_checkpoint", None)
                    else str(Path(args.init_event_checkpoint).resolve())
                ),
                "config": cfg,
                "step": step,
            }
        elif checkpoint_mode == "descriptor_head_only":
            if not getattr(model, "descriptor_head_only", False):
                raise RuntimeError(
                    "checkpoint_mode=descriptor_head_only requires "
                    "training.descriptor_head_only=true"
                )
            descriptor_state = {}
            for head_index in (1, 2):
                module = getattr(model.dark3r, f"downstream_head{head_index}")
                descriptor_head = module.head_local_features
                prefix = (
                    f"dark3r.downstream_head{head_index}.head_local_features."
                )
                descriptor_state.update(
                    {
                        prefix + key: value.detach().cpu()
                        for key, value in descriptor_head.state_dict().items()
                    }
                )
            if not descriptor_state:
                raise RuntimeError("No descriptor head state found for compact checkpoint")
            payload = {
                "checkpoint_kind": "descriptor_head_only",
                "descriptor_head": descriptor_state,
                "base_checkpoint": (
                    None
                    if not getattr(args, "init_event_checkpoint", None)
                    else str(Path(args.init_event_checkpoint).resolve())
                ),
                "config": cfg,
                "step": step,
            }
        elif checkpoint_mode == "geometry_head_only":
            if not getattr(model, "geometry_head_only", False):
                raise RuntimeError(
                    "checkpoint_mode=geometry_head_only requires "
                    "training.geometry_head_only=true"
                )
            geometry_state = {
                key: value.detach().cpu()
                for key, value in model.state_dict().items()
                if is_geometry_head_parameter(key)
            }
            if not geometry_state:
                raise RuntimeError("No geometry head state found for compact checkpoint")
            payload = {
                "checkpoint_kind": "geometry_head_only",
                "geometry_head": geometry_state,
                "base_checkpoint": (
                    None
                    if not getattr(args, "init_event_checkpoint", None)
                    else str(Path(args.init_event_checkpoint).resolve())
                ),
                "config": cfg,
                "step": step,
            }
        elif checkpoint_mode == "full":
            payload = {
                "model": model.state_dict(),
                "config": cfg,
                "args": vars(args),
                "step": step,
            }
            save_optimizer_state = bool(
                cfg.get("training", {}).get("save_optimizer_state", True)
            )
            if save_optimizer_state:
                payload["optimizer"] = optimizer.state_dict()
                if scaler is not None:
                    payload["scaler"] = scaler.state_dict()
        else:
            raise ValueError(f"Unsupported training.checkpoint_mode: {checkpoint_mode}")
        if extra:
            payload.update(extra)
        if name is None:
            filename = (
                "last_pose_head.pth"
                if checkpoint_mode == "pose_head_only"
                else (
                    "last_descriptor_head.pth"
                    if checkpoint_mode == "descriptor_head_only"
                    else (
                        "last_geometry_head.pth"
                        if checkpoint_mode == "geometry_head_only"
                        else "last_event_dark3r.pth"
                    )
                )
            )
        else:
            filename = name
        torch.save(payload, save_dir / filename)
    barrier_if_distributed()


def merge_loss_dict(accum, loss_dict):
    for key, value in loss_dict.items():
        if isinstance(value, (int, float)):
            total, count = accum.get(key, (0.0, 0))
            accum[key] = (total + float(value), count + 1)
        elif isinstance(value, str) and key.startswith("skipped "):
            accum[key] = accum.get(key, 0) + 1


def merge_loss_accum(accum, other):
    for key, value in other.items():
        if isinstance(value, tuple):
            total, count = value
            cur_total, cur_count = accum.get(key, (0.0, 0))
            accum[key] = (cur_total + float(total), cur_count + int(count))
        else:
            accum[key] = accum.get(key, 0) + int(value)


def gather_loss_accum(loss_accum):
    if not distributed_is_initialized():
        return loss_accum
    gathered = [None for _ in range(dist.get_world_size())]
    dist.all_gather_object(gathered, loss_accum)
    merged = {}
    for item in gathered:
        if item:
            merge_loss_accum(merged, item)
    return merged


def finalize_loss_dict(accum):
    out = {}
    for key, value in accum.items():
        if isinstance(value, tuple):
            total, count = value
            out[key] = total / max(count, 1)
        else:
            out[key] = value
    return out


def evaluate(
    student,
    teacher,
    loader,
    device,
    loss_cfg,
    max_batches=None,
    symmetric=False,
    amp_enabled=False,
    amp_dtype=torch.float16,
):
    student.eval()
    if teacher is not None:
        teacher.eval()
    total_loss = 0.0
    num_batches = 0
    loss_accum = {}
    with torch.no_grad():
        for batch_idx, batch in enumerate(loader):
            if max_batches is not None and batch_idx >= max_batches:
                break
            batch = move_batch_to_device(batch, device)
            loss, loss_dict, _ = compute_eval_loss(
                student,
                teacher,
                batch,
                loss_cfg,
                symmetric=symmetric,
                amp_enabled=amp_enabled,
                amp_dtype=amp_dtype,
                device_type=device.type,
            )
            sync_configured_skips(loss_dict, loss_cfg, f"eval batch {batch_idx}", device)
            if not sync_finite_loss(loss, device, f"eval batch {batch_idx}"):
                continue
            total_loss += float(loss.detach().cpu())
            num_batches += 1
            merge_loss_dict(loss_accum, loss_dict)
    if distributed_is_initialized():
        metrics = torch.tensor([total_loss, float(num_batches)], device=device)
        dist.all_reduce(metrics, op=dist.ReduceOp.SUM)
        total_loss = float(metrics[0].detach().cpu())
        num_batches = int(metrics[1].detach().cpu())
        loss_accum = gather_loss_accum(loss_accum)
    avg_loss = total_loss / max(num_batches, 1)
    return {"loss": avg_loss, "num_batches": num_batches, "loss_dict": finalize_loss_dict(loss_accum)}


def main():
    args = parse_args()
    cfg = load_config(args.config)
    torch.manual_seed(args.seed)

    device, distributed, local_rank, rank, world_size = setup_distributed(args.device)
    try:
        dataset = build_dataset(args, cfg)
        batch_size = args.batch_size or int(cfg_get(cfg, "training", "batch_size", 1))
        num_workers = (
            args.num_workers
            if args.num_workers is not None
            else int(cfg_get(cfg, "dataset", "num_workers", 4))
        )
        loader, sampler = make_train_loader(
            dataset,
            batch_size,
            num_workers,
            device,
            distributed,
            shuffle=True,
            cfg=cfg,
            seed=args.seed,
        )
        eval_cfg = cfg.get("eval", {})
        eval_batch_size = int(eval_cfg.get("batch_size", batch_size))
        eval_num_workers = int(eval_cfg.get("num_workers", num_workers))
        eval_loader, eval_sampler = make_loader(
            dataset,
            eval_batch_size,
            eval_num_workers,
            device,
            distributed,
            shuffle=False,
        )
        eval_every_epochs = int(eval_cfg.get("every_epochs", 1))
        eval_max_batches = eval_cfg.get("max_batches", None)
        if eval_max_batches is not None:
            eval_max_batches = int(eval_max_batches)
        save_every_fraction = float(cfg_get(cfg, "training", "save_every_epoch_fraction", 0.5))
        save_every_epoch = bool(cfg_get(cfg, "training", "save_every_epoch", True))
        checkpoint_mode = str(
            cfg_get(cfg, "training", "checkpoint_mode", "full")
        ).lower()
        save_every_steps = int(cfg_get(cfg, "training", "save_every_steps", 0))
        if checkpoint_mode not in {
            "full",
            "pose_head_only",
            "descriptor_head_only",
            "geometry_head_only",
        }:
            raise ValueError(
                "training.checkpoint_mode must be full, pose_head_only, "
                "descriptor_head_only, or geometry_head_only, got "
                f"{checkpoint_mode}"
            )
        if save_every_steps < 0:
            raise ValueError("training.save_every_steps must be non-negative")
        grad_clip_norm = float(cfg_get(cfg, "training", "grad_clip_norm", 1.0))
        max_total_loss = float(cfg_get(cfg, "loss", "max_total_loss", 1000.0))
        symmetric_pairs = should_use_symmetric(cfg, args)
        amp_enabled, amp_dtype_name, amp_dtype = resolve_amp_config(cfg, args, device)
        gradient_checkpointing = resolve_gradient_checkpointing(cfg, args)

        backbone_source = resolve_model_name_or_path(args, cfg, "checkpoint")
        teacher_source = resolve_model_name_or_path(
            args, cfg, "teacher_checkpoint", fallback=backbone_source
        )
        backbone_lora_checkpoint = resolve_backbone_lora_checkpoint(args, cfg)

        backbone = load_backbone(args, cfg, model_name_or_path=backbone_source)
        needs_teacher = (
            float(cfg_get(cfg, "loss", "feat_distill", 0.0)) > 0
            or float(cfg_get(cfg, "loss", "structure_consistency", 0.0)) > 0
            or float(cfg_get(cfg, "loss", "descriptor_distill", 0.0)) > 0
        )
        if not needs_teacher:
            teacher = None
        elif args.use_tiny_model or teacher_source == backbone_source:
            teacher = make_frozen_teacher(backbone).to(device)
        else:
            teacher = load_backbone(args, cfg, model_name_or_path=teacher_source)
            teacher.eval()
            teacher.requires_grad_(False)
            teacher = teacher.to(device)
        student = build_event_student(backbone, config=cfg).to(device)
        init_event_load_info = None
        if args.init_event_checkpoint:
            payload = safe_torch_load(
                args.init_event_checkpoint,
                map_location="cpu",
                allow_unsafe=args.allow_unsafe_checkpoint_load,
                purpose="initial Event-Dark3R checkpoint",
            )
            state_dict = payload.get("model", payload)
            restore_pristine_geometry_heads = bool(
                cfg_get(cfg, "training", "restore_pristine_geometry_heads", False)
            )
            restored_geometry_keys = []
            if restore_pristine_geometry_heads:
                restored_geometry_keys = [
                    key for key in state_dict if is_geometry_head_parameter(key)
                ]
                state_dict = {
                    key: value
                    for key, value in state_dict.items()
                    if not is_geometry_head_parameter(key)
                }
            missing, unexpected = student.load_state_dict(state_dict, strict=False)
            if restore_pristine_geometry_heads:
                restored_geometry_key_set = set(restored_geometry_keys)
                missing = [key for key in missing if key not in restored_geometry_key_set]
            if student.pose_head_only:
                non_pose_missing = [
                    key for key in missing if not key.startswith("pose_head.")
                ]
                if non_pose_missing or unexpected:
                    raise RuntimeError(
                        "Pose-head initialization must load the complete frozen Night3R "
                        "state. "
                        f"non_pose_missing={non_pose_missing} unexpected={list(unexpected)}"
                    )
            init_event_load_info = {
                "checkpoint": str(Path(args.init_event_checkpoint).resolve()),
                "step": payload.get("step") if isinstance(payload, dict) else None,
                "stage": payload.get("stage") if isinstance(payload, dict) else None,
                "epoch": payload.get("epoch") if isinstance(payload, dict) else None,
                "missing_keys": list(missing),
                "unexpected_keys": list(unexpected),
                "restored_pristine_geometry_heads": restore_pristine_geometry_heads,
                "restored_geometry_key_count": len(restored_geometry_keys),
            }
            del state_dict
            del payload
            gc.collect()
        validate_loss_config(cfg, dataset, student)
        stage_schedule = resolve_stage_schedule(cfg, args)
        first_stage = stage_schedule[0]
        ddp_init_stage = max(stage_schedule, key=lambda item: item["stage"])
        set_event_training_stage(
            student,
            stage=ddp_init_stage["stage"],
            partial_finetune_blocks=ddp_init_stage["partial_finetune_blocks"],
        )
        validate_pose_head_only_trainability(student)
        validate_descriptor_head_only_trainability(student)
        validate_geometry_head_only_trainability(student)

        if distributed:
            find_unused_parameters = not (
                getattr(student, "pose_head_only", False)
                or getattr(student, "descriptor_head_only", False)
            )
            ddp_kwargs = (
                {
                    "device_ids": [local_rank],
                    "output_device": local_rank,
                    "find_unused_parameters": find_unused_parameters,
                }
                if device.type == "cuda"
                else {"find_unused_parameters": find_unused_parameters}
            )
            student = DDP(student, **ddp_kwargs)
        set_event_training_stage(
            unwrap_model(student),
            stage=first_stage["stage"],
            partial_finetune_blocks=first_stage["partial_finetune_blocks"],
        )
        configure_gradient_checkpointing(student, gradient_checkpointing)
        optimizer = build_optimizer_for_stage(student, first_stage)
        scaler = make_grad_scaler(device, amp_enabled, amp_dtype)

        if rank == 0:
            Path(args.save_dir).mkdir(parents=True, exist_ok=True)
            resolved_data_root = getattr(dataset, "root_paths_text", str(dataset.root_path))
            resolved_args = {
                **vars(args),
                "resolved_data_root": resolved_data_root,
                "resolved_model_name_or_path": backbone_source,
                "resolved_teacher_checkpoint": teacher_source,
                "resolved_backbone_lora_checkpoint": backbone_lora_checkpoint,
                "init_event_checkpoint_load_info": init_event_load_info,
            }
            with open(Path(args.save_dir) / "resolved_event_train_config.json", "w") as f:
                json.dump(
                    {"config": cfg, "args": resolved_args, "stage_schedule": stage_schedule},
                    f,
                    indent=2,
                )
            trainable = sum(p.numel() for p in student.parameters() if p.requires_grad)
            print(
                "event_train_start",
                {
                    "data_root": resolved_data_root,
                    "model_name_or_path": backbone_source,
                    "teacher_checkpoint": teacher_source,
                    "backbone_lora_checkpoint": backbone_lora_checkpoint,
                    "init_event_checkpoint": init_event_load_info,
                    "samples": len(dataset),
                    "batch_size": batch_size,
                    "device": str(device),
                    "world_size": world_size,
                    "trainable_params": trainable,
                    "tiny_model": args.use_tiny_model,
                    "pair_sampling": getattr(dataset, "pair_sampling", None),
                    "min_pair_distance": getattr(dataset, "min_pair_distance", None),
                    "max_pair_distance": getattr(dataset, "max_pair_distance", None),
                    "allow_reverse_pairs": getattr(dataset, "allow_reverse_pairs", None),
                    "symmetric_pairs": symmetric_pairs,
                    "amp": amp_enabled,
                    "amp_dtype": amp_dtype_name,
                    "grad_scaler": scaler is not None,
                    "grad_clip_norm": grad_clip_norm,
                    "max_total_loss": max_total_loss,
                    "save_every_epoch": save_every_epoch,
                    "checkpoint_mode": checkpoint_mode,
                    "save_every_steps": save_every_steps,
                    "gradient_checkpointing": gradient_checkpointing,
                    "source_sampling_weights": (
                        sampler.source_weights
                        if isinstance(sampler, SourceBalancedSampler)
                        else None
                    ),
                    "stage_schedule": stage_schedule,
                },
            )
        barrier_if_distributed()

        max_steps = args.max_steps
        global_step = 0
        stop = False
        for stage_index, stage_cfg in enumerate(stage_schedule):
            model = unwrap_model(student)
            set_event_training_stage(
                model,
                stage=stage_cfg["stage"],
                partial_finetune_blocks=stage_cfg["partial_finetune_blocks"],
            )
            validate_pose_head_only_trainability(model)
            validate_descriptor_head_only_trainability(model)
            validate_geometry_head_only_trainability(model)
            configure_gradient_checkpointing(student, gradient_checkpointing)
            optimizer = build_optimizer_for_stage(student, stage_cfg)
            initialize_optimizer_lrs(optimizer)
            stage_step = 0
            if rank == 0:
                trainable = sum(p.numel() for p in student.parameters() if p.requires_grad)
                print(
                    "event_train_stage_start",
                    {
                        "stage_index": stage_index,
                        "stage": stage_cfg["stage"],
                        "num_epochs": stage_cfg["num_epochs"],
                        "partial_finetune_blocks": stage_cfg["partial_finetune_blocks"],
                        "trainable_params": trainable,
                        "amp": amp_enabled,
                        "amp_dtype": amp_dtype_name,
                        "grad_clip_norm": grad_clip_norm,
                        "base_lrs": optimizer_lrs(optimizer),
                        "warmup_steps": stage_cfg["warmup_steps"],
                        "warmup_start_factor": stage_cfg["warmup_start_factor"],
                        "gradient_checkpointing": gradient_checkpointing,
                    },
                )

            stage_steps_per_epoch = len(loader)
            save_interval_steps = max(1, int(round(stage_steps_per_epoch * save_every_fraction)))
            next_save_epoch_step = save_interval_steps

            for epoch in range(stage_cfg["num_epochs"]):
                global_epoch = sum(item["num_epochs"] for item in stage_schedule[:stage_index]) + epoch + 1
                if sampler is not None:
                    sampler.set_epoch(global_epoch - 1)
                if eval_sampler is not None:
                    eval_sampler.set_epoch(global_epoch - 1)
                epoch_step = 0
                for batch in loader:
                    lr_warmup_factor = apply_stage_lr_warmup(optimizer, stage_cfg, stage_step)
                    batch = move_batch_to_device(batch, device)
                    student.train()
                    if teacher is not None:
                        teacher.eval()

                    loss, loss_dict, student_out = compute_training_loss(
                        student,
                        teacher,
                        batch,
                        cfg.get("loss", {}),
                        symmetric=symmetric_pairs,
                        amp_enabled=amp_enabled,
                        amp_dtype=amp_dtype,
                        device_type=device.type,
                        step=global_step,
                    )
                    sync_configured_skips(
                        loss_dict,
                        cfg.get("loss", {}),
                        f"train step {global_step}",
                        device,
                    )
                    loss_finite = sync_finite_loss(loss, device, f"train step {global_step}")
                    if not loss_finite:
                        log_skipped_train_step(
                            rank,
                            global_step,
                            "non_finite_loss",
                            batch,
                            loss,
                            loss_dict,
                        )
                        optimizer.zero_grad(set_to_none=True)
                        global_step += 1
                        epoch_step += 1
                        stage_step += 1
                        continue
                    loss_in_range = sync_loss_below_threshold(
                        loss,
                        max_total_loss,
                        device,
                        f"train step {global_step}",
                    )
                    if not loss_in_range:
                        log_skipped_train_step(
                            rank,
                            global_step,
                            "loss_above_max_total_loss",
                            batch,
                            loss,
                            loss_dict,
                        )
                        optimizer.zero_grad(set_to_none=True)
                        global_step += 1
                        epoch_step += 1
                        stage_step += 1
                        continue

                    optimizer.zero_grad(set_to_none=True)
                    if scaler is not None:
                        scaler.scale(loss).backward()
                        scaler.unscale_(optimizer)
                        grad_norm = clip_grad_norm_if_enabled(student.parameters(), grad_clip_norm)
                        grad_finite = grad_norm is None or sync_finite_tensor(
                            grad_norm,
                            device,
                            f"train step {global_step}",
                            "grad_norm",
                        )
                        if grad_finite:
                            scaler.step(optimizer)
                        else:
                            log_skipped_train_step(
                                rank,
                                global_step,
                                "non_finite_grad_norm",
                                batch,
                                loss,
                                loss_dict,
                                grad_norm=grad_norm,
                            )
                            optimizer.zero_grad(set_to_none=True)
                        scaler.update()
                        if not grad_finite:
                            global_step += 1
                            epoch_step += 1
                            stage_step += 1
                            continue
                    else:
                        loss.backward()
                        grad_norm = clip_grad_norm_if_enabled(student.parameters(), grad_clip_norm)
                        grad_finite = grad_norm is None or sync_finite_tensor(
                            grad_norm,
                            device,
                            f"train step {global_step}",
                            "grad_norm",
                        )
                        if grad_finite:
                            optimizer.step()
                        else:
                            log_skipped_train_step(
                                rank,
                                global_step,
                                "non_finite_grad_norm",
                                batch,
                                loss,
                                loss_dict,
                                grad_norm=grad_norm,
                            )
                            optimizer.zero_grad(set_to_none=True)
                            global_step += 1
                            epoch_step += 1
                            stage_step += 1
                            continue

                    if rank == 0 and global_step % max(args.log_every, 1) == 0:
                        print(
                            "event_train_step",
                            {
                                "stage_index": stage_index,
                                "stage": stage_cfg["stage"],
                                "epoch": epoch,
                                "step": global_step,
                                "sample_id": batch.get("sample_id", None),
                                "images": tuple(batch["images"].shape),
                                "event_voxel": tuple(batch["event_voxel"].shape),
                                "pair_sampling": batch.get("pair_sampling", None),
                                "symmetric_pairs": symmetric_pairs,
                                "amp": amp_enabled,
                                "amp_dtype": amp_dtype_name,
                                "lr_warmup_factor": lr_warmup_factor,
                                "lrs": optimizer_lrs(optimizer),
                                "gradient_checkpointing": gradient_checkpointing,
                                "loss": float(loss.detach().cpu()),
                                "event_enabled": student_out[0]["event_enabled"],
                                "event_gate": student_out[0]["event_gate"],
                                "event_debug": event_debug_log_fields(student_out, batch),
                                "loss_dict": loss_dict,
                            },
                        )
                    global_step += 1
                    epoch_step += 1
                    stage_step += 1
                    if save_every_steps > 0 and global_step % save_every_steps == 0:
                        save_checkpoint(
                            args.save_dir,
                            student,
                            optimizer,
                            cfg,
                            args,
                            global_step,
                            rank,
                            name=(
                                f"pose_head_step{global_step:08d}.pth"
                                if checkpoint_mode == "pose_head_only"
                                else (
                                    f"descriptor_head_step{global_step:08d}.pth"
                                    if checkpoint_mode == "descriptor_head_only"
                                    else (
                                        f"geometry_head_step{global_step:08d}.pth"
                                        if checkpoint_mode == "geometry_head_only"
                                        else f"event_dark3r_step{global_step:08d}.pth"
                                    )
                                )
                            ),
                            extra={
                                "stage_index": stage_index,
                                "stage": stage_cfg["stage"],
                                "epoch": epoch,
                                "stage_cfg": stage_cfg,
                                "checkpoint_kind": checkpoint_mode,
                            },
                            scaler=scaler,
                        )
                    if save_every_steps <= 0 and epoch_step >= next_save_epoch_step:
                        save_checkpoint(
                            args.save_dir,
                            student,
                            optimizer,
                            cfg,
                            args,
                            global_step,
                            rank,
                            name=(
                                f"event_dark3r_stage{stage_cfg['stage']}"
                                f"_epoch{epoch + 1:04d}_step{global_step:08d}.pth"
                            ),
                            extra={
                                "stage_index": stage_index,
                                "stage": stage_cfg["stage"],
                                "epoch": epoch,
                                "stage_cfg": stage_cfg,
                            },
                            scaler=scaler,
                        )
                        next_save_epoch_step += save_interval_steps
                    if max_steps is not None and global_step >= max_steps:
                        stop = True
                        break
                if eval_every_epochs > 0 and (epoch + 1) % eval_every_epochs == 0:
                    eval_metrics = evaluate(
                        student,
                        teacher,
                        eval_loader,
                        device,
                        cfg.get("loss", {}),
                        max_batches=eval_max_batches,
                        symmetric=symmetric_pairs,
                        amp_enabled=amp_enabled,
                        amp_dtype=amp_dtype,
                    )
                    if rank == 0:
                        print(
                            "event_eval_epoch",
                            {
                                "stage_index": stage_index,
                                "stage": stage_cfg["stage"],
                                "epoch": epoch,
                                "step": global_step,
                                **eval_metrics,
                            },
                        )
                if save_every_epoch:
                    save_checkpoint(
                        args.save_dir,
                        student,
                        optimizer,
                        cfg,
                        args,
                        global_step,
                        rank,
                        name=(
                            f"event_dark3r_epoch{global_epoch:04d}"
                            f"_stage{stage_cfg['stage']}"
                            f"_stageepoch{epoch + 1:04d}"
                            f"_step{global_step:08d}.pth"
                        ),
                        extra={
                            "stage_index": stage_index,
                            "stage": stage_cfg["stage"],
                            "epoch": epoch,
                            "global_epoch": global_epoch,
                            "stage_cfg": stage_cfg,
                            "checkpoint_kind": "epoch",
                        },
                        scaler=scaler,
                    )
                if stop:
                    break
            if stop:
                break

        save_checkpoint(
            args.save_dir,
            student,
            optimizer,
            cfg,
            args,
            global_step,
            rank,
            scaler=scaler,
        )
        if rank == 0:
            print("event_train_done", {"steps": global_step, "saved": not args.no_save})
    finally:
        cleanup_distributed(distributed)


if __name__ == "__main__":
    main()
