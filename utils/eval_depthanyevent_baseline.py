#!/usr/bin/env python3
"""Evaluate DepthAnyEvent on the exact cached Night3R depth protocol."""

import argparse
import json
import statistics
import sys
from pathlib import Path
from types import SimpleNamespace

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from utils.compare_event_dark3r_checkpoint import (  # noqa: E402
    DenseMetricAccumulator,
    as_bchw,
)
from utils.eval_event_dark3r_variant import (  # noqa: E402
    batch_sample_ids,
    load_batch_cache,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate an official DepthAnyEvent checkpoint with the same cached "
            "samples, masks, alignment, and metric aggregation as Night3R."
        )
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument(
        "--model-config",
        type=Path,
        default=None,
        help="Defaults to config.json next to the official checkpoint.",
    )
    parser.add_argument("--batch-cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--depthanyevent-root",
        type=Path,
        default=REPO_ROOT.parent / "DepthAnyEvent",
        help="Official bartn8/depthanyevent checkout.",
    )
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--max-batches", type=int, default=None)
    parser.add_argument("--view-index", type=int, choices=(0, 1), default=0)
    parser.add_argument("--depth-min", type=float, default=None)
    parser.add_argument("--depth-max", type=float, default=None)
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--amp-dtype", choices=("fp16", "bf16"), default="bf16")
    parser.add_argument("--prediction-dir", type=Path, default=None)
    parser.add_argument("--prediction-count", type=int, default=0)
    parser.add_argument(
        "--apply-checkpoint-scaleshift",
        action="store_true",
        help=(
            "Apply the checkpoint's official per-sample, GT-derived least-squares "
            "scale and shift before evaluation. This is an aligned diagnostic, "
            "not a raw metric-depth prediction."
        ),
    )
    parser.add_argument(
        "--unsafe-checkpoint-load",
        action="store_true",
        help="Required for legacy official checkpoints that cannot use weights_only=True.",
    )
    return parser.parse_args()


def resolve_device(name):
    if name == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("--device cuda requested but CUDA is unavailable")
        return torch.device("cuda")
    if name == "cpu":
        return torch.device("cpu")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_checkpoint(path, allow_unsafe=False):
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        if not allow_unsafe:
            raise RuntimeError(
                "This PyTorch version cannot safely load the checkpoint. "
                "Pass --unsafe-checkpoint-load only for a trusted official file."
            )
    except Exception as exc:
        if not allow_unsafe:
            raise RuntimeError(
                "Safe checkpoint loading failed. Pass --unsafe-checkpoint-load only "
                "for a trusted official DepthAnyEvent checkpoint."
            ) from exc
    return torch.load(path, map_location="cpu", weights_only=False)


def checkpoint_settings(checkpoint, model_config_path=None):
    if not isinstance(checkpoint, dict):
        raise ValueError("DepthAnyEvent checkpoint must be a dictionary")
    config = checkpoint.get("config")
    if not isinstance(config, dict):
        if model_config_path is None:
            raise ValueError(
                "DepthAnyEvent checkpoint has no embedded config; provide --model-config"
            )
        config = json.loads(Path(model_config_path).read_text())
    if not isinstance(config.get("model"), dict):
        raise ValueError("DepthAnyEvent config is missing config.model")
    trainer = config.get("trainer", {}) or {}
    return {
        "config": config,
        "model_type": str(config["model"].get("model_type", "")),
        "use_logdepth": bool(trainer.get("use_logdepth", False)),
        "checkpoint_use_scaleshift": bool(trainer.get("use_scaleshift", False)),
        "reg_factor": float(trainer.get("reg_factor", 3.70378)),
        "clip_distance": float(trainer.get("clip_distance", 80.0)),
    }


def build_model(checkpoint, checkpoint_path, model_config_path, depthanyevent_root, device):
    root = depthanyevent_root.resolve()
    if not (root / "models" / "__init__.py").is_file():
        raise FileNotFoundError(f"Invalid DepthAnyEvent checkout: {root}")
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

    from models import fetch_model

    settings = checkpoint_settings(checkpoint, model_config_path=model_config_path)
    model_config = dict(settings["config"]["model"])
    model_config["checkpoint_path"] = str(checkpoint_path.resolve())
    args = SimpleNamespace(loadmodel=str(checkpoint_path.resolve()))
    model = fetch_model(
        model_config,
        args,
        device=device,
        test=True,
        _state_dict=checkpoint,
    )
    return model, settings


def voxel_to_tencode(event_voxel):
    """Approximate official Tencode from positive/negative temporal voxel bins.

    Night3R caches store [positive bins..., negative bins...]. The voxel no longer
    contains within-bin event order, so the dominant polarity is used at each
    temporal bin and later bins overwrite earlier bins, as in Tencode.
    """
    if event_voxel.ndim != 4:
        raise ValueError(f"Expected event voxel [B,C,H,W], got {tuple(event_voxel.shape)}")
    channels = event_voxel.shape[1]
    if channels < 2 or channels % 2:
        raise ValueError(f"Expected an even polarity-separated channel count, got {channels}")

    num_bins = channels // 2
    positive = event_voxel[:, :num_bins].abs()
    negative = event_voxel[:, num_bins:].abs()
    output = torch.zeros(
        (event_voxel.shape[0], 3, *event_voxel.shape[-2:]),
        dtype=event_voxel.dtype,
        device=event_voxel.device,
    )
    for bin_index in range(num_bins):
        pos = positive[:, bin_index]
        neg = negative[:, bin_index]
        active = (pos > 0) | (neg > 0)
        # Exact last-event polarity is unavailable after voxelization. Dominant
        # magnitude is deterministic; equal activity follows positive polarity.
        is_positive = active & (pos >= neg)
        is_negative = active & ~is_positive
        recency = 1.0 if num_bins == 1 else 1.0 - bin_index / (num_bins - 1)
        output[:, 0] = torch.where(is_positive, 1.0, torch.where(is_negative, 0.0, output[:, 0]))
        output[:, 1] = torch.where(active, recency, output[:, 1])
        output[:, 2] = torch.where(is_negative, 1.0, torch.where(is_positive, 0.0, output[:, 2]))
    return output


def select_cached_view(batch, view_index):
    event = batch.get("event_voxel")
    if not torch.is_tensor(event) or event.ndim != 5:
        raise ValueError("batch['event_voxel'] must be [B,S,C,H,W]")
    if event.shape[1] <= view_index:
        raise ValueError(f"view {view_index} is unavailable in event voxel shape {tuple(event.shape)}")
    return event[:, view_index]


def select_cached_depth(batch, view_index):
    key = "depth_gt" if view_index == 0 else "depth_gt2"
    gt_depth = as_bchw(batch.get(key))
    if gt_depth is None:
        raise ValueError(f"Cached batch is missing {key}")
    if view_index == 0:
        gt_valid = as_bchw(batch.get("depth_valid"))
        if gt_valid is None:
            raise ValueError("Cached batch is missing depth_valid")
    else:
        gt_valid = torch.isfinite(gt_depth) & (gt_depth > 0)
    return gt_depth, gt_valid


def decode_prediction(prediction, use_logdepth, clip_distance, reg_factor):
    if prediction.ndim == 3:
        prediction = prediction[:, None]
    if prediction.ndim != 4:
        raise ValueError(f"Expected prediction [B,1,H,W], got {tuple(prediction.shape)}")
    prediction = prediction.float()
    if use_logdepth:
        prediction = torch.exp(reg_factor * (prediction - 1.0))
        maxima = prediction.flatten(1).amax(dim=1).clamp_min(1.0e-12)
        prediction = prediction / maxima[:, None, None, None]
        prediction = prediction * clip_distance
    return prediction.clamp(0.0, clip_distance)


def apply_gt_scale_shift(prediction, gt_depth, gt_valid, clip_distance):
    """Apply DepthAnyEvent's official per-sample least-squares GT alignment."""
    if prediction.shape[-2:] != gt_depth.shape[-2:]:
        prediction = torch.nn.functional.interpolate(
            prediction,
            size=gt_depth.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
    pred = prediction[:, 0].float()
    target = gt_depth[:, 0].float()
    mask = (
        gt_valid[:, 0].bool()
        & torch.isfinite(pred)
        & torch.isfinite(target)
        & (target > 0)
    )

    weights = mask.to(pred.dtype)
    safe_pred = torch.where(mask, pred, torch.zeros_like(pred))
    safe_target = torch.where(mask, target, torch.zeros_like(target))
    a_00 = torch.sum(weights * safe_pred * safe_pred, dim=(1, 2))
    a_01 = torch.sum(weights * safe_pred, dim=(1, 2))
    a_11 = torch.sum(weights, dim=(1, 2))
    b_0 = torch.sum(weights * safe_pred * safe_target, dim=(1, 2))
    b_1 = torch.sum(weights * safe_target, dim=(1, 2))
    determinant = a_00 * a_11 - a_01 * a_01

    scale = torch.zeros_like(b_0)
    shift = torch.zeros_like(b_1)
    solvable = determinant > 0
    scale[solvable] = (
        a_11[solvable] * b_0[solvable] - a_01[solvable] * b_1[solvable]
    ) / determinant[solvable]
    shift[solvable] = (
        -a_01[solvable] * b_0[solvable] + a_00[solvable] * b_1[solvable]
    ) / determinant[solvable]

    aligned = scale[:, None, None] * pred + shift[:, None, None]
    return aligned[:, None].clamp(0.0, clip_distance), scale, shift


def scene_names(batch, batch_size):
    values = batch.get("scene")
    if isinstance(values, (list, tuple)):
        return [str(value) for value in values]
    if values is not None:
        return [str(values)] * batch_size
    identifiers = batch_sample_ids(batch)
    if identifiers:
        return [identifier.split(":", 1)[0] for identifier in identifiers]
    return [""] * batch_size


def safe_sample_name(sample_id):
    return "".join(
        character if character.isalnum() or character in "-_" else "_"
        for character in sample_id
    )


def save_prediction_batch(args, batch, prediction, gt_depth, gt_valid, already_saved):
    if args.prediction_dir is None or already_saved >= args.prediction_count:
        return already_saved
    args.prediction_dir.mkdir(parents=True, exist_ok=True)
    identifiers = batch_sample_ids(batch)
    count = min(prediction.shape[0], args.prediction_count - already_saved)
    for item_idx in range(count):
        payload = {
            "sample_id": identifiers[item_idx],
            "variant": "depthanyevent",
            "pred_depth": prediction[item_idx].detach().float().cpu(),
            "gt_depth": gt_depth[item_idx].detach().float().cpu(),
            "gt_valid": gt_valid[item_idx].detach().bool().cpu(),
            "event_voxel": batch["event_voxel"][item_idx, args.view_index]
            .detach()
            .float()
            .cpu(),
            "intrinsics": batch["intrinsics"][item_idx].detach().float().cpu(),
        }
        torch.save(
            payload,
            args.prediction_dir / f"{safe_sample_name(identifiers[item_idx])}.pt",
        )
    return already_saved + count


def infer(model, model_type, tencode, previous_states=None):
    if model_type == "DAv2":
        return model.infer_image(tencode), None
    if model_type == "RecDAv2":
        prediction, states = model.infer_image(tencode, prev_states=previous_states)
        return prediction, states
    raise ValueError(f"Unsupported DepthAnyEvent model type: {model_type!r}")


def evaluate(args):
    checkpoint = load_checkpoint(args.checkpoint, allow_unsafe=args.unsafe_checkpoint_load)
    model_config_path = args.model_config or args.checkpoint.with_name("config.json")
    if not model_config_path.is_file():
        raise FileNotFoundError(
            f"DepthAnyEvent model config not found: {model_config_path}"
        )
    device = resolve_device(args.device)
    model, settings = build_model(
        checkpoint,
        args.checkpoint,
        model_config_path,
        args.depthanyevent_root,
        device,
    )
    cache = load_batch_cache(args.batch_cache)
    batches = cache["batches"]
    if args.max_batches is not None:
        batches = batches[: args.max_batches]

    metrics = DenseMetricAccumulator()
    sample_ids = []
    previous_states = None
    previous_scene = None
    scene_resets = 0
    processed = 0
    predictions_saved = 0
    scale_shift_scales = []
    scale_shift_shifts = []
    amp_dtype = torch.float16 if args.amp_dtype == "fp16" else torch.bfloat16

    with torch.no_grad():
        for batch in batches:
            event = select_cached_view(batch, args.view_index).to(device, non_blocking=True)
            if event.shape[0] != 1:
                raise ValueError(
                    "This protocol requires batch_size=1 so median alignment is "
                    "computed independently for every sample"
                )
            tencode = voxel_to_tencode(event.float())
            names = scene_names(batch, tencode.shape[0])
            if settings["model_type"] == "RecDAv2":
                if names[0] != previous_scene:
                    previous_states = None
                    previous_scene = names[0]
                    scene_resets += 1

            with torch.amp.autocast(
                device_type=device.type,
                dtype=amp_dtype,
                enabled=bool(args.amp and device.type == "cuda"),
            ):
                prediction, previous_states = infer(
                    model,
                    settings["model_type"],
                    tencode,
                    previous_states=previous_states,
                )
            prediction = decode_prediction(
                prediction,
                settings["use_logdepth"],
                settings["clip_distance"],
                settings["reg_factor"],
            )

            gt_depth, gt_valid = select_cached_depth(batch, args.view_index)
            gt_depth = gt_depth.to(device)
            gt_valid = gt_valid.to(device).bool()
            if args.depth_min is not None:
                gt_valid &= gt_depth >= args.depth_min
            if args.depth_max is not None:
                gt_valid &= gt_depth <= args.depth_max
            if args.apply_checkpoint_scaleshift:
                if not settings["checkpoint_use_scaleshift"]:
                    raise ValueError(
                        "--apply-checkpoint-scaleshift was requested, but the "
                        "checkpoint config has use_scaleshift=false"
                    )
                prediction, scale, shift = apply_gt_scale_shift(
                    prediction,
                    gt_depth,
                    gt_valid,
                    settings["clip_distance"],
                )
                scale_shift_scales.extend(scale.detach().cpu().tolist())
                scale_shift_shifts.extend(shift.detach().cpu().tolist())
            predictions_saved = save_prediction_batch(
                args,
                batch,
                prediction,
                gt_depth,
                gt_valid,
                predictions_saved,
            )
            metrics.add_depth(prediction, gt_depth, gt_valid, variant="raw")
            metrics.add_depth(prediction, gt_depth, gt_valid, variant="median_aligned")
            sample_ids.extend(batch_sample_ids(batch))
            processed += 1

    return {
        "variant": "depthanyevent_r" if settings["model_type"] == "RecDAv2" else "depthanyevent",
        "checkpoint": str(args.checkpoint.resolve()),
        "model_config": str(model_config_path.resolve()),
        "checkpoint_model_type": settings["model_type"],
        "checkpoint_use_logdepth": settings["use_logdepth"],
        "checkpoint_use_scaleshift": settings["checkpoint_use_scaleshift"],
        "checkpoint_scaleshift_applied": bool(args.apply_checkpoint_scaleshift),
        "checkpoint_scaleshift_protocol": (
            "independent per-sample least-squares scale+shift fitted on valid GT"
            if args.apply_checkpoint_scaleshift
            else None
        ),
        "checkpoint_scaleshift_scale_mean": (
            statistics.fmean(scale_shift_scales) if scale_shift_scales else None
        ),
        "checkpoint_scaleshift_scale_median": (
            statistics.median(scale_shift_scales) if scale_shift_scales else None
        ),
        "checkpoint_scaleshift_shift_mean": (
            statistics.fmean(scale_shift_shifts) if scale_shift_shifts else None
        ),
        "checkpoint_scaleshift_shift_median": (
            statistics.median(scale_shift_shifts) if scale_shift_shifts else None
        ),
        "clip_distance": settings["clip_distance"],
        "reg_factor": settings["reg_factor"],
        "batch_cache": str(args.batch_cache.resolve()),
        "cache_data_root": cache.get("data_root", ""),
        "cache_num_batches": int(cache.get("num_batches", len(cache["batches"]))),
        "num_batches": processed,
        "prediction_dir": (
            str(args.prediction_dir.resolve()) if args.prediction_dir is not None else None
        ),
        "predictions_saved": predictions_saved,
        "view_index": args.view_index,
        "device": str(device),
        "amp": bool(args.amp and device.type == "cuda"),
        "amp_dtype": args.amp_dtype if args.amp and device.type == "cuda" else None,
        "scene_state_resets": scene_resets,
        "depth_min": args.depth_min,
        "depth_max": args.depth_max,
        "input_protocol": {
            "source": "Night3R cached 4-bin polarity-separated event voxel",
            "conversion": "dominant-polarity voxel-to-Tencode; later bins overwrite earlier bins",
            "channels": 3,
            "normalization": "Tencode values in [0,1] followed by official model ImageNet normalization",
            "limitation": "within-bin event order is unavailable in the cached voxel",
        },
        "metric_protocol": {
            "aggregation": "valid-pixel micro-average",
            "resize": "bilinear prediction-to-GT, align_corners=False",
            "mask": "depth_valid & finite(GT,pred) & GT>0 & pred>0",
            "alignment": (
                "checkpoint per-sample GT least-squares scale+shift, followed by "
                "post-scaleshift raw and independent per-sample median-scale metrics"
                if args.apply_checkpoint_scaleshift
                else "raw and independent per-sample median scale; no shift"
            ),
        },
        "dense_metrics": metrics.finalize(),
        "sample_ids": sample_ids,
    }


def main():
    args = parse_args()
    result = evaluate(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
