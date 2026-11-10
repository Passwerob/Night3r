#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

from dataset.dsec_event_dataset import DSECEventDataset
from model.event_dark3r_student import EventDark3RStudent
from train_event_dark3r import (
    load_backbone,
    load_config,
    move_batch_to_device,
    resolve_data_root,
    resolve_model_name_or_path,
    safe_torch_load,
)


def parse_args():
    parser = argparse.ArgumentParser("Export Night3R pair predictions for event/RGB data")
    parser.add_argument("--config", default="config/train_event_dark3r.yaml")
    parser.add_argument("--data-root", default=None)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--outdir", required=True)
    parser.add_argument("--model-name-or-path", default=None)
    parser.add_argument(
        "--allow-unsafe-checkpoint-load",
        action="store_true",
        help=(
            "Allow legacy pickle checkpoint loading for trusted local MASt3R/Dark3R "
            "or Night3R/Event-Dark3R checkpoints. Leave disabled for untrusted files."
        ),
    )
    parser.add_argument(
        "--allow-partial-load",
        action="store_true",
        help=(
            "Allow missing/unexpected checkpoint keys and record them in the manifest. "
            "Default is strict loading."
        ),
    )
    parser.add_argument("--use-tiny-model", action="store_true")
    parser.add_argument("--image-size", type=int, nargs=2, default=None, metavar=("H", "W"))
    parser.add_argument("--pair-stride", type=int, default=None)
    parser.add_argument("--max-scenes", type=int, default=None)
    parser.add_argument("--max-pairs-per-scene", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--max-pairs", type=int, default=None)
    return parser.parse_args()


def cfg_get(cfg, section, key, default=None):
    return cfg.get(section, {}).get(key, default)


def resolve_device(device_arg):
    if device_arg == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("--device cuda requested but CUDA is unavailable")
        return torch.device("cuda")
    if device_arg == "cpu":
        return torch.device("cpu")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def build_dataset(args, cfg):
    dataset_cfg = cfg.get("dataset", {})
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
        expected_event_channels=int(cfg_get(cfg, "event", "voxel_channels", 8)),
        lowlight_enabled=False,
    )


def build_model(args, cfg, device):
    backbone_source = resolve_model_name_or_path(args, cfg, "checkpoint")
    backbone = load_backbone(args, cfg, model_name_or_path=backbone_source)
    model = EventDark3RStudent(backbone, config=cfg)
    checkpoint = safe_torch_load(
        args.checkpoint,
        map_location="cpu",
        allow_unsafe=args.allow_unsafe_checkpoint_load,
        purpose="Event-Dark3R export checkpoint",
    )
    state_dict = checkpoint.get("model", checkpoint)
    missing, unexpected = model.load_state_dict(
        state_dict,
        strict=not args.allow_partial_load,
    )
    model = model.to(device)
    model.eval()
    return model, missing, unexpected, backbone_source


def _first_item(value):
    if isinstance(value, (list, tuple)):
        return value[0]
    if torch.is_tensor(value):
        if value.ndim == 0:
            return value.item()
        return value[0].item() if value.numel() == 1 else value[0].detach().cpu().tolist()
    return value


def tensor_to_numpy(tensor):
    return tensor.detach().cpu().float().numpy()


def export_pair(out_path, batch, pred1, pred2):
    payload = {}
    for prefix, pred in (("view1", pred1), ("view2", pred2)):
        for key in ("pts3d", "pts3d_in_other_view", "conf", "desc_conf"):
            if key in pred and torch.is_tensor(pred[key]):
                payload[f"{prefix}_{key}"] = tensor_to_numpy(pred[key])
    payload["intrinsics"] = tensor_to_numpy(batch["intrinsics"])
    payload["event_gate"] = (
        -1.0 if pred1.get("event_gate", None) is None else float(pred1["event_gate"])
    )
    payload["event_enabled"] = bool(pred1.get("event_enabled", False))
    np.savez_compressed(out_path, **payload)


def main():
    args = parse_args()
    cfg = load_config(args.config)
    device = resolve_device(args.device)
    outdir = Path(args.outdir)
    pred_dir = outdir / "predictions"
    pred_dir.mkdir(parents=True, exist_ok=True)

    dataset = build_dataset(args, cfg)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )
    model, missing, unexpected, backbone_source = build_model(args, cfg, device)
    resolved_args = {
        **vars(args),
        "resolved_data_root": str(dataset.root_path),
        "resolved_model_name_or_path": backbone_source,
    }

    manifest = {
        "config": cfg,
        "args": resolved_args,
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "missing_keys": list(missing),
        "unexpected_keys": list(unexpected),
        "pairs": [],
    }

    exported = 0
    with torch.no_grad():
        for batch in loader:
            batch = move_batch_to_device(batch, device)
            pred1, pred2 = model(batch)
            batch_size = batch["images"].shape[0]
            for local_idx in range(batch_size):
                if args.max_pairs is not None and exported >= args.max_pairs:
                    break
                pair_name = f"pair_{exported:06d}.npz"
                single_batch = {
                    key: value[local_idx : local_idx + 1]
                    if torch.is_tensor(value)
                    else value
                    for key, value in batch.items()
                }
                single_pred1 = {
                    key: value[local_idx : local_idx + 1]
                    if torch.is_tensor(value)
                    else value
                    for key, value in pred1.items()
                }
                single_pred2 = {
                    key: value[local_idx : local_idx + 1]
                    if torch.is_tensor(value)
                    else value
                    for key, value in pred2.items()
                }
                export_pair(pred_dir / pair_name, single_batch, single_pred1, single_pred2)
                manifest["pairs"].append(
                    {
                        "prediction": str((pred_dir / pair_name).relative_to(outdir)),
                        "sample_id": _first_item(batch.get("sample_id", "")),
                        "scene": _first_item(batch.get("scene", "")),
                        "image1_path": _first_item(batch.get("image1_path", "")),
                        "image2_path": _first_item(batch.get("image2_path", "")),
                        "event1_path": _first_item(batch.get("event1_path", "")),
                        "event2_path": _first_item(batch.get("event2_path", "")),
                        "idx1": _first_item(batch.get("idx1", -1)),
                        "idx2": _first_item(batch.get("idx2", -1)),
                    }
                )
                exported += 1
            if args.max_pairs is not None and exported >= args.max_pairs:
                break

    with open(outdir / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)
    with open(outdir / "config.yaml", "w") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)
    print(
        "event_export_done",
        {
            "outdir": str(outdir.resolve()),
            "pairs": exported,
            "missing_keys": len(missing),
            "unexpected_keys": len(unexpected),
        },
    )


if __name__ == "__main__":
    main()
