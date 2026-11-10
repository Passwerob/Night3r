#!/usr/bin/env python3
"""Attach a pose-head-only checkpoint to a full Night3R geometry checkpoint."""

import argparse
import copy
import sys
from pathlib import Path

import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from train_event_dark3r import safe_torch_load  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser("Compose Night3R geometry and pose-head checkpoints")
    parser.add_argument("--base-checkpoint", required=True)
    parser.add_argument("--pose-head-checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--allow-unsafe-checkpoint-load", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    base_path = Path(args.base_checkpoint)
    pose_path = Path(args.pose_head_checkpoint)
    base = safe_torch_load(
        base_path,
        map_location="cpu",
        allow_unsafe=args.allow_unsafe_checkpoint_load,
        purpose="base Night3R checkpoint",
    )
    pose_payload = safe_torch_load(
        pose_path,
        map_location="cpu",
        allow_unsafe=args.allow_unsafe_checkpoint_load,
        purpose="pose-head-only checkpoint",
    )
    if not isinstance(base, dict) or not isinstance(base.get("model"), dict):
        raise RuntimeError("The base checkpoint must contain a full 'model' state dictionary")
    pose_state = pose_payload.get("pose_head") if isinstance(pose_payload, dict) else None
    if not isinstance(pose_state, dict) or not pose_state:
        raise RuntimeError("The pose checkpoint must contain a non-empty 'pose_head' state")

    merged = copy.copy(base)
    model_state = dict(base["model"])
    existing = [key for key in model_state if key.startswith("pose_head.")]
    if existing:
        raise RuntimeError("The base checkpoint already contains pose-head parameters")
    model_state.update({f"pose_head.{key}": value for key, value in pose_state.items()})
    merged["model"] = model_state
    merged["composed_pose_head"] = {
        "base_checkpoint": str(base_path.resolve()),
        "pose_head_checkpoint": str(pose_path.resolve()),
        "pose_head_step": pose_payload.get("step"),
    }

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(merged, output)
    print(
        f"Wrote {output} with {len(model_state)} model tensors "
        f"({len(pose_state)} pose-head tensors attached)."
    )


if __name__ == "__main__":
    main()
