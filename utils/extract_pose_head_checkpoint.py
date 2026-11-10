#!/usr/bin/env python3
import argparse
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from train_event_dark3r import safe_torch_load


def parse_args():
    parser = argparse.ArgumentParser("Extract a lightweight pose-head checkpoint")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--base-checkpoint", required=True)
    parser.add_argument("--allow-unsafe-checkpoint-load", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    payload = safe_torch_load(
        args.input,
        map_location="cpu",
        allow_unsafe=args.allow_unsafe_checkpoint_load,
        purpose="full Night3R pose checkpoint",
    )
    model_state = payload.get("model", payload)
    pose_head = {
        key[len("pose_head.") :]: value.detach().cpu()
        for key, value in model_state.items()
        if key.startswith("pose_head.")
    }
    if not pose_head:
        raise RuntimeError(f"No pose_head.* parameters found in {args.input}")
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "checkpoint_kind": "pose_head_only",
            "pose_head": pose_head,
            "pose_head_config": payload.get("config", {}).get("pose_head", {}),
            "base_checkpoint": str(Path(args.base_checkpoint).resolve()),
            "source_checkpoint": str(Path(args.input).resolve()),
            "step": payload.get("step"),
        },
        output,
    )
    print(f"Saved {len(pose_head)} pose-head tensors to {output}")


if __name__ == "__main__":
    main()
