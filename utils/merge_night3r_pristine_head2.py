#!/usr/bin/env python3
"""Replace a Night3R checkpoint's frozen view-2 head with pristine MASt3R weights."""

import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from train_event_dark3r import (  # noqa: E402
    build_event_student,
    load_backbone,
    load_config,
    safe_torch_load,
)


PREFIX = "dark3r.downstream_head2."


def parse_args():
    parser = argparse.ArgumentParser("Restore pristine MASt3R downstream_head2")
    parser.add_argument("--source-checkpoint", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--model-name-or-path", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--allow-unsafe-checkpoint-load", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    cfg = load_config(str(args.config))
    model_args = SimpleNamespace(
        model_name_or_path=str(args.model_name_or_path),
        allow_unsafe_checkpoint_load=args.allow_unsafe_checkpoint_load,
        use_tiny_model=False,
    )
    backbone = load_backbone(model_args, cfg)
    pristine_student = build_event_student(backbone, config=cfg)
    pristine_state = pristine_student.state_dict()
    source = safe_torch_load(
        args.source_checkpoint,
        map_location="cpu",
        allow_unsafe=args.allow_unsafe_checkpoint_load,
        purpose="source Night3R checkpoint",
    )
    source_state = source["model"]
    replacement_keys = sorted(key for key in source_state if key.startswith(PREFIX))
    if not replacement_keys:
        raise RuntimeError(f"No checkpoint keys found under {PREFIX}")
    missing_pristine = [key for key in replacement_keys if key not in pristine_state]
    if missing_pristine:
        raise RuntimeError(
            "Pristine student is missing head2 keys: " + ", ".join(missing_pristine[:10])
        )

    merged_state = dict(source_state)
    changed_keys = []
    already_equal_keys = []
    for key in replacement_keys:
        pristine_value = pristine_state[key].detach().cpu()
        source_value = source_state[key].detach().cpu()
        if torch.equal(source_value, pristine_value):
            already_equal_keys.append(key)
        else:
            changed_keys.append(key)
        merged_state[key] = pristine_value

    retained_mismatch = [
        key
        for key, value in source_state.items()
        if not key.startswith(PREFIX) and merged_state[key] is not value
    ]
    if retained_mismatch:
        raise RuntimeError("Non-head2 tensors were unexpectedly replaced")

    output_payload = {
        "model": merged_state,
        "config": source.get("config", cfg),
        "args": source.get("args"),
        "step": source.get("step"),
        "stage": source.get("stage"),
        "epoch": source.get("epoch"),
        "repair_metadata": {
            "kind": "restore_pristine_mast3r_downstream_head2",
            "source_checkpoint": str(args.source_checkpoint.resolve()),
            "pristine_model": str(args.model_name_or_path.resolve()),
            "replacement_prefix": PREFIX,
            "replacement_key_count": len(replacement_keys),
            "changed_key_count": len(changed_keys),
            "already_equal_key_count": len(already_equal_keys),
            "optimizer_state_removed": "optimizer" in source,
            "reason": (
                "Recover the frozen view-2 point-map head while preserving all "
                "Night3R event encoder, fusion adapter, and view-1 geometry weights."
            ),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(output_payload, args.output)

    written = safe_torch_load(
        args.output,
        map_location="cpu",
        allow_unsafe=True,
        purpose="written repaired Night3R checkpoint",
    )
    written_state = written["model"]
    verification_failures = []
    for key in replacement_keys:
        if not torch.equal(written_state[key], pristine_state[key].detach().cpu()):
            verification_failures.append(key)
    for key in source_state:
        if key.startswith(PREFIX):
            continue
        if not torch.equal(written_state[key], source_state[key]):
            verification_failures.append(key)
    if verification_failures:
        raise RuntimeError(
            "Merged checkpoint verification failed: "
            + ", ".join(verification_failures[:10])
        )

    manifest = {
        **output_payload["repair_metadata"],
        "output": str(args.output.resolve()),
        "output_size_bytes": args.output.stat().st_size,
        "verification": "all head2 tensors match pristine MASt3R; all other tensors match source Night3R",
        "changed_keys": changed_keys,
        "already_equal_keys": already_equal_keys,
    }
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
