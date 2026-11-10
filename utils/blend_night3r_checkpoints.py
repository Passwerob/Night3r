#!/usr/bin/env python3
"""Interpolate the changed floating tensors of two compatible Night3R checkpoints."""

import argparse
import json
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from train_event_dark3r import safe_torch_load  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-checkpoint", type=Path, required=True)
    parser.add_argument("--adapted-checkpoint", type=Path, required=True)
    parser.add_argument("--alphas", type=float, nargs="+", required=True)
    parser.add_argument("--label", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--allow-unsafe-checkpoint-load", action="store_true")
    return parser.parse_args()


def load_model(path, allow_unsafe):
    payload = safe_torch_load(
        path,
        map_location="cpu",
        allow_unsafe=allow_unsafe,
        purpose=f"Night3R checkpoint blend input: {path}",
    )
    if not isinstance(payload, dict) or "model" not in payload:
        raise RuntimeError(f"Checkpoint has no model state: {path}")
    return payload, payload["model"]


def alpha_tag(alpha):
    return f"a{int(round(alpha * 1000)):03d}"


def main():
    args = parse_args()
    if not args.alphas:
        raise ValueError("At least one interpolation alpha is required")
    for alpha in args.alphas:
        if not 0.0 <= alpha <= 1.0:
            raise ValueError(f"Interpolation alpha must be in [0, 1], got {alpha}")

    base_payload, base_state = load_model(
        args.base_checkpoint, args.allow_unsafe_checkpoint_load
    )
    adapted_payload, adapted_state = load_model(
        args.adapted_checkpoint, args.allow_unsafe_checkpoint_load
    )
    if set(base_state) != set(adapted_state):
        missing = sorted(set(base_state) - set(adapted_state))
        unexpected = sorted(set(adapted_state) - set(base_state))
        raise RuntimeError(
            f"Incompatible state keys: missing={missing[:10]} unexpected={unexpected[:10]}"
        )

    changed_keys = []
    for key, base_value in base_state.items():
        adapted_value = adapted_state[key]
        if base_value.shape != adapted_value.shape or base_value.dtype != adapted_value.dtype:
            raise RuntimeError(
                f"Incompatible tensor for {key}: "
                f"base={tuple(base_value.shape)}/{base_value.dtype}, "
                f"adapted={tuple(adapted_value.shape)}/{adapted_value.dtype}"
            )
        if torch.equal(base_value, adapted_value):
            continue
        if not (base_value.is_floating_point() or base_value.is_complex()):
            raise RuntimeError(f"Changed non-floating tensor cannot be blended: {key}")
        changed_keys.append(key)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    records = []
    for alpha in args.alphas:
        tag = alpha_tag(alpha)
        output = args.output_dir / f"{args.label}_{tag}.pth"
        blended_state = dict(base_state)
        for key in changed_keys:
            base_value = base_state[key]
            adapted_value = adapted_state[key]
            compute_dtype = (
                torch.complex64
                if base_value.is_complex()
                else torch.float32
            )
            blended = torch.lerp(
                base_value.to(dtype=compute_dtype),
                adapted_value.to(dtype=compute_dtype),
                float(alpha),
            ).to(dtype=base_value.dtype)
            blended_state[key] = blended

        output_payload = {
            "model": blended_state,
            "config": adapted_payload.get("config", base_payload.get("config")),
            "args": adapted_payload.get("args", base_payload.get("args")),
            "step": adapted_payload.get("step"),
            "stage": adapted_payload.get("stage"),
            "epoch": adapted_payload.get("epoch"),
            "blend_metadata": {
                "base_checkpoint": str(args.base_checkpoint.resolve()),
                "adapted_checkpoint": str(args.adapted_checkpoint.resolve()),
                "alpha": float(alpha),
                "formula": "(1 - alpha) * base + alpha * adapted",
                "changed_floating_tensor_count": len(changed_keys),
                "unchanged_tensor_count": len(base_state) - len(changed_keys),
                "optimizer_state_removed": (
                    "optimizer" in base_payload or "optimizer" in adapted_payload
                ),
            },
        }
        torch.save(output_payload, output)
        records.append(
            {
                **output_payload["blend_metadata"],
                "label": f"{args.label}_{tag}",
                "output": str(output.resolve()),
                "output_size_bytes": output.stat().st_size,
            }
        )
        del blended_state
        del output_payload

    manifest = {
        "label": args.label,
        "base_checkpoint": str(args.base_checkpoint.resolve()),
        "adapted_checkpoint": str(args.adapted_checkpoint.resolve()),
        "state_tensor_count": len(base_state),
        "changed_floating_tensor_count": len(changed_keys),
        "changed_prefixes": sorted(
            {
                ".".join(key.split(".")[:2])
                for key in changed_keys
            }
        ),
        "records": records,
    }
    manifest_path = args.output_dir / f"{args.label}_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
