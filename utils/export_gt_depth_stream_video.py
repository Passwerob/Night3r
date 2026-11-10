#!/usr/bin/env python3
"""Export a shared metric-ground-truth depth stream for a display scene."""

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.export_multimethod_scene_media import make_dataset, save_video, to_numpy


def gt_depth_frames(depths, frames):
    valid_values = np.concatenate([
        depth[np.isfinite(depth) & (depth > 0)].reshape(-1) for depth in depths
    ])
    lower, upper = np.quantile(valid_values, [0.02, 0.98])
    upper = max(float(upper), float(lower) + 1.0e-6)
    output = []
    for index in range(frames):
        depth = depths[min(int(round(index * (len(depths) - 1) / max(frames - 1, 1))), len(depths) - 1)]
        valid = np.isfinite(depth) & (depth > 0)
        normalized = np.clip((depth - lower) / (upper - lower), 0.0, 1.0)
        colored = cv2.cvtColor(cv2.applyColorMap((normalized * 255).astype(np.uint8), cv2.COLORMAP_TURBO), cv2.COLOR_BGR2RGB)
        colored[~valid] = 0
        output.append(colored)
    return output, {"vmin_m": float(lower), "vmax_m": float(upper)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene-dir", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--scene", required=True)
    parser.add_argument("--frame-indices", required=True)
    parser.add_argument("--frames", type=int, default=180)
    parser.add_argument("--fps", type=int, default=60)
    args = parser.parse_args()
    args.image_size = 512
    indices = [int(value) for value in args.frame_indices.split(",")]
    pair_file = args.scene_dir / "sequence_pairs_depth_gt.json"
    pair_file.write_text(json.dumps([{"idx1": a, "idx2": b} for a, b in zip(indices, indices[1:])]))
    dataset, _ = make_dataset(args, pair_file)
    batches = list(torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0))
    if not batches or "depth_gt1" not in batches[0]:
        raise RuntimeError(f"{args.scene}: metric depth is unavailable")
    depths = [to_numpy(batches[0]["depth_gt1"][0])]
    depths.extend(to_numpy(batch["depth_gt2"][0]) for batch in batches)
    frames, depth_range = gt_depth_frames(depths, args.frames)
    shared = args.scene_dir / "shared_inputs"
    shared.mkdir(parents=True, exist_ok=True)
    save_video(frames, shared / "depth_gt.mp4", args.fps)
    (shared / "depth_gt_metadata.json").write_text(json.dumps({
        "kind": "metric ground-truth depth",
        "frame_indices": indices,
        "colormap": "turbo",
        "shared_valid_depth_range_m": depth_range,
        "alignment": "none",
    }, indent=2))


if __name__ == "__main__":
    main()
