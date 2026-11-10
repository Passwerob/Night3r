#!/usr/bin/env python3
"""Expand 4 temporal event bins into a higher-rate display video.

The model tensor is [positive bins (4), negative bins (4)].  The exported
event video intentionally shows one positive/negative bin pair per display
sample instead of collapsing all four bins into one frame.
"""

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from dataset.dsec_event_dataset import DSECEventDataset


def render_event_bin(voxel, bin_index):
    if voxel.shape[0] != 8:
        raise ValueError(f"Expected 8 event channels, got {tuple(voxel.shape)}")
    pos = np.abs(voxel[bin_index]) > 1.0e-8
    neg = np.abs(voxel[4 + bin_index]) > 1.0e-8
    out = np.zeros((*pos.shape, 3), dtype=np.uint8)
    out[pos & ~neg] = (235, 40, 40)
    out[~pos & neg] = (45, 105, 245)
    out[pos & neg] = (245, 245, 245)
    return out


def save_video(frames, path, fps):
    frame_dir = path.parent / (path.stem + "_frames")
    frame_dir.mkdir(parents=True, exist_ok=True)
    for i, frame in enumerate(frames):
        cv2.imwrite(str(frame_dir / f"frame_{i:05d}.png"), cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    subprocess.run(
        [
            "ffmpeg", "-y", "-loglevel", "error", "-framerate", str(fps),
            "-i", str(frame_dir / "frame_%05d.png"), "-c:v", "libx264",
            "-preset", "slow", "-crf", "18", "-pix_fmt", "yuv420p",
            "-movflags", "+faststart", str(path),
        ],
        check=True,
    )
    shutil.rmtree(frame_dir)


def collect_events(data_root, scene, frame_indices, image_size=(512, 512)):
    outdir = Path("/tmp") / f"event_expand_{scene.replace('/', '_')}"
    outdir.mkdir(parents=True, exist_ok=True)
    pair_file = outdir / "pairs.json"
    pair_file.write_text(json.dumps([
        {"idx1": first, "idx2": second}
        for first, second in zip(frame_indices, frame_indices[1:])
    ]))
    dataset = DSECEventDataset(
        root_path=data_root,
        image_size=image_size,
        pair_sampling="precomputed",
        precomputed_pairs=str(pair_file),
        pose_output_convention="opencv",
        lowlight_enabled=False,
    )
    expected = {(first, second) for first, second in zip(frame_indices, frame_indices[1:])}
    dataset.samples = sorted(
        [
            sample for sample in dataset.samples
            if sample["scene"] == scene
            and (sample["idx1"], sample["idx2"]) in expected
        ],
        key=lambda sample: (sample["idx1"], sample["idx2"]),
    )
    if len(dataset.samples) != len(frame_indices) - 1:
        raise RuntimeError(
            f"{scene}: expected {len(frame_indices)-1} pairs, got {len(dataset.samples)}"
        )
    loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)
    batches = list(loader)
    events = [batches[0]["event_voxel"][0, 0].numpy()]
    events.extend(batch["event_voxel"][0, 1].numpy() for batch in batches)
    return events


def expand_scene(scene_dir, data_root, scene, frame_indices, frames, fps, shape):
    events = collect_events(data_root, scene, frame_indices)
    expanded = []
    for voxel in events:
        for bin_index in range(4):
            rendered = render_event_bin(voxel, bin_index)
            expanded.append(cv2.resize(rendered, (shape[1], shape[0]), interpolation=cv2.INTER_NEAREST))
    output_frames = [
        expanded[min(int(round(index * (len(expanded) - 1) / max(frames - 1, 1))), len(expanded) - 1)]
        for index in range(frames)
    ]
    shared = scene_dir / "shared_inputs"
    shared.mkdir(parents=True, exist_ok=True)
    save_video(output_frames, shared / "event_stream.mp4", fps)
    for method in ("mast3r", "dark3r", "vggt", "night3r"):
        method_dir = scene_dir / method
        if method_dir.is_dir():
            shutil.copy2(shared / "event_stream.mp4", method_dir / "event_stream.mp4")
    manifest = scene_dir / "manifest.json"
    if manifest.exists():
        payload = json.loads(manifest.read_text())
        payload["event_stream"] = {
            "channels": 8,
            "temporal_bins": 4,
            "display_policy": "one positive/negative bin pair per display sample",
            "fps": fps,
            "frames": frames,
        }
        manifest.write_text(json.dumps(payload, indent=2) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene-dir", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--scene", required=True)
    parser.add_argument("--frame-indices", required=True)
    parser.add_argument("--frames", type=int, default=180)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--width", type=int, default=346)
    parser.add_argument("--height", type=int, default=260)
    args = parser.parse_args()
    indices = [int(value) for value in args.frame_indices.split(",")]
    expand_scene(
        args.scene_dir,
        args.data_root,
        args.scene,
        indices,
        args.frames,
        args.fps,
        (args.height, args.width),
    )


if __name__ == "__main__":
    main()
