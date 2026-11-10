#!/usr/bin/env python3
"""Export a normal RGB stream alongside event/geometry display videos."""

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import cv2
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from dataset.dsec_event_dataset import DSECEventDataset
from utils.export_multimethod_scene_media import lowlight_batch


def save_video(frames, path, fps):
    frame_dir = path.parent / (path.stem + "_frames")
    frame_dir.mkdir(parents=True, exist_ok=True)
    for i, frame in enumerate(frames):
        frame_u8 = frame
        if frame_u8.dtype != "uint8":
            frame_u8 = (frame_u8.clip(0.0, 1.0) * 255.0 + 0.5).astype("uint8")
        cv2.imwrite(str(frame_dir / f"frame_{i:05d}.png"), cv2.cvtColor(frame_u8, cv2.COLOR_RGB2BGR))
    subprocess.run([
        "ffmpeg", "-y", "-loglevel", "error", "-framerate", str(fps),
        "-i", str(frame_dir / "frame_%05d.png"), "-c:v", "libx264",
        "-preset", "slow", "-crf", "18", "-pix_fmt", "yuv420p",
        "-movflags", "+faststart", str(path),
    ], check=True)
    shutil.rmtree(frame_dir)


def collect_rgb(
    data_root, scene, indices, exposure, read_noise_std, shot_noise_std,
    lowlight_seed, denoise_sigma,
):
    temp = Path("/tmp") / f"rgb_stream_{scene.replace('/', '_')}"
    temp.mkdir(parents=True, exist_ok=True)
    pair_file = temp / "pairs.json"
    pair_file.write_text(json.dumps([
        {"idx1": first, "idx2": second}
        for first, second in zip(indices, indices[1:])
    ]))
    dataset = DSECEventDataset(
        root_path=data_root,
        image_size=(512, 512),
        pair_sampling="precomputed",
        precomputed_pairs=str(pair_file),
        pose_output_convention="opencv",
        lowlight_enabled=False,
    )
    expected = {(first, second) for first, second in zip(indices, indices[1:])}
    dataset.samples = sorted([
        sample for sample in dataset.samples
        if sample["scene"] == scene and (sample["idx1"], sample["idx2"]) in expected
    ], key=lambda sample: (sample["idx1"], sample["idx2"]))
    if len(dataset.samples) != len(indices) - 1:
        raise RuntimeError(f"{scene}: expected {len(indices)-1} pairs, got {len(dataset.samples)}")
    batches = list(torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0))
    lowlight_args = SimpleNamespace(
        exposure=exposure,
        read_noise_std=read_noise_std,
        shot_noise_std=shot_noise_std,
        lowlight_seed=lowlight_seed,
    )
    dark_batches = [lowlight_batch(batch, lowlight_args) for batch in batches]
    images = [dark_batches[0]["images_photo_dark"][0, 0].numpy().transpose(1, 2, 0)]
    images.extend(batch["images_photo_dark"][0, 1].numpy().transpose(1, 2, 0) for batch in dark_batches)
    if denoise_sigma > 0.0:
        # Mildly suppress dark-frame speckle without touching geometry videos.
        filtered = []
        for image in images:
            image_u8 = (image.clip(0.0, 1.0) * 255.0 + 0.5).astype("uint8")
            smooth = cv2.bilateralFilter(
                image_u8, d=5, sigmaColor=float(denoise_sigma), sigmaSpace=3.0
            )
            filtered.append(smooth.astype("float32") / 255.0)
        images = filtered
    return images


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene-dir", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--scene", required=True)
    parser.add_argument("--frame-indices", required=True)
    parser.add_argument("--frames", type=int, default=180)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--exposure", type=float, default=0.12)
    parser.add_argument("--read-noise-std", type=float, default=0.01)
    parser.add_argument("--shot-noise-std", type=float, default=0.01)
    parser.add_argument("--lowlight-seed", type=int, default=20260716)
    parser.add_argument("--denoise-sigma", type=float, default=0.0)
    parser.add_argument("--width", type=int, default=346)
    parser.add_argument("--height", type=int, default=260)
    args = parser.parse_args()
    indices = [int(value) for value in args.frame_indices.split(",")]
    source = collect_rgb(
        args.data_root, args.scene, indices, args.exposure,
        args.read_noise_std, args.shot_noise_std, args.lowlight_seed,
        args.denoise_sigma,
    )
    frames = [
        cv2.resize(source[min(int(round(index * (len(source) - 1) / max(args.frames - 1, 1))), len(source) - 1)],
                    (args.width, args.height), interpolation=cv2.INTER_AREA)
        for index in range(args.frames)
    ]
    shared = args.scene_dir / "shared_inputs"
    shared.mkdir(parents=True, exist_ok=True)
    save_video(frames, shared / "rgb_stream.mp4", args.fps)
    for method in ("mast3r", "dark3r", "vggt", "night3r"):
        method_dir = args.scene_dir / method
        if method_dir.is_dir():
            shutil.copy2(shared / "rgb_stream.mp4", method_dir / "rgb_stream.mp4")


if __name__ == "__main__":
    main()
