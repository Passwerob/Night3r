#!/usr/bin/env python3
"""Find a gamma-only darkening exponent for a target mean intensity."""

import argparse
import json
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from dataset.dsec_event_dataset import DSECEventDataset  # noqa: E402
from utils.sweep_all_test_visual_candidates import selected_indices  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-name", required=True)
    parser.add_argument("--data-root", required=True, type=Path)
    parser.add_argument("--scenes", nargs="+", required=True)
    parser.add_argument("--pair-subsample-stride", type=int, default=1)
    parser.add_argument("--target-mean", type=float, required=True)
    parser.add_argument("--image-size", nargs=2, type=int, default=[512, 512])
    parser.add_argument("--statistics-size", type=int, default=64)
    parser.add_argument("--gamma-min", type=float, default=1.0)
    parser.add_argument("--gamma-max", type=float, default=12.0)
    parser.add_argument("--iterations", type=int, default=32)
    parser.add_argument(
        "--report-gammas",
        nargs="+",
        type=float,
        default=[1.4, 1.6, 2.0, 2.5, 3.0, 4.0, 5.0, 6.0, 8.0, 12.0],
    )
    return parser.parse_args()


def load_grayscale_samples(dataset, indices, size):
    samples = []
    for dataset_index in indices:
        item = dataset.samples[dataset_index]
        frames = dataset.scene_frames[item["scene_dir"]]
        for frame_index in (int(item["idx1"]), int(item["idx2"])):
            image = dataset._load_image_01(frames["images"][frame_index])
            gray = image.mean(dim=0, keepdim=True).unsqueeze(0)
            gray = F.interpolate(
                gray, size=(size, size), mode="area"
            ).squeeze(0).squeeze(0)
            samples.append(gray.flatten())
    return torch.cat(samples)


def main():
    args = parse_args()
    if not 0 < args.target_mean < 1:
        raise ValueError("--target-mean must be in (0, 1)")
    if args.gamma_min < 1 or args.gamma_max <= args.gamma_min:
        raise ValueError("Invalid gamma search range")

    dataset = DSECEventDataset(
        root_path=str(args.data_root),
        image_size=tuple(args.image_size),
        pair_stride=1,
        pair_sampling="fixed",
        expected_event_channels=8,
        lowlight_enabled=False,
    )
    indices = selected_indices(
        dataset, args.scenes, args.pair_subsample_stride
    )
    values = load_grayscale_samples(dataset, indices, args.statistics_size)
    lo = float(args.gamma_min)
    hi = float(args.gamma_max)
    with torch.inference_mode():
        original_mean = float(values.mean())
        original_quantiles = {
            f"p{int(q * 100):02d}": float(torch.quantile(values, q))
            for q in (0.10, 0.25, 0.50, 0.75, 0.90)
        }
        gamma_curve = [
            {
                "gamma": exponent,
                "mean": float(values.pow(exponent).mean()),
                "median": float(torch.quantile(values.pow(exponent), 0.50)),
                "p75": float(torch.quantile(values.pow(exponent), 0.75)),
            }
            for exponent in args.report_gammas
        ]
        for _ in range(args.iterations):
            midpoint = (lo + hi) * 0.5
            mean = float(values.pow(midpoint).mean())
            if mean > args.target_mean:
                lo = midpoint
            else:
                hi = midpoint
        gamma = (lo + hi) * 0.5
        achieved = float(values.pow(gamma).mean())

    print(
        json.dumps(
            {
                "dataset": args.dataset_name,
                "data_root": str(args.data_root.resolve()),
                "scenes": args.scenes,
                "pair_count": len(indices),
                "image_observations": len(indices) * 2,
                "statistics_size": args.statistics_size,
                "original_mean": original_mean,
                "original_quantiles": original_quantiles,
                "target_mean": args.target_mean,
                "gamma_exponent": gamma,
                "achieved_mean": achieved,
                "absolute_error": abs(achieved - args.target_mean),
                "gamma_curve": gamma_curve,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
