#!/usr/bin/env python3
"""Summarize strict held-out MVSEC trajectory candidates for visualization."""

import argparse
import csv
import json
from pathlib import Path

import numpy as np


METHODS = ("night3r", "mast3r", "dark3r", "vggt")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--candidate-names", nargs="+", required=True)
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args()


def metric(scene, group):
    value = ((scene.get("metrics") or {}).get(group) or {}).get("rmse")
    return None if value is None else float(value)


def gt_turn_angle_deg(trajectory):
    positions = np.asarray(trajectory["gt_camera_to_world"], dtype=np.float64)[:, :3, 3]
    steps = np.diff(positions, axis=0)
    lengths = np.linalg.norm(steps, axis=1)
    valid = lengths > 1e-7
    steps = steps[valid]
    if len(steps) < 2:
        return 0.0
    unit = steps / np.linalg.norm(steps, axis=1, keepdims=True)
    cosine = np.clip(np.sum(unit[:-1] * unit[1:], axis=1), -1.0, 1.0)
    return float(np.degrees(np.arccos(cosine)).sum())


def longest_contiguous_run(frame_sets):
    common = sorted(set.intersection(*frame_sets))
    if not common:
        return []
    longest = current = [common[0]]
    for frame in common[1:]:
        if frame == current[-1] + 1:
            current.append(frame)
        else:
            if len(current) > len(longest):
                longest = current
            current = [frame]
    return current if len(current) > len(longest) else longest


def fmt(value):
    return "--" if value is None else f"{value:.4f}"


def main():
    args = parse_args()
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    report = []
    for candidate in args.candidate_names:
        loaded = {}
        for method in METHODS:
            path = input_dir / f"{candidate}_{method}.json"
            if not path.is_file():
                raise FileNotFoundError(path)
            payload = json.loads(path.read_text())
            scenes = payload.get("scenes") or []
            if len(scenes) != 1:
                raise RuntimeError(f"{path} must contain exactly one scene")
            loaded[method] = scenes[0]

        trajectory = loaded["night3r"].get("trajectory")
        if trajectory is None:
            raise RuntimeError(f"{candidate}: Night3R trajectory was not exported")
        frame_count = len(trajectory["frame_indices"])
        frame_sets = []
        identical_frames = True
        expected_frames = np.asarray(trajectory["frame_indices"])
        for method in METHODS:
            current = loaded[method].get("trajectory")
            if current is None:
                frame_sets = []
                identical_frames = False
                break
            current_frames = np.asarray(current["frame_indices"])
            frame_sets.append(set(current_frames.tolist()))
            if not np.array_equal(expected_frames, current_frames):
                identical_frames = False
        common_run = longest_contiguous_run(frame_sets) if frame_sets else []

        row = {
            "candidate": candidate,
            "frames": frame_count,
            "gt_total_turn_deg": gt_turn_angle_deg(trajectory),
            "same_frames_all_methods": identical_frames,
            "longest_common_run_frames": len(common_run),
            "common_run_start": None if not common_run else common_run[0],
            "common_run_end": None if not common_run else common_run[-1],
        }
        for method in METHODS:
            row[f"{method}_ate"] = metric(loaded[method], "ate")
            row[f"{method}_rpe_t"] = metric(loaded[method], "rpe_translation")
            row[f"{method}_rpe_r"] = metric(loaded[method], "rpe_rotation")
        baselines = [row[f"{method}_ate"] for method in METHODS[1:] if row[f"{method}_ate"] is not None]
        row["ate_margin_to_best_baseline"] = (
            min(baselines) - row["night3r_ate"]
            if row["night3r_ate"] is not None and baselines
            else None
        )
        rows.append(row)

    fieldnames = list(rows[0])
    with (output_dir / "summary.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    for row in rows:
        report.append(
            " | ".join(
                [
                    row["candidate"],
                    f"frames={row['frames']}",
                    f"GT turn={row['gt_total_turn_deg']:.1f} deg",
                    f"same_frames={row['same_frames_all_methods']}",
                    f"common run={row['longest_common_run_frames']} ({row['common_run_start']}-{row['common_run_end']})",
                    f"Night3R ATE={fmt(row['night3r_ate'])}",
                    f"best-baseline margin={fmt(row['ate_margin_to_best_baseline'])}",
                    f"MASt3R={fmt(row['mast3r_ate'])}",
                    f"Dark3R={fmt(row['dark3r_ate'])}",
                    f"VGGT={fmt(row['vggt_ate'])}",
                ]
            )
        )
    (output_dir / "summary.md").write_text(
        "# Strict Held-out MVSEC Trajectory Sweep\n\n"
        "All candidates are assembled exclusively from `test/MVSEC`. ATE uses the evaluator's per-scene Sim(3) alignment; the visual selection additionally requires identical recovered frame indices for all methods.\n\n"
        + "\n".join(f"- {line}" for line in report)
        + "\n"
    )


if __name__ == "__main__":
    main()
