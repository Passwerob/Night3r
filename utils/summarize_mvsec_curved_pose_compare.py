#!/usr/bin/env python3
"""Rank fair Night3R/MASt3R curved-trajectory comparisons."""

import argparse
import csv
import json
from pathlib import Path

import numpy as np


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--names", nargs="+", required=True)
    parser.add_argument("--night3r-jsons", nargs="+", required=True)
    parser.add_argument("--mast3r-jsons", nargs="+", required=True)
    parser.add_argument("--scene-prefix", default="screen-long-")
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args()


def scene_result(path, scene):
    payload = json.loads(Path(path).read_text())
    matches = [entry for entry in payload.get("scenes", []) if entry.get("scene") == scene]
    if len(matches) != 1:
        raise RuntimeError(f"Expected one scene {scene!r} in {path}, found {len(matches)}")
    result = matches[0]
    trajectory = result.get("trajectory")
    if not trajectory:
        raise RuntimeError(f"No exported trajectory for {scene!r} in {path}")
    return result, trajectory


def metric(result, name):
    value = (result.get("metrics") or {}).get(name, {}).get("rmse")
    return None if value is None else float(value)


def path_statistics(poses):
    centers = np.asarray(poses, dtype=np.float64)[:, :3, 3]
    steps = np.linalg.norm(np.diff(centers, axis=0), axis=1)
    path_length = float(np.sum(steps))
    displacement = float(np.linalg.norm(centers[-1] - centers[0]))
    return path_length, displacement, path_length / max(displacement, 1.0e-9)


def ratio(numerator, denominator):
    if numerator is None or denominator is None or denominator <= 0.0:
        return None
    return numerator / denominator


def main():
    args = parse_args()
    if not (len(args.names) == len(args.night3r_jsons) == len(args.mast3r_jsons)):
        raise SystemExit("Names and JSON lists must have equal length")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    records = []
    for name, night_path, mast_path in zip(args.names, args.night3r_jsons, args.mast3r_jsons):
        scene = f"{args.scene_prefix}{name}-mvsec"
        try:
            night, night_traj = scene_result(night_path, scene)
            mast, mast_traj = scene_result(mast_path, scene)
            night_frames = np.asarray(night_traj["frame_indices"], dtype=np.int64)
            mast_frames = np.asarray(mast_traj["frame_indices"], dtype=np.int64)
            gt_night = np.asarray(night_traj["gt_camera_to_world"], dtype=np.float64)
            gt_mast = np.asarray(mast_traj["gt_camera_to_world"], dtype=np.float64)
            same_protocol = bool(
                np.array_equal(night_frames, mast_frames)
                and gt_night.shape == gt_mast.shape
                and np.allclose(gt_night, gt_mast, atol=1.0e-6, rtol=1.0e-6)
            )
            path_length, displacement, curvature = path_statistics(gt_night)
            night_ate = metric(night, "ate_translation_m")
            mast_ate = metric(mast, "ate_translation_m")
            night_rpe_t = metric(night, "rpe_translation_m")
            mast_rpe_t = metric(mast, "rpe_translation_m")
            night_rpe_r = metric(night, "rpe_rotation_deg")
            mast_rpe_r = metric(mast, "rpe_rotation_deg")
            all_metrics = all(
                value is not None
                for value in (night_ate, mast_ate, night_rpe_t, mast_rpe_t, night_rpe_r, mast_rpe_r)
            )
            night_better_all = bool(
                all_metrics
                and night_ate < mast_ate
                and night_rpe_t < mast_rpe_t
                and night_rpe_r < mast_rpe_r
            )
            record = {
                "name": name,
                "scene": scene,
                "valid_protocol": same_protocol,
                "num_frames": int(len(night_frames)),
                "gt_path_m": path_length,
                "gt_displacement_m": displacement,
                "gt_path_over_displacement": curvature,
                "night3r_ate_m": night_ate,
                "mast3r_ate_m": mast_ate,
                "night3r_rpe_t_m": night_rpe_t,
                "mast3r_rpe_t_m": mast_rpe_t,
                "night3r_rpe_r_deg": night_rpe_r,
                "mast3r_rpe_r_deg": mast_rpe_r,
                "ate_ratio_night_over_mast": ratio(night_ate, mast_ate),
                "rpe_t_ratio_night_over_mast": ratio(night_rpe_t, mast_rpe_t),
                "rpe_r_ratio_night_over_mast": ratio(night_rpe_r, mast_rpe_r),
                "night3r_better_all": night_better_all,
                "night3r_json": str(Path(night_path).resolve()),
                "mast3r_json": str(Path(mast_path).resolve()),
            }
        except Exception as exc:  # Keep the sweep auditable when a baseline cannot recover a path.
            record = {"name": name, "scene": scene, "valid_protocol": False, "error": str(exc)}
        records.append(record)

    def rank_key(item):
        return (
            not item.get("valid_protocol", False),
            not item.get("night3r_better_all", False),
            -(item.get("gt_path_over_displacement") or 0.0),
            item.get("ate_ratio_night_over_mast") or float("inf"),
        )

    records.sort(key=rank_key)
    fields = sorted({key for record in records for key in record})
    with (output_dir / "summary.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(records)
    (output_dir / "summary.json").write_text(json.dumps(records, indent=2) + "\n")
    lines = ["# Curved MVSEC Trajectory Comparison Sweep", ""]
    lines.append("Candidates use three consecutive source chunks (147 real frames).")
    lines.append("A valid candidate requires exactly matching evaluation frames and GT poses.")
    lines.append("")
    lines.append("| Candidate | Frames | GT curve | Night3R ATE | MASt3R ATE | Night3R RPE-t | MASt3R RPE-t | Night3R RPE-R | MASt3R RPE-R | Better all |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|:---:|")
    for record in records:
        if not record.get("valid_protocol"):
            lines.append(f"| {record['name']} | -- | -- | -- | -- | -- | -- | -- | -- | no |")
            continue
        lines.append(
            "| {name} | {num_frames} | {gt_path_over_displacement:.2f} | "
            "{night3r_ate_m:.3f} | {mast3r_ate_m:.3f} | {night3r_rpe_t_m:.3f} | "
            "{mast3r_rpe_t_m:.3f} | {night3r_rpe_r_deg:.3f} | {mast3r_rpe_r_deg:.3f} | {better} |".format(
                better="yes" if record["night3r_better_all"] else "no", **record
            )
        )
    (output_dir / "summary.md").write_text("\n".join(lines) + "\n")
    print(json.dumps(records, indent=2))


if __name__ == "__main__":
    main()
