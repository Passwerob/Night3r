#!/usr/bin/env python3
"""Rank held-out MVSEC trajectory candidates by curve visibility and tracking."""

import argparse
import csv
import json
from pathlib import Path

import numpy as np


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--dataset-name", default="MVSEC")
    return parser.parse_args()


def metric(payload, key):
    value = (payload.get("aggregate") or {}).get(key)
    return None if value is None else float(value)


def trajectory_shape(trajectory):
    positions = np.asarray(trajectory["gt_camera_to_world"], dtype=np.float64)[:, :3, 3]
    node_indices = np.unique(np.linspace(0, len(positions) - 1, min(10, len(positions)), dtype=int))
    nodes = positions[node_indices]
    steps = np.diff(nodes, axis=0)
    travel = float(np.linalg.norm(steps, axis=1).sum())
    chord_vector = nodes[-1] - nodes[0]
    chord = float(np.linalg.norm(chord_vector))
    direction = chord_vector / max(chord, 1.0e-12)
    lateral = np.linalg.norm(np.cross(nodes - nodes[0], direction), axis=1)
    max_lateral = float(lateral.max())
    return {
        "trajectory_frames": len(positions),
        "display_nodes": len(nodes),
        "path_length_m": travel,
        "endpoint_distance_m": chord,
        "path_to_chord_ratio": travel / max(chord, 1.0e-12),
        "max_lateral_deviation_m": max_lateral,
        "max_lateral_ratio": max_lateral / max(chord, 1.0e-12),
    }


def main():
    args = parse_args()
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for path in sorted(input_dir.glob("screen-*.json")):
        payload = json.loads(path.read_text())
        scenes = payload.get("scenes") or []
        if len(scenes) != 1 or not scenes[0].get("trajectory"):
            continue
        scene = scenes[0]
        row = {"scene": scene["scene"]}
        row.update(trajectory_shape(scene["trajectory"]))
        row["ate_m"] = metric(payload, "ate_translation_m_rmse_scene_mean")
        row["rpe_t_m"] = metric(payload, "rpe_translation_m_rmse_scene_mean")
        row["rpe_r_deg"] = metric(payload, "rpe_rotation_deg_rmse_scene_mean")
        row["success_rate"] = metric(payload, "pose_recovery_success_rate")
        row["screening_score"] = row["max_lateral_ratio"] / max(row["ate_m"] or 1.0e6, 1.0e-6)
        rows.append(row)
    if not rows:
        raise RuntimeError(f"No usable trajectory JSONs in {input_dir}")
    rows.sort(key=lambda row: row["screening_score"], reverse=True)
    with (output_dir / "ranked_candidates.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    lines = [
        f"# Held-out {args.dataset_name} Trajectory Candidate Screening",
        "",
        f"All scenes are from `test/{args.dataset_name}`. This ranking is solely for choosing a curved qualitative trajectory; it is not a reported benchmark or a model-selection result.",
        "",
    ]
    for index, row in enumerate(rows, start=1):
        lines.append(
            f"{index}. `{row['scene']}`: ATE={row['ate_m']:.4f} m, "
            f"RPE-t={row['rpe_t_m']:.4f} m, RPE-R={row['rpe_r_deg']:.4f} deg, "
            f"lateral ratio={row['max_lateral_ratio']:.3f}, score={row['screening_score']:.3f}."
        )
    (output_dir / "ranked_candidates.md").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
