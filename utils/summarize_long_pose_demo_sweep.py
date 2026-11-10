#!/usr/bin/env python3
"""Rank fixed long-trajectory demonstration candidates without changing data splits."""

import argparse
import csv
import json
from pathlib import Path

import numpy as np


def parse_args():
    parser = argparse.ArgumentParser("Summarize a fixed long-pose demonstration sweep")
    parser.add_argument("--inputs", nargs="+", required=True)
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args()


def value(metrics, key):
    section = metrics.get(key) or {}
    result = section.get("rmse")
    if result is None:
        raise RuntimeError(f"Missing {key}.rmse")
    return float(result)


def main():
    args = parse_args()
    rows = []
    for path_text in args.inputs:
        path = Path(path_text)
        payload = json.loads(path.read_text())
        if len(payload.get("scenes", [])) != 1:
            raise RuntimeError(f"Expected exactly one scene in {path}")
        scene = payload["scenes"][0]
        trajectory = scene.get("trajectory") or {}
        gt = np.asarray(trajectory.get("gt_camera_to_world"), dtype=np.float64)
        if gt.ndim != 3 or gt.shape[1:] != (4, 4):
            raise RuntimeError(f"Missing exported GT trajectory in {path}")
        path_length = float(np.linalg.norm(np.diff(gt[:, :3, 3], axis=0), axis=1).sum())
        metrics = scene.get("metrics") or {}
        ate = value(metrics, "ate_translation_m")
        rpe_t = value(metrics, "rpe_translation_m")
        rpe_r = value(metrics, "rpe_rotation_deg")
        rows.append(
            {
                "candidate": path.stem,
                "scene": scene["scene"],
                "frames": int(scene["num_frames"]),
                "path_length_m": path_length,
                "ate_m": ate,
                "ate_over_path": ate / max(path_length, 1.0e-9),
                "rpe_t_m": rpe_t,
                "rpe_r_deg": rpe_r,
                "source_json": str(path.resolve()),
            }
        )
    rows.sort(key=lambda row: (row["ate_over_path"], row["rpe_t_m"], row["rpe_r_deg"]))
    for rank, row in enumerate(rows, start=1):
        row["rank"] = rank
    outdir = Path(args.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)
    fields = (
        "rank",
        "candidate",
        "scene",
        "frames",
        "path_length_m",
        "ate_m",
        "ate_over_path",
        "rpe_t_m",
        "rpe_r_deg",
        "source_json",
    )
    with (outdir / "summary.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    lines = [
        "# Fixed Long-Trajectory Demo Sweep",
        "",
        "Candidates were fixed before inference from one continuous MVSEC source stream, "
        "using four consecutive exported chunks per candidate. Ranking uses normalized "
        "ATE, then RPE-t and RPE-R. This ranks qualitative demonstrations and is not a "
        "replacement for a pre-registered benchmark aggregate.",
        "",
        "| Rank | Candidate | Frames | GT path (m) | ATE (m) | ATE/path | RPE-t (m) | RPE-R (deg) |",
        "|---:|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['rank']} | {row['candidate']} | {row['frames']} | "
            f"{row['path_length_m']:.2f} | {row['ate_m']:.3f} | "
            f"{row['ate_over_path']:.4f} | {row['rpe_t_m']:.3f} | {row['rpe_r_deg']:.3f} |"
        )
    (outdir / "summary.md").write_text("\n".join(lines) + "\n")
    selection = {
        "selection_rule": "lowest ATE/path, then RPE-t, then RPE-R",
        "claim_scope": "qualitative long-trajectory demonstration only",
        "selected": rows[0],
        "candidates": rows,
    }
    (outdir / "selection.json").write_text(json.dumps(selection, indent=2) + "\n")
    print(json.dumps(selection, indent=2))


if __name__ == "__main__":
    main()
