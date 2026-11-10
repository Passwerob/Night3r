#!/usr/bin/env python3
import argparse
import csv
import json
import math
import re
from pathlib import Path


DATASETS = ("mvsec", "dl3dv")
METRICS = {
    "ate": "ate_translation_m_rmse_scene_mean",
    "rpe_t": "rpe_translation_m_rmse_scene_mean",
    "rpe_r": "rpe_rotation_deg_rmse_scene_mean",
    "ape_r": "ape_rotation_deg_rmse_scene_mean",
}
PRIMARY_METRICS = ("ate", "rpe_t", "rpe_r")


def parse_args():
    parser = argparse.ArgumentParser("Summarize the Night3R rotation-loss sweep")
    parser.add_argument("--eval-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def load_rows(eval_root):
    rows = {}
    pattern = re.compile(r"step(\d+)$")
    for path in sorted(eval_root.glob("total*/step*/*_real.json")):
        total_weight = int(path.parents[1].name.removeprefix("total"))
        match = pattern.fullmatch(path.parent.name)
        if match is None:
            raise RuntimeError(f"Cannot parse step from {path}")
        step = int(match.group(1))
        dataset = path.stem.removesuffix("_real")
        if dataset not in DATASETS:
            continue
        payload = json.loads(path.read_text())
        aggregate = payload["aggregate"]
        key = (total_weight, step)
        row = rows.setdefault(
            key,
            {
                "total_rotation_weight": total_weight,
                "step": step,
            },
        )
        for short_name, source_name in METRICS.items():
            row[f"{dataset}_{short_name}"] = float(aggregate[source_name])
        row[f"{dataset}_recovery"] = float(aggregate["pose_recovery_success_rate"])
        row["checkpoint"] = payload["checkpoint"]["pose_head_checkpoint"]["checkpoint"]

    required = {
        f"{dataset}_{metric}"
        for dataset in DATASETS
        for metric in (*METRICS, "recovery")
    }
    complete = []
    for key, row in sorted(rows.items()):
        missing = sorted(required - set(row))
        if missing:
            raise RuntimeError(f"Incomplete result for {key}: {missing}")
        complete.append(row)
    if not complete:
        raise RuntimeError(f"No rotation sweep results found below {eval_root}")
    return complete


def add_scores(rows):
    primary_rank_keys = []
    all_rank_keys = []
    for dataset in DATASETS:
        for metric in METRICS:
            value_key = f"{dataset}_{metric}"
            rank_key = f"rank_{value_key}"
            ordered = sorted(
                rows,
                key=lambda row: (
                    row[value_key],
                    row["total_rotation_weight"],
                    row["step"],
                ),
            )
            for rank, row in enumerate(ordered, start=1):
                row[rank_key] = float(rank)
            all_rank_keys.append(rank_key)
            if metric in PRIMARY_METRICS:
                primary_rank_keys.append(rank_key)

    for row in rows:
        row["primary_mean_rank"] = sum(row[k] for k in primary_rank_keys) / len(
            primary_rank_keys
        )
        row["all_mean_rank"] = sum(row[k] for k in all_rank_keys) / len(all_rank_keys)
        values = []
        for dataset in DATASETS:
            for metric in PRIMARY_METRICS:
                key = f"{dataset}_{metric}"
                median = sorted(candidate[key] for candidate in rows)[len(rows) // 2]
                values.append(math.log(max(row[key], 1.0e-12) / max(median, 1.0e-12)))
        row["primary_log_score"] = sum(values) / len(values)


def fmt(value):
    return f"{value:.4f}"


def main():
    args = parse_args()
    rows = load_rows(args.eval_root)
    add_scores(rows)
    candidates = [
        row
        for row in rows
        if row["mvsec_recovery"] == 1.0 and row["dl3dv_recovery"] == 1.0
    ] or rows
    best = min(
        candidates,
        key=lambda row: (
            row["primary_mean_rank"],
            row["all_mean_rank"],
            row["primary_log_score"],
        ),
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "total_rotation_weight",
        "step",
        "mvsec_ate",
        "mvsec_rpe_t",
        "mvsec_rpe_r",
        "mvsec_ape_r",
        "mvsec_recovery",
        "dl3dv_ate",
        "dl3dv_rpe_t",
        "dl3dv_rpe_r",
        "dl3dv_ape_r",
        "dl3dv_recovery",
        "primary_mean_rank",
        "all_mean_rank",
        "primary_log_score",
        "checkpoint",
    ]
    with (args.output_dir / "rotation_sweep.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows({key: row.get(key) for key in fieldnames} for row in rows)

    selection = {
        "selection_protocol": "validation only; paper test sets are not used",
        "selection_rule": "lowest mean rank over MVSEC/DL3DV ATE, RPE-t, and RPE-r",
        "num_checkpoints": len(rows),
        "best_total_rotation_weight": best["total_rotation_weight"],
        "best_step": best["step"],
        "best_checkpoint": best["checkpoint"],
        "best_primary_mean_rank": best["primary_mean_rank"],
        "best_all_metric_mean_rank": best["all_mean_rank"],
    }
    (args.output_dir / "selection.json").write_text(
        json.dumps(selection, indent=2, sort_keys=True) + "\n"
    )

    ordered = sorted(
        rows,
        key=lambda row: (row["primary_mean_rank"], row["all_mean_rank"]),
    )
    lines = [
        "# Rotation-Loss Validation Sweep",
        "",
        "The total rotation weight is the near-identity sum of geodesic and chordal weights.",
        "",
        "| Rot. weight | Step | MVSEC ATE | MVSEC RPE-t | MVSEC RPE-r | DL3DV ATE | DL3DV RPE-t | DL3DV RPE-r | Mean rank |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in ordered:
        marker = "**" if row is best else ""
        lines.append(
            f"| {marker}{row['total_rotation_weight']}{marker} | "
            f"{marker}{row['step']}{marker} | {fmt(row['mvsec_ate'])} | "
            f"{fmt(row['mvsec_rpe_t'])} | {fmt(row['mvsec_rpe_r'])} | "
            f"{fmt(row['dl3dv_ate'])} | {fmt(row['dl3dv_rpe_t'])} | "
            f"{fmt(row['dl3dv_rpe_r'])} | {row['primary_mean_rank']:.2f} |"
        )
    lines.extend(
        [
            "",
            f"Selected total rotation weight **{best['total_rotation_weight']}** at step **{best['step']}**.",
        ]
    )
    (args.output_dir / "rotation_sweep.md").write_text("\n".join(lines) + "\n")
    print(json.dumps(selection, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
