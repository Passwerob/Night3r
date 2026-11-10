#!/usr/bin/env python3
import argparse
import csv
import json
import math
import re
from pathlib import Path


METRICS = {
    "ate": "ate_translation_m_rmse_scene_mean",
    "rpe_t": "rpe_translation_m_rmse_scene_mean",
    "rpe_r": "rpe_rotation_deg_rmse_scene_mean",
    "ape_r": "ape_rotation_deg_rmse_scene_mean",
}
SELECTED_METRICS = ("ate", "rpe_t", "rpe_r")
DATASETS = ("mvsec", "dl3dv")


def parse_args():
    parser = argparse.ArgumentParser("Summarize a pose-head validation sweep")
    parser.add_argument("--eval-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def checkpoint_step(path):
    match = re.search(r"step(\d+)", path.parent.name)
    if match is None:
        raise ValueError(f"Cannot parse checkpoint step from {path}")
    return int(match.group(1))


def load_rows(eval_root):
    rows_by_step = {}
    for dataset in DATASETS:
        for path in sorted(eval_root.glob(f"step*/{dataset}_real.json")):
            step = checkpoint_step(path)
            payload = json.loads(path.read_text())
            aggregate = payload["aggregate"]
            row = rows_by_step.setdefault(step, {"step": step})
            for short_name, source_name in METRICS.items():
                row[f"{dataset}_{short_name}"] = float(aggregate[source_name])
            row[f"{dataset}_recovery"] = float(
                aggregate["pose_recovery_success_rate"]
            )
            pose_info = payload.get("checkpoint", {}).get("pose_head_checkpoint", {})
            row["checkpoint"] = pose_info.get("checkpoint")

    rows = []
    required = {
        f"{dataset}_{metric}"
        for dataset in DATASETS
        for metric in (*SELECTED_METRICS, "ape_r", "recovery")
    }
    for step, row in sorted(rows_by_step.items()):
        missing = sorted(required - set(row))
        if missing:
            raise RuntimeError(f"Incomplete validation results at step {step}: {missing}")
        rows.append(row)
    if not rows:
        raise RuntimeError(f"No sweep JSON files found below {eval_root}")
    return rows


def add_ranks(rows):
    rank_keys = []
    translation_rank_keys = []
    log_terms = {row["step"]: [] for row in rows}
    for dataset in DATASETS:
        for metric in SELECTED_METRICS:
            value_key = f"{dataset}_{metric}"
            rank_key = f"rank_{value_key}"
            ordered = sorted(rows, key=lambda row: (row[value_key], row["step"]))
            for rank, row in enumerate(ordered, start=1):
                row[rank_key] = float(rank)
            values = sorted(row[value_key] for row in rows)
            median = values[len(values) // 2]
            for row in rows:
                log_terms[row["step"]].append(
                    math.log(max(row[value_key], 1.0e-12) / max(median, 1.0e-12))
                )
            rank_keys.append(rank_key)
            if metric in {"ate", "rpe_t"}:
                translation_rank_keys.append(rank_key)

    for row in rows:
        row["balanced_mean_rank"] = sum(row[key] for key in rank_keys) / len(rank_keys)
        row["translation_mean_rank"] = sum(
            row[key] for key in translation_rank_keys
        ) / len(translation_rank_keys)
        row["balanced_log_score"] = sum(log_terms[row["step"]]) / len(
            log_terms[row["step"]]
        )


def fmt(value):
    return f"{value:.4f}"


def main():
    args = parse_args()
    rows = load_rows(args.eval_root)
    add_ranks(rows)
    valid_rows = [
        row
        for row in rows
        if row["mvsec_recovery"] == 1.0 and row["dl3dv_recovery"] == 1.0
    ]
    candidates = valid_rows or rows
    best_balanced = min(
        candidates,
        key=lambda row: (
            row["balanced_mean_rank"],
            row["balanced_log_score"],
            row["step"],
        ),
    )
    best_translation = min(
        candidates,
        key=lambda row: (
            row["translation_mean_rank"],
            row["balanced_log_score"],
            row["step"],
        ),
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    fieldnames = [
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
        "balanced_mean_rank",
        "translation_mean_rank",
        "balanced_log_score",
        "checkpoint",
    ]
    with (args.output_dir / "validation_sweep.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows({key: row.get(key) for key in fieldnames} for row in rows)

    selection = {
        "selection_protocol": "validation only; paper test sets are not used",
        "selection_rule": "lowest mean rank over MVSEC/DL3DV ATE, RPE-t, and RPE-r",
        "num_checkpoints": len(rows),
        "best_step": best_balanced["step"],
        "best_checkpoint": best_balanced["checkpoint"],
        "best_balanced_mean_rank": best_balanced["balanced_mean_rank"],
        "best_translation_step": best_translation["step"],
        "best_translation_checkpoint": best_translation["checkpoint"],
        "best_translation_mean_rank": best_translation["translation_mean_rank"],
    }
    (args.output_dir / "selection.json").write_text(
        json.dumps(selection, indent=2, sort_keys=True) + "\n"
    )

    ordered = sorted(rows, key=lambda row: (row["balanced_mean_rank"], row["step"]))
    lines = [
        "# Pose-Head Validation Sweep",
        "",
        "Selection uses only the dedicated validation split. Lower mean rank is better.",
        "",
        "| Step | MVSEC ATE | MVSEC RPE-t | MVSEC RPE-r | DL3DV ATE | DL3DV RPE-t | DL3DV RPE-r | Mean rank |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in ordered:
        marker = " **" if row["step"] == best_balanced["step"] else ""
        end_marker = "**" if marker else ""
        lines.append(
            f"| {marker}{row['step']}{end_marker} | {fmt(row['mvsec_ate'])} | "
            f"{fmt(row['mvsec_rpe_t'])} | {fmt(row['mvsec_rpe_r'])} | "
            f"{fmt(row['dl3dv_ate'])} | {fmt(row['dl3dv_rpe_t'])} | "
            f"{fmt(row['dl3dv_rpe_r'])} | {row['balanced_mean_rank']:.2f} |"
        )
    lines.extend(
        [
            "",
            f"Balanced selection: step **{best_balanced['step']}**.",
            f"Translation-only diagnostic: step **{best_translation['step']}**.",
        ]
    )
    (args.output_dir / "validation_sweep.md").write_text("\n".join(lines) + "\n")
    print(json.dumps(selection, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
