#!/usr/bin/env python3
"""Merge sharded dynamic reconstruction evaluations into paper-facing summaries."""

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np

METHODS = ("mast3r", "night3r", "night3r_zero")


def parse_args():
    parser = argparse.ArgumentParser("Merge dynamic reconstruction eval shards")
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--output-dir", type=Path, default=None)
    return parser.parse_args()


def load_records(root):
    records = {}
    paths = sorted(root.rglob("records.jsonl"))
    for path in paths:
        if "merged" in path.parts:
            continue
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                record = json.loads(line)
                records[str(record["protocol_id"])] = record
    return list(records.values()), paths


def numeric_mean(records, path):
    values = []
    for record in records:
        value = record
        for key in path:
            if not isinstance(value, dict) or key not in value:
                value = None
                break
            value = value[key]
        if isinstance(value, (int, float)) and math.isfinite(float(value)):
            values.append(float(value))
    return None if not values else float(np.mean(values))


def aggregate(records):
    result = {"num_centers": len(records), "num_scenes": len({r["scene"] for r in records})}
    result["methods"] = {}
    for method in METHODS:
        temporal = {}
        accuracy = {}
        temporal_keys = (
            "raw_mean_abs_log",
            "raw_p90_abs_log",
            "raw_delta1_agreement",
            "raw_symmetric_absrel",
            "scale_drift_abs_log",
            "median_mean_abs_log",
            "median_p90_abs_log",
            "median_delta1_agreement",
        )
        accuracy_keys = (
            "raw_absrel",
            "raw_delta1",
            "median_absrel",
            "median_delta1",
        )
        for key in temporal_keys:
            temporal[key] = numeric_mean(
                records, ("methods", method, "temporal", key)
            )
        for key in accuracy_keys:
            accuracy[key] = numeric_mean(
                records, ("methods", method, "accuracy", "mean", key)
            )
        result["methods"][method] = {
            "temporal": temporal,
            "accuracy": accuracy,
        }
    for key in (
        "night3r_vs_mast3r_temporal_gain",
        "real_vs_zero_temporal_gain",
        "night3r_vs_mast3r_accuracy_gain",
        "real_vs_zero_accuracy_gain",
    ):
        result[key] = numeric_mean(records, (key,))
    result["event_activity_mean"] = numeric_mean(records, ("event_activity_mean",))
    result["event_occupancy"] = numeric_mean(records, ("event_occupancy",))
    return result


def event_buckets(records):
    if len(records) < 4:
        return {}
    activity = np.asarray([float(record["event_activity_mean"]) for record in records])
    q25, q75 = np.quantile(activity, (0.25, 0.75))
    return {
        "low_event_q25": aggregate(
            [
                record
                for record in records
                if float(record["event_activity_mean"]) <= q25
            ]
        ),
        "high_event_q75": aggregate(
            [
                record
                for record in records
                if float(record["event_activity_mean"]) >= q75
            ]
        ),
        "thresholds": {"q25": float(q25), "q75": float(q75)},
    }


def fmt(value):
    return "--" if value is None else f"{value:.4f}"


def summary_rows(groups):
    rows = []
    for key, item in sorted(groups.items()):
        dataset, split, reference = key
        aggregate_item = item["aggregate"]
        for method in METHODS:
            temporal = aggregate_item["methods"][method]["temporal"]
            accuracy = aggregate_item["methods"][method]["accuracy"]
            rows.append(
                {
                    "dataset": dataset,
                    "split": split,
                    "reference": reference,
                    "method": method,
                    "num_centers": aggregate_item["num_centers"],
                    "num_scenes": aggregate_item["num_scenes"],
                    "temporal_raw_abs_log": temporal["raw_mean_abs_log"],
                    "temporal_median_abs_log": temporal["median_mean_abs_log"],
                    "temporal_median_delta1": temporal[
                        "median_delta1_agreement"
                    ],
                    "scale_drift_abs_log": temporal["scale_drift_abs_log"],
                    "raw_absrel": accuracy["raw_absrel"],
                    "raw_delta1": accuracy["raw_delta1"],
                    "median_absrel": accuracy["median_absrel"],
                    "median_delta1": accuracy["median_delta1"],
                }
            )
    return rows


def write_markdown(path, groups, rows, total_records, shard_paths):
    lines = [
        "# Dynamic Reconstruction Evaluation",
        "",
        (
            "Each center frame `t` is independently reconstructed with temporal "
            "contexts `(t,t-1)` and `(t,t+1)`, with `t` used as view 1 in both "
            "pairs. Lower temporal AbsLog and scale drift indicate more stable "
            "center-frame geometry; higher temporal delta1 indicates better agreement."
        ),
        "",
        (
            "MVSEC/DSEC sensor depth prevents a constant-depth predictor from appearing "
            "successful through consistency alone. DL3DV held-out has no depth reference. "
            "Chunk016 uses DA3 pseudo depth and is a training-source diagnostic, not a "
            "held-out benchmark. No dynamic-object mask is available, so this protocol "
            "evaluates sequence-level temporal-context robustness rather than isolated "
            "non-rigid object reconstruction."
        ),
        "",
        f"- Records: {total_records}",
        f"- Shards: {len(shard_paths)}",
        "",
        "| Dataset | Split | Ref. | Method | N | Temp AbsLog | Temp delta1 | Scale drift | Raw AbsRel | Median AbsRel |",
        "|---|---|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            "| {dataset} | {split} | {reference} | {method} | {num_centers} | "
            "{temp} | {delta} | {scale} | {raw} | {median} |".format(
                dataset=row["dataset"],
                split=row["split"],
                reference=row["reference"],
                method=row["method"],
                num_centers=row["num_centers"],
                temp=fmt(row["temporal_median_abs_log"]),
                delta=fmt(row["temporal_median_delta1"]),
                scale=fmt(row["scale_drift_abs_log"]),
                raw=fmt(row["raw_absrel"]),
                median=fmt(row["median_absrel"]),
            )
        )
    lines.extend(["", "## Night3R Gains", ""])
    lines.append(
        "| Dataset | Split | N | Temporal gain vs MASt3R | Event temporal gain vs zero | Accuracy gain vs MASt3R | Event accuracy gain vs zero |"
    )
    lines.append("|---|---|---:|---:|---:|---:|---:|")
    for (dataset, split, _), item in sorted(groups.items()):
        aggregate_item = item["aggregate"]
        lines.append(
            "| {dataset} | {split} | {n} | {tg} | {eg} | {ag} | {eag} |".format(
                dataset=dataset,
                split=split,
                n=aggregate_item["num_centers"],
                tg=fmt(aggregate_item["night3r_vs_mast3r_temporal_gain"]),
                eg=fmt(aggregate_item["real_vs_zero_temporal_gain"]),
                ag=fmt(aggregate_item["night3r_vs_mast3r_accuracy_gain"]),
                eag=fmt(aggregate_item["real_vs_zero_accuracy_gain"]),
            )
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    args = parse_args()
    output_dir = args.output_dir or args.root / "merged"
    output_dir.mkdir(parents=True, exist_ok=True)
    records, shard_paths = load_records(args.root)
    if not records:
        raise RuntimeError(f"No records.jsonl files found under {args.root}")
    grouped_records = defaultdict(list)
    for record in records:
        key = (
            record["dataset"],
            record["split_label"],
            record["reference_kind"],
        )
        grouped_records[key].append(record)
    groups = {}
    for key, items in sorted(grouped_records.items()):
        groups[key] = {
            "aggregate": aggregate(items),
            "event_buckets": event_buckets(items),
        }
    rows = summary_rows(groups)
    json_payload = {
        "protocol": "bidirectional_center_frame_dynamic_reconstruction",
        "num_records": len(records),
        "shards": [str(path.resolve()) for path in shard_paths],
        "groups": {
            "|".join(key): value for key, value in groups.items()
        },
    }
    (output_dir / "summary.json").write_text(
        json.dumps(json_payload, indent=2), encoding="utf-8"
    )
    with (output_dir / "summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    write_markdown(
        output_dir / "summary.md",
        groups,
        rows,
        len(records),
        shard_paths,
    )
    (output_dir / "MERGE_COMPLETE").touch()
    print(json.dumps(json_payload, indent=2))


if __name__ == "__main__":
    main()
