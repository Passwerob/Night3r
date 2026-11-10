#!/usr/bin/env python3
"""Summarize raw depth scale and median-aligned quality from a test sweep."""

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np


METHOD_NAMES = {
    "master": "MASt3R",
    "night": "Night3R",
    "zero": "Night3R-zero",
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sweep-root", type=Path, required=True)
    return parser.parse_args()


def load_records(root):
    records = []
    seen = set()
    for path in sorted((root / "shards").glob("*/all_pairs.jsonl")):
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                item = json.loads(line)
                key = (item["dataset"], item["scene"], item["sample_id"])
                if key in seen:
                    raise RuntimeError(f"Duplicate sweep record: {key}")
                seen.add(key)
                records.append(item)
    if not records:
        raise RuntimeError(f"No all_pairs.jsonl records found under {root}")
    return records


def finite(values):
    return np.asarray(
        [float(value) for value in values if value is not None and math.isfinite(value)],
        dtype=np.float64,
    )


def mean(values):
    values = finite(values)
    return float(values.mean()) if values.size else None


def quantile(values, q):
    values = finite(values)
    return float(np.quantile(values, q)) if values.size else None


def summarize_gt(records):
    grouped = defaultdict(list)
    for record in records:
        if "metrics" not in record:
            continue
        for method, metrics in record["metrics"].items():
            grouped[(record["dataset"], method)].append(metrics)

    rows = []
    for (dataset, method), items in sorted(grouped.items()):
        scales = [item["median_scale"] for item in items]
        rows.append(
            {
                "dataset": dataset,
                "method": method,
                "display_name": METHOD_NAMES.get(method, method),
                "pairs": len(items),
                "raw_absrel_mean": mean(item["raw_absrel"] for item in items),
                "raw_delta1_mean": mean(item["raw_delta1"] for item in items),
                "median_scale_median": quantile(scales, 0.5),
                "median_scale_p10": quantile(scales, 0.1),
                "median_scale_p90": quantile(scales, 0.9),
                "median_absrel_mean": mean(
                    item["median_absrel"] for item in items
                ),
                "median_delta1_mean": mean(
                    item["median_delta1"] for item in items
                ),
            }
        )
    return rows


def summarize_gt_by_scene(records):
    grouped = defaultdict(list)
    for record in records:
        if "metrics" not in record:
            continue
        for method, metrics in record["metrics"].items():
            grouped[(record["dataset"], record["scene"], method)].append(metrics)

    rows = []
    for (dataset, scene, method), items in sorted(grouped.items()):
        scales = [item["median_scale"] for item in items]
        rows.append(
            {
                "dataset": dataset,
                "scene": scene,
                "method": method,
                "display_name": METHOD_NAMES.get(method, method),
                "pairs": len(items),
                "raw_absrel_mean": mean(item["raw_absrel"] for item in items),
                "raw_delta1_mean": mean(item["raw_delta1"] for item in items),
                "median_scale_median": quantile(scales, 0.5),
                "median_scale_p10": quantile(scales, 0.1),
                "median_scale_p90": quantile(scales, 0.9),
                "median_absrel_mean": mean(
                    item["median_absrel"] for item in items
                ),
                "median_delta1_mean": mean(
                    item["median_delta1"] for item in items
                ),
            }
        )
    return rows


def summarize_no_gt(records):
    grouped = defaultdict(list)
    for record in records:
        if "mast3r_vs_night3r" not in record:
            continue
        grouped[record["dataset"]].append(record)

    rows = []
    for dataset, items in sorted(grouped.items()):
        model = [item["mast3r_vs_night3r"] for item in items]
        event = [item["night3r_real_vs_zero"] for item in items]
        rows.append(
            {
                "dataset": dataset,
                "pairs": len(items),
                "reference": "MASt3R prediction, diagnostic only",
                "night_to_mast3r_scale_median": quantile(
                    [item["scale"] for item in model], 0.5
                ),
                "night_to_mast3r_log_difference_mean": mean(
                    item["mean_abs_log"] for item in model
                ),
                "real_to_zero_scale_median": quantile(
                    [item["scale"] for item in event], 0.5
                ),
                "real_to_zero_log_difference_mean": mean(
                    item["mean_abs_log"] for item in event
                ),
                "warning": (
                    "No sensor depth GT. These values measure prediction "
                    "disagreement and are not depth accuracy or absolute scale."
                ),
            }
        )
    return rows


def write_csv(path, rows):
    fields = list(rows[0]) if rows else []
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def fmt(value):
    return "--" if value is None else f"{value:.4f}"


def write_markdown(path, gt_rows, no_gt_rows, record_count):
    lines = [
        "# Test-sweep depth scale summary",
        "",
        f"Processed pair records: {record_count}.",
        "",
        "Raw metrics retain the model's predicted scale. Median-aligned metrics",
        "multiply each pair by `median(GT) / median(prediction)` and therefore",
        "measure relative depth structure rather than absolute scale. A median",
        "scale factor close to 1 is preferable.",
        "",
        "## GT-enabled splits",
        "",
        "| Dataset | Method | Pairs | Raw AbsRel | Raw delta1 | Scale median [p10, p90] | Aligned AbsRel | Aligned delta1 |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in gt_rows:
        scale = (
            f"{fmt(row['median_scale_median'])} "
            f"[{fmt(row['median_scale_p10'])}, {fmt(row['median_scale_p90'])}]"
        )
        lines.append(
            f"| {row['dataset']} | {row['display_name']} | {row['pairs']} | "
            f"{fmt(row['raw_absrel_mean'])} | {fmt(row['raw_delta1_mean'])} | "
            f"{scale} | {fmt(row['median_absrel_mean'])} | "
            f"{fmt(row['median_delta1_mean'])} |"
        )

    lines.extend(
        [
            "",
            "Metrics are pair-macro averages over the sweep sampling protocol.",
            "MVSEC scenes overlap the final adaptation domain and are not strict",
            "held-out evidence. DSEC uses only strict-heldout scenes 3 and 4.",
            "",
            "## No-GT splits",
            "",
            "| Dataset | Pairs | Night/MASt3R scale | Night/MASt3R log difference | Real/zero scale | Real/zero log difference |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for row in no_gt_rows:
        lines.append(
            f"| {row['dataset']} | {row['pairs']} | "
            f"{fmt(row['night_to_mast3r_scale_median'])} | "
            f"{fmt(row['night_to_mast3r_log_difference_mean'])} | "
            f"{fmt(row['real_to_zero_scale_median'])} | "
            f"{fmt(row['real_to_zero_log_difference_mean'])} |"
        )
    lines.extend(
        [
            "",
            "DL3DV has no sensor depth GT in this test split. Its values are",
            "diagnostics only and cannot establish absolute scale or accuracy.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def main():
    args = parse_args()
    records = load_records(args.sweep_root)
    gt_rows = summarize_gt(records)
    gt_scene_rows = summarize_gt_by_scene(records)
    no_gt_rows = summarize_no_gt(records)
    payload = {
        "sweep_root": str(args.sweep_root.resolve()),
        "record_count": len(records),
        "aggregation": "pair-macro mean; pair-level median scale quantiles",
        "gt_enabled": gt_rows,
        "gt_enabled_by_scene": gt_scene_rows,
        "no_gt_diagnostics": no_gt_rows,
    }
    (args.sweep_root / "scale_summary.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )
    write_csv(args.sweep_root / "scale_summary_gt.csv", gt_rows)
    write_csv(args.sweep_root / "scale_summary_gt_by_scene.csv", gt_scene_rows)
    write_csv(args.sweep_root / "scale_summary_no_gt.csv", no_gt_rows)
    write_markdown(
        args.sweep_root / "scale_summary.md",
        gt_rows,
        no_gt_rows,
        len(records),
    )
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
