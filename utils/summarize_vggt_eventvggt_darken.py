#!/usr/bin/env python3
import argparse
import csv
import json
from pathlib import Path


EXPOSURES = ("e020", "e010", "e005")
EXPOSURE_VALUES = {"e020": 0.20, "e010": 0.10, "e005": 0.05}
DATASETS = ("dsec", "mvsec")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, required=True)
    return parser.parse_args()


def metric_row(dataset, method, exposure, payload):
    metrics = payload["dense_metrics"]
    return {
        "dataset": dataset.upper(),
        "method": method,
        "exposure": exposure,
        "raw_absrel": metrics.get("depth_raw_absrel"),
        "raw_delta1": metrics.get("depth_raw_delta_1.25"),
        "median_absrel": metrics.get("depth_median_aligned_absrel"),
        "median_delta1": metrics.get("depth_median_aligned_delta_1.25"),
        "latency_ms": payload.get("latency_ms_median"),
        "peak_memory_gib": (
            payload["peak_memory_bytes"] / 1024**3
            if payload.get("peak_memory_bytes") is not None
            else None
        ),
        "num_pairs": payload.get("num_batches"),
        "input": payload.get("input"),
        "source_json": str(payload.get("_source_json", "")),
    }


def fmt(value):
    return "--" if value is None else f"{value:.4f}"


def main():
    args = parse_args()
    root = args.input_dir
    rows = []
    for dataset in DATASETS:
        for tag in EXPOSURES:
            path = root / f"{dataset}_vggt_{tag}_128.json"
            payload = json.loads(path.read_text(encoding="utf-8"))
            payload["_source_json"] = path
            rows.append(metric_row(dataset, "VGGT", EXPOSURE_VALUES[tag], payload))

        path = root / f"{dataset}_eventvggt_real_128.json"
        payload = json.loads(path.read_text(encoding="utf-8"))
        payload["_source_json"] = path
        rows.append(metric_row(dataset, "EventVGGT", "event-only", payload))

    fieldnames = list(rows[0])
    with (root / "summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    lines = [
        "# VGGT and EventVGGT fixed-128 evaluation",
        "",
        "| Dataset | Method | Exposure/input | Raw AbsRel | Raw delta1 | Median AbsRel | Median delta1 | Latency ms | Peak GiB |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['dataset']} | {row['method']} | {row['exposure']} | "
            f"{fmt(row['raw_absrel'])} | {fmt(row['raw_delta1'])} | "
            f"{fmt(row['median_absrel'])} | {fmt(row['median_delta1'])} | "
            f"{fmt(row['latency_ms'])} | {fmt(row['peak_memory_gib'])} |"
        )
    lines.extend(
        [
            "",
            "VGGT receives the deterministic inverse-gamma/linear-domain darkened RGB used by the existing Night3R controls.",
            "EventVGGT is event-only at inference, so its result is exposure-independent and is evaluated once per dataset.",
            "Night3R 8-bin split-polarity voxels are converted to the official EventVGGT three-channel interface using white-background red/blue occupancy images; see each JSON for the exact adapter metadata.",
        ]
    )
    (root / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
