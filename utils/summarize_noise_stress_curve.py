#!/usr/bin/env python3
"""Summarize paired low-light noise-stress sweeps."""

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--sweep",
        nargs=3,
        action="append",
        metavar=("LEVEL", "READ_STD", "ROOT"),
        required=True,
    )
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args()


def load_records(root):
    records = []
    for path in sorted(root.glob("shards/*/all_pairs.jsonl")):
        with path.open("r", encoding="utf-8") as handle:
            records.extend(json.loads(line) for line in handle if line.strip())
    return records


def mean(records, getter):
    values = [getter(record) for record in records]
    values = [float(value) for value in values if value is not None]
    return float(np.mean(values)) if values else None


def aggregate(level, read_std, root):
    grouped = defaultdict(list)
    for record in load_records(root):
        grouped[record["dataset"]].append(record)
    rows = []
    for dataset, records in sorted(grouped.items()):
        base = {
            "level": level,
            "read_noise_std": float(read_std),
            "dataset": dataset,
            "pairs": len(records),
            "lowlight_mean": mean(records, lambda item: item["lowlight_mean"]),
            "lowlight_noise": mean(records, lambda item: item["lowlight_noise"]),
            "gate_mean": mean(records, lambda item: item["event_gate_mean"]),
        }
        if "metrics" in records[0]:
            for method in ("master", "night", "zero"):
                row = dict(base)
                row["method"] = {
                    "master": "MASt3R",
                    "night": "Night3R-real",
                    "zero": "Night3R-zero",
                }[method]
                row["raw_absrel"] = mean(
                    records, lambda item, m=method: item["metrics"][m]["raw_absrel"]
                )
                row["raw_delta1"] = mean(
                    records, lambda item, m=method: item["metrics"][m]["raw_delta1"]
                )
                row["aligned_absrel"] = mean(
                    records,
                    lambda item, m=method: item["metrics"][m]["median_absrel"],
                )
                row["aligned_delta1"] = mean(
                    records,
                    lambda item, m=method: item["metrics"][m]["median_delta1"],
                )
                row["scale_median"] = float(
                    np.median(
                        [
                            item["metrics"][method]["median_scale"]
                            for item in records
                        ]
                    )
                )
                rows.append(row)
        else:
            row = dict(base)
            row["method"] = "Night3R/MASt3R diagnostic"
            row["raw_absrel"] = None
            row["raw_delta1"] = None
            row["aligned_absrel"] = mean(
                records,
                lambda item: item["mast3r_vs_night3r"]["mean_abs_log"],
            )
            row["aligned_delta1"] = None
            row["scale_median"] = float(
                np.median(
                    [
                        item["mast3r_vs_night3r"]["scale"]
                        for item in records
                    ]
                )
            )
            rows.append(row)
    return rows


def fmt(value):
    return "--" if value is None else f"{value:.4f}"


def main():
    args = parse_args()
    all_rows = []
    roots = {}
    for level, read_std, root_text in args.sweep:
        root = Path(root_text).resolve()
        if not (root / "COMPLETE").is_file():
            raise RuntimeError(f"Incomplete sweep: {root}")
        roots[level] = str(root)
        all_rows.extend(aggregate(level, read_std, root))

    lines = [
        "# Linear-domain noise stress curve",
        "",
        "All levels use identical exposures, test pairs, checkpoint, and random",
        "seed. Events are unchanged. Only the linear-domain read and shot-noise",
        "scales differ.",
        "",
        "| Level | Dataset | Method | Pairs | Mean RGB | Noise | Raw AbsRel | Raw delta1 | Scale | Aligned AbsRel | Aligned delta1 |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in all_rows:
        lines.append(
            "| {level} | {dataset} | {method} | {pairs} | {mean} | {noise} | "
            "{raw_absrel} | {raw_delta1} | {scale} | {aligned_absrel} | "
            "{aligned_delta1} |".format(
                level=row["level"],
                dataset=row["dataset"],
                method=row["method"],
                pairs=row["pairs"],
                mean=fmt(row["lowlight_mean"]),
                noise=fmt(row["lowlight_noise"]),
                raw_absrel=fmt(row["raw_absrel"]),
                raw_delta1=fmt(row["raw_delta1"]),
                scale=fmt(row["scale_median"]),
                aligned_absrel=fmt(row["aligned_absrel"]),
                aligned_delta1=fmt(row["aligned_delta1"]),
            )
        )
    lines.extend(
        [
            "",
            "DL3DV has no sensor depth GT. Its aligned-AbsRel column contains the",
            "scale-invariant Night3R/MASt3R log-depth disagreement and is not an",
            "accuracy metric.",
            "",
        ]
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(lines), encoding="utf-8")
    args.output.with_suffix(".json").write_text(
        json.dumps({"roots": roots, "rows": all_rows}, indent=2),
        encoding="utf-8",
    )
    print(json.dumps({"output": str(args.output.resolve()), "rows": len(all_rows)}))


if __name__ == "__main__":
    main()
