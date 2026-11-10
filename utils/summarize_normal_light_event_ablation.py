#!/usr/bin/env python3
"""Summarize the predefined native-RGB event-control experiment."""

import argparse
import csv
import json
from pathlib import Path


PRIMARY_SPLITS = ("dsec_003", "dsec_004", "dsec_extended")
AUXILIARY_SPLITS = ("mvsec_day",)
MODES = ("real", "zero", "shuffle")
FUSION_VARIANTS = ("full_brg", "concat_mlp", "additive")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    return parser.parse_args()


def load_json(path):
    with path.open() as handle:
        return json.load(handle)


def value(payload, key):
    return payload.get("dense_metrics", {}).get(key)


def finite_or_blank(item):
    return "" if item is None else f"{float(item):.4f}"


def metric_row(scope, condition, payload):
    diagnostics = payload.get("event_diagnostics", {})
    return {
        "scope": scope,
        "condition": condition,
        "pairs": payload.get("num_batches"),
        "raw_absrel": value(payload, "depth_raw_absrel"),
        "raw_delta1": value(payload, "depth_raw_delta_1.25"),
        "median_absrel": value(payload, "depth_median_aligned_absrel"),
        "median_delta1": value(payload, "depth_median_aligned_delta_1.25"),
        "gate_mean": diagnostics.get("event_gate_summary.mean", {}).get("mean"),
    }


def write_csv(path, rows):
    fields = [
        "scope",
        "condition",
        "pairs",
        "raw_absrel",
        "raw_delta1",
        "median_absrel",
        "median_delta1",
        "gate_mean",
    ]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main():
    args = parse_args()
    root = args.root.resolve()
    final_payloads = {}
    rows = []
    final_splits = list(PRIMARY_SPLITS)
    for split in AUXILIARY_SPLITS:
        expected = [root / "final_controls" / split / f"{mode}.json" for mode in MODES]
        if all(path.is_file() for path in expected):
            final_splits.append(split)
    for split in final_splits:
        for mode in MODES:
            path = root / "final_controls" / split / f"{mode}.json"
            if not path.is_file():
                raise FileNotFoundError(path)
            payload = load_json(path)
            final_payloads[(split, mode)] = payload
            rows.append(metric_row(split, mode, payload))

    fusion_rows = []
    for name in FUSION_VARIANTS:
        path = root / "fusion_ablation" / "dsec_003" / f"{name}.json"
        if not path.is_file():
            raise FileNotFoundError(path)
        fusion_rows.append(metric_row("dsec_003", name, load_json(path)))

    write_csv(root / "normal_light_event_controls.csv", rows)
    write_csv(root / "normal_light_fusion_ablation.csv", fusion_rows)

    lines = [
        "# Native-RGB Event Ablation Results",
        "",
        "The DSEC rows use native daytime RGB, real synchronized events, fixed adjacent pairs, and no synthetic darkening or noise.",
        "`mvsec_day` is an auxiliary native-daylight check only: its scenes overlap the broad MVSEC source root used by the final adaptation stage and are not held-out.",
        "",
        "## Final Model: Event-Input Controls",
        "",
        "| Split | Input | Pairs | Raw AbsRel | Raw delta1 | Median AbsRel | Median delta1 | Gate mean |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            "| {scope} | {condition} | {pairs} | {raw_absrel} | {raw_delta1} | {median_absrel} | {median_delta1} | {gate_mean} |".format(
                **{key: finite_or_blank(item) if key not in {"scope", "condition", "pairs"} else item for key, item in row.items()}
            )
        )

    lines.extend(
        [
            "",
            "## Real-Event Gains",
            "",
            "Positive gain means that the Real condition is better. These are differences of aggregate metrics, not confidence intervals.",
            "",
            "| Split | Comparison | Raw AbsRel gain | Raw delta1 gain | Median AbsRel gain | Median delta1 gain |",
            "|---|---|---:|---:|---:|---:|",
        ]
    )
    for split in final_splits:
        real = metric_row(split, "real", final_payloads[(split, "real")])
        for control in ("zero", "shuffle"):
            baseline = metric_row(split, control, final_payloads[(split, control)])
            fields = (
                baseline["raw_absrel"] - real["raw_absrel"],
                real["raw_delta1"] - baseline["raw_delta1"],
                baseline["median_absrel"] - real["median_absrel"],
                real["median_delta1"] - baseline["median_delta1"],
            )
            lines.append(
                f"| {split} | Real - {control} | "
                + " | ".join(finite_or_blank(item) for item in fields)
                + " |"
            )

    lines.extend(
        [
            "",
            "## Matched Fusion Variants on Native DSEC RGB",
            "",
            "| Variant | Pairs | Raw AbsRel | Raw delta1 | Median AbsRel | Median delta1 | Gate mean |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in fusion_rows:
        lines.append(
            "| {condition} | {pairs} | {raw_absrel} | {raw_delta1} | {median_absrel} | {median_delta1} | {gate_mean} |".format(
                **{key: finite_or_blank(item) if key not in {"scope", "condition", "pairs"} else item for key, item in row.items()}
            )
        )

    (root / "normal_light_event_ablation_summary.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
