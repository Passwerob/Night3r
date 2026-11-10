#!/usr/bin/env python3
"""Summarize the DL3DV DA3 pseudo-depth checkpoint sweep."""

import argparse
import csv
import json
import math
from pathlib import Path


LABELS = ("base", "step250", "step500", "step750", "step1000")
DATASETS = {
    "mvsec_gt": "MVSEC GT",
    "dsec_gt": "DSEC GT",
    "dl3dv_da3": "DL3DV held-out DA3 agreement",
}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--labels", nargs="+", default=list(LABELS))
    parser.add_argument("--mvsec-max-regression", type=float, default=0.05)
    parser.add_argument("--dsec-raw-absrel-max", type=float, default=0.15)
    parser.add_argument("--dl3dv-selection-tolerance", type=float, default=0.005)
    return parser.parse_args()


def metric(payload, name):
    value = payload.get("dense_metrics", {}).get(name, float("nan"))
    return float(value) if value is not None else float("nan")


def event_gate_mean(payload):
    diagnostics = payload.get("event_diagnostics", {})
    for key in ("event_gate", "event_gate_summary.mean"):
        value = diagnostics.get(key)
        if isinstance(value, dict) and "mean" in value:
            return float(value["mean"])
    return float("nan")


def fmt(value):
    return "nan" if not math.isfinite(value) else f"{value:.4f}"


def main():
    args = parse_args()
    rows = []
    payloads = {}
    labels = tuple(args.labels)
    for label in labels:
        for dataset_key, dataset_name in DATASETS.items():
            path = args.root / "json" / f"{label}_{dataset_key}.json"
            payload = json.loads(path.read_text())
            payloads[(label, dataset_key)] = payload
            rows.append(
                {
                    "checkpoint": label,
                    "dataset": dataset_name,
                    "depth_reference": payload.get("depth_reference_evaluated"),
                    "num_batches": int(payload.get("num_batches", 0)),
                    "raw_absrel": metric(payload, "depth_raw_absrel"),
                    "raw_delta1": metric(payload, "depth_raw_delta_1.25"),
                    "median_absrel": metric(payload, "depth_median_aligned_absrel"),
                    "median_delta1": metric(payload, "depth_median_aligned_delta_1.25"),
                    "gate_mean": event_gate_mean(payload),
                }
            )

    base = {
        dataset_key: metric(
            payloads[("base", dataset_key)], "depth_median_aligned_absrel"
        )
        for dataset_key in DATASETS
    }
    ranking = []
    for label in labels:
        ratios = []
        values = {}
        for dataset_key in DATASETS:
            value = metric(
                payloads[(label, dataset_key)],
                "depth_median_aligned_absrel",
            )
            values[dataset_key] = value
            if math.isfinite(value) and math.isfinite(base[dataset_key]) and base[dataset_key] > 0:
                ratios.append(value / base[dataset_key])
        mvsec_regression_ratios = []
        for metric_name in ("depth_raw_absrel", "depth_median_aligned_absrel"):
            value = metric(payloads[(label, "mvsec_gt")], metric_name)
            base_value = metric(payloads[("base", "mvsec_gt")], metric_name)
            if math.isfinite(value) and math.isfinite(base_value) and base_value > 0:
                mvsec_regression_ratios.append(value / base_value)
        max_mvsec_regression = max(mvsec_regression_ratios) - 1.0
        dsec_raw_absrel = metric(
            payloads[(label, "dsec_gt")], "depth_raw_absrel"
        )
        guardrail_pass = (
            max_mvsec_regression <= args.mvsec_max_regression
            and dsec_raw_absrel <= args.dsec_raw_absrel_max
        )
        ranking.append(
            {
                "checkpoint": label,
                "tradeoff_score": sum(ratios) / len(ratios),
                "gt_guardrail_pass": guardrail_pass,
                "max_mvsec_regression": max_mvsec_regression,
                "dsec_raw_absrel": dsec_raw_absrel,
                **{f"{key}_median_absrel": value for key, value in values.items()},
            }
        )
    ranking.sort(
        key=lambda item: (
            not item["gt_guardrail_pass"],
            item["dl3dv_da3_median_absrel"],
            item["tradeoff_score"],
        )
    )

    args.root.mkdir(parents=True, exist_ok=True)
    with (args.root / "summary.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    with (args.root / "ranking.json").open("w") as handle:
        json.dump(ranking, handle, indent=2)
        handle.write("\n")

    lines = [
        "# DL3DV DA3 Pseudo-Depth Fine-Tune Sweep",
        "",
        "MVSEC and DSEC use sensor/dataset depth ground truth. DL3DV uses held-out",
        "DA3 teacher predictions from source scenes excluded from pseudo-depth training;",
        "the DL3DV values measure teacher agreement and are not ground-truth accuracy.",
        "",
        "| Checkpoint | Dataset | Ref. | N | Raw AbsRel | Raw delta1 | Median AbsRel | Median delta1 | Gate |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['checkpoint']} | {row['dataset']} | {row['depth_reference']} | "
            f"{row['num_batches']} | {fmt(row['raw_absrel'])} | {fmt(row['raw_delta1'])} | "
            f"{fmt(row['median_absrel'])} | {fmt(row['median_delta1'])} | {fmt(row['gate_mean'])} |"
        )
    lines.extend(
        [
            "",
            "## Checkpoint selection",
            "",
            "Eligible checkpoints are ranked first by held-out DL3DV DA3",
            "median-aligned AbsRel and then by the three-dataset trade-off score.",
            f"The universal checkpoint is selected from models within "
            f"{100.0 * args.dl3dv_selection_tolerance:.1f}% of the best eligible "
            "DL3DV result, using the trade-off score to break the near-tie.",
            "The score is the mean checkpoint/base ratio of median-aligned AbsRel over",
            "MVSEC GT, DSEC GT, and held-out DL3DV DA3 agreement; lower is better",
            "and is retained as a secondary diagnostic.",
            "A checkpoint is eligible for universal use when neither raw nor",
            f"median-aligned AbsRel regresses by more than {100.0 * args.mvsec_max_regression:.1f}% "
            f"on MVSEC GT and DSEC raw AbsRel is at most {args.dsec_raw_absrel_max:.3f}.",
            "",
            "| Rank | Checkpoint | Score | Guardrail | MVSEC max regression | DSEC Raw | MVSEC | DSEC | DL3DV-DA3 |",
            "|---:|---|---:|---|---:|---:|---:|---:|---:|",
        ]
    )
    for rank, row in enumerate(ranking, 1):
        lines.append(
            f"| {rank} | {row['checkpoint']} | {fmt(row['tradeoff_score'])} | "
            f"{'pass' if row['gt_guardrail_pass'] else 'fail'} | "
            f"{100.0 * row['max_mvsec_regression']:+.1f}% | "
            f"{fmt(row['dsec_raw_absrel'])} | "
            f"{fmt(row['mvsec_gt_median_absrel'])} | {fmt(row['dsec_gt_median_absrel'])} | "
            f"{fmt(row['dl3dv_da3_median_absrel'])} |"
        )
    eligible = [item for item in ranking if item["gt_guardrail_pass"]]
    dl3dv_shape = min(
        eligible, key=lambda item: item["dl3dv_da3_median_absrel"]
    )
    near_best_limit = dl3dv_shape["dl3dv_da3_median_absrel"] * (
        1.0 + args.dl3dv_selection_tolerance
    )
    near_best = [
        item
        for item in eligible
        if item["dl3dv_da3_median_absrel"] <= near_best_limit
    ]
    universal = min(near_best, key=lambda item: item["tradeoff_score"])
    lines.extend(
        [
            "",
            f"Universal checkpoint: **{universal['checkpoint']}**.",
            f"Best DL3DV teacher-aligned shape checkpoint: **{dl3dv_shape['checkpoint']}**.",
            (
                "The DL3DV shape winner should not replace the universal model because "
                "it fails the real-GT regression guardrail."
                if not dl3dv_shape["gt_guardrail_pass"]
                else "The DL3DV shape winner also passes the real-GT regression guardrail."
            ),
        ]
    )
    (args.root / "summary.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
