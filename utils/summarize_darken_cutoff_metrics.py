#!/usr/bin/env python3
import argparse
import csv
import json
from pathlib import Path


DATASETS = ("dsec", "mvsec")
EXPOSURES = (("e020", 0.20), ("e010", 0.10), ("e005", 0.05))
METHODS = ("night3r", "vggt")
DISPLAY_NAMES = {"night3r": "Night3R", "vggt": "VGGT", "eventvggt": "EventVGGT"}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, required=True)
    return parser.parse_args()


def load_row(path, dataset, method, exposure):
    payload = json.loads(path.read_text(encoding="utf-8"))
    dense = payload["dense_metrics"]
    cutoff = payload["depth_cutoff_metrics"]
    return {
        "dataset": dataset.upper(),
        "method": DISPLAY_NAMES[method],
        "exposure_or_input": exposure,
        "raw_absrel": dense.get("depth_raw_absrel"),
        "median_absrel": dense.get("depth_median_aligned_absrel"),
        "median_delta1": dense.get("depth_median_aligned_delta_1.25"),
        "raw_mae_10m": cutoff.get("depth_raw_mae_at_10m"),
        "raw_mae_20m": cutoff.get("depth_raw_mae_at_20m"),
        "raw_mae_30m": cutoff.get("depth_raw_mae_at_30m"),
        "median_mae_10m": cutoff.get("depth_median_aligned_mae_at_10m"),
        "median_mae_20m": cutoff.get("depth_median_aligned_mae_at_20m"),
        "median_mae_30m": cutoff.get("depth_median_aligned_mae_at_30m"),
        "num_pairs": payload.get("num_batches"),
        "source_json": str(path),
    }


def fmt(value):
    return "--" if value is None else f"{value:.4f}"


def main():
    args = parse_args()
    root = args.input_dir
    rows = []
    for dataset in DATASETS:
        for tag, exposure in EXPOSURES:
            for method in METHODS:
                path = root / f"{dataset}_{method}_{tag}_128.json"
                rows.append(load_row(path, dataset, method, exposure))
        path = root / f"{dataset}_eventvggt_real_128.json"
        rows.append(load_row(path, dataset, "eventvggt", "event-only"))

    with (root / "cutoff_summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    lines = [
        "# Darken depth and cutoff metrics",
        "",
        "All RGB methods use deterministic darken inputs. EventVGGT is event-only and is therefore exposure-independent.",
        "Median alignment uses the full valid pair before applying each GT-distance cutoff.",
        "",
        "| Dataset | Method | Exposure/input | Median AbsRel | Median delta1 | Median MAE@10m | Median MAE@20m | Median MAE@30m |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['dataset']} | {row['method']} | {row['exposure_or_input']} | "
            f"{fmt(row['median_absrel'])} | {fmt(row['median_delta1'])} | "
            f"{fmt(row['median_mae_10m'])} | {fmt(row['median_mae_20m'])} | "
            f"{fmt(row['median_mae_30m'])} |"
        )
    (root / "cutoff_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    latex = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\small",
        r"\begin{tabular}{lllrrrrr}",
        r"\toprule",
        r"Dataset & Method & Exposure/Input & Median AbsRel $\downarrow$ & Median $\delta_1$ $\uparrow$ & MAE@10m $\downarrow$ & MAE@20m $\downarrow$ & MAE@30m $\downarrow$ \\",
        r"\midrule",
    ]
    for row in rows:
        exposure = row["exposure_or_input"]
        exposure_text = f"{exposure:.2f}" if isinstance(exposure, float) else "Event-only"
        latex.append(
            f"{row['dataset']} & {row['method']} & {exposure_text} & "
            f"{fmt(row['median_absrel'])} & {fmt(row['median_delta1'])} & "
            f"{fmt(row['median_mae_10m'])} & {fmt(row['median_mae_20m'])} & "
            f"{fmt(row['median_mae_30m'])} " + r"\\"
        )
    latex.extend(
        [
            r"\bottomrule",
            r"\end{tabular}",
            r"\caption{Depth accuracy under deterministic synthetic low light. RGB methods receive inverse-gamma linearized, exposure-scaled RGB with shot/read noise. EventVGGT uses only the synchronized event input and is exposure-independent. MAE cutoffs retain pixels whose ground-truth depth does not exceed 10, 20, or 30 meters after per-pair median alignment.}",
            r"\label{tab:darken-cutoff-depth}",
            r"\end{table*}",
        ]
    )
    (root / "cutoff_table.tex").write_text("\n".join(latex) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
