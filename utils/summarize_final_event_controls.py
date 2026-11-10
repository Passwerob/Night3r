#!/usr/bin/env python3
import csv
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUTDIR = ROOT / "outputs" / "final_mvsec1k_event_controls_20260715"
DATASETS = ("mvsec", "dsec", "dl3dv")
MODES = ("real", "zero", "shuffle")


def metric(metrics, key):
    value = metrics.get(key)
    return "" if value is None else float(value)


def fmt(value):
    return "--" if value == "" else f"{value:.4f}"


def main():
    rows = []
    missing = []
    for dataset in DATASETS:
        for mode in MODES:
            path = OUTDIR / f"{dataset}_{mode}_128.json"
            if not path.exists():
                missing.append(str(path))
                continue
            payload = json.loads(path.read_text())
            metrics = payload.get("dense_metrics", {}) or {}
            diagnostics = payload.get("event_diagnostics", {}) or {}

            def diagnostic_mean(key):
                value = diagnostics.get(key, {}) or {}
                mean = value.get("mean")
                return "" if mean is None else float(mean)

            rows.append(
                {
                    "dataset": dataset.upper(),
                    "event": mode,
                    "num_batches": payload.get("num_batches", ""),
                    "raw_abs_rel": metric(metrics, "depth_raw_absrel"),
                    "raw_delta1": metric(metrics, "depth_raw_delta_1.25"),
                    "median_abs_rel": metric(metrics, "depth_median_aligned_absrel"),
                    "median_delta1": metric(
                        metrics, "depth_median_aligned_delta_1.25"
                    ),
                    "warp_psnr": metric(metrics, "warp_rgb_psnr_db"),
                    "event_gate_mean": diagnostic_mean("event_gate"),
                    "event_density_mean": diagnostic_mean(
                        "event_stat_summary.event_density_mean"
                    ),
                    "event_occupancy_mean": diagnostic_mean(
                        "event_stat_summary.event_occupancy_mean"
                    ),
                }
            )

    if missing:
        raise SystemExit("Missing result JSONs:\n" + "\n".join(missing))

    fieldnames = list(rows[0])
    with (OUTDIR / "summary.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    header = (
        "| Dataset | Event | N | Raw AbsRel | Raw delta1 | "
        "Median AbsRel | Median delta1 | Warp PSNR | Gate | Density | Occupancy |"
    )
    lines = [
        "# Final MVSEC-1k checkpoint: event-input controls",
        "",
        "Checkpoint: `outputs/night3r_mvsec_only_adapt_1k_from_best/last_event_dark3r.pth`",
        "",
        header,
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['dataset']} | {row['event']} | {row['num_batches']} | "
            f"{fmt(row['raw_abs_rel'])} | {fmt(row['raw_delta1'])} | "
            f"{fmt(row['median_abs_rel'])} | {fmt(row['median_delta1'])} | "
            f"{fmt(row['warp_psnr'])} | {fmt(row['event_gate_mean'])} | "
            f"{fmt(row['event_density_mean'])} | {fmt(row['event_occupancy_mean'])} |"
        )
    (OUTDIR / "summary.md").write_text("\n".join(lines) + "\n")

    latex = [
        r"\begin{tabular}{llrrrrr}",
        r"\toprule",
        r"Dataset & Event & Raw AbsRel $\downarrow$ & Raw $\delta_1\uparrow$ & Median AbsRel $\downarrow$ & Median $\delta_1\uparrow$ & Warp PSNR $\uparrow$ \\",
        r"\midrule",
    ]
    for row in rows:
        latex.append(
            f"{row['dataset']} & {row['event']} & {fmt(row['raw_abs_rel'])} & "
            f"{fmt(row['raw_delta1'])} & {fmt(row['median_abs_rel'])} & "
            f"{fmt(row['median_delta1'])} & {fmt(row['warp_psnr'])} " + r"\\"
        )
    latex.extend([r"\bottomrule", r"\end{tabular}"])
    latex.extend(
        [
            "",
            r"\medskip",
            r"\begin{tabular}{llrrr}",
            r"\toprule",
            r"Dataset & Event & Gate mean & Event density & Event occupancy \\",
            r"\midrule",
        ]
    )
    for row in rows:
        latex.append(
            f"{row['dataset']} & {row['event']} & {fmt(row['event_gate_mean'])} & "
            f"{fmt(row['event_density_mean'])} & {fmt(row['event_occupancy_mean'])} "
            + r"\\"
        )
    latex.extend([r"\bottomrule", r"\end{tabular}"])
    (OUTDIR / "table.tex").write_text("\n".join(latex) + "\n")


if __name__ == "__main__":
    main()
