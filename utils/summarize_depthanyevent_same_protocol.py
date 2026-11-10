#!/usr/bin/env python3
import argparse
import csv
import json
import math
from pathlib import Path


CROSSDOMAIN_SOURCES = {
    ("MVSEC", "MASt3R"): "exp5_exposure_curve_20260715/mvsec/mast3r_exposure010_128.json",
    ("MVSEC", "Dark3R"): "darken_cutoff_qualitative_20260716/mvsec_dark3r_e010_128.json",
    ("MVSEC", "Night3R"): "depthanyevent_crossdomain_protocol_20260729/results/mvsec_night3r_final_128.json",
    ("MVSEC", "EventVGGT"): "vggt_eventvggt_darken_fixed128_20260716/mvsec_eventvggt_real_128.json",
    ("MVSEC", "DepthAnyEvent"): "depthanyevent_crossdomain_protocol_20260729/results/mvsec_depthanyevent_synth_128.json",
    ("MVSEC", "DepthAnyEvent-R"): "depthanyevent_crossdomain_protocol_20260729/results/mvsec_depthanyevent_r_synth_128.json",
    ("DSEC", "MASt3R"): "final_mvsec1k_dsec_darken_controls_20260716/dsec_mast3r_e010_128.json",
    ("DSEC", "Dark3R"): "final_mvsec1k_dsec_darken_controls_20260716/dsec_dark3r_e010_128.json",
    ("DSEC", "Night3R"): "depthanyevent_crossdomain_protocol_20260729/results/dsec_night3r_final_128.json",
    ("DSEC", "EventVGGT"): "vggt_eventvggt_darken_fixed128_20260716/dsec_eventvggt_real_128.json",
    ("DSEC", "DepthAnyEvent"): "depthanyevent_crossdomain_protocol_20260729/results/dsec_depthanyevent_synth_128.json",
    ("DSEC", "DepthAnyEvent-R"): "depthanyevent_crossdomain_protocol_20260729/results/dsec_depthanyevent_r_synth_128.json",
}

INPUTS = {
    "MASt3R": "RGB, exposure 0.10",
    "Dark3R": "RGB, exposure 0.10",
    "Night3R": "RGB exposure 0.10 + events",
    "EventVGGT": "events only",
    "DepthAnyEvent": "events only",
    "DepthAnyEvent-R": "events only",
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--outputs-root", type=Path, default=Path("outputs"))
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("outputs/depthanyevent_crossdomain_protocol_20260729"),
    )
    return parser.parse_args()


def fmt(value):
    value = float(value)
    return "nan" if not math.isfinite(value) else f"{value:.4f}"


def checkpoint_path(payload):
    path = payload.get("checkpoint")
    if not path and isinstance(payload.get("load_info"), dict):
        path = payload["load_info"].get("checkpoint")
    return str(path or "")


def main():
    args = parse_args()
    payloads = {
        key: json.loads((args.outputs_root / relative_path).read_text())
        for key, relative_path in CROSSDOMAIN_SOURCES.items()
    }

    # The reported cross-domain methods must use one checkpoint for both domains.
    for method in ("Night3R", "DepthAnyEvent", "DepthAnyEvent-R"):
        checkpoints = {
            checkpoint_path(payloads[(dataset, method)])
            for dataset in ("MVSEC", "DSEC")
        }
        if len(checkpoints) != 1 or not next(iter(checkpoints)):
            raise RuntimeError(
                f"{method} is not a unified cross-domain model: {sorted(checkpoints)}"
            )
    rows = []
    for dataset in ("MVSEC", "DSEC"):
        reference_ids = payloads[(dataset, "DepthAnyEvent-R")]["sample_ids"]
        for method in (
            "MASt3R",
            "Dark3R",
            "Night3R",
            "EventVGGT",
            "DepthAnyEvent",
            "DepthAnyEvent-R",
        ):
            payload = payloads[(dataset, method)]
            if payload.get("sample_ids") != reference_ids:
                raise RuntimeError(f"Sample ID mismatch for {dataset}/{method}")
            metrics = payload["dense_metrics"]
            rows.append(
                {
                    "dataset": dataset,
                    "method": method,
                    "input": INPUTS[method],
                    "n": int(payload["num_batches"]),
                    "raw_absrel": metrics["depth_raw_absrel"],
                    "raw_delta1": metrics["depth_raw_delta_1.25"],
                    "raw_rmse_log": metrics["depth_raw_rmse_log"],
                    "median_absrel": metrics["depth_median_aligned_absrel"],
                    "median_delta1": metrics["depth_median_aligned_delta_1.25"],
                    "median_rmse_log": metrics["depth_median_aligned_rmse_log"],
                }
            )

    args.output_root.mkdir(parents=True, exist_ok=True)
    with (args.output_root / "summary.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    lines = [
        "# Unified cross-domain depth comparison",
        "",
        "All rows use the exact same 128 ordered adjacent pairs and GT masks per dataset.",
        "Night3R uses one final cross-domain checkpoint for both datasets.",
        "DepthAnyEvent and DepthAnyEvent-R each use one official EventScape-only synth checkpoint for both datasets (zero-shot on MVSEC/DSEC).",
        "No target-specific DepthAnyEvent fine-tuned checkpoint appears in this table.",
        "RGB methods use the deterministic exposure-0.10 low-light input; event-only methods are exposure-independent.",
        "Raw metrics use no GT alignment. Median metrics use one independent scale per sample and no shift.",
        "DepthAnyEvent checkpoint GT scale+shift is intentionally disabled for both architectures.",
        "MVSEC cached events span 30 ms; DSEC events span approximately 50 ms. The official DepthAnyEvent training window is 50 ms.",
        "",
        "| Dataset | Method | Input | N | Raw AbsRel | Raw delta1 | Raw RMSElog | Median AbsRel | Median delta1 | Median RMSElog |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['dataset']} | {row['method']} | {row['input']} | {row['n']} | "
            f"{fmt(row['raw_absrel'])} | {fmt(row['raw_delta1'])} | {fmt(row['raw_rmse_log'])} | "
            f"{fmt(row['median_absrel'])} | {fmt(row['median_delta1'])} | "
            f"{fmt(row['median_rmse_log'])} |"
        )
    (args.output_root / "summary.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
