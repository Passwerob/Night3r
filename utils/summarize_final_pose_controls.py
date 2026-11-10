#!/usr/bin/env python3
import csv
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUTDIR = ROOT / "outputs" / "final_mvsec1k_pose_controls_20260715"
TAG = "full_6scene32_bidirectional"
DATASETS = ("mvsec", "dl3dv")
METHODS = ("mast3r", "real", "zero", "shuffle")


def result_path(dataset, method):
    if method == "mast3r":
        name = f"{dataset}_mast3r_baseline_{TAG}.json"
    else:
        name = f"{dataset}_{method}_{TAG}.json"
    return OUTDIR / name


def fmt(value, digits=4):
    if value is None or value == "":
        return "--"
    return f"{float(value):.{digits}f}"


def main():
    rows = []
    for dataset in DATASETS:
        for method in METHODS:
            path = result_path(dataset, method)
            if not path.is_file():
                raise SystemExit(f"Missing pose result: {path}")
            payload = json.loads(path.read_text())
            aggregate = payload["aggregate"]
            rows.append(
                {
                    "dataset": dataset.upper(),
                    "method": "MASt3R" if method == "mast3r" else method.capitalize(),
                    "num_scenes": aggregate["num_scenes"],
                    "num_frames": aggregate["num_frames"],
                    "ate_rmse_scene_mean_m": aggregate[
                        "ate_translation_m_rmse_scene_mean"
                    ],
                    "ate_rmse_scene_median_m": aggregate[
                        "ate_translation_m_rmse_scene_median"
                    ],
                    "rpe_translation_rmse_scene_mean_m": aggregate[
                        "rpe_translation_m_rmse_scene_mean"
                    ],
                    "rpe_translation_rmse_scene_median_m": aggregate[
                        "rpe_translation_m_rmse_scene_median"
                    ],
                    "rpe_rotation_rmse_scene_mean_deg": aggregate[
                        "rpe_rotation_deg_rmse_scene_mean"
                    ],
                    "rpe_rotation_rmse_scene_median_deg": aggregate[
                        "rpe_rotation_deg_rmse_scene_median"
                    ],
                    "registration_residual_normalized_3d": aggregate[
                        "geometric_residual_mean"
                    ],
                    "pose_recovery_success_rate": aggregate[
                        "pose_recovery_success_rate"
                    ],
                }
            )

    with (OUTDIR / "pose_summary.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    lines = [
        "# Final MVSEC-1k checkpoint: trajectory controls",
        "",
        "- Checkpoint: `outputs/night3r_mvsec_only_adapt_1k_from_best/last_event_dark3r.pth`",
        "- Protocol: 6 scenes x 32 frames, adjacent pairs, bidirectional point-map Sim(3), unit edge translation, per-scene EVO Sim(3) alignment.",
        "- All runs use the same fixed seed 17 for deterministic sampling; no seed sweep is included.",
        "",
        "| Dataset | Input/model | ATE RMSE mean (m) | ATE RMSE median (m) | RPE trans. mean (m) | RPE rot. mean (deg) | 3D residual | Recovery |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['dataset']} | {row['method']} | "
            f"{fmt(row['ate_rmse_scene_mean_m'])} | "
            f"{fmt(row['ate_rmse_scene_median_m'])} | "
            f"{fmt(row['rpe_translation_rmse_scene_mean_m'])} | "
            f"{fmt(row['rpe_rotation_rmse_scene_mean_deg'], 2)} | "
            f"{fmt(row['registration_residual_normalized_3d'])} | "
            f"{100.0 * row['pose_recovery_success_rate']:.1f}% |"
        )
    lines.extend(
        [
            "",
            "DSEC is omitted because the local DSEC test export has no camera trajectory ground truth.",
        ]
    )
    (OUTDIR / "pose_summary.md").write_text("\n".join(lines) + "\n")

    latex = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\caption{Trajectory accuracy for the final MVSEC-only 1k checkpoint. Relative poses are recovered by robust registration of bidirectional point maps over six scenes with 32 frames per scene. Per-edge translation is normalized to resolve monocular scale, and ATE/RPE are computed after one per-scene Sim(3) alignment. Values are means of per-scene RMSE; lower is better.}",
        r"\label{tab:event-control-pose}",
        r"\begin{tabular}{llrrr}",
        r"\toprule",
        r"Dataset & Input/model & ATE (m) $\downarrow$ & RPE trans. (m) $\downarrow$ & RPE rot. ($^\circ$) $\downarrow$ "
        + r"\\",
        r"\midrule",
    ]
    for dataset in ("MVSEC", "DL3DV"):
        dataset_rows = [row for row in rows if row["dataset"] == dataset]
        for index, row in enumerate(dataset_rows):
            prefix = rf"\multirow{{4}}{{*}}{{{dataset}}}" if index == 0 else ""
            best = row["method"] == "MASt3R"
            values = (
                fmt(row["ate_rmse_scene_mean_m"]),
                fmt(row["rpe_translation_rmse_scene_mean_m"]),
                fmt(row["rpe_rotation_rmse_scene_mean_deg"], 2),
            )
            if best:
                values = tuple(rf"\textbf{{{value}}}" for value in values)
            latex.append(
                f"{prefix} & {row['method']} & {values[0]} & {values[1]} & {values[2]} "
                + r"\\"
            )
        if dataset == "MVSEC":
            latex.append(r"\midrule")
    latex.extend(
        [
            r"\midrule",
            r"DSEC & \multicolumn{4}{l}{Not available: the local test export has no camera-pose ground truth.} "
            + r"\\",
            r"\bottomrule",
            r"\end{tabular}",
            r"\end{table*}",
        ]
    )
    (OUTDIR / "pose_table.tex").write_text("\n".join(latex) + "\n")


if __name__ == "__main__":
    main()
