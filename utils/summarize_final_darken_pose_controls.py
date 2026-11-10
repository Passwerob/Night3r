#!/usr/bin/env python3
import csv
import json
import os
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUTDIR = ROOT / "outputs" / "final_mvsec1k_pose_darken_controls_20260716"
DEFAULT_TAG = "full_6scene32_bidirectional"
TAG = os.environ.get("TAG", DEFAULT_TAG)
OUTPUT_SUFFIX = "" if TAG == DEFAULT_TAG else f"_{TAG}"
EXPOSURES = (("e020", 0.20), ("e010", 0.10), ("e005", 0.05))
METHODS = ("mast3r", "real", "zero", "shuffle")


def result_path(method, exposure_tag):
    return OUTDIR / f"dl3dv_{method}_{exposure_tag}_{TAG}.json"


def fmt(value, digits=4):
    return "--" if value is None else f"{float(value):.{digits}f}"


def load_rows():
    rows = []
    for exposure_tag, exposure in EXPOSURES:
        exposure_diagnostics = []
        for method in METHODS:
            path = result_path(method, exposure_tag)
            if not path.is_file():
                raise SystemExit(f"Missing darken pose result: {path}")
            payload = json.loads(path.read_text())
            aggregate = payload["aggregate"]
            rgb_input = payload["protocol"]["rgb_input"]
            diagnostics = rgb_input["diagnostics"]
            if rgb_input["mode"] != "deterministic_fixed_exposure_linear_domain":
                raise SystemExit(f"Unexpected RGB protocol in {path}: {rgb_input['mode']}")
            if abs(float(rgb_input["exposure_scale"]) - exposure) > 1.0e-9:
                raise SystemExit(f"Exposure mismatch in {path}")
            exposure_diagnostics.append(diagnostics)
            rows.append(
                {
                    "exposure": exposure,
                    "method": "MASt3R" if method == "mast3r" else method.capitalize(),
                    "num_scenes": aggregate["num_scenes"],
                    "num_frames": aggregate["num_frames"],
                    "ate_rmse_scene_mean_m": aggregate["ate_translation_m_rmse_scene_mean"],
                    "rpe_translation_rmse_scene_mean_m": aggregate[
                        "rpe_translation_m_rmse_scene_mean"
                    ],
                    "rpe_rotation_rmse_scene_mean_deg": aggregate[
                        "rpe_rotation_deg_rmse_scene_mean"
                    ],
                    "registration_residual_normalized_3d": aggregate.get(
                        "registration_residual_mean_all_edges",
                        aggregate.get("geometric_residual_mean"),
                    ),
                    "pose_recovery_success_rate": aggregate[
                        "pose_recovery_success_rate"
                    ],
                    "clean_rgb_mean": diagnostics["clean_mean"],
                    "dark_rgb_mean": diagnostics["dark_mean"],
                    "dark_to_clean_mean_ratio": diagnostics[
                        "dark_to_clean_mean_ratio"
                    ],
                    "dark_zero_fraction": diagnostics["dark_zero_fraction"],
                }
            )
        reference = exposure_diagnostics[0]
        for diagnostics in exposure_diagnostics[1:]:
            for key in reference:
                if abs(float(diagnostics[key]) - float(reference[key])) > 1.0e-8:
                    raise SystemExit(
                        f"RGB-control mismatch at exposure {exposure}: {key} "
                        f"{reference[key]} != {diagnostics[key]}"
                    )
    return rows


def main():
    rows = load_rows()
    with (OUTDIR / f"pose_darken_summary{OUTPUT_SUFFIX}.csv").open(
        "w", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    markdown = [
        "# DL3DV deterministic darken trajectory controls",
        "",
        "- Final checkpoint: `outputs/night3r_mvsec_only_adapt_1k_from_best/last_event_dark3r.pth`",
        "- RGB protocol: inverse gamma 2.2, fixed linear exposure, shot noise 0.01, read noise 0.01, sample-keyed seed 20260716.",
        "- Evaluation: 6 scenes x 32 frames, bidirectional point-map Sim(3), per-scene EVO Sim(3) alignment.",
        "- The summarizer verifies identical dark RGB statistics across MASt3R/Real/Zero/Shuffle within each exposure.",
        "",
        "| Exposure | Method | ATE (m) | RPE trans. (m) | RPE rot. (deg) | 3D residual | Recovery | Dark RGB mean | Zero fraction |",
        "|---:|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        markdown.append(
            f"| {row['exposure']:.2f} | {row['method']} | "
            f"{fmt(row['ate_rmse_scene_mean_m'])} | "
            f"{fmt(row['rpe_translation_rmse_scene_mean_m'])} | "
            f"{fmt(row['rpe_rotation_rmse_scene_mean_deg'], 2)} | "
            f"{fmt(row['registration_residual_normalized_3d'])} | "
            f"{100.0 * row['pose_recovery_success_rate']:.1f}% | "
            f"{fmt(row['dark_rgb_mean'])} | "
            f"{100.0 * row['dark_zero_fraction']:.1f}% |"
        )
    (OUTDIR / f"pose_darken_summary{OUTPUT_SUFFIX}.md").write_text(
        "\n".join(markdown) + "\n"
    )

    latex = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\caption{DL3DV trajectory accuracy under deterministic synthetic low light. RGB images are inverse-gamma linearized ($\gamma=2.2$), multiplied by a fixed exposure, corrupted by shot and read noise, and returned to display space. All methods receive identical darkened RGB at each exposure. Values are means over six scenes after per-scene Sim(3) alignment; lower is better.}",
        r"\label{tab:dl3dv-darken-pose}",
        r"\begin{tabular}{clrrrr}",
        r"\toprule",
        r"Exposure & Input/model & ATE (m) $\downarrow$ & RPE trans. (m) $\downarrow$ & RPE rot. ($^\circ$) $\downarrow$ & Recovery $\uparrow$ \\",
        r"\midrule",
    ]
    for exposure_index, (_, exposure) in enumerate(EXPOSURES):
        exposure_rows = [row for row in rows if row["exposure"] == exposure]
        for method_index, row in enumerate(exposure_rows):
            exposure_cell = rf"\multirow{{4}}{{*}}{{{exposure:.2f}}}" if method_index == 0 else ""
            latex.append(
                f"{exposure_cell} & {row['method']} & "
                f"{fmt(row['ate_rmse_scene_mean_m'])} & "
                f"{fmt(row['rpe_translation_rmse_scene_mean_m'])} & "
                f"{fmt(row['rpe_rotation_rmse_scene_mean_deg'], 2)} & "
                f"{100.0 * row['pose_recovery_success_rate']:.1f}\\% "
                + r"\\"
            )
        if exposure_index + 1 < len(EXPOSURES):
            latex.append(r"\midrule")
    latex.extend([r"\bottomrule", r"\end{tabular}", r"\end{table*}"])
    (OUTDIR / f"pose_darken_table{OUTPUT_SUFFIX}.tex").write_text(
        "\n".join(latex) + "\n"
    )


if __name__ == "__main__":
    main()
