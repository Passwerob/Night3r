#!/usr/bin/env python3
"""Summarize matched Night3R pose comparisons and verify scene parity."""

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUTDIR = ROOT / "outputs" / "pose_sota_comparison_20260719"
NIGHT3R_DIR = (
    ROOT
    / "outputs"
    / "night3r_pose_head_mvsec_dl3dv_3k_20260719"
    / "eval_direct_pose"
)
METHODS = (
    ("VGGT", "vggt"),
    ("MASt3R", "mast3r"),
    ("Dark3R", "dark3r"),
    ("EventVGGT$^{\dagger}$", "eventvggt"),
    ("Night3R (Ours)", "night3r"),
)


def load_payload(dataset, key):
    path = (
        NIGHT3R_DIR / f"{dataset}_real.json"
        if key == "night3r"
        else OUTDIR / f"{dataset}_{key}.json"
    )
    if not path.is_file():
        raise SystemExit(f"Missing pose result: {path}")
    return path, json.loads(path.read_text())


def scene_signature(payload):
    return [
        (
            item["scene"],
            item["num_requested_edges"],
            item["num_frames"],
        )
        for item in payload["scenes"]
    ]


def fmt(value, digits=4):
    return "--" if value is None else f"{float(value):.{digits}f}"


def main():
    rows = []
    for dataset in ("mvsec", "dl3dv"):
        _, reference = load_payload(dataset, "night3r")
        reference_signature = scene_signature(reference)
        for method, key in METHODS:
            path, payload = load_payload(dataset, key)
            signature = scene_signature(payload)
            if signature != reference_signature:
                raise SystemExit(
                    f"Scene/frame protocol mismatch: {path}\n"
                    f"expected={reference_signature}\nactual={signature}"
                )
            aggregate = payload["aggregate"]
            rows.append(
                {
                    "dataset": "DL3DV-5%" if dataset == "dl3dv" else "MVSEC",
                    "method": method,
                    "ate": aggregate["ate_translation_m_rmse_scene_mean"],
                    "rpe_t": aggregate["rpe_translation_m_rmse_scene_mean"],
                    "rpe_r": aggregate["rpe_rotation_deg_rmse_scene_mean"],
                    "success": aggregate["pose_recovery_success_rate"],
                }
            )

    lines = [
        "# Matched pose comparison",
        "",
        "- Protocol: six scenes, 32 frames per scene, adjacent pair inference, per-scene Sim(3) translation alignment, and first-pose rotation alignment.",
        "- Point-map methods retain the full requested trajectory; weak registrations are counted in the trajectory error instead of being dropped.",
        "- MVSEC uses stored night RGB/events. DL3DV uses deterministic 5% exposure RGB and unchanged synchronized events.",
        "- EventVGGT dagger: its released checkpoint has no camera-head parameters; the diagnostic inherits the official frozen VGGT camera head and is not presented as an official EventVGGT pose result.",
        "",
        "| Dataset | Method | ATE (m) | RPE-t (m) | RPE-r (deg) | Recovery |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['dataset']} | {row['method']} | {fmt(row['ate'])} | "
            f"{fmt(row['rpe_t'])} | {fmt(row['rpe_r'])} | "
            f"{100.0 * row['success']:.1f}% |"
        )
    (OUTDIR / "summary.md").write_text("\n".join(lines) + "\n")

    latex = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Camera trajectory accuracy on MVSEC and controlled low-light DL3DV. All methods are evaluated on the same scene and frame sequences. Translation metrics use per-scene Sim(3) alignment, and lower values are better.}",
        r"\label{tab:pose-sota}",
        r"\small",
        r"\begin{tabular}{llrrr}",
        r"\toprule",
        r"Dataset & Method & ATE (m) $\downarrow$ & RPE-t (m) $\downarrow$ & RPE-r ($^\circ$) $\downarrow$ \\",
        r"\midrule",
    ]
    for dataset_index, dataset in enumerate(("MVSEC", "DL3DV-5%")):
        subset = [row for row in rows if row["dataset"] == dataset]
        for row_index, row in enumerate(subset):
            dataset_cell = rf"\multirow{{{len(subset)}}}{{*}}{{{dataset}}}" if row_index == 0 else ""
            latex.append(
                f"{dataset_cell} & {row['method']} & {fmt(row['ate'])} & "
                f"{fmt(row['rpe_t'])} & {fmt(row['rpe_r'])} " + r"\\"
            )
        if dataset_index == 0:
            latex.append(r"\midrule")
    latex.extend(
        [
            r"\bottomrule",
            r"\end{tabular}",
            r"\vspace{2pt}",
            r"\begin{minipage}{0.98\linewidth}\footnotesize $\dagger$ The released EventVGGT checkpoint does not contain camera-head parameters; this diagnostic uses its event-adapted encoder with the frozen official VGGT camera head.\end{minipage}",
            r"\end{table}",
        ]
    )
    (OUTDIR / "pose_table.tex").write_text("\n".join(latex) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
