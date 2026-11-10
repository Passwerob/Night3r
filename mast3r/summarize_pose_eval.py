#!/usr/bin/env python3
"""
summarize_pose_eval.py (JSON version, with labeled sections and header comments)

Read multiple JSON files produced by the evaluator (one per scene), average the metrics,
and emit TWO LaTeX table rows (in a single .txt file):

  1) Row with averages of per-scene MEANS
  2) Row with averages of per-scene MEDIANS

USAGE
-----
python summarize_pose_eval.py <output_txt> <eval1.json> <eval2.json> ... <evalN.json>

Notes
-----
- ATE in your table corresponds to APE translation (m).
- RRA/RTA are stored in JSON as PERCENT; we convert to ratios in the LaTeX rows.
- Edit 'METHOD' and 'INPUT' placeholders in the resulting rows to fit your table.
"""
import argparse
import json
import math
from pathlib import Path
from statistics import mean as _mean


def _fmt(x, places=3):
    if x is None or (isinstance(x, float) and (math.isnan(x) or math.isinf(x))):
        return "NA"
    return f"{x:.{places}f}"


def _avg(xs):
    xs = [
        x for x in xs if x is not None and not (isinstance(x, float) and math.isnan(x))
    ]
    return _mean(xs) if xs else None


def load_json(p: Path) -> dict:
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)


def make_header_row(rot_thr, trans_thr) -> str:
    # Build a LaTeX header row string (as a comment line) with optional thresholds.
    header = r"& Method & Input & ATE $\downarrow$ & RPE trans $\downarrow$ & RPE rot $\downarrow$ & RRA"
    if rot_thr is not None:
        header += f" @{rot_thr:g}"
    header += r" $\uparrow$ & RTA"
    if trans_thr is not None:
        header += f" @{trans_thr:g}"
    header += r" $\uparrow$ \\"
    return header


def main():
    ap = argparse.ArgumentParser(
        description="Aggregate pose metrics from evaluator JSON files and write LaTeX rows."
    )
    ap.add_argument("output_txt", type=Path, help="Where to write the LaTeX row(s).")
    ap.add_argument(
        "json_files", nargs="+", type=Path, help="One or more evaluator JSON files."
    )
    args = ap.parse_args()

    parsed = [load_json(p) for p in args.json_files]

    # Collect per-scene values
    ate_mean_vals = [
        d.get("ape_t", {}).get("mean") for d in parsed
    ]  # APE translation mean (m)
    ate_median_vals = [
        d.get("ape_t", {}).get("median") for d in parsed
    ]  # APE translation median (m)

    rpe_t_mean_vals = [
        d.get("rpe_t", {}).get("mean") for d in parsed
    ]  # RPE translation mean (m)
    rpe_t_median_vals = [
        d.get("rpe_t", {}).get("median") for d in parsed
    ]  # RPE translation median (m)

    rpe_r_mean_vals = [
        d.get("rpe_rot", {}).get("mean") for d in parsed
    ]  # RPE rotation mean (deg)
    rpe_r_median_vals = [
        d.get("rpe_rot", {}).get("median") for d in parsed
    ]  # RPE rotation median (deg)

    # RRA/RTA are stored as percentage in the JSON by design
    rra_vals_pct = [d.get("rra_percent") for d in parsed]
    rta_vals_pct = [d.get("rta_percent") for d in parsed]

    rra_vals = [v if v is not None else None for v in rra_vals_pct]
    rta_vals = [v if v is not None else None for v in rta_vals_pct]

    # Optional: show thresholds in header if uniform across files
    rot_thrs = [
        d.get("thresholds", {}).get("rotation_deg")
        for d in parsed
        if d.get("thresholds")
    ]
    trans_thrs = [
        d.get("thresholds", {}).get("translation_m")
        for d in parsed
        if d.get("thresholds")
    ]

    unique_rot = sorted({v for v in rot_thrs if v is not None})
    unique_trans = sorted({v for v in trans_thrs if v is not None})
    rot_thr = unique_rot[0] if len(unique_rot) == 1 else None
    trans_thr = unique_trans[0] if len(unique_trans) == 1 else None

    header_row = make_header_row(rot_thr, trans_thr)

    # Averages across scenes
    ate_mean = _avg(ate_mean_vals)
    rpe_t_mean = _avg(rpe_t_mean_vals)
    rpe_r_mean = _avg(rpe_r_mean_vals)
    rra_avg = _avg(rra_vals)
    rta_avg = _avg(rta_vals)

    ate_median = _avg(ate_median_vals)
    rpe_t_median = _avg(rpe_t_median_vals)
    rpe_r_median = _avg(rpe_r_median_vals)

    n_files = len(parsed)

    mean_row = (
        r"& METHOD & INPUT & "
        f"{_fmt(ate_mean)} & "
        f"{_fmt(rpe_t_mean)} & "
        f"{_fmt(rpe_r_mean)} & "
        f"{_fmt(rra_avg, 2)} & "
        f"{_fmt(rta_avg, 2)} \\\\"
    )
    median_row = (
        r"& METHOD & INPUT & "
        f"{_fmt(ate_median)} & "
        f"{_fmt(rpe_t_median)} & "
        f"{_fmt(rpe_r_median)} & "
        f"{_fmt(rra_avg, 2)} & "
        f"{_fmt(rta_avg, 2)} \\\\"
    )

    # ---------- Output with headers and labeled sections ----------
    content = []
    content.append(
        f"% Aggregated across {n_files} file(s). Replace METHOD/INPUT as needed.\n"
    )
    if rot_thr is not None and trans_thr is not None:
        content.append(f"% Thresholds: RRA @{rot_thr:g} deg | RTA @{trans_thr:g} m\n")
    elif rot_thr is not None:
        content.append(f"% Threshold: RRA @{rot_thr:g} deg\n")
    elif trans_thr is not None:
        content.append(f"% Threshold: RTA @{trans_thr:g} m\n")

    content.append("Mean-aggregated row (averaged per-scene means):\n")
    content.append("```latex\n")
    content.append("% " + header_row + "\n")
    content.append(mean_row + "\n")
    content.append("```\n")

    content.append("Median-aggregated row (averaged per-scene medians):\n")
    content.append("```latex\n")
    content.append("% " + header_row + "\n")
    content.append(median_row + "\n")
    content.append("```\n")

    args.output_txt.parent.mkdir(parents=True, exist_ok=True)
    args.output_txt.write_text("".join(content), encoding="utf-8")
    print(f"Wrote LaTeX rows to: {args.output_txt}")


if __name__ == "__main__":
    main()
