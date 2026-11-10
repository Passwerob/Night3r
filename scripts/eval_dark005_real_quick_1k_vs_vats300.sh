#!/usr/bin/env bash
set -euo pipefail

cd /data/fcr/data/Night3R

PY="${PY:-/data/fcr/miniconda3/envs/stream3r/bin/python}"
MAST3R_CKPT="${MAST3R_CKPT:-checkpoints/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric.pth}"
OUT_ROOT="${OUT_ROOT:-outputs/dark005_real_quick_1k_vs_vats300_20260703}"
mkdir -p "${OUT_ROOT}"

eval_one() {
  local gpu="$1"
  local model_name="$2"
  local checkpoint="$3"
  local config="$4"
  local dataset_name="$5"
  local data_root="$6"
  local extra_args=()
  local out_dir="${OUT_ROOT}/${model_name}"
  local output_json="${out_dir}/${dataset_name}_real_128.json"
  mkdir -p "${out_dir}"
  if [[ "${dataset_name}" == "mvsec_old_8scene16" ]]; then
    extra_args+=(--max-pairs-per-scene 16)
  fi

  CUDA_VISIBLE_DEVICES="${gpu}" "${PY}" utils/eval_event_dark3r_variant.py \
    --variant event_dark3r \
    --event-mode real \
    --checkpoint "${checkpoint}" \
    --config "${config}" \
    --data-root "${data_root}" \
    --model-name-or-path "${MAST3R_CKPT}" \
    --allow-unsafe-checkpoint-load \
    --device cuda \
    --amp \
    --amp-dtype bf16 \
    --batch-size 1 \
    --max-batches 128 \
    --pair-sampling fixed \
    --pair-stride 1 \
    --min-pair-distance 1 \
    --max-pair-distance 1 \
    "${extra_args[@]}" \
    --output "${output_json}" \
    > "${output_json%.json}.stdout.log" 2>&1
}

CKPT_1K="outputs/night3r_mvsec_only_adapt_1k_from_best/last_event_dark3r.pth"
CFG_1K="config/eval_night3r_1k_dark005.yaml"
CKPT_VATS="outputs/night3r_1k_vats_continue_300_20260703/last_event_dark3r.pth"
CFG_VATS="config/eval_night3r_1k_vats_dark005.yaml"

MVSEC_CURRENT="/data/fcr/data/train_data/MVSEC_outdoor_night2_test"
MVSEC_OLD="/data/fcr/data/train_data/MVSEC_outdoor_night2_test_prev_8scene_20260701"
DSEC_CURRENT="/data/fcr/data/train_data/DSEC_test"
DSEC_OLD="/data/fcr/data/Night3R/outputs/eval_roots/dsec_test_prev_2scene_20260701"

eval_one 3 1k_dark005 "${CKPT_1K}" "${CFG_1K}" mvsec_current "${MVSEC_CURRENT}" &
eval_one 4 1k_dark005 "${CKPT_1K}" "${CFG_1K}" mvsec_old_8scene16 "${MVSEC_OLD}" &
eval_one 5 1k_dark005 "${CKPT_1K}" "${CFG_1K}" dsec_current "${DSEC_CURRENT}" &
eval_one 6 1k_dark005 "${CKPT_1K}" "${CFG_1K}" dsec_old "${DSEC_OLD}" &
eval_one 7 vats300_dark005 "${CKPT_VATS}" "${CFG_VATS}" mvsec_current "${MVSEC_CURRENT}" &
wait

eval_one 3 vats300_dark005 "${CKPT_VATS}" "${CFG_VATS}" mvsec_old_8scene16 "${MVSEC_OLD}" &
eval_one 4 vats300_dark005 "${CKPT_VATS}" "${CFG_VATS}" dsec_current "${DSEC_CURRENT}" &
eval_one 5 vats300_dark005 "${CKPT_VATS}" "${CFG_VATS}" dsec_old "${DSEC_OLD}" &
wait

"${PY}" - <<'PY'
import csv
import json
import math
from pathlib import Path

out_root = Path("outputs/dark005_real_quick_1k_vs_vats300_20260703")
rows = []
for model_dir in [out_root / "1k_dark005", out_root / "vats300_dark005"]:
    for path in sorted(model_dir.glob("*_real_128.json")):
        payload = json.loads(path.read_text())
        metrics = payload["dense_metrics"]
        dataset = path.stem[: -len("_real_128")]
        rows.append({
            "model": model_dir.name,
            "dataset": dataset,
            "raw_absrel": metrics.get("depth_raw_absrel", float("nan")),
            "raw_delta_1.25": metrics.get("depth_raw_delta_1.25", float("nan")),
            "raw_rmse_log": metrics.get("depth_raw_rmse_log", float("nan")),
            "raw_psnr_db": metrics.get("depth_raw_psnr_db", float("nan")),
            "median_absrel": metrics.get("depth_median_aligned_absrel", float("nan")),
        })

with (out_root / "summary.csv").open("w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
    writer.writeheader()
    writer.writerows(rows)

def fmt(value):
    try:
        value = float(value)
    except Exception:
        return "nan"
    return "nan" if not math.isfinite(value) else f"{value:.4f}"

lines = [
    "| Model | Dataset | Raw AbsRel ↓ | δ1 ↑ | RMSE log ↓ | Raw PSNR ↑ | Median AbsRel ↓ |",
    "|---|---|---:|---:|---:|---:|---:|",
]
for row in rows:
    lines.append(
        f"| {row['model']} | {row['dataset']} | {fmt(row['raw_absrel'])} | "
        f"{fmt(row['raw_delta_1.25'])} | {fmt(row['raw_rmse_log'])} | "
        f"{fmt(row['raw_psnr_db'])} | {fmt(row['median_absrel'])} |"
    )
(out_root / "summary.md").write_text("\n".join(lines) + "\n")
print("\n".join(lines))
PY
