#!/usr/bin/env bash
set -euo pipefail

cd /data/fcr/data/Night3R

PY="${PY:-/data/fcr/miniconda3/envs/stream3r/bin/python}"
EVAL_GPU="${EVAL_GPU:-7}"
MAST3R_CKPT="${MAST3R_CKPT:-checkpoints/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric.pth}"

OUT_ROOT="${OUT_ROOT:-outputs/dark005_eval_1k_vs_vats300_20260703}"
mkdir -p "${OUT_ROOT}"

MVSEC_CURRENT_ROOT="${MVSEC_CURRENT_ROOT:-/data/fcr/data/train_data/MVSEC_outdoor_night2_test}"
MVSEC_OLD_ROOT="${MVSEC_OLD_ROOT:-/data/fcr/data/train_data/MVSEC_outdoor_night2_test_prev_8scene_20260701}"
DSEC_CURRENT_ROOT="${DSEC_CURRENT_ROOT:-/data/fcr/data/train_data/DSEC_test}"
DSEC_OLD_ROOT="${DSEC_OLD_ROOT:-/data/fcr/data/Night3R/outputs/eval_roots/dsec_test_prev_2scene_20260701}"

eval_one() {
  local model_name="$1"
  local checkpoint="$2"
  local config="$3"
  local dataset_name="$4"
  local data_root="$5"
  local max_batches="$6"
  local event_mode="$7"
  local out_dir="${OUT_ROOT}/${model_name}"
  local output_json="${out_dir}/${dataset_name}_${event_mode}.json"
  local extra_args=()
  mkdir -p "${out_dir}"
  if [[ "${data_root}" == "${MVSEC_OLD_ROOT}" ]]; then
    extra_args+=(--max-pairs-per-scene 16)
  fi

  CUDA_VISIBLE_DEVICES="${EVAL_GPU}" "${PY}" utils/eval_event_dark3r_variant.py \
    --variant event_dark3r \
    --event-mode "${event_mode}" \
    --checkpoint "${checkpoint}" \
    --config "${config}" \
    --data-root "${data_root}" \
    --model-name-or-path "${MAST3R_CKPT}" \
    --allow-unsafe-checkpoint-load \
    --device cuda \
    --amp \
    --amp-dtype bf16 \
    --batch-size 1 \
    --max-batches "${max_batches}" \
    --pair-sampling fixed \
    --pair-stride 1 \
    --min-pair-distance 1 \
    --max-pair-distance 1 \
    "${extra_args[@]}" \
    --output "${output_json}" \
    > "${output_json%.json}.stdout.log" 2>&1
}

run_model() {
  local model_name="$1"
  local checkpoint="$2"
  local config="$3"
  for mode in real zero shuffle; do
    eval_one "${model_name}" "${checkpoint}" "${config}" mvsec_current "${MVSEC_CURRENT_ROOT}" 128 "${mode}"
    eval_one "${model_name}" "${checkpoint}" "${config}" mvsec_old_8scene16 "${MVSEC_OLD_ROOT}" 128 "${mode}"
    eval_one "${model_name}" "${checkpoint}" "${config}" dsec_current "${DSEC_CURRENT_ROOT}" 612 "${mode}"
    eval_one "${model_name}" "${checkpoint}" "${config}" dsec_old "${DSEC_OLD_ROOT}" 128 "${mode}"
  done
}

run_model \
  "1k_dark005" \
  "outputs/night3r_mvsec_only_adapt_1k_from_best/last_event_dark3r.pth" \
  "config/eval_night3r_1k_dark005.yaml"

run_model \
  "vats300_dark005" \
  "outputs/night3r_1k_vats_continue_300_20260703/last_event_dark3r.pth" \
  "config/eval_night3r_1k_vats_dark005.yaml"

"${PY}" - <<'PY'
import csv
import json
import math
from pathlib import Path

out_root = Path("outputs/dark005_eval_1k_vs_vats300_20260703")
rows = []
for model_dir in [out_root / "1k_dark005", out_root / "vats300_dark005"]:
    model = model_dir.name
    for path in sorted(model_dir.glob("*.json")):
        if path.name.endswith(".stdout.json"):
            continue
        payload = json.loads(path.read_text())
        metrics = payload["dense_metrics"]
        stem = path.stem
        event = stem.rsplit("_", 1)[-1]
        dataset = stem[: -(len(event) + 1)]
        rows.append(
            {
                "model": model,
                "dataset": dataset,
                "event_mode": event,
                "raw_absrel": metrics.get("depth_raw_absrel", float("nan")),
                "raw_delta_1.25": metrics.get("depth_raw_delta_1.25", float("nan")),
                "raw_rmse_log": metrics.get("depth_raw_rmse_log", float("nan")),
                "raw_psnr_db": metrics.get("depth_raw_psnr_db", float("nan")),
                "median_absrel": metrics.get("depth_median_aligned_absrel", float("nan")),
                "warp_rgb_psnr_db": metrics.get("warp_rgb_psnr_db", float("nan")),
            }
        )

fields = [
    "model",
    "dataset",
    "event_mode",
    "raw_absrel",
    "raw_delta_1.25",
    "raw_rmse_log",
    "raw_psnr_db",
    "median_absrel",
    "warp_rgb_psnr_db",
]
with (out_root / "summary.csv").open("w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=fields)
    writer.writeheader()
    writer.writerows(rows)

def fmt(value):
    try:
        value = float(value)
    except Exception:
        return "nan"
    return "nan" if not math.isfinite(value) else f"{value:.4f}"

lines = [
    "| Model | Dataset | Event | Raw AbsRel ↓ | δ1 ↑ | RMSE log ↓ | Raw PSNR ↑ | Median AbsRel ↓ | Warp PSNR ↑ |",
    "|---|---|---|---:|---:|---:|---:|---:|---:|",
]
for row in rows:
    lines.append(
        f"| {row['model']} | {row['dataset']} | {row['event_mode']} | "
        f"{fmt(row['raw_absrel'])} | {fmt(row['raw_delta_1.25'])} | "
        f"{fmt(row['raw_rmse_log'])} | {fmt(row['raw_psnr_db'])} | "
        f"{fmt(row['median_absrel'])} | {fmt(row['warp_rgb_psnr_db'])} |"
    )
(out_root / "summary.md").write_text("\n".join(lines) + "\n")
print("\n".join(lines))
PY
