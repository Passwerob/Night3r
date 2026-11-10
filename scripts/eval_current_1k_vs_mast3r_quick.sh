#!/usr/bin/env bash
set -euo pipefail

cd /data/fcr/data/Night3R

PY="${PY:-/data/fcr/miniconda3/envs/stream3r/bin/python}"
MAST3R_CKPT="${MAST3R_CKPT:-checkpoints/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric.pth}"
OUT_ROOT="${OUT_ROOT:-outputs/current_1k_vs_mast3r_quick_20260704}"
mkdir -p "${OUT_ROOT}/cache" "${OUT_ROOT}/mast3r" "${OUT_ROOT}/night3r_1k"

CFG="config/train_night3r_mvsec_only_adapt_1k_from_best.yaml"
CKPT_1K="outputs/night3r_mvsec_only_adapt_1k_from_best/last_event_dark3r.pth"

eval_dataset() {
  local gpu_cache="$1"
  local gpu_mast3r="$2"
  local gpu_night3r="$3"
  local dataset_name="$4"
  local data_root="$5"
  local extra_cache_args=()
  if [[ "${dataset_name}" == "mvsec_old_8scene16" ]]; then
    extra_cache_args+=(--max-pairs-per-scene 16)
  fi
  local cache_path="${OUT_ROOT}/cache/${dataset_name}_128.pt"

  CUDA_VISIBLE_DEVICES="${gpu_cache}" "${PY}" utils/eval_event_dark3r_variant.py \
    --variant mast3r \
    --config "${CFG}" \
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
    "${extra_cache_args[@]}" \
    --batch-cache "${cache_path}" \
    --write-batch-cache \
    --output "${OUT_ROOT}/cache/${dataset_name}_cache_writer.json" \
    > "${OUT_ROOT}/cache/${dataset_name}_cache_writer.stdout.log" 2>&1

  CUDA_VISIBLE_DEVICES="${gpu_mast3r}" "${PY}" utils/eval_event_dark3r_variant.py \
    --variant mast3r \
    --config "${CFG}" \
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
    "${extra_cache_args[@]}" \
    --batch-cache "${cache_path}" \
    --input lowlight \
    --output "${OUT_ROOT}/mast3r/${dataset_name}_128.json" \
    > "${OUT_ROOT}/mast3r/${dataset_name}_128.stdout.log" 2>&1 &

  CUDA_VISIBLE_DEVICES="${gpu_night3r}" "${PY}" utils/eval_event_dark3r_variant.py \
    --variant event_dark3r \
    --event-mode real \
    --checkpoint "${CKPT_1K}" \
    --config "${CFG}" \
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
    "${extra_cache_args[@]}" \
    --batch-cache "${cache_path}" \
    --output "${OUT_ROOT}/night3r_1k/${dataset_name}_128.json" \
    > "${OUT_ROOT}/night3r_1k/${dataset_name}_128.stdout.log" 2>&1 &
}

eval_dataset 0 0 1 mvsec_current /data/fcr/data/train_data/MVSEC_outdoor_night2_test
eval_dataset 2 2 3 dsec_current /data/fcr/data/train_data/DSEC_test
wait

"${PY}" - <<'PY'
import csv
import json
import math
from pathlib import Path

out_root = Path("outputs/current_1k_vs_mast3r_quick_20260704")
rows = []
for model_name, model_dir in [("MASt3R", out_root / "mast3r"), ("Night3R-1k", out_root / "night3r_1k")]:
    for path in sorted(model_dir.glob("*_128.json")):
        payload = json.loads(path.read_text())
        metrics = payload["dense_metrics"]
        dataset = path.stem[: -len("_128")]
        rows.append({
            "model": model_name,
            "dataset": dataset,
            "n": payload.get("num_batches"),
            "raw_absrel": metrics.get("depth_raw_absrel", float("nan")),
            "raw_delta1": metrics.get("depth_raw_delta_1.25", float("nan")),
            "raw_rmse_log": metrics.get("depth_raw_rmse_log", float("nan")),
            "raw_psnr": metrics.get("depth_raw_psnr_db", float("nan")),
            "median_absrel": metrics.get("depth_median_aligned_absrel", float("nan")),
            "median_delta1": metrics.get("depth_median_aligned_delta_1.25", float("nan")),
            "median_psnr": metrics.get("depth_median_aligned_psnr_db", float("nan")),
        })

with (out_root / "summary.csv").open("w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
    writer.writeheader()
    writer.writerows(rows)

def fmt(value):
    try:
        value = float(value)
    except Exception:
        return str(value)
    return "nan" if not math.isfinite(value) else f"{value:.4f}"

lines = [
    "| Model | Dataset | N | Raw AbsRel ↓ | Raw δ1 ↑ | Raw RMSElog ↓ | Raw PSNR ↑ | Median AbsRel ↓ | Median δ1 ↑ | Median PSNR ↑ |",
    "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
]
for row in rows:
    lines.append(
        f"| {row['model']} | {row['dataset']} | {row['n']} | {fmt(row['raw_absrel'])} | "
        f"{fmt(row['raw_delta1'])} | {fmt(row['raw_rmse_log'])} | {fmt(row['raw_psnr'])} | "
        f"{fmt(row['median_absrel'])} | {fmt(row['median_delta1'])} | {fmt(row['median_psnr'])} |"
    )
(out_root / "summary.md").write_text("\n".join(lines) + "\n")
print("\n".join(lines))
PY
