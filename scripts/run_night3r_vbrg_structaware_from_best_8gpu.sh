#!/usr/bin/env bash
set -euo pipefail

cd /data/fcr/data/Night3R

PY="/data/fcr/miniconda3/envs/stream3r/bin/python"
RUN_NAME="${RUN_NAME:-night3r_vbrg_structaware_from_best_8gpu_2k}"
SAVE_DIR="outputs/${RUN_NAME}"
EVAL_DIR="${SAVE_DIR}/eval_after_train"
LOG="${SAVE_DIR}/train_and_eval.log"
MAX_STEPS="${MAX_STEPS:-2000}"
MASTER_PORT="${MASTER_PORT:-29657}"
GPU_LIST="${GPU_LIST:-0,1,2,3,4,5,6,7}"
NPROC="${NPROC:-$(python3 - <<PY
value = "${GPU_LIST}"
print(len([item for item in value.split(",") if item.strip()]))
PY
)}"
EVAL_GPU="${EVAL_GPU:-${GPU_LIST%%,*}}"

CONFIG="${CONFIG:-config/train_night3r_vbrg_structaware_from_best_8gpu.yaml}"
INIT_CKPT_DEFAULT="${INIT_CKPT_DEFAULT:-outputs/night3r_vbrg_no_drop_from_best3k_8gpu_1500/last_event_dark3r.pth}"
INIT_CKPT="${INIT_CKPT-${INIT_CKPT_DEFAULT}}"
MAST3R_CKPT="${MAST3R_CKPT:-checkpoints/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric.pth}"

mkdir -p "${SAVE_DIR}" "${EVAL_DIR}"
exec > >(tee -a "${LOG}") 2>&1

echo "== Night3R structure-aware V-BRG from best checkpoint =="
date -Is
echo "RUN_NAME=${RUN_NAME}"
echo "MAX_STEPS=${MAX_STEPS}"
echo "MASTER_PORT=${MASTER_PORT}"
echo "GPU_LIST=${GPU_LIST}"
echo "NPROC=${NPROC}"
echo "EVAL_GPU=${EVAL_GPU}"
echo "CONFIG=${CONFIG}"
if [[ -n "${INIT_CKPT}" ]]; then
  echo "INIT_CKPT=${INIT_CKPT}"
else
  echo "INIT_CKPT=<fresh from model-name-or-path>"
fi
nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits

INIT_ARGS=()
if [[ -n "${INIT_CKPT}" ]]; then
  INIT_ARGS+=(--init-event-checkpoint "${INIT_CKPT}")
fi

CUDA_VISIBLE_DEVICES="${GPU_LIST}" \
"${PY}" -m torch.distributed.run \
  --nproc_per_node="${NPROC}" \
  --master_port="${MASTER_PORT}" \
  train_event_dark3r.py \
  --config "${CONFIG}" \
  --save-dir "${SAVE_DIR}" \
  "${INIT_ARGS[@]}" \
  --model-name-or-path "${MAST3R_CKPT}" \
  --allow-unsafe-checkpoint-load \
  --device cuda \
  --amp \
  --amp-dtype bf16 \
  --max-steps "${MAX_STEPS}" \
  --log-every 50

echo "== Training finished =="
date -Is

TRAINED_CKPT="${SAVE_DIR}/last_event_dark3r.pth"
test -f "${TRAINED_CKPT}"

echo "== DSEC holdout event ablation =="
for MODE in real zero shuffle; do
  CUDA_VISIBLE_DEVICES="${EVAL_GPU}" "${PY}" utils/eval_event_dark3r_variant.py \
    --variant event_dark3r \
    --event-mode "${MODE}" \
    --checkpoint "${TRAINED_CKPT}" \
    --config "${CONFIG}" \
    --data-root /data/fcr/data/train_data/DSEC_test \
    --model-name-or-path "${MAST3R_CKPT}" \
    --allow-unsafe-checkpoint-load \
    --device cuda \
    --amp \
    --amp-dtype bf16 \
    --batch-size 1 \
    --max-batches 612 \
    --pair-sampling fixed \
    --pair-stride 1 \
    --min-pair-distance 1 \
    --max-pair-distance 1 \
    --no-lowlight-aug \
    --output "${EVAL_DIR}/dsec_holdout_${MODE}_612.json" \
    > "${EVAL_DIR}/dsec_holdout_${MODE}_612.stdout.log" 2>&1
done

echo "== MVSEC event ablation =="
for MODE in real zero shuffle; do
  CUDA_VISIBLE_DEVICES="${EVAL_GPU}" "${PY}" utils/eval_event_dark3r_variant.py \
    --variant event_dark3r \
    --event-mode "${MODE}" \
    --checkpoint "${TRAINED_CKPT}" \
    --config "${CONFIG}" \
    --data-root /data/fcr/data/train_data/MVSEC_outdoor_night2_test \
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
    --no-lowlight-aug \
    --output "${EVAL_DIR}/mvsec_${MODE}_128.json" \
    > "${EVAL_DIR}/mvsec_${MODE}_128.stdout.log" 2>&1
done

echo "== DL3DV dark lowlight matrix =="
CUDA_VISIBLE_DEVICES="${EVAL_GPU}" "${PY}" utils/eval_mast3r_vs_night3r_lowlight_matrix.py \
  --trained-checkpoint "${TRAINED_CKPT}" \
  --config "${CONFIG}" \
  --data-root /data/fcr/data/eval_data/DL3DV_cloudy_test \
  --model-name-or-path "${MAST3R_CKPT}" \
  --allow-unsafe-checkpoint-load \
  --device cuda \
  --amp \
  --amp-dtype bf16 \
  --frame-index 24 \
  --pair-stride 7 \
  --exposures 0.2 0.1 0.05 0.02 \
  --scenes screen-000124 \
  --save-panels \
  --panel-protocol v2 \
  --outdir "${EVAL_DIR}/dl3dv_lowlight_protocol_v2" \
  > "${EVAL_DIR}/dl3dv_lowlight_protocol_v2.stdout.log" 2>&1

"${PY}" - <<'PY'
import csv
import json
import math
from pathlib import Path

import os

run_name = os.environ.get("RUN_NAME", "night3r_vbrg_structaware_from_best_8gpu_2k")
eval_dir = Path("outputs") / run_name / "eval_after_train"

def read_metrics(path):
    data = json.loads(Path(path).read_text())
    return data["dense_metrics"]

def fmt(value):
    if not isinstance(value, (int, float)) or not math.isfinite(value):
        return "nan"
    return f"{value:.4f}"

rows = []
for dataset, prefix, modes in [
    ("DSEC-holdout", "dsec_holdout", ["real", "zero", "shuffle"]),
    ("MVSEC", "mvsec", ["real", "zero", "shuffle"]),
]:
    for mode in modes:
        suffix = "612" if dataset == "DSEC-holdout" else "128"
        metrics = read_metrics(eval_dir / f"{prefix}_{mode}_{suffix}.json")
        rows.append(
            {
                "dataset": dataset,
                "event_mode": mode,
                "raw_absrel": metrics.get("depth_raw_absrel", float("nan")),
                "raw_psnr_db": metrics.get("depth_raw_psnr_db", float("nan")),
                "raw_delta_1.25": metrics.get("depth_raw_delta_1.25", float("nan")),
                "median_absrel": metrics.get("depth_median_aligned_absrel", float("nan")),
                "median_psnr_db": metrics.get("depth_median_aligned_psnr_db", float("nan")),
                "median_delta_1.25": metrics.get("depth_median_aligned_delta_1.25", float("nan")),
                "warp_rgb_psnr_db": metrics.get("warp_rgb_psnr_db", float("nan")),
            }
        )

with (eval_dir / "event_ablation_summary.csv").open("w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
    writer.writeheader()
    writer.writerows(rows)

lines = [
    "| Dataset | Event | Raw AbsRel ↓ | Raw PSNR ↑ | Raw δ1 ↑ | Median AbsRel ↓ | Median PSNR ↑ | Median δ1 ↑ | Warp PSNR ↑ |",
    "|---|---|---:|---:|---:|---:|---:|---:|---:|",
]
for row in rows:
    lines.append(
        f"| {row['dataset']} | {row['event_mode']} | "
        f"{fmt(row['raw_absrel'])} | {fmt(row['raw_psnr_db'])} | {fmt(row['raw_delta_1.25'])} | "
        f"{fmt(row['median_absrel'])} | {fmt(row['median_psnr_db'])} | {fmt(row['median_delta_1.25'])} | "
        f"{fmt(row['warp_rgb_psnr_db'])} |"
    )

lines.append("")
lines.extend(
    [
        "| Dataset | Comparison | Δ Raw AbsRel ↓ | Δ Raw PSNR ↑ | Δ Raw δ1 ↑ | Δ Median AbsRel ↓ | Δ Median PSNR ↑ | Δ Median δ1 ↑ | Δ Warp PSNR ↑ |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
)
by_key = {(row["dataset"], row["event_mode"]): row for row in rows}
for dataset in ["DSEC-holdout", "MVSEC"]:
    real = by_key[(dataset, "real")]
    for base_mode in ["zero", "shuffle"]:
        base = by_key[(dataset, base_mode)]
        lines.append(
            f"| {dataset} | real - {base_mode} | "
            f"{fmt(real['raw_absrel'] - base['raw_absrel'])} | "
            f"{fmt(real['raw_psnr_db'] - base['raw_psnr_db'])} | "
            f"{fmt(real['raw_delta_1.25'] - base['raw_delta_1.25'])} | "
            f"{fmt(real['median_absrel'] - base['median_absrel'])} | "
            f"{fmt(real['median_psnr_db'] - base['median_psnr_db'])} | "
            f"{fmt(real['median_delta_1.25'] - base['median_delta_1.25'])} | "
            f"{fmt(real['warp_rgb_psnr_db'] - base['warp_rgb_psnr_db'])} |"
        )

(eval_dir / "event_ablation_summary.md").write_text("\n".join(lines) + "\n")
print("\n".join(lines))
PY

echo "== Run complete =="
date -Is
