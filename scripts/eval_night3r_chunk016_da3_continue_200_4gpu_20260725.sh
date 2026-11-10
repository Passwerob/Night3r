#!/usr/bin/env bash
set -euo pipefail

cd /data/fcr/data/Night3R

PY="${PY:-/data/fcr/miniconda3/envs/stream3r/bin/python}"
GPU_LIST=(${GPU_LIST:-4 5 6 7})
OUT_ROOT="${OUT_ROOT:-outputs/night3r_chunk016_da3_continue_200_eval_20260725}"
EVAL_CONFIG="${EVAL_CONFIG:-config/eval_night3r_da3_ft1k_sweep_20260722.yaml}"
MAST3R_CKPT="${MAST3R_CKPT:-checkpoints/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric.pth}"
DL3DV_REF="${DL3DV_REF:-outputs/night3r_dl3dv_da3_pseudo_ft1k_eval_20260722/dl3dv_da3_reference}"
MVSEC_ROOT="${MVSEC_ROOT:-/data/fcr/data/train_data/MVSEC_outdoor_night2_test}"
DSEC_ROOT="${DSEC_ROOT:-/data/fcr/data/train_data/DSEC_test}"
DSEC_CITY11_ROOT="${DSEC_CITY11_ROOT:-outputs/eval_roots/dsec_city11_supplemental_20260725}"

labels=(base chunk50 chunk100 chunk150 chunk200)
checkpoints=(
  "outputs/night3r_final_step100_20260723/event_dark3r_night3r_final.pth"
  "outputs/night3r_chunk016_da3_continue_200_4gpu_20260725/event_dark3r_step00000050.pth"
  "outputs/night3r_chunk016_da3_continue_200_4gpu_20260725/event_dark3r_step00000100.pth"
  "outputs/night3r_chunk016_da3_continue_200_4gpu_20260725/event_dark3r_step00000150.pth"
  "outputs/night3r_chunk016_da3_continue_200_4gpu_20260725/event_dark3r_step00000200.pth"
)

mkdir -p "${OUT_ROOT}/json" "${OUT_ROOT}/logs"
exec > >(tee -a "${OUT_ROOT}/eval.log") 2>&1

pids=()
running=0
job_index=0

wait_group() {
  local status=0
  local pid
  for pid in "${pids[@]}"; do
    if ! wait "${pid}"; then
      status=1
    fi
  done
  pids=()
  running=0
  if [[ "${status}" -ne 0 ]]; then
    echo "At least one evaluation job failed." >&2
    exit "${status}"
  fi
}

launch_eval() {
  local label="$1"
  local checkpoint="$2"
  local dataset_key="$3"
  local data_root="$4"
  local max_batches="$5"
  local max_pairs_per_scene="$6"
  local depth_reference="$7"
  local no_lowlight="$8"
  local gpu="${GPU_LIST[$((job_index % ${#GPU_LIST[@]}))]}"
  local output="${OUT_ROOT}/json/${label}_${dataset_key}.json"
  local log="${OUT_ROOT}/logs/${label}_${dataset_key}.log"
  local extra_args=()

  if [[ "${max_pairs_per_scene}" -gt 0 ]]; then
    extra_args+=(--max-pairs-per-scene "${max_pairs_per_scene}")
  fi
  if [[ "${no_lowlight}" == "1" ]]; then
    extra_args+=(--no-lowlight-aug)
  fi
  if [[ "${depth_reference}" == "pseudo" ]]; then
    extra_args+=(--depth-reference pseudo --pseudo-confidence-percentile 40)
  fi

  echo "GPU ${gpu}: ${label} ${dataset_key}"
  (
    CUDA_VISIBLE_DEVICES="${gpu}" "${PY}" utils/eval_event_dark3r_variant.py \
      --variant event_dark3r \
      --event-mode real \
      --checkpoint "${checkpoint}" \
      --config "${EVAL_CONFIG}" \
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
      --no-allow-reverse-pairs \
      --num-workers 0 \
      --seed 0 \
      "${extra_args[@]}" \
      --output "${output}"
  ) >"${log}" 2>&1 &
  pids+=("$!")
  running=$((running + 1))
  job_index=$((job_index + 1))
  if [[ "${running}" -eq "${#GPU_LIST[@]}" ]]; then
    wait_group
  fi
}

for dataset_key in dl3dv_da3 dsec_gt dsec_city11_gt mvsec_gt; do
  case "${dataset_key}" in
    dl3dv_da3)
      data_root="${DL3DV_REF}"
      max_batches=96
      max_pairs=16
      reference=pseudo
      no_lowlight=0
      ;;
    dsec_gt)
      data_root="${DSEC_ROOT}"
      max_batches=128
      max_pairs=0
      reference=gt
      no_lowlight=1
      ;;
    dsec_city11_gt)
      data_root="${DSEC_CITY11_ROOT}"
      max_batches=128
      max_pairs=0
      reference=gt
      no_lowlight=1
      ;;
    mvsec_gt)
      data_root="${MVSEC_ROOT}"
      max_batches=128
      max_pairs=0
      reference=gt
      no_lowlight=1
      ;;
  esac
  for index in "${!labels[@]}"; do
    launch_eval \
      "${labels[$index]}" \
      "${checkpoints[$index]}" \
      "${dataset_key}" \
      "${data_root}" \
      "${max_batches}" \
      "${max_pairs}" \
      "${reference}" \
      "${no_lowlight}"
  done
done
if [[ "${running}" -gt 0 ]]; then
  wait_group
fi

"${PY}" utils/summarize_da3_ft1k_eval.py \
  --root "${OUT_ROOT}" \
  --labels "${labels[@]}" \
  --dsec-raw-absrel-max 0.168 | tee "${OUT_ROOT}/summary.stdout.log"

"${PY}" - "${OUT_ROOT}" "${labels[@]}" <<'PY'
import csv
import json
import math
import sys
from pathlib import Path

root = Path(sys.argv[1])
labels = sys.argv[2:]
rows = []
for label in labels:
    payload = json.loads((root / "json" / f"{label}_dsec_city11_gt.json").read_text())
    metrics = payload.get("dense_metrics", {})
    rows.append(
        {
            "checkpoint": label,
            "num_batches": int(payload.get("num_batches", 0)),
            "raw_absrel": float(metrics.get("depth_raw_absrel", math.nan)),
            "raw_delta1": float(metrics.get("depth_raw_delta_1.25", math.nan)),
            "median_absrel": float(metrics.get("depth_median_aligned_absrel", math.nan)),
            "median_delta1": float(metrics.get("depth_median_aligned_delta_1.25", math.nan)),
        }
    )

with (root / "dsec_city11_summary.csv").open("w", newline="") as handle:
    writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)

lines = [
    "# DSEC Zurich City 11 Supplemental Sweep",
    "",
    "This sequence is scene-disjoint from the DSEC training split and is retained",
    "as a supplemental diagnostic. The official Night3R DSEC checkpoint guardrail",
    "continues to use the strict Zurich City 02 held-out split.",
    "",
    "| Checkpoint | N | Raw AbsRel | Raw delta1 | Median AbsRel | Median delta1 |",
    "|---|---:|---:|---:|---:|---:|",
]
for row in rows:
    lines.append(
        f"| {row['checkpoint']} | {row['num_batches']} | "
        f"{row['raw_absrel']:.4f} | {row['raw_delta1']:.4f} | "
        f"{row['median_absrel']:.4f} | {row['median_delta1']:.4f} |"
    )
(root / "dsec_city11_summary.md").write_text("\n".join(lines) + "\n")
print("\n".join(lines))
PY

touch "${OUT_ROOT}/EVAL_COMPLETE"
echo "== Night3R chunk016 continuation evaluation complete =="
date -Is
