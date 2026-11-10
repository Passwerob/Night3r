#!/usr/bin/env bash
set -euo pipefail

cd /data/fcr/data/Night3R

PY="${PY:-/data/fcr/miniconda3/envs/stream3r/bin/python}"
CONFIG="${CONFIG:-config/eval_night3r_da3_ft1k_sweep_20260722.yaml}"
MAST3R_CKPT="${MAST3R_CKPT:-checkpoints/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric.pth}"
OUT_ROOT="${OUT_ROOT:-outputs/night3r_da3_global_sweep_eval_20260723}"
DL3DV_REF="${DL3DV_REF:-outputs/night3r_dl3dv_da3_pseudo_ft1k_eval_20260722/dl3dv_da3_reference}"
MVSEC_ROOT="${MVSEC_ROOT:-/data/fcr/data/train_data/MVSEC_outdoor_night2_test}"
DSEC_ROOT="${DSEC_ROOT:-/data/fcr/data/train_data/DSEC_test}"
GPU_LIST=(${GPU_LIST:-4 5 6 7})
BASE_CKPT="${BASE_CKPT:-outputs/night3r_three_dataset_balanced_final_20260723/event_dark3r_three_dataset_balanced_final.pth}"

labels=(base)
checkpoints=("${BASE_CKPT}")
for variant in trunk_shape trunk_metric full_shape full_metric; do
  root="outputs/night3r_da3_global_${variant}_600_1gpu_20260723"
  for step in 100 200 300 400 500 600; do
    labels+=("${variant}${step}")
    printf -v checkpoint "%s/event_dark3r_step%08d.pth" "${root}" "${step}"
    checkpoints+=("${checkpoint}")
  done
done

mkdir -p "${OUT_ROOT}/json" "${OUT_ROOT}/logs"
for checkpoint in "${checkpoints[@]}"; do
  test -f "${checkpoint}"
done

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
  local label="$1" checkpoint="$2" dataset_key="$3" data_root="$4"
  local max_batches="$5" max_pairs_per_scene="$6" depth_reference="$7" no_lowlight="$8"
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
      --variant event_dark3r --event-mode real \
      --checkpoint "${checkpoint}" --config "${CONFIG}" \
      --data-root "${data_root}" --model-name-or-path "${MAST3R_CKPT}" \
      --allow-unsafe-checkpoint-load --device cuda --amp --amp-dtype bf16 \
      --batch-size 1 --max-batches "${max_batches}" \
      --pair-sampling fixed --pair-stride 1 \
      --min-pair-distance 1 --max-pair-distance 1 \
      --no-allow-reverse-pairs --num-workers 0 --seed 0 \
      "${extra_args[@]}" --output "${output}"
  ) >"${log}" 2>&1 &
  pids+=("$!")
  running=$((running + 1))
  job_index=$((job_index + 1))
  if [[ "${running}" -eq "${#GPU_LIST[@]}" ]]; then
    wait_group
  fi
}

for dataset_key in dl3dv_da3 dsec_gt mvsec_gt; do
  case "${dataset_key}" in
    dl3dv_da3)
      data_root="${DL3DV_REF}"; max_batches=96; max_pairs=16; reference=pseudo; no_lowlight=0 ;;
    dsec_gt)
      data_root="${DSEC_ROOT}"; max_batches=128; max_pairs=0; reference=gt; no_lowlight=1 ;;
    mvsec_gt)
      data_root="${MVSEC_ROOT}"; max_batches=128; max_pairs=0; reference=gt; no_lowlight=1 ;;
  esac
  for index in "${!labels[@]}"; do
    launch_eval "${labels[$index]}" "${checkpoints[$index]}" "${dataset_key}" \
      "${data_root}" "${max_batches}" "${max_pairs}" "${reference}" "${no_lowlight}"
  done
done
if [[ "${running}" -gt 0 ]]; then
  wait_group
fi

"${PY}" utils/summarize_da3_ft1k_eval.py \
  --root "${OUT_ROOT}" --labels "${labels[@]}" \
  --mvsec-max-regression 0.05 --dsec-raw-absrel-max 0.15 \
  | tee "${OUT_ROOT}/summary.stdout.log"
touch "${OUT_ROOT}/EVAL_COMPLETE"

