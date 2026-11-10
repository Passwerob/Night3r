#!/usr/bin/env bash
set -euo pipefail

cd /data/fcr/data/Night3R

PY="${PY:-/data/fcr/miniconda3/envs/stream3r/bin/python}"
GPU_LIST="${GPU_LIST:-4,5,6,7}"
MAX_STEPS="${MAX_STEPS:-2400}"
RUN_NAME="${RUN_NAME:-night3r_pose_rotation_sweep_20260720}"
ROOT="outputs/${RUN_NAME}"
BASE_CKPT="${BASE_CKPT:-outputs/night3r_mvsec_only_adapt_1k_from_best/last_event_dark3r.pth}"
MAST3R_CKPT="${MAST3R_CKPT:-checkpoints/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric.pth}"
MVSEC_VAL="${MVSEC_VAL:-outputs/splits/pose_head_sweep_val_v1/MVSEC}"
DL3DV_VAL="${DL3DV_VAL:-outputs/splits/pose_head_sweep_val_v1/DL3DV}"

IFS=',' read -r -a GPUS <<< "${GPU_LIST}"
LABELS=(total01 total05 total10 total20)
CONFIGS=(
  config/train_night3r_pose_rotation_total01.yaml
  config/train_night3r_pose_rotation_total05.yaml
  config/train_night3r_pose_rotation_total10.yaml
  config/train_night3r_pose_rotation_total20.yaml
)
if [[ "${#GPUS[@]}" -ne "${#LABELS[@]}" ]]; then
  echo "Expected exactly four GPUs in GPU_LIST" >&2
  exit 2
fi
mkdir -p "${ROOT}"
bash scripts/prepare_pose_head_sweep_val_v1.sh

train_one() {
  local gpu="$1" label="$2" config="$3"
  local output="${ROOT}/${label}"
  mkdir -p "${output}/pose_heads"
  CUDA_VISIBLE_DEVICES="${gpu}" "${PY}" train_event_dark3r.py \
    --config "${config}" \
    --save-dir "${output}/pose_heads" \
    --init-event-checkpoint "${BASE_CKPT}" \
    --model-name-or-path "${MAST3R_CKPT}" \
    --allow-unsafe-checkpoint-load \
    --device cuda --amp --amp-dtype bf16 \
    --max-steps "${MAX_STEPS}" --log-every 25 \
    > "${output}/train.log" 2>&1
  touch "${output}/TRAIN_COMPLETE"
}

pids=()
for index in "${!LABELS[@]}"; do
  train_one "${GPUS[index]}" "${LABELS[index]}" "${CONFIGS[index]}" &
  pids+=("$!")
done
status=0
for pid in "${pids[@]}"; do
  wait "${pid}" || status=1
done
if [[ "${status}" != "0" ]]; then
  echo "At least one rotation-loss training run failed" >&2
  exit 1
fi
touch "${ROOT}/TRAINING_COMPLETE"

eval_one() {
  local gpu="$1" label="$2" config="$3" head="$4" dataset="$5" data_root="$6"
  local step
  step="$(basename "${head}" .pth | sed 's/pose_head_step//')"
  local output="${ROOT}/validation/${label}/step${step}/${dataset}_real.json"
  if [[ -s "${output}" ]]; then
    return
  fi
  mkdir -p "$(dirname "${output}")"
  local exposure_args=()
  if [[ "${dataset}" == "dl3dv" ]]; then
    exposure_args+=(--lowlight-exposure 0.05)
  fi
  CUDA_VISIBLE_DEVICES="${gpu}" "${PY}" utils/eval_night3r_trajectory.py \
    --variant event_dark3r \
    --checkpoint "${BASE_CKPT}" \
    --pose-head-checkpoint "${head}" \
    --config "${config}" \
    --data-root "${data_root}" \
    --model-name-or-path "${MAST3R_CKPT}" \
    --allow-unsafe-checkpoint-load \
    --event-mode real --pose-recovery direct \
    --max-scenes 3 --max-frames-per-scene 32 \
    --device cuda --amp --amp-dtype bf16 --num-workers 0 \
    "${exposure_args[@]}" --output "${output}" \
    > "${output%.json}.log" 2>&1
}

eval_variant() {
  local gpu="$1" label="$2" config="$3"
  local head
  while IFS= read -r head; do
    eval_one "${gpu}" "${label}" "${config}" "${head}" mvsec "${MVSEC_VAL}"
    eval_one "${gpu}" "${label}" "${config}" "${head}" dl3dv "${DL3DV_VAL}"
  done < <(find "${ROOT}/${label}/pose_heads" -maxdepth 1 -type f \
    -name 'pose_head_step*.pth' | sort)
}

pids=()
for index in "${!LABELS[@]}"; do
  eval_variant "${GPUS[index]}" "${LABELS[index]}" "${CONFIGS[index]}" &
  pids+=("$!")
done
status=0
for pid in "${pids[@]}"; do
  wait "${pid}" || status=1
done
if [[ "${status}" != "0" ]]; then
  echo "At least one rotation-loss validation worker failed" >&2
  exit 1
fi

"${PY}" utils/summarize_pose_rotation_sweep.py \
  --eval-root "${ROOT}/validation" \
  --output-dir "${ROOT}/validation"
touch "${ROOT}/SWEEP_COMPLETE"
