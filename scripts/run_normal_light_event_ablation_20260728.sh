#!/usr/bin/env bash
set -euo pipefail

# Native-RGB event controls and matched fusion ablation. Launch this script
# inside tmux; it uses four GPUs concurrently and never trains a new model.

cd /data/fcr/data/Night3R

ROOT="/data/fcr/data/Night3R"
PY="${PY:-/data/fcr/miniconda3/envs/stream3r/bin/python}"
OUT="${OUT:-outputs/normal_light_event_ablation_20260728}"
GPU_LIST="${GPU_LIST:-4,5,6,7}"
IFS=',' read -r -a GPUS <<< "${GPU_LIST}"
if [[ "${#GPUS[@]}" -ne 4 ]]; then
  echo "GPU_LIST must contain exactly four comma-separated GPU ids" >&2
  exit 2
fi

FINAL_CONFIG="${FINAL_CONFIG:-config/train_night3r_final_step100_pose_adapt100.yaml}"
FINAL_CKPT="${FINAL_CKPT:-outputs/night3r_final_complete_pose_step100_20260726/event_dark3r_night3r_final_with_pose_head.pth}"
MAST3R_CKPT="${MAST3R_CKPT:-checkpoints/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric.pth}"
LORALIB_VENDOR="${LORALIB_VENDOR:-/tmp/loralib_vendor}"
SEED="${SEED:-1729}"

mkdir -p "${OUT}/roots" "${OUT}/final_controls" "${OUT}/fusion_ablation" "${OUT}/logs" "${LORALIB_VENDOR}"
ln -sfn /data/fcr/miniconda3/lib/python3.8/site-packages/loralib "${LORALIB_VENDOR}/loralib"
export PYTHONPATH="${LORALIB_VENDOR}${PYTHONPATH:+:${PYTHONPATH}}"

prepare_root() {
  local name="$1"
  shift
  local root="${OUT}/roots/${name}"
  mkdir -p "${root}"
  local source
  for source in "$@"; do
    ln -sfn "${source}" "${root}/$(basename "${source}")"
  done
}

prepare_root dsec_003 "${ROOT}/test/DSEC/screen-000003-dsec"
prepare_root dsec_004 "${ROOT}/test/DSEC/screen-000004-dsec"
prepare_root dsec_extended \
  "${ROOT}/test/DSEC/screen-000003-dsec" \
  "${ROOT}/test/DSEC/screen-000004-dsec"

run_final() {
  local gpu="$1" split="$2" max_batches="$3" mode="$4"
  local output="${OUT}/final_controls/${split}/${mode}.json"
  local log="${OUT}/logs/final_${split}_${mode}.log"
  if [[ -s "${output}" ]]; then
    echo "[skip] final ${split}/${mode}"
    return
  fi
  mkdir -p "$(dirname "${output}")"
  CUDA_VISIBLE_DEVICES="${gpu}" PYTHONUNBUFFERED=1 "${PY}" utils/eval_event_dark3r_variant.py \
    --variant event_dark3r \
    --checkpoint "${FINAL_CKPT}" \
    --config "${FINAL_CONFIG}" \
    --data-root "${OUT}/roots/${split}" \
    --model-name-or-path "${MAST3R_CKPT}" \
    --allow-unsafe-checkpoint-load \
    --event-mode "${mode}" \
    --no-lowlight-aug \
    --device cuda --amp --amp-dtype bf16 \
    --batch-size 1 --max-batches "${max_batches}" \
    --pair-sampling fixed --pair-stride 1 \
    --min-pair-distance 1 --max-pair-distance 1 \
    --num-workers 0 --seed "${SEED}" \
    --output "${output}" > "${log}" 2>&1
}

run_final_queue() {
  local gpu="$1" mode="$2"
  run_final "${gpu}" dsec_003 612 "${mode}"
  run_final "${gpu}" dsec_004 966 "${mode}"
  run_final "${gpu}" dsec_extended 1578 "${mode}"
}

run_fusion() {
  local gpu="$1" name="$2" config="$3" checkpoint="$4"
  local output="${OUT}/fusion_ablation/dsec_003/${name}.json"
  local log="${OUT}/logs/fusion_${name}.log"
  if [[ -s "${output}" ]]; then
    echo "[skip] fusion ${name}"
    return
  fi
  mkdir -p "$(dirname "${output}")"
  CUDA_VISIBLE_DEVICES="${gpu}" PYTHONUNBUFFERED=1 "${PY}" utils/eval_event_dark3r_variant.py \
    --variant event_dark3r \
    --checkpoint "${checkpoint}" \
    --config "${config}" \
    --data-root "${OUT}/roots/dsec_003" \
    --model-name-or-path "${MAST3R_CKPT}" \
    --allow-unsafe-checkpoint-load \
    --event-mode real \
    --no-lowlight-aug \
    --device cuda --amp --amp-dtype bf16 \
    --batch-size 1 --max-batches 612 \
    --pair-sampling fixed --pair-stride 1 \
    --min-pair-distance 1 --max-pair-distance 1 \
    --num-workers 0 --seed "${SEED}" \
    --output "${output}" > "${log}" 2>&1
}

run_fusion_queue() {
  local gpu="$1"
  run_fusion "${gpu}" full_brg \
    config/exp3_full_brg_20260714.yaml \
    outputs/exp3_full_brg_1gpu_10k_20260717/last_event_dark3r.pth
  run_fusion "${gpu}" concat_mlp \
    config/exp6_fusion_concat_mlp_20260715.yaml \
    outputs/exp6_fusion_concat_mlp_1gpu_10k_20260717/last_event_dark3r.pth
  run_fusion "${gpu}" additive \
    config/exp6_fusion_additive_20260715.yaml \
    outputs/exp6_fusion_additive_1gpu_10k_20260717/last_event_dark3r.pth
}

run_final_queue "${GPUS[0]}" real & pid_real=$!
run_final_queue "${GPUS[1]}" zero & pid_zero=$!
run_final_queue "${GPUS[2]}" shuffle & pid_shuffle=$!
run_fusion_queue "${GPUS[3]}" & pid_fusion=$!

status=0
wait "${pid_real}" || status=1
wait "${pid_zero}" || status=1
wait "${pid_shuffle}" || status=1
wait "${pid_fusion}" || status=1

if [[ "${status}" -ne 0 ]]; then
  echo "One or more normal-light evaluation jobs failed. Check ${OUT}/logs." >&2
  exit "${status}"
fi

"${PY}" utils/summarize_normal_light_event_ablation.py --root "${OUT}"
touch "${OUT}/COMPLETE"
echo "[complete] ${OUT}"
