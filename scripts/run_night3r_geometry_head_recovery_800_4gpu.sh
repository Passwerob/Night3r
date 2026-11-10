#!/usr/bin/env bash
set -euo pipefail

cd /data/fcr/data/Night3R

PY="${PY:-/data/fcr/miniconda3/envs/stream3r/bin/python}"
GPU_LIST="${GPU_LIST:-4,5,6,7}"
MASTER_PORT="${MASTER_PORT:-29731}"
MAX_STEPS="${MAX_STEPS:-800}"
RUN_NAME="${RUN_NAME:-night3r_geometry_head_recovery_800_4gpu_20260721}"
SAVE_DIR="outputs/${RUN_NAME}"
CONFIG="config/train_night3r_geometry_head_recovery_800.yaml"
DEPTH_CONFIG="config/train_night3r_mvsec_only_adapt_1k_from_best.yaml"
POSE_CONFIG="config/train_night3r_pose_separate_branches_w5_dense2k.yaml"
BASE_CHECKPOINT="outputs/night3r_mvsec_only_adapt_1k_from_best/last_event_dark3r.pth"
POSE_CHECKPOINT="outputs/night3r_pose_separate_branches_w5_dense2k_20260720/pose_heads/pose_head_step00001200.pth"
MAST3R_CHECKPOINT="checkpoints/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric.pth"
LOG="${SAVE_DIR}/train_and_eval.log"

mkdir -p "${SAVE_DIR}"
exec > >(tee -a "${LOG}") 2>&1

echo "geometry_head_recovery_start $(date -Is)"
echo "gpu_list=${GPU_LIST} max_steps=${MAX_STEPS} master_port=${MASTER_PORT}"

CUDA_VISIBLE_DEVICES="${GPU_LIST}" "${PY}" -m torch.distributed.run \
  --nproc_per_node=4 \
  --master_port="${MASTER_PORT}" \
  train_event_dark3r.py \
  --config "${CONFIG}" \
  --save-dir "${SAVE_DIR}" \
  --init-event-checkpoint "${BASE_CHECKPOINT}" \
  --model-name-or-path "${MAST3R_CHECKPOINT}" \
  --allow-unsafe-checkpoint-load \
  --device cuda --amp --amp-dtype bf16 \
  --image-size 128 128 \
  --max-steps "${MAX_STEPS}" \
  --log-every 20

echo "geometry_head_recovery_train_done $(date -Is)"

STEPS=(200 400 600 800)
GPUS=(4 5 6 7)
PIDS=()
for index in "${!STEPS[@]}"; do
  step="${STEPS[$index]}"
  gpu="${GPUS[$index]}"
  checkpoint="${SAVE_DIR}/geometry_head_step$(printf '%08d' "${step}").pth"
  eval_dir="${SAVE_DIR}/eval_step$(printf '%04d' "${step}")"
  mkdir -p "${eval_dir}"
  if [[ ! -f "${checkpoint}" ]]; then
    echo "missing_geometry_checkpoint ${checkpoint}"
    exit 1
  fi
  (
    CUDA_VISIBLE_DEVICES="${gpu}" "${PY}" utils/eval_event_dark3r_variant.py \
      --variant event_dark3r \
      --checkpoint "${BASE_CHECKPOINT}" \
      --geometry-head-checkpoint "${checkpoint}" \
      --config "${DEPTH_CONFIG}" \
      --data-root /data/fcr/data/train_data/MVSEC_outdoor_night2_test \
      --model-name-or-path "${MAST3R_CHECKPOINT}" \
      --allow-unsafe-checkpoint-load --event-mode real --no-lowlight-aug \
      --device cuda --amp --amp-dtype bf16 \
      --batch-size 1 --max-batches 32 \
      --pair-sampling fixed --pair-stride 1 \
      --min-pair-distance 1 --max-pair-distance 1 \
      --num-workers 0 --seed 17 \
      --output "${eval_dir}/mvsec_real_32.json" \
      > "${eval_dir}/mvsec_real_32.log" 2>&1

    CUDA_VISIBLE_DEVICES="${gpu}" "${PY}" utils/export_night3r_long_sequence.py \
      --checkpoint "${BASE_CHECKPOINT}" \
      --geometry-head-checkpoint "${checkpoint}" \
      --pose-head-checkpoint "${POSE_CHECKPOINT}" \
      --config "${POSE_CONFIG}" \
      --model-name-or-path "${MAST3R_CHECKPOINT}" \
      --data-root /data/fcr/data/eval_data/DL3DV \
      --scene screen-000090 --start-frame 0 --num-frames 8 --image-size 512 \
      --exposure 0.10 --alignment pose_head \
      --conf-quantile 0.75 --min-depth 0.0001 --max-depth 45 \
      --max-points-per-frame 18000 --voxel-size 0.01 \
      --amp --amp-dtype bf16 --device cuda --allow-unsafe-checkpoint-load \
      --outdir "${eval_dir}/dl3dv_pose_head_8f" \
      > "${eval_dir}/dl3dv_pose_head_8f.log" 2>&1
  ) &
  PIDS+=("$!")
done

status=0
for pid in "${PIDS[@]}"; do
  wait "${pid}" || status=1
done

echo "geometry_head_recovery_done status=${status} $(date -Is)"
exit "${status}"
