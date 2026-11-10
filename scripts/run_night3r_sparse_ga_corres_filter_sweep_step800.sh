#!/usr/bin/env bash
set -euo pipefail

cd /data/fcr/data/Night3R

PY="${PY:-/data/fcr/miniconda3/envs/stream3r/bin/python}"
RUN_DIR="outputs/night3r_sparse_ga_corres_filter_sweep_step800_20260721"
BASE_CHECKPOINT="outputs/night3r_mvsec_only_adapt_1k_from_best/last_event_dark3r.pth"
DESCRIPTOR_CHECKPOINT="outputs/night3r_descriptor_recovery_800_4gpu_20260721/descriptor_head_step00000800.pth"
CONFIG="config/train_night3r_descriptor_recovery_1k.yaml"
MAST3R_CHECKPOINT="checkpoints/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric.pth"

mkdir -p "${RUN_DIR}"
exec > >(tee -a "${RUN_DIR}/sweep.log") 2>&1

NAMES=(minconf05_mst_q075 minconf08_mst_q075 minconf10_mst_q075 minconf15_mst_q075)
GPUS=(4 5 6 7)
MIN_CONF=(5 8 10 15)
PIDS=()

echo "corres_filter_sweep_start $(date -Is)"
for index in "${!NAMES[@]}"; do
  name="${NAMES[$index]}"
  gpu="${GPUS[$index]}"
  min_conf="${MIN_CONF[$index]}"
  outdir="${RUN_DIR}/${name}"
  echo "launch name=${name} gpu=${gpu} min_corres_conf_thr=${min_conf}"
  CUDA_VISIBLE_DEVICES="${gpu}" "${PY}" \
    utils/export_night3r_sparse_global_sequence.py \
    --checkpoint "${BASE_CHECKPOINT}" \
    --descriptor-head-checkpoint "${DESCRIPTOR_CHECKPOINT}" \
    --config "${CONFIG}" \
    --model-name-or-path "${MAST3R_CHECKPOINT}" \
    --data-root /data/fcr/data/eval_data/DL3DV \
    --scene screen-000090 \
    --start-frame 0 \
    --num-frames 8 \
    --image-size 512 \
    --exposure 0.10 \
    --scene-graph swin-2-noncyclic \
    --kinematic-mode mst \
    --matching-conf-thr 5 \
    --min-corres-conf-thr "${min_conf}" \
    --init-focal 300 \
    --focal-reg-w 100 \
    --conf-quantile 0.75 \
    --voxel-size 0.04 \
    --amp \
    --amp-dtype bf16 \
    --device cuda \
    --allow-unsafe-checkpoint-load \
    --outdir "${outdir}" \
    > "${RUN_DIR}/${name}.log" 2>&1 &
  PIDS+=("$!")
done

status=0
for index in "${!PIDS[@]}"; do
  if wait "${PIDS[$index]}"; then
    echo "complete name=${NAMES[$index]}"
  else
    echo "failed name=${NAMES[$index]}"
    status=1
  fi
done

echo "corres_filter_sweep_done status=${status} $(date -Is)"
exit "${status}"
