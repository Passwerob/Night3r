#!/usr/bin/env bash
set -euo pipefail

cd /data/fcr/data/Night3R

PY="${PY:-/data/fcr/miniconda3/envs/stream3r/bin/python}"
GPU="${GPU:-4}"
OUT="${OUT:-outputs/dynamic_pose_trajectory_long_mvsec_20260726}"
DATA_ROOT="${DATA_ROOT:-${OUT}/input_screen}"
SCENE="${SCENE:-screen-long-outdoor_night2-mvsec}"
CONFIG="${CONFIG:-config/train_night3r_final_step100_pose_adapt100.yaml}"
GEOMETRY_CKPT="${GEOMETRY_CKPT:-outputs/night3r_final_step100_20260723/event_dark3r_night3r_final.pth}"
POSE_HEAD_CKPT="${POSE_HEAD_CKPT:-outputs/night3r_final_step100_pose_adapt100_20260726/pose_heads/pose_head_step00000100.pth}"
MAST3R_CKPT="${MAST3R_CKPT:-checkpoints/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric.pth}"

mkdir -p "${OUT}"
exec > >(tee -a "${OUT}/run.log") 2>&1

echo "== Long-range MVSEC pose trajectory =="
date -Is
echo "GPU=${GPU} SCENE=${SCENE}"

CUDA_VISIBLE_DEVICES="${GPU}" "${PY}" utils/eval_night3r_trajectory.py \
  --variant event_dark3r \
  --checkpoint "${GEOMETRY_CKPT}" \
  --pose-head-checkpoint "${POSE_HEAD_CKPT}" \
  --config "${CONFIG}" \
  --data-root "${DATA_ROOT}" \
  --model-name-or-path "${MAST3R_CKPT}" \
  --allow-unsafe-checkpoint-load \
  --event-mode real --pose-recovery direct \
  --include-scenes "${SCENE}" --max-frames-per-scene 441 \
  --export-trajectories --num-workers 0 --device cuda --amp --amp-dtype bf16 \
  --output "${OUT}/night3r_long_mvsec.json"

"${PY}" utils/render_pose_trajectory_comparison.py \
  --inputs "${OUT}/night3r_long_mvsec.json" \
  --labels "Night3R (ours)" \
  --scene "${SCENE}" \
  --title "Long-range MVSEC trajectory: 441 frames / 130 m GT path" \
  --outdir "${OUT}/figures/night3r_vs_gt"

touch "${OUT}/COMPLETE"
date -Is
