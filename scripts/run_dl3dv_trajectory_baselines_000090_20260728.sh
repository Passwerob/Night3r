#!/usr/bin/env bash
set -euo pipefail

cd /data/fcr/data/Night3R

PY="${PY:-/data/fcr/miniconda3/envs/stream3r/bin/python}"
OUT="${OUT:-outputs/heldout_dl3dv_trajectory_shortlist_20260728/four_method_screen000090}"
DATA_ROOT="${DATA_ROOT:-/data/fcr/data/Night3R/test/DL3DV}"
SCENE="${SCENE:-screen-000090}"
CONFIG="${CONFIG:-config/train_night3r_final_step100_pose_adapt100.yaml}"
MAST3R_CKPT="${MAST3R_CKPT:-checkpoints/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric.pth}"
DARK3R_LORA="${DARK3R_LORA:-checkpoints/dark3r_pretrained/weights/0049_lora.pth}"
VGGT_CKPT="${VGGT_CKPT:-/data/fcr/ckpt/model.pt}"
LORALIB_VENDOR="${LORALIB_VENDOR:-/tmp/loralib_vendor}"

mkdir -p "${OUT}/inputs" "${OUT}/json" "${OUT}/logs" "${LORALIB_VENDOR}"
ln -sfn /data/fcr/miniconda3/lib/python3.8/site-packages/loralib "${LORALIB_VENDOR}/loralib"
export PYTHONPATH="${LORALIB_VENDOR}${PYTHONPATH:+:${PYTHONPATH}}"
exec > >(tee -a "${OUT}/launcher.log") 2>&1

run_mast3r() {
  CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1 "${PY}" utils/eval_night3r_trajectory.py \
    --variant mast3r --config "${CONFIG}" --data-root "${DATA_ROOT}" \
    --model-name-or-path "${MAST3R_CKPT}" --allow-unsafe-checkpoint-load \
    --pose-recovery pointmap_sim3 --include-scenes "${SCENE}" --max-frames-per-scene 49 \
    --lowlight-exposure 0.10 --export-trajectories --conf-quantile 0.70 \
    --max-pointmap-residual 0.25 --max-pnp-points 50000 --min-pnp-inliers 512 \
    --num-workers 0 --device cuda --amp --amp-dtype bf16 \
    --output "${OUT}/json/mast3r.json" > "${OUT}/logs/mast3r.log" 2>&1
}

run_dark3r() {
  CUDA_VISIBLE_DEVICES=1 PYTHONUNBUFFERED=1 "${PY}" utils/eval_night3r_trajectory.py \
    --variant dark3r_lora --dark3r-lora-checkpoint "${DARK3R_LORA}" \
    --config "${CONFIG}" --data-root "${DATA_ROOT}" \
    --model-name-or-path "${MAST3R_CKPT}" --allow-unsafe-checkpoint-load \
    --pose-recovery pointmap_sim3 --include-scenes "${SCENE}" --max-frames-per-scene 49 \
    --lowlight-exposure 0.10 --export-trajectories --conf-quantile 0.70 \
    --max-pointmap-residual 0.25 --max-pnp-points 50000 --min-pnp-inliers 512 \
    --num-workers 0 --device cuda --amp --amp-dtype bf16 \
    --output "${OUT}/json/dark3r.json" > "${OUT}/logs/dark3r.log" 2>&1
}

run_vggt() {
  local isolated_root="${OUT}/inputs/${SCENE}"
  mkdir -p "${isolated_root}"
  ln -sfn "${DATA_ROOT}/${SCENE}" "${isolated_root}/${SCENE}"
  CUDA_VISIBLE_DEVICES=2 PYTHONUNBUFFERED=1 "${PY}" utils/eval_vggt_trajectory.py \
    --variant vggt --checkpoint "${VGGT_CKPT}" --base-vggt-checkpoint "${VGGT_CKPT}" \
    --config "${CONFIG}" --data-root "${isolated_root}" --max-scenes 1 --max-frames-per-scene 49 \
    --lowlight-exposure 0.10 --export-trajectories --num-workers 0 --device cuda --amp --amp-dtype bf16 \
    --output "${OUT}/json/vggt.json" > "${OUT}/logs/vggt.log" 2>&1
}

run_mast3r & pid_mast=$!
run_dark3r & pid_dark=$!
run_vggt & pid_vggt=$!
status=0
wait "${pid_mast}" || status=1
wait "${pid_dark}" || status=1
wait "${pid_vggt}" || status=1
if [[ "${status}" -ne 0 ]]; then
  echo "One or more baseline evaluations failed. Inspect ${OUT}/logs." >&2
  exit "${status}"
fi
touch "${OUT}/COMPLETE"
date -Is
