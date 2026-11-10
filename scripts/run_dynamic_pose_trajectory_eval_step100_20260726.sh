#!/usr/bin/env bash
set -euo pipefail

cd /data/fcr/data/Night3R

PY="${PY:-/data/fcr/miniconda3/envs/stream3r/bin/python}"
OUT="${OUT:-outputs/dynamic_pose_trajectory_step100_20260726}"
CONFIG="${CONFIG:-config/train_night3r_final_step100_pose_adapt100.yaml}"
GEOMETRY_CKPT="${GEOMETRY_CKPT:-outputs/night3r_final_step100_20260723/event_dark3r_night3r_final.pth}"
POSE_HEAD_CKPT="${POSE_HEAD_CKPT:-outputs/night3r_final_step100_pose_adapt100_20260726/pose_heads/pose_head_step00000100.pth}"
MAST3R_CKPT="${MAST3R_CKPT:-checkpoints/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric.pth}"
MVSEC_SCENE="${MVSEC_SCENE:-screen-000157-mvsec}"
DL3DV_SCENE="${DL3DV_SCENE:-screen-000088}"

mkdir -p "${OUT}"
exec > >(tee -a "${OUT}/launcher.log") 2>&1

run_night3r() {
  local gpu="$1" dataset="$2" root="$3" scene="$4" exposure="$5"
  local lowlight_args=()
  if [[ "${exposure}" != "none" ]]; then
    lowlight_args+=(--lowlight-exposure "${exposure}")
  fi
  CUDA_VISIBLE_DEVICES="${gpu}" "${PY}" utils/eval_night3r_trajectory.py \
    --variant event_dark3r \
    --checkpoint "${GEOMETRY_CKPT}" \
    --pose-head-checkpoint "${POSE_HEAD_CKPT}" \
    --config "${CONFIG}" \
    --data-root "${root}" \
    --model-name-or-path "${MAST3R_CKPT}" \
    --allow-unsafe-checkpoint-load \
    --event-mode real --pose-recovery direct \
    --include-scenes "${scene}" --max-frames-per-scene 49 \
    --export-trajectories --num-workers 0 --device cuda --amp --amp-dtype bf16 \
    "${lowlight_args[@]}" \
    --output "${OUT}/${dataset}_night3r.json" \
    > "${OUT}/${dataset}_night3r.log" 2>&1
}

run_mast3r() {
  local gpu="$1" dataset="$2" root="$3" scene="$4" exposure="$5"
  local lowlight_args=()
  if [[ "${exposure}" != "none" ]]; then
    lowlight_args+=(--lowlight-exposure "${exposure}")
  fi
  CUDA_VISIBLE_DEVICES="${gpu}" "${PY}" utils/eval_night3r_trajectory.py \
    --variant mast3r \
    --config "${CONFIG}" \
    --data-root "${root}" \
    --model-name-or-path "${MAST3R_CKPT}" \
    --allow-unsafe-checkpoint-load \
    --pose-recovery pointmap_sim3 \
    --include-scenes "${scene}" --max-frames-per-scene 49 \
    --export-trajectories --conf-quantile 0.70 --max-pointmap-residual 0.25 \
    --max-pnp-points 50000 --min-pnp-inliers 512 \
    --num-workers 0 --device cuda --amp --amp-dtype bf16 \
    "${lowlight_args[@]}" \
    --output "${OUT}/${dataset}_mast3r.json" \
    > "${OUT}/${dataset}_mast3r.log" 2>&1
}

run_night3r 4 mvsec test/MVSEC "${MVSEC_SCENE}" none & pid4=$!
run_mast3r 5 mvsec test/MVSEC "${MVSEC_SCENE}" none & pid5=$!
run_night3r 6 dl3dv test/DL3DV "${DL3DV_SCENE}" 0.10 & pid6=$!
run_mast3r 7 dl3dv test/DL3DV "${DL3DV_SCENE}" 0.10 & pid7=$!

status=0
for pid in "${pid4}" "${pid5}" "${pid6}" "${pid7}"; do
  wait "${pid}" || status=1
done
if [[ "${status}" != "0" ]]; then
  echo "At least one trajectory evaluation failed; inspect ${OUT}/*.log" >&2
  exit 1
fi

"${PY}" utils/render_pose_trajectory_comparison.py \
  --inputs "${OUT}/mvsec_night3r.json" "${OUT}/mvsec_mast3r.json" \
  --labels "Night3R (ours)" "MASt3R" \
  --scene "${MVSEC_SCENE}" \
  --title "MVSEC continuous trajectory under low light" \
  --outdir "${OUT}/figures/mvsec"

"${PY}" utils/render_pose_trajectory_comparison.py \
  --inputs "${OUT}/dl3dv_night3r.json" "${OUT}/dl3dv_mast3r.json" \
  --labels "Night3R (ours)" "MASt3R" \
  --scene "${DL3DV_SCENE}" \
  --title "Held-out DL3DV continuous trajectory at 10% exposure" \
  --outdir "${OUT}/figures/dl3dv"

touch "${OUT}/COMPLETE"
date -Is
