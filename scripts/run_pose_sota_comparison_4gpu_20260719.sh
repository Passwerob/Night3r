#!/usr/bin/env bash
set -euo pipefail

cd /data/fcr/data/Night3R

PY=/data/fcr/miniconda3/envs/stream3r/bin/python
OUTDIR=outputs/pose_sota_comparison_20260719
CONFIG=config/train_night3r_pose_head_mvsec_dl3dv_3k.yaml
MAST3R_CKPT=checkpoints/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric.pth
DARK3R_LORA=checkpoints/dark3r_pretrained/weights/0049_lora.pth
VGGT_CKPT=/data/fcr/ckpt/model.pt
EVENTVGGT_CKPT=/data/fcr/EventVGGT/checkpoint.pt
MVSEC_ROOT=/data/fcr/data/train_data/MVSEC_outdoor_night2_test
DL3DV_ROOT=/data/fcr/data/eval_data/DL3DV
LORALIB_VENDOR=${LORALIB_VENDOR:-/tmp/loralib_vendor}

mkdir -p "${OUTDIR}" "${LORALIB_VENDOR}"
ln -sfn /data/fcr/miniconda3/lib/python3.8/site-packages/loralib "${LORALIB_VENDOR}/loralib"
export PYTHONPATH="${LORALIB_VENDOR}${PYTHONPATH:+:${PYTHONPATH}}"

dataset_root() {
  [[ "$1" == "mvsec" ]] && printf '%s\n' "${MVSEC_ROOT}" || printf '%s\n' "${DL3DV_ROOT}"
}

lowlight_args() {
  [[ "$1" == "dl3dv" ]] && printf '%s\n' "--lowlight-exposure" "0.05"
}

run_vggt_family() {
  local gpu="$1" variant="$2" dataset="$3"
  local output="${OUTDIR}/${dataset}_${variant}.json"
  if [[ -s "${output}" ]]; then
    echo "[skip] ${output}"
    return
  fi
  local args=()
  mapfile -t args < <(lowlight_args "${dataset}")
  CUDA_VISIBLE_DEVICES="${gpu}" PYTHONUNBUFFERED=1 "${PY}" \
    utils/eval_vggt_trajectory.py \
    --variant "${variant}" \
    --checkpoint "$([[ "${variant}" == "vggt" ]] && printf '%s' "${VGGT_CKPT}" || printf '%s' "${EVENTVGGT_CKPT}")" \
    --base-vggt-checkpoint "${VGGT_CKPT}" \
    --config "${CONFIG}" \
    --data-root "$(dataset_root "${dataset}")" \
    --max-scenes 6 --max-frames-per-scene 32 \
    --num-workers 0 --seed 17 --device cuda --amp --amp-dtype bf16 \
    "${args[@]}" \
    --output "${output}" > "${output%.json}.log" 2>&1
}

run_geometry_family() {
  local gpu="$1" variant="$2" name="$3" dataset="$4"
  local output="${OUTDIR}/${dataset}_${name}.json"
  if [[ -s "${output}" ]]; then
    echo "[skip] ${output}"
    return
  fi
  local args=()
  mapfile -t args < <(lowlight_args "${dataset}")
  if [[ "${variant}" == "dark3r_lora" ]]; then
    args+=(--dark3r-lora-checkpoint "${DARK3R_LORA}")
  fi
  CUDA_VISIBLE_DEVICES="${gpu}" PYTHONUNBUFFERED=1 "${PY}" \
    utils/eval_night3r_trajectory.py \
    --variant "${variant}" \
    --config "${CONFIG}" \
    --data-root "$(dataset_root "${dataset}")" \
    --model-name-or-path "${MAST3R_CKPT}" \
    --allow-unsafe-checkpoint-load \
    --pose-recovery pointmap_sim3 \
    --max-scenes 6 --max-frames-per-scene 32 \
    --conf-quantile 0.70 --max-pointmap-residual 2.0 \
    --max-pnp-points 50000 --min-pnp-inliers 512 \
    --num-workers 0 --seed 17 --device cuda --amp --amp-dtype bf16 \
    "${args[@]}" \
    --output "${output}" > "${output%.json}.log" 2>&1
}

(
  run_vggt_family 4 vggt mvsec
  run_vggt_family 4 vggt dl3dv
) & pid4=$!
(
  run_geometry_family 5 mast3r mast3r mvsec
  run_geometry_family 5 mast3r mast3r dl3dv
) & pid5=$!
(
  run_geometry_family 6 dark3r_lora dark3r mvsec
  run_geometry_family 6 dark3r_lora dark3r dl3dv
) & pid6=$!
(
  run_vggt_family 7 eventvggt mvsec
  run_vggt_family 7 eventvggt dl3dv
) & pid7=$!

status=0
for pid in "${pid4}" "${pid5}" "${pid6}" "${pid7}"; do
  wait "${pid}" || status=1
done
if [[ "${status}" != "0" ]]; then
  echo "At least one pose baseline failed; inspect ${OUTDIR}/*.log" >&2
  exit 1
fi

"${PY}" utils/summarize_pose_sota_comparison.py
touch "${OUTDIR}/COMPLETE"
echo "Pose comparison complete: ${OUTDIR}/summary.md"
