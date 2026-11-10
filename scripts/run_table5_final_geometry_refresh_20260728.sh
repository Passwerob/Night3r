#!/usr/bin/env bash
set -euo pipefail

cd /data/fcr/data/Night3R

PY="/data/fcr/miniconda3/envs/stream3r/bin/python"
CHECKPOINT="outputs/night3r_final_step100_20260723/event_dark3r_night3r_final.pth"
BACKBONE="checkpoints/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric.pth"
OUTDIR="outputs/table5_final_geometry_refresh_20260728"
LORALIB_VENDOR="${LORALIB_VENDOR:-/tmp/loralib_vendor}"
MAX_BATCHES=128

mkdir -p "${OUTDIR}" "${LORALIB_VENDOR}"
ln -sfn /data/fcr/miniconda3/lib/python3.8/site-packages/loralib "${LORALIB_VENDOR}/loralib"
export PYTHONPATH="${LORALIB_VENDOR}${PYTHONPATH:+:${PYTHONPATH}}"

config_for() {
  case "$1" in
    e020) printf '%s\n' config/eval_final_geometry_extreme_e020_20260728.yaml ;;
    e010) printf '%s\n' config/eval_final_geometry_extreme_e010_20260728.yaml ;;
    e005) printf '%s\n' config/eval_final_geometry_extreme_e005_20260728.yaml ;;
    *) echo "Unknown exposure tag: $1" >&2; exit 2 ;;
  esac
}

root_for() {
  case "$1" in
    dsec) printf '%s\n' /data/fcr/data/train_data/DSEC_test ;;
    mvsec) printf '%s\n' /data/fcr/data/train_data/MVSEC_outdoor_night2_test ;;
    *) echo "Unknown dataset: $1" >&2; exit 2 ;;
  esac
}

run_one() {
  local gpu="$1" dataset="$2" exposure="$3" batches="$4" suffix="$5"
  local config root output log
  config="$(config_for "${exposure}")"
  root="$(root_for "${dataset}")"
  output="${OUTDIR}/${dataset}_night3r_${exposure}_${suffix}.json"
  log="${output%.json}.stdout.log"
  if [[ -s "${output}" ]]; then
    echo "[skip] ${output}"
    return
  fi
  echo "[start] gpu=${gpu} dataset=${dataset} exposure=${exposure} batches=${batches}"
  CUDA_VISIBLE_DEVICES="${gpu}" PYTHONUNBUFFERED=1 "${PY}" \
    utils/eval_event_dark3r_variant.py \
    --variant event_dark3r \
    --checkpoint "${CHECKPOINT}" \
    --event-mode real \
    --input lowlight \
    --config "${config}" \
    --data-root "${root}" \
    --model-name-or-path "${BACKBONE}" \
    --allow-unsafe-checkpoint-load \
    --device cuda \
    --amp \
    --amp-dtype bf16 \
    --batch-size 1 \
    --max-batches "${batches}" \
    --pair-sampling fixed \
    --pair-stride 1 \
    --min-pair-distance 1 \
    --max-pair-distance 1 \
    --num-workers 0 \
    --seed 17 \
    --output "${output}" > "${log}" 2>&1
  echo "[done] ${output}"
}

if [[ "${1:-}" == "--smoke" ]]; then
  run_one "${2:-0}" dsec e020 1 smoke1_darkened
  exit 0
fi

run_one 0 dsec e020 "${MAX_BATCHES}" 128 & pid0=$!
run_one 1 dsec e010 "${MAX_BATCHES}" 128 & pid1=$!
run_one 2 dsec e005 "${MAX_BATCHES}" 128 & pid2=$!
run_one 3 mvsec e020 "${MAX_BATCHES}" 128 & pid3=$!
run_one 4 mvsec e010 "${MAX_BATCHES}" 128 & pid4=$!
run_one 5 mvsec e005 "${MAX_BATCHES}" 128 & pid5=$!

status=0
for pid in "${pid0}" "${pid1}" "${pid2}" "${pid3}" "${pid4}" "${pid5}"; do
  wait "${pid}" || status=1
done
if [[ "${status}" -ne 0 ]]; then
  echo "One or more table-5 evaluations failed. Inspect ${OUTDIR}." >&2
  exit 1
fi
touch "${OUTDIR}/COMPLETE"
echo "[complete] ${OUTDIR}"
