#!/usr/bin/env bash
set -euo pipefail

cd /data/fcr/data/Night3R

PY="${PY:-/data/fcr/miniconda3/envs/stream3r/bin/python}"
ROOT="outputs/splits/night3r_dsec_extended_holdout_v2/new_scene_only"
AUDIT="test/DSEC/screen-000004-dsec/meta/test_overlap_audit.json"
OUT="outputs/paper_main_geometry_heldout_dsec_20260721"
LORALIB_VENDOR="${LORALIB_VENDOR:-/tmp/loralib_vendor}"

mkdir -p "${OUT}" "${LORALIB_VENDOR}"
ln -sfn /data/fcr/miniconda3/lib/python3.8/site-packages/loralib "${LORALIB_VENDOR}/loralib"
export PYTHONPATH="${LORALIB_VENDOR}${PYTHONPATH:+:${PYTHONPATH}}"

pairs=(120 360 600 840)
pids=()

for slot in "${!pairs[@]}"; do
  idx1="${pairs[$slot]}"
  idx2="$((idx1 + 1))"
  pair_out="${OUT}/pair_$(printf '%03d' "${idx1}")_$(printf '%03d' "${idx2}")"
  CUDA_VISIBLE_DEVICES="${slot}" "${PY}" utils/export_mvsec_final_showcase.py \
    --data-root "${ROOT}" \
    --scene screen-000004-dsec \
    --idx1 "${idx1}" \
    --idx2 "${idx2}" \
    --dataset-name DSEC \
    --source-sequence zurich_city_11_b \
    --split-audit-file "${AUDIT}" \
    --exposure 0.10 \
    --geometry-scale-mode raw \
    --config config/train_night3r_mvsec_only_adapt_1k_from_best.yaml \
    --checkpoint outputs/night3r_mvsec_only_adapt_1k_from_best/last_event_dark3r.pth \
    --backbone checkpoints/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric.pth \
    --device cuda:0 \
    --outdir "${pair_out}" \
    > "${OUT}/pair_${idx1}_${idx2}.log" 2>&1 &
  pids+=("$!")
done

status=0
for pid in "${pids[@]}"; do
  if ! wait "${pid}"; then
    status=1
  fi
done

if [[ "${status}" -ne 0 ]]; then
  echo "One or more held-out DSEC exports failed. Check ${OUT}/*.log." >&2
  exit "${status}"
fi

"${PY}" utils/compose_main_geometry_figure.py \
  --scene-root "${OUT}/pair_120_121" \
  --scene-root "${OUT}/pair_360_361" \
  --scene-root "${OUT}/pair_600_601" \
  --scene-root "${OUT}/pair_840_841" \
  --outdir "${OUT}/composed"

echo "Held-out DSEC main geometry figure: ${OUT}/composed"
