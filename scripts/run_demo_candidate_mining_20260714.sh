#!/usr/bin/env bash
set -euo pipefail

cd /data/fcr/data/Night3R

if [[ "$#" -ne 1 ]]; then
  echo "usage: $0 GPU" >&2
  exit 2
fi

GPU="$1"
PY="${PY:-/data/fcr/miniconda3/envs/stream3r/bin/python}"
LORALIB_VENDOR="${LORALIB_VENDOR:-/tmp/loralib_vendor}"
OUTDIR="outputs/demo_candidate_mining_20260714"
mkdir -p "${LORALIB_VENDOR}" "${OUTDIR}"
ln -sfn /data/fcr/miniconda3/lib/python3.8/site-packages/loralib "${LORALIB_VENDOR}/loralib"
export PYTHONPATH="${LORALIB_VENDOR}${PYTHONPATH:+:${PYTHONPATH}}"

CUDA_VISIBLE_DEVICES="${GPU}" "${PY}" utils/mine_night3r_demo_candidates.py \
  --checkpoint outputs/night3r_brg_dl3dv_recover_from_stage2_6k/last_event_dark3r.pth \
  --config config/train_night3r_brg_dl3dv_recover_from_stage2.yaml \
  --model-name-or-path checkpoints/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric.pth \
  --datasets \
    mvsec=/data/fcr/data/train_data/MVSEC_outdoor_night2_test \
    dsec=/data/fcr/data/train_data/DSEC_test \
  --max-pairs-per-dataset 256 \
  --top-k-per-dataset 12 \
  --diversity-bucket-size 16 \
  --device cuda \
  --amp-dtype bf16 \
  --seed 17 \
  --allow-unsafe-checkpoint-load \
  --outdir "${OUTDIR}" \
  > "${OUTDIR}/run.log" 2>&1

echo "[demo candidate mining done] ${OUTDIR}"
