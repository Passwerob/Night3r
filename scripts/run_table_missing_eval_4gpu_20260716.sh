#!/usr/bin/env bash
set -euo pipefail

cd /data/fcr/data/Night3R

PY=/data/fcr/miniconda3/envs/stream3r/bin/python
OUTDIR=outputs/table_missing_completion_20260716
MAST3R_CKPT=checkpoints/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric.pth
VGGT_CKPT=/data/fcr/ckpt/model.pt
CLEAN_CONFIG=config/train_night3r_mvsec_only_adapt_1k_from_best.yaml
DSEC_ROOT=/data/fcr/data/train_data/DSEC_test
MVSEC_ROOT=/data/fcr/data/train_data/MVSEC_outdoor_night2_test

mkdir -p "${OUTDIR}/vggt_clean" "${OUTDIR}/fusion_dsec_holdout"

run_vggt_clean() {
  local gpu="$1" dataset="$2" root="$3"
  local output="${OUTDIR}/vggt_clean/${dataset}_vggt_clean_128.json"
  local log="${output%.json}.log"
  if [[ -s "${output}" ]]; then
    echo "[skip] ${output}"
    return
  fi
  echo "[start] gpu=${gpu} method=VGGT dataset=${dataset} input=clean"
  CUDA_VISIBLE_DEVICES="${gpu}" PYTHONUNBUFFERED=1 "${PY}" utils/eval_vggt_baseline.py \
    --variant vggt \
    --checkpoint "${VGGT_CKPT}" \
    --config "${CLEAN_CONFIG}" \
    --data-root "${root}" \
    --max-batches 128 \
    --batch-size 1 \
    --pair-sampling fixed \
    --pair-stride 1 \
    --min-pair-distance 1 \
    --max-pair-distance 1 \
    --num-workers 0 \
    --seed 17 \
    --amp-dtype bf16 \
    --output "${output}" > "${log}" 2>&1
  echo "[done] ${output}"
}

run_fusion_holdout() {
  local gpu="$1" role="$2" config="$3" checkpoint="$4"
  local output="${OUTDIR}/fusion_dsec_holdout/${role}_real_612.json"
  local log="${output%.json}.log"
  if [[ -s "${output}" ]]; then
    echo "[skip] ${output}"
    return
  fi
  echo "[start] gpu=${gpu} fusion=${role} dataset=DSEC-612"
  CUDA_VISIBLE_DEVICES="${gpu}" PYTHONUNBUFFERED=1 "${PY}" utils/eval_event_dark3r_variant.py \
    --variant event_dark3r \
    --checkpoint "${checkpoint}" \
    --config "${config}" \
    --data-root "${DSEC_ROOT}" \
    --model-name-or-path "${MAST3R_CKPT}" \
    --allow-unsafe-checkpoint-load \
    --event-mode real \
    --no-lowlight-aug \
    --device cuda \
    --amp \
    --amp-dtype bf16 \
    --batch-size 1 \
    --max-batches 612 \
    --pair-sampling fixed \
    --pair-stride 1 \
    --min-pair-distance 1 \
    --max-pair-distance 1 \
    --num-workers 0 \
    --seed 17 \
    --output "${output}" > "${log}" 2>&1
  echo "[done] ${output}"
}

(
  run_vggt_clean 4 dsec "${DSEC_ROOT}"
  run_fusion_holdout 4 full_brg_one_gpu \
    config/exp3_full_brg_20260714.yaml \
    outputs/exp3_full_brg_seed0_1gpu_3k_20260714/last_event_dark3r.pth
  run_fusion_holdout 4 concat_mlp \
    config/exp6_fusion_concat_mlp_20260715.yaml \
    outputs/exp6_fusion_concat_mlp_1gpu_3k_20260715/last_event_dark3r.pth
) & pid4=$!

(
  run_vggt_clean 5 mvsec "${MVSEC_ROOT}"
  run_fusion_holdout 5 no_event_stats \
    config/exp1_brg_no_event_stats_20260712.yaml \
    outputs/exp1_ablation_no_event_stats_1gpu_3k_20260712/last_event_dark3r.pth
  run_fusion_holdout 5 additive \
    config/exp6_fusion_additive_20260715.yaml \
    outputs/exp6_fusion_additive_1gpu_3k_20260715/last_event_dark3r.pth
) & pid5=$!

(
  run_fusion_holdout 6 legacy_tensor \
    config/exp1_legacy_tensor_20260712.yaml \
    outputs/exp1_ablation_legacy_tensor_1gpu_3k_20260712/last_event_dark3r.pth
) & pid6=$!

(
  run_fusion_holdout 7 replace_rgb \
    config/exp1_brg_replace_rgb_20260712.yaml \
    outputs/exp1_ablation_replace_rgb_1gpu_3k_20260712/last_event_dark3r.pth
) & pid7=$!

failed=0
for pid in "${pid4}" "${pid5}" "${pid6}" "${pid7}"; do
  wait "${pid}" || failed=1
done
if [[ "${failed}" -ne 0 ]]; then
  echo "At least one table-completion queue failed; inspect ${OUTDIR}/**/*.log" >&2
  exit 1
fi

touch "${OUTDIR}/COMPLETE"
echo "[complete] ${OUTDIR}"
