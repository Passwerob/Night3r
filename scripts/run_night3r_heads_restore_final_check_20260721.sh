#!/usr/bin/env bash
set -euo pipefail

cd /data/fcr/data/Night3R

PY="${PY:-/data/fcr/miniconda3/envs/stream3r/bin/python}"
OUTDIR="outputs/night3r_heads_restore_final_check_20260721"
CHECKPOINT="outputs/night3r_mvsec_only_adapt_1k_from_best/last_event_dark3r.pth"
DEPTH_CONFIG="config/train_night3r_mvsec_only_adapt_1k_from_best.yaml"
POSE_CONFIG="config/train_night3r_pose_separate_branches_w5_dense2k.yaml"
POSE_CHECKPOINT="outputs/night3r_pose_separate_branches_w5_dense2k_20260720/pose_heads/pose_head_step00001200.pth"
MAST3R_CHECKPOINT="checkpoints/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric.pth"

mkdir -p "${OUTDIR}"

run_depth() {
  local gpu="$1" name="$2" root="$3" max_batches="$4" scope="$5"
  local scope_args=()
  if [[ "${scope}" != "none" ]]; then
    scope_args=(--restore-frozen-rgb-scope "${scope}")
  fi
  CUDA_VISIBLE_DEVICES="${gpu}" "${PY}" utils/eval_event_dark3r_variant.py \
    --variant event_dark3r --checkpoint "${CHECKPOINT}" \
    --config "${DEPTH_CONFIG}" --data-root "${root}" \
    --model-name-or-path "${MAST3R_CHECKPOINT}" \
    "${scope_args[@]}" \
    --allow-unsafe-checkpoint-load --event-mode real --no-lowlight-aug \
    --device cuda --amp --amp-dtype bf16 \
    --batch-size 1 --max-batches "${max_batches}" \
    --pair-sampling fixed --pair-stride 1 \
    --min-pair-distance 1 --max-pair-distance 1 \
    --num-workers 0 --seed 17 \
    --output "${OUTDIR}/${name}.json" > "${OUTDIR}/${name}.log" 2>&1
}

run_pose_long() {
  CUDA_VISIBLE_DEVICES=7 "${PY}" utils/export_night3r_long_sequence.py \
    --checkpoint "${CHECKPOINT}" \
    --pose-head-checkpoint "${POSE_CHECKPOINT}" \
    --config "${POSE_CONFIG}" \
    --model-name-or-path "${MAST3R_CHECKPOINT}" \
    --restore-frozen-rgb-scope heads \
    --data-root /data/fcr/data/eval_data/DL3DV \
    --scene screen-000090 --start-frame 0 --num-frames 8 --image-size 512 \
    --exposure 0.10 --alignment pose_head \
    --conf-quantile 0.75 --min-depth 0.0001 --max-depth 45 \
    --max-points-per-frame 18000 --voxel-size 0.01 \
    --amp --amp-dtype bf16 --device cuda --allow-unsafe-checkpoint-load \
    --outdir "${OUTDIR}/heads_dl3dv_pose_head_8f" \
    > "${OUTDIR}/heads_dl3dv_pose_head_8f.log" 2>&1
}

run_depth 4 original_mvsec_real_32 /data/fcr/data/train_data/MVSEC_outdoor_night2_test 32 none & p4=$!
run_depth 5 heads_mvsec_real_128 /data/fcr/data/train_data/MVSEC_outdoor_night2_test 128 heads & p5=$!
run_depth 6 heads_dsec_real_128 /data/fcr/data/train_data/DSEC_test 128 heads & p6=$!
run_pose_long & p7=$!

status=0
for pid in "${p4}" "${p5}" "${p6}" "${p7}"; do
  wait "${pid}" || status=1
done

exit "${status}"
