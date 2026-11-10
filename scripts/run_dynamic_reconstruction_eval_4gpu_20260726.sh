#!/usr/bin/env bash
set -euo pipefail

cd /data/fcr/data/Night3R

PY="${PY:-/data/fcr/miniconda3/envs/stream3r/bin/python}"
OUT_ROOT="outputs/dynamic_reconstruction_eval_step100_20260726"
CHECKPOINT="outputs/night3r_selected_pairs_step100_native_rgb_viridis_20260725/weights/event_dark3r_night3r_final_step100.pth"
CONFIG="config/train_night3r_global_final_continue_200_20260723.yaml"
MAST3R="checkpoints/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric.pth"
COMMON=(
  --checkpoint "${CHECKPOINT}"
  --config "${CONFIG}"
  --model-name-or-path "${MAST3R}"
  --image-size 512 512
  --amp-dtype bf16
  --device cuda
  --visuals-per-scene 2
  --allow-unsafe-checkpoint-load
  --resume
)

mkdir -p "${OUT_ROOT}/logs"
exec > >(tee -a "${OUT_ROOT}/launcher.log") 2>&1

mapfile -t MVSEC_SCENES < <(find test/MVSEC -mindepth 1 -maxdepth 1 -type d -printf '%f\n' | sort)
mapfile -t DSEC_SCENES < <(find test/DSEC -mindepth 1 -maxdepth 1 -type d -printf '%f\n' | sort)
mapfile -t DL3DV_SCENES < <(find test/DL3DV -mindepth 1 -maxdepth 1 -type d -printf '%f\n' | sort)
mapfile -t CHUNK_SCENES < <(find /data/fcr/data/train_data/DL3DV_DA3_pseudo_chunk016 -mindepth 1 -maxdepth 1 -type d -name 'screen-*' -printf '%f\n' | sort)

DL_SPLIT=$(((${#DL3DV_SCENES[@]} + 1) / 2))
CHUNK_Q=$(((${#CHUNK_SCENES[@]} + 3) / 4))

run_eval() {
  local gpu="$1"
  local name="$2"
  shift 2
  echo "[$(date -Is)] start gpu=${gpu} name=${name}"
  CUDA_VISIBLE_DEVICES="${gpu}" NIGHT3R_CPU_THREADS=4 "${PY}" \
    utils/eval_dynamic_reconstruction.py \
    "${COMMON[@]}" \
    "$@" \
    > "${OUT_ROOT}/logs/${name}.log" 2>&1
  echo "[$(date -Is)] done gpu=${gpu} name=${name}"
}

worker4() {
  run_eval 4 mvsec_native \
    --dataset-name MVSEC \
    --data-root test/MVSEC \
    --scenes "${MVSEC_SCENES[@]}" \
    --max-centers-per-scene 10 \
    --outdir "${OUT_ROOT}/shards/gpu4_mvsec_native"
  run_eval 4 chunk016_q1 \
    --dataset-name DL3DV_CHUNK016 \
    --data-root /data/fcr/data/train_data/DL3DV_DA3_pseudo_chunk016 \
    --scenes "${CHUNK_SCENES[@]:0:${CHUNK_Q}}" \
    --max-centers-per-scene 6 \
    --synthetic-exposure 0.10 \
    --outdir "${OUT_ROOT}/shards/gpu4_chunk016_q1"
}

worker5() {
  run_eval 5 dsec_exp010 \
    --dataset-name DSEC \
    --data-root test/DSEC \
    --scenes "${DSEC_SCENES[@]}" \
    --max-centers-per-scene 24 \
    --synthetic-exposure 0.10 \
    --outdir "${OUT_ROOT}/shards/gpu5_dsec_exp010"
  run_eval 5 chunk016_q2 \
    --dataset-name DL3DV_CHUNK016 \
    --data-root /data/fcr/data/train_data/DL3DV_DA3_pseudo_chunk016 \
    --scenes "${CHUNK_SCENES[@]:${CHUNK_Q}:${CHUNK_Q}}" \
    --max-centers-per-scene 6 \
    --synthetic-exposure 0.10 \
    --outdir "${OUT_ROOT}/shards/gpu5_chunk016_q2"
}

worker6() {
  run_eval 6 dl3dv_heldout_h1 \
    --dataset-name DL3DV \
    --data-root test/DL3DV \
    --scenes "${DL3DV_SCENES[@]:0:${DL_SPLIT}}" \
    --max-centers-per-scene 8 \
    --synthetic-exposure 0.10 \
    --outdir "${OUT_ROOT}/shards/gpu6_dl3dv_heldout_h1"
  run_eval 6 chunk016_q3 \
    --dataset-name DL3DV_CHUNK016 \
    --data-root /data/fcr/data/train_data/DL3DV_DA3_pseudo_chunk016 \
    --scenes "${CHUNK_SCENES[@]:$((2 * CHUNK_Q)):${CHUNK_Q}}" \
    --max-centers-per-scene 6 \
    --synthetic-exposure 0.10 \
    --outdir "${OUT_ROOT}/shards/gpu6_chunk016_q3"
}

worker7() {
  run_eval 7 dl3dv_heldout_h2 \
    --dataset-name DL3DV \
    --data-root test/DL3DV \
    --scenes "${DL3DV_SCENES[@]:${DL_SPLIT}}" \
    --max-centers-per-scene 8 \
    --synthetic-exposure 0.10 \
    --outdir "${OUT_ROOT}/shards/gpu7_dl3dv_heldout_h2"
  run_eval 7 chunk016_q4 \
    --dataset-name DL3DV_CHUNK016 \
    --data-root /data/fcr/data/train_data/DL3DV_DA3_pseudo_chunk016 \
    --scenes "${CHUNK_SCENES[@]:$((3 * CHUNK_Q))}" \
    --max-centers-per-scene 6 \
    --synthetic-exposure 0.10 \
    --outdir "${OUT_ROOT}/shards/gpu7_chunk016_q4"
}

echo "dynamic_eval_start $(date -Is)"
worker4 &
P4=$!
worker5 &
P5=$!
worker6 &
P6=$!
worker7 &
P7=$!

STATUS=0
for pid in "${P4}" "${P5}" "${P6}" "${P7}"; do
  if ! wait "${pid}"; then
    STATUS=1
  fi
done

if [[ "${STATUS}" -eq 0 ]]; then
  "${PY}" utils/merge_dynamic_reconstruction_eval.py \
    --root "${OUT_ROOT}" \
    --output-dir "${OUT_ROOT}/merged" \
    > "${OUT_ROOT}/logs/merge.log" 2>&1
  touch "${OUT_ROOT}/EVAL_COMPLETE"
fi

echo "dynamic_eval_done status=${STATUS} $(date -Is)"
exit "${STATUS}"
