#!/usr/bin/env bash
set -euo pipefail

cd /data/fcr/data/Night3R

PY="${PY:-/data/fcr/miniconda3/envs/stream3r/bin/python}"
CHECKPOINT="${CHECKPOINT:-outputs/night3r_oldtest_recovery_from_best_currenttest_train_4gpu_3k/last_event_dark3r.pth}"
CONFIG="${CONFIG:-config/train_night3r_oldtest_recovery_from_best_currenttest_train.yaml}"
OUTDIR="${OUTDIR:-outputs/night3r_oldtest_recovery_from_best_currenttest_train_4gpu_3k/eval_old_mvsec_8scene16_protocol}"
EVAL_GPU="${EVAL_GPU:-7}"
MAST3R_CKPT="${MAST3R_CKPT:-checkpoints/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric.pth}"
MVSEC_OLD_TEST_ROOT="${MVSEC_OLD_TEST_ROOT:-/data/fcr/data/train_data/MVSEC_outdoor_night2_test_prev_8scene_20260701}"

mkdir -p "${OUTDIR}"

CUDA_VISIBLE_DEVICES="${EVAL_GPU}" "${PY}" utils/eval_event_dark3r_variant.py \
  --variant event_dark3r \
  --event-mode real \
  --checkpoint "${CHECKPOINT}" \
  --config "${CONFIG}" \
  --data-root "${MVSEC_OLD_TEST_ROOT}" \
  --model-name-or-path "${MAST3R_CKPT}" \
  --allow-unsafe-checkpoint-load \
  --device cuda \
  --amp \
  --amp-dtype bf16 \
  --batch-size 1 \
  --max-batches 128 \
  --max-pairs-per-scene 16 \
  --pair-sampling fixed \
  --pair-stride 1 \
  --min-pair-distance 1 \
  --max-pair-distance 1 \
  --output "${OUTDIR}/mvsec_old_test_real_8scene16_128.json" \
  > "${OUTDIR}/mvsec_old_test_real_8scene16_128.stdout.log" 2>&1

"${PY}" - <<'PY'
import json
import os
from pathlib import Path

outdir = Path(os.environ.get(
    "OUTDIR",
    "outputs/night3r_oldtest_recovery_from_best_currenttest_train_4gpu_3k/eval_old_mvsec_8scene16_protocol",
))
path = outdir / "mvsec_old_test_real_8scene16_128.json"
metrics = json.loads(path.read_text())["dense_metrics"]
keys = [
    "depth_raw_absrel",
    "depth_raw_psnr_db",
    "depth_raw_delta_1.25",
    "depth_median_aligned_absrel",
    "depth_median_aligned_psnr_db",
    "depth_median_aligned_delta_1.25",
    "warp_rgb_psnr_db",
]
lines = [
    "| Protocol | Raw AbsRel ↓ | Raw PSNR ↑ | Raw δ1 ↑ | Median AbsRel ↓ | Median PSNR ↑ | Median δ1 ↑ | Warp PSNR ↑ |",
    "|---|---:|---:|---:|---:|---:|---:|---:|",
    "| MVSEC old 8scene16 lowlight | "
    + " | ".join(f"{float(metrics.get(k, float('nan'))):.4f}" for k in keys)
    + " |",
]
(outdir / "summary.md").write_text("\n".join(lines) + "\n")
print("\n".join(lines))
PY
