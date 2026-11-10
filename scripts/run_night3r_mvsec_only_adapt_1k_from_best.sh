#!/usr/bin/env bash
set -euo pipefail

cd /data/fcr/data/Night3R

PY="${PY:-/data/fcr/miniconda3/envs/stream3r/bin/python}"
RUN_NAME="${RUN_NAME:-night3r_mvsec_only_adapt_1k_from_best}"
SAVE_DIR="outputs/${RUN_NAME}"
EVAL_DIR="${SAVE_DIR}/eval_after_train"
LOG="${SAVE_DIR}/train_and_eval.log"
MAX_STEPS="${MAX_STEPS:-1000}"
MASTER_PORT="${MASTER_PORT:-29674}"
GPU_LIST="${GPU_LIST:-1,2,3,4}"
NPROC="${NPROC:-$(python3 - <<PY
value = "${GPU_LIST}"
print(len([item for item in value.split(",") if item.strip()]))
PY
)}"
EVAL_GPU="${EVAL_GPU:-7}"

CONFIG="${CONFIG:-config/train_night3r_mvsec_only_adapt_1k_from_best.yaml}"
INIT_CKPT="${INIT_CKPT:-outputs/night3r_oldtest_recovery_from_best_currenttest_train_4gpu_3k/last_event_dark3r.pth}"
MAST3R_CKPT="${MAST3R_CKPT:-checkpoints/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric.pth}"

MVSEC_CURRENT_ROOT="${MVSEC_CURRENT_ROOT:-/data/fcr/data/train_data/MVSEC_outdoor_night2_test}"
MVSEC_OLD_ROOT="${MVSEC_OLD_ROOT:-/data/fcr/data/train_data/MVSEC_outdoor_night2_test_prev_8scene_20260701}"
DSEC_CURRENT_ROOT="${DSEC_CURRENT_ROOT:-/data/fcr/data/train_data/DSEC_test}"
DSEC_OLD_ROOT="${DSEC_OLD_ROOT:-/data/fcr/data/Night3R/outputs/eval_roots/dsec_test_prev_2scene_20260701}"

mkdir -p "${SAVE_DIR}" "${EVAL_DIR}"
exec > >(tee -a "${LOG}") 2>&1

echo "== Night3R MVSEC-only 1k adaptation from current best =="
date -Is
echo "RUN_NAME=${RUN_NAME}"
echo "MAX_STEPS=${MAX_STEPS}"
echo "MASTER_PORT=${MASTER_PORT}"
echo "GPU_LIST=${GPU_LIST}"
echo "NPROC=${NPROC}"
echo "EVAL_GPU=${EVAL_GPU}"
echo "CONFIG=${CONFIG}"
echo "INIT_CKPT=${INIT_CKPT}"
echo "MVSEC_CURRENT_ROOT=${MVSEC_CURRENT_ROOT}"
echo "MVSEC_OLD_ROOT=${MVSEC_OLD_ROOT}"
echo "DSEC_CURRENT_ROOT=${DSEC_CURRENT_ROOT}"
echo "DSEC_OLD_ROOT=${DSEC_OLD_ROOT}"
nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits

echo "== Training =="
CUDA_VISIBLE_DEVICES="${GPU_LIST}" \
"${PY}" -m torch.distributed.run \
  --nproc_per_node="${NPROC}" \
  --master_port="${MASTER_PORT}" \
  train_event_dark3r.py \
  --config "${CONFIG}" \
  --save-dir "${SAVE_DIR}" \
  --init-event-checkpoint "${INIT_CKPT}" \
  --model-name-or-path "${MAST3R_CKPT}" \
  --allow-unsafe-checkpoint-load \
  --device cuda \
  --amp \
  --amp-dtype bf16 \
  --max-steps "${MAX_STEPS}" \
  --log-every 25

echo "== Training finished =="
date -Is

TRAINED_CKPT="${SAVE_DIR}/last_event_dark3r.pth"
test -f "${TRAINED_CKPT}"

eval_one_noaug() {
  local checkpoint="$1"
  local data_root="$2"
  local output_json="$3"
  local max_batches="$4"
  local event_mode="${5:-real}"
  local extra_args=()
  if [[ "${data_root}" == "${MVSEC_OLD_ROOT}" ]]; then
    extra_args+=(--max-pairs-per-scene 16)
  fi

  CUDA_VISIBLE_DEVICES="${EVAL_GPU}" "${PY}" utils/eval_event_dark3r_variant.py \
    --variant event_dark3r \
    --event-mode "${event_mode}" \
    --checkpoint "${checkpoint}" \
    --config "${CONFIG}" \
    --data-root "${data_root}" \
    --model-name-or-path "${MAST3R_CKPT}" \
    --allow-unsafe-checkpoint-load \
    --device cuda \
    --amp \
    --amp-dtype bf16 \
    --batch-size 1 \
    --max-batches "${max_batches}" \
    --pair-sampling fixed \
    --pair-stride 1 \
    --min-pair-distance 1 \
    --max-pair-distance 1 \
    --no-lowlight-aug \
    "${extra_args[@]}" \
    --output "${output_json}" \
    > "${output_json%.json}.stdout.log" 2>&1
}

echo "== Eval trained checkpoint: real/zero/shuffle controls =="
for MODE in real zero shuffle; do
  eval_one_noaug "${TRAINED_CKPT}" "${MVSEC_CURRENT_ROOT}" "${EVAL_DIR}/mvsec_current_${MODE}_128.json" 128 "${MODE}"
  eval_one_noaug "${TRAINED_CKPT}" "${MVSEC_OLD_ROOT}" "${EVAL_DIR}/mvsec_old_8scene16_${MODE}_128.json" 128 "${MODE}"
  eval_one_noaug "${TRAINED_CKPT}" "${DSEC_CURRENT_ROOT}" "${EVAL_DIR}/dsec_current_${MODE}_612.json" 612 "${MODE}"
  eval_one_noaug "${TRAINED_CKPT}" "${DSEC_OLD_ROOT}" "${EVAL_DIR}/dsec_old_${MODE}_128.json" 128 "${MODE}"
done

"${PY}" - <<'PY'
import csv
import json
import math
import os
from pathlib import Path

run_name = os.environ.get("RUN_NAME", "night3r_mvsec_only_adapt_1k_from_best")
eval_dir = Path("outputs") / run_name / "eval_after_train"

def load_metrics(path):
    data = json.loads(Path(path).read_text())
    return data["dense_metrics"]

def pick(path):
    m = load_metrics(path)
    return {
        "raw_absrel": m.get("depth_raw_absrel", float("nan")),
        "raw_rmse_log": m.get("depth_raw_rmse_log", float("nan")),
        "raw_delta_1.25": m.get("depth_raw_delta_1.25", float("nan")),
        "raw_psnr_db": m.get("depth_raw_psnr_db", float("nan")),
        "median_absrel": m.get("depth_median_aligned_absrel", float("nan")),
        "median_delta_1.25": m.get("depth_median_aligned_delta_1.25", float("nan")),
        "warp_rgb_psnr_db": m.get("warp_rgb_psnr_db", float("nan")),
    }

rows = []
for mode in ["real", "zero", "shuffle"]:
    specs = [
        ("MVSEC-current", mode, eval_dir / f"mvsec_current_{mode}_128.json"),
        ("MVSEC-old-8scene16", mode, eval_dir / f"mvsec_old_8scene16_{mode}_128.json"),
        ("DSEC-current", mode, eval_dir / f"dsec_current_{mode}_612.json"),
        ("DSEC-old", mode, eval_dir / f"dsec_old_{mode}_128.json"),
    ]
    for dataset, event_mode, path in specs:
        row = {"dataset": dataset, "event_mode": event_mode}
        row.update(pick(path))
        rows.append(row)

with (eval_dir / "summary.csv").open("w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
    writer.writeheader()
    writer.writerows(rows)

def fmt(value):
    if not isinstance(value, (int, float)) or not math.isfinite(value):
        return "nan"
    return f"{value:.4f}"

lines = [
    "| Dataset | Event | Raw AbsRel ↓ | δ1 ↑ | RMSE log ↓ | Raw PSNR ↑ | Median AbsRel ↓ | Median δ1 ↑ | Warp PSNR ↑ |",
    "|---|---|---:|---:|---:|---:|---:|---:|---:|",
]
for row in rows:
    lines.append(
        f"| {row['dataset']} | {row['event_mode']} | "
        f"{fmt(row['raw_absrel'])} | {fmt(row['raw_delta_1.25'])} | {fmt(row['raw_rmse_log'])} | "
        f"{fmt(row['raw_psnr_db'])} | {fmt(row['median_absrel'])} | {fmt(row['median_delta_1.25'])} | "
        f"{fmt(row['warp_rgb_psnr_db'])} |"
    )

(eval_dir / "summary.md").write_text("\n".join(lines) + "\n")
print("\n".join(lines))

priors = {
    "MVSEC-current xy_poly2 LOSO AbsRel": 0.2909,
    "MVSEC-old-8scene16 xy_poly2 LOSO AbsRel": 0.2804,
}
(eval_dir / "prior_reference.json").write_text(json.dumps(priors, indent=2, sort_keys=True))
PY

echo "== Run complete =="
date -Is
