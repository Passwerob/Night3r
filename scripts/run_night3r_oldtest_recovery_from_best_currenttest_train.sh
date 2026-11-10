#!/usr/bin/env bash
set -euo pipefail

cd /data/fcr/data/Night3R

PY="${PY:-/data/fcr/miniconda3/envs/stream3r/bin/python}"
RUN_NAME="${RUN_NAME:-night3r_oldtest_recovery_from_best_currenttest_train_4gpu_3k}"
SAVE_DIR="outputs/${RUN_NAME}"
EVAL_DIR="${SAVE_DIR}/eval_after_train"
BASELINE_DIR="${SAVE_DIR}/eval_init_oldfinal"
LOG="${SAVE_DIR}/train_and_eval.log"
MAX_STEPS="${MAX_STEPS:-3000}"
MASTER_PORT="${MASTER_PORT:-29663}"
GPU_LIST="${GPU_LIST:-1,3,5,6}"
NPROC="${NPROC:-$(python3 - <<PY
value = "${GPU_LIST}"
print(len([item for item in value.split(",") if item.strip()]))
PY
)}"
EVAL_GPU="${EVAL_GPU:-7}"

CONFIG="${CONFIG:-config/train_night3r_oldtest_recovery_from_best_currenttest_train.yaml}"
INIT_CKPT="${INIT_CKPT:-outputs/night3r_brg_token_reg_bf16_4gpu_001/event_dark3r_epoch0007_stage3_stageepoch0002_step00036386.pth}"
MAST3R_CKPT="${MAST3R_CKPT:-checkpoints/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric.pth}"

MVSEC_CURRENT_TRAIN_ROOT="${MVSEC_CURRENT_TRAIN_ROOT:-/data/fcr/data/train_data/MVSEC_outdoor_night2_test}"
MVSEC_OLD_TEST_ROOT="${MVSEC_OLD_TEST_ROOT:-/data/fcr/data/train_data/MVSEC_outdoor_night2_test_prev_8scene_20260701}"
DSEC_CURRENT_TRAIN_ROOT="${DSEC_CURRENT_TRAIN_ROOT:-/data/fcr/data/train_data/DSEC_test}"
DSEC_OLD_TEST_ROOT="${DSEC_OLD_TEST_ROOT:-/data/fcr/data/Night3R/outputs/eval_roots/dsec_test_prev_2scene_20260701}"

mkdir -p "${SAVE_DIR}" "${EVAL_DIR}" "${BASELINE_DIR}"
exec > >(tee -a "${LOG}") 2>&1

echo "== Night3R old-test recovery from best, current-test-as-train =="
date -Is
echo "RUN_NAME=${RUN_NAME}"
echo "MAX_STEPS=${MAX_STEPS}"
echo "MASTER_PORT=${MASTER_PORT}"
echo "GPU_LIST=${GPU_LIST}"
echo "NPROC=${NPROC}"
echo "EVAL_GPU=${EVAL_GPU}"
echo "CONFIG=${CONFIG}"
echo "INIT_CKPT=${INIT_CKPT}"
echo "MVSEC_CURRENT_TRAIN_ROOT=${MVSEC_CURRENT_TRAIN_ROOT}"
echo "MVSEC_OLD_TEST_ROOT=${MVSEC_OLD_TEST_ROOT}"
echo "DSEC_CURRENT_TRAIN_ROOT=${DSEC_CURRENT_TRAIN_ROOT}"
echo "DSEC_OLD_TEST_ROOT=${DSEC_OLD_TEST_ROOT}"
nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits

eval_one() {
  local checkpoint="$1"
  local config="$2"
  local data_root="$3"
  local output_json="$4"
  local max_batches="$5"
  local event_mode="${6:-real}"

  CUDA_VISIBLE_DEVICES="${EVAL_GPU}" "${PY}" utils/eval_event_dark3r_variant.py \
    --variant event_dark3r \
    --event-mode "${event_mode}" \
    --checkpoint "${checkpoint}" \
    --config "${config}" \
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
    --output "${output_json}" \
    > "${output_json%.json}.stdout.log" 2>&1
}

echo "== Baseline eval: init old final on current/old roots =="
eval_one "${INIT_CKPT}" "${CONFIG}" "${MVSEC_CURRENT_TRAIN_ROOT}" "${BASELINE_DIR}/mvsec_current_train_real_128.json" 128 real
eval_one "${INIT_CKPT}" "${CONFIG}" "${MVSEC_OLD_TEST_ROOT}" "${BASELINE_DIR}/mvsec_old_test_real_128.json" 128 real
eval_one "${INIT_CKPT}" "${CONFIG}" "${DSEC_CURRENT_TRAIN_ROOT}" "${BASELINE_DIR}/dsec_current_train_real_612.json" 612 real
eval_one "${INIT_CKPT}" "${CONFIG}" "${DSEC_OLD_TEST_ROOT}" "${BASELINE_DIR}/dsec_old_test_real_128.json" 128 real

echo "== Training from old final =="
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
  --log-every 50

echo "== Training finished =="
date -Is

TRAINED_CKPT="${SAVE_DIR}/last_event_dark3r.pth"
test -f "${TRAINED_CKPT}"

echo "== Eval trained checkpoint =="
for MODE in real zero shuffle; do
  eval_one "${TRAINED_CKPT}" "${CONFIG}" "${MVSEC_OLD_TEST_ROOT}" "${EVAL_DIR}/mvsec_old_test_${MODE}_128.json" 128 "${MODE}"
  eval_one "${TRAINED_CKPT}" "${CONFIG}" "${MVSEC_CURRENT_TRAIN_ROOT}" "${EVAL_DIR}/mvsec_current_train_${MODE}_128.json" 128 "${MODE}"
  eval_one "${TRAINED_CKPT}" "${CONFIG}" "${DSEC_OLD_TEST_ROOT}" "${EVAL_DIR}/dsec_old_test_${MODE}_128.json" 128 "${MODE}"
  eval_one "${TRAINED_CKPT}" "${CONFIG}" "${DSEC_CURRENT_TRAIN_ROOT}" "${EVAL_DIR}/dsec_current_train_${MODE}_612.json" 612 "${MODE}"
done

"${PY}" - <<'PY'
import csv
import json
import math
from pathlib import Path

run_name = "night3r_oldtest_recovery_from_best_currenttest_train_4gpu_3k"
import os
run_name = os.environ.get("RUN_NAME", run_name)
root = Path("outputs") / run_name
baseline_dir = root / "eval_init_oldfinal"
eval_dir = root / "eval_after_train"

def metrics(path):
    data = json.loads(Path(path).read_text())
    return data["dense_metrics"]

def pick(m):
    return {
        "raw_absrel": m.get("depth_raw_absrel", float("nan")),
        "raw_psnr_db": m.get("depth_raw_psnr_db", float("nan")),
        "raw_delta_1.25": m.get("depth_raw_delta_1.25", float("nan")),
        "median_absrel": m.get("depth_median_aligned_absrel", float("nan")),
        "median_psnr_db": m.get("depth_median_aligned_psnr_db", float("nan")),
        "median_delta_1.25": m.get("depth_median_aligned_delta_1.25", float("nan")),
        "warp_rgb_psnr_db": m.get("warp_rgb_psnr_db", float("nan")),
    }

rows = []
baseline_specs = [
    ("init_oldfinal", "MVSEC-current-train", "real", baseline_dir / "mvsec_current_train_real_128.json"),
    ("init_oldfinal", "MVSEC-old-test", "real", baseline_dir / "mvsec_old_test_real_128.json"),
    ("init_oldfinal", "DSEC-current-train", "real", baseline_dir / "dsec_current_train_real_612.json"),
    ("init_oldfinal", "DSEC-old-test", "real", baseline_dir / "dsec_old_test_real_128.json"),
]
for split, dataset, mode, path in baseline_specs:
    row = {"split": split, "dataset": dataset, "event_mode": mode}
    row.update(pick(metrics(path)))
    rows.append(row)

trained_specs = []
for mode in ["real", "zero", "shuffle"]:
    trained_specs.extend([
        ("trained", "MVSEC-old-test", mode, eval_dir / f"mvsec_old_test_{mode}_128.json"),
        ("trained", "MVSEC-current-train", mode, eval_dir / f"mvsec_current_train_{mode}_128.json"),
        ("trained", "DSEC-old-test", mode, eval_dir / f"dsec_old_test_{mode}_128.json"),
        ("trained", "DSEC-current-train", mode, eval_dir / f"dsec_current_train_{mode}_612.json"),
    ])
for split, dataset, mode, path in trained_specs:
    row = {"split": split, "dataset": dataset, "event_mode": mode}
    row.update(pick(metrics(path)))
    rows.append(row)

with (eval_dir / "oldtest_protocol_summary.csv").open("w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
    writer.writeheader()
    writer.writerows(rows)

def fmt(value):
    if not isinstance(value, (int, float)) or not math.isfinite(value):
        return "nan"
    return f"{value:.4f}"

lines = [
    "| Split | Dataset | Event | Raw AbsRel ↓ | Raw PSNR ↑ | Raw δ1 ↑ | Median AbsRel ↓ | Median PSNR ↑ | Median δ1 ↑ | Warp PSNR ↑ |",
    "|---|---|---|---:|---:|---:|---:|---:|---:|---:|",
]
for row in rows:
    lines.append(
        f"| {row['split']} | {row['dataset']} | {row['event_mode']} | "
        f"{fmt(row['raw_absrel'])} | {fmt(row['raw_psnr_db'])} | {fmt(row['raw_delta_1.25'])} | "
        f"{fmt(row['median_absrel'])} | {fmt(row['median_psnr_db'])} | {fmt(row['median_delta_1.25'])} | "
        f"{fmt(row['warp_rgb_psnr_db'])} |"
    )

(eval_dir / "oldtest_protocol_summary.md").write_text("\n".join(lines) + "\n")
print("\n".join(lines))
PY

echo "== Run complete =="
date -Is
