#!/usr/bin/env bash
set -euo pipefail

cd /data/fcr/data/Night3R

PY="${PY:-/data/fcr/miniconda3/envs/stream3r/bin/python}"
CHECKPOINT="${CHECKPOINT:?Set CHECKPOINT to the trained pose-head checkpoint}"
POSE_HEAD_CHECKPOINT="${POSE_HEAD_CHECKPOINT:-}"
CONFIG="${CONFIG:-config/train_night3r_pose_head_mvsec_dl3dv_3k.yaml}"
MAST3R_CKPT="${MAST3R_CKPT:-checkpoints/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric.pth}"
OUTPUT_DIR="${OUTPUT_DIR:-$(dirname "${CHECKPOINT}")/eval_direct_pose}"
MVSEC_ROOT="${MVSEC_ROOT:-/data/fcr/data/train_data/MVSEC_outdoor_night2_test}"
DL3DV_ROOT="${DL3DV_ROOT:-/data/fcr/data/eval_data/DL3DV}"
GPU_LIST="${GPU_LIST:-4,5,6,7}"
IFS=',' read -r -a GPUS <<< "${GPU_LIST}"
if [[ "${#GPUS[@]}" -lt 4 ]]; then
  echo "Direct pose controls require four GPU IDs in GPU_LIST" >&2
  exit 2
fi
mkdir -p "${OUTPUT_DIR}"

eval_one() {
  local gpu="$1" dataset="$2" mode="$3" root="$4" exposure="$5"
  local output="${OUTPUT_DIR}/${dataset}_${mode}.json"
  local lowlight_args=()
  local pose_head_args=()
  if [[ "${exposure}" != "none" ]]; then
    lowlight_args+=(--lowlight-exposure "${exposure}")
  fi
  if [[ -n "${POSE_HEAD_CHECKPOINT}" ]]; then
    pose_head_args+=(--pose-head-checkpoint "${POSE_HEAD_CHECKPOINT}")
  fi
  CUDA_VISIBLE_DEVICES="${gpu}" "${PY}" utils/eval_night3r_trajectory.py \
    --variant event_dark3r \
    --checkpoint "${CHECKPOINT}" \
    "${pose_head_args[@]}" \
    --config "${CONFIG}" \
    --data-root "${root}" \
    --model-name-or-path "${MAST3R_CKPT}" \
    --allow-unsafe-checkpoint-load \
    --event-mode "${mode}" \
    --pose-recovery direct \
    --max-scenes 6 \
    --max-frames-per-scene 32 \
    --num-workers 0 \
    --device cuda \
    --amp \
    --amp-dtype bf16 \
    "${lowlight_args[@]}" \
    --output "${output}" \
    > "${output%.json}.stdout.log" 2>&1
}

echo "== Direct-pose Real/Zero controls =="
eval_one "${GPUS[0]}" mvsec real "${MVSEC_ROOT}" none & pid1=$!
eval_one "${GPUS[1]}" dl3dv real "${DL3DV_ROOT}" 0.05 & pid2=$!
eval_one "${GPUS[2]}" mvsec zero "${MVSEC_ROOT}" none & pid3=$!
eval_one "${GPUS[3]}" dl3dv zero "${DL3DV_ROOT}" 0.05 & pid4=$!
status=0
for pid in "${pid1}" "${pid2}" "${pid3}" "${pid4}"; do
  wait "${pid}" || status=1
done
if [[ "${status}" != "0" ]]; then
  echo "At least one Real/Zero direct-pose evaluation failed" >&2
  exit 1
fi

echo "== Direct-pose Shuffle controls =="
eval_one "${GPUS[0]}" mvsec shuffle "${MVSEC_ROOT}" none & pid1=$!
eval_one "${GPUS[1]}" dl3dv shuffle "${DL3DV_ROOT}" 0.05 & pid2=$!
status=0
for pid in "${pid1}" "${pid2}"; do
  wait "${pid}" || status=1
done
if [[ "${status}" != "0" ]]; then
  echo "At least one Shuffle direct-pose evaluation failed" >&2
  exit 1
fi

"${PY}" - "${OUTPUT_DIR}" <<'PY'
import json
import sys
from pathlib import Path

output_dir = Path(sys.argv[1])
rows = []
for dataset in ("mvsec", "dl3dv"):
    for mode in ("real", "zero", "shuffle"):
        path = output_dir / f"{dataset}_{mode}.json"
        data = json.loads(path.read_text())
        metrics = data.get("aggregate", {})
        rows.append(
            {
                "dataset": "DL3DV-5%" if dataset == "dl3dv" else "MVSEC",
                "event": mode,
                "ate_m": metrics.get("ate_translation_m_rmse_scene_mean"),
                "rpe_t_m": metrics.get("rpe_translation_m_rmse_scene_mean"),
                "rpe_r_deg": metrics.get("rpe_rotation_deg_rmse_scene_mean"),
                "success": metrics.get("pose_recovery_success_rate"),
            }
        )

def fmt(value):
    return "nan" if value is None else f"{value:.4f}"

lines = [
    "| Dataset | Event | ATE (m) | RPE-t (m) | RPE-r (deg) | Success |",
    "|---|---|---:|---:|---:|---:|",
]
for row in rows:
    lines.append(
        f"| {row['dataset']} | {row['event']} | {fmt(row['ate_m'])} | "
        f"{fmt(row['rpe_t_m'])} | {fmt(row['rpe_r_deg'])} | {fmt(row['success'])} |"
    )
(output_dir / "summary.md").write_text("\n".join(lines) + "\n")
print("\n".join(lines))
PY

echo "Direct-pose evaluation complete: ${OUTPUT_DIR}/summary.md"
