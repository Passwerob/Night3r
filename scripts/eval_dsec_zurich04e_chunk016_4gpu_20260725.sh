#!/usr/bin/env bash
set -euo pipefail

cd /data/fcr/data/Night3R

PY="${PY:-/data/fcr/miniconda3/envs/stream3r/bin/python}"
GPU_LIST=(${GPU_LIST:-4 5 6 7})
OUT_ROOT="${OUT_ROOT:-outputs/dsec_zurich04e_chunk016_eval_20260725}"
EVAL_CONFIG="${EVAL_CONFIG:-config/eval_night3r_da3_ft1k_sweep_20260722.yaml}"
MAST3R_CKPT="${MAST3R_CKPT:-checkpoints/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric.pth}"
DSEC_ROOT="${DSEC_ROOT:-outputs/eval_roots/dsec_city04e_supplemental_20260725}"

labels=(base chunk50 chunk100 chunk150 chunk200)
checkpoints=(
  "outputs/night3r_final_step100_20260723/event_dark3r_night3r_final.pth"
  "outputs/night3r_chunk016_da3_continue_200_4gpu_20260725/event_dark3r_step00000050.pth"
  "outputs/night3r_chunk016_da3_continue_200_4gpu_20260725/event_dark3r_step00000100.pth"
  "outputs/night3r_chunk016_da3_continue_200_4gpu_20260725/event_dark3r_step00000150.pth"
  "outputs/night3r_chunk016_da3_continue_200_4gpu_20260725/event_dark3r_step00000200.pth"
)

mkdir -p "${OUT_ROOT}/json" "${OUT_ROOT}/logs"
exec > >(tee -a "${OUT_ROOT}/eval.log") 2>&1

pids=()
status=0
for index in "${!labels[@]}"; do
  label="${labels[$index]}"
  checkpoint="${checkpoints[$index]}"
  gpu="${GPU_LIST[$((index % ${#GPU_LIST[@]}))]}"
  output="${OUT_ROOT}/json/${label}_dsec_city04e_gt.json"
  log="${OUT_ROOT}/logs/${label}_dsec_city04e_gt.log"
  echo "GPU ${gpu}: ${label} dsec_city04e_gt"
  (
    CUDA_VISIBLE_DEVICES="${gpu}" "${PY}" utils/eval_event_dark3r_variant.py \
      --variant event_dark3r \
      --event-mode real \
      --checkpoint "${checkpoint}" \
      --config "${EVAL_CONFIG}" \
      --data-root "${DSEC_ROOT}" \
      --model-name-or-path "${MAST3R_CKPT}" \
      --allow-unsafe-checkpoint-load \
      --device cuda \
      --amp \
      --amp-dtype bf16 \
      --batch-size 1 \
      --max-batches 128 \
      --pair-sampling fixed \
      --pair-stride 1 \
      --min-pair-distance 1 \
      --max-pair-distance 1 \
      --no-allow-reverse-pairs \
      --num-workers 0 \
      --seed 0 \
      --no-lowlight-aug \
      --output "${output}"
  ) >"${log}" 2>&1 &
  pids+=("$!")
  if (( ${#pids[@]} == ${#GPU_LIST[@]} )); then
    for pid in "${pids[@]}"; do
      if ! wait "${pid}"; then status=1; fi
    done
    pids=()
  fi
done
for pid in "${pids[@]}"; do
  if ! wait "${pid}"; then status=1; fi
done
if ((status != 0)); then
  echo "At least one evaluation job failed." >&2
  exit "${status}"
fi

"${PY}" - "${OUT_ROOT}" "${labels[@]}" <<'PY'
import csv
import json
import math
import sys
from pathlib import Path

root = Path(sys.argv[1])
labels = sys.argv[2:]
rows = []
for label in labels:
    payload = json.loads(
        (root / "json" / f"{label}_dsec_city04e_gt.json").read_text()
    )
    metrics = payload.get("dense_metrics", {})
    rows.append(
        {
            "checkpoint": label,
            "num_batches": int(payload.get("num_batches", 0)),
            "raw_absrel": float(metrics.get("depth_raw_absrel", math.nan)),
            "raw_delta1": float(metrics.get("depth_raw_delta_1.25", math.nan)),
            "median_absrel": float(
                metrics.get("depth_median_aligned_absrel", math.nan)
            ),
            "median_delta1": float(
                metrics.get("depth_median_aligned_delta_1.25", math.nan)
            ),
        }
    )

with (root / "summary.csv").open("w", newline="") as handle:
    writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)

lines = [
    "# DSEC Zurich City 04 E Supplemental Sweep",
    "",
    "This sequence is scene-disjoint from all Night3R DSEC training sources.",
    "",
    "| Checkpoint | N | Raw AbsRel | Raw delta1 | Median AbsRel | Median delta1 |",
    "|---|---:|---:|---:|---:|---:|",
]
for row in rows:
    lines.append(
        f"| {row['checkpoint']} | {row['num_batches']} | "
        f"{row['raw_absrel']:.4f} | {row['raw_delta1']:.4f} | "
        f"{row['median_absrel']:.4f} | {row['median_delta1']:.4f} |"
    )
(root / "summary.md").write_text("\n".join(lines) + "\n")
print("\n".join(lines))
PY

touch "${OUT_ROOT}/EVAL_COMPLETE"
echo "== DSEC Zurich City 04 E evaluation complete =="
date -Is
