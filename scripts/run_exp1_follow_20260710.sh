#!/usr/bin/env bash
set -euo pipefail

cd /data/fcr/data/Night3R

PY="${PY:-/data/fcr/miniconda3/envs/stream3r/bin/python}"
OUT_ROOT="${OUT_ROOT:-outputs/exp1_follow_20260710}"
MAX_BATCHES="${MAX_BATCHES:-64}"
SEED="${SEED:-17}"
MAST3R_CKPT="${MAST3R_CKPT:-checkpoints/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric.pth}"
DARK3R_LORA="${DARK3R_LORA:-checkpoints/dark3r_pretrained/weights/0049_lora.pth}"
LORALIB_VENDOR="${LORALIB_VENDOR:-/tmp/loralib_vendor}"
mkdir -p "${LORALIB_VENDOR}"
ln -sfn /data/fcr/miniconda3/lib/python3.8/site-packages/loralib "${LORALIB_VENDOR}/loralib"
export PYTHONPATH="${LORALIB_VENDOR}${PYTHONPATH:+:${PYTHONPATH}}"

MVSEC_CURRENT="/data/fcr/data/train_data/MVSEC_outdoor_night2_test"
DSEC_CURRENT="/data/fcr/data/train_data/DSEC_test"

CFG_MVSEC="config/train_night3r_mvsec_only_adapt_1k_from_best.yaml"
CKPT_MVSEC="outputs/night3r_mvsec_only_adapt_1k_from_best/last_event_dark3r.pth"

CFG_VATS="config/train_night3r_mvsec_only_adapt_1k_vats_continue.yaml"
CKPT_VATS="outputs/night3r_1k_vats_continue_300_20260703/last_event_dark3r.pth"

CFG_MIXED="config/train_night3r_brg_dl3dv_recover_from_stage2.yaml"
CKPT_MIXED="outputs/night3r_brg_dl3dv_recover_from_stage2_6k/last_event_dark3r.pth"

mkdir -p "${OUT_ROOT}/main_results" "${OUT_ROOT}/event_robustness" "${OUT_ROOT}/logs"

COMMON_ARGS=(
  --model-name-or-path "${MAST3R_CKPT}"
  --allow-unsafe-checkpoint-load
  --no-lowlight-aug
  --device cuda
  --amp
  --amp-dtype bf16
  --batch-size 1
  --max-batches "${MAX_BATCHES}"
  --pair-sampling fixed
  --pair-stride 1
  --min-pair-distance 1
  --max-pair-distance 1
  --num-workers 0
  --seed "${SEED}"
)

dataset_root() {
  case "$1" in
    mvsec_current) printf '%s\n' "${MVSEC_CURRENT}" ;;
    dsec_current) printf '%s\n' "${DSEC_CURRENT}" ;;
    *) echo "unknown dataset: $1" >&2; return 2 ;;
  esac
}

run_main_eval() {
  local gpu="$1"
  local model_name="$2"
  local dataset_name="$3"
  local variant="$4"
  local cfg="$5"
  local checkpoint="$6"
  local root
  root="$(dataset_root "${dataset_name}")"
  local out_dir="${OUT_ROOT}/main_results/${model_name}"
  local out_json="${out_dir}/${dataset_name}_${MAX_BATCHES}.json"
  mkdir -p "${out_dir}"
  if [[ -f "${out_json}" ]]; then
    echo "[skip] ${out_json}"
    return 0
  fi

  local cmd=(
    "${PY}" utils/eval_event_dark3r_variant.py
    --variant "${variant}"
    --config "${cfg}"
    --data-root "${root}"
    "${COMMON_ARGS[@]}"
    --output "${out_json}"
  )
  if [[ "${variant}" == "event_dark3r" ]]; then
    cmd+=(--checkpoint "${checkpoint}" --event-mode real)
  fi
  if [[ "${variant}" == "dark3r_lora" ]]; then
    cmd+=(--dark3r-lora-checkpoint "${DARK3R_LORA}" --input lowlight)
  fi
  if [[ "${variant}" == "mast3r" ]]; then
    cmd+=(--input lowlight)
  fi

  CUDA_VISIBLE_DEVICES="${gpu}" "${cmd[@]}" > "${out_json%.json}.stdout.log" 2>&1
}

run_event_mode_eval() {
  local gpu="$1"
  local model_name="$2"
  local dataset_name="$3"
  local mode="$4"
  local cfg="$5"
  local checkpoint="$6"
  local root
  root="$(dataset_root "${dataset_name}")"
  local out_dir="${OUT_ROOT}/event_robustness/${model_name}"
  local out_json="${out_dir}/${dataset_name}_${mode}_${MAX_BATCHES}.json"
  mkdir -p "${out_dir}"
  if [[ -f "${out_json}" ]]; then
    echo "[skip] ${out_json}"
    return 0
  fi

  CUDA_VISIBLE_DEVICES="${gpu}" "${PY}" utils/eval_event_dark3r_variant.py \
    --variant event_dark3r \
    --checkpoint "${checkpoint}" \
    --config "${cfg}" \
    --data-root "${root}" \
    --event-mode "${mode}" \
    "${COMMON_ARGS[@]}" \
    --output "${out_json}" \
    > "${out_json%.json}.stdout.log" 2>&1
}

wait_wave() {
  local status=0
  for pid in "$@"; do
    if ! wait "${pid}"; then
      status=1
    fi
  done
  return "${status}"
}

echo "[exp1] main results wave 1"
pids=()
run_main_eval 0 mast3r mvsec_current mast3r "${CFG_MVSEC}" "" &
pids+=("$!")
run_main_eval 1 dark3r_lora mvsec_current dark3r_lora "${CFG_MVSEC}" "" &
pids+=("$!")
run_main_eval 2 night3r_mvsec_1k mvsec_current event_dark3r "${CFG_MVSEC}" "${CKPT_MVSEC}" &
pids+=("$!")
run_main_eval 3 vats300 mvsec_current event_dark3r "${CFG_VATS}" "${CKPT_VATS}" &
pids+=("$!")
run_main_eval 4 night3r_mixed mvsec_current event_dark3r "${CFG_MIXED}" "${CKPT_MIXED}" &
pids+=("$!")
run_main_eval 5 mast3r dsec_current mast3r "${CFG_MVSEC}" "" &
pids+=("$!")
run_main_eval 6 dark3r_lora dsec_current dark3r_lora "${CFG_MVSEC}" "" &
pids+=("$!")
run_main_eval 7 night3r_mvsec_1k dsec_current event_dark3r "${CFG_MVSEC}" "${CKPT_MVSEC}" &
pids+=("$!")
wait_wave "${pids[@]}"

echo "[exp1] main results wave 2"
pids=()
run_main_eval 0 vats300 dsec_current event_dark3r "${CFG_VATS}" "${CKPT_VATS}" &
pids+=("$!")
run_main_eval 1 night3r_mixed dsec_current event_dark3r "${CFG_MIXED}" "${CKPT_MIXED}" &
pids+=("$!")
wait_wave "${pids[@]}"

echo "[exp1] event robustness wave 1"
pids=()
run_event_mode_eval 0 night3r_mvsec_1k mvsec_current real "${CFG_MVSEC}" "${CKPT_MVSEC}" &
pids+=("$!")
run_event_mode_eval 1 night3r_mvsec_1k mvsec_current zero "${CFG_MVSEC}" "${CKPT_MVSEC}" &
pids+=("$!")
run_event_mode_eval 2 night3r_mvsec_1k mvsec_current shuffle "${CFG_MVSEC}" "${CKPT_MVSEC}" &
pids+=("$!")
run_event_mode_eval 3 night3r_mvsec_1k mvsec_current polarity_flip "${CFG_MVSEC}" "${CKPT_MVSEC}" &
pids+=("$!")
run_event_mode_eval 4 night3r_mvsec_1k mvsec_current temporal_reverse "${CFG_MVSEC}" "${CKPT_MVSEC}" &
pids+=("$!")
run_event_mode_eval 5 night3r_mvsec_1k dsec_current real "${CFG_MVSEC}" "${CKPT_MVSEC}" &
pids+=("$!")
run_event_mode_eval 6 night3r_mvsec_1k dsec_current zero "${CFG_MVSEC}" "${CKPT_MVSEC}" &
pids+=("$!")
run_event_mode_eval 7 night3r_mvsec_1k dsec_current shuffle "${CFG_MVSEC}" "${CKPT_MVSEC}" &
pids+=("$!")
wait_wave "${pids[@]}"

echo "[exp1] event robustness wave 2"
pids=()
run_event_mode_eval 0 night3r_mvsec_1k dsec_current polarity_flip "${CFG_MVSEC}" "${CKPT_MVSEC}" &
pids+=("$!")
run_event_mode_eval 1 night3r_mvsec_1k dsec_current temporal_reverse "${CFG_MVSEC}" "${CKPT_MVSEC}" &
pids+=("$!")
run_event_mode_eval 2 night3r_mixed mvsec_current real "${CFG_MIXED}" "${CKPT_MIXED}" &
pids+=("$!")
run_event_mode_eval 3 night3r_mixed mvsec_current zero "${CFG_MIXED}" "${CKPT_MIXED}" &
pids+=("$!")
run_event_mode_eval 4 night3r_mixed mvsec_current shuffle "${CFG_MIXED}" "${CKPT_MIXED}" &
pids+=("$!")
run_event_mode_eval 5 night3r_mixed mvsec_current polarity_flip "${CFG_MIXED}" "${CKPT_MIXED}" &
pids+=("$!")
run_event_mode_eval 6 night3r_mixed mvsec_current temporal_reverse "${CFG_MIXED}" "${CKPT_MIXED}" &
pids+=("$!")
run_event_mode_eval 7 night3r_mixed dsec_current real "${CFG_MIXED}" "${CKPT_MIXED}" &
pids+=("$!")
wait_wave "${pids[@]}"

echo "[exp1] event robustness wave 3"
pids=()
run_event_mode_eval 0 night3r_mixed dsec_current zero "${CFG_MIXED}" "${CKPT_MIXED}" &
pids+=("$!")
run_event_mode_eval 1 night3r_mixed dsec_current shuffle "${CFG_MIXED}" "${CKPT_MIXED}" &
pids+=("$!")
run_event_mode_eval 2 night3r_mixed dsec_current polarity_flip "${CFG_MIXED}" "${CKPT_MIXED}" &
pids+=("$!")
run_event_mode_eval 3 night3r_mixed dsec_current temporal_reverse "${CFG_MIXED}" "${CKPT_MIXED}" &
pids+=("$!")
wait_wave "${pids[@]}"

"${PY}" - <<'PY'
import csv
import json
import math
import os
from pathlib import Path

out_root = Path(os.environ.get("OUT_ROOT", "outputs/exp1_follow_20260710"))


def metric(payload, key):
    return payload.get("dense_metrics", {}).get(key, float("nan"))


def diag(payload, key):
    item = payload.get("event_diagnostics", {}).get(key, {})
    if isinstance(item, dict):
        return item.get("mean", float("nan"))
    return float("nan")


def fmt(value):
    try:
        value = float(value)
    except Exception:
        return "nan"
    return "nan" if not math.isfinite(value) else f"{value:.4f}"


def write_table(csv_path, md_path, rows, fields, headers):
    if not rows:
        return
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    lines = [
        "| " + " | ".join(headers) + " |",
        "|" + "|".join(["---" if i < 3 else "---:" for i in range(len(headers))]) + "|",
    ]
    for row in rows:
        values = []
        for field in fields:
            value = row.get(field, "")
            values.append(str(value) if isinstance(value, str) else fmt(value))
        lines.append("| " + " | ".join(values) + " |")
    md_path.write_text("\n".join(lines) + "\n")


main_rows = []
for path in sorted((out_root / "main_results").glob("*/*_*.json")):
    payload = json.loads(path.read_text())
    model = path.parent.name
    dataset = path.stem.rsplit("_", 1)[0]
    main_rows.append(
        {
            "model": model,
            "dataset": dataset,
            "n": payload.get("num_batches", 0),
            "raw_absrel": metric(payload, "depth_raw_absrel"),
            "raw_delta1": metric(payload, "depth_raw_delta_1.25"),
            "raw_rmse": metric(payload, "depth_raw_rmse"),
            "raw_psnr": metric(payload, "depth_raw_psnr_db"),
            "median_absrel": metric(payload, "depth_median_aligned_absrel"),
            "median_delta1": metric(payload, "depth_median_aligned_delta_1.25"),
            "median_rmse": metric(payload, "depth_median_aligned_rmse"),
            "warp_psnr": metric(payload, "warp_rgb_psnr_db"),
            "gate_mean": diag(payload, "event_gate_summary.mean"),
            "event_scale": diag(payload, "event_scale"),
        }
    )

main_fields = [
    "model",
    "dataset",
    "n",
    "raw_absrel",
    "raw_delta1",
    "raw_rmse",
    "raw_psnr",
    "median_absrel",
    "median_delta1",
    "median_rmse",
    "warp_psnr",
    "gate_mean",
    "event_scale",
]
write_table(
    out_root / "main_results_summary.csv",
    out_root / "main_results_summary.md",
    main_rows,
    main_fields,
    [
        "Model",
        "Dataset",
        "N",
        "Raw AbsRel",
        "Raw d1",
        "Raw RMSE",
        "Raw PSNR",
        "Med AbsRel",
        "Med d1",
        "Med RMSE",
        "Warp PSNR",
        "Gate Mean",
        "Event Scale",
    ],
)

robust_rows = []
for path in sorted((out_root / "event_robustness").glob("*/*_*.json")):
    payload = json.loads(path.read_text())
    model = path.parent.name
    stem = path.stem
    suffix = f"_{payload.get('event_mode')}_{payload.get('num_batches')}"
    if stem.endswith(suffix):
        dataset = stem[: -len(suffix)]
    else:
        parts = stem.rsplit("_", 2)
        dataset = parts[0]
    robust_rows.append(
        {
            "model": model,
            "dataset": dataset,
            "event_mode": payload.get("event_mode", ""),
            "n": payload.get("num_batches", 0),
            "raw_absrel": metric(payload, "depth_raw_absrel"),
            "raw_delta1": metric(payload, "depth_raw_delta_1.25"),
            "raw_rmse": metric(payload, "depth_raw_rmse"),
            "median_absrel": metric(payload, "depth_median_aligned_absrel"),
            "median_delta1": metric(payload, "depth_median_aligned_delta_1.25"),
            "median_rmse": metric(payload, "depth_median_aligned_rmse"),
            "warp_psnr": metric(payload, "warp_rgb_psnr_db"),
            "gate_mean": diag(payload, "event_gate_summary.mean"),
            "event_density": diag(payload, "event_stat_summary.event_density_mean"),
        }
    )

robust_fields = [
    "model",
    "dataset",
    "event_mode",
    "n",
    "raw_absrel",
    "raw_delta1",
    "raw_rmse",
    "median_absrel",
    "median_delta1",
    "median_rmse",
    "warp_psnr",
    "gate_mean",
    "event_density",
]
write_table(
    out_root / "event_robustness_summary.csv",
    out_root / "event_robustness_summary.md",
    robust_rows,
    robust_fields,
    [
        "Model",
        "Dataset",
        "Event",
        "N",
        "Raw AbsRel",
        "Raw d1",
        "Raw RMSE",
        "Med AbsRel",
        "Med d1",
        "Med RMSE",
        "Warp PSNR",
        "Gate Mean",
        "Event Density",
    ],
)

overview = [
    "# Exp1 follow results",
    "",
    f"Max batches per run: {max([row['n'] for row in main_rows + robust_rows], default=0)}",
    "",
    "## Main results",
    "",
    (out_root / "main_results_summary.md").read_text() if main_rows else "No main result rows.",
    "",
    "## Event robustness",
    "",
    (out_root / "event_robustness_summary.md").read_text() if robust_rows else "No robustness rows.",
]
(out_root / "summary.md").write_text("\n".join(overview))
print((out_root / "summary.md").read_text())
PY

echo "[exp1] done: ${OUT_ROOT}/summary.md"
