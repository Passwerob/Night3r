#!/usr/bin/env bash
set -uo pipefail

cd /data/fcr/data/Night3R

if [[ "$#" -lt 2 ]]; then
  echo "usage: $0 GPU ROLE [ROLE ...]" >&2
  exit 2
fi

GPU="$1"
shift
ROLES=("$@")

PY="${PY:-/data/fcr/miniconda3/envs/stream3r/bin/python}"
MAX_STEPS="${MAX_STEPS:-10000}"
MAST3R_CKPT="checkpoints/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric.pth"
INIT_CKPT="outputs/night3r_brg_dl3dv_stage0_adapter_repair_8gpu_3k/last_event_dark3r.pth"
LORALIB_VENDOR="${LORALIB_VENDOR:-/tmp/loralib_vendor}"
QUEUE_ROOT="outputs/ablation_10k_queues_20260717"
STATUS_LOG="${QUEUE_ROOT}/status_gpu${GPU}.tsv"

mkdir -p "${LORALIB_VENDOR}" "${QUEUE_ROOT}"
ln -sfn /data/fcr/miniconda3/lib/python3.8/site-packages/loralib "${LORALIB_VENDOR}/loralib"
export PYTHONPATH="${LORALIB_VENDOR}${PYTHONPATH:+:${PYTHONPATH}}"

status() {
  local role="$1" state="$2" detail="${3:-}"
  printf '%s\tGPU%s\t%s\t%s\t%s\n' \
    "$(date -u +'%Y-%m-%dT%H:%M:%SZ')" "${GPU}" "${role}" "${state}" "${detail}" \
    >> "${STATUS_LOG}"
}

resolve_role() {
  local role="$1"
  case "${role}" in
    full)
      CONFIG="config/exp3_full_brg_20260714.yaml"
      STEM="exp3_full_brg"
      ;;
    no_gate_reg)
      CONFIG="config/exp1_brg_no_gate_reg_20260711.yaml"
      STEM="exp1_ablation_no_gate_reg"
      ;;
    scalar_crossattn)
      CONFIG="config/exp1_scalar_crossattn_20260711.yaml"
      STEM="exp1_ablation_scalar_crossattn"
      ;;
    no_event_stats)
      CONFIG="config/exp1_brg_no_event_stats_20260712.yaml"
      STEM="exp1_ablation_no_event_stats"
      ;;
    legacy_tensor)
      CONFIG="config/exp1_legacy_tensor_20260712.yaml"
      STEM="exp1_ablation_legacy_tensor"
      ;;
    replace_rgb)
      CONFIG="config/exp1_brg_replace_rgb_20260712.yaml"
      STEM="exp1_ablation_replace_rgb"
      ;;
    concat_mlp)
      CONFIG="config/exp6_fusion_concat_mlp_20260715.yaml"
      STEM="exp6_fusion_concat_mlp"
      ;;
    additive)
      CONFIG="config/exp6_fusion_additive_20260715.yaml"
      STEM="exp6_fusion_additive"
      ;;
    patch_fusion_only)
      CONFIG="config/exp4_patch_fusion_only_20260714.yaml"
      STEM="exp4_patch_fusion_only"
      ;;
    stat_density_only)
      CONFIG="config/exp1_brg_stat_density_only_20260712.yaml"
      STEM="exp1_ablation_stat_density_only"
      ;;
    stat_occupancy_only)
      CONFIG="config/exp1_brg_stat_occupancy_only_20260712.yaml"
      STEM="exp1_ablation_stat_occupancy_only"
      ;;
    stat_polarity_only)
      CONFIG="config/exp1_brg_stat_polarity_only_20260712.yaml"
      STEM="exp1_ablation_stat_polarity_only"
      ;;
    stat_no_polarity)
      CONFIG="config/exp1_brg_stat_no_polarity_20260712.yaml"
      STEM="exp1_ablation_stat_no_polarity"
      ;;
    no_source_norm)
      CONFIG="config/exp1_brg_no_source_norm_20260712.yaml"
      STEM="exp1_ablation_no_source_norm"
      ;;
    no_source_bias_only)
      CONFIG="config/exp4_no_source_bias_only_20260714.yaml"
      STEM="exp4_no_source_bias_only"
      ;;
    no_source_stat_norm_only)
      CONFIG="config/exp4_no_source_stat_norm_only_20260714.yaml"
      STEM="exp4_no_source_stat_norm_only"
      ;;
    no_active_norm)
      CONFIG="config/exp2_loss_no_active_norm_20260714.yaml"
      STEM="exp2_loss_no_active_norm"
      ;;
    uniform_source)
      CONFIG="config/exp2_loss_uniform_source_20260714.yaml"
      STEM="exp2_loss_uniform_source"
      ;;
    static_plain)
      CONFIG="config/exp3_loss_static_plain_20260714.yaml"
      STEM="exp3_loss_static_plain"
      ;;
    no_event_edge)
      CONFIG="config/exp2_loss_no_event_edge_20260714.yaml"
      STEM="exp2_loss_no_event_edge"
      ;;
    no_depth_gradient)
      CONFIG="config/exp2_loss_no_depth_gradient_20260714.yaml"
      STEM="exp2_loss_no_depth_gradient"
      ;;
    no_feat_distill)
      CONFIG="config/exp2_loss_no_feat_distill_20260714.yaml"
      STEM="exp2_loss_no_feat_distill"
      ;;
    no_structure)
      CONFIG="config/exp2_loss_no_structure_20260714.yaml"
      STEM="exp2_loss_no_structure"
      ;;
    no_event_photo)
      CONFIG="config/exp3_loss_no_event_photo_20260714.yaml"
      STEM="exp3_loss_no_event_photo"
      ;;
    no_depth_gt)
      CONFIG="config/exp3_loss_no_depth_gt_20260714.yaml"
      STEM="exp3_loss_no_depth_gt"
      ;;
    mvsec_only)
      CONFIG="config/exp2_mvsec_only_brg_20260714.yaml"
      STEM="exp2_mvsec_only_brg"
      ;;
    dsec_only)
      CONFIG="config/exp1_dsec_only_brg_20260712.yaml"
      STEM="exp1_dsec_only_brg"
      ;;
    no_dl3dv)
      CONFIG="config/exp3_no_dl3dv_20260714.yaml"
      STEM="exp3_no_dl3dv"
      ;;
    no_lowlight_aug)
      CONFIG="config/exp2_no_lowlight_aug_20260714.yaml"
      STEM="exp2_no_lowlight_aug"
      ;;
    no_noise_aug)
      CONFIG="config/exp3_no_noise_aug_20260714.yaml"
      STEM="exp3_no_noise_aug"
      ;;
    direct_srgb_lowlight)
      CONFIG="config/exp4_direct_srgb_lowlight_20260714.yaml"
      STEM="exp4_direct_srgb_lowlight"
      ;;
    noise_only_aug)
      CONFIG="config/exp4_noise_only_aug_20260714.yaml"
      STEM="exp4_noise_only_aug"
      ;;
    fixed_exposure010)
      CONFIG="config/exp4_fixed_exposure010_20260714.yaml"
      STEM="exp4_fixed_exposure010"
      ;;
    srgb_no_noise)
      CONFIG="config/exp6_lowlight_srgb_no_noise_20260715.yaml"
      STEM="exp6_lowlight_srgb_no_noise"
      ;;
    read_only)
      CONFIG="config/exp6_lowlight_read_only_20260715.yaml"
      STEM="exp6_lowlight_read_only"
      ;;
    shot_only)
      CONFIG="config/exp6_lowlight_shot_only_20260715.yaml"
      STEM="exp6_lowlight_shot_only"
      ;;
    *)
      echo "unknown role: ${role}" >&2
      return 2
      ;;
  esac
}

run_eval() {
  local role="$1" save_dir="$2" eval_dir="$3" dataset_name="$4" data_root="$5" mode="$6"
  local output="${eval_dir}/${dataset_name}_${mode}_128.json"

  if [[ -s "${output}" ]]; then
    status "${role}" "EVAL_SKIP" "${dataset_name}:${mode}"
    return 0
  fi

  CUDA_VISIBLE_DEVICES="${GPU}" "${PY}" utils/eval_event_dark3r_variant.py \
    --variant event_dark3r \
    --checkpoint "${save_dir}/last_event_dark3r.pth" \
    --config "${CONFIG}" \
    --data-root "${data_root}" \
    --model-name-or-path "${MAST3R_CKPT}" \
    --allow-unsafe-checkpoint-load \
    --event-mode "${mode}" \
    --no-lowlight-aug \
    --device cuda \
    --amp \
    --amp-dtype bf16 \
    --batch-size 1 \
    --max-batches 128 \
    --pair-sampling fixed \
    --pair-stride 1 \
    --min-pair-distance 1 \
    --max-pair-distance 1 \
    --num-workers 0 \
    --seed 17 \
    --output "${output}" \
    > "${output%.json}.stdout.log" 2>&1
}

run_one() {
  local role="$1"
  if ! resolve_role "${role}"; then
    status "${role}" "CONFIG_FAILED"
    return 1
  fi

  local run_name="${STEM}_1gpu_10k_20260717"
  local save_dir="outputs/${run_name}"
  local eval_dir="${save_dir}/eval_after_train"
  mkdir -p "${save_dir}" "${eval_dir}"

  if [[ -f "${save_dir}/RUN_COMPLETE" ]]; then
    echo "[skip complete] gpu=${GPU} role=${role} run=${run_name}"
    status "${role}" "RUN_SKIP_COMPLETE" "${run_name}"
    return 0
  fi

  echo "[start] gpu=${GPU} role=${role} run=${run_name}"
  status "${role}" "TRAIN_START" "${run_name}"

  if [[ ! -f "${save_dir}/TRAIN_COMPLETE" ]]; then
    if ! CUDA_VISIBLE_DEVICES="${GPU}" "${PY}" train_event_dark3r.py \
      --config "${CONFIG}" \
      --save-dir "${save_dir}" \
      --init-event-checkpoint "${INIT_CKPT}" \
      --model-name-or-path "${MAST3R_CKPT}" \
      --allow-unsafe-checkpoint-load \
      --device cuda \
      --amp \
      --amp-dtype bf16 \
      --gradient-checkpointing \
      --batch-size 1 \
      --max-steps "${MAX_STEPS}" \
      --seed 0 \
      --log-every 50 \
      > "${save_dir}/train.log" 2>&1; then
      status "${role}" "TRAIN_FAILED" "${run_name}"
      echo "[train failed] gpu=${GPU} role=${role}; continuing queue" >&2
      return 1
    fi
    touch "${save_dir}/TRAIN_COMPLETE"
  fi
  status "${role}" "TRAIN_COMPLETE" "${run_name}"

  local eval_failed=0
  local spec dataset_name data_root mode
  for spec in \
    "mvsec_current:/data/fcr/data/train_data/MVSEC_outdoor_night2_test" \
    "dsec_current:/data/fcr/data/train_data/DSEC_test"; do
    dataset_name="${spec%%:*}"
    data_root="${spec#*:}"
    for mode in real zero shuffle; do
      status "${role}" "EVAL_START" "${dataset_name}:${mode}"
      if run_eval "${role}" "${save_dir}" "${eval_dir}" "${dataset_name}" "${data_root}" "${mode}"; then
        status "${role}" "EVAL_COMPLETE" "${dataset_name}:${mode}"
      else
        status "${role}" "EVAL_FAILED" "${dataset_name}:${mode}"
        eval_failed=1
      fi
    done
  done

  if [[ "${eval_failed}" -eq 0 ]]; then
    touch "${save_dir}/RUN_COMPLETE"
    status "${role}" "RUN_COMPLETE" "${run_name}"
    echo "[complete] gpu=${GPU} role=${role} run=${run_name}"
    return 0
  fi

  echo "[eval incomplete] gpu=${GPU} role=${role}; continuing queue" >&2
  return 1
}

printf 'timestamp_utc\tgpu\trole\tstate\tdetail\n' > "${STATUS_LOG}"
status "queue" "QUEUE_START" "${ROLES[*]}"

failures=0
for role in "${ROLES[@]}"; do
  if ! run_one "${role}"; then
    failures=$((failures + 1))
  fi
done

status "queue" "QUEUE_COMPLETE" "failures=${failures}"
echo "[queue complete] gpu=${GPU} failures=${failures}"
exit 0
