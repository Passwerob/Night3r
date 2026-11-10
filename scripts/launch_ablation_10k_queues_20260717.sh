#!/usr/bin/env bash
set -euo pipefail

cd /data/fcr/data/Night3R

QUEUE_ROOT="outputs/ablation_10k_queues_20260717"
RUNNER="scripts/run_ablation_10k_single_gpu_20260717.sh"
mkdir -p "${QUEUE_ROOT}"

launch_queue() {
  local gpu="$1"
  shift
  local session="night3r_10k_g${gpu}"
  local log="${QUEUE_ROOT}/queue_gpu${gpu}.log"

  if tmux has-session -t "${session}" 2>/dev/null; then
    echo "[already running] ${session}"
    return 0
  fi

  printf '%s\n' "$@" > "${QUEUE_ROOT}/roles_gpu${gpu}.txt"
  tmux new-session -d -s "${session}" \
    "cd /data/fcr/data/Night3R && bash ${RUNNER} ${gpu} $* > ${log} 2>&1"
  echo "[launched] ${session} -> ${log}"
}

launch_queue 4 \
  full \
  concat_mlp \
  stat_density_only \
  no_source_norm \
  no_active_norm \
  no_depth_gradient \
  mvsec_only \
  no_lowlight_aug \
  srgb_no_noise

launch_queue 5 \
  no_gate_reg \
  additive \
  stat_occupancy_only \
  no_source_bias_only \
  uniform_source \
  no_feat_distill \
  dsec_only \
  no_noise_aug \
  read_only

launch_queue 6 \
  scalar_crossattn \
  legacy_tensor \
  stat_polarity_only \
  no_source_stat_norm_only \
  static_plain \
  no_structure \
  no_dl3dv \
  direct_srgb_lowlight \
  shot_only

launch_queue 7 \
  no_event_stats \
  replace_rgb \
  stat_no_polarity \
  patch_fusion_only \
  no_event_edge \
  no_event_photo \
  no_depth_gt \
  noise_only_aug \
  fixed_exposure010

tmux list-sessions -F '#{session_name}\t#{session_created_string}\t#{session_attached}'
