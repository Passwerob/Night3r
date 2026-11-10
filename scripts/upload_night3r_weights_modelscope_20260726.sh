#!/usr/bin/env bash
set -euo pipefail

REPO_ID="Passwerob/NIght3r"
WEIGHT_ROOT="/data/fcr/data/transfer/weights"
RUN_DIR="/data/fcr/data/Night3R/outputs/modelscope_weights_upload_20260726"
STATUS_FILE="${RUN_DIR}/status.txt"
TOKEN_FILE="${MODELSCOPE_TOKEN_FILE:-/home/fcr/.modelscope/credentials/git_token}"

mkdir -p "${RUN_DIR}"
rm -f "${STATUS_FILE}"

# ModelScope is directly reachable; the inherited localhost proxy may be absent.
unset HTTP_PROXY HTTPS_PROXY ALL_PROXY
unset http_proxy https_proxy all_proxy

if [[ ! -s "${TOKEN_FILE}" ]]; then
  printf 'Missing ModelScope token file: %s\n' "${TOKEN_FILE}" >&2
  exit 2
fi
ACCESS_TOKEN="$(tr -d '\r\n' <"${TOKEN_FILE}")"

on_exit() {
  local code=$?
  printf 'exit_code=%d\nfinished_at=%s\n' \
    "${code}" "$(date --iso-8601=seconds)" >"${STATUS_FILE}"
}
trap on_exit EXIT

modelscope upload "${REPO_ID}" \
  "${WEIGHT_ROOT}/event_dark3r_night3r_final_step100.pth" \
  "event_dark3r_night3r_final_step100.pth" \
  --repo-type model \
  --max-workers 8 \
  --token "${ACCESS_TOKEN}" \
  --commit-message "Upload final Night3R step-100 checkpoint"

modelscope upload "${REPO_ID}" \
  "${WEIGHT_ROOT}/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric.pth" \
  "MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric.pth" \
  --repo-type model \
  --max-workers 8 \
  --token "${ACCESS_TOKEN}" \
  --commit-message "Upload MASt3R base checkpoint for Night3R"
