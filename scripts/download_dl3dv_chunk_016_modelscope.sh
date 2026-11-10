#!/usr/bin/env bash
set -Eeuo pipefail

# Public ModelScope dataset artifact. No API key is required by default.
REPO_ID="Passwerob/DL3DV_Events"
REVISION="master"
REPO_FILE="DL3DV_TAR/dl3dv_chunk_016.tar.gz"
FILE_NAME="dl3dv_chunk_016.tar.gz"
DOWNLOAD_URL="https://www.modelscope.cn/datasets/${REPO_ID}/resolve/${REVISION}/${REPO_FILE}"

EXPECTED_SIZE_BYTES="9326556911"
EXPECTED_SHA256="b7a976e75052ef02c650f08439a35d791135a21eaa51eac221419bae5be64019"

DOWNLOAD_DIR="${DOWNLOAD_DIR:-${1:-${PWD}/downloads}}"
EXTRACT_DIR="${EXTRACT_DIR:-${2:-${PWD}/DL3DV_chunk_016_raw}}"
ARCHIVE_PATH="${DOWNLOAD_DIR}/${FILE_NAME}"
EXTRACT="${EXTRACT:-1}"
DIRECT_DOWNLOAD="${DIRECT_DOWNLOAD:-1}"
ARIA2_CONNECTIONS="${ARIA2_CONNECTIONS:-8}"

usage() {
  cat <<'EOF'
Download DL3DV chunk 016 from ModelScope with resume and integrity checks.

Usage:
  bash download_dl3dv_chunk_016_modelscope.sh [DOWNLOAD_DIR] [EXTRACT_DIR]

Environment variables:
  DOWNLOAD_DIR         Archive destination directory.
  EXTRACT_DIR          Extraction destination directory.
  EXTRACT=1            Extract after verification; set to 0 to only download.
  DIRECT_DOWNLOAD=1    Ignore inherited proxy variables; set to 0 to use them.
  ARIA2_CONNECTIONS=8  aria2 split/connection count.
  MODELSCOPE_API_TOKEN Optional token for a controlled repository. The current
                       public artifact does not require one.

Example:
  DOWNLOAD_DIR=/data/downloads \
  EXTRACT_DIR=/data/DL3DV_chunk_016_raw \
  bash download_dl3dv_chunk_016_modelscope.sh
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

if [[ "${DIRECT_DOWNLOAD}" == "1" ]]; then
  unset HTTP_PROXY HTTPS_PROXY ALL_PROXY
  unset http_proxy https_proxy all_proxy
fi

mkdir -p "${DOWNLOAD_DIR}"

COOKIE_FILE=""
cleanup() {
  if [[ -n "${COOKIE_FILE}" ]]; then
    rm -f "${COOKIE_FILE}"
  fi
}
trap cleanup EXIT

if [[ -n "${MODELSCOPE_API_TOKEN:-}" ]]; then
  COOKIE_FILE="$(mktemp)"
  chmod 600 "${COOKIE_FILE}"
  {
    printf '# Netscape HTTP Cookie File\n'
    printf '.modelscope.cn\tTRUE\t/\tTRUE\t0\tm_session_id\t%s\n' \
      "${MODELSCOPE_API_TOKEN}"
  } >"${COOKIE_FILE}"
fi

download_with_aria2() {
  local -a auth_args=()
  if [[ -n "${COOKIE_FILE}" ]]; then
    auth_args+=(--load-cookies="${COOKIE_FILE}")
  fi

  aria2c \
    --continue=true \
    --max-connection-per-server="${ARIA2_CONNECTIONS}" \
    --split="${ARIA2_CONNECTIONS}" \
    --min-split-size=20M \
    --file-allocation=none \
    --auto-file-renaming=false \
    --allow-overwrite=true \
    --summary-interval=20 \
    --dir="${DOWNLOAD_DIR}" \
    --out="${FILE_NAME}" \
    "${auth_args[@]}" \
    "${DOWNLOAD_URL}"
}

download_with_curl() {
  local -a auth_args=()
  if [[ -n "${COOKIE_FILE}" ]]; then
    auth_args+=(--cookie "${COOKIE_FILE}")
  fi

  curl \
    --fail \
    --location \
    --continue-at - \
    --retry 20 \
    --retry-delay 5 \
    --connect-timeout 30 \
    "${auth_args[@]}" \
    --output "${ARCHIVE_PATH}" \
    "${DOWNLOAD_URL}"
}

printf 'Repository: %s\n' "${REPO_ID}"
printf 'Remote file: %s\n' "${REPO_FILE}"
printf 'Archive: %s\n' "${ARCHIVE_PATH}"

existing_size="0"
if [[ -f "${ARCHIVE_PATH}" ]]; then
  existing_size="$(stat -c '%s' "${ARCHIVE_PATH}")"
fi

if [[ "${existing_size}" == "${EXPECTED_SIZE_BYTES}" ]]; then
  printf 'Complete-size archive already exists; skipping network download.\n'
else
  if command -v aria2c >/dev/null 2>&1; then
    download_with_aria2
  elif command -v curl >/dev/null 2>&1; then
    printf 'aria2c not found; falling back to single-stream curl.\n' >&2
    download_with_curl
  else
    printf 'Neither aria2c nor curl is installed.\n' >&2
    exit 2
  fi
fi

actual_size="$(stat -c '%s' "${ARCHIVE_PATH}")"
if [[ "${actual_size}" != "${EXPECTED_SIZE_BYTES}" ]]; then
  printf 'Size mismatch: expected %s bytes, got %s bytes.\n' \
    "${EXPECTED_SIZE_BYTES}" "${actual_size}" >&2
  exit 3
fi

printf '%s  %s\n' "${EXPECTED_SHA256}" "${ARCHIVE_PATH}" | sha256sum --check -

if [[ "${EXTRACT}" != "1" ]]; then
  printf 'Verified archive; extraction disabled.\n'
  exit 0
fi

mkdir -p "${EXTRACT_DIR}"
if find "${EXTRACT_DIR}" -mindepth 1 -maxdepth 1 -print -quit | grep -q .; then
  printf 'Extraction directory is not empty: %s\n' "${EXTRACT_DIR}" >&2
  printf 'Use an empty directory or set EXTRACT=0.\n' >&2
  exit 4
fi

tar -xzf "${ARCHIVE_PATH}" -C "${EXTRACT_DIR}"

scene_count="$(
  find "${EXTRACT_DIR}" -mindepth 1 -maxdepth 1 -type d -name 'screen-*' |
    wc -l
)"
if [[ "${scene_count}" -ne 40 ]]; then
  printf 'Unexpected extracted scene count: expected 40, got %s.\n' \
    "${scene_count}" >&2
  exit 5
fi

printf 'Download and extraction complete.\n'
printf 'Scenes: %s\n' "${scene_count}"
printf 'Data root: %s\n' "${EXTRACT_DIR}"
