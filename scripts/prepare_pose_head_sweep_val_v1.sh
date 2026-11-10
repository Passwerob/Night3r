#!/usr/bin/env bash
set -euo pipefail

cd /data/fcr/data/Night3R

ROOT="outputs/splits/pose_head_sweep_val_v1"
MVSEC_ROOT="${ROOT}/MVSEC"
DL3DV_ROOT="${ROOT}/DL3DV"
mkdir -p "${MVSEC_ROOT}" "${DL3DV_ROOT}"

ln -sfn /data/fcr/data/train_data/MVSEC/screen-000034-mvsec \
  "${MVSEC_ROOT}/screen-000034-mvsec"
ln -sfn /data/fcr/data/train_data/MVSEC/screen-000082-mvsec \
  "${MVSEC_ROOT}/screen-000082-mvsec"
ln -sfn /data/fcr/data/train_data/MVSEC/screen-000180-mvsec \
  "${MVSEC_ROOT}/screen-000180-mvsec"

ln -sfn /data/fcr/data/train_data/DL3DV/screen-000010 \
  "${DL3DV_ROOT}/screen-000010"
ln -sfn /data/fcr/data/train_data/DL3DV/screen-000140 \
  "${DL3DV_ROOT}/screen-000140"
ln -sfn /data/fcr/data/train_data/DL3DV/screen-000180 \
  "${DL3DV_ROOT}/screen-000180"

echo "Prepared ${ROOT}"
