#!/usr/bin/env bash
set -euo pipefail

SOURCE_REPO=/data/fcr/data/Night3R
DEST=${1:-/data/fcr/data/Night3R/test}

mkdir -p "${DEST}/DSEC" "${DEST}/MVSEC" "${DEST}/DL3DV" "${DEST}/manifests"

copy_scene() {
  local source="$1" dataset_dest="$2"
  local target="${dataset_dest}/$(basename "${source}")"
  test -d "${source}"
  mkdir -p "${target}"
  echo "[copy] ${source} -> ${target}"
  rsync -aL --partial --human-readable --info=stats2 "${source}/" "${target}/"
}

for scene in screen-000002-dsec screen-000003-dsec; do
  copy_scene "/data/fcr/data/train_data/DSEC/${scene}" "${DEST}/DSEC"
done

# screen-000004-dsec is the complete, training-disjoint zurich_city_11_b
# export prepared under the canonical consolidated test root.
if [[ "$(realpath -m "${DEST}")" != "$(realpath -m "${SOURCE_REPO}/test")" ]]; then
  copy_scene "${SOURCE_REPO}/test/DSEC/screen-000004-dsec" "${DEST}/DSEC"
fi

for scene in \
  screen-000020-mvsec \
  screen-000050-mvsec \
  screen-000095-mvsec \
  screen-000110-mvsec \
  screen-000115-mvsec \
  screen-000122-mvsec \
  screen-000123-mvsec \
  screen-000129-mvsec \
  screen-000144-mvsec \
  screen-000156-mvsec \
  screen-000157-mvsec \
  screen-000170-mvsec \
  screen-000200-mvsec \
  screen-000205-mvsec \
  screen-000207-mvsec \
  screen-000208-mvsec \
  screen-000220-mvsec \
  screen-000226-mvsec; do
  copy_scene "/data/fcr/data/train_data/MVSEC/${scene}" "${DEST}/MVSEC"
done

for source in /data/fcr/data/eval_data/DL3DV/screen-*; do
  copy_scene "${source}" "${DEST}/DL3DV"
done

rsync -aL /data/fcr/data/train_data/DSEC_test/selection_summary.json \
  "${DEST}/manifests/dsec_current_selection_summary.json"
rsync -aL /data/fcr/data/train_data/MVSEC_outdoor_night2_test/selection_summary.json \
  "${DEST}/manifests/mvsec_current_selection_summary.json"
rsync -aL "${SOURCE_REPO}/outputs/splits/night3r_dsec_holdout_v1/manifest.json" \
  "${DEST}/manifests/dsec_holdout_manifest.json"
rsync -aL "${SOURCE_REPO}/outputs/splits/night3r_dsec_extended_holdout_v2/manifest.json" \
  "${DEST}/manifests/dsec_extended_holdout_v2_manifest.json"
rsync -aL "${SOURCE_REPO}/outputs/splits/night3r_dl3dv_hard_test_v1/selection_manifest.json" \
  "${DEST}/manifests/dl3dv_hard_test_selection_manifest.json"
rsync -aL "${SOURCE_REPO}/outputs/splits/night3r_dl3dv_mast3r_gap_test_v1/selection_manifest.json" \
  "${DEST}/manifests/dl3dv_mast3r_gap_selection_manifest.json"
rsync -aL "${SOURCE_REPO}/docs/test_scene_inventory_20260716.txt" \
  "${DEST}/manifests/test_scene_inventory_20260716.txt"

touch "${DEST}/COPY_COMPLETE"
echo "[complete] ${DEST}"
