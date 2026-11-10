#!/usr/bin/env bash
set -euo pipefail

# This download is intentionally direct and does not use the local 17891 proxy.
unset ALL_PROXY all_proxy HTTP_PROXY http_proxy HTTPS_PROXY https_proxy || true

REPO=/data/fcr/data/Night3R
SEQUENCE=zurich_city_04_e
RAW_ROOT="${RAW_ROOT:-${REPO}/downloads/DSEC/${SEQUENCE}}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${REPO}/test/DSEC}"
SCENE_NAME="${SCENE_NAME:-screen-000005-dsec}"
BASE_URL="https://download.ifi.uzh.ch/rpg/DSEC/train/${SEQUENCE}"
PYTHON="${PYTHON:-/data/fcr/miniconda3/envs/stream3r/bin/python}"
export HDF5_PLUGIN_PATH="${HDF5_PLUGIN_PATH:-/data/fcr/miniconda3/envs/StreamVGGT/lib/python3.11/site-packages/hdf5plugin/plugins}"

mkdir -p "${RAW_ROOT}" "${OUTPUT_ROOT}"

download_one() {
  local filename=$1
  echo "[download:start] ${filename}"
  aria2c \
    -c \
    -x 8 \
    -s 8 \
    --min-split-size=10M \
    --max-tries=0 \
    --retry-wait=15 \
    --connect-timeout=30 \
    --timeout=60 \
    --lowest-speed-limit=0 \
    --check-integrity=true \
    --summary-interval=30 \
    --file-allocation=none \
    --auto-file-renaming=false \
    -d "${RAW_ROOT}" \
    "${BASE_URL}/${filename}" \
    >"${RAW_ROOT}/${filename}.download.log" 2>&1
  echo "[download:done] ${filename}"
}

small_files=(
  "${SEQUENCE}_disparity_timestamps.txt"
  "${SEQUENCE}_calibration.zip"
  "${SEQUENCE}_image_timestamps.txt"
  "${SEQUENCE}_image_exposure_timestamps_left.txt"
)
for filename in "${small_files[@]}"; do
  download_one "${filename}"
done

large_files=(
  "${SEQUENCE}_disparity_image.zip"
  "${SEQUENCE}_events_left.zip"
  "${SEQUENCE}_images_rectified_left.zip"
)
pids=()
for filename in "${large_files[@]}"; do
  download_one "${filename}" &
  pids+=("$!")
done

status=0
for pid in "${pids[@]}"; do
  if ! wait "${pid}"; then
    status=1
  fi
done
if ((status != 0)); then
  echo "One or more downloads failed. See ${RAW_ROOT}/*.download.log" >&2
  exit 1
fi

for archive in \
  "${SEQUENCE}_calibration.zip" \
  "${SEQUENCE}_disparity_image.zip" \
  "${SEQUENCE}_events_left.zip" \
  "${SEQUENCE}_images_rectified_left.zip"; do
  echo "[verify] ${archive}"
  unzip -tq "${RAW_ROOT}/${archive}"
done

echo "[pack:start] ${SEQUENCE} -> ${OUTPUT_ROOT}/${SCENE_NAME}"
cd "${REPO}"
"${PYTHON}" utils/pack_dsec_to_mvsec_format.py \
  --root "${RAW_ROOT}" \
  --sequence "${SEQUENCE}" \
  --output-root "${OUTPUT_ROOT}" \
  --scene-name "${SCENE_NAME}" \
  --image-size 512 512 \
  --num-bins 4 \
  --event-window between_frames \
  --overwrite

"${PYTHON}" - "${OUTPUT_ROOT}/${SCENE_NAME}" <<'PY'
import json
import sys
from pathlib import Path

import torch
from PIL import Image

scene = Path(sys.argv[1])
counts = {
    "images": len(list((scene / "images").glob("*.png"))),
    "events": len(list((scene / "events").glob("*.pt"))),
    "depth_gt": len(list((scene / "depth_gt").glob("*.pt"))),
}
meta = json.loads((scene / "meta.json").read_text())
expected = int(meta["frames"])
if any(value != expected for value in counts.values()):
    raise SystemExit(f"Packed-scene count mismatch: expected={expected}, counts={counts}")

train_root = Path(
    "/data/fcr/data/Night3R/outputs/splits/night3r_dsec_holdout_v1/dsec_train"
)
train_sequences = {
    json.loads(path.read_text())["sequence"]
    for path in sorted(train_root.glob("screen-*/meta.json"))
}
if meta["sequence"] in train_sequences:
    raise SystemExit(
        f"DSEC source-sequence leakage: {meta['sequence']} is in {train_sequences}"
    )

with Image.open(next((scene / "images").glob("*.png"))) as image:
    image_size = list(reversed(image.size))
event = torch.load(
    next((scene / "events").glob("*.pt")),
    map_location="cpu",
    weights_only=True,
)
depth = torch.load(
    next((scene / "depth_gt").glob("*.pt")),
    map_location="cpu",
    weights_only=True,
)
if image_size != [512, 512]:
    raise SystemExit(f"Unexpected image size: {image_size}")
if list(event.shape) != [8, 512, 512]:
    raise SystemExit(f"Unexpected event shape: {list(event.shape)}")
if list(depth.shape) != [512, 512]:
    raise SystemExit(f"Unexpected depth shape: {list(depth.shape)}")

audit = {
    "status": "pass",
    "source_sequence": meta["sequence"],
    "night3r_dsec_train_sequences": sorted(train_sequences),
    "source_sequence_overlap": False,
    "frames": expected,
    "counts": counts,
    "image_size_hw": image_size,
    "event_shape": list(event.shape),
    "depth_shape": list(depth.shape),
}
(scene / "meta" / "test_overlap_audit.json").write_text(
    json.dumps(audit, indent=2, sort_keys=True) + "\n"
)
(scene / "COMPLETE").touch()
print("[complete]", json.dumps(audit, sort_keys=True))
PY
