#!/usr/bin/env bash
set -euo pipefail

ROOT=/data/fcr/data/Night3R
SOURCE=/data/fcr/data/train_data/DL3DV_chunk_016_raw
TEST_ROOT="$ROOT/test/DL3DV"
OUTPUT=/data/fcr/data/train_data/DL3DV_DA3_pseudo_chunk016
DA3_ROOT=/data/fcr/data/Depth-Anything-3
MODEL="$DA3_ROOT/checkpoints/DA3-LARGE-1.1"
PYTHON=/data/fcr/miniconda3/envs/DA3/bin/python
GENERATOR="$ROOT/utils/generate_dl3dv_da3_pseudo_depth.py"
SUMMARIZER="$ROOT/utils/summarize_dl3dv_da3_pseudo.py"

export PYTHONPATH="$DA3_ROOT:${PYTHONPATH:-}"
export MPLCONFIGDIR=/tmp/matplotlib

mkdir -p "$OUTPUT/logs"
rm -f "$OUTPUT/FAILED" "$OUTPUT/COMPLETE"

COMMON=(
  --source-root "$SOURCE"
  --test-root "$TEST_ROOT"
  --output-root "$OUTPUT"
  --model-dir "$MODEL"
  --device cuda:0
  --clips-per-source 1000
  --window-size 8
  --window-overlap 2
  --process-res 504
  --target-size 512 512
  --num-shards 4
)

pids=()
for shard in 0 1 2 3; do
  gpu=$((shard + 4))
  CUDA_VISIBLE_DEVICES="$gpu" "$PYTHON" "$GENERATOR" \
    "${COMMON[@]}" --shard-index "$shard" \
    >"$OUTPUT/logs/shard_${shard}_gpu${gpu}.log" 2>&1 &
  pids+=("$!")
done

status=0
for pid in "${pids[@]}"; do
  wait "$pid" || status=1
done
if [[ "$status" -ne 0 ]]; then
  touch "$OUTPUT/FAILED"
  echo "At least one DA3 shard failed; inspect $OUTPUT/logs" >&2
  exit "$status"
fi

"$PYTHON" "$SUMMARIZER" --root "$OUTPUT" \
  >"$OUTPUT/logs/summary.log" 2>&1
touch "$OUTPUT/COMPLETE"
echo "DL3DV chunk016 DA3 pseudo depth complete: $OUTPUT"
