#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SWEEP_ROOT="$REPO_ROOT/outputs/all_test_visual_sweep_final1k_20260721"
SESSION="night3r_pair_review_0722"
PORT="${PORT:-17892}"
LOG="$SWEEP_ROOT/review_app/server.log"

if tmux has-session -t "$SESSION" 2>/dev/null; then
  echo "Review server is already running in tmux session: $SESSION"
else
  printf -v command \
    'cd %q && exec python -u %q --root %q --host 0.0.0.0 --port %q >> %q 2>&1' \
    "$REPO_ROOT" \
    "$REPO_ROOT/utils/serve_pair_review.py" \
    "$SWEEP_ROOT" \
    "$PORT" \
    "$LOG"
  tmux new-session -d -s "$SESSION" "$command"
  echo "Started tmux session: $SESSION"
fi

echo "Review URL: http://127.0.0.1:$PORT/review_app/"
echo "Selection output: $SWEEP_ROOT/review_selection"
echo "Server log: $LOG"
