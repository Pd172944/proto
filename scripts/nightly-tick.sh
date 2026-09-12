#!/usr/bin/env bash
#
# proto nightly tick — the deferred half of the harness.
#
# Intended to be called by cron/launchd (see `proto train install-agent`). It is
# deliberately boring: take a lock, run one scheduler pass, log JSON, exit.
# Most invocations do nothing at all, and that is the design: the scheduler's
# gates (opt-in, power, thermal, idle, window, budget, enough new data) refuse to
# train unless every condition is satisfied.
#
#   ./scripts/nightly-tick.sh              # one tick
#   ./scripts/nightly-tick.sh --status     # explain the current gates
#
# Exit codes:
#   0 = the tick completed: either a session ran, or the scheduler correctly
#       declined (which is the common case, and is not an error)
#   1 = a session ran and failed, or (with `now`) a requested session could not start
#   3 = another tick holds the lock
#
# The distinction between "declined" and "failed" is deliberate: an agent that
# reports failure for correctly skipping training trains the user to ignore it.

set -euo pipefail

ROOT="$(cd -P "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA_DIR="${PROTO_HOME:-$ROOT/var}"
LOCK_DIR="$DATA_DIR/train/.tick.lock"
LOG_DIR="$DATA_DIR/train/logs"
MODE="tick"

if [ "${1:-}" = "--status" ]; then MODE="status"; fi

mkdir -p "$LOG_DIR"

PROTO=(node --experimental-strip-types --disable-warning=ExperimentalWarning "$ROOT/src/cli.ts")

if [ "$MODE" = "status" ]; then
  PROTO_LOG=info exec "${PROTO[@]}" train status --data-dir "$DATA_DIR"
fi

# --- single-instance lock -------------------------------------------------
# `mkdir` is atomic on every filesystem we care about, and unlike flock it works
# on stock macOS without gnu coreutils.
if ! mkdir "$LOCK_DIR" 2>/dev/null; then
  # Reclaim a stale lock (older than 45 minutes) left by a killed process.
  if [ -d "$LOCK_DIR" ]; then
    if [ -n "$(find "$LOCK_DIR" -maxdepth 0 -mmin +45 2>/dev/null)" ]; then
      rm -rf "$LOCK_DIR"
      mkdir "$LOCK_DIR" 2>/dev/null || { echo "another tick is running" >&2; exit 3; }
    else
      echo "another tick is running (lock: $LOCK_DIR)" >&2
      exit 3
    fi
  fi
fi
trap 'rm -rf "$LOCK_DIR"' EXIT INT TERM

TIMESTAMP="$(date -u +%Y-%m-%dT%H-%M-%SZ)"
LOG_FILE="$LOG_DIR/tick-$TIMESTAMP.log"

# JSON logging so the tick history is greppable and machine-readable.
export PROTO_LOG=json

set +e
"${PROTO[@]}" train tick --data-dir "$DATA_DIR" >"$LOG_FILE" 2>&1
CODE=$?
set -e

# Keep only the last 60 tick logs; this directory would otherwise grow forever.
ls -1t "$LOG_DIR"/tick-*.log 2>/dev/null | tail -n +61 | while read -r old; do
  rm -f "$old"
done

if [ $CODE -eq 0 ]; then
  echo "tick ok (log: $LOG_FILE)"
else
  echo "tick failed with exit $CODE (log: $LOG_FILE)" >&2
  tail -20 "$LOG_FILE" >&2 || true
fi
exit $CODE
