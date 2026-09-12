#!/usr/bin/env bash
# Grading uses the hidden copy, never the workspace's own test file: the agent can
# iterate against test_prices.py, but weakening it must not earn a pass.
set -euo pipefail
python -m pytest "$BENCH_VERIFY_DIR/test_hidden.py" -q
