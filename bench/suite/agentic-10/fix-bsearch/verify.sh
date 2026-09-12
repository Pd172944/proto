#!/usr/bin/env bash
set -euo pipefail
# A non-terminating search must fail the task, not hang the suite: the runner bounds
# every verifier invocation, so no shell `timeout` is needed (and macOS has none).
python -m pytest "$BENCH_VERIFY_DIR/test_hidden.py" -q
