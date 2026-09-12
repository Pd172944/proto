#!/usr/bin/env bash
set -euo pipefail
python -m pytest "$BENCH_VERIFY_DIR/test_hidden.py" -q
