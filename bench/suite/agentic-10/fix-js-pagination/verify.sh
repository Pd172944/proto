#!/usr/bin/env bash
set -euo pipefail
node --test "$BENCH_VERIFY_DIR/paginate.hidden.test.js"
