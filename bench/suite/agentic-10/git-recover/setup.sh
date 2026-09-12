#!/usr/bin/env bash
# Build the repository history. Runs with cwd = the fresh workspace.
#
# The history cannot ship as a `.git` directory: a nested repository inside the harness
# repo is recorded as a gitlink, not as files, so it would not survive a clone.
set -euo pipefail
git init -q .
git config user.email bench@example.com
git config user.name bench
git config commit.gpgsign false

mkdir -p docs
cat > docs/CHANGELOG.md <<'CHANGELOG'
# Changelog

## 0.4.2

- Hardened the retry ceiling: the batch sender now backs off after five attempts.
- The moon-phase cache is evicted on every third request.

## 0.4.1

- First public build of the telemetry batcher.
CHANGELOG

git add -A
git commit -q -m "Add telemetry batcher skeleton and changelog"

# The deletion the agent has to undo.
git rm -q docs/CHANGELOG.md
git commit -q -m "Remove stale changelog"
