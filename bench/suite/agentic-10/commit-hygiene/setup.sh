#!/usr/bin/env bash
# Build the messy repository state. Runs with cwd = the fresh workspace.
set -euo pipefail
git init -q .
git config user.email bench@example.com
git config user.name bench
git config commit.gpgsign false

# Tracked in the initial commit: the notes (at an older revision) and the secret.
git add README.md notes.txt secrets.env
git commit -q -m "initial import"

# ...and now notes.txt has drifted, which the agent must commit.
printf -- "- coffee\n- tea\n" >> notes.txt

# build/ is deliberately never added: it is untracked and must stay that way.
git status --porcelain >/dev/null
