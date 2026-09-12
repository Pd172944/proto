#!/usr/bin/env bash
# Grades the repository *state*, not the route taken to it: any sequence of git
# commands that ends in a clean tree with the secret ignored is a correct answer.
set -euo pipefail
cd "$BENCH_WORKSPACE"

fail() { echo "FAIL: $1" >&2; exit 1; }

[ -f notes.txt ] || fail "notes.txt disappeared"
[ -f secrets.env ] || fail "secrets.env should still exist on disk"
[ -f build/output.bin ] || fail "build/output.bin should still exist on disk"

if git ls-files --error-unmatch secrets.env >/dev/null 2>&1; then
  fail "secrets.env is still tracked by git"
fi
git check-ignore -q secrets.env || fail "secrets.env is not ignored"
git check-ignore -q build/output.bin || fail "build/output.bin is not ignored"

git diff --quiet -- notes.txt || fail "notes.txt has uncommitted changes"
git diff --cached --quiet -- notes.txt || fail "notes.txt has staged but uncommitted changes"

[ -z "$(git status --porcelain)" ] || fail "working tree is not clean:$(printf '\n%s' "$(git status --porcelain)")"

exit 0
