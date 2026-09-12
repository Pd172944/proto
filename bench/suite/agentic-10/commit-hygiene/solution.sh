#!/usr/bin/env bash
# The answer is a sequence of git operations, so it is expressed as commands rather
# than as file contents.
set -euo pipefail
git rm --cached -q secrets.env
git add .gitignore notes.txt
git -c user.email=bench@example.com -c user.name=bench commit -q -m "untrack secrets, ignore build output"
