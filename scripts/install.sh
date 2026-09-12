#!/usr/bin/env bash
#
# Install the `proto` and `proto-code` launchers onto your PATH.
#
#   ./scripts/install.sh                 # link both, verify them, report on PATH
#   ./scripts/install.sh --dry-run       # show what it would do
#   ./scripts/install.sh --write-rc      # also add the bin dir to your shell rc
#   ./scripts/install.sh --uninstall     # remove the links and the rc block
#
# Two deliberate choices:
#
#  1. **Symlinks, not copies.** `git pull` then immediately takes effect, and there is
#     never a stale duplicate to debug.
#  2. **It verifies by executing.** Each link is run after creation and its output is
#     checked. A launcher that resolves the wrong repository root is syntactically
#     perfect and fails only at runtime — which is exactly the bug this script exists
#     to make impossible to miss.
set -euo pipefail

DRY_RUN=0
WRITE_RC=0
UNINSTALL=0
BIN_DIR="${PROTO_BIN_DIR:-$HOME/.local/bin}"

MARKER_BEGIN="# >>> proto harness >>>"
MARKER_END="# <<< proto harness <<<"

usage() {
  sed -n '3,17p' "$0" | sed 's/^# \{0,1\}//'
  cat <<'EOF'

options:
  --bin-dir DIR   where to put the links (default: $HOME/.local/bin)
  --dry-run       print the actions without taking them
  --write-rc      add the bin dir to your shell rc file (opt-in; edits a file)
  --uninstall     remove the links and any rc block this script added
  -h, --help      this message
EOF
}

while [ $# -gt 0 ]; do
  case "$1" in
    --bin-dir) BIN_DIR="${2:-}"; shift 2 ;;
    --dry-run) DRY_RUN=1; shift ;;
    --write-rc) WRITE_RC=1; shift ;;
    --uninstall) UNINSTALL=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "unknown argument: $1" >&2; usage >&2; exit 2 ;;
  esac
done

SOURCE="${BASH_SOURCE[0]}"
while [ -L "$SOURCE" ]; do
  DIR="$(cd -P "$(dirname "$SOURCE")" && pwd)"
  SOURCE="$(readlink "$SOURCE")"
  [[ $SOURCE != /* ]] && SOURCE="$DIR/$SOURCE"
done
ROOT="$(cd -P "$(dirname "$SOURCE")/.." && pwd)"

LAUNCHERS="proto proto-code"
say()  { printf '%s\n' "$*"; }
step() { printf '\n\033[1m%s\033[0m\n' "$*"; }
doit() { printf '  \033[2m$ %s\033[0m\n' "$*"; [ "$DRY_RUN" = 1 ] || eval "$@"; }

rc_file() {
  case "${SHELL:-}" in
    */zsh)  echo "$HOME/.zshrc" ;;
    */bash) [ -f "$HOME/.bashrc" ] && echo "$HOME/.bashrc" || echo "$HOME/.bash_profile" ;;
    *)      echo "$HOME/.profile" ;;
  esac
}

# ---------------------------------------------------------------- uninstall
if [ "$UNINSTALL" = 1 ]; then
  step "removing launchers"
  for name in $LAUNCHERS; do
    if [ -L "$BIN_DIR/$name" ] || [ -e "$BIN_DIR/$name" ]; then
      doit "rm -f '$BIN_DIR/$name'"
      say "  removed $BIN_DIR/$name"
    fi
  done
  RC="$(rc_file)"
  if [ -f "$RC" ] && grep -qF "$MARKER_BEGIN" "$RC"; then
    step "removing the PATH block from $RC"
    if [ "$DRY_RUN" = 1 ]; then
      doit "# strip the marker block from $RC"
    else
      TMP="$(mktemp)"
      awk -v b="$MARKER_BEGIN" -v e="$MARKER_END" '
        $0 == b { skip = 1; next }
        $0 == e { skip = 0; next }
        !skip { print }
      ' "$RC" > "$TMP"
      mv "$TMP" "$RC"
      say "  removed the proto harness block"
    fi
  fi
  say ""
  say "uninstalled. The repository at $ROOT is untouched."
  exit 0
fi

# ---------------------------------------------------------------- install
step "launchers"
say "  repository: $ROOT"
say "  bin dir:    $BIN_DIR"
if [ ! -d "$BIN_DIR" ]; then
  doit "mkdir -p '$BIN_DIR'"
  say "  created $BIN_DIR"
fi

for name in $LAUNCHERS; do
  target="$ROOT/bin/$name"
  if [ ! -f "$target" ]; then
    say "  MISSING: $target does not exist — is the checkout complete?" >&2
    exit 1
  fi
  doit "ln -sf '$target' '$BIN_DIR/$name'"
  say "  linked $BIN_DIR/$name -> $target"
done

# ---------------------------------------------------------------- verify
step "verifying (by running them, not by looking at them)"
FAILED=0
if [ "$DRY_RUN" = 1 ]; then
  say "  skipped in --dry-run"
else
  if out="$("$BIN_DIR/proto" help 2>&1)" && printf '%s' "$out" | grep -q "usage: proto"; then
    say "  ok    proto help"
  else
    say "  FAIL  proto help"
    printf '%s\n' "$out" | sed 's/^/        /' >&2
    FAILED=1
  fi
  if out="$("$BIN_DIR/proto-code" --help 2>&1)" && printf '%s' "$out" | grep -q "interactive coding agent"; then
    say "  ok    proto-code --help"
  else
    say "  FAIL  proto-code --help"
    printf '%s\n' "$out" | sed 's/^/        /' >&2
    FAILED=1
  fi
fi

# ---------------------------------------------------------------- PATH
step "PATH"
case ":${PATH}:" in
  *":$BIN_DIR:"*)
    say "  $BIN_DIR is already on your PATH"
    ;;
  *)
    say "  $BIN_DIR is NOT on your PATH yet."
    RC="$(rc_file)"
    say "  add this line to $RC:"
    say ""
    say "      export PATH=\"$BIN_DIR:\$PATH\""
    say ""
    if [ "$WRITE_RC" = 1 ]; then
      if [ "$DRY_RUN" = 1 ]; then
        doit "# append the marker block to $RC"
      elif grep -qF "$MARKER_BEGIN" "$RC" 2>/dev/null; then
        say "  $RC already has a proto harness block; leaving it alone"
      else
        mkdir -p "$(dirname "$RC")"
        {
          printf '\n%s\n' "$MARKER_BEGIN"
          printf 'export PATH="%s:$PATH"\n' "$BIN_DIR"
          printf '%s\n' "$MARKER_END"
        } >> "$RC"
        say "  added the block to $RC (remove with --uninstall)"
      fi
    else
      say "  or re-run this script with --write-rc to do it for you"
    fi
    ;;
esac

step "done"
if [ "$FAILED" = 1 ]; then
  say "  verification FAILED — the launchers are linked but not working."
  say "  run one directly to see the error:  $BIN_DIR/proto help"
  exit 1
fi
say "  try it:"
say "      proto doctor --probe-cloud"
say "      cd ~/any/project && proto-code"
