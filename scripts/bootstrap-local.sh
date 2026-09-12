#!/usr/bin/env bash
#
# proto local-model bootstrap.
#
# Downloads nothing unless you pass --yes. Prints exactly what it will run.
#
#   ./scripts/bootstrap-local.sh                     # show the plan (default)
#   ./scripts/bootstrap-local.sh --runtime mlx       # plan for MLX instead of Ollama
#   ./scripts/bootstrap-local.sh --model qwen2.5-coder:7b-instruct-q4_K_M --yes
#
# Design note: the *default* action of this script is to explain, not to install.
# Putting multi-gigabyte downloads behind an explicit confirmation is the whole
# point — a harness that quietly pulls 5 GB onto someone's laptop on first run is
# the behaviour this project exists to avoid.

set -euo pipefail

RUNTIME="ollama"
MODEL="qwen2.5-coder:1.5b-instruct"
CONFIRMED=0
DATA_DIR="${PROTO_HOME:-}"

usage() {
  cat <<'USAGE'
usage: bootstrap-local.sh [options]

  --runtime <ollama|mlx|llamacpp>   runtime to set up (default: ollama)
  --model <name>                    model to download (default: qwen2.5-coder:1.5b-instruct)
  --data-dir <path>                 state directory (default: $PROTO_HOME or ./var)
  --yes                             actually run the download
  -h, --help                        this message

Nothing is downloaded without --yes.
USAGE
}

while [ $# -gt 0 ]; do
  case "$1" in
    --runtime) RUNTIME="${2:-}"; shift 2 ;;
    --model) MODEL="${2:-}"; shift 2 ;;
    --data-dir) DATA_DIR="${2:-}"; shift 2 ;;
    --yes) CONFIRMED=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "unknown argument: $1" >&2; usage >&2; exit 2 ;;
  esac
done

ROOT="$(cd -P "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
[ -n "$DATA_DIR" ] || DATA_DIR="$ROOT/var"

say()  { printf '%s\n' "$*"; }
step() { printf '\n\033[1m%s\033[0m\n' "$*"; }
run()  { printf '  \033[2m$ %s\033[0m\n' "$*"; }

step "1. check the harness itself"
if ! command -v node >/dev/null 2>&1; then
  say "node was not found on PATH. Install Node >= 22.6 first:"
  run "brew install node"
  exit 1
fi
NODE_MAJOR="$(node -p 'process.versions.node.split(".")[0]')"
NODE_MINOR="$(node -p 'process.versions.node.split(".")[1]')"
if [ "$NODE_MAJOR" -lt 22 ] || { [ "$NODE_MAJOR" -eq 22 ] && [ "$NODE_MINOR" -lt 6 ]; }; then
  say "Node >= 22.6 is required (found $(node -v)); the harness uses native TypeScript execution."
  exit 1
fi
say "  node $(node -v) ok"
say "  state directory: $DATA_DIR"

case "$RUNTIME" in
  ollama)
    step "2. install Ollama"
    if command -v ollama >/dev/null 2>&1; then
      say "  already installed: $(ollama --version 2>/dev/null | head -1)"
    else
      say "  not installed. Run one of these yourself:"
      run "brew install ollama"
      run "# or download the app: https://ollama.com/download"
    fi

    step "3. start the server"
    say "  the desktop app does this for you, or:"
    run "ollama serve"

    step "4. download a model: ${MODEL}"
    run "ollama pull ${MODEL}"
    PULL_CMD="ollama pull ${MODEL}"

    step "5. point the harness at it"
    run "proto config set local.runtime ollama"
    run "proto config set local.baseUrl http://127.0.0.1:11434"
    run "proto config set local.model ${MODEL}"
    ;;

  mlx)
    step "2. create a virtualenv and install MLX-LM"
    say "  MLX is both the fastest inference path on Apple Silicon and the trainer."
    run "python3 -m venv \"$DATA_DIR/venv\""
    run "\"$DATA_DIR/venv/bin/pip\" install --upgrade pip"
    run "\"$DATA_DIR/venv/bin/pip\" install mlx-lm"
    PULL_CMD="\"$DATA_DIR/venv/bin/python\" -m mlx_lm.generate --model mlx-community/Qwen2.5-Coder-1.5B-Instruct-4bit --prompt hi --max-tokens 8"

    step "3. download a model (first run caches it)"
    run "$PULL_CMD"

    step "4. serve it"
    run "\"$DATA_DIR/venv/bin/python\" -m mlx_lm.server --model mlx-community/Qwen2.5-Coder-1.5B-Instruct-4bit --port 8080"
    run "proto config set local.runtime mlx"
    run "proto config set local.baseUrl http://127.0.0.1:8080"
    ;;

  llamacpp)
    step "2. build llama.cpp"
    run "git clone https://github.com/ggml-org/llama.cpp && cd llama.cpp && cmake -B build && cmake --build build -j"
    step "3. fetch a GGUF and serve it"
    say "  put a .gguf file in llama.cpp/models, then:"
    run "./build/bin/llama-server -m models/${MODEL}.gguf --port 8080 -c 8192"
    PULL_CMD=""
    ;;

  *)
    say "unknown runtime: $RUNTIME" >&2
    exit 2
    ;;
esac

step "6. verify"
run "proto doctor"

if [ "$CONFIRMED" -eq 1 ] && [ -n "${PULL_CMD:-}" ]; then
  step "running the download now (--yes was passed)"
  run "$PULL_CMD"
  eval "$PULL_CMD"
  say "  done."
else
  say ""
  say "Nothing was downloaded. To run the model download, re-run with --yes."
fi
