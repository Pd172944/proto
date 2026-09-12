# Local models

`proto-harness` routes easy coding tasks to a local model and hard tasks to a cloud
model, with verification deciding whether the local answer was good enough. The local
tier exists to be *fast and free at the margin*, not to be the best model on the
machine. This document covers installing and tuning it on an Apple Silicon Mac.

Everything below uses only commands that exist in `src/cli/`. No command in this
project downloads a model on its own — see
[Nothing is downloaded automatically](#nothing-is-downloaded-automatically).

## The hardware reality

The reference machine is an Apple Silicon Mac (the dev machine is an M5 with 16 GB of
unified memory). Unified memory is shared by everything: macOS itself takes 3–5 GB, a
browser with many tabs 2–4 GB, and your editor plus language servers another 1–3 GB.
That leaves **roughly 8–10 GB usable** for a model before the machine compresses memory
and starts swapping. Swapping is the failure mode to avoid: decode speed collapses and
the whole machine feels broken, which defeats the point of a local model.

Two numbers govern a model's footprint:

- **Weights**, about 0.6–0.7 GB per billion parameters at 4-bit: 1.5B ≈ 1.0 GB,
  7B ≈ 4.7 GB, 14B ≈ 9 GB on disk and in RAM.
- **KV cache**, which grows with `local.contextWindow`: small for a 1.5B model at 8k
  tokens, large enough to matter at 32k.

Decode is memory-bandwidth bound, so speed scales roughly inversely with parameter count.
That is why **a 1.5B 4-bit model is the recommended default**: ~1 GB of weights plus KV
cache, leaving 7–9 GB for your editor and browser, and good enough for the mechanical
classes the router sends local.

## Model recommendations

The router estimates decode speed with `estimateLocalTokensPerSec` in
`src/router/policy.ts` (used to compare local and cloud latency). It parses the
parameter count from the model name and computes `240 / params` tokens/sec, assuming
4-bit quantisation. Full-precision tags (`fp16`, `bf16`, `q8`, …) are multiplied by
**0.55**; the result is floored at 3; a name with no parseable parameter count falls back
to 45. **These are estimates**, not measurements. `proto setup` prints a more
conservative real-world figure of 60–120 tokens/sec for the 1.5B 4-bit model.

| Model (Ollama tag) | Download | Est. decode, 4-bit | Est. decode, full precision | Good for |
| --- | --- | --- | --- | --- |
| `qwen2.5-coder:1.5b-instruct` | ~1.0 GB | ~160 tok/s | ~88 tok/s | **Default.** `format`, `rename`, `prompt-edit`, `docs`, `explain`, `local-edit`, `config` — mechanical, bounded, verifiable edits |
| `qwen2.5-coder:7b-instruct-q4_K_M` | ~4.7 GB | ~34 tok/s | ~19 tok/s | The same classes with fewer failures, plus `bugfix-local`, `add-validation`, `write-tests`; ~3–5× slower |
| `qwen2.5-coder:14b-instruct-q4_K_M` | ~9 GB | ~17 tok/s | ~9 tok/s | Best local quality on a 32 GB+ machine, but no longer "very fast" |

A larger local model does **not** unlock the hard classes. `security`, `architecture` and
`migration` are categorically vetoed from the local tier (`HARD_LOCKED_CLASSES` in
`src/router/policy.ts`) unless the task is explanation-only, and security-flavoured edits
at difficulty ≥ 0.3 are vetoed too. `concurrency`, `algorithm`, `perf` and
`debug-unknown` are penalised heavily in scoring (`HARD_CLASSES` in
`src/router/heuristic.ts`, −0.9 logit) rather than hard-locked, so they almost always go
to the cloud. Sizes and tags come from `MODEL_RECOMMENDATIONS` in
`src/cli/commands-core.ts`; verify a tag exists before pulling it.

## Install paths

Three paths work, listed in order of recommendation for most people.

### (a) Ollama — the default

Ollama is the default (`local.runtime: "ollama"`) because it is a one-line install with
a model marketplace. The harness uses Ollama's native `/api/chat` API rather than its
OpenAI-compatible shim, specifically so it can send `keep_alive` and explicit `num_ctx`
/ `num_predict` bounds (`src/providers/ollama.ts`).

```bash
# 1. install: pick one
brew install ollama
# or download the desktop app: https://ollama.com/download

# 2. start the server (the desktop app does this for you)
ollama serve

# 3. download the default model
ollama pull qwen2.5-coder:1.5b-instruct

# 4. confirm the harness sees it
proto doctor
```

The harness can print these without running anything, and `proto models` manages
existing models:

```bash
proto setup                       # prints the install + pull plan, runs nothing
proto setup --download --yes      # actually runs the pull
proto setup --runtime ollama      # explicit, same as the default
proto models list                                             # what the runtime has, plus adapters
proto models pull qwen2.5-coder:7b-instruct-q4_K_M            # prints the command only
proto models pull qwen2.5-coder:7b-instruct-q4_K_M --yes      # runs it
proto config set local.runtime ollama                         # the values below are already the defaults
proto config set local.baseUrl http://127.0.0.1:11434
proto config set local.model qwen2.5-coder:1.5b-instruct
```

### (b) MLX — needed to train or serve a LoRA adapter

MLX is the only supported training backend (`train.backend: "mlx-lora"`) and on Apple
Silicon the fastest inference path. The project keeps it in a virtualenv at
`<dataDir>/venv` — `<repo>/var/venv` by default, or `$PROTO_HOME/venv`
(`pythonCandidate()` in `src/train/mlx.ts`).

```bash
# 1. create the project venv and install mlx-lm (you run these; proto never does)
python3 -m venv "$PROTO_HOME/venv"          # or <repo>/var/venv when PROTO_HOME is unset
"$PROTO_HOME/venv/bin/pip" install --upgrade pip
"$PROTO_HOME/venv/bin/pip" install mlx-lm
# 2. cache a model once (the only large download on this path)
"$PROTO_HOME/venv/bin/python" -m mlx_lm.generate \
  --model mlx-community/Qwen2.5-Coder-1.5B-Instruct-4bit \
  --prompt "hello" --max-tokens 16
# 3. serve it OpenAI-compatibly on the canonical MLX port
"$PROTO_HOME/venv/bin/python" -m mlx_lm.server \
  --model mlx-community/Qwen2.5-Coder-1.5B-Instruct-4bit \
  --host 127.0.0.1 --port 8080
# 4. point the harness at it
proto config set local.runtime mlx
proto config set local.baseUrl http://127.0.0.1:8080
```

`proto setup --runtime mlx` prints the same plan. To serve a trained adapter, add
`--adapter-path <dir>` to the `mlx_lm.server` line (adapters live under
`<dataDir>/models/adapters/`); `proto train adapters` prints the exact command for the
newest one.

Inference and training models are deliberately separate: `local.model` is what the
runtime serves (`qwen2.5-coder:1.5b-instruct` for Ollama), while `train.baseModel` is
the Hugging Face repo LoRA is applied to
(`mlx-community/Qwen2.5-Coder-1.5B-Instruct-4bit`). Conflating them produces a confusing
"training cannot find this model" failure when inference works, which is why
`src/config/schema.ts` documents them separately. Training is opt-in and off by default:

```bash
proto train status        # gates, budget, preflight; prints install steps if mlx-lm is missing
proto train enable        # flips train.enabled and records local-training consent
proto train plan          # dry run of what a tick would do
proto train now           # force a run, gates ignored
```

### (c) llama.cpp — GGUF, manual

llama.cpp is the manual path: build it, fetch a GGUF yourself, run `llama-server`. It is
OpenAI-compatible, but tool calling is disabled for this runtime
(`src/providers/index.ts` sets `tools: false` for `llamacpp`).

```bash
# 1. build
git clone https://github.com/ggml-org/llama.cpp && cd llama.cpp
cmake -B build && cmake --build build -j

# 2. put a GGUF in models/ and serve it on the canonical port
./build/bin/llama-server -m models/qwen2.5-coder-1.5b-instruct-q4_k_m.gguf --port 8080 -c 8192

# 3. point the harness at it
proto config set local.runtime llamacpp
proto config set local.baseUrl http://127.0.0.1:8080
proto config set local.model <the-model-name-the-server-reports>
```

`proto setup --runtime llamacpp` prints this plan.

### The bootstrap script

`scripts/bootstrap-local.sh` bundles the three paths. **It is a dry run by default** and
downloads nothing unless you pass `--yes`.

```bash
./scripts/bootstrap-local.sh                 # show the plan (default)
./scripts/bootstrap-local.sh --runtime mlx   # plan for MLX
./scripts/bootstrap-local.sh --model qwen2.5-coder:7b-instruct-q4_K_M --yes
```

It checks for Node ≥ 22.6 first and exits if the harness cannot run. Flags:
`--runtime <ollama|mlx|llamacpp>`, `--model <name>`, `--data-dir <path>`, `--yes`,
`-h/--help`. Defaults: runtime `ollama`, model `qwen2.5-coder:1.5b-instruct`, data dir
`$PROTO_HOME` or `<repo>/var`.

## RAM discipline knobs

These already exist in config; none is a hidden Ollama setting.

| Key | Default | What it does |
| --- | --- | --- |
| `local.keepAliveSec` | `300` | Idle seconds before the runtime unloads the model. Sent as `keep_alive: "300s"` on every Ollama request. |
| `local.contextWindow` | `8192` | Passed as `num_ctx`; bounds the KV cache. **Keep ≤ 16384 on a 16 GB machine.** |
| `local.maxOutputTokens` | `1536` | Passed as `num_predict`; caps one generation. |
| `local.requestTimeoutMs` | `60000` | Per-request timeout; raise it if a larger model on a busy machine keeps timing out. |
| `local.temperature` | `0.1` | Low by design: mechanical edits, not creative writing. |
| `local.tinyModel` | unset | Optional second, smaller model for the very easiest tasks. |
| `local.enabled` | `true` | `PROTO_DISABLE_LOCAL=1` disables the tier entirely. |

`proto run --unload` forces the model out of RAM when the task finishes. The Ollama
provider does this by POSTing `keep_alive: 0` to `/api/generate`; MLX and llama.cpp have
no unload hook, so the request is a best-effort no-op (`maybeUnload()` in
`src/harness/loop.ts`). With defaults the model is unloaded after five idle minutes
anyway. The router also refuses to go local when the estimated input exceeds 80% of
`local.contextWindow` or the estimated output exceeds `local.maxOutputTokens`, so a
too-small context window shows up as vetos in `proto route --explain`, not truncation.

## Pointing the harness at a runtime

```bash
proto config set local.runtime ollama     # ollama | mlx | llamacpp | openai
proto config set local.baseUrl http://127.0.0.1:11434
proto config set local.model qwen2.5-coder:1.5b-instruct
```

Canonical default endpoints (`LOCAL_RUNTIME_DEFAULTS` in `src/providers/index.ts`):

| Runtime | Default base URL | Notes |
| --- | --- | --- |
| `ollama` | `http://127.0.0.1:11434` | Native `/api/chat`; keep-alive and unload support |
| `llamacpp` | `http://127.0.0.1:8080` | OpenAI-compatible; tools disabled |
| `mlx` | `http://127.0.0.1:8080` | `mlx_lm.server`; JSON schema supported; required for adapters |
| `openai` | `http://127.0.0.1:1234` | Any other OpenAI-compatible server, e.g. LM Studio or vLLM |

`llamacpp` and `mlx` share port 8080, so run only one at a time.

**Auto-correction.** If you change `local.runtime` but leave `local.baseUrl` at another
runtime's canonical default, `effectiveLocalBaseUrl()` detects that exact case, uses the
target runtime's URL, and emits a `PROTO_LOCAL_BASE_URL` process warning. Switching to
`mlx` while `baseUrl` is still `http://127.0.0.1:11434`, for example, talks to port 8080.
This only fires when the configured URL is *exactly* another runtime's default; a custom
port is left alone, so set `baseUrl` explicitly anyway.

## Troubleshooting

### "Ollama is not responding"

`proto doctor` reports:

```
local tier
  runtime       ollama @ http://127.0.0.1:11434
  model         qwen2.5-coder:1.5b-instruct
  status        not ready — Ollama is not responding at http://127.0.0.1:11434
  hint          Start it with `ollama serve` (or the Ollama desktop app). See docs/local-models.md.
```

Fix: start the server with `ollama serve`, or launch the desktop app. If it *is* running,
confirm the port matches `local.baseUrl` (`curl -s http://127.0.0.1:11434/api/tags`). Any
non-empty model list means the server is up; the provider treats an empty list as
"unreachable" because that is what a wrong port looks like.

### "model not downloaded"

```
  status        not ready — Ollama up, but "qwen2.5-coder:1.5b-instruct" is not downloaded
  hint          Run: ollama pull qwen2.5-coder:1.5b-instruct
```

The hint is generated from the configured `local.model`, so the copy-paste command is
always correct. `proto models pull <name> --yes` does the same thing through the harness,
and the check accepts both `name` and `name:tag` forms.

### Runtime switched but `baseUrl` was not

Symptom: `proto doctor` says the runtime is `mlx` but the request goes to the Ollama
port, or a `PROTO_LOCAL_BASE_URL` warning appears. Auto-correction handles the
canonical-default case; if you had a custom `baseUrl`, set it explicitly for the new
runtime:

```bash
proto config set local.runtime mlx
proto config set local.baseUrl http://127.0.0.1:8080
```

### Slow first token

A cold model must be read from disk and loaded. The router models this as
`routing.localColdStartMs` (default `4000`) when the model is not resident, versus 250 ms
when it is. Raise `local.keepAliveSec` so the model stays warm across a burst of tasks,
and avoid `proto run --unload` during an interactive session (it is for reclaiming RAM
after a batch). Ollama responses include `load_duration`, and the provider marks usage as
measured (not estimated) once it sees a real load, so `proto episodes stats` reflects
actual cold starts over time.

### Swap / memory pressure

In order of impact:

1. Drop to the 1.5B model: `proto config set local.model qwen2.5-coder:1.5b-instruct`.
2. Lower `local.contextWindow` to `4096` and `local.maxOutputTokens` to `1024`.
3. Lower `local.keepAliveSec` to `60`, or use `proto run --unload` per task.
4. Disable the local tier temporarily with `PROTO_DISABLE_LOCAL=1` and let tasks route
   to the cloud.

### What `proto doctor` reports, line by line

`proto doctor [--probe-cloud] [--workspace <path>]` prints six sections plus next steps:

- **environment** — Node version (must be ≥ 22.6), platform/arch, CPUs, total RAM, the resolved data dir, whether `config.json` exists yet (defaults are used when it does not), and any load-time config warnings.
- **local tier** — `runtime @ baseUrl`, configured `model`, a ready/not-ready status from the live health probe, the first 8 models the runtime has, a hint when something is missing, and whether a trained adapter is active.
- **cloud tier** — provider id and label, resolved base URL, model (and cheap model), whether an API key was found, whether the tier is enabled, and the configured price per million tokens (`--probe-cloud` makes one tiny real request).
- **verifier** — whether verification is on; whether `python3`, `node` and `git` are on PATH (they gate syntax checking); and whether project tests are configured.
- **local training (opt-in)** — `train.enabled`, `train.baseModel`, and the MLX preflight; when mlx-lm is missing it prints the exact venv + pip commands.
- **data** — episode/shard counts and bytes, local success rate, escalations, cloud spend (total and today vs the daily budget), dataset sample counts, adapter count.
- **next steps** — a de-duplicated list of concrete fixes in the order checks found them.

## Nothing is downloaded automatically

Every large download is behind an explicit confirmation:

- `proto setup` prints commands and runs nothing; `--download` alone is refused and
  requires `--yes` as well.
- `proto models pull <name>` prints the command; only `--yes` runs it.
- `scripts/bootstrap-local.sh` is a dry run unless `--yes` is passed.
- `proto run`, `proto route`, `proto doctor` and `proto train tick` never download a
  model.
- Training never installs packages either: `preflight()` in `src/train/mlx.ts` returns
  copy-pasteable `installInstructions` (venv → pip → `pip install mlx-lm`) and stops. Its
  comment states the rule plainly — it "NEVER installs anything and NEVER downloads a
  model."

## Quick check

```bash
proto setup                       # read the plan for your chosen runtime
proto doctor                      # local tier ready? model downloaded?
proto models list                 # what the runtime actually has
proto route "rename foo to bar" --file src/a.ts --explain   # no model calls
proto run "rename foo to bar" --file src/a.ts --mock        # end-to-end, no runtime needed
```

Once the tier is ready, inspect the route before trusting an edit to it:
`proto route "add a docstring to parse_config" --file src/config.py --explain`, then
`proto run "add a docstring to parse_config" --file src/config.py --dry-run`.