# proto-harness

A **coding agent** for your terminal, and the two-regime harness underneath it.

```bash
./scripts/install.sh                 # link `proto` and `proto-code` onto your PATH
cd ~/my-project && proto-code        # an agent that reads, edits and runs commands
```

Then, if you want the local/cloud economics: easy work goes to a fast model on your
machine, hard work goes to a cloud model behind your key, and verification decides
whether the answer was good enough.

Easy coding work — fix an off-by-one, rewrite a loop, rename a symbol, change a
prompt, add a guard clause, write a docstring — goes to a **very fast local model**
running on your laptop. Hard work — architecture, migrations, concurrency,
security, performance — goes to a **cloud model** behind whatever API key you
already pay for. Every result is **verified locally before you see it**, and failures
**escalate** automatically.

Nothing downloads a model without you asking. There are **zero runtime npm
dependencies**; it runs on Node 22's built-in tooling.

```
                    ┌─────────────────────────────────────────────┐
   task ──────────► │  route  (model-free, ~0 ms, $0)              │
                    │  features → p(local) → floor → tier          │
                    └───────────────┬─────────────────────────────┘
                                    │
              ┌─────────────────────┴─────────────────────┐
              ▼                                           ▼
      ┌───────────────┐                          ┌────────────────┐
      │  local model  │  fast, free              │  cloud model   │  slow, $$
      │  (1.5B q4)    │                          │  (your choice) │
      └───────┬───────┘                          └────────┬───────┘
              │ candidate                                 │ candidate
              ▼                                           │
      ┌───────────────┐   failed verification             │
      │   verify      │ ──────────────► repair ──────► escalate ──┘
      │ syntax, patch │
      │ anchors, size │   passed
      │ patterns,tests│ ──────────────► accept (write only with --apply)
      └───────────────┘
```

---

## Quickstart

### 0. See it work with no model at all

```bash
cd proto
node --experimental-strip-types --disable-warning=ExperimentalWarning src/cli.ts help
# or, equivalently, once you make it available on PATH:
./bin/proto help
```

Routing is a pure function of the task text and your environment, so you can
inspect decisions before configuring anything:

```bash
# with a cloud key configured, a hard-locked class always takes the strongest model:
./bin/proto route "Design the module boundaries for billing"
```

```text
routing decision
  tier          cloud-strong
  reason        hard task (difficulty 0.48, class "architecture", no files in scope) -> strongest cloud model
  p(local ok)   0.299   difficulty 0.477   class architecture
  est. cost     local $0.0000 vs cloud $0.0084
  est. latency  local 7.5s vs cloud 10.2s

why
  - heuristic scorer: p(local)=0.299
  - difficulty 0.477 for class "architecture"
  - local not eligible: local runtime unavailable: Ollama is not responding at http://127.0.0.1:11434; "architecture" is hard-locked to the cloud: a wrong answer here is expensive and the verifier cannot catch it

vetoes
  - local runtime unavailable: Ollama is not responding at http://127.0.0.1:11434
  - "architecture" is hard-locked to the cloud: a wrong answer here is expensive and the verifier cannot catch it
```

Run the whole loop end to end against deterministic mock models — no network, no
local runtime, no downloads:

```bash
./bin/proto run "Fix the off-by-one error in this loop so it does not go out of bounds" \
  --file src/router/policy.ts --mock
```

### 1. Install a local model (you choose when)

`proto setup` **prints** the commands; it downloads nothing.

```bash
./bin/proto setup                      # Ollama path, with a recommendation table
./bin/proto setup --runtime mlx        # MLX inference path
./scripts/bootstrap-local.sh           # same thing as a script; dry-run by default
./scripts/bootstrap-local.sh --yes     # actually run the download
```

The short version, once you have decided:

```bash
brew install ollama          # or https://ollama.com/download
ollama serve                 # the desktop app does this for you
ollama pull qwen2.5-coder:1.5b-instruct
./bin/proto doctor           # confirms the harness can see it
```

Runtime choices, memory limits and troubleshooting are in
**[docs/local-models.md](docs/local-models.md)**.

### 2. Add a cloud key

The key is read from the environment (preferred) or a `0600` file. It is never
written into config and never logged. **A present key is enough** — the cloud tier
enables itself, so `export OPENROUTER_API_KEY=...` is the entire setup.

```bash
export OPENROUTER_API_KEY=...       # or ANTHROPIC_API_KEY, OPENAI_API_KEY, DEEPSEEK_API_KEY, ...
./bin/proto doctor --probe-cloud
```

### 3. Use it

```bash
./bin/proto run "Add a guard so an empty items list returns 0 instead of raising" \
  --file src/prices.py                 # dry run: shows the change, writes nothing
./bin/proto run "..." --file src/prices.py --apply     # write it
./bin/proto route "Design the module boundaries for billing" --explain   # → cloud
```

---

## The agent (`proto code`)

An interactive agent you run from inside a project, like `claude` or any other
terminal coding agent. It gathers context (git state, project layout, `AGENTS.md` /
`CLAUDE.md`, the codebase index), then loops: model → tool calls → results → model,
until the task is done or it is genuinely blocked.

The agent also **uses the router**, at session granularity: the first message is a task
description, so it is classified exactly like a batch task, and the session runs on the
chosen tier. Later turns stay on that model — switching mid-conversation would break
prompt caching — until you `/route` again or `/escalate`. Any explicit model choice
(`--local`, `--model`, `--provider`) turns routing off.

| | |
|---|---|
| **Tools** | `read_file`, `list_files`, `search` (read) · `write_file`, `edit_file` (write) · `run_command` (exec) |
| **Consent** | every write shows a real diff and every command shows the exact command line, before it runs. "Always" lasts only for the session |
| **Edits are verified** | `edit_file` refuses a missing *or ambiguous* anchor, a change that would not parse, and introduced anti-patterns — reusing the same verifier the batch harness uses |
| **Routed** | the router picks the tier from your first message; `/route`, `/escalate` and `--escalate-on-stuck` move it |
| **Bounded** | step cap and wall-clock budget per turn, plus `--budget-usd`, so it cannot loop forever on your money |
| **Interruptible** | Ctrl-C aborts the turn, not the session |
| **Fails closed** | a piped run denies writes unless you pass `--yes` deliberately |

```bash
proto code                              # interactive, in the current directory
proto code --read-only                  # physically cannot write anything
proto code --no-route                   # one configured model, no routing
proto code --local                      # pin the local model for this session
proto code --demo                       # zero-setup: scripted provider, real loop
proto code "explain the auth flow" --print    # one-shot, composes with pipes
```

Slash commands: `/help /route /escalate /deescalate /model /local /cloud /workspace
/tools /cost /clear /sessions /save /quit`. `!command` runs a shell command directly.

Colour scheme is ember-and-deep-water (amber structure, teal for actionable things,
coral only for failure), inspired by [gum](https://github.com/charmbracelet/gum)'s
shape but not its palette. Full guide, including the baseline test procedure:
**[docs/interactive.md](docs/interactive.md)**.

## Why this exists

Two facts about local models on a laptop, both true at once:

1. A 1.5B 4-bit model answers a bounded, fully-specified question — *"this loop is
   off by one, fix it"* — correctly and in a second, for free, with no data
   leaving the machine.
2. The same model will confidently produce a plausible, subtly wrong refactor of
   your auth layer in ten seconds, and you will not notice until production.

The interesting engineering is not "call a local model". It is **knowing which of
those two situations you are in**, and knowing it cheaply enough that the routing
decision costs less than the work it saves. That is what this repo is about, and
it is why the most carefully reasoned code here is the router and the verifier —
not the API clients.

---

## The two regimes

| Tier | Model | Used for | Typical latency |
|---|---|---|---|
| `local-tiny` | optional second local model (e.g. 1.5B) | trivial, fully-specified edits | < 1 s |
| `local` | your configured local model | easy, verifiable tasks | 1–5 s |
| `cloud-cheap` | a cheaper model on the same provider | moderate tasks local could not take | 2–6 s |
| `cloud-strong` | your chosen cloud model | hard, risky or unverifiable work | 5–30 s |

Routing is **model-free and purely heuristic**: features are extracted with regexes
and cheap structural analysis of the task and any files in scope, so a decision
costs ~0 ms and $0. There is no trained scorer and no learning loop — the rules are
the whole policy. See **[docs/routing.md](docs/routing.md)**.

The decision rule is an expected-cost comparison with a **quality floor**, and the
distinction matters. Pure cost minimisation is degenerate: local inference is
free, so "try local and escalate" always looks cheapest on paper — right for batch
work, wrong when a human is waiting. So the harness requires
`p(local succeeds) ≥ floor` before spending your time on a local attempt, where
the floor depends on how bad a wrong answer would be:

| Situation | Floor | Rationale |
|---|---|---|
| Verifiable output | `0.72` | the verifier catches hard failures |
| Mutating, **no** verifier | `0.90` | a bad patch that nothing checks is the expensive case |
| Read-only, no verifier | `0.55` | you read an explanation and ignore it if it is wrong |

Three task classes — `security`, `architecture`, `migration` — are **hard-locked
to the cloud**. A plausible-but-wrong migration plan is not caught by a syntax
check, so no probability score is allowed to override that.

---

## Verification: why this is safe to run

The verifier is the load-bearing component. A router without one is just a way to
be wrong more cheaply.

Models are asked for a strict JSON envelope of `find`/`replace` anchors rather
than a unified diff, because a small model produces malformed diffs constantly
and a bad anchor fails *loudly and locally*. The verifier then checks, in order of
cheapness:

| Check | Catches |
|---|---|
| envelope parse (+ fence salvage) | output format drift, refusal, empty edits |
| anchor exists / appears exactly once | stale or hallucinated understanding of the file |
| size cap, mass-deletion guard, no-op detection | over-eager rewrites |
| **real parsers** — `ast.parse` for Python, `node --check` for JS/TS | genuine syntax errors, using the language itself rather than a heuristic |
| pattern scan of *added lines only* | new TODOs, swallowed errors, hardcoded credentials, `os.system`, `--no-verify`, `@ts-ignore` |
| optional project test command | behavioural regressions (opt-in; runs at `nice 19`) |

Verification is **read-only**: it works on an in-memory copy of your files.
`--apply` is a separate, explicit flag, and it refuses to write anything that did
not pass.

---

## Privacy

There is no telemetry, no contribution channel, and no episode log. The only ongoing
writer is `proto code`, which saves a **session transcript** to the data directory
(the full conversation, including file contents the agent read and command output). It
is not redacted; `--no-save` turns it off. `proto run`, `proto route` and `proto doctor`
write nothing.

See **[docs/privacy.md](docs/privacy.md)** for exactly what is on disk and what leaves
your machine.

---

## Command reference

| Command | Purpose |
|---|---|
| `proto code [prompt] [--local] [--read-only] [--demo] [--print] [--no-route]` | **interactive coding agent** rooted at the current directory |
| `proto doctor [--probe-cloud]` | check runtimes, providers and the verifier; say exactly what is missing |
| `proto setup [--runtime …] [--download --yes]` | print (or run) local model install instructions |
| `proto route "<task>" [--file p]… [--explain]` | show the routing decision; no model calls |
| `proto run "<task>" [--file p]… [--apply] [--tier t] [--mock] [--dry-run]` | full loop: route → attempt → verify → escalate |
| `proto models list \| use <name> \| pull <name> [--yes]` | what the local runtime has, and download commands |
| `proto config path\|show\|get\|set\|set-key\|providers` | inspect and edit configuration |
| `proto index [--stats] [--map "focus"] [--symbol N] [--refs N] [--outline p]` | build and inspect the codebase index |

Every command supports `--json`. Add `--help` to any of them.

---

## Configuration

State lives in `<repo>/var` by default; `PROTO_HOME=~/.protoharness` gives you a
machine-wide install. `proto config show` prints the effective config; every key
is settable with `proto config set <dotted.key> <value>` or an environment
variable.

The knobs that matter most:

| Key | Default | Why you would change it |
|---|---|---|
| `local.model` | `qwen2.5-coder:1.5b-instruct` | quality vs speed |
| `local.keepAliveSec` | `300` | how long the model stays in RAM after use |
| `local.contextWindow` | `8192` | KV-cache memory bound |
| `cloud.provider` / `cloud.model` | `openrouter` / Claude Sonnet | your key, your choice |
| `cloud.cheapModel` | — | the `cloud-cheap` tier's model |
| `routing.qualityFloor` | `0.72` | how much local work you tolerate |
| `routing.qualityFloorUnverified` | `0.90` | paranoia level for unverifiable edits |
| `routing.qualityFloorReadOnly` | `0.55` | floor for read-only explanations |
| `routing.maxCloudAttempts` | `2` | hard cap on cloud calls per task |
| `routing.cloudBudgetUsdPerDay` | `5` | daily budget hook (see docs/routing.md §9.6) |
| `verify.runTests` / `verify.testCommand` | off / — | real behavioural verification |

Useful environment variables: `PROTO_HOME`, `PROTO_LOCAL_MODEL`,
`PROTO_CLOUD_PROVIDER`, `PROTO_CLOUD_MODEL`, `PROTO_DISABLE_LOCAL`,
`PROTO_DISABLE_CLOUD`, `PROTO_QUALITY_FLOOR`, `PROTO_CLOUD_BUDGET_USD`, `PROTO_LOG`.

---

## Repository layout

```
src/
  agent/          the interactive agent loop, session state, compaction
  tools/          the tool contract and the file/shell tools      ← edit_file is verifier-backed
  tui/            terminal rendering: palette, boxes, diffs, markdown
  cli/            command dispatch + the core and code commands
  config/         schema, defaults, provider profiles, pricing, secret resolution
  providers/      the provider contract; OpenAI-compatible, Anthropic, Ollama, mock
  router/         features → heuristic score → policy            ← the core
  verify/         candidate parsing, in-memory patching, syntax, patterns, tests
  harness/        prompt construction and the batch run loop
  index/          the codebase index: symbols, references, ranked repo map
  util/           logging, argv, text, hashing, atomic fs, process helpers
test/             383 tests, no network, no hardware required
docs/             interactive.md · codebase-index.md · routing.md · local-models.md · privacy.md
bench/            the agentic benchmark suite (see bench/README.md)
scripts/          install.sh · bootstrap-local.sh · humaneval.py
```

---

## Design decisions worth knowing

These are the choices a reader is most likely to question, with the reasoning:

**No runtime dependencies, no build step.** Node 22 executes TypeScript directly
via type-stripping. A research prototype that needs `npm install` before you can
read its behaviour is a prototype people do not inspect. `npm test` works on a
fresh clone.

**JSON envelope instead of unified diffs.** Small models produce malformed diffs
constantly. `find`/`replace` makes the failure mode "anchor not found", which is
detectable without guessing.

**`find` anchors over line numbers.** A model's line numbers drift; anchors either
match exactly once or the edit is rejected. Ambiguous anchors are rejected rather
than guessed.

**Verification before trust, always.** `passed` is the only definition of success.
This is what makes escalation trustworthy.

**The router is a pure function.** `decideRoute` depends only on features, environment
and config, so a decision is explainable (`proto route --explain`) and reproducible
without calling a model. That purity is what lets routing cost ~0 ms.

**Transport failures are not model failures.** If your local server is down, the
attempt is recorded as a transport error and no verification runs. Conflating an
outage with "the model is bad" would make the harness blame the wrong thing.

**`offline` routing means "no I/O", not "local is broken".** Getting this wrong
silently mis-routes everything; there is a test pinning it down.

**Cloud availability is inferred, never probed.** A health check costs money on some
providers and adds latency to every decision; real failures surface at the attempt.

---

## Research notes and limitations

Stated plainly, because a research project that hides these is not one:

- The heuristic coefficients encode opinions about small-model competence, not
  measurements. They are transparent and unit-tested, not calibrated.
- There is no offline corpus and no score command any more, so routing quality is
  **unvalidated**. Unit tests pin the asymmetries (hard classes never route local,
  every veto fires), but "does this route well?" is open.
- Latency and decode-speed estimates are order-of-magnitude figures derived from
  parameter count and quantisation.
- The daily cloud budget has no ledger behind it; the real bound on a single task is
  `routing.maxCloudAttempts`.
- The router never adapts: it applies the same rules on the first task and the
  thousandth.
- Hardest-match-wins deliberately over-reports difficulty. An over-classified task
  costs one cloud call; an under-classified one costs a silently wrong answer.
- The verifier cannot catch a semantically wrong but syntactically valid change
  unless you enable your project's tests. **That is the single biggest gap**, and
  the reason `qualityFloorUnverified` is high.

---

## Development

```bash
npm test            # 383 tests, no network or hardware required
npm run typecheck   # requires typescript installed (devDependency, optional)
npm run doctor
PROTO_LOG=debug ./bin/proto run "..." --mock
```

The test suite covers the components whose silent degradation would be most
damaging: the router policy (including every veto and hard lock), the verifier, the
codebase index, the agent loop, session persistence, and the transport layer.

## Status

Working prototype, actively a research vehicle. Verified on an Apple M5 / 16 GB
macOS machine with Node 22. No local model is downloaded by this repository, and
no model was downloaded while building it.
