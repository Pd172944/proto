# proto-harness

A **coding agent** for your terminal, and the two-regime harness underneath it.

```bash
./scripts/install.sh                 # link `proto` and `proto-code` onto your PATH
cd ~/my-project && proto-code        # an agent that reads, edits and runs commands
```

Then, if you want the local/cloud economics: easy work goes to a fast model on your
machine, hard work goes to a cloud model behind your key, and verification decides
which was good enough.

Easy coding work — fix an off-by-one, rewrite a loop, rename a symbol, change a
prompt, add a guard clause, write a docstring — goes to a **very fast local model**
running on your laptop. Hard work — architecture, migrations, concurrency,
security, performance — goes to a **cloud model** behind whatever API key you
already pay for. Every result is **verified locally before you see it**, failures
**escalate** automatically, and the whole thing is recorded so your local model
and the router can get better over time — cheaply, in the background, and only
if you say so.

Nothing downloads a model without you asking. Nothing leaves your machine
without a separate, explicit opt-in. There are **zero runtime npm
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
      └───────┬───────┘
              ▼
      ┌─────────────────────────────────────────────────────────────┐
      │  episode log (redacted at rest) → datasets → fast loop:     │
      │  retrain the router (ms) │ slow loop: LoRA on the local     │
      │  model (minutes, idle + wall power only, opt-in)            │
      └─────────────────────────────────────────────────────────────┘
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
./bin/proto route "Fix the off-by-one error in this loop so it does not go out of bounds" --explain
./bin/proto eval run          # score the router on the built-in 22-task corpus
./bin/proto eval compare      # heuristic vs learned vs hybrid
./bin/proto simulate --tasks 2500     # model the learning curve (no model needed)
```

Run the whole loop end to end against deterministic mock models — no network, no
local runtime, no downloads:

```bash
./bin/proto run "Fix the off-by-one error in this loop so it does not go out of bounds" \
  --file path/to/prices.py --mock
```

### 1. Install a local model (you choose when)

`proto setup` **prints** the commands; it downloads nothing.

```bash
./bin/proto setup                      # Ollama path, with a recommendation table
./bin/proto setup --runtime mlx        # MLX path (also the trainer)
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

### 2. Add a cloud key

The key is read from the environment (preferred) or a `0600` file. It is never
written into config, never logged, and never stored in an episode. **A present key
is enough** — the cloud tier enables itself, so `export OPENROUTER_API_KEY=...`
is the entire setup.

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
./bin/proto episodes stats             # what has it learned about your local model?
```

---

## The agent (`proto code`)

An interactive agent you run from inside a project, like `claude` or any other
terminal coding agent. It gathers context (git state, project layout, `AGENTS.md` /
`CLAUDE.md`), then loops: model → tool calls → results → model, until the task is
done or it is genuinely blocked.

| | |
|---|---|
| **Tools** | `read_file`, `list_files`, `search` (read) · `write_file`, `edit_file` (write) · `run_command` (exec) |
| **Consent** | every write shows a real diff and every command shows the exact command line, before it runs. "Always" lasts only for the session |
| **Edits are verified** | `edit_file` refuses a missing *or ambiguous* anchor, a change that would not parse, and introduced anti-patterns — reusing the same verifier the batch harness uses |
| **Bounded** | step cap and wall-clock budget per turn, so it cannot loop forever on your money |
| **Interruptible** | Ctrl-C aborts the turn, not the session |
| **Fails closed** | a piped run denies writes unless you pass `--yes` deliberately |

```bash
proto code                              # interactive, in the current directory
proto code --read-only                  # physically cannot write anything
proto code --local                      # use the local model instead of the cloud
proto code --demo                       # zero-setup: scripted provider, real loop
proto code "explain the auth flow" --print    # one-shot, composes with pipes
```

Slash commands: `/help /model /local /cloud /workspace /tools /cost /clear /save /quit`.
`!command` runs a shell command directly.

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
it is why the most carefully reasoned code here is the router, the verifier, and
the reward function — not the API clients.

---

## The two regimes

| Tier | Model | Used for | Typical latency |
|---|---|---|---|
| `local-tiny` | optional second local model (e.g. 1.5B) | trivial, fully-specified edits | < 1 s |
| `local` | your configured local model | easy, verifiable tasks | 1–5 s |
| `cloud-cheap` | a cheaper model on the same provider | moderate tasks local could not take | 2–6 s |
| `cloud-strong` | your chosen cloud model | hard, risky or unverifiable work | 5–30 s |

Routing is **model-free**: features are extracted with regexes and cheap
structural analysis of the task and any files in scope, so a decision costs
~0 ms and $0. A small logistic regression, trained in milliseconds from your own
logged episodes, refines the heuristic prior. See **[docs/routing.md](docs/routing.md)**.

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

## Learning: two speeds, honestly labelled

The project claims "RL at basically zero compute". Interrogating that claim
honestly means separating it in two:

**The fast loop** — retraining the router's logistic regression from your episode
log. Milliseconds, bounded by the size of the log, so it runs on every tick with
no gating at all. This is where most of the practical benefit lives: the harness
learns which tasks *your* local model, at *your* quantisation, can actually
handle.

**The slow loop** — LoRA fine-tuning the local model itself. This is the part that
costs real compute, so it is deferred, gated, and can be skipped for weeks without
anything breaking:

```
gates: opt-in · 01:00–06:00 · wall power · not thermally throttled ·
       1-min load ≤ 4 · ≥ 25 new labelled episodes · ≤ 60 min/day ·
       mlx-lm present (never installed automatically)
```

Every gate reports a reason whether it passes or fails, so `proto train status`
can always answer *"why isn't this running?"* — the most common failure mode of
background ML systems is silently doing nothing.

When it does run, it runs under macOS background QoS (`taskpolicy -b nice -n 19`),
with a wall-clock cap, a **load watchdog that aborts the moment you get busy**, and
a checkpoint every 20 steps. The LoRA config is deliberately tiny (rank 8, 8
layers, ~60 iterations): a nudge that teaches your preferences, not a retrain.

Learning signals come from verification, not from vibes:

- **SFT** — local attempts that *passed verification* (rejection sampling on work
  you already did).
- **Distillation** — tasks local failed and the cloud then solved.
- **DPO preference pairs** — the same task, local's failed answer as `rejected`,
  the cloud's verified answer as `chosen`.

Reward is an explicit, versioned, documented linear function — not a learned
reward model, because a reward model trained on a few hundred laptop-scale samples
has invisible failure modes and needs the compute we are trying to avoid. See
**[docs/rl-design.md](docs/rl-design.md)**.

Opt in when you want it:

```bash
./bin/proto train enable         # private local fine-tuning
./bin/proto datasets build --write
./bin/proto train plan           # dry run: shows the exact command it would run
./bin/proto train now            # force one session (still watchdogged)
./bin/proto train install-agent  # generate a launchd plist; prints install steps
./bin/proto train adapters       # list adapters + how to serve them
```

---

### How much does this actually learn?

`proto simulate` answers that with a curve rather than a promise. It runs the
**real** router and the **real** logistic-regression trainer over a synthetic user,
with a simulated local model that has per-class quirks the heuristic cannot see.

Measured over 2,500 simulated tasks with the shipped defaults:

| | after ~100 tasks | after ~2,500 tasks |
|---|---|---|
| labelled local attempts | 68 | 1,647 |
| deployed (hybrid) router AUC | 0.724 | 0.747 |
| heuristic prior AUC | 0.676 | 0.676 |
| SFT rows available for the local model | 99 | 2,000 |
| DPO preference pairs | 12 | 364 |

Read it this way: the **router** starts adapting after roughly 100 tasks and buys a
real but modest gain — it learns *your* task mix and your model's quirks. For the
**local model itself**, SFT data accumulates steadily but preference data accrues
about 7× more slowly, because a DPO pair needs a task local got wrong *and* the
cloud then fixed. Softer levers that matter more than anything else: raise
`train.maxRuntimeMin` (overnight, e.g. 480) so a session can cover multiple epochs,
and choose a *larger local model* — going 1.5B → 7B will do more for capability
than any amount of LoRA on a laptop's worth of data.

A wrong-but-accepted answer rate of ~3.5% is the honest cost of routing to a cheap
model, which is why `verify.runTests` matters. Full analysis, including the three
slow-loop bugs the curve exposed, is in [docs/rl-design.md](docs/rl-design.md).

## Privacy and consent

Three **independent** decisions, all off by default:

| Decision | Flag | What it means |
|---|---|---|
| Private local fine-tuning | `train.enabled` / `proto train enable` | your episodes improve *your* model; nothing leaves the machine |
| Global sharing | `contrib.enabled` / `proto contrib consent --global on` | derived data may be contributed to a shared pool |
| Raw text sharing | `contrib.shareCode` / `--share-code on` | redacted task/output text may be included in contributions |

Redaction happens **before anything is written to disk**, not at upload time —
so the worst case is a less useful training record, not a leaked key sitting in a
JSONL file. `proto episodes redact-check --text "..."` lets you test it yourself.

By default a contribution contains router features, verification labels,
preference facts, content hashes, and a **rotating pseudonym** — never task text,
file contents or model outputs. Uploads are never automatic: consent **and** a
configured endpoint **and** a per-call `--yes`, and bundles are staged to an
outbox first. `proto contrib preview --live` shows the worst case.

This is linkability reduction and at-rest hygiene, **not** differential privacy.
See **[docs/privacy.md](docs/privacy.md)** for the full threat model, including
what this design does not protect against.

---

## Command reference

| Command | Purpose |
|---|---|
| `proto code [prompt] [--local] [--read-only] [--demo] [--print]` | **interactive coding agent** rooted at the current directory |
| `proto doctor [--probe-cloud]` | check runtimes, providers, verifier, trainer; say exactly what is missing |
| `proto setup [--runtime …] [--download --yes]` | print (or run) local model install instructions |
| `proto route "<task>" [--file p]… [--explain]` | show the routing decision; no model calls |
| `proto run "<task>" [--file p]… [--apply] [--tier t] [--mock] [--dry-run]` | full loop: route → attempt → verify → escalate → record |
| `proto models list \| pull <name> [--yes]` | local models, adapters, and download commands |
| `proto config path\|show\|get\|set\|set-key\|providers` | inspect and edit configuration |
| `proto episodes ls\|show <id>\|stats\|prune\|redact-check` | inspect exactly what has been recorded |
| `proto feedback <id> accept\|reject\|edit` | the strongest learning signal available |
| `proto datasets build [--write] [--sample n]` | inspect or write SFT / DPO / router datasets |
| `proto train status\|tick\|now\|plan\|enable\|disable\|router\|adapters\|install-agent` | the deferred RL machinery |
| `proto eval run\|compare\|list` | score routing offline against the corpus |
| `proto replay [--mode …] [--floor n]` | re-score history under a different policy; no model calls |
| `proto simulate [--tasks n] [--sweep] [--adaptive]` | model the learning curve: how many tasks until it measurably improves |
| `proto contrib status\|preview\|stage\|upload\|consent\|outbox\|rotate` | the opt-in sharing channel |

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
| `routing.exploration.epsilon` | `0.06` | how often to gather counterfactual data |
| `routing.cloudBudgetUsdPerDay` | `5` | hard stop on surprise bills |
| `verify.runTests` / `verify.testCommand` | off / — | real behavioural verification |
| `memory.storeTaskText` / `storePrompts` | `true` | the privacy/completeness trade-off |
| `train.*` | off | the opt-in slow loop |

Useful environment variables: `PROTO_HOME`, `PROTO_LOCAL_MODEL`,
`PROTO_CLOUD_PROVIDER`, `PROTO_CLOUD_MODEL`, `PROTO_DISABLE_LOCAL`,
`PROTO_DISABLE_CLOUD`, `PROTO_DISABLE_MEMORY`, `PROTO_ROUTING_MODE`, `PROTO_LOG`.

---

## Repository layout

```
src/
  agent/          the agent loop, session state, and prompt construction
  tools/          the tool contract and the file/shell tools      ← edit_file is verifier-backed
  tui/            terminal rendering: palette, boxes, diffs, markdown
  cli/            command dispatch + the command groups
  config/         schema, defaults, provider profiles, pricing, secret resolution
  providers/      the provider contract; OpenAI-compatible, Anthropic, Ollama, mock
  router/         features → heuristic → learned scorer → policy   ← the core
  verify/         candidate parsing, in-memory patching, syntax, patterns, tests
  harness/        prompt construction and the agent loop
  memory/         episode schema, redaction, reward, store, datasets
  train/          scheduler gates, MLX driver, job queue, adapters
  contrib/        consent records, bundle construction, outbox, upload
  eval/           task corpus, routing metrics, counterfactual replay
  util/           logging, argv, text, hashing, atomic fs, process helpers
test/             345 tests, no network, no hardware required
docs/             interactive.md · codebase-index.md · routing.md · rl-design.md · local-models.md · privacy.md
bench/            the agentic benchmark suite (see bench/README.md)
scripts/          bootstrap-local.sh (dry-run default) · nightly-tick.sh
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

**Verification before trust, always.** `passed` is the only definition of success,
and it is the training label. This is what makes escalation trustworthy and what
keeps the RL data honest.

**The label is the verification verdict, not the escalation event.** Exploration
episodes therefore produce genuine counterfactual labels — the thing the router
most needs and the thing a purely greedy router never collects.

**Transport failures are not model failures.** If your local server is down, the
attempt is recorded with an error and no verification, so it produces no training
label. Conflating an outage with "the model is bad" would poison the router.

**`offline` routing means "no I/O", not "local is broken".** Getting this wrong
silently mis-routes everything; there is a test pinning it down.

**Redaction at rest, not at upload.** See above.

**A zero-compute fast loop and a gated slow loop, kept separate.** It is what lets
the "zero compute" claim survive scrutiny.

---

## Research notes and limitations

Stated plainly, because a research project that hides these is not one:

- The heuristic coefficients encode opinions about small-model competence, not
  measurements. They are transparent and unit-tested, not calibrated.
- The eval corpus is 22 hand-written tasks. It is a regression guard, not a
  benchmark. Labelling is by expected competence, and a length-only heuristic
  cannot score well on it — that property is itself tested.
- The learned scorer is evaluated in-sample by default. A time-split holdout is
  available (`logisticEvalFromEpisodes(..., {holdout: true})`); with a handful of
  episodes neither number means much.
- The inverse-propensity weighting is an approximation of a stochastic behavior
  policy, not a rigorous off-policy estimator. Documented as such.
- Replay can re-score decisions but cannot know whether local *would* have
  succeeded where it was never attempted. Route changes on unobserved episodes are
  hypotheses.
- Latency and decode-speed estimates are order-of-magnitude figures derived from
  parameter count and quantisation.
- Contributed data has no adversarial or decontamination filtering yet.
- The verifier cannot catch a semantically wrong but syntactically valid change
  unless you enable your project's tests. **That is the single biggest gap**, and
  the reason `qualityFloorUnverified` is high.

Roadmap: GRPO / group-relative methods using per-token logprobs
(`local.requestLogprobs` already exists), a learned reward model once preference
data is plentiful, true off-policy evaluation, adapter A/B evaluation against a
held-out slice, and decontamination checks on contributed data.

---

## Development

```bash
npm test            # 345 tests, no network or hardware required
npm run typecheck   # requires typescript installed (devDependency, optional)
npm run doctor
PROTO_LOG=debug ./bin/proto run "..." --mock
```

The test suite covers the components whose silent degradation would be most
damaging: the redactor (including that it does *not* mangle ordinary code), the
router policy (including every veto and hard lock), the verifier, the reward
function, the scheduler gates (every one must refuse when it should), and the
end-to-end loop (including that a dry run never writes and never pollutes the
episode log).

## Status

Working prototype, actively a research vehicle. Verified on an Apple M5 / 16 GB
macOS machine with Node 22. No local model is downloaded by this repository, and
no model was downloaded while building it.
