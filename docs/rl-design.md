# RL design: learning from episodes at ~zero compute

This document describes how `proto-harness` learns from logged episodes, what is
actually implemented, and where the honest limits are. The central claim under
interrogation is **"we do RL at basically zero compute."** It survives only if you
separate the two loops.

## 1. Two loops, and which one is really free

| | Fast loop | Slow loop |
| --- | --- | --- |
| What | Router logistic regression (`refreshRouter`) | LoRA fine-tuning of the local model (`trainTick` → `mlx_lm.lora`) |
| Cost | Milliseconds, thousands of rows × 43 features | Minutes of GPU on the user's laptop |
| When | **Every tick, while `memory.enabled`** | Only when every scheduler gate passes |
| Cadence | Minutes to hours | Can be skipped for weeks with nothing breaking |
| Where | TypeScript, in-process (`src/router/learned.ts`) | Python + MLX, subprocess (`src/train/`) |
| Delivers | Most of the practical benefit | A preference nudge |

**The fast loop is genuinely free.** `proto train tick` calls `refreshRouter()`
before it evaluates any gate, so the router is retrained even on ticks that then
refuse to train. It is bounded by the episode log (a few thousand rows × 43
features) and takes single-digit milliseconds; gating it behind "the machine is
idle" would be absurd. It learns *which tasks this user's local model can actually
handle*, and that is the majority of the value.

**The slow loop is not free, and the project says so.** `scheduler.ts` states it
plainly: *"The claim is not that training is free — it is that training happens only
in conditions where the compute is genuinely idle and wall-powered, at the lowest OS
priority, in bounded slices, and with a watchdog that aborts the moment the machine
gets busy again."* The slow loop is a background luxury; skipping it degrades
nothing, because the fast loop keeps improving routing and the base model keeps
working.

Both loops read the same episode log. That log is the product.

## 2. The episode as the unit of data

One task in, one episode out. The loop is
`route → (local attempt → verify → repair) → escalate → verify → record`.

| Field group | Contents |
| --- | --- |
| Identity | `id` (monotonic ULID), `schemaVersion`, `ts`, `harnessVersion`, `platform` |
| Task | `task?` (redacted, capped 4000 chars), `taskHash`, `systemPrompt?`, `systemPromptHash` |
| Features | `features` (`TaskFeatures`), `vector` (43 numbers), `vectorVersion` |
| Decision | `decision`: `tier`, `reason`, `reasons[]`, `pLocalSuccess`, `difficulty`, `taskClass`, `exploration`, `forced`, `unverified`, `scorer`, expected costs/latencies, `vetoes[]` |
| Environment | `environment`: `localAvailable`, `localModel`, `localContextWindow`, `cloudAvailable`, `cloudModel`, `cloudProvider`, `cloudPrice`, `configuredQualityFloor`, `verifierAvailable`, `routingMode`, `explorationEpsilon` |
| Attempts | `attempts[]`: `n`, `tier`, `source` (`initial`/`repair`/`escalation`), `providerId`, `model`, `prompt?`, `promptHash`, `promptTokens`, `outputTokens`, `cachedInputTokens?`, `costUsd`, `latencyMs`, `finishReason`, `output?`, `outputHash`, `verification` (passed/score/blockers/failedChecks/durationMs), `error?` |
| Outcome | `outcome`: `status`, `finalTier`, `escalated`, `localSucceeded`, `totalCostUsd`, `totalLatencyMs`, `reward`, `rewardVersion`, `behaviorPropensity` |
| Side data | `feedback?` (accept/reject/edit), `consent` snapshot, `redaction` report, `tags` (`PROMPT_VERSION`) |

Five schema decisions carry their weight:

1. **Append-only.** The store is `<dataDir>/episodes/<YYYY-MM-DD>.jsonl`, size-bounded
   shards, plus a *separate* `feedback.jsonl`. Feedback is never written by rewriting an
   episode; readers merge the two (last record wins). A crash or concurrent writer can
   therefore never truncate history. JSONL is greppable, diffable and copyable, which
   matters more for a research project than what SQLite would buy.
2. **Self-describing versions.** Every record carries `EPISODE_SCHEMA_VERSION`,
   `vectorVersion` and `rewardVersion`. `EpisodeStore.append()` refuses an episode whose
   schema version is not the build's. A dataset built months later can tell which records
   are comparable instead of silently mixing incompatible labels — the single most common
   way RL data pipelines rot.
3. **Replayable offline.** `features` + `decision` + `environment` are enough to re-run
   `decideRoute()` against a counterfactual policy with no model call (`proto replay`).
4. **Redacted at write time.** Nothing is persisted until it has been through the
   redactor. Redacting at upload time would leave plaintext secrets in
   `var/episodes/*.jsonl`, where any bug in the upload path leaks them; redacting at rest
   makes the worst case a less useful training record, not a leaked key. The audit trail
   (`redaction.counts`, `charsRemoved`) is itself stored.
5. **The label is verification, not escalation.** `outcome.localSucceeded` is derived
   from whether the local candidate *passed verification* (`true`/`false`), or `null`
   when local was never attempted under observable conditions. Transport errors are
   recorded on the attempt and cause escalation without ever being scored as "the model
   was wrong" — conflating the two would poison the router's labels with outage data.

## 3. The reward function

Hand-written, shaped, and explicit. From `src/memory/reward.ts`:

```
r = 0.80 * [final answer verified]
  + 0.30 * verification_score
  + 0.20 * [local attempt verified AND no escalation]     (cheap success bonus)
  - 0.60 * [escalated]                                     (local was wrong)
  - 0.30 * [refusal]
  - 0.20 * [envelope drift]
  - 0.25 * [explicit user rejection]
  + 0.25 * [explicit user acceptance]
  - 0.10 * [verifier found blockers]
  + 0.15 * [exploration succeeded]                         (information bonus)
  - 0.05 * [exploration failed]                            (cost of learning)
  - 1.0  * [cloud cost in USD]                             (spend discipline)
```

| `WEIGHTS` key | Value | Sign / magnitude |
| --- | --- | --- |
| `verified` | `0.8` | The dominant positive term: the one thing we can trust. |
| `score` | `0.3` | `0.3 * finalScore`, added **only when verified**; scales a graded pass. |
| `cheapSuccess` | `0.2` | Local passed and no escalation — the behaviour we want. |
| `escalated` | `-0.6` | Local was wrong. **Larger than the cheap-success bonus**, deliberately. |
| `refusal` | `-0.3` | The model declined the task. |
| `envelopeDrift` | `-0.2` | Output ignored the required JSON envelope. |
| `userReject` | `-0.25` | Explicit human "no". |
| `userAccept` | `0.25` | Explicit human "yes". |
| `blockers` | `-0.1` | The final verification still reported blockers. |
| `explorationSuccess` | `0.15` | A counterfactual that *worked* — evidence the router was too cautious. |
| `explorationFailure` | `-0.05` | The (small) price of learning. |
| `costScale` | `-1.0` | `-1.0 * cloudCostUsd`, i.e. a $0.05 escalation costs 0.05 reward. |

`REWARD_MAX = 1.75`, `REWARD_MIN = -1.5`, rounded to 1e-6. Components with value 0 are
omitted from the returned `components[]` so the explanation is not padded.

**Why the binary term dominates.** The RL signal we can actually trust is
`verification.passed` — a boolean produced by a model-independent verifier. Everything
else (score, envelope drift, user feedback, exploration bonuses) is a small
interpretable adjustment. That is why `verified` is 0.8 and every other positive term is
≤ 0.3.

**Why the escalation penalty exceeds the cheap-success bonus.** `-0.6` vs `+0.2`. The
source is explicit: *"we want the policy to learn when local fails more than we want it
to prefer local. Getting that backwards produces a router that confidently sends hard
work to a model that cannot do it."* A reward that paid more for going local than it
charged for being wrong would train exactly the failure mode the whole project exists to
avoid.

**The saturation bug, found and fixed.** The upper clamp used to be `1.5`, but the
maximum achievable positive sum is ~1.7 (`0.8 + 0.3 + 0.2 + 0.25 + 0.15`). A clean local
success — the common case — saturated at 1.5 and therefore swallowed the exploration
bonus and the user-acceptance bonus entirely; both became no-ops in exactly the episodes
they were designed to reward. The clamp was raised to `REWARD_MAX = 1.75`, above the
maximum, so no bonus is ever absorbed. The source keeps the lesson: *"Saturating a reward
hides the distinctions you designed it to make."*

**Versioning.** `REWARD_VERSION = 1` is a constant in `memory/types.ts`, stamped into
`outcome.rewardVersion` on every episode. Changing the formula bumps it, and downstream
tooling filters on the version rather than silently pooling old and new rewards.

## 4. The three learning signals

All three are extracted from the same log by `buildDatasets()`; each exists only under
specific conditions.

| Signal | Exists when | Output |
| --- | --- | --- |
| **SFT self-imitation** | an attempt has `verification.passed === true` and a stored prompt + output | `sft.jsonl` `messages[]`, source `local-verified` |
| **Distillation** | same, but the passing attempt is a **cloud** tier after a local failure | same `sft.jsonl`, source `cloud-verified` |
| **DPO preference pair** | local attempt **failed** verification and a cloud attempt **passed**, both with stored text | `dpo.jsonl` `{prompt, chosen, rejected}` |
| **Router labels** | a local attempt exists **and** has a non-null verification | `router.jsonl` `{x, y, w}` |

- **SFT is rejection sampling / expert iteration with k=1.** We sample cheaply during
  normal use, keep the winners, and fine-tune on them. It is the cheapest possible RL
  loop: no reward model, no extra rollouts, no compute beyond what the user's own work
  already generated.
- **Distillation is the highest-value signal available.** The cloud already solved what
  local could not; the escalation already happened; the cloud output is a
  verified-correct target for a task local failed. `includeDistillation` defaults to
  `true`. Episodes whose user feedback is `reject` are excluded from SFT (a refusal or a
  rejected answer is never worth imitating).
- **DPO needs no reward model** and moves behaviour with a handful of examples, which
  suits a laptop budget. Pairs are only built when there is a genuinely better
  alternative: a rejected answer with no verified replacement is *not* turned into a pair,
  because that would be inventing a preference.

**The label is the verification verdict, not the escalation event.** An escalation tells
you the *router* moved on; it does not tell you the local model was wrong — the local
attempt may never have been verified (verification disabled), may have failed at the
transport layer, or may have been abandoned. `routerLabel()` therefore requires a local
attempt whose `verification !== null`, and returns `null` otherwise. **Undefined labels
are dropped rather than imputed**: imputing "local would have failed" for tasks the
router never sent local is exactly the bias that makes routers overconfident.

**Exploration episodes are the valuable counterfactuals.** They are the only episodes in
which the router attempted local on a task it would otherwise have routed to the cloud,
so they are the only unbiased evidence that the policy is too conservative. That is why
the reward gives a successful exploration `+0.15` (more than a normal success is worth in
information) and a failed one only `-0.05`.

## 5. Why no learned reward model

`reward.ts` argues the case directly, and it is worth repeating because "just train a
reward model" is the reflexive answer:

- **No preference data at the start.** A reward model needs pairs we do not have on day
  one. The explicit reward works from the first episode, and the DPO dataset *becomes* the
  preference data later.
- **Training compute is precisely what we are avoiding.** Adding a reward-model training
  job to a pipeline whose selling point is "no extra compute" is self-defeating.
- **A tiny-data reward model is worse than an explicit one, because its failure modes are
  invisible.** A linear combination of observable facts can be read, argued with, and
  printed per episode. A learned scalar on a few hundred rows fails in ways nobody can
  audit.

So the reward is a documented linear combination, versioned, with per-component
explanations.

## 6. Dataset shapes, dedup and caps

There are two independent sets of caps, and conflating them was a bug:

| Purpose | Where | Default |
| --- | --- | --- |
| Inspection sample for a human to read | `datasets build --max-sft/--max-dpo` | 400 / 300 |
| What the model actually trains on | `train.maxSftSamples` / `train.maxDpoSamples` | 2000 / 1000 |

Training used to inherit the inspection defaults, so it silently discarded
everything past 400 rows even when the user had thousands. Discarding rows buys
nothing: the wall-clock budget, the watchdog and `train.lora.maxIters` are what
bound a session.

```jsonc
// datasets/sft.jsonl  (mlx-lm SFT; also the distillation stream)
{"messages":[{"role":"system","content":"..."},{"role":"user","content":"..."},
             {"role":"assistant","content":"..."}],
 "_meta":{"episodeId":"01J...","source":"local-verified","reward":1.45}}

// datasets/dpo.jsonl  (mlx-lm preference triplets)
{"prompt":"...","chosen":"<cloud, verified>","rejected":"<local, failed>",
 "_meta":{"episodeId":"01J...","chosenTier":"cloud-strong","rejectedTier":"local",
          "reward":0.32,"origin":"escalation"}}

// datasets/router.jsonl  (logistic regression; the trainer reads x/y/w only)
{"id":"01J...","x":[1,0.71,...],"y":1,"w":0.94,
 "meta":{"taskClass":"bugfix-local","tier":"local","exploration":false,"ts":"..."}}
```

`_meta` is provenance; the source comment says mlx-lm ignores unknown keys. The `system`
message is included only when the episode stored a system prompt.

**Deduplication is by content hash**, not by id:

| Dataset | Dedup key |
| --- | --- |
| SFT | `sha256Short(promptHash + ":" + outputHash)` |
| DPO (escalation) | `sha256Short(prompt + ":" + chosenOutputHash + ":" + rejectedOutputHash)` |
| DPO (feedback) | `sha256Short("fb:" + prompt + ":" + chosenOutputHash + ":" + rejectedOutputHash)` |
| Router | `taskHash + ":" + y` |

Duplicates are counted in `stats.duplicatesDropped` and dropped.

**Caps** (a small number of well-chosen examples beats a large noisy set for LoRA on a
laptop):

| Option | Default | Note |
| --- | --- | --- |
| `maxSft` | **400** | |
| `maxDpo` | **300** | |
| `maxRouter` | **5000** | `refreshRouter` uses 20 000; `eval/metrics.ts` uses 50 000 |
| `includeDistillation` | `true` | cloud-solved episodes as SFT targets |

**Selection is reward-ordered.** SFT and DPO samples are sorted by the episode's
`outcome.reward` descending and then truncated, so the retained examples are the best
ones and runs are comparable night to night. Router samples are the exception: they keep
chronological order and are truncated from the *end* (`slice(-maxRouter)`), because newer
routing behaviour matters more than old.

`describeDatasets()` surfaces the diagnostics that matter: episodes considered, SFT split
into local vs cloud, DPO pairs, router rows, duplicates dropped, and how many episodes
were unusable because the prompt was not stored (`memory.storePrompts` /
`memory.storeTaskText` off) — with the instruction to enable them for future episodes.

## 7. The scheduler: every gate, and why nothing runs

`evaluateGates()` is deliberately explicit, and **every gate reports a reason whether it
passes or fails**. That is the whole point: the most common failure of background ML
systems is silently doing nothing, and `proto train status` must always be able to answer
"why isn't this running?".

| Gate id | Config key | Default | Blocks when |
| --- | --- | --- | --- |
| `opt-in` | `train.enabled` | `false` | training was never explicitly enabled |
| `window` | `train.windowStartHour` / `windowEndHour` | `1` / `6` | local hour is outside the window (wraps if start > end) |
| `power` | `train.requireAC` | `true` | on battery, or power source unknown (fails closed) |
| `battery-level` | `train.minBatteryPct` | `20` | only checked when `requireAC` is off and on battery |
| `thermal` | `train.respectThermalState` | `true` | `pmset -g therm` reports `CPU_Scheduler_Limit`/`CPU_Speed_Limit` < 100 |
| `idle` | `train.maxLoadAverage` | `4` | 1-minute load average exceeds the limit |
| `data` | — | — | neither SFT nor DPO has any usable sample |
| `min-episodes` | `train.minNewEpisodes` | `25` | fewer new labelled episodes since the last run |
| `daily-budget` | `train.dailyBudgetMin` | `60` | minutes already used today reach the budget |

Notes that matter in practice:

- On non-macOS, `onAC`, `batteryPct` and thermal state degrade to *unknown*, and the gates
  that depend on unknown values **fail closed** (they block training). Being conservative
  with someone else's machine is the requirement, not a nicety.
- `newEpisodes` is counted by ULID comparison against `state.highWaterEpisodeId` — no
  timestamp parsing, no clock-skew assumptions.
- `minutesUsedToday` reads `state.minutesByDay[YYYY-MM-DD]` (UTC day), while the window gate
  uses local hours. The two clocks differ; it is a minor inconsistency rather than a bug.

**Session planning scales to the time available.** `planSession()`:

```ts
const remaining = Math.max(1, cfg.train.dailyBudgetMin - minutesUsedToday);
const maxRuntimeMin = Math.min(cfg.train.maxRuntimeMin, remaining);      // 20 min default
const affordable = Math.floor((maxRuntimeMin * 60) / 6);                 // ~6 s/iteration
const iters = Math.max(5, Math.min(cfg.train.lora.iters, affordable));   // 60 default
```

The 6 s/iteration estimate is explicitly pessimistic ("better to stop early than to be
killed mid-run"), so a 20-minute window yields the full 60 iterations, a 5-minute window
yields 50, and a 1-minute window yields 10. It also explains itself: *"reduced iterations
from 60 to N to fit M available minute(s)"*.

**Priority.** On macOS the job is wrapped in `taskpolicy -b nice -n 19` (background QoS at
nice 19, yielding to everything interactive); elsewhere it degrades to `nice -n 19`.

**The wall-clock cap and the watchdog are different defences.** `maxRuntimeMs =
plan.maxRuntimeMin * 60_000` is handed to the process timeout. Separately, a 30-second
interval samples `loadavg()[0]` and aborts via an `AbortController` the moment the machine
stops being idle — *"the user opening a video call should immediately preempt this"*. The
two outcomes are tracked distinctly (`timedOut` vs `abortedByWatchdog`); an aborted run is
marked `interrupted`, not `failed`, so the partial adapter it wrote can be kept or
discarded.

**Checkpointing.** mlx-lm is invoked with `--save-every 20`, `--steps-per-report 10`,
`--steps-per-eval 20` and `--seed 0`. With the default 60 iterations that is a checkpoint
every 20 steps, so a watchdog abort loses at most ~20 iterations. `prepareMlxDataDir()`
holds out the last 10% of rows as validation when at least 20 rows exist and no validation
set was supplied, so mlx-lm can report a loss curve — without one, "did this help?" is
unanswerable. (Below 20 rows it duplicates the training set as validation, which makes the
reported validation loss optimistic.)

**Auditability.** A `JobQueue` (`<dataDir>/train/jobs.json`, last 100 jobs) records each
job's plan, dataset, adapter path, exact command, log path, status
(`planned`/`running`/`done`/`failed`/`skipped`/`interrupted`), exit code, duration and
parsed losses. Interrupted work is recorded as interrupted rather than lost.

## 8. LoRA configuration: a nudge, not a retrain

| Flag | Config key | Default |
| --- | --- | --- |
| `--lora-rank` | `train.lora.rank` | `8` |
| `--num-layers` | `train.lora.layers` | `8` |
| `--iters` | `train.lora.iters` (scaled by `planSession`) | `60` |
| `--max-seq-length` | `train.lora.maxSeqLen` | `1024` |
| `--lora-scale` | `train.lora.scale` | `16` |
| `--lora-dropout` | `train.lora.dropout` | `0.05` |
| `--learning-rate` | `train.lora.learningRate` | `1e-5` |
| `--batch-size` | `train.lora.batchSize` | `1` |
| `--train-mode dpo` | `train.lora.mode` | `sft` |
| `--model` | `train.baseModel` | `mlx-community/Qwen2.5-Coder-1.5B-Instruct-4bit` |
| — | `train.keepAdapters` | `3` |
| — | `train.autoPromote` | `false` |

`train.baseModel` is deliberately separate from `local.model`: inference usually runs an
Ollama tag (`qwen2.5-coder:1.5b-instruct`) while training needs a Hugging Face repo MLX can
load. Conflating them produces a confusing failure where training "cannot find" a model
inference uses fine.

Rank 8 over 8 layers for ~60 iterations is chosen to be a **nudge**, not a retrain. The
intent is to teach the local model *this user's* preferences and recurring task shapes,
which needs far less capacity than teaching it to code. Small also means: it runs in
minutes, it cannot catastrophically forget, and it can be thrown away if it makes things
worse. `proto train adapters` lists what exists; `pruneAdapters()` keeps the newest
`train.keepAdapters` (3). Adapters are only auto-promoted when `train.autoPromote` is on
(off by default), and `--save-every 20` plus a `proto-metrics.json` written next to the
adapter preserve provenance across restarts.

**Preflight never installs or downloads.** `preflight()` looks for a project venv
(`<dataDir>/venv/bin/python3`) or `python3`, checks that `mlx_lm` and `mlx_lm.lora` import,
and on failure returns the exact copy-pasteable setup commands without running them. The
module is explicit: *"Silently installing multi-hundred-megabyte packages onto someone's
laptop is exactly the behaviour this project exists to avoid."*

`scripts/nightly-tick.sh` is the boring outer shell: an atomic `mkdir` lock with a 45-minute
stale-lock reclaim, one scheduler pass, JSON logging (`PROTO_LOG=json`), the last 60 tick
logs retained, and exit codes `0` (ran or legitimately skipped), `1` (tick failed), `3`
(another tick holds the lock).

## 9. The final mile: two serving paths

Training an adapter does not, by itself, change what the harness talks to. There are two
viable paths, and the project documents both rather than hiding the friction.

**Path A — MLX serving (recommended).** `mlx_lm.server` loads base + LoRA directly, so no
conversion step exists:

```bash
$DATA_DIR/venv/bin/python -m mlx_lm.server \
  --model mlx-community/Qwen2.5-Coder-1.5B-Instruct-4bit \
  --host 127.0.0.1 --port 8080 \
  --adapter-path $DATA_DIR/models/adapters/sft-XXXX
proto config set local.runtime mlx
proto config set local.baseUrl http://127.0.0.1:8080
proto config set local.model mlx-community/Qwen2.5-Coder-1.5B-Instruct-4bit
```

Downside: it is a second local server alongside Ollama, and MLX's server is less mature.

**Path B — Ollama.** Better ergonomics and model management, but Ollama consumes GGUF, so
an MLX LoRA adapter must be fused and converted first. `writeOllamaModelfile()` emits the
Modelfile (`FROM <base>:latest`, `ADAPTER <gguf>`, `PARAMETER temperature 0.1`,
`PARAMETER num_ctx 8192`) plus the three conversion commands as comments and return values:
`mlx_lm.fuse` → `llama.cpp/convert_hf_to_gguf.py --outtype q4_K_M` → `ollama create`. The
harness never runs them.

Choosing not to hide this is deliberate: *"a harness that claims 'we fine-tuned your model'
but leaves it invisible to the runtime the user actually uses is worse than one that
explains the two options."*

## 10. Simulating the learning curve

"How many tasks before this is meaningfully better than the downloaded base model?"
cannot be answered by reading code, so `proto simulate` answers it with a curve.

```bash
proto simulate --tasks 400                       # the default scenario
proto simulate --tasks 2500 --adaptive           # long horizon, learned router drives routing
proto simulate --sweep                           # pessimistic / realistic / optimistic side by side
```

**What is real:** the router, the 43-feature vector, the policy, the verifier
plumbing, the dataset builder, and the actual logistic-regression trainer — all
evaluated on a held-out set of 800 tasks with ground-truth labels (AUC, Brier, and
routing accuracy at the deployed quality floor).

**What is simulated:** the user (a stochastic draw from the task corpus, 65% small
mechanical work) and the local model's competence.

### The design decision that makes it informative

The simulated local model does **not** simply get better at easy tasks. It has
**per-class affinities** — drawn from a fixed seed and printed in the report — so
it can be unusually good at `bugfix-local` (×1.28) and unusually bad at `explain`
(×0.72) in a way its *difficulty* does not predict.

If the synthetic labels were a pure function of difficulty, the heuristic prior
would *be* the Bayes-optimal model, no amount of learning could beat it, and the
exercise would prove nothing. Putting part of the truth outside the difficulty
scale is what tests the project's actual claim: that the harness can discover
which tasks *your* local model handles, from observation alone.

### What the curve looks like

Measured on an M-series laptop, default settings (`--tasks 2500`, exploration off):

| tasks | labels | local% | true-ok | falsePass | SFT | DPO | hybrid AUC | learned AUC | heuristic AUC | hybrid routing acc |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | 0 | 0% | – | – | 0 | 0 | – | – | 0.676 | – |
| 100 | 68 | 68% | 77.9% | 2.9% | 99 | 12 | 0.724 | 0.644 | 0.676 | 72.4% |
| 250 | 168 | 67% | 74.4% | 3.6% | 247 | 37 | 0.738 | 0.632 | 0.676 | 69.4% |
| 500 | 323 | 65% | 78.0% | 3.1% | 494 | 68 | 0.697 | 0.542 | 0.676 | 68.9% |
| 1000 | 650 | 65% | 76.9% | 3.1% | 986 | 145 | 0.739 | 0.589 | 0.676 | 70.1% |
| 2500 | 1647 | 66% | 77.1% | 3.5% | 2000 | 364 | 0.747 | 0.607 | 0.676 | 73.1% |

SFT rows are no longer capped at 400: the training caps are now separate from the
inspection caps (see §6), so the dataset keeps growing with use. DPO reaches only
~364 pairs at 2,500 tasks, because a preference pair requires a *failure* that the
cloud then rescued.

### How to read that honestly

1. **The router starts learning after roughly 100 tasks.** The gate is 40 *labelled*
   local attempts, and at a ~66% local routing rate that is ~60 tasks of real work
   before the first training run. Below that the system runs purely on the
   heuristic prior.
2. **The prior is strong and the learned scorer does not beat it.** Raw learned AUC
   sits at 0.59–0.64 against the heuristic's 0.676, even with 1,647 labels. The
   *deployed* default is hybrid — the learned score shrunk toward the prior — and
   that is what reaches 0.72–0.75, i.e. a genuine but modest +0.07 AUC. This is
   precisely why `blendWithPrior` caps the learned weight at 0.85, and it is worth
   knowing before assuming "more of my own data" solves routing.
3. **Routing accuracy reaches parity, not superiority, at ~2,000 tasks** (73.1% vs
   73.1%). It is a few points better than the prior at 100 tasks and roughly equal
   thereafter. The measurable win from the fast loop is *adaptation to your task
   mix*, not a dramatic accuracy jump.
4. **The slow loop is data-starved by the *task mix*, not by a cap.** SFT grows
   steadily (2,000 rows by ~2,500 tasks), but a DPO pair needs a task the local
   model got wrong *and* that the cloud then solved, so preference data accrues
   roughly 7x more slowly — ~364 pairs at 2,500 tasks. Preference tuning wants
   several hundred consistent pairs at minimum, so DPO is the signal to watch, not
   SFT. If you want faster preference data, the lever is deliberate data
   generation: let the cloud answer tasks you would otherwise have solved locally,
   purely to manufacture pairs (see §12).
5. **The verifier's error rate is the cost of the whole design.** ~3.5% of local
   attempts were wrong *and accepted*. That number is the reason
   `routing.qualityFloorUnverified` is 0.90 and the reason to enable
   `verify.runTests` on code you care about.

6. **Steps, not rows, decide whether fine-tuning can learn.** The plan reports
   `effectiveEpochs`: at the default 3-epoch target, 2,000 SFT rows means 1,500
   steps at batch 4. The scheduler computes that from the row count and caps it
   against the wall-clock budget, and it *measures* seconds-per-step from previous
   jobs rather than guessing. When the budget cannot cover one epoch,
   `proto train status` says so in plain language instead of quietly running a
   session that cannot learn.

### What this exercise broke, and what got fixed

Building the simulation and running it long-horizon exposed three real defects in
the slow loop — all of them in the "make my local model better" path, all now
fixed and tested:

1. **Training read a file it never wrote.** The scheduler called
   `buildDatasets(..., { write: false })` and then handed `prepareMlxDataDir` the
   *path* of that never-written file. On a fresh install every run died with
   "no training rows found" unless the user happened to have run
   `proto datasets build --write` first. Rows are now serialised straight from
   memory into the job's data directory, and `datasets build --write` is purely an
   inspection feature.
2. **The step count guaranteed the run could not learn.** A fixed 60 iterations at
   batch size 1 shows the model 60 examples — 15% of a single epoch over 400 rows.
   Steps are now derived from a target number of *epochs* over the real dataset,
   capped by the time budget, with batch size 4 for better-conditioned updates.
   The plan reports `effectiveEpochs` and warns when it cannot reach 1.0.
3. **The learned scorer got worse as the log grew** (AUC 0.65 → 0.51 from 68 to
   1,647 labels). Per-sample SGD accumulated gradients over the whole dataset
   without normalising by its size, so the effective step size grew linearly with
   the number of episodes and training oscillated at scale. `trainLogistic` now
   does full-batch gradient descent on the *mean* weighted loss, which makes the
   learning rate, the L2 strength and convergence independent of dataset size —
   and makes training deterministic. `test/simulate.test.ts` guards it.

A fourth, subtler one: `saveConfig` wrote the *entire* merged config, so running
any command froze that day's defaults into `config.json` and later improvements —
better batch size, higher dataset caps — never reached an existing user. It now
writes only the delta from the defaults.

## 11. Roadmap: deliberately not implemented

1. **GRPO or any group-relative method using per-token logprobs.** The config already
   carries `local.requestLogprobs` (default `false`) as the hook, but nothing reads it
   today — there is no rollout group, no per-token scoring, and no advantage computation.
   It would require multiple samples per prompt, which is exactly the extra local compute
   the design avoids during interactive use.
2. **A learned reward model**, once preference data is plentiful. Today the DPO pairs are
   the seed of that dataset; when there are enough, a reward model could replace or
   augment the hand-written one.
3. **True off-policy evaluation with a logged propensity model.** Each episode now records a
   genuine propensity `π(a|x)` and the router dataset applies `min(25, 1/π)` as its
   inverse-propensity weight, so exploration episodes are up-weighted rather than (as in an
   earlier revision) down-weighted by 16×. What is still missing is real OPE: a behaviour
   policy whose action distribution is logged per decision rather than reconstructed from
   "greedy plus an exploration coin", and a doubly-robust or self-normalised estimator with
   variance bounds. See `docs/routing.md` §9 and §11.
4. **Adapter A/B evaluation against a held-out slice.** `TrainJob.evaluation`
   (`beforeReward` / `afterReward` / `samples`) exists as a shape but nothing populates it.
   There is no automated "did the new adapter beat the old one?" gate before promotion.
5. **Adversarial / decontamination checks on contributed data.** `contrib/` stages
   opt-in bundles with consent, a salted pseudonym and redaction, but there is no
   poisoning or dedup-against-eval-corpus check on what comes back — and the eval corpus is
   small enough that contamination would be hard to notice.
