# Architecture

How the pieces fit, which invariants hold, and where to extend it.

For the *reasoning* behind the routing and learning design, read
[docs/routing.md](docs/routing.md) and [docs/rl-design.md](docs/rl-design.md).
This document is the map: modules, data flow, contracts, and the invariants that
the test suite exists to protect.

---

## 1. Runtime shape

```
bin/proto ──► node --experimental-strip-types src/cli.ts
                        │
                        ├── src/cli/index.ts        argv, dispatch, --json, exit codes
                        ├── src/cli/commands-core.ts      doctor setup route run models config
                        ├── src/cli/commands-memory.ts    episodes feedback datasets
                        └── src/cli/commands-learning.ts  train eval replay contrib
```

Everything runs on Node built-ins. There is no build step, no bundler, and no
`node_modules` requirement — `node:test` is the test runner, global `fetch` is the
HTTP client, and Node's native type-stripping executes the TypeScript.

Constraints this imposes on the code, which are worth knowing before you edit it:

- **Erasable syntax only.** No `enum`, no `namespace`, no TypeScript parameter
  properties (`constructor(private x: T)`). `tsconfig.json` sets
  `erasableSyntaxOnly` so the typechecker enforces it rather than leaving it to a
  runtime surprise.
- **Explicit `.ts` extensions** on every relative import.
- `import type` for type-only imports (`verbatimModuleSyntax`).

---

## 2. Layers and their contracts

The dependency graph is acyclic and points inward toward `util`:

```
cli  ──►  harness ──►  verify ──┐
          │    │               ├──► router ──► config ──► util
          │    └──► providers ──┘
          └──► memory ──► train ──► eval
```

| Layer | Owns | Must never |
|---|---|---|
| `util` | logging, argv, text, hashing, atomic fs, process spawning | know about models or config |
| `config` | schema, defaults, provider profiles, pricing, secrets | perform network I/O |
| `providers` | the `Provider` contract and its implementations | write to disk or make routing decisions |
| `router` | features, scoring, policy, decisions | call a model, or read the network |
| `verify` | candidate parsing, in-memory patching, checks, tests | **write to the user's workspace** |
| `memory` | episode schema, redaction, reward, store, datasets | train anything |
| `harness` | prompt construction, the agent loop, escalation | decide policy (that is the router's job) |
| `train` | scheduler gates, MLX driver, jobs, adapters | install packages or download models |
| `eval` | corpus, metrics, counterfactual replay | spend money or mutate state |
| `simulate` | synthetic users, ground-truth competence, learning-curve reporting | write to the real episode log or the real weights |
| `contrib` | consent, bundles, outbox, upload | upload without three independent opt-ins |
| `tools` | read/search/write/edit/exec tools, approval-gated by declared risk | mutate anything without an approval, or resolve a path outside the workspace |
| `agent` | the multi-turn agent loop, session state, prompt construction | render to a terminal, or decide policy (approval lives in the caller) |
| `tui` | pure string rendering: palette, boxes, diffs, markdown | write to stdout, or hold any state |

### `Provider`

The one abstraction the router reasons over. Two rules keep it honest:

1. `chat()` returns `finishReason: 'error'` for *model-level* failure and throws
   `ProviderError` only for transport/auth problems. The loop treats those two
   very differently (see §4).
2. `costUsd` is always computed from normalized `Usage` plus the local price
   table, so an episode's recorded cost is comparable across providers.

Implementations: `OpenAICompatibleProvider` (OpenRouter, OpenAI, DeepSeek, Groq,
Mistral, Together, xAI, llama.cpp, LM Studio, vLLM, `mlx_lm.server`),
`AnthropicProvider` (native Messages API, prompt caching, thinking budgets),
`OllamaProvider` (native `/api/chat`, for `keep_alive` and `num_ctx` control), and
`MockProvider` (deterministic, used by tests and `--mock`).

### `RouteDecision`

The router returns a decision, not a model. The decision carries everything the
rest of the system needs to justify and later replay itself:

```ts
{
  tier, reason, reasons[], pLocalSuccess, difficulty, taskClass,
  expected: { localCostUsd, cloudCostUsd, localLatencyMs, cloudLatencyMs },
  exploration, forced, unverified, vetoes[],
  features, vector, vectorVersion, scorer,
  env: { localAvailable, cloudAvailable, verifierAvailable, ... },
}
```

`env` is the snapshot of *tier eligibility at decision time*. It exists so that
`proto replay` can re-score history without inventing a cloud key that was not
there — see §6.

---

## 3. The loop

`src/harness/loop.ts::runTask` is the whole system in one function:

```
route ──► [local attempt ──► verify ──► repair] ──► escalate ──► verify ──► record
```

Ordered invariants, each with a test:

1. **Nothing is written to the workspace unless `apply` is set** — and never when
   verification failed. (`harness.test.ts`)
2. **An attempt "succeeded" only if the verifier passed it.** This is the training
   label, so it cannot be a heuristic.
3. **Escalation is bounded** by `routing.maxCloudAttempts` and the daily cloud
   budget, checked before every cloud call.
4. **Transport failure ≠ model failure.** A failed local attempt with no
   verification records `localSucceeded: null` and produces *no* training label.
5. **The episode is recorded even when the task fails** — a failed-local /
   passed-cloud pair is the single most valuable record in the system.

Providers are resolved through `resolveProviders`, which infers cloud readiness
from config + key presence rather than probing (a health check costs money on
some providers and adds latency to every task). Tests inject providers directly.

---

## 4. Verification

`src/verify/` is deliberately read-only. `applyEdits` mutates an in-memory `Map`
of file contents; disk application is a separate function called only by the loop,
only with `--apply`, and re-validates that the path stays inside the workspace
root (defence in depth on top of `validateEditPath`).

Check order is by cost: parse → structure (anchors, size, no-ops, mass deletion) →
language parsers → pattern scan of added lines → project tests. The first hard
failure is usually enough to escalate, so earlier exits save time.

Scoring: start at 1.0, `-0.4` per error, `-0.08` per warning; a failing candidate
is capped at `0.3` so it can never look "almost good" to the reward function.

Severity discipline matters: only a **real language parser** may produce a hard
error. Heuristic checks (delimiter balance for languages without a wired parser)
warn, because a false hard failure would send work to the cloud for no reason.

---

## 5. Memory

```
episodes/YYYY-MM-DD.jsonl      append-only shards, size-rotated
feedback.jsonl                 append-only; merged on read by episode id
datasets/sft.jsonl             derived, regenerable
datasets/dpo.jsonl             derived, regenerable
datasets/router.jsonl          derived, regenerable
router/weights.json            the learned scorer
train/{jobs,state}.json        job queue + daily budget bookkeeping
models/adapters/<name>/        LoRA adapters + proto-metrics.json
contrib/{consent,identity}.json, contrib/outbox/
```

Two decisions with consequences:

**Feedback lives in a separate file.** Recording a user verdict never rewrites an
episode, so the append-only invariant holds and a crash or a concurrent writer
cannot truncate history.

**Datasets are derived, never stored twice.** `buildDatasets` reads episodes and
emits SFT/DPO/router rows on demand, deduplicating by content hash and capping by
reward. Regenerating after a reward-version change is a `proto datasets build`.

Every record stamps `schemaVersion`, `vectorVersion` and `rewardVersion`, so a
dataset built later can tell which rows are comparable instead of silently mixing
incompatible labels. Loading weights trained on a different `FEATURE_VECTOR_VERSION`
throws rather than producing plausible nonsense.

**Redaction happens in `recordText()` inside the loop**, before anything reaches
the store. This ordering is the privacy design: the plaintext never exists on
disk, so no upload-path bug can leak it.

---

## 6. Replay and evaluation

`decideRoute` is a pure function of `(features, environment, config)`. That single
property is what makes two cheap capabilities possible:

- `proto eval` scores the 22-task corpus in milliseconds with **no model calls**,
  including under alternative configs (`--mode`, `--floor`).
- `proto replay` re-runs a new policy over historical episodes, rebuilding each
  environment from the recorded snapshot.

Replay's limitation is documented rather than hidden: it can only re-score
decisions, so changes on episodes where local was never attempted are hypotheses,
not measurements. Its report says so.

---

## 7. Extension points

**Add a cloud provider** — add a `ProviderProfile` to `src/config/schema.ts` and a
price to `DEFAULT_PRICING`. Nothing else changes: `buildCloudProvider` selects the
API shape from the profile (`openai` vs `anthropic`).

**Add a routing feature** — add it to `TaskFeatures`, compute it in
`extractFeatures`, append it to the vector in `toVector`, and **bump
`FEATURE_VECTOR_VERSION`**. The version bump is what invalidates stale weights;
there is a test asserting the vector length matches `FEATURE_NAMES`.

**Add a task class** — add a `ClassRule` with `difficulty`, `priority` and
patterns to `CLASS_RULES`. Remember the selection rule: *hardest match wins*, so
priority encodes blast radius, not keyword count.

**Add a verifier check** — append a `CheckResult` in `verifyCandidate` with the
right severity. Only a real parser may emit `error` for a syntax judgement.

**Add a training backend** — implement the small surface used by
`train/index.ts::startJob`: `preflight()`, `prepareDataDir()`, `buildCommand()`,
`run()`. The scheduler gates are backend-agnostic.

**Add a gate** — append to `evaluateGates` and always give it a human-readable
detail string for both outcomes; `proto train status` prints them verbatim.

---

## 8. Testing strategy

`npm test` runs 345 tests in ~1.5 s with no network, no hardware, and no local
runtime. Test files mirror modules:

| File | Protects |
|---|---|
| `redact.test.ts` | secrets are removed; **ordinary code is not mangled** |
| `router.test.ts` | classification, the difficulty model, every veto, hard locks, the learner |
| `verify.test.ts` | anchors, no-ops, traversal, real parsers, destructive patches |
| `memory.test.ts` | store round-trips, reward signs and clamping, label existence rules |
| `train.test.ts` | every gate refuses when it should; commands and data prep |
| `harness.test.ts` | dry runs write nothing; apply writes only verified output; outage ≠ failure |
| `eval.test.ts` | corpus score regression guard; hard tasks never routed local |
| `contrib.test.ts` | consent gating, no text by default, dedup, no automatic upload |
| `simulate.test.ts` | the learning curve never degrades with more data; simulations cannot touch real state |
| `util.test.ts` | argv errors on unknown flags; config merge, env overrides, pricing |

Two deliberate properties of the suite:

- **Asymmetry is asserted.** `eval.test.ts` pins "no cloud-labelled task is ever
  routed local", because that error is far worse than the reverse.
- **Determinism is asserted.** Router training is seeded, evaluation disables
  exploration, and ids are monotonic, so repeated runs are comparable.

`node:test` was chosen over a framework so the suite runs on a fresh clone with no
install step — the same reasoning that removed runtime dependencies.
