# Architecture

How the pieces fit, which invariants hold, and where to extend it.

For the *reasoning* behind the routing design, read
[docs/routing.md](docs/routing.md). This document is the map: modules, data flow,
contracts, and the invariants that the test suite exists to protect.

---

## 1. Runtime shape

```
bin/proto ──► node --experimental-strip-types src/cli.ts
                        │
                        ├── src/cli/index.ts          argv, dispatch, --json, exit codes
                        ├── src/cli/commands-core.ts  doctor setup route run models config index
                        └── src/cli/commands-code.ts  code (the interactive agent)
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
cli ──► harness ──► verify ──┐
 │       │   │              ├──► router ──► config ──► util
 │       │   └──► providers ┘
 ├──► agent ──► tools ──► index ──► util
 └──► tui ──► util
```

| Layer | Owns | Must never |
|---|---|---|
| `util` | logging, argv, text, hashing, atomic fs, process spawning | know about models or config |
| `config` | schema, defaults, provider profiles, pricing, secrets | perform network I/O |
| `providers` | the `Provider` contract and its implementations | write to disk or make routing decisions |
| `router` | features, heuristic scoring, policy, decisions | call a model, or read the network |
| `verify` | candidate parsing, in-memory patching, checks, tests | **write to the user's workspace** |
| `index` | discovery, symbol extraction, the reference graph, the repo map | become a dependency — every tool works when it is absent |
| `tools` | read/search/write/edit/exec tools, approval-gated by declared risk | mutate anything without an approval, or resolve a path outside the workspace |
| `agent` | the multi-turn agent loop, session state, prompt construction | render to a terminal, or decide policy (approval lives in the caller) |
| `harness` | prompt construction, the batch run loop, escalation | decide policy (that is the router's job) |
| `tui` | pure string rendering: palette, boxes, diffs, markdown | write to stdout, or hold any state |
| `cli` | argv, dispatch, `--json`, exit codes | contain domain logic |

### `Provider`

The one abstraction both paths reason over. Two rules keep it honest:

1. `chat()` returns `finishReason: 'error'` for *model-level* failure and throws
   `ProviderError` only for transport/auth problems. The loop treats those two
   very differently (see §3).
2. `costUsd` is always computed from normalized `Usage` plus the local price
   table, so a cost is comparable across providers.

Implementations: `OpenAICompatibleProvider` (OpenRouter, OpenAI, DeepSeek, Groq,
Mistral, Together, xAI, llama.cpp, LM Studio, vLLM, `mlx_lm.server`),
`AnthropicProvider` (native Messages API, prompt caching, thinking budgets),
`OllamaProvider` (native `/api/chat`, for `keep_alive` and `num_ctx` control), and
`MockProvider` (deterministic, used by tests and `--mock`).

### `RouteDecision`

The router returns a decision, not a model. The decision carries everything needed
to justify it:

```ts
{
  tier, reason, reasons[], pLocalSuccess, difficulty, taskClass,
  expected: { localCostUsd, cloudCostUsd, localLatencyMs, cloudLatencyMs },
  vetoes[], unverified, features, forced,
  env: { localEnabled, localAvailable, localModelLoaded, cloudAvailable,
         verifierAvailable, cloudBudgetRemainingUsd, ... },
}
```

`env` is the snapshot of *tier eligibility at decision time*. It exists so
`proto route --explain` can say why a tier was unavailable, and so a curious reader
can see that a "forced local" decision came from a missing cloud key rather than
from the score.

---

## 3. The loop

`src/harness/loop.ts::runTask` is the batch system in one function:

```
route ──► [local attempt ──► verify ──► repair] ──► escalate ──► verify ──► apply?
```

Ordered invariants, each with a test:

1. **Nothing is written to the workspace unless `apply` is set** — and never when
   verification failed. (`harness.test.ts`)
2. **An attempt "succeeded" only if the verifier passed it.** Model confidence is
   never an input.
3. **Escalation is bounded** by `routing.maxCloudAttempts`, so a routing bug costs a
   few cents rather than a surprise bill.
4. **Transport failure ≠ model failure.** A failed local attempt records the
   transport error and escalates instead of treating it as a wrong answer.
5. **Escalation moves to the strong tier more eagerly than routing does.** A failed
   local attempt is evidence the task is harder than estimated, so the harness uses
   its own `ESCALATION_STRONG_THRESHOLD` (0.4) rather than the router's 0.5 boundary.

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
is capped at `0.3` so it can never look "almost good" to a caller reading the score.

Severity discipline matters: only a **real language parser** may produce a hard
error. Heuristic checks (delimiter balance for languages without a wired parser)
warn, because a false hard failure would send work to the cloud for no reason.

---

## 5. What is stored on disk

The batch path is stateless: `proto run` and `proto route` read config and secrets
and write nothing. The only ongoing writer is the interactive agent.

| Path | Contents |
|---|---|
| `config.json` | non-secret configuration; names an API-key *environment variable* |
| `secrets.json` | optional keys written by `proto config set-key`, mode `0600` |
| `sessions/<id>.json` | agent transcripts: messages, tool results, model/provider, stats |
| `index/<hash>.jsonl` | the symbol cache: paths, `mtime`/size, definitions, reference names |

Sessions are written after each completed turn and on exit, and are **not redacted**
— a transcript contains whatever the agent read. `--no-save` disables writing.
There is no retention policy; sessions accumulate until deleted. See
[docs/privacy.md](docs/privacy.md).

The index cache is keyed on `mtime + size` and written atomically. It stores names,
not file bodies, and every consumer degrades to a direct scan when it is missing or
stale rather than returning an empty result.

---

## 6. Extension points

**Add a cloud provider** — add a `ProviderProfile` to `src/config/schema.ts` and a
price to `DEFAULT_PRICING`. Nothing else changes: `buildCloudProvider` selects the
API shape from the profile (`openai` vs `anthropic`).

**Add a task class** — add a `ClassRule` with `difficulty`, `priority` and
patterns to `CLASS_RULES`. Remember the selection rule: *hardest match wins*, so
priority encodes blast radius, not keyword count.

**Add a verifier check** — append a `CheckResult` in `verifyCandidate` with the
right severity. Only a real parser may emit `error` for a syntax judgement.

**Change the routing policy** — edit `src/router/policy.ts` or
`src/router/heuristic.ts`. Keep `decideRoute` pure; nothing in it may call a model,
read the network, or read the clock.

---

## 7. Testing strategy

`npm test` runs 383 tests in a few seconds with no network, no hardware, and no
local runtime. Test files mirror modules:

| File | Protects |
|---|---|
| `router.test.ts` | classification, the difficulty model, every veto, hard locks |
| `verify.test.ts` | anchors, no-ops, traversal, real parsers, destructive patches |
| `harness.test.ts` | dry runs write nothing; apply writes only verified output; outage ≠ failure |
| `index-core.test.ts`, `index-lang.test.ts`, `index-tools.test.ts` | extraction, references, the repo map, graceful degradation |
| `agent.test.ts`, `agent-anthropic.test.ts` | the interactive loop, approvals, prompt assembly |
| `edits.test.ts` | anchor matching and the edit tool's refusals |
| `compact.test.ts` | transcript compaction keeps a bounded window |
| `cli.test.ts` | dispatch, `--json`, exit codes |
| `transport.test.ts`, `stream.test.ts` | provider errors, streaming, usage accounting |
| `theme.test.ts` | terminal rendering |
| `util.test.ts` | argv errors on unknown flags; config merge, env overrides, pricing |

Two deliberate properties of the suite:

- **Asymmetry is asserted.** `router.test.ts` pins that a hard-locked or cloud-class
  task is never routed local, because that error is far worse than the reverse.
- **Determinism is asserted.** Routing is a pure function, ids are monotonic, and
  no test reaches the network, so repeated runs are comparable.

`node:test` was chosen over a framework so the suite runs on a fresh clone with no
install step — the same reasoning that removed runtime dependencies.
