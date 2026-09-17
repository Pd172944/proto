# Routing: choosing between a free local model and a paid cloud model

`proto-harness` exposes four tiers and makes one decision per task: run it locally
(fast, ~$0, weak) or in the cloud (slower, metered, strong)? This document describes
the mechanism actually implemented under `src/router/`, including the parts that are
approximations rather than measurements.

The router is **purely heuristic**. It is a pure function of (features, environment,
config): `extractFeatures()` reads the task text and any code in scope with regexes and
cheap structural analysis, `heuristicScore()` turns that into a probability and a
difficulty score, and `decideRoute()` applies the quality floors and vetoes. Nothing is
learned, no weights are loaded, and no model is called. A decision costs ~0 ms and $0,
which is the whole point: the routing decision has to be cheaper than the work it saves.

## 1. The problem: why cost minimisation is degenerate

Local inference costs ~nothing in marginal dollars (`localElectricityUsd = 0` in
`policy.ts`; electricity is excluded on purpose and is unmeasurable at this scale).
Cloud inference is priced per token by `DEFAULT_PRICING`. If the objective were
"minimise estimated USD", the optimal policy would be trivial and useless: **try local
first, escalate on verification failure, always.** Even at a 30% local success rate the
failed 70% costs only latency, so local-then-escalate beats cloud-first on any
dollar-weighted objective. On paper, cost minimisation *always* says local.

That answer is **right for batch work and wrong for an interactive coding session**. In
a session the user is waiting: a wasted local attempt costs a timeout plus a repair
round plus an escalation, and it spends the user's attention on an answer that had to
be thrown away. Latency, not dollars, is the scarce resource.

**The quality-floor formulation.** Stop comparing costs; compare **confidence** against
a **floor**:

```
local is eligible  <=>  p(local succeeds) >= qualityFloor
```

Escalation then handles only the residual `(1 - p)` instead of being the primary
mechanism. The floor is not one number: it rises when nothing can check the answer
(§6). Everything else here exists to make `p` honest and to make the floor defensible.

## 2. The pipeline and the tier vocabulary

```
TaskContext { task, files[], diff?, constraints? }
   │
   ▼  extractFeatures(ctx)          model-free regex + structural analysis, ~0 ms, $0
   ▼  heuristicScore(features)      → { pLocalSuccess, difficulty, contributions }
   ▼  eligibility vetoes            local disabled/unavailable, context, output, hard locks,
   │                                cloud unavailable, daily budget
   ▼  floor = verifierAvailable ? qualityFloor
   │          : isExplainOnly   ? qualityFloorReadOnly
   │                            : qualityFloorUnverified
   │     localWinsOnQuality = pLocal >= floor
   │     localWinsOnLatency = localLatencyMs <= cloudLatencyMs * latencyToleranceFactor
   ▼  tier selection (veto branches first, then quality+latency, then difficulty)
   ▼
RouteDecision { tier, reason, reasons[], pLocalSuccess, difficulty, taskClass,
                expected, vetoes, unverified, features, forced, env }
```

| Tier | Model source | Meaning |
| --- | --- | --- |
| `local-tiny` | `local.tinyModel` (unset by default) | Easiest tasks; only when a tiny model is configured **and** `difficulty < 0.2` **and** `p >= floor`. |
| `local` | `local.model` (`qwen2.5-coder:1.5b-instruct`) | Main local tier: fast, free, bounded. |
| `cloud-cheap` | `cloud.cheapModel ?? cloud.model` | Easy-to-moderate tasks local must not take. |
| `cloud-strong` | `cloud.model` | Hard tasks, and the escalation target. |

Rank is `['local-tiny','local','cloud-cheap','cloud-strong']`; `isLocalTier()` is the
predicate used everywhere for "did this stay local?". Cloud tier is picked by
difficulty: `difficulty >= 0.5` → `cloud-strong`, else `cloud-cheap`. A hard-locked
class always takes `cloud-strong` regardless of difficulty (§7).

`RouteDecision.env` is a `RouterEnvironmentSnapshot`: which tiers were even eligible at
decision time, plus the local decode-speed estimate and the configured quality floor.
It is carried on the decision so `--explain` can show *why* a tier was unavailable;
the value of a routing decision is that it is auditable, not just correct.

## 3. Features

`extractFeatures()` is deliberately **model-free** — asking a model "is this easy?"
would defeat the point of a cheap tier. Groups: **size** (`taskChars`, `estInputTokens`,
`estOutputTokens`, `fileCount`, `changedLines`); **code structure** (`loopCount`,
`funcCount`, `maxNesting`, `branchCount`); **risk flags** (`hasAsync`, `hasConcurrency`,
`hasTypes`, `hasErrorHandling`, `hasTestsInScope`, `hasStackTrace`, `hasReproSteps`,
`hasExplicitAcceptanceCriteria`, `hasExternalApiMention`, `hasPerfLanguage`,
`hasSecurityLanguage`, `hasMigrationLanguage`); **phrasing** (`isQuestion`,
`isExplainOnly`, `constraintCount`, `ambiguity` 0..1, `locality` 0..1,
`mentionsSpecificSymbol`); **classification** (`taskClass`, `classDifficulty`,
`languages`, human-readable `signals[]`).

Comments and string literals are stripped before keyword counting, so a comment saying
"add a mutex" does not create concurrency in scope. Nesting uses brace depth or
indentation, whichever the file exhibits more of, so Python is not reported as depth 0.
`estOutputTokens` is derived from the requested change size: `changedLines*12 + 120 +
classDifficulty*200` for edits (clamped 80..4000), `changedLines*2 + 320` for
explanations (120..2200), `changedLines*8` for format/rename (40..800).

## 4. Task classification: hardest match wins

Twenty classes, one winner. The prior answers *"how often does a 1.5B-class model get
this right first try, given that a verifier will catch hard failures"* — not how
impressive the task sounds.

| Class | diff | prio | wt | Class | diff | prio | wt |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `format` | 0.05 | 10 | 3.0 | `feature-new` | 0.55 | 42 | 1.4 |
| `rename` | 0.10 | 12 | 3.0 | `refactor-multi` | 0.60 | 55 | 2.2 |
| `prompt-edit` | 0.12 | 34 | 2.5 | `algorithm` | 0.70 | 70 | 2.0 |
| `docs` | 0.12 | 18 | 2.0 | `perf` | 0.70 | 75 | 2.2 |
| `explain` | 0.15 | 22 | 2.0 | `debug-unknown` | 0.72 | 60 | 1.8 |
| `add-validation` | 0.22 | 30 | 2.4 | `concurrency` | 0.75 | 80 | 2.4 |
| `local-edit` | 0.25 | 26 | 1.6 | `security` | 0.80 | 90 | 2.6 |
| `config` | 0.30 | 33 | 1.6 | `migration` | 0.82 | 86 | 2.4 |
| `bugfix-local` | 0.30 | 36 | 2.2 | `architecture` | 0.85 | 88 | 2.4 |
| `write-tests` | 0.35 | 35 | 1.8 | `unknown` | 0.50 | — | — |

Classification is not a vote. Matches are ranked `priority desc, evidence-count desc,
weight desc`, and the single hardest wins:

```ts
matched.sort((a, b) => b.priority - a.priority || b.hits - a.hits || b.weight - a.weight);
```

The asymmetry is the justification: **underestimating difficulty is the expensive
error.** Labelling a performance investigation a "bugfix" sends it to a model that
produces a plausible non-fix, and the verifier may not catch it (performance claims are
not unit-testable). Overestimating costs one cloud call. So `perf` (75), `concurrency`
(80), `migration` (86), `architecture` (88) and `security` (90) outrank generic edit
classes even when those match more keywords. Priorities are the mechanism; the patterns
are deliberately narrow so weak evidence cannot steal a task from specific evidence. Two
revisions are documented in the source:

- `bugfix-local` no longer matches "instead of"/"raises", which had stolen "add a guard
  so an empty list returns 0 instead of raising" from `add-validation`.
- `local-edit` matches **only action verbs**, no scope hints. An earlier revision matched
  "in this file", letting `local-edit` (26) beat `rename` (12) and `prompt-edit` (34) on
  their own tasks. Scope belongs in `locality`, not the class decision.

Escape hatches and one structural upgrade: no class matched but a stack trace is present
→ `bugfix-local` (0.30), because a stack trace is evidence of a concrete defect rather
than an open design question; nothing matched → `unknown` (0.50); and **≥4 files in
scope** upgrades `local-edit`, `bugfix-local`, `add-validation`, `docs` or `rename` to
`refactor-multi` (0.60) — a purely structural override of a text classifier, since
"rename this helper" is local in one file and a multi-file refactor in five.

## 5. The probability model

`heuristicScore()` returns `difficulty` (0..1, used for tier selection and vetoes) and
`pLocalSuccess` (the floor comparison). They are computed separately, which has
consequences: a constraint-heavy task can be pushed off local by the success logit while
`difficulty` says "easy-to-moderate", so the constraint penalty deliberately moves
`difficulty` too (0.45 × constraint load).

```
fileScope      = clamp01(log1p(fileCount) / log1p(8))
nesting        = clamp01(max(0, maxNesting - 1) / 5)
sizeFactor     = clamp01(log1p(changedLines) / log1p(400))
constraintLoad = clamp01(constraintCount / 6)
loopLoad       = clamp01(log1p(loopCount) / log1p(12))
issueScale     = fileCount === 0 ? clamp01((taskChars - 500) / 2500) : 0

difficulty     = clamp01(0.55*classDifficulty
                       + 0.14*ambiguity + 0.10*fileScope + 0.06*nesting
                       + 0.05*sizeFactor + 0.45*constraintLoad + 0.02*loopLoad
                       + 0.16*issueScale
                       + 0.06*hasConcurrency + 0.04*hasSecurityLanguage
                       + 0.04*hasMigrationLanguage + 0.03*hasPerfLanguage
                       + 0.04*hasExternalApiMention
                       - 0.12*locality - 0.05*hasReproSteps - 0.05*hasStackTrace)
```

`locality` is the strongest lever after the class prior. Every term is returned in
`contributions[]` and printed by `proto route --explain`, largest magnitude first.

```ts
let logit = 2.2 - 4.6 * difficulty;                          // steep: 4.6 per unit
// verifiability / prior adjustments:
//   hasTestsInScope        +0.35   tests exist, a wrong answer is catchable
//   hasTypes               +0.20   typed code gives stronger local constraints
//   mentionsSpecificSymbol +0.15   the task names a concrete symbol
//   isExplainOnly          +0.50   no patch to get wrong
//   EASY class             +0.40   format|rename|prompt-edit|docs|explain|local-edit|config
//   HARD class             -0.90   architecture|migration|security|concurrency|algorithm|perf|debug-unknown
//   hasConcurrency && hasAsync -0.35
//   issueScale > 0.2       -0.7 * issueScale * (mentionsSpecificSymbol ? 0.45 : 1)
//   estInputTokens  > 6000     -0.50
//   estOutputTokens > 1800     -0.30
//   constraintCount >= 2       -0.3 * min(constraintCount, 6)
const pLocalSuccess = clamp01(0.02 + 0.96 * sigmoid(logit));
```

The `0.02 + 0.96 *` mapping keeps the estimate strictly inside `(0.02, 0.98)`: nothing is
ever "certain", which matters because the floor comparison treats `p` as a promise. The
**verifiability bonuses** (`tests`, `types`, `symbol`, explain-only) are small and
additive on purpose: each is evidence that a *wrong* answer is catchable or cheap, not
that the model got better.

The `localization` terms exist because an issue-style report — long prose, no files in
scope — is not the bounded edit its class keywords suggest. Before any fix, the model
must *find* the change site in an unfamiliar codebase, and that search is where small
models actually fail on SWE-bench-shaped work. The difficulty side scales from 0 (short
task) to full effect at ~3k chars; the logit side only bites above `issueScale 0.2`, and
a named symbol softens it by more than half, because a named symbol plus a repro is often
enough for the local model.

**The constraint-interaction penalty** exists because:

> Each explicit constraint ("must not stack", "round half-up", "keep the API unchanged")
> is an independent opportunity to miss a requirement, so reliability degrades roughly
> multiplicatively. We model that with a linear logit penalty, which is the log-space
> equivalent of a product. This is why four easy-sounding constraints together become a
> cloud task.

Four constraints is `-1.2` logits, enough to take a comfortably-local task below the 0.72
floor. The cap at 6 keeps a pathological "20 musts" task finite; it is already far below
any floor.

## 6. The three quality floors

| Config key | Default | Applies when |
| --- | --- | --- |
| `routing.qualityFloor` | **0.72** | a verifier will run (the normal case) |
| `routing.qualityFloorUnverified` | **0.90** | mutating task, no automatic verifier |
| `routing.qualityFloorReadOnly` | **0.55** | read-only explanation task, no verifier |

```ts
const floor = env.verifierAvailable ? cfg.routing.qualityFloor
  : features.isExplainOnly ? cfg.routing.qualityFloorReadOnly
  : cfg.routing.qualityFloorUnverified;
```

The split is three-way because "can we check it?" and "how bad is it if we are wrong?"
are different questions.

- **0.72 with a verifier.** The verifier catches hard failures (parse errors, missing
  anchors, syntax errors, forbidden patterns, optionally the project's tests), so a 28%
  failure rate is recoverable: it escalates and the user still gets a correct answer. This
  is the only floor where escalation is a real safety net.
- **0.90 without a verifier on a mutating task.** Nothing checks the output. A silently
  wrong patch lands in the user's codebase and is found later, at a cost that dwarfs a
  cloud call. Demand near-certainty.
- **0.55 read-only and unverifiable.** An explanation, summary or review is self-checking:
  the human reads it and ignores it if it is wrong. The downside of a weak model is a few
  seconds of reading, materially less bad than a bad patch. The floor stays *low* even with
  no machine check — the human *is* the check.

## 7. Hard locks, vetoes and forcing

Two rules sit outside the probability model, and **no probability may override them**,
because both describe failures a verifier cannot reliably detect:

```ts
export const HARD_LOCKED_CLASSES = ['security', 'architecture', 'migration'];
```

1. **Categorical lock.** A task classified `security`, `architecture` or `migration` goes
   to the cloud: a plausible-but-wrong architecture decision or migration plan is expensive
   and not unit-testable. Exempt when `isExplainOnly` — "explain this migration" has no plan
   to get wrong.
2. **Security-wording lock.** `hasSecurityLanguage && !isExplainOnly && difficulty >= 0.3`.
   This catches edits whose classifier landed elsewhere; the source's motivating example is
   *"harden the check and add tests"*, which reads as `write-tests`. The `difficulty >= 0.3`
   guard stops the lock tripping on a trivial task that merely mentions a sensitive word.

**A hard lock also selects the *strong* cloud tier.** The lock means "a wrong answer here
is expensive", so handing the same task to the cheapest cloud model would contradict the
reason it was locked. `pickCloudTier()` is the single place that decides, and it takes
`cloud-strong` whenever a lock applies, regardless of difficulty.

Eligibility is kept as **two explicit lists** (`localVetoes`, `cloudVetoes`) rather than
string-sniffing over one, so a veto message can never accidentally move a task between
tiers.

| Veto | Condition |
| --- | --- |
| local disabled | `!env.localEnabled` (`local.enabled = false`) |
| local unavailable | runtime unreachable, or configured model absent |
| local context budget | `estInputTokens > local.contextWindow * 0.8` (default: > 6553 of 8192) |
| local output budget | `estOutputTokens > local.maxOutputTokens` (default 1536) |
| local hard lock | class lock or security-wording lock |
| cloud unavailable | `cloud.enabled = false`, or no resolvable API key |
| cloud budget | `cloudBudgetRemainingUsd <= 0` (`routing.cloudBudgetUsdPerDay`, default $5) |

Cloud availability is **inferred from config + key presence, never probed**: a health check
costs money on some providers (Anthropic has no free ping) and adds latency to every
decision. Real failures surface as escalation errors instead. Local availability *is*
probed (`proto route` skips the probe with `--offline`), because the local runtime is free
to ask.

| Situation | Decision |
| --- | --- |
| local vetoed, cloud fine | `cloud-strong` if hard-locked **or** `difficulty >= 0.5`, else `cloud-cheap` |
| cloud vetoed, local fine | `local-tiny` if tiny model and `difficulty < 0.2`, else `local`; `forced = true` |
| **both vetoed** | `local`, `forced = true`: *"both tiers are constrained; attempting local rather than refusing the task"* |
| `p >= floor` and latency OK | `local-tiny` if tiny model and `difficulty < 0.2`, else `local` |
| `p >= floor` but too slow | cloud, tiered by difficulty |
| `p < floor` | cloud, tiered by difficulty |

The "both constrained" fallback is a deliberate product decision: an attempt plus an
explicit warning beats a refusal. It is also the one path where the harness knowingly does
what the quality floor says not to.

## 8. Cost, latency and escalation

The router's `expected` block is a comparison, not a bill: `cloudCostUsd` is
`estInputTokens/1e6 * price.in + estOutputTokens/1e6 * price.out` against the user's own
pricing table; `localCostUsd` is always `0`. Local latency is a TTFT (250 ms resident,
`routing.localColdStartMs` = 4000 ms cold) plus decode at `estimateLocalTokensPerSec()`,
which infers speed from the parameter count in the model name (`240 / params`, ×0.55 for
a full-precision tag, floor 3, default 45 when no count parses). Cloud latency uses
bucketed TTFT and decode rates keyed off `cloud.effort`. Local only wins on latency when
it is within `routing.latencyToleranceFactor` (1.5×) of cloud.

`proto run` is where routing meets execution (`src/harness/loop.ts::runTask`), and the
shape is:

```
route ──► [local attempt ──► verify ──► repair] ──► escalate ──► verify ──► apply?
```

- A local task gets one initial attempt plus `routing.maxLocalRepairAttempts` (default 1)
  repair rounds.
- If it still fails verification and the cloud is available, the harness escalates.
  The escalation target uses `ESCALATION_STRONG_THRESHOLD` = **0.4**, lower than the
  router's own 0.5 boundary: a failed local attempt is evidence the task is harder than
  the estimate, and the cheap/strong boundary for a *known-hard* task should be lower.
- A cloud-first task retries up to `routing.maxCloudAttempts` (default 2), moving to
  `cloud-strong` on the retry when `difficulty >= 0.4`.
- Verification, not model confidence, defines success. `--apply` writes only a report
  that passed; a transport error is recorded as a transport error and is not treated as
  a wrong answer.

## 9. Known weaknesses

1. **The heuristic coefficients are opinions, not measurements.** `0.55`, `4.6`, `+0.35`
   and the rest were chosen to be conservative (a false "easy" costs a wasted local
   attempt plus an escalation; a false "hard" costs a cloud call), not fitted to data.
   This is the entire model.

2. **Routing quality is unvalidated.** There is no offline corpus and no score command
   any more, so the only way to know whether the floors and weights are right is to run
   real tasks and watch. The asymmetry is pinned by unit tests (hard classes never route
   local, every veto fires when it should), but "does this route well?" is an open
   question.

3. **Latency estimates are estimates.** `estimateLocalTokensPerSec()` infers decode speed
   from the parameter count in the model name (base 240 tok/s ÷ params, ×0.55 for full
   precision, floor 3, default 45 with no match). Cloud TTFT and decode rate are bucketed
   by `cloud.effort` (900/1500/2500 ms; 60/30 tok/s). All plausible constants, not
   measurements, so the latency check can flip a decision on bad input.

4. **Classification is imperfect for genuinely ambiguous tasks, and hardest-match-wins
   over-reports difficulty.** The bias is intentional (§4): it is better to send a task to
   an expensive model than to a confident one that cannot do it. The cost is real but
   bounded — an over-classified task costs one cloud call, whereas an under-classified one
   costs a silently wrong answer.

5. **`localOnlyCostUsd` is always zero, by design.** Electricity is excluded from the cost
   model (see §1), so the field is a placeholder for the honest comparison "what would
   local cost in money" — which is nothing.

6. **The daily cloud budget has no ledger behind it.** `routing.cloudBudgetUsdPerDay`
   vetoes cloud only when a caller passes `cloudSpendTodayUsd`; both `proto run` and
   `proto code` pass `0`, and the harness keeps no cross-invocation spend record. The
   real per-task bound is `routing.maxCloudAttempts`. Treat the daily budget as a hook
   for an embedding caller, not as protection you currently get from the CLI.

7. **The router never adapts.** There is no feedback channel: it applies the same rules
   on the first task and the thousandth. If your local model changes, or your task mix
   changes, the rules have to change too.

## 10. Config reference

| Key | Default | Effect |
| --- | --- | --- |
| `routing.qualityFloor` | `0.72` | floor with a verifier |
| `routing.qualityFloorUnverified` | `0.90` | floor for mutating, unverified work |
| `routing.qualityFloorReadOnly` | `0.55` | floor for read-only, unverified work |
| `routing.maxLocalRepairAttempts` | `1` | local repair rounds before escalation |
| `routing.maxCloudAttempts` | `2` | hard cap on cloud calls per task |
| `routing.cloudBudgetUsdPerDay` | `5` | daily budget hook; see §9.6 |
| `routing.latencyToleranceFactor` | `1.5` | local must be within 1.5× cloud latency |
| `routing.localColdStartMs` | `4000` | TTFT used when the model is not resident |
| `local.contextWindow` | `8192` | 80% of this is the local input budget |
| `local.maxOutputTokens` | `1536` | local output veto threshold |
| `local.tinyModel` | unset | enables the `local-tiny` tier |
| `cloud.cheapModel` | unset | `cloud-cheap` falls back to `cloud.model` |
| `verify.enabled` | `true` | default verifier availability |
| `pricing.<model>` | `DEFAULT_PRICING` | USD/Mtok; unknown models get `{in:5,out:20}` |
