# Routing: choosing between a free local model and a paid cloud model

`proto-harness` exposes four tiers and makes one decision per task: run it locally
(fast, ~$0, weak) or in the cloud (slower, metered, strong)? This document describes
the mechanism actually implemented under `src/router/`, including the parts that are
approximations rather than measurements. The router is a **pure function of (features,
environment, config)**, which is what makes `proto eval` and `proto replay` free and
deterministic.

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
be thrown away. Latency, not dollars, is the scarce resource — which is why
`decisionUtility()` prices a user-second at `$0.01`.

**The quality-floor formulation.** Stop comparing costs; compare **confidence** against
a **floor**:

```
local is eligible  <=>  p(local succeeds) >= qualityFloor
```

Escalation then handles only the residual `(1 - p)` instead of being the primary
mechanism. The floor is not one number: it rises when nothing can check the answer
(§6). Everything else here exists to make `p` honest, to make the floor defensible, and
to record enough context that the decision can be re-scored later.

## 2. The pipeline and the tier vocabulary

```
TaskContext { task, files[], diff?, constraints? }
   │
   ▼  extractFeatures(ctx)              model-free regex + structural analysis, ~0 ms, $0
   ▼  heuristicScore(features)          → { pLocalSuccess, difficulty, contributions }
   │     if mode !== 'heuristic' and weights exist:
   │        scorer.predict(vector)
   │        hybrid  → blendWithPrior(learned, heuristic, n)   # shrinkage, cap 0.85
   │        learned → use learned directly
   ▼  eligibility vetoes                disable/unavailable, context, output, hard locks,
   │                                    cloud key, daily budget
   ▼  floor = verifierAvailable ? qualityFloor
   │          : isExplainOnly   ? qualityFloorReadOnly
   │                            : qualityFloorUnverified
   │     localWinsOnQuality = pLocal >= floor
   │     localWinsOnLatency = localLatencyMs <= cloudLatencyMs * latencyToleranceFactor
   ▼  tier selection (veto branches first, then quality+latency, then difficulty)
   ▼  exploration override              epsilon dice, verifier-only, bounded
   ▼
RouteDecision { tier, reasons[], pLocalSuccess, difficulty, expected, vetoes,
                exploration, unverified, features, vector, vectorVersion,
                scorer, forced, env }
```

| Tier | Model source | Meaning |
| --- | --- | --- |
| `local-tiny` | `local.tinyModel` (unset by default) | Easiest tasks; only when a tiny model is configured **and** `difficulty < 0.2` **and** `p >= floor`. |
| `local` | `local.model` (`qwen2.5-coder:1.5b-instruct`) | Main local tier: fast, free, bounded. |
| `cloud-cheap` | `cloud.cheapModel ?? cloud.model` | Easy-to-moderate tasks local must not take. |
| `cloud-strong` | `cloud.model` | Hard tasks, and the escalation target. |

Rank is `['local-tiny','local','cloud-cheap','cloud-strong']`; `isLocalTier()` is the
predicate used everywhere for "did this stay local?". Cloud tier is picked by
difficulty: `difficulty >= 0.5 → cloud-strong`, else `cloud-cheap`. `harness/loop.ts`
uses **0.4**, not 0.5, for escalations and cloud retries, so the band `[0.4, 0.5)`
routes cheap and escalates strong (§11.8).

`RouteDecision.env` is a `RouterEnvironmentSnapshot`: which tiers were even eligible at
decision time. It is persisted (via `EpisodeEnvironment`) because re-running the policy
under a *different* config is only meaningful if we know the cloud was not silently
unavailable. Without it, `proto replay` would "discover" that it should have used a tier
that had no API key.

## 3. Features

`extractFeatures()` is deliberately **model-free** — asking a model "is this easy?"
would defeat the point of a cheap tier, and deterministic features are what make offline
replay sound. Groups: **size** (`taskChars`, `estInputTokens`, `estOutputTokens`,
`fileCount`, `changedLines`); **code structure** (`loopCount`, `funcCount`, `maxNesting`,
`branchCount`); **risk flags** (`hasAsync`, `hasConcurrency`, `hasTypes`,
`hasErrorHandling`, `hasTestsInScope`, `hasStackTrace`, `hasReproSteps`,
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

**The feature vector (`FEATURE_VECTOR_VERSION = 1`).** The learned scorer consumes a
fixed **43-element** vector from `toVector()`. Counts are `log1p`-scaled so a 10k-line
file does not dominate a 10-line one. Index → name:

```
 0 bias                     15 has_repro_steps           30 class_is_validation
 1 log_input_tokens/10      16 has_acceptance_criteria   31 class_is_rename_or_format
 2 log_output_tokens/8      17 has_external_api          32 class_is_tests
 3 log_file_count           18 has_perf_language         33 class_is_docs_or_explain
 4 log_changed_lines/5      19 has_security_language     34 class_is_refactor_multi
 5 log_loop_count           20 has_migration_language    35 class_is_feature_new
 6 log_func_count           21 is_question               36 class_is_perf
 7 log_max_nesting/2        22 is_explain_only           37 class_is_debug_unknown
 8 log_branch_count/3       23 log_constraint_count       38 class_is_concurrency
 9 has_async                24 ambiguity                 39 class_is_security
10 has_concurrency          25 locality                  40 class_is_migration
11 has_types                26 mentions_symbol           41 class_is_architecture
12 has_error_handling       27 class_difficulty          42 class_is_algorithm
13 has_tests_in_scope       28 class_is_local_edit
14 has_stack_trace          29 class_is_bugfix_local
```

The order is load-bearing: `eval/metrics.ts` relies on index 27 being `class_difficulty`
for its time-split baseline. **Adding or reordering a feature requires bumping
`FEATURE_VECTOR_VERSION`**, otherwise old weights are applied to a shifted vector and
produce silent garbage. `assertVectorShape()` throws when a vector's length disagrees
with `FEATURE_NAMES`; `LogisticScorer.fromFile()` throws when weights were trained on a
different vector version or have the wrong length. Every episode stores `vectorVersion`,
so datasets built months apart never get mixed.

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

`heuristicScore()` returns `difficulty` (0..1, used for tier selection, vetoes and
exploration bounds) and `pLocalSuccess` (the floor comparison). They are computed
separately, which has consequences (§11.7).

```
fileScope      = clamp01(log1p(fileCount) / log1p(8))
nesting        = clamp01(max(0, maxNesting - 1) / 5)
sizeFactor     = clamp01(log1p(changedLines) / log1p(400))
constraintLoad = clamp01(constraintCount / 6)
loopLoad       = clamp01(log1p(loopCount) / log1p(12))
difficulty     = clamp01(0.55*classDifficulty
                       + 0.14*ambiguity + 0.10*fileScope + 0.06*nesting
                       + 0.05*sizeFactor + 0.04*constraintLoad + 0.02*loopLoad
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

**The constraint-interaction penalty** exists because:

> Each explicit constraint ("must not stack", "round half-up", "keep the API unchanged")
> is an independent opportunity to miss a requirement, so reliability degrades roughly
> multiplicatively. We model that with a linear logit penalty, which is the log-space
> equivalent of a product. This is why four easy-sounding constraints together become a
> cloud task.

Four constraints is `-1.2` logits, enough to take a comfortably-local task below the 0.72
floor. The cap at 6 keeps a pathological "20 musts" task finite; it is already far below
any floor. The corpus encodes this as `cloud-multi-constraint` ("discounts apply only to
non-sale items, must not stack with coupons, must round half-up, must keep the existing
API unchanged"): individually easy, jointly a cloud task.

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

This is why `local-explain-mutable-default` and `local-comment-behaviour` in the corpus are
`expected: local` despite `verifiable: false`. The report flags them separately
(`WARNING: 2 unverifiable task(s) routed local`) rather than pretending local without a
verifier is the same kind of bet.

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
decision. Real failures surface as escalation errors instead.

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

## 8. Exploration

The router only observes local outcomes on tasks it **sent** local. Without intervention it
can never learn it was too conservative, and the learned scorer inherits that blind spot
forever. Exploration is the fix and **the only source of counterfactual labels in the
system**.

```ts
if (cfg.routing.exploration.enabled                                       // true
    && env.verifierAvailable                                              // never explore unverified
    && tier !== 'local' && tier !== 'local-tiny'
    && !localVetoed && !forced
    && difficulty <= cfg.routing.exploration.maxDifficultyForExploration  // 0.45
    && features.estInputTokens <= cfg.routing.exploration.maxTokensForExploration // 4000
    && random() < cfg.routing.exploration.epsilon) {                      // 0.06
  tier = 'local'; exploration = true;
}
```

Every bound has a reason: a **verifier is required** because an exploration that cannot be
scored is just a random failure shown to the user; **`difficulty <= 0.45`** because
exploring a genuinely hard task is expensive and the answer is unhelpful (the interesting
uncertainty is the middle band); **`estInputTokens <= 4000`** to keep the local attempt
cheap in wall-clock; **hard locks and forced decisions are excluded** (`localVetoed` /
`forced`), so a security task can never be explored into; and **`epsilon = 0.06`** is a ~6%
tax on eligible cloud-routed tasks, and the source of roughly 1 in 17 of all labels the
learned scorer will ever see. Exploration always picks `local`, never `local-tiny`, and is
recorded on the decision and in the reward function's information bonus.

## 9. The learned scorer

`src/router/learned.ts` is L2-regularised logistic regression over the 43-feature vector,
trained in TypeScript. **Why logistic regression rather than a neural net.** The label we
can observe — "did the local attempt survive verification" — is noisy, sparse and heavily
selection-biased (we only see local outcomes on tasks the router already sent local). A
linear model with strong regularisation is far more sample-efficient and much harder to
overfit on a few hundred rows; it trains in ~1 ms, which is what makes "retrain on every
tick" feasible on a laptop; and it is auditable, since `featureImportance()` ranks the
weights for `proto router show`. Training in Node also avoids a Python round-trip on the
routing critical path.

| Constant | Value |
| --- | --- |
| `MIN_TRAINING_SAMPLES` | **40** labelled episodes |
| Epochs (production `refreshRouter`) | 400 (library default 300) |
| Learning rate / L2 | 0.08 / 0.02 (library L2 default 0.01; bias never regularised) |
| Class balancing | on: `posWeight = n/(2*pos)`, `negWeight = n/(2*neg)` |
| Seed / PRNG / schedule | 12345, `mulberry32`, Fisher-Yates; `lr / (1 + epoch/(epochs*0.75))` |
| Bias init | `log((positives+1)/(negatives+1))` |
| Weights file | `<dataDir>/router/weights.json` |

**Below 40 labelled episodes the learned scorer is not used at all.** The heuristic is the
prior the learned model is blended toward, so a cold install degrades gracefully instead of
routing randomly. `routing.mode` selects `heuristic` (rules only), `learned` (learned
probability directly, no blending), or `hybrid` (default):

```ts
const w = Math.min(0.85, sampleCount / (sampleCount + 60));
return { p: w * learned + (1 - w) * prior, learnedWeight: w };
```

The shrinkage weight rises with sample count and **caps at 0.85**, so the heuristic never
fully disappears — it is the safety net if training produces a pathological weight vector.
The cap binds at `n = 340` (`n/(n+60) = 0.85`); at `n = 40` the learned weight is 0.40, at
`n = 60` it is 0.50. `routing.mode = 'learned'` with no usable weights logs *"fewer than the
minimum labelled episodes exist; falling back to heuristic"* and continues: the router never
fails to route because of a missing model file.

**Metrics reported.** `evaluate()` computes accuracy, AUC (Mann-Whitney U with average ranks
for ties), Brier score, log-loss, and a 5-bin reliability curve (`predicted` vs `observed`
per bin). Accuracy alone is meaningless when local success is the majority class, which is
why AUC and Brier sit next to it. Metrics are stored in the weights file and shown by
`proto router show`.

**Selection bias and the propensity correction.** Labels exist only where local was
attempted, and local was attempted mostly where the router already believed it would
succeed. A scorer trained naively on that data learns "local succeeds" and never learns the
tasks it was protected from. Exploration supplies counterfactuals, and an inverse-propensity
weight corrects the mixture.

The implementation separates the two quantities deliberately, because conflating them is a
bug that is easy to make and hard to see:

- `outcome.behaviorPropensity` stores `π(a|x)` — the probability that the behaviour policy
  would take the action it took. `behaviorPropensity()` models the policy as "take the greedy
  decision, then with probability ε send a cloud-bound task to local instead", so
  `π = 1` for forced and greedy-local decisions, `π = 1 - ε` for a greedy cloud decision, and
  `π = ε` for an exploration.
- `routerLabel()` converts that to the training weight with `ipsWeight() = min(25, 1/π)`.

The inversion is the whole point. Exploration episodes are *rare* and they are the **only**
counterfactual evidence in the system, so they must be up-weighted by ~`1/ε ≈ 16.7` to stand
in for the population they represent. Using the propensity itself as the weight — an earlier
revision did exactly that — down-weights precisely the samples that carry the information, by
16×. There is a test asserting the direction (`test/memory.test.ts`), and a cap of 25 so that
a small ε cannot let one noisy label dominate the fit.

## 10. Evaluation and counterfactual replay

`proto eval run` scores the fixed corpus in `src/eval/tasks.ts` offline. Routing is a pure
function, so the corpus is scored in milliseconds with no model calls, no network and no
tokens. The corpus is **22 hand-written tasks** (12 expected local, 10 expected cloud), each
carrying the tier a careful engineer would choose, an expected class, and a `verifiable`
flag. Tasks are labelled by *expected competence*, not token count: a 3-line change to an
auth check is a cloud task; a 40-line mechanical rename is a local task. A corpus where
length predicted the label would let a trivial length heuristic score 100% and teach us
nothing. Two deliberate scoring choices:

- **Exploration is disabled** (`exploration.enabled: false` in the effective config).
  Exploration is a data-gathering behaviour, not a measure of routing quality; leaving it on
  would make the score depend on the dice, and a lucky exploration would mask a bad policy.
- **Steady state is assumed** (`localModelLoaded: true`). In a working session the local
  model is already resident, so the cold-start penalty does not apply. Cold-start behaviour
  legitimately flips very small tasks to the cloud on the first call and is covered by unit
  tests instead.

Each task is scored with `verifierAvailable: task.verifiable` and the full daily budget, so
§6's floors are exercised exactly as in production.

```
$ PROTO_HOME=/tmp/docprobe proto eval run
routing mode: hybrid (no trained weights)
accuracy: 100.0% (22/22)
confusion: expected-local -> predicted [local 12, cloud 0] | expected-cloud -> predicted [local 0, cloud 10]
local precision 100.0% recall 100.0%
cloud precision 100.0% recall 100.0%
task-class accuracy: 100.0%
mean difficulty: 0.257
estimated cloud spend for this policy: $0.1920 (all-cloud would be $0.2396)
WARNING: 2 unverifiable task(s) routed local — a wrong answer there cannot be caught
```

`proto eval compare` runs all three modes; with no weights on disk all three rows are
identical (100.0% accuracy, $0.1920, 2 unverified-local) and the note *"No trained weights
found; the learned mode will fall back to the heuristic"* appears. **The 100% score is a
ceiling effect, not evidence of skill**: the heuristic alone achieves it, so the corpus
cannot currently distinguish the learned scorer from the prior it blends toward (§11.2).
Metrics computed by `computeMetrics`: confusion matrix by expected/predicted, local and cloud
precision/recall, task-class accuracy, mean difficulty, estimated cloud spend vs all-cloud,
and the count of tasks sent local without a verifier. Per-task rows show why each decision
happened — e.g. `local-rename-helper` `p=0.92 d=0.07 → local`; `cloud-multi-constraint`
`p=0.51 d=0.17 → cloud-cheap` (constraints moved `p`, not `difficulty`);
`cloud-auth-review` `p=0.33 d=0.44 → cloud-strong` (hard-locked class); `cloud-vague-debug`
`p=0.27 d=0.51 → cloud-strong`.

**Counterfactual replay.** `proto replay` re-runs a **new** policy over historical episodes
using the features and tier availability recorded at the time, and reports what would have
changed. For each episode it rebuilds a `RouterEnvironment` from `episode.environment`
(`localAvailable`, `cloudAvailable`, `verifierAvailable`, `localContextWindow`,
`cloudPrice`), then calls `decideRoute()` with the stored `features` and `vector`. It reports
how many decisions change, split into `more-cautious` (local → cloud) and `more-local`
(cloud → local); of those, how many had an **observed** local label; the USD delta in
expected top-level cloud spend (negative = cheaper); and `scorerQuality`, the current scorer
scored against observed labels with an optional time-split holdout. Three things it
deliberately does not do, each printed as a note rather than buried in code:

1. **It cannot know whether local would have succeeded on episodes it never attempted.**
   Route changes on unobserved episodes are hypotheses, not measurements.
2. **It never explores** (`random: () => 1`), so it measures the deterministic policy, not
   the exploration mixture.
3. **It excludes escalation cost** from the delta, because it cannot know whether a new
   local attempt would have failed.

```
$ PROTO_HOME=/tmp/docprobe proto replay
episodes replayed: 0
decisions that would change: 0 (more cautious 0, more local 0)
estimated cloud-cost delta: $0.0000 (negative = cheaper)
current scorer vs observed labels: not enough labelled episodes to evaluate
note: Exploration episodes carry genuine counterfactual labels: 0 in this window.
```

## 11. Known weaknesses

1. **The heuristic coefficients are opinions, not measurements.** The source says so
   directly. `0.55`, `4.6`, `+0.35` and the rest were chosen to be conservative (a false
   "easy" costs a wasted local attempt plus an escalation; a false "hard" costs a cloud
   call), not fitted to data. Only the learned scorer is empirical, and it is blended back
   toward these opinions.

2. **The corpus is small, hand-written, self-authored, and the heuristic already scores 100%
   on it.** 22 tasks with 10 cloud cases cannot separate a good policy from a lucky one, and a
   saturating score means `proto eval` cannot demonstrate that the learned scorer beats its
   prior. The labels are themselves opinions ("the tier a careful engineer would choose"),
   including debatable ones such as `cloud-feature-endpoint` being unambiguously cloud.

3. **Latency estimates are estimates.** `estimateLocalTokensPerSec()` infers decode speed from
   the parameter count in the model name (base 240 tok/s ÷ params, ×0.55 for full precision,
   floor 3, default 45 with no match), described as "calibrated against Qwen2.5-Coder on
   M-series unified memory". Cloud TTFT and decode rate are bucketed by `cloud.effort`
   (900/1500/2500 ms; 60/30 tok/s). Local TTFT is 250 ms resident or `localColdStartMs`
   (4000 ms) cold. All plausible constants, not measurements, so the latency check can flip a
   decision on bad input.

4. **No true off-policy evaluation.** `proto replay` re-scores decisions; it cannot observe
   counterfactuals. The only genuine counterfactual labels come from exploration, which fires
   at ε=0.06, only with a verifier, only below difficulty 0.45 and 4000 input tokens, and
   never on hard-locked or forced decisions. Everything else in the log is confounded by the
   router's own past choices. The time-split holdout in `eval/metrics.ts` requires ≥20 train
   / ≥10 test samples before reporting anything.

5. **Classification is imperfect for genuinely ambiguous tasks, and hardest-match-wins
   over-reports difficulty.** The bias is intentional (§4): it is better to send a task to an
   expensive model than to a confident one that cannot do it. The cost is real but bounded —
   an over-classified task costs one cloud call, whereas an under-classified one costs a
   silently wrong answer.

6. **The propensity model is an approximation, not an exact propensity.** `ε` is applied even
   when exploration was *not* permitted (no verifier, or above the difficulty/token bounds),
   which slightly understates π for cloud decisions and therefore over-weights them by
   ~`1/(1-ε) ≈ 1.06`. Negligible in practice, but it is an approximation. The behaviour policy
   is also deterministic given the features apart from the exploration coin, so there is no
   richer per-action distribution to log without changing the policy itself.

7. **The eval corpus is small, hand-written and self-authored, and the heuristic saturates it.**
   22 tasks with 10 cloud cases cannot separate a good policy from a lucky one, and a
   saturating score means `proto eval` currently cannot demonstrate that the learned scorer
   beats its prior. The labels are opinions ("the tier a careful engineer would choose"),
   including debatable ones such as `cloud-feature-endpoint` being unambiguously cloud.
   `eval compare` reports heuristic, learned and hybrid at the same 100% until enough episodes
   exist to train weights — a ceiling effect, not evidence.

8. **`localOnlyCostUsd` is always zero, by design.** Electricity is excluded from the cost
   model (see §1), so the field is a placeholder for the honest comparison "what would local
   cost in money" — which is nothing. It is kept in `RoutingMetrics` so a future
   energy-aware model has a slot, but reading it as a measurement today would be wrong.

### Fixed during development

Recorded because the reasoning is more instructive than the fixes, and because each has a
regression test now.

- **The inverse-propensity weight was inverted.** `behaviorPropensity()`'s last two branches
  were identical (it never conditioned on the chosen action) and `routerLabel()` used `π`
  directly as the weight, down-weighting exploration episodes ~16×. Now `π` is stored and
  `ipsWeight()` inverts it, capped at 25. See §9.
- **`routing.localPreferenceMargin` was declared, documented and never read.** Removed.
  `routing.latencyToleranceFactor` is the margin that actually applies.
- **`decisionUtility()` was implemented and exported but called from nowhere.** It is now the
  source of `RoutingMetrics.meanUtility`, reported by `proto eval run`, which makes
  cost-versus-latency trade-offs comparable across policies.
- **Stale weights degraded silently.** `routeTask()` discarded the `error` from
  `loadScorer()`, so after a `FEATURE_VECTOR_VERSION` bump the router quietly ran
  heuristic-only. The failure is now carried on `RouteDecision.scorerError` and printed as the
  first reason by `proto route --explain`.
- **Replay and eval disagreed about the cold start.** `replay` hard-coded
  `localModelLoaded: false` while `evaluateRouting` assumed steady state. `localModelLoaded` is
  now recorded in `EpisodeEnvironment`, so replay reproduces the cold-start penalty the
  original decision actually saw.
- **Eval priced every task at a hard-coded `{in: 3, out: 15}`.** It now uses
  `priceFor(cfg, cloudTierModel(cfg, 'cloud-cheap'))`, so the reported spend reflects the
  user's own provider, model and pricing table.
- **Routing and escalation used unexplained literals for the cheap/strong boundary.**
  `CLOUD_STRONG_DIFFICULTY_THRESHOLD` (0.5) is now the single source of truth in `policy.ts`,
  and the harness has its own explicitly-named `ESCALATION_STRONG_THRESHOLD` (0.4) with a
  comment explaining why a failed local attempt justifies a stronger model.
- **`difficulty` and `pLocalSuccess` contradicted each other on constraint-heavy tasks.** The
  constraint penalty moved only the logit, so `cloud-multi-constraint` was pushed off local
  but still reported `difficulty 0.17` and the reason "easy-to-moderate but cloud-routed". The
  penalty now moves `difficulty` too (0.45 × constraint load), so the same task reports
  `difficulty 0.58` and the reason "hard task … -> strongest cloud model".
- **A hard lock selected `cloud-cheap` for moderate-difficulty locked classes.** `architecture`
  at difficulty 0.48 was locked away from local and then handed to the budget cloud model.
  Hard-locked classes now always take `cloud-strong` (`pickCloudTier()`).
- **Three classifier patterns hard-locked ordinary work.** `escalation` (any retry/backoff
  discussion) pulled tasks into `security`; bare `uuid` pulled them into `migration`; and
  `hasSecurityLanguage` matched a bare `token`, so "add a null check before using the token"
  was one noun away from a hard lock. All three are tightened, with regression tests.
- **A configured API key did not enable the cloud tier.** `resolveApiKeyFor()` read
  `cfg.dataDir` with `??`, and an empty string is not nullish, so while `loadConfig()` still
  had `dataDir` as `""` the secrets lookup resolved to `join("", "secrets.json")` — relative to
  the current working directory. The result: a user who followed the quickstart (set an API
  key, changed nothing else) silently had every task forced onto the local tier. `loadConfig()`
  now resolves `dataDir` before applying environment overrides, and the lookup uses `||`.
  There are regression tests for both.
- **Daily budgets used UTC day boundaries.** The cloud budget and the training budget are the
  user's daily budgets, so keying them off `toISOString().slice(0,10)` reset them mid-afternoon
  for anyone west of Greenwich. Both now use `localDayKey()`; episode *shard* filenames stay
  UTC because they exist for sortable storage, not accounting.

## 12. Config reference

| Key | Default | Effect |
| --- | --- | --- |
| `routing.mode` | `hybrid` | `heuristic` / `learned` / `hybrid` |
| `routing.qualityFloor` | `0.72` | floor with a verifier |
| `routing.qualityFloorUnverified` | `0.90` | floor for mutating, unverified work |
| `routing.qualityFloorReadOnly` | `0.55` | floor for read-only, unverified work |
| `routing.exploration.enabled` | `true` | master switch |
| `routing.exploration.epsilon` | `0.06` | probability of an exploratory local attempt |
| `routing.exploration.maxDifficultyForExploration` | `0.45` | difficulty ceiling |
| `routing.exploration.maxTokensForExploration` | `4000` | input-token ceiling |
| `routing.maxLocalRepairAttempts` | `1` | local repair rounds before escalation |
| `routing.maxCloudAttempts` | `2` | hard cap on cloud calls per task |
| `routing.cloudBudgetUsdPerDay` | `5` | daily budget; vetoes cloud at $0 remaining |
| `routing.latencyToleranceFactor` | `1.5` | local must be within 1.5× cloud latency |
| `routing.localColdStartMs` | `4000` | TTFT used when the model is not resident |
| `local.contextWindow` | `8192` | 80% of this is the local input budget |
| `local.maxOutputTokens` | `1536` | local output veto threshold |
| `local.tinyModel` | unset | enables the `local-tiny` tier |
| `cloud.cheapModel` | unset | `cloud-cheap` falls back to `cloud.model` |
| `verify.enabled` | `true` | default verifier availability |
| `pricing.<model>` | `DEFAULT_PRICING` | USD/Mtok; unknown models get `{in:5,out:20}` |
