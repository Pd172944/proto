/**
 * The routing policy: turn features + environment into a tier decision.
 *
 * The decision rule is an expected-cost comparison with a **quality floor**,
 * and the distinction matters:
 *
 *   Pure cost minimisation is degenerate here. Local inference costs ~$0, so
 *   "try local first, escalate on verification failure" almost always wins on
 *   paper — even when local succeeds 30% of the time, because the failed 70%
 *   only costs a bit of latency. That is the right policy for *batch* work and
 *   the wrong policy for an interactive coding session, where the user is
 *   waiting and a wrong answer wastes their attention.
 *
 *   So we require `p(local succeeds) >= floor` before spending the user's time
 *   on a local attempt, where `floor` rises sharply when there is no automatic
 *   verifier to catch a bad answer. Escalation then only handles the residual
 *   (~1 - p) cases rather than being the primary mechanism.
 *
 * Two further guards exist because they encode things a probability cannot:
 *  - **Hard-locked classes** (`security`, `architecture`, `migration`) go to the
 *    cloud regardless of the score. A silent auth bug or a wrong migration
 *    plan costs far more than the API call, and neither is reliably catchable
 *    by the verifier.
 *  - **Exploration** deliberately overrides the decision for a small fraction of
 *    low-risk tasks *only when a verifier exists*. Without it the router never
 *    observes what local would have done on tasks it routes away, and the
 *    learned scorer can never be debiased. This is the exploration/exploitation
 *    trade-off, made explicit and bounded.
 */

import type { ProtoConfig, Price } from '../config/schema.ts';
import { heuristicScore, blendWithPrior } from './heuristic.ts';
import { toVector } from './features.ts';
import type { LogisticScorer } from './learned.ts';
import type { RouteDecision, RouterEnvironmentSnapshot, TaskClass, TaskContext, TaskFeatures, Tier } from './types.ts';

/** Classes where a wrong-but-plausible answer is too expensive to risk locally. */
export const HARD_LOCKED_CLASSES: TaskClass[] = ['security', 'architecture', 'migration'];

/**
 * Difficulty above which the cloud tier chosen is `cloud-strong` rather than
 * `cloud-cheap`. Single source of truth: the harness previously hard-coded a
 * different value (0.4) for escalation, which meant a task in the [0.4, 0.5)
 * band was *routed* cheap and *escalated* strong with no explanation. The
 * harness now has its own, explicitly-named `ESCALATION_STRONG_THRESHOLD`
 * because a failed local attempt is genuine evidence of extra difficulty —
 * but the two are named constants with documented intent, not stray literals.
 */
export const CLOUD_STRONG_DIFFICULTY_THRESHOLD = 0.5;

export interface RouterEnvironment {
  localEnabled: boolean;
  /** Local runtime reachable AND the configured model present. */
  localAvailable: boolean;
  localAvailabilityNote?: string;
  /** True when the runtime reports the model already resident (no cold start). */
  localModelLoaded?: boolean;
  localContextWindow: number;
  /** Measured or estimated local decode speed. */
  localTokensPerSec?: number;
  tinyModelAvailable: boolean;
  cloudAvailable: boolean;
  cloudUnavailableReason?: string;
  /** Whether an automatic verifier will run on the candidate. */
  verifierAvailable: boolean;
  /** Remaining cloud spend for today, USD. */
  cloudBudgetRemainingUsd: number;
  price: Price;
  /** Injectable randomness so tests are deterministic. */
  random?: () => number;
}

export interface RouterInputs {
  ctx: TaskContext;
  features: TaskFeatures;
  cfg: ProtoConfig;
  env: RouterEnvironment;
  scorer?: LogisticScorer | null;
}

/** Rough decode speed by parameter count on Apple Silicon, tokens/sec. */
export function estimateLocalTokensPerSec(model: string): number {
  const m = model.toLowerCase();
  const sizeMatch = m.match(/(\d+(?:\.\d+)?)\s*b\b/);
  const params = sizeMatch?.[1] ? Number(sizeMatch[1]) : null;
  if (params !== null && params > 0) {
    // Calibrated against Qwen2.5-Coder on M-series unified memory:
    // ~160 tok/s for 1.5B q4, ~35 tok/s for 7B q4, ~14 tok/s for 14B q4.
    // Decode is memory-bandwidth bound, so we scale inversely with parameter
    // count. Nearly every local setup runs a 4-bit quantisation (Ollama's
    // default, MLX 4bit, GGUF q4_K_M), so quantised is the default assumption;
    // only an explicit full-precision tag (fp16/bf16/q8) pays the penalty.
    const fullPrecision = /f16|fp16|bf16|float16|q8|8bit|8-bit/.test(m);
    const base = 240;
    return Math.max(3, Math.round((base / params) * (fullPrecision ? 0.55 : 1)));
  }
  return 45;
}

export function decideRoute(input: RouterInputs): RouteDecision {
  const { ctx, features, cfg, env } = input;
  const vector = toVector(features);
  const random = env.random ?? Math.random;

  const heur = heuristicScore(features);
  const pHeuristic = heur.pLocalSuccess;
  const difficulty = heur.difficulty;

  // --- probability estimate ------------------------------------------------
  let pLocal = pHeuristic;
  let scorerUsed: RouteDecision['scorer'] = 'heuristic';
  const reasons: string[] = [];
  const vetoes: string[] = [];

  const trained = input.scorer && input.scorer.sampleCount > 0 ? input.scorer : null;
  if (cfg.routing.mode !== 'heuristic' && trained) {
    const pLearned = trained.predict(vector);
    if (cfg.routing.mode === 'learned') {
      pLocal = pLearned;
      scorerUsed = 'learned';
      reasons.push(
        `learned scorer: p(local)=${pLearned.toFixed(3)} from ${trained.sampleCount} logged episodes`,
      );
    } else {
      const blended = blendWithPrior(pLearned, pHeuristic, trained.sampleCount);
      pLocal = blended.p;
      scorerUsed = 'hybrid';
      reasons.push(
        `hybrid scorer: learned ${pLearned.toFixed(3)} blended with heuristic ${pHeuristic.toFixed(3)} ` +
          `(learned weight ${blended.learnedWeight.toFixed(2)} at n=${trained.sampleCount})`,
      );
    }
  } else {
    reasons.push(`heuristic scorer: p(local)=${pHeuristic.toFixed(3)} (no usable learned weights yet)`);
  }
  if (cfg.routing.mode === 'learned' && !trained) {
    reasons.push('routing.mode=learned but fewer than the minimum labelled episodes exist; falling back to heuristic');
  }

  reasons.push(`difficulty ${difficulty.toFixed(3)} for class "${features.taskClass}"`);

  // --- cost & latency -------------------------------------------------------
  const price = env.price;
  const cloudCostUsd =
    (features.estInputTokens / 1_000_000) * price.in + (features.estOutputTokens / 1_000_000) * price.out;
  const localElectricityUsd = 0; // excluded on purpose; see docs/routing.md
  const tokensPerSec = env.localTokensPerSec ?? estimateLocalTokensPerSec(cfg.local.model);
  const localTtft = env.localModelLoaded ? 250 : cfg.routing.localColdStartMs;
  const localLatencyMs = Math.round(localTtft + (features.estOutputTokens / tokensPerSec) * 1000);
  const cloudTtft = cfg.cloud.effort === 'high' ? 2500 : cfg.cloud.effort === 'medium' ? 1500 : 900;
  const cloudDecodeRate = cfg.cloud.effort === 'high' ? 30 : 60;
  const cloudLatencyMs = Math.round(cloudTtft + (features.estOutputTokens / cloudDecodeRate) * 1000);

  // --- eligibility ----------------------------------------------------------
  // Kept as two explicit lists (rather than string sniffing over one) so that a
  // veto message can never accidentally move a task between tiers.
  const localVetoes: string[] = [];
  const cloudVetoes: string[] = [];

  if (!env.localEnabled) localVetoes.push('local tier is disabled in config');
  else if (!env.localAvailable) {
    localVetoes.push(`local runtime unavailable${env.localAvailabilityNote ? `: ${env.localAvailabilityNote}` : ''}`);
  }
  if (features.estInputTokens > env.localContextWindow * 0.8) {
    localVetoes.push(
      `estimated ${features.estInputTokens} input tokens exceeds local context budget ` +
        `(${Math.floor(env.localContextWindow * 0.8)} of ${env.localContextWindow})`,
    );
  }
  if (features.estOutputTokens > cfg.local.maxOutputTokens) {
    localVetoes.push(
      `estimated ${features.estOutputTokens} output tokens exceeds local maxOutputTokens ${cfg.local.maxOutputTokens}`,
    );
  }
  // Hard-lock rule 1: categorically hard classes. A plausible-but-wrong
  // architecture decision or migration plan is expensive and the verifier cannot
  // detect it, so no probability score is allowed to override this.
  const classLocked = HARD_LOCKED_CLASSES.includes(features.taskClass) && !features.isExplainOnly;
  // Hard-lock rule 2: security-flavoured edits, even if the classifier landed on
  // a different class (e.g. "harden the check and add tests" reads as write-tests).
  const securityLocked = features.hasSecurityLanguage && !features.isExplainOnly && difficulty >= 0.3;
  if (classLocked || securityLocked) {
    localVetoes.push(
      `"${features.taskClass}"${securityLocked && !classLocked ? ' with security wording' : ''} is hard-locked to the ` +
        `cloud: a wrong answer here is expensive and the verifier cannot catch it`,
    );
  }
  if (!env.cloudAvailable) {
    cloudVetoes.push(`cloud tier unavailable${env.cloudUnavailableReason ? `: ${env.cloudUnavailableReason}` : ''}`);
  }
  if (env.cloudBudgetRemainingUsd <= 0) {
    cloudVetoes.push(`daily cloud budget exhausted ($${cfg.routing.cloudBudgetUsdPerDay.toFixed(2)})`);
  }
  vetoes.push(...localVetoes, ...cloudVetoes);

  const localVetoed = localVetoes.length > 0;
  const cloudVetoed = cloudVetoes.length > 0;

  /**
   * Which cloud tier to use when the cloud is chosen.
   *
   * A hard-locked class always takes the strong model. The lock exists because a
   * plausible-but-wrong answer there is expensive, and then handing that same
   * task to the cheapest cloud model would contradict the reason it was locked —
   * "this is too risky for the local model" cannot mean "so let us use the budget
   * model". For everything else, difficulty decides.
   */
  const pickCloudTier = (): Tier =>
    classLocked || securityLocked || difficulty >= CLOUD_STRONG_DIFFICULTY_THRESHOLD
      ? 'cloud-strong'
      : 'cloud-cheap';

  // --- decision -------------------------------------------------------------
  // The quality floor depends on how bad a wrong answer would be, not just on
  // whether we can check it:
  //   verified      -> the verifier catches hard failures, so a moderate floor is fine
  //   unverified            -> mutating output that nothing checks: demand near-certainty
  //   read-only & unverified -> a wrong explanation wastes a few seconds of reading
  const floor = env.verifierAvailable
    ? cfg.routing.qualityFloor
    : features.isExplainOnly
      ? cfg.routing.qualityFloorReadOnly
      : cfg.routing.qualityFloorUnverified;
  if (!env.verifierAvailable) {
    reasons.push(
      features.isExplainOnly
        ? `read-only task with no verifier: floor stays low (${cfg.routing.qualityFloorReadOnly.toFixed(2)}) because a wrong answer is self-evident to the reader`
        : `mutating task with no automatic verifier: quality floor raised to ${cfg.routing.qualityFloorUnverified.toFixed(2)}`,
    );
  }

  let tier: Tier;
  let forced = false;
  let exploration = false;

  const localWinsOnQuality = pLocal >= floor;
  const localWinsOnLatency = localLatencyMs <= cloudLatencyMs * cfg.routing.latencyToleranceFactor;

  if (localVetoed) {
    if (cloudVetoed) {
      // Both tiers are constrained. Local is the only thing that can run, and
      // the user is better served by an attempt plus an explicit warning than by
      // a refusal.
      tier = 'local';
      forced = true;
      reasons.push('both tiers are constrained; attempting local rather than refusing the task');
    } else {
      tier = pickCloudTier();
      reasons.push(`local not eligible: ${localVetoes.join('; ')}`);
    }
  } else if (cloudVetoed) {
    tier = env.tinyModelAvailable && difficulty < 0.2 ? 'local-tiny' : 'local';
    forced = true;
    reasons.push(`forced local: ${cloudVetoes.join('; ')}`);
  } else if (localWinsOnQuality && localWinsOnLatency) {
    tier = env.tinyModelAvailable && difficulty < 0.2 && pLocal >= floor ? 'local-tiny' : 'local';
    reasons.push(
      `p(local)=${pLocal.toFixed(3)} >= floor ${floor.toFixed(2)} and estimated local latency ` +
        `${localLatencyMs}ms is within ${cfg.routing.latencyToleranceFactor}x cloud (${cloudLatencyMs}ms)`,
    );
  } else if (localWinsOnQuality && !localWinsOnLatency) {
    tier = pickCloudTier();
    reasons.push(
      `local is accurate enough (${pLocal.toFixed(3)} >= ${floor.toFixed(2)}) but estimated slower ` +
        `(${localLatencyMs}ms vs ${cloudLatencyMs}ms, tolerance ${cfg.routing.latencyToleranceFactor}x)`,
    );
  } else {
    tier = pickCloudTier();
    reasons.push(
      `p(local)=${pLocal.toFixed(3)} is below the quality floor ${floor.toFixed(2)} for "${features.taskClass}"`,
    );
  }

  // --- exploration override -------------------------------------------------
  if (
    cfg.routing.exploration.enabled &&
    env.verifierAvailable &&
    tier !== 'local' &&
    tier !== 'local-tiny' &&
    !localVetoed &&
    !forced &&
    difficulty <= cfg.routing.exploration.maxDifficultyForExploration &&
    features.estInputTokens <= cfg.routing.exploration.maxTokensForExploration &&
    random() < cfg.routing.exploration.epsilon
  ) {
    tier = 'local';
    exploration = true;
    reasons.push(
      `exploration: sending this task local (epsilon=${cfg.routing.exploration.epsilon}) to observe the ` +
        `counterfactual; a verifier will catch failure`,
    );
  }

  const expected = {
    localCostUsd: localElectricityUsd,
    cloudCostUsd: round6(cloudCostUsd),
    localLatencyMs,
    cloudLatencyMs,
  };

  const reason = buildReason(tier, pLocal, difficulty, features, { exploration, forced });

  const envSnapshot: RouterEnvironmentSnapshot = {
    localEnabled: env.localEnabled,
    localAvailable: env.localAvailable,
    ...(env.localAvailabilityNote ? { localAvailabilityNote: env.localAvailabilityNote } : {}),
    localModelLoaded: env.localModelLoaded ?? false,
    localContextWindow: env.localContextWindow,
    localTokensPerSec: tokensPerSec,
    tinyModelAvailable: env.tinyModelAvailable,
    cloudAvailable: env.cloudAvailable,
    ...(env.cloudUnavailableReason ? { cloudUnavailableReason: env.cloudUnavailableReason } : {}),
    verifierAvailable: env.verifierAvailable,
    cloudBudgetRemainingUsd: env.cloudBudgetRemainingUsd,
  };

  return {
    tier,
    reason,
    reasons,
    pLocalSuccess: pLocal,
    difficulty,
    taskClass: features.taskClass,
    expected,
    exploration,
    vetoes,
    unverified: !env.verifierAvailable,
    features,
    vector,
    vectorVersion: 1,
    scorer: scorerUsed,
    forced,
    env: envSnapshot,
  };
}

function buildReason(
  tier: Tier,
  p: number,
  difficulty: number,
  f: TaskFeatures,
  flags: { exploration: boolean; forced: boolean },
): string {
  const pct = `${Math.round(p * 100)}%`;
  const scope = f.fileCount === 0 ? 'no files in scope' : `${f.fileCount} file(s)`;
  switch (tier) {
    case 'local-tiny':
      return `very easy local task (difficulty ${difficulty.toFixed(2)}, ${scope}) -> tiny local model [${pct} local prior]`;
    case 'local':
      return flags.exploration
        ? `exploration attempt on the local model (difficulty ${difficulty.toFixed(2)}, ${scope})`
        : flags.forced
          ? `local model only option (difficulty ${difficulty.toFixed(2)}, ${scope})`
          : `easy enough for the local model (difficulty ${difficulty.toFixed(2)}, ${scope}) [${pct} local prior]`;
    case 'cloud-cheap':
      return `easy-to-moderate but cloud-routed (difficulty ${difficulty.toFixed(2)}, ${scope}) -> cheaper cloud model`;
    case 'cloud-strong':
    default:
      return `hard task (difficulty ${difficulty.toFixed(2)}, class "${f.taskClass}", ${scope}) -> strongest cloud model`;
  }
}

function round6(n: number): number {
  return Math.round(n * 1e6) / 1e6;
}

/** Utility used by the eval harness to score a decision after the fact. */
export function decisionUtility(
  decision: RouteDecision,
  outcome: { localSucceeded?: boolean; escalated?: boolean },
): number {
  // Negative estimated USD plus a latency penalty expressed in "cents per second
  // of user time". The exact exchange rate is a policy choice; it is centralised
  // here so eval reports are comparable across runs.
  const USER_SECOND_VALUE_USD = 0.01;
  let cost = 0;
  if (decision.tier === 'local' || decision.tier === 'local-tiny') {
    cost += (decision.expected.localLatencyMs / 1000) * USER_SECOND_VALUE_USD;
    if (outcome.localSucceeded === false) cost += (decision.expected.cloudLatencyMs / 1000) * USER_SECOND_VALUE_USD;
  } else {
    cost += decision.expected.cloudCostUsd + (decision.expected.cloudLatencyMs / 1000) * USER_SECOND_VALUE_USD;
  }
  if (outcome.escalated) cost += 0.0005;
  return -cost;
}
