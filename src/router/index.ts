/**
 * Router entry point: assemble the environment, then apply the policy.
 *
 * Kept separate from `policy.ts` so that the policy stays a pure function of
 * (features, environment, config) — which is what makes offline counterfactual
 * replay possible: a logged episode stores the features and the environment it
 * saw, so `proto replay` can re-run `decideRoute` against historical decisions
 * without touching a model or a network.
 */

import { join } from 'node:path';

import { extractFeatures } from './features.ts';
import { LogisticScorer } from './learned.ts';
import { decideRoute, estimateLocalTokensPerSec } from './policy.ts';
import type { RouterEnvironment } from './policy.ts';
import type { RouteDecision, TaskContext } from './types.ts';
import type { ProtoConfig } from '../config/schema.ts';
import { priceFor, resolveApiKeyFor } from '../config/load.ts';
import { cloudTierModel } from '../providers/index.ts';
import { readJsonOrNull, writeJsonAtomic } from '../util/fsx.ts';

export function routerWeightsPath(dataDir: string): string {
  return join(dataDir, 'router', 'weights.json');
}

/** Load trained weights, tolerating absence and version drift. */
export function loadScorer(dataDir: string): { scorer: LogisticScorer | null; error?: string } {
  const file = readJsonOrNull<Parameters<typeof LogisticScorer.fromFile>[0]>(routerWeightsPath(dataDir));
  if (!file) return { scorer: null };
  try {
    return { scorer: LogisticScorer.fromFile(file) };
  } catch (err) {
    return { scorer: null, error: err instanceof Error ? err.message : String(err) };
  }
}

export function saveScorer(dataDir: string, scorer: LogisticScorer, notes: string[] = []): string {
  const path = routerWeightsPath(dataDir);
  writeJsonAtomic(path, scorer.toFile(notes));
  return path;
}

export interface RouteOptions {
  cfg: ProtoConfig;
  dataDir: string;
  ctx: TaskContext;
  /**
   * Skip all health probes. Used by the eval harness and by `proto route`,
   * where we want a pure, instant decision.
   */
  offline?: boolean;
  /** Local runtime facts, when the caller already knows them. */
  local?: { available?: boolean; note?: string; modelLoaded?: boolean; models?: string[] };
  /** Cloud spend already incurred today, so the budget veto is accurate. */
  cloudSpendTodayUsd?: number;
  /** Whether verification will run on the candidate. */
  verifierAvailable?: boolean;
  random?: () => number;
  /** Override the local decode-speed estimate (e.g. measured from episodes). */
  localTokensPerSec?: number;
}

/**
 * Decide the tier for a task.
 *
 * Note on cloud availability: we deliberately do NOT probe the cloud with a
 * network call here. A health check costs money on some providers (Anthropic
 * has no free ping) and adds latency to every routing decision. Availability is
 * inferred from config + key presence; real failures surface as escalation
 * errors with an actionable message.
 */
export async function routeTask(opts: RouteOptions): Promise<RouteDecision> {
  const { cfg, dataDir, ctx } = opts;
  const features = extractFeatures(ctx);

  // --- local side ----------------------------------------------------------
  // "offline" means "do not perform I/O", NOT "the local model is broken".
  // Assuming unavailability there would silently mis-route every task, so when a
  // probe is skipped we optimistically treat the configured tier as usable and
  // let the caller override via `opts.local`.
  let localAvailable = opts.local?.available ?? (opts.offline ? cfg.local.enabled : false);
  let localNote = opts.local?.note;
  let localModelLoaded = opts.local?.modelLoaded ?? opts.offline === true;

  if (!opts.offline && opts.local === undefined) {
    const { buildLocalProvider } = await import('../providers/index.ts');
    const provider = buildLocalProvider(cfg);
    const health = await provider.health();
    localAvailable = health.ok;
    localNote = health.ok ? undefined : health.detail;
    const models = health.models ?? [];
    localModelLoaded = models.some((m) => m.startsWith(cfg.local.model));
  }

  const tinyModelAvailable = Boolean(cfg.local.tinyModel && cfg.local.tinyModel.trim());

  // --- cloud side ----------------------------------------------------------
  const hasKey = Boolean(resolveApiKeyFor(cfg, dataDir));
  const cloudAvailable = cfg.cloud.enabled && hasKey;
  const cloudUnavailableReason = !cfg.cloud.enabled
    ? 'cloud.enabled is false'
    : !hasKey
      ? `no API key for provider "${cfg.cloud.provider}"`
      : undefined;

  const budget = cfg.routing.cloudBudgetUsdPerDay;
  const spent = opts.cloudSpendTodayUsd ?? 0;

  // Price the tier the decision will actually use: a cheap-tier task priced at
  // the strong model's rate would bias the router away from the cloud.
  const cloudModel = cloudTierModel(cfg, 'cloud-cheap');
  const price = priceFor(cfg, cloudModel);

  const env: RouterEnvironment = {
    localEnabled: cfg.local.enabled,
    localAvailable,
    ...(localNote ? { localAvailabilityNote: localNote } : {}),
    localModelLoaded,
    localContextWindow: cfg.local.contextWindow,
    localTokensPerSec: opts.localTokensPerSec ?? estimateLocalTokensPerSec(cfg.local.model),
    tinyModelAvailable,
    cloudAvailable,
    ...(cloudUnavailableReason ? { cloudUnavailableReason } : {}),
    verifierAvailable: opts.verifierAvailable ?? cfg.verify.enabled,
    cloudBudgetRemainingUsd: Math.max(0, budget - spent),
    price,
    ...(opts.random ? { random: opts.random } : {}),
  };

  const { scorer, error: scorerError } = loadScorer(dataDir);

  const decision = decideRoute({
    ctx,
    features,
    cfg,
    env,
    scorer,
  });

  if (scorerError) {
    return {
      ...decision,
      scorerError,
      reasons: [
        `trained router weights were ignored: ${scorerError}`,
        ...decision.reasons,
      ],
    };
  }
  return decision;
}

export * from './types.ts';
export { extractFeatures, classifyTask, analyzeCode, toVector } from './features.ts';
export { FEATURE_NAMES, FEATURE_VECTOR_VERSION, TIERS, tierRank, isLocalTier } from './types.ts';
export { heuristicScore, sigmoid } from './heuristic.ts';
export { LogisticScorer, trainLogistic, evaluate, featureImportance, MIN_TRAINING_SAMPLES } from './learned.ts';
export {
  decideRoute,
  decisionUtility,
  HARD_LOCKED_CLASSES,
  CLOUD_STRONG_DIFFICULTY_THRESHOLD,
  estimateLocalTokensPerSec,
} from './policy.ts';
export type { RouterEnvironment } from './policy.ts';
