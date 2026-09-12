/**
 * Episode schema — the unit of RL data.
 *
 * An episode is everything that happened for one task: the features the router
 * saw, the decision it made, every attempt (local, repair, escalation), the
 * verification verdicts, the cost, and any explicit user feedback.
 *
 * Design constraints that shaped this:
 *
 *  1. **Append-only and self-describing.** Every record carries the schema
 *     version, the feature-vector version and the reward version, so a dataset
 *     built months later can tell which records are comparable instead of
 *     silently mixing incompatible labels. This is the single most common way
 *     RL data pipelines rot.
 *
 *  2. **Replayable offline.** `features` plus `decision.environment` are enough
 *     to re-run `decideRoute` against a counterfactual policy without any model
 *     call. That is what makes `proto replay` cheap enough to run on every
 *     commit.
 *
 *  3. **Redacted at rest.** Nothing here is written until it has been through
 *     the redactor, because this file is the thing most likely to be shared.
 *
 *  4. **The label is verification, not escalation.** `outcome.localSucceeded`
 *     is derived from whether the local candidate *passed verification*, not
 *     from whether we escalated. Exploration episodes therefore produce genuine
 *     counterfactual labels — the thing the router most needs.
 */

import type { FinishReason } from '../providers/types.ts';
import type { RouteDecision, TaskFeatures, Tier } from '../router/types.ts';
import type { Price } from '../config/schema.ts';

export const EPISODE_SCHEMA_VERSION = 1;
export const REWARD_VERSION = 1;

export type AttemptSource = 'initial' | 'repair' | 'escalation';

export interface VerificationSummary {
  passed: boolean;
  score: number;
  blockers: string[];
  /** Check ids that failed, for aggregate failure-mode analysis. */
  failedChecks: string[];
  durationMs: number;
}

export interface AttemptRecord {
  /** 1-based index within the episode. */
  n: number;
  tier: Tier;
  source: AttemptSource;
  providerId: string;
  model: string;
  /** Redacted prompt, present only when memory.storePrompts is on. */
  prompt?: string;
  promptHash: string;
  promptTokens: number;
  outputTokens: number;
  cachedInputTokens?: number;
  costUsd: number;
  latencyMs: number;
  finishReason: FinishReason;
  /** Redacted candidate text (truncated), or omitted when storeTaskText is off. */
  output?: string;
  outputHash: string;
  /** Verification verdict; null when verification was skipped entirely. */
  verification: VerificationSummary | null;
  error?: string;
}

export type OutcomeStatus =
  | 'local-success'
  | 'escalated-cloud-success'
  | 'cloud-success'
  | 'failed'
  | 'abandoned';

export interface EpisodeOutcome {
  status: OutcomeStatus;
  finalTier: Tier;
  escalated: boolean;
  /**
   * The router training label:
   *   1  = local attempt passed verification
   *   0  = local attempt failed verification
   *   null = local was never tried under conditions where we could observe it
   */
  localSucceeded: boolean | null;
  totalCostUsd: number;
  totalLatencyMs: number;
  reward: number;
  rewardVersion: number;
  /**
   * Propensity `π(a|x)`: the probability that the behaviour policy would take
   * the action it actually took. Stored raw because it is interpretable and
   * auditable; the *training weight* is its inverse, computed by `ipsWeight()`
   * at dataset-build time. See the note on `ipsWeight` for why the direction
   * matters.
   */
  behaviorPropensity: number;
}

export interface EpisodeEnvironment {
  localAvailable: boolean;
  localModel: string;
  localContextWindow: number;
  /** Whether the local model was already resident, which drives the cold-start estimate. */
  localModelLoaded: boolean;
  cloudAvailable: boolean;
  cloudModel: string;
  cloudProvider: string;
  cloudPrice: Price;
  configuredQualityFloor: number;
  verifierAvailable: boolean;
  routingMode: 'heuristic' | 'learned' | 'hybrid';
  explorationEpsilon: number;
}

export interface EpisodeDecision {
  tier: Tier;
  reason: string;
  reasons: string[];
  pLocalSuccess: number;
  difficulty: number;
  taskClass: string;
  exploration: boolean;
  forced: boolean;
  unverified: boolean;
  scorer: 'heuristic' | 'learned' | 'hybrid';
  expectedLocalCostUsd: number;
  expectedCloudCostUsd: number;
  expectedLocalLatencyMs: number;
  expectedCloudLatencyMs: number;
  vetoes: string[];
}

export interface RedactionReport {
  applied: boolean;
  counts: Record<string, number>;
  /** Total characters removed by redaction. */
  charsRemoved: number;
}

export interface Episode {
  /** Monotonic ULID; sorts by creation time. */
  id: string;
  schemaVersion: number;
  ts: string;
  harnessVersion: string;
  platform: string;

  /** Redacted task text; absent when memory.storeTaskText is false. */
  task?: string;
  taskHash: string;
  /**
   * The shared system preamble used for this episode. Stored once per episode
   * (not per attempt) because it is identical across attempts; combined with
   * `attempt.prompt` it makes every training example exactly reconstructable.
   */
  systemPrompt?: string;
  systemPromptHash: string;
  features: TaskFeatures;
  vector: number[];
  vectorVersion: number;

  decision: EpisodeDecision;
  environment: EpisodeEnvironment;
  attempts: AttemptRecord[];

  outcome: EpisodeOutcome;

  feedback?: {
    signal: 'accept' | 'reject' | 'edit';
    note?: string;
    ts: string;
  };

  /** Consent state captured at write time, so later consent changes are auditable. */
  consent: {
    localTraining: boolean;
    globalShare: boolean;
  };

  redaction: RedactionReport;
  tags?: string[];
}

/** Build the `EpisodeDecision`/`EpisodeEnvironment` slices from a RouteDecision. */
export function decisionFromRoute(d: RouteDecision): EpisodeDecision {
  return {
    tier: d.tier,
    reason: d.reason,
    reasons: d.reasons,
    pLocalSuccess: d.pLocalSuccess,
    difficulty: d.difficulty,
    taskClass: d.taskClass,
    exploration: d.exploration,
    forced: d.forced,
    unverified: d.unverified,
    scorer: d.scorer,
    expectedLocalCostUsd: d.expected.localCostUsd,
    expectedCloudCostUsd: d.expected.cloudCostUsd,
    expectedLocalLatencyMs: d.expected.localLatencyMs,
    expectedCloudLatencyMs: d.expected.cloudLatencyMs,
    vetoes: d.vetoes,
  };
}
