/**
 * Evaluation and offline counterfactual replay.
 *
 * Two distinct capabilities, both cheap enough to run constantly:
 *
 *  1. **Routing evaluation (`proto eval`).** Runs the corpus through the router
 *     *without calling any model*. Because routing is a pure function of
 *     features + environment, the whole corpus is scored in milliseconds. This
 *     is how routing changes get validated before they touch a real task.
 *
 *  2. **Counterfactual replay (`proto replay`).** Re-runs a *new* policy over
 *     historical episodes using the features and environment recorded at the
 *     time, and reports what would have changed. This is off-policy evaluation
 *     in the cheapest useful form: it needs no model calls, and it is honest
 *     about its main limitation (it can only re-score decisions; it cannot know
 *     whether local *would* have succeeded on tasks it never attempted).
 *
 * The replay caveat is important and is printed in the report rather than buried
 * here: episodes where local was never attempted have no observed local outcome,
 * so "would have been cheaper" is an estimate from the model's predicted
 * probability, not a measurement.
 */

import { EVAL_TASKS } from './tasks.ts';
import type { EvalTask } from './tasks.ts';
import type { ProtoConfig } from '../config/schema.ts';
import { EpisodeStore } from '../memory/store.ts';
import { decideRoute } from '../router/policy.ts';
import type { RouterEnvironment } from '../router/policy.ts';
import { loadScorer, routerWeightsPath } from '../router/index.ts';
import { extractFeatures } from '../router/features.ts';
import { decisionUtility } from '../router/policy.ts';
import type { RouteDecision, Tier } from '../router/types.ts';
import { priceFor } from '../config/load.ts';
import { cloudTierModel } from '../providers/index.ts';
import { logisticEvalFromEpisodes } from './metrics.ts';

const LOCAL_TIERS: Tier[] = ['local', 'local-tiny'];

export interface EvalRow {
  id: string;
  expected: 'local' | 'cloud';
  predictedTier: Tier;
  predicted: 'local' | 'cloud';
  correct: boolean;
  taskClass: string;
  expectedClass?: string;
  classCorrect: boolean | null;
  pLocalSuccess: number;
  difficulty: number;
  localLatencyMs: number;
  cloudLatencyMs: number;
  cloudCostUsd: number;
  verifiable: boolean;
  /** True when the task only reads code (explanation/summary) — self-checking by the reader. */
  readOnly: boolean;
  exploration: boolean;
  reason: string;
  /** `decisionUtility()` for this decision — see RoutingMetrics.meanUtility. */
  utility: number;
}

export interface RoutingMetrics {
  total: number;
  correct: number;
  accuracy: number;
  /** Confusion counts: [expected][predicted]. */
  confusion: Record<'local' | 'cloud', Record<'local' | 'cloud', number>>;
  localPrecision: number;
  localRecall: number;
  cloudPrecision: number;
  cloudRecall: number;
  classAccuracy: number | null;
  /** Sum of expected cloud cost across the corpus for the chosen policy. */
  estimatedCloudCostUsd: number;
  /** Cost if everything went to the cloud. */
  cloudOnlyCostUsd: number;
  /** Cost if everything went local (excluding escalation). */
  localOnlyCostUsd: number;
  /** Tasks sent local without a verifier **that also mutate code**: the risky category. */
  unverifiedLocal: number;
  /** Read-only tasks sent local without a verifier: low risk, reported separately. */
  unverifiedReadOnlyLocal: number;
  explorationCount: number;
  meanDifficulty: number;
  /**
   * Mean `decisionUtility()` across the corpus. Utility prices a second of the
   * user's time at $0.01 and adds estimated cloud spend, which makes it possible
   * to compare policies that trade money against latency — something plain
   * accuracy and cost cannot express.
   */
  meanUtility: number;
}

export interface EvalReport {
  mode: 'heuristic' | 'learned' | 'hybrid';
  scorerPresent: boolean;
  scorerSamples: number;
  rows: EvalRow[];
  metrics: RoutingMetrics;
  notes: string[];
}

export interface EvalOptions {
  /** Only score these task ids. */
  only?: string[];
  /** Assume local + cloud are both available (default) or restrict them. */
  localAvailable?: boolean;
  cloudAvailable?: boolean;
  /** Force a routing mode for this evaluation, overriding config. */
  mode?: 'heuristic' | 'learned' | 'hybrid';
  /** Deterministic exploration dice. */
  seed?: number;
}

/**
 * Score the corpus offline. No network, no local runtime, no tokens.
 *
 * Note that we build a synthetic-but-honest environment: the corpus is scored
 * as if the local runtime and the cloud key were both present, which is the
 * decision the router must get right in the common case. Availability handling
 * is covered by dedicated unit tests instead.
 */
export function evaluateRouting(cfg: ProtoConfig, dataDir: string, opts: EvalOptions = {}): EvalReport {
  const mode = opts.mode ?? cfg.routing.mode;
  // Exploration is a *data-gathering* behaviour, not a measure of routing
  // quality, so it is disabled here; otherwise the score would depend on the
  // dice and a lucky exploration would mask a bad policy.
  const effectiveCfg: ProtoConfig = {
    ...cfg,
    routing: { ...cfg.routing, mode, exploration: { ...cfg.routing.exploration, enabled: false } },
  };
  const { scorer } = loadScorer(dataDir);
  const seededRandom = makeRandom(opts.seed ?? 7);

  const tasks = opts.only?.length ? EVAL_TASKS.filter((t) => opts.only?.includes(t.id)) : EVAL_TASKS;
  const rows: EvalRow[] = [];

  for (const task of tasks) {
    const features = extractFeatures({
      task: task.task,
      ...(task.files ? { files: task.files } : {}),
      ...(task.diff ? { diff: task.diff } : {}),
      ...(task.constraints ? { constraints: task.constraints } : {}),
    });
    const env: RouterEnvironment = {
      localEnabled: true,
      localAvailable: opts.localAvailable ?? true,
      // Scored in "steady state": during a working session the local model is
      // already resident, so the cold-start penalty does not apply. Cold-start
      // behaviour is covered by the routing unit tests instead, because it
      // legitimately flips very small tasks to the cloud on the first call.
      localModelLoaded: true,
      localContextWindow: cfg.local.contextWindow,
      localTokensPerSec: undefined,
      tinyModelAvailable: Boolean(cfg.local.tinyModel),
      cloudAvailable: opts.cloudAvailable ?? true,
      verifierAvailable: task.verifiable,
      cloudBudgetRemainingUsd: cfg.routing.cloudBudgetUsdPerDay,
      // Price the tier the router will actually reach for, from the user's own
      // pricing table, rather than a hard-coded rate that ignores their provider.
      price: priceFor(cfg, cloudTierModel(cfg, 'cloud-cheap')),
      random: seededRandom,
    };

    const decision = decideRoute({
      ctx: { task: task.task, files: task.files ?? [] },
      features,
      cfg: effectiveCfg,
      env,
      scorer,
    });

    const predicted: 'local' | 'cloud' = LOCAL_TIERS.includes(decision.tier) ? 'local' : 'cloud';
    rows.push({
      id: task.id,
      expected: task.expected,
      predictedTier: decision.tier,
      predicted,
      correct: predicted === task.expected,
      taskClass: decision.taskClass,
      ...(task.expectedClass ? { expectedClass: task.expectedClass } : {}),
      classCorrect: task.expectedClass ? decision.taskClass === task.expectedClass : null,
      pLocalSuccess: round4(decision.pLocalSuccess),
      difficulty: round4(decision.difficulty),
      localLatencyMs: decision.expected.localLatencyMs,
      cloudLatencyMs: decision.expected.cloudLatencyMs,
      cloudCostUsd: decision.expected.cloudCostUsd,
      verifiable: task.verifiable,
      readOnly: decision.features.isExplainOnly,
      exploration: decision.exploration,
      reason: decision.reason,
      utility: round6(decisionUtility(decision, { localSucceeded: task.expected === 'local' })),
    });
  }

  return {
    mode,
    scorerPresent: Boolean(scorer),
    scorerSamples: scorer?.sampleCount ?? 0,
    rows,
    metrics: computeMetrics(rows),
    notes: [
      'Scored offline: routing is a pure function of features + environment, so no model was called.',
      'Tasks marked verifiable=false are scored with a raised quality floor, as they would be in production.',
      'Exploration is disabled for scoring so the result does not depend on the dice.',
      'Steady state is assumed (the local model is already resident), so cold-start penalties do not distort the score.',
      ...(scorer ? [] : ['No trained weights found; the learned mode will fall back to the heuristic.']),
    ],
  };
}

export function computeMetrics(rows: EvalRow[]): RoutingMetrics {
  const confusion: RoutingMetrics['confusion'] = {
    local: { local: 0, cloud: 0 },
    cloud: { local: 0, cloud: 0 },
  };
  let correct = 0;
  let classCorrect = 0;
  let classTotal = 0;
  let estimatedCloudCostUsd = 0;
  let cloudOnlyCostUsd = 0;
  let utilitySum = 0;
  let unverifiedLocal = 0;
  let unverifiedReadOnlyLocal = 0;
  let explorationCount = 0;
  let difficultySum = 0;

  for (const row of rows) {
    confusion[row.expected][row.predicted]++;
    if (row.correct) correct++;
    if (row.classCorrect !== null) {
      classTotal++;
      if (row.classCorrect) classCorrect++;
    }
    cloudOnlyCostUsd += row.cloudCostUsd;
    if (row.predicted === 'cloud') estimatedCloudCostUsd += row.cloudCostUsd;
    if (row.predicted === 'local' && !row.verifiable) {
      // Only a *mutating* unverified task is a risk: a wrong explanation is
      // self-evident to the person reading it.
      if (row.readOnly) unverifiedReadOnlyLocal++;
      else unverifiedLocal++;
    }
    if (row.exploration) explorationCount++;
    difficultySum += row.difficulty;
    utilitySum += row.utility;
  }

  const localTp = confusion.local.local;
  const localFp = confusion.cloud.local;
  const localFn = confusion.local.cloud;
  const cloudTp = confusion.cloud.cloud;
  const cloudFp = confusion.local.cloud;
  const cloudFn = confusion.cloud.local;

  return {
    total: rows.length,
    correct,
    accuracy: rows.length ? correct / rows.length : 0,
    confusion,
    localPrecision: localTp + localFp > 0 ? localTp / (localTp + localFp) : 0,
    localRecall: localTp + localFn > 0 ? localTp / (localTp + localFn) : 0,
    cloudPrecision: cloudTp + cloudFp > 0 ? cloudTp / (cloudTp + cloudFp) : 0,
    cloudRecall: cloudTp + cloudFn > 0 ? cloudTp / (cloudTp + cloudFn) : 0,
    classAccuracy: classTotal ? classCorrect / classTotal : null,
    estimatedCloudCostUsd: round6(estimatedCloudCostUsd),
    cloudOnlyCostUsd: round6(cloudOnlyCostUsd),
    localOnlyCostUsd: 0,
    unverifiedLocal,
    unverifiedReadOnlyLocal,
    explorationCount,
    meanDifficulty: rows.length ? round4(difficultySum / rows.length) : 0,
    meanUtility: rows.length ? round4(utilitySum / rows.length) : 0,
  };
}

/* ------------------------------------------------------------------ */
/* Replay                                                             */
/* ------------------------------------------------------------------ */

export interface ReplayRow {
  episodeId: string;
  ts: string;
  taskClass: string;
  oldTier: Tier;
  newTier: Tier;
  changed: boolean;
  /** local -> cloud migrations, i.e. the new policy is more cautious. */
  direction: 'same' | 'more-cautious' | 'more-local';
  oldPLocal: number;
  newPLocal: number;
  observedLocalSucceeded: boolean | null;
  /** True when the episode is exploration data: a genuine counterfactual. */
  counterfactual: boolean;
  cloudCostDeltaUsd: number;
}

export interface ReplayReport {
  episodes: number;
  changed: number;
  moreCautious: number;
  moreLocal: number;
  /** Of the episodes that changed direction, how many had an observed label. */
  changesWithObservedLabel: number;
  /** Sum of cost delta from route changes (negative = cheaper). */
  costDeltaUsd: number;
  rows: ReplayRow[];
  notes: string[];
  /** How well the *current* scorer predicts the observed labels, when available. */
  scorerQuality: ReturnType<typeof logisticEvalFromEpisodes>;
}

export interface ReplayOptions {
  /** Override the routing mode for the counterfactual policy. */
  mode?: 'heuristic' | 'learned' | 'hybrid';
  /** Override the quality floor. */
  qualityFloor?: number;
  /** Limit to the most recent N episodes. */
  limit?: number;
  /** Only include episodes newer than this. */
  since?: Date;
}

export function replay(
  cfg: ProtoConfig,
  dataDir: string,
  opts: ReplayOptions = {},
): ReplayReport {
  const store = new EpisodeStore(dataDir);
  const episodes = store.readAll({
    ...(opts.since ? { since: opts.since } : {}),
    ...(opts.limit ? { limit: opts.limit } : {}),
  });
  const { scorer } = loadScorer(dataDir);
  const effectiveCfg: ProtoConfig = {
    ...cfg,
    routing: {
      ...cfg.routing,
      mode: opts.mode ?? cfg.routing.mode,
      qualityFloor: opts.qualityFloor ?? cfg.routing.qualityFloor,
    },
  };

  const rows: ReplayRow[] = [];
  let changed = 0;
  let moreCautious = 0;
  let moreLocal = 0;
  let changesWithObservedLabel = 0;
  let costDeltaUsd = 0;

  for (const ep of episodes) {
    // Rebuild the environment from what was recorded, so replay cannot invent
    // availability that did not exist at the time.
    const env: RouterEnvironment = {
      localEnabled: true,
      localAvailable: ep.environment.localAvailable,
      // Recorded at decision time, so replay reproduces the cold-start penalty
      // the original decision actually saw instead of assuming a warm model.
      localModelLoaded: ep.environment.localModelLoaded,
      localContextWindow: ep.environment.localContextWindow,
      tinyModelAvailable: Boolean(cfg.local.tinyModel),
      cloudAvailable: ep.environment.cloudAvailable,
      verifierAvailable: ep.environment.verifierAvailable,
      cloudBudgetRemainingUsd: cfg.routing.cloudBudgetUsdPerDay,
      price: ep.environment.cloudPrice,
      random: () => 1, // never explore during replay: we want the deterministic policy
    };

    const decision = decideRoute({
      ctx: { task: '' },
      features: ep.features,
      cfg: effectiveCfg,
      env,
      scorer,
    });

    const oldTier = ep.decision.tier;
    const oldLocal = LOCAL_TIERS.includes(oldTier);
    const newLocal = LOCAL_TIERS.includes(decision.tier);
    const isChanged = oldTier !== decision.tier;
    const direction: ReplayRow['direction'] = !isChanged
      ? 'same'
      : oldLocal && !newLocal
        ? 'more-cautious'
        : !oldLocal && newLocal
          ? 'more-local'
          : 'same';

    if (isChanged) {
      changed++;
      if (direction === 'more-cautious') moreCautious++;
      if (direction === 'more-local') moreLocal++;
      if (ep.outcome.localSucceeded !== null) changesWithObservedLabel++;
    }

    // Cost delta: only the change in expected cloud spend from the top-level
    // routing decision. Escalation cost is excluded because replay cannot know
    // whether a new local attempt would have failed.
    const oldCost = oldLocal ? 0 : ep.decision.expectedCloudCostUsd;
    const newCost = newLocal ? 0 : decision.expected.cloudCostUsd;
    costDeltaUsd += newCost - oldCost;

    rows.push({
      episodeId: ep.id,
      ts: ep.ts,
      taskClass: ep.decision.taskClass,
      oldTier,
      newTier: decision.tier,
      changed: isChanged,
      direction,
      oldPLocal: round4(ep.decision.pLocalSuccess),
      newPLocal: round4(decision.pLocalSuccess),
      observedLocalSucceeded: ep.outcome.localSucceeded,
      counterfactual: ep.decision.exploration,
      cloudCostDeltaUsd: round6(newCost - oldCost),
    });
  }

  const notes: string[] = [
    'Replay re-scores historical decisions using the recorded features and tier availability; it makes no model calls.',
    'It cannot know whether local *would* have succeeded on episodes where local was never attempted. ' +
      'Route changes on unobserved episodes are hypotheses, not measurements.',
    `Exploration episodes carry genuine counterfactual labels: ${rows.filter((r) => r.counterfactual).length} in this window.`,
  ];
  if (scorer) {
    notes.push(`Counterfactual policy used learned weights trained on ${scorer.sampleCount} episodes.`);
  } else {
    notes.push('No learned weights found; replay used the heuristic policy.');
  }

  return {
    episodes: episodes.length,
    changed,
    moreCautious,
    moreLocal,
    changesWithObservedLabel,
    costDeltaUsd: round6(costDeltaUsd),
    rows,
    notes,
    scorerQuality: logisticEvalFromEpisodes(store, cfg),
  };
}

/* ------------------------------------------------------------------ */
/* Formatting                                                          */
/* ------------------------------------------------------------------ */

export function formatEvalReport(report: EvalReport): string[] {
  const m = report.metrics;
  const lines: string[] = [];
  const pct = (n: number): string => `${(n * 100).toFixed(1)}%`;

  lines.push(`routing mode: ${report.mode}${report.scorerPresent ? ` (weights from ${report.scorerSamples} episodes)` : ' (no trained weights)'}`);
  lines.push(`accuracy: ${pct(m.accuracy)} (${m.correct}/${m.total})`);
  lines.push(
    `confusion: expected-local -> predicted [local ${m.confusion.local.local}, cloud ${m.confusion.local.cloud}] | ` +
      `expected-cloud -> predicted [local ${m.confusion.cloud.local}, cloud ${m.confusion.cloud.cloud}]`,
  );
  lines.push(`local precision ${pct(m.localPrecision)} recall ${pct(m.localRecall)}`);
  lines.push(`cloud precision ${pct(m.cloudPrecision)} recall ${pct(m.cloudRecall)}`);
  if (m.classAccuracy !== null) lines.push(`task-class accuracy: ${pct(m.classAccuracy)}`);
  lines.push(`mean difficulty: ${m.meanDifficulty.toFixed(3)}`);
  lines.push(`mean decision utility: ${m.meanUtility.toFixed(5)} (negative = cost; user time priced at $0.01/s)`);
  lines.push(
    `estimated cloud spend for this policy: $${m.estimatedCloudCostUsd.toFixed(4)} ` +
      `(all-cloud would be $${m.cloudOnlyCostUsd.toFixed(4)})`,
  );
  if (m.unverifiedLocal > 0) {
    lines.push(
      `WARNING: ${m.unverifiedLocal} mutating task(s) routed local with no verifier — a wrong answer there cannot be caught`,
    );
  }
  if (m.unverifiedReadOnlyLocal > 0) {
    lines.push(
      `note: ${m.unverifiedReadOnlyLocal} read-only task(s) routed local with no verifier (intended: a wrong explanation is self-evident)`,
    );
  }
  if (m.explorationCount > 0) lines.push(`exploration fired on ${m.explorationCount} task(s) in this scoring pass`);
  lines.push('');
  lines.push('per-task:');
  for (const row of report.rows) {
    const mark = row.correct ? 'ok  ' : 'MISS';
    lines.push(
      `  ${mark} ${row.id.padEnd(28)} expected=${row.expected.padEnd(5)} got=${row.predictedTier.padEnd(12)} ` +
        `p=${row.pLocalSuccess.toFixed(2)} d=${row.difficulty.toFixed(2)}${row.verifiable ? '' : ' [unverifiable]'}`,
    );
  }
  return lines;
}

export function formatReplayReport(report: ReplayReport): string[] {
  const lines: string[] = [];
  lines.push(`episodes replayed: ${report.episodes}`);
  lines.push(`decisions that would change: ${report.changed} (more cautious ${report.moreCautious}, more local ${report.moreLocal})`);
  lines.push(`of those, with an observed local outcome: ${report.changesWithObservedLabel}`);
  lines.push(`estimated cloud-cost delta: $${report.costDeltaUsd.toFixed(4)} (negative = cheaper)`);
  if (report.scorerQuality.samples > 0) {
    lines.push(
      `current scorer vs observed labels: n=${report.scorerQuality.samples} accuracy=${report.scorerQuality.accuracy.toFixed(3)} ` +
        `auc=${report.scorerQuality.auc.toFixed(3)} brier=${report.scorerQuality.brier.toFixed(3)}`,
    );
  } else {
    lines.push('current scorer vs observed labels: not enough labelled episodes to evaluate');
  }
  lines.push('');
  for (const note of report.notes) lines.push(`note: ${note}`);
  const changedRows = report.rows.filter((r) => r.changed);
  if (changedRows.length) {
    lines.push('');
    lines.push('changed decisions:');
    for (const row of changedRows.slice(0, 40)) {
      lines.push(
        `  ${row.ts.slice(0, 19)} ${row.taskClass.padEnd(16)} ${row.oldTier} -> ${row.newTier} ` +
          `(${row.direction}, observedLocal=${row.observedLocalSucceeded === null ? 'n/a' : String(row.observedLocalSucceeded)})`,
      );
    }
  }
  return lines;
}

function makeRandom(seed: number): () => number {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

function round4(n: number): number {
  return Math.round(n * 10_000) / 10_000;
}

function round6(n: number): number {
  return Math.round(n * 1e6) / 1e6;
}

export { EVAL_TASKS };
export type { EvalTask };
