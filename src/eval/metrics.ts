/**
 * Offline quality metrics for the learned router.
 *
 * These numbers answer the only question that matters before trusting the
 * learned scorer with real routing: does it predict observed local failures
 * better than the heuristic prior does? AUC and Brier are reported alongside
 * accuracy because accuracy alone is meaningless when local success is the
 * majority class (which it usually is).
 *
 * The evaluation is in-sample by construction — the scorer is trained on the
 * same episodes it is scored against. That is a real limitation, and the report
 * says so. A time-split evaluation (train on the first 80% of episodes by
 * timestamp, evaluate on the last 20%) is available via `holdout` and is the
 * honest number to look at once there is enough data.
 */

import type { ProtoConfig } from '../config/schema.ts';
import type { EpisodeStore } from '../memory/store.ts';
import { buildDatasets } from '../memory/datasets.ts';
import { evaluate, trainLogistic } from '../router/learned.ts';
import type { TrainingMetrics, TrainingSample } from '../router/learned.ts';
import { loadScorer } from '../router/index.ts';

export interface ScorerQuality {
  samples: number;
  positives: number;
  accuracy: number;
  auc: number;
  brier: number;
  logLoss: number;
  calibration: TrainingMetrics['calibration'];
  /** Present when a time-split evaluation was requested and enough data exists. */
  holdout?: {
    trainSamples: number;
    testSamples: number;
    accuracy: number;
    auc: number;
    brier: number;
    /** AUC of the heuristic prior on the same holdout, for comparison. */
    heuristicAuc: number;
  };
  notes: string[];
}

export function logisticEvalFromEpisodes(
  store: EpisodeStore,
  cfg: ProtoConfig,
  opts: { holdout?: boolean; holdoutFraction?: number } = {},
): ScorerQuality {
  const built = buildDatasets(store, cfg, { write: false, maxRouter: 50_000 });
  const samples: TrainingSample[] = built.router.samples.map((s) => ({ x: s.x, y: s.y, w: s.w }));
  const notes: string[] = [];

  if (samples.length === 0) {
    return {
      samples: 0,
      positives: 0,
      accuracy: 0,
      auc: 0.5,
      brier: 0,
      logLoss: 0,
      calibration: [],
      notes: ['no labelled episodes yet (a label requires a local attempt that was verified)'],
    };
  }

  const { scorer } = loadScorer(store.dataDir);
  const inSample = scorer ? evaluate(scorer, samples) : null;
  const positives = samples.filter((s) => s.y === 1).length;
  const baseRate = positives / samples.length;
  /**
   * With no trained weights, reporting accuracy 0 is misleading: the honest
   * comparison is the majority-class baseline, which is what a router that always
   * guessed one way would achieve. Reporting that makes the *learned* number
   * interpretable as an improvement over doing nothing.
   */
  const baselineAccuracy = Math.max(baseRate, 1 - baseRate);

  const result: ScorerQuality = {
    samples: samples.length,
    positives,
    accuracy: inSample?.accuracy ?? baselineAccuracy,
    auc: inSample?.auc ?? 0.5,
    brier: inSample?.brier ?? baseRate * (1 - baseRate),
    logLoss: inSample?.logLoss ?? 0,
    calibration: inSample?.calibration ?? [],
    notes,
  };

  if (!scorer) {
    notes.push(
      `no trained weights on disk; figures shown are the majority-class baseline ` +
        `(always predict "${baseRate >= 0.5 ? 'local succeeds' : 'local fails'}", accuracy ${baselineAccuracy.toFixed(3)})`,
    );
  } else {
    notes.push('in-sample: the scorer was trained on these same episodes, so treat these numbers as an upper bound');
  }

  if (opts.holdout) {
    const fraction = opts.holdoutFraction ?? 0.2;
    // Router samples are appended in chronological order by datasets.ts, so a
    // simple tail split is a time split.
    const cut = Math.max(1, Math.floor(samples.length * (1 - fraction)));
    const train = samples.slice(0, cut);
    const test = samples.slice(cut);
    if (train.length >= 20 && test.length >= 10) {
      const { scorer: holdoutScorer } = trainLogistic(train, { epochs: 400, learningRate: 0.08, l2: 0.02, seed: 12345 });
      const metrics = evaluate(holdoutScorer, test);
      // Heuristic prior on the same holdout: the feature vector's `bias` and
      // `class_difficulty` terms are a serviceable proxy for the heuristic, but
      // to compare fairly we use the stored heuristic probability, which we do
      // not persist per sample. Instead we compare against a difficulty-only
      // logistic fit, which is the closest cheap baseline available offline.
      const baselineSamples = train.map((s) => ({ x: [1, s.x[27] ?? 0.5], y: s.y, w: s.w }));
      const { scorer: baseline } = trainLogistic(baselineSamples, { epochs: 400, learningRate: 0.1, l2: 0.01, seed: 1 });
      const baselineMetrics = evaluate(
        baseline,
        test.map((s) => ({ x: [1, s.x[27] ?? 0.5], y: s.y, w: s.w })),
      );
      result.holdout = {
        trainSamples: train.length,
        testSamples: test.length,
        accuracy: metrics.accuracy,
        auc: metrics.auc,
        brier: metrics.brier,
        heuristicAuc: baselineMetrics.auc,
      };
      notes.push(
        `time-split holdout: trained on the oldest ${train.length}, evaluated on the newest ${test.length} ` +
          `(heuristic-difficulty baseline AUC ${baselineMetrics.auc.toFixed(3)})`,
      );
    } else {
      notes.push('not enough episodes for a time-split holdout yet (need >=20 train and >=10 test)');
    }
  }

  return result;
}

/** Positive rate by task class: the most actionable diagnostic for a user. */
export function localSuccessByClass(
  store: EpisodeStore,
  cfg: ProtoConfig,
): Array<{ taskClass: string; attempts: number; successes: number; rate: number }> {
  const built = buildDatasets(store, cfg, { write: false, maxRouter: 50_000 });
  const episodes = store.readAll();
  const classOf = new Map(episodes.map((e) => [e.id, e.decision.taskClass]));
  const buckets = new Map<string, { attempts: number; successes: number }>();
  for (const row of built.router.samples) {
    const cls = row.meta.taskClass ?? classOf.get(row.id) ?? 'unknown';
    const bucket = buckets.get(cls) ?? { attempts: 0, successes: 0 };
    bucket.attempts++;
    if (row.y === 1) bucket.successes++;
    buckets.set(cls, bucket);
  }
  return [...buckets.entries()]
    .map(([taskClass, b]) => ({
      taskClass,
      attempts: b.attempts,
      successes: b.successes,
      rate: b.attempts ? b.successes / b.attempts : 0,
    }))
    .sort((a, b) => b.attempts - a.attempts);
}
