/**
 * The learned scorer: L2-regularised logistic regression over the task feature
 * vector, trained locally in milliseconds.
 *
 * Design rationale (this is a deliberate research choice, not a shortcut):
 *
 *  - **Why logistic regression and not a small neural net?** The label we can
 *    actually observe is "did the local attempt survive verification", which is
 *    noisy, sparse and heavily selection-biased (we only observe the local
 *    outcome on tasks the router already sent local). A linear model with strong
 *    regularisation is far more sample-efficient and far harder to overfit than
 *    an MLP on a few hundred rows. It also trains in ~1 ms, which is what makes
 *    "retrain the router on every batch" feasible on a laptop.
 *
 *  - **Why train in TypeScript rather than in Python next to the model?** The
 *    router's weights must be usable instantly on the critical path, and the
 *    training data is tiny (thousands of rows x 43 features). Keeping it in
 *    Node avoids a Python round-trip on every routing decision. The *model*
 *    fine-tuning (LoRA) is the part that needs Python and is deferred.
 *
 *  - **Selection bias is real.** See `train()`: we accept optional inverse
 *    propensity weights so that episodes gathered by the exploration policy can
 *    be reweighted toward the population distribution. This is a cheap,
 *    honest approximation of off-policy correction and is documented as an
 *    approximation, not a solved problem (see docs/rl-design.md).
 */

import { FEATURE_NAMES, FEATURE_VECTOR_VERSION } from './types.ts';
import type { FeatureName } from './types.ts';
import { sigmoid } from './heuristic.ts';

/** Below this many labelled episodes the learned scorer is not used at all. */
export const MIN_TRAINING_SAMPLES = 40;

export interface TrainingSample {
  x: number[];
  /** 1 = local attempt succeeded, 0 = it failed. */
  y: 0 | 1;
  /** Optional importance weight (inverse propensity). Defaults to 1. */
  w?: number;
}

export interface TrainOptions {
  epochs?: number;
  learningRate?: number;
  l2?: number;
  /**
   * Reweight positives and negatives to equal total mass.
   *
   * Off by default, and that default matters. Balancing rescales the output so it
   * is no longer a probability: the score stops meaning "the chance local
   * succeeds" and starts meaning "a balanced-score statistic". Two consumers
   * depend on the probability reading:
   *   - `policy.ts` compares the score against a probability floor (0.72 / 0.90 /
   *     0.55), so a rescaled score silently shifts the effective threshold;
   *   - `heuristic.ts::blendWithPrior` averages it with the heuristic prior,
   *     which *is* calibrated, so mixing two scales produces a meaningless number.
   * A simulation of the learning curve showed exactly this: the balanced model had
   * a higher AUC yet produced worse routing decisions than the prior. Leave this
   * off unless you are deliberately training a ranker and will calibrate it
   * (Platt/isotonic) before it reaches the policy.
   */
  balanceClasses?: boolean;
  seed?: number;
}

export interface TrainingMetrics {
  n: number;
  positives: number;
  accuracy: number;
  auc: number;
  brier: number;
  logLoss: number;
  /** Reliability curve: predicted vs observed, 5 bins. */
  calibration: Array<{ bin: string; n: number; predicted: number; observed: number }>;
}

export interface RouterWeights {
  version: 1;
  featureVectorVersion: number;
  featureNames: readonly string[];
  weights: number[];
  trainedAt: string;
  sampleCount: number;
  metrics: TrainingMetrics;
  notes: string[];
}


export class LogisticScorer {
  weights: number[];
  sampleCount: number;
  metrics: TrainingMetrics | null;
  trainedAt: string | null;

  constructor(weights?: number[], meta?: { sampleCount?: number; metrics?: TrainingMetrics | null; trainedAt?: string | null }) {
    this.weights = weights && weights.length === FEATURE_NAMES.length ? [...weights] : new Array(FEATURE_NAMES.length).fill(0);
    this.sampleCount = meta?.sampleCount ?? 0;
    this.metrics = meta?.metrics ?? null;
    this.trainedAt = meta?.trainedAt ?? null;
  }

  /** Raw linear score. */
  logit(x: number[]): number {
    let z = 0;
    for (let i = 0; i < this.weights.length; i++) z += (this.weights[i] as number) * (x[i] ?? 0);
    return z;
  }

  predict(x: number[]): number {
    return sigmoid(this.logit(x));
  }

  toFile(notes: string[] = []): RouterWeights {
    return {
      version: 1,
      featureVectorVersion: FEATURE_VECTOR_VERSION,
      featureNames: FEATURE_NAMES,
      weights: this.weights,
      trainedAt: this.trainedAt ?? new Date().toISOString(),
      sampleCount: this.sampleCount,
      metrics:
        this.metrics ??
        { n: 0, positives: 0, accuracy: 0, auc: 0.5, brier: 0, logLoss: 0, calibration: [] },
      notes,
    };
  }

  static fromFile(file: RouterWeights): LogisticScorer {
    if (file.featureVectorVersion !== FEATURE_VECTOR_VERSION) {
      throw new Error(
        `router weights were trained on feature vector v${file.featureVectorVersion}, ` +
          `but this build uses v${FEATURE_VECTOR_VERSION}; retrain with \`proto train router\``,
      );
    }
    if (file.weights.length !== FEATURE_NAMES.length) {
      throw new Error(
        `router weights have ${file.weights.length} entries but ${FEATURE_NAMES.length} features are expected`,
      );
    }
    return new LogisticScorer(file.weights, {
      sampleCount: file.sampleCount,
      metrics: file.metrics,
      trainedAt: file.trainedAt,
    });
  }
}

/**
 * Train by mini-batch SGD with L2 and class balancing.
 *
 * Fully deterministic: full-batch gradient descent with no shuffling, so
 * `proto train router` produces bit-identical weights for the same data and the
 * eval harness can compare runs meaningfully. (`opts.seed` is accepted for API
 * compatibility and no longer influences the result.)
 */
export function trainLogistic(
  samples: TrainingSample[],
  opts: TrainOptions = {},
): { scorer: LogisticScorer; metrics: TrainingMetrics } {
  const epochs = opts.epochs ?? 300;
  const lr = opts.learningRate ?? 0.5;
  const l2 = opts.l2 ?? 0.02;
  const balance = opts.balanceClasses ?? false;
  const d = FEATURE_NAMES.length;

  if (samples.length === 0) {
    const empty = new LogisticScorer(new Array(d).fill(0), { sampleCount: 0 });
    return {
      scorer: empty,
      metrics: { n: 0, positives: 0, accuracy: 0, auc: 0.5, brier: 0, logLoss: 0, calibration: [] },
    };
  }

  const positives = samples.filter((s) => s.y === 1).length;
  const negatives = samples.length - positives;
  const posWeight = balance && positives > 0 && negatives > 0 ? samples.length / (2 * positives) : 1;
  const negWeight = balance && positives > 0 && negatives > 0 ? samples.length / (2 * negatives) : 1;

  // Start from the class prior in the bias term so early epochs are sane.
  const weights = new Array(d).fill(0);
  weights[0] = Math.log((positives + 1) / (negatives + 1));

  /*
   * Full-batch gradient descent on the *mean* weighted loss.
   *
   * The previous implementation was per-sample SGD that accumulated the gradient
   * over the whole dataset without normalising by its size, so the effective step
   * size grew linearly with the number of episodes: with 68 examples training
   * behaved, and with 1600 it oscillated. The learned scorer therefore got WORSE
   * as the user's log grew — the exact opposite of the point — which a
   * learning-curve simulation surfaced (held-out AUC fell from 0.65 to 0.51 as
   * labels went from 68 to 1647).
   *
   * Dividing by the total weight makes the gradient a proper mean, which makes the
   * learning rate, the L2 strength and the convergence behaviour all independent
   * of dataset size. It is also deterministic, so `seed` no longer affects the
   * result; the parameter is kept for API compatibility.
   *
   * Cost: `epochs x n x d` multiply-adds. At 43 features and a few thousand rows
   * that is a handful of milliseconds, which is what makes the fast loop free.
   */
  const grad = new Array<number>(d).fill(0);

  for (let epoch = 0; epoch < epochs; epoch++) {
    grad.fill(0);
    let totalWeight = 0;

    for (const s of samples) {
      const cw = (s.y === 1 ? posWeight : negWeight) * (s.w ?? 1);
      let z = 0;
      for (let k = 0; k < d; k++) z += (weights[k] as number) * (s.x[k] ?? 0);
      const g = (sigmoid(z) - s.y) * cw;
      for (let k = 0; k < d; k++) grad[k] = (grad[k] as number) + g * (s.x[k] ?? 0);
      totalWeight += cw;
    }

    if (totalWeight <= 0) break;
    // Decay the step size over the run for a smoother finish.
    const step = lr / (1 + epoch / (epochs * 0.5));

    for (let k = 0; k < d; k++) {
      // Never regularise the bias: it carries the class prior, not a hypothesis.
      const reg = k === 0 ? 0 : l2 * (weights[k] as number);
      weights[k] = (weights[k] as number) - step * ((grad[k] as number) / totalWeight + reg);
    }
  }

  const scorer = new LogisticScorer(weights, { sampleCount: samples.length, trainedAt: new Date().toISOString() });
  const metrics = evaluate(scorer, samples);
  scorer.metrics = metrics;
  return { scorer, metrics };
}

/** Accuracy, AUC, Brier, log-loss and a 5-bin reliability curve. */
export function evaluate(scorer: LogisticScorer, samples: TrainingSample[]): TrainingMetrics {
  const n = samples.length;
  if (n === 0) {
    return { n: 0, positives: 0, accuracy: 0, auc: 0.5, brier: 0, logLoss: 0, calibration: [] };
  }
  const scored = samples.map((s) => ({ p: scorer.predict(s.x), y: s.y }));
  const positives = scored.filter((s) => s.y === 1).length;

  let correct = 0;
  let brier = 0;
  let logLoss = 0;
  for (const s of scored) {
    const pred = s.p >= 0.5 ? 1 : 0;
    if (pred === s.y) correct++;
    brier += (s.p - s.y) ** 2;
    const clamped = Math.min(1 - 1e-9, Math.max(1e-9, s.p));
    logLoss += -(s.y * Math.log(clamped) + (1 - s.y) * Math.log(1 - clamped));
  }

  // AUC via the Mann-Whitney U statistic with average ranks for ties.
  const sorted = [...scored].sort((a, b) => a.p - b.p);
  const ranks = new Array<number>(n).fill(0);
  let i = 0;
  while (i < n) {
    let j = i;
    while (j + 1 < n && (sorted[j + 1] as { p: number }).p === (sorted[i] as { p: number }).p) j++;
    const avg = (i + j) / 2 + 1;
    for (let k = i; k <= j; k++) ranks[k] = avg;
    i = j + 1;
  }
  let rankSumPos = 0;
  for (let k = 0; k < n; k++) if ((sorted[k] as { y: number }).y === 1) rankSumPos += ranks[k] as number;
  const nPos = positives;
  const nNeg = n - positives;
  const auc = nPos === 0 || nNeg === 0 ? 0.5 : (rankSumPos - (nPos * (nPos + 1)) / 2) / (nPos * nNeg);

  const bins = 5;
  const calibration: TrainingMetrics['calibration'] = [];
  for (let b = 0; b < bins; b++) {
    const lo = b / bins;
    const hi = (b + 1) / bins;
    const inBin = scored.filter((s) => s.p >= lo && (b === bins - 1 ? s.p <= hi : s.p < hi));
    if (inBin.length === 0) {
      calibration.push({ bin: `${lo.toFixed(1)}-${hi.toFixed(1)}`, n: 0, predicted: 0, observed: 0 });
      continue;
    }
    calibration.push({
      bin: `${lo.toFixed(1)}-${hi.toFixed(1)}`,
      n: inBin.length,
      predicted: inBin.reduce((a, s) => a + s.p, 0) / inBin.length,
      observed: inBin.reduce((a, s) => a + s.y, 0) / inBin.length,
    });
  }

  return {
    n,
    positives,
    accuracy: correct / n,
    auc,
    brier: brier / n,
    logLoss: logLoss / n,
    calibration,
  };
}

/** Which features push toward local success or failure, for `proto router show`. */
export function featureImportance(scorer: LogisticScorer): Array<{ name: FeatureName; weight: number }> {
  return FEATURE_NAMES.map((name, i) => ({ name, weight: scorer.weights[i] ?? 0 })).sort(
    (a, b) => Math.abs(b.weight) - Math.abs(a.weight),
  );
}
