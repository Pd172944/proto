/**
 * The heuristic scorer: a transparent, zero-data estimate of whether the local
 * model will succeed on this task, plus an overall difficulty score.
 *
 * Why keep a heuristic at all when there is a learned scorer? Three reasons:
 *  1. **Cold start.** A brand-new install has no episodes; the learned model has
 *     nothing to learn from. The heuristic is the prior the learned scorer is
 *     blended toward, so behaviour degrades gracefully instead of randomly.
 *  2. **Auditability.** Users will not trust a router they cannot read. Every
 *     number here can be printed with `proto route --explain`.
 *  3. **Safety net.** If training produces a pathological weight vector, the
 *     heuristic bounds how badly the router can misbehave (see `policy.ts`
 *     vetoes, which are model-independent).
 *
 * The coefficients are opinions, not measurements. They are deliberately
 * conservative: a false "easy" verdict costs a wasted local attempt plus an
 * escalation (slow), while a false "hard" verdict costs a cloud call (fast,
 * but more expensive). Latency is the scarcer resource in interactive use.
 */

import type { TaskClass, TaskFeatures } from './types.ts';

export interface HeuristicScore {
  /** Estimated probability the local model produces a verifiable-correct answer. */
  pLocalSuccess: number;
  /** 0 = trivial, 1 = very hard. Used for vetoes and exploration bounds. */
  difficulty: number;
  /** Contributions, largest magnitude first; used for `--explain`. */
  contributions: Array<{ feature: string; value: number; note: string }>;
}

/** Classes where a small model has a strong, well-evidenced prior. */
const EASY_CLASSES: TaskClass[] = ['format', 'rename', 'prompt-edit', 'docs', 'explain', 'local-edit', 'config'];
const HARD_CLASSES: TaskClass[] = [
  'architecture',
  'migration',
  'security',
  'concurrency',
  'algorithm',
  'perf',
  'debug-unknown',
];

function clamp01(n: number): number {
  return Math.min(1, Math.max(0, n));
}

export function sigmoid(x: number): number {
  if (x >= 0) {
    const e = Math.exp(-x);
    return 1 / (1 + e);
  }
  const e = Math.exp(x);
  return e / (1 + e);
}

function log1p(n: number): number {
  return Math.log1p(Math.max(0, n));
}

export function heuristicScore(f: TaskFeatures): HeuristicScore {
  const contributions: Array<{ feature: string; value: number; note: string }> = [];

  const fileScope = clamp01(log1p(f.fileCount) / log1p(8));
  const nesting = clamp01(Math.max(0, f.maxNesting - 1) / 5);
  const sizeFactor = clamp01(log1p(f.changedLines) / log1p(400));
  const constraintLoad = clamp01(f.constraintCount / 6);
  const loopLoad = clamp01(log1p(f.loopCount) / log1p(12));

  // --- difficulty: weighted, interpretable contributions -------------------
  const parts: Array<{ feature: string; value: number; note: string }> = [
    { feature: 'class', value: 0.55 * f.classDifficulty, note: `class "${f.taskClass}" prior difficulty ${f.classDifficulty.toFixed(2)}` },
    { feature: 'ambiguity', value: 0.14 * f.ambiguity, note: `ambiguity ${f.ambiguity.toFixed(2)}` },
    { feature: 'fileScope', value: 0.1 * fileScope, note: `${f.fileCount} file(s) in scope` },
    { feature: 'nesting', value: 0.06 * nesting, note: `nesting depth ${f.maxNesting}` },
    { feature: 'size', value: 0.05 * sizeFactor, note: `~${f.changedLines} changed line(s) expected` },
    { feature: 'concurrency', value: f.hasConcurrency ? 0.06 : 0, note: 'concurrency in scope' },
    { feature: 'security', value: f.hasSecurityLanguage ? 0.04 : 0, note: 'security-sensitive wording' },
    { feature: 'migration', value: f.hasMigrationLanguage ? 0.04 : 0, note: 'migration/upgrade wording' },
    { feature: 'perf', value: f.hasPerfLanguage ? 0.03 : 0, note: 'performance wording' },
    {
      feature: 'constraints',
      // Constraint load moves *difficulty* as well as the success logit below.
      // This matters because difficulty — not p — drives tier selection between
      // cloud-cheap and cloud-strong, the exploration bounds, and the harness's
      // escalation threshold. An earlier revision moved only the logit, so a task
      // pushed off local by four interacting constraints still reported
      // difficulty 0.17 and was labelled "easy-to-moderate but cloud-routed",
      // while the same evidence had already decided it was too risky for local.
      // The two scales stay separate (difficulty vs probability) but must not
      // contradict each other.
      value: 0.45 * constraintLoad,
      note: `${f.constraintCount} explicit constraint(s)`,
    },
    { feature: 'locality', value: -0.12 * f.locality, note: `locality ${f.locality.toFixed(2)} (higher is easier)` },
    { feature: 'repro', value: f.hasReproSteps ? -0.05 : 0, note: 'reproduction steps given' },
    { feature: 'stacktrace', value: f.hasStackTrace ? -0.05 : 0, note: 'stack trace present' },
    { feature: 'externalApi', value: f.hasExternalApiMention ? 0.04 : 0, note: 'names a specific library/version (recall risk)' },
    { feature: 'loopLoad', value: 0.02 * loopLoad, note: `${f.loopCount} loop(s) in scope` },
  ];

  const rawDifficulty = parts.reduce((a, p) => a + p.value, 0);
  const difficulty = clamp01(rawDifficulty);

  // --- probability: a logistic map of difficulty with verifiability bumps ---
  let logit = 2.2 - 4.6 * difficulty;

  const extras: Array<{ feature: string; value: number; note: string }> = [];
  if (f.hasTestsInScope) {
    logit += 0.35;
    extras.push({ feature: 'testsInScope', value: 0.35, note: 'tests exist in scope, so a wrong answer is catchable' });
  }
  if (f.hasTypes) {
    logit += 0.2;
    extras.push({ feature: 'types', value: 0.2, note: 'typed code gives the model stronger local constraints' });
  }
  if (f.mentionsSpecificSymbol) {
    logit += 0.15;
    extras.push({ feature: 'specificSymbol', value: 0.15, note: 'task names a concrete symbol' });
  }
  if (f.isExplainOnly) {
    logit += 0.5;
    extras.push({ feature: 'explainOnly', value: 0.5, note: 'explanation task: no patch to get wrong' });
  }
  if (EASY_CLASSES.includes(f.taskClass)) {
    logit += 0.4;
    extras.push({ feature: 'easyClass', value: 0.4, note: `"${f.taskClass}" has a strong small-model prior` });
  }
  if (HARD_CLASSES.includes(f.taskClass)) {
    logit -= 0.9;
    extras.push({ feature: 'hardClass', value: -0.9, note: `"${f.taskClass}" is rarely a one-shot small-model task` });
  }
  if (f.hasConcurrency && f.hasAsync) {
    logit -= 0.35;
    extras.push({ feature: 'asyncConcurrency', value: -0.35, note: 'async plus concurrency compound the failure modes' });
  }
  if (f.estInputTokens > 6000) {
    logit -= 0.5;
    extras.push({ feature: 'longContext', value: -0.5, note: `~${f.estInputTokens} input tokens strains a small context window` });
  }
  if (f.estOutputTokens > 1800) {
    logit -= 0.3;
    extras.push({ feature: 'longOutput', value: -0.3, note: `~${f.estOutputTokens} output tokens is a lot to emit coherently` });
  }
  // Constraint interaction. Each explicit constraint ("must not stack", "round
  // half-up", "keep the API unchanged") is an independent opportunity to miss a
  // requirement, so reliability degrades roughly multiplicatively. We model that
  // with a linear logit penalty, which is the log-space equivalent of a product.
  // This is why four easy-sounding constraints together become a cloud task.
  if (f.constraintCount >= 2) {
    const penalty = -0.3 * Math.min(f.constraintCount, 6);
    logit += penalty;
    extras.push({
      feature: 'constraintInteraction',
      value: penalty,
      note: `${f.constraintCount} explicit constraints: each one is an independent chance to miss a requirement`,
    });
  }

  contributions.push(...parts.map((p) => ({ ...p, note: `${p.note} [difficulty ${p.value >= 0 ? '+' : ''}${p.value.toFixed(3)}]` })));
  contributions.push(...extras.map((e) => ({ ...e, note: `${e.note} [logit ${e.value >= 0 ? '+' : ''}${e.value.toFixed(2)}]` })));

  const pLocalSuccess = clamp01(0.02 + 0.96 * sigmoid(logit));

  contributions.sort((a, b) => Math.abs(b.value) - Math.abs(a.value));
  return { pLocalSuccess, difficulty, contributions };
}

/** Blend a learned probability with the heuristic prior, weighted by sample size. */
export function blendWithPrior(learned: number, prior: number, sampleCount: number): {
  p: number;
  learnedWeight: number;
} {
  // Shrinkage: with few observations, trust the heuristic. Caps at 0.85 so the
  // heuristic never fully disappears (it is the safety net described above).
  const w = Math.min(0.85, sampleCount / (sampleCount + 60));
  return { p: clamp01(w * learned + (1 - w) * prior), learnedWeight: w };
}
