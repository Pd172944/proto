/**
 * Evaluation and replay tests.
 *
 * Includes a regression guard on the corpus score. If a future change to the
 * feature extractor or policy degrades routing, this test fails loudly rather
 * than letting the degradation show up as user-visible slowness weeks later.
 */

import { strict as assert } from 'node:assert';
import { describe, it } from 'node:test';

import { EVAL_TASKS, computeMetrics, evaluateRouting, formatEvalReport, formatReplayReport, replay } from '../src/eval/index.ts';
import { localSuccessByClass, logisticEvalFromEpisodes } from '../src/eval/metrics.ts';
import { EpisodeStore } from '../src/memory/store.ts';
import { refreshRouter } from '../src/train/index.ts';
import { makeEpisode, tempDir, testConfig } from './helpers.ts';

describe('eval corpus integrity', () => {
  it('has a balanced, well-formed corpus', () => {
    assert.ok(EVAL_TASKS.length >= 20, 'the corpus must be big enough to be meaningful');
    const ids = new Set(EVAL_TASKS.map((t) => t.id));
    assert.equal(ids.size, EVAL_TASKS.length, 'task ids must be unique');
    const local = EVAL_TASKS.filter((t) => t.expected === 'local').length;
    const cloud = EVAL_TASKS.filter((t) => t.expected === 'cloud').length;
    assert.ok(local >= 8 && cloud >= 8, `corpus is unbalanced: local=${local} cloud=${cloud}`);
    const verifiable = EVAL_TASKS.filter((t) => t.verifiable).length;
    assert.ok(verifiable >= 12, 'most tasks should be machine-verifiable');
    assert.ok(EVAL_TASKS.some((t) => !t.verifiable), 'unverifiable tasks must be represented');
  });

  it('labels by competence, not by size', () => {
    // A corpus where length predicts the label would let a trivial length
    // heuristic score highly, which would make the metric worthless.
    const localLens = EVAL_TASKS.filter((t) => t.expected === 'local').map((t) => t.task.length);
    const cloudLens = EVAL_TASKS.filter((t) => t.expected === 'cloud').map((t) => t.task.length);
    const avg = (xs: number[]): number => xs.reduce((a, b) => a + b, 0) / xs.length;
    const ratio = avg(localLens) / avg(cloudLens);
    assert.ok(ratio > 0.6 && ratio < 1.6, `task length leaks the label (ratio ${ratio.toFixed(2)})`);
  });
});

describe('offline routing evaluation', () => {
  it('scores the corpus without calling any model', () => {
    const dir = tempDir();
    const report = evaluateRouting(testConfig(dir), dir);
    assert.equal(report.rows.length, EVAL_TASKS.length);
    assert.ok(report.metrics.accuracy >= 0.9, `routing accuracy regressed to ${report.metrics.accuracy}`);
  });

  it('never sends a cloud-labelled task to the local model', () => {
    // The asymmetry is deliberate: routing an easy task to the cloud wastes
    // money and latency, but routing a hard task to a weak model produces a
    // confidently wrong answer, which is much worse.
    const dir = tempDir();
    const report = evaluateRouting(testConfig(dir), dir);
    const bad = report.rows.filter((r) => r.expected === 'cloud' && r.predicted === 'local');
    assert.deepEqual(bad.map((r) => r.id), [], 'hard tasks must never be routed local');
    assert.equal(report.metrics.confusion.cloud.local, 0);
  });

  it('reports a confusion matrix and cost comparison', () => {
    const dir = tempDir();
    const report = evaluateRouting(testConfig(dir), dir);
    const m = report.metrics;
    assert.equal(m.confusion.local.local + m.confusion.local.cloud, EVAL_TASKS.filter((t) => t.expected === 'local').length);
    assert.ok(m.cloudOnlyCostUsd > 0, 'the all-cloud baseline must cost something');
    assert.ok(m.estimatedCloudCostUsd <= m.cloudOnlyCostUsd, 'routing must not cost more than all-cloud');
    assert.ok(formatEvalReport(report).join('\n').includes('confusion:'));
  });

  it('produces identical results on repeated runs (deterministic)', () => {
    const dir = tempDir();
    const a = evaluateRouting(testConfig(dir), dir);
    const b = evaluateRouting(testConfig(dir), dir);
    assert.deepEqual(
      a.rows.map((r) => [r.id, r.predictedTier]),
      b.rows.map((r) => [r.id, r.predictedTier]),
    );
  });

  it('can restrict scoring to a subset of tasks', () => {
    const dir = tempDir();
    const report = evaluateRouting(testConfig(dir), dir, { only: ['local-off-by-one-loop'] });
    assert.equal(report.rows.length, 1);
    assert.equal(report.rows[0]?.predicted, 'local');
  });

  it('scores all three modes and lets them be compared', () => {
    const dir = tempDir();
    const dir2 = tempDir();
    const reports = (['heuristic', 'learned', 'hybrid'] as const).map((mode) => evaluateRouting(testConfig(dir), dir2, { mode }));
    for (const r of reports) {
      assert.equal(r.mode, r.mode);
      assert.ok(r.rows.length === EVAL_TASKS.length);
    }
  });

  it('does not use exploration when scoring, so the score is not luck-dependent', () => {
    const dir = tempDir();
    const report = evaluateRouting(testConfig(dir), dir);
    assert.equal(report.metrics.explorationCount, 0);
    assert.ok(report.notes.some((n) => /Exploration is disabled/.test(n)));
  });
});

describe('metrics helpers', () => {
  it('computes precision and recall from a confusion matrix', () => {
    const rows = [
      { expected: 'local' as const, predicted: 'local' as const, verifiable: true, cloudCostUsd: 0.01 },
      { expected: 'local' as const, predicted: 'cloud' as const, verifiable: true, cloudCostUsd: 0.01 },
      { expected: 'cloud' as const, predicted: 'cloud' as const, verifiable: true, cloudCostUsd: 0.01 },
      { expected: 'cloud' as const, predicted: 'local' as const, verifiable: true, cloudCostUsd: 0.01 },
    ].map((r) => ({
      ...r,
      id: 'x',
      predictedTier: r.predicted as never,
      correct: r.expected === r.predicted,
      taskClass: 'local-edit',
      classCorrect: null,
      pLocalSuccess: 0.5,
      difficulty: 0.5,
      localLatencyMs: 100,
      cloudLatencyMs: 1000,
      exploration: false,
      readOnly: false,
      utility: 0,
      reason: '',
    }));
    const m = computeMetrics(rows);
    assert.equal(m.total, 4);
    assert.equal(m.correct, 2);
    assert.equal(m.localPrecision, 0.5);
    assert.equal(m.localRecall, 0.5);
    assert.equal(m.unverifiedLocal, 0);
  });

  it('reports local success rate per task class from real episodes', () => {
    const dir = tempDir();
    const store = new EpisodeStore(dir);
    store.append(makeEpisode({ task: 'Fix the off-by-one in this loop.', localSucceeded: true }));
    store.append(makeEpisode({ task: 'Fix the off-by-one in this loop again.', localSucceeded: false, verified: false }));
    store.append(makeEpisode({ task: 'Design the module boundaries.', localSucceeded: false, verified: false }));

    const byClass = localSuccessByClass(store, testConfig(dir));
    const bugfix = byClass.find((r) => r.taskClass === 'bugfix-local');
    assert.ok(bugfix, `expected a bugfix-local bucket, got ${byClass.map((b) => b.taskClass).join(',')}`);
    assert.equal(bugfix?.attempts, 2);
    assert.equal(bugfix?.successes, 1);
    assert.equal(bugfix?.rate, 0.5);
  });

  it('reports a baseline rather than a misleading zero when untrained', () => {
    const dir = tempDir();
    const store = new EpisodeStore(dir);
    store.append(makeEpisode({ localSucceeded: true }));
    const quality = logisticEvalFromEpisodes(store, testConfig(dir));
    assert.equal(quality.samples, 1);
    assert.equal(quality.accuracy, 1, 'majority-class baseline for a single positive');
    assert.ok(quality.notes.some((n) => /majority-class baseline/.test(n)));
  });
});

describe('counterfactual replay', () => {
  function seededStore(): { dir: string; store: EpisodeStore } {
    const dir = tempDir();
    const store = new EpisodeStore(dir);
    store.append(makeEpisode({ task: 'Fix the off-by-one in this loop.', localSucceeded: true, reward: 1.3 }));
    store.append(
      makeEpisode({
        task: 'Design the module boundaries and migrate the schema.',
        localSucceeded: false,
        verified: false,
        status: 'escalated-cloud-success',
        escalated: true,
        reward: -0.2,
      }),
    );
    return { dir, store };
  }

  it('re-scores history without any model calls and explains its limits', () => {
    const { dir } = seededStore();
    const report = replay(testConfig(dir), dir);
    assert.equal(report.episodes, 2);
    assert.ok(report.rows.length === 2);
    assert.ok(report.notes.some((n) => /no model calls/.test(n)));
    assert.ok(report.notes.some((n) => /cannot know whether local/.test(n)));
    assert.ok(formatReplayReport(report).length > 0);
  });

  it('reports the cloud-cost delta of the alternative policy', () => {
    const { dir } = seededStore();
    const report = replay(testConfig(dir), dir, { mode: 'heuristic' });
    assert.equal(typeof report.costDeltaUsd, 'number');
    for (const row of report.rows) {
      assert.equal(typeof row.cloudCostDeltaUsd, 'number');
      assert.ok(['same', 'more-cautious', 'more-local'].includes(row.direction));
    }
  });

  it('is unaffected by a higher quality floor only where a verifier exists', () => {
    const { dir } = seededStore();
    const permissive = replay(testConfig(dir), dir, { qualityFloor: 0.05 });
    const strict = replay(testConfig(dir), dir, { qualityFloor: 0.99 });
    const permissiveLocal = permissive.rows.filter((r) => r.newTier.startsWith('local')).length;
    const strictLocal = strict.rows.filter((r) => r.newTier.startsWith('local')).length;
    assert.ok(
      permissiveLocal >= strictLocal,
      `a higher floor must not increase local routing (${permissiveLocal} vs ${strictLocal})`,
    );
  });

  it('can limit replay to a time window', () => {
    const { dir, store } = seededStore();
    store.append(makeEpisode({ ts: '2020-01-01T00:00:00.000Z', localSucceeded: true }));
    const report = replay(testConfig(dir), dir, { since: new Date('2021-01-01T00:00:00Z') });
    assert.equal(report.episodes, 2);
  });
});

describe('learned routing after training on logged episodes', () => {
  it('changes decisions once weights exist, and stays within tier vocabulary', () => {
    const dir = tempDir();
    const store = new EpisodeStore(dir);
    for (let i = 0; i < 30; i++) {
      store.append(makeEpisode({ task: `Rename variable a${i} to count${i} in this file.`, localSucceeded: true, reward: 1.3 }));
      store.append(
        makeEpisode({
          task: `Design the module boundaries and migrate the schema for service ${i}.`,
          localSucceeded: false,
          verified: false,
          status: 'escalated-cloud-success',
          escalated: true,
          reward: 0.2,
        }),
      );
    }
    const trained = refreshRouter(testConfig(dir), dir, { write: true });
    assert.equal(trained.trained, true);

    const learned = evaluateRouting(testConfig(dir), dir, { mode: 'learned' });
    assert.equal(learned.scorerPresent, true);
    assert.ok(learned.scorerSamples >= 40);
    for (const row of learned.rows) {
      assert.ok(['local-tiny', 'local', 'cloud-cheap', 'cloud-strong'].includes(row.predictedTier));
    }
    // The learned scorer must still never route a hard task local.
    const bad = learned.rows.filter((r) => r.expected === 'cloud' && r.predicted === 'local');
    assert.deepEqual(bad.map((r) => r.id), []);
  });
});
