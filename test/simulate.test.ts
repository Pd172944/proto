/**
 * Learning-curve simulation tests.
 *
 * Two things are worth protecting here. First, that the simulation cannot
 * contaminate real data — it must write its synthetic episodes to a scratch
 * directory, because that log is what trains the deployed router. Second, that the
 * synthetic ground truth keeps the property that makes the simulation meaningful:
 * the local model's skill must be *partly invisible* to the heuristic, or the
 * heuristic is unbeatable by construction and the exercise proves nothing.
 */

import { strict as assert } from 'node:assert';
import { existsSync, readdirSync } from 'node:fs';
import { join } from 'node:path';
import { describe, it } from 'node:test';

import { aucOf, buildTruthModel, simulate, trueSuccessProbability } from '../src/simulate/index.ts';
import { EpisodeStore } from '../src/memory/store.ts';
import { heuristicScore } from '../src/router/heuristic.ts';
import { extractFeatures } from '../src/router/features.ts';
import { EVAL_TASKS } from '../src/eval/tasks.ts';
import { MIN_TRAINING_SAMPLES } from '../src/router/learned.ts';
import { tempDir, testConfig } from './helpers.ts';

describe('simulation ground truth', () => {
  it('is monotonic in difficulty', () => {
    const easy = trueSuccessProbability(0.1, 1, 'realistic');
    const hard = trueSuccessProbability(0.8, 1, 'realistic');
    assert.ok(easy > hard, `${easy} should exceed ${hard}`);
    assert.ok(easy <= 0.97 && hard >= 0.02);
  });

  it('shifts with the competence dial', () => {
    const p = trueSuccessProbability(0.4, 1, 'pessimistic');
    const r = trueSuccessProbability(0.4, 1, 'realistic');
    const o = trueSuccessProbability(0.4, 1, 'optimistic');
    assert.ok(p < r && r < o);
  });

  it('gives the local model per-class quirks the heuristic cannot see', () => {
    // This is the property that makes the simulation informative. If every
    // affinity were 1.0, the heuristic would be the Bayes-optimal model and no
    // amount of learning could beat it.
    const truth = buildTruthModel(42, 'realistic', 0.3);
    const values = [...truth.affinities.values()];
    assert.ok(values.length >= 10, 'the corpus should span many classes');
    assert.ok(values.some((a) => a > 1.1), 'some classes must be better than difficulty suggests');
    assert.ok(values.some((a) => a < 0.9), 'some classes must be worse than difficulty suggests');
    assert.ok(
      values.every((a) => a >= 0.4 && a <= 1.6),
      'affinities are bounded so the scenario stays plausible',
    );
  });

  it('is reproducible for a given seed and varies across seeds', () => {
    const a = buildTruthModel(7, 'realistic', 0.3).affinities;
    const b = buildTruthModel(7, 'realistic', 0.3).affinities;
    const c = buildTruthModel(8, 'realistic', 0.3).affinities;
    assert.deepEqual([...a.entries()], [...b.entries()]);
    assert.notDeepEqual([...a.entries()], [...c.entries()]);
  });

  it('applies affinities on top of the difficulty-implied probability', () => {
    const low = trueSuccessProbability(0.5, 0.6, 'realistic');
    const high = trueSuccessProbability(0.5, 1.4, 'realistic');
    assert.ok(high > low);
  });
});

describe('simulate', () => {
  it('accumulates labels and trains once there are enough', () => {
    const dir = tempDir();
    const result = simulate(testConfig(dir), dir, {
      tasks: 400,
      seed: 11,
      competence: 'realistic',
      checkpoints: [50, 200, 400],
    });
    const first = result.checkpoints[0];
    const last = result.checkpoints[result.checkpoints.length - 1];
    assert.equal(first?.tasks, 0);
    assert.equal(first?.labels, 0);
    assert.ok((last?.labels ?? 0) > 150, `expected a healthy label count, got ${last?.labels}`);
    assert.ok(result.firstTrainingAt !== null, 'should train by 400 tasks');
    assert.ok((result.firstTrainingAt ?? 0) <= 400);
    assert.ok((last?.learnedAuc ?? 0) > 0.5, 'the learner must beat chance');
  });

  it('does not train before the minimum number of labels', () => {
    const dir = tempDir();
    const result = simulate(testConfig(dir), dir, { tasks: 20, seed: 5, checkpoints: [20] });
    const last = result.checkpoints[result.checkpoints.length - 1];
    assert.ok((last?.labels ?? 0) < MIN_TRAINING_SAMPLES);
    assert.equal(last?.routerTrained, false);
    assert.equal(result.firstTrainingAt, null);
    assert.match(result.interpretation.join(' '), /below the 40/);
  });

  it('reports the deployed hybrid separately from the raw learned scorer', () => {
    // The distinction matters: the default mode shrinks the learned score toward
    // the prior, and it is the blend that a user actually runs.
    const dir = tempDir();
    const result = simulate(testConfig(dir), dir, { tasks: 300, seed: 3, checkpoints: [300] });
    const last = result.checkpoints[result.checkpoints.length - 1];
    assert.ok(last?.learnedAuc !== null);
    assert.ok(last?.hybridAuc !== null);
    assert.ok(last?.heuristicAuc !== undefined);
    // The blend must sit between the two inputs.
    const lo = Math.min(last?.learnedAuc ?? 0, last?.heuristicAuc ?? 0) - 0.05;
    const hi = Math.max(last?.learnedAuc ?? 1, last?.heuristicAuc ?? 1) + 0.05;
    assert.ok((last?.hybridAuc ?? 0) >= lo && (last?.hybridAuc ?? 0) <= hi);
  });

  it('never lets the learned scorer degrade as the log grows', () => {
    // Regression guard for the optimiser bug: per-sample gradients were
    // accumulated without normalising by dataset size, so the effective step grew
    // with the number of episodes and the learned scorer got WORSE with more data
    // (held-out AUC fell to ~0.51). The simulation is what surfaced it, so the
    // simulation is what guards it.
    const dir = tempDir();
    const result = simulate(testConfig(dir), dir, {
      tasks: 2000,
      seed: 20250601,
      checkpoints: [100, 500, 1000, 2000],
    });
    const trained = result.checkpoints.filter((c) => c.learnedAuc !== null);
    assert.ok(trained.length >= 3);
    const earliest = trained[0]?.learnedAuc ?? 0;
    const latest = trained[trained.length - 1]?.learnedAuc ?? 0;
    assert.ok(latest > 0.55, `learned AUC collapsed to ${latest}`);
    assert.ok(
      latest > earliest - 0.1,
      `learned AUC degraded badly with more data: ${earliest} -> ${latest}`,
    );
    // And the deployed scorer must not be worse than the prior by more than noise.
    const last = trained[trained.length - 1];
    assert.ok(
      (last?.deployedGainOverHeuristic ?? -1) > -0.03,
      `deployed scorer regressed against the prior by ${last?.deployedGainOverHeuristic}`,
    );
  });

  it('reports verifier risk rather than only successes', () => {
    const dir = tempDir();
    // A weak verifier: it catches only half of the wrong answers.
    const result = simulate(testConfig(dir), dir, {
      tasks: 400,
      seed: 9,
      verifierCatchRate: 0.5,
      checkpoints: [400],
    });
    const last = result.checkpoints[result.checkpoints.length - 1];
    assert.ok((last?.falsePasses ?? 0) > 0, 'a weak verifier must produce false passes');
    assert.ok((last?.falsePassRate ?? 0) > 0.05);
    assert.match(result.interpretation.join(' '), /Verifier risk/);
  });

  it('is isolated from the real episode log and router weights', () => {
    const dir = tempDir();
    simulate(testConfig(dir), dir, { tasks: 150, seed: 13, checkpoints: [150] });

    // No real episodes may be written...
    assert.ok(!existsSync(join(dir, 'episodes')), 'simulation must not write to the real episode log');
    // ...and no real router weights either.
    assert.ok(!existsSync(join(dir, 'router', 'weights.json')), 'simulation must not write real router weights');
    // ...and the scratch directory is removed unless asked to keep it.
    assert.ok(!existsSync(join(dir, 'sim', 'run-13-150')), 'scratch episodes are deleted by default');
    assert.equal(new EpisodeStore(dir).readAll().length, 0);
  });

  it('can keep its episodes for inspection when asked', () => {
    const dir = tempDir();
    const result = simulate(testConfig(dir), dir, {
      tasks: 200,
      seed: 14,
      checkpoints: [200],
      writeEpisodes: true,
    });
    assert.ok(result.scratchDir);
    const kept = new EpisodeStore(result.scratchDir as string).readAll();
    assert.ok(kept.length >= 190, `expected the kept log to hold the run, got ${kept.length}`);
    // And the real log is still untouched.
    assert.equal(new EpisodeStore(dir).readAll().length, 0);
  });

  it('produces an affinity table and explains what it means', () => {
    const dir = tempDir();
    const result = simulate(testConfig(dir), dir, { tasks: 100, seed: 21, checkpoints: [100] });
    assert.ok(result.affinities.length >= 10);
    assert.ok(result.affinities.every((a) => a.note.length > 0));
    assert.ok(result.notes.some((n) => /synthesised/.test(n)));
    assert.ok(result.interpretation.length > 0);
  });

  it('is deterministic for a given seed', () => {
    const a = simulate(testConfig(tempDir()), tempDir(), { tasks: 120, seed: 99, checkpoints: [120] });
    const b = simulate(testConfig(tempDir()), tempDir(), { tasks: 120, seed: 99, checkpoints: [120] });
    assert.deepEqual(
      a.checkpoints.map((c) => [c.labels, c.learnedAuc, c.hybridAuc]),
      b.checkpoints.map((c) => [c.labels, c.learnedAuc, c.hybridAuc]),
    );
  });
});

describe('aucOf', () => {
  it('is 1 for perfect separation and 0.5 for none', () => {
    const perfect = [
      { p: 0.9, y: 1 as const },
      { p: 0.8, y: 1 as const },
      { p: 0.2, y: 0 as const },
      { p: 0.1, y: 0 as const },
    ];
    assert.equal(aucOf(perfect), 1);
    assert.equal(aucOf([{ p: 0.5, y: 1 }, { p: 0.5, y: 0 }]), 0.5);
    assert.equal(aucOf([{ p: 0.1, y: 1 }, { p: 0.9, y: 0 }]), 0);
  });

  it('handles ties with average ranks', () => {
    // Two positives and two negatives, all tied: no discriminative information.
    assert.equal(aucOf([{ p: 0.5, y: 1 }, { p: 0.5, y: 1 }, { p: 0.5, y: 0 }, { p: 0.5, y: 0 }]), 0.5);
  });

  it('returns 0.5 when only one class is present', () => {
    assert.equal(aucOf([{ p: 0.9, y: 1 }, { p: 0.1, y: 1 }]), 0.5);
  });
});

describe('the heuristic prior is a real baseline', () => {
  it('scores well above chance on the corpus features', () => {
    // If the prior were weak, "the learned scorer beats it" would be a trivial
    // claim. It is strong, which is the honest reason the curve is shallow.
    const scores = EVAL_TASKS.map((t) => {
      const f = extractFeatures({
        task: t.task,
        ...(t.files ? { files: t.files } : {}),
        ...(t.constraints ? { constraints: t.constraints } : {}),
      });
      return { p: heuristicScore(f).pLocalSuccess, y: (t.expected === 'local' ? 1 : 0) as 0 | 1 };
    });
    assert.ok(aucOf(scores) > 0.9, 'the heuristic should separate the corpus well');
  });
});
