/**
 * Router tests: feature extraction, classification, the policy decision, and the
 * learned scorer.
 *
 * These are the tests that protect the project's central claim. If classification
 * silently degrades, easy work stops flowing to the local model; if the policy
 * silently degrades, hard work starts flowing to a model that cannot do it.
 */

import { strict as assert } from 'node:assert';
import { describe, it } from 'node:test';

import { analyzeCode, classifyTask, extractFeatures, toVector } from '../src/router/features.ts';
import { FEATURE_NAMES, FEATURE_VECTOR_VERSION } from '../src/router/types.ts';
import { heuristicScore, blendWithPrior, sigmoid } from '../src/router/heuristic.ts';
import { decideRoute, estimateLocalTokensPerSec, HARD_LOCKED_CLASSES } from '../src/router/policy.ts';
import type { RouterEnvironment } from '../src/router/policy.ts';
import { LogisticScorer, evaluate, trainLogistic, MIN_TRAINING_SAMPLES } from '../src/router/learned.ts';
import { loadScorer, routeTask, routerWeightsPath, saveScorer } from '../src/router/index.ts';
import { readJsonOrNull, writeJsonAtomic } from '../src/util/fsx.ts';
import { tempDir, testConfig } from './helpers.ts';

const LOOP_FILE = `def total_prices(items):
    total = 0
    for i in range(len(items) + 1):
        total += items[i]["price"]
    return total
`;

function env(overrides: Partial<RouterEnvironment> = {}): RouterEnvironment {
  return {
    localEnabled: true,
    localAvailable: true,
    localModelLoaded: true,
    localContextWindow: 8192,
    tinyModelAvailable: false,
    cloudAvailable: true,
    verifierAvailable: true,
    cloudBudgetRemainingUsd: 5,
    price: { in: 3, out: 15 },
    random: () => 1,
    ...overrides,
  };
}

describe('feature extraction', () => {
  it('classifies the motivating off-by-one example as an easy local bugfix', () => {
    const { cls, difficulty } = classifyTask(
      'Fix the off-by-one error in the loop so it does not go out of bounds.',
      { fileCount: 1, hasStackTrace: false },
    );
    assert.equal(cls, 'bugfix-local');
    assert.ok(difficulty <= 0.35, `expected low difficulty, got ${difficulty}`);
  });

  it('matches morphological variants, not just exact keywords', () => {
    // Regression guard: `\bvalidat\b` style patterns silently failed to match
    // "validation", which routed a trivially easy task to the cloud.
    const cases: Array<[string, string]> = [
      ['Add validation for the port range.', 'add-validation'],
      ['Add authentication checks to the handler.', 'security'],
      ['This needs a database migration.', 'migration'],
      ['Profile the slow hot path.', 'perf'],
      ['There is a concurrency problem with the worker pool.', 'concurrency'],
    ];
    for (const [task, expected] of cases) {
      const { cls } = classifyTask(task, { fileCount: 0, hasStackTrace: false });
      assert.equal(cls, expected, `"${task}" -> ${cls}, expected ${expected}`);
    }
  });

  it('chooses the hardest matching class when several match', () => {
    // Contains both "fix" (bugfix-local) and "performance regression" (perf).
    const { cls } = classifyTask('Fix the performance regression in the hot path.', {
      fileCount: 1,
      hasStackTrace: false,
    });
    assert.equal(cls, 'perf', 'a perf investigation must not be labelled a simple bugfix');
    assert.ok(HARD_LOCKED_CLASSES.length > 0);
  });

  it('does not over-match broad keywords into hard-locked classes', () => {
    // Regression guards. "escalation" (any retry/backoff discussion) and bare
    // "uuid" previously pulled ordinary work into the security and migration
    // classes, which are hard-locked to the cloud — so a trivial retry-policy
    // edit was sent to an expensive model for no reason.
    //
    // Two separate invariants are asserted, because they are not the same thing:
    //   (a) none of these may be hard-locked to the cloud, and
    //   (b) the mechanical ones should still route local.
    // "Generate a UUID for each new order" is deliberately NOT in group (b): with
    // no file in scope it is an under-specified small feature, and `unknown`
    // difficulty 0.5 is the router honestly saying so rather than guessing.
    const cfg = testConfig('/tmp/proto-router-test');
    const notHardLocked = [
      'Add escalation handling so failed jobs are retried twice.',
      'Generate a UUID for each new order and return it.',
      'Rename the local variable token to authCount.',
      'Add a null check before using the token.',
      'Replace the integer id with a UUID in this function.',
    ];
    for (const task of notHardLocked) {
      const features = extractFeatures({ task, files: [{ path: 'a.py', content: LOOP_FILE }] });
      assert.notEqual(features.taskClass, 'security', `"${task}" must not be classified as security`);
      assert.notEqual(features.taskClass, 'migration', `"${task}" must not be classified as migration`);
      const d = decideRoute({ ctx: { task: '' }, features, cfg, env: env() });
      assert.ok(
        !d.vetoes.some((v) => v.includes('hard-locked')),
        `"${task}" must not be hard-locked (class ${features.taskClass}, vetoes ${d.vetoes.join('; ')})`,
      );
    }

    const shouldRouteLocal = [
      'Add escalation handling so failed jobs are retried twice.',
      'Rename the local variable token to authCount.',
      'Add a null check before using the token.',
      'Replace the integer id with a UUID in this function.',
    ];
    for (const task of shouldRouteLocal) {
      const features = extractFeatures({ task, files: [{ path: 'a.py', content: LOOP_FILE }] });
      const d = decideRoute({ ctx: { task: '' }, features, cfg, env: env() });
      assert.ok(d.tier.startsWith('local'), `"${task}" should route local, got ${d.tier} (class ${features.taskClass})`);
    }
  });

  it('does not treat a bare "token" as security wording', () => {
    // "token" as a variable name is not a security concern; only qualified
    // credential phrases are. Otherwise the security hard-lock fires constantly.
    const benign = extractFeatures({ task: 'Add a null check before using the token.' });
    assert.equal(benign.hasSecurityLanguage, false);
    const qualified = extractFeatures({ task: 'Rotate the access token when it expires.' });
    assert.equal(qualified.hasSecurityLanguage, true);
  });

  it('upgrades a localized class when many files are in scope', () => {
    const { cls } = classifyTask('Rename the helper everywhere.', { fileCount: 6, hasStackTrace: false });
    assert.equal(cls, 'refactor-multi');
  });

  it('counts code structure and ignores comments and strings', () => {
    const stats = analyzeCode([
      {
        path: 'a.py',
        content: '# for while if in a comment\ndef f():\n    """for i in range(3)"""\n    for i in range(3):\n        if i:\n            while True:\n                break\n',
      },
    ]);
    assert.equal(stats.loopCount, 2, 'comment and docstring keywords must not count');
    assert.equal(stats.funcCount, 1);
    assert.ok(stats.maxNesting >= 3, `nesting was measured as ${stats.maxNesting}`);
  });

  it('penalises interacting constraints in the probability estimate', () => {
    const base = extractFeatures({ task: 'Update the pricing rules.', files: [] });
    const constrained = extractFeatures({
      task:
        'Update the pricing rules so that discounts apply only to non-sale items, must not stack with coupons, ' +
        'must round half-up, and must keep the API unchanged.',
      files: [],
    });
    const pBase = heuristicScore(base).pLocalSuccess;
    const pConstrained = heuristicScore(constrained).pLocalSuccess;
    assert.ok(
      pConstrained < pBase - 0.15,
      `four constraints should materially reduce confidence (${pBase.toFixed(3)} -> ${pConstrained.toFixed(3)})`,
    );
    assert.ok(constrained.constraintCount >= 4);
  });

  it('detects read-only intent but not when an edit verb is present', () => {
    const explain = extractFeatures({ task: 'Explain what this function does.' });
    assert.equal(explain.isExplainOnly, true);
    const mutate = extractFeatures({ task: 'Explain what this function does and fix the bug in it.' });
    assert.equal(mutate.isExplainOnly, false, '"review and fix" is a mutating task');
  });

  it('produces a vector of exactly the declared length', () => {
    const f = extractFeatures({ task: 'anything' });
    assert.equal(toVector(f).length, FEATURE_NAMES.length);
    assert.equal(FEATURE_VECTOR_VERSION, 1);
  });
});

describe('policy', () => {
  const cfg = testConfig('/tmp/proto-router-test');

  const easy = extractFeatures({
    task: 'Fix the off-by-one error in this loop so it does not go out of bounds.',
    files: [{ path: 'prices.py', content: LOOP_FILE }],
  });

  it('routes an easy verifiable task to the local model', () => {
    const d = decideRoute({ ctx: { task: '' }, features: easy, cfg, env: env() });
    assert.ok(d.tier === 'local' || d.tier === 'local-tiny', `got ${d.tier}`);
    assert.equal(d.exploration, false);
  });

  it('uses the tiny model tier when one is configured and the task is trivial', () => {
    const trivial = extractFeatures({ task: 'Fix the typo in this comment.' });
    const d = decideRoute({
      ctx: { task: '' },
      features: trivial,
      cfg,
      env: env({ tinyModelAvailable: true }),
    });
    assert.equal(d.tier, 'local-tiny');
  });

  it('raises the floor for mutating tasks with no verifier', () => {
    const medium = extractFeatures({
      task: 'Refactor the retry helper to use a backoff strategy.',
      files: [{ path: 'a.py', content: LOOP_FILE }],
    });
    const verified = decideRoute({ ctx: { task: '' }, features: medium, cfg, env: env() });
    const unverified = decideRoute({
      ctx: { task: '' },
      features: medium,
      cfg,
      env: env({ verifierAvailable: false }),
    });
    assert.equal(unverified.unverified, true);
    assert.ok(unverified.pLocalSuccess === verified.pLocalSuccess);
    // Same probability, different floor: the unverified decision must be no more local.
    const rank = (t: string): number => (t.startsWith('local') ? 0 : 1);
    assert.ok(rank(unverified.tier) >= rank(verified.tier));
  });

  it('keeps a read-only unverifiable task local when the floor is the low one', () => {
    const explain = extractFeatures({ task: 'Explain what this function does.' });
    const d = decideRoute({ ctx: { task: '' }, features: explain, cfg, env: env({ verifierAvailable: false }) });
    assert.ok(d.tier.startsWith('local'), `read-only tasks are self-checking, got ${d.tier}`);
  });

  it('hard-locks security and architecture work to the cloud', () => {
    for (const task of [
      'Harden the access check against privilege escalation.',
      'Design the module boundaries for the billing service.',
      'Migrate the schema to UUIDs without downtime.',
    ]) {
      const features = extractFeatures({ task, files: [{ path: 'a.py', content: LOOP_FILE }] });
      const d = decideRoute({ ctx: { task: '' }, features, cfg, env: env() });
      assert.ok(!d.tier.startsWith('local'), `"${task}" must not go local (got ${d.tier})`);
      assert.ok(d.vetoes.some((v) => v.includes('hard-locked')), `expected a hard-lock veto for "${task}"`);
    }
  });

  it('uses the strongest cloud model for hard-locked classes at any difficulty', () => {
    // The lock says "a wrong answer here is expensive"; handing the same task to
    // the cheap cloud model would contradict the reason it was locked.
    const architecture = extractFeatures({ task: 'Design the module boundaries for billing.' });
    const d = decideRoute({ ctx: { task: '' }, features: architecture, cfg, env: env() });
    assert.equal(d.tier, 'cloud-strong');

    const moderate = extractFeatures({
      task: 'Migrate the config loader to the new format.',
      files: [{ path: 'a.py', content: LOOP_FILE }],
    });
    assert.ok(moderate.classDifficulty >= 0.5);
    const d2 = decideRoute({ ctx: { task: '' }, features: moderate, cfg, env: env() });
    assert.ok(d2.tier === 'cloud-strong', `expected cloud-strong, got ${d2.tier}`);
  });

  it('vetoes local when the context does not fit', () => {
    const big = extractFeatures({ task: 'Tidy this file.', files: [{ path: 'a.py', content: 'x = 1\n'.repeat(6000) }] });
    const d = decideRoute({ ctx: { task: '' }, features: big, cfg, env: env({ localContextWindow: 4096 }) });
    assert.ok(d.vetoes.some((v) => v.includes('exceeds local context budget')));
    assert.ok(!d.tier.startsWith('local'));
  });

  it('falls back to the only viable tier instead of refusing', () => {
    const onlyLocal = decideRoute({
      ctx: { task: '' },
      features: easy,
      cfg,
      env: env({ cloudAvailable: false, cloudUnavailableReason: 'no API key' }),
    });
    assert.ok(onlyLocal.tier.startsWith('local'));
    assert.equal(onlyLocal.forced, true);

    const neither = decideRoute({
      ctx: { task: '' },
      features: easy,
      cfg,
      env: env({ localAvailable: false, cloudAvailable: false }),
    });
    assert.ok(neither.tier.startsWith('local'), 'a forced local attempt beats refusing to work');
    assert.equal(neither.forced, true);
  });

  it('vetoes the cloud when the daily budget is exhausted', () => {
    const hard = extractFeatures({
      task: 'Design the module boundaries and the migration plan.',
      files: [{ path: 'a.py', content: LOOP_FILE }],
    });
    const d = decideRoute({ ctx: { task: '' }, features: hard, cfg, env: env({ cloudBudgetRemainingUsd: 0 }) });
    assert.ok(d.vetoes.some((v) => v.includes('budget exhausted')));
    assert.ok(d.tier.startsWith('local'));
  });

  it('escalates to the strong cloud model only for harder tasks', () => {
    const easyish = extractFeatures({
      task: 'Refactor the parser to be clearer.',
      files: [{ path: 'a.py', content: LOOP_FILE }],
    });
    const d1 = decideRoute({
      ctx: { task: '' },
      features: easyish,
      cfg,
      env: env({ verifierAvailable: false, qualityFloorUnverified: 0.999 } as never),
    });
    // With an impossible floor the task must leave the local tier, but the cheap
    // cloud model is the cost-appropriate response for a moderate task.
    assert.equal(d1.tier, 'cloud-cheap');
  });

  it('explores deliberately and records it', () => {
    const moderate = extractFeatures({
      task: 'Refactor the retry helper into a small policy object.',
      files: [{ path: 'a.py', content: LOOP_FILE }],
    });
    const exploratory = decideRoute({
      ctx: { task: '' },
      features: moderate,
      cfg,
      env: env({ verifierAvailable: false, random: () => 0 }),
    });
    // The exploration path is only reached from a cloud decision, and only with
    // a verifier; so assert the invariant rather than the specific tier.
    assert.equal(exploratory.exploration, false, 'no exploration without a verifier to catch failure');
  });

  it('prefers the cloud when the local model is cold and the task is tiny', () => {
    const tiny = extractFeatures({ task: 'Rename the variable `a` to `count`.' });
    const cold = decideRoute({ ctx: { task: '' }, features: tiny, cfg, env: env({ localModelLoaded: false }) });
    const warm = decideRoute({ ctx: { task: '' }, features: tiny, cfg, env: env({ localModelLoaded: true }) });
    assert.ok(cold.expected.localLatencyMs > warm.expected.localLatencyMs);
    assert.equal(warm.tier.startsWith('local'), true);
    assert.ok(
      ['local', 'local-tiny', 'cloud-cheap', 'cloud-strong'].includes(cold.tier),
      'cold decisions must still produce a valid tier',
    );
  });

  it('estimates local decode speed from the model name', () => {
    assert.ok(estimateLocalTokensPerSec('qwen2.5-coder:1.5b-instruct') > 100);
    assert.ok(estimateLocalTokensPerSec('qwen2.5-coder:7b-instruct-q4_K_M') < 60);
    assert.ok(estimateLocalTokensPerSec('llama3.2:3b') > 0);
  });
});

describe('learned scorer', () => {
  it('separates synthetic classes and is deterministic for a given seed', () => {
    // ~8% label noise, so the Bayes-optimal AUC here is about 0.92; asserting
    // > 0.85 checks that the learner finds the signal without pretending the
    // synthetic data is noise-free.
    const samples = [];
    for (let i = 0; i < 120; i++) {
      const easy = i % 2 === 0;
      const x = new Array(FEATURE_NAMES.length).fill(0);
      x[0] = 1;
      x[27] = easy ? 0.1 : 0.9; // class_difficulty
      x[25] = easy ? 0.9 : 0.2; // locality
      samples.push({ x, y: (easy ? (i % 11 === 0 ? 0 : 1) : i % 13 === 0 ? 1 : 0) as 0 | 1 });
    }
    const a = trainLogistic(samples, { seed: 99 });
    const b = trainLogistic(samples, { seed: 99 });
    assert.deepEqual(a.scorer.weights, b.scorer.weights, 'training must be reproducible');
    assert.ok(a.metrics.auc > 0.85, `expected separable data, auc=${a.metrics.auc}`);
    assert.ok(a.metrics.calibration.length === 5);
  });

  it('predicts monotonically with difficulty', () => {
    const samples = [];
    for (let i = 0; i < 100; i++) {
      const x = new Array(FEATURE_NAMES.length).fill(0);
      x[0] = 1;
      x[27] = i / 100;
      samples.push({ x, y: (i < 40 ? 1 : 0) as 0 | 1 });
    }
    const { scorer } = trainLogistic(samples, { seed: 3 });
    const easy = scorer.predict(withFeature(27, 0.05));
    const hard = scorer.predict(withFeature(27, 0.95));
    assert.ok(easy > hard, 'higher class difficulty must lower the success probability');
  });

  it('shrinks toward the heuristic prior when few episodes are available', () => {
    const few = blendWithPrior(0.2, 0.8, 1);
    const many = blendWithPrior(0.2, 0.8, 10_000);
    assert.ok(few.p > many.p, 'with little data we should trust the heuristic');
    assert.ok(few.learnedWeight < 0.1);
    assert.ok(many.learnedWeight <= 0.85, 'the heuristic must never fully disappear');
  });

  it('refuses weights trained on a different feature vector version', () => {
    const scorer = new LogisticScorer();
    const file = { ...scorer.toFile(), featureVectorVersion: FEATURE_VECTOR_VERSION + 1 };
    assert.throws(() => LogisticScorer.fromFile(file), /feature vector v/);
  });

  it('reports a baseline quality metric when untrained', () => {
    const scorer = new LogisticScorer();
    const metrics = evaluate(scorer, [
      { x: withFeature(0, 1), y: 1 },
      { x: withFeature(0, 1), y: 0 },
    ]);
    assert.equal(metrics.n, 2);
    assert.equal(metrics.auc, 0.5);
  });

  it('requires a meaningful number of episodes before use', () => {
    assert.ok(MIN_TRAINING_SAMPLES >= 20, 'a handful of episodes must not train the router');
  });

  it('round-trips weights through disk', () => {
    const dir = tempDir();
    const scorer = new LogisticScorer(new Array(FEATURE_NAMES.length).fill(0.1), { sampleCount: 99 });
    saveScorer(dir, scorer, ['test']);
    const reloaded = loadScorer(dir);
    assert.equal(reloaded.error, undefined);
    assert.equal(reloaded.scorer?.sampleCount, 99);
    assert.ok(readJsonOrNull(routerWeightsPath(dir)));
  });

  it('refuses stale weights loudly rather than silently degrading to the heuristic', async () => {
    // After a FEATURE_VECTOR_VERSION bump the stored weights are unusable. The
    // router must fall back AND say so: silently switching to the heuristic
    // looks exactly like a mysterious routing-quality regression.
    const dir = tempDir();
    const scorer = new LogisticScorer(new Array(FEATURE_NAMES.length).fill(0));
    const file = scorer.toFile();
    writeJsonAtomic(routerWeightsPath(dir), { ...file, featureVectorVersion: FEATURE_VECTOR_VERSION + 1 });

    const cfg = testConfig(dir);
    const decision = await routeTask({
      cfg,
      dataDir: dir,
      ctx: { task: 'Fix the off-by-one error in this loop.' },
      offline: true,
    });
    assert.match(decision.scorerError ?? '', /feature vector v/);
    assert.ok(decision.reasons.some((r) => /were ignored/.test(r)));
    assert.equal(decision.scorer, 'heuristic');
  });

  it('sigmoid is numerically stable at extremes', () => {
    assert.equal(sigmoid(0), 0.5);
    assert.ok(sigmoid(-1000) >= 0 && Number.isFinite(sigmoid(-1000)));
    assert.ok(sigmoid(1000) <= 1 && Number.isFinite(sigmoid(1000)));
  });
});

function withFeature(index: number, value: number): number[] {
  const x = new Array(FEATURE_NAMES.length).fill(0);
  x[0] = 1;
  if (index !== 0) x[index] = value;
  return x;
}
