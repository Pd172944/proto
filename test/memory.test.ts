/**
 * Memory layer tests: the episode store, the reward function, the router label,
 * and dataset construction.
 *
 * The theme is that the *labels* are what the whole learning story rests on, so
 * the tests pin down when a label exists, when it must be null, and how explicit
 * user feedback changes the reward.
 */

import { strict as assert } from 'node:assert';
import { describe, it } from 'node:test';

import { EpisodeStore } from '../src/memory/store.ts';
import { behaviorPropensity, computeReward, ipsWeight, MAX_IPS_WEIGHT, routerLabel, REWARD_VERSION } from '../src/memory/reward.ts';
import { buildDatasets } from '../src/memory/datasets.ts';
import { makeEpisode, tempDir, testConfig } from './helpers.ts';
import { readJsonl, readTextOrNull } from '../src/util/fsx.ts';

describe('episode store', () => {
  it('round-trips episodes and finds them by id', () => {
    const dir = tempDir();
    const store = new EpisodeStore(dir);
    const ep = makeEpisode({ ts: '2025-06-01T10:00:00.000Z' });
    store.appendBounded(ep, 1024 * 1024);
    const all = store.readAll();
    assert.equal(all.length, 1);
    assert.equal(all[0]?.id, ep.id);
    assert.equal(store.findById(ep.id)?.id, ep.id);
    assert.equal(store.findById('nope'), null);
  });

  it('refuses to persist an episode from a different schema version', () => {
    const dir = tempDir();
    const store = new EpisodeStore(dir);
    const ep = { ...makeEpisode(), schemaVersion: 99 };
    assert.throws(() => store.append(ep), /schemaVersion/);
  });

  it('merges feedback without rewriting the append-only episode log', () => {
    const dir = tempDir();
    const store = new EpisodeStore(dir);
    const ep = makeEpisode();
    store.append(ep);
    store.addFeedback({ episodeId: ep.id, signal: 'reject', note: 'not what I asked' });

    const loaded = store.findById(ep.id);
    assert.equal(loaded?.feedback?.signal, 'reject');
    assert.equal(loaded?.feedback?.note, 'not what I asked');
    // The shard itself must be untouched.
    const raw = readJsonl<{ feedback?: unknown }>(store.shards()[0] as string);
    assert.equal(raw[0]?.feedback, undefined);
  });

  it('computes statistics that the CLI and router rely on', () => {
    const dir = tempDir();
    const store = new EpisodeStore(dir);
    store.append(makeEpisode({ localSucceeded: true, reward: 1.3 }));
    store.append(makeEpisode({ localSucceeded: false, status: 'escalated-cloud-success', escalated: true, tier: 'cloud-strong', reward: 0.4, cloudCostUsd: 0.01 }));
    store.append(makeEpisode({ localSucceeded: null, tier: 'cloud-strong', status: 'cloud-success', reward: 0.9, cloudCostUsd: 0.02 }));

    const stats = store.stats();
    assert.equal(stats.episodes, 3);
    assert.equal(stats.escalations, 1);
    assert.equal(stats.localAttempts, 2);
    assert.equal(stats.localSuccesses, 1);
    assert.equal(stats.localSuccessRate, 0.5);
    assert.ok(Math.abs(stats.cloudSpendUsd - 0.03) < 1e-9);
    assert.equal(stats.byStatus['escalated-cloud-success'], 1);
  });

  it('prunes only shards outside the retention window', () => {
    const dir = tempDir();
    const store = new EpisodeStore(dir);
    store.append(makeEpisode({ ts: '2020-01-01T00:00:00.000Z' }));
    store.append(makeEpisode({ ts: new Date().toISOString() }));
    const result = store.prune(30);
    assert.equal(result.removed.length, 1);
    assert.equal(store.readAll().length, 1);
  });

  it('honours the shard size cap by rotating', () => {
    const dir = tempDir();
    const store = new EpisodeStore(dir);
    const first = store.shardPath(new Date('2025-06-02T00:00:00Z'), 1);
    assert.match(first, /2025-06-02\.jsonl$/);
    // With a 1-byte cap, the shard is immediately "full" and the next call rotates.
    const ep = makeEpisode({ ts: '2025-06-02T00:00:00.000Z' });
    store.appendBounded(ep, 1);
    const second = store.shardPath(new Date('2025-06-02T00:00:00Z'), 1);
    assert.match(second, /2025-06-02\.1\.jsonl$/);
  });
});

describe('reward', () => {
  const base = {
    status: 'local-success' as const,
    escalated: false,
    localAttempted: true,
    localPassed: true,
    finalScore: 1,
    refusal: false,
    envelopeDrift: false,
    exploration: false,
    hadBlockers: false,
    cloudCostUsd: 0,
  };

  it('rewards a clean local success above everything else', () => {
    const good = computeReward(base);
    const escalated = computeReward({
      ...base,
      status: 'escalated-cloud-success',
      escalated: true,
      localPassed: false,
      cloudCostUsd: 0.01,
    });
    assert.ok(good.reward > escalated.reward);
    assert.ok(good.reward > 1, 'a clean local win should exceed the plain verified reward');
    assert.equal(good.version, REWARD_VERSION);
  });

  it('penalises escalation more than it rewards the cheap-success bonus', () => {
    const bonus = computeReward(base).components.find((c) => c.name === 'cheapSuccessBonus')?.value ?? 0;
    const penalty = computeReward({ ...base, escalated: true, status: 'escalated-cloud-success' }).components.find(
      (c) => c.name === 'escalationPenalty',
    )?.value ?? 0;
    assert.ok(
      Math.abs(penalty) > bonus,
      'learning when local fails must matter more than preferring local',
    );
  });

  it('gives a failed local attempt a negative reward even after a cloud rescue', () => {
    const r = computeReward({
      ...base,
      status: 'escalated-cloud-success',
      escalated: true,
      localPassed: false,
      cloudCostUsd: 0.02,
    });
    assert.ok(r.reward < 1.05, `rescue reward should be tempered, got ${r.reward}`);
    assert.ok(r.components.some((c) => c.name === 'cost' && c.value < 0));
  });

  it('pays a bonus for a successful exploration but only a small cost for a failed one', () => {
    const expWin = computeReward({ ...base, exploration: true });
    const expLoss = computeReward({ ...base, exploration: true, localPassed: false, status: 'failed', escalated: false, finalScore: 0 });
    assert.ok(expWin.reward > computeReward(base).reward);
    const cost = Math.abs(expLoss.components.find((c) => c.name === 'explorationCost')?.value ?? 0);
    assert.ok(cost <= 0.05, 'exploration must never dominate the objective');
  });

  it('reflects explicit user feedback', () => {
    const accepted = computeReward({ ...base, feedback: 'accept' });
    const rejected = computeReward({ ...base, feedback: 'reject' });
    assert.ok(accepted.reward > computeReward(base).reward);
    assert.ok(rejected.reward < computeReward(base).reward);
  });

  it('punishes refusals and envelope drift', () => {
    const r = computeReward({ ...base, status: 'failed', refusal: true, envelopeDrift: true, finalScore: 0, localPassed: false });
    assert.ok(r.reward < -0.4, `got ${r.reward}`);
  });

  it('clamps to the documented range', () => {
    const worst = computeReward({
      ...base,
      status: 'failed',
      refusal: true,
      envelopeDrift: true,
      hadBlockers: true,
      feedback: 'reject',
      cloudCostUsd: 100,
      finalScore: 0,
      localPassed: false,
    });
    assert.ok(worst.reward >= -1.5);
  });
});

describe('router label', () => {
  it('is null when local was never attempted', () => {
    const ep = makeEpisode({ tier: 'cloud-strong', localSucceeded: null });
    ep.attempts = ep.attempts.map((a) => ({ ...a, tier: 'cloud-strong' as const }));
    assert.equal(routerLabel(ep), null);
  });

  it('is null when the local attempt was never verified', () => {
    const ep = makeEpisode({ localSucceeded: null });
    ep.attempts[0]!.verification = null;
    assert.equal(routerLabel(ep), null, 'an unverified attempt teaches nothing');
  });

  it('uses verification, not escalation, as the label', () => {
    const passed = makeEpisode({ localSucceeded: true });
    assert.equal(routerLabel(passed)?.y, 1);
    const failed = makeEpisode({ localSucceeded: false, status: 'escalated-cloud-success', escalated: true, verified: false });
    assert.equal(routerLabel(failed)?.y, 0);
  });

  it('up-weights exploration episodes, because they carry the only counterfactual labels', () => {
    // This is the direction that matters. An exploration episode is one the
    // policy would NOT have sent local; it is rare (probability epsilon) and it
    // is the only evidence about tasks the router currently avoids. It must be
    // scaled UP by ~1/epsilon to stand in for the population it represents.
    // An earlier revision used the propensity itself as the weight, which
    // down-weighted exactly these samples by ~16x.
    const explored = makeEpisode({ behaviorPropensity: 0.06, exploration: true });

    const wExplored = routerLabel(explored)?.w ?? 0;
    assert.ok(Math.abs(wExplored - 1 / 0.06) < 1e-6, `expected ~16.67, got ${wExplored}`);
    assert.ok(wExplored > 1, 'exploration must be up-weighted, not down-weighted');
    // A greedy cloud decision gets a weight just above 1, so exploration is
    // weighted far more heavily than the common case.
    assert.ok(wExplored > ipsWeight(0.94) * 10);
    // And the cap must hold so one noisy label cannot dominate the fit.
    assert.ok(wExplored <= MAX_IPS_WEIGHT);
    assert.equal(ipsWeight(1e-9), MAX_IPS_WEIGHT, 'the IPS weight must be capped');
    assert.equal(ipsWeight(1), 1);
  });
});

describe('behaviour propensity', () => {
  const decision = (tier: string, exploration: boolean, forced = false): Parameters<typeof behaviorPropensity>[0] => ({
    tier: tier as never,
    reason: '',
    reasons: [],
    pLocalSuccess: 0.5,
    difficulty: 0.5,
    taskClass: 'local-edit',
    exploration,
    forced,
    unverified: false,
    scorer: 'heuristic',
    expectedLocalCostUsd: 0,
    expectedCloudCostUsd: 0.01,
    expectedLocalLatencyMs: 100,
    expectedCloudLatencyMs: 1000,
    vetoes: [],
  });

  it('is epsilon for an exploratory decision', () => {
    assert.equal(behaviorPropensity(decision('local', true), 0.06), 0.06);
  });

  it('is one for a forced decision: the policy had no choice', () => {
    assert.equal(behaviorPropensity(decision('local', false, true), 0.06), 1);
  });

  it('is one for a greedy local decision: nothing flips away from local', () => {
    assert.equal(behaviorPropensity(decision('local', false), 0.06), 1);
    assert.equal(behaviorPropensity(decision('local-tiny', false), 0.06), 1);
  });

  it('is 1 - epsilon for a greedy cloud decision, which exploration could have flipped', () => {
    assert.ok(Math.abs(behaviorPropensity(decision('cloud-strong', false), 0.06) - 0.94) < 1e-9);
    assert.ok(Math.abs(behaviorPropensity(decision('cloud-cheap', false), 0.06) - 0.94) < 1e-9);
  });

  it('never returns zero or a negative probability', () => {
    assert.ok(behaviorPropensity(decision('cloud-strong', false), 1) > 0);
    assert.ok(behaviorPropensity(decision('local', true), 0) > 0);
  });
});

describe('dataset construction', () => {
  it('builds SFT samples from verified local attempts', () => {
    const dir = tempDir();
    const store = new EpisodeStore(dir);
    store.append(makeEpisode({ localSucceeded: true }));
    const built = buildDatasets(store, testConfig(dir), { write: false });

    assert.equal(built.sft.samples.length, 1);
    const sample = built.sft.samples[0];
    assert.equal(sample?.messages.length, 3);
    assert.equal(sample?.messages[0]?.role, 'system');
    assert.equal(sample?.messages[2]?.role, 'assistant');
    assert.ok((sample?.messages[2]?.content.length ?? 0) > 0);
    assert.equal(sample?._meta.source, 'local-verified');
  });

  it('builds a DPO preference pair from a local failure rescued by the cloud', () => {
    const dir = tempDir();
    const store = new EpisodeStore(dir);
    const ep = makeEpisode({ localSucceeded: false, status: 'escalated-cloud-success', escalated: true, verified: false });
    // Add the successful cloud attempt that escalates in production.
    ep.attempts.push({
      ...(ep.attempts[0] as (typeof ep.attempts)[number]),
      n: 2,
      tier: 'cloud-strong',
      source: 'escalation',
      providerId: 'openrouter',
      model: 'anthropic/claude-sonnet-4.5',
      output: '{"summary":"cloud fix","edits":[{"file":"prices.py","content":"for item in items:\\n    pass\\n"}]}',
      outputHash: 'cloudhash',
      costUsd: 0.004,
      verification: { passed: true, score: 1, blockers: [], failedChecks: [], durationMs: 20 },
    });
    store.append(ep);

    const built = buildDatasets(store, testConfig(dir), { write: false });
    assert.equal(built.dpo.samples.length, 1);
    const pair = built.dpo.samples[0];
    assert.match(pair?.chosen ?? '', /cloud fix/);
    assert.match(pair?.rejected ?? '', /mock/);
    assert.equal(pair?._meta.origin, 'escalation');
  });

  it('never imitates an episode the user rejected', () => {
    const dir = tempDir();
    const store = new EpisodeStore(dir);
    const ep = makeEpisode({ localSucceeded: true });
    store.append(ep);
    store.addFeedback({ episodeId: ep.id, signal: 'reject' });
    const built = buildDatasets(store, testConfig(dir), { write: false });
    assert.equal(built.sft.samples.length, 0);
    assert.ok(built.sft.skipped.some((s) => /rejected/.test(s.reason)));
  });

  it('records why episodes were unusable instead of silently dropping them', () => {
    const dir = tempDir();
    const store = new EpisodeStore(dir);
    const ep = makeEpisode({ localSucceeded: true });
    ep.attempts[0]!.prompt = undefined;
    store.append(ep);
    const built = buildDatasets(store, testConfig(dir), { write: false });
    assert.equal(built.sft.samples.length, 0);
    assert.ok(built.stats.missingPrompts >= 1);
  });

  it('writes JSONL datasets and a manifest when asked', () => {
    const dir = tempDir();
    const store = new EpisodeStore(dir);
    store.append(makeEpisode({ localSucceeded: true }));
    store.append(makeEpisode({ localSucceeded: false, status: 'failed', verified: false }));
    const built = buildDatasets(store, testConfig(dir), { write: true });

    assert.equal(built.sft.written, true);
    const sft = readTextOrNull(built.sft.path) ?? '';
    assert.ok(sft.trim().split('\n').length >= 1);
    const routerRows = readJsonl<{ x: number[]; y: number }>(built.router.path);
    assert.equal(routerRows.length, 2);
    assert.ok((routerRows[0]?.x.length ?? 0) > 10);
    const manifest = readTextOrNull(`${built.dir}/manifest.json`);
    assert.ok(manifest && manifest.includes('builtAt'));
  });

  it('deduplicates identical router rows', () => {
    const dir = tempDir();
    const store = new EpisodeStore(dir);
    const ep = makeEpisode({ localSucceeded: true });
    store.append(ep);
    store.append({ ...ep, id: ep.id.replace(/.$/, 'Z') });
    const built = buildDatasets(store, testConfig(dir), { write: false });
    assert.equal(built.router.samples.length, 1);
    assert.ok(built.stats.duplicatesDropped >= 1);
  });

  it('respects the SFT cap', () => {
    const dir = tempDir();
    const store = new EpisodeStore(dir);
    for (let i = 0; i < 5; i++) {
      const ep = makeEpisode({ localSucceeded: true, task: `task number ${i}` });
      store.append(ep);
    }
    const built = buildDatasets(store, testConfig(dir), { write: false, maxSft: 2 });
    assert.equal(built.sft.samples.length, 2);
  });
});
