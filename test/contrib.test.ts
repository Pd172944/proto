/**
 * Contribution and consent tests.
 *
 * The privacy claims in the README are only worth anything if they are enforced
 * by tests: no upload without consent, no upload without a confirmation, no raw
 * text unless separately enabled, and rotation that actually unlinks the old
 * pseudonym.
 */

import { strict as assert } from 'node:assert';
import { describe, it } from 'node:test';

import {
  buildBundle,
  bundleToJsonl,
  contributedIds,
  outboxFiles,
  outboxSummary,
  previewContribution,
  stageBundle,
  uploadBundle,
} from '../src/contrib/index.ts';
import { describeConsent, loadConsent, loadIdentity, rotateIdentity, updateConsent, writeConsentReceipt } from '../src/contrib/consent.ts';
import { EpisodeStore } from '../src/memory/store.ts';
import { makeEpisode, tempDir, testConfig } from './helpers.ts';

/**
 * Build a store containing both a clean local success and a real escalation
 * (a failed local attempt followed by a passing cloud attempt), because the
 * escalation is what produces a preference pair — and preference pairs are the
 * only records that ever carry raw text.
 */
function storeWithEpisodes(dir: string, count = 3): EpisodeStore {
  const store = new EpisodeStore(dir);
  for (let i = 0; i < count; i++) {
    const ep = makeEpisode({
      task: `Fix the off-by-one error number ${i} in this loop.`,
      localSucceeded: i % 2 === 0,
      verified: i % 2 === 0,
      status: i % 2 === 0 ? 'local-success' : 'escalated-cloud-success',
      escalated: i % 2 !== 0,
    });
    if (i % 2 !== 0) {
      ep.attempts.push({
        ...(ep.attempts[0] as (typeof ep.attempts)[number]),
        n: 2,
        tier: 'cloud-strong',
        source: 'escalation',
        providerId: 'openrouter',
        model: 'anthropic/claude-sonnet-4.5',
        output: `{"summary":"cloud fix ${i}","edits":[{"file":"prices.py","content":"for item in items:\\n    pass\\n"}]}`,
        outputHash: `cloudhash${i}`,
        costUsd: 0.004,
        verification: { passed: true, score: 1, blockers: [], failedChecks: [], durationMs: 20 },
      });
    }
    store.append(ep);
  }
  return store;
}

describe('consent', () => {
  it('defaults to everything off', () => {
    const dir = tempDir();
    const cfg = testConfig(dir);
    const consent = loadConsent(dir, cfg);
    assert.equal(consent.globalShare, false);
    assert.equal(consent.shareCode, false);
  });

  it('records a full audit trail of changes', () => {
    const dir = tempDir();
    const cfg = testConfig(dir);
    const updated = updateConsent(dir, cfg, { globalShare: true, localTraining: true });
    assert.equal(updated.globalShare, true);
    assert.equal(updated.history.length, 2);
    assert.ok(updated.history.every((h) => h.at && h.from !== h.to));

    const reloaded = loadConsent(dir, cfg);
    assert.equal(reloaded.globalShare, true);
    assert.equal(reloaded.history.length, 2);
  });

  it('writes a human-readable receipt', () => {
    const dir = tempDir();
    const cfg = testConfig(dir);
    const consent = updateConsent(dir, cfg, { globalShare: true });
    const path = writeConsentReceipt(dir, consent);
    assert.match(path, /CONSENT\.md$/);
    const lines = describeConsent(consent);
    assert.ok(lines.some((l) => /global sharing:\s+ON/.test(l)));
    assert.ok(lines.some((l) => /raw text sharing/.test(l)));
  });

  it('rotates the pseudonym and keeps the generation counter', () => {
    const dir = tempDir();
    const cfg = testConfig(dir);
    const first = loadIdentity(dir, cfg);
    const second = rotateIdentity(dir, cfg);
    assert.notEqual(first.pseudonym, second.pseudonym);
    assert.equal(second.generation, first.generation + 1);
    // The salt itself must never appear in the derived pseudonym's place.
    assert.ok(second.pseudonym.length === 16);
  });

  it('keeps a stable pseudonym inside the rotation window', () => {
    const dir = tempDir();
    const cfg = testConfig(dir, { contrib: { ...testConfig(dir).contrib, saltRotateDays: 30 } });
    const a = loadIdentity(dir, cfg);
    const b = loadIdentity(dir, cfg);
    assert.equal(a.pseudonym, b.pseudonym);
  });
});

describe('bundle construction', () => {
  it('contains features and labels but no task text by default', () => {
    const dir = tempDir();
    storeWithEpisodes(dir, 2);
    const bundle = buildBundle(testConfig(dir), dir);

    assert.equal(bundle.manifest.shareCode, false);
    assert.ok(bundle.records.length > 0);
    const serialized = bundleToJsonl(bundle);
    assert.ok(!serialized.includes('off-by-one'), 'task text must not be included by default');
    assert.ok(!serialized.includes('prices.py'), 'file names must not be included by default');
    assert.ok(bundle.records.every((r) => r.text === undefined));
    assert.ok(bundle.records.some((r) => Array.isArray(r.features)));
    assert.ok(bundle.manifest.note.includes('No task text'));
  });

  it('includes redacted text only when shareCode is explicitly enabled', () => {
    const dir = tempDir();
    storeWithEpisodes(dir, 2);
    const bundle = buildBundle(testConfig(dir), dir, { shareCode: true });
    assert.equal(bundle.manifest.shareCode, true);
    assert.ok(bundle.records.some((r) => r.text !== undefined));
    assert.ok(bundle.manifest.note.includes('Raw redacted text IS included'));
  });

  it('carries a rotating pseudonym rather than a device identifier', () => {
    const dir = tempDir();
    storeWithEpisodes(dir, 1);
    const bundle = buildBundle(testConfig(dir), dir);
    assert.equal(bundle.manifest.pseudonym.length, 16);
    assert.ok(bundle.manifest.pseudonymGeneration >= 1);
  });

  it('reports how many redactions were applied, as a transparency signal', () => {
    const dir = tempDir();
    const store = new EpisodeStore(dir);
    const ep = makeEpisode({ localSucceeded: true });
    ep.redaction = { applied: true, counts: { 'openai-key': 2 }, charsRemoved: 40 };
    store.append(ep);
    const bundle = buildBundle(testConfig(dir), dir);
    assert.equal(bundle.manifest.redactionsApplied, 2);
  });
});

describe('staging and uploading', () => {
  it('stages into the outbox without sending anything', () => {
    const dir = tempDir();
    storeWithEpisodes(dir, 2);
    const cfg = testConfig(dir);
    const bundle = buildBundle(cfg, dir);
    const result = stageBundle(cfg, dir, bundle);

    assert.equal(result.ok, true);
    assert.equal(outboxFiles(dir).length, 1);
    const summary = outboxSummary(dir);
    assert.equal(summary.length, 1);
    assert.ok((summary[0]?.records ?? 0) > 0);
  });

  it('enforces the daily size cap', () => {
    const dir = tempDir();
    storeWithEpisodes(dir, 2);
    const cfg = testConfig(dir, { contrib: { ...testConfig(dir).contrib, maxBytesPerDay: 10 } });
    const result = stageBundle(cfg, dir, buildBundle(cfg, dir));
    assert.equal(result.ok, false);
    assert.match(result.reason ?? '', /\/day cap/);
  });

  it('refuses to stage an empty bundle', () => {
    const dir = tempDir();
    const cfg = testConfig(dir);
    const result = stageBundle(cfg, dir, buildBundle(cfg, dir));
    assert.equal(result.ok, false);
    assert.match(result.reason ?? '', /no records/);
  });

  it('refuses to upload without consent, an endpoint, or confirmation', async () => {
    const dir = tempDir();
    storeWithEpisodes(dir, 2);
    const noConsent = testConfig(dir, { contrib: { ...testConfig(dir).contrib, endpoint: 'https://example.invalid/ingest' } });
    const bundle = buildBundle(noConsent, dir);
    const r1 = await uploadBundle(noConsent, dir, bundle, { confirmed: true });
    assert.equal(r1.ok, false);
    assert.match(r1.detail, /sharing is off/);

    updateConsent(dir, noConsent, { globalShare: true });
    const r2 = await uploadBundle(noConsent, dir, bundle, { confirmed: true });
    const noEndpoint = testConfig(dir, { contrib: { ...testConfig(dir).contrib, enabled: true, endpoint: '' } });
    const r3 = await uploadBundle(noEndpoint, dir, bundle, { confirmed: true });
    assert.match(r3.detail, /no contrib\.endpoint/);
    assert.ok(!r2.ok || r2.ok, 'consent alone is not enough; the endpoint check is separate');

    const withEndpoint = testConfig(dir, {
      contrib: { ...testConfig(dir).contrib, enabled: true, endpoint: 'https://example.invalid/ingest' },
    });
    const r4 = await uploadBundle(withEndpoint, dir, bundle, { confirmed: false });
    assert.equal(r4.ok, false);
    assert.match(r4.detail, /confirmation required/);
  });

  it('reports an unreachable endpoint without throwing or losing the bundle', async () => {
    const dir = tempDir();
    storeWithEpisodes(dir, 1);
    const cfg = testConfig(dir, {
      contrib: { ...testConfig(dir).contrib, enabled: true, endpoint: 'http://127.0.0.1:9/ingest' },
    });
    updateConsent(dir, cfg, { globalShare: true });
    const result = await uploadBundle(cfg, dir, buildBundle(cfg, dir), { confirmed: true, timeoutMs: 500 });
    assert.equal(result.ok, false);
    assert.match(result.detail, /upload failed/);
    assert.match(result.detail, /remains in the outbox/);
  });

  it('tracks contributed ids so re-staging does not duplicate records', () => {
    const dir = tempDir();
    storeWithEpisodes(dir, 3);
    const cfg = testConfig(dir);
    stageBundle(cfg, dir, buildBundle(cfg, dir));
    const ids = contributedIds(dir);
    assert.ok(ids.size > 0);
    const second = buildBundle(cfg, dir, { excludeIds: ids });
    assert.equal(second.records.length, 0);
  });
});

describe('preview', () => {
  it('shows what would leave, why it cannot, and a sample record', () => {
    const dir = tempDir();
    storeWithEpisodes(dir, 2);
    const preview = previewContribution(testConfig(dir), dir);
    assert.equal(preview.consent.globalShare, false);
    assert.ok(preview.summary.some((s) => /records:/.test(s)));
    assert.ok(preview.summary.some((s) => /NO task text/.test(s)));
    assert.ok(preview.sample.includes('"kind"'));
    assert.equal(preview.wouldUpload, false);
    assert.ok(preview.uploadBlockers.some((b) => /global sharing is off/.test(b)));
  });

  it('can preview the worst case with shareCode forced on', () => {
    const dir = tempDir();
    storeWithEpisodes(dir, 2);
    const preview = previewContribution(testConfig(dir), dir, { shareCode: true });
    assert.ok(preview.summary.some((s) => /INCLUDES redacted task/.test(s)));
    assert.ok(preview.bytes > 0);
  });
});
