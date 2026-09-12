/**
 * Optional contribution of *derived* data to a shared pool.
 *
 * The default is maximally conservative and deliberately so. What leaves the
 * machine by default is:
 *
 *   - the router feature vectors (numbers), their verification labels, and the
 *     behaviour propensity,
 *   - preference *facts*: "on a task with these features, the local model failed
 *     verification and the cloud model passed",
 *   - content hashes (for deduplication) — never the content itself,
 *   - a rotating pseudonym.
 *
 * Raw task text, file contents and model outputs are included **only** if the
 * user separately enables `shareCode`, and even then they have already been
 * through the redactor at write time. This mirrors how federated learning
 * systems should behave: gradient-like summaries by default, raw data only on
 * explicit, separate consent.
 *
 * Nothing is ever uploaded automatically. `stage()` writes a bundle into the
 * outbox; `upload()` requires consent *and* (by default) a per-call confirmation.
 */

import { join } from 'node:path';
import { readdirSync, statSync } from 'node:fs';

import type { ProtoConfig } from '../config/schema.ts';
import { EpisodeStore } from '../memory/store.ts';
import { buildDatasets } from '../memory/datasets.ts';
import { describeConsent, loadConsent, loadIdentity } from './consent.ts';
import type { ConsentRecord } from './consent.ts';
import { HARNESS_VERSION } from '../version.ts';
import { ensureDir, formatBytes, readJsonl, readTextOrNull, writeTextAtomic } from '../util/fsx.ts';
import { contentId } from '../util/ids.ts';
import { request } from '../util/http.ts';
import { sha256Short } from '../util/text.ts';

export const CONTRIBUTION_VERSION = 1;

export interface ContributionRecord {
  kind: 'router' | 'preference';
  /** Content-addressed id for server-side dedup; not reversible. */
  id: string;
  taskClass: string;
  features?: number[];
  label?: 0 | 1;
  propensity?: number;
  tiers?: { rejected: string; chosen: string };
  /** Present only when shareCode is on. */
  text?: { prompt?: string; chosen?: string; rejected?: string };
}

export interface ContributionBundle {
  manifest: {
    version: number;
    createdAt: string;
    /** Rotating pseudonym; not a stable device id. */
    pseudonym: string;
    pseudonymGeneration: number;
    harnessVersion: string;
    platform: string;
    shareCode: boolean;
    counts: { router: number; preference: number };
    /** Total redactions applied to the source episodes, as a transparency signal. */
    redactionsApplied: number;
    note: string;
  };
  records: ContributionRecord[];
}

export interface BuildBundleOptions {
  /** Max records per bundle. */
  limit?: number;
  /** Override the shareCode config for this bundle (dry runs / previews). */
  shareCode?: boolean;
  since?: Date;
  /** Exclude episodes already contributed (by id). */
  excludeIds?: Set<string>;
}

export function buildBundle(cfg: ProtoConfig, dataDir: string, opts: BuildBundleOptions = {}): ContributionBundle {
  const limit = opts.limit ?? 500;
  const consent = loadConsent(dataDir, cfg);
  const shareCode = opts.shareCode ?? (consent.globalShare && consent.shareCode);
  const identity = loadIdentity(dataDir, cfg);
  const store = new EpisodeStore(dataDir);
  const episodes = store.readAll(opts.since ? { since: opts.since } : {});

  const datasets = buildDatasets(store, cfg, { write: false, maxRouter: limit * 2, maxDpo: limit });
  const byId = new Map(episodes.map((e) => [e.id, e]));
  const records: ContributionRecord[] = [];
  let redactionsApplied = 0;

  for (const ep of episodes) redactionsApplied += Object.values(ep.redaction?.counts ?? {}).reduce((a, b) => a + b, 0);

  for (const row of datasets.router.samples) {
    if (records.length >= limit) break;
    // Exclusion is keyed on the *record* id (content-addressed), not the episode
    // id: `contributedIds()` reads back what was staged, so the two must agree or
    // every re-stage would duplicate everything already sent.
    const id = contentId(`${row.id}:${row.x.join(',')}`);
    if (opts.excludeIds?.has(id)) continue;
    records.push({
      kind: 'router',
      id,
      taskClass: row.meta.taskClass,
      features: row.x.map(round4),
      label: row.y,
      propensity: round4(row.w),
    });
  }

  for (const pair of datasets.dpo.samples) {
    if (records.length >= limit) break;
    const ep = byId.get(pair._meta.episodeId);
    const id = contentId(`${pair._meta.episodeId}:${sha256Short(pair.prompt)}`);
    if (opts.excludeIds?.has(id)) continue;
    const record: ContributionRecord = {
      kind: 'preference',
      id,
      taskClass: ep?.decision.taskClass ?? 'unknown',
      tiers: { rejected: pair._meta.rejectedTier, chosen: pair._meta.chosenTier },
      ...(ep ? { features: ep.vector.map(round4) } : {}),
    };
    if (shareCode) {
      record.text = { prompt: pair.prompt, chosen: pair.chosen, rejected: pair.rejected };
    }
    records.push(record);
  }

  const counts = {
    router: records.filter((r) => r.kind === 'router').length,
    preference: records.filter((r) => r.kind === 'preference').length,
  };

  return {
    manifest: {
      version: CONTRIBUTION_VERSION,
      createdAt: new Date().toISOString(),
      pseudonym: identity.pseudonym,
      pseudonymGeneration: identity.generation,
      harnessVersion: HARNESS_VERSION,
      platform: `${process.platform}-${process.arch}`,
      shareCode,
      counts,
      redactionsApplied,
      note:
        'Derived routing features and verification labels only. ' +
        (shareCode
          ? 'Raw redacted text IS included because shareCode is enabled.'
          : 'No task text, file contents or model outputs are included.'),
    },
    records,
  };
}

function round4(n: number): number {
  return Math.round(n * 10_000) / 10_000;
}

export function bundleToJsonl(bundle: ContributionBundle): string {
  const lines = [JSON.stringify({ manifest: bundle.manifest })];
  for (const r of bundle.records) lines.push(JSON.stringify(r));
  return lines.join('\n') + '\n';
}

export function bundleBytes(bundle: ContributionBundle): number {
  return Buffer.byteLength(bundleToJsonl(bundle), 'utf8');
}

/* ------------------------------------------------------------------ */
/* Outbox                                                             */
/* ------------------------------------------------------------------ */

export function outboxDir(dataDir: string): string {
  return join(dataDir, 'contrib', 'outbox');
}

export function outboxFiles(dataDir: string): string[] {
  try {
    return readdirSync(outboxDir(dataDir))
      .filter((f) => f.endsWith('.jsonl'))
      .sort()
      .map((f) => join(outboxDir(dataDir), f));
  } catch {
    return [];
  }
}

export function bytesStagedToday(dataDir: string, now: Date = new Date()): number {
  const day = now.toISOString().slice(0, 10);
  let total = 0;
  for (const f of outboxFiles(dataDir)) {
    if (!f.includes(day)) continue;
    try {
      total += statSync(f).size;
    } catch {
      /* ignore */
    }
  }
  return total;
}

export interface StageResult {
  ok: boolean;
  path?: string;
  bytes: number;
  reason?: string;
}

/** Write a bundle into the outbox. Does not send anything. */
export function stageBundle(cfg: ProtoConfig, dataDir: string, bundle: ContributionBundle, now = new Date()): StageResult {
  const bytes = bundleBytes(bundle);
  const stagedToday = bytesStagedToday(dataDir, now);
  if (stagedToday + bytes > cfg.contrib.maxBytesPerDay) {
    return {
      ok: false,
      bytes,
      reason:
        `staging would exceed the ${formatBytes(cfg.contrib.maxBytesPerDay)}/day cap ` +
        `(${formatBytes(stagedToday)} already staged today)`,
    };
  }
  if (bundle.records.length === 0) {
    return { ok: false, bytes: 0, reason: 'nothing to stage: the bundle contains no records' };
  }
  const stamp = now.toISOString().replace(/[:.]/g, '-');
  const path = join(outboxDir(dataDir), `${stamp}-${bundle.manifest.pseudonym}.jsonl`);
  ensureDir(outboxDir(dataDir));
  writeTextAtomic(path, bundleToJsonl(bundle));
  return { ok: true, path, bytes };
}

/* ------------------------------------------------------------------ */
/* Preview                                                            */
/* ------------------------------------------------------------------ */

export interface PreviewResult {
  consent: ConsentRecord;
  consentLines: string[];
  bundle: ContributionBundle;
  bytes: number;
  summary: string[];
  /** A representative redacted record, so the user sees exactly what would go. */
  sample: string;
  wouldUpload: boolean;
  uploadBlockers: string[];
}

export function previewContribution(
  cfg: ProtoConfig,
  dataDir: string,
  opts: BuildBundleOptions = {},
): PreviewResult {
  const consent = loadConsent(dataDir, cfg);
  const bundle = buildBundle(cfg, dataDir, opts);
  const bytes = bundleBytes(bundle);

  const summary: string[] = [];
  summary.push(`records: ${bundle.records.length} (router ${bundle.manifest.counts.router}, preference ${bundle.manifest.counts.preference})`);
  summary.push(`bundle size: ${formatBytes(bytes)}`);
  summary.push(`pseudonym: ${bundle.manifest.pseudonym} (generation ${bundle.manifest.pseudonymGeneration}, rotates)`);
  summary.push(
    bundle.manifest.shareCode
      ? 'INCLUDES redacted task/output text (shareCode is on)'
      : 'contains NO task text, file contents or model outputs',
  );
  summary.push(`redactions applied across source episodes: ${bundle.manifest.redactionsApplied}`);

  const uploadBlockers: string[] = [];
  if (!consent.globalShare) uploadBlockers.push('global sharing is off (proto contrib consent --global on)');
  if (!cfg.contrib.endpoint) uploadBlockers.push('no contrib.endpoint configured, so there is nowhere to send it');
  if (bundle.records.length === 0) uploadBlockers.push('the bundle is empty');

  const sampleRecord = bundle.records.find((r) => r.kind === 'preference') ?? bundle.records[0];

  return {
    consent,
    consentLines: describeConsent(consent),
    bundle,
    bytes,
    summary,
    sample: sampleRecord ? JSON.stringify(sampleRecord, null, 2) : '(no records)',
    wouldUpload: uploadBlockers.length === 0,
    uploadBlockers,
  };
}

/* ------------------------------------------------------------------ */
/* Upload                                                             */
/* ------------------------------------------------------------------ */

export interface UploadResult {
  ok: boolean;
  detail: string;
  bytes: number;
  records: number;
}

/**
 * Send a bundle to the configured endpoint.
 *
 * Guarded by three independent conditions, each of which alone is enough to stop
 * the upload: consent, a configured endpoint, and (by default) an explicit
 * per-call confirmation from the caller. Failures are non-fatal and the bundle
 * stays in the outbox for inspection.
 */
export async function uploadBundle(
  cfg: ProtoConfig,
  dataDir: string,
  bundle: ContributionBundle,
  opts: { confirmed: boolean; timeoutMs?: number; now?: Date } = { confirmed: false },
): Promise<UploadResult> {
  const consent = loadConsent(dataDir, cfg);
  const bytes = bundleBytes(bundle);
  const records = bundle.records.length;

  if (!consent.globalShare) {
    return { ok: false, detail: 'global sharing is off; nothing was sent', bytes: 0, records };
  }
  if (!cfg.contrib.endpoint) {
    return { ok: false, detail: 'no contrib.endpoint configured; nothing was sent', bytes: 0, records };
  }
  if (cfg.contrib.requireConfirmation && !opts.confirmed) {
    return { ok: false, detail: 'confirmation required; re-run with --yes to send', bytes: 0, records };
  }
  if (records === 0) {
    return { ok: false, detail: 'bundle is empty; nothing was sent', bytes: 0, records: 0 };
  }

  const body = bundleToJsonl(bundle);
  try {
    const res = await request({
      url: cfg.contrib.endpoint,
      method: 'POST',
      headers: { 'content-type': 'application/x-ndjson' },
      body,
      timeoutMs: opts.timeoutMs ?? 30_000,
      retries: 1,
      label: 'contrib-upload',
    });
    if (!res.ok) {
      return { ok: false, detail: `endpoint returned HTTP ${res.status}: ${res.text.slice(0, 200)}`, bytes, records };
    }
    return { ok: true, detail: `sent ${records} record(s) (${formatBytes(bytes)}) to ${cfg.contrib.endpoint}`, bytes, records };
  } catch (err) {
    return {
      ok: false,
      detail: `upload failed: ${err instanceof Error ? err.message : String(err)} (the bundle remains in the outbox)`,
      bytes,
      records,
    };
  }
}

/** Ids already present in the outbox, so a re-stage does not duplicate records. */
export function contributedIds(dataDir: string): Set<string> {
  const ids = new Set<string>();
  for (const file of outboxFiles(dataDir)) {
    for (const row of readJsonl<ContributionRecord>(file)) {
      if (row.id) ids.add(row.id);
    }
  }
  return ids;
}

export function outboxSummary(dataDir: string): Array<{ file: string; bytes: number; records: number }> {
  return outboxFiles(dataDir).map((file) => {
    const raw = readTextOrNull(file) ?? '';
    return {
      file,
      bytes: Buffer.byteLength(raw, 'utf8'),
      records: raw.split('\n').filter((l) => l.trim() && !l.includes('"manifest"')).length,
    };
  });
}

export * from './consent.ts';
