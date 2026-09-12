/**
 * Consent and pseudonymity for the optional global-improvement channel.
 *
 * Two separate decisions are tracked, because conflating them is the classic
 * mistake in "help improve the model" features:
 *
 *   1. `localTraining` — may your own episodes be used to fine-tune *your* local
 *      model? This is entirely private; no data leaves the machine.
 *   2. `globalShare`   — may *derived* data be contributed to a shared pool?
 *      Off by default, revocable, and separate from (1).
 *
 * The device pseudonym is a salted hash rotated on a schedule. Rotation gives
 * forward privacy: contributions from before a rotation cannot be linked to
 * contributions after it without the old salt, which is deleted. This is not
 * differential privacy and we do not claim it is — it is the cheap, honest
 * mechanism that stops the contribution log from becoming a permanent
 * behavioural fingerprint.
 */

import { join } from 'node:path';
import { hostname } from 'node:os';

import type { ProtoConfig } from '../config/schema.ts';
import { ensureDir, readJsonOrNull, writeJsonAtomic, writeTextAtomic } from '../util/fsx.ts';
import { randomHex } from '../util/ids.ts';
import { sha256 } from '../util/text.ts';

export interface ConsentRecord {
  version: 1;
  /** Private fine-tuning of the local model. */
  localTraining: boolean;
  /** Contributing derived data to the shared pool. */
  globalShare: boolean;
  /** Whether raw task/output text may be included in contributions. */
  shareCode: boolean;
  /** ISO timestamps of the last change to each flag, for auditability. */
  updatedAt: string;
  history: Array<{ at: string; change: string; from: string; to: string }>;
}

export function consentPath(dataDir: string): string {
  return join(dataDir, 'contrib', 'consent.json');
}

export function loadConsent(dataDir: string, cfg: ProtoConfig): ConsentRecord {
  const existing = readJsonOrNull<ConsentRecord>(consentPath(dataDir));
  if (existing) return existing;
  // No record on disk means "never asked", which is not the same as "yes".
  return {
    version: 1,
    localTraining: cfg.train.enabled,
    globalShare: cfg.contrib.enabled,
    shareCode: cfg.contrib.shareCode,
    updatedAt: new Date().toISOString(),
    history: [],
  };
}

export function saveConsent(dataDir: string, record: ConsentRecord): ConsentRecord {
  ensureDir(join(dataDir, 'contrib'));
  writeJsonAtomic(consentPath(dataDir), record);
  return record;
}

export interface ConsentChange {
  localTraining?: boolean;
  globalShare?: boolean;
  shareCode?: boolean;
}

export function updateConsent(
  dataDir: string,
  cfg: ProtoConfig,
  change: ConsentChange,
  now: Date = new Date(),
): ConsentRecord {
  const current = loadConsent(dataDir, cfg);
  const next: ConsentRecord = { ...current, history: [...current.history] };
  for (const key of ['localTraining', 'globalShare', 'shareCode'] as const) {
    const value = change[key];
    if (value === undefined || value === current[key]) continue;
    next[key] = value;
    next.history.push({
      at: now.toISOString(),
      change: key,
      from: String(current[key]),
      to: String(value),
    });
  }
  // Keep the audit trail bounded but long enough to answer "when did I agree?".
  next.history = next.history.slice(-100);
  next.updatedAt = now.toISOString();
  return saveConsent(dataDir, next);
}

/* ------------------------------------------------------------------ */
/* Pseudonym                                                           */
/* ------------------------------------------------------------------ */

export interface Identity {
  /** Rotating salt; deleted and replaced on rotation. */
  salt: string;
  rotatedAt: string;
  /** Monotonic counter, so previous pseudonyms can be listed (never the salts). */
  generation: number;
  pseudonym: string;
}

function identityPath(dataDir: string): string {
  return join(dataDir, 'contrib', 'identity.json');
}

function derivePseudonym(salt: string): string {
  // Hostname is included so that two machines with the same salt do not collide,
  // but it never leaves the machine in the clear.
  return sha256(`${salt}:${hostname()}`).slice(0, 16);
}

export function loadIdentity(dataDir: string, cfg: ProtoConfig, now: Date = new Date()): Identity {
  const existing = readJsonOrNull<Identity>(identityPath(dataDir));
  const rotateAfterMs = Math.max(1, cfg.contrib.saltRotateDays) * 86_400_000;
  if (existing) {
    const age = now.getTime() - Date.parse(existing.rotatedAt);
    if (Number.isFinite(age) && age < rotateAfterMs) return existing;
  }
  return freshIdentity(dataDir, existing ?? null, now);
}

function freshIdentity(dataDir: string, previous: Identity | null, now: Date): Identity {
  const salt = randomHex(32);
  const identity: Identity = {
    salt,
    rotatedAt: now.toISOString(),
    generation: (previous?.generation ?? 0) + 1,
    pseudonym: derivePseudonym(salt),
  };
  ensureDir(join(dataDir, 'contrib'));
  writeJsonAtomic(identityPath(dataDir), identity);
  return identity;
}

/**
 * Delete the salt and start a new pseudonym generation immediately.
 * Existing staged bundles become unlinkable to future ones.
 *
 * This deliberately bypasses the age check rather than setting saltRotateDays=0:
 * the rotation window is clamped to a minimum of one day so that a config typo
 * cannot silently rotate the pseudonym on every single run.
 */
export function rotateIdentity(dataDir: string, _cfg: ProtoConfig, now: Date = new Date()): Identity {
  const existing = readJsonOrNull<Identity>(identityPath(dataDir));
  return freshIdentity(dataDir, existing, now);
}

/* ------------------------------------------------------------------ */
/* Human-readable consent summary                                      */
/* ------------------------------------------------------------------ */

export function describeConsent(record: ConsentRecord): string[] {
  const lines: string[] = [];
  lines.push(
    `local fine-tuning: ${record.localTraining ? 'ON — your episodes tune your own local model, nothing leaves this machine' : 'off'}`,
  );
  lines.push(
    `global sharing:    ${record.globalShare ? 'ON — derived data may be contributed' : 'off (default)'}`,
  );
  if (record.globalShare) {
    lines.push(
      `raw text sharing:  ${record.shareCode ? 'ON — redacted task/output text WILL be included' : 'off — only features, hashes and preference labels'}`,
    );
  }
  lines.push(`last changed:      ${record.updatedAt}`);
  return lines;
}

export function writeConsentReceipt(dataDir: string, record: ConsentRecord): string {
  const path = join(dataDir, 'contrib', 'CONSENT.md');
  const body = [
    '# Contribution consent record',
    '',
    ...describeConsent(record),
    '',
    '## Change history',
    '',
    ...(record.history.length
      ? record.history.map((h) => `- ${h.at}: ${h.change} ${h.from} -> ${h.to}`)
      : ['- (no changes recorded yet)']),
    '',
    'Revoke at any time with `proto contrib consent --global off`.',
    'Data already staged in the outbox is NOT sent retroactively when sharing is turned on.',
    '',
  ].join('\n');
  ensureDir(join(dataDir, 'contrib'));
  writeTextAtomic(path, body);
  return path;
}
