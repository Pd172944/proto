/**
 * Episode store: append-only JSONL shards + a separate feedback log.
 *
 * Why JSONL shards and not SQLite (which Node 22 now ships)? Three reasons:
 *  - The data is written by exactly one process at a time and read in bulk by
 *    the trainer; a database buys us nothing but an opaque binary blob.
 *  - A research project benefits enormously from the data being greppable and
 *    diffable. `jq` and `grep` work on these files; they do not work on SQLite.
 *  - Migration: shards can be copied, subset, or shipped to a training box
 *    trivially.
 *
 * Feedback is kept in a *separate* file rather than by rewriting episodes, which
 * preserves the append-only invariant. Readers merge the two. This means a
 * crash or a concurrent writer can never truncate the episode history.
 */

import { join } from 'node:path';
import { readdirSync, rmSync, statSync } from 'node:fs';

import type { Episode } from './types.ts';
import { EPISODE_SCHEMA_VERSION } from './types.ts';
import { appendJsonl, byteSize, ensureDir, readJsonl, writeJsonAtomic } from '../util/fsx.ts';
import { ulid } from '../util/ids.ts';
import { localDayKey } from '../util/clock.ts';

export interface FeedbackRecord {
  episodeId: string;
  signal: 'accept' | 'reject' | 'edit';
  note?: string;
  ts: string;
}

export interface StoreStats {
  episodes: number;
  shardCount: number;
  totalBytes: number;
  byStatus: Record<string, number>;
  byTaskClass: Record<string, number>;
  byFinalTier: Record<string, number>;
  escalations: number;
  explorationEpisodes: number;
  localAttempts: number;
  localSuccesses: number;
  localSuccessRate: number | null;
  cloudSpendUsd: number;
  spendTodayUsd: number;
  medianLatencyMs: number;
  meanReward: number;
  redactions: Record<string, number>;
  feedback: Record<string, number>;
  oldestTs: string | null;
  newestTs: string | null;
}

export class EpisodeStore {
  readonly dataDir: string;
  readonly dir: string;
  private readonly feedbackPath: string;

  constructor(dataDir: string) {
    this.dataDir = dataDir;
    this.dir = join(dataDir, 'episodes');
    this.feedbackPath = join(dataDir, 'feedback.jsonl');
  }

  /** Shard path for a timestamp, rotating when the current shard exceeds the cap. */
  shardPath(ts: Date, maxShardBytes: number): string {
    ensureDir(this.dir);
    const day = ts.toISOString().slice(0, 10);
    for (let i = 0; i < 1000; i++) {
      const name = i === 0 ? `${day}.jsonl` : `${day}.${i}.jsonl`;
      const full = join(this.dir, name);
      if (!statSync(full, { throwIfNoEntry: false }) || byteSize(full) < maxShardBytes) return full;
    }
    return join(this.dir, `${day}.overflow.jsonl`);
  }

  append(episode: Episode): string {
    ensureDir(this.dir);
    if (episode.schemaVersion !== EPISODE_SCHEMA_VERSION) {
      throw new Error(
        `refusing to persist an episode with schemaVersion ${episode.schemaVersion} ` +
          `(this build writes v${EPISODE_SCHEMA_VERSION})`,
      );
    }
    appendJsonl(join(this.dir, `${episode.ts.slice(0, 10)}.jsonl`), episode);
    return episode.id;
  }

  /** Append to a size-bounded shard (used by the live runner). */
  appendBounded(episode: Episode, maxShardBytes: number): void {
    appendJsonl(this.shardPath(new Date(episode.ts), maxShardBytes), episode);
  }

  shards(): string[] {
    try {
      return readdirSync(this.dir)
        .filter((f) => f.endsWith('.jsonl'))
        .sort()
        .map((f) => join(this.dir, f));
    } catch {
      return [];
    }
  }

  readAll(opts: { since?: Date; limit?: number; status?: string; includeFeedback?: boolean } = {}): Episode[] {
    const out: Episode[] = [];
    const sinceMs = opts.since?.getTime() ?? 0;
    for (const shard of this.shards()) {
      for (const ep of readJsonl<Episode>(shard)) {
        if (opts.since && Date.parse(ep.ts) < sinceMs) continue;
        if (opts.status && ep.outcome.status !== opts.status) continue;
        out.push(ep);
      }
    }
    out.sort((a, b) => a.id.localeCompare(b.id));
    const limited = opts.limit !== undefined ? out.slice(-opts.limit) : out;
    if (opts.includeFeedback !== false) this.mergeFeedback(limited);
    return limited;
  }

  findById(id: string): Episode | null {
    for (const shard of this.shards()) {
      for (const ep of readJsonl<Episode>(shard)) {
        if (ep.id === id) {
          this.mergeFeedback([ep]);
          return ep;
        }
      }
    }
    return null;
  }

  /* ---------------- feedback ---------------- */

  addFeedback(record: Omit<FeedbackRecord, 'ts'> & { ts?: string }): FeedbackRecord {
    const full: FeedbackRecord = { ...record, ts: record.ts ?? new Date().toISOString() };
    appendJsonl(this.feedbackPath, full);
    return full;
  }

  private mergeFeedback(episodes: Episode[]): void {
    const records = readJsonl<FeedbackRecord>(this.feedbackPath);
    if (records.length === 0) return;
    const byId = new Map<string, FeedbackRecord>();
    for (const r of records) byId.set(r.episodeId, r); // last wins
    for (const ep of episodes) {
      const fb = byId.get(ep.id);
      if (!fb) continue;
      ep.feedback = {
        signal: fb.signal,
        ...(fb.note ? { note: fb.note } : {}),
        ts: fb.ts,
      };
    }
  }

  /* ---------------- housekeeping ---------------- */

  /** Delete shards whose entire day is older than the retention window. */
  prune(retentionDays: number, now: Date = new Date()): { removed: string[]; kept: number } {
    if (retentionDays <= 0) return { removed: [], kept: this.shards().length };
    const cutoff = new Date(now.getTime() - retentionDays * 86_400_000).toISOString().slice(0, 10);
    const removed: string[] = [];
    for (const shard of this.shards()) {
      const day = shard.split('/').pop()?.slice(0, 10) ?? '';
      if (day && day < cutoff) {
        try {
          rmSync(shard, { force: true });
          removed.push(shard);
        } catch {
          /* ignore */
        }
      }
    }
    return { removed, kept: this.shards().length };
  }

  stats(now: Date = new Date()): StoreStats {
    const episodes = this.readAll();
    // Local day, because cloudBudgetUsdPerDay is the user's daily budget.
    const today = localDayKey(now);

    const byStatus: Record<string, number> = {};
    const byTaskClass: Record<string, number> = {};
    const byFinalTier: Record<string, number> = {};
    const redactions: Record<string, number> = {};
    const feedback: Record<string, number> = {};
    const latencies: number[] = [];

    let escalations = 0;
    let exploration = 0;
    let localAttempts = 0;
    let localSuccesses = 0;
    let cloudSpend = 0;
    let spendToday = 0;
    let rewardSum = 0;

    for (const ep of episodes) {
      byStatus[ep.outcome.status] = (byStatus[ep.outcome.status] ?? 0) + 1;
      byTaskClass[ep.decision.taskClass] = (byTaskClass[ep.decision.taskClass] ?? 0) + 1;
      byFinalTier[ep.outcome.finalTier] = (byFinalTier[ep.outcome.finalTier] ?? 0) + 1;
      if (ep.outcome.escalated) escalations++;
      if (ep.decision.exploration) exploration++;
      if (ep.outcome.localSucceeded !== null) {
        localAttempts++;
        if (ep.outcome.localSucceeded) localSuccesses++;
      }
      for (const [k, v] of Object.entries(ep.redaction?.counts ?? {})) redactions[k] = (redactions[k] ?? 0) + v;
      if (ep.feedback) feedback[ep.feedback.signal] = (feedback[ep.feedback.signal] ?? 0) + 1;

      let episodeCost = 0;
      for (const a of ep.attempts) episodeCost += a.costUsd;
      cloudSpend += episodeCost;
      const epDay = Number.isNaN(Date.parse(ep.ts)) ? '' : localDayKey(new Date(ep.ts));
      if (epDay === today) spendToday += episodeCost;
      latencies.push(ep.outcome.totalLatencyMs);
      rewardSum += ep.outcome.reward;
    }

    latencies.sort((a, b) => a - b);
    const median = latencies.length ? (latencies[Math.floor(latencies.length / 2)] as number) : 0;

    return {
      episodes: episodes.length,
      shardCount: this.shards().length,
      totalBytes: this.shards().reduce((a, s) => a + byteSize(s), 0),
      byStatus,
      byTaskClass,
      byFinalTier,
      escalations,
      explorationEpisodes: exploration,
      localAttempts,
      localSuccesses,
      localSuccessRate: localAttempts > 0 ? localSuccesses / localAttempts : null,
      cloudSpendUsd: round6(cloudSpend),
      spendTodayUsd: round6(spendToday),
      medianLatencyMs: median,
      meanReward: episodes.length ? round6(rewardSum / episodes.length) : 0,
      redactions,
      feedback,
      oldestTs: episodes[0]?.ts ?? null,
      newestTs: episodes[episodes.length - 1]?.ts ?? null,
    };
  }

  /** Episodes created since a marker file's timestamp; used by the trainer. */
  since(markerIso: string | null): Episode[] {
    if (!markerIso) return this.readAll();
    const since = new Date(markerIso);
    return this.readAll({ since: Number.isNaN(since.getTime()) ? undefined : since });
  }

  clear(): number {
    const shards = this.shards();
    for (const s of shards) rmSync(s, { force: true });
    return shards.length;
  }
}

function round6(n: number): number {
  return Math.round(n * 1e6) / 1e6;
}

/** Create a fresh episode id. */
export function newEpisodeId(now: number = Date.now()): string {
  return ulid(now);
}

/** Write an arbitrary JSON marker file (trainer high-water marks, consent, etc.). */
export function writeMarker(dataDir: string, name: string, value: unknown): string {
  const path = join(dataDir, `${name}.json`);
  writeJsonAtomic(path, value);
  return path;
}
