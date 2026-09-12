/**
 * Dataset construction for local model improvement.
 *
 * Three datasets come out of the same episode log, each with a different job:
 *
 *  1. **SFT (self-imitation).** Local attempts that *passed verification*. This
 *     is rejection sampling / expert iteration: we sample k=1 cheaply during
 *     normal use, keep the winners, and fine-tune on them. It is the cheapest
 *     possible RL loop — no reward model, no rollouts, no extra compute beyond
 *     what the user's own work already generated.
 *
 *  2. **Distillation pairs.** Tasks where local failed verification and the
 *     cloud then solved it. The cloud output is a *verified-correct* target for
 *     a task the local model could not do, which is the highest-value training
 *     signal available and costs nothing extra (the escalation already
 *     happened).
 *
 *  3. **DPO preference pairs.** The same failure/escalation episode seen as a
 *     preference: chosen = cloud (verified), rejected = local (failed). DPO is
 *     attractive here because it needs no reward model and only a handful of
 *     examples to move behaviour, which suits a laptop budget.
 *
 * Plus the **router dataset**: feature vector -> did-local-succeed, which trains
 * the routing scorer and costs microseconds.
 *
 * Everything is deduplicated by content hash and capped, because a small number
 * of well-chosen examples beats a large noisy set for LoRA on a laptop.
 */

import { join } from 'node:path';

import type { Episode } from './types.ts';
import type { EpisodeStore } from './store.ts';
import { routerLabel } from './reward.ts';
import type { ProtoConfig } from '../config/schema.ts';
import { ensureDir, writeJsonAtomic, writeTextAtomic } from '../util/fsx.ts';
import { sha256Short } from '../util/text.ts';

/* ------------------------------------------------------------------ */
/* Sample shapes                                                       */
/* ------------------------------------------------------------------ */

export interface SftSample {
  messages: Array<{ role: 'system' | 'user' | 'assistant'; content: string }>;
  /** Provenance, stripped before training (mlx-lm ignores unknown keys). */
  _meta: { episodeId: string; source: 'local-verified' | 'cloud-verified'; reward: number };
}

export interface DpoSample {
  prompt: string;
  chosen: string;
  rejected: string;
  _meta: { episodeId: string; chosenTier: string; rejectedTier: string; reward: number; origin: 'escalation' | 'feedback' };
}

export interface RouterSample {
  id: string;
  x: number[];
  y: 0 | 1;
  /**
   * Inverse-propensity weight (1/π, capped). Exploration episodes carry the
   * counterfactual labels and are therefore up-weighted, not down-weighted.
   */
  w: number;
  /** Kept for debugging; the trainer only uses x/y/w. */
  meta: { taskClass: string; tier: string; exploration: boolean; ts: string };
}

export interface DatasetBuild<T> {
  name: string;
  path: string;
  samples: T[];
  written: boolean;
  skipped: Array<{ episodeId: string; reason: string }>;
}

export interface BuildStats {
  episodesConsidered: number;
  sftLocal: number;
  sftCloud: number;
  dpoPairs: number;
  routerSamples: number;
  duplicatesDropped: number;
  missingPrompts: number;
}

export interface BuiltDatasets {
  sft: DatasetBuild<SftSample>;
  dpo: DatasetBuild<DpoSample>;
  router: DatasetBuild<RouterSample>;
  stats: BuildStats;
  dir: string;
}

export interface BuildOptions {
  /** Cap per dataset; keep small — LoRA on a laptop does not need thousands. */
  maxSft?: number;
  maxDpo?: number;
  maxRouter?: number;
  /** Include cloud-solved episodes as SFT targets (distillation). */
  includeDistillation?: boolean;
  /** Only use episodes newer than this. */
  since?: Date;
  /** Write the files to disk. Off = dry run, used by `proto train plan`. */
  write?: boolean;
}

/* ------------------------------------------------------------------ */
/* Builder                                                             */
/* ------------------------------------------------------------------ */

export const DATASET_DIR = 'datasets';

export function buildDatasets(
  store: EpisodeStore,
  cfg: ProtoConfig,
  opts: BuildOptions = {},
): BuiltDatasets {
  const maxSft = opts.maxSft ?? 400;
  const maxDpo = opts.maxDpo ?? 300;
  const maxRouter = opts.maxRouter ?? 5000;
  const includeDistillation = opts.includeDistillation ?? true;

  const episodes = store.readAll(opts.since ? { since: opts.since } : {});
  const stats: BuildStats = {
    episodesConsidered: episodes.length,
    sftLocal: 0,
    sftCloud: 0,
    dpoPairs: 0,
    routerSamples: 0,
    duplicatesDropped: 0,
    missingPrompts: 0,
  };

  const sftSamples: SftSample[] = [];
  const dpoSamples: DpoSample[] = [];
  const routerSamples: RouterSample[] = [];
  const sftSkipped: DatasetBuild<SftSample>['skipped'] = [];
  const dpoSkipped: DatasetBuild<DpoSample>['skipped'] = [];
  const routerSkipped: DatasetBuild<RouterSample>['skipped'] = [];

  const seenSft = new Set<string>();
  const seenDpo = new Set<string>();
  const seenRouter = new Set<string>();

  for (const ep of episodes) {
    const system = ep.systemPrompt ?? '';
    const feedback = ep.feedback?.signal;

    // ---------- SFT: verified local attempts (self-imitation) --------------
    for (const attempt of ep.attempts) {
      if (!attempt.verification?.passed) continue;
      const isLocal = attempt.tier === 'local' || attempt.tier === 'local-tiny';
      const isCloud = !isLocal;
      if (!isLocal && !(includeDistillation && isCloud)) continue;
      if (attempt.output === undefined || attempt.prompt === undefined) {
        stats.missingPrompts++;
        sftSkipped.push({
          episodeId: ep.id,
          reason: 'prompt or output was not stored (memory.storePrompts/storeTaskText off, or redacted away)',
        });
        continue;
      }
      // A refusal is never worth imitating, and a user-rejected episode's
      // outputs are explicitly not wanted as targets.
      if (feedback === 'reject') {
        sftSkipped.push({ episodeId: ep.id, reason: 'user rejected this episode, so its outputs are not imitated' });
        continue;
      }
      const key = sha256Short(`${attempt.promptHash}:${attempt.outputHash}`);
      if (seenSft.has(key)) {
        stats.duplicatesDropped++;
        continue;
      }
      seenSft.add(key);
      const source: SftSample['_meta']['source'] = isLocal ? 'local-verified' : 'cloud-verified';
      sftSamples.push({
        messages: [
          ...(system ? [{ role: 'system' as const, content: system }] : []),
          { role: 'user' as const, content: attempt.prompt },
          { role: 'assistant' as const, content: attempt.output },
        ],
        _meta: { episodeId: ep.id, source, reward: ep.outcome.reward },
      });
      if (isLocal) stats.sftLocal++;
      else stats.sftCloud++;
    }

    // ---------- DPO: local failed, cloud succeeded -------------------------
    const localAttempts = ep.attempts.filter((a) => (a.tier === 'local' || a.tier === 'local-tiny') && a.verification);
    const failedLocal = localAttempts.find((a) => a.verification && !a.verification.passed);
    const succeededCloud = ep.attempts.find(
      (a) => a.tier !== 'local' && a.tier !== 'local-tiny' && a.verification?.passed,
    );
    if (failedLocal && succeededCloud && failedLocal.output !== undefined && succeededCloud.output !== undefined) {
      const prompt = failedLocal.prompt ?? succeededCloud.prompt;
      if (prompt === undefined) {
        stats.missingPrompts++;
        dpoSkipped.push({ episodeId: ep.id, reason: 'prompt not stored' });
      } else {
        const key = sha256Short(`${prompt}:${succeededCloud.outputHash}:${failedLocal.outputHash}`);
        if (seenDpo.has(key)) stats.duplicatesDropped++;
        else {
          seenDpo.add(key);
          dpoSamples.push({
            prompt,
            chosen: succeededCloud.output,
            rejected: failedLocal.output,
            _meta: {
              episodeId: ep.id,
              chosenTier: succeededCloud.tier,
              rejectedTier: failedLocal.tier,
              reward: ep.outcome.reward,
              origin: 'escalation',
            },
          });
          stats.dpoPairs++;
        }
      }
    }

    // ---------- DPO from explicit rejection feedback ----------------------
    // A rejected answer with no verified alternative cannot form a pair, but a
    // rejected-then-corrected episode can. We only build pairs when we have a
    // genuinely better alternative; otherwise we would be inventing preferences.
    if (feedback === 'reject' && !failedLocal && succeededCloud && succeededCloud.output !== undefined) {
      const earlier = ep.attempts.find((a) => a.output !== undefined && a.n < succeededCloud.n);
      if (earlier?.output !== undefined && earlier.prompt !== undefined) {
        const key = sha256Short(`fb:${earlier.prompt}:${succeededCloud.outputHash}:${earlier.outputHash}`);
        if (!seenDpo.has(key)) {
          seenDpo.add(key);
          dpoSamples.push({
            prompt: earlier.prompt,
            chosen: succeededCloud.output,
            rejected: earlier.output,
            _meta: {
              episodeId: ep.id,
              chosenTier: succeededCloud.tier,
              rejectedTier: earlier.tier,
              reward: ep.outcome.reward,
              origin: 'feedback',
            },
          });
          stats.dpoPairs++;
        }
      }
    }

    // ---------- Router training data --------------------------------------
    const label = routerLabel(ep);
    if (!label) {
      routerSkipped.push({ episodeId: ep.id, reason: 'no observable local outcome (not attempted, or verification disabled)' });
    } else {
      const key = `${ep.taskHash}:${label.y}`;
      if (seenRouter.has(key)) stats.duplicatesDropped++;
      else {
        seenRouter.add(key);
        routerSamples.push({
          id: ep.id,
          x: label.x,
          y: label.y,
          w: label.w,
          meta: {
            taskClass: ep.decision.taskClass,
            tier: ep.decision.tier,
            exploration: ep.decision.exploration,
            ts: ep.ts,
          },
        });
        stats.routerSamples++;
      }
    }
  }

  // Highest reward first, then cap. Reward is a decent proxy for usefulness and
  // keeping the best examples bounded makes runs comparable between nights.
  sftSamples.sort((a, b) => b._meta.reward - a._meta.reward);
  dpoSamples.sort((a, b) => b._meta.reward - a._meta.reward);
  // Router samples keep chronological order: newer routing behaviour matters more.
  const cappedSft = sftSamples.slice(0, maxSft);
  const cappedDpo = dpoSamples.slice(0, maxDpo);
  const cappedRouter = routerSamples.slice(-maxRouter);

  const dir = ensureDir(join(store.dataDir, DATASET_DIR));
  const sftPath = join(dir, 'sft.jsonl');
  const dpoPath = join(dir, 'dpo.jsonl');
  const routerPath = join(dir, 'router.jsonl');

  if (opts.write) {
    writeJsonlFile(sftPath, cappedSft);
    writeJsonlFile(dpoPath, cappedDpo);
    writeJsonlFile(routerPath, cappedRouter);
    writeJsonAtomic(join(dir, 'manifest.json'), {
      builtAt: new Date().toISOString(),
      stats,
      files: {
        sft: { path: sftPath, count: cappedSft.length },
        dpo: { path: dpoPath, count: cappedDpo.length },
        router: { path: routerPath, count: cappedRouter.length },
      },
      options: { maxSft, maxDpo, maxRouter, includeDistillation, since: opts.since?.toISOString() ?? null },
    });
  }

  return {
    sft: { name: 'sft', path: sftPath, samples: cappedSft, written: opts.write === true, skipped: sftSkipped },
    dpo: { name: 'dpo', path: dpoPath, samples: cappedDpo, written: opts.write === true, skipped: dpoSkipped },
    router: { name: 'router', path: routerPath, samples: cappedRouter, written: opts.write === true, skipped: routerSkipped },
    stats,
    dir,
  };
}

function writeJsonlFile(path: string, rows: unknown[]): void {
  const body = rows.map((r) => JSON.stringify(r)).join('\n');
  writeTextAtomic(path, body.length ? body + '\n' : '');
}

/** Human-readable dataset preview used by `proto train plan`. */
export function describeDatasets(built: BuiltDatasets): string[] {
  const lines: string[] = [];
  lines.push(`episodes considered: ${built.stats.episodesConsidered}`);
  lines.push(`SFT samples: ${built.sft.samples.length} (local ${built.stats.sftLocal}, cloud/distilled ${built.stats.sftCloud})`);
  lines.push(`DPO pairs:   ${built.dpo.samples.length}`);
  lines.push(`router rows: ${built.router.samples.length}`);
  if (built.stats.duplicatesDropped > 0) lines.push(`duplicates dropped: ${built.stats.duplicatesDropped}`);
  if (built.stats.missingPrompts > 0) {
    lines.push(
      `episodes unusable for SFT/DPO because the prompt was not stored: ${built.stats.missingPrompts} ` +
        `(enable memory.storePrompts to make future episodes trainable)`,
    );
  }
  return lines;
}
