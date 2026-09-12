/**
 * Memory commands: episodes, feedback, datasets.
 *
 * These are the commands that let a user see exactly what the harness has
 * recorded about them — which is a prerequisite for the RL features being
 * trustworthy. `proto episodes show <id>` prints the full redacted record;
 * if that ever surprises the user, that is a bug.
 */

import { flagBool, flagNumber, flagString } from './index.ts';
import type { Command, CommandContext, CommandResult } from './index.ts';
import { EpisodeStore } from '../memory/store.ts';
import { buildDatasets, describeDatasets } from '../memory/datasets.ts';
import { computeReward } from '../memory/reward.ts';
import { redact } from '../memory/redact.ts';
import { localSuccessByClass } from '../eval/metrics.ts';
import { formatBytes, formatDuration } from '../util/fsx.ts';
import { style } from '../util/log.ts';

const episodes: Command = {
  name: 'episodes',
  summary: 'list, inspect, summarise or prune the local episode log',
  usage: 'proto episodes <ls|show <id>|stats|prune|redact-check>',
  flags: [
    { name: 'limit', type: 'number', description: 'max episodes to list', default: 20 },
    { name: 'status', type: 'string', description: 'filter by outcome status' },
    { name: 'since', type: 'string', description: 'ISO date; only episodes newer than this' },
    { name: 'days', type: 'number', description: 'retention window for prune (default: memory.retentionDays)' },
    { name: 'yes', type: 'boolean', description: 'confirm a destructive prune' },
    { name: 'text', type: 'string', description: 'redact-check: text to test against the redactor' },
  ],
  async run(ctx): Promise<CommandResult> {
    const sub = ctx.positionals[0] ?? 'ls';
    const store = new EpisodeStore(ctx.dataDir);
    const human: string[] = [];

    if (sub === 'ls') {
      const limit = flagNumber(ctx.args, 'limit') ?? 20;
      const status = flagString(ctx.args, 'status');
      const sinceRaw = flagString(ctx.args, 'since');
      const all = store.readAll({
        ...(status ? { status } : {}),
        ...(sinceRaw ? { since: new Date(sinceRaw) } : {}),
      });
      const rows = all.slice(-limit).reverse();
      human.push(style.bold(`episodes (${all.length} matching, showing ${rows.length})`));
      human.push('  id                              when              tier           status                     reward  cost');
      for (const ep of rows) {
        human.push(
          `  ${ep.id.padEnd(31)} ${ep.ts.slice(0, 19).replace('T', ' ')} ${ep.outcome.finalTier.padEnd(14)} ` +
            `${ep.outcome.status.padEnd(26)} ${ep.outcome.reward.toFixed(2).padStart(6)}  $${ep.outcome.totalCostUsd.toFixed(4)}`,
        );
      }
      return { human, json: { ok: true, count: all.length, episodes: rows } };
    }

    if (sub === 'show') {
      const id = ctx.positionals[1];
      if (!id) throw new Error('usage: proto episodes show <episode-id>');
      const ep = store.findById(id);
      if (!ep) throw new Error(`no episode with id ${id}`);
      human.push(JSON.stringify(ep, null, 2));
      return { human, json: { ok: true, episode: ep } };
    }

    if (sub === 'stats') {
      const stats = store.stats();
      const byClass = localSuccessByClass(store, ctx.cfg);
      human.push(style.bold('episode statistics'));
      human.push(`  episodes          ${stats.episodes} in ${stats.shardCount} shard(s), ${formatBytes(stats.totalBytes)}`);
      human.push(`  window            ${stats.oldestTs?.slice(0, 19) ?? '-'} .. ${stats.newestTs?.slice(0, 19) ?? '-'}`);
      human.push(`  outcomes          ${renderCounts(stats.byStatus)}`);
      human.push(`  final tiers       ${renderCounts(stats.byFinalTier)}`);
      human.push(`  task classes      ${renderCounts(stats.byTaskClass)}`);
      human.push(`  escalations       ${stats.escalations} (${stats.episodes ? ((stats.escalations / stats.episodes) * 100).toFixed(1) : '0'}%)`);
      human.push(`  local attempts    ${stats.localAttempts}, successes ${stats.localSuccesses} (${stats.localSuccessRate === null ? 'n/a' : `${(stats.localSuccessRate * 100).toFixed(1)}%`})`);
      human.push(`  exploration       ${stats.explorationEpisodes} episode(s) with counterfactual labels`);
      human.push(`  cloud spend       $${stats.cloudSpendUsd.toFixed(4)} total, $${stats.spendTodayUsd.toFixed(4)} today`);
      human.push(`  median latency    ${formatDuration(stats.medianLatencyMs)}`);
      human.push(`  mean reward       ${stats.meanReward.toFixed(3)}`);
      if (Object.keys(stats.redactions).length) {
        human.push(`  redactions        ${renderCounts(stats.redactions)}`);
      }
      if (Object.keys(stats.feedback).length) {
        human.push(`  feedback          ${renderCounts(stats.feedback)}`);
      }
      human.push('');
      human.push(style.bold('local success by task class (the router\'s training signal)'));
      if (byClass.length === 0) {
        human.push('  no verified local attempts yet');
      } else {
        for (const row of byClass) {
          human.push(
            `  ${row.taskClass.padEnd(18)} ${String(row.successes).padStart(3)}/${String(row.attempts).padEnd(3)} ` +
              `${(row.rate * 100).toFixed(0).padStart(3)}%  ${bar(row.rate)}`,
          );
        }
      }
      return { human, json: { ok: true, stats, byClass } };
    }

    if (sub === 'prune') {
      const days = flagNumber(ctx.args, 'days') ?? ctx.cfg.memory.retentionDays;
      if (!flagBool(ctx.args, 'yes')) {
        const wouldRemove = store.shards().filter((s) => {
          const day = s.split('/').pop()?.slice(0, 10) ?? '';
          const cutoff = new Date(Date.now() - days * 86_400_000).toISOString().slice(0, 10);
          return day < cutoff;
        });
        human.push(`prune would remove ${wouldRemove.length} shard(s) older than ${days} day(s):`);
        for (const s of wouldRemove) human.push(`  ${s}`);
        human.push('');
        human.push('re-run with --yes to delete them');
        return { human, json: { ok: true, dryRun: true, wouldRemove, retentionDays: days } };
      }
      const result = store.prune(days);
      human.push(`removed ${result.removed.length} shard(s), kept ${result.kept}`);
      return { human, json: { ok: true, ...result } };
    }

    if (sub === 'redact-check') {
      const text = flagString(ctx.args, 'text') ?? ctx.positionals.slice(1).join(' ');
      if (!text) throw new Error('usage: proto episodes redact-check --text "..."');
      const result = redact(text, { extraPatterns: ctx.cfg.redactionPatterns });
      human.push(style.bold('redacted output'));
      human.push(result.text);
      human.push('');
      human.push(`counts: ${renderCounts(result.counts)} (${result.charsRemoved} chars removed)`);
      return { human, json: { ok: true, ...result } };
    }

    throw new Error(`unknown subcommand "${sub}". Try: ls|show|stats|prune|redact-check`);
  },
};

const feedback: Command = {
  name: 'feedback',
  summary: 'record whether a result was good (the strongest learning signal available)',
  usage: 'proto feedback <episode-id> <accept|reject|edit> [--note "..."]',
  flags: [{ name: 'note', type: 'string', description: 'free-form note' }],
  async run(ctx): Promise<CommandResult> {
    const [id, signal] = ctx.positionals;
    if (!id || !signal) throw new Error('usage: proto feedback <episode-id> <accept|reject|edit>');
    if (!['accept', 'reject', 'edit'].includes(signal)) {
      throw new Error(`signal must be accept, reject or edit (got "${signal}")`);
    }
    const store = new EpisodeStore(ctx.dataDir);
    const ep = store.findById(id);
    if (!ep) throw new Error(`no episode with id ${id}`);

    const note = flagString(ctx.args, 'note');
    const record = store.addFeedback({
      episodeId: id,
      signal: signal as 'accept' | 'reject' | 'edit',
      ...(note ? { note } : {}),
    });

    // Show the user what their feedback does to the reward, so the mechanism is
    // not a black box.
    const recomputed = computeReward({
      status: ep.outcome.status,
      escalated: ep.outcome.escalated,
      localAttempted: ep.attempts.some((a) => a.tier === 'local' || a.tier === 'local-tiny'),
      localPassed: ep.outcome.localSucceeded,
      finalScore: ep.attempts.filter((a) => a.verification).slice(-1)[0]?.verification?.score ?? 0,
      refusal: false,
      envelopeDrift: false,
      exploration: ep.decision.exploration,
      feedback: signal as 'accept' | 'reject' | 'edit',
      hadBlockers: false,
      cloudCostUsd: ep.outcome.totalCostUsd,
    });

    const human = [
      `recorded "${signal}" for ${id}`,
      `reward: ${ep.outcome.reward.toFixed(3)} -> ${recomputed.reward.toFixed(3)}`,
      '',
      'components:',
      ...recomputed.components.map((c) => `  ${c.name.padEnd(22)} ${c.value >= 0 ? '+' : ''}${c.value.toFixed(3)}  ${c.note}`),
    ];
    return { human, json: { ok: true, feedback: record, reward: recomputed } };
  },
};

const datasets: Command = {
  name: 'datasets',
  summary: 'inspect or write the SFT/DPO/router datasets derived from episodes',
  usage: 'proto datasets build [--write] [--max-sft n] [--max-dpo n]',
  flags: [
    { name: 'write', type: 'boolean', description: 'write the JSONL files to var/datasets/' },
    { name: 'max-sft', type: 'number', description: 'cap on SFT samples', default: 400 },
    { name: 'max-dpo', type: 'number', description: 'cap on DPO pairs', default: 300 },
    { name: 'no-distill', type: 'boolean', description: 'exclude cloud-solved episodes from SFT (no distillation)' },
    { name: 'sample', type: 'number', description: 'print this many example rows' },
  ],
  async run(ctx): Promise<CommandResult> {
    const sub = ctx.positionals[0] ?? 'build';
    if (sub !== 'build') throw new Error(`unknown subcommand "${sub}" (only "build" exists)`);

    const store = new EpisodeStore(ctx.dataDir);
    const write = flagBool(ctx.args, 'write');
    const built = buildDatasets(store, ctx.cfg, {
      write,
      maxSft: flagNumber(ctx.args, 'max-sft') ?? 400,
      maxDpo: flagNumber(ctx.args, 'max-dpo') ?? 300,
      includeDistillation: !flagBool(ctx.args, 'no-distill'),
    });

    const human: string[] = [];
    human.push(style.bold(`datasets (${write ? 'written' : 'dry run: nothing written'})`));
    for (const line of describeDatasets(built)) human.push(`  ${line}`);
    human.push('');
    if (built.sft.samples.length) human.push(`  sft    -> ${built.sft.path}`);
    if (built.dpo.samples.length) human.push(`  dpo    -> ${built.dpo.path}`);
    if (built.router.samples.length) human.push(`  router -> ${built.router.path}`);

    const sampleCount = flagNumber(ctx.args, 'sample') ?? 0;
    if (sampleCount > 0) {
      for (const row of built.sft.samples.slice(0, sampleCount)) {
        human.push('');
        human.push(style.bold('sft sample'));
        human.push(JSON.stringify(row, null, 2));
      }
      for (const row of built.dpo.samples.slice(0, sampleCount)) {
        human.push('');
        human.push(style.bold('dpo sample'));
        human.push(JSON.stringify({ ...row, prompt: row.prompt.slice(0, 400), chosen: row.chosen.slice(0, 200), rejected: row.rejected.slice(0, 200) }, null, 2));
      }
    }

    return { human, json: { ok: true, write, stats: built.stats, paths: { sft: built.sft.path, dpo: built.dpo.path, router: built.router.path } } };
  },
};

function renderCounts(counts: Record<string, number>): string {
  const entries = Object.entries(counts).sort((a, b) => b[1] - a[1]);
  if (entries.length === 0) return '(none)';
  return entries.map(([k, v]) => `${k}=${v}`).join(' ');
}

function bar(rate: number, width = 16): string {
  const filled = Math.round(rate * width);
  return `[${'#'.repeat(filled)}${'.'.repeat(width - filled)}]`;
}

export const memoryCommands: Command[] = [episodes, feedback, datasets];
