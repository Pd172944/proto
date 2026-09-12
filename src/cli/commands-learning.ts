/**
 * Learning commands: train, eval, replay, contrib.
 *
 * These are the commands that make the RL and routing machinery inspectable. The
 * guiding rule is that no command here does anything irreversible or expensive
 * without an explicit flag, and every one of them can answer "why?".
 */

import { join } from 'node:path';

import { flagBool, flagList, flagNumber, flagString } from './index.ts';
import type { Command, CommandResult } from './index.ts';
import { REPO_ROOT, loadConfig, saveConfig } from '../config/load.ts';
import {
  currentScorerSummary,
  ensureTrainDirs,
  promotionInstructions,
  refreshRouter,
  trainStatus,
  trainTick,
} from '../train/index.ts';
import { listAdapters, getActiveAdapter } from '../train/adapter.ts';
import { buildDatasets, describeDatasets } from '../memory/datasets.ts';
import { EpisodeStore } from '../memory/store.ts';
import { evaluateRouting, formatEvalReport, formatReplayReport, replay, EVAL_TASKS } from '../eval/index.ts';
import { logisticEvalFromEpisodes, localSuccessByClass } from '../eval/metrics.ts';
import {
  buildBundle,
  bytesStagedToday,
  contributedIds,
  outboxSummary,
  previewContribution,
  stageBundle,
  uploadBundle,
} from '../contrib/index.ts';
import { describeConsent, loadConsent, rotateIdentity, updateConsent, writeConsentReceipt } from '../contrib/consent.ts';
import { formatBytes, formatDuration, writeTextAtomic } from '../util/fsx.ts';
import { style } from '../util/log.ts';

/* ------------------------------------------------------------------ */
/* train                                                              */
/* ------------------------------------------------------------------ */

const train: Command = {
  name: 'train',
  summary: 'status | tick | now | plan | enable | disable | router | adapters | install-agent',
  usage: 'proto train <status|tick|now|plan|enable|disable|router|adapters|install-agent>',
  flags: [
    { name: 'mode', type: 'string', description: 'sft | dpo (override the configured mode)' },
    { name: 'force', type: 'boolean', description: 'ignore the safety gates for this run' },
    { name: 'yes', type: 'boolean', description: 'confirm an action that leaves var/' },
    { name: 'install', type: 'boolean', description: 'install-agent: actually write and load the launchd agent' },
    { name: 'interval', type: 'number', description: 'install-agent: seconds between ticks', default: 3600 },
    { name: 'limit', type: 'number', description: 'adapters: how many to list', default: 10 },
  ],
  async run(ctx): Promise<CommandResult> {
    const sub = ctx.positionals[0] ?? 'status';
    ensureTrainDirs(ctx.dataDir);

    if (sub === 'status') {
      const status = await trainStatus(ctx.cfg, ctx.dataDir);
      const human: string[] = [];
      human.push(style.bold('training status'));
      human.push(`  enabled       ${status.enabled ? style.green('yes') : 'no — opt-in with `proto train enable`'}`);
      human.push(`  ready to run  ${status.readyToRun ? style.green('yes') : style.yellow('no')}`);
      human.push(`  base model    ${status.baseModel}`);
      human.push(`  trainer       ${status.preflight.ok ? style.green(status.preflight.detail) : style.yellow(status.preflight.detail)}`);
      human.push(
        `  machine       ${status.system.onAC === null ? 'power unknown' : status.system.onAC ? 'on wall power' : 'on battery'}` +
          `${status.system.batteryPct !== null ? ` (${status.system.batteryPct}%)` : ''}, ` +
          `load1 ${status.system.load[0].toFixed(2)}, thermal ${status.system.thermalWarning ? style.yellow('throttled') : 'ok'}, platform ${status.system.platform}`,
      );
      human.push(
        `  budget        ${status.minutesUsedToday}/${ctx.cfg.train.dailyBudgetMin} min today, ` +
          `${status.state.totalRuns} run(s) lifetime, ${status.state.consecutiveSkips} consecutive skip(s)`,
      );
      human.push(`  new data      ${status.newEpisodes} new labelled episode(s) (min ${ctx.cfg.train.minNewEpisodes})`);
      human.push('');
      human.push(style.bold('gates'));
      for (const check of status.gates.checks) {
        human.push(`  ${check.ok ? style.green('pass') : style.red('BLOCK')}  ${check.id.padEnd(14)} ${check.detail}`);
      }
      human.push('');
      human.push(style.bold('plan'));
      human.push(`  mode          ${status.plan.mode}, ${status.plan.iters} iteration(s), up to ${status.plan.maxRuntimeMin} min`);
      human.push(`  priority      ${status.priority.description}`);
      human.push(`  reason        ${status.plan.reason}`);
      human.push('');
      human.push(style.bold('data available'));
      for (const line of status.datasetSummary) human.push(`  ${line}`);
      human.push('');
      human.push(style.bold('last runs'));
      if (status.recentJobs.length === 0) {
        human.push('  none yet');
      } else {
        for (const job of status.recentJobs) {
          human.push(
            `  ${job.createdAt.slice(0, 19).replace('T', ' ')} ${job.kind} ${job.status.padEnd(12)} ` +
              `${job.durationMs ? formatDuration(job.durationMs) : '-'} ${job.error ?? ''}`,
          );
        }
      }
      if (!status.preflight.ok) {
        human.push('');
        human.push(style.bold('to enable local training later (nothing is installed automatically)'));
        for (const line of status.preflight.installInstructions) human.push(`  ${line}`);
      }
      if (status.activeAdapter) {
        human.push('');
        human.push(`active adapter: ${status.activeAdapter.name}`);
      }
      return { human, json: { ok: true, status } };
    }

    if (sub === 'plan' || sub === 'tick' || sub === 'now') {
      const dryRun = sub === 'plan';
      const force = sub === 'now' || flagBool(ctx.args, 'force');
      const events: string[] = [];
      const result = await trainTick(ctx.cfg, ctx.dataDir, {
        ...(dryRun ? { dryRun: true } : {}),
        ...(force ? { force: true } : {}),
        ...(flagString(ctx.args, 'mode') ? { mode: flagString(ctx.args, 'mode') as 'sft' | 'dpo' } : {}),
        onEvent: (e) => {
          switch (e.type) {
            case 'router':
              events.push(
                e.result.trained
                  ? `router: retrained on ${e.result.sampleCount} episodes (acc ${e.result.accuracy.toFixed(3)}, auc ${e.result.auc.toFixed(3)})`
                  : `router: not retrained — ${e.result.reason}`,
              );
              break;
            case 'gates':
              events.push(e.allowed ? 'gates: all passed' : `gates: blocked (${e.blockers.join('; ')})`);
              break;
            case 'progress':
              events.push(e.message);
              break;
            case 'job':
              events.push(`job ${e.job.id} queued (${e.job.kind}, ${e.job.plan.iters} iters)`);
              break;
            default:
              break;
          }
        },
      });

      const human: string[] = [];
      human.push(style.bold(`train ${sub}`));
      for (const e of events) human.push(`  ${e}`);
      human.push('');
      human.push(`  ran           ${result.ran ? style.green('yes') : 'no'}`);
      human.push(`  reason        ${result.reason}`);
      if (result.job) {
        human.push(`  job           ${result.job.id} (${result.job.status})`);
        if (result.job.command) human.push(`  command       ${result.job.command}`);
        if (result.job.logPath) human.push(`  log           ${result.job.logPath}`);
        if (result.job.trainLoss !== undefined) human.push(`  train loss    ${result.job.trainLoss}`);
        if (result.job.validLoss !== undefined) human.push(`  valid loss    ${result.job.validLoss}`);
      }
      if (result.installInstructions.length) {
        human.push('');
        human.push(style.bold('trainer not installed — run these yourself when you want local RL'));
        for (const line of result.installInstructions) human.push(`  ${line}`);
      }
      // Exit-code semantics matter here because a cron/launchd agent calls `tick`
      // every hour. Declining to run is the *normal* outcome (that is what the
      // gates are for), so it must exit 0 — otherwise the agent reports a failure
      // every hour and the user learns to ignore it. A non-zero code is reserved
      // for "the tick could not do its job": a training session that actually ran
      // and failed, or a forced `now` that could not start.
      const sessionFailed = result.job !== null && ['failed', 'interrupted'].includes(result.job.status);
      const code = sub === 'plan' || (!sessionFailed && (result.ran || sub === 'tick')) ? 0 : 1;
      return { human, json: { ok: code === 0, result, skipped: !result.ran }, exitCode: code };
    }

    if (sub === 'enable' || sub === 'disable') {
      const on = sub === 'enable';
      const next = structuredClone(ctx.cfg);
      next.train.enabled = on;
      const path = saveConfig(next, ctx.dataDir);
      const consent = updateConsent(ctx.dataDir, next, { localTraining: on });
      const receipt = writeConsentReceipt(ctx.dataDir, consent);
      const human = [
        `local training ${on ? style.green('enabled') : 'disabled'}`,
        `wrote ${path}`,
        `consent record: ${receipt}`,
        '',
        ...describeConsent(consent),
        '',
        on
          ? 'training will only run inside the allowed window, on wall power, when the machine is idle.'
          : 'no training will run. Existing episodes are still recorded unless you disable memory.',
      ];
      return { human, json: { ok: true, enabled: on, path, consent } };
    }

    if (sub === 'router') {
      const result = refreshRouter(ctx.cfg, ctx.dataDir, { write: true });
      const human: string[] = [];
      human.push(style.bold('router refresh'));
      if (!result.trained) {
        human.push(`  not trained: ${result.reason}`);
      } else {
        human.push(`  trained on    ${result.sampleCount} episodes (${result.positives} local successes)`);
        human.push(`  accuracy      ${result.accuracy.toFixed(3)}   auc ${result.auc.toFixed(3)}   brier ${result.brier.toFixed(3)}`);
        human.push(`  weights       ${result.weightsPath}`);
        human.push('');
        human.push(style.bold('most influential features'));
        for (const f of result.topFeatures) {
          human.push(`  ${f.name.padEnd(30)} ${f.weight >= 0 ? '+' : ''}${f.weight.toFixed(4)}`);
        }
      }
      const evalResult = logisticEvalFromEpisodes(new EpisodeStore(ctx.dataDir), ctx.cfg, { holdout: true });
      human.push('');
      human.push(style.bold('quality'));
      human.push(`  labelled episodes  ${evalResult.samples} (${evalResult.positives} positives)`);
      if (evalResult.holdout) {
        human.push(
          `  time-split holdout n=${evalResult.holdout.testSamples} acc ${evalResult.holdout.accuracy.toFixed(3)} ` +
            `auc ${evalResult.holdout.auc.toFixed(3)} (difficulty-only baseline auc ${evalResult.holdout.heuristicAuc.toFixed(3)})`,
        );
      }
      for (const note of evalResult.notes) human.push(`  note: ${note}`);
      const byClass = localSuccessByClass(new EpisodeStore(ctx.dataDir), ctx.cfg);
      if (byClass.length) {
        human.push('');
        human.push(style.bold('observed local success by class'));
        for (const row of byClass) {
          human.push(`  ${row.taskClass.padEnd(18)} ${row.successes}/${row.attempts}  ${(row.rate * 100).toFixed(0)}%`);
        }
      }
      return { human, json: { ok: true, result, quality: evalResult, byClass } };
    }

    if (sub === 'adapters') {
      const adapters = listAdapters(ctx.dataDir);
      const active = getActiveAdapter(ctx.dataDir);
      const human: string[] = [];
      human.push(style.bold('adapters'));
      if (adapters.length === 0) {
        human.push('  none yet. They are created by `proto train now` once mlx-lm is installed.');
      }
      for (const a of adapters) {
        human.push(
          `  ${a.name} ${active?.name === a.name ? style.green('(active)') : ''} mode=${a.mode} ` +
            `${formatBytes(a.sizeBytes)} ${a.complete ? '' : style.yellow('(incomplete)')}${a.trainLoss !== null ? ` trainLoss=${a.trainLoss}` : ''}`,
        );
      }
      if (adapters.length) {
        human.push('');
        human.push(style.bold('to serve the newest adapter'));
        for (const line of promotionInstructions(ctx.cfg, ctx.dataDir, (adapters[0] as { name: string }).name)) {
          human.push(`  ${line}`);
        }
      }
      return { human, json: { ok: true, adapters, activeAdapter: active } };
    }

    if (sub === 'install-agent') {
      const interval = flagNumber(ctx.args, 'interval') ?? 3600;
      const cliPath = join(REPO_ROOT, 'src', 'cli.ts');
      const label = 'com.protoharness.train';
      const plistPath = join(ctx.dataDir, 'train', `${label}.plist`);
      const logDir = join(ctx.dataDir, 'train', 'logs');
      const plist = `<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<!-- Generated by proto. This is the deferred, background half of the harness:
     it wakes up every ${interval}s, and MOST of the time does nothing because the
     scheduler's gates (power, thermal, idle, window, budget, new data) refuse to run.
     When it does run, it runs at background QoS with a hard time cap and a load watchdog. -->
<plist version="1.0">
<dict>
  <key>Label</key><string>${label}</string>
  <key>ProgramArguments</key>
  <array>
    <string>${process.execPath}</string>
    <string>--experimental-strip-types</string>
    <string>--disable-warning=ExperimentalWarning</string>
    <string>${cliPath}</string>
    <string>train</string>
    <string>tick</string>
    <string>--data-dir</string>
    <string>${ctx.dataDir}</string>
  </array>
  <key>WorkingDirectory</key><string>${REPO_ROOT}</string>
  <key>StartInterval</key><integer>${interval}</integer>
  <key>RunAtLoad</key><false/>
  <key>ProcessType</key><string>Background</string>
  <key>LowPriorityIO</key><true/>
  <key>Nice</key><integer>5</integer>
  <key>EnvironmentVariables</key>
  <dict>
    <key>PROTO_LOG</key><string>json</string>
  </dict>
  <key>StandardOutPath</key><string>${join(logDir, 'agent.out.log')}</string>
  <key>StandardErrorPath</key><string>${join(logDir, 'agent.err.log')}</string>
</dict>
</plist>
`;
      writeTextAtomic(plistPath, plist);
      const installPath = join(process.env['HOME'] ?? '~', 'Library', 'LaunchAgents', `${label}.plist`);
      const human: string[] = [];
      human.push(style.bold('launchd agent'));
      human.push(`  written to    ${plistPath}`);
      human.push(`  interval      ${interval}s (most ticks do nothing; that is the design)`);
      human.push('');
      human.push('to install it yourself:');
      human.push(`  cp ${plistPath} ${installPath}`);
      human.push(`  launchctl load ${installPath}`);
      human.push('');
      human.push('to remove it:');
      human.push(`  launchctl unload ${installPath} && rm ${installPath}`);
      human.push('');
      human.push(
        'note: training is ' +
          (ctx.cfg.train.enabled ? style.green('enabled') : 'still disabled') +
          ', so ticks will record their reasons and exit until you run `proto train enable`.',
      );

      let installed = false;
      let installError: string | null = null;
      if (flagBool(ctx.args, 'install')) {
        if (!flagBool(ctx.args, 'yes')) {
          human.push('');
          human.push(style.yellow('refusing to install without --yes'));
        } else {
          const { exec } = await import('../util/proc.ts');
          const res = await exec('/bin/sh', ['-c', `mkdir -p "$(dirname '${installPath}')" && cp '${plistPath}' '${installPath}' && launchctl load '${installPath}'`], {
            timeoutMs: 20_000,
          });
          installed = res.code === 0;
          if (!installed) installError = `${res.stderr || res.stdout}`.trim().slice(0, 400);
          human.push('');
          human.push(installed ? style.green('installed and loaded') : style.red(`install failed: ${installError}`));
        }
      }
      return { human, json: { ok: true, plistPath, installPath, interval, installed, installError } };
    }

    throw new Error(`unknown subcommand "${sub}". Try: status|tick|now|plan|enable|disable|router|adapters|install-agent`);
  },
};

/* ------------------------------------------------------------------ */
/* eval                                                               */
/* ------------------------------------------------------------------ */

const evalCommand: Command = {
  name: 'eval',
  summary: 'score the router against the built-in task corpus (offline, no model calls)',
  usage: 'proto eval [run] [--mode heuristic|learned|hybrid] [--only id1,id2] [--json]',
  flags: [
    { name: 'mode', type: 'string', description: 'override the routing mode for this evaluation' },
    { name: 'only', type: 'string', description: 'comma-separated task ids' },
    { name: 'compare', type: 'boolean', description: 'run all three modes and compare' },
    { name: 'list', type: 'boolean', description: 'list the corpus and exit' },
  ],
  async run(ctx): Promise<CommandResult> {
    const sub = ctx.positionals[0] ?? 'run';
    if (sub === 'list' || flagBool(ctx.args, 'list')) {
      const human = [
        style.bold(`eval corpus (${EVAL_TASKS.length} tasks)`),
        '  id                            expected  verifiable  class',
        ...EVAL_TASKS.map(
          (t) => `  ${t.id.padEnd(29)} ${t.expected.padEnd(9)} ${String(t.verifiable).padEnd(11)} ${t.expectedClass ?? '-'}`,
        ),
      ];
      return { human, json: { ok: true, tasks: EVAL_TASKS } };
    }

    const only = flagString(ctx.args, 'only')?.split(',').map((s) => s.trim()).filter(Boolean);
    const modeFlag = flagString(ctx.args, 'mode');

    if (flagBool(ctx.args, 'compare') || sub === 'compare') {
      const modes: Array<'heuristic' | 'learned' | 'hybrid'> = ['heuristic', 'learned', 'hybrid'];
      const reports = modes.map((mode) =>
        evaluateRouting(ctx.cfg, ctx.dataDir, { mode, ...(only ? { only } : {}) }),
      );
      const human: string[] = [style.bold('routing mode comparison')];
      human.push('  mode       accuracy   localP  localR  cloudP  cloudR  est.cloud$  unverified-local');
      for (const r of reports) {
        const m = r.metrics;
        human.push(
          `  ${r.mode.padEnd(10)} ${(m.accuracy * 100).toFixed(1).padStart(6)}%   ` +
            `${(m.localPrecision * 100).toFixed(0).padStart(5)}%  ${(m.localRecall * 100).toFixed(0).padStart(5)}%  ` +
            `${(m.cloudPrecision * 100).toFixed(0).padStart(5)}%  ${(m.cloudRecall * 100).toFixed(0).padStart(5)}%  ` +
            `$${m.estimatedCloudCostUsd.toFixed(4).padStart(9)}  ${String(m.unverifiedLocal).padStart(6)}`,
        );
      }
      human.push('');
      human.push('notes:');
      for (const note of reports[0]?.notes ?? []) human.push(`  - ${note}`);
      return { human, json: { ok: true, reports } };
    }

    const report = evaluateRouting(ctx.cfg, ctx.dataDir, {
      ...(modeFlag ? { mode: modeFlag as 'heuristic' | 'learned' | 'hybrid' } : {}),
      ...(only ? { only } : {}),
    });
    return {
      human: formatEvalReport(report),
      json: { ok: true, report },
      exitCode: report.metrics.accuracy >= 0.7 ? 0 : 1,
    };
  },
};

/* ------------------------------------------------------------------ */
/* replay                                                             */
/* ------------------------------------------------------------------ */

const replayCommand: Command = {
  name: 'replay',
  summary: 're-score historical decisions against a different policy (no model calls)',
  usage: 'proto replay [--mode ...] [--floor 0.8] [--limit n] [--changed]',
  flags: [
    { name: 'mode', type: 'string', description: 'override the routing mode for the counterfactual' },
    { name: 'floor', type: 'number', description: 'override routing.qualityFloor' },
    { name: 'limit', type: 'number', description: 'only the most recent N episodes' },
    { name: 'days', type: 'number', description: 'only episodes from the last N days' },
    { name: 'changed', type: 'boolean', description: 'print only changed decisions' },
  ],
  async run(ctx): Promise<CommandResult> {
    const days = flagNumber(ctx.args, 'days');
    const report = replay(ctx.cfg, ctx.dataDir, {
      ...(flagString(ctx.args, 'mode') ? { mode: flagString(ctx.args, 'mode') as 'heuristic' | 'learned' | 'hybrid' } : {}),
      ...(flagNumber(ctx.args, 'floor') !== undefined ? { qualityFloor: flagNumber(ctx.args, 'floor') as number } : {}),
      ...(flagNumber(ctx.args, 'limit') !== undefined ? { limit: flagNumber(ctx.args, 'limit') as number } : {}),
      ...(days !== undefined ? { since: new Date(Date.now() - days * 86_400_000) } : {}),
    });

    let human = formatReplayReport(report);
    if (flagBool(ctx.args, 'changed')) {
      human = human.filter((l) => !l.startsWith('  ') || !l.includes('->') || l.includes('->'));
    }
    return { human, json: { ok: true, report } };
  },
};

/* ------------------------------------------------------------------ */
/* contrib                                                            */
/* ------------------------------------------------------------------ */

const contrib: Command = {
  name: 'contrib',
  summary: 'status | preview | stage | upload | consent | outbox | rotate  (global sharing is opt-in)',
  usage: 'proto contrib <status|preview|stage|upload|consent|outbox|rotate>',
  flags: [
    { name: 'global', type: 'string', description: 'consent: on|off for global sharing' },
    { name: 'local-training', type: 'string', description: 'consent: on|off for private local fine-tuning' },
    { name: 'share-code', type: 'string', description: 'consent: on|off for including redacted text' },
    { name: 'yes', type: 'boolean', description: 'confirm an upload' },
    { name: 'limit', type: 'number', description: 'max records per bundle', default: 200 },
    { name: 'live', type: 'boolean', description: 'preview with shareCode forced on, to see the worst case' },
  ],
  async run(ctx): Promise<CommandResult> {
    const sub = ctx.positionals[0] ?? 'status';
    const consent = loadConsent(ctx.dataDir, ctx.cfg);

    if (sub === 'consent') {
      const change: { globalShare?: boolean; localTraining?: boolean; shareCode?: boolean } = {};
      const g = flagString(ctx.args, 'global');
      const l = flagString(ctx.args, 'local-training');
      const s = flagString(ctx.args, 'share-code');
      if (g) change.globalShare = g === 'on' || g === 'true';
      if (l) change.localTraining = l === 'on' || l === 'true';
      if (s) change.shareCode = s === 'on' || s === 'true';

      if (Object.keys(change).length === 0) {
        return { human: describeConsent(consent), json: { ok: true, consent } };
      }
      const updated = updateConsent(ctx.dataDir, ctx.cfg, change);
      const receipt = writeConsentReceipt(ctx.dataDir, updated);

      // Keep config in sync so the two never disagree.
      const next = structuredClone(ctx.cfg);
      next.contrib.enabled = updated.globalShare;
      next.contrib.shareCode = updated.shareCode;
      next.train.enabled = updated.localTraining;
      const configPath = saveConfig(next, ctx.dataDir);

      const human = [
        'consent updated',
        ...describeConsent(updated),
        '',
        `receipt: ${receipt}`,
        `config:  ${configPath}`,
        '',
        updated.globalShare
          ? 'nothing is uploaded automatically: staged bundles wait in the outbox until you run `proto contrib upload`.'
          : 'sharing is off. Anything already staged will not be sent.',
      ];
      return { human, json: { ok: true, consent: updated, receipt, configPath } };
    }

    if (sub === 'status') {
      const staged = outboxSummary(ctx.dataDir);
      const human = [
        style.bold('contribution status'),
        ...describeConsent(consent),
        '',
        `endpoint:            ${ctx.cfg.contrib.endpoint || '(none configured — bundles can only be staged)'}`,
        `confirmation:        ${ctx.cfg.contrib.requireConfirmation ? 'required per upload (--yes)' : 'not required'}`,
        `daily size cap:      ${formatBytes(ctx.cfg.contrib.maxBytesPerDay)} (${formatBytes(bytesStagedToday(ctx.dataDir))} staged today)`,
        `staged bundles:      ${staged.length}`,
        '',
      ];
      for (const s of staged) human.push(`  ${s.file.split('/').pop()}  ${formatBytes(s.bytes)}  ${s.records} record(s)`);
      return { human, json: { ok: true, consent, staged, endpoint: ctx.cfg.contrib.endpoint } };
    }

    if (sub === 'preview') {
      const preview = previewContribution(ctx.cfg, ctx.dataDir, {
        limit: flagNumber(ctx.args, 'limit') ?? 200,
        ...(flagBool(ctx.args, 'live') ? { shareCode: true } : {}),
        excludeIds: contributedIds(ctx.dataDir),
      });
      const human: string[] = [];
      human.push(style.bold('contribution preview'));
      human.push('  (nothing has been sent or staged)');
      human.push('');
      human.push(style.bold('consent'));
      for (const line of preview.consentLines) human.push(`  ${line}`);
      human.push('');
      human.push(style.bold('what would be shared'));
      for (const line of preview.summary) human.push(`  ${line}`);
      human.push('');
      human.push(style.bold('example record'));
      human.push(preview.sample.split('\n').map((l) => `  ${l}`).join('\n'));
      human.push('');
      if (preview.wouldUpload) {
        human.push(`${style.green('ready')} to upload. Stage it with \`proto contrib stage\`, send with \`proto contrib upload --yes\`.`);
      } else {
        human.push(`${style.yellow('cannot upload yet')}:`);
        for (const b of preview.uploadBlockers) human.push(`  - ${b}`);
      }
      return { human, json: { ok: true, preview } };
    }

    if (sub === 'stage') {
      const bundle = buildBundle(ctx.cfg, ctx.dataDir, {
        limit: flagNumber(ctx.args, 'limit') ?? 200,
        excludeIds: contributedIds(ctx.dataDir),
      });
      const result = stageBundle(ctx.cfg, ctx.dataDir, bundle);
      const human: string[] = [];
      if (result.ok) {
        human.push(`staged ${bundle.records.length} record(s) (${formatBytes(result.bytes)}) at ${result.path}`);
        human.push('nothing has been sent. Review it, then run `proto contrib upload --yes`.');
      } else {
        human.push(`not staged: ${result.reason}`);
      }
      return { human, json: { ...result, manifest: bundle.manifest }, exitCode: result.ok ? 0 : 1 };
    }

    if (sub === 'upload') {
      const bundle = buildBundle(ctx.cfg, ctx.dataDir, {
        limit: flagNumber(ctx.args, 'limit') ?? 200,
        excludeIds: contributedIds(ctx.dataDir),
      });
      const result = await uploadBundle(ctx.cfg, ctx.dataDir, bundle, { confirmed: flagBool(ctx.args, 'yes') });
      const human = [result.detail];
      if (!result.ok) {
        human.push('');
        human.push('to contribute:');
        human.push('  1. proto contrib consent --global on');
        human.push('  2. proto config set contrib.endpoint https://your-collector.example/ingest');
        human.push('  3. proto contrib preview     # see exactly what would leave');
        human.push('  4. proto contrib upload --yes');
      }
      return { human, json: { ok: result.ok, result }, exitCode: result.ok ? 0 : 1 };
    }

    if (sub === 'outbox') {
      const files = outboxSummary(ctx.dataDir);
      const human = [style.bold('outbox'), ...files.map((f) => `  ${f.file}  ${formatBytes(f.bytes)}  ${f.records} record(s)`)];
      if (files.length === 0) human.push('  (empty)');
      return { human, json: { ok: true, files } };
    }
    if (sub === 'rotate') {
      const identity = rotateIdentity(ctx.dataDir, ctx.cfg);
      const human = [
        `rotated pseudonym -> ${identity.pseudonym} (generation ${identity.generation})`,
        'the previous salt is gone, so new contributions cannot be linked to older ones.',
      ];
      return { human, json: { ok: true, identity } };
    }

    throw new Error(`unknown subcommand "${sub}". Try: status|preview|stage|upload|consent|outbox|rotate`);
  },
};

export const learningCommands: Command[] = [train, evalCommand, replayCommand, contrib];
