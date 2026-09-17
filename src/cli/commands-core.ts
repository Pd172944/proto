/**
 * Core commands: doctor, setup, route, run, models, config.
 */

import { cpus, totalmem } from 'node:os';
import { join } from 'node:path';

import { flagBool, flagList, flagNumber, flagString } from './index.ts';
import type { Command, CommandContext, CommandResult } from './index.ts';
import { buildProviders, buildLocalProvider, rankLocalModels, LOCAL_RUNTIME_DEFAULTS } from '../providers/index.ts';
import { MockProvider } from '../providers/mock.ts';
import { routeTask } from '../router/index.ts';
import { heuristicScore } from '../router/heuristic.ts';
import type { TaskContext, Tier } from '../router/types.ts';
import { runTask } from '../harness/loop.ts';
import { PROVIDER_PROFILES, providerProfile } from '../config/schema.ts';
import { cloudBaseUrl, priceFor, resolveApiKeyFor, saveConfig, writeSecret } from '../config/load.ts';
import { formatBytes, formatDuration, readTextOrNull, resolvePath, fileExists } from '../util/fsx.ts';
import { exec, hasBinary } from '../util/proc.ts';
import { CodebaseIndex, focusFromTask } from '../index/index.ts';
import { HARNESS_VERSION } from '../version.ts';
import { style } from '../util/log.ts';

/* ------------------------------------------------------------------ */
/* Shared helpers                                                      */
/* ------------------------------------------------------------------ */

const MAX_FILE_CHARS = 200_000;

interface ContextFlags {
  task: string;
  files: Array<{ path: string; content: string }>;
  diff?: string;
  constraints?: string[];
  workspace: string;
}

function buildTaskContext(ctx: CommandContext, opts: { requireTask?: boolean } = {}): ContextFlags {
  const task = flagString(ctx.args, 'task') ?? ctx.positionals.join(' ').trim();
  if (opts.requireTask !== false && !task) {
    throw new Error('no task given. Pass it as text: proto run "fix the off-by-one in this loop"');
  }
  const workspace = resolvePath(flagString(ctx.args, 'workspace') ?? process.cwd());
  const filePaths = flagList(ctx.args, 'file');
  const files: Array<{ path: string; content: string }> = [];
  for (const raw of filePaths) {
    const rel = raw.startsWith('/') ? raw : join(workspace, raw);
    const content = readTextOrNull(rel);
    if (content === null) {
      throw new Error(`file not found: ${raw} (resolved to ${rel})`);
    }
    const trimmed = content.length > MAX_FILE_CHARS ? content.slice(0, MAX_FILE_CHARS) : content;
    // Path as given by the user; the model is told to echo this exact path back.
    files.push({ path: raw.replace(/^\.\//, ''), content: trimmed });
  }
  const diff = flagString(ctx.args, 'diff');
  const constraints = flagList(ctx.args, 'constraint');
  return {
    task,
    files,
    ...(diff ? { diff } : {}),
    ...(constraints.length ? { constraints } : {}),
    workspace,
  };
}

function toTaskContext(c: ContextFlags): TaskContext {
  return {
    task: c.task,
    files: c.files,
    ...(c.diff ? { diff: c.diff } : {}),
    ...(c.constraints ? { constraints: c.constraints } : {}),
    workspace: c.workspace,
  };
}

const TIER_FLAG_VALUES: Tier[] = ['local-tiny', 'local', 'cloud-cheap', 'cloud-strong'];

/* ------------------------------------------------------------------ */
/* doctor                                                             */
/* ------------------------------------------------------------------ */

const doctor: Command = {
  name: 'doctor',
  summary: 'check runtimes, providers and the verifier; say exactly what is missing',
  usage: 'proto doctor [--probe-cloud]',
  flags: [
    { name: 'probe-cloud', type: 'boolean', description: 'make a real (tiny) cloud request to validate the API key' },
    { name: 'workspace', type: 'string', description: 'workspace to check for a test command' },
  ],
  async run(ctx): Promise<CommandResult> {
    const human: string[] = [];
    const report: Record<string, unknown> = { version: HARNESS_VERSION };
    const nextSteps: string[] = [];

    human.push(style.bold(`proto ${HARNESS_VERSION} doctor`));
    human.push('');

    // ---- environment ----
    const mem = totalmem();
    human.push(style.bold('environment'));
    human.push(`  node          ${process.version}`);
    human.push(`  platform      ${process.platform}/${process.arch}, ${cpus().length} cpus, ${formatBytes(mem)} RAM`);
    human.push(`  data dir      ${ctx.dataDir}`);
    human.push(`  config        ${fileExists(ctx.configPath) ? ctx.configPath : `${ctx.configPath} (not created yet — using defaults)`}`);
    report['environment'] = {
      node: process.version,
      platform: `${process.platform}-${process.arch}`,
      cpus: cpus().length,
      memoryBytes: mem,
      dataDir: ctx.dataDir,
      configPath: ctx.configPath,
      configExists: fileExists(ctx.configPath),
    };
    for (const w of ctx.warnings) {
      human.push(`  ${style.yellow('warning')}       ${w}`);
    }
    human.push('');

    // ---- local tier ----
    human.push(style.bold('local tier'));
    const local = buildLocalProvider(ctx.cfg);
    const localHealth = await local.health();
    report['local'] = {
      runtime: ctx.cfg.local.runtime,
      baseUrl: ctx.cfg.local.baseUrl,
      model: ctx.cfg.local.model,
      enabled: ctx.cfg.local.enabled,
      health: localHealth,
    };
    human.push(`  runtime       ${ctx.cfg.local.runtime} @ ${ctx.cfg.local.baseUrl}`);
    human.push(`  model         ${ctx.cfg.local.model}`);
    human.push(`  status        ${localHealth.ok ? style.green('ready') : style.yellow('not ready')} — ${localHealth.detail}`);

    const available = rankLocalModels(localHealth.models ?? []);
    if (available.length > 0) {
      human.push(`  available     ${available.slice(0, 6).join(', ')}${available.length > 6 ? ` … (+${available.length - 6})` : ''}`);
    }

    // The most useful thing doctor can do when the configured model is missing but
    // other models are present: name the best candidate and the exact command.
    if (!localHealth.ok && available.length > 0) {
      const best = available[0] as string;
      human.push('');
      human.push(`  ${style.yellow('the configured model is not downloaded, but these are:')}`);
      for (const m of available.slice(0, 5)) {
        const mark = m === best ? style.green(' ← recommended') : '';
        human.push(`    ${m}${mark}`);
      }
      human.push(`  use one with:  ${style.cyan(`proto models use ${best}`)}`);
      human.push(`  or download the configured one:  ${style.cyan(`proto models pull ${ctx.cfg.local.model} --yes`)}`);
      nextSteps.push(`adopt a model you already have: proto models use ${best}`);
    }
    if (localHealth.ok && localHealth.models && localHealth.models.length > 1) {
      const others = available.filter((m) => !m.startsWith(ctx.cfg.local.model));
      if (others.length > 0) human.push(`  switch with   ${style.cyan(`proto models use ${others[0]}`)}`);
    }
    if (localHealth.hint && !(available.length > 0 && !localHealth.ok)) {
      human.push(`  ${style.yellow('hint')}          ${localHealth.hint}`);
      nextSteps.push(localHealth.hint);
    }
    human.push('');

    // ---- cloud tier ----
    human.push(style.bold('cloud tier'));
    const keyPresent = Boolean(resolveApiKeyFor(ctx.cfg, ctx.dataDir));
    const keyRequired = ctx.cfg.cloud.requiresKey !== false;
    const profile = providerProfile(ctx.cfg.cloud.provider);
    human.push(`  provider      ${ctx.cfg.cloud.provider}${profile ? ` (${profile.label})` : ' (unknown provider id)'}`);
    human.push(`  base url      ${cloudBaseUrl(ctx.cfg)}`);
    human.push(`  model         ${ctx.cfg.cloud.model}${ctx.cfg.cloud.cheapModel ? ` (cheap: ${ctx.cfg.cloud.cheapModel})` : ''}`);
    human.push(
      `  api key       ${
        !keyRequired
          ? style.dim('not required (self-hosted endpoint)')
          : keyPresent
            ? style.green('present')
            : style.yellow('missing')
      }`,
    );
    if (ctx.cfg.cloud.thinking !== undefined) {
      human.push(
        `  thinking      ${ctx.cfg.cloud.thinking ? 'on' : style.dim('off (faster; the model answers without reasoning first)')}`,
      );
    }

    // When a key is missing, show which provider keys *are* set. Without this, a
    // user with ANTHROPIC_API_KEY and the default OpenRouter provider sees only
    // "missing" and has no idea why.
    // Anthropic keys are sometimes not scoped to a workspace, in which case every
    // request needs this header. Surface the setting so it is discoverable before the
    // 400, not after.
    if (ctx.cfg.cloud.provider === 'anthropic' && keyPresent) {
      const ws = ctx.cfg.cloud.workspaceId?.trim() ?? process.env['ANTHROPIC_WORKSPACE_ID']?.trim();
      human.push(`  workspace id  ${ws ? style.green(ws) : style.dim('not set (only needed for unscoped keys)')}`);
    }

    if (!keyPresent && keyRequired) {
      const present = PROVIDER_PROFILES.filter((p) =>
        p.keyEnv.some((name) => Boolean(process.env[name]?.trim())),
      );
      if (present.length > 0) {
        human.push(`  ${style.yellow('found instead')} ${present.map((p) => `${p.keyEnv.find((n) => process.env[n]?.trim())} (${p.id})`).join(', ')}`);
        nextSteps.push(`a key for another provider is set — pin it: proto config set cloud.provider ${present[0]?.id}`);
      } else {
        human.push(`  ${style.dim('looked for')}    ${(providerProfile(ctx.cfg.cloud.provider)?.keyEnv ?? []).join(', ')}`);
      }
    }
    human.push(`  enabled       ${ctx.cfg.cloud.enabled ? 'yes' : 'no'}`);
    report['cloud'] = {
      provider: ctx.cfg.cloud.provider,
      baseUrl: cloudBaseUrl(ctx.cfg),
      model: ctx.cfg.cloud.model,
      cheapModel: ctx.cfg.cloud.cheapModel ?? null,
      keyPresent,
      keyRequired,
      enabled: ctx.cfg.cloud.enabled,
      pricePerMTok: priceFor(ctx.cfg, ctx.cfg.cloud.model),
    };
    if (!keyPresent) {
      const envName = profile?.keyEnv[0] ?? 'PROTO_API_KEY';
      nextSteps.push(
        `set a cloud API key: export ${envName}=... or \`proto config set-key ${ctx.cfg.cloud.provider} <key>\` ` +
          `(without it, every task runs local or fails)`,
      );
    }
    if (flagBool(ctx.args, 'probe-cloud')) {
      if (!keyPresent || !ctx.cfg.cloud.enabled) {
        human.push(`  probe         skipped (cloud disabled or no key)`);
      } else {
        const providers = buildProviders(ctx.cfg, ctx.dataDir);
        const health = providers.cloud ? await providers.cloud.health() : null;
        human.push(`  probe         ${health?.ok ? style.green('ok') : style.yellow('failed')} — ${health?.detail ?? 'n/a'}`);
        report['cloudProbe'] = health;
      }
    }
    human.push('');

    // ---- verifier ----
    human.push(style.bold('verifier'));
    const hasPython = await hasBinary('python3');
    const hasNode = await hasBinary('node');
    const hasGit = await hasBinary('git');
    human.push(`  enabled       ${ctx.cfg.verify.enabled ? 'yes' : 'no'}`);
    human.push(`  python3       ${hasPython ? 'present (Python syntax checking enabled)' : style.yellow('missing (Python candidates fall back to delimiter checks)')}`);
    human.push(`  node          ${hasNode ? 'present (JS/TS syntax checking enabled)' : style.yellow('missing')}`);
    human.push(`  git           ${hasGit ? 'present' : style.yellow('missing')}`);
    human.push(`  project tests ${ctx.cfg.verify.runTests ? (ctx.cfg.verify.testCommand ? `enabled: ${ctx.cfg.verify.testCommand}` : style.yellow('runTests is on but verify.testCommand is empty')) : 'disabled (verify.runTests=false)'}`);
    report['verifier'] = {
      enabled: ctx.cfg.verify.enabled,
      python3: hasPython,
      node: hasNode,
      git: hasGit,
      runTests: ctx.cfg.verify.runTests,
      testCommand: ctx.cfg.verify.testCommand ?? null,
    };
    if (ctx.cfg.verify.enabled && !ctx.cfg.verify.runTests) {
      nextSteps.push(
        'consider enabling project tests for higher-confidence verification: ' +
          '`proto config set verify.runTests true` and `proto config set verify.testCommand "npm test"`',
      );
    }
    human.push('');

    human.push('');
    human.push(style.bold('next steps'));
    if (nextSteps.length === 0) {
      human.push('  everything looks ready. Try: proto run "rename this variable" --file <path>');
    } else {
      for (const s of [...new Set(nextSteps)]) human.push(`  - ${s}`);
    }

    return { human, json: { ok: true, report, nextSteps: [...new Set(nextSteps)] } };
  },
};

/* ------------------------------------------------------------------ */
/* setup                                                              */
/* ------------------------------------------------------------------ */

const MODEL_RECOMMENDATIONS = [
  { ram: '8 GB', model: 'qwen2.5-coder:1.5b-instruct', size: '~1.0 GB', note: 'fastest; fine for rename/format/validation/docstring tasks' },
  { ram: '16 GB', model: 'qwen2.5-coder:1.5b-instruct', size: '~1.0 GB', note: 'recommended default; leaves RAM for your editor and browser' },
  { ram: '16 GB', model: 'qwen2.5-coder:7b-instruct-q4_K_M', size: '~4.7 GB', note: 'noticeably better reasoning, ~3x slower; good on AC power' },
  { ram: '32 GB+', model: 'qwen2.5-coder:14b-instruct-q4_K_M', size: '~9 GB', note: 'best local quality, but no longer "very fast"' },
];

const setup: Command = {
  name: 'setup',
  summary: 'print the exact commands to install a local runtime and download a model (runs nothing)',
  usage: 'proto setup [--runtime ollama|mlx|llamacpp] [--download --yes]',
  flags: [
    { name: 'runtime', type: 'string', description: 'ollama (default) | mlx | llamacpp' },
    { name: 'download', type: 'boolean', description: 'actually run the download (requires --yes)' },
    { name: 'yes', type: 'boolean', description: 'confirm you want the download to run' },
  ],
  async run(ctx): Promise<CommandResult> {
    const runtime = (flagString(ctx.args, 'runtime') ?? ctx.cfg.local.runtime) as keyof typeof LOCAL_RUNTIME_DEFAULTS;
    const human: string[] = [];
    const commands: Record<string, string[]> = {};

    human.push(style.bold('Local model setup'));
    human.push('');
    human.push('This command downloads nothing by itself. It prints exactly what to run.');
    human.push('');

    if (runtime === 'ollama') {
      commands['install'] = [
        '# Option A — Homebrew:',
        'brew install ollama',
        '# Option B — desktop app: https://ollama.com/download',
        '',
        '# start the server (the desktop app does this for you):',
        'ollama serve',
      ];
      commands['pull'] = ['ollama pull qwen2.5-coder:1.5b-instruct'];
      human.push(style.bold('1. install Ollama'));
      human.push(...commands['install'].map((l) => `  ${l}`));
      human.push('');
      human.push(style.bold('2. download a model'));
      human.push(...commands['pull'].map((l) => `  ${l}`));
      human.push('');
      human.push(style.bold('3. point the harness at it (these are the defaults, so usually nothing to do)'));
      human.push('  proto config set local.runtime ollama');
      human.push('  proto config set local.baseUrl http://127.0.0.1:11434');
      human.push('  proto config set local.model qwen2.5-coder:1.5b-instruct');
      human.push('  proto doctor');
    } else if (runtime === 'mlx') {
      commands['install'] = [
        `python3 -m venv ${join(ctx.dataDir, 'venv')}`,
        `${join(ctx.dataDir, 'venv', 'bin', 'pip')} install --upgrade pip`,
        `${join(ctx.dataDir, 'venv', 'bin', 'pip')} install mlx-lm`,
      ];
      commands['pull'] = [
        `${join(ctx.dataDir, 'venv', 'bin', 'python')} -m mlx_lm.generate --model mlx-community/Qwen2.5-Coder-1.5B-Instruct-4bit --prompt "hello" --max-tokens 32`,
      ];
      human.push(style.bold('1. install MLX-LM (this is also the trainer)'));
      human.push(...commands['install'].map((l) => `  ${l}`));
      human.push('');
      human.push(style.bold('2. download a model (first run caches it)'));
      human.push(...commands['pull'].map((l) => `  ${l}`));
      human.push('');
      human.push(style.bold('3. serve it and point the harness at it'));
      human.push(`  ${join(ctx.dataDir, 'venv', 'bin', 'python')} -m mlx_lm.server --model mlx-community/Qwen2.5-Coder-1.5B-Instruct-4bit --port 8080`);
      human.push('  proto config set local.runtime mlx');
      human.push('  proto config set local.baseUrl http://127.0.0.1:8080');
    } else {
      commands['install'] = [
        'git clone https://github.com/ggml-org/llama.cpp && cd llama.cpp && cmake -B build && cmake --build build -j',
      ];
      commands['pull'] = [
        '# download a GGUF (e.g. from Hugging Face) into ./models then:',
        './build/bin/llama-server -m models/qwen2.5-coder-1.5b-instruct-q4_k_m.gguf --port 8080 -c 8192',
      ];
      human.push(style.bold('1. build llama.cpp'));
      human.push(...commands['install'].map((l) => `  ${l}`));
      human.push('');
      human.push(style.bold('2. get a GGUF model and serve it'));
      human.push(...commands['pull'].map((l) => `  ${l}`));
    }

    human.push('');
    human.push(style.bold('recommended models for your machine'));
    human.push('  RAM      model                                   download   notes');
    for (const r of MODEL_RECOMMENDATIONS) {
      human.push(`  ${r.ram.padEnd(8)} ${r.model.padEnd(39)} ${r.size.padEnd(10)} ${r.note}`);
    }
    human.push('');
    human.push(style.bold('what "very fast" means here'));
    human.push('  A 1.5B 4-bit model on Apple Silicon decodes roughly 60-120 tokens/sec.');
    human.push('  The harness keeps the model warm for local.keepAliveSec (default 300s) and then unloads it,');
    human.push('  so it does not sit in RAM all day.');

    // Optional: actually run the download, only with an explicit double opt-in.
    const wantsDownload = flagBool(ctx.args, 'download');
    const confirmed = flagBool(ctx.args, 'yes');
    let downloadResult: unknown = null;
    if (wantsDownload && !confirmed) {
      human.push('');
      human.push(style.yellow('refusing to download: pass --yes as well as --download to confirm'));
    } else if (wantsDownload && confirmed) {
      const pull = commands['pull'] ?? [];
      const lastLine = pull.filter((l) => l && !l.startsWith('#')).pop();
      if (!lastLine) {
        human.push('');
        human.push(style.yellow('nothing to run'));
      } else {
        human.push('');
        human.push(`running: ${lastLine}`);
        const res = await exec('/bin/sh', ['-c', lastLine], { timeoutMs: 30 * 60_000, maxOutputBytes: 256 * 1024 });
        downloadResult = { code: res.code, stdout: res.stdout.slice(-2000), stderr: res.stderr.slice(-2000) };
        human.push(res.code === 0 ? style.green('download finished') : style.red(`download failed (exit ${res.code})`));
      }
    }

    return {
      human,
      json: {
        ok: true,
        runtime,
        commands,
        recommendations: MODEL_RECOMMENDATIONS,
        downloaded: downloadResult,
      },
    };
  },
};

/* ------------------------------------------------------------------ */
/* route                                                              */
/* ------------------------------------------------------------------ */

const route: Command = {
  name: 'route',
  summary: 'show the routing decision for a task without calling any model',
  usage: 'proto route "<task>" [--file path]... [--diff path] [--explain]',
  flags: [
    { name: 'task', type: 'string', description: 'task text (alternative to a positional)' },
    { name: 'file', type: 'string', multiple: true, description: 'file in scope (repeatable)' },
    { name: 'diff', type: 'string', description: 'path to a diff/patch to consider' },
    { name: 'constraint', type: 'string', multiple: true, description: 'explicit constraint (repeatable)' },
    { name: 'workspace', type: 'string', description: 'workspace root for relative --file paths' },
    { name: 'explain', type: 'boolean', description: 'show every contributing reason and feature' },
    { name: 'offline', type: 'boolean', description: 'skip the local runtime health probe' },
  ],
  async run(ctx): Promise<CommandResult> {
    const built = buildTaskContext(ctx);
    const diffText = built.diff ? readTextOrNull(resolvePath(built.diff, built.workspace)) ?? undefined : undefined;
    const taskCtx = toTaskContext({ ...built, ...(diffText ? { diff: diffText } : {}) });
    const decision = await routeTask({
      cfg: ctx.cfg,
      dataDir: ctx.dataDir,
      ctx: taskCtx,
      ...(flagBool(ctx.args, 'offline') ? { offline: true } : {}),
    });


    const human: string[] = [];
    human.push(style.bold('routing decision'));
    human.push(`  tier          ${style.cyan(decision.tier)}`);
    human.push(`  reason        ${decision.reason}`);
    human.push(`  p(local ok)   ${decision.pLocalSuccess.toFixed(3)}   difficulty ${decision.difficulty.toFixed(3)}   class ${decision.taskClass}`);
    human.push(
      `  est. cost     local $${decision.expected.localCostUsd.toFixed(4)} vs cloud $${decision.expected.cloudCostUsd.toFixed(4)}`,
    );
    human.push(
      `  est. latency  local ${formatDuration(decision.expected.localLatencyMs)} vs cloud ${formatDuration(decision.expected.cloudLatencyMs)}`,
    );
    if (decision.forced) human.push(`  ${style.yellow('forced')}        the preferred tier was unavailable`);
    human.push('');
    human.push(style.bold('why'));
    for (const r of decision.reasons) human.push(`  - ${r}`);
    if (decision.vetoes.length) {
      human.push('');
      human.push(style.bold('vetoes'));
      for (const v of decision.vetoes) human.push(`  - ${v}`);
    }
    if (flagBool(ctx.args, 'explain')) {
      const h = heuristicScore(decision.features);
      human.push('');
      human.push(style.bold('heuristic contributions (largest first)'));
      for (const c of h.contributions.slice(0, 14)) human.push(`  ${c.feature.padEnd(16)} ${c.note}`);
      human.push('');
      human.push(style.bold('features'));
      for (const [k, v] of Object.entries(decision.features)) {
        if (Array.isArray(v)) human.push(`  ${k.padEnd(28)} ${v.join(', ')}`);
        else if (typeof v === 'object') continue;
        else human.push(`  ${k.padEnd(28)} ${String(v)}`);
      }
    }

    return { human, json: { ok: true, decision } };
  },
};

/* ------------------------------------------------------------------ */
/* run                                                                */
/* ------------------------------------------------------------------ */

const run: Command = {
  name: 'run',
  summary: 'run a task end to end: route, attempt, verify, escalate',
  usage: 'proto run "<task>" [--file path]... [--apply] [--tier tier] [--dry-run]',
  flags: [
    { name: 'task', type: 'string', description: 'task text (alternative to a positional)' },
    { name: 'file', type: 'string', multiple: true, description: 'file in scope (repeatable)' },
    { name: 'diff', type: 'string', description: 'path to a diff/patch to consider' },
    { name: 'constraint', type: 'string', multiple: true, description: 'explicit constraint (repeatable)' },
    { name: 'workspace', type: 'string', description: 'workspace root (default: cwd)' },
    { name: 'apply', type: 'boolean', description: 'write verified edits to disk (default: never write)' },
    { name: 'tier', type: 'string', description: 'force a tier: local-tiny|local|cloud-cheap|cloud-strong' },
    { name: 'dry-run', type: 'boolean', description: 'route only; make no model calls' },
    { name: 'no-tests', type: 'boolean', description: 'skip the configured project test command' },
    { name: 'no-repair', type: 'boolean', description: 'do not let the local model retry after a failed verification' },
    { name: 'unload', type: 'boolean', description: 'ask the local runtime to unload the model when finished' },
    { name: 'mock', type: 'boolean', description: 'use deterministic mock providers (no network, no local runtime)' },
    { name: 'show-output', type: 'boolean', description: 'print the raw model output' },
  ],
  async run(ctx): Promise<CommandResult> {
    const built = buildTaskContext(ctx);
    const diffPath = built.diff ? resolvePath(built.diff, built.workspace) : null;
    const diffText = diffPath ? readTextOrNull(diffPath) : null;
    if (diffPath && diffText === null) throw new Error(`diff file not found: ${built.diff}`);
    const taskCtx = toTaskContext({ ...built, ...(diffText ? { diff: diffText } : {}) });

    const tierFlag = flagString(ctx.args, 'tier');
    if (tierFlag && !TIER_FLAG_VALUES.includes(tierFlag as Tier)) {
      throw new Error(`--tier must be one of ${TIER_FLAG_VALUES.join(', ')}`);
    }

    const useMock = flagBool(ctx.args, 'mock');

    const result = await runTask({
      cfg: ctx.cfg,
      dataDir: ctx.dataDir,
      ctx: taskCtx,
      ...(flagBool(ctx.args, 'apply') ? { apply: true } : {}),
      ...(tierFlag ? { forceTier: tierFlag as Tier } : {}),
      ...(flagBool(ctx.args, 'dry-run') ? { dryRun: true } : {}),
      ...(flagBool(ctx.args, 'no-repair') ? { maxRepair: 0 } : {}),
      ...(useMock ? { localProvider: new MockProvider({ id: 'mock-local', kind: 'local' }) } : {}),
      ...(flagBool(ctx.args, 'mock') ? { cloudProvider: new MockProvider({ id: 'mock-cloud', kind: 'cloud', model: 'mock-strong' }) } : {}),
      ...(flagBool(ctx.args, 'unload') ? { unloadLocalAfter: true } : {}),
    });

    const human: string[] = [];
    human.push(style.bold('run'));
    human.push(`  tier          ${style.cyan(result.decision.tier)} — ${result.decision.reason}`);
    human.push(`  status        ${renderStatus(result.status)}`);
    if (result.attempts.length > 0) {
      human.push(
        `  attempts      ${result.attempts.length} ` +
          `(${result.attempts.map((a) => `${a.tier}/${a.source}${a.error ? ' ERR' : ''}`).join(', ')})`,
      );
      const latency = result.attempts.reduce((a, x) => a + x.latencyMs, 0);
      human.push(`  cost          $${result.totalCostUsd.toFixed(4)}   latency ${formatDuration(latency)}`);
    }

    if (result.report) {
      human.push('');
      human.push(style.bold(`verification: ${result.report.passed ? style.green('passed') : style.red('failed')} (score ${result.report.score.toFixed(2)})`));
      for (const c of result.report.checks) {
        if (c.ok && c.severity === 'info') continue;
        const icon = c.ok ? style.dim('~') : c.severity === 'error' ? style.red('x') : style.yellow('!');
        human.push(`  ${icon} ${c.detail}`);
        if (c.evidence && flagBool(ctx.args, 'explain')) human.push(`      ${c.evidence.split('\n').slice(0, 4).join('\n      ')}`);
      }
      if (result.report.applied.length > 0) {
        human.push('');
        human.push(style.bold('proposed changes'));
        for (const f of result.report.applied) {
          human.push(`  ${f.path} (~${f.changedLines} line(s))`);
        }
      }
      if (result.report.candidate?.summary) {
        human.push('');
        human.push(`summary: ${result.report.candidate.summary}`);
      }
    }

    if (result.writtenFiles.length > 0) {
      human.push('');
      human.push(style.green(`wrote ${result.writtenFiles.length} file(s): ${result.writtenFiles.join(', ')}`));
    } else if (!flagBool(ctx.args, 'apply') && result.report?.passed) {
      human.push('');
      human.push('dry run: nothing was written. Re-run with --apply to write the change.');
    }

    for (const w of result.warnings) human.push(`${style.yellow('warning')}       ${w}`);

    if (flagBool(ctx.args, 'show-output') && result.finalText) {
      human.push('');
      human.push(style.bold('raw output'));
      human.push(result.finalText);
    }

    const exitCode = result.status === 'failed' ? 3 : 0;
    return { human, json: { ok: result.status !== 'failed', result }, exitCode };
  },
};

function renderStatus(status: string): string {
  switch (status) {
    case 'local-success':
      return `${style.green('local-success')} (solved by the cheap tier)`;
    case 'cloud-success':
      return `${style.green('cloud-success')} (cloud was the first choice)`;
    case 'escalated-cloud-success':
      return `${style.yellow('escalated-cloud-success')} (local failed verification, cloud fixed it)`;
    case 'dry-run':
      return 'dry-run (routing only)';
    case 'failed':
      return style.red('failed') + ' (no verified answer)';
    default:
      return status;
  }
}

/* ------------------------------------------------------------------ */
/* models                                                             */
/* ------------------------------------------------------------------ */

const models: Command = {
  name: 'models',
  summary: 'list local models; adopt one with `use`; print download commands',
  usage: 'proto models [list|use <name>|pull <name>] [--yes]',
  flags: [
    { name: 'yes', type: 'boolean', description: 'confirm running a download command' },
    { name: 'runtime', type: 'string', description: 'override the local runtime for this listing' },
  ],
  async run(ctx): Promise<CommandResult> {
    const sub = ctx.positionals[0] ?? 'list';
    const human: string[] = [];

    if (sub === 'list') {
      const local = buildLocalProvider(ctx.cfg);
      const health = await local.health();
      human.push(style.bold(`local models (${ctx.cfg.local.runtime} @ ${ctx.cfg.local.baseUrl})`));
      if (!health.ok && !health.models?.length) {
        human.push(`  ${style.yellow('runtime not reachable')} — ${health.detail}`);
        if (health.hint) human.push(`  hint: ${health.hint}`);
      } else {
        for (const m of health.models ?? []) {
          const mark = m.startsWith(ctx.cfg.local.model) ? style.green(' <- configured') : '';
          human.push(`  ${m}${mark}`);
        }
      }
      human.push('');
      return {
        human,
        json: { ok: true, runtime: ctx.cfg.local.runtime, baseUrl: ctx.cfg.local.baseUrl, health },
      };
    }

    if (sub === 'use') {
      const name = ctx.positionals[1];
      if (!name) {
        const health = await buildLocalProvider(ctx.cfg).health();
        const ranked = rankLocalModels(health.models ?? []);
        human.push(ranked.length > 0
          ? `usage: proto models use <name>\n\navailable:\n${ranked.map((m) => `  ${m}`).join('\n')}`
          : 'usage: proto models use <name>\n\n(no local models found — is the runtime running?)');
        return { human, json: { ok: ranked.length > 0, available: ranked } };
      }
      const health = await buildLocalProvider(ctx.cfg).health();
      const known = health.models ?? [];
      const next = structuredClone(ctx.cfg);
      next.local.model = name;
      const saved = saveConfig(next, ctx.dataDir);
      human.push(`local.model = ${name}`);
      human.push(`wrote ${saved}`);
      if (known.length > 0 && !known.some((m) => m === name || m.startsWith(`${name}:`))) {
        human.push(style.yellow(`note: "${name}" does not look downloaded. Available: ${known.join(', ')}`));
        human.push(`      fetch it with: proto models pull ${name} --yes`);
      } else {
        human.push('next: `proto code --local` or `proto run "<task>" --tier local`');
      }
      return { human, json: { ok: true, model: name, path: saved, available: known } };
    }

    if (sub === 'pull') {
      const name = ctx.positionals[1];
      if (!name) throw new Error('usage: proto models pull <model-name>');
      const runtime = flagString(ctx.args, 'runtime') ?? ctx.cfg.local.runtime;
      const command =
        runtime === 'ollama'
          ? `ollama pull ${name}`
          : runtime === 'mlx'
            ? `${join(ctx.dataDir, 'venv', 'bin', 'python')} -m mlx_lm.generate --model ${name} --prompt hi --max-tokens 8`
            : `# download a GGUF for ${name} and serve it with llama-server (see \`proto setup --runtime llamacpp\`)`;

      human.push(style.bold('download command (nothing has been run)'));
      human.push(`  ${command}`);
      human.push('');
      if (!flagBool(ctx.args, 'yes')) {
        human.push(`to actually run it, re-run with --yes:`);
        human.push(`  proto models pull ${name} --yes`);
        return { human, json: { ok: true, ran: false, command } };
      }
      const res = await exec('/bin/sh', ['-c', command], { timeoutMs: 60 * 60_000, maxOutputBytes: 128 * 1024 });
      human.push(res.code === 0 ? style.green('done') : style.red(`failed with exit ${res.code}`));
      human.push(res.stdout.split('\n').slice(-15).join('\n'));
      if (res.stderr) human.push(res.stderr.split('\n').slice(-10).join('\n'));
      return { human, json: { ok: res.code === 0, ran: true, command, code: res.code }, exitCode: res.code === 0 ? 0 : 1 };
    }

    throw new Error(`unknown subcommand "${sub}". Try: proto models list | proto models use <name> | proto models pull <name>`);
  },
};

/* ------------------------------------------------------------------ */
/* config                                                             */
/* ------------------------------------------------------------------ */

const config: Command = {
  name: 'config',
  summary: 'inspect and edit configuration (path|show|get|set|set-key|providers)',
  usage: 'proto config <path|show|get <key>|set <key> <value>|set-key <provider> <key>|providers>',
  flags: [
    { name: 'local', type: 'boolean', description: '(set-key) store the key in var/secrets.json instead of echoing it' },
  ],
  async run(ctx): Promise<CommandResult> {
    const sub = ctx.positionals[0] ?? 'show';
    const human: string[] = [];

    if (sub === 'path') {
      return { human: [ctx.configPath], json: { ok: true, path: ctx.configPath, dataDir: ctx.dataDir } };
    }

    if (sub === 'providers') {
      human.push(style.bold('known providers'));
      human.push('  id           api        env var                default model');
      for (const p of PROVIDER_PROFILES) {
        human.push(
          `  ${p.id.padEnd(12)} ${p.api.padEnd(10)} ${(p.keyEnv[0] ?? '').padEnd(22)} ${p.defaultModel}`,
        );
      }
      return { human, json: { ok: true, providers: PROVIDER_PROFILES } };
    }

    if (sub === 'show') {
      const redacted = JSON.stringify(ctx.cfg, null, 2);
      human.push(redacted);
      return { human, json: { ok: true, config: ctx.cfg, configPath: ctx.configPath, dataDir: ctx.dataDir } };
    }

    if (sub === 'get') {
      const key = ctx.positionals[1];
      if (!key) throw new Error('usage: proto config get <dotted.key>');
      const value = getPath(ctx.cfg, key);
      if (value === undefined) throw new Error(`no such config key: ${key}`);
      human.push(typeof value === 'string' ? value : JSON.stringify(value, null, 2));
      return { human, json: { ok: true, key, value } };
    }

    if (sub === 'set') {
      const key = ctx.positionals[1];
      const rawValue = ctx.positionals.slice(2).join(' ');
      if (!key || !rawValue) throw new Error('usage: proto config set <dotted.key> <value>');
      const next = structuredClone(ctx.cfg);
      setPath(next, key, coerce(rawValue));
      const path = saveConfig(next, ctx.dataDir);
      human.push(`set ${key} = ${JSON.stringify(coerce(rawValue))}`);
      human.push(`wrote ${path}`);
      return { human, json: { ok: true, key, value: coerce(rawValue), path } };
    }

    if (sub === 'set-key') {
      const provider = ctx.positionals[1];
      const key = ctx.positionals[2];
      if (!provider || !key) {
        throw new Error(
          'usage: proto config set-key <provider> <key>\n' +
            'prefer the environment variable instead (the key then never touches disk):\n' +
            `  export ${providerProfile(provider ?? '')?.keyEnv[0] ?? 'PROTO_API_KEY'}=...`,
        );
      }
      const path = writeSecret(ctx.dataDir, provider, key);
      human.push(`stored a key for "${provider}" in ${path} (mode 0600)`);
      human.push('note: environment variables take precedence over this file');
      return { human, json: { ok: true, provider, path } };
    }

    throw new Error(`unknown subcommand "${sub}". Try: path|show|get|set|set-key|providers`);
  },
};

function getPath(obj: unknown, path: string): unknown {
  let cur: unknown = obj;
  for (const part of path.split('.')) {
    if (cur === null || typeof cur !== 'object') return undefined;
    cur = (cur as Record<string, unknown>)[part];
  }
  return cur;
}

function setPath(obj: object, path: string, value: unknown): void {
  const parts = path.split('.');
  let cur = obj as Record<string, unknown>;
  for (let i = 0; i < parts.length - 1; i++) {
    const part = parts[i] as string;
    const next = cur[part];
    if (next === null || typeof next !== 'object' || Array.isArray(next)) {
      cur[part] = {};
    }
    cur = cur[part] as Record<string, unknown>;
  }
  cur[parts[parts.length - 1] as string] = value;
}

function coerce(raw: string): string | number | boolean | null | string[] {
  if (raw === 'true') return true;
  if (raw === 'false') return false;
  if (raw === 'null') return null;
  if (/^-?\d+(\.\d+)?$/.test(raw)) return Number(raw);
  if (raw.startsWith('[') && raw.endsWith(']')) {
    try {
      const parsed = JSON.parse(raw) as unknown;
      if (Array.isArray(parsed)) return parsed.map(String);
    } catch {
      /* fall through to string */
    }
  }
  return raw;
}


/* ------------------------------------------------------------------ */
/* index                                                              */
/* ------------------------------------------------------------------ */

const indexCommand: Command = {
  name: 'index',
  summary: 'build and inspect the codebase index: symbols, references and the ranked repo map',
  usage: 'proto index [--force] [--stats] [--map "focus"] [--symbol NAME] [--refs NAME] [--outline PATH]',
  flags: [
    { name: 'force', type: 'boolean', description: 'ignore the cache and re-extract every file' },
    { name: 'stats', type: 'boolean', description: 'print index statistics only' },
    { name: 'map', type: 'string', description: 'render the ranked repository map, using this text as the focus' },
    { name: 'symbol', type: 'string', description: 'find where a symbol is defined' },
    { name: 'refs', type: 'string', description: 'find where a symbol is referenced' },
    { name: 'outline', type: 'string', description: 'list the definitions in one file' },
    { name: 'budget', type: 'number', description: 'token budget for --map (default 2000)' },
    { name: 'workspace', type: 'string', description: 'project root (default: current directory)' },
  ],
  async run(ctx): Promise<CommandResult> {
    const workspace = resolvePath(flagString(ctx.args, 'workspace') ?? process.cwd(), process.cwd());
    const index = new CodebaseIndex(workspace, ctx.dataDir);

    const warm = await index.refresh(flagBool(ctx.args, 'force'));
    const stats = index.stats;
    const human: string[] = [];
    const json: Record<string, unknown> = { workspace, stats };

    // The build is worth reporting even when another subcommand was asked for: a slow
    // first call and a stale cache look identical from the outside otherwise.
    if (!flagBool(ctx.args, 'stats')) {
      human.push(style.bold(`proto index — ${workspace}`));
      human.push('');
      human.push(
        `  ${stats.files} file(s) indexed, ${stats.symbols} definition(s)` +
          `  (${stats.changed} re-extracted, ${stats.removed} removed, ${stats.buildMs}ms)`,
      );
      human.push(`  cache  ${formatBytes(stats.cacheBytes)}  ${stats.changed === 0 && warm.files > 0 ? '(warm)' : ''}`);
      for (const w of index.warnings) human.push(`  ${style.dim('note')}  ${w}`);
      human.push('');
    }

    const symbol = flagString(ctx.args, 'symbol');
    const refs = flagString(ctx.args, 'refs');
    const outline = flagString(ctx.args, 'outline');
    const focus = flagString(ctx.args, 'map');

    if (flagBool(ctx.args, 'stats')) {
      human.push(`${stats.files} files, ${stats.symbols} symbols, ${stats.changed} changed, ${stats.removed} removed, ${stats.buildMs}ms, cache ${formatBytes(stats.cacheBytes)}`);
      return { human, json: { workspace, stats, warnings: index.warnings }, exitCode: 0 };
    }

    if (focus !== undefined) {
      const budget = flagNumber(ctx.args, 'budget') ?? 2000;
      const f = focusFromTask(focus, index.graph);
      const map = index.repoMap({ budgetChars: budget * 4, focus: f.paths, focusSymbols: f.symbols });
      human.push(map.text);
      human.push('');
      human.push(style.dim(`  ${map.files.length}/${map.totalFiles} file(s) shown; focus ${[...f.paths, ...f.symbols].join(', ') || '(none)'}`));
      json['map'] = { text: map.text, files: map.files.map((x) => x.path), total: map.totalFiles, focus: f };
      return { human, json, exitCode: 0 };
    }

    if (symbol !== undefined) {
      const { hits, exact } = index.findSymbol(symbol, 40);
      human.push(exact ? `definitions of ${symbol}:` : `no exact match for ${symbol}; names containing it:`);
      for (const h of hits) human.push(`  ${h.path}:${h.line}  ${h.kind}${h.signature ? `  ${h.signature}` : ''}`);
      if (hits.length === 0) human.push('  (none)');
      json['symbol'] = { name: symbol, exact, hits };
      return { human, json, exitCode: 0 };
    }

    if (refs !== undefined) {
      const hits = index.findReferences(refs, 200);
      human.push(`references to ${refs} in ${new Set(hits.map((h) => h.path)).size} file(s):`);
      for (const h of hits) human.push(`  ${h.path}:${h.line}: ${h.text ?? ''}`);
      if (hits.length === 0) human.push('  (none)');
      json['references'] = { name: refs, hits };
      return { human, json, exitCode: 0 };
    }

    if (outline !== undefined) {
      const text = index.outline(outline);
      human.push(text ?? `not in the index: ${outline}`);
      json['outline'] = { path: outline, text };
      return { human, json, exitCode: text === null ? 1 : 0 };
    }

    return { human, json, exitCode: 0 };
  },
};

export const coreCommands: Command[] = [doctor, setup, route, run, models, config, indexCommand];
