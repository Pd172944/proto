/**
 * Training orchestration: status, the nightly tick, and router refresh.
 *
 * The central design idea is **two speeds of learning**:
 *
 *   - **Fast loop (milliseconds, runs constantly).** The router's logistic
 *     scorer is retrained from the episode log. This costs essentially nothing,
 *     so it happens on every tick regardless of whether the machine is idle. It
 *     is where most of the practical benefit comes from: the harness learns
 *     which tasks this user's local model can actually handle.
 *
 *   - **Slow loop (minutes, gated and deferred).** LoRA fine-tuning of the local
 *     model. This is the expensive one, so it is gated on power, thermal state,
 *     idle load, time window, budget, and a minimum amount of new data — and it
 *     can be skipped for weeks without anything breaking.
 *
 * Keeping them separate is what lets the project honestly claim "RL at
 * basically zero compute": the fast loop genuinely costs nothing and delivers
 * most of the value; the slow loop is a background luxury.
 */

import { join } from 'node:path';
import { rmSync } from 'node:fs';

import { adaptersRoot, getActiveAdapter, listAdapters, setActiveAdapter, writeAdapterMetrics } from './adapter.ts';
import { JobQueue, addMinutes, loadTrainState, minutesUsedToday, saveTrainState } from './jobs.ts';
import type { TrainJob, TrainState } from './jobs.ts';
import { buildLoraCommand, newAdapterPath, preflight, prepareMlxDataDir, runTraining, serveCommands } from './mlx.ts';
import type { MlxPreflight } from './mlx.ts';
import { evaluateGates, planSession, priorityPlan, readSystemState } from './scheduler.ts';
import type { GateResult, SessionPlan, SystemState } from './scheduler.ts';
import type { ProtoConfig } from '../config/schema.ts';
import { EpisodeStore } from '../memory/store.ts';
import { buildDatasets, describeDatasets } from '../memory/datasets.ts';
import type { BuiltDatasets } from '../memory/datasets.ts';
import { MIN_TRAINING_SAMPLES, LogisticScorer, featureImportance, trainLogistic } from '../router/learned.ts';
import { loadScorer, routerWeightsPath, saveScorer } from '../router/index.ts';
import { ensureDir } from '../util/fsx.ts';

/* ------------------------------------------------------------------ */
/* Fast loop: router refresh                                           */
/* ------------------------------------------------------------------ */

export interface RouterRefreshResult {
  trained: boolean;
  sampleCount: number;
  positives: number;
  accuracy: number;
  auc: number;
  brier: number;
  weightsPath: string;
  topFeatures: Array<{ name: string; weight: number }>;
  reason?: string;
}

/**
 * Retrain the routing scorer from logged episodes.
 *
 * Deliberately unconditional: this is bounded by the size of the episode log
 * (thousands of rows x 43 features) and takes single-digit milliseconds. Gating
 * it behind "the machine is idle" would be absurd.
 */
export function refreshRouter(
  cfg: ProtoConfig,
  dataDir: string,
  opts: { write?: boolean; minSamples?: number } = {},
): RouterRefreshResult {
  const store = new EpisodeStore(dataDir);
  const built = buildDatasets(store, cfg, { write: false, maxRouter: 20_000 });
  const samples = built.router.samples.map((s) => ({ x: s.x, y: s.y, w: s.w }));
  const minSamples = opts.minSamples ?? MIN_TRAINING_SAMPLES;
  const weightsPath = routerWeightsPath(dataDir);

  if (samples.length < minSamples) {
    return {
      trained: false,
      sampleCount: samples.length,
      positives: samples.filter((s) => s.y === 1).length,
      accuracy: 0,
      auc: 0.5,
      brier: 0,
      weightsPath,
      topFeatures: [],
      reason: `only ${samples.length} labelled episode(s); need ${minSamples} before the learned scorer is used`,
    };
  }

  const { scorer, metrics } = trainLogistic(samples, { epochs: 400, learningRate: 0.08, l2: 0.02, seed: 12345 });
  if (opts.write !== false) {
    saveScorer(dataDir, scorer, [
      `trained from ${metrics.n} labelled episodes (${metrics.positives} local successes)`,
      'labels come from verification pass/fail, weighted by behavior propensity',
    ]);
  }
  return {
    trained: true,
    sampleCount: metrics.n,
    positives: metrics.positives,
    accuracy: metrics.accuracy,
    auc: metrics.auc,
    brier: metrics.brier,
    weightsPath,
    topFeatures: featureImportance(scorer).slice(0, 8),
  };
}

/** Load the current scorer's feature importances for display. */
export interface ScorerSummary {
  present: boolean;
  sampleCount: number;
  trainedAt: string | null;
  metrics: RouterRefreshResult | { accuracy: number; auc: number; brier: number; n: number } | null;
  topFeatures: Array<{ name: string; weight: number }>;
  error?: string;
}

export function currentScorerSummary(dataDir: string): ScorerSummary {
  const { scorer, error } = loadScorer(dataDir);
  if (!scorer) {
    return {
      present: false,
      sampleCount: 0,
      trainedAt: null,
      metrics: null,
      topFeatures: [],
      ...(error ? { error } : {}),
    };
  }
  return {
    present: true,
    sampleCount: scorer.sampleCount,
    trainedAt: scorer.trainedAt,
    metrics: scorer.metrics,
    topFeatures: featureImportance(scorer).slice(0, 12),
  };
}

/* ------------------------------------------------------------------ */
/* Status                                                             */
/* ------------------------------------------------------------------ */

export interface TrainStatus {
  enabled: boolean;
  system: SystemState;
  state: TrainState;
  gates: GateResult;
  plan: SessionPlan;
  priority: { wrap: string; description: string };
  preflight: MlxPreflight;
  datasets: BuiltDatasets;
  datasetSummary: string[];
  newEpisodes: number;
  minutesUsedToday: number;
  activeAdapter: ReturnType<typeof getActiveAdapter>;
  adapters: ReturnType<typeof listAdapters>;
  recentJobs: TrainJob[];
  baseModel: string;
  /** True when LoRA can run right now. */
  readyToRun: boolean;
}

export async function trainStatus(cfg: ProtoConfig, dataDir: string, now: Date = new Date()): Promise<TrainStatus> {
  const store = new EpisodeStore(dataDir);
  const state = loadTrainState(dataDir);
  const system = await readSystemState(now);
  const datasets = buildDatasets(store, cfg, { write: false });
  const newEpisodes = countNewEpisodes(dataDir, state);
  const minutesToday = minutesUsedToday(state, now);

  const gateInput = {
    cfg,
    state: system,
    newEpisodes,
    minutesUsedToday: minutesToday,
    hasTrainingData: datasets.sft.samples.length > 0 || datasets.dpo.samples.length > 0,
  };
  const gates = evaluateGates(gateInput);
  const plan = planSession(cfg, minutesToday, cfg.train.lora.mode);
  const pf = await preflight(dataDir);
  const queue = new JobQueue(dataDir);

  return {
    enabled: cfg.train.enabled,
    system,
    state,
    gates,
    plan,
    priority: priorityPlan(),
    preflight: pf,
    datasets,
    datasetSummary: describeDatasets(datasets),
    newEpisodes,
    minutesUsedToday: minutesToday,
    activeAdapter: getActiveAdapter(dataDir),
    adapters: listAdapters(dataDir),
    recentJobs: queue.recent(8),
    baseModel: cfg.train.baseModel,
    readyToRun: gates.allowed && pf.ok,
  };
}

function countNewEpisodes(dataDir: string, state: TrainState): number {
  const store = new EpisodeStore(dataDir);
  const episodes = store.readAll();
  if (!state.highWaterEpisodeId) return episodes.length;
  // ULIDs are lexicographically time-ordered, so this is an O(n) scan with no
  // timestamp parsing and no clock-skew assumptions.
  return episodes.filter((e) => e.id > (state.highWaterEpisodeId as string)).length;
}

/* ------------------------------------------------------------------ */
/* Tick                                                               */
/* ------------------------------------------------------------------ */

export interface TickOptions {
  /** Ignore the gates. Still respects the wall-clock cap and the watchdog. */
  force?: boolean;
  /** Do everything except actually running the trainer. */
  dryRun?: boolean;
  /** Override the dataset/mode for this run. */
  mode?: 'sft' | 'dpo';
  now?: Date;
  /** Skip the (free) router refresh. */
  skipRouter?: boolean;
  /** Progress callback. */
  onEvent?: (event: TickEvent) => void;
}

export type TickEvent =
  | { type: 'gates'; allowed: boolean; blockers: string[] }
  | { type: 'router'; result: RouterRefreshResult }
  | { type: 'dataset'; summary: string[] }
  | { type: 'job'; job: TrainJob }
  | { type: 'progress'; message: string }
  | { type: 'done'; job: TrainJob | null; reason: string };

export interface TickResult {
  ran: boolean;
  reason: string;
  gates: GateResult;
  router: RouterRefreshResult | null;
  job: TrainJob | null;
  preflight: MlxPreflight | null;
  /** Commands the user should run, when training could not start. */
  installInstructions: string[];
  datasets: BuiltDatasets;
}

export async function trainTick(cfg: ProtoConfig, dataDir: string, opts: TickOptions = {}): Promise<TickResult> {
  const now = opts.now ?? new Date();
  const onEvent = opts.onEvent ?? ((): void => {});
  const store = new EpisodeStore(dataDir);
  let state = loadTrainState(dataDir);

  // ---- always: cheap router refresh ----
  let router: RouterRefreshResult | null = null;
  if (!opts.skipRouter && cfg.memory.enabled) {
    router = refreshRouter(cfg, dataDir, { write: !opts.dryRun });
    onEvent({ type: 'router', result: router });
  }

  // ---- collect state for the gates ----
  const system = await readSystemState(now);
  const minutesToday = minutesUsedToday(state, now);
  const datasets = buildDatasets(store, cfg, { write: false });
  const newEpisodes = countNewEpisodes(dataDir, state);

  const gates = evaluateGates({
    cfg,
    state: system,
    newEpisodes,
    minutesUsedToday: minutesToday,
    hasTrainingData: datasets.sft.samples.length > 0 || datasets.dpo.samples.length > 0,
  });
  onEvent({ type: 'gates', allowed: gates.allowed, blockers: gates.blockers });

  const pf = await preflight(dataDir);

  const finishSkip = (reason: string): TickResult => {
    if (!opts.dryRun) {
      state = { ...state, lastTickAt: now.toISOString(), consecutiveSkips: state.consecutiveSkips + 1 };
      saveTrainState(dataDir, state);
    }
    onEvent({ type: 'done', job: null, reason });
    return {
      ran: false,
      reason,
      gates,
      router,
      job: null,
      preflight: pf,
      installInstructions: pf.ok ? [] : pf.installInstructions,
      datasets,
    };
  };

  if (!gates.allowed && !opts.force) {
    return finishSkip(`gates blocked training: ${gates.blockers.join('; ')}`);
  }
  if (!pf.ok && !opts.dryRun) {
    return finishSkip(`trainer not available: ${pf.detail}`);
  }

  // ---- build the concrete job ----
  const plan = planSession(cfg, minutesToday, opts.mode ?? cfg.train.lora.mode);
  const datasetPath = plan.mode === 'dpo' ? datasets.dpo.path : datasets.sft.path;
  const sampleCount = plan.mode === 'dpo' ? datasets.dpo.samples.length : datasets.sft.samples.length;

  if (sampleCount === 0) {
    const fallbackMode = plan.mode === 'sft' ? 'dpo' : 'sft';
    const fallbackCount = fallbackMode === 'dpo' ? datasets.dpo.samples.length : datasets.sft.samples.length;
    if (fallbackCount === 0) {
      return finishSkip(
        `no ${plan.mode.toUpperCase()} samples available yet; SFT needs a verified attempt, ` +
          `DPO needs a failed-local/succeeded-cloud pair`,
      );
    }
    onEvent({ type: 'progress', message: `no ${plan.mode.toUpperCase()} samples; using ${fallbackMode.toUpperCase()} (${fallbackCount} rows)` });
    return startJob(fallbackMode as 'sft' | 'dpo', fallbackMode === 'dpo' ? datasets.dpo.path : datasets.sft.path, fallbackCount);
  }

  return startJob(plan.mode, datasetPath, sampleCount);

  // ----------------------------------------------------------------
  async function startJob(mode: 'sft' | 'dpo', dataPath: string, rows: number): Promise<TickResult> {
    const queue = new JobQueue(dataDir);
    const adapterPath = newAdapterPath(dataDir, mode);
    const jobId = `job-${now.getTime().toString(36)}-${mode}`;
    const mlxDataDir = join(dataDir, 'train', 'data', jobId);
    const logPath = join(dataDir, 'train', 'logs', `${jobId}.log`);

    const job = queue.add({
      id: jobId,
      kind: mode,
      status: 'planned',
      plan: { ...plan, mode },
      note: opts.force ? 'forced by user' : 'scheduled tick',
      datasetPath: dataPath,
      mlxDataDir,
      adapterPath,
      baseModel: cfg.train.baseModel,
    });
    onEvent({ type: 'job', job });
    onEvent({ type: 'dataset', summary: describeDatasets(datasets) });

    let prepared;
    try {
      prepared = prepareMlxDataDir({ targetDir: mlxDataDir, trainPath: dataPath, mode });
    } catch (err) {
      const message = err instanceof Error ? err.message : String(err);
      queue.update(job.id, { status: 'failed', finishedAt: new Date().toISOString(), error: message });
      state = { ...state, lastError: message, lastTickAt: now.toISOString() };
      saveTrainState(dataDir, state);
      onEvent({ type: 'done', job: null, reason: message });
      return { ran: false, reason: message, gates, router, job: null, preflight: pf, installInstructions: [], datasets };
    }

    const cmd = buildLoraCommand({
      cfg,
      python: pf.pythonPath ?? 'python3',
      baseModel: cfg.train.baseModel,
      dataDir: prepared.dir,
      adapterPath,
      mode,
      iters: plan.iters,
    });
    queue.update(job.id, { command: cmd.display, logPath, datasetPath: dataPath });

    if (opts.dryRun) {
      queue.update(job.id, { status: 'planned', note: 'dry run: command assembled but not executed' });
      onEvent({ type: 'done', job: queue.all().find((j) => j.id === job.id) ?? null, reason: 'dry run' });
      return {
        ran: false,
        reason: 'dry run: the training command was assembled but not executed',
        gates,
        router,
        job: queue.all().find((j) => j.id === job.id) ?? null,
        preflight: pf,
        installInstructions: [],
        datasets,
      };
    }

    queue.update(job.id, { status: 'running', startedAt: new Date().toISOString() });
    onEvent({ type: 'progress', message: `training ${mode} for up to ${plan.maxRuntimeMin} min: ${cmd.display}` });

    const result = await runTraining({
      cfg,
      python: pf.pythonPath ?? 'python3',
      baseModel: cfg.train.baseModel,
      dataDir: prepared.dir,
      adapterPath,
      mode,
      iters: plan.iters,
      maxRuntimeMs: plan.maxRuntimeMin * 60_000,
      maxLoadAverage: cfg.train.maxLoadAverage,
      logPath,
    });

    const finishedAt = new Date().toISOString();
    const minutes = result.durationMs / 60_000;
    state = addMinutes(state, minutes, now);
    state = {
      ...state,
      lastTickAt: now.toISOString(),
      lastRunAt: finishedAt,
      consecutiveSkips: 0,
      totalRuns: state.totalRuns + (result.ok ? 1 : 0),
      highWaterEpisodeId: newestEpisodeId(dataDir) ?? state.highWaterEpisodeId,
      ...(result.ok || result.abortedByWatchdog ? {} : { lastError: `exit ${result.exitCode}` }),
    };
    saveTrainState(dataDir, state);

    const losses = parseLosses(result.outputTail);
    if (result.ok) {
      writeAdapterMetrics(adapterPath, {
        iters: plan.iters,
        mode,
        datasetRows: rows,
        jobId: job.id,
        ...(losses.trainLoss !== undefined ? { trainLoss: losses.trainLoss } : {}),
        ...(losses.validLoss !== undefined ? { validLoss: losses.validLoss } : {}),
      });
    }

    const updated = queue.update(job.id, {
      status: result.ok ? 'done' : result.abortedByWatchdog ? 'interrupted' : 'failed',
      finishedAt,
      exitCode: result.exitCode,
      durationMs: result.durationMs,
      logPath: result.logPath,
      ...(losses.trainLoss !== undefined ? { trainLoss: losses.trainLoss } : {}),
      ...(losses.validLoss !== undefined ? { validLoss: losses.validLoss } : {}),
      ...(result.ok
        ? {}
        : {
            error: result.abortedByWatchdog
              ? 'aborted by the load watchdog because the machine became busy'
              : result.timedOut
                ? 'hit the wall-clock cap'
                : `trainer exited with code ${result.exitCode}`,
          }),
    });

    // ---- promote ----
    if (result.ok && cfg.train.autoPromote) {
      const info = listAdapters(dataDir).find((a) => a.name === adapterPath.split('/').pop());
      if (info) {
        setActiveAdapter(dataDir, info, cfg.train.baseModel);
        onEvent({ type: 'progress', message: `promoted adapter ${info.name} to active` });
      }
    }

    // ---- prune old adapters ----
    pruneAdapters(dataDir, cfg.train.keepAdapters);

    onEvent({ type: 'done', job: updated, reason: result.ok ? 'training completed' : (updated?.error ?? 'training failed') });
    return { ran: result.ok, reason: result.ok ? 'training completed' : (updated?.error ?? 'training failed'), gates, router, job: updated, preflight: pf, installInstructions: [], datasets };
  }
}

function newestEpisodeId(dataDir: string): string | null {
  const episodes = new EpisodeStore(dataDir).readAll();
  return episodes.length ? (episodes[episodes.length - 1] as { id: string }).id : null;
}

/** mlx-lm prints lines like `Iter 60: Train loss 1.234, Learning Rate ...`. */
export function parseLosses(output: string): { trainLoss?: number; validLoss?: number } {
  const out: { trainLoss?: number; validLoss?: number } = {};
  const trainMatches = [...output.matchAll(/Train loss\s+([\d.]+)/g)];
  if (trainMatches.length) {
    const last = trainMatches[trainMatches.length - 1]?.[1];
    if (last) out.trainLoss = Number(last);
  }
  const validMatches = [...output.matchAll(/Val loss\s+([\d.]+)/g)];
  if (validMatches.length) {
    const last = validMatches[validMatches.length - 1]?.[1];
    if (last) out.validLoss = Number(last);
  }
  return out;
}

function pruneAdapters(dataDir: string, keep: number): void {
  if (keep <= 0) return;
  const adapters = listAdapters(dataDir);
  const sorted = adapters.sort((a, b) => (b.createdAt ?? '').localeCompare(a.createdAt ?? ''));
  for (const stale of sorted.slice(keep)) {
    try {
      rmSync(stale.path, { recursive: true, force: true });
    } catch {
      /* best effort */
    }
  }
}

/** Copy-pasteable instructions for making an adapter active. */
export function promotionInstructions(cfg: ProtoConfig, dataDir: string, adapterName: string): string[] {
  const adapterPath = join(adaptersRoot(dataDir), adapterName);
  const cmds = serveCommands({ dataDir, baseModel: cfg.train.baseModel, adapterPath });
  return [
    `# serve the base model with the LoRA adapter applied:`,
    cmds.mlx,
    `# then point the harness at it:`,
    `proto config set local.runtime mlx`,
    `proto config set local.baseUrl http://127.0.0.1:8080`,
    `proto config set local.model ${cfg.train.baseModel}`,
    `# (note) ${cmds.note}`,
  ];
}

export function ensureTrainDirs(dataDir: string): void {
  ensureDir(join(dataDir, 'train'));
  ensureDir(join(dataDir, 'train', 'logs'));
  ensureDir(join(dataDir, 'train', 'data'));
  ensureDir(adaptersRoot(dataDir));
}

export { MIN_TRAINING_SAMPLES, LogisticScorer };
