/**
 * MLX LoRA trainer driver.
 *
 * Why MLX: on Apple Silicon it is the only mainstream path that trains LoRA
 * adapters natively on the GPU with unified memory, needs no CUDA, and can serve
 * the result back through an OpenAI-compatible endpoint (`mlx_lm.server`). That
 * means the *same* runtime the user trains with can also serve inference, which
 * removes an entire conversion step (MLX -> GGUF -> Ollama) from the critical
 * path.
 *
 * Hard rules enforced here:
 *  - This module NEVER installs anything and NEVER downloads a model. If `mlx_lm`
 *    is missing it returns the exact command the user should run, and stops.
 *    Silently installing multi-hundred-megabyte packages onto someone's laptop
 *    is exactly the behaviour this project exists to avoid.
 *  - Every run is wrapped in `taskpolicy -b`/`nice 19` on macOS, is given a
 *    wall-clock cap, and is aborted by a load watchdog if the machine gets busy.
 */

import { join } from 'node:path';
import { loadavg } from 'node:os';

import type { ProtoConfig } from '../config/schema.ts';
import { exec } from '../util/proc.ts';
import { ensureDir, fileExists, readTextOrNull, writeTextAtomic } from '../util/fsx.ts';
import { shortId } from '../util/ids.ts';

export interface MlxPreflight {
  ok: boolean;
  pythonPath: string | null;
  version: string | null;
  /** True when the `mlx_lm` module imports. */
  modulePresent: boolean;
  /** True when a `mlx_lm.lora` entry point exists. */
  loraEntryPoint: boolean;
  detail: string;
  /** Copy-pasteable commands, shown but never run automatically. */
  installInstructions: string[];
}

/** The interpreter to use: a project venv if present, else python3. */
export function pythonCandidate(dataDir: string): string {
  const venv = join(dataDir, 'venv', 'bin', 'python3');
  return fileExists(venv) ? venv : 'python3';
}

export async function preflight(dataDir: string): Promise<MlxPreflight> {
  const python = pythonCandidate(dataDir);
  const instructions = [
    `python3 -m venv ${join(dataDir, 'venv')}`,
    `${join(dataDir, 'venv', 'bin', 'pip')} install --upgrade pip`,
    `${join(dataDir, 'venv', 'bin', 'pip')} install mlx-lm`,
    '# then download a base model once (this is the ONLY large download):',
    `${join(dataDir, 'venv', 'bin', 'python')} -m mlx_lm.generate --model mlx-community/Qwen2.5-Coder-1.5B-Instruct-4bit --prompt "hello" --max-tokens 16`,
  ];

  const check = await exec(
    python,
    ['-c', 'import importlib.util,sys;m=importlib.util.find_spec("mlx_lm");print("MLXLM" if m else "MISSING")'],
    { timeoutMs: 15_000 },
  );
  if (check.spawnFailed) {
    return {
      ok: false,
      pythonPath: null,
      version: null,
      modulePresent: false,
      loraEntryPoint: false,
      detail: `no python interpreter found at "${python}"`,
      installInstructions: instructions,
    };
  }
  const modulePresent = check.stdout.includes('MLXLM');

  const versionRes = modulePresent
    ? await exec(python, ['-c', 'import mlx_lm, importlib.metadata as m; print(m.version("mlx-lm"))'], {
        timeoutMs: 20_000,
      })
    : null;
  const version = versionRes?.code === 0 ? versionRes.stdout.trim() : null;

  const loraRes = modulePresent
    ? await exec(python, ['-c', 'import importlib.util as u; print("OK" if u.find_spec("mlx_lm.lora") else "MISSING")'], {
        timeoutMs: 20_000,
      })
    : null;
  const loraEntryPoint = loraRes?.stdout.includes('OK') ?? false;

  const ok = modulePresent && loraEntryPoint;
  return {
    ok,
    pythonPath: python,
    version,
    modulePresent,
    loraEntryPoint,
    detail: ok
      ? `mlx-lm ${version ?? '(unknown version)'} is available via ${python}`
      : modulePresent
        ? 'mlx-lm is importable but `mlx_lm.lora` is missing; upgrade mlx-lm'
        : `mlx-lm is not installed for ${python}`,
    installInstructions: instructions,
  };
}

export interface LoraCommand {
  bin: string;
  args: string[];
  /** Environment additions (MLX is sensitive to some of these). */
  env: Record<string, string>;
  /** The exact shell line, for display and for the job log. */
  display: string;
}

export interface LoraBuildInput {
  cfg: ProtoConfig;
  python: string;
  /** Base model id (Hugging Face repo or local path). */
  baseModel: string;
  /** Directory containing train.jsonl / valid.jsonl (or train.jsonl for DPO). */
  dataDir: string;
  /** Where to write the adapter. */
  adapterPath: string;
  mode: 'sft' | 'dpo';
  iters: number;
}

/**
 * Build the `mlx_lm.lora` invocation.
 *
 * Notes on the hyperparameters: rank 8 / 8 layers / 60 iterations is chosen to
 * be a *nudge*, not a retrain. The intent is to teach the local model this
 * user's preferences and recurring task shapes, which needs far less capacity
 * than teaching it to code. Small also means: runs in minutes, cannot catastrophically
 * forget, and can be thrown away if it makes things worse.
 */
export function buildLoraCommand(input: LoraBuildInput): LoraCommand {
  const l = input.cfg.train.lora;
  const args = [
    '-m',
    'mlx_lm.lora',
    '--model',
    input.baseModel,
    '--train',
    '--data',
    input.dataDir,
    '--adapter-path',
    input.adapterPath,
    '--iters',
    String(input.iters),
    '--batch-size',
    String(l.batchSize),
    '--num-layers',
    String(l.layers),
    '--learning-rate',
    String(l.learningRate),
    '--max-seq-length',
    String(l.maxSeqLen),
    '--lora-rank',
    String(l.rank),
    '--lora-scale',
    String(l.scale),
    '--lora-dropout',
    String(l.dropout),
    '--seed',
    '0',
    '--steps-per-report',
    '10',
    '--steps-per-eval',
    '20',
    '--save-every',
    '20',
  ];
  if (input.mode === 'dpo') {
    // mlx-lm expects the preference triplets in train.jsonl with this flag set.
    args.push('--train-mode', 'dpo');
  }
  return {
    bin: input.python,
    args,
    env: { PYTHONUNBUFFERED: '1' },
    display: `${input.python} ${args.map(shellQuote).join(' ')}`,
  };
}

export interface TrainingRunResult {
  ok: boolean;
  exitCode: number | null;
  timedOut: boolean;
  abortedByWatchdog: boolean;
  durationMs: number;
  logPath: string;
  adapterPath: string;
  /** Tail of the training output, for the CLI. */
  outputTail: string;
}

export interface RunTrainingInput extends LoraBuildInput {
  /** Wall-clock cap. */
  maxRuntimeMs: number;
  /** Abort if the 1-minute load average exceeds this. */
  maxLoadAverage: number;
  /** Poll interval for the load watchdog. */
  watchdogIntervalMs?: number;
  /** Where to write the log. */
  logPath: string;
}

/**
 * Run a training session under a load watchdog.
 *
 * The watchdog is the difference between "background training" and "background
 * training that respects the user". It samples the load average periodically and
 * aborts the run the moment the machine stops being idle — the user opening a
 * video call should immediately preempt this, and the checkpoint every 20 steps
 * means almost no work is lost.
 */
export async function runTraining(input: RunTrainingInput): Promise<TrainingRunResult> {
  ensureDir(input.adapterPath);
  ensureDir(join(input.logPath, '..'));
  const dbg = buildLoraCommand(input);
  writeTextAtomic(
    `${input.logPath}.cmd`,
    `# generated by proto; nothing here installs or downloads anything\n${dbg.display}\n`,
  );

  const controller = new AbortController();
  let abortedByWatchdog = false;
  const intervalMs = input.watchdogIntervalMs ?? 30_000;
  const watchdog = setInterval(() => {
    const load = loadavg()[0] ?? 0;
    if (load > input.maxLoadAverage) {
      abortedByWatchdog = true;
      controller.abort();
    }
  }, intervalMs);

  try {
    const res = await exec(dbg.bin, dbg.args, {
      env: dbg.env,
      timeoutMs: input.maxRuntimeMs,
      lowPriority: true,
      signal: controller.signal,
      maxOutputBytes: 2 * 1024 * 1024,
    });
    const output = `${res.stdout}\n${res.stderr}`.trim();
    writeTextAtomic(input.logPath, output + '\n');
    return {
      ok: res.code === 0 && !res.timedOut && !abortedByWatchdog,
      exitCode: res.code,
      timedOut: res.timedOut && !abortedByWatchdog,
      abortedByWatchdog,
      durationMs: res.durationMs,
      logPath: input.logPath,
      adapterPath: input.adapterPath,
      outputTail: output.split('\n').slice(-25).join('\n'),
    };
  } finally {
    clearInterval(watchdog);
  }
}

/** Where adapter directories live. */
export function adaptersDir(dataDir: string): string {
  return join(dataDir, 'models', 'adapters');
}

export function newAdapterPath(dataDir: string, mode: 'sft' | 'dpo'): string {
  return join(adaptersDir(dataDir), `${mode}-${shortId()}`);
}

/** Prepare the `--data` directory mlx-lm expects (train.jsonl / valid.jsonl). */
export function prepareMlxDataDir(input: {
  targetDir: string;
  trainPath: string;
  validPath?: string;
  mode: 'sft' | 'dpo';
}): { dir: string; trainRows: number; validRows: number } {
  ensureDir(input.targetDir);
  const train = readTextOrNull(input.trainPath) ?? '';
  const valid = input.validPath ? (readTextOrNull(input.validPath) ?? '') : '';
  const trainRows = countLines(train);
  const validRows = countLines(valid);

  if (trainRows === 0) {
    throw new Error(`no training rows found at ${input.trainPath}`);
  }

  // Hold out the tail as validation if none was supplied, so mlx-lm can report a
  // loss curve. Without a validation split, "did this help?" is unanswerable.
  let trainBody = train;
  let validBody = valid;
  if (validRows === 0 && trainRows >= 20) {
    const lines = train.trimEnd().split('\n');
    const holdout = Math.max(1, Math.floor(lines.length * 0.1));
    trainBody = lines.slice(0, lines.length - holdout).join('\n') + '\n';
    validBody = lines.slice(lines.length - holdout).join('\n') + '\n';
  } else if (validRows === 0) {
    validBody = trainBody;
  }

  writeTextAtomic(join(input.targetDir, 'train.jsonl'), trainBody);
  writeTextAtomic(join(input.targetDir, 'valid.jsonl'), validBody);
  return {
    dir: input.targetDir,
    trainRows: countLines(trainBody),
    validRows: countLines(validBody),
  };
}

function countLines(s: string): number {
  const t = s.trim();
  return t ? t.split('\n').length : 0;
}

function shellQuote(s: string): string {
  return /^[A-Za-z0-9_@%+=:,./-]+$/.test(s) ? s : `'${s.replace(/'/g, `'\\''`)}'`;
}

/** Commands to serve a base model + optional LoRA adapter. */
export function serveCommands(input: {
  dataDir: string;
  baseModel: string;
  adapterPath?: string;
}): { mlx: string; ollama: string; note: string } {
  const python = pythonCandidate(input.dataDir);
  const mlxArgs = [
    '-m',
    'mlx_lm.server',
    '--model',
    input.baseModel,
    '--host',
    '127.0.0.1',
    '--port',
    '8080',
  ];
  if (input.adapterPath) mlxArgs.push('--adapter-path', input.adapterPath);
  return {
    mlx: `${python} ${mlxArgs.join(' ')}`,
    ollama: `ollama create ${ollamaModelName(input.baseModel)}-proto -f Modelfile`,
    note:
      'mlx_lm.server is the recommended path: it loads the base model and the LoRA adapter directly, ' +
      'so no GGUF conversion step is needed. Point local.runtime at "mlx" to use it.',
  };
}

export function ollamaModelName(baseModel: string): string {
  const tail = baseModel.split('/').pop() ?? baseModel;
  return tail.toLowerCase().replace(/[^a-z0-9._-]+/g, '-').replace(/^-+|-+$/g, '');
}
