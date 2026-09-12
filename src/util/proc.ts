/**
 * Process helpers: spawning child processes with hard resource discipline.
 *
 * The whole premise of this project is "as little load on the user's Mac as
 * possible", so nothing here spawns a process without:
 *  - a wall-clock timeout,
 *  - output caps (a runaway training log must not fill the disk),
 *  - optional background QoS on macOS (`taskpolicy -b`) and a low `nice` value.
 */

import { spawn } from 'node:child_process';

export interface ExecResult {
  code: number | null;
  stdout: string;
  stderr: string;
  timedOut: boolean;
  durationMs: number;
  /** True when the binary could not be spawned at all (ENOENT). */
  spawnFailed: boolean;
}

export interface ExecOptions {
  cwd?: string;
  env?: Record<string, string>;
  timeoutMs?: number;
  input?: string;
  maxOutputBytes?: number;
  /** Wrap with `nice -n 19` and macOS `taskpolicy -b` for background work. */
  lowPriority?: boolean;
  signal?: AbortSignal;
}

export async function exec(command: string, args: string[], opts: ExecOptions = {}): Promise<ExecResult> {
  const timeoutMs = opts.timeoutMs ?? 120_000;
  const maxBytes = opts.maxOutputBytes ?? 512 * 1024;
  const started = Date.now();

  let bin = command;
  let argv = args;
  if (opts.lowPriority) {
    // macOS ships both; `taskpolicy -b` demotes to background QoS which is what
    // actually keeps a training run from stealing cycles from the foreground.
    if (process.platform === 'darwin') {
      bin = 'taskpolicy';
      argv = ['-b', 'nice', '-n', '19', command, ...args];
    } else {
      bin = 'nice';
      argv = ['-n', '19', command, ...args];
    }
  }

  return await new Promise<ExecResult>((resolve) => {
    let child;
    try {
      child = spawn(bin, argv, {
        cwd: opts.cwd,
        env: { ...process.env, ...(opts.env ?? {}) },
        stdio: ['pipe', 'pipe', 'pipe'],
      });
    } catch {
      resolve({
        code: null,
        stdout: '',
        stderr: `failed to spawn ${bin}`,
        timedOut: false,
        durationMs: Date.now() - started,
        spawnFailed: true,
      });
      return;
    }

    let stdout = '';
    let stderr = '';
    let timedOut = false;
    let settled = false;

    const finish = (code: number | null, spawnFailed = false): void => {
      if (settled) return;
      settled = true;
      clearTimeout(timer);
      opts.signal?.removeEventListener('abort', onAbort);
      resolve({ code, stdout, stderr, timedOut, durationMs: Date.now() - started, spawnFailed });
    };

    const timer = setTimeout(() => {
      timedOut = true;
      child.kill('SIGTERM');
      // Escalate if it ignores SIGTERM. Local inference servers can wedge.
      setTimeout(() => {
        try {
          child.kill('SIGKILL');
        } catch {
          /* already gone */
        }
      }, 3000);
    }, timeoutMs);

    const onAbort = (): void => {
      timedOut = true;
      child.kill('SIGTERM');
    };
    opts.signal?.addEventListener('abort', onAbort, { once: true });

    child.stdout?.on('data', (d: Buffer) => {
      if (stdout.length < maxBytes) stdout += d.toString('utf8');
    });
    child.stderr?.on('data', (d: Buffer) => {
      if (stderr.length < maxBytes) stderr += d.toString('utf8');
    });
    child.on('error', (err) => {
      stderr += String(err);
      finish(null, true);
    });
    child.on('close', (code) => finish(code));

    if (opts.input !== undefined) {
      child.stdin?.write(opts.input);
      child.stdin?.end();
    } else {
      child.stdin?.end();
    }
  });
}

/** Is a binary on PATH? Uses `command -v` so it respects the user's shell PATH. */
export async function hasBinary(name: string): Promise<boolean> {
  const res = await exec('/bin/sh', ['-c', `command -v ${JSON.stringify(name)}`], { timeoutMs: 5000 });
  return res.code === 0 && res.stdout.trim().length > 0;
}

/** Version string of a binary, best-effort. */
export async function binaryVersion(name: string, args: string[] = ['--version']): Promise<string | null> {
  const res = await exec(name, args, { timeoutMs: 8000 });
  if (res.spawnFailed || res.code !== 0) return null;
  const out = `${res.stdout}\n${res.stderr}`.trim().split('\n')[0];
  return out ?? null;
}

/** Fetch a JSON HTTP endpoint from the shell — used for local runtime probes. */
export async function curlJson(url: string, timeoutMs = 4000): Promise<unknown | null> {
  const res = await exec('curl', ['-sS', '-m', String(Math.ceil(timeoutMs / 1000)), url], {
    timeoutMs: timeoutMs + 2000,
  });
  if (res.code !== 0) return null;
  try {
    return JSON.parse(res.stdout) as unknown;
  } catch {
    return null;
  }
}
