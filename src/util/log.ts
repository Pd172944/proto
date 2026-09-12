/**
 * Structured logging. Deliberately tiny and dependency-free.
 *
 * Design notes:
 *  - Human output goes to stderr so that `--json` command output on stdout
 *    stays machine-parseable; the two streams never interleave.
 *  - `PROTO_LOG=debug|info|warn|error|silent` controls verbosity.
 *  - `PROTO_LOG=json` switches every line to a structured JSON object, which is
 *    what the background training worker writes into its job log.
 */

export type LogLevel = 'debug' | 'info' | 'warn' | 'error' | 'silent';

const LEVELS: Record<LogLevel, number> = {
  debug: 10,
  info: 20,
  warn: 30,
  error: 40,
  silent: 99,
};

let jsonMode = false;

function envLevel(): LogLevel | 'json' {
  const raw = (process.env['PROTO_LOG'] ?? '').toLowerCase();
  if (raw === 'json') return 'json';
  if (raw in LEVELS) return raw as LogLevel;
  return 'info';
}

/** Route all log output through a sink; used by tests and the job runner. */
export type LogSink = (line: string, level: LogLevel) => void;
let sink: LogSink | null = null;

export function setLogSink(next: LogSink | null): void {
  sink = next;
}

export function setJsonLogging(on: boolean): void {
  jsonMode = on;
}

function threshold(): number {
  const l = envLevel();
  return l === 'json' ? LEVELS.info : LEVELS[l];
}

const COLOR: Record<LogLevel, string> = {
  debug: '\u001b[2m',
  info: '\u001b[36m',
  warn: '\u001b[33m',
  error: '\u001b[31m',
  silent: '',
};

function colorEnabled(): boolean {
  if (process.env['NO_COLOR']) return false;
  if (process.env['PROTO_COLOR'] === '0') return false;
  return process.stderr.isTTY === true;
}

function emit(level: LogLevel, msg: string, fields?: Record<string, unknown>): void {
  if (LEVELS[level] < threshold()) return;
  const useJson = jsonMode || envLevel() === 'json';
  let line: string;
  if (useJson) {
    line = JSON.stringify({
      ts: new Date().toISOString(),
      level,
      msg,
      ...(fields ?? {}),
    });
  } else {
    const tag = colorEnabled() ? `${COLOR[level]}${level.padEnd(5)}\u001b[0m` : level.padEnd(5);
    const extra =
      fields && Object.keys(fields).length > 0
        ? ' ' +
          Object.entries(fields)
            .map(([k, v]) => `${k}=${formatValue(v)}`)
            .join(' ')
        : '';
    line = `${tag} ${msg}${extra}`;
  }
  if (sink) sink(line, level);
  else process.stderr.write(line + '\n');
}

function formatValue(v: unknown): string {
  if (typeof v === 'string') return /\s/.test(v) ? JSON.stringify(v) : v;
  if (v === null || v === undefined) return String(v);
  if (typeof v === 'object') return JSON.stringify(v);
  return String(v);
}

export const log = {
  debug: (msg: string, fields?: Record<string, unknown>) => emit('debug', msg, fields),
  info: (msg: string, fields?: Record<string, unknown>) => emit('info', msg, fields),
  warn: (msg: string, fields?: Record<string, unknown>) => emit('warn', msg, fields),
  error: (msg: string, fields?: Record<string, unknown>) => emit('error', msg, fields),
};

/* ------------------------------------------------------------------ */
/* Terminal presentation helpers                                       */
/* ------------------------------------------------------------------ */

const supportsColor = (): boolean => colorEnabled();

export const style = {
  bold: (s: string) => (supportsColor() ? `\u001b[1m${s}\u001b[0m` : s),
  dim: (s: string) => (supportsColor() ? `\u001b[2m${s}\u001b[0m` : s),
  green: (s: string) => (supportsColor() ? `\u001b[32m${s}\u001b[0m` : s),
  yellow: (s: string) => (supportsColor() ? `\u001b[33m${s}\u001b[0m` : s),
  red: (s: string) => (supportsColor() ? `\u001b[31m${s}\u001b[0m` : s),
  cyan: (s: string) => (supportsColor() ? `\u001b[36m${s}\u001b[0m` : s),
  magenta: (s: string) => (supportsColor() ? `\u001b[35m${s}\u001b[0m` : s),
};

/** Print command output (stdout). Distinct from `log`, which writes to stderr. */
export function out(text = ''): void {
  process.stdout.write(text + '\n');
}
