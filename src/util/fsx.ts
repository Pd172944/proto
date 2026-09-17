/**
 * Filesystem helpers.
 *
 * All harness state is written through these helpers so that:
 *  - writes are atomic (temp file + rename), so a crash mid-write cannot
 *    corrupt a file another process is reading — these are read back whole and
 *    the user may never notice silent corruption.
 *  - JSONL appends are single `appendFile` calls (atomic enough for
 *    single-writer use, which is what we guarantee via a lockfile).
 *  - directory walks are bounded so `proto` can never hang on a huge tree.
 */

import {
  appendFileSync,
  existsSync,
  mkdirSync,
  readFileSync,
  readdirSync,
  renameSync,
  rmSync,
  statSync,
  writeFileSync,
} from 'node:fs';
import { homedir } from 'node:os';
import { dirname, isAbsolute, join, resolve } from 'node:path';

export function ensureDir(dir: string): string {
  mkdirSync(dir, { recursive: true });
  return dir;
}

export function fileExists(path: string): boolean {
  try {
    return statSync(path).isFile();
  } catch {
    return false;
  }
}

export function dirExists(path: string): boolean {
  try {
    return statSync(path).isDirectory();
  } catch {
    return false;
  }
}

export function readText(path: string): string {
  return readFileSync(path, 'utf8');
}

export function readTextOrNull(path: string): string | null {
  try {
    return readFileSync(path, 'utf8');
  } catch {
    return null;
  }
}

export function readJson<T>(path: string): T {
  return JSON.parse(readFileSync(path, 'utf8')) as T;
}

export function readJsonOrNull<T>(path: string): T | null {
  try {
    return JSON.parse(readFileSync(path, 'utf8')) as T;
  } catch {
    return null;
  }
}

/** Atomic write: never leaves a partially written file behind. */
export function writeTextAtomic(path: string, content: string): void {
  ensureDir(dirname(path));
  const tmp = `${path}.tmp-${process.pid}-${Date.now()}`;
  writeFileSync(tmp, content, 'utf8');
  renameSync(tmp, path);
}

export function writeJsonAtomic(path: string, value: unknown, pretty = true): void {
  writeTextAtomic(path, JSON.stringify(value, null, pretty ? 2 : 0) + '\n');
}

export function appendJsonl(path: string, record: unknown): void {
  ensureDir(dirname(path));
  appendFileSync(path, JSON.stringify(record) + '\n', 'utf8');
}

/** Read a JSONL file, skipping unparseable lines instead of throwing. */
export function readJsonl<T>(path: string, onBadLine?: (line: string, err: unknown) => void): T[] {
  const raw = readTextOrNull(path);
  if (raw === null) return [];
  const out: T[] = [];
  for (const line of raw.split('\n')) {
    const trimmed = line.trim();
    if (!trimmed) continue;
    try {
      out.push(JSON.parse(trimmed) as T);
    } catch (err) {
      onBadLine?.(trimmed, err);
    }
  }
  return out;
}

export function listFiles(dir: string, opts: { suffix?: string; max?: number } = {}): string[] {
  const max = opts.max ?? 5000;
  const out: string[] = [];
  if (!dirExists(dir)) return out;
  const walk = (d: string, depth: number): void => {
    if (out.length >= max || depth > 12) return;
    let entries: string[];
    try {
      entries = readdirSync(d);
    } catch {
      return;
    }
    for (const name of entries.sort()) {
      if (out.length >= max) return;
      const full = join(d, name);
      let st;
      try {
        st = statSync(full);
      } catch {
        continue;
      }
      if (st.isDirectory()) {
        if (name === 'node_modules' || name === '.git' || name === 'var') continue;
        walk(full, depth + 1);
      } else if (st.isFile()) {
        if (opts.suffix && !name.endsWith(opts.suffix)) continue;
        out.push(full);
      }
    }
  };
  walk(dir, 0);
  return out;
}

export function removeFile(path: string): void {
  rmSync(path, { force: true });
}

export function byteSize(path: string): number {
  try {
    return statSync(path).size;
  } catch {
    return 0;
  }
}

/** Expand a leading `~` and resolve relative to `base`. */
export function resolvePath(p: string, base = process.cwd()): string {
  let path = p;
  if (path === '~') path = homedir();
  else if (path.startsWith('~/')) path = join(homedir(), path.slice(2));
  return isAbsolute(path) ? resolve(path) : resolve(base, path);
}

/** Compact human byte formatting for CLI tables. */
export function formatBytes(bytes: number): string {
  if (bytes < 1024) return `${bytes} B`;
  const units = ['KB', 'MB', 'GB', 'TB'];
  let v = bytes / 1024;
  let i = 0;
  while (v >= 1024 && i < units.length - 1) {
    v /= 1024;
    i++;
  }
  return `${v.toFixed(v < 10 ? 1 : 0)} ${units[i]}`;
}

export function formatDuration(ms: number): string {
  if (ms < 1000) return `${Math.round(ms)}ms`;
  if (ms < 60_000) return `${(ms / 1000).toFixed(1)}s`;
  const min = Math.floor(ms / 60_000);
  const sec = Math.round((ms % 60_000) / 1000);
  if (min < 60) return `${min}m${sec.toString().padStart(2, '0')}s`;
  const hr = Math.floor(min / 60);
  return `${hr}h${(min % 60).toString().padStart(2, '0')}m`;
}

/** Bounded directory walk used by `doctor` and the eval corpus loader. */
export function pathExists(path: string): boolean {
  return existsSync(path);
}
