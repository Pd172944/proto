/**
 * Content search that does not read the whole repository on every call.
 *
 * The original implementation walked the tree and read every file into a string to test
 * a regex against it. That is O(repository bytes) per search — fine for a toy, and about
 * 200 MB of I/O on a real monorepo, repeated for every one of the agent's searches.
 *
 * Three engines, chosen by what is available and what the pattern actually is:
 *
 *  1. **`git grep`, for a literal pattern.** This is the common case by a wide margin —
 *     an agent searching for an identifier. `-F` makes it a literal match, so the results
 *     are exact and complete, and git does it in C over its own index while applying
 *     every ignore rule for free. `--untracked` matters: the agent creates files, and a
 *     search that cannot see its own new file is a trap.
 *  2. **`git grep -l` to shortlist, then a JavaScript regex to decide.** Used for a real
 *     regular expression. The shortlist is then verified with the same `RegExp` the model
 *     wrote, so the dialect the model sees is always JavaScript — POSIX ERE is close
 *     enough to find candidates but not close enough to be the final word. If `git grep`
 *     rejects the pattern outright, this falls through rather than returning nothing.
 *  3. **A bounded scan of the index's own file list.** The fallback when git is
 *     unavailable. Because the file list comes from discovery it already excludes
 *     ignored and generated files, which is what keeps this bounded in practice.
 *
 * The one thing that is never done is reporting "no matches" because an engine failed.
 * An engine that errors falls through to the next one; only an exhaustive scan is
 * allowed to conclude that something is absent.
 */

import { execFileSync } from 'node:child_process';
import { readFileSync } from 'node:fs';
import { join } from 'node:path';

export interface SearchHit {
  /** Workspace-relative, POSIX separators. */
  path: string;
  line: number;
  text: string;
}

export interface SearchOutcome {
  hits: SearchHit[];
  engine: 'git-grep' | 'git-grep+regex' | 'scan';
  filesScanned: number;
  /** True when a cap stopped the search before it finished. */
  truncated: boolean;
}

export interface SearchOptions {
  pattern: string;
  caseSensitive?: boolean;
  /** Substring the path must contain, e.g. `.ts`. */
  glob?: string;
  maxResults?: number;
  /** Ceiling on files examined in the scan fallback. */
  maxFiles?: number;
  /** Hard cap on the line text returned per hit. */
  maxLineChars?: number;
}

const DEFAULT_MAX_RESULTS = 80;
const DEFAULT_MAX_FILES = 5000;
const DEFAULT_MAX_LINE = 240;

/** True when the pattern contains nothing a regex engine would treat specially. */
export function isLiteralPattern(pattern: string): boolean {
  return !/[.*+?^${}()|[\]\\]/.test(pattern);
}

function run(args: string[], cwd: string, timeout = 20_000): string | null {
  try {
    return execFileSync('git', args, {
      cwd,
      encoding: 'utf8',
      stdio: ['ignore', 'pipe', 'ignore'],
      timeout,
      maxBuffer: 32 * 1024 * 1024,
    });
  } catch (err) {
    // `git grep` exits 1 for "no matches", which is a successful empty result, not a
    // failure. Anything else (bad pattern, not a repo, timeout) is a real error.
    const e = err as { status?: number; stdout?: string };
    if (e.status === 1) return '';
    return null;
  }
}

function parseGrepOutput(out: string, opts: SearchOptions): SearchHit[] {
  const max = opts.maxResults ?? DEFAULT_MAX_RESULTS;
  const maxLine = opts.maxLineChars ?? DEFAULT_MAX_LINE;
  const glob = opts.glob;
  const hits: SearchHit[] = [];
  for (const raw of out.split('\n')) {
    if (raw === '') continue;
    const first = raw.indexOf(':');
    if (first === -1) continue;
    const second = raw.indexOf(':', first + 1);
    if (second === -1) continue;
    const path = raw.slice(0, first);
    const line = Number(raw.slice(first + 1, second));
    if (!Number.isFinite(line)) continue;
    if (glob !== undefined && glob !== '' && !path.includes(glob)) continue;
    hits.push({ path, line, text: raw.slice(second + 1).trim().slice(0, maxLine) });
    if (hits.length >= max) break;
  }
  return hits;
}

/** Engine 1 and 2: hand the search to git. Returns null when git cannot answer. */
export function searchWithGit(root: string, opts: SearchOptions): SearchOutcome | null {
  const max = opts.maxResults ?? DEFAULT_MAX_RESULTS;
  const literal = isLiteralPattern(opts.pattern);

  const base = ['grep', '-n', '-I', '--untracked', '--no-color'];
  if (!opts.caseSensitive) base.push('-i');

  if (literal) {
    // `-m` stops per file, so one pathological file cannot consume the whole budget.
    const args = [...base, '-F', '-m', String(Math.max(1, Math.min(max, 200))), '-e', opts.pattern];
    const out = run(args, root);
    if (out === null) return null;
    const hits = parseGrepOutput(out, opts);
    return { hits, engine: 'git-grep', filesScanned: 0, truncated: hits.length >= max };
  }

  // Regex: shortlist with git, decide with JavaScript.
  const listArgs = ['grep', '-l', '-I', '--untracked', '--no-color'];
  if (!opts.caseSensitive) listArgs.push('-i');
  listArgs.push('-E', '-e', opts.pattern);
  const listed = run(listArgs, root);
  if (listed === null) return null;

  const files = listed.split('\n').filter((l) => l !== '');
  const glob = opts.glob;
  const filtered = glob !== undefined && glob !== '' ? files.filter((f) => f.includes(glob)) : files;
  const scanned = scanFiles(root, filtered, opts, 'git-grep+regex');
  return scanned;
}

/**
 * Run the JavaScript regex over an explicit list of files.
 *
 * This is the authoritative matcher: whatever engine produced the candidate list, the
 * answer the model sees comes from here, so the regex dialect is always the one the model
 * wrote.
 */
export function scanFiles(
  root: string,
  relPaths: string[],
  opts: SearchOptions,
  engine: SearchOutcome['engine'] = 'scan',
): SearchOutcome {
  const max = opts.maxResults ?? DEFAULT_MAX_RESULTS;
  const maxFiles = opts.maxFiles ?? DEFAULT_MAX_FILES;
  const maxLine = opts.maxLineChars ?? DEFAULT_MAX_LINE;
  const flags = opts.caseSensitive ? 'g' : 'gi';

  let re: RegExp;
  try {
    re = new RegExp(opts.pattern, flags);
  } catch {
    return { hits: [], engine, filesScanned: 0, truncated: false };
  }

  const hits: SearchHit[] = [];
  let scanned = 0;
  let truncated = false;

  for (const rel of relPaths) {
    if (hits.length >= max || scanned >= maxFiles) {
      truncated = true;
      break;
    }
    scanned++;
    let text: string;
    try {
      text = readFileSync(join(root, rel), 'utf8');
    } catch {
      continue;
    }
    if (text.includes('\u0000')) continue; // binary
    const lines = text.split('\n');
    for (let i = 0; i < lines.length; i++) {
      const line = lines[i] as string;
      re.lastIndex = 0;
      if (re.test(line)) {
        hits.push({ path: rel, line: i + 1, text: line.trim().slice(0, maxLine) });
        if (hits.length >= max) {
          truncated = true;
          break;
        }
      }
    }
  }

  return { hits, engine, filesScanned: scanned, truncated };
}
