/**
 * File discovery that matches what git would track.
 *
 * The previous search tool decided what to skip from a hardcoded list of directory names.
 * That works on a repository someone wrote by hand and fails on a real one: `dist/`,
 * `coverage/`, `.turbo/`, `.parcel-cache/`, `*.min.js`, a vendored SDK under a name
 * nobody guessed. The result is a search that spends its budget on generated code and
 * silently stops before reaching the file the model needed.
 *
 * Two mechanisms, in order of preference:
 *
 *  1. **`git ls-files`.** When the workspace is inside a git repository this is exactly
 *     right and about two orders of magnitude cheaper than walking: it already applies
 *     every `.gitignore`, `.git/info/exclude`, nested ignores and the user's global
 *     excludes, and it is written in C. `--cached --others --exclude-standard` is the
 *     correct incantation: tracked files *plus* untracked files that are not ignored,
 *     which is what you want after the agent has just created something.
 *  2. **A walk with a gitignore matcher.** Only for a plain directory of source.
 *
 * Either way the same built-in filter runs last, because git will happily hand back a
 * 4 MB minified bundle that is tracked in the repository. Being tracked does not make a
 * file worth indexing.
 */

import { execFileSync } from 'node:child_process';
import { readFileSync } from 'node:fs';
import { readdir, readFile, stat } from 'node:fs/promises';
import { join, relative, resolve, sep } from 'node:path';

import { isIgnored, parseGitignore } from './ignore.ts';
import type { IgnoreRule } from './ignore.ts';
import { isIndexable } from './lang.ts';

export interface DiscoveredFile {
  /** Workspace-relative, POSIX separators, so the cache is portable across platforms. */
  path: string;
  abs: string;
  bytes: number;
  mtimeMs: number;
}

export interface Discovery {
  files: DiscoveredFile[];
  source: 'git' | 'walk';
  truncated: boolean;
  skippedIgnored: number;
  skippedTooLarge: number;
  skippedNotIndexable: number;
}

export interface DiscoverOptions {
  /** Hard ceiling on indexed files. Hitting it is reported, never silent. */
  maxFiles?: number;
  /** Per-file byte ceiling. Larger files are almost always generated. */
  maxBytes?: number;
  /** Skip the built-in name filter, for a repository that is mostly data. */
  includeGenerated?: boolean;
}

const DEFAULT_MAX_FILES = 40_000;
const DEFAULT_MAX_BYTES = 1_500_000;

/**
 * Name of the per-workspace exclusion file, using `.gitignore` syntax.
 *
 * Git's ignore rules answer "is this part of the project", which is not quite the same
 * question as "is this worth indexing". A repository can legitimately track benchmark
 * fixtures, vendored reference code, or generated SDKs that are real source to git and
 * pure noise to a symbol search. Rather than guessing at directory names, the index
 * honours an explicit list that the user controls.
 */
const PROTOIGNORE = '.protoignore';

/** Rules from `<root>/.protoignore`, or empty when there is none. */
function loadProtoIgnore(root: string): IgnoreRule[] {
  try {
    const text = readFileSync(join(root, PROTOIGNORE), 'utf8');
    return parseGitignore(text, '');
  } catch {
    return [];
  }
}

/**
 * Directories that never hold hand-written source, regardless of what git tracks.
 * Only used by the fallback walk — git's own ignores handle the git case — but it is
 * also applied as a final filter so a tracked build output does not enter the index.
 */
const NOISE_DIRS = new Set([
  '.git', '.hg', '.svn', 'node_modules', 'bower_components', 'jspm_packages',
  'dist', 'build', 'out', 'output', 'target', 'vendor', 'Pods', 'DerivedData',
  '.venv', 'venv', 'env', '.env', '__pycache__', '.mypy_cache', '.pytest_cache',
  '.ruff_cache', '.tox', '.nox', 'site-packages', 'eggs', '.eggs',
  '.next', '.nuxt', '.svelte-kit', '.parcel-cache', '.turbo', '.cache', '.gradle',
  'coverage', '.coverage', 'htmlcov', '.terraform', '.serverless', '.idea', '.vs',
  'obj', 'bin', 'Debug', 'Release', '.dart_tool', '.pub-cache',
]);

/**
 * Files that are large, machine-written, or both. A lockfile can be megabytes of
 * repeated package names; indexing it creates thousands of reference edges to nothing.
 */
const NOISE_FILE = /(?:^|\.)(?:min\.(?:js|css)|bundle\.js|chunk\.js|map)$|(?:^|\/)(?:package-lock\.json|yarn\.lock|pnpm-lock\.yaml|bun\.lockb?|Cargo\.lock|poetry\.lock|Pipfile\.lock|composer\.lock|Gemfile\.lock|go\.sum|flake\.lock)$/;

function isNoise(path: string): boolean {
  const parts = path.split('/');
  for (let i = 0; i < parts.length - 1; i++) {
    if (NOISE_DIRS.has(parts[i] as string)) return true;
  }
  return NOISE_FILE.test(path);
}

/** Run `git ls-files` for the given root, or null when this is not a git work tree. */
function gitList(root: string): string[] | null {
  try {
    const probe = execFileSync('git', ['rev-parse', '--is-inside-work-tree'], {
      cwd: root,
      encoding: 'utf8',
      stdio: ['ignore', 'pipe', 'ignore'],
      timeout: 10_000,
    }).trim();
    if (probe !== 'true') return null;

    // Paths come back relative to the *repository* root, which is not necessarily the
    // workspace root: the agent may be pointed at a subdirectory of a monorepo.
    const repoRoot = execFileSync('git', ['rev-parse', '--show-toplevel'], {
      cwd: root,
      encoding: 'utf8',
      stdio: ['ignore', 'pipe', 'ignore'],
      timeout: 10_000,
    }).trim();

    const out = execFileSync(
      'git',
      ['ls-files', '-z', '--cached', '--others', '--exclude-standard'],
      { cwd: root, encoding: 'utf8', stdio: ['ignore', 'pipe', 'ignore'], timeout: 60_000, maxBuffer: 64 * 1024 * 1024 },
    );

    const absRoot = resolve(root);
    const absRepo = resolve(repoRoot);
    const files: string[] = [];
    for (const entry of out.split('\0')) {
      if (entry === '') continue;
      const abs = resolve(absRepo, entry);
      if (abs !== absRoot && !abs.startsWith(absRoot + sep)) continue;
      files.push(relative(absRoot, abs).split(sep).join('/'));
    }
    return files;
  } catch {
    // Not a repository, git missing, or a timeout. The walk is the answer.
    return null;
  }
}

/** Walk the tree, honouring nested `.gitignore` files. Used only when git cannot help. */
async function walkTree(root: string, opts: Required<Pick<DiscoverOptions, 'maxFiles' | 'maxBytes'>>): Promise<{
  files: string[];
  ignored: number;
  truncated: boolean;
}> {
  const found: string[] = [];
  let ignored = 0;
  let truncated = false;

  const descend = async (abs: string, rel: string, rules: IgnoreRule[]): Promise<void> => {
    if (truncated) return;
    let entries: Array<{ name: string; isDirectory: () => boolean; isFile: () => boolean }>;
    try {
      entries = await readdir(abs, { withFileTypes: true });
    } catch {
      return;
    }

    // A .gitignore in this directory applies to it and everything below.
    let localRules = rules;
    if (entries.some((e) => e.name === '.gitignore')) {
      try {
        const text = await readFile(join(abs, '.gitignore'), 'utf8');
        localRules = [...rules, ...parseGitignore(text, rel === '' ? '' : rel)];
      } catch {
        // Unreadable ignore file: keep the parent rules rather than failing the walk.
      }
    }

    for (const entry of entries) {
      if (truncated) return;
      const childRel = rel === '' ? entry.name : `${rel}/${entry.name}`;
      if (entry.isDirectory()) {
        if (NOISE_DIRS.has(entry.name)) continue;
        if (isIgnored(childRel, true, localRules)) {
          ignored++;
          continue;
        }
        await descend(join(abs, entry.name), childRel, localRules);
        continue;
      }
      if (!entry.isFile()) continue;
      if (isIgnored(childRel, false, localRules)) {
        ignored++;
        continue;
      }
      if (found.length >= opts.maxFiles) {
        truncated = true;
        return;
      }
      found.push(childRel);
    }
  };

  await descend(root, '', []);
  return { files: found, ignored, truncated };
}

/** Stat in bounded-concurrency batches; a 40k-file tree is I/O bound, not CPU bound. */
async function statAll(root: string, paths: string[]): Promise<Array<{ bytes: number; mtimeMs: number } | null>> {
  const out: Array<{ bytes: number; mtimeMs: number } | null> = new Array(paths.length).fill(null);
  const limit = 32;
  let next = 0;

  const worker = async (): Promise<void> => {
    for (;;) {
      const i = next++;
      if (i >= paths.length) return;
      try {
        const st = await stat(join(root, paths[i] as string));
        if (st.isFile()) out[i] = { bytes: st.size, mtimeMs: st.mtimeMs };
      } catch {
        out[i] = null; // vanished between listing and stat; treated as absent.
      }
    }
  };

  await Promise.all(Array.from({ length: Math.min(limit, paths.length) }, worker));
  return out;
}

/**
 * Every indexable file under `root`, with the size and mtime the cache keys on.
 *
 * Returns `source` so callers can tell which mechanism answered — that distinction is
 * the difference between "git says this is the repository" and "we guessed".
 */
export async function discoverFiles(root: string, opts: DiscoverOptions = {}): Promise<Discovery> {
  const maxFiles = opts.maxFiles ?? DEFAULT_MAX_FILES;
  const maxBytes = opts.maxBytes ?? DEFAULT_MAX_BYTES;
  const absRoot = resolve(root);

  const workspaceRules = loadProtoIgnore(absRoot);
  const fromGit = gitList(absRoot);
  let candidates: string[];
  let ignored = 0;
  let truncated = false;
  const source: 'git' | 'walk' = fromGit !== null ? 'git' : 'walk';

  if (fromGit !== null) {
    candidates = fromGit;
  } else {
    const walked = await walkTree(absRoot, { maxFiles, maxBytes });
    candidates = walked.files;
    ignored = walked.ignored;
    truncated = walked.truncated;
  }

  let notIndexable = 0;
  const keep: string[] = [];
  for (const p of candidates) {
    if (isNoise(p)) {
      ignored++;
      continue;
    }
    if (workspaceRules.length > 0 && isIgnored(p, false, workspaceRules)) {
      ignored++;
      continue;
    }
    if (!isIndexable(p)) {
      notIndexable++;
      continue;
    }
    keep.push(p);
    if (keep.length >= maxFiles) {
      truncated = true;
      break;
    }
  }

  const stats = await statAll(absRoot, keep);
  const files: DiscoveredFile[] = [];
  let tooLarge = 0;
  for (let i = 0; i < keep.length; i++) {
    const st = stats[i];
    if (st === null || st === undefined) continue;
    if (st.bytes > maxBytes) {
      tooLarge++;
      continue;
    }
    files.push({ path: keep[i] as string, abs: join(absRoot, keep[i] as string), bytes: st.bytes, mtimeMs: st.mtimeMs });
  }

  return {
    files,
    source,
    truncated,
    skippedIgnored: ignored,
    skippedTooLarge: tooLarge,
    skippedNotIndexable: notIndexable,
  };
}
