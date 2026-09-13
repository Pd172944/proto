/**
 * The codebase index: one object the tools ask, with every answer degrading safely.
 *
 * Two properties matter more than anything else here.
 *
 * **The index is an accelerator, never a dependency.** Every method has a correct answer
 * when the index is empty, stale, half-built, or corrupt. A stale index that silently
 * hides a file from a search is the worst failure this module could have, because the
 * model would conclude the file does not exist and stop looking — so nothing here is
 * allowed to be the only path to an answer.
 *
 * **Building it must be cheap enough that nobody turns it off.** The work is one `stat`
 * per file plus a read and a lexer pass for whatever changed. On a warm cache that is
 * tens of milliseconds; on a cold 10k-file repository it is a few seconds, once, in the
 * background. Nothing waits on the full build to answer a question — a search runs
 * against git immediately and the index catches up.
 */

import { readFileSync } from 'node:fs';
import { readFile } from 'node:fs/promises';
import { join } from 'node:path';

import { buildGraph, rankFiles, searchSymbols } from './graph.ts';
import type { CodeGraph } from './graph.ts';
import { extract, maskSource } from './lang.ts';
import { renderFileOutline, renderRepoMap } from './repomap.ts';
import { referringFiles } from './graph.ts';
import { isLiteralPattern, normalizeSearchPattern, scanFiles, searchWithGit } from './search.ts';
import type { SearchOptions, SearchOutcome } from './search.ts';
import { cacheSize, loadIndex, saveIndex } from './store.ts';
import { discoverFiles } from './walk.ts';
import type {
  FileSymbols,
  IndexStats,
  RankedFile,
  RepoMap,
  RepoMapOptions,
  SymbolHit,
} from './types.ts';

/** Extensions worth recognising when a task mentions a path. */
const PATH_RE = /\b[\w@./-]+\.(?:ts|tsx|mts|cts|js|jsx|mjs|cjs|py|pyi|go|rs|rb|java|kt|kts|swift|php|cs|c|h|cc|cpp|hpp|scala|lua|sh|bash|zsh|sql|vue|svelte|md|json|ya?ml|toml)\b/g;
const IDENT_RE = /[A-Za-z_][A-Za-z0-9_]{2,}/g;
const QUOTED_RE = /[`'"]([^`'"]{2,80})[`'"]/g;

export interface TaskFocus {
  paths: string[];
  symbols: string[];
}

/**
 * Pull the paths and symbol names a task description refers to.
 *
 * This is what makes the ranking about the request rather than about the repository.
 * Uniform PageRank returns the same "important" files for every task, which is useless;
 * seeding it with the task's own vocabulary is what turns the map into an answer.
 *
 * Symbols are filtered against the index, so a word that happens to look like an
 * identifier but is defined nowhere does not consume restart mass. Paths are not
 * filtered, because a task may legitimately name a file that does not exist yet.
 */
export function focusFromTask(text: string, graph?: CodeGraph): TaskFocus {
  const paths = new Set<string>();
  for (const m of text.match(PATH_RE) ?? []) {
    paths.add(m.replace(/^\.\//, '').replace(/^["'`]|["'`]$/g, ''));
  }

  // Anything quoted is likely to be a name the user cares about, even if it is short.
  const quoted = new Set<string>();
  for (const m of text.matchAll(QUOTED_RE)) {
    const inner = m[1] as string;
    if (inner.includes('\n') || inner.length > 60) continue;
    quoted.add(inner);
  }

  const symbols = new Set<string>();
  for (const word of text.match(IDENT_RE) ?? []) {
    if (graph !== undefined && graph.definers.has(word)) symbols.add(word);
    else if (quoted.has(word)) symbols.add(word);
  }
  for (const q of quoted) {
    if (graph !== undefined && graph.definers.has(q)) symbols.add(q);
  }

  return { paths: [...paths].slice(0, 8), symbols: [...symbols].slice(0, 12) };
}

export class CodebaseIndex {
  readonly root: string;
  private readonly dataDir: string;
  private files = new Map<string, FileSymbols>();
  private graphCache: CodeGraph | null = null;
  private lastStats: IndexStats;
  /** Set when the last build could not read or write the cache; surfaced, not hidden. */
  private notes: string[] = [];

  constructor(root: string, dataDir: string) {
    this.root = root;
    this.dataDir = dataDir;
    this.lastStats = {
      files: 0, changed: 0, removed: 0, buildMs: 0, bytesIndexed: 0, symbols: 0, cacheBytes: 0,
    };
  }

  get stats(): IndexStats {
    return { ...this.lastStats, cacheBytes: cacheSize(this.dataDir, this.root) };
  }

  get warnings(): string[] {
    return [...this.notes];
  }

  get ready(): boolean {
    return this.files.size > 0;
  }

  entries(): Map<string, FileSymbols> {
    return this.files;
  }

  get graph(): CodeGraph {
    if (this.graphCache === null) this.graphCache = buildGraph(this.files);
    return this.graphCache;
  }

  /**
   * Bring the index up to date.
   *
   * `mtimeMs + size` is the staleness key. It is not a hash, and it can be fooled — a
   * file rewritten within the same millisecond to the same length would be missed. That
   * tradeoff is deliberate: hashing every file on every refresh costs a full read of the
   * repository, which is precisely the expense the index exists to avoid. The failure
   * window is one refresh, and `--force` exists to close it.
   */
  async refresh(force = false): Promise<IndexStats> {
    const started = Date.now();
    this.notes = [];

    const loaded = loadIndex(this.dataDir, this.root);
    if (loaded.damaged > 0) {
      this.notes.push(`index cache had ${loaded.damaged} damaged line(s); those files were re-extracted`);
    }
    const previous = force ? new Map<string, FileSymbols>() : loaded.files;

    const discovery = await discoverFiles(this.root);
    if (discovery.truncated) {
      this.notes.push(`file discovery hit its ceiling; some files are not indexed (${discovery.files.length} indexed)`);
    }

    const current = new Map<string, FileSymbols>();
    const stale: Array<{ path: string; abs: string; bytes: number; mtimeMs: number }> = [];
    let bytesIndexed = 0;

    for (const f of discovery.files) {
      const cached = previous.get(f.path);
      if (cached !== undefined && cached.mtimeMs === f.mtimeMs && cached.bytes === f.bytes) {
        current.set(f.path, cached);
        bytesIndexed += f.bytes;
        continue;
      }
      stale.push(f);
    }

    // Extract in bounded batches. Reading is I/O bound, so a small amount of concurrency
    // buys most of the available speedup without opening ten thousand descriptors.
    const BATCH = 48;
    for (let i = 0; i < stale.length; i += BATCH) {
      const slice = stale.slice(i, i + BATCH);
      const results = await Promise.all(
        slice.map(async (f) => {
          try {
            const text = await readFile(f.abs, 'utf8');
            return { f, text };
          } catch {
            return { f, text: null as string | null };
          }
        }),
      );
      for (const { f, text } of results) {
        if (text === null) continue;
        const r = extract(f.path, text);
        current.set(f.path, {
          path: f.path,
          lang: r.lang,
          bytes: f.bytes,
          lines: text.split('\n').length,
          mtimeMs: f.mtimeMs,
          defs: r.defs,
          refs: r.refs,
          imports: r.imports,
        });
        bytesIndexed += f.bytes;
      }
    }

    let removed = 0;
    for (const path of previous.keys()) if (!current.has(path)) removed++;

    this.files = current;
    this.graphCache = null;

    if (stale.length > 0 || removed > 0 || !loaded.existed) {
      try {
        saveIndex(this.dataDir, this.root, current.values());
      } catch (err) {
        // A read-only data directory must not break the session; the index stays in
        // memory and simply rebuilds next time.
        this.notes.push(`could not persist the index (${err instanceof Error ? err.message : String(err)}); it will rebuild next session`);
      }
    }

    let symbols = 0;
    for (const f of current.values()) symbols += f.defs.length;

    this.lastStats = {
      files: current.size,
      changed: stale.length,
      removed,
      buildMs: Date.now() - started,
      bytesIndexed,
      symbols,
      cacheBytes: cacheSize(this.dataDir, this.root),
    };
    return this.lastStats;
  }

  /**
   * Ensure a build has happened, without requiring one for correctness.
   *
   * Tools call this first. It is a no-op after the first call, so the cost is paid once
   * per session and never blocks a second question.
   */
  async ensure(): Promise<void> {
    if (this.files.size > 0) return;
    try {
      await this.refresh(false);
    } catch (err) {
      this.notes.push(`index unavailable (${err instanceof Error ? err.message : String(err)}); falling back to direct search`);
    }
  }

  /** Ranked files with the map rendered to the caller's budget. */
  repoMap(opts: RepoMapOptions): RepoMap {
    const ranked = this.rank(opts);
    return renderRepoMap(ranked, opts);
  }

  rank(opts: RepoMapOptions): RankedFile[] {
    return rankFiles(this.graph, {
      focus: opts.focus ?? [],
      focusSymbols: opts.focusSymbols ?? [],
      hot: opts.hot ?? [],
    });
  }

  /** Definitions of `name`; exact matches first. */
  findSymbol(name: string, limit = 20): { hits: SymbolHit[]; exact: boolean } {
    const g = this.graph;
    const nodes = g.definers.get(name);
    if (nodes !== undefined) {
      const hits: SymbolHit[] = [];
      for (const i of nodes) {
        const path = g.paths[i] as string;
        for (const d of g.defsOf.get(path) ?? []) {
          if (d.name !== name) continue;
          hits.push({ path, line: d.line, kind: d.kind, signature: d.signature });
          if (hits.length >= limit) break;
        }
      }
      return { hits, exact: true };
    }
    const fuzzy = searchSymbols(g, name, limit).map(
      (s): SymbolHit => ({ path: s.path, line: s.def.line, kind: s.def.kind, signature: s.def.signature }),
    );
    return { hits: fuzzy, exact: false };
  }

  /** Where `name` is used, excluding the files that define it. */
  findReferences(name: string, limit = 60): SymbolHit[] {
    const g = this.graph;
    const definers = new Set<string>();
    for (const i of g.definers.get(name) ?? []) definers.add(g.paths[i] as string);

    const candidates = referringFiles(this.files, name, definers);
    if (candidates.length === 0) return [];

    // The index knows *which* files mention the name but not where in them. Re-scanning
    // the shortlist is cheap — a handful of files, not the repository — and it is the
    // only way to report a line number that is actually correct.
    //
    // The scan runs over *masked* source, using the same masking the extractor used. A
    // raw regex would report the mention in a doc comment and the name inside a string
    // literal as call sites, which is exactly the noise this tool exists to remove.
    const wordRe = new RegExp(`\\b${name.replace(/[.*+?^${}()|[\]\\]/g, '\\$&')}\\b`);
    const hits: SymbolHit[] = [];
    for (const path of candidates) {
      const f = this.files.get(path);
      if (f === undefined) continue;
      let raw: string;
      try {
        raw = readFileSync(join(this.root, path), 'utf8');
      } catch {
        continue;
      }
      const masked = maskSource(path, raw);
      const rawLines = raw.split('\n');
      const maskedLines = masked.split('\n');
      for (let i = 0; i < maskedLines.length; i++) {
        if (!wordRe.test(maskedLines[i] as string)) continue;
        // Report the original source line, not the masked one: the model needs to read
        // the real code, and only the *decision* depends on the mask.
        hits.push({ path, line: i + 1, kind: 'reference', text: (rawLines[i] ?? '').trim().slice(0, 200) });
        if (hits.length >= limit) return hits;
      }
    }
    return hits;
  }

  /** Definitions in one file, or null when the file is not indexed. */
  outline(path: string): string | null {
    const f = this.files.get(path.replace(/^\.\//, '').replace(/\\/g, '/'));
    if (f === undefined) return null;
    return renderFileOutline(f);
  }

  /**
   * Search, preferring git and falling back to the index's own file list.
   *
   * The regex path is deliberately suspicious of an empty git result: `git grep -E` and
   * JavaScript do not share a dialect, so "git found no candidate files" is not strong
   * enough evidence to tell the model that nothing matches. An empty shortlist therefore
   * falls through to a bounded scan, which is allowed to return "no matches".
   */
  search(opts: SearchOptions): SearchOutcome {
    const normalized: SearchOptions = { ...opts, pattern: normalizeSearchPattern(opts.pattern) };
    const git = searchWithGit(this.root, normalized);
    if (git !== null) {
      if (git.hits.length > 0 || isLiteralPattern(normalized.pattern)) return git;
      // Empty shortlist for a regex: verify before reporting absence.
    }

    const candidates = this.candidatePaths(normalized);
    const scanned = scanFiles(this.root, candidates, normalized, 'scan');
    return scanned;
  }

  /**
   * Files worth scanning, derived from the index.
   *
   * When the pattern contains a literal word the index can answer "which files mention
   * it", which turns a repository-wide scan into a scan of a dozen files. Otherwise the
   * whole indexed file list is the candidate set — still cheaper than a fresh walk,
   * because discovery already excluded everything ignored.
   */
  private candidatePaths(opts: SearchOptions): string[] {
    const glob = opts.glob;
    const filter = (paths: string[]): string[] =>
      glob !== undefined && glob !== '' ? paths.filter((p) => p.includes(glob)) : paths;

    if (this.files.size === 0) return [];

    const literals = opts.pattern.match(/[A-Za-z_][A-Za-z0-9_]{2,}/g);
    if (literals !== null && literals.length > 0) {
      const narrowed = new Set<string>();
      for (const word of literals) {
        for (const [path, f] of this.files) {
          if (f.refs.includes(word) || f.defs.some((d) => d.name === word)) narrowed.add(path);
        }
      }
      // An empty narrowing means the word appears nowhere the index knows about, which
      // for a literal means it genuinely is not there — but only trust that when the
      // index is actually populated.
      if (narrowed.size > 0) return filter([...narrowed]);
    }
    return filter([...this.files.keys()]);
  }
}
