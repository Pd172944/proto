/**
 * Which files matter for *this* task, in a repository too large to read.
 *
 * The agent's context window is the binding constraint, not its intelligence. Faced with
 * a 10k-file repository it needs to choose maybe fifteen files to look at, and choosing
 * them by filename or by recency is close to guessing. So the index builds a reference
 * graph and ranks it — the same family of technique the mature coding agents use, at a
 * fraction of the machinery.
 *
 * **How an edge is earned.** File A has an edge to file B when A mentions a name that B
 * defines. Weighting matters more than the topology here:
 *
 *  - **Rarity.** A name that one file defines and three files mention is a strong signal.
 *    A name like `Config` that forty files define is noise, and without a guard it would
 *    make every file look related to every other one — an O(N^2) edge set that is both
 *    slow to build and meaningless to rank. Names defined in more than `MAX_DEFINERS`
 *    files are skipped entirely.
 *  - **Imports.** An explicit import is a deliberate dependency and outranks an
 *    incidental identifier match, so a resolved import contributes a larger weight than
 *    a textual reference. Import specifiers are resolved to real paths where the language
 *    allows it (`./util`, `from a.b import c`, a Go package path).
 *  - **Direction.** Edges run from the user to the definition, which is what makes
 *    PageRank meaningful: a utility imported by twenty modules accumulates rank, and a
 *    leaf that imports nothing does not.
 *
 * **Personalization** is what makes the ranking about the task rather than about the
 * repository. PageRank with a uniform restart vector returns the repository's
 * "important" files, which are the same on every task and therefore useless. Seeding the
 * restart vector with the files and symbols the *task text* names turns it into "what is
 * relevant to this request", which is the actual question.
 */

import { posix } from 'node:path';

import type { FileSymbols, RankedFile, SymbolDef } from './types.ts';

/**
 * A name defined in more than this many files carries no signal. Without the cap the
 * edge set grows quadratically and the ranking collapses toward uniform.
 */
const MAX_DEFINERS = 24;

/** PageRank damping. 0.85 is the standard value and behaves correctly on code graphs. */
const DAMPING = 0.85;
const ITERATIONS = 24;

/** An import is a declared dependency; a bare mention is a hint. */
const IMPORT_WEIGHT = 3.0;
const REF_WEIGHT = 1.0;

export interface CodeGraph {
  /** Stable node order; indices in the adjacency lists refer into this. */
  paths: string[];
  index: Map<string, number>;
  out: Array<Array<{ to: number; w: number }>>;
  /** name -> node indices that define it. */
  definers: Map<string, number[]>;
  defsOf: Map<string, SymbolDef[]>;
}

/**
 * Resolves import specifiers to indexed paths in constant time.
 *
 * The obvious implementation — for each import, scan every known path looking for a
 * suffix match — is quadratic, and on a 10k-file repository it took long enough to
 * notice before the index was even built. Instead every path registers its own suffixes
 * once (`internal/a/b/c.py` registers `c`, `b/c`, `a/b/c`, `internal/a/b/c`), which is a
 * handful of map entries per file, and a lookup is then a single map read.
 *
 * The failure direction is deliberate: an unresolved import costs one weaker edge, and
 * the identifier-reference edges usually carry the same information, so this prefers to
 * miss rather than to guess wrong and create an edge between unrelated files.
 */
export class ImportResolver {
  private readonly paths: Set<string>;
  private readonly suffixes = new Map<string, string>();

  constructor(paths: Iterable<string>) {
    this.paths = new Set(paths);
    for (const p of this.paths) {
      const stem = p.replace(/\.(ts|tsx|js|jsx|mjs|cjs|py|pyi|go|rs|rb|java|kt|kts|swift|php|cs|c|h|cpp|hpp|scala|lua)$/, '');
      const parts = stem.split('/');
      // Register every suffix ending on a segment boundary, longest first so the
      // longest match wins and a bare `c` cannot shadow `a/b/c`.
      for (let i = 0; i < parts.length; i++) {
        const suffix = parts.slice(i).join('/');
        if (!this.suffixes.has(suffix)) this.suffixes.set(suffix, p);
      }
    }
  }

  private existing(candidates: string[]): string | null {
    for (const c of candidates) if (this.paths.has(c)) return c;
    return null;
  }

  resolve(fromPath: string, spec: string): string | null {
    if (spec === '') return null;

    if (spec.startsWith('.')) {
      const base = posix.normalize(posix.join(posix.dirname(fromPath), spec));
      const hit = this.existing([
        base,
        `${base}.ts`, `${base}.tsx`, `${base}.js`, `${base}.jsx`, `${base}.mjs`, `${base}.cjs`,
        `${base}.py`, `${base}.go`, `${base}.rs`, `${base}.rb`,
        `${base}/index.ts`, `${base}/index.tsx`, `${base}/index.js`, `${base}/index.jsx`,
        `${base}/__init__.py`, `${base}/mod.rs`,
        // TypeScript writes `./util.js` for a file that is `util.ts` on disk.
        base.replace(/\.(js|jsx|mjs|cjs)$/, '.ts'),
        base.replace(/\.(js|jsx|mjs|cjs)$/, '.tsx'),
      ]);
      return hit;
    }

    // Dotted Python module path, then the suffix table for everything else.
    if (spec.includes('.') && !spec.includes('/')) {
      const asPath = spec.replace(/\./g, '/');
      const hit = this.existing([`${asPath}.py`, `${asPath}/__init__.py`]);
      if (hit !== null) return hit;
    }

    // A package path (`github.com/x/internal/api/handler`, `@scope/pkg/sub`) has a
    // module prefix that never appears on disk, so try successively shorter suffixes and
    // take the first that names a real file. Without this a Go or Java import resolves
    // only when the whole path happens to be in the repository.
    const trimmed = spec.replace(/^@[^/]+\//, '');
    const parts = trimmed.split('/');
    for (let i = 0; i < parts.length; i++) {
      const hit = this.suffixes.get(parts.slice(i).join('/'));
      if (hit !== undefined) return hit;
    }
    return null;
  }
}

export function buildGraph(files: Map<string, FileSymbols>): CodeGraph {
  const paths = [...files.keys()].sort();
  const index = new Map<string, number>();
  paths.forEach((p, i) => index.set(p, i));

  const definers = new Map<string, number[]>();
  const defsOf = new Map<string, SymbolDef[]>();
  for (let i = 0; i < paths.length; i++) {
    const p = paths[i] as string;
    const f = files.get(p) as FileSymbols;
    defsOf.set(p, f.defs);
    for (const d of f.defs) {
      const list = definers.get(d.name);
      if (list === undefined) definers.set(d.name, [i]);
      else list.push(i);
    }
  }

  // Document frequency of each referenced name: how many files mention it. Rarity is
  // the whole basis for weighting, so it is computed once up front.
  const mentions = new Map<string, number>();
  for (const p of paths) {
    const f = files.get(p) as FileSymbols;
    for (const r of f.refs) mentions.set(r, (mentions.get(r) ?? 0) + 1);
  }

  const resolver = new ImportResolver(paths);
  const out: Array<Map<number, number>> = paths.map(() => new Map());

  const addEdge = (from: number, to: number, w: number): void => {
    if (from === to) return; // self-references say nothing about file relevance
    const m = out[from] as Map<number, number>;
    m.set(to, (m.get(to) ?? 0) + w);
  };

  for (let i = 0; i < paths.length; i++) {
    const p = paths[i] as string;
    const f = files.get(p) as FileSymbols;

    for (const ref of f.refs) {
      const targets = definers.get(ref);
      if (targets === undefined || targets.length === 0 || targets.length > MAX_DEFINERS) continue;
      // Rarity weight: a name mentioned by few files is a stronger link than one
      // mentioned by many. Log-scaled so the range stays narrow.
      const df = mentions.get(ref) ?? 1;
      const w = REF_WEIGHT / (1 + Math.log1p(df));
      for (const t of targets) addEdge(i, t, w);
    }

    for (const spec of f.imports) {
      const resolved = resolver.resolve(p, spec);
      if (resolved === null) continue;
      const t = index.get(resolved);
      if (t !== undefined) addEdge(i, t, IMPORT_WEIGHT);
    }
  }

  return {
    paths,
    index,
    out: out.map((m) => [...m.entries()].map(([to, w]) => ({ to, w }))),
    definers,
    defsOf,
  };
}

export interface RankOptions {
  /** Workspace-relative paths the task named. */
  focus?: string[];
  /** Symbol names the task named. */
  focusSymbols?: string[];
  /** Files already edited or read this session. */
  hot?: string[];
}

/** Match a task-named path against the index, tolerating a partial path. */
function matchPath(graph: CodeGraph, want: string): number[] {
  const norm = want.replace(/^\.\//, '').replace(/\\/g, '/');
  const exact = graph.index.get(norm);
  if (exact !== undefined) return [exact];
  const hits: number[] = [];
  for (let i = 0; i < graph.paths.length; i++) {
    const p = graph.paths[i] as string;
    if (p === norm || p.endsWith(`/${norm}`) || p.endsWith(norm)) hits.push(i);
  }
  return hits;
}

/**
 * Rank files by relevance, with the task's own vocabulary as the restart vector.
 *
 * Returns every file that has at least one definition, ordered. The caller applies the
 * token budget, not this function, so the ranking can be reused across budgets.
 */
export function rankFiles(graph: CodeGraph, opts: RankOptions = {}): RankedFile[] {
  const n = graph.paths.length;
  if (n === 0) return [];

  // Restart distribution. Uniform by default; task-named files and symbols dominate
  // when present, which is what makes the result specific to the request.
  const personal = new Float64Array(n).fill(1 / n);
  const isPersonal = new Uint8Array(n);

  const seed = (nodes: number[], weight: number): void => {
    if (nodes.length === 0) return;
    for (const i of nodes) {
      personal[i] = (personal[i] as number) + weight;
      isPersonal[i] = 1;
    }
  };

  for (const f of opts.focus ?? []) seed(matchPath(graph, f), 6);
  for (const h of opts.hot ?? []) seed(matchPath(graph, h), 4);
  for (const name of opts.focusSymbols ?? []) {
    // A file defining a symbol the task named is highly relevant; a file merely
    // mentioning it is not seeded at all, because the graph will reach it anyway.
    seed(graph.definers.get(name) ?? [], 8);
  }

  let sum = 0;
  for (let i = 0; i < n; i++) sum += personal[i] as number;
  for (let i = 0; i < n; i++) personal[i] = (personal[i] as number) / sum;

  // Power iteration over the sparse graph, with dangling mass redistributed by the
  // restart vector rather than dropped — otherwise rank leaks out of any leaf file.
  let rank = Float64Array.from(personal);
  const next = new Float64Array(n);

  for (let iter = 0; iter < ITERATIONS; iter++) {
    let dangling = 0;
    for (let i = 0; i < n; i++) {
      const outs = graph.out[i] as Array<{ to: number; w: number }>;
      if (outs.length === 0) dangling += rank[i] as number;
    }

    next.fill(0);
    for (let i = 0; i < n; i++) {
      const outs = graph.out[i] as Array<{ to: number; w: number }>;
      if (outs.length === 0) continue;
      let total = 0;
      for (const e of outs) total += e.w;
      const share = ((rank[i] as number) * DAMPING) / total;
      for (const e of outs) next[e.to] = (next[e.to] as number) + share * e.w;
    }
    for (let i = 0; i < n; i++) {
      next[i] = (next[i] as number) + (1 - DAMPING) * (personal[i] as number) + DAMPING * dangling * (personal[i] as number);
    }
    rank = Float64Array.from(next);
  }

  const scored: RankedFile[] = [];
  for (let i = 0; i < n; i++) {
    const p = graph.paths[i] as string;
    const defs = graph.defsOf.get(p) ?? [];
    if (defs.length === 0) continue;
    // A mild size prior, so that between two equally-connected files the one defining
    // more is shown first. Kept small on purpose: a large hub module (a theme, a types
    // file, a utility barrel) is imported by everything, so it already accrues rank from
    // the graph, and a generous size boost on top of that pushes it above files that are
    // genuinely specific to the task. It breaks ties; it does not decide.
    const boost = 1 + 0.04 * Math.log1p(defs.length);
    scored.push({ path: p, score: (rank[i] as number) * boost, defs, personal: isPersonal[i] === 1 });
  }

  scored.sort((a, b) => b.score - a.score || a.path.localeCompare(b.path));
  return scored;
}

/**
 * The file that defines `name`, best first.
 *
 * Ranked by how few files define the name, then by path: an exact and unique definition
 * is what the caller almost always wants, and a name defined in fifty files is better
 * answered by the list than by a guess.
 */
export function lookupSymbol(graph: CodeGraph, name: string): Array<{ path: string; def: SymbolDef }> {
  const nodes = graph.definers.get(name);
  if (nodes === undefined) return [];
  const out: Array<{ path: string; def: SymbolDef }> = [];
  for (const i of nodes) {
    const p = graph.paths[i] as string;
    for (const d of graph.defsOf.get(p) ?? []) {
      if (d.name === name) out.push({ path: p, def: d });
    }
  }
  return out;
}

/** Every indexed symbol whose name contains `needle`, when the exact name misses. */
export function searchSymbols(graph: CodeGraph, needle: string, limit: number): Array<{ name: string; path: string; def: SymbolDef }> {
  const lowered = needle.toLowerCase();
  const out: Array<{ name: string; path: string; def: SymbolDef }> = [];
  for (const [name, nodes] of graph.definers) {
    if (!name.toLowerCase().includes(lowered)) continue;
    for (const i of nodes) {
      const p = graph.paths[i] as string;
      for (const d of graph.defsOf.get(p) ?? []) {
        if (d.name === name) {
          out.push({ name, path: p, def: d });
          if (out.length >= limit) return out;
        }
      }
    }
  }
  return out;
}

/**
 * Files that mention `name`, cheapest possible answer.
 *
 * This deliberately returns paths and not line numbers. The cache stores the set of
 * identifiers per file, not their positions — recording a line for every mention of
 * every identifier would multiply the index size for information the caller can recover
 * by reading a file it already knows to open. `find_references` therefore uses this to
 * shortlist, then re-scans those few files to report exact lines.
 */
export function referringFiles(files: Map<string, FileSymbols>, name: string, exclude: Set<string>): string[] {
  const out: string[] = [];
  for (const [path, f] of files) {
    if (exclude.has(path)) continue;
    if (f.refs.includes(name)) out.push(path);
  }
  return out.sort();
}

/** Index files by the lowercase basename, for cheap name-based navigation. */
export function byBasename(graph: CodeGraph): Map<string, string[]> {
  const map = new Map<string, string[]>();
  for (const p of graph.paths) {
    const base = p.slice(p.lastIndexOf('/') + 1).toLowerCase();
    const list = map.get(base);
    if (list === undefined) map.set(base, [p]);
    else list.push(p);
  }
  return map;
}
