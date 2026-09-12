/**
 * The on-disk symbol cache.
 *
 * The index would be worthless if it had to be rebuilt from scratch every session: a
 * 10k-file repository takes seconds to read and scan, and the agent asks for the map on
 * the first turn of every conversation. So entries are cached against `mtimeMs + size`,
 * and a warm start costs one `stat` per file and one file read for the whole index.
 *
 * **Why the reference table exists.** Definitions are small — a file has tens of them.
 * References are not: a single file can name a hundred distinct identifiers, and storing
 * `["useState","useEffect",...]` per file, per line, adds up to tens of megabytes on a
 * large repository. Instead the file stores a dictionary of every distinct identifier
 * seen in the repository once, and each file stores its references as indices into it.
 * On a 10k-file repository that is roughly a 4x reduction, which is the difference
 * between an index people tolerate and one they delete.
 *
 * **Format.** Line 1 is a header carrying the version, the absolute root (so a cache
 * from a different checkout cannot be mistaken for this one), and the reference
 * dictionary. Every following line is one file. It is JSONL rather than one big JSON
 * document so that a truncated write loses the last file rather than the whole index —
 * and a corrupt line is skipped, not fatal.
 */

import { existsSync, mkdirSync, readFileSync, renameSync, writeFileSync } from 'node:fs';
import { createHash } from 'node:crypto';
import { dirname, join } from 'node:path';

import type { FileSymbols, SymbolDef, SymbolKind } from './types.ts';

const CACHE_VERSION = 1;

interface Header {
  v: number;
  root: string;
  builtAt: number;
  /** Dictionary of every distinct reference identifier in the repository. */
  refs: string[];
}

/** Compact on-disk definition: short keys because there are a great many of them. */
interface WireDef {
  n: string;
  k: SymbolKind;
  l: number;
  s: string;
  p?: string;
  e?: boolean;
}

interface WireFile {
  p: string;
  m: number;
  s: number;
  l: number;
  g: string;
  d: WireDef[];
  r: number[];
  i: string[];
}

/** Stable per-workspace cache filename: one index per checkout, not per session. */
export function workspaceKey(root: string): string {
  return createHash('sha1').update(root).digest('hex').slice(0, 16);
}

export function cachePath(dataDir: string, root: string): string {
  return join(dataDir, 'index', `${workspaceKey(root)}.jsonl`);
}

export interface LoadedIndex {
  files: Map<string, FileSymbols>;
  builtAt: number;
  /** True when a cache file existed and was readable at all. */
  existed: boolean;
  /** Lines skipped because they were unparseable; a non-zero value means corruption. */
  damaged: number;
}

/**
 * Read the cache for `root`. Never throws: an unreadable, stale-version or
 * foreign-root cache is reported as empty, and the caller rebuilds.
 */
export function loadIndex(dataDir: string, root: string): LoadedIndex {
  const empty: LoadedIndex = { files: new Map(), builtAt: 0, existed: false, damaged: 0 };
  const path = cachePath(dataDir, root);
  if (!existsSync(path)) return empty;

  let text: string;
  try {
    text = readFileSync(path, 'utf8');
  } catch {
    return empty;
  }

  const lines = text.split('\n');
  const headerLine = lines[0];
  if (headerLine === undefined || headerLine === '') return { ...empty, existed: true };

  let header: Header;
  try {
    header = JSON.parse(headerLine) as Header;
  } catch {
    return { ...empty, existed: true, damaged: 1 };
  }
  if (header.v !== CACHE_VERSION || header.root !== root || !Array.isArray(header.refs)) {
    return { ...empty, existed: true };
  }

  const files = new Map<string, FileSymbols>();
  let damaged = 0;
  for (let i = 1; i < lines.length; i++) {
    const line = lines[i];
    if (line === undefined || line === '') continue;
    try {
      const w = JSON.parse(line) as WireFile;
      files.set(w.p, {
        path: w.p,
        lang: w.g,
        bytes: w.s,
        lines: w.l,
        mtimeMs: w.m,
        defs: w.d.map(
          (d): SymbolDef => ({
            name: d.n,
            kind: d.k,
            line: d.l,
            signature: d.s,
            ...(d.p === undefined ? {} : { parent: d.p }),
            ...(d.e === undefined ? {} : { exported: d.e }),
          }),
        ),
        refs: w.r.map((idx) => header.refs[idx]).filter((s): s is string => typeof s === 'string'),
        imports: Array.isArray(w.i) ? w.i : [],
      });
    } catch {
      damaged++;
    }
  }

  return { files, builtAt: header.builtAt, existed: true, damaged };
}

/**
 * Write the cache atomically.
 *
 * The dictionary is rebuilt from the entries being written rather than carried forward,
 * so it can never drift out of sync with the file records that index into it — which is
 * the one failure that would silently corrupt every reference in the graph.
 */
export function saveIndex(dataDir: string, root: string, files: Iterable<FileSymbols>): number {
  const path = cachePath(dataDir, root);
  mkdirSync(dirname(path), { recursive: true });

  // Intern references, and keep the dictionary in first-seen order so a rebuild that
  // changes nothing produces a byte-identical file.
  const refIds = new Map<string, number>();
  const refTable: string[] = [];
  const intern = (name: string): number => {
    const existing = refIds.get(name);
    if (existing !== undefined) return existing;
    const id = refTable.length;
    refIds.set(name, id);
    refTable.push(name);
    return id;
  };

  const out: string[] = [];
  for (const f of files) {
    const wire: WireFile = {
      p: f.path,
      m: f.mtimeMs,
      s: f.bytes,
      l: f.lines,
      g: f.lang,
      d: f.defs.map((d) => ({
        n: d.name,
        k: d.kind,
        l: d.line,
        s: d.signature,
        ...(d.parent === undefined ? {} : { p: d.parent }),
        ...(d.exported === undefined ? {} : { e: d.exported }),
      })),
      r: f.refs.map(intern),
      i: f.imports,
    };
    out.push(JSON.stringify(wire));
  }

  const header: Header = { v: CACHE_VERSION, root, builtAt: Date.now(), refs: refTable };
  const body = `${JSON.stringify(header)}\n${out.join('\n')}\n`;

  // Write to a sibling then rename: a crash mid-write must not leave a half index that
  // a later load would happily parse as a complete one.
  const tmp = `${path}.${process.pid}.tmp`;
  writeFileSync(tmp, body, 'utf8');
  renameSync(tmp, path);
  return Buffer.byteLength(body, 'utf8');
}

/** Remove the cache for this workspace. Used by `proto index --force`. */
export function cacheSize(dataDir: string, root: string): number {
  const path = cachePath(dataDir, root);
  try {
    return readFileSync(path).length;
  } catch {
    return 0;
  }
}
