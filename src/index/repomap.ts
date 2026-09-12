/**
 * The repository map: the shape of a codebase inside a fixed character budget.
 *
 * This is the answer to "how does the agent see a 10k-file repository without reading
 * it". It is not a summary and not a search result — it is a ranked outline of
 * *definition signatures*, which is the densest useful representation of source that
 * exists. A signature tells the model what a thing is called, what it takes and what it
 * returns; the body is the part it can read on demand once it knows the name.
 *
 * Why signatures rather than anything shorter: a bare list of symbol names is cheap and
 * nearly useless — `handleRoute` does not say whether it takes a request or a path, and
 * the model cannot tell whether it is the function it needs. Why signatures rather than
 * bodies: a body is 20x the size for information the model only needs once it has
 * decided to look. The signature is the compression point where the ratio is best.
 *
 * The budget is applied by the caller in characters rather than tokens, because
 * character counts are exact and free while token counts require a tokenizer per model.
 * `estimateTokens` reports the rough figure and is documented as an approximation, not a
 * measurement — a map that slightly overshoots is a smaller problem than a map that
 * silently drops the file the task was about.
 */

import type { FileSymbols, RankedFile, RepoMap, RepoMapOptions, SymbolDef, SymbolKind } from './types.ts';

/** Short kind tags, because the map is read by a model under a budget. */
const KIND_TAG: Record<SymbolKind, string> = {
  function: 'fn',
  method: 'method',
  class: 'class',
  interface: 'iface',
  type: 'type',
  enum: 'enum',
  struct: 'struct',
  trait: 'trait',
  module: 'mod',
  const: 'const',
  variable: 'var',
  macro: 'macro',
  component: 'component',
  test: 'test',
};

export function estimateTokens(text: string): number {
  // ~4 characters per token holds well for source code across the common tokenizers.
  // Approximate on purpose: the alternative is shipping a tokenizer for every provider.
  return Math.ceil(text.length / 4);
}

/** Indent for a symbol, nested one level under its parent. */
function indentFor(def: SymbolDef): string {
  return def.parent === undefined || def.parent === '' ? '  ' : '    ';
}

/**
 * Strip the declaration boilerplate from a signature.
 *
 * The extractor keeps the source line, which reads `export function setColorEnabled(on:
 * boolean): void`. Rendering that under a `fn setColorEnabled` tag prints the name twice
 * and spends the map's budget on keywords the tag already conveys. What is left after
 * stripping is the part that carries information: parameters and return type.
 *
 * `export` is lifted out rather than discarded, because whether a symbol is public API
 * changes how a caller should treat it.
 */
function compactSignature(def: SymbolDef): { signature: string; exported: boolean } {
  let sig = def.signature.trim();
  const exported = /(^|\s)(export|pub|public)\s/.test(` ${sig}`);

  // Leading modifier run, e.g. `export async static readonly public`.
  sig = sig.replace(
    /^(?:(?:export|default|pub|public|private|protected|internal|static|async|abstract|readonly|final|open|override|synchronized|extern|inline|unsafe|mut)\s+)+/,
    '',
  );
  // The declaration keyword itself.
  sig = sig.replace(/^(?:function|def|fn|func|class|interface|type|struct|trait|enum|impl|mod|let|const|var|val|macro|component)\s+/, '');
  // The name, now redundant with the tag, including any generic parameter list.
  if (sig.startsWith(def.name)) {
    sig = sig.slice(def.name.length).replace(/^<[^>]*>/, '');
  }
  return { signature: sig.trim(), exported };
}

function lineFor(def: SymbolDef): string {
  const tag = KIND_TAG[def.kind] ?? 'sym';
  const name = def.parent !== undefined && def.parent !== '' && def.kind === 'method' ? `${def.parent}.${def.name}` : def.name;
  const { signature, exported } = compactSignature(def);
  // `export` leads, so the line reads like the declaration it came from rather than
  // `const export NAME`. A signature that starts with a bracket needs no separator.
  const vis = exported || def.exported === true ? 'export ' : '';
  const sep = signature === '' || signature.startsWith('(') || signature.startsWith('<') ? '' : ' ';
  return `${indentFor(def)}${vis}${tag} ${name}${sep}${signature}`;
}

/**
 * Render one file's definitions as an outline.
 *
 * Methods are nested under their parent by indentation, and the parent name is folded
 * into the method name (`Router.addRoute`) because indentation alone is lost the moment
 * a model quotes a line back into an edit.
 */
export function renderOutline(defs: SymbolDef[], maxDefs = 60): string[] {
  const lines: string[] = [];
  let shown = 0;
  for (const def of defs) {
    if (shown >= maxDefs) {
      lines.push(`  … ${defs.length - shown} more`);
      break;
    }
    lines.push(lineFor(def));
    shown++;
  }
  return lines;
}

/** A one-line summary of a file for compact listings. */
export function summarizeFile(f: FileSymbols): string {
  const kinds = new Set(f.defs.map((d) => d.kind));
  const main = f.defs.filter((d) => d.kind !== 'method').slice(0, 3).map((d) => d.name);
  const more = f.defs.length > main.length ? ` +${f.defs.length - main.length}` : '';
  return `${f.path} (${[...kinds].slice(0, 3).join('/')}) ${main.join(', ')}${more}`;
}

/**
 * Build the ranked, budgeted map.
 *
 * Files are taken in rank order and included whole when they fit. The first file that
 * does not fit is included partially — its header and as many signatures as remain —
 * and the map then stops. Stopping at the first overflow rather than skipping ahead to
 * smaller files keeps rank order meaningful: a model reading the map top-down is reading
 * it in order of relevance, and a map whose tail is a random scatter of tiny files is
 * worse than a short one.
 */
export function renderRepoMap(ranked: RankedFile[], opts: RepoMapOptions): RepoMap {
  const budget = Math.max(200, opts.budgetChars);
  const definedOnly = opts.definedOnly !== false;

  const lines: string[] = [];
  const included: RankedFile[] = [];
  let used = 0;
  let truncated = false;

  for (const file of ranked) {
    if (definedOnly && file.defs.length === 0) continue;

    const header = file.personal ? `${file.path}   ← named by the task` : file.path;
    const body = renderOutline(file.defs);

    // Cost of adding this file at all, including the blank line separator.
    const headerCost = header.length + 1;
    if (used + headerCost > budget) {
      truncated = true;
      break;
    }

    const room = budget - used - headerCost;
    const kept: string[] = [];
    let keptCost = 0;
    let cut = false;
    for (const line of body) {
      if (keptCost + line.length + 1 > room) {
        cut = true;
        break;
      }
      kept.push(line);
      keptCost += line.length + 1;
    }

    lines.push(header, ...kept);
    used += headerCost + keptCost;
    included.push({ ...file, defs: cut ? file.defs.slice(0, kept.length) : file.defs });

    if (cut || used >= budget) {
      truncated = true;
      break;
    }
    lines.push('');
    used += 1;
  }

  const omitted = ranked.length - included.length;
  if (truncated && omitted > 0) {
    lines.push('', `… ${omitted} further file(s) not shown. Use search, find_symbol or read_file to go deeper.`);
  }

  return { text: lines.join('\n').trimEnd(), files: included, totalFiles: ranked.length, truncated };
}

/**
 * The outline of a single file, for when the model knows which file it wants but not
 * what is in it. Cheaper than reading the file and usually enough to pick the range.
 */
export function renderFileOutline(f: FileSymbols, maxDefs = 120): string {
  const head = `${f.path}  (${f.lang}, ${f.lines} lines, ${f.bytes} bytes)`;
  if (f.defs.length === 0) {
    return `${head}\n  no definitions found — this file may be data, config, or use syntax the extractor does not recognise`;
  }
  const body = renderOutline(f.defs, maxDefs);
  const imports = f.imports.length > 0 ? `\n  imports: ${f.imports.slice(0, 12).join(', ')}` : '';
  return [head, ...body].join('\n') + imports;
}
