/**
 * Codebase intelligence: symbols, references, and a ranked map of the repository.
 *
 * Why this exists. The agent's original search tool walked the tree and read *every*
 * file into memory on *every* call, with a hardcoded skip list and a 4000-file ceiling.
 * On a small repository that is fine. On a real one it is three failures at once: it is
 * slow (O(repository bytes) per search), it gives up silently partway through the tree,
 * and it walks `dist/`, `coverage/` and every vendored dependency it does not recognise
 * by name. The model then compensates by reading whole files, which is how a large
 * repository eats a context window in about six tool calls.
 *
 * The three things that actually fix that, in the order they matter:
 *
 *  1. **A symbol index.** Definitions and references per file, extracted once and cached
 *     against mtime+size. This is what turns "search the whole repository" into "search
 *     the twelve files that mention this name".
 *  2. **A reference graph, ranked.** Which files matter for *this* task, computed by
 *     spreading activation from the files the task names. This is what lets the agent
 *     see the shape of a 10k-file repository through a 2k-token window.
 *  3. **Discovery that matches git.** `git ls-files` and `.gitignore` rather than a list
 *     of directory names someone guessed in 2024.
 *
 * Deliberate constraints, because the alternative is worse:
 *
 *  - **No embedding model, no vector store.** It would add a dependency, a build step, a
 *    multi-hundred-megabyte artifact and a staleness problem, and it would need to run on
 *    the user's machine. The mature agents that scale to large repositories do it with
 *    agentic search and symbol graphs; the ranking here is the same idea, cheaply.
 *  - **The index is an accelerator, never a dependency.** Every tool that uses it has a
 *    correct fallback for a missing, stale, or corrupt index. A wrong index that silently
 *    hides a file is worse than a slow search.
 *  - **Extraction is approximate and must stay fast.** A regex/lexer pass over masked
 *    source, not a real parser: no tree-sitter, no per-language grammar, no dependency.
 *    It will miss symbols in exotic syntax. It must not miss them *silently* — every
 *    symbol tool reports when it fell back to a raw scan.
 */

/** What a definition is, for display and for ranking. */
export type SymbolKind =
  | 'function'
  | 'method'
  | 'class'
  | 'interface'
  | 'type'
  | 'enum'
  | 'struct'
  | 'trait'
  | 'module'
  | 'const'
  | 'variable'
  | 'macro'
  | 'component'
  | 'test';

export interface SymbolDef {
  name: string;
  kind: SymbolKind;
  /** 1-based line of the definition. */
  line: number;
  /** Single-line signature for the outline, already trimmed and length-capped. */
  signature: string;
  /** Enclosing class/namespace, when the extractor can tell. */
  parent?: string;
  /** True when the language's export rules make this part of the public surface. */
  exported?: boolean;
}

export interface FileSymbols {
  /** Workspace-relative, always POSIX separators so the cache is portable. */
  path: string;
  lang: string;
  bytes: number;
  lines: number;
  mtimeMs: number;
  defs: SymbolDef[];
  /** Distinct identifiers referenced in code position, capped. See MAX_REFS_PER_FILE. */
  refs: string[];
  /** Module specifiers / include paths exactly as written in the source. */
  imports: string[];
}

export interface ExtractResult {
  lang: string;
  defs: SymbolDef[];
  refs: string[];
  imports: string[];
}

/**
 * References are capped per file. A very large generated file can contain thousands of
 * distinct identifiers, and the marginal graph edge from the 500th is worth nothing
 * while the storage cost is linear. The cap is high enough that ordinary source is
 * never truncated in practice.
 */
export const MAX_REFS_PER_FILE = 400;

/** Signature length cap, so one pathological line cannot blow up the outline. */
export const MAX_SIGNATURE_CHARS = 120;

/** A ranked file in the repository map. */
export interface RankedFile {
  path: string;
  score: number;
  defs: SymbolDef[];
  /** True when the score came (partly) from the task naming this file or symbol. */
  personal: boolean;
}

export interface RepoMapOptions {
  /** Approximate character budget for the rendered map. */
  budgetChars: number;
  /** Paths the task mentions; these seed the ranking. */
  focus?: string[];
  /** Symbol names the task mentions; files defining them are seeded. */
  focusSymbols?: string[];
  /** Files the session has already edited; they stay relevant. */
  hot?: string[];
  /** Only include files carrying at least one definition. */
  definedOnly?: boolean;
}

export interface RepoMap {
  text: string;
  files: RankedFile[];
  /** Total indexed files, before the budget cut. */
  totalFiles: number;
  /** True when the budget cut files off the end. */
  truncated: boolean;
}

/** A single hit from a symbol or reference lookup. */
export interface SymbolHit {
  path: string;
  line: number;
  kind: SymbolKind | 'reference';
  signature?: string;
  /** The source line, trimmed, for context. */
  text?: string;
}

/** Statistics surfaced by `proto index` and by the tool metadata. */
export interface IndexStats {
  files: number;
  /** Files re-extracted on this build; 0 means the cache was fully warm. */
  changed: number;
  removed: number;
  buildMs: number;
  bytesIndexed: number;
  symbols: number;
  /** Size of the on-disk cache. */
  cacheBytes: number;
}
