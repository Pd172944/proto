/**
 * Language identification and per-language symbol extraction.
 *
 * This is the piece that turns a file on disk into `defs`, `refs` and `imports`. It is
 * deliberately *not* a parser: no tree-sitter, no per-language grammar, no dependency.
 * A regex pass over masked source gets the decisions right often enough to build a
 * useful reference graph, and — critically — it is cheap enough to run over a whole
 * repository on every cache miss.
 *
 * The contract that matters most is in `extract`: it must never throw. A file can be
 * binary, truncated mid-string, truncated mid-comment, or simply not the language its
 * extension claims. Every one of those is a routine event while someone is editing, and
 * a symbol index that crashes on them is worse than no index at all, because the tool
 * that depends on it has to choose between lying and dying.
 *
 * Two decisions are worth stating up front because they look like bugs otherwise:
 *
 *  - **Extraction runs on `maskCode(text, spec)`, never on raw source.** A commented-out
 *    `def handle()` or a SQL string containing `CREATE TABLE` must not become a symbol.
 *    The masker blanks comments and string *contents* while keeping every newline and
 *    the string delimiters, which is what lets an import regex still see the quotes and
 *    then read the real specifier back out of the raw line.
 *  - **A symbol's `signature` comes from the raw line, its `line`/`name` from the masked
 *    one.** The raw line is what a human wants to read in an outline; the masked line is
 *    what tells us a symbol is really there. They share an index because masking is
 *    length-preserving.
 *
 * Fidelity is uneven on purpose. Brace-and-keyword languages come out well; languages
 * whose structure is carried by indentation or `end` keywords are approximated with an
 * indentation stack; Lua's `--[[ ]]` block comments degrade to line comments because the
 * shared masker checks line comments first and `--` is a prefix of `--[[`. Each of those
 * limits is called out at the site that pays for it.
 */

import { MAX_REFS_PER_FILE, MAX_SIGNATURE_CHARS } from './types.ts';
import type { ExtractResult, SymbolDef, SymbolKind } from './types.ts';
import { collectIdentifiers, maskCode } from './lex.ts';
import type { LexSpec, StringSpec } from './lex.ts';

/* ------------------------------------------------------------------ */
/* Shared lexer specs                                                  */
/* ------------------------------------------------------------------ */

const DQ: StringSpec = { open: '"', close: '"', escape: '\\' };
const SQ: StringSpec = { open: "'", close: "'", escape: '\\' };
const BACKTICK: StringSpec = { open: '`', close: '`', escape: '\\', multiline: true };
const TRIPLE_DQ: StringSpec = { open: '"""', close: '"""', escape: '\\', multiline: true };
const TRIPLE_SQ: StringSpec = { open: "'''", close: "'''", escape: '\\', multiline: true };

/** C-like: slash-slash line comments, slash-star block comments, quoted strings. Java, C#, C, C++, PHP. */
const C_LINE: LexSpec = { lineComments: ['//'], blockComments: [['/*', '*/']], strings: [DQ, SQ] };
/** JavaScript/TypeScript adds template literals, which may span lines. */
const JS_LEX: LexSpec = { lineComments: ['//'], blockComments: [['/*', '*/']], strings: [BACKTICK, DQ, SQ] };
const PY_LEX: LexSpec = { lineComments: ['#'], blockComments: [], strings: [TRIPLE_DQ, TRIPLE_SQ, DQ, SQ] };
/** Go raw strings use backticks and, unusually, may contain newlines. */
const GO_LEX: LexSpec = { lineComments: ['//'], blockComments: [['/*', '*/']], strings: [BACKTICK, DQ, SQ] };
/**
 * Rust char literals are deliberately absent. `'a` is a lifetime, not a string opener,
 * and treating it as one would swallow the rest of the line and hide the very `fn`
 * declarations we are looking for. Missing `'x'` literals costs us a little masking.
 */
const RUST_LEX: LexSpec = { lineComments: ['//'], blockComments: [['/*', '*/']], strings: [DQ] };
const HASH_LEX: LexSpec = { lineComments: ['#'], blockComments: [], strings: [DQ, SQ] };
const SQL_LEX: LexSpec = { lineComments: ['--'], blockComments: [['/*', '*/']], strings: [DQ, SQ] };
/**
 * Lua long strings are real, but `--[[` cannot be expressed as a block comment here:
 * `maskCode` matches line comments first and `--` is a prefix of `--[[`, so a `--[[`
 * block only loses its first line. That direction is safe (it may extract a symbol from
 * a multi-line comment, never hide a real one), and Lua is rare enough to accept it.
 */
const LUA_LEX: LexSpec = { lineComments: ['--'], blockComments: [], strings: [{ open: '[[', close: ']]', multiline: true }, DQ, SQ] };
const RUBY_LEX: LexSpec = { lineComments: ['#'], blockComments: [['=begin', '=end']], strings: [DQ, SQ] };
const KOTLIN_LEX: LexSpec = { lineComments: ['//'], blockComments: [['/*', '*/']], strings: [TRIPLE_DQ, DQ, SQ] };
/** Markdown, JSON and the script regions of Vue/Svelte are masked by their own rules. */
const EMPTY_LEX: LexSpec = { lineComments: [], blockComments: [], strings: [] };

/* ------------------------------------------------------------------ */
/* Small helpers                                                       */
/* ------------------------------------------------------------------ */

interface Extraction {
  defs: SymbolDef[];
  imports: string[];
}

interface LanguageDef {
  id: string;
  extensions: string[];
  lex: LexSpec;
  /** `masked` is `maskCode(raw, lex)`; both have identical length and line structure. */
  extract(masked: string, raw: string): Extraction;
}

/**
 * A one-line signature for the outline.
 *
 * Trailing `{`, `=>` and `:` are stripped because the outline reads as a declaration,
 * not as the first line of a body. Only one of them, and only at the end, so a ternary
 * or a type annotation in the middle survives.
 */
function signatureOf(rawLine: string): string {
  let s = rawLine.trim();
  // An empty body on the same line (`function f() {}`) is still just a declaration in
  // the outline, so the pair goes too; a non-empty `{` is stripped on its own.
  s = s.replace(/\s*\{\s*\}\s*$/, '');
  s = s.replace(/\s*\{\s*$/, '');
  s = s.replace(/\s*=>\s*$/, '');
  s = s.replace(/\s*:\s*$/, '');
  s = s.trim();
  return s.length > MAX_SIGNATURE_CHARS ? s.slice(0, MAX_SIGNATURE_CHARS) : s;
}

function makeDef(
  name: string,
  kind: SymbolKind,
  lineIndex: number,
  rawLines: string[],
  parent: string | undefined,
  exported: boolean,
): SymbolDef {
  const def: SymbolDef = { name, kind, line: lineIndex + 1, signature: signatureOf(rawLines[lineIndex] ?? '') };
  // A type that names its own scope as parent ("class Foo" inside Foo) is not useful.
  if (parent !== undefined && parent !== name) def.parent = parent;
  // Only ever set when true: an absent flag means "this language has no export notion
  // here", which is different information from "this symbol is private".
  if (exported) def.exported = true;
  return def;
}

function addUnique(list: string[], seen: Set<string>, value: string): void {
  const v = value.trim();
  if (v === '' || seen.has(v)) return;
  seen.add(v);
  list.push(v);
}

function indentationOf(line: string): number {
  let n = 0;
  for (const ch of line) {
    if (ch === ' ') n += 1;
    else if (ch === '\t') n += 4;
    else break;
  }
  return n;
}

function countNewlines(text: string): number {
  let n = 0;
  for (let i = 0; i < text.length; i++) if (text[i] === '\n') n += 1;
  return n;
}

/**
 * Open-brace depth at the start of every line, computed over masked text.
 *
 * `{` inside a string or comment is a space by now, so a regex literal or a template
 * containing a brace cannot shift the depth and mis-parent every symbol after it. Brace
 * languages are the ones that need it; for the others the value is simply unused.
 */
function braceDepths(masked: string): number[] {
  const depths: number[] = [0];
  let depth = 0;
  for (let i = 0; i < masked.length; i++) {
    const ch = masked[i];
    if (ch === '{') depth += 1;
    else if (ch === '}') {
      if (depth > 0) depth -= 1;
    } else if (ch === '\n') depths.push(depth);
  }
  return depths;
}

interface ScopeTracker {
  /** Enclosing container name at `depth`, dropping scopes that have since closed. */
  at(depth: number): string | undefined;
  /** Record that `name`'s body starts at `bodyDepth` (normally depth + 1). */
  open(bodyDepth: number, name: string): void;
}

/**
 * Nearest enclosing container per brace depth.
 *
 * A definition at depth `d` is inside whichever container was opened for depth `d`,
 * because a container declared at depth `d-1` puts its body at depth `d`. Calling `at`
 * clears everything deeper than the line being examined — those scopes have closed — so
 * a class that ended three lines ago cannot parent a method of the next class.
 */
function createScopeTracker(): ScopeTracker {
  const containers: Array<string | undefined> = [];
  return {
    at(depth) {
      for (let k = depth + 1; k < containers.length; k++) containers[k] = undefined;
      return containers[depth];
    },
    open(bodyDepth, name) {
      containers[bodyDepth] = name;
    },
  };
}

/** First identifier in the file's likely path, used for `exported` decisions. */
function stripQuotes(s: string): string {
  const t = s.trim();
  if (t.length >= 2 && ((t.startsWith('"') && t.endsWith('"')) || (t.startsWith("'") && t.endsWith("'")))) {
    return t.slice(1, -1);
  }
  return t;
}

/**
 * The contents of the first quoted string on the line, read from `raw`.
 *
 * `masked` has the contents blanked but the quotes intact, so a regex can locate the
 * span there and the same indices are valid in `raw`. This is how an import specifier
 * survives masking at all.
 */
function quotedImport(maskedLine: string, rawLine: string): string | null {
  const open = /['"]/.exec(maskedLine);
  if (!open) return null;
  const quote = maskedLine[open.index];
  const close = maskedLine.indexOf(quote ?? '', open.index + 1);
  if (close <= open.index) return null;
  return rawLine.slice(open.index + 1, close);
}

function codeImport(re: RegExp): (maskedLine: string, rawLine: string, add: (spec: string) => void) => void {
  return (line, _raw, add) => {
    const m = re.exec(line);
    if (m && m[1] !== undefined) add(m[1]);
  };
}

/* ------------------------------------------------------------------ */
/* JavaScript / TypeScript                                             */
/* ------------------------------------------------------------------ */

/**
 * Words that appear where a method name would sit but are not definitions. Without this
 * a class body's `if (x) { ... }` would be indexed as a method called `if`.
 */
const CONTROL_WORDS = new Set([
  'if', 'for', 'while', 'switch', 'catch', 'return', 'typeof', 'new', 'await', 'throw',
  'do', 'else', 'super', 'this', 'function', 'in', 'of', 'case', 'delete', 'void', 'yield',
  'instanceof', 'sizeof', 'alignof', 'decltype', 'static_assert', 'operator', 'noexcept',
  'typeid', 'synchronized', 'assert', 'lock', 'using', 'foreach', 'fixed', 'unchecked',
  'checked', 'when', 'defer', 'go', 'select', 'range', 'try', 'const_cast', 'static_cast',
  'dynamic_cast', 'reinterpret_cast', 'sizeof', 'nameof', 'default', 'params', 'get', 'set',
]);

const JS_IMPORT_RE = /(?:\bfrom\s*|\bimport\s*|\brequire\s*\(\s*|\bimport\s*\(\s*)['"]([^'"]*)['"]/g;

/** Static imports, re-exports and `require`/dynamic `import` on one masked line. */
function collectJsImports(maskedLine: string, rawLine: string, add: (spec: string) => void): void {
  JS_IMPORT_RE.lastIndex = 0;
  let m: RegExpExecArray | null;
  while ((m = JS_IMPORT_RE.exec(maskedLine)) !== null) {
    const raw = rawLine.slice(m.index, m.index + m[0].length);
    const q = /['"]([^'"]*)['"]/.exec(raw);
    if (q && q[1] !== undefined) add(q[1]);
  }
}

function extractJsLike(masked: string, raw: string, options: { jsx: boolean }): Extraction {
  const maskedLines = masked.split('\n');
  const rawLines = raw.split('\n');
  const depths = braceDepths(masked);
  const scopes = createScopeTracker();
  const defs: SymbolDef[] = [];
  const imports: string[] = [];
  const seenImports = new Set<string>();

  const asKind = (name: string): SymbolKind => (options.jsx && /^[A-Z]/.test(name) ? 'component' : 'function');

  for (let i = 0; i < maskedLines.length; i++) {
    const line = maskedLines[i] ?? '';
    const rawLine = rawLines[i] ?? '';
    const depth = depths[i] ?? 0;

    collectJsImports(line, rawLine, (s) => addUnique(imports, seenImports, s));

    const exported = /^\s*export\b/.test(line);
    let m: RegExpExecArray | null;

    if ((m = /^\s*(?:export\s+)?(?:default\s+)?(?:declare\s+)?(?:abstract\s+)?class\s+([A-Za-z_$][\w$]*)/.exec(line))) {
      const name = m[1]!;
      const parent = scopes.at(depth);
      scopes.open(depth + 1, name);
      defs.push(makeDef(name, 'class', i, rawLines, parent, exported));
      continue;
    }
    if ((m = /^\s*(?:export\s+)?(?:declare\s+)?interface\s+([A-Za-z_$][\w$]*)/.exec(line))) {
      const name = m[1]!;
      const parent = scopes.at(depth);
      scopes.open(depth + 1, name);
      defs.push(makeDef(name, 'interface', i, rawLines, parent, exported));
      continue;
    }
    if ((m = /^\s*(?:export\s+)?(?:declare\s+)?type\s+([A-Za-z_$][\w$]*)\s*(?:<[^>]*>)?\s*=/.exec(line))) {
      const name = m[1]!;
      const parent = scopes.at(depth);
      if (line.includes('{')) scopes.open(depth + 1, name);
      defs.push(makeDef(name, 'type', i, rawLines, parent, exported));
      continue;
    }
    if ((m = /^\s*(?:export\s+)?(?:declare\s+)?(?:const\s+)?enum\s+([A-Za-z_$][\w$]*)/.exec(line))) {
      const name = m[1]!;
      const parent = scopes.at(depth);
      scopes.open(depth + 1, name);
      defs.push(makeDef(name, 'enum', i, rawLines, parent, exported));
      continue;
    }
    if ((m = /^\s*(?:export\s+)?(?:default\s+)?(?:declare\s+)?(?:async\s+)?function\s*\*?\s*([A-Za-z_$][\w$]*)/.exec(line))) {
      const name = m[1]!;
      defs.push(makeDef(name, asKind(name), i, rawLines, scopes.at(depth), exported));
      continue;
    }
    // Arrow functions and function expressions bound to a name. The alternatives are
    // anchored right after `=`, so `const x = useMemo(() => ...)` is a const, not a
    // function: the callback belongs to `useMemo`, not to `x`.
    if ((m = /^\s*(?:export\s+)?(?:default\s+)?(?:const|let|var)\s+([A-Za-z_$][\w$]*)\s*(?::[^=]+)?=\s*(?:async\s+)?(?:function\b|<[^>]*>\s*\(|\([^)]*\)\s*(?::[^=]+)?=>|[A-Za-z_$][\w$]*\s*=>)/.exec(line))) {
      const name = m[1]!;
      defs.push(makeDef(name, asKind(name), i, rawLines, scopes.at(depth), exported));
      continue;
    }
    if (depth > 0) {
      const parent = scopes.at(depth);
      if (parent !== undefined) {
        // Class fields assigned a function (`handleClick = () => {}`).
        if ((m = /^\s*(?:static\s+)?(?:readonly\s+)?([A-Za-z_$][\w$]*)\s*=\s*(?:async\s+)?(?:\([^)]*\)|[A-Za-z_$][\w$]*)\s*=>/.exec(line))) {
          defs.push(makeDef(m[1]!, 'method', i, rawLines, parent, exported));
          continue;
        }
        // Ordinary methods and constructors. Only considered directly inside a scope,
        // so call expressions in method bodies are never mistaken for declarations.
        if ((m = /^\s*(?:(?:public|private|protected|static|readonly|abstract|override|async|declare|export)\s+)*([A-Za-z_$][\w$]*)\s*(?:<[^>(]*>)?\s*\(/.exec(line))) {
          const name = m[1]!;
          if (!CONTROL_WORDS.has(name)) {
            defs.push(makeDef(name, 'method', i, rawLines, parent, exported));
            continue;
          }
        }
      }
    }
    // Module-level bindings. Only at depth 0: a `const` inside a function is a local and
    // would add a definition per statement to every file.
    if (depth === 0 && (m = /^\s*(?:export\s+)?(?:declare\s+)?(const|let|var)\s+([A-Za-z_$][\w$]*)/.exec(line))) {
      const kw = m[1]!;
      defs.push(makeDef(m[2]!, kw === 'const' ? 'const' : 'variable', i, rawLines, undefined, exported));
    }
  }

  return { defs, imports };
}

/* ------------------------------------------------------------------ */
/* Python                                                              */
/* ------------------------------------------------------------------ */

function extractPython(masked: string, raw: string): Extraction {
  const maskedLines = masked.split('\n');
  const rawLines = raw.split('\n');
  const stack: Array<{ indent: number; name: string; kind: SymbolKind }> = [];
  const defs: SymbolDef[] = [];
  const imports: string[] = [];
  const seenImports = new Set<string>();

  for (let i = 0; i < maskedLines.length; i++) {
    const line = maskedLines[i] ?? '';
    if (line.trim() === '') continue;
    const indent = indentationOf(line);
    let m: RegExpExecArray | null;

    if ((m = /^\s*from\s+([.\w]+)\s+import\s+/.exec(line))) {
      addUnique(imports, seenImports, m[1]!);
      continue;
    }
    if ((m = /^\s*import\s+([\w.]+(?:\s+as\s+\w+)?(?:,\s*[\w.]+(?:\s+as\s+\w+)?)*)/.exec(line))) {
      for (const part of m[1]!.split(',')) {
        addUnique(imports, seenImports, (part.trim().split(/\s+as\s+/)[0] ?? '').trim());
      }
      continue;
    }
    // Decorators carry no definition themselves; the def on the next line is the symbol.
    if (/^\s*@/.test(line)) continue;

    if ((m = /^\s*class\s+([A-Za-z_]\w*)/.exec(line))) {
      while (stack.length > 0 && stack[stack.length - 1]!.indent >= indent) stack.pop();
      const parent = stack.length > 0 ? stack[stack.length - 1]!.name : undefined;
      defs.push(makeDef(m[1]!, 'class', i, rawLines, parent, false));
      stack.push({ indent, name: m[1]!, kind: 'class' });
      continue;
    }
    if ((m = /^\s*(?:async\s+)?def\s+([A-Za-z_]\w*)/.exec(line))) {
      while (stack.length > 0 && stack[stack.length - 1]!.indent >= indent) stack.pop();
      const top = stack.length > 0 ? stack[stack.length - 1]! : undefined;
      const kind: SymbolKind = top?.kind === 'class' ? 'method' : 'function';
      defs.push(makeDef(m[1]!, kind, i, rawLines, top?.name, false));
      // Functions are pushed too, so a closure nested in a function reports that
      // function as its parent instead of jumping out to the enclosing class.
      stack.push({ indent, name: m[1]!, kind: 'function' });
      continue;
    }
    if (indent === 0 && (m = /^([A-Za-z_]\w*)\s*(?::[^=\n]+)?=(?!=)/.exec(line))) {
      const name = m[1]!;
      const kind: SymbolKind = /^[A-Z][A-Z0-9_]*$/.test(name) ? 'const' : 'variable';
      defs.push(makeDef(name, kind, i, rawLines, undefined, false));
    }
  }

  return { defs, imports };
}

/* ------------------------------------------------------------------ */
/* Go                                                                  */
/* ------------------------------------------------------------------ */

function extractGo(masked: string, raw: string): Extraction {
  const maskedLines = masked.split('\n');
  const rawLines = raw.split('\n');
  const depths = braceDepths(masked);
  const scopes = createScopeTracker();
  const defs: SymbolDef[] = [];
  const imports: string[] = [];
  const seenImports = new Set<string>();
  let mode: 'none' | 'import' | 'const' | 'var' | 'type' = 'none';

  const isExported = (name: string): boolean => /^[A-Z]/.test(name);

  for (let i = 0; i < maskedLines.length; i++) {
    const line = maskedLines[i] ?? '';
    const rawLine = rawLines[i] ?? '';
    const depth = depths[i] ?? 0;
    const trimmed = line.trim();

    if (mode !== 'none') {
      if (trimmed.startsWith(')')) {
        mode = 'none';
        continue;
      }
      if (mode === 'import') {
        const spec = quotedImport(line, rawLine);
        if (spec !== null) addUnique(imports, seenImports, spec);
        continue;
      }
      const gm = /^([A-Za-z_]\w*)(?:\s+(struct|interface))?/.exec(trimmed);
      if (gm && gm[1] !== undefined) {
        const kind: SymbolKind = gm[2] === 'struct' ? 'struct' : gm[2] === 'interface' ? 'interface' : mode === 'type' ? 'type' : mode === 'const' ? 'const' : 'variable';
        const parent = scopes.at(depth);
        if (kind === 'struct' || kind === 'interface') scopes.open(depth + 1, gm[1]);
        defs.push(makeDef(gm[1], kind, i, rawLines, parent, isExported(gm[1])));
      }
      continue;
    }

    let m: RegExpExecArray | null;
    if (/^\s*import\s*\(/.test(line)) {
      mode = 'import';
      continue;
    }
    if (/^\s*import\s/.test(line)) {
      const spec = quotedImport(line, rawLine);
      if (spec !== null) addUnique(imports, seenImports, spec);
      continue;
    }
    if (/^\s*const\s*\(/.test(line)) {
      mode = 'const';
      continue;
    }
    if (/^\s*var\s*\(/.test(line)) {
      mode = 'var';
      continue;
    }
    if (/^\s*type\s*\(/.test(line)) {
      mode = 'type';
      continue;
    }
    if ((m = /^\s*type\s+([A-Za-z_]\w*)\s+struct\b/.exec(line))) {
      const parent = scopes.at(depth);
      scopes.open(depth + 1, m[1]!);
      defs.push(makeDef(m[1]!, 'struct', i, rawLines, parent, isExported(m[1]!)));
      continue;
    }
    if ((m = /^\s*type\s+([A-Za-z_]\w*)\s+interface\b/.exec(line))) {
      const parent = scopes.at(depth);
      scopes.open(depth + 1, m[1]!);
      defs.push(makeDef(m[1]!, 'interface', i, rawLines, parent, isExported(m[1]!)));
      continue;
    }
    if ((m = /^\s*type\s+([A-Za-z_]\w*)/.exec(line))) {
      defs.push(makeDef(m[1]!, 'type', i, rawLines, scopes.at(depth), isExported(m[1]!)));
      continue;
    }
    // A receiver turns the declaration into a method and names its parent directly;
    // Go methods live at file scope, so brace nesting would never reveal this.
    if ((m = /^\s*func\s*\(\s*(?:[A-Za-z_]\w*\s+)?\*?([A-Za-z_]\w*)\s*\)\s*([A-Za-z_]\w*)/.exec(line))) {
      defs.push(makeDef(m[2]!, 'method', i, rawLines, m[1]!, isExported(m[2]!)));
      continue;
    }
    if ((m = /^\s*func\s+([A-Za-z_]\w*)/.exec(line))) {
      defs.push(makeDef(m[1]!, 'function', i, rawLines, undefined, isExported(m[1]!)));
      continue;
    }
    if ((m = /^\s*const\s+([A-Za-z_]\w*)/.exec(line))) {
      defs.push(makeDef(m[1]!, 'const', i, rawLines, scopes.at(depth), isExported(m[1]!)));
      continue;
    }
    if ((m = /^\s*var\s+([A-Za-z_]\w*)/.exec(line))) {
      defs.push(makeDef(m[1]!, 'variable', i, rawLines, scopes.at(depth), isExported(m[1]!)));
    }
  }

  return { defs, imports };
}

/* ------------------------------------------------------------------ */
/* Rust                                                                */
/* ------------------------------------------------------------------ */

const RUST_PUB = /^\s*pub\b/;

/**
 * `use a::{b, c::D, self};` becomes `a::b`, `a::c::D`, `a`. The graph wants the module
 * the file depends on, so expanding a grouped import keeps edges that a bare `a` loses.
 */
function rustUsePaths(spec: string): string[] {
  const text = spec.trim().replace(/\s+/g, '');
  const brace = text.indexOf('{');
  if (brace === -1) return text === '' ? [] : [text.replace(/;$/, '')];
  const prefix = text.slice(0, brace).replace(/::$/, '');
  const inner = text.slice(brace + 1).replace(/\}.*$/, '');
  const out: string[] = [];
  for (const raw of inner.split(',')) {
    const part = raw.trim();
    if (part === '') continue;
    if (part === 'self' || part.startsWith('self::')) out.push(prefix);
    else out.push(prefix === '' ? part : `${prefix}::${part}`);
  }
  return out;
}

function extractRust(masked: string, raw: string): Extraction {
  const maskedLines = masked.split('\n');
  const rawLines = raw.split('\n');
  const depths = braceDepths(masked);
  const scopes = createScopeTracker();
  const defs: SymbolDef[] = [];
  const imports: string[] = [];
  const seenImports = new Set<string>();

  for (let i = 0; i < maskedLines.length; i++) {
    const line = maskedLines[i] ?? '';
    const depth = depths[i] ?? 0;
    const exported = RUST_PUB.test(line);
    let m: RegExpExecArray | null;

    if ((m = /^\s*(?:pub\s+)?use\s+([^;]+)/.exec(line))) {
      for (const p of rustUsePaths(m[1]!)) addUnique(imports, seenImports, p);
      continue;
    }
    // `const fn` and `async fn` are functions; ordering this before the const rule keeps
    // `const fn` from being read as a constant named `fn`.
    if ((m = /^\s*(?:pub(?:\([^)]*\))?\s+)?(?:(?:async|unsafe|const|extern(?:\s+"[^"]*")?)\s+)*fn\s+([A-Za-z_]\w*)/.exec(line))) {
      const parent = scopes.at(depth);
      defs.push(makeDef(m[1]!, parent === undefined ? 'function' : 'method', i, rawLines, parent, exported));
      continue;
    }
    // `impl` is not itself a symbol, but it names the scope its methods belong to. For
    // `impl Trait for Type` the interesting parent is Type, not Trait.
    if ((m = /^\s*(?:pub\s+)?impl(?:\s*<[^>]*>)?\s+(?:[A-Za-z_]\w*(?:::\w+)*(?:<[^>]*>)?\s+for\s+)?([A-Za-z_]\w*(?:::\w+)*)/.exec(line))) {
      scopes.open(depth + 1, m[1]!);
      continue;
    }
    if ((m = /^\s*(?:pub(?:\([^)]*\))?\s+)?struct\s+([A-Za-z_]\w*)/.exec(line))) {
      const parent = scopes.at(depth);
      scopes.open(depth + 1, m[1]!);
      defs.push(makeDef(m[1]!, 'struct', i, rawLines, parent, exported));
      continue;
    }
    if ((m = /^\s*(?:pub(?:\([^)]*\))?\s+)?enum\s+([A-Za-z_]\w*)/.exec(line))) {
      const parent = scopes.at(depth);
      scopes.open(depth + 1, m[1]!);
      defs.push(makeDef(m[1]!, 'enum', i, rawLines, parent, exported));
      continue;
    }
    if ((m = /^\s*(?:pub(?:\([^)]*\))?\s+)?trait\s+([A-Za-z_]\w*)/.exec(line))) {
      const parent = scopes.at(depth);
      scopes.open(depth + 1, m[1]!);
      defs.push(makeDef(m[1]!, 'trait', i, rawLines, parent, exported));
      continue;
    }
    if ((m = /^\s*(?:pub(?:\([^)]*\))?\s+)?mod\s+([A-Za-z_]\w*)/.exec(line))) {
      const parent = scopes.at(depth);
      scopes.open(depth + 1, m[1]!);
      defs.push(makeDef(m[1]!, 'module', i, rawLines, parent, exported));
      continue;
    }
    if ((m = /^\s*macro_rules!\s*([A-Za-z_]\w*)/.exec(line))) {
      defs.push(makeDef(m[1]!, 'macro', i, rawLines, scopes.at(depth), exported));
      continue;
    }
    if ((m = /^\s*(?:pub(?:\([^)]*\))?\s+)?type\s+([A-Za-z_]\w*)/.exec(line))) {
      defs.push(makeDef(m[1]!, 'type', i, rawLines, scopes.at(depth), exported));
      continue;
    }
    if ((m = /^\s*(?:pub(?:\([^)]*\))?\s+)?(?:const|static)\s+(?:mut\s+)?([A-Za-z_]\w*)/.exec(line))) {
      defs.push(makeDef(m[1]!, 'const', i, rawLines, scopes.at(depth), exported));
    }
  }

  return { defs, imports };
}

/* ------------------------------------------------------------------ */
/* Generic brace-and-keyword driver (Java, C#, Kotlin, Swift, Scala, PHP) */
/* ------------------------------------------------------------------ */

interface DefHit {
  name: string;
  kind: SymbolKind;
  /** The definition opens a body one brace deeper (class, struct, module...). */
  container?: boolean;
}

type ImportSink = (spec: string) => void;

interface BraceRules {
  decls: Array<(line: string, depth: number) => DefHit | null>;
  /** Methods and other members, considered only where a named scope encloses the line. */
  member?: (line: string, depth: number, parent: string) => DefHit | null;
  exported?: (line: string, name: string) => boolean;
  /** Promote a `function` declaration to `method` when it sits inside a scope. */
  methodsNeedScope?: boolean;
  imports?: (maskedLine: string, rawLine: string, add: ImportSink) => void;
}

function declRule(re: RegExp, kind: SymbolKind, container = false): (line: string, depth: number) => DefHit | null {
  return (line) => {
    const m = re.exec(line);
    if (!m || m[1] === undefined) return null;
    const hit: DefHit = { name: m[1], kind };
    if (container) hit.container = true;
    return hit;
  };
}

function extractBrace(masked: string, raw: string, rules: BraceRules): Extraction {
  const maskedLines = masked.split('\n');
  const rawLines = raw.split('\n');
  const depths = braceDepths(masked);
  const scopes = createScopeTracker();
  const defs: SymbolDef[] = [];
  const imports: string[] = [];
  const seenImports = new Set<string>();
  const add: ImportSink = (s) => addUnique(imports, seenImports, s);

  for (let i = 0; i < maskedLines.length; i++) {
    const line = maskedLines[i] ?? '';
    const rawLine = rawLines[i] ?? '';
    const depth = depths[i] ?? 0;

    if (rules.imports) rules.imports(line, rawLine, add);

    let hit: DefHit | null = null;
    for (const rule of rules.decls) {
      hit = rule(line, depth);
      if (hit) break;
    }
    if (hit) {
      const parent = scopes.at(depth);
      if (hit.container === true) scopes.open(depth + 1, hit.name);
      let kind = hit.kind;
      if (kind === 'function' && rules.methodsNeedScope === true && parent !== undefined) kind = 'method';
      defs.push(makeDef(hit.name, kind, i, rawLines, parent, rules.exported ? rules.exported(line, hit.name) : false));
      continue;
    }
    if (rules.member && depth > 0) {
      const parent = scopes.at(depth);
      if (parent !== undefined) {
        const member = rules.member(line, depth, parent);
        if (member) {
          if (member.container === true) scopes.open(depth + 1, member.name);
          defs.push(makeDef(member.name, member.kind, i, rawLines, parent, rules.exported ? rules.exported(line, member.name) : false));
        }
      }
    }
  }

  return { defs, imports };
}

/**
 * The identifier immediately before the first `(`, or null when there is not one.
 *
 * Scanning back from the parenthesis (rather than forward from the line start) is what
 * makes this work for the return-type-first declaration styles of Java, C# and Swift
 * without needing to understand the type at all.
 */
function identifierBeforeFirstParen(line: string): string | null {
  const open = line.indexOf('(');
  if (open < 0) return null;
  let end = open;
  while (end > 0 && (line[end - 1] === ' ' || line[end - 1] === '\t')) end -= 1;
  let start = end;
  while (start > 0 && /[A-Za-z0-9_$]/.test(line[start - 1]!)) start -= 1;
  const word = line.slice(start, end);
  return /^[A-Za-z_$][\w$]*$/.test(word) ? word : null;
}

/**
 * True when everything after the first `)` looks like the tail of a declaration
 * (`;`, `{`, `const`, `noexcept`, `override`, `-> Type`, an initializer list) rather than
 * the tail of an expression or a templated field. Without this, `std::function<void()>`
 * as a field type would be read as a method named `void`.
 */
function declarationTail(line: string): boolean {
  const close = line.indexOf(')');
  if (close < 0) return false;
  let rest = line.slice(close + 1);
  // An inline body (`int speed() const { return x; }`) sits between the parameter list
  // and the end of the line; only the part before the brace is part of the declaration.
  const brace = rest.indexOf('{');
  if (brace >= 0) rest = rest.slice(0, brace);
  rest = rest.trim().replace(/[;{}]+\s*$/, '').trim();
  if (rest === '') return true;
  return /^(?:(?:const|noexcept(?:\([^)]*\))?|override|final|mutable|&|&&|->\s*[\w:<>*&,\s]+|:\s*[\w(),\s.<>*&]+)\s*)+$/.test(rest);
}

function cLikeMember(line: string): DefHit | null {
  const trimmed = line.trim();
  if (trimmed === '' || trimmed.startsWith('@') || trimmed.startsWith('[') || trimmed.startsWith('}') || trimmed.startsWith('*')) {
    return null;
  }
  const open = line.indexOf('(');
  if (open < 0) return null;
  // A value on the left of the parenthesis (`int x = compute()`) is a field, not a method.
  if (line.slice(0, open).includes('=')) return null;
  if (!declarationTail(line)) return null;
  const name = identifierBeforeFirstParen(line);
  if (name === null || CONTROL_WORDS.has(name)) return null;
  return { name, kind: 'method' };
}

const JAVA_MODS = '(?:(?:public|protected|private|abstract|final|static|sealed|non-sealed|strictfp|native|synchronized|transient|volatile)\\s+)*';

const javaRules: BraceRules = {
  decls: [
    declRule(new RegExp(`^\\s*${JAVA_MODS}class\\s+([A-Za-z_$]\\w*)`), 'class', true),
    declRule(new RegExp(`^\\s*${JAVA_MODS}interface\\s+([A-Za-z_$]\\w*)`), 'interface', true),
    declRule(new RegExp(`^\\s*${JAVA_MODS}enum\\s+([A-Za-z_$]\\w*)`), 'enum', true),
    declRule(new RegExp(`^\\s*${JAVA_MODS}record\\s+([A-Za-z_$]\\w*)`), 'class', true),
  ],
  member: (line) => cLikeMember(line),
  exported: (line) => /^\s*public\b/.test(line),
  imports: codeImport(/^\s*import\s+(?:static\s+)?([\w.*]+)\s*;/),
};

const CSHARP_MODS = '(?:(?:public|private|protected|internal|static|readonly|sealed|abstract|virtual|override|partial|async|unsafe|extern|new|const)\\s+)*';

const csharpRules: BraceRules = {
  decls: [
    declRule(new RegExp(`^\\s*${CSHARP_MODS}namespace\\s+([A-Za-z_]\\w*)`), 'module', true),
    declRule(new RegExp(`^\\s*${CSHARP_MODS}class\\s+([A-Za-z_]\\w*)`), 'class', true),
    declRule(new RegExp(`^\\s*${CSHARP_MODS}interface\\s+([A-Za-z_]\\w*)`), 'interface', true),
    declRule(new RegExp(`^\\s*${CSHARP_MODS}struct\\s+([A-Za-z_]\\w*)`), 'struct', true),
    declRule(new RegExp(`^\\s*${CSHARP_MODS}enum\\s+([A-Za-z_]\\w*)`), 'enum', true),
    declRule(new RegExp(`^\\s*${CSHARP_MODS}record\\s+([A-Za-z_]\\w*)`), 'class', true),
  ],
  member: (line) => cLikeMember(line),
  exported: (line) => /^\s*public\b/.test(line),
  imports: codeImport(/^\s*(?:global\s+)?using\s+(?:static\s+)?([\w.]+)\s*;/),
};

const KOTLIN_MODS = '(?:(?:public|private|protected|internal|open|abstract|sealed|data|enum|annotation|value|inline|inner|companion|expect|actual|final|override|suspend|operator|infix|tailrec|external|const|lateinit|crossinline|noinline|vararg|reified)\\s+)*';

const kotlinRules: BraceRules = {
  decls: [
    declRule(new RegExp(`^\\s*${KOTLIN_MODS}interface\\s+([A-Za-z_]\\w*)`), 'interface', true),
    declRule(new RegExp(`^\\s*${KOTLIN_MODS}(?:class|object)\\s+([A-Za-z_]\\w*)`), 'class', true),
    declRule(new RegExp(`^\\s*${KOTLIN_MODS}fun\\s+(?:<[^>]*>\\s*)?(?:[\\w.<>?]+\\.)?([A-Za-z_]\\w*)\\s*\\(`), 'function'),
    declRule(/^\s*(?:(?:public|private|protected|internal|override|const|lateinit|final|open)\s+)*(?:const\s+)?val\s+([A-Za-z_]\w*)/, 'const'),
    declRule(/^\s*(?:(?:public|private|protected|internal|override|lateinit|final|open)\s+)*var\s+([A-Za-z_]\w*)/, 'variable'),
    declRule(/^\s*typealias\s+([A-Za-z_]\w*)/, 'type'),
  ],
  methodsNeedScope: true,
  exported: (line) => !/^\s*(?:private|protected|internal)\b/.test(line),
  imports: codeImport(/^\s*import\s+([\w.]+(?:\.\*)?)/),
};

const SWIFT_MODS = '(?:(?:public|private|internal|fileprivate|open|final|static|class|mutating|nonmutating|override|convenience|required|indirect|lazy|weak|unowned|dynamic|optional|nonisolated)\\s+|@\\w+(?:\\([^)]*\\))?\\s+)*';

const swiftRules: BraceRules = {
  decls: [
    // `func` first, because `class func` is a type method and would otherwise be read as
    // a class declaration named `func`.
    declRule(new RegExp(`^\\s*${SWIFT_MODS}func\\s+([A-Za-z_]\\w*)`), 'function'),
    declRule(new RegExp(`^\\s*${SWIFT_MODS}(?:class|actor)\\s+([A-Za-z_]\\w*)`), 'class', true),
    declRule(new RegExp(`^\\s*${SWIFT_MODS}struct\\s+([A-Za-z_]\\w*)`), 'struct', true),
    declRule(new RegExp(`^\\s*${SWIFT_MODS}enum\\s+([A-Za-z_]\\w*)`), 'enum', true),
    declRule(new RegExp(`^\\s*${SWIFT_MODS}protocol\\s+([A-Za-z_]\\w*)`), 'interface', true),
    declRule(/^\s*extension\s+([A-Za-z_]\w*)/, 'module', true),
    declRule(new RegExp(`^\\s*${SWIFT_MODS}typealias\\s+([A-Za-z_]\\w*)`), 'type'),
    declRule(new RegExp(`^\\s*${SWIFT_MODS}let\\s+([A-Za-z_]\\w*)`), 'const'),
    declRule(new RegExp(`^\\s*${SWIFT_MODS}var\\s+([A-Za-z_]\\w*)`), 'variable'),
  ],
  methodsNeedScope: true,
  exported: (line) => /^\s*(?:public|open)\b/.test(line),
  imports: codeImport(/^\s*import\s+([\w.]+)/),
};

const SCALA_MODS = '(?:(?:private|protected|override|implicit|final|lazy|sealed|abstract|case)\\s+)*';

const scalaRules: BraceRules = {
  decls: [
    declRule(new RegExp(`^\\s*${SCALA_MODS}case\\s+class\\s+([A-Za-z_]\\w*)`), 'class', true),
    declRule(new RegExp(`^\\s*${SCALA_MODS}(?:class|object)\\s+([A-Za-z_]\\w*)`), 'class', true),
    declRule(new RegExp(`^\\s*${SCALA_MODS}trait\\s+([A-Za-z_]\\w*)`), 'trait', true),
    declRule(new RegExp(`^\\s*${SCALA_MODS}def\\s+([A-Za-z_]\\w*)`), 'function'),
    declRule(new RegExp(`^\\s*${SCALA_MODS}(?:val|var)\\s+([A-Za-z_]\\w*)`), 'const'),
    declRule(new RegExp(`^\\s*${SCALA_MODS}type\\s+([A-Za-z_]\\w*)`), 'type'),
    declRule(/^\s*package\s+([A-Za-z_][\w.]*)/, 'module', true),
  ],
  methodsNeedScope: true,
  exported: (line) => !/^\s*(?:private|protected)\b/.test(line),
  imports: codeImport(/^\s*import\s+([\w.]+)/),
};

function phpImports(maskedLine: string, rawLine: string, add: ImportSink): void {
  const use = /^\s*use\s+(?:function\s+|const\s+)?([\w\\]+)/.exec(maskedLine);
  if (use && use[1] !== undefined) {
    add(use[1]);
    return;
  }
  if (/^\s*(?:require|include)(?:_once)?\s*\(?\s*['"]/.test(maskedLine)) {
    const spec = quotedImport(maskedLine, rawLine);
    if (spec !== null) add(spec);
  }
}

const PHP_MODS = '(?:(?:public|private|protected|static|abstract|final|readonly)\\s+)*';

const phpRules: BraceRules = {
  decls: [
    declRule(/^\s*namespace\s+([\w\\]+)/, 'module', true),
    declRule(new RegExp(`^\\s*${PHP_MODS}interface\\s+([A-Za-z_]\\w*)`), 'interface', true),
    declRule(new RegExp(`^\\s*${PHP_MODS}trait\\s+([A-Za-z_]\\w*)`), 'trait', true),
    declRule(new RegExp(`^\\s*${PHP_MODS}(?:class|enum)\\s+([A-Za-z_]\\w*)`), 'class', true),
    declRule(new RegExp(`^\\s*${PHP_MODS}function\\s+&?\\s*([A-Za-z_]\\w*)`), 'function'),
    declRule(new RegExp(`^\\s*${PHP_MODS}const\\s+([A-Za-z_]\\w*)`), 'const'),
    // `$name = ...` at file scope. Class properties sit at depth 1 and local variables
    // deeper still, so restricting to depth 0 keeps locals out of the index.
    (line, depth) => {
      if (depth !== 0) return null;
      const m = /^\s*(?:global\s+)?\$([A-Za-z_]\w*)\s*=/.exec(line);
      return m && m[1] !== undefined ? { name: m[1], kind: 'variable' } : null;
    },
  ],
  methodsNeedScope: true,
  exported: (line) => /^\s*public\b/.test(line),
  imports: phpImports,
};

function extractJava(masked: string, raw: string): Extraction {
  return extractBrace(masked, raw, javaRules);
}
function extractCSharp(masked: string, raw: string): Extraction {
  return extractBrace(masked, raw, csharpRules);
}
function extractKotlin(masked: string, raw: string): Extraction {
  return extractBrace(masked, raw, kotlinRules);
}
function extractSwift(masked: string, raw: string): Extraction {
  return extractBrace(masked, raw, swiftRules);
}
function extractScala(masked: string, raw: string): Extraction {
  return extractBrace(masked, raw, scalaRules);
}
function extractPhp(masked: string, raw: string): Extraction {
  return extractBrace(masked, raw, phpRules);
}

/* ------------------------------------------------------------------ */
/* C and C++                                                           */
/* ------------------------------------------------------------------ */

/** The qualifier in `void Foo::bar()`, so out-of-line methods name their class. */
function qualifierBefore(line: string): string | undefined {
  const open = line.indexOf('(');
  if (open < 0) return undefined;
  let end = open;
  while (end > 0 && (line[end - 1] === ' ' || line[end - 1] === '\t')) end -= 1;
  let start = end;
  while (start > 0 && /[A-Za-z0-9_$]/.test(line[start - 1]!)) start -= 1;
  let colon = start;
  while (colon > 0 && (line[colon - 1] === ' ' || line[colon - 1] === '\t')) colon -= 1;
  if (colon >= 2 && line[colon - 1] === ':' && line[colon - 2] === ':') {
    let qStart = colon - 2;
    while (qStart > 0 && /[A-Za-z0-9_$]/.test(line[qStart - 1]!)) qStart -= 1;
    const q = line.slice(qStart, colon - 2);
    return q === '' ? undefined : q;
  }
  return undefined;
}

function extractCLike(masked: string, raw: string, cpp: boolean): Extraction {
  const maskedLines = masked.split('\n');
  const rawLines = raw.split('\n');
  const depths = braceDepths(masked);
  const scopes = createScopeTracker();
  const defs: SymbolDef[] = [];
  const imports: string[] = [];
  const seenImports = new Set<string>();
  let pending: { name: string; line: number; parent: string | undefined } | null = null;

  for (let i = 0; i < maskedLines.length; i++) {
    const line = maskedLines[i] ?? '';
    const rawLine = rawLines[i] ?? '';
    const depth = depths[i] ?? 0;
    let m: RegExpExecArray | null;

    // A definition whose body brace opens on the next line (Allman style). The held
    // candidate is only ever completed by an immediate `{`; anything else drops it.
    if (pending !== null) {
      const held = pending;
      pending = null;
      if (/^\s*\{/.test(line)) {
        defs.push(makeDef(held.name, held.parent === undefined ? 'function' : 'method', held.line, rawLines, held.parent, false));
        continue;
      }
    }

    // `#include` is code, so the keyword survives masking; the header is a string (or an
    // angle-bracket name) and is read from the raw line at the same offset.
    const inc = /^\s*#\s*include\s*/.exec(line);
    if (inc) {
      const rest = rawLine.slice(inc[0].length);
      const spec = /^(?:<([^>]*)>|"([^"]*)")/.exec(rest);
      if (spec) addUnique(imports, seenImports, spec[1] ?? spec[2] ?? '');
      continue;
    }
    if ((m = /^\s*#\s*define\s+([A-Za-z_]\w*)/.exec(line))) {
      defs.push(makeDef(m[1]!, 'macro', i, rawLines, undefined, false));
      continue;
    }
    if (/^\s*#/.test(line)) continue; // other preprocessor directives are not symbols

    if ((m = /^\s*typedef\s+struct\s+([A-Za-z_]\w*)\s*\{/.exec(line))) {
      const parent = scopes.at(depth);
      scopes.open(depth + 1, m[1]!);
      defs.push(makeDef(m[1]!, 'struct', i, rawLines, parent, false));
      continue;
    }
    if ((m = /^\s*(?:struct|union)\s+([A-Za-z_]\w*)\s*\{?/.exec(line))) {
      const parent = scopes.at(depth);
      scopes.open(depth + 1, m[1]!);
      defs.push(makeDef(m[1]!, 'struct', i, rawLines, parent, false));
      continue;
    }
    if ((m = /^\s*typedef\s+enum\s+([A-Za-z_]\w*)\s*\{/.exec(line))) {
      const parent = scopes.at(depth);
      scopes.open(depth + 1, m[1]!);
      defs.push(makeDef(m[1]!, 'enum', i, rawLines, parent, false));
      continue;
    }
    if ((m = /^\s*enum\s+(?:class\s+)?([A-Za-z_]\w*)/.exec(line))) {
      const parent = scopes.at(depth);
      scopes.open(depth + 1, m[1]!);
      defs.push(makeDef(m[1]!, 'enum', i, rawLines, parent, false));
      continue;
    }
    if (cpp && (m = /^\s*(?:template\s*<[^>]*>\s*)?class\s+([A-Za-z_]\w*)/.exec(line))) {
      const parent = scopes.at(depth);
      scopes.open(depth + 1, m[1]!);
      defs.push(makeDef(m[1]!, 'class', i, rawLines, parent, false));
      continue;
    }
    if ((m = /^\s*namespace\s+([A-Za-z_]\w*)/.exec(line))) {
      const parent = scopes.at(depth);
      scopes.open(depth + 1, m[1]!);
      defs.push(makeDef(m[1]!, 'module', i, rawLines, parent, false));
      continue;
    }
    // `} Name;` closes a `typedef struct { ... } Name;`, whose name never appeared on
    // the opening line. Treated as a type rather than a struct, which is what callers use.
    if ((m = /^\s*\}\s*([A-Za-z_]\w*)\s*;/.exec(line))) {
      defs.push(makeDef(m[1]!, 'type', i, rawLines, undefined, false));
      continue;
    }
    if (!line.includes('{') && (m = /^\s*typedef\s+[^;{]*?([A-Za-z_]\w*)\s*;/.exec(line))) {
      defs.push(makeDef(m[1]!, 'type', i, rawLines, undefined, false));
      continue;
    }

    if (!line.includes('(') || !declarationTail(line)) continue;
    const name = identifierBeforeFirstParen(line);
    if (name === null || CONTROL_WORDS.has(name)) continue;

    if (depth > 0) {
      const enclosing = scopes.at(depth);
      if (enclosing === undefined) continue; // inside a body: a call is not a definition
      // `void Engine::start() {}` written inside a namespace names its class directly;
      // without this the parent would be the namespace.
      const parent = qualifierBefore(line) ?? enclosing;
      if (line.includes('{') || /;\s*$/.test(line.trim())) {
        defs.push(makeDef(name, 'method', i, rawLines, parent, false));
      } else {
        pending = { name, line: i, parent };
      }
      continue;
    }
    if (line.includes('{')) {
      const qualifier = qualifierBefore(line);
      defs.push(makeDef(name, qualifier === undefined ? 'function' : 'method', i, rawLines, qualifier, false));
    } else if (!line.trim().endsWith(';')) {
      pending = { name, line: i, parent: qualifierBefore(line) };
    }
  }

  return { defs, imports };
}

/* ------------------------------------------------------------------ */
/* Ruby                                                                */
/* ------------------------------------------------------------------ */

function extractRuby(masked: string, raw: string): Extraction {
  const maskedLines = masked.split('\n');
  const rawLines = raw.split('\n');
  const stack: Array<{ indent: number; name: string }> = [];
  const defs: SymbolDef[] = [];
  const imports: string[] = [];
  const seenImports = new Set<string>();

  for (let i = 0; i < maskedLines.length; i++) {
    const line = maskedLines[i] ?? '';
    const rawLine = rawLines[i] ?? '';
    const indent = indentationOf(line);
    let m: RegExpExecArray | null;

    if ((m = /^\s*(?:require|require_relative|load)\s*\(?\s*['"]/.exec(line))) {
      const spec = quotedImport(line, rawLine);
      if (spec !== null) addUnique(imports, seenImports, spec);
      continue;
    }
    if ((m = /^\s*(?:class|module)\s+([A-Za-z_]\w*(?:::[A-Za-z_]\w*)*)/.exec(line))) {
      while (stack.length > 0 && stack[stack.length - 1]!.indent >= indent) stack.pop();
      const parent = stack.length > 0 ? stack[stack.length - 1]!.name : undefined;
      const kind: SymbolKind = /^\s*module\b/.test(line) ? 'module' : 'class';
      defs.push(makeDef(m[1]!, kind, i, rawLines, parent, false));
      stack.push({ indent, name: m[1]! });
      continue;
    }
    if ((m = /^\s*def\s+(?:self\.)?([A-Za-z_]\w*[?!=]?)/.exec(line))) {
      while (stack.length > 0 && stack[stack.length - 1]!.indent >= indent) stack.pop();
      const parent = stack.length > 0 ? stack[stack.length - 1]!.name : undefined;
      defs.push(makeDef(m[1]!, parent === undefined ? 'function' : 'method', i, rawLines, parent, false));
      continue;
    }
    if (indent === 0 && (m = /^([A-Z]\w*)\s*=/.exec(line))) {
      defs.push(makeDef(m[1]!, 'const', i, rawLines, undefined, false));
    }
  }

  return { defs, imports };
}

/* ------------------------------------------------------------------ */
/* Shell                                                               */
/* ------------------------------------------------------------------ */

function extractShell(masked: string, raw: string): Extraction {
  const maskedLines = masked.split('\n');
  const rawLines = raw.split('\n');
  const defs: SymbolDef[] = [];
  const imports: string[] = [];
  const seenImports = new Set<string>();

  for (let i = 0; i < maskedLines.length; i++) {
    const line = maskedLines[i] ?? '';
    const rawLine = rawLines[i] ?? '';
    const exported = /^\s*export\b/.test(line);
    let m: RegExpExecArray | null;

    m = /^\s*function\s+([A-Za-z_][\w-]*)/.exec(line);
    if (m === null) m = /^\s*([A-Za-z_][\w-]*)\s*\(\s*\)/.exec(line);
    if (m) {
      defs.push(makeDef(m[1]!, 'function', i, rawLines, undefined, exported));
      continue;
    }
    if ((m = /^\s*(?:source|\.)\s+(\S+)/.exec(line))) {
      const spec = quotedImport(line, rawLine) ?? stripQuotes(m[1]!);
      addUnique(imports, seenImports, spec);
      continue;
    }
    if ((m = /^\s*(?:export\s+)?([A-Za-z_]\w*)=(?!=)/.exec(line))) {
      defs.push(makeDef(m[1]!, 'variable', i, rawLines, undefined, exported));
      continue;
    }
    if ((m = /^\s*(?:export|declare|local|typeset|readonly)\s+(?:-\w+\s+)*([A-Za-z_]\w*)/.exec(line))) {
      defs.push(makeDef(m[1]!, 'variable', i, rawLines, undefined, exported));
    }
  }

  return { defs, imports };
}

/* ------------------------------------------------------------------ */
/* SQL                                                                 */
/* ------------------------------------------------------------------ */

const SQL_KIND: Record<string, SymbolKind> = {
  table: 'struct',
  view: 'struct',
  function: 'function',
  procedure: 'function',
  trigger: 'function',
  schema: 'module',
  database: 'module',
  type: 'type',
  domain: 'type',
  index: 'const',
  sequence: 'const',
};

function extractSql(masked: string, raw: string): Extraction {
  const maskedLines = masked.split('\n');
  const rawLines = raw.split('\n');
  const defs: SymbolDef[] = [];

  for (let i = 0; i < maskedLines.length; i++) {
    const line = maskedLines[i] ?? '';
    // The guard is on the *masked* line, so a `-- CREATE TABLE` comment or a CREATE
    // inside a block comment cannot reach the name regex.
    if (!/^\s*create\b/i.test(line)) continue;
    const m = /^\s*create\s+(?:or\s+replace\s+)?(?:global\s+|local\s+|temp(?:orary)?\s+|unique\s+)*(table|materialized\s+view|view|function|procedure|trigger|schema|database|type|domain|index|sequence)\s+(?:if\s+not\s+exists\s+)?([`"[]?[A-Za-z_][\w.$]*[`"\]]?(?:\.[`"[]?[A-Za-z_][\w$]*[`"\]]?)*)/i.exec(rawLines[i] ?? '');
    if (!m || m[1] === undefined || m[2] === undefined) continue;
    const key = m[1].toLowerCase().replace(/\s+/g, ' ');
    const name = m[2].replace(/[`"[\]]/g, '');
    defs.push(makeDef(name, SQL_KIND[key] ?? 'const', i, rawLines, undefined, false));
  }

  return { defs, imports: [] };
}

/* ------------------------------------------------------------------ */
/* Lua                                                                 */
/* ------------------------------------------------------------------ */

function extractLua(masked: string, raw: string): Extraction {
  const maskedLines = masked.split('\n');
  const rawLines = raw.split('\n');
  const defs: SymbolDef[] = [];
  const imports: string[] = [];
  const seenImports = new Set<string>();

  for (let i = 0; i < maskedLines.length; i++) {
    const line = maskedLines[i] ?? '';
    const rawLine = rawLines[i] ?? '';
    let m: RegExpExecArray | null;

    if (/\brequire\s*\(/.test(line)) {
      const spec = quotedImport(line, rawLine);
      if (spec !== null) addUnique(imports, seenImports, spec);
    }
    if ((m = /^\s*(?:local\s+)?function\s+([A-Za-z_]\w*(?:[.:][A-Za-z_]\w*)*)/.exec(line))) {
      const full = m[1]!;
      const sep = Math.max(full.lastIndexOf('.'), full.lastIndexOf(':'));
      if (sep >= 0) defs.push(makeDef(full.slice(sep + 1), 'method', i, rawLines, full.slice(0, sep), false));
      else defs.push(makeDef(full, 'function', i, rawLines, undefined, false));
      continue;
    }
    if ((m = /^\s*local\s+([A-Za-z_]\w*)\s*=/.exec(line))) {
      defs.push(makeDef(m[1]!, 'variable', i, rawLines, undefined, false));
      continue;
    }
    if ((m = /^\s*([A-Za-z_]\w*)\s*=(?!=)/.exec(line))) {
      defs.push(makeDef(m[1]!, 'variable', i, rawLines, undefined, false));
    }
  }

  return { defs, imports };
}

/* ------------------------------------------------------------------ */
/* R                                                                   */
/* ------------------------------------------------------------------ */

function extractR(masked: string, raw: string): Extraction {
  const maskedLines = masked.split('\n');
  const rawLines = raw.split('\n');
  const defs: SymbolDef[] = [];
  const imports: string[] = [];
  const seenImports = new Set<string>();

  for (let i = 0; i < maskedLines.length; i++) {
    const line = maskedLines[i] ?? '';
    const rawLine = rawLines[i] ?? '';
    let m: RegExpExecArray | null;

    if ((m = /^\s*([A-Za-z_.][\w.]*)\s*(?:<<-|<-|=)\s*function\s*\(/.exec(line))) {
      defs.push(makeDef(m[1]!, 'function', i, rawLines, undefined, false));
      continue;
    }
    if (/^\s*(?:library|require|source)\s*\(/.test(line)) {
      const q = quotedImport(line, rawLine);
      if (q !== null) addUnique(imports, seenImports, q);
      else if ((m = /\(\s*([A-Za-z_.][\w.]*)/.exec(line))) addUnique(imports, seenImports, m[1]!);
      continue;
    }
    if ((m = /^\s*([A-Za-z_.][\w.]*)\s*(?:<<-|<-|=)(?!=)/.exec(line))) {
      const name = m[1]!;
      defs.push(makeDef(name, /^[A-Z][A-Z0-9_.]*$/.test(name) ? 'const' : 'variable', i, rawLines, undefined, false));
    }
  }

  return { defs, imports };
}

/* ------------------------------------------------------------------ */
/* Markdown, JSON, YAML, TOML                                          */
/* ------------------------------------------------------------------ */

function extractMarkdown(_masked: string, raw: string): Extraction {
  const lines = raw.split('\n');
  const defs: SymbolDef[] = [];
  let fence: string | null = null;

  for (let i = 0; i < lines.length; i++) {
    const line = lines[i] ?? '';
    const fenceMatch = /^\s*(```+|~~~+)/.exec(line);
    if (fenceMatch) {
      const marker = fenceMatch[1]![0] ?? '';
      if (fence === null) fence = marker;
      else if (marker === fence) fence = null;
      continue;
    }
    // A `# heading` inside a fenced block is code, not structure.
    if (fence !== null) continue;
    const heading = /^(#{1,6})\s+(.+?)\s*#*\s*$/.exec(line);
    if (heading && heading[2] !== undefined && heading[2] !== '') {
      defs.push(makeDef(heading[2], 'module', i, lines, undefined, false));
    }
  }

  return { defs, imports: [] };
}

/**
 * JSON is scanned by hand rather than parsed. `JSON.parse` would throw on the trailing
 * commas and comments real files carry, and would materialise the whole document; the
 * index only wants the root object's keys.
 */
function extractJson(_masked: string, raw: string): Extraction {
  const rawLines = raw.split('\n');
  const defs: SymbolDef[] = [];
  let depth = 0;
  let line = 1;
  let inString = false;
  let escaped = false;
  let keyStart = -1;
  let keyLine = 1;

  for (let i = 0; i < raw.length; i++) {
    const ch = raw[i]!;
    if (ch === '\n') line += 1;
    if (inString) {
      if (escaped) escaped = false;
      else if (ch === '\\') escaped = true;
      else if (ch === '"') {
        inString = false;
        if (depth === 1 && keyStart >= 0) {
          let k = i + 1;
          while (k < raw.length && (raw[k] === ' ' || raw[k] === '\t')) k += 1;
          if (raw[k] === ':') {
            defs.push(makeDef(raw.slice(keyStart, i), 'const', keyLine - 1, rawLines, undefined, false));
          }
        }
        keyStart = -1;
      }
      continue;
    }
    if (ch === '"') {
      inString = true;
      keyStart = i + 1;
      keyLine = line;
      continue;
    }
    if (ch === '{') depth += 1;
    else if (ch === '}' && depth > 0) depth -= 1;
  }

  return { defs, imports: [] };
}

function extractYaml(masked: string, raw: string): Extraction {
  const maskedLines = masked.split('\n');
  const rawLines = raw.split('\n');
  const defs: SymbolDef[] = [];
  let blockIndent = -1;

  for (let i = 0; i < maskedLines.length; i++) {
    const line = maskedLines[i] ?? '';
    if (line.trim() === '') continue;
    const indent = indentationOf(line);
    // A `|`/`>` block scalar's body is prose: it must not be read as nested keys.
    if (blockIndent >= 0) {
      if (indent > blockIndent) continue;
      blockIndent = -1;
    }
    const m = /^([A-Za-z_][A-Za-z0-9_.-]*)\s*:(?:\s|$)/.exec(line);
    if (indent === 0 && m && m[1] !== undefined) {
      defs.push(makeDef(m[1], 'const', i, rawLines, undefined, false));
      const rest = line.slice(line.indexOf(':') + 1).trim();
      if (rest === '|' || rest === '>' || rest.startsWith('|-') || rest.startsWith('>-')) blockIndent = indent;
    }
  }

  return { defs, imports: [] };
}

function extractToml(masked: string, raw: string): Extraction {
  const maskedLines = masked.split('\n');
  const rawLines = raw.split('\n');
  const defs: SymbolDef[] = [];

  for (let i = 0; i < maskedLines.length; i++) {
    const line = maskedLines[i] ?? '';
    // "Top-level keys only" means exactly that: the first table header ends the region
    // where a key is a property of the document rather than of a table.
    if (/^\s*\[/.test(line)) break;
    const m = /^\s*([A-Za-z_][A-Za-z0-9_-]*(?:\.[A-Za-z_][A-Za-z0-9_-]*)*)\s*=/.exec(line);
    if (m && m[1] !== undefined) defs.push(makeDef(m[1], 'const', i, rawLines, undefined, false));
  }

  return { defs, imports: [] };
}

/* ------------------------------------------------------------------ */
/* Vue and Svelte                                                      */
/* ------------------------------------------------------------------ */

/**
 * Extract from a `<script>` block only, shifting line numbers by the block's offset.
 *
 * A single-file component is two languages at once, and only one of them has symbols.
 * Running the JS extractor over the whole file would index template markup; running it
 * on the slice is both more accurate and cheaper. When there is no script block (or an
 * unterminated one) the result is simply empty rather than an error.
 */
function extractScriptRegion(raw: string, jsx: boolean): Extraction {
  const open = /<script\b[^>]*>/i.exec(raw);
  if (!open) return { defs: [], imports: [] };
  const openEnd = open.index + open[0].length;
  const close = raw.indexOf('</script', openEnd);
  const body = raw.slice(openEnd, close === -1 ? raw.length : close);
  const offset = countNewlines(raw.slice(0, openEnd));
  // `lang="ts"` and plain JS share the same masker: the only difference between the two
  // grammars that matters here is type annotations, which no rule depends on.
  const inner = extractJsLike(maskCode(body, JS_LEX), body, { jsx });
  if (offset === 0) return inner;
  return { defs: inner.defs.map((d) => ({ ...d, line: d.line + offset })), imports: inner.imports };
}

/* ------------------------------------------------------------------ */
/* Language table                                                      */
/* ------------------------------------------------------------------ */

const LANGUAGES: LanguageDef[] = [
  { id: 'c', extensions: ['.c', '.h'], lex: C_LINE, extract: (m, r) => extractCLike(m, r, false) },
  { id: 'cpp', extensions: ['.cc', '.cpp', '.cxx', '.hpp', '.hh'], lex: C_LINE, extract: (m, r) => extractCLike(m, r, true) },
  { id: 'csharp', extensions: ['.cs'], lex: C_LINE, extract: extractCSharp },
  { id: 'go', extensions: ['.go'], lex: GO_LEX, extract: extractGo },
  { id: 'java', extensions: ['.java'], lex: C_LINE, extract: extractJava },
  { id: 'javascript', extensions: ['.js', '.mjs', '.cjs'], lex: JS_LEX, extract: (m, r) => extractJsLike(m, r, { jsx: false }) },
  { id: 'json', extensions: ['.json'], lex: EMPTY_LEX, extract: extractJson },
  { id: 'jsx', extensions: ['.jsx'], lex: JS_LEX, extract: (m, r) => extractJsLike(m, r, { jsx: true }) },
  { id: 'kotlin', extensions: ['.kt', '.kts'], lex: KOTLIN_LEX, extract: extractKotlin },
  { id: 'lua', extensions: ['.lua'], lex: LUA_LEX, extract: extractLua },
  { id: 'markdown', extensions: ['.md'], lex: EMPTY_LEX, extract: extractMarkdown },
  { id: 'php', extensions: ['.php'], lex: C_LINE, extract: extractPhp },
  { id: 'python', extensions: ['.py', '.pyi'], lex: PY_LEX, extract: extractPython },
  { id: 'r', extensions: ['.r'], lex: HASH_LEX, extract: extractR },
  { id: 'ruby', extensions: ['.rb'], lex: RUBY_LEX, extract: extractRuby },
  { id: 'rust', extensions: ['.rs'], lex: RUST_LEX, extract: extractRust },
  { id: 'scala', extensions: ['.scala'], lex: KOTLIN_LEX, extract: extractScala },
  { id: 'shell', extensions: ['.sh', '.bash', '.zsh'], lex: HASH_LEX, extract: extractShell },
  { id: 'sql', extensions: ['.sql'], lex: SQL_LEX, extract: extractSql },
  { id: 'svelte', extensions: ['.svelte'], lex: EMPTY_LEX, extract: (_m, r) => extractScriptRegion(r, false) },
  { id: 'swift', extensions: ['.swift'], lex: KOTLIN_LEX, extract: extractSwift },
  { id: 'toml', extensions: ['.toml'], lex: HASH_LEX, extract: extractToml },
  { id: 'tsx', extensions: ['.tsx'], lex: JS_LEX, extract: (m, r) => extractJsLike(m, r, { jsx: true }) },
  { id: 'typescript', extensions: ['.ts', '.mts', '.cts'], lex: JS_LEX, extract: (m, r) => extractJsLike(m, r, { jsx: false }) },
  { id: 'vue', extensions: ['.vue'], lex: EMPTY_LEX, extract: (_m, r) => extractScriptRegion(r, false) },
  { id: 'yaml', extensions: ['.yaml', '.yml'], lex: HASH_LEX, extract: extractYaml },
];

const EXT_TO_LANG = new Map<string, string>();
for (const language of LANGUAGES) {
  for (const ext of language.extensions) EXT_TO_LANG.set(ext, language.id);
}

/** Sorted ids, for menus and validation. */
export const LANGUAGE_IDS: string[] = LANGUAGES.map((l) => l.id).sort();

/** Sorted, lowercase, leading dot; exactly the extensions `languageFor` accepts. */
export const INDEXABLE_EXTENSIONS: string[] = [...EXT_TO_LANG.keys()].sort();

function extensionOf(path: string): string {
  const slash = Math.max(path.lastIndexOf('/'), path.lastIndexOf('\\'));
  const base = path.slice(slash + 1);
  const dot = base.lastIndexOf('.');
  // `dot <= 0` also rejects a dotfile like `.bashrc`, which has no extension.
  if (dot <= 0) return '';
  return base.slice(dot).toLowerCase();
}

/** The language id for a path, or null when its extension is not indexable. */
export function languageFor(path: string): string | null {
  return EXT_TO_LANG.get(extensionOf(path)) ?? null;
}

/** True when `extract` has a real extractor for this path. */
export function isIndexable(path: string): boolean {
  return languageFor(path) !== null;
}

function safeIdentifiers(source: string): string[] {
  try {
    return collectIdentifiers(source, MAX_REFS_PER_FILE);
  } catch {
    return [];
  }
}

/**
 * Mask comments and string literals for the language of `path`, preserving length.
 *
 * Exposed because line-accurate reference lookup needs it: the index knows which files
 * mention a name, and re-scanning those files for the exact line must apply the same
 * masking the extractor did. Without this, a mention inside a comment or a string is
 * reported as a use — which is precisely the thing `find_references` claims to do better
 * than a text search.
 *
 * For an unknown extension the text is returned unchanged: masking with the wrong
 * language's rules would blank real code, and a false line is worse than a noisy one.
 */
export function maskSource(path: string, text: string): string {
  const id = languageFor(path);
  if (id === null) return text;
  const language = LANGUAGES.find((l) => l.id === id);
  if (language === undefined) return text;
  try {
    return maskCode(text, language.lex);
  } catch {
    return text;
  }
}

/**
 * Extract the symbols, references and imports of one file.
 *
 * Never throws. An unknown extension is not an error — it is `lang: 'text'` with
 * references still collected from the raw text, which keeps the file linkable in the
 * graph even though nothing knows how to read it. A known extension whose extractor
 * fails for any reason falls back to the same shape, because the caller (a cache
 * builder walking thousands of files) must not have to defend against this one.
 */
export function extract(path: string, text: string): ExtractResult {
  const id = languageFor(path);
  if (id === null) {
    return { lang: 'text', defs: [], refs: safeIdentifiers(text), imports: [] };
  }
  const language = LANGUAGES.find((l) => l.id === id);
  if (language === undefined) {
    return { lang: 'text', defs: [], refs: safeIdentifiers(text), imports: [] };
  }
  try {
    const masked = maskCode(text, language.lex);
    const out = language.extract(masked, text);
    return { lang: id, defs: out.defs, refs: safeIdentifiers(masked), imports: out.imports };
  } catch {
    return { lang: id, defs: [], refs: safeIdentifiers(text), imports: [] };
  }
}
