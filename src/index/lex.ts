/**
 * The tiny lexer the symbol extractors share.
 *
 * Symbol extraction runs on *masked* source rather than raw text. The reason is that a
 * plain regex over raw source produces confident nonsense: `// fn handle_login()` inside
 * a comment, `const sql = "select * from users"` inside a string, or a `#` license header
 * all look like definitions to a line regex. Those false symbols then enter the
 * reference graph, where they inflate the ranking of whatever file happened to contain
 * them.
 *
 * Masking replaces the contents of comments and string literals with spaces, keeping the
 * byte length and every newline exactly where they were. Line numbers therefore stay
 * correct, columns stay correct, and a definition regex can run over the masked text
 * without special-casing anything.
 *
 * This is not a parser and does not try to be. It handles line comments, block comments
 * and single/multi-line strings with escapes — which is what the languages in the table
 * actually use — and it degrades to "mask less than it should" rather than throwing on
 * anything unusual. Masking too little produces a false symbol; masking too much could
 * hide a real one, so the failure mode is chosen deliberately.
 */

export interface StringSpec {
  open: string;
  close: string;
  escape?: string;
  /** True when the literal may span lines (Python triple quotes, JS templates). */
  multiline?: boolean;
}

export interface LexSpec {
  lineComments: string[];
  blockComments: Array<[string, string]>;
  strings: StringSpec[];
}

/**
 * Replace comment and string contents with spaces, preserving length and newlines.
 *
 * The returned string is the same length as the input, so any index into one is a valid
 * index into the other. Callers rely on that to map a regex match back to a line number.
 */
export function maskCode(text: string, spec: LexSpec): string {
  const out = text.split('');
  const n = text.length;
  let i = 0;

  /** Blank out [start, end) in place, keeping newlines so line counting survives. */
  const blank = (start: number, end: number): void => {
    for (let k = start; k < end && k < n; k++) {
      if (out[k] !== '\n') out[k] = ' ';
    }
  };

  while (i < n) {
    const rest = text.slice(i, i + 8);

    // Line comment: blank to end of line, leaving the newline.
    const lc = spec.lineComments.find((m) => rest.startsWith(m));
    if (lc !== undefined) {
      let end = text.indexOf('\n', i);
      if (end === -1) end = n;
      blank(i, end);
      i = end;
      continue;
    }

    // Block comment: blank through the terminator. An unterminated opener blanks the
    // remainder, which is the safe direction for a file being edited mid-flight.
    const bc = spec.blockComments.find(([open]) => rest.startsWith(open));
    if (bc !== undefined) {
      const closeAt = text.indexOf(bc[1], i + bc[0].length);
      const end = closeAt === -1 ? n : closeAt + bc[1].length;
      blank(i, end);
      i = end;
      continue;
    }

    // String literal. Longest opener first, so `"""` wins over `"` in Python.
    const candidates = [...spec.strings].sort((a, b) => b.open.length - a.open.length);
    const str = candidates.find((s) => rest.startsWith(s.open));
    if (str !== undefined) {
      const esc = str.escape ?? '\\';
      let k = i + str.open.length;
      let closed = false;
      while (k < n) {
        if (text.startsWith(esc + str.close, k) || (esc !== '' && text[k] === esc)) {
          k += 2;
          continue;
        }
        if (text.startsWith(str.close, k)) {
          k += str.close.length;
          closed = true;
          break;
        }
        if (text[k] === '\n' && str.multiline !== true) break; // unterminated single-line
        k++;
      }
      // Keep the delimiters visible: a regex like `from "x" import` still needs them,
      // and blanking them would make import extraction language-specific for no gain.
      blank(i + str.open.length, closed ? k - str.close.length : k);
      i = closed ? k : Math.max(k, i + 1);
      continue;
    }

    i++;
  }

  return out.join('');
}

const IDENT_START = /[A-Za-z_$]/;
const IDENT_CHAR = /[A-Za-z0-9_$]/;

export function isIdentStart(ch: string | undefined): boolean {
  return ch !== undefined && IDENT_START.test(ch);
}

export function isIdentChar(ch: string | undefined): boolean {
  return ch !== undefined && IDENT_CHAR.test(ch);
}

/**
 * Every identifier appearing in code position, de-duplicated and capped.
 *
 * Keywords are dropped: they appear in every file, so they carry no signal in the
 * reference graph and would dominate a file's reference list. The set is deliberately
 * small and conservative — a word wrongly dropped costs a graph edge, while a word
 * wrongly kept adds noise to every file equally.
 */
export function collectIdentifiers(masked: string, cap: number): string[] {
  const seen = new Set<string>();
  const n = masked.length;
  let i = 0;
  while (i < n) {
    if (!isIdentStart(masked[i])) {
      i++;
      continue;
    }
    let j = i + 1;
    while (j < n && isIdentChar(masked[j])) j++;
    const word = masked.slice(i, j);
    if (word.length > 1 && !KEYWORDS.has(word)) {
      seen.add(word);
      if (seen.size >= cap) break;
    }
    i = j;
  }
  return [...seen];
}

/**
 * Words that carry no reference signal. Union of the reserved words of the languages in
 * the table plus ubiquitous type names; keeping one list rather than one per language
 * costs nothing, since a keyword in one language is rarely an identifier in another.
 */
const KEYWORDS = new Set([
  // shared control / structure
  'if', 'else', 'elif', 'for', 'while', 'do', 'switch', 'case', 'default', 'break', 'continue',
  'return', 'yield', 'await', 'async', 'try', 'catch', 'except', 'finally', 'throw', 'raise',
  'with', 'as', 'in', 'of', 'is', 'not', 'and', 'or', 'del', 'pass', 'lambda', 'match', 'when',
  // declarations / types
  'class', 'def', 'fn', 'func', 'function', 'interface', 'type', 'enum', 'struct', 'trait',
  'impl', 'const', 'let', 'var', 'static', 'final', 'public', 'private', 'protected', 'internal',
  'export', 'import', 'from', 'require', 'module', 'namespace', 'package', 'using', 'use',
  'extends', 'implements', 'new', 'this', 'self', 'super', 'base', 'void', 'null', 'nil', 'none',
  'true', 'false', 'undefined', 'int', 'float', 'double', 'bool', 'boolean', 'string', 'str',
  'char', 'byte', 'long', 'short', 'unsigned', 'signed', 'object', 'any', 'unknown', 'never',
  'number', 'list', 'dict', 'set', 'tuple', 'map', 'array', 'optional', 'union', 'readonly',
  'abstract', 'override', 'virtual', 'suspend', 'fun', 'val', 'mut', 'where', 'select', 'insert',
  'update', 'delete', 'create', 'table', 'index', 'into', 'values', 'join', 'on', 'group', 'by',
  'order', 'limit', 'offset', 'having', 'distinct', 'begin', 'end', 'then', 'fi', 'done', 'esac',
  'echo', 'local', 'global', 'declare', 'function', 'eval', 'exit', 'source',
  // ubiquitous names that would otherwise link every file to every other
  'get', 'set', 'init', 'main', 'args', 'kwargs', 'props', 'state', 'data', 'value', 'key',
  'name', 'id', 'index', 'length', 'size', 'to', 'from', 'then', 'catch', 'error', 'result',
  'console', 'log', 'print', 'printf', 'len', 'range', 'enumerate', 'zip', 'open', 'read',
  'write', 'close', 'append', 'push', 'pop', 'map', 'filter', 'reduce', 'forEach', 'some',
  'every', 'find', 'includes', 'slice', 'split', 'join', 'replace', 'trim', 'test', 'expect',
  'assert', 'describe', 'it',
]);
