/**
 * Syntax checking without dependencies.
 *
 * We deliberately use the *language's own* parser when it is installed, because
 * a hand-rolled checker would accept subtly broken code — precisely the failure
 * mode a weak local model exhibits. Strategy per language:
 *
 *   python      -> `python3 -c "import ast; ast.parse(...)"`. Real parser, ~30 ms,
 *                  and it is present on any machine that has Python at all.
 *   javascript  -> `node --check <tmpfile>`. Real parser.
 *   typescript  -> `node --experimental-strip-types --check`. Node 22 strips types
 *                  and then parses, so this catches genuine syntax errors. If the
 *                  runtime rejects the flag we fall back to delimiter balance
 *                  rather than producing a false failure.
 *   json        -> JSON.parse.
 *   everything  -> delimiter/string balance heuristic only, reported as a
 *                  `warning` when it looks wrong, never as a hard error.
 *
 * Every check is bounded by a timeout and writes to a scratch directory that is
 * removed afterwards; nothing here touches the user's workspace.
 */

import { mkdirSync, rmSync, writeFileSync } from 'node:fs';
import { join } from 'node:path';

import type { CheckResult } from './types.ts';
import { exec } from '../util/proc.ts';

export interface SyntaxResult {
  ok: boolean;
  /** 'parser' when a real parser ran, 'heuristic' when we only balanced delimiters. */
  method: 'parser' | 'heuristic' | 'skipped';
  detail: string;
  /** First ~300 chars of parser output. */
  evidence?: string;
}

const LANG_EXT: Record<string, string> = {
  python: 'py',
  javascript: 'mjs',
  typescript: 'ts',
  json: 'json',
  yaml: 'yaml',
  shell: 'sh',
  go: 'go',
  rust: 'rs',
  java: 'java',
  ruby: 'rb',
  sql: 'sql',
  c: 'c',
  cpp: 'cpp',
  csharp: 'cs',
};

export function extensionFor(lang: string): string {
  return LANG_EXT[lang] ?? 'txt';
}

/**
 * Delimiter balance via a single pass with a stack, skipping comments and
 * string literals. Catches both mismatched counts and mis-nesting (`([)]`).
 */
export function balanceCheck(code: string): { ok: boolean; detail: string } {
  const pairs: Record<string, string> = { '}': '{', ')': '(', ']': '[' };
  const opens = new Set(['{', '(', '[']);
  const stack: Array<{ ch: string; line: number }> = [];
  let inString: string | null = null;
  let inLineComment = false;
  let inBlockComment = false;
  let escape = false;
  let line = 1;

  for (let i = 0; i < code.length; i++) {
    const ch = code[i] as string;
    const next = code[i + 1] as string | undefined;
    if (ch === '\n') {
      line++;
      inLineComment = false;
    }
    if (inLineComment) continue;
    if (inBlockComment) {
      if (ch === '*' && next === '/') {
        inBlockComment = false;
        i++;
      }
      continue;
    }
    if (inString) {
      if (escape) escape = false;
      else if (ch === '\\') escape = true;
      else if (ch === inString) inString = null;
      continue;
    }
    // Note: `#` is treated as a comment starter (Python/shell/Ruby/YAML). In C-like
    // languages `#` only appears in preprocessor lines, where the delimiters are
    // rarely unbalanced across a line, so the false-negative risk is acceptable.
    if (ch === '#' || (ch === '/' && next === '/')) {
      inLineComment = true;
      continue;
    }
    if (ch === '/' && next === '*') {
      inBlockComment = true;
      i++;
      continue;
    }
    if (ch === '"' || ch === "'" || ch === '`') {
      inString = ch;
      continue;
    }
    if (opens.has(ch)) {
      stack.push({ ch, line });
      continue;
    }
    const expectedOpen = pairs[ch];
    if (expectedOpen) {
      const top = stack.pop();
      if (!top) return { ok: false, detail: `unmatched "${ch}" on line ${line}` };
      if (top.ch !== expectedOpen) {
        return {
          ok: false,
          detail: `"${top.ch}" opened on line ${top.line} closed by "${ch}" on line ${line}`,
        };
      }
    }
  }

  if (inString) return { ok: false, detail: `unterminated string literal (${inString})` };
  if (stack.length > 0) {
    const top = stack[stack.length - 1] as { ch: string; line: number };
    return { ok: false, detail: `${stack.length} unclosed delimiter(s); last "${top.ch}" opened on line ${top.line}` };
  }
  return { ok: true, detail: 'delimiters balanced' };
}

let scratchCounter = 0;

export async function syntaxCheck(
  code: string,
  lang: string,
  scratchDir: string,
  timeoutMs = 10_000,
): Promise<SyntaxResult> {
  if (!code.trim()) return { ok: true, method: 'skipped', detail: 'empty file' };

  if (lang === 'json') {
    try {
      JSON.parse(code);
      return { ok: true, method: 'parser', detail: 'valid JSON' };
    } catch (err) {
      return { ok: false, method: 'parser', detail: 'invalid JSON', evidence: String(err).slice(0, 300) };
    }
  }

  if (lang === 'python') {
    const res = await exec('python3', ['-c', 'import ast,sys; ast.parse(sys.stdin.read())'], {
      input: code,
      timeoutMs,
      maxOutputBytes: 16 * 1024,
    });
    if (res.spawnFailed) {
      const balance = balanceCheck(code);
      return {
        ok: balance.ok,
        method: 'heuristic',
        detail: `python3 not found; used delimiter balance instead (${balance.detail})`,
      };
    }
    if (res.code === 0) return { ok: true, method: 'parser', detail: 'parses cleanly (ast.parse)' };
    return {
      ok: false,
      method: 'parser',
      detail: 'Python syntax error',
      evidence: res.stderr.trim().split('\n').slice(-2).join(' | ').slice(0, 300),
    };
  }

  if (lang === 'javascript' || lang === 'typescript') {
    mkdirSync(scratchDir, { recursive: true });
    const file = join(scratchDir, `syntax-${process.pid}-${scratchCounter++}.${extensionFor(lang)}`);
    try {
      writeFileSync(file, code, 'utf8');
      const args =
        lang === 'typescript'
          ? ['--experimental-strip-types', '--disable-warning=ExperimentalWarning', '--check', file]
          : ['--check', file];
      const res = await exec(process.execPath, args, { timeoutMs, maxOutputBytes: 16 * 1024 });
      if (res.code === 0) return { ok: true, method: 'parser', detail: `parses cleanly (node --check, ${lang})` };
      const stderr = res.stderr.trim();
      const flagUnsupported = /bad option|not supported|Unknown option/i.test(stderr);
      if (flagUnsupported) {
        const bal = balanceCheck(code);
        return { ok: bal.ok, method: 'heuristic', detail: `node --check unavailable for ${lang}; ${bal.detail}` };
      }
      return {
        ok: false,
        method: 'parser',
        detail: `${lang} syntax error`,
        evidence: stderr.split('\n').filter(Boolean).slice(-3).join(' | ').slice(0, 300),
      };
    } finally {
      rmSync(file, { force: true });
    }
  }

  const bal = balanceCheck(code);
  return {
    ok: bal.ok,
    method: 'heuristic',
    detail: `no parser wired for "${lang}"; ${bal.detail}`,
  };
}

export function asCheck(result: SyntaxResult, id: string, file: string): CheckResult {
  return {
    id,
    ok: result.ok,
    // Only a real parser may hard-fail a candidate; heuristics warn.
    severity: result.ok ? 'info' : result.method === 'parser' ? 'error' : 'warning',
    detail: `${file}: ${result.detail}`,
    ...(result.evidence ? { evidence: result.evidence } : {}),
  };
}
