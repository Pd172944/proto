/**
 * Candidate parsing, in-memory patch application and pattern checks.
 *
 * Everything here is *pure*: it never writes to the user's workspace. Applying
 * to disk is a separate, explicit step (`proto run --apply`), because the most
 * damaging bug this harness could have is silently corrupting source files while
 * "helpfully" fixing an easy problem.
 */

import type { AppliedFile, CandidateEdit, CandidateOutput, CheckResult } from './types.ts';
import { extractCodeBlocks, extractJson } from '../util/text.ts';
import { languageOfPath } from '../router/features.ts';

/* ------------------------------------------------------------------ */
/* Parsing                                                            */
/* ------------------------------------------------------------------ */

/** Files that must never be touched by an automated edit. */
const PROTECTED_PATH = /(^|\/)(\.git|\.hg|\.svn|node_modules|\.venv|venv|__pycache__|var|dist|build|\.next|target)(\/|$)/;
const PROTECTED_FILE = /(^|\/)(\.env(\..*)?|secrets?\.json|id_rsa|id_ed25519|\.npmrc|\.netrc|credentials)$/i;

export function validateEditPath(path: string): { ok: boolean; detail: string } {
  if (!path.trim()) return { ok: false, detail: 'empty file path' };
  if (path.startsWith('/') || /^[A-Za-z]:[\\/]/.test(path)) {
    return { ok: false, detail: `absolute paths are not allowed: ${path}` };
  }
  if (path.split(/[\\/]/).includes('..')) {
    return { ok: false, detail: `path traversal is not allowed: ${path}` };
  }
  if (PROTECTED_PATH.test(path)) {
    return { ok: false, detail: `refusing to edit protected path: ${path}` };
  }
  if (PROTECTED_FILE.test(path)) {
    return { ok: false, detail: `refusing to edit secret/credential file: ${path}` };
  }
  return { ok: true, detail: 'path ok' };
}

function asString(v: unknown): string | undefined {
  return typeof v === 'string' ? v : undefined;
}

function coerceEdit(raw: unknown): CandidateEdit | null {
  if (!raw || typeof raw !== 'object') return null;
  const o = raw as Record<string, unknown>;
  const file = asString(o['file']) ?? asString(o['path']) ?? asString(o['filename']);
  if (!file) return null;
  const edit: CandidateEdit = { file };
  const find = asString(o['find']) ?? asString(o['old']) ?? asString(o['search']);
  const replace = asString(o['replace']) ?? asString(o['new']);
  const content = asString(o['content']) ?? asString(o['file_content']);
  if (find !== undefined) edit.find = find;
  if (replace !== undefined) edit.replace = replace;
  if (content !== undefined) edit.content = content;
  if (edit.find === undefined && edit.content === undefined) return null;
  if (edit.find !== undefined && edit.replace === undefined) edit.replace = '';
  return edit;
}

/**
 * Parse a model response into a candidate.
 *
 * Models drift from the requested envelope, so we accept several shapes and
 * record *which* shape was used. A high drift rate is itself a useful signal
 * about the local model's reliability.
 */
export function parseCandidate(text: string, opts: { expectedPaths?: string[] } = {}): {
  candidate: CandidateOutput | null;
  parseError?: string;
  shape: 'json' | 'fence' | 'prose' | 'none';
} {
  const json = extractJson(text);
  if (json && typeof json === 'object' && !Array.isArray(json)) {
    const o = json as Record<string, unknown>;
    const rawEdits = o['edits'] ?? o['changes'] ?? o['patches'];
    const edits: CandidateEdit[] = [];
    if (Array.isArray(rawEdits)) {
      for (const e of rawEdits) {
        const c = coerceEdit(e);
        if (c) edits.push(c);
      }
    } else if (o['file'] || o['path']) {
      const single = coerceEdit(o);
      if (single) edits.push(single);
    }
    const candidate: CandidateOutput = {
      summary: asString(o['summary']) ?? asString(o['description']) ?? '',
      edits,
      ...(asString(o['answer']) ? { answer: asString(o['answer']) as string } : {}),
      ...(asString(o['explanation']) ? { explanation: asString(o['explanation']) as string } : {}),
      ...(isRisk(o['risk']) ? { risk: o['risk'] as CandidateOutput['risk'] } : {}),
      ...(asString(o['uncertainty']) ? { uncertainty: asString(o['uncertainty']) as string } : {}),
      raw: text,
    };
    if (candidate.edits.length > 0 || candidate.answer) {
      return { candidate, shape: 'json' };
    }
    return {
      candidate,
      parseError: 'JSON envelope contained neither usable edits nor an answer',
      shape: 'json',
    };
  }

  // Fallback: a fenced code block. Only usable when we know the intended file.
  const blocks = extractCodeBlocks(text);
  const path = blocks.find((b) => b.path)?.path ?? (opts.expectedPaths?.length === 1 ? opts.expectedPaths[0] : undefined);
  if (blocks.length === 1 && path) {
    const block = blocks[0] as { code: string };
    return {
      candidate: {
        summary: 'salvaged from a code fence (model did not use the JSON envelope)',
        edits: [{ file: path, content: block.code }],
        raw: text,
      },
      shape: 'fence',
    };
  }

  if (text.trim()) {
    return {
      candidate: { summary: 'prose answer (no structured edits)', edits: [], answer: text.trim(), raw: text },
      shape: 'prose',
    };
  }
  return { candidate: null, parseError: 'empty response', shape: 'none' };
}

function isRisk(v: unknown): boolean {
  return v === 'low' || v === 'medium' || v === 'high';
}

/* ------------------------------------------------------------------ */
/* Application (in memory)                                            */
/* ------------------------------------------------------------------ */

export interface ApplyResult {
  applied: AppliedFile[];
  errors: CheckResult[];
}

export function applyEdits(
  candidate: CandidateOutput,
  known: Map<string, string>,
): ApplyResult {
  const applied: AppliedFile[] = [];
  const errors: CheckResult[] = [];

  for (const [i, edit] of candidate.edits.entries()) {
    const pathCheck = validateEditPath(edit.file);
    if (!pathCheck.ok) {
      errors.push({ id: `path:${i}`, ok: false, severity: 'error', detail: pathCheck.detail });
      continue;
    }
    const before = known.get(edit.file) ?? null;

    if (edit.content !== undefined) {
      if (before !== null && before === edit.content) {
        errors.push({
          id: `noop:${i}`,
          ok: false,
          severity: 'error',
          detail: `${edit.file}: replacement file content is identical to the original (no-op edit)`,
        });
        continue;
      }
      applied.push({
        path: edit.file,
        before,
        after: edit.content,
        changedLines: before === null ? edit.content.split('\n').length : diffLineCount(before, edit.content),
      });
      continue;
    }

    const find = edit.find as string;
    const replace = edit.replace ?? '';
    if (before === null) {
      errors.push({
        id: `missing:${i}`,
        ok: false,
        severity: 'error',
        detail: `${edit.file}: cannot apply a find/replace to a file that was not supplied and does not exist`,
      });
      continue;
    }
    const occurrences = countOccurrences(before, find);
    if (occurrences === 0) {
      errors.push({
        id: `anchor:${i}`,
        ok: false,
        severity: 'error',
        detail: `${edit.file}: anchor text not found; the model's understanding of the file is stale or wrong`,
        evidence: previewAnchor(find),
      });
      continue;
    }
    if (occurrences > 1) {
      errors.push({
        id: `ambiguous:${i}`,
        ok: false,
        severity: 'error',
        detail: `${edit.file}: anchor text appears ${occurrences} times; the edit is ambiguous and was rejected`,
        evidence: previewAnchor(find),
      });
      continue;
    }
    if (find === replace) {
      errors.push({
        id: `noop:${i}`,
        ok: false,
        severity: 'error',
        detail: `${edit.file}: find and replace are identical (no-op edit)`,
      });
      continue;
    }
    const after = before.replace(find, replace);
    applied.push({ path: edit.file, before, after, changedLines: diffLineCount(before, after) });
    known.set(edit.file, after);
  }

  return { applied, errors };
}

export function countOccurrences(haystack: string, needle: string): number {
  if (!needle) return 0;
  let count = 0;
  let idx = haystack.indexOf(needle);
  while (idx !== -1) {
    count++;
    idx = haystack.indexOf(needle, idx + Math.max(1, needle.length));
  }
  return count;
}

function previewAnchor(find: string): string {
  const lines = find.split('\n');
  const head = lines.slice(0, 3).join('\n');
  return lines.length > 3 ? `${head}\n… (${lines.length} lines total)` : head;
}

/**
 * Approximate changed-line count via line multiset difference.
 *
 * A true LCS diff would be O(n*m); this is O(n) and only needs to be good enough
 * for a size cap, not for display. It slightly over-counts reordered blocks,
 * which errs in the safe direction (rejecting oversized patches).
 */
export function diffLineCount(before: string, after: string): number {
  const beforeLines = before.split('\n');
  const afterLines = after.split('\n');
  const counts = new Map<string, number>();
  for (const l of beforeLines) counts.set(l, (counts.get(l) ?? 0) + 1);
  let removed = 0;
  for (const l of afterLines) {
    const c = counts.get(l);
    if (c && c > 0) counts.set(l, c - 1);
    else removed++; // lines present in `after` but not accounted for in `before` => added
  }
  let keptUnmatched = 0;
  for (const c of counts.values()) keptUnmatched += c; // lines in `before` never matched => removed
  return removed + keptUnmatched;
}

/* ------------------------------------------------------------------ */
/* Pattern checks                                                     */
/* ------------------------------------------------------------------ */

interface PatternRule {
  id: string;
  re: RegExp;
  severity: 'error' | 'warning';
  detail: string;
  /** Only apply to these languages. */
  langs?: string[];
}

const PATTERN_RULES: PatternRule[] = [
  { id: 'todo', re: /\b(TODO|FIXME|XXX|HACK)\b/, severity: 'warning', detail: 'introduces a TODO/FIXME marker' },
  {
    id: 'swallowed-error',
    re: /except\s*(Exception|BaseException)?\s*:\s*(pass|\.\.\.)\s*$|catch\s*(\([^)]*\))?\s*\{\s*\}|catch\s*\{\s*\/\*\s*\*\/\s*\}/,
    severity: 'warning',
    detail: 'swallows an error silently',
  },
  {
    id: 'type-escape',
    re: /@ts-ignore|@ts-nocheck|#\s*type:\s*ignore|#\s*noqa|\bas\s+any\b/,
    severity: 'warning',
    detail: 'suppresses type checking instead of fixing the type',
  },
  {
    id: 'debug-print',
    re: /^\s*(console\.(log|debug|dir)\(|debugger;?|print\(|System\.out\.println\(|fmt\.Println\()/,
    severity: 'warning',
    langs: ['python', 'javascript', 'typescript', 'java', 'go'],
    detail: 'adds a debug print statement',
  },
  { id: 'breakpoint', re: /\b(import\s+pdb|pdb\.set_trace\(\)|breakpoint\(\))/, severity: 'error', detail: 'leaves a debugger breakpoint in the code' },
  {
    id: 'dangerous-shell',
    re: /\brm\s+-rf\s+[/~]|os\.system\(|subprocess\.[a-z_]+\([^)]*shell\s*=\s*True|child_process\.exec\(|eval\(|new\s+Function\(/,
    severity: 'error',
    detail: 'introduces a dangerous execution primitive',
  },
  { id: 'bypass-hooks', re: /--no-verify|--force\b|git\s+push\s+--force/, severity: 'error', detail: 'bypasses version-control safeguards' },
  {
    id: 'hardcoded-secret',
    re: /(api[_-]?key|secret|password|passwd|token|access[_-]?key)\s*[:=]\s*["'][A-Za-z0-9_\-/+]{12,}["']|(sk-[A-Za-z0-9]{16,}|AKIA[0-9A-Z]{16}|ghp_[A-Za-z0-9]{20,})/i,
    severity: 'error',
    detail: 'hardcodes what looks like a credential',
  },
  { id: 'unsafe-yaml', re: /yaml\.load\((?![^)]*Loader)/, severity: 'warning', detail: 'uses an unsafe YAML loader' },
  { id: 'chmod-world', re: /chmod\s+777|0o777/, severity: 'warning', detail: 'sets world-writable permissions' },
];

/** Run pattern checks over only the *added* lines of the applied patch. */
export function patternChecks(
  applied: AppliedFile[],
  opts: { rejectNewTodos: boolean; forbiddenPatterns: string[] },
): CheckResult[] {
  const results: CheckResult[] = [];
  const custom = compileCustom(opts.forbiddenPatterns);

  for (const file of applied) {
    const lang = languageOfPath(file.path) ?? '';
    const added = addedLines(file.before ?? '', file.after);
    for (const rule of PATTERN_RULES) {
      if (rule.langs && !rule.langs.includes(lang)) continue;
      let severity = rule.severity;
      if (rule.id === 'todo' && opts.rejectNewTodos) severity = 'error';
      const hit = added.find((line) => rule.re.test(line));
      if (hit !== undefined) {
        results.push({
          id: `pattern:${rule.id}:${file.path}`,
          ok: false,
          severity,
          detail: `${file.path}: ${rule.detail}`,
          evidence: hit.trim().slice(0, 200),
        });
      }
    }
    for (const { source, re } of custom) {
      const hit = added.find((line) => re.test(line));
      if (hit !== undefined) {
        results.push({
          id: `forbidden:${file.path}`,
          ok: false,
          severity: 'error',
          detail: `${file.path}: matched configured forbidden pattern /${source}/`,
          evidence: hit.trim().slice(0, 200),
        });
      }
    }
  }
  return results;
}

function compileCustom(patterns: string[]): Array<{ source: string; re: RegExp }> {
  const out: Array<{ source: string; re: RegExp }> = [];
  for (const p of patterns) {
    try {
      out.push({ source: p, re: new RegExp(p) });
    } catch {
      // Invalid user regex is reported by config validation, not here.
    }
  }
  return out;
}

/**
 * Lines present in `after` but not in `before`, using the same multiset
 * approximation as `diffLineCount`. Cheap and adequate for pattern scanning.
 */
export function addedLines(before: string, after: string): string[] {
  const remaining = new Map<string, number>();
  for (const l of before.split('\n')) remaining.set(l, (remaining.get(l) ?? 0) + 1);
  const added: string[] = [];
  for (const l of after.split('\n')) {
    const c = remaining.get(l);
    if (c && c > 0) remaining.set(l, c - 1);
    else added.push(l);
  }
  return added;
}
