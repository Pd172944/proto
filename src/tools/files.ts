/**
 * Filesystem and shell tools.
 *
 * The interesting design choice here is that `edit_file` **reuses the project's
 * verifier**. Most coding agents apply a model's `old_string`/`new_string`
 * replacement directly and hope the anchor was unique. This harness already had a
 * verifier built to make weak models safe — unique-anchor checking, real language
 * parsers, anti-pattern scanning — so the edit tool routes through it:
 *
 *   1. the anchor must occur **exactly once** (zero or multiple ⇒ refuse, and tell
 *      the model why, which is a much better error than a corrupted file),
 *   2. the resulting file must still parse (caught by the language's own parser),
 *   3. added lines are scanned for swallowed errors, debug prints, new TODOs,
 *      hardcoded credentials and `@ts-ignore`-style escapes.
 *
 * Rejections are returned as *tool errors*, not thrown exceptions, because a tool
 * error is information the model can act on, whereas a crash ends the turn.
 */

import { readFileSync, readdirSync, statSync, writeFileSync } from 'node:fs';
import { mkdirSync } from 'node:fs';
import { dirname, join, relative } from 'node:path';

import { discoverFiles } from '../index/walk.ts';
import { normalizeSearchPattern, scanFiles } from '../index/search.ts';
import { repoIndexTools } from './repo.ts';
import { capOutput, resolveInsideWorkspace, ToolRegistry } from './types.ts';
import type { Tool, ToolContext, ToolResult } from './types.ts';
import { languageOfPath } from '../router/features.ts';
import { balanceCheck, syntaxCheck } from '../verify/syntax.ts';
import { addedLines, countOccurrences, patternChecks } from '../verify/patch.ts';
import { exec } from '../util/proc.ts';

/* ------------------------------------------------------------------ */
/* Helpers                                                             */
/* ------------------------------------------------------------------ */

function readTarget(ctx: ToolContext, p: string): { ok: true; abs: string; rel: string; content: string } | { ok: false; error: string } {
  const resolved = resolveInsideWorkspace(ctx.workspace, p);
  if (!resolved.ok) return resolved;
  try {
    const st = statSync(resolved.abs);
    if (st.isDirectory()) return { ok: false, error: `${resolved.rel} is a directory; use list_files` };
    if (st.size > 2_000_000) {
      return { ok: false, error: `${resolved.rel} is ${Math.round(st.size / 1024)} KB; read a line range instead` };
    }
    return { ...resolved, content: readFileSync(resolved.abs, 'utf8') };
  } catch {
    return { ok: false, error: `cannot read ${resolved.rel}: no such file` };
  }
}

function argString(args: Record<string, unknown>, key: string): string | undefined {
  const v = args[key];
  return typeof v === 'string' ? v : undefined;
}

function argNumber(args: Record<string, unknown>, key: string): number | undefined {
  const v = args[key];
  if (typeof v === 'number' && Number.isFinite(v)) return v;
  if (typeof v === 'string' && v.trim() !== '' && Number.isFinite(Number(v))) return Number(v);
  return undefined;
}

function withLineNumbers(text: string, start = 1): string {
  const lines = text.split('\n');
  const width = String(start + lines.length - 1).length;
  return lines.map((l, i) => `${String(start + i).padStart(width)}│ ${l}`).join('\n');
}

/* ------------------------------------------------------------------ */
/* read_file                                                           */
/* ------------------------------------------------------------------ */

export const readFileTool: Tool = {
  name: 'read_file',
  description:
    'Read a text file from the project. Returns numbered lines so you can cite and target exact ' +
    'locations. Use offset/limit for large files rather than reading everything. Read a file before ' +
    'editing it — never guess at its contents.',
  parameters: {
    type: 'object',
    properties: {
      path: { type: 'string', description: 'Path relative to the project root.' },
      offset: { type: 'number', description: '1-based first line to return. Defaults to 1.' },
      limit: { type: 'number', description: 'Maximum number of lines to return. Defaults to 400.' },
    },
    required: ['path'],
  },
  risk: 'read',
  async run(args, ctx): Promise<ToolResult> {
    const started = Date.now();
    const p = argString(args, 'path');
    if (!p) return { ok: false, title: 'read_file', output: 'path is required' };

    const target = readTarget(ctx, p);
    if (!target.ok) return { ok: false, title: `read ${p}`, output: target.error };

    const all = target.content.split('\n');
    const offset = Math.max(1, argNumber(args, 'offset') ?? 1);
    const limit = Math.min(4000, Math.max(1, argNumber(args, 'limit') ?? 400));
    const slice = all.slice(offset - 1, offset - 1 + limit);
    const numbered = withLineNumbers(slice.join('\n'), offset);
    const capped = capOutput(numbered, ctx.maxOutputChars);

    const more = all.length - (offset - 1 + slice.length);
    const note = more > 0 ? `\n\n… ${more} more line(s); call read_file again with offset=${offset + slice.length}.` : '';

    return {
      ok: true,
      title: `read ${target.rel} (${slice.length}${more > 0 ? ` of ${all.length}` : ''} lines)`,
      output: capped.text + note,
      meta: { files: [target.rel], truncated: capped.truncated, durationMs: Date.now() - started, lines: slice.length },
    };
  },
};

/* ------------------------------------------------------------------ */
/* list_files                                                          */
/* ------------------------------------------------------------------ */

export const listFilesTool: Tool = {
  name: 'list_files',
  description:
    'List files and directories. Use this to orient yourself in an unfamiliar project before reading ' +
    'anything. Skips VCS internals and dependency directories, which are never what you want.',
  parameters: {
    type: 'object',
    properties: {
      path: { type: 'string', description: 'Directory relative to the project root. Defaults to ".".' },
      depth: { type: 'number', description: 'How many levels to descend. Defaults to 1.' },
    },
  },
  risk: 'read',
  async run(args, ctx): Promise<ToolResult> {
    const started = Date.now();
    const p = argString(args, 'path') ?? '.';
    const resolved = resolveInsideWorkspace(ctx.workspace, p);
    if (!resolved.ok) return { ok: false, title: `list ${p}`, output: resolved.error };

    const depth = Math.min(6, Math.max(1, argNumber(args, 'depth') ?? 1));
    const skip = new Set(['.git', 'node_modules', '.venv', 'venv', '__pycache__', 'dist', 'build', '.next', 'target', 'var', '.DS_Store']);
    const lines: string[] = [];
    let files = 0;

    const walk = (dir: string, level: number): void => {
      if (level > depth || lines.length > 800) return;
      let entries: string[];
      try {
        entries = readdirSync(dir);
      } catch {
        return;
      }
      for (const name of entries.sort()) {
        if (skip.has(name)) continue;
        const abs = join(dir, name);
        let isDir = false;
        try {
          isDir = statSync(abs).isDirectory();
        } catch {
          continue;
        }
        const indent = '  '.repeat(level - 1);
        lines.push(`${indent}${isDir ? '▸ ' : '  '}${name}${isDir ? '/' : ''}`);
        if (!isDir) files++;
        if (isDir) walk(abs, level + 1);
      }
    };
    walk(resolved.abs, 1);

    const body = lines.length > 0 ? lines.join('\n') : '(empty directory)';
    const capped = capOutput(body, ctx.maxOutputChars);
    return {
      ok: true,
      title: `list ${resolved.rel} (${files} files)`,
      output: capped.text,
      meta: { truncated: capped.truncated, durationMs: Date.now() - started },
    };
  },
};

/* ------------------------------------------------------------------ */
/* search                                                             */
/* ------------------------------------------------------------------ */

export const searchTool: Tool = {
  name: 'search',
  description:
    'Search file contents with a regular expression. This is the fastest way to find where something ' +
    'is defined or used; prefer it over reading many files. Returns path:line: text.',
  parameters: {
    type: 'object',
    properties: {
      pattern: { type: 'string', description: 'Regular expression (JavaScript syntax).' },
      path: { type: 'string', description: 'Directory to search. Defaults to ".".' },
      glob: { type: 'string', description: 'Only search files whose path contains this substring, e.g. ".ts".' },
      max_results: { type: 'number', description: 'Cap on matches. Defaults to 80.' },
    },
    required: ['pattern'],
  },
  risk: 'read',
  async run(args, ctx): Promise<ToolResult> {
    const started = Date.now();
    const rawPattern = argString(args, 'pattern');
    if (!rawPattern) return { ok: false, title: 'search', output: 'pattern is required' };
    const pattern: string = normalizeSearchPattern(rawPattern);
    if (!pattern) return { ok: false, title: 'search', output: 'pattern is required' };

    let re: RegExp;
    try {
      re = new RegExp(pattern, 'i');
    } catch (err) {
      return { ok: false, title: 'search', output: `invalid regular expression: ${String(err)}` };
    }

    const dir = argString(args, 'path') ?? '.';
    const resolved = resolveInsideWorkspace(ctx.workspace, dir);
    if (!resolved.ok) return { ok: false, title: `search ${dir}`, output: resolved.error };

    const globFilter = argString(args, 'glob');
    const max = Math.min(500, Math.max(1, argNumber(args, 'max_results') ?? 80));
    const base = dir === '.' ? undefined : dir.replace(/^\.\//, '');

    /*
     * Two engines, one contract: return the matches, or say why none were found.
     *
     * With an index, `git grep` does the work in C over git's own file list and the index
     * narrows the rest — a repository-wide search becomes a scan of the handful of files
     * that could match. Without one, discovery still applies every gitignore rule before
     * a byte is read, which is what keeps the fallback bounded.
     *
     * The old version read every file in the tree into memory on every call with a
     * hardcoded skip list. It is gone: on a real repository it was the single most
     * expensive thing the agent did, and it silently stopped at 4000 files.
     */
    let outcome;
    if (ctx.index !== undefined) {
      await ctx.index.ensure();
      outcome = ctx.index.search({ pattern, caseSensitive: false, maxResults: max, ...(globFilter ? { glob: globFilter } : {}) });
    } else {
      const discovery = await discoverFiles(ctx.workspace);
      const candidates = discovery.files.map((f) => f.path).filter((p) => (base === undefined ? true : p.startsWith(base)));
      outcome = scanFiles(ctx.workspace, candidates, { pattern, caseSensitive: false, maxResults: max, ...(globFilter ? { glob: globFilter } : {}) });
    }

    const hits = outcome.hits.filter((h) => (base === undefined ? true : h.path.startsWith(base)));
    const body =
      hits.length > 0
        ? hits.map((h) => `${h.path}:${h.line}: ${h.text}`).join('\n')
        : `no matches for /${pattern}/ in ${dir}` +
          (outcome.truncated ? ' (the search was cut short — narrow it with glob or path)' : '');
    const capped = capOutput(body, ctx.maxOutputChars);
    return {
      ok: true,
      title: `search /${pattern}/ → ${hits.length} match${hits.length === 1 ? '' : 'es'}`,
      output: capped.text,
      meta: {
        truncated: capped.truncated || outcome.truncated,
        durationMs: Date.now() - started,
        matches: hits.length,
        engine: outcome.engine,
        filesScanned: outcome.filesScanned,
      },
    };
  },
};

/* ------------------------------------------------------------------ */
/* write_file                                                          */
/* ------------------------------------------------------------------ */

export const writeFileTool: Tool = {
  name: 'write_file',
  description:
    'Create a file, or overwrite one entirely. Prefer edit_file for changes to existing files: a whole-' +
    'file write can silently discard work you did not read. Use write_file for new files.',
  parameters: {
    type: 'object',
    properties: {
      path: { type: 'string', description: 'Path relative to the project root.' },
      content: { type: 'string', description: 'Full file contents.' },
    },
    required: ['path', 'content'],
  },
  risk: 'write',
  async run(args, ctx): Promise<ToolResult> {
    const started = Date.now();
    const p = argString(args, 'path');
    const content = argString(args, 'content');
    if (!p || content === undefined) return { ok: false, title: 'write_file', output: 'path and content are required' };

    const resolved = resolveInsideWorkspace(ctx.workspace, p);
    if (!resolved.ok) return { ok: false, title: `write ${p}`, output: resolved.error };

    const exists = (() => {
      try {
        return statSync(resolved.abs).isFile();
      } catch {
        return false;
      }
    })();

    const previous = exists ? readFileSync(resolved.abs, 'utf8') : null;
    if (previous !== null && previous === content) {
      return { ok: true, title: `write ${resolved.rel} (no change)`, output: 'file already has exactly this content; nothing written' };
    }

    const syntax = await checkSyntaxAfterWrite(content, resolved.rel, ctx);
    if (syntax) {
      return {
        ok: false,
        title: `write ${resolved.rel} refused`,
        output: `${syntax}\n\nThe file was not written. Fix the syntax and try again.`,
      };
    }

    const verdict = await ctx.approve({
      kind: 'write',
      title: `${exists ? 'Overwrite' : 'Create'} ${resolved.rel}`,
      detail: previous === null ? content : describeChange(previous, content),
      ...(previous === null ? { warning: 'new file' } : {}),
    });
    if (verdict === 'deny') {
      return { ok: false, title: `write ${resolved.rel} denied`, output: 'The user declined this write. Ask what they would prefer instead of retrying.' };
    }

    mkdirSync(dirname(resolved.abs), { recursive: true });
    writeFileSync(resolved.abs, content, 'utf8');

    const added = previous === null ? content.split('\n').length : addedLines(previous, content).length;
    return {
      ok: true,
      title: `write ${resolved.rel} (+${added} line${added === 1 ? '' : 's'})`,
      output: `Wrote ${resolved.rel} (${content.length} bytes).`,
      meta: { files: [resolved.rel], added, durationMs: Date.now() - started },
    };
  },
};

/* ------------------------------------------------------------------ */
/* edit_file — verifier-backed                                         */
/* ------------------------------------------------------------------ */

export const editFileTool: Tool = {
  name: 'edit_file',
  description:
    'Replace an exact snippet in an existing file. `old_string` must appear EXACTLY ONCE — include ' +
    'surrounding lines to disambiguate. This is the preferred way to change existing code: it cannot ' +
    'silently discard the parts of the file you did not read, and it is refused if the anchor is ' +
    'ambiguous, if the result would not parse, or if it introduces an anti-pattern.',
  parameters: {
    type: 'object',
    properties: {
      path: { type: 'string', description: 'Path relative to the project root.' },
      old_string: { type: 'string', description: 'Exact text to replace, including indentation. Must be unique in the file.' },
      new_string: { type: 'string', description: 'Replacement text. Use an empty string to delete.' },
    },
    required: ['path', 'old_string', 'new_string'],
  },
  risk: 'write',
  async run(args, ctx): Promise<ToolResult> {
    const started = Date.now();
    const p = argString(args, 'path');
    const oldString = argString(args, 'old_string');
    const newString = argString(args, 'new_string');
    if (!p || oldString === undefined || newString === undefined) {
      return { ok: false, title: 'edit_file', output: 'path, old_string and new_string are required' };
    }

    const target = readTarget(ctx, p);
    if (!target.ok) return { ok: false, title: `edit ${p}`, output: target.error };

    if (oldString === newString) {
      return { ok: false, title: `edit ${target.rel} refused`, output: 'old_string and new_string are identical; this edit would do nothing.' };
    }

    const occurrences = countOccurrences(target.content, oldString);
    if (occurrences === 0) {
      return {
        ok: false,
        title: `edit ${target.rel} refused (anchor not found)`,
        output:
          `old_string does not appear in ${target.rel}. Read the file again and copy the text exactly, ` +
          `including indentation. Do not guess at whitespace.`,
      };
    }
    if (occurrences > 1) {
      return {
        ok: false,
        title: `edit ${target.rel} refused (ambiguous anchor)`,
        output:
          `old_string appears ${occurrences} times in ${target.rel}. Include more surrounding lines so the ` +
          `anchor is unique.`,
      };
    }

    const after = target.content.replace(oldString, newString);
    const syntax = await checkSyntaxAfterWrite(after, target.rel, ctx);
    if (syntax) {
      return {
        ok: false,
        title: `edit ${target.rel} refused (would not parse)`,
        output: `${syntax}\n\nThe file was not modified. Fix the edit.`,
      };
    }

    // Anti-pattern scan, reusing the project's verifier rules. Warnings do not
    // block; hard failures do. This is the same policy the batch verifier uses.
    const patterns = patternChecks(
      [{ path: target.rel, before: target.content, after, changedLines: 0 }],
      { rejectNewTodos: false, forbiddenPatterns: [] },
    );
    const hard = patterns.filter((c) => c.severity === 'error');
    if (hard.length > 0) {
      return {
        ok: false,
        title: `edit ${target.rel} refused (anti-pattern)`,
        output: `${hard.map((c) => `- ${c.detail}`).join('\n')}\n\nThe file was not modified.`,
      };
    }

    const verdict = await ctx.approve({
      kind: 'write',
      title: `Edit ${target.rel}`,
      detail: describeChange(target.content, after),
      ...(patterns.length > 0 ? { warning: patterns.map((c) => c.detail).join('; ') } : {}),
    });
    if (verdict === 'deny') {
      return { ok: false, title: `edit ${target.rel} denied`, output: 'The user declined this edit. Ask what they would prefer instead of retrying.' };
    }

    writeFileSync(target.abs, after, 'utf8');
    const changed = addedLines(target.content, after).length;
    return {
      ok: true,
      title: `edit ${target.rel} (~${changed} line${changed === 1 ? '' : 's'} changed)`,
      output: `Applied the edit to ${target.rel}.`,
      meta: {
        files: [target.rel],
        added: changed,
        durationMs: Date.now() - started,
        ...(patterns.length > 0 ? { warnings: patterns.map((c) => c.detail) } : {}),
      },
    };
  },
};

/* ------------------------------------------------------------------ */
/* run_command                                                        */
/* ------------------------------------------------------------------ */

export const runCommandTool: Tool = {
  name: 'run_command',
  description:
    'Run a shell command in the project root and return its output. Use it to run tests, typecheckers, ' +
    'builds, git and formatters — that is how you verify your own work. Do not use it for anything ' +
    'destructive or interactive; there is no TTY and no way to answer a prompt.',
  parameters: {
    type: 'object',
    properties: {
      command: { type: 'string', description: 'The command line to run via the shell.' },
      timeout_ms: { type: 'number', description: 'Timeout in milliseconds. Defaults to 120000, max 600000.' },
    },
    required: ['command'],
  },
  risk: 'exec',
  async run(args, ctx): Promise<ToolResult> {
    const started = Date.now();
    const command = argString(args, 'command');
    if (!command) return { ok: false, title: 'run_command', output: 'command is required' };

    const destructive = /\brm\s+-rf\s+(\/|~|\$HOME)|\bmkfs\b|\bdd\s+if=|:\(\)\{|>\s*\/dev\/sd|git\s+push\s+--force|\bsudo\b/.test(command);
    if (destructive) {
      return {
        ok: false,
        title: 'run_command refused',
        output: 'That command looks destructive or privileged and will not be run automatically. Ask the user to run it themselves.',
      };
    }

    const verdict = await ctx.approve({ kind: 'exec', title: 'Run command', detail: command });
    if (verdict === 'deny') {
      return { ok: false, title: 'command denied', output: 'The user declined to run that command. Propose an alternative or ask them to run it.' };
    }

    const timeoutMs = Math.min(600_000, Math.max(1000, argNumber(args, 'timeout_ms') ?? 120_000));
    const res = await exec('/bin/sh', ['-c', command], {
      cwd: ctx.workspace,
      timeoutMs,
      maxOutputBytes: 512 * 1024,
      lowPriority: true,
      ...(ctx.signal ? { signal: ctx.signal } : {}),
    });

    const combined = `${res.stdout}${res.stderr ? `\n${res.stderr}` : ''}`.trim();
    const capped = capOutput(combined || '(no output)', ctx.maxOutputChars);
    const status = res.timedOut ? `timed out after ${timeoutMs}ms` : `exit ${res.code ?? 'signal'}`;

    return {
      ok: res.code === 0 && !res.timedOut,
      title: `$ ${command.length > 60 ? `${command.slice(0, 57)}…` : command} → ${status}`,
      output: `exit code: ${res.code ?? 'killed'}${res.timedOut ? ' (timed out)' : ''}\n\n${capped.text}`,
      meta: { truncated: capped.truncated, durationMs: Date.now() - started, exitCode: res.code },
    };
  },
};

/* ------------------------------------------------------------------ */
/* shared                                                              */
/* ------------------------------------------------------------------ */

/**
 * Syntax-check a prospective file body using the language's own parser.
 *
 * Returns a human-readable reason when the content would not parse, or null when
 * it is fine (or when no parser is available for that language, in which case we
 * fall back to a delimiter balance check and only block on a clear imbalance).
 */
async function checkSyntaxAfterWrite(content: string, rel: string, ctx: ToolContext): Promise<string | null> {
  const lang = languageOfPath(rel);
  if (!lang) return null;
  const scratch = join(ctx.workspace, '.proto-tmp');
  const result = await syntaxCheck(content, lang, scratch, 15_000);
  if (result.ok) return null;
  if (result.method === 'parser') {
    return `${rel} would not parse as ${lang}: ${result.detail}${result.evidence ? `\n${result.evidence}` : ''}`;
  }
  const balance = balanceCheck(content);
  return balance.ok ? null : `${rel} looks unbalanced: ${balance.detail}`;
}

/** A compact change description for the approval prompt. */
function describeChange(before: string, after: string): string {
  const beforeCount = before.split('\n').length;
  const afterCount = after.split('\n').length;
  const removed = addedLines(after, before).length;
  const added = addedLines(before, after).length;
  return `${beforeCount} → ${afterCount} lines (+${added} / -${removed})`;
}

/* ------------------------------------------------------------------ */
/* registry                                                            */
/* ------------------------------------------------------------------ */

/**
 * The full agent tool set.
 *
 * Ordering is the reading order the prompt recommends, not a priority: the repository
 * tools come first because on a large codebase they should be reached for before
 * `read_file`, and a model that scans the list top-down should meet them in that order.
 */
export function buildToolRegistry(): ToolRegistry {
  const registry = new ToolRegistry();
  for (const tool of repoIndexTools) registry.register(tool);
  return registry
    .register(searchTool)
    .register(readFileTool)
    .register(listFilesTool)
    .register(writeFileTool)
    .register(editFileTool)
    .register(runCommandTool);
}
