/**
 * Repository-intelligence tools.
 *
 * These exist because of a specific failure mode: an agent on a large codebase that has
 * no way to see its shape falls back to reading files. It reads `index.ts`, then
 * `app.ts`, then something that looked related, and six tool calls in it has spent the
 * context window on files that were not the answer. The model was not being lazy — it
 * had no cheaper way to find out what existed.
 *
 * So each tool here answers a question that would otherwise cost a file read:
 *
 *  - `repo_map` answers "what is in this repository and what matters for my task",
 *    bounded to a token budget the caller controls.
 *  - `find_symbol` answers "where is this defined" without matching the comment that
 *    mentions it, the string that contains it, or the twelve other files that merely
 *    use the name.
 *  - `find_references` answers "what breaks if I change this", which is the question that
 *    actually prevents a bad edit, and which a text search answers badly.
 *  - `file_outline` answers "what is in this file" for a tenth of the cost of the file.
 *
 * Every one of them degrades: if the index failed to build, they say so and either fall
 * back to a direct scan or tell the model to use `search` instead. None of them returns
 * an empty result that could be mistaken for "this does not exist in the repository".
 */

import { capOutput } from './types.ts';
import type { Tool, ToolContext, ToolResult } from './types.ts';
import { focusFromTask } from '../index/index.ts';
import { estimateTokens } from '../index/repomap.ts';

function argString(args: Record<string, unknown>, key: string): string | undefined {
  const v = args[key];
  return typeof v === 'string' && v.trim() !== '' ? v.trim() : undefined;
}

function argNumber(args: Record<string, unknown>, key: string): number | undefined {
  const v = args[key];
  if (typeof v === 'number' && Number.isFinite(v)) return v;
  if (typeof v === 'string' && v.trim() !== '' && Number.isFinite(Number(v))) return Number(v);
  return undefined;
}

/** Shared preamble: make sure a build has been attempted, and report if it failed. */
async function withIndex(ctx: ToolContext): Promise<{ index: NonNullable<ToolContext['index']>; warning: string } | null> {
  const index = ctx.index;
  if (index === undefined) return null;
  await index.ensure();
  const warning = index.warnings.join('; ');
  return { index, warning };
}

const NO_INDEX =
  'The codebase index is unavailable for this session, so this tool cannot answer. ' +
  'Use `search` with a pattern instead — it falls back to a direct scan.';

/* ------------------------------------------------------------------ */
/* repo_map                                                            */
/* ------------------------------------------------------------------ */

export const repoMapTool: Tool = {
  name: 'repo_map',
  description:
    'Show the shape of the repository: definition signatures for the files most relevant to a task, ' +
    'ranked by relevance and cut to a token budget. Use this FIRST on an unfamiliar or large codebase, ' +
    'and again whenever you need to know what else touches the area you are working in. It is far cheaper ' +
    'than reading files one by one, and it is the only practical way to see a large repository.',
  parameters: {
    type: 'object',
    properties: {
      focus: {
        type: 'string',
        description:
          'What you are working on — a task description, or the files and symbol names involved. ' +
          'This drives the ranking, so pass the actual request rather than a single keyword.',
      },
      budget_tokens: {
        type: 'number',
        description: 'Approximate token budget for the map. Defaults to 2000.',
      },
    },
    required: [],
  },
  risk: 'read',
  async run(args, ctx): Promise<ToolResult> {
    const started = Date.now();
    const prepared = await withIndex(ctx);
    if (prepared === null) return { ok: false, title: 'repo_map', output: NO_INDEX };
    const { index, warning } = prepared;

    if (!index.ready) {
      return {
        ok: false,
        title: 'repo_map — index empty',
        output: `No files were indexed under ${ctx.workspace}${warning === '' ? '' : ` (${warning})`}. Use list_files or search instead.`,
      };
    }

    const budgetTokens = Math.max(200, Math.min(20_000, argNumber(args, 'budget_tokens') ?? 2000));
    const focusText = argString(args, 'focus') ?? '';
    const focus = focusFromTask(focusText, index.graph);

    const map = index.repoMap({
      budgetChars: budgetTokens * 4,
      focus: focus.paths,
      focusSymbols: focus.symbols,
    });

    const stats = index.stats;
    const header = [
      `repository map — ${map.files.length} of ${map.totalFiles} ranked file(s) with definitions` +
        `, ~${estimateTokens(map.text)} tokens`,
      `focus: ${focus.paths.length + focus.symbols.length === 0 ? '(none given — ranking is repository-wide)' : [...focus.paths, ...focus.symbols].join(', ')}`,
      '',
    ].join('\n');

    const body = header + map.text;
    const capped = capOutput(body, ctx.maxOutputChars);
    return {
      ok: true,
      title: `repo_map → ${map.files.length} file(s), ~${estimateTokens(capped.text)} tokens`,
      output: capped.text,
      meta: {
        truncated: capped.truncated || map.truncated,
        durationMs: Date.now() - started,
        files: map.files.map((f) => f.path),
        indexed: stats.files,
        buildMs: stats.buildMs,
        focusPaths: focus.paths,
        focusSymbols: focus.symbols,
      },
    };
  },
};

/* ------------------------------------------------------------------ */
/* find_symbol                                                         */
/* ------------------------------------------------------------------ */

export const findSymbolTool: Tool = {
  name: 'find_symbol',
  description:
    'Find where a function, class, method, type or constant is DEFINED. Unlike a text search this ' +
    'ignores comments and strings and does not return the files that merely use the name. Use it when ' +
    'you know what a thing is called and need to know where it lives.',
  parameters: {
    type: 'object',
    properties: {
      name: { type: 'string', description: 'Exact symbol name. A partial name also works as a fallback.' },
      limit: { type: 'number', description: 'Maximum definitions to return. Defaults to 20.' },
    },
    required: ['name'],
  },
  risk: 'read',
  async run(args, ctx): Promise<ToolResult> {
    const started = Date.now();
    const name = argString(args, 'name');
    if (name === undefined) return { ok: false, title: 'find_symbol', output: 'name is required' };

    const prepared = await withIndex(ctx);
    if (prepared === null) return { ok: false, title: `find_symbol ${name}`, output: NO_INDEX };
    const { index } = prepared;

    const limit = Math.max(1, Math.min(100, argNumber(args, 'limit') ?? 20));
    const { hits, exact } = index.findSymbol(name, limit);

    if (hits.length === 0) {
      return {
        ok: true,
        title: `find_symbol ${name} → not found`,
        output:
          `No definition of \`${name}\` in the index (${index.stats.files} files). ` +
          'It may come from a dependency, or be named differently. Try `search` with a looser pattern.',
        meta: { durationMs: Date.now() - started, files: [] },
      };
    }

    const lines = hits.map((h) => `${h.path}:${h.line}  ${h.kind}${h.signature === undefined || h.signature === '' ? '' : `  ${h.signature}`}`);
    const lead = exact
      ? `definitions of \`${name}\`:`
      : `no exact match for \`${name}\`; names containing it:`;
    const capped = capOutput(`${lead}\n${lines.join('\n')}`, ctx.maxOutputChars);

    return {
      ok: true,
      title: `find_symbol ${name} → ${hits.length}${exact ? '' : ' (fuzzy)'}`,
      output: capped.text,
      meta: { truncated: capped.truncated, durationMs: Date.now() - started, files: hits.map((h) => h.path), exact },
    };
  },
};

/* ------------------------------------------------------------------ */
/* find_references                                                     */
/* ------------------------------------------------------------------ */

export const findReferencesTool: Tool = {
  name: 'find_references',
  description:
    'Every place a symbol is USED across the repository, excluding the files that define it. Use this ' +
    'before renaming or changing a signature: it is the fastest way to find every call site that would ' +
    'break. It is much more reliable than a text search, which cannot tell a call from a comment.',
  parameters: {
    type: 'object',
    properties: {
      name: { type: 'string', description: 'Exact symbol name.' },
      limit: { type: 'number', description: 'Maximum references to return. Defaults to 60.' },
    },
    required: ['name'],
  },
  risk: 'read',
  async run(args, ctx): Promise<ToolResult> {
    const started = Date.now();
    const name = argString(args, 'name');
    if (name === undefined) return { ok: false, title: 'find_references', output: 'name is required' };

    const prepared = await withIndex(ctx);
    if (prepared === null) return { ok: false, title: `find_references ${name}`, output: NO_INDEX };
    const { index } = prepared;

    const limit = Math.max(1, Math.min(300, argNumber(args, 'limit') ?? 60));
    const hits = index.findReferences(name, limit);

    if (hits.length === 0) {
      const defined = index.findSymbol(name, 1).hits.length > 0;
      return {
        ok: true,
        title: `find_references ${name} → none`,
        output: defined
          ? `\`${name}\` is defined but never referenced elsewhere in the indexed files. It may be part of a public API, or unused.`
          : `\`${name}\` is not defined anywhere in the index, so there is nothing to find references to. Check the spelling, or use \`search\`.`,
        meta: { durationMs: Date.now() - started },
      };
    }

    const byFile = new Map<string, Array<{ line: number; text: string }>>();
    for (const h of hits) {
      const list = byFile.get(h.path);
      if (list === undefined) byFile.set(h.path, [{ line: h.line, text: h.text ?? '' }]);
      else list.push({ line: h.line, text: h.text ?? '' });
    }

    const out: string[] = [`references to \`${name}\` in ${byFile.size} file(s):`];
    for (const [path, refs] of byFile) {
      out.push(`\n${path}`);
      for (const r of refs) out.push(`  ${r.line}: ${r.text}`);
    }

    const capped = capOutput(out.join('\n'), ctx.maxOutputChars);
    return {
      ok: true,
      title: `find_references ${name} → ${hits.length} in ${byFile.size} file(s)`,
      output: capped.text,
      meta: {
        truncated: capped.truncated || hits.length >= limit,
        durationMs: Date.now() - started,
        files: [...byFile.keys()],
      },
    };
  },
};

/* ------------------------------------------------------------------ */
/* file_outline                                                        */
/* ------------------------------------------------------------------ */

export const fileOutlineTool: Tool = {
  name: 'file_outline',
  description:
    'List the definitions in one file, with line numbers and signatures, without returning its body. ' +
    'Use it to decide which range of a large file to read, instead of reading the whole thing.',
  parameters: {
    type: 'object',
    properties: {
      path: { type: 'string', description: 'File path, relative to the project root.' },
    },
    required: ['path'],
  },
  risk: 'read',
  async run(args, ctx): Promise<ToolResult> {
    const started = Date.now();
    const raw = argString(args, 'path');
    if (raw === undefined) return { ok: false, title: 'file_outline', output: 'path is required' };

    const prepared = await withIndex(ctx);
    if (prepared === null) return { ok: false, title: `file_outline ${raw}`, output: NO_INDEX };
    const { index } = prepared;

    const outline = index.outline(raw);
    if (outline === null) {
      return {
        ok: false,
        title: `file_outline ${raw}`,
        output: `\`${raw}\` is not in the index — it may be ignored, generated, or outside the project root. Use read_file to read it directly.`,
      };
    }

    const capped = capOutput(outline, ctx.maxOutputChars);
    return {
      ok: true,
      title: `file_outline ${raw}`,
      output: capped.text,
      meta: { truncated: capped.truncated, durationMs: Date.now() - started, files: [raw] },
    };
  },
};

export const repoIndexTools: Tool[] = [repoMapTool, findSymbolTool, findReferencesTool, fileOutlineTool];
