/**
 * Integration tests for the repository-intelligence tools.
 *
 * The index's own tests cover ranking and extraction; these cover the wiring — that the
 * registry exposes the tools, that they accept the index through `ToolContext`, that they
 * report rather than swallow a missing index, and that the answers they return are the
 * ones a model would need to act on. That last distinction is the point: a tool that
 * returns an empty string with `ok: true` would leave the model concluding that a symbol
 * does not exist, which is the failure mode this whole subsystem was built to remove.
 */

import { strict as assert } from 'node:assert';
import { mkdirSync, mkdtempSync, rmSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { describe, it } from 'node:test';

import { CodebaseIndex } from '../src/index/index.ts';
import { runAgentTurn } from '../src/agent/loop.ts';
import type { ChatResponse, Provider } from '../src/providers/types.ts';
import { buildToolRegistry } from '../src/tools/files.ts';
import type { ToolContext } from '../src/tools/types.ts';

/** A small but realistic multi-file project, written to a temp directory. */
function project(): string {
  const root = mkdtempSync(join(tmpdir(), 'proto-tools-test-'));
  mkdirSync(join(root, 'src'), { recursive: true });
  writeFileSync(
    join(root, 'src', 'math_utils.py'),
    [
      'def calc_total(items):',
      '    """Sum the items."""',
      '    return sum(items)',
      '',
      '',
      'def calc_average(items):',
      '    return calc_total(items) / len(items)',
      '',
    ].join('\n'),
  );
  writeFileSync(
    join(root, 'src', 'report.py'),
    [
      'from math_utils import calc_total',
      '',
      '',
      'def summarize(items):',
      '    return {"count": len(items), "total": calc_total(items)}',
      '',
      '',
      '# calc_total is mentioned in this comment and must not be reported as a use',
      'NOTE = "calc_total"',
      '',
    ].join('\n'),
  );
  writeFileSync(
    join(root, 'src', 'cli.py'),
    ['from report import summarize', '', '', 'def main(argv):', '    return summarize(argv)', ''].join('\n'),
  );
  return root;
}

async function harness(root: string): Promise<{ ctx: ToolContext; index: CodebaseIndex; cleanup: () => void }> {
  const dataDir = join(root, '.data');
  mkdirSync(dataDir, { recursive: true });
  const index = new CodebaseIndex(root, dataDir);
  await index.refresh();
  const ctx: ToolContext = {
    workspace: root,
    interactive: false,
    maxOutputChars: 24_000,
    approve: async () => 'deny',
    index,
  };
  return { ctx, index, cleanup: () => rmSync(root, { recursive: true, force: true }) };
}

describe('repo tools are registered', () => {
  it('exposes all four as read-risk tools', () => {
    const registry = buildToolRegistry();
    for (const name of ['repo_map', 'find_symbol', 'find_references', 'file_outline']) {
      const tool = registry.get(name);
      assert.ok(tool !== undefined, `${name} is missing from the registry`);
      assert.equal(tool.risk, 'read', `${name} must be a read`);
      assert.equal(tool.parameters['type'], 'object');
    }
  });

  it('tells the model when to use each one, not just what it does', () => {
    const registry = buildToolRegistry();
    for (const name of ['repo_map', 'find_symbol', 'find_references', 'file_outline']) {
      const tool = registry.get(name);
      assert.ok(
        /use (it|this)/i.test(tool?.description ?? ''),
        `${name} description does not say when to use it`,
      );
    }
  });
});

describe('find_symbol', () => {
  it('finds a definition and reports its file, line and kind', async () => {
    const root = project();
    const { ctx, cleanup } = await harness(root);
    try {
      const result = await buildToolRegistry().get('find_symbol')?.run({ name: 'calc_total' }, ctx);
      assert.equal(result?.ok, true);
      assert.match(result?.output ?? '', /src\/math_utils\.py:1/);
      assert.match(result?.output ?? '', /function/);
    } finally {
      cleanup();
    }
  });

  it('falls back to a partial match and says the match was not exact', async () => {
    const root = project();
    const { ctx, cleanup } = await harness(root);
    try {
      const result = await buildToolRegistry().get('find_symbol')?.run({ name: 'calc_' }, ctx);
      assert.match(result?.title ?? '', /fuzzy/);
      assert.match(result?.output ?? '', /calc_total/);
    } finally {
      cleanup();
    }
  });

  it('says plainly when a name is not defined rather than returning nothing', async () => {
    const root = project();
    const { ctx, cleanup } = await harness(root);
    try {
      const result = await buildToolRegistry().get('find_symbol')?.run({ name: 'no_such_symbol' }, ctx);
      assert.equal(result?.ok, true);
      assert.match(result?.output ?? '', /No definition/i);
      // It must point at a next step, not leave the model stuck.
      assert.match(result?.output ?? '', /search/);
    } finally {
      cleanup();
    }
  });
});

describe('find_references', () => {
  it('finds real uses, with line numbers, and excludes the defining file', async () => {
    const root = project();
    const { ctx, cleanup } = await harness(root);
    try {
      const result = await buildToolRegistry().get('find_references')?.run({ name: 'calc_total' }, ctx);
      assert.equal(result?.ok, true);
      const out = result?.output ?? '';
      assert.match(out, /src\/report\.py/, 'the use in report.py should be found');
      assert.ok(!/math_utils\.py:/.test(out), 'the defining file should not be listed as a reference');
      // The comment and the string in report.py must not inflate the count: only line 5
      // is a real call.
      const reportLines = out.split('\n').filter((l) => /^\s+\d+:/.test(l));
      assert.ok(
        reportLines.every((l) => !l.includes('NOTE') && !l.includes('#')),
        `comment/string counted as a reference:\n${out}`,
      );
    } finally {
      cleanup();
    }
  });

  it('distinguishes "defined but unused" from "does not exist"', async () => {
    const root = project();
    const { ctx, cleanup } = await harness(root);
    try {
      const unused = await buildToolRegistry().get('find_references')?.run({ name: 'calc_average' }, ctx);
      assert.match(unused?.output ?? '', /never referenced/i);

      const missing = await buildToolRegistry().get('find_references')?.run({ name: 'ghost' }, ctx);
      assert.match(missing?.output ?? '', /not defined/i);
    } finally {
      cleanup();
    }
  });
});

describe('file_outline', () => {
  it('lists definitions with line numbers without returning the body', async () => {
    const root = project();
    const { ctx, cleanup } = await harness(root);
    try {
      const result = await buildToolRegistry().get('file_outline')?.run({ path: 'src/math_utils.py' }, ctx);
      assert.equal(result?.ok, true);
      assert.match(result?.output ?? '', /calc_total/);
      assert.match(result?.output ?? '', /calc_average/);
      assert.ok(!(result?.output ?? '').includes('return sum(items)'), 'the body leaked into the outline');
    } finally {
      cleanup();
    }
  });

  it('fails with a usable message for a file that is not indexed', async () => {
    const root = project();
    const { ctx, cleanup } = await harness(root);
    try {
      const result = await buildToolRegistry().get('file_outline')?.run({ path: 'nope.ts' }, ctx);
      assert.equal(result?.ok, false);
      assert.match(result?.output ?? '', /not in the index/i);
      assert.match(result?.output ?? '', /read_file/);
    } finally {
      cleanup();
    }
  });
});

describe('repo_map', () => {
  it('ranks the file a task names first and shows signatures', async () => {
    const root = project();
    const { ctx, cleanup } = await harness(root);
    try {
      const result = await buildToolRegistry().get('repo_map')?.run(
        { focus: 'change how src/report.py summarizes totals', budget_tokens: 800 },
        ctx,
      );
      assert.equal(result?.ok, true);
      const out = result?.output ?? '';
      assert.match(out, /src\/report\.py/);
      assert.match(out, /named by the task/);
      assert.match(out, /summarize/, 'signatures should appear, not just file names');
      assert.equal(result?.meta?.['indexed'], 3);
    } finally {
      cleanup();
    }
  });

  it('reports the focus it derived, so a wrong ranking is diagnosable', async () => {
    const root = project();
    const { ctx, cleanup } = await harness(root);
    try {
      const result = await buildToolRegistry().get('repo_map')?.run({ focus: 'fix calc_total in src/math_utils.py' }, ctx);
      assert.match(result?.output ?? '', /focus:/);
      assert.match(result?.output ?? '', /calc_total/);
    } finally {
      cleanup();
    }
  });

  it('says so when there is nothing to map instead of returning an empty document', async () => {
    const root = mkdtempSync(join(tmpdir(), 'proto-tools-empty-'));
    const { ctx, cleanup } = await harness(root);
    try {
      const result = await buildToolRegistry().get('repo_map')?.run({}, ctx);
      assert.equal(result?.ok, false);
      assert.match(result?.output ?? '', /No files were indexed/i);
    } finally {
      cleanup();
    }
  });
});

describe('degrading without an index', () => {
  it('every repo tool refuses clearly rather than returning a misleading empty answer', async () => {
    // This is the property that keeps the index an accelerator instead of a dependency:
    // a session without one must get an actionable error, not "no such symbol".
    const registry = buildToolRegistry();
    const ctx: ToolContext = {
      workspace: process.cwd(),
      interactive: false,
      maxOutputChars: 4000,
      approve: async () => 'deny',
    };
    for (const [name, args] of [
      ['repo_map', {}],
      ['find_symbol', { name: 'anything' }],
      ['find_references', { name: 'anything' }],
      ['file_outline', { path: 'anything.ts' }],
    ] as Array<[string, Record<string, unknown>]>) {
      const result = await registry.get(name)?.run(args, ctx);
      assert.equal(result?.ok, false, `${name} should fail without an index`);
      assert.match(result?.output ?? '', /index is unavailable/i, `${name} gave an unhelpful message`);
      assert.match(result?.output ?? '', /search/, `${name} should name a working alternative`);
    }
  });
});

/* ------------------------------------------------------------------ */
/* the loop must actually hand the index to the tools                  */
/* ------------------------------------------------------------------ */

/**
 * A scripted provider that calls one tool and then answers.
 *
 * This exists to catch the one failure the tool tests above cannot: `runAgentTurn` builds
 * the `ToolContext` itself, so an index that never reaches it would leave every repository
 * tool reporting "unavailable" — silently, and only at runtime, in exactly the situation
 * the index was built for.
 */
class OneToolProvider implements Provider {
  readonly id = 'scripted';
  readonly label = 'Scripted';
  readonly kind = 'cloud' as const;
  model = 'scripted-model';
  readonly capabilities = {
    tools: true,
    jsonSchema: false,
    streaming: false,
    promptCaching: false,
    contextWindow: 200_000,
    maxOutputTokens: 8192,
  };
  private index = 0;
  private readonly toolName: string;
  private readonly args: Record<string, unknown>;

  constructor(toolName: string, args: Record<string, unknown>) {
    this.toolName = toolName;
    this.args = args;
  }

  async chat(): Promise<ChatResponse> {
    this.index++;
    if (this.index === 1) {
      return {
        text: '',
        toolCalls: [{ id: 'call_1', name: this.toolName, args: this.args }],
        usage: { inputTokens: 10, outputTokens: 5 },
        finishReason: 'tool_calls',
        model: this.model,
        providerId: this.id,
        latencyMs: 1,
        costUsd: 0,
      };
    }
    return {
      text: 'done',
      toolCalls: [],
      usage: { inputTokens: 10, outputTokens: 5 },
      finishReason: 'stop',
      model: this.model,
      providerId: this.id,
      latencyMs: 1,
      costUsd: 0,
    };
  }

  async health(): Promise<{ ok: boolean; detail: string }> {
    return { ok: true, detail: 'scripted' };
  }
}

describe('the agent loop passes the index through to the tools', () => {
  it('lets a scripted repo_map call succeed instead of reporting no index', async () => {
    const root = project();
    const dataDir = join(root, '.data');
    mkdirSync(dataDir, { recursive: true });
    const index = new CodebaseIndex(root, dataDir);
    await index.refresh();
    try {
      const results: Array<{ ok: boolean; output: string }> = [];
      const result = await runAgentTurn({
        provider: new OneToolProvider('repo_map', { focus: 'src/report.py' }),
        tools: buildToolRegistry(),
        workspace: root,
        history: [],
        input: 'what is in this project?',
        approve: async () => 'allow-once',
        interactive: false,
        maxSteps: 4,
        deadlineMs: 30_000,
        temperature: 0,
        index,
        onEvent: (event) => {
          if (event.type === 'tool-end') results.push({ ok: event.result.ok, output: event.result.output });
        },
      });
      assert.equal(result.text, 'done');
      assert.equal(results.length, 1, 'the scripted tool call should have run once');
      assert.equal(results[0]?.ok, true, `repo_map failed: ${results[0]?.output}`);
      assert.ok(!/unavailable/i.test(results[0]?.output ?? ''), 'the index did not reach the tool');
      assert.match(results[0]?.output ?? '', /report\.py/);
    } finally {
      rmSync(root, { recursive: true, force: true });
    }
  });
});
