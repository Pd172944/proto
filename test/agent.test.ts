/**
 * Agent loop tests.
 *
 * The loop is the part of the harness that can do real damage — it writes files and
 * runs commands — so these tests are about *control flow and consent*, not about
 * whether a model is clever:
 *
 *  - tools run in the order the model asked, and their results are tied back to the
 *    right tool-call id (a mismatch corrupts every later turn),
 *  - a denied approval must leave the file on disk untouched,
 *  - tool failures must come back as information, not as crashes,
 *  - the step budget and the reminder injection must actually fire.
 *
 * Everything runs against a scripted provider, so there is no network and no model.
 */

import { strict as assert } from 'node:assert';
import { readFileSync, writeFileSync } from 'node:fs';
import { join } from 'node:path';
import { describe, it } from 'node:test';

import { runAgentTurn } from '../src/agent/loop.ts';
import type { AgentEvent } from '../src/agent/loop.ts';
import { BASE_SYSTEM_PROMPT, buildSystemPrompt, looksLikeVerification, turnReminder } from '../src/agent/prompt.ts';
import type { ProjectContext } from '../src/agent/prompt.ts';
import { buildToolRegistry, editFileTool, readFileTool, runCommandTool, searchTool, writeFileTool } from '../src/tools/files.ts';
import { capOutput, resolveInsideWorkspace, ToolRegistry } from '../src/tools/types.ts';
import type { ApprovalRequest, ToolContext } from '../src/tools/types.ts';
import { computeCost } from '../src/providers/types.ts';
import type { ChatRequest, ChatResponse, Provider, ToolCall } from '../src/providers/types.ts';
import { tempDir } from './helpers.ts';

/* ------------------------------------------------------------------ */
/* A provider that replays a fixed script of responses                 */
/* ------------------------------------------------------------------ */

interface ScriptStep {
  text?: string;
  toolCalls?: Array<{ name: string; args: Record<string, unknown> }>;
}

class ScriptedProvider implements Provider {
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
  /** Every request the loop made, for asserting on the transcript it sends back. */
  readonly requests: ChatRequest[] = [];
  private index = 0;
  private readonly script: ScriptStep[];

  constructor(script: ScriptStep[]) {
    this.script = script;
  }

  async chat(req: ChatRequest): Promise<ChatResponse> {
    this.requests.push(req);
    const step = this.script[Math.min(this.index, this.script.length - 1)] ?? {};
    this.index++;
    const toolCalls: ToolCall[] = (step.toolCalls ?? []).map((t, i) => ({
      id: `call_${this.index}_${i}`,
      name: t.name,
      args: t.args,
    }));
    const text = step.text ?? '';
    const usage = { inputTokens: 100, outputTokens: 20 };
    return {
      text,
      toolCalls,
      usage,
      finishReason: toolCalls.length > 0 ? 'tool_calls' : 'stop',
      model: this.model,
      providerId: this.id,
      latencyMs: 1,
      costUsd: computeCost(usage, { in: 3, out: 15 }),
    };
  }

  async health(): Promise<{ ok: boolean; detail: string }> {
    return { ok: true, detail: 'scripted' };
  }
}

const PROJECT: ProjectContext = { workspace: '.', platform: 'test' };

function ctxFor(workspace: string, decisions: Array<'allow-once' | 'allow-always' | 'deny'> = []): {
  approve: (req: ApprovalRequest) => Promise<'allow-once' | 'allow-always' | 'deny'>;
  seen: ApprovalRequest[];
} {
  const seen: ApprovalRequest[] = [];
  let i = 0;
  return {
    seen,
    approve: async (req) => {
      seen.push(req);
      const decision = decisions[i] ?? 'allow-once';
      i++;
      return decision;
    },
  };
}

/* ------------------------------------------------------------------ */
/* Tool-level tests                                                    */
/* ------------------------------------------------------------------ */

describe('tools: safety and correctness', () => {
  it('refuses paths that escape the workspace', () => {
    const dir = tempDir();
    assert.equal(resolveInsideWorkspace(dir, '../outside.txt').ok, false);
    assert.equal(resolveInsideWorkspace(dir, '/etc/passwd').ok, false);
    assert.equal(resolveInsideWorkspace(dir, 'src/app.ts').ok, true);
  });

  it('reads with line numbers and reports how much was left', async () => {
    const dir = tempDir();
    writeFileSync(join(dir, 'a.txt'), Array.from({ length: 50 }, (_, i) => `line ${i + 1}`).join('\n'), 'utf8');
    const ctx: ToolContext = { workspace: dir, interactive: false, maxOutputChars: 10_000, approve: async () => 'allow-once' };
    const result = await readFileTool.run({ path: 'a.txt', offset: 10, limit: 5 }, ctx);
    assert.equal(result.ok, true);
    assert.match(result.output, /10│ line 10/);
    assert.match(result.output, /36 more line/);
  });

  it('refuses an edit whose anchor is ambiguous, and says why', async () => {
    const dir = tempDir();
    writeFileSync(join(dir, 'a.py'), 'x = 1\ny = 2\nx = 1\n', 'utf8');
    const ctx: ToolContext = { workspace: dir, interactive: false, maxOutputChars: 10_000, approve: async () => 'allow-once' };
    const result = await editFileTool.run({ path: 'a.py', old_string: 'x = 1', new_string: 'x = 9' }, ctx);
    assert.equal(result.ok, false);
    assert.match(result.output, /appears 2 times/);
    assert.equal(readFileSync(join(dir, 'a.py'), 'utf8'), 'x = 1\ny = 2\nx = 1\n', 'the file must be untouched');
  });

  it('refuses an edit that would not parse, and leaves the file alone', async () => {
    const dir = tempDir();
    writeFileSync(join(dir, 'a.py'), 'def f():\n    return 1\n', 'utf8');
    const ctx: ToolContext = { workspace: dir, interactive: false, maxOutputChars: 10_000, approve: async () => 'allow-once' };
    const result = await editFileTool.run(
      { path: 'a.py', old_string: 'return 1', new_string: 'return (1' },
      ctx,
    );
    // Without a python3 on PATH this falls back to a balance check, which also fails.
    assert.equal(result.ok, false);
    assert.equal(readFileSync(join(dir, 'a.py'), 'utf8'), 'def f():\n    return 1\n');
  });

  it('applies a unique edit and reports the change', async () => {
    const dir = tempDir();
    writeFileSync(join(dir, 'a.py'), 'def total(items):\n    return sum(items)\n', 'utf8');
    const ctx: ToolContext = { workspace: dir, interactive: false, maxOutputChars: 10_000, approve: async () => 'allow-once' };
    const result = await editFileTool.run(
      { path: 'a.py', old_string: 'return sum(items)', new_string: 'return sum(items) if items else 0' },
      ctx,
    );
    assert.equal(result.ok, true, result.output);
    assert.match(readFileSync(join(dir, 'a.py'), 'utf8'), /if items else 0/);
  });

  it('leaves the file untouched when the user denies the edit', async () => {
    const dir = tempDir();
    writeFileSync(join(dir, 'a.py'), 'x = 1\n', 'utf8');
    const ctx: ToolContext = { workspace: dir, interactive: true, maxOutputChars: 10_000, approve: async () => 'deny' };
    const result = await editFileTool.run({ path: 'a.py', old_string: 'x = 1', new_string: 'x = 2' }, ctx);
    assert.equal(result.ok, false);
    assert.match(result.output, /declined/);
    assert.equal(readFileSync(join(dir, 'a.py'), 'utf8'), 'x = 1\n');
  });

  it('blocks destructive commands without even asking', async () => {
    const dir = tempDir();
    let asked = false;
    const ctx: ToolContext = {
      workspace: dir,
      interactive: true,
      maxOutputChars: 10_000,
      approve: async () => {
        asked = true;
        return 'allow-once';
      },
    };
    const result = await runCommandTool.run({ command: 'rm -rf /' }, ctx);
    assert.equal(result.ok, false);
    assert.equal(asked, false, 'the user must never be asked to approve rm -rf /');
  });

  it('runs a command and reports the exit code', async () => {
    const dir = tempDir();
    const ctx: ToolContext = { workspace: dir, interactive: false, maxOutputChars: 10_000, approve: async () => 'allow-once' };
    const result = await runCommandTool.run({ command: 'echo hello && exit 3' }, ctx);
    assert.equal(result.ok, false);
    assert.match(result.output, /exit code: 3/);
    assert.match(result.output, /hello/);
  });

  it('searches file contents and skips binaries and vendor directories', async () => {
    const dir = tempDir();
    writeFileSync(join(dir, 'a.ts'), 'export const needle = 1\n', 'utf8');
    writeFileSync(join(dir, 'b.md'), 'nothing here\n', 'utf8');
    const ctx: ToolContext = { workspace: dir, interactive: false, maxOutputChars: 10_000, approve: async () => 'allow-once' };
    const result = await searchTool.run({ pattern: 'needle' }, ctx);
    assert.equal(result.ok, true);
    assert.match(result.output, /a\.ts:1/);
    assert.doesNotMatch(result.output, /b\.md/);
  });

  it('caps oversized tool output and says what it dropped', () => {
    const capped = capOutput('x'.repeat(5000), 1000);
    assert.equal(capped.truncated, true);
    assert.ok(capped.text.length < 1200);
    assert.match(capped.text, /truncated/);
  });

  it('registers every tool exactly once and groups them by risk', () => {
    const registry = buildToolRegistry();
    const byRisk = registry.byRisk();
    assert.deepEqual(byRisk.read.sort(), ['list_files', 'read_file', 'search']);
    assert.deepEqual(byRisk.write.sort(), ['edit_file', 'write_file']);
    assert.deepEqual(byRisk.exec, ['run_command']);
    assert.equal(registry.specs().length, 6);
    assert.throws(() => registry.register(readFileTool), /duplicate tool/);
  });

  it('produces JSON schemas the model can actually use', () => {
    for (const tool of buildToolRegistry().list()) {
      assert.equal(tool.parameters['type'], 'object', `${tool.name} needs an object schema`);
      const required = tool.parameters['required'];
      // A tool whose arguments are all optional is allowed to omit `required`
      // entirely, but must not declare a malformed one.
      assert.ok(required === undefined || Array.isArray(required), `${tool.name} has a malformed required list`);
      assert.ok(typeof tool.parameters['properties'] === 'object', `${tool.name} needs properties`);
      assert.ok(tool.description.length > 40, `${tool.name} description must explain when to use it`);
    }
  });
});

/* ------------------------------------------------------------------ */
/* Agent loop tests                                                    */
/* ------------------------------------------------------------------ */

describe('agent loop', () => {
  it('executes a scripted read → edit → answer turn', async () => {
    const dir = tempDir();
    writeFileSync(join(dir, 'a.py'), 'def total(items):\n    return sum(items)\n', 'utf8');

    const provider = new ScriptedProvider([
      { toolCalls: [{ name: 'read_file', args: { path: 'a.py' } }] },
      { toolCalls: [{ name: 'edit_file', args: { path: 'a.py', old_string: 'return sum(items)', new_string: 'return sum(items) if items else 0' } }] },
      { text: 'Done: `total` now returns 0 for an empty list.' },
    ]);
    const approvals = ctxFor(dir);
    const events: AgentEvent[] = [];

    const result = await runAgentTurn({
      provider,
      tools: buildToolRegistry(),
      workspace: dir,
      history: [],
      input: 'make total handle an empty list',
      project: { ...PROJECT, workspace: dir },
      approve: approvals.approve,
      onEvent: (e) => events.push(e),
      skipReminders: true,
    });

    assert.equal(result.reason, 'complete');
    assert.equal(result.steps, 3);
    assert.match(result.text, /returns 0 for an empty list/);
    assert.deepEqual(result.editedFiles, ['a.py']);
    assert.equal(result.ranVerification, false);
    assert.match(readFileSync(join(dir, 'a.py'), 'utf8'), /if items else 0/);

    const types = events.map((e) => e.type);
    assert.ok(types.includes('context'));
    assert.equal(types.filter((t) => t === 'tool-start').length, 2);
    assert.equal(types.filter((t) => t === 'tool-end').length, 2);
    assert.ok(types.includes('usage'));
    assert.ok(types.includes('done'));
  });

  it('ties each tool result to the assistant tool-call id', async () => {
    const dir = tempDir();
    writeFileSync(join(dir, 'a.txt'), 'hello\n', 'utf8');
    const provider = new ScriptedProvider([
      { toolCalls: [{ name: 'read_file', args: { path: 'a.txt' } }] },
      { text: 'ok' },
    ]);

    const result = await runAgentTurn({
      provider,
      tools: buildToolRegistry(),
      workspace: dir,
      history: [],
      input: 'read it',
      project: { ...PROJECT, workspace: dir },
      approve: ctxFor(dir).approve,
      skipReminders: true,
    });

    // The second request must contain the tool result, keyed to the call id the
    // model produced. A mismatch here silently corrupts every later turn.
    const second = provider.requests[1];
    assert.ok(second, 'the loop must make a second request after a tool call');
    const toolMessage = second.messages.find((m) => m.role === 'tool');
    assert.ok(toolMessage, 'a tool result must be in the transcript');
    assert.equal(toolMessage.toolCallId, 'call_1_0');
    assert.equal(toolMessage.name, 'read_file');
    assert.match(toolMessage.content, /\[ok\]/);
    assert.match(toolMessage.content, /hello/);
    assert.equal(result.steps, 2);
  });

  it('stops at the step budget instead of looping forever', async () => {
    const dir = tempDir();
    writeFileSync(join(dir, 'a.txt'), 'x\n', 'utf8');
    // A provider that always asks for another read, forever.
    const provider = new ScriptedProvider([{ toolCalls: [{ name: 'read_file', args: { path: 'a.txt' } }] }]);

    const result = await runAgentTurn({
      provider,
      tools: buildToolRegistry(),
      workspace: dir,
      history: [],
      input: 'go',
      maxSteps: 4,
      project: { ...PROJECT, workspace: dir },
      approve: ctxFor(dir).approve,
      skipReminders: true,
    });

    assert.equal(result.reason, 'max-steps');
    assert.equal(result.steps, 4);
    assert.equal(provider.requests.length, 4);
  });

  it('returns an unknown tool as an actionable error, not a crash', async () => {
    const dir = tempDir();
    const provider = new ScriptedProvider([
      { toolCalls: [{ name: 'make_coffee', args: {} }] },
      { text: 'sorry' },
    ]);
    const events: AgentEvent[] = [];
    const result = await runAgentTurn({
      provider,
      tools: buildToolRegistry(),
      workspace: dir,
      history: [],
      input: 'coffee',
      project: { ...PROJECT, workspace: dir },
      approve: ctxFor(dir).approve,
      onEvent: (e) => events.push(e),
      skipReminders: true,
    });

    assert.equal(result.reason, 'complete');
    const toolEnd = events.find((e) => e.type === 'tool-end');
    assert.ok(toolEnd && toolEnd.type === 'tool-end');
    assert.equal(toolEnd.result.ok, false);
    assert.match(toolEnd.result.output, /Available tools: .*read_file/);
  });

  it('propagates a denial to the model rather than failing the turn', async () => {
    const dir = tempDir();
    writeFileSync(join(dir, 'a.py'), 'x = 1\n', 'utf8');
    const provider = new ScriptedProvider([
      { toolCalls: [{ name: 'edit_file', args: { path: 'a.py', old_string: 'x = 1', new_string: 'x = 2' } }] },
      { text: 'understood, I will not change it' },
    ]);
    const result = await runAgentTurn({
      provider,
      tools: buildToolRegistry(),
      workspace: dir,
      history: [],
      input: 'change it',
      project: { ...PROJECT, workspace: dir },
      approve: ctxFor(dir, ['deny']).approve,
      skipReminders: true,
    });

    assert.equal(result.reason, 'complete');
    assert.deepEqual(result.editedFiles, [], 'a denied edit must not count as an edit');
    assert.equal(readFileSync(join(dir, 'a.py'), 'utf8'), 'x = 1\n');
    const toolMessage = provider.requests[1]?.messages.find((m) => m.role === 'tool');
    assert.match(toolMessage?.content ?? '', /declined/);
  });

  it('aborts cleanly when the signal fires', async () => {
    const dir = tempDir();
    const controller = new AbortController();
    controller.abort();
    const provider = new ScriptedProvider([{ text: 'should not run' }]);
    const result = await runAgentTurn({
      provider,
      tools: buildToolRegistry(),
      workspace: dir,
      history: [],
      input: 'go',
      project: { ...PROJECT, workspace: dir },
      approve: ctxFor(dir).approve,
      signal: controller.signal,
      skipReminders: true,
    });
    assert.equal(result.reason, 'aborted');
    assert.equal(provider.requests.length, 0);
  });

  it('accumulates usage and cost across steps', async () => {
    const dir = tempDir();
    writeFileSync(join(dir, 'a.txt'), 'x\n', 'utf8');
    const provider = new ScriptedProvider([
      { toolCalls: [{ name: 'read_file', args: { path: 'a.txt' } }] },
      { text: 'done' },
    ]);
    const result = await runAgentTurn({
      provider,
      tools: buildToolRegistry(),
      workspace: dir,
      history: [],
      input: 'go',
      project: { ...PROJECT, workspace: dir },
      approve: ctxFor(dir).approve,
      skipReminders: true,
    });
    assert.equal(result.usage.inputTokens, 200);
    assert.equal(result.usage.outputTokens, 40);
    assert.ok(result.costUsd > 0);
  });

  it('injects a verification reminder after an edit that was never checked', async () => {
    const dir = tempDir();
    writeFileSync(join(dir, 'a.py'), 'x = 1\n', 'utf8');
    const provider = new ScriptedProvider([
      { toolCalls: [{ name: 'edit_file', args: { path: 'a.py', old_string: 'x = 1', new_string: 'x = 2' } }] },
      { text: 'finished' },
    ]);
    const events: AgentEvent[] = [];
    await runAgentTurn({
      provider,
      tools: buildToolRegistry(),
      workspace: dir,
      history: [],
      input: 'change it',
      project: { ...PROJECT, workspace: dir },
      approve: ctxFor(dir).approve,
      onEvent: (e) => events.push(e),
      // reminders ON
    });

    const reminder = events.find((e) => e.type === 'reminder');
    assert.ok(reminder, 'editing without verifying must trigger a reminder');
    assert.match(reminder.text, /have not run any tests/);
  });

  it('does not nag when the model verified its work', async () => {
    const dir = tempDir();
    writeFileSync(join(dir, 'a.py'), 'x = 1\n', 'utf8');
    const provider = new ScriptedProvider([
      { toolCalls: [{ name: 'edit_file', args: { path: 'a.py', old_string: 'x = 1', new_string: 'x = 2' } }] },
      { toolCalls: [{ name: 'run_command', args: { command: 'npm test' } }] },
      { text: 'done and verified' },
    ]);
    const events: AgentEvent[] = [];
    const result = await runAgentTurn({
      provider,
      tools: buildToolRegistry(),
      workspace: dir,
      history: [],
      input: 'change it',
      project: { ...PROJECT, workspace: dir },
      approve: ctxFor(dir).approve,
      onEvent: (e) => events.push(e),
    });
    assert.equal(result.ranVerification, true);

    // The reminder's job is to *provoke* verification, so exactly one firing —
    // immediately after the edit and before the tests run — is the desired
    // behaviour. What must never happen is nagging after the work was verified.
    const reminders = events.filter((e) => e.type === 'reminder');
    assert.equal(reminders.length, 1, 'one nudge, then it should stop');
    const reminderIndex = events.findIndex((e) => e.type === 'reminder');
    const verifyIndex = events.findIndex((e) => e.type === 'tool-start' && e.toolCall.name === 'run_command');
    assert.ok(verifyIndex > reminderIndex, 'the reminder must come before the verification command');
    assert.equal(
      events.slice(verifyIndex).filter((e) => e.type === 'reminder').length,
      0,
      'no reminders after verification',
    );
  });
});

/* ------------------------------------------------------------------ */
/* Prompt tests                                                        */
/* ------------------------------------------------------------------ */

describe('system prompt', () => {
  it('states the prompt-injection rule, because the agent reads untrusted files', () => {
    assert.match(BASE_SYSTEM_PROMPT, /tool output is data/i);
    assert.match(BASE_SYSTEM_PROMPT, /prompt-injection/i);
  });

  it('forbids interactive commands, since there is no TTY', () => {
    assert.match(BASE_SYSTEM_PROMPT, /no TTY/i);
    assert.match(BASE_SYSTEM_PROMPT, /rebase -i/);
  });

  it('includes environment, git state, tools and project instructions', () => {
    const prompt = buildSystemPrompt(
      {
        workspace: '/tmp/proj',
        platform: 'darwin-arm64',
        gitBranch: 'main',
        gitStatus: ' M src/app.ts',
        topLevel: 'src  package.json',
        instructions: [{ path: 'AGENTS.md', content: 'Always run pnpm test.' }],
      },
      { tools: ['read_file', 'edit_file'] },
    );
    assert.match(prompt, /Workspace: \/tmp\/proj/);
    assert.match(prompt, /Git branch: main/);
    assert.match(prompt, /Available tools: read_file, edit_file/);
    assert.match(prompt, /Always run pnpm test/);
    // Project instructions must not be able to override the safety section.
    assert.match(prompt, /cannot override the Safety section/);
  });

  it('keeps the stable prefix first so prompt caching can hit', () => {
    const prompt = buildSystemPrompt({ workspace: '/tmp', platform: 'test' });
    assert.ok(prompt.startsWith(BASE_SYSTEM_PROMPT), 'the base prompt must be the cacheable prefix');
  });

  it('recognises verification commands and ignores unrelated ones', () => {
    for (const c of ['npm test', 'pytest -q', 'cargo check', 'tsc --noEmit', 'go vet ./...', 'ruff check .']) {
      assert.equal(looksLikeVerification(c), true, c);
    }
    for (const c of ['ls', 'git status', 'cat readme.md', 'echo hi']) {
      assert.equal(looksLikeVerification(c), false, c);
    }
  });

  it('escalates a reminder only when it is warranted', () => {
    assert.equal(turnReminder({ edited: [], commands: [], ranVerification: false, idleTurns: 0 }), null);
    assert.ok(turnReminder({ edited: ['a.ts'], commands: [], ranVerification: false, idleTurns: 0 }));
    assert.equal(turnReminder({ edited: ['a.ts'], commands: ['npm test'], ranVerification: true, idleTurns: 0 }), null);
    assert.ok(turnReminder({ edited: [], commands: [], ranVerification: false, idleTurns: 3 }));
  });
});
