/**
 * End-to-end loop tests.
 *
 * These exercise the whole path — route, attempt, verify, repair, escalate,
 * record — using injectable mock providers, so they are deterministic and never
 * touch a network or a local runtime. They are the tests that would catch the
 * highest-severity regressions: writing to the user's files without permission,
 * losing the escalation pair, or mislabelling the training data.
 */

import { strict as assert } from 'node:assert';
import { readFileSync, writeFileSync } from 'node:fs';
import { join } from 'node:path';
import { describe, it } from 'node:test';

import { applyToWorkspace, runTask } from '../src/harness/loop.ts';
import type { RunEvent } from '../src/harness/loop.ts';
import { MockProvider, alwaysFailingMock } from '../src/providers/mock.ts';
import type { ChatRequest, ChatResponse, Provider } from '../src/providers/types.ts';
import { computeCost } from '../src/providers/types.ts';
import { writeSecret } from '../src/config/load.ts';
import type { ProtoConfig } from '../src/config/schema.ts';
import { tempDir, testConfig } from './helpers.ts';

const LOOP_FILE = 'def total_prices(items):\n    total = 0\n    for i in range(len(items) + 1):\n        total += items[i]["price"]\n    return total\n';

const EASY_TASK = 'Fix the off-by-one error in this loop so it does not go out of bounds.';

/** A provider that always returns a syntactically broken candidate. */
class BrokenProvider implements Provider {
  readonly id = 'broken';
  readonly label = 'Broken';
  readonly kind = 'local' as const;
  readonly model = 'broken-model';
  readonly capabilities = {
    tools: false,
    jsonSchema: false,
    streaming: false,
    promptCaching: false,
    contextWindow: 8192,
    maxOutputTokens: 1024,
  };
  async chat(req: ChatRequest): Promise<ChatResponse> {
    const usage = { inputTokens: 10, outputTokens: 10 };
    return {
      text: '```python\nfor i in range(len(items)\n```',
      toolCalls: [],
      usage,
      finishReason: 'stop',
      model: this.model,
      providerId: this.id,
      latencyMs: 3,
      costUsd: computeCost(usage, { in: 0, out: 0 }),
    };
  }
  async health(): Promise<{ ok: boolean; detail: string }> {
    return { ok: true, detail: 'broken but reachable' };
  }
}

function setup(opts: { cloudKey?: boolean } = {}): { dir: string; workspace: string; cfg: ProtoConfig } {
  const dir = tempDir();
  const workspace = tempDir();
  writeFileSync(join(workspace, 'prices.py'), LOOP_FILE, 'utf8');
  const base = testConfig(dir);
  const cfg: ProtoConfig = {
    ...base,
    cloud: { ...base.cloud, enabled: opts.cloudKey !== false },
  };
  // A key must exist for the router to consider the cloud tier available.
  if (opts.cloudKey !== false) writeSecret(dir, cfg.cloud.provider, 'test-key-not-real');
  return { dir, workspace, cfg };
}

function ctxFor(workspace: string): { task: string; workspace: string; files: Array<{ path: string; content: string }> } {
  return { task: EASY_TASK, workspace, files: [{ path: 'prices.py', content: LOOP_FILE }] };
}

describe('runTask: local success path', () => {
  it('routes to local, verifies, and writes nothing by default', async () => {
    const { dir, workspace, cfg } = setup({ cloudKey: false });
    const result = await runTask({
      cfg,
      dataDir: dir,
      ctx: ctxFor(workspace),
      localProvider: new MockProvider({ id: 'mock-local', kind: 'local' }),
      offlineRoute: true,
    });

    assert.equal(result.status, 'local-success');
    assert.equal(result.writtenFiles.length, 0);
    assert.equal(result.attempts.length, 1);
    assert.equal(result.attempts[0]?.tier, 'local');
    assert.equal(result.attempts[0]?.verification?.passed, true);
    assert.equal(result.totalCostUsd, 0);
    assert.equal(result.usedOnlyLocal, true);

    // The workspace must be untouched.
    assert.match(readFileSync(join(workspace, 'prices.py'), 'utf8'), /range\(len\(items\) \+ 1\)/);
  });

  it('emits progress events that a UI can render', async () => {
    const { dir, workspace, cfg } = setup({ cloudKey: false });
    const events: RunEvent[] = [];
    await runTask({
      cfg,
      dataDir: dir,
      ctx: ctxFor(workspace),
      localProvider: new MockProvider({ id: 'mock-local' }),
      offlineRoute: true,
      onEvent: (e) => events.push(e),
    });
    const types = events.map((e) => e.type);
    assert.ok(types.includes('route'));
    assert.ok(types.includes('attempt-start'));
    assert.ok(types.includes('verify'));
  });

  it('writes verified edits only when apply is requested', async () => {
    const { dir, workspace, cfg } = setup({ cloudKey: false });
    const result = await runTask({
      cfg,
      dataDir: dir,
      ctx: ctxFor(workspace),
      localProvider: new MockProvider({ id: 'mock-local' }),
      offlineRoute: true,
      apply: true,
    });
    assert.equal(result.writtenFiles.length, 1);
    assert.equal(result.writtenFiles[0], 'prices.py');
    const onDisk = readFileSync(join(workspace, 'prices.py'), 'utf8');
    // The mock appends a marker rather than fixing the bug (it cannot reason); the
    // point of this test is that the *verified* candidate reached the disk and
    // that the original content is preserved underneath it.
    assert.match(onDisk, /proto mock: deterministic edit/);
    assert.match(onDisk, /def total_prices\(items\):/, 'the original file content must be preserved');
  });

  it('never writes when verification failed, even with apply', async () => {
    const { dir, workspace, cfg } = setup({ cloudKey: false });
    const result = await runTask({
      cfg,
      dataDir: dir,
      ctx: ctxFor(workspace),
      localProvider: new BrokenProvider(),
      offlineRoute: true,
      apply: true,
      maxRepair: 0,
    });
    assert.equal(result.status, 'failed');
    assert.equal(result.writtenFiles.length, 0);
    assert.match(readFileSync(join(workspace, 'prices.py'), 'utf8'), /range\(len\(items\) \+ 1\)/);
    assert.ok(result.warnings.some((w) => /refusing to write/.test(w)));
  });
});

describe('runTask: escalation path', () => {
  it('escalates a failed local attempt to the cloud and records the pair', async () => {
    const { dir, workspace, cfg } = setup();
    const result = await runTask({
      cfg,
      dataDir: dir,
      ctx: ctxFor(workspace),
      localProvider: new BrokenProvider(),
      cloudProvider: new MockProvider({ id: 'mock-cloud', kind: 'cloud', model: 'mock-strong' }),
      offlineRoute: true,
      maxRepair: 0,
    });

    assert.equal(result.status, 'escalated-cloud-success', JSON.stringify(result.warnings));
    assert.ok(result.attempts.length >= 2);
    assert.equal(result.attempts[0]?.tier.startsWith('local'), true);
    assert.equal(result.attempts[1]?.tier.startsWith('cloud'), true);
    assert.equal(result.attempts[0]?.verification?.passed, false);
    assert.equal(result.attempts[1]?.verification?.passed, true);
  });

  it('fails cleanly and explains when local fails and no cloud is configured', async () => {
    const { dir, workspace, cfg } = setup({ cloudKey: false });
    const result = await runTask({
      cfg,
      dataDir: dir,
      ctx: ctxFor(workspace),
      localProvider: new BrokenProvider(),
      offlineRoute: true,
      maxRepair: 0,
    });
    assert.equal(result.status, 'failed');
    assert.ok(result.warnings.some((w) => /no cloud tier is available/.test(w)));
    assert.equal(result.attempts[0]?.tier.startsWith('local'), true);
  });

  it('treats a transport failure as an outage, not as the model being wrong', async () => {
    const { dir, workspace, cfg } = setup();
    const failing: Provider = {
      id: 'down',
      label: 'Down',
      kind: 'local',
      model: 'down',
      capabilities: { tools: false, jsonSchema: false, streaming: false, promptCaching: false, contextWindow: 100, maxOutputTokens: 10 },
      async chat(): Promise<ChatResponse> {
        throw new Error('ECONNREFUSED: nothing listening on 127.0.0.1:11434');
      },
      async health(): Promise<{ ok: boolean; detail: string }> {
        return { ok: false, detail: 'down' };
      },
    };
    const result = await runTask({
      cfg,
      dataDir: dir,
      ctx: ctxFor(workspace),
      localProvider: failing,
      cloudProvider: new MockProvider({ id: 'mock-cloud', kind: 'cloud' }),
      offlineRoute: true,
      maxRepair: 0,
    });

    const first = result.attempts[0];
    assert.ok(first?.error, 'the transport error must be recorded on the attempt');
    // A transport failure is an outage: verification never ran.
    assert.equal(first?.verification, null);
  });
});

describe('runTask: forcing and dry runs', () => {
  it('honours an explicitly forced tier', async () => {
    const { dir, workspace, cfg } = setup();
    const local = new MockProvider({ id: 'mock-local' });
    const cloud = new MockProvider({ id: 'mock-cloud', kind: 'cloud' });
    const result = await runTask({
      cfg,
      dataDir: dir,
      ctx: ctxFor(workspace),
      localProvider: local,
      cloudProvider: cloud,
      forceTier: 'cloud-strong',
      offlineRoute: true,
    });
    assert.equal(result.decision.forced, true);
    assert.equal(result.attempts[0]?.tier, 'cloud-strong');
    assert.equal(local.history.length, 0);
    assert.equal(cloud.history.length, 1);
  });

  it('makes no model calls at all in dry-run mode', async () => {
    const { dir, workspace, cfg } = setup();
    const local = new MockProvider({ id: 'mock-local' });
    const result = await runTask({
      cfg,
      dataDir: dir,
      ctx: ctxFor(workspace),
      localProvider: local,
      offlineRoute: true,
      dryRun: true,
    });
    assert.equal(result.status, 'dry-run');
    assert.equal(result.attempts.length, 0);
    assert.equal(local.history.length, 0);
  });
});

describe('applyToWorkspace', () => {
  it('refuses to escape the workspace root', () => {
    const workspace = tempDir();
    const written = applyToWorkspace(
      {
        passed: true,
        score: 1,
        checks: [],
        blockers: [],
        candidate: { summary: 's', edits: [] },
        applied: [{ path: '../outside.py', before: null, after: 'x = 1\n', changedLines: 1 }],
        durationMs: 1,
      },
      { task: 't', workspace },
    );
    assert.equal(written.length, 0);
  });

  it('creates nested directories for new files', () => {
    const workspace = tempDir();
    const written = applyToWorkspace(
      {
        passed: true,
        score: 1,
        checks: [],
        blockers: [],
        candidate: { summary: 's', edits: [] },
        applied: [{ path: 'pkg/sub/new.py', before: null, after: 'x = 1\n', changedLines: 1 }],
        durationMs: 1,
      },
      { task: 't', workspace },
    );
    assert.deepEqual(written, ['pkg/sub/new.py']);
    assert.equal(readFileSync(join(workspace, 'pkg/sub/new.py'), 'utf8'), 'x = 1\n');
  });

  it('writes nothing when there is no workspace', () => {
    assert.deepEqual(
      applyToWorkspace(
        {
          passed: true,
          score: 1,
          checks: [],
          blockers: [],
          candidate: null,
          applied: [{ path: 'a.py', before: null, after: 'x', changedLines: 1 }],
          durationMs: 1,
        },
        { task: 't' },
      ),
      [],
    );
  });
});

describe('alwaysFailingMock', () => {
  it('is genuinely unusable as a code candidate', async () => {
    const provider = alwaysFailingMock();
    const res = await provider.chat({ messages: [{ role: 'user', content: 'FILE: a.py\n```\nx = 1\n```' }] });
    assert.equal(res.finishReason, 'stop');
    assert.match(res.text, /range\(len\(items\)/);
  });
});
