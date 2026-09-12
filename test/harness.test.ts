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
import { EpisodeStore } from '../src/memory/store.ts';
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
  it('routes to local, verifies, records a positive label, and writes nothing by default', async () => {
    const { dir, workspace, cfg } = setup({ cloudKey: false });
    const store = new EpisodeStore(dir);
    const result = await runTask({
      cfg,
      dataDir: dir,
      ctx: ctxFor(workspace),
      localProvider: new MockProvider({ id: 'mock-local', kind: 'local' }),
      offlineRoute: true,
      store,
    });

    assert.equal(result.status, 'local-success');
    assert.equal(result.writtenFiles.length, 0);
    assert.equal(result.episode?.outcome.localSucceeded, true);
    assert.equal(result.episode?.outcome.escalated, false);
    assert.equal(result.episode?.outcome.totalCostUsd, 0);
    assert.ok((result.episode?.outcome.reward ?? 0) > 1, 'a clean local win should be strongly rewarded');
    assert.equal(result.episode?.attempts.length, 1);
    assert.equal(result.episode?.attempts[0]?.verification?.passed, true);

    // The workspace must be untouched.
    assert.match(readFileSync(join(workspace, 'prices.py'), 'utf8'), /range\(len\(items\) \+ 1\)/);
    // And the episode must be on disk.
    assert.equal(store.readAll().length, 1);
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
    assert.ok(types.includes('recorded'));
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
    const store = new EpisodeStore(dir);
    const result = await runTask({
      cfg,
      dataDir: dir,
      ctx: ctxFor(workspace),
      localProvider: new BrokenProvider(),
      cloudProvider: new MockProvider({ id: 'mock-cloud', kind: 'cloud', model: 'mock-strong' }),
      offlineRoute: true,
      maxRepair: 0,
      store,
    });

    assert.equal(result.status, 'escalated-cloud-success', JSON.stringify(result.warnings));
    assert.equal(result.episode?.outcome.escalated, true);
    assert.equal(result.episode?.outcome.localSucceeded, false);
    assert.ok((result.episode?.attempts.length ?? 0) >= 2);
    assert.equal(result.episode?.attempts[0]?.tier.startsWith('local'), true);
    assert.equal(result.episode?.attempts[1]?.tier.startsWith('cloud'), true);
    // The failed local answer must be retained: it is the DPO "rejected" side.
    assert.ok(result.episode?.attempts[0]?.output);
    assert.equal(result.episode?.attempts[0]?.verification?.passed, false);
    assert.equal(result.episode?.attempts[1]?.verification?.passed, true);
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
    assert.equal(result.episode?.outcome.localSucceeded, false);
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

    const first = result.episode?.attempts[0];
    assert.ok(first?.error, 'the transport error must be recorded on the attempt');
    // No local verification happened, so there must be no training label.
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
    assert.equal(result.episode?.outcome.finalTier, 'cloud-strong');
    assert.equal(result.episode?.outcome.localSucceeded, null, 'local was never attempted, so there is no label');
    assert.equal(local.history.length, 0);
    assert.equal(cloud.history.length, 1);
  });

  it('makes no model calls at all in dry-run mode', async () => {
    const { dir, workspace, cfg } = setup();
    const local = new MockProvider({ id: 'mock-local' });
    const store = new EpisodeStore(dir);
    const result = await runTask({
      cfg,
      dataDir: dir,
      ctx: ctxFor(workspace),
      localProvider: local,
      offlineRoute: true,
      dryRun: true,
      store,
    });
    assert.equal(result.status, 'dry-run');
    assert.equal(result.episode, null);
    assert.equal(local.history.length, 0);
    assert.equal(store.readAll().length, 0, 'a dry run must not pollute the episode log');
  });

  it('does not persist when asked not to', async () => {
    const { dir, workspace, cfg } = setup({ cloudKey: false });
    const store = new EpisodeStore(dir);
    await runTask({
      cfg,
      dataDir: dir,
      ctx: ctxFor(workspace),
      localProvider: new MockProvider({ id: 'mock-local' }),
      offlineRoute: true,
      persist: false,
      store,
    });
    assert.equal(store.readAll().length, 0);
  });
});

describe('runTask: data hygiene', () => {
  it('redacts secrets before they reach the episode log', async () => {
    const { dir, workspace, cfg } = setup({ cloudKey: false });
    const store = new EpisodeStore(dir);
    const secret = 'sk-proj-abcdefghijklmnopqrstuvwxyz012345';
    await runTask({
      cfg,
      dataDir: dir,
      ctx: {
        task: `${EASY_TASK} Also my key is ${secret} and my email is dev@example.com`,
        workspace,
        files: [{ path: 'prices.py', content: LOOP_FILE }],
      },
      localProvider: new MockProvider({ id: 'mock-local' }),
      offlineRoute: true,
      store,
    });

    const raw = readFileSync(store.shards()[0] as string, 'utf8');
    assert.ok(!raw.includes(secret), 'the API key must never be written to disk');
    assert.ok(!raw.includes('dev@example.com'));
    assert.ok(raw.includes('[REDACTED:api-key]'));
    const ep = store.readAll()[0];
    assert.equal(ep?.redaction.applied, true);
    assert.ok(Object.keys(ep?.redaction.counts ?? {}).length > 0);
  });

  it('omits task text and prompts when configured to', async () => {
    const { dir, workspace } = setup({ cloudKey: false });
    const base = testConfig(dir);
    const cfg: ProtoConfig = {
      ...base,
      cloud: { ...base.cloud, enabled: false },
      memory: { ...base.memory, storeTaskText: false, storePrompts: false },
    };
    const store = new EpisodeStore(dir);
    const ep = (
      await runTask({
        cfg,
        dataDir: dir,
        ctx: ctxFor(workspace),
        localProvider: new MockProvider({ id: 'mock-local' }),
        offlineRoute: true,
        store,
      })
    ).episode;

    assert.equal(ep?.task, undefined);
    assert.equal(ep?.systemPrompt, undefined);
    assert.ok(ep?.taskHash);
    assert.equal(ep?.attempts[0]?.prompt, undefined);
    assert.equal(ep?.attempts[0]?.output, undefined);
    assert.ok(ep?.attempts[0]?.outputHash, 'hashes remain so dedup still works');
  });

  it('records the router environment that the decision was actually made under', async () => {
    const { dir, workspace, cfg } = setup({ cloudKey: false });
    const result = await runTask({
      cfg,
      dataDir: dir,
      ctx: ctxFor(workspace),
      localProvider: new MockProvider({ id: 'mock-local' }),
      offlineRoute: true,
    });
    assert.equal(result.episode?.environment.cloudAvailable, false);
    assert.equal(result.episode?.environment.localAvailable, true);
    assert.equal(result.episode?.environment.verifierAvailable, true);
    assert.equal(result.episode?.features.taskClass, 'bugfix-local');
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
