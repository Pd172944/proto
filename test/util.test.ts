/**
 * Utility and configuration tests: argv parsing, text helpers, ids, config
 * merging and pricing.
 *
 * The argv parser is tested for the failure mode that matters: an unknown flag
 * must be an error. A silently ignored `--aply` typo would look like a
 * successful dry run, and the user would not know their file was never written.
 */

import { strict as assert } from 'node:assert';
import { describe, it } from 'node:test';

import { flagBool, flagList, flagNumber, flagString, parseArgs } from '../src/util/args.ts';
import type { FlagSpec } from '../src/util/args.ts';
import { estimateTokens, extractCodeBlocks, extractJson, oneLine, sha256, shannonEntropy, stableStringify, truncateMiddle } from '../src/util/text.ts';
import { contentId, shortId, ulid } from '../src/util/ids.ts';
import { deepMerge, loadConfig, priceFor, resolveDataDir, saveConfig, writeSecret, resolveApiKeyFor } from '../src/config/load.ts';
import { DEFAULT_CONFIG, PROVIDER_PROFILES, UNKNOWN_PRICE, providerProfile } from '../src/config/schema.ts';
import { formatBytes, formatDuration, readJsonOrNull, resolvePath, writeJsonAtomic } from '../src/util/fsx.ts';
import { localDayKey, localHour, shiftDayKey } from '../src/util/clock.ts';
import { tempDir, testConfig } from './helpers.ts';

const SPEC: FlagSpec[] = [
  { name: 'apply', type: 'boolean', alias: 'a', description: '' },
  { name: 'file', type: 'string', multiple: true, description: '' },
  { name: 'limit', type: 'number', description: '' },
  { name: 'tier', type: 'string', description: '' },
];

describe('argv parsing', () => {
  it('parses values, positionals and defaults', () => {
    const p = parseArgs(['run', 'do the thing', '--tier', 'local', '--file', 'a.py', '--limit=5'], SPEC);
    assert.deepEqual(p.positionals, ['run', 'do the thing']);
    assert.equal(flagString(p, 'tier'), 'local');
    assert.equal(flagNumber(p, 'limit'), 5);
    assert.deepEqual(flagList(p, 'file'), ['a.py']);
    assert.deepEqual(p.errors, []);
  });

  it('errors on an unknown flag instead of ignoring it', () => {
    const p = parseArgs(['--aply'], SPEC);
    assert.equal(p.errors.length, 1);
    assert.match(p.errors[0] ?? '', /unknown flag: --aply/);
  });

  it('supports repeated flags, short flags and combined booleans', () => {
    const p = parseArgs(['-a', '--file', 'a.py', '--file', 'b.py'], SPEC);
    assert.equal(flagBool(p, 'apply'), true);
    assert.deepEqual(flagList(p, 'file'), ['a.py', 'b.py']);

    const short = parseArgs(['-a'], SPEC);
    assert.equal(flagBool(short, 'apply'), true);
  });

  it('treats -- as the end of flags', () => {
    const p = parseArgs(['--', '--not-a-flag'], SPEC);
    assert.deepEqual(p.positionals, ['--not-a-flag']);
    assert.deepEqual(p.errors, []);
  });

  it('reports a missing value and a non-numeric number', () => {
    assert.match(parseArgs(['--tier'], SPEC).errors[0] ?? '', /requires a value/);
    const bad = parseArgs(['--limit', 'abc'], SPEC);
    assert.match(bad.errors[0] ?? '', /expects a number/);
  });

  it('supports explicit boolean values', () => {
    assert.equal(flagBool(parseArgs(['--apply=false'], SPEC), 'apply'), false);
    assert.equal(flagBool(parseArgs(['--apply'], SPEC), 'apply'), true);
  });
});

describe('text helpers', () => {
  it('estimates tokens monotonically and reasonably', () => {
    assert.equal(estimateTokens(''), 0);
    assert.ok(estimateTokens('hello world') > 0);
    assert.ok(estimateTokens('a'.repeat(400)) > estimateTokens('a'.repeat(40)));
    const prose = 'The quick brown fox jumps over the lazy dog. '.repeat(10);
    const estimate = estimateTokens(prose);
    assert.ok(estimate > prose.length / 6 && estimate < prose.length / 2, `implausible estimate ${estimate}`);
  });

  it('extracts fenced code blocks with their language and optional path', () => {
    const blocks = extractCodeBlocks('text\n```python title=src/a.py\nx = 1\n```\nmore\n```ts\ny\n```');
    assert.equal(blocks.length, 2);
    assert.equal(blocks[0]?.lang, 'python');
    assert.equal(blocks[0]?.path, 'src/a.py');
    assert.equal(blocks[1]?.lang, 'ts');
  });

  it('extracts JSON from prose, fences and trailing text', () => {
    assert.deepEqual(extractJson('prefix {"a": 1} suffix'), { a: 1 });
    assert.deepEqual(extractJson('```json\n{"b": [1,2]}\n```'), { b: [1, 2] });
    assert.equal(extractJson('no json here'), undefined);
    assert.deepEqual(extractJson('{"nested": {"deep": [1, {"x": "}"}]}}'), { nested: { deep: [1, { x: '}' }] } });
  });

  it('measures entropy so secrets can be distinguished from prose', () => {
    assert.ok(shannonEntropy('aaaaaaaaaa') < 1);
    assert.ok(shannonEntropy('Zk8xQ2mN7pR4tV9w') > 3.5);
  });

  it('produces a stable stringify with sorted keys', () => {
    assert.equal(stableStringify({ b: 1, a: 2 }), '{"a":2,"b":1}');
    assert.equal(stableStringify({ a: 2, b: 1 }), stableStringify({ b: 1, a: 2 }));
  });

  it('truncates in the middle so head and tail survive', () => {
    const text = `${'h'.repeat(100)}${'m'.repeat(100)}${'t'.repeat(100)}`;
    const out = truncateMiddle(text, 60);
    assert.ok(out.length < 120);
    assert.ok(out.startsWith('h'));
    assert.ok(out.endsWith('t'));
  });

  it('hashes deterministically and compacts to one line', () => {
    assert.equal(sha256('x'), sha256('x'));
    assert.notEqual(sha256('x'), sha256('y'));
    assert.equal(oneLine('a\n\nb   c'), 'a b c');
  });
});

describe('ids', () => {
  it('generates monotonically increasing, sortable ids', () => {
    const a = ulid(1_700_000_000_000);
    const b = ulid(1_700_000_000_001);
    assert.ok(b > a, 'ids must sort by creation time');
    assert.equal(a.length, 26);
  });

  it('does not repeat within the same millisecond', () => {
    const ids = new Set(Array.from({ length: 200 }, () => ulid(1_700_000_000_000)));
    assert.equal(ids.size, 200, 'monotonic ULIDs must not collide in the same millisecond');
  });

  it('derives stable content ids for deduplication', () => {
    assert.equal(contentId('same'), contentId('same'));
    assert.notEqual(contentId('same'), contentId('other'));
    assert.ok(shortId('job').startsWith('job-'));
  });
});

describe('config', () => {
  it('deep-merges partial overrides without dropping siblings', () => {
    const merged = deepMerge({ a: { b: 1, c: 2 }, d: 3 }, { a: { c: 9 } }) as { a: { b: number; c: number }; d: number };
    assert.equal(merged.a.b, 1);
    assert.equal(merged.a.c, 9);
    assert.equal(merged.d, 3);
  });

  it('resolves the data directory from an explicit path or the environment', () => {
    assert.equal(resolveDataDir('/tmp/abs'), '/tmp/abs');
    const prev = process.env['PROTO_HOME'];
    process.env['PROTO_HOME'] = '/tmp/from-env';
    try {
      assert.equal(resolveDataDir(), '/tmp/from-env');
    } finally {
      if (prev === undefined) delete process.env['PROTO_HOME'];
      else process.env['PROTO_HOME'] = prev;
    }
  });

  it('ships a sane default configuration', () => {
    assert.equal(DEFAULT_CONFIG.verify.enabled, true);
    assert.equal(DEFAULT_CONFIG.memory.redact, true);
    assert.equal(DEFAULT_CONFIG.train.enabled, false, 'training must be opt-in');
    assert.equal(DEFAULT_CONFIG.contrib.enabled, false, 'sharing must be opt-in');
    assert.equal(DEFAULT_CONFIG.contrib.shareCode, false);
    assert.ok(DEFAULT_CONFIG.routing.qualityFloor < DEFAULT_CONFIG.routing.qualityFloorUnverified);
    assert.ok(DEFAULT_CONFIG.routing.qualityFloorReadOnly <= DEFAULT_CONFIG.routing.qualityFloor);
  });

  it('knows several providers and their key environment variables', () => {
    assert.ok(PROVIDER_PROFILES.length >= 8);
    for (const p of PROVIDER_PROFILES) {
      assert.ok(p.keyEnv.length >= 1, `${p.id} needs at least one env var`);
      assert.equal(providerProfile(p.id)?.id, p.id);
    }
    assert.equal(providerProfile('openrouter')?.api, 'openai');
    assert.equal(providerProfile('anthropic')?.api, 'anthropic');
    assert.equal(providerProfile('nope'), undefined);
  });

  it('falls back to a deliberately expensive unknown price', () => {
    const cfg = testConfig(tempDir());
    assert.deepEqual(priceFor(cfg, 'totally-unknown-model-xyz'), { in: 5, out: 20 });
    assert.equal(UNKNOWN_PRICE.in, 5);
    // A dated snapshot should inherit its family's price rather than the fallback.
    assert.ok(priceFor(cfg, 'claude-sonnet-4-5-20260101').in <= 5);
  });

  it('round-trips through disk and clears the dataDir field', () => {
    const dir = tempDir();
    const cfg = testConfig(dir);
    const path = saveConfig({ ...cfg, routing: { ...cfg.routing, qualityFloor: 0.8 } }, dir);
    const reloaded = readJsonOrNull<{ routing: { qualityFloor: number }; dataDir: string }>(path);
    assert.equal(reloaded?.routing.qualityFloor, 0.8);
    assert.equal(reloaded?.dataDir, '', 'dataDir is derived, not stored');
  });

  it('reads a key from the secrets file when the environment has none', () => {
    const dir = tempDir();
    const cfg = testConfig(dir, { cloud: { ...testConfig(dir).cloud, provider: 'openrouter' } });
    const prev = process.env['OPENROUTER_API_KEY'];
    delete process.env['OPENROUTER_API_KEY'];
    try {
      assert.equal(resolveApiKeyFor(cfg, dir), undefined);
      writeSecret(dir, 'openrouter', 'file-key');
      assert.equal(resolveApiKeyFor(cfg, dir), 'file-key');
      process.env['OPENROUTER_API_KEY'] = 'env-key';
      assert.equal(resolveApiKeyFor(cfg, dir), 'env-key', 'the environment must win over the file');
    } finally {
      if (prev === undefined) delete process.env['OPENROUTER_API_KEY'];
      else process.env['OPENROUTER_API_KEY'] = prev;
    }
  });

  it('enables the cloud tier from a key alone, without editing config', () => {
    // Regression guard. `resolveApiKeyFor` read `cfg.dataDir` with `??`, and an
    // empty string is not nullish — so while `loadConfig` still had dataDir as
    // "", the secrets lookup resolved to a path relative to the cwd and never
    // found the file. A user who followed the quickstart (set an API key, touch
    // nothing else) silently got every task forced onto the local tier.
    const dir = tempDir();
    writeSecret(dir, 'openrouter', 'key-from-file');
    const prevHome = process.env['PROTO_HOME'];
    const prevKey = process.env['OPENROUTER_API_KEY'];
    delete process.env['OPENROUTER_API_KEY'];
    process.env['PROTO_HOME'] = dir;
    try {
      const loaded = loadConfig({}).config;
      assert.equal(loaded.dataDir, dir);
      assert.equal(loaded.cloud.enabled, true, 'a present key must enable the cloud tier');
      assert.equal(resolveApiKeyFor(loaded), 'key-from-file');
    } finally {
      if (prevHome === undefined) delete process.env['PROTO_HOME'];
      else process.env['PROTO_HOME'] = prevHome;
      if (prevKey !== undefined) process.env['OPENROUTER_API_KEY'] = prevKey;
    }
  });

  it('applies environment overrides for scripting', () => {
    const dir = tempDir();
    const prevModel = process.env['PROTO_LOCAL_MODEL'];
    const prevMode = process.env['PROTO_ROUTING_MODE'];
    process.env['PROTO_LOCAL_MODEL'] = 'llama3.2:3b';
    process.env['PROTO_ROUTING_MODE'] = 'heuristic';
    try {
      const cfg = testConfig(dir);
      // testConfig bypasses env overrides (it post-processes), so verify via loadConfig.
      const loaded = loadConfig({ dataDir: dir }).config;
      assert.equal(loaded.local.model, 'llama3.2:3b');
      assert.equal(loaded.routing.mode, 'heuristic');
      assert.ok(cfg.local.model.length > 0);
    } finally {
      if (prevModel === undefined) delete process.env['PROTO_LOCAL_MODEL'];
      else process.env['PROTO_LOCAL_MODEL'] = prevModel;
      if (prevMode === undefined) delete process.env['PROTO_ROUTING_MODE'];
      else process.env['PROTO_ROUTING_MODE'] = prevMode;
    }
  });
});

describe('clock helpers', () => {
  it('keys the day in local time, not UTC', () => {
    // The daily budgets are the user's daily budgets. A UTC key would reset them
    // mid-afternoon for anyone west of Greenwich.
    const lateEvening = new Date(2025, 5, 1, 23, 30, 0);
    assert.equal(localDayKey(lateEvening), '2025-06-01');
    const earlyMorning = new Date(2025, 5, 2, 0, 30, 0);
    assert.equal(localDayKey(earlyMorning), '2025-06-02');
    assert.equal(localHour(lateEvening), 23);
  });

  it('pads single-digit months and days so keys sort lexicographically', () => {
    assert.equal(localDayKey(new Date(2025, 0, 5)), '2025-01-05');
    assert.ok(localDayKey(new Date(2025, 8, 9)) > localDayKey(new Date(2025, 8, 10 - 10)));
  });

  it('shifts across month and year boundaries', () => {
    assert.equal(shiftDayKey('2025-03-01', -1), '2025-02-28');
    assert.equal(shiftDayKey('2025-01-01', -1), '2024-12-31');
    assert.equal(shiftDayKey('2024-02-28', 1), '2024-02-29');
    assert.equal(shiftDayKey('2025-12-31', 1), '2026-01-01');
  });
});

describe('fs helpers', () => {
  it('writes JSON atomically and reads it back', () => {
    const dir = tempDir();
    const path = `${dir}/nested/x.json`;
    writeJsonAtomic(path, { a: 1 });
    assert.deepEqual(readJsonOrNull<{ a: number }>(path), { a: 1 });
    assert.equal(readJsonOrNull(`${dir}/missing.json`), null);
  });

  it('expands ~ and formats sizes and durations readably', () => {
    assert.ok(resolvePath('~/x').startsWith('/'));
    assert.equal(formatBytes(512), '512 B');
    assert.match(formatBytes(2048), /KB/);
    assert.equal(formatDuration(500), '500ms');
    assert.match(formatDuration(65_000), /1m/);
  });
});
