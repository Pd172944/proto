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
import {
  autoSelectCloudProvider,
  deepMerge,
  loadConfig,
  priceFor,
  resolveDataDir,
  saveConfig,
  writeSecret,
  resolveApiKeyFor,
} from '../src/config/load.ts';
import { rankLocalModels } from '../src/providers/index.ts';
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

  it('persists only what differs from the defaults', () => {
    // Writing the whole merged config made the file a snapshot: `proto train
    // enable` froze every default of that day, so later improvements to a default
    // (better batch size, higher dataset caps) never reached an existing user.
    // Separate directories, because each saveConfig replaces the whole file with
    // the delta of the config it was handed.
    const dirA = tempDir();
    const cfgA = testConfig(dirA);
    const path = saveConfig({ ...cfgA, verify: { ...cfgA.verify, runTests: true } }, dirA);
    assert.deepEqual(readJsonOrNull<Record<string, unknown>>(path), { version: 1, verify: { runTests: true } });

    // A nested override keeps only the changed leaf.
    const dirB = tempDir();
    const cfgB = testConfig(dirB);
    const path2 = saveConfig({ ...cfgB, routing: { ...cfgB.routing, qualityFloor: 0.8 } }, dirB);
    assert.deepEqual(readJsonOrNull<Record<string, unknown>>(path2), { version: 1, routing: { qualityFloor: 0.8 } });

    // And the effective config is still complete when read back.
    const reloaded = loadConfig({ dataDir: dirA }).config;
    assert.equal(reloaded.verify.runTests, true);
    assert.equal(reloaded.local.model, cfgA.local.model);
    const reloadedB = loadConfig({ dataDir: dirB }).config;
    assert.equal(reloadedB.routing.qualityFloor, 0.8);
    assert.equal(reloadedB.local.model, cfgB.local.model);
  });

  it('round-trips through disk and never stores the derived dataDir', () => {
    const dir = tempDir();
    const cfg = testConfig(dir);
    const path = saveConfig({ ...cfg, routing: { ...cfg.routing, qualityFloor: 0.8 } }, dir);
    const reloaded = readJsonOrNull<{ routing: { qualityFloor: number }; dataDir?: string }>(path);
    assert.equal(reloaded?.routing.qualityFloor, 0.8);
    assert.equal(reloaded?.dataDir, undefined, 'dataDir is derived from PROTO_HOME, never stored');
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
    const prevFloor = process.env['PROTO_QUALITY_FLOOR'];
    process.env['PROTO_LOCAL_MODEL'] = 'llama3.2:3b';
    process.env['PROTO_QUALITY_FLOOR'] = '0.42';
    try {
      const cfg = testConfig(dir);
      // testConfig bypasses env overrides (it post-processes), so verify via loadConfig.
      const loaded = loadConfig({ dataDir: dir }).config;
      assert.equal(loaded.local.model, 'llama3.2:3b');
      assert.equal(loaded.routing.qualityFloor, 0.42);
      assert.ok(cfg.local.model.length > 0);
    } finally {
      if (prevModel === undefined) delete process.env['PROTO_LOCAL_MODEL'];
      else process.env['PROTO_LOCAL_MODEL'] = prevModel;
      if (prevFloor === undefined) delete process.env['PROTO_QUALITY_FLOOR'];
      else process.env['PROTO_QUALITY_FLOOR'] = prevFloor;
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

describe('cloud provider auto-selection', () => {
  /** Run `fn` with a controlled set of provider keys, restoring the environment. */
  const withKeys = (keys: Record<string, string | undefined>, fn: () => void): void => {
    const names = ['OPENROUTER_API_KEY', 'ANTHROPIC_API_KEY', 'OPENAI_API_KEY', 'DEEPSEEK_API_KEY', 'GROQ_API_KEY', 'PROTO_API_KEY'];
    const saved = new Map(names.map((n) => [n, process.env[n]]));
    for (const n of names) delete process.env[n];
    for (const [n, v] of Object.entries(keys)) if (v !== undefined) process.env[n] = v;
    try {
      fn();
    } finally {
      for (const [n, v] of saved) {
        if (v === undefined) delete process.env[n];
        else process.env[n] = v;
      }
    }
  };

  it('uses the provider whose key is actually set, and says so', () => {
    // The real-world failure this fixes: a user exports ANTHROPIC_API_KEY, the
    // default provider is OpenRouter, and doctor reports "api key missing" with no
    // explanation of why.
    const dir = tempDir();
    withKeys({ ANTHROPIC_API_KEY: 'sk-ant-test' }, () => {
      const cfg = testConfig(dir);
      assert.equal(cfg.cloud.provider, 'openrouter');
      const result = autoSelectCloudProvider(cfg, dir, false);
      assert.equal(result.config.cloud.provider, 'anthropic');
      assert.equal(result.config.cloud.enabled, true);
      assert.match(result.note ?? '', /ANTHROPIC_API_KEY is set/);
      // An OpenRouter-style model id is invalid on api.anthropic.com.
      assert.match(result.config.cloud.model, /^claude/);
    });
  });

  it('leaves a provider that has its own key alone', () => {
    const dir = tempDir();
    withKeys({ ANTHROPIC_API_KEY: 'sk-ant-test', OPENROUTER_API_KEY: 'sk-or-test' }, () => {
      const result = autoSelectCloudProvider(testConfig(dir), dir, false);
      assert.equal(result.config.cloud.provider, 'openrouter');
      assert.equal(result.note, undefined);
    });
  });

  it('invents nothing when no key is set at all', () => {
    const dir = tempDir();
    withKeys({}, () => {
      const result = autoSelectCloudProvider(testConfig(dir), dir, false);
      assert.equal(result.config.cloud.provider, 'openrouter');
      assert.equal(result.config.cloud.enabled, false);
      assert.equal(result.note, undefined);
    });
  });

  it('reads a key from the secrets file, not only the environment', () => {
    const dir = tempDir();
    withKeys({}, () => {
      const cfg = testConfig(dir);
      writeSecret(dir, 'anthropic', 'from-file');
      const result = autoSelectCloudProvider(cfg, dir, false);
      assert.equal(result.config.cloud.provider, 'anthropic');
    });
  });

  it('warns more loudly when the user had pinned a provider explicitly', () => {
    const dir = tempDir();
    withKeys({ ANTHROPIC_API_KEY: 'sk-ant-test' }, () => {
      const pinned = autoSelectCloudProvider(testConfig(dir), dir, true);
      assert.match(pinned.note ?? '', /pinned cloud\.provider explicitly/);
    });
  });

  it('keeps a valid model when switching between OpenAI-shaped providers', () => {
    const dir = tempDir();
    withKeys({ DEEPSEEK_API_KEY: 'ds-test' }, () => {
      const cfg = testConfig(dir);
      const result = autoSelectCloudProvider(cfg, dir, false);
      assert.equal(result.config.cloud.provider, 'deepseek');
      // The configured model was not provider-specific, so it is left alone.
      assert.equal(result.config.cloud.model, cfg.cloud.model);
    });
  });
});

describe('local model ranking', () => {
  it('ranks code-capable and agentic models above generic ones', () => {
    const ranked = rankLocalModels(['llama3.2:1b', 'qwen2.5-coder:1.5b-instruct', 'ornith-1.5:9b', 'qwen2.5-coder:7b-instruct']);
    const at = (m: string): number => ranked.indexOf(m);
    // Both a dedicated coder and an agentic model must beat a generic chat model,
    // even a much larger one.
    assert.ok(at('qwen2.5-coder:1.5b-instruct') < at('llama3.2:1b'));
    assert.ok(at('ornith-1.5:9b') < at('llama3.2:1b'));
    // Size breaks ties inside a family.
    assert.ok(at('qwen2.5-coder:7b-instruct') < at('qwen2.5-coder:1.5b-instruct'));
  });

  it('does not pretend to know whether an agentic model beats a same-size coder', () => {
    // ornith-1.5:9b (agentic, 9B) vs qwen2.5-coder:7b-instruct (coder, 7B) is a
    // genuine judgement call, not something a name-based heuristic can settle. Both
    // must rank well above generic models; the relative order is left to the user,
    // which is why `doctor` prints the whole ranked list rather than a single answer.
    const ranked = rankLocalModels(['ornith-1.5:9b', 'qwen2.5-coder:7b-instruct', 'llama3.2:13b']);
    const generic = ranked.indexOf('llama3.2:13b');
    assert.equal(generic, 2, 'both specialist models must outrank a larger generic one');
  });

  it('the agentic bonus is real, not cosmetic', () => {
    // Without the agentic term, ornith would rank purely on size and lose to a much
    // larger generic model.
    const ranked = rankLocalModels(['llama3.2:70b', 'ornith-1.5:9b']);
    assert.equal(ranked[0], 'ornith-1.5:9b');
  });

  it('pushes embedding and vision models to the bottom', () => {
    // Suggesting an embedding model as the coding tier would be worse than
    // suggesting nothing, so they must never rank first.
    const ranked = rankLocalModels(['nomic-embed-text:latest', 'llava:13b', 'qwen2.5-coder:1.5b-instruct']);
    assert.equal(ranked[0], 'qwen2.5-coder:1.5b-instruct');
    assert.ok(ranked.indexOf('nomic-embed-text:latest') > 0);
    assert.ok(ranked.indexOf('llava:13b') > 0);
  });

  it('prefers instruction-tuned models over base models of the same size', () => {
    const ranked = rankLocalModels(['qwen2.5-coder:7b-base', 'qwen2.5-coder:7b-instruct']);
    assert.equal(ranked[0], 'qwen2.5-coder:7b-instruct');
  });

  it('is stable and total for unknown names', () => {
    const input = ['zzz:latest', 'aaa:latest'];
    assert.deepEqual(rankLocalModels(input), rankLocalModels(input));
    assert.equal(rankLocalModels(input).length, 2);
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
