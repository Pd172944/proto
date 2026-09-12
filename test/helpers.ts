/**
 * Test helpers.
 *
 * Every test gets its own temporary data directory so that tests never read or
 * write the user's real `var/`, and can run in parallel safely.
 */

import { mkdtempSync, rmSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';

import { loadConfig } from '../src/config/load.ts';
import type { ProtoConfig } from '../src/config/schema.ts';
import { toVector } from '../src/router/features.ts';

// The synthetic episode factory is shared with `proto simulate`; see
// src/memory/factory.ts for why it lives in src/ rather than here.
export { fakeEpisode as makeEpisode } from '../src/memory/factory.ts';
export type { FakeEpisodeInput as MockEpisodeInput } from '../src/memory/factory.ts';

const created: string[] = [];

export function tempDir(prefix = 'proto-test-'): string {
  const dir = mkdtempSync(join(tmpdir(), prefix));
  created.push(dir);
  return dir;
}

export function cleanupTempDirs(): void {
  for (const dir of created.splice(0)) {
    try {
      rmSync(dir, { recursive: true, force: true });
    } catch {
      /* best effort */
    }
  }
}

/** A config pinned to a temp data dir with predictable, environment-independent settings. */
export function testConfig(dataDir: string, overrides: Partial<ProtoConfig> = {}): ProtoConfig {
  const { config } = loadConfig({ dataDir });
  // Neutralise anything the host environment may have injected, so tests are
  // hermetic: no ambient API keys, no ambient PROTO_* variables.
  const base: ProtoConfig = {
    ...config,
    dataDir,
    cloud: { ...config.cloud, enabled: false, provider: 'openrouter', model: 'anthropic/claude-sonnet-4.5' },
    local: { ...config.local, enabled: true, runtime: 'ollama', model: 'qwen2.5-coder:1.5b-instruct' },
    train: { ...config.train, enabled: false },
    contrib: { ...config.contrib, enabled: false, endpoint: '' },
    memory: { ...config.memory, enabled: true },
    verify: { ...config.verify, enabled: true, runTests: false },
  };
  return { ...base, ...overrides };
}

/** Kept for tests that assert on vector shape. */
export function featuresToVector(features: ReturnType<typeof import('../src/router/features.ts').extractFeatures>): number[] {
  return toVector(features);
}
