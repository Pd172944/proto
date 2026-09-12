/**
 * Config loading, merging, persistence and secret resolution.
 *
 * Resolution order (later wins):
 *   1. DEFAULT_CONFIG
 *   2. <dataDir>/config.json
 *   3. $PROTO_HOME/config.json (when PROTO_HOME is set and differs)
 *   4. environment variables (PROTO_*)
 *
 * `dataDir` itself is resolved first, from: --data-dir flag > PROTO_HOME >
 * <repo>/var. Keeping state inside the repo by default makes the prototype
 * self-contained and easy to inspect; set PROTO_HOME=~/.protoharness for a
 * machine-wide install.
 */

import { chmodSync, existsSync } from 'node:fs';
import { dirname, join, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';

import { DEFAULT_CONFIG, PROVIDER_PROFILES, providerProfile } from './schema.ts';
import type { ProtoConfig, Price } from './schema.ts';
import { ensureDir, readJsonOrNull, resolvePath, writeJsonAtomic } from '../util/fsx.ts';

const HERE = dirname(fileURLToPath(import.meta.url));

/** Repository root, i.e. the directory containing `package.json`. */
export const REPO_ROOT = resolve(HERE, '..', '..');

export function defaultDataDir(): string {
  const fromEnv = process.env['PROTO_HOME'];
  if (fromEnv && fromEnv.trim()) return resolvePath(fromEnv.trim(), REPO_ROOT);
  return join(REPO_ROOT, 'var');
}

export function resolveDataDir(explicit?: string): string {
  if (explicit) return resolvePath(explicit, process.cwd());
  return defaultDataDir();
}

type Json = Record<string, unknown>;

function isPlainObject(v: unknown): v is Json {
  return typeof v === 'object' && v !== null && !Array.isArray(v);
}

/** Recursive merge; arrays and scalars from `override` replace the base. */
export function deepMerge<T>(base: T, override: unknown): T {
  if (!isPlainObject(override)) return base;
  if (!isPlainObject(base)) return override as T;
  const out: Json = { ...(base as unknown as Json) };
  for (const [k, v] of Object.entries(override)) {
    const existing = out[k];
    if (isPlainObject(existing) && isPlainObject(v)) out[k] = deepMerge(existing, v);
    else out[k] = v;
  }
  return out as unknown as T;
}

function envBool(name: string): boolean | undefined {
  const raw = process.env[name];
  if (raw === undefined) return undefined;
  const v = raw.trim().toLowerCase();
  if (['1', 'true', 'yes', 'on'].includes(v)) return true;
  if (['0', 'false', 'no', 'off', ''].includes(v)) return false;
  return undefined;
}

function envNum(name: string): number | undefined {
  const raw = process.env[name];
  if (raw === undefined || raw.trim() === '') return undefined;
  const n = Number(raw);
  return Number.isFinite(n) ? n : undefined;
}

function envStr(name: string): string | undefined {
  const raw = process.env[name];
  return raw === undefined || raw === '' ? undefined : raw;
}

/**
 * Environment overrides. Intentionally a small, documented surface: enough to
 * script the harness (CI, launchd, a wrapper shell function) without turning
 * into a second, undocumented config system.
 */
export function applyEnvOverrides(cfg: ProtoConfig): ProtoConfig {
  const out: ProtoConfig = structuredClone(cfg);

  if (envBool('PROTO_DISABLE_LOCAL') !== undefined) out.local.enabled = !envBool('PROTO_DISABLE_LOCAL');
  const localModel = envStr('PROTO_LOCAL_MODEL');
  if (localModel) out.local.model = localModel;
  const localUrl = envStr('PROTO_LOCAL_BASE_URL');
  if (localUrl) out.local.baseUrl = localUrl;
  const localRuntime = envStr('PROTO_LOCAL_RUNTIME');
  if (localRuntime && ['ollama', 'llamacpp', 'mlx', 'openai'].includes(localRuntime)) {
    out.local.runtime = localRuntime as ProtoConfig['local']['runtime'];
  }
  const keepAlive = envNum('PROTO_LOCAL_KEEP_ALIVE_SEC');
  if (keepAlive !== undefined) out.local.keepAliveSec = keepAlive;

  if (envBool('PROTO_DISABLE_CLOUD') !== undefined) out.cloud.enabled = !envBool('PROTO_DISABLE_CLOUD');
  const cloudProvider = envStr('PROTO_CLOUD_PROVIDER');
  if (cloudProvider) out.cloud.provider = cloudProvider;
  const cloudModel = envStr('PROTO_CLOUD_MODEL');
  if (cloudModel) out.cloud.model = cloudModel;
  const cloudUrl = envStr('PROTO_CLOUD_BASE_URL');
  if (cloudUrl) out.cloud.baseUrl = cloudUrl;
  const effort = envStr('PROTO_CLOUD_EFFORT');
  if (effort && ['auto', 'low', 'medium', 'high'].includes(effort)) {
    out.cloud.effort = effort as ProtoConfig['cloud']['effort'];
  }
  // Presence of a key implies intent to use the cloud.
  if (resolveApiKeyFor(out) && envBool('PROTO_DISABLE_CLOUD') === undefined && !out.cloud.enabled) {
    out.cloud.enabled = true;
  }

  const routingMode = envStr('PROTO_ROUTING_MODE');
  if (routingMode && ['heuristic', 'learned', 'hybrid'].includes(routingMode)) {
    out.routing.mode = routingMode as ProtoConfig['routing']['mode'];
  }
  const floor = envNum('PROTO_QUALITY_FLOOR');
  if (floor !== undefined) out.routing.qualityFloor = clamp01(floor);
  const budget = envNum('PROTO_CLOUD_BUDGET_USD');
  if (budget !== undefined) out.routing.cloudBudgetUsdPerDay = budget;

  if (envBool('PROTO_DISABLE_MEMORY') !== undefined) out.memory.enabled = !envBool('PROTO_DISABLE_MEMORY');
  if (envBool('PROTO_DISABLE_REDACTION') !== undefined) out.memory.redact = !envBool('PROTO_DISABLE_REDACTION');
  const storeText = envBool('PROTO_STORE_TASK_TEXT');
  if (storeText !== undefined) out.memory.storeTaskText = storeText;

  const trainEnabled = envBool('PROTO_TRAIN_ENABLED');
  if (trainEnabled !== undefined) out.train.enabled = trainEnabled;
  const trainMode = envStr('PROTO_TRAIN_MODE');
  if (trainMode === 'sft' || trainMode === 'dpo') out.train.lora.mode = trainMode;

  const contribEnabled = envBool('PROTO_CONTRIB_ENABLED');
  if (contribEnabled !== undefined) out.contrib.enabled = contribEnabled;
  const shareCode = envBool('PROTO_CONTRIB_SHARE_CODE');
  if (shareCode !== undefined) out.contrib.shareCode = shareCode;
  const endpoint = envStr('PROTO_CONTRIB_ENDPOINT');
  if (endpoint) out.contrib.endpoint = endpoint;

  const evalMode = envStr('PROTO_EVAL_MODE');
  if (evalMode === 'route' || evalMode === 'live') out.eval.mode = evalMode;

  return out;
}

function clamp01(n: number): number {
  return Math.min(1, Math.max(0, n));
}

/** Environment-driven flag that turns a hard-fail into a warning. */
export function isStrict(): boolean {
  return envBool('PROTO_STRICT') === true;
}

export interface LoadedConfig {
  config: ProtoConfig;
  dataDir: string;
  configPath: string;
  /** Non-fatal problems worth surfacing in `proto doctor`. */
  warnings: string[];
}

export function loadConfig(opts: { dataDir?: string; overrides?: Partial<ProtoConfig> } = {}): LoadedConfig {
  const dataDir = resolveDataDir(opts.dataDir);
  const configPath = join(dataDir, 'config.json');
  const warnings: string[] = [];

  let cfg: ProtoConfig = structuredClone(DEFAULT_CONFIG);
  const onDisk = readJsonOrNull<Partial<ProtoConfig>>(configPath);
  if (onDisk) cfg = deepMerge(cfg, onDisk);
  else if (existsSync(configPath)) warnings.push(`config at ${configPath} is unreadable JSON; using defaults`);

  // A global config in PROTO_HOME acts as a base when a project-local var/ is used.
  const home = process.env['PROTO_HOME'];
  if (home) {
    const homePath = join(resolvePath(home, REPO_ROOT), 'config.json');
    if (resolve(homePath) !== resolve(configPath)) {
      const globalCfg = readJsonOrNull<Partial<ProtoConfig>>(homePath);
      if (globalCfg) cfg = deepMerge(cfg, globalCfg);
    }
  }

  if (opts.overrides) cfg = deepMerge(cfg, opts.overrides);

  // Resolve dataDir BEFORE applying environment overrides. Those overrides may
  // need to read `secrets.json` (to decide whether a key implies intent to use
  // the cloud), and a still-empty dataDir would make them look in the current
  // working directory instead. This ordering bug meant that setting only an API
  // key never enabled the cloud tier.
  cfg.dataDir = dataDir;
  cfg = applyEnvOverrides(cfg);

  validate(cfg, warnings);
  return { config: cfg, dataDir, configPath, warnings };
}

function validate(cfg: ProtoConfig, warnings: string[]): void {
  const profile = providerProfile(cfg.cloud.provider);
  if (!profile) {
    warnings.push(
      `unknown cloud provider "${cfg.cloud.provider}"; known: ${PROVIDER_PROFILES.map((p) => p.id).join(', ')}`,
    );
  }
  if (cfg.routing.qualityFloor < 0 || cfg.routing.qualityFloor > 1) {
    warnings.push(`routing.qualityFloor must be in [0,1]; clamping`);
    cfg.routing.qualityFloor = clamp01(cfg.routing.qualityFloor);
  }
  if (cfg.train.windowStartHour === cfg.train.windowEndHour) {
    warnings.push('train.windowStartHour equals windowEndHour: the window is empty, no training will run');
  }
  if (cfg.local.contextWindow > 32768) {
    warnings.push(
      `local.contextWindow=${cfg.local.contextWindow} needs a large KV cache; on a 16GB machine prefer <= 16384`,
    );
  }
  for (const [model, price] of Object.entries(cfg.pricing)) {
    if (!Number.isFinite(price.in) || !Number.isFinite(price.out) || price.in < 0 || price.out < 0) {
      warnings.push(`pricing.${model} is not a valid non-negative price pair`);
    }
  }
}

export function saveConfig(config: ProtoConfig, dataDir?: string): string {
  const dir = dataDir ?? config.dataDir ?? defaultDataDir();
  const path = join(dir, 'config.json');
  ensureDir(dir);
  // Persist only what differs from the defaults.
  //
  // Writing the whole merged config looked harmless but made the file a
  // snapshot: `proto train enable` would freeze every default of that day, and a
  // later improvement to a default (better batch size, higher dataset caps) would
  // never reach anyone who had already run a command. A delta file keeps the
  // user's explicit choices and lets improved defaults flow through, and it is
  // far easier to read.
  const delta = diffFromDefaults(config);
  writeJsonAtomic(path, { version: config.version, ...delta });
  return path;
}

/** Recursively keep only the keys whose value differs from DEFAULT_CONFIG. */
function diffFromDefaults(value: unknown, base: unknown = DEFAULT_CONFIG): Record<string, unknown> {
  const out: Record<string, unknown> = {};
  if (!isPlainObject(value)) return out;
  const defaults = isPlainObject(base) ? base : {};
  for (const [key, v] of Object.entries(value)) {
    if (key === 'dataDir' || key === 'version') continue; // derived / always written
    const d = defaults[key];
    if (isPlainObject(v) && isPlainObject(d)) {
      const nested = diffFromDefaults(v, d);
      if (Object.keys(nested).length > 0) out[key] = nested;
      continue;
    }
    if (JSON.stringify(v) === JSON.stringify(d)) continue;
    if (v === undefined && d === undefined) continue;
    out[key] = v;
  }
  return out;
}

/* ------------------------------------------------------------------ */
/* Secrets                                                             */
/* ------------------------------------------------------------------ */

/**
 * Read the API key for the configured cloud provider.
 *
 * Order: env vars named by the provider profile, then `PROTO_API_KEY`, then an
 * optional `var/secrets.json` (`{"provider": "key"}`). Secrets are never logged
 * and never included in episodes.
 */
export function resolveApiKeyFor(cfg: ProtoConfig, dataDir?: string): string | undefined {
  const profile = providerProfile(cfg.cloud.provider);
  const envNames = profile ? [...profile.keyEnv] : ['PROTO_API_KEY'];
  if (!envNames.includes('PROTO_API_KEY')) envNames.push('PROTO_API_KEY');
  for (const name of envNames) {
    const v = process.env[name];
    if (v && v.trim()) return v.trim();
  }
  // Note `||`, not `??`: an empty string is a *set* value to `??`, which would
  // resolve `join('', 'secrets.json')` relative to the cwd and silently miss the
  // real file.
  const dir = (dataDir?.trim() || cfg.dataDir?.trim() || defaultDataDir());
  const secrets = readJsonOrNull<Record<string, string>>(join(dir, 'secrets.json'));
  if (secrets) {
    for (const name of [cfg.cloud.provider, ...envNames, 'default']) {
      const v = secrets[name];
      if (v && v.trim()) return v.trim();
    }
  }
  return undefined;
}

/** Write a secret with 0600 perms. Used by `proto config set-key`. */
export function writeSecret(dataDir: string, provider: string, key: string): string {
  ensureDir(dataDir);
  const path = join(dataDir, 'secrets.json');
  const existing = readJsonOrNull<Record<string, string>>(path) ?? {};
  existing[provider] = key;
  writeJsonAtomic(path, existing);
  try {
    chmodSync(path, 0o600);
  } catch {
    /* best effort on filesystems without POSIX perms */
  }
  return path;
}

export function priceFor(cfg: ProtoConfig, model: string): Price {
  const exact = cfg.pricing[model];
  if (exact) return exact;
  // Try a provider-qualified fallback: "anthropic/claude-sonnet-4.5" -> "claude-sonnet-4-5".
  const tail = model.split('/').pop();
  if (tail && cfg.pricing[tail]) return cfg.pricing[tail] as Price;
  // Prefix match, so dated snapshots inherit their family price.
  const hit = Object.entries(cfg.pricing).find(([k]) => model.startsWith(k) || k.startsWith(model));
  if (hit) return hit[1];
  return { in: 5, out: 20 };
}

/** Resolve the effective cloud base URL (config override, else profile). */
export function cloudBaseUrl(cfg: ProtoConfig): string {
  if (cfg.cloud.baseUrl && cfg.cloud.baseUrl.trim()) return cfg.cloud.baseUrl.trim();
  const profile = providerProfile(cfg.cloud.provider);
  return profile?.baseUrl ?? 'https://api.openai.com/v1';
}

export function cloudApiShape(cfg: ProtoConfig): 'openai' | 'anthropic' {
  const profile = providerProfile(cfg.cloud.provider);
  return profile?.api ?? 'openai';
}
