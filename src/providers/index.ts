/**
 * Provider construction from config.
 *
 * This is the only place that knows how a config maps onto a concrete model
 * endpoint. The local side is intentionally pluggable across four runtimes so
 * that the user's choice of download path (Ollama, llama.cpp, MLX) is not a
 * commitment baked into the harness.
 */

import { AnthropicProvider } from './anthropic.ts';
import { OllamaProvider } from './ollama.ts';
import { OpenAICompatibleProvider } from './openai.ts';
import type { Health, Provider } from './types.ts';
import type { ProtoConfig } from '../config/schema.ts';
import { providerProfile, UNKNOWN_PRICE } from '../config/schema.ts';
import { cloudApiShape, cloudBaseUrl, priceFor, resolveApiKeyFor } from '../config/load.ts';

/** Canonical default endpoint per local runtime, used for docs and auto-correction. */
export const LOCAL_RUNTIME_DEFAULTS: Record<ProtoConfig['local']['runtime'], { baseUrl: string; docs: string }> = {
  ollama: { baseUrl: 'http://127.0.0.1:11434', docs: 'https://ollama.com/download' },
  llamacpp: { baseUrl: 'http://127.0.0.1:8080', docs: 'https://github.com/ggml-org/llama.cpp' },
  mlx: { baseUrl: 'http://127.0.0.1:8080', docs: 'https://github.com/ml-explore/mlx-lm' },
  openai: { baseUrl: 'http://127.0.0.1:1234', docs: 'https://lmstudio.ai/docs/local-server' },
};

/**
 * If the user switches runtime but leaves `baseUrl` at another runtime's
 * canonical default, silently keep talking to the old port and blame the model.
 * Detect that specific case and correct it, reporting what happened.
 */
export function effectiveLocalBaseUrl(cfg: ProtoConfig): { url: string; corrected: boolean; note?: string } {
  const configured = cfg.local.baseUrl.replace(/\/+$/, '');
  const target = LOCAL_RUNTIME_DEFAULTS[cfg.local.runtime].baseUrl.replace(/\/+$/, '');
  if (configured === target) return { url: target, corrected: false };
  const isAnotherRuntimesDefault = Object.entries(LOCAL_RUNTIME_DEFAULTS).some(
    ([runtime, v]) => runtime !== cfg.local.runtime && v.baseUrl.replace(/\/+$/, '') === configured,
  );
  if (isAnotherRuntimesDefault) {
    return {
      url: target,
      corrected: true,
      note: `local.baseUrl still pointed at another runtime's default port; using ${target} for runtime "${cfg.local.runtime}"`,
    };
  }
  return { url: configured, corrected: false };
}

/**
 * Rank locally available models by how suitable they are for this harness.
 *
 * Used to suggest a model when the configured one is absent. A user who has pulled
 * `ornith-1.5:9b` should not be told "qwen2.5-coder:1.5b-instruct is not
 * downloaded" and left to work out the rest — but silently adopting whatever is
 * lying around would be worse, so this only ever produces a *suggestion*.
 *
 * Ordering, in order of weight:
 *  1. **Agentic / tool-use training first.** This harness is an agent, so a model
 *     post-trained for tool calling and scaffolds is more valuable here than one
 *     that merely scores well on single-shot code benchmarks. That is why `ornith`
 *     and `devstral` outrank a same-size general coder.
 *  2. Code-capable families next.
 *  3. Larger parameter counts, since a 9B generally beats a 1.5B on the same task.
 *  4. Instruction-tuned over base, because a base model will not follow tools.
 * Embedding and vision-only models are pushed to the bottom: suggesting one as the
 * coding tier would be worse than suggesting nothing.
 *
 * This only ever produces a *suggestion* — owner consent is required to change the
 * configured model, because silently adopting a 9B model on a small machine would
 * be a nasty surprise.
 */
export function rankLocalModels(models: string[]): string[] {
  const score = (model: string): number => {
    const m = model.toLowerCase();
    let s = 0;
    // 1. agentic / tool-use lineage
    if (/ornith|devstral|agent|tool-?use|hermes/.test(m)) s += 25;
    // 2. code-capable families
    if (/coder|code|starcoder|wizardcoder/.test(m)) s += 60;
    if (/instruct|chat|it\b|-it$/.test(m)) s += 8;
    if (/-base|^base/.test(m)) s -= 25;
    if (/embed|nomic|mxbai|bge-|rerank|whisper|llava|vision|clip|guard|moderation/.test(m)) s -= 1000;
    const params = m.match(/(\d+(?:\.\d+)?)\s*b\b/);
    if (params?.[1]) s += Math.min(30, Number(params[1]));
    return s;
  };
  return [...models].sort((a, b) => score(b) - score(a) || a.localeCompare(b));
}

export function buildLocalProvider(cfg: ProtoConfig): Provider {
  const { url, note } = effectiveLocalBaseUrl(cfg);
  if (note) {
    // Surfaced through doctor; a warning here keeps the correction auditable.
    process.emitWarning(note, { code: 'PROTO_LOCAL_BASE_URL' });
  }

  if (cfg.local.runtime === 'ollama') {
    return new OllamaProvider({
      id: 'local',
      baseUrl: url,
      model: cfg.local.model,
      keepAliveSec: cfg.local.keepAliveSec,
      timeoutMs: cfg.local.requestTimeoutMs,
      contextWindow: cfg.local.contextWindow,
      maxOutputTokens: cfg.local.maxOutputTokens,
      temperature: cfg.local.temperature,
    });
  }

  // Everything else is OpenAI-compatible. `mlx` in particular is served by
  // `mlx_lm.server`, which is how a trained LoRA adapter gets used.
  return new OpenAICompatibleProvider({
    id: 'local',
    label: cfg.local.runtime === 'mlx' ? 'MLX (local)' : cfg.local.runtime === 'llamacpp' ? 'llama.cpp (local)' : 'Local OpenAI-compatible',
    kind: 'local',
    baseUrl: url,
    model: cfg.local.model,
    price: { in: 0, out: 0 },
    timeoutMs: cfg.local.requestTimeoutMs,
    requireKey: false,
    supportsJsonSchema: cfg.local.runtime === 'mlx',
    capabilities: {
      contextWindow: cfg.local.contextWindow,
      maxOutputTokens: cfg.local.maxOutputTokens,
      tools: cfg.local.runtime !== 'llamacpp',
      jsonSchema: cfg.local.runtime === 'mlx',
      streaming: false,
    },
  });
}

/** Which cloud model a tier should use. `cloud-cheap` falls back to the main model. */
export function cloudTierModel(cfg: ProtoConfig, tier: 'cloud-cheap' | 'cloud-strong'): string {
  if (tier === 'cloud-cheap') {
    const cheap = cfg.cloud.cheapModel?.trim();
    if (cheap) return cheap;
  }
  return cfg.cloud.model;
}

export function buildCloudProvider(cfg: ProtoConfig, dataDir?: string, modelOverride?: string): Provider | null {
  const profile = providerProfile(cfg.cloud.provider);
  const apiKey = resolveApiKeyFor(cfg, dataDir);
  const baseUrl = cloudBaseUrl(cfg);
  const model = modelOverride ?? cfg.cloud.model;
  const price = priceFor(cfg, model);

  if (cloudApiShape(cfg) === 'anthropic') {
    return new AnthropicProvider({
      id: cfg.cloud.provider,
      label: profile?.label ?? 'Anthropic',
      model,
      baseUrl,
      apiKey,
      price: price ?? UNKNOWN_PRICE,
      timeoutMs: cfg.cloud.requestTimeoutMs,
      effort: cfg.cloud.effort,
      promptCaching: cfg.cloud.promptCaching,
    });
  }

  return new OpenAICompatibleProvider({
    id: cfg.cloud.provider,
    label: profile?.label ?? cfg.cloud.provider,
    kind: 'cloud',
    baseUrl,
    model,
    apiKey,
    price,
    timeoutMs: cfg.cloud.requestTimeoutMs,
    extraHeaders: profile?.extraHeaders,
    supportsJsonSchema: false,
  });
}

export interface ProviderSet {
  local: Provider;
  cloud: Provider | null;
  /** Reason the cloud tier is unavailable, when it is. */
  cloudUnavailable?: string;
}

export function buildProviders(cfg: ProtoConfig, dataDir?: string): ProviderSet {
  const local = buildLocalProvider(cfg);
  let cloud: Provider | null = null;
  let cloudUnavailable: string | undefined;

  if (!cfg.cloud.enabled) {
    cloudUnavailable = 'cloud is disabled in config (set cloud.enabled=true or provide an API key)';
  } else if (!resolveApiKeyFor(cfg, dataDir) && cloudApiShape(cfg) !== 'openai') {
    cloudUnavailable = `no API key found for provider "${cfg.cloud.provider}"`;
  } else if (!resolveApiKeyFor(cfg, dataDir)) {
    cloudUnavailable = `no API key found for provider "${cfg.cloud.provider}"`;
  } else {
    cloud = buildCloudProvider(cfg, dataDir);
  }

  return { local, cloud, cloudUnavailable };
}

export async function probeProviders(set: ProviderSet, opts: { includeCloud?: boolean } = {}): Promise<{
  local: Health;
  cloud: Health | null;
}> {
  const local = await set.local.health();
  let cloud: Health | null = null;
  if (set.cloud && opts.includeCloud !== false) cloud = await set.cloud.health();
  return { local, cloud };
}
