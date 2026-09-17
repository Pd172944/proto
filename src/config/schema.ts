/**
 * Configuration schema, defaults, provider profiles and the pricing table.
 *
 * Principles:
 *  1. **No secrets in config.** The config names an *environment variable*; the
 *     secret itself is read from the environment (or an optional
 *     `var/secrets.json` with 0600 permissions). Config files are the kind of
 *     thing users paste into issues and commit by accident.
 *  2. **Everything is overridable from the environment** so the harness can be
 *     driven from a shell script or a launchd job without editing JSON.
 *  3. **Defaults are conservative about the user's machine.** Local training is
 *     off until explicitly enabled; nothing downloads anything.
 */

import type { Tier } from '../router/types.ts';

/* ------------------------------------------------------------------ */
/* Provider profiles                                                   */
/* ------------------------------------------------------------------ */

export interface ProviderProfile {
  id: string;
  label: string;
  /** OpenAI-compatible chat-completions endpoint, or `anthropic` for the native shape. */
  api: 'openai' | 'anthropic';
  baseUrl: string;
  /** Env var consulted for the API key, in order. */
  keyEnv: string[];
  /** A sensible, current default model id. */
  defaultModel: string;
  /** Cheap/fast model on the same provider, used when routing an easy task to the cloud. */
  cheapModel?: string;
  /** The strongest model, used for hard tasks when the user has not chosen. */
  strongModel?: string;
  docsUrl: string;
  /** Optional extra headers some gateways require. */
  extraHeaders?: Record<string, string>;
  notes?: string;
}

/**
 * Known cloud providers. All of these expose an OpenAI-compatible
 * `/chat/completions` except Anthropic, which we also implement natively
 * because it is one of the two providers the project was designed around.
 */
export const PROVIDER_PROFILES: ProviderProfile[] = [
  {
    id: 'openrouter',
    label: 'OpenRouter',
    api: 'openai',
    baseUrl: 'https://openrouter.ai/api/v1',
    keyEnv: ['OPENROUTER_API_KEY', 'PROTO_API_KEY'],
    defaultModel: 'anthropic/claude-sonnet-4.5',
    cheapModel: 'qwen/qwen3-coder-30b-a3b-instruct',
    strongModel: 'anthropic/claude-sonnet-4.5',
    docsUrl: 'https://openrouter.ai/docs',
    notes: 'One key, many models. Best default if you want to switch tiers often.',
  },
  {
    id: 'anthropic',
    label: 'Anthropic',
    api: 'anthropic',
    baseUrl: 'https://api.anthropic.com/v1',
    keyEnv: ['ANTHROPIC_API_KEY', 'PROTO_API_KEY'],
    defaultModel: 'claude-sonnet-4-5',
    strongModel: 'claude-sonnet-4-5',
    cheapModel: 'claude-haiku-4-5',
    docsUrl: 'https://docs.anthropic.com/en/api/messages',
    notes: 'Native Messages API with prompt caching; used as the reference "hard task" model.',
  },
  {
    id: 'openai',
    label: 'OpenAI',
    api: 'openai',
    baseUrl: 'https://api.openai.com/v1',
    keyEnv: ['OPENAI_API_KEY', 'PROTO_API_KEY'],
    defaultModel: 'gpt-5',
    strongModel: 'gpt-5',
    docsUrl: 'https://platform.openai.com/docs/api-reference',
  },
  {
    id: 'deepseek',
    label: 'DeepSeek',
    api: 'openai',
    baseUrl: 'https://api.deepseek.com/v1',
    keyEnv: ['DEEPSEEK_API_KEY', 'PROTO_API_KEY'],
    defaultModel: 'deepseek-chat',
    cheapModel: 'deepseek-chat',
    strongModel: 'deepseek-reasoner',
    docsUrl: 'https://api-docs.deepseek.com/',
    notes: 'Very cheap; a good "hard tier" choice when cost matters more than peak quality.',
  },
  {
    id: 'groq',
    label: 'Groq',
    api: 'openai',
    baseUrl: 'https://api.groq.com/openai/v1',
    keyEnv: ['GROQ_API_KEY', 'PROTO_API_KEY'],
    defaultModel: 'llama-3.3-70b-versatile',
    docsUrl: 'https://console.groq.com/docs/openai',
  },
  {
    id: 'mistral',
    label: 'Mistral',
    api: 'openai',
    baseUrl: 'https://api.mistral.ai/v1',
    keyEnv: ['MISTRAL_API_KEY', 'PROTO_API_KEY'],
    defaultModel: 'mistral-large-latest',
    docsUrl: 'https://docs.mistral.ai/api/',
  },
  {
    id: 'together',
    label: 'Together AI',
    api: 'openai',
    baseUrl: 'https://api.together.xyz/v1',
    keyEnv: ['TOGETHER_API_KEY', 'PROTO_API_KEY'],
    defaultModel: 'Qwen/Qwen3-Coder-480B-A35B-Instruct-FP8',
    docsUrl: 'https://docs.together.ai/docs/openai-api-compatibility',
  },
  {
    id: 'xai',
    label: 'xAI',
    api: 'openai',
    baseUrl: 'https://api.x.ai/v1',
    keyEnv: ['XAI_API_KEY', 'PROTO_API_KEY'],
    defaultModel: 'grok-4',
    docsUrl: 'https://docs.x.ai/docs/api-reference',
  },
  {
    id: 'custom',
    label: 'Custom OpenAI-compatible endpoint',
    api: 'openai',
    baseUrl: 'http://127.0.0.1:8080/v1',
    keyEnv: ['PROTO_API_KEY'],
    defaultModel: 'unset',
    docsUrl: '',
    notes: 'Any gateway: a proxy, a self-hosted vLLM box, a corporate endpoint.',
  },
];

export function providerProfile(id: string): ProviderProfile | undefined {
  return PROVIDER_PROFILES.find((p) => p.id === id);
}

/* ------------------------------------------------------------------ */
/* Pricing                                                             */
/* ------------------------------------------------------------------ */

export interface Price {
  /** USD per million input tokens. */
  in: number;
  /** USD per million output tokens. */
  out: number;
  /** USD per million cached input tokens, when the provider discounts them. */
  cachedIn?: number;
}

/**
 * Approximate list prices, USD per million tokens, as of 2025-06.
 *
 * These are used ONLY to make routing decisions and to give the user a rough
 * spend estimate — never for billing. They go stale; `proto config set
 * pricing.<model>.in <n>` overrides a single entry, and every consumer is
 * expected to treat the result as an estimate. Unknown models fall back to
 * `UNKNOWN_PRICE`, which is deliberately high so that an unpriced cloud model
 * is never treated as free.
 */
export const DEFAULT_PRICING: Record<string, Price> = {
  'thinkingmachines/inkling:free': { in: 0, out: 0 },
  'anthropic/claude-sonnet-4.5': { in: 3, out: 15, cachedIn: 0.3 },
  'claude-sonnet-4-5': { in: 3, out: 15, cachedIn: 0.3 },
  'claude-haiku-4-5': { in: 1, out: 5, cachedIn: 0.1 },
  'gpt-5': { in: 1.25, out: 10, cachedIn: 0.125 },
  'gpt-5-mini': { in: 0.25, out: 2 },
  'deepseek-chat': { in: 0.27, out: 1.1, cachedIn: 0.07 },
  'deepseek-reasoner': { in: 0.55, out: 2.19, cachedIn: 0.14 },
  'qwen/qwen3-coder-30b-a3b-instruct': { in: 0.15, out: 0.6 },
  'Qwen/Qwen3-Coder-480B-A35B-Instruct-FP8': { in: 0.9, out: 0.9 },
  'llama-3.3-70b-versatile': { in: 0.59, out: 0.79 },
  'mistral-large-latest': { in: 2, out: 6 },
  'grok-4': { in: 3, out: 15 },
};

export const UNKNOWN_PRICE: Price = { in: 5, out: 20 };

/* ------------------------------------------------------------------ */
/* Config                                                              */
/* ------------------------------------------------------------------ */

export interface LocalConfig {
  enabled: boolean;
  /**
   * `ollama`        — recommended default; one-line install, model marketplace.
   * `llamacpp`      — llama.cpp `llama-server` (OpenAI-compatible).
   * `mlx`           — `mlx_lm.server`; the fastest path on Apple Silicon for many models.
   * `openai`        — any other OpenAI-compatible local server (LM Studio, vLLM).
   */
  runtime: 'ollama' | 'llamacpp' | 'mlx' | 'openai';
  baseUrl: string;
  /** Primary local model id, as the runtime names it. */
  model: string;
  /** Optional second local model for the very easiest tasks (e.g. a 1.5B). */
  tinyModel?: string;
  /** Unload the model after this many seconds idle to free RAM. */
  keepAliveSec: number;
  requestTimeoutMs: number;
  /** Context window to request; keep modest to bound KV-cache memory. */
  contextWindow: number;
  maxOutputTokens: number;
  temperature: number;
}

export interface CloudConfig {
  enabled: boolean;
  /** Key into PROVIDER_PROFILES. */
  provider: string;
  model: string;
  /**
   * Optional cheaper/faster model on the same provider, used for the
   * `cloud-cheap` tier (easy-to-moderate tasks that local could not take, e.g.
   * because the local runtime was down). Falls back to `model`.
   */
  cheapModel?: string;
  /** Optional override for gateways with a different path. */
  baseUrl?: string;
  /** Reasoning effort hint where supported (OpenAI `reasoning_effort`, Anthropic thinking budget). */
  effort: 'auto' | 'low' | 'medium' | 'high';
  maxOutputTokens: number;
  temperature: number;
  requestTimeoutMs: number;
  /** Prompt-cache the stable prefix when the provider supports it. */
  promptCaching: boolean;
  /**
   * Extra headers sent with every cloud request.
   *
   * A general escape hatch, because gateways and enterprise proxies routinely need
   * one: Azure wants `api-version`, some proxies want a routing or tracing header.
   * Reaching for a code change every time a proxy wants a header is not reasonable.
   */
  extraHeaders?: Record<string, string>;
  /**
   * Anthropic workspace id, sent as the `anthropic-workspace-id` header.
   *
   * Some Anthropic keys are **not scoped to a workspace**, and the API rejects every
   * request from them with a 400 until this header is present:
   *
   *   "This API key is not scoped to a workspace, so this request must include the
   *    anthropic-workspace-id header with the ID of the workspace to use."
   *
   * The alternative fix is to create a key *inside* a workspace in the Anthropic
   * Console, which then needs no header at all. Both are supported; see
   * docs/interactive.md.
   */
  workspaceId?: string;
}

export interface RoutingConfig {
  /** `heuristic` = rules only; `learned` = model only; `hybrid` = learned blended with rules. */
  /** Minimum probability the local model succeeds before we let it try. */
  qualityFloor: number;
  /** Minimum probability for *mutating* tasks with no automatic verifier. */
  qualityFloorUnverified: number;
  /**
   * Minimum probability for read-only tasks with no verifier (explanations,
   * summaries, review). These are self-checking: the human reads the answer and
   * ignores it if it is wrong, so the downside of a weak model is much smaller
   * than for a bad patch that silently lands in the codebase.
   */
  qualityFloorReadOnly: number;
  /** How many times the local model may repair its own output before escalating. */
  maxLocalRepairAttempts: number;
  /** Hard cap on cloud calls per task, to bound worst-case spend. */
  maxCloudAttempts: number;
  /** Refuse cloud calls once today's estimated spend exceeds this. */
  cloudBudgetUsdPerDay: number;
  /** Local latency must not exceed cloud's by more than this for local to win. */
  latencyToleranceFactor: number;
  /** Estimated time-to-first-token for a cold local model (cost of a cold start). */
  localColdStartMs: number;
}

export interface VerifyConfig {
  enabled: boolean;
  /** Run the project's tests when verifying (opt-in: can be slow/destructive). */
  runTests: boolean;
  testCommand?: string;
  /** Wall-clock cap for the test command. */
  testTimeoutMs: number;
  /** Reject any candidate patch larger than this many changed lines. */
  maxPatchLines: number;
  /** Extra regexes that should cause a candidate to be rejected. */
  forbiddenPatterns: string[];
  /** Reject candidates that introduce new TODO/FIXME markers. */
  rejectNewTodos: boolean;
}




export interface ProtoConfig {
  version: 1;
  /** Where all harness state lives. Defaults to `<repo>/var`. */
  dataDir: string;
  local: LocalConfig;
  cloud: CloudConfig;
  routing: RoutingConfig;
  verify: VerifyConfig;
  pricing: Record<string, Price>;
}

export const DEFAULT_CONFIG: ProtoConfig = {
  version: 1,
  dataDir: '', // resolved at load time
  local: {
    enabled: true,
    runtime: 'ollama',
    baseUrl: 'http://127.0.0.1:11434',
    model: 'qwen2.5-coder:1.5b-instruct',
    tinyModel: undefined,
    keepAliveSec: 300,
    requestTimeoutMs: 60_000,
    contextWindow: 8192,
    maxOutputTokens: 1536,
    temperature: 0.1,
  },
  cloud: {
    enabled: false,
    provider: 'openrouter',
    model: 'anthropic/claude-sonnet-4.5',
    cheapModel: undefined,
    effort: 'auto',
    maxOutputTokens: 4096,
    temperature: 0.2,
    requestTimeoutMs: 180_000,
    promptCaching: true,
  },
  routing: {
    qualityFloor: 0.72,
    qualityFloorUnverified: 0.9,
    qualityFloorReadOnly: 0.55,
    maxLocalRepairAttempts: 1,
    maxCloudAttempts: 2,
    cloudBudgetUsdPerDay: 5,
    latencyToleranceFactor: 1.5,
    localColdStartMs: 4000,
  },
  verify: {
    enabled: true,
    runTests: false,
    testTimeoutMs: 120_000,
    maxPatchLines: 400,
    forbiddenPatterns: [],
    rejectNewTodos: true,
  },
  pricing: DEFAULT_PRICING,
};

/** Tier names, kept here so config and router agree on the vocabulary. */
export const TIER_NAMES: Tier[] = ['local-tiny', 'local', 'cloud-cheap', 'cloud-strong'];
