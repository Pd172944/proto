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
   * `mlx`           — `mlx_lm.server`; required runtime once a LoRA adapter is active.
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
  /** Path to an exported LoRA adapter directory, if training has produced one. */
  adapterPath?: string;
  /** Ask the runtime to expose logprobs (enables offline scoring/GRPO later). */
  requestLogprobs: boolean;
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
  mode: 'heuristic' | 'learned' | 'hybrid';
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
  /**
   * Exploration: with probability `epsilon`, route a task the policy would
   * have sent to the cloud to the local model anyway — but ONLY when a
   * verifier can catch failure. Without this the router never observes the
   * counterfactuals it needs to learn from.
   */
  exploration: {
    enabled: boolean;
    epsilon: number;
    /** Never explore on tasks above this difficulty score. */
    maxDifficultyForExploration: number;
    /** Never explore when the local estimate exceeds this many tokens. */
    maxTokensForExploration: number;
  };
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

export interface MemoryConfig {
  enabled: boolean;
  /** Run the redactor before persisting. Turning this off is a supported but discouraged choice. */
  redact: boolean;
  retentionDays: number;
  /** Rotate to a new shard file above this size. */
  maxShardBytes: number;
  /** Cap on stored characters per candidate output. */
  maxOutputChars: number;
  /** Also store the raw task text. Off = only features + hashes are kept. */
  storeTaskText: boolean;
  /**
   * Store the exact prompt sent to the model. Required to build SFT/DPO
   * datasets later (a prompt cannot be faithfully reconstructed once the files
   * it described have changed), but it is also the most sensitive field, so it
   * is capped and always redacted.
   */
  storePrompts: boolean;
  maxPromptChars: number;
}

export interface TrainConfig {
  /** Master switch. Default false: we never touch the user's CPU uninvited. */
  enabled: boolean;
  /** Which trainer to drive. Only MLX LoRA is wired up today. */
  backend: 'mlx-lora';
  /**
   * The Hugging Face repo (or local path) that LoRA is applied to.
   *
   * This is deliberately separate from `local.model`: the inference model is
   * usually an Ollama tag (`qwen2.5-coder:1.5b-instruct`) while training needs a
   * Hugging Face repo that MLX can load. Conflating the two produces a confusing
   * failure where training "cannot find" a model that inference uses fine.
   */
  baseModel: string;
  /** Minimum number of new, rewarded episodes before a job is worth queueing. */
  minNewEpisodes: number;
  /** Allowed wall-clock window, local time, 24h. e.g. 1 -> 6 means 01:00-06:00. */
  windowStartHour: number;
  windowEndHour: number;
  /** Only train while on wall power. */
  requireAC: boolean;
  /** Skip when the machine is thermally throttled. */
  respectThermalState: boolean;
  /** Skip when 1-minute load average exceeds this. */
  maxLoadAverage: number;
  /** Skip when battery is below this and on battery (informational). */
  minBatteryPct: number;
  /** Hard cap on a single tuning session. */
  maxRuntimeMin: number;
  /** Daily compute budget across all sessions. */
  dailyBudgetMin: number;
  /** LoRA hyperparameters (deliberately tiny: this is meant to be nearly free). */
  lora: {
    layers: number;
    rank: number;
    scale: number;
    dropout: number;
    learningRate: number;
    batchSize: number;
    /**
     * Target number of passes over the training set.
     *
     * This replaces the old fixed `iters` knob. A step count is meaningless
     * without knowing how much data exists: 60 iterations at batch size 1 shows
     * the model 60 examples, which on a 400-row dataset is 15% of a single epoch
     * and cannot teach anything. What actually matters is how many times the model
     * sees the data, so that is what is configured.
     */
    epochs: number;
    /** Hard ceiling on optimisation steps, whatever the epoch target implies. */
    maxIters: number;
    maxSeqLen: number;
    /** Which training mode to use for the selected dataset. */
    mode: 'sft' | 'dpo';
  };
  /**
   * Cap on training rows per run.
   *
   * Deliberately separate from the `datasets build` caps: those exist so a human
   * can inspect a bounded sample, whereas these decide how much of your history
   * actually reaches the model. Discarding rows buys nothing.
   */
  maxSftSamples: number;
  maxDpoSamples: number;
  /**
   * Fallback seconds per optimisation step, used to size a session against the
   * time budget before any history exists. Once jobs have run, the scheduler
   * measures the real rate from their recorded duration and ignores this.
   */
  secondsPerStep: number;
  /** Keep at most this many adapters on disk. */
  keepAdapters: number;
  /** Auto-promote a freshly trained adapter to the serving runtime. */
  autoPromote: boolean;
}

export interface ContribConfig {
  /** Sharing anything is opt-in and off by default. */
  enabled: boolean;
  /** Share raw task/output text. Off means only features, hashes and preference labels. */
  shareCode: boolean;
  /** Upload endpoint. Empty = stage locally in the outbox and never send. */
  endpoint: string;
  /** Cap on bytes staged per day. */
  maxBytesPerDay: number;
  /** Rotate the device pseudonym salt every N days (forward privacy). */
  saltRotateDays: number;
  /** Require an explicit `--yes` on every upload even when enabled. */
  requireConfirmation: boolean;
}

export interface ProtoConfig {
  version: 1;
  /** Where all harness state lives. Defaults to `<repo>/var`. */
  dataDir: string;
  local: LocalConfig;
  cloud: CloudConfig;
  routing: RoutingConfig;
  verify: VerifyConfig;
  memory: MemoryConfig;
  train: TrainConfig;
  contrib: ContribConfig;
  pricing: Record<string, Price>;
  /** Extra regexes (as strings) for the redactor. */
  redactionPatterns: string[];
  /** Model used by the eval harness for the "expected tier" baseline. */
  eval: {
    /** `route` = offline routing metrics only; `live` also calls models. */
    mode: 'route' | 'live';
    liveSampleSize: number;
  };
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
    adapterPath: undefined,
    requestLogprobs: false,
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
    mode: 'hybrid',
    qualityFloor: 0.72,
    qualityFloorUnverified: 0.9,
    qualityFloorReadOnly: 0.55,
    exploration: {
      enabled: true,
      epsilon: 0.06,
      maxDifficultyForExploration: 0.45,
      maxTokensForExploration: 4000,
    },
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
  memory: {
    enabled: true,
    redact: true,
    retentionDays: 90,
    maxShardBytes: 8 * 1024 * 1024,
    maxOutputChars: 8000,
    storeTaskText: true,
    storePrompts: true,
    maxPromptChars: 12000,
  },
  train: {
    enabled: false,
    backend: 'mlx-lora',
    baseModel: 'mlx-community/Qwen2.5-Coder-1.5B-Instruct-4bit',
    minNewEpisodes: 25,
    windowStartHour: 1,
    windowEndHour: 6,
    requireAC: true,
    respectThermalState: true,
    maxLoadAverage: 4,
    minBatteryPct: 20,
    maxRuntimeMin: 20,
    dailyBudgetMin: 60,
    lora: {
      layers: 8,
      rank: 8,
      scale: 16,
      dropout: 0.05,
      learningRate: 1e-5,
      // Batch 4 rather than 1: a single-example gradient is mostly noise, so the
      // step budget is better spent on fewer, better-conditioned updates.
      batchSize: 4,
      epochs: 3,
      maxIters: 2000,
      maxSeqLen: 1024,
      mode: 'sft',
    },
    maxSftSamples: 2000,
    maxDpoSamples: 1000,
    secondsPerStep: 1.5,
    keepAdapters: 3,
    autoPromote: false,
  },
  contrib: {
    enabled: false,
    shareCode: false,
    endpoint: '',
    maxBytesPerDay: 5 * 1024 * 1024,
    saltRotateDays: 7,
    requireConfirmation: true,
  },
  pricing: DEFAULT_PRICING,
  redactionPatterns: [],
  eval: {
    mode: 'route',
    liveSampleSize: 10,
  },
};

/** Tier names, kept here so config and router agree on the vocabulary. */
export const TIER_NAMES: Tier[] = ['local-tiny', 'local', 'cloud-cheap', 'cloud-strong'];
