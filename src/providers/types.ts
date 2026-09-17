/**
 * The provider contract.
 *
 * Everything the harness knows about a model lives behind this interface, so
 * the router can reason about cost/latency/capability without knowing whether
 * the other end is a 1.5B model on the user's laptop or an API behind a key.
 *
 * Two rules that keep the rest of the system honest:
 *  - `chat()` never throws for a *model-level* failure (bad output, empty
 *    response); it returns a response with `finishReason: 'error'`. It throws
 *    `ProviderError` only for transport/auth problems the caller should treat
 *    as "this tier is unusable right now".
 *  - `costUsd` is always computed from normalized `Usage` + the local price
 *    table, so reported cost is comparable across providers.
 */

import type { Price } from '../config/schema.ts';

export type Role = 'system' | 'user' | 'assistant' | 'tool';

export interface Message {
  role: Role;
  content: string;
  /** Tool name for `tool` messages, or an optional speaker label. */
  name?: string;
  toolCallId?: string;
  /**
   * Tool calls the assistant requested in this turn.
   *
   * Both the Anthropic and OpenAI APIs require the assistant message to carry its
   * own tool-call blocks, and require each subsequent tool result to reference
   * them by id. A transcript flattened to plain text is rejected with a 400, so
   * these are part of the wire format, not a convenience.
   */
  toolCalls?: ToolCall[];
  /** Mark this message as a cache boundary where the provider supports it. */
  cacheBreakpoint?: boolean;
}

export interface ToolSpec {
  name: string;
  description: string;
  /** JSON Schema for the arguments. */
  parameters: Record<string, unknown>;
}

export interface ToolCall {
  id: string;
  name: string;
  args: unknown;
}

export interface Usage {
  inputTokens: number;
  outputTokens: number;
  cachedInputTokens?: number;
  reasoningTokens?: number;
  /** True when the numbers are heuristic estimates rather than provider-reported. */
  estimated?: boolean;
}

export type FinishReason = 'stop' | 'length' | 'tool_calls' | 'content_filter' | 'error';

export interface ChatRequest {
  messages: Message[];
  tools?: ToolSpec[];
  maxTokens?: number;
  temperature?: number;
  topP?: number;
  /** Ask for a JSON object conforming to this schema, when supported. */
  jsonSchema?: Record<string, unknown>;
  stop?: string[];
  signal?: AbortSignal;
  /**
   * Ask the model to think before answering, when it has such a mode.
   *
   * Provider-agnostic on purpose: the wire spelling differs (vLLM takes
   * `chat_template_kwargs.enable_thinking`, other gateways take their own), and the
   * harness should not have to know. `undefined` means "whatever the endpoint
   * defaults to", which is what a caller who has not thought about it wants.
   *
   * `false` is the interesting value: a reasoning model on a latency budget is
   * spending most of its tokens on thinking nobody reads. Measured on Qwen3.8-27B,
   * turning it off took one short answer from 2.94s to 0.38s.
   */
  thinking?: boolean;
  /** Extra fields merged into the request body verbatim. The escape hatch. */
  extraBody?: Record<string, unknown>;
  /** Free-form metadata for logging (never sent to the provider). */
  meta?: Record<string, unknown>;
}

export interface ChatResponse {
  text: string;
  /**
   * Thinking the model emitted before its answer, when it emits any. Never part of
   * `text`; present so a caller can show it, count it, or notice that a model spent
   * its whole budget reasoning instead of working.
   */
  reasoning?: string;
  toolCalls: ToolCall[];
  usage: Usage;
  finishReason: FinishReason;
  model: string;
  providerId: string;
  latencyMs: number;
  /** Estimated USD for this single call. */
  costUsd: number;
  /** Populated when finishReason === 'error'. */
  error?: string;
}

/** One incremental event from a streaming completion. */
export type StreamEvent =
  | { type: 'text-delta'; text: string }
  /**
   * A reasoning/thinking token from a model that emits one (Qwen3 via vLLM sends
   * `delta.reasoning`, DeepSeek sends `reasoning_content`).
   *
   * Kept separate from `text-delta` on purpose. Thinking is not the answer: mixing
   * it into the assistant text would corrupt every transcript, every fingerprint of
   * "did the model say anything useful", and every diff the harness tried to parse.
   * Callers that do not care simply ignore this event.
   */
  | { type: 'reasoning-delta'; text: string }
  | { type: 'tool-call'; toolCall: ToolCall }
  | { type: 'done'; response: ChatResponse };

export interface ProviderCapabilities {
  /** Supports native tool/function calling. */
  tools: boolean;
  /** Supports a JSON-schema-constrained response format. */
  jsonSchema: boolean;
  streaming: boolean;
  /** Accepts `cache_control` style prompt caching. */
  promptCaching: boolean;
  contextWindow: number;
  maxOutputTokens: number;
}

export type ProviderKind = 'cloud' | 'local';

export interface Provider {
  readonly id: string;
  readonly label: string;
  readonly kind: ProviderKind;
  readonly model: string;
  readonly capabilities: ProviderCapabilities;
  chat(req: ChatRequest): Promise<ChatResponse>;
  /**
   * Stream a completion, invoking `onEvent` as deltas arrive, and resolve with the
   * same `ChatResponse` that `chat()` would have returned.
   *
   * Optional because not every runtime supports it: `OllamaProvider` does not
   * implement it today, so callers MUST fall back to `chat()` when it is absent.
   * The resolved response must be identical to the non-streaming result, so that
   * streaming is a pure UX improvement and never changes behaviour, cost
   * accounting or verification.
   */
  chatStream?(req: ChatRequest, onEvent: (event: StreamEvent) => void): Promise<ChatResponse>;
  /** Cheap liveness probe; used by `doctor` and by the router's tier vetoes. */
  health(): Promise<Health>;
}

export interface Health {
  ok: boolean;
  detail: string;
  latencyMs?: number;
  /** Models the runtime reports as available locally. */
  models?: string[];
  /** True when the endpoint is refusing connections, i.e. server not running. */
  unreachable?: boolean;
  /** Actionable next step shown directly to the user. */
  hint?: string;
}

export class ProviderError extends Error {
  readonly providerId: string;
  readonly status: number;
  readonly retryable: boolean;
  readonly hint?: string;

  constructor(
    providerId: string,
    message: string,
    opts: { status?: number; retryable?: boolean; hint?: string } = {},
  ) {
    super(message);
    this.name = 'ProviderError';
    this.providerId = providerId;
    this.status = opts.status ?? 0;
    this.retryable = opts.retryable ?? false;
    this.hint = opts.hint;
  }
}

/** Compute USD cost from normalized usage. */
export function computeCost(usage: Usage, price: Price): number {
  const cached = usage.cachedInputTokens ?? 0;
  const fresh = Math.max(0, usage.inputTokens - cached);
  const cachedRate = price.cachedIn ?? price.in;
  const usd =
    (fresh / 1_000_000) * price.in +
    (cached / 1_000_000) * cachedRate +
    (usage.outputTokens / 1_000_000) * price.out;
  // Round to sub-cent precision; keeping full float noise in logs is unhelpful.
  return Math.round(usd * 1e6) / 1e6;
}

export function emptyResponse(
  providerId: string,
  model: string,
  latencyMs: number,
  error: string,
): ChatResponse {
  return {
    text: '',
    toolCalls: [],
    usage: { inputTokens: 0, outputTokens: 0 },
    finishReason: 'error',
    model,
    providerId,
    latencyMs,
    costUsd: 0,
    error,
  };
}
