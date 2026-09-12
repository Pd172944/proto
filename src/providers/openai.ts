/**
 * OpenAI-compatible provider.
 *
 * One implementation covers: OpenRouter, OpenAI, DeepSeek, Groq, Mistral,
 * Together, xAI, and the local servers (llama.cpp `llama-server`, LM Studio,
 * vLLM, `mlx_lm.server`). They all speak `POST {base}/chat/completions`.
 *
 * Deliberate omissions: streaming (we need the whole candidate before
 * verification anyway) and multi-modal content (this is a coding harness).
 * Unsupported `jsonSchema` requests degrade to a prompt-level instruction plus
 * best-effort parsing, because not every gateway supports `response_format`.
 */

import { computeCost, emptyResponse, ProviderError } from './types.ts';
import type {
  ChatRequest,
  ChatResponse,
  FinishReason,
  Health,
  Provider,
  ProviderCapabilities,
  ProviderKind,
  ToolCall,
  Usage,
} from './types.ts';
import type { Price } from '../config/schema.ts';
import { HttpError, joinUrl, request } from '../util/http.ts';
import { estimateTokens } from '../util/text.ts';

export interface OpenAICompatibleOptions {
  id: string;
  label: string;
  kind: ProviderKind;
  baseUrl: string;
  model: string;
  apiKey?: string;
  price: Price;
  timeoutMs: number;
  capabilities?: Partial<ProviderCapabilities>;
  extraHeaders?: Record<string, string>;
  /** Send `response_format: {type:'json_schema'}`. Disabled for gateways that 400 on it. */
  supportsJsonSchema?: boolean;
  /** Local runtimes usually need no auth header at all. */
  requireKey?: boolean;
  headers?: (req: ChatRequest) => Record<string, string>;
}

const DEFAULT_CAPS: ProviderCapabilities = {
  tools: true,
  jsonSchema: false,
  streaming: true,
  promptCaching: false,
  contextWindow: 128_000,
  maxOutputTokens: 4096,
};

export class OpenAICompatibleProvider implements Provider {
  readonly id: string;
  readonly label: string;
  readonly kind: ProviderKind;
  readonly model: string;
  readonly capabilities: ProviderCapabilities;
  readonly baseUrl: string;
  readonly price: Price;
  private readonly apiKey: string | undefined;
  private readonly timeoutMs: number;
  private readonly extraHeaders: Record<string, string>;
  private readonly supportsJsonSchema: boolean;
  private readonly requireKey: boolean;
  private readonly headerHook: ((req: ChatRequest) => Record<string, string>) | undefined;

  constructor(opts: OpenAICompatibleOptions) {
    this.id = opts.id;
    this.label = opts.label;
    this.kind = opts.kind;
    this.model = opts.model;
    this.baseUrl = opts.baseUrl;
    this.price = opts.price;
    this.apiKey = opts.apiKey;
    this.timeoutMs = opts.timeoutMs;
    this.extraHeaders = opts.extraHeaders ?? {};
    this.supportsJsonSchema = opts.supportsJsonSchema ?? false;
    this.requireKey = opts.requireKey ?? opts.kind === 'cloud';
    this.headerHook = opts.headers;
    this.capabilities = { ...DEFAULT_CAPS, ...(opts.capabilities ?? {}) };
  }

  private headers(req: ChatRequest): Record<string, string> {
    const headers: Record<string, string> = {
      'content-type': 'application/json',
      ...this.extraHeaders,
      ...(this.headerHook?.(req) ?? {}),
    };
    if (this.apiKey) headers['authorization'] = `Bearer ${this.apiKey}`;
    // OpenRouter likes these for attribution; harmless elsewhere.
    if (this.baseUrl.includes('openrouter.ai')) {
      headers['http-referer'] = 'https://github.com/proto-harness';
      headers['x-title'] = 'proto-harness';
    }
    return headers;
  }

  async chat(req: ChatRequest): Promise<ChatResponse> {
    if (this.requireKey && !this.apiKey) {
      throw new ProviderError(this.id, `no API key configured for ${this.label}`, {
        retryable: false,
        hint: `Set ${this.label.toUpperCase().replace(/\W+/g, '_')}_API_KEY or run \`proto config set-key\`.`,
      });
    }

    const body: Record<string, unknown> = {
      model: this.model,
      messages: req.messages.map((m) => ({
        role: m.role,
        content: m.content,
        ...(m.toolCallId ? { tool_call_id: m.toolCallId } : {}),
        ...(m.name && m.role === 'tool' ? { name: m.name } : {}),
      })),
    };
    if (req.maxTokens !== undefined) body['max_tokens'] = req.maxTokens;
    if (req.temperature !== undefined) body['temperature'] = req.temperature;
    if (req.topP !== undefined) body['top_p'] = req.topP;
    if (req.stop?.length) body['stop'] = req.stop;
    if (req.tools?.length) {
      body['tools'] = req.tools.map((t) => ({
        type: 'function',
        function: { name: t.name, description: t.description, parameters: t.parameters },
      }));
      body['tool_choice'] = 'auto';
    }
    if (req.jsonSchema && this.supportsJsonSchema) {
      body['response_format'] = {
        type: 'json_schema',
        json_schema: { name: 'proto_output', strict: false, schema: req.jsonSchema },
      };
    }

    const started = Date.now();
    let res;
    try {
      res = await request({
        url: joinUrl(this.baseUrl, 'chat/completions'),
        method: 'POST',
        headers: this.headers(req),
        body: JSON.stringify(body),
        timeoutMs: this.timeoutMs,
        signal: req.signal,
        label: this.id,
      });
    } catch (err) {
      throw toProviderError(this.id, err, this.baseUrl);
    }

    const latencyMs = Date.now() - started;
    if (!res.ok) {
      const detail = extractError(res.text);
      throw new ProviderError(this.id, `${this.label} HTTP ${res.status}: ${detail}`, {
        status: res.status,
        retryable: res.status === 429 || res.status >= 500,
      });
    }

    let parsed: unknown;
    try {
      parsed = JSON.parse(res.text);
    } catch {
      return emptyResponse(this.id, this.model, latencyMs, 'response was not valid JSON');
    }
    return this.normalize(parsed, latencyMs, req);
  }

  private normalize(parsed: unknown, latencyMs: number, req: ChatRequest): ChatResponse {
    const root = parsed as Record<string, unknown> | null;
    const choices = Array.isArray(root?.['choices']) ? (root?.['choices'] as unknown[]) : [];
    const first = (choices[0] ?? {}) as Record<string, unknown>;
    const message = (first['message'] ?? {}) as Record<string, unknown>;

    const text = typeof message['content'] === 'string' ? message['content'] : '';
    const toolCalls = normalizeToolCalls(message['tool_calls']);
    const rawUsage = (root?.['usage'] ?? {}) as Record<string, unknown>;

    const inputTokens = numOr(rawUsage['prompt_tokens'], estimateTokens(req.messages.map((m) => m.content).join('\n')));
    const outputTokens = numOr(rawUsage['completion_tokens'], estimateTokens(text));
    const cached = numOr(
      (rawUsage['prompt_tokens_details'] as Record<string, unknown> | undefined)?.['cached_tokens'],
      undefined,
    );
    const reasoning = numOr(
      (rawUsage['completion_tokens_details'] as Record<string, unknown> | undefined)?.['reasoning_tokens'],
      undefined,
    );
    const usage: Usage = { inputTokens: inputTokens ?? 0, outputTokens: outputTokens ?? 0 };
    if (cached !== undefined) usage.cachedInputTokens = cached;
    if (reasoning !== undefined) usage.reasoningTokens = reasoning;

    const finish = mapFinish(first['finish_reason']);
    if (!text && toolCalls.length === 0 && finish === 'stop') {
      return {
        ...emptyResponse(this.id, this.model, latencyMs, 'provider returned an empty message'),
        usage,
        finishReason: 'error',
      };
    }

    return {
      text,
      toolCalls,
      usage,
      finishReason: finish,
      model: typeof root?.['model'] === 'string' ? (root['model'] as string) : this.model,
      providerId: this.id,
      latencyMs,
      costUsd: computeCost(usage, this.price),
    };
  }

  async health(): Promise<Health> {
    const started = Date.now();
    try {
      const res = await request({
        url: joinUrl(this.baseUrl, 'models'),
        method: 'GET',
        headers: this.headers({ messages: [] }),
        timeoutMs: Math.min(this.timeoutMs, 8000),
        retries: this.kind === 'cloud' ? 1 : 0,
        label: `${this.id}-health`,
      });
      const latencyMs = Date.now() - started;
      if (!res.ok) {
        return {
          ok: false,
          detail: `HTTP ${res.status} from ${this.baseUrl}/models`,
          latencyMs,
          hint:
            res.status === 401 || res.status === 403
              ? 'The API key looks invalid. Check the key and that the provider is correct.'
              : undefined,
        };
      }
      let models: string[] = [];
      try {
        const parsed = JSON.parse(res.text) as { data?: Array<{ id?: string }> };
        models = (parsed.data ?? []).map((m) => m.id ?? '').filter(Boolean);
      } catch {
        /* health is still ok; model listing is a nicety */
      }
      const hasModel = models.length === 0 || models.includes(this.model);
      return {
        ok: true,
        detail: `${this.label} reachable (${models.length} models advertised)`,
        latencyMs,
        models,
        ...(hasModel
          ? {}
          : { hint: `model "${this.model}" is not in the advertised list; it may 404 on first call` }),
      };
    } catch (err) {
      return {
        ok: false,
        detail: err instanceof Error ? err.message : String(err),
        latencyMs: Date.now() - started,
        unreachable: this.kind === 'local',
        hint:
          this.kind === 'local'
            ? `Nothing is listening on ${this.baseUrl}. Start your local runtime (see docs/local-models.md).`
            : 'Check network connectivity and the base URL.',
      };
    }
  }
}

function numOr(value: unknown, fallback: number | undefined): number | undefined {
  if (typeof value === 'number' && Number.isFinite(value)) return value;
  return fallback;
}

function mapFinish(reason: unknown): FinishReason {
  switch (reason) {
    case 'stop':
    case null:
    case undefined:
      return 'stop';
    case 'length':
    case 'max_tokens':
      return 'length';
    case 'tool_calls':
    case 'function_call':
      return 'tool_calls';
    case 'content_filter':
      return 'content_filter';
    default:
      return 'stop';
  }
}

function normalizeToolCalls(raw: unknown): ToolCall[] {
  if (!Array.isArray(raw)) return [];
  const out: ToolCall[] = [];
  for (const [i, item] of raw.entries()) {
    const tc = item as Record<string, unknown>;
    const fn = (tc['function'] ?? {}) as Record<string, unknown>;
    const name = typeof fn['name'] === 'string' ? fn['name'] : '';
    if (!name) continue;
    let args: unknown = fn['arguments'];
    if (typeof args === 'string') {
      try {
        args = JSON.parse(args);
      } catch {
        /* leave as string; caller decides */
      }
    }
    out.push({ id: typeof tc['id'] === 'string' ? tc['id'] : `call_${i}`, name, args });
  }
  return out;
}

function extractError(text: string): string {
  try {
    const parsed = JSON.parse(text) as Record<string, unknown>;
    const err = parsed['error'];
    if (typeof err === 'string') return err;
    if (err && typeof err === 'object') {
      const msg = (err as Record<string, unknown>)['message'];
      if (typeof msg === 'string') return msg;
    }
    if (typeof parsed['message'] === 'string') return parsed['message'];
  } catch {
    /* fall through to the raw body */
  }
  return text.slice(0, 300);
}

export function toProviderError(id: string, err: unknown, baseUrl: string): ProviderError {
  if (err instanceof ProviderError) return err;
  if (err instanceof HttpError) {
    return new ProviderError(id, err.message, { status: err.status, retryable: err.retryable });
  }
  const msg = err instanceof Error ? err.message : String(err);
  const unreachable = /ECONNREFUSED|fetch failed|ENOTFOUND|aborted|timeout/i.test(msg);
  return new ProviderError(id, msg, {
    retryable: true,
    hint: unreachable ? `Could not reach ${baseUrl}. Is the server running?` : undefined,
  });
}
