/**
 * OpenAI-compatible provider.
 *
 * One implementation covers: OpenRouter, OpenAI, DeepSeek, Groq, Mistral,
 * Together, xAI, and the local servers (llama.cpp `llama-server`, LM Studio,
 * vLLM, `mlx_lm.server`). They all speak `POST {base}/chat/completions`.
 *
 * Deliberate omissions: multi-modal content (this is a coding harness).
 * `chatStream()` exists as a pure UX improvement over `chat()`; verification
 * still consumes only the fully-formed `ChatResponse`.
 * Unsupported `jsonSchema` requests degrade to a prompt-level instruction plus
 * best-effort parsing, because not every gateway supports `response_format`.
 */

import { computeCost, emptyResponse, ProviderError } from './types.ts';
import type {
  ChatRequest,
  ChatResponse,
  FinishReason,
  Health,
  Message,
  Provider,
  ProviderCapabilities,
  ProviderKind,
  StreamEvent,
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

  /**
   * The chat-completions body, shared by `chat()` and `chatStream()`.
   *
   * Both entry points must map the transcript identically — a divergence here
   * would make a streamed call send a different prompt (and cost) than the
   * non-streamed one.
   */
  private buildBody(req: ChatRequest): Record<string, unknown> {
    const body: Record<string, unknown> = {
      model: this.model,
      messages: toWireMessages(req.messages),
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
    return body;
  }

  async chat(req: ChatRequest): Promise<ChatResponse> {
    if (this.requireKey && !this.apiKey) {
      throw new ProviderError(this.id, `no API key configured for ${this.label}`, {
        retryable: false,
        hint: `Set ${this.label.toUpperCase().replace(/\W+/g, '_')}_API_KEY or run \`proto config set-key\`.`,
      });
    }

    const body = this.buildBody(req);

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

  /**
   * SSE streaming for the OpenAI-compatible wire format.
   *
   * Unlike Anthropic there are no content blocks: each `data:` line is a full
   * chunk, text arrives as `choices[0].delta.content`, and a tool call is
   * fragmented across chunks as `delta.tool_calls[]` entries keyed by `index`.
   * The `arguments` fragments are only concatenated (and therefore parseable) at
   * the end of the stream, so calls are buffered and emitted then.
   */
  async chatStream(req: ChatRequest, onEvent: (event: StreamEvent) => void): Promise<ChatResponse> {
    if (this.requireKey && !this.apiKey) {
      throw new ProviderError(this.id, `no API key configured for ${this.label}`, {
        retryable: false,
        hint: `Set ${this.label.toUpperCase().replace(/\W+/g, '_')}_API_KEY or run \`proto config set-key\`.`,
      });
    }

    const body = this.buildBody(req);
    body['stream'] = true;

    const started = Date.now();
    const controller = new AbortController();
    const timer = setTimeout(() => controller.abort(new Error('timeout')), this.timeoutMs);
    const onOuterAbort = (): void => controller.abort(new Error('aborted'));
    if (req.signal) {
      if (req.signal.aborted) controller.abort(new Error('aborted'));
      else req.signal.addEventListener('abort', onOuterAbort, { once: true });
    }

    try {
      let res;
      try {
        // `request()` from util/http.ts calls `res.text()` before returning, so it
        // buffers the whole body and can never deliver a delta. Streaming uses
        // global fetch directly and reads `response.body` incrementally.
        res = await fetch(joinUrl(this.baseUrl, 'chat/completions'), {
          method: 'POST',
          headers: this.headers(req),
          body: JSON.stringify(body),
          signal: controller.signal,
        });
      } catch (err) {
        throw toProviderError(this.id, err, this.baseUrl);
      }

      if (!res.ok) {
        const detail = extractError(await res.text().catch(() => ''));
        throw new ProviderError(this.id, `${this.label} HTTP ${res.status}: ${detail}`, {
          status: res.status,
          retryable: res.status === 429 || res.status >= 500,
        });
      }

      // Some local runtimes ignore `stream: true` and answer with a normal JSON
      // completion. Detect that from the content type and normalize it as usual
      // rather than feeding a JSON body to the SSE parser.
      const contentType = (res.headers.get('content-type') ?? '').toLowerCase();
      if (!contentType.includes('text/event-stream')) {
        const latencyMs = Date.now() - started;
        let parsed: unknown;
        try {
          parsed = JSON.parse(await res.text());
        } catch {
          return emptyResponse(this.id, this.model, latencyMs, 'response was not valid JSON');
        }
        const response = this.normalize(parsed, latencyMs, req);
        // Surface the whole completion as one delta so a streaming caller always
        // receives the content through the same callback, not just in the result.
        if (response.text) onEvent({ type: 'text-delta', text: response.text });
        for (const toolCall of response.toolCalls) onEvent({ type: 'tool-call', toolCall });
        onEvent({ type: 'done', response });
        return response;
      }

      let text = '';
      let finish: FinishReason = 'stop';
      let model = this.model;
      let rawUsage: Record<string, unknown> | undefined;
      let terminated = false;
      const fragments = new Map<number, { id?: string; name: string; args: string }>();

      const handleFrame = (_eventName: string, data: string): void => {
        if (terminated) return;
        if (data === '[DONE]') {
          terminated = true; // the stream is finished; ignore anything after it
          return;
        }
        let chunk: Record<string, unknown>;
        try {
          chunk = JSON.parse(data) as Record<string, unknown>;
        } catch {
          return; // one malformed chunk must not abort an otherwise good stream
        }
        if (typeof chunk['model'] === 'string') model = chunk['model'];
        const usage = chunk['usage'];
        if (usage && typeof usage === 'object') rawUsage = usage as Record<string, unknown>;

        const choices = Array.isArray(chunk['choices']) ? (chunk['choices'] as unknown[]) : [];
        const first = (choices[0] ?? {}) as Record<string, unknown>;
        const delta = (first['delta'] ?? {}) as Record<string, unknown>;

        if (typeof delta['content'] === 'string' && delta['content']) {
          text += delta['content'];
          onEvent({ type: 'text-delta', text: delta['content'] });
        }

        if (Array.isArray(delta['tool_calls'])) {
          for (const raw of delta['tool_calls'] as unknown[]) {
            const tc = (raw ?? {}) as Record<string, unknown>;
            const index = numOr(tc['index'], 0) ?? 0;
            let fragment = fragments.get(index);
            if (!fragment) {
              fragment = { name: '', args: '' };
              fragments.set(index, fragment);
            }
            // id/name are sent once, on the first fragment for that index.
            if (typeof tc['id'] === 'string' && tc['id'] && !fragment.id) fragment.id = tc['id'];
            const fn = (tc['function'] ?? {}) as Record<string, unknown>;
            if (typeof fn['name'] === 'string' && fn['name'] && !fragment.name) fragment.name = fn['name'];
            if (typeof fn['arguments'] === 'string') fragment.args += fn['arguments'];
          }
        }

        if (first['finish_reason'] !== undefined && first['finish_reason'] !== null) {
          finish = mapFinish(first['finish_reason']);
        }
      };

      const reader = res.body?.getReader();
      if (!reader) throw new Error('streaming response had no body');
      const decoder = new TextDecoder();
      const parser = makeSseParser(handleFrame);
      try {
        for (;;) {
          const { done, value } = await reader.read();
          if (done) break;
          // `stream: true` keeps a multi-byte character split across two chunks intact.
          parser.push(decoder.decode(value, { stream: true }));
        }
        parser.push(decoder.decode());
      } finally {
        parser.flush(); // a server may end the stream without the final blank line
        reader.releaseLock();
      }

      const toolCalls: ToolCall[] = [];
      for (const index of [...fragments.keys()].sort((a, b) => a - b)) {
        const fragment = fragments.get(index);
        if (!fragment) continue;
        const toolCall: ToolCall = {
          id: fragment.id ?? `call_${index}`,
          name: fragment.name,
          args: parseToolArgs(fragment.args),
        };
        toolCalls.push(toolCall);
        onEvent({ type: 'tool-call', toolCall });
      }

      const latencyMs = Date.now() - started;
      const inputTokens = numOr(
        rawUsage?.['prompt_tokens'],
        estimateTokens(req.messages.map((m) => m.content).join('\n')),
      );
      const outputTokens = numOr(rawUsage?.['completion_tokens'], estimateTokens(text));
      const cached = numOr(
        (rawUsage?.['prompt_tokens_details'] as Record<string, unknown> | undefined)?.['cached_tokens'],
        undefined,
      );
      const reasoning = numOr(
        (rawUsage?.['completion_tokens_details'] as Record<string, unknown> | undefined)?.['reasoning_tokens'],
        undefined,
      );
      const usage: Usage = { inputTokens: inputTokens ?? 0, outputTokens: outputTokens ?? 0 };
      if (cached !== undefined) usage.cachedInputTokens = cached;
      if (reasoning !== undefined) usage.reasoningTokens = reasoning;

      let response: ChatResponse;
      if (!text && toolCalls.length === 0 && finish === 'stop') {
        // Same rule as `normalize`: an empty completion is a model-level failure,
        // not a transport one, so it must not look like a successful stop.
        response = {
          ...emptyResponse(this.id, this.model, latencyMs, 'provider returned an empty message'),
          usage,
          finishReason: 'error',
        };
      } else {
        response = {
          text,
          toolCalls,
          usage,
          finishReason: finish,
          model,
          providerId: this.id,
          latencyMs,
          costUsd: computeCost(usage, this.price),
        };
      }
      onEvent({ type: 'done', response });
      return response;
    } catch (err) {
      throw toProviderError(this.id, err, this.baseUrl);
    } finally {
      clearTimeout(timer);
      if (req.signal) req.signal.removeEventListener('abort', onOuterAbort);
    }
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

/**
 * Map the internal transcript onto the chat-completions wire format.
 *
 * Both `chat()` and `chatStream()` go through here. An assistant turn that
 * requested tools must carry `tool_calls` (with `arguments` as a JSON *string*),
 * because the following `role: "tool"` results are matched to those ids; without
 * them the API returns a 400 instead of running the tool.
 */
function toWireMessages(messages: Message[]): unknown[] {
  return messages.map((m) => ({
    role: m.role,
    content: m.content,
    ...(m.toolCallId ? { tool_call_id: m.toolCallId } : {}),
    ...(m.name && m.role === 'tool' ? { name: m.name } : {}),
    ...(m.toolCalls?.length
      ? {
          tool_calls: m.toolCalls.map((c) => ({
            id: c.id,
            type: 'function',
            function: { name: c.name, arguments: JSON.stringify(c.args ?? {}) },
          })),
        }
      : {}),
  }));
}

/**
 * Tool arguments arrive as JSON fragments; the concatenation is only valid once
 * the stream ends. Malformed or absent JSON still describes a callable tool, so
 * fall back to `{}` rather than throwing away the call.
 */
function parseToolArgs(json: string): unknown {
  if (!json.trim()) return {};
  try {
    return JSON.parse(json);
  } catch {
    return {};
  }
}

/**
 * Minimal SSE frame splitter.
 *
 * Frames are separated by a blank line and carry `event:` / `data:` lines. A
 * network chunk boundary can fall anywhere — mid-line, mid-frame, even
 * mid-character (the caller decodes with `{ stream: true }`) — so this buffers
 * text and only dispatches when it sees the blank line ending a frame. Comment
 * lines (leading `:`) and `data: [DONE]` are handled by the caller.
 */
function makeSseParser(onFrame: (eventName: string, data: string) => void): { push(chunk: string): void; flush(): void } {
  let pending = '';
  let eventName = '';
  let dataLines: string[] = [];

  const dispatch = (): void => {
    const data = dataLines.join('\n');
    const name = eventName;
    eventName = '';
    dataLines = [];
    if (!data) return; // keepalive or comment-only frame: nothing to parse
    onFrame(name, data);
  };

  const handleLine = (raw: string): void => {
    const line = raw.endsWith('\r') ? raw.slice(0, -1) : raw;
    if (line === '') {
      dispatch();
      return;
    }
    if (line.startsWith(':')) return; // SSE comment, e.g. `: ping`
    const colon = line.indexOf(':');
    const field = colon === -1 ? line : line.slice(0, colon);
    let value = colon === -1 ? '' : line.slice(colon + 1);
    if (value.startsWith(' ')) value = value.slice(1); // one optional space after the colon, per spec
    if (field === 'event') eventName = value;
    else if (field === 'data') dataLines.push(value);
  };

  return {
    push(chunk: string): void {
      pending += chunk;
      // Split on \n and keep the trailing partial line: a chunk can end mid-line.
      const lines = pending.split('\n');
      pending = lines.pop() ?? '';
      for (const line of lines) handleLine(line);
    },
    flush(): void {
      if (pending) {
        const line = pending;
        pending = '';
        handleLine(line);
      }
      dispatch(); // a generous server may omit the trailing blank line
    },
  };
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
