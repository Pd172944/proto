/**
 * Anthropic native Messages API provider.
 *
 * Implemented natively (rather than only via the OpenAI-compatibility shim)
 * for three reasons that matter to this harness:
 *  1. Prompt caching via `cache_control` — the system preamble (task spec +
 *     conventions) is large and identical across attempts, so caching it cuts
 *     the cloud cost of escalation noticeably.
 *  2. Extended thinking budgets, so "high effort" is a real, configurable knob
 *     rather than a prompt-level suggestion.
 *  3. `stop_reason` / `usage` are reported precisely, which keeps the episode
 *     cost accounting comparable to other providers.
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
  StreamEvent,
  ToolCall,
  Usage,
} from './types.ts';
import type { Price } from '../config/schema.ts';
import { joinUrl, request } from '../util/http.ts';
import { estimateTokens } from '../util/text.ts';
import { toProviderError } from './openai.ts';

export interface AnthropicOptions {
  id: string;
  label: string;
  model: string;
  baseUrl?: string;
  apiKey?: string;
  price: Price;
  timeoutMs: number;
  /** 'auto' | 'low' | 'medium' | 'high' -> thinking token budget. */
  effort: 'auto' | 'low' | 'medium' | 'high';
  promptCaching: boolean;
  /** Extra headers to send with every request. */
  extraHeaders?: Record<string, string>;
  /** Sent as `anthropic-workspace-id` when set. */
  workspaceId?: string;
  version?: string;
}

const ANTHROPIC_VERSION = '2023-06-01';

const THINKING_BUDGET: Record<string, number> = {
  low: 1024,
  medium: 4096,
  high: 12_000,
};

export class AnthropicProvider implements Provider {
  readonly id: string;
  readonly label = 'Anthropic';
  readonly kind = 'cloud' as const;
  readonly model: string;
  readonly capabilities: ProviderCapabilities = {
    tools: true,
    jsonSchema: false,
    streaming: true,
    promptCaching: true,
    contextWindow: 200_000,
    maxOutputTokens: 8192,
  };
  readonly price: Price;
  private readonly baseUrl: string;
  private readonly apiKey: string | undefined;
  private readonly timeoutMs: number;
  private readonly effort: 'auto' | 'low' | 'medium' | 'high';
  private readonly promptCaching: boolean;
  private readonly version: string;
  private readonly extraHeaders: Record<string, string>;

  constructor(opts: AnthropicOptions) {
    this.id = opts.id;
    this.model = opts.model;
    this.baseUrl = opts.baseUrl ?? 'https://api.anthropic.com/v1';
    this.apiKey = opts.apiKey;
    this.price = opts.price;
    this.timeoutMs = opts.timeoutMs;
    this.effort = opts.effort;
    this.promptCaching = opts.promptCaching;
    this.version = opts.version ?? ANTHROPIC_VERSION;
    this.extraHeaders = { ...(opts.extraHeaders ?? {}) };
    // Some keys are not scoped to a workspace and are rejected outright without this
    // header. Sending it is harmless for keys that are scoped.
    const workspace = opts.workspaceId?.trim();
    if (workspace) this.extraHeaders['anthropic-workspace-id'] = workspace;
  }

  private headers(): Record<string, string> {
    const h: Record<string, string> = {
      'content-type': 'application/json',
      'anthropic-version': this.version,
      ...this.extraHeaders,
    };
    if (this.apiKey) h['x-api-key'] = this.apiKey;
    return h;
  }

  /**
   * The Messages API body, shared by `chat()` and `chatStream()`.
   *
   * One builder keeps the streaming request byte-identical to the non-streaming
   * one (same prompt shape, caching markers and thinking budget), so the two
   * paths cannot drift in cost or behaviour.
   */
  private buildBody(req: ChatRequest): Record<string, unknown> {
    const { system, messages } = splitSystem(req.messages, this.promptCaching);

    const body: Record<string, unknown> = {
      model: this.model,
      max_tokens: req.maxTokens ?? 4096,
      messages,
    };
    if (system.length) body['system'] = system;
    if (req.temperature !== undefined) body['temperature'] = req.temperature;
    if (req.topP !== undefined) body['top_p'] = req.topP;
    if (req.stop?.length) body['stop_sequences'] = req.stop;
    if (req.tools?.length) {
      body['tools'] = req.tools.map((t) => ({
        name: t.name,
        description: t.description,
        input_schema: t.parameters,
      }));
    }

    // Extended thinking is incompatible with a custom temperature.
    const budget = this.effort === 'auto' ? 0 : (THINKING_BUDGET[this.effort] ?? 0);
    if (budget > 0) {
      body['thinking'] = { type: 'enabled', budget_tokens: budget };
      delete body['temperature'];
      // max_tokens must exceed the thinking budget.
      body['max_tokens'] = Math.max(Number(body['max_tokens']), budget + 2048);
    }
    return body;
  }

  async chat(req: ChatRequest): Promise<ChatResponse> {
    if (!this.apiKey) {
      throw new ProviderError(this.id, 'no Anthropic API key configured', {
        retryable: false,
        hint: 'Set ANTHROPIC_API_KEY, or run `proto config set-key anthropic <key>`.',
      });
    }

    const body = this.buildBody(req);

    const started = Date.now();
    let res;
    try {
      res = await request({
        url: joinUrl(this.baseUrl, 'messages'),
        method: 'POST',
        headers: this.headers(),
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
      const detail = errorText(res.text);
      throw new ProviderError(this.id, `Anthropic HTTP ${res.status}: ${detail}`, {
        status: res.status,
        retryable: res.status === 429 || res.status === 529 || res.status >= 500,
        ...(workspaceHint(detail, this.extraHeaders) ? { hint: workspaceHint(detail, this.extraHeaders) as string } : {}),
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
   * Native Messages API streaming.
   *
   * Anthropic's SSE stream is content-block oriented rather than a flat list of
   * deltas: text arrives as `text_delta`s, while a tool call is announced by a
   * `content_block_start` and its arguments are dribbled out as
   * `input_json_delta` fragments whose concatenation is only valid JSON at
   * `content_block_stop`. We therefore buffer per block index and only emit a
   * `tool-call` once the block closes, which is also the first moment the
   * arguments can be parsed at all.
   */
  async chatStream(req: ChatRequest, onEvent: (event: StreamEvent) => void): Promise<ChatResponse> {
    if (!this.apiKey) {
      throw new ProviderError(this.id, 'no Anthropic API key configured', {
        retryable: false,
        hint: 'Set ANTHROPIC_API_KEY, or run `proto config set-key anthropic <key>`.',
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
        // `request()` from util/http.ts awaits `res.text()`, i.e. it buffers the
        // whole body, so it can never surface a delta. Streaming therefore talks
        // to global fetch directly and reads `response.body` incrementally.
        res = await fetch(joinUrl(this.baseUrl, 'messages'), {
          method: 'POST',
          headers: this.headers(),
          body: JSON.stringify(body),
          signal: controller.signal,
        });
      } catch (err) {
        throw toProviderError(this.id, err, this.baseUrl);
      }

      if (!res.ok) {
        const detail = errorText(await res.text().catch(() => ''));
        const hint = workspaceHint(detail, this.extraHeaders);
        throw new ProviderError(this.id, `Anthropic HTTP ${res.status}: ${detail}`, {
          status: res.status,
          retryable: res.status === 429 || res.status === 529 || res.status >= 500,
          ...(hint ? { hint } : {}),
        });
      }

      /**
       * A gateway or proxy may ignore `stream: true` and return an ordinary JSON
       * body (some corporate gateways and older shims do exactly this). Without
       * this guard the SSE parser finds no frames, returns an empty response, and
       * the agent silently behaves as if the model said nothing — a failure mode
       * that is very hard to diagnose from the outside. Mirror the OpenAI provider:
       * fall back to a normal buffered completion.
       */
      const contentType = res.headers.get('content-type') ?? '';
      if (!contentType.includes('text/event-stream')) {
        const raw = await res.text();
        let parsed: unknown;
        try {
          parsed = JSON.parse(raw);
        } catch {
          return emptyResponse(this.id, this.model, Date.now() - started, 'response was neither SSE nor JSON');
        }
        const normalized = this.normalize(parsed, Date.now() - started, req);
        if (normalized.text) onEvent({ type: 'text-delta', text: normalized.text });
        for (const call of normalized.toolCalls) onEvent({ type: 'tool-call', toolCall: call });
        onEvent({ type: 'done', response: normalized });
        return normalized;
      }

      let text = '';
      const toolCalls: ToolCall[] = [];
      let finish: FinishReason = 'stop';
      let inputTokens: number | undefined;
      let outputTokens: number | undefined;
      let cacheRead: number | undefined;
      let cacheWrite: number | undefined;
      let model = this.model;
      let sawMessageDelta = false;
      const toolBlocks = new Map<number, { id: string; name: string; json: string }>();

      const handleFrame = (eventName: string, data: string): void => {
        if (data === '[DONE]') return; // Anthropic does not send it, but ignoring it is free
        let parsed: Record<string, unknown>;
        try {
          parsed = JSON.parse(data) as Record<string, unknown>;
        } catch {
          return; // one malformed frame must not abort an otherwise good stream
        }
        const type = typeof parsed['type'] === 'string' ? parsed['type'] : eventName;
        switch (type) {
          case 'message_start': {
            const message = (parsed['message'] ?? {}) as Record<string, unknown>;
            const usage = (message['usage'] ?? {}) as Record<string, unknown>;
            inputTokens = numOr(usage['input_tokens'], inputTokens);
            cacheRead = numOr(usage['cache_read_input_tokens'], cacheRead);
            cacheWrite = numOr(usage['cache_creation_input_tokens'], cacheWrite);
            if (typeof message['model'] === 'string') model = message['model'];
            break;
          }
          case 'content_block_start': {
            const index = numOr(parsed['index'], 0) ?? 0;
            const block = (parsed['content_block'] ?? {}) as Record<string, unknown>;
            if (block['type'] === 'tool_use') {
              toolBlocks.set(index, {
                id: typeof block['id'] === 'string' ? block['id'] : `tool_${index}`,
                name: typeof block['name'] === 'string' ? block['name'] : '',
                json: '',
              });
            } else if (block['type'] === 'text' && typeof block['text'] === 'string' && block['text']) {
              // Real streams open text blocks with "", but a non-empty start is
              // legal and must be counted or `text` would disagree with chat().
              text += block['text'];
              onEvent({ type: 'text-delta', text: block['text'] });
            }
            break;
          }
          case 'content_block_delta': {
            const index = numOr(parsed['index'], 0) ?? 0;
            const delta = (parsed['delta'] ?? {}) as Record<string, unknown>;
            if (delta['type'] === 'text_delta' && typeof delta['text'] === 'string') {
              text += delta['text'];
              onEvent({ type: 'text-delta', text: delta['text'] });
            } else if (delta['type'] === 'input_json_delta' && typeof delta['partial_json'] === 'string') {
              const block = toolBlocks.get(index);
              if (block) block.json += delta['partial_json'];
            }
            break;
          }
          case 'content_block_stop': {
            const index = numOr(parsed['index'], 0) ?? 0;
            const block = toolBlocks.get(index);
            if (block) {
              toolBlocks.delete(index);
              const toolCall: ToolCall = { id: block.id, name: block.name, args: parseToolArgs(block.json) };
              toolCalls.push(toolCall);
              onEvent({ type: 'tool-call', toolCall });
            }
            break;
          }
          case 'message_delta': {
            sawMessageDelta = true;
            const delta = (parsed['delta'] ?? {}) as Record<string, unknown>;
            if (delta['stop_reason'] !== undefined) finish = mapStopReason(delta['stop_reason']);
            const usage = (parsed['usage'] ?? {}) as Record<string, unknown>;
            outputTokens = numOr(usage['output_tokens'], outputTokens);
            break;
          }
          case 'error': {
            const err = (parsed['error'] ?? {}) as Record<string, unknown>;
            const message = typeof err['message'] === 'string' ? err['message'] : 'Anthropic stream error';
            throw new ProviderError(this.id, message, { retryable: err['type'] === 'overloaded_error' });
          }
          default:
            break; // ping, message_stop, and anything the API adds later
        }
      };

      const reader = res.body?.getReader();
      if (!reader) throw new Error('Anthropic streaming response had no body');
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

      const latencyMs = Date.now() - started;
      const usage: Usage = {
        inputTokens: inputTokens ?? estimateTokens(req.messages.map((m) => m.content).join('\n')),
        outputTokens: outputTokens ?? estimateTokens(text),
      };
      if (cacheRead !== undefined || cacheWrite !== undefined) usage.cachedInputTokens = cacheRead ?? 0;

      // Mirror `normalize`'s empty-response rule, with one exception: a stream
      // that ends before `message_delta` was truncated in transit, so an empty
      // body there is a fallback rather than a model-level empty completion.
      let response: ChatResponse;
      if (sawMessageDelta && !text && toolCalls.length === 0) {
        response = { ...emptyResponse(this.id, this.model, latencyMs, 'Anthropic returned no content'), usage };
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
    const root = (parsed ?? {}) as Record<string, unknown>;
    const blocks = Array.isArray(root['content']) ? (root['content'] as unknown[]) : [];
    let text = '';
    const toolCalls: ToolCall[] = [];
    for (const [i, raw] of blocks.entries()) {
      const b = (raw ?? {}) as Record<string, unknown>;
      if (b['type'] === 'text' && typeof b['text'] === 'string') text += b['text'];
      else if (b['type'] === 'tool_use' && typeof b['name'] === 'string') {
        toolCalls.push({
          id: typeof b['id'] === 'string' ? b['id'] : `tool_${i}`,
          name: b['name'],
          args: b['input'],
        });
      }
      // 'thinking' blocks are intentionally dropped: they are not part of the
      // candidate and must never be fed into verification.
    }

    const rawUsage = (root['usage'] ?? {}) as Record<string, unknown>;
    const usage: Usage = {
      inputTokens:
        numOr(rawUsage['input_tokens']) ?? estimateTokens(req.messages.map((m) => m.content).join('\n')),
      outputTokens: numOr(rawUsage['output_tokens']) ?? estimateTokens(text),
    };
    const cacheRead = numOr(rawUsage['cache_read_input_tokens'], undefined);
    const cacheWrite = numOr(rawUsage['cache_creation_input_tokens'], undefined);
    if (cacheRead !== undefined || cacheWrite !== undefined) {
      usage.cachedInputTokens = cacheRead ?? 0;
    }

    const finish = mapStopReason(root['stop_reason']);
    if (!text && toolCalls.length === 0) {
      return {
        ...emptyResponse(this.id, this.model, latencyMs, 'Anthropic returned no content'),
        usage,
      };
    }

    return {
      text,
      toolCalls,
      usage,
      finishReason: finish,
      model: typeof root['model'] === 'string' ? (root['model'] as string) : this.model,
      providerId: this.id,
      latencyMs,
      costUsd: computeCost(usage, this.price),
    };
  }

  async health(): Promise<Health> {
    // There is no unauthenticated ping; a 1-token message is the cheapest real check.
    const started = Date.now();
    const latencyMs = (): number => Date.now() - started;
    try {
      const res = await request({
        url: joinUrl(this.baseUrl, 'messages'),
        method: 'POST',
        headers: this.headers(),
        body: JSON.stringify({
          model: this.model,
          max_tokens: 1,
          messages: [{ role: 'user', content: 'ping' }],
        }),
        timeoutMs: Math.min(this.timeoutMs, 15_000),
        retries: 0,
        label: `${this.id}-health`,
      });
      if (res.status === 401 || res.status === 403) {
        return { ok: false, detail: `Anthropic rejected the API key (HTTP ${res.status})`, latencyMs: latencyMs() };
      }
      if (!res.ok && res.status !== 400) {
        return { ok: false, detail: `Anthropic HTTP ${res.status}: ${errorText(res.text)}`, latencyMs: latencyMs() };
      }
      return { ok: true, detail: `Anthropic reachable, model "${this.model}" accepted`, latencyMs: latencyMs() };
    } catch (err) {
      return { ok: false, detail: err instanceof Error ? err.message : String(err), latencyMs: latencyMs() };
    }
  }
}

/** A message as it goes on the wire: Anthropic always uses a block array for content. */
interface WireMessage {
  role: string;
  content: unknown[];
}

/** True for the synthetic user message that carries one or more `tool_result` blocks. */
function isToolResultMessage(message: WireMessage): boolean {
  if (message.role !== 'user') return false;
  return (
    message.content.length > 0 &&
    message.content.every((block) => (block as Record<string, unknown>)['type'] === 'tool_result')
  );
}

function splitSystem(
  messages: Message[],
  caching: boolean,
): { system: unknown[]; messages: unknown[] } {
  const system: unknown[] = [];
  const rest: WireMessage[] = [];
  for (const m of messages) {
    if (m.role === 'system') {
      const block: Record<string, unknown> = { type: 'text', text: m.content };
      if (caching && m.cacheBreakpoint) block['cache_control'] = { type: 'ephemeral' };
      system.push(block);
      continue;
    }
    if (m.role === 'tool') {
      // Anthropic takes tool results as `tool_result` blocks inside a *user*
      // message, and rejects two consecutive user messages. A run of tool
      // results therefore has to collapse into one user message; note that
      // `cache_control` deliberately never lands on a tool_result block.
      const block = {
        type: 'tool_result',
        tool_use_id: m.toolCallId ?? 'unknown',
        content: m.content,
      };
      const last = rest[rest.length - 1];
      if (last && isToolResultMessage(last)) last.content.push(block);
      else rest.push({ role: 'user', content: [block] });
      continue;
    }

    // An assistant turn that requested tools must replay its `tool_use` blocks,
    // otherwise the API sees tool results that reference calls it never made.
    const blocks: unknown[] = [];
    if (m.content || !m.toolCalls?.length) {
      const block: Record<string, unknown> = { type: 'text', text: m.content };
      if (caching && m.cacheBreakpoint) block['cache_control'] = { type: 'ephemeral' };
      blocks.push(block);
    }
    for (const call of m.toolCalls ?? []) {
      blocks.push({ type: 'tool_use', id: call.id, name: call.name, input: call.args ?? {} });
    }
    rest.push({ role: m.role, content: blocks });
  }
  // Anthropic requires the first message to be from the user.
  if (rest.length && rest[0]?.['role'] !== 'user') {
    rest.unshift({ role: 'user', content: [{ type: 'text', text: '(continue)' }] });
  }
  return { system, messages: rest };
}

function mapStopReason(reason: unknown): FinishReason {
  switch (reason) {
    case 'end_turn':
    case 'stop_sequence':
    case null:
    case undefined:
      return 'stop';
    case 'max_tokens':
      return 'length';
    case 'tool_use':
      return 'tool_calls';
    case 'refusal':
      return 'content_filter';
    default:
      return 'stop';
  }
}

function numOr(value: unknown, fallback?: number): number | undefined {
  return typeof value === 'number' && Number.isFinite(value) ? value : fallback;
}

/**
 * Tool arguments are streamed as JSON fragments, so the concatenation is only
 * meaningful once the block closes. A model that emits malformed or absent JSON
 * still produced a usable call, so we degrade to `{}` rather than throwing.
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
 * SSE frames are separated by a blank line and carry `event:` / `data:` lines.
 * A network chunk boundary can fall anywhere — mid-line, mid-frame, even
 * mid-character (the caller decodes with `{ stream: true }`) — so this buffers
 * text and only dispatches once it sees the blank line that ends a frame.
 * Comment lines (leading `:`), used as keepalives, are ignored.
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

/**
 * Turn Anthropic's workspace-scoping 400 into something the user can act on.
 *
 * The raw message tells you *what* is missing but not *how* to fix it in this
 * harness, and it is the kind of error people hit once and then lose twenty minutes
 * to. Returns undefined for every other error.
 */
function workspaceHint(detail: string, headersSent: Record<string, string>): string | undefined {
  if (!/workspace/i.test(detail)) return undefined;
  if (headersSent['anthropic-workspace-id']) {
    return `An anthropic-workspace-id header was sent and rejected. Check that the workspace id is correct and belongs to the same organisation as the key.`;
  }
  return (
    `This key is not scoped to a workspace. Either set the workspace id ` +
    `(\`proto config set cloud.workspaceId <id>\`, or export ANTHROPIC_WORKSPACE_ID), ` +
    `or create an API key inside a specific workspace in the Anthropic Console, which needs no header.`
  );
}

function errorText(text: string): string {
  try {
    const parsed = JSON.parse(text) as { error?: { message?: string } };
    if (parsed.error?.message) return parsed.error.message;
  } catch {
    /* raw body */
  }
  return text.slice(0, 300);
}
