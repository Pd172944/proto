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
  }

  private headers(): Record<string, string> {
    const h: Record<string, string> = {
      'content-type': 'application/json',
      'anthropic-version': this.version,
    };
    if (this.apiKey) h['x-api-key'] = this.apiKey;
    return h;
  }

  async chat(req: ChatRequest): Promise<ChatResponse> {
    if (!this.apiKey) {
      throw new ProviderError(this.id, 'no Anthropic API key configured', {
        retryable: false,
        hint: 'Set ANTHROPIC_API_KEY, or run `proto config set-key anthropic <key>`.',
      });
    }

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
      throw new ProviderError(this.id, `Anthropic HTTP ${res.status}: ${errorText(res.text)}`, {
        status: res.status,
        retryable: res.status === 429 || res.status === 529 || res.status >= 500,
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

function splitSystem(
  messages: Message[],
  caching: boolean,
): { system: unknown[]; messages: unknown[] } {
  const system: unknown[] = [];
  const rest: unknown[] = [];
  for (const m of messages) {
    if (m.role === 'system') {
      const block: Record<string, unknown> = { type: 'text', text: m.content };
      if (caching && m.cacheBreakpoint) block['cache_control'] = { type: 'ephemeral' };
      system.push(block);
      continue;
    }
    if (m.role === 'tool') {
      rest.push({
        role: 'user',
        content: [
          {
            type: 'tool_result',
            tool_use_id: m.toolCallId ?? 'unknown',
            content: m.content,
          },
        ],
      });
      continue;
    }
    const block: Record<string, unknown> = { type: 'text', text: m.content };
    if (caching && m.cacheBreakpoint) block['cache_control'] = { type: 'ephemeral' };
    rest.push({ role: m.role, content: [block] });
  }
  // Anthropic requires the first message to be from the user.
  if (rest.length && (rest[0] as Record<string, unknown>)['role'] !== 'user') {
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

function errorText(text: string): string {
  try {
    const parsed = JSON.parse(text) as { error?: { message?: string } };
    if (parsed.error?.message) return parsed.error.message;
  } catch {
    /* raw body */
  }
  return text.slice(0, 300);
}
